# Phase 17 — Execution status & queue correctness (archived 2026-07-12)

> ✅ STATUS: IMPLEMENTATION COMPLETE; core behaviours user-confirmed
> (per-cell timing, pending/cancelled statuses, batch supersede, stop-on-error,
> no-kernel prompting). Kind: **change to existing behaviour**.
> The last few user-test items (interrupted-cell ✕, rerun-during-run hang fix,
> re-run after finish, stuck-Running monitoring) are tracked in
> `to-do/awaiting_testing.md` under "Phase 17".

These items were all reported by the user 2026-07-11 while running cells with
shift-enter and Run All; they share a root cause (queued cells were marked
"Running" and started their timer the moment they were queued, not when the
kernel actually began executing them).

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (execute_cell,
run_cell_batch/advance_run_queue, route, restart/interrupt),
`crates/repl/src/notebook/cell.rs` (start_execution / finish_execution /
is_executing / execution_start_time, plus a new "pending" state).

## Problem summary (observed)

- Shift-enter queueing: pressing shift-enter down a stack immediately marks
  every following cell "Running". They should be PENDING; only the cell the
  kernel is actually executing should be "Running".
- Because queued cells enter the "Running" state (and start their timer)
  early, their reported execution time includes the wait behind a long cell:
  two ~20ms cells queued behind a 20s cell both showed ~20s. Alone they show
  the correct time.
- Run All ("Execute all cells") does NOT show this timing bug (it starts each
  cell properly) but it does NOT set a pending status on the waiting cells, and
  it does NOT reset a previously-executed cell's ✓ tick to a pending state when
  re-running (the stale ✓ lingers).
- Restart (and interrupt) mid-execution: a cell that was 5s into a 20s run got
  a COMPLETED ✓ with a 5s time instead of a cancelled/failed state; the queued
  cells below it also got ✓ + the same 5s time despite never executing.
- Shift-enter with NO kernel selected: holding shift+enter through the stack
  selected a kernel but SKIPPED the first cell (only happens when no kernel was
  pre-selected; see also bug #16, same span-kernel-selection path).

## Implemented

- [x] Explicit per-cell execution state: `CellExecutionStatus` — `Idle →
      Pending → Running → {Finished | Cancelled}` (replaces the `is_executing`
      bool). Queued cells are Pending (muted "…" indicator), not Running.
- [x] The timer starts (and the previous output is dropped) only when the
      kernel's `execute_input` for that cell arrives — i.e. when it actually
      begins executing. Queued cells no longer inherit the wait behind a long
      cell (the two ~20ms cells behind a 20s cell now report ~20ms).
- [x] Shift-enter stacking and the kernel-starting flush both leave cells
      Pending until their own `execute_input`.
- [x] Run All / batch runs (`run_cell_batch`): every code cell in the batch is
      marked Pending immediately — a previously-executed cell's ✓ makes way
      for the pending marker, but its OUTPUT stays until it re-executes.
- [x] Restart / kernel loss / kernel switch (`stop_executing_cells`) now
      CANCEL running/queued cells (muted ✕, no time) instead of finishing
      them with a bogus ✓ + elapsed time. `cancel_run_queue` clears the
      pending markers of still-queued batch cells.
- [x] Kernel `aborted` replies (the requests a kernel discards after an error
      or interrupt) mark their cells Cancelled instead of Finished, and abort
      the batch like an error.
- [x] Shift-enter with no kernel skipping the first cell: root cause found —
      after the picker opened, `enter_command_mode`/`advance_in_command_mode`
      re-focused the notebook, which DISMISSED the picker popover, whose
      dismiss callback drops the awaiting cells (so cell 1 was discarded).
      Both now skip the focus grab while the kernel picker is deployed.
- [x] Debug logging around promote/dismiss/flush of queued cells (also serves
      as the bug #16 instrumentation pass).

## Follow-up fixes from user testing (2026-07-11)

User CONFIRMED: pending appears, per-cell times behind a long cell are correct,
cancelled status works, Run All marks pending. Remaining issues found and fixed:

- [x] Cells stuck "Running" (and two showing Running at once): shell
      `ExecuteReply` could arrive before iopub `ExecuteInput` for a fast cell,
      and `begin_running` then resurrected the finished cell. Made the status
      transition monotonic (begin_running won't revive a Finished/Cancelled
      cell; it still clears the previous outputs). (commit 5fcfca1)
- [x] Reworded the pending indicator "…Pending" → "Pending…".
- [x] Run All (batch) while a run is in flight left every new cell stuck
      Pending, because `advance_run_queue` won't start while a cell is active.
      A batch now SUPERSEDES the in-flight run: interrupt it, drop its routing
      (`execution_requests.clear()`), cancel its cells, then start fresh. A
      single-cell run still just queues at the kernel (Jupyter behaviour).

## Follow-up round 2 (user testing 2026-07-11)

- [x] Escape on the kernel picker left the triggering cell stuck "Pending":
      `clear_awaiting_cells` now cancels the awaiting cells' status too.
- [x] Pending icon changed from Ellipsis to Clock (kept the "Pending…" text).
- [x] Shift-enter cycling past an already running/queued cell re-ran it (and
      reset the running cell to Pending). Added `is_cell_in_flight` and the
      single-cell run paths skip cells already in flight — only eligible
      (idle/finished) cells get re-queued.
- [x] REGRESSION from the above (user 2026-07-11): `is_cell_in_flight` keyed
      off `execution_requests`, which is never pruned per-cell, so finished
      cells stayed "in flight" forever — nothing could be re-run without a
      kernel restart. Re-based the check on the cell's own Pending/Running
      status (resolves to Finished/Cancelled). (commit 93f8a96)
- [x] Run All while running cancelled the NEW run: interrupting puts ipykernel
      into an "aborting" state, so requests submitted immediately came back
      Aborted. The batch now defers submitting until the kernel returns to Idle
      (`resume_run_queue_on_idle`, resolved in `route`'s Status handler).

## Follow-up round 3 (user testing 2026-07-12)

- [x] Rerun-during-a-run could HANG (everything stuck): the superseding batch
      waited for a Status(idle) that never arrives when the kernel wasn't
      actually busy at supersede time. Only wait when Busy; submit immediately
      otherwise. (commit 051c3c6)
- [x] Interrupted cell showed ✓ + time; user wants ✕: a KeyboardInterrupt
      error now marks the cell Cancelled (traceback still shown), and
      finish_execution won't flip a Cancelled cell back to Finished.
      (commit 051c3c6)

## Manual test checklist (for the user)

- [x] Shift-enter down a stack with a kernel running: only the executing cell
      shows "Running", each cell's finished time is ITS OWN (cells after a 20s
      cell show ~ms). ✅ CONFIRMED (per-cell times + pending).
- [x] Run All over previously-executed cells: ✓ ticks become Pending, old
      outputs stay until each cell re-executes. ✅ CONFIRMED ("Run all does
      mark everything pending").
- [x] Restart mid-batch: the running cell and the queued cells show a muted ✕
      "Cancelled" — no ✓, no inherited time. ✅ CONFIRMED ("Cancelled status
      works perfectly").
- [x] Interrupt mid-batch works ✅ CONFIRMED 2026-07-12 (with one change
      requested and made: the interrupted cell now shows ✕ instead of ✓ —
      retest tracked in awaiting_testing.md).
- [x] Execute All with no kernel selected prompts kernel selection.
      ✅ CONFIRMED 2026-07-12.
- [x] Shift-enter through a stack with NO kernel selected: picker appears,
      cells queue and execute once a kernel is selected. ✅ CONFIRMED
      2026-07-12. ("Sometimes kernels do load in instantly" — the empty-picker
      case remains bug #20.)
- [x] Stop-on-error still works. ✅ CONFIRMED 2026-07-12.

Remaining user-test items moved to `to-do/awaiting_testing.md` (phase 17
section): interrupted-cell ✕, rerun-during-run no longer hangs, re-run after
finish, stuck-Running monitoring.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; 42 tests pass.
