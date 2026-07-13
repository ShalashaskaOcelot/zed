# Phase 17 — Execution status & queue correctness

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass.
> Kind: **change to existing behaviour** — keep OPEN until the user confirms
> the statuses/timings actually behave as described.

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

## Manual test checklist (for the user)

- [ ] Shift-enter down a stack with a kernel running: waiting cells show a
      muted "… Pending", only the executing cell shows "Running", and each
      cell's finished time is ITS OWN (the cells after a 20s cell show ~ms).
- [ ] Run All over previously-executed cells: ✓ ticks become Pending, old
      outputs stay until each cell re-executes.
- [ ] Restart mid-batch: the running cell and the queued cells show a muted ✕
      "Cancelled" — no ✓, no inherited time.
- [ ] Interrupt mid-batch: the running cell finishes with the
      KeyboardInterrupt error (✓ + its real time); the rest are Cancelled.
- [ ] Shift-enter through a stack with NO kernel selected, pick a kernel from
      the picker: ALL queued cells run, including the first.
- [ ] Stop-on-error still works (a failing cell cancels the rest).

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; 42 tests pass.
