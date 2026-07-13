# Phase 20 — Per-cell scoped stop / interrupt

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass. Kind: **change to existing behaviour** — keep OPEN until the user
> confirms the scoped behaviour.

Requested by the user 2026-07-11.

Problem: the gutter stop button (shown on a running/queued cell) dispatches the
global `InterruptKernel`, which interrupts the kernel AND cancels the entire
batch. Stopping should be scoped to the cell that owns the button.

Desired behaviour:
- Stop on a RUNNING cell: interrupt the kernel (interrupt is inherently global —
  only one cell runs at a time), but do NOT cancel the rest of the queued
  batch beyond what the interrupt implies. (Open question: after interrupting
  the running cell, should the queue continue to the next cell, or stop? VS
  Code / Jupyter "interrupt" stops the current cell; the batch's stop-on-error
  path already halts the rest. Decide: interrupting the active cell halts the
  batch, same as today, since that IS the running cell.)
- Stop on a PENDING/queued cell: remove ONLY that cell from the queue (mark it
  Idle/Cancelled) and leave the cells above and below it to run. No kernel
  interrupt.

Primary files: `crates/repl/src/notebook/cell.rs` (the gutter control's
on_click currently dispatches `InterruptKernel`), `notebook_ui.rs`
(`run_queue`, `active_run_cell`, a new per-cell stop handler; likely a new
`CellEvent`/`CellToolbarAction` or a dedicated event).

## Implemented

- [x] The gutter stop button emits `CellEvent::Stop(cell_id)` instead of the
      global `InterruptKernel`.
- [x] `handle_cell_stop`: the active (running) cell → interrupt the kernel
      (halts the batch, as interrupt always has — this IS the running cell).
- [x] A queued (pending) batch cell → removed from `run_queue` only, its status
      cancelled; the rest of the batch keeps running.
- [x] A single running cell (not in a batch) → interrupt; a pending/awaiting
      single cell → cancelled and removed from `pending_executions` /
      `cells_awaiting_kernel_choice`.
- [x] Toolbar / kernel-status Stop affordances unchanged; only the per-cell
      gutter button changed scope.

## Decision (from the plan's open question)

Interrupting the ACTIVE running cell still halts the batch — an interrupt is
kernel-global and the user previously confirmed (phase 10) that interrupting
mid-batch stopping the rest is correct. Only the PENDING-cell case changed:
stopping a not-yet-running cell drops just that one.

## Manual test checklist (for the user)

- [ ] Run a batch; click stop on a still-PENDING cell → only that cell is
      removed/cancelled, the cells above and below keep running.
- [ ] Click stop on the RUNNING cell → it interrupts (as before).
- [ ] Stop on a single (non-batch) running cell → interrupts it.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; 42 tests pass.
