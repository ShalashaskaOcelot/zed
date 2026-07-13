# Phase 20 — Per-cell scoped stop / interrupt

Kind: **change to existing behaviour**. Not yet started — this is a plan.
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

## Tasks

- [ ] The gutter stop button emits a per-cell "stop" (carrying the cell id)
      instead of the global `InterruptKernel`.
- [ ] Handler: if the cell is the `active_run_cell` (running) → interrupt the
      kernel (current behaviour for the running cell).
- [ ] If the cell is merely queued (in `run_queue`) → remove just that cell
      from `run_queue`, cancel its status, and leave the rest of the queue
      intact so it keeps running.
- [ ] Keep the toolbar/kernel-status Stop affordances working; only the
      per-cell gutter button changes scope.

## Risks / gaps

- Removing a single cell from a batch mid-run must not desync `active_run_cell`
  or the stop-on-error/advance logic.
- Confirm the interrupt-vs-continue decision for the running cell with the user
  if ambiguous.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: stop a queued (not-yet-running) cell → it's removed, others still
  run; stop the running cell → it interrupts.
