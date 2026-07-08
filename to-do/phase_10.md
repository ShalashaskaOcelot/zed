# Phase 10 — Sequential multi-cell execution with stop-on-error

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (2026-07-08). Compiles,
> clippy-clean, unit tests pass. Runtime behaviour NOT yet confirmed. Do NOT
> archive until the user confirms.
>
> Kind: **change to existing behaviour** — changes how "Run all" / "Run cells
> above" / "Run cell and below" behave when a cell fails. Keep OPEN until the
> user confirms the batch actually stops on failure.
>
> ## Implementation summary
> - Added a `run_queue: Vec<CellId>` + `active_run_cell: Option<CellId>` to
>   `NotebookEditor`. Batch runs (`run_cells` / `run_cells_above` /
>   `run_cell_and_below`) now go through `run_cell_batch` → `advance_run_queue`,
>   which submits ONE code cell at a time and waits.
> - `route` watches for the active cell's `ExecuteReply`: `ReplyStatus::Ok`
>   advances to the next queued cell; `ReplyStatus::Error` clears the rest of
>   the queue (stop-on-error).
> - `cancel_run_queue` aborts the batch on interrupt, restart, kernel error,
>   clean exit, launch failure, picker dismiss, and delete/convert of a queued
>   cell.
> - Batch + no-kernel: the first cell is held awaiting a kernel choice and the
>   queue is preserved; `change_kernel` keeps the queue when it's satisfying a
>   prompt (awaiting non-empty) but cancels it on a deliberate switch.
> - Single-cell runs keep their direct `execute_cell` path.
>
> ## Manual test checklist (for the user)
> - [ ] Run All with a mid-notebook cell that raises: cells after it do NOT run.
> - [ ] Run All with a dead kernel: only the first errors; the rest don't run.
> - [ ] Run All with no errors: all cells run in order.
> - [ ] Interrupt / restart mid-batch stops the remaining cells.
> - [ ] Run All with no kernel selected: prompt → pick → whole batch runs;
>       prompt → Esc → nothing runs.

Goal: when running multiple cells, stop the batch as soon as a cell fails, and
clear the remaining queued cells — instead of blindly running every cell even
after a kernel error or an exception.

User report: running multiple cells (run all / run above / run cell and below)
does not stop on failure. If cell 1 dies with a kernel error, every subsequent
cell is still attempted (and hits the same error). If cell 1 raises a code
error, the rest still run. All remaining queued cells should be cleared on a
failure.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`
(`execute_cell`, `run_cells`, `run_cells_above`, `run_cell_and_below`,
`pending_executions`, the message router `route`/`on ... execute_reply`),
reference `crates/repl/src/session.rs` for how replies/status are handled.

## Why this needs real work (not a one-liner)

Today `run_*` loops call `execute_cell` for every cell up front, so all
`execute_request`s are handed to the kernel's shell channel at once. Once sent,
they cannot be "unsent" — the kernel executes them in order regardless of
earlier errors. To stop on error the FRONTEND must submit cells one at a time,
waiting for each to finish before sending the next.

## Design sketch

- Introduce an explicit run queue: an ordered list of cell ids representing the
  current batch, plus the id of the cell currently executing.
- Submit only the current cell's `execute_request`; on its `execute_reply`
  (or status idle) inspect the result:
  - `status == "ok"` → pop and submit the next cell in the batch.
  - `status == "error"` (exception) → clear the rest of the batch.
  - kernel error / kernel died (`kernel_errored`) → clear the whole batch.
- `run_cells` / `run_cells_above` / `run_cell_and_below` populate the batch
  rather than calling `execute_cell` in a loop.
- Single-cell run (ctrl-enter) is just a batch of one — unify the path.
- Interrupt / restart / change-kernel must clear the batch (extends the
  existing `execution_requests`/`pending_executions` clearing).

## Tasks

- [ ] Add a run-queue structure + "currently executing" tracking.
- [ ] Detect per-cell completion from the kernel messages (execute_reply
      status, or Status: idle for the cell's msg_id) and advance the queue.
- [ ] Stop-on-error: clear the remaining batch on cell error and on
      `kernel_errored`.
- [ ] Route `run_*` actions through the queue; keep single-run working.
- [ ] Ensure interrupt/restart/change-kernel/delete clear the queue and reset
      affected cell spinners.

## Open questions

- Make stop-on-error configurable later (some users want run-all to continue)?
  Default to stop-on-error (JupyterLab-style) for now; a setting is backlog.

## Manual test checklist (for the user)

- [ ] Run all with a dead kernel: only the first cell errors; the rest are not
      attempted / are cleared.
- [ ] Run all where cell 1 raises an exception: cells below do not run.
- [ ] Run all with no errors: all cells run in order.
- [ ] Interrupt / restart mid-batch clears the remaining queue.
