# Phase 10 — Sequential multi-cell execution with stop-on-error (COMPLETE, archived 2026-07-08)

Kind: change to existing behaviour. Confirmed by the user 2026-07-08:
- "stop on error is working" (a failed cell cancels the rest of the batch);
- restart mid-batch "kills all" (stops the running cell, doesn't run the rest);
- interrupt mid-batch stopped the running cell's execution and "did not run
  the rest of the cells below".

## Delivered (confirmed)

- Batch runs (Run All / Run Cells Above / Run Cell and Below) submit ONE code
  cell at a time via `run_cell_batch` → `advance_run_queue`.
- `route` advances the queue on `ExecuteReply` Ok, clears it on Error.
- `cancel_run_queue` aborts the batch on interrupt / restart / change-kernel /
  kernel error / clean exit / launch failure / picker dismiss / delete or
  convert of a queued cell.
- Batch + no-kernel: queue preserved across the kernel prompt; a deliberate
  kernel switch cancels it. Single-cell runs unchanged.

## Related follow-up (not part of this phase)

- Interrupt TIMING: interrupting a C-level blocking call (e.g. `time.sleep`)
  on Windows is delayed until the call returns — a known ipykernel/Windows
  event-interrupt limitation, tracked with bug #3 / backlog (immediate
  interrupt via `GenerateConsoleCtrlEvent`).

## Verification

- Confirmed at runtime by the user. `cargo test -p repl` passes; clippy clean.
