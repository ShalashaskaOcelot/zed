# Phase 17 — Execution status & queue correctness

Kind: **change to existing behaviour** (fix how cell execution status and
timing are tracked across queued/batch runs). Not yet started — this is a plan.
These items were all reported by the user 2026-07-11 while running cells with
shift-enter and Run All; they share a root cause (queued cells are marked
"Running" and start their timer the moment they are queued, not when the kernel
actually begins executing them).

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

## Tasks

- [ ] Introduce an explicit per-cell execution state: `Idle → Pending →
      Running → {Done | Cancelled | Error}` (replace the single `is_executing`
      bool). Queued cells are Pending (distinct icon), not Running.
- [ ] Start the execution timer only when the cell actually begins running
      (on ExecuteInput / when submitted to the kernel as the active cell), not
      when it is queued. Fixes the inherited long-cell time.
- [ ] `advance_run_queue` / the shift-enter queue: mark queued cells Pending
      up front; promote to Running one at a time as each is submitted.
- [ ] Run All: set Pending on all target cells; on a previously-executed cell,
      replace the ✓ with Pending but KEEP its old output until it re-executes
      and overwrites it.
- [ ] Restart / interrupt: any Running or Pending cell becomes Cancelled (no ✓,
      no bogus time); do not mark cells that never ran as complete. (Refines
      the phase-10 cancel path + bug #3/#1 restart path.)
- [ ] Shift-enter with no kernel: ensure the first queued cell is not dropped
      when the kernel is chosen (fix the promote/skip off-by-one; verify against
      bug #16's misrouting in the same path).

## Risks / gaps

- Touches the batch state machine (phase 10) and the restart/interrupt paths
  (bug #1/#3) — regression-test stop-on-error and restart-cancels-batch.
- Coordinate the new state with the status display work in phase 18 (icons).

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean; add unit tests for
  the state transitions where feasible (timer start, pending vs running).
- User test: shift-enter a stack (correct per-cell times, pending icons);
  Run All (pending + tick reset, output kept); restart mid-run (cancelled, no
  bogus times); shift-enter with no kernel (first cell runs).
