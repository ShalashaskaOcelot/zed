# Phase 21 — Live elapsed-time counter while a cell runs

Kind: **change to existing behaviour** (extends the phase-17/18 status
display). Not yet started — this is a plan. Requested by the user 2026-07-11.

Goal: while a cell is running, show a live ticking elapsed time (e.g.
"Running… 3.2s") instead of only revealing the total once it finishes. On
completion it settles to the final "✓ <time>" already implemented in phase 18.

Primary files: `crates/repl/src/notebook/cell.rs`
(`execution_status_element`, `execution_start_time`), and a per-second (or
finer) refresh mechanism.

## Tasks

- [ ] While `CellExecutionStatus::Running`, render the elapsed time from
      `execution_start_time` next to the "Running…" label, updating live.
- [ ] Drive the refresh with a lightweight timer (e.g. a repeating
      `cx.spawn` + `cx.background_executor().timer` that calls `cx.notify()`
      while any cell is running) rather than per-frame work; stop it when
      nothing is running.
- [ ] Format consistently with the finished time (`format_duration`): sub-second
      as ms, then `s`, then `m s`. A running counter probably updates ~10x/s
      for sub-second cells and ~1x/s beyond.
- [ ] Ensure the timer is cancelled on finish/cancel and doesn't leak or keep
      the view awake when idle.

## Risks / gaps

- Don't spin a timer when no cell is running (battery / wakeups).
- The timer must live on the notebook (or cell) and be dropped appropriately;
  avoid multiple overlapping timers.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: run a multi-second cell → the time ticks up live, then settles to
  the final value with the ✓.
