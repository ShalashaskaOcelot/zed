# Phase 21 — Live elapsed-time counter while a cell runs

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass. Kind: **change to existing behaviour** — keep OPEN until the user
> confirms the counter ticks.

Requested by the user 2026-07-11.

Goal: while a cell is running, show a live ticking elapsed time (e.g.
"Running… 3.2s") instead of only revealing the total once it finishes. On
completion it settles to the final "✓ <time>" already implemented in phase 18.

Primary files: `crates/repl/src/notebook/cell.rs`
(`execution_status_element`, `execution_start_time`), and a per-second (or
finer) refresh mechanism.

## Implemented

- [x] While Running, the in-cell status shows "Running… <elapsed>" from
      `execution_start_time`, updating live.
- [x] Refresh driven by a per-cell 100ms `cx.spawn` +
      `cx.background_executor().timer` loop (`_run_timer` on `CodeCell`) that
      notifies only while the status is Running and ends itself otherwise —
      no timer runs when nothing is running.
- [x] Formatting shares `format_duration` with the finished time (ms →
      s → m s).
- [x] The task is dropped (cancelled) by every terminal transition
      (finish / cancel / re-pending), and self-terminates as a backstop.

## Manual test checklist (for the user)

- [ ] Run a multi-second cell → the time ticks up live next to "Running…",
      then settles to the final ✓ + time.
- [ ] No stray ticking/refreshing when nothing is running.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; 42 tests pass.
