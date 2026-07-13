# Phase 18 — In-cell execution status display (VS Code style) (COMPLETE, archived 2026-07-11)

> ✅ STATUS: CONFIRMED by the user 2026-07-11: status/time render inside the
> cell ("looks good"), no empty output box ("looks good"), and the button
> position is "now good... close enough that I can't tell just by eye" after
> the rightward nudge. Kind: **change to existing behaviour**.

Requested by the user 2026-07-11.

Goal: match VS Code's placement — show the run status (running spinner /
completed ✓ / pending / cancelled) and the execution time INSIDE the cell, in
its bottom-left corner with padding, instead of floating outside the cell.
Also finish centering the gutter run button, which the user still finds
slightly left of center.

Primary files: `crates/repl/src/notebook/cell.rs` (the code cell render — the
status/time row currently rendered above/around the output block, and the
gutter run button), possibly `notebook_ui.rs` for spacing constants.

## Implemented

- [x] The execution status + time now render INSIDE the cell as a VS Code-
      style status bar row in the bottom-left (below the code, within the
      cell's border/padding), for all phase-17 states: pending / running /
      finished ✓ + time / cancelled ✕. Removed from the old position above the
      output block — the output section now appears only when there are actual
      outputs (no more empty output box just to show a time).
- [x] Gutter centering: widened the gutter 26→30px so the run button and the
      `[N]` number center in the bar-to-cell-edge span with symmetric (~3.5px)
      clearance on both sides, instead of sitting tight against the bar.
- [x] Centering: after trying bar-to-edge and full-gutter-width variants, the
      controls sit at `left(px(7.))` + `w(GUTTER_WIDTH - 7.0)` (nudged right of
      the accent bar). User: "now good."
- [x] Status row does not collide with the toolbar (top-right) or the language
      badge (bottom-right, absolute) — the status row is in normal flow on the
      left.

## Manual test checklist (for the user)

- [x] Run a cell: "Running…" appears inside the cell, bottom-left; on finish
      it becomes ✓ + time in the same spot. ✅ CONFIRMED.
- [x] Pending/cancelled states (batch runs, restart) show in the same in-cell
      spot. ✅ CONFIRMED.
- [x] A cell with no outputs no longer grows an empty output box — the status
      lives in the cell itself. ✅ CONFIRMED.
- [x] The gutter run button and `[N]` now read as centered. ✅ CONFIRMED
      ("now good... close enough").

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; 42 tests pass.
