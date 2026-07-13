# Phase 18 — In-cell execution status display (VS Code style)

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass.
> Kind: **change to existing behaviour** — keep OPEN until the user confirms
> the status sits inside the cell and the button reads centered.

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
- [x] Status row does not collide with the toolbar (top-right) or the language
      badge (bottom-right, absolute) — the status row is in normal flow on the
      left.

## Manual test checklist (for the user)

- [ ] Run a cell: "Running…" appears inside the cell, bottom-left; on finish
      it becomes ✓ + time in the same spot.
- [ ] Pending/cancelled states (batch runs, restart) show in the same in-cell
      spot.
- [ ] A cell with no outputs no longer grows an empty output box — the status
      lives in the cell itself.
- [ ] The gutter run button and `[N]` now read as centered between the accent
      bar and the cell.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; 42 tests pass.
