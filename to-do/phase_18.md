# Phase 18 — In-cell execution status display (VS Code style)

Kind: **change to existing behaviour** (relocate/restyle the execution status
chrome). Not yet started — this is a plan. Requested by the user 2026-07-11.

Goal: match VS Code's placement — show the run status (running spinner /
completed ✓ / pending / cancelled) and the execution time INSIDE the cell, in
its bottom-left corner with padding, instead of floating outside the cell.
Also finish centering the gutter run button, which the user still finds
slightly left of center.

Primary files: `crates/repl/src/notebook/cell.rs` (the code cell render — the
status/time row currently rendered above/around the output block, and the
gutter run button), possibly `notebook_ui.rs` for spacing constants.

## Tasks

- [ ] Move the execution status indicator + execution time out of the current
      out-of-cell position into the cell's bottom-left corner, inside the cell
      padding (VS Code layout). Applies to running / completed / pending /
      cancelled states (statuses come from phase 17).
- [ ] Center the gutter run button (and the `[N]` number) properly between the
      indicator bar and the cell edge — the current `left(px(3.))` +
      `GUTTER_WIDTH - 3` still reads as left-of-center (likely the icon
      button's own metrics); tune until it looks centered at runtime.
- [ ] Keep the status readable in both selected/hovered and idle states.

## Dependencies / ordering

- Best done WITH or AFTER phase 17, since the set of statuses (pending,
  cancelled) it must display is defined there.

## Risks / gaps

- Bottom-left placement must not collide with output content or the output
  gutter's "..." menu.
- Pure styling — verify in-app; no unit coverage.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: status + time sit neatly in the cell's bottom-left; run button
  visually centered in the gutter.
