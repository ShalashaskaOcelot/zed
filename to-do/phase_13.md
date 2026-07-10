# Phase 13 — Per-cell hover/selection toolbar

Kind: **new feature** (discoverability layer over existing actions).
Not yet started — this is a plan.

Goal: a VS Code-style per-cell toolbar shown on the selected/hovered cell so the
common actions are discoverable without the "More options" menu or memorising
keys. USER PREFERENCE (2026-07-08): wants run-above / run-cell-and-below (and
likely delete / add) shown individually in the focused cell's top-right, NOT
tucked inside "More options".

Primary files: `crates/repl/src/notebook/cell.rs` (cell render), reusing the
phase-4 actions already wired on the notebook root.

## Tasks

- [ ] Render a small action bar in the top-right of the selected (and/or
      hovered) cell. Buttons dispatch the EXISTING actions — no new logic:
      Run, Run Above, Run Cell & Below, Delete, and Add (above/below).
- [ ] Icon buttons with `Tooltip::for_action` so each shows its keybinding.
- [ ] Show on selection and on hover; keep it out of the way when the cell is
      neither (avoid clutter on long notebooks).
- [ ] Make sure clicking a toolbar button returns focus to the notebook so
      command-mode shortcuts keep working (ties into bug #15 follow-up).

## Risks / gaps

- Focus: popovers/buttons that steal focus are the suspected cause of the
  bug #15 desync — route clicks through handlers that re-focus the notebook.
- Don't duplicate the whole "More options" menu inline; surface only the
  high-frequency actions and leave the rest in the menu.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: buttons appear on the focused/hovered cell, each fires the right
  action, and shortcuts still work afterwards.
