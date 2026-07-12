# Phase 13 — Per-cell hover/selection toolbar

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass. Kind: **new feature** — on confirmation the toolbar appears and each
> button fires the right action, archive; tweaks become new items.

Goal: a VS Code-style per-cell toolbar shown on the selected/hovered cell so the
common actions are discoverable without the "More options" menu or memorising
keys. USER PREFERENCE (2026-07-08): wants run-above / run-cell-and-below (and
likely delete / add) shown individually in the focused cell's top-right, NOT
tucked inside "More options".

Primary files: `crates/repl/src/notebook/cell.rs` (cell render), reusing the
phase-4 actions already wired on the notebook root.

## Implemented

- [x] `CodeCell::cell_toolbar` renders a small action bar (icon buttons) in the
      code cell's top-right: Run cells above, Run cell and below, Add cell
      below, Delete. Each has a `Tooltip::for_action` so its keybinding shows.
      (Run-cell was intentionally NOT included — the left gutter run button is
      the single home for running the current cell; user 2026-07-11.)
- [x] Polished the left gutter "execution box": run/stop button now sits in a
      subtle rounded well, and the execution number renders Jupyter-style as
      `[N]` (it is the kernel's session-global `In [N]` execution count /
      order, confirmed 2026-07-11 — see phase 15).
- [x] Gutter redesign round 2 (user feedback 2026-07-11: the selection line
      clipped through the button and counter): the selection indicator is now a
      slim bar at the FAR-LEFT edge of the gutter (3px rounded accent bar when
      selected, 1px hairline otherwise — shared `gutter_indicator_bar` used by
      code, markdown, and output gutters), the gutter widened 19→26px, and the
      button/counter/ellipsis are inset to the right of the bar so nothing
      overlaps. Removed the old opaque-background masking hack and the well's
      border (was too busy at that size).
- [x] Gutter redesign round 3, VS Code-style (user feedback 2026-07-11 with
      VS Code screenshots): (1) run button is now a bare hovering button — no
      well — centered in the full gutter width so it clears the accent bar;
      (2) the run/edit/ellipsis gutter controls only show on the SELECTED or
      HOVERED cell (a running cell always shows its stop button); (3) the edge
      bar is now: accent on selected, grey on hovered, NOTHING otherwise (the
      old always-on hairline is gone); (4) hover is tracked per-cell via a
      shared `CELL_HOVER_GROUP` gpui group on every cell root (code, markdown,
      raw), which the toolbar also uses.
- [x] Shown when the cell is selected, and on hover via `group("code-cell")` +
      `group_hover(... .visible())` (invisible otherwise).
- [x] Buttons emit `CellEvent::ToolbarAction(cell_id, action)`; the notebook's
      `handle_cell_toolbar_action` FIRST selects that cell (by id, command mode,
      focuses the notebook handle) THEN runs the mapped action — so a button
      always acts on its own cell even when shown on hover of a non-selected
      cell, and focus returns to the notebook so shortcuts keep working.
- [x] The language badge moved from top-right to bottom-right to make room.

## Scope notes / follow-ups (candidate backlog)

- Toolbar is on CODE cells only. Markdown/raw cells could get a smaller
  (Add/Delete) toolbar later.
- Uses `ArrowUp`/`ArrowDown` icons for run-above/run-below (tooltips clarify);
  swap for more specific icons if any are added.

## Manual test checklist (for the user)

- [ ] Select a code cell → the toolbar appears top-right; hover a non-selected
      code cell → it appears there too.
- [ ] Each button fires the right action (run / run above / run below / add
      below / delete) on the CORRECT cell, including when clicked via hover on a
      cell that wasn't selected.
- [ ] After clicking a toolbar button, command-mode keyboard shortcuts still
      work (focus returned to the notebook).
- [ ] The language badge (now bottom-right) doesn't overlap the toolbar.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; `cargo test -p repl
  notebook` passes.
