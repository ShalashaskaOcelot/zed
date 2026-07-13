# Phase 22 — Multi-select cells

Kind: **new feature** (sizeable). Not yet started — this is a plan. Requested
by the user 2026-07-11. VS Code / file-explorer / Excel-style multi-selection.

Goal: select multiple cells and act on them together.

Desired behaviour:
- shift + down/up (command mode): keep the anchor cell selected and extend the
  selection to the adjacent cell (contiguous range grows/shrinks).
- shift + click a cell: select the whole contiguous range from the anchor to
  the clicked cell, inclusive.
- ctrl/cmd + click: toggle an individual cell into a discontiguous selection.
- ctrl/cmd + arrows: do nothing (no selection change).
- Cell actions (delete, copy/cut, run, move, convert) operate on the FULL
  selection.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (selection state,
navigation handlers, every cell action), `cell.rs` (selected rendering),
`assets/keymaps/*` (shift+arrow bindings).

## Tasks

- [ ] Replace/augment the single `selected_cell_index` with a selection SET +
      an anchor (keep a "primary"/active cell for command-mode single-key ops
      and rendering focus).
- [ ] shift+down/up extends the contiguous range from the anchor; plain
      arrows collapse to a single selection (current behaviour).
- [ ] shift+click selects the anchor→clicked range; ctrl/cmd+click toggles a
      cell; ctrl/cmd+arrows are no-ops.
- [ ] Render all selected cells with the selection treatment (accent bar /
      background), distinguishing the primary cell if needed.
- [ ] Make the cell actions operate over the selection: delete removes all
      selected; copy/cut capture all; run runs all (in order); move moves the
      block; convert converts all.
- [ ] Undo/redo must treat a multi-cell action as one logical operation
      (group the `CellEdit`s) — or clearly document if grouped.

## Risks / gaps

- Large surface: every action and the render path assume a single index today.
  Do it incrementally (selection state + nav first, then wire actions).
- Undo grouping for multi-cell delete/move is the trickiest part.
- Interaction with the per-cell toolbar / hover.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean; unit-test the
  selection-set math (extend, toggle, range) where feasible.
- User test: each selection gesture matches VS Code; actions apply to the whole
  selection; undo restores a multi-cell action in one step.
