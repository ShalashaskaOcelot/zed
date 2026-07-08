# Phase 4 — Cell actions, Jupyter shortcuts, and the two dead menus

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (commit b018b45, 2026-07-08).
> Code is written, compiles, clippy-clean, and unit tests pass, but the
> runtime behaviour has NOT been confirmed by the user yet. Do NOT archive
> this phase until the user confirms. Ask periodically.

Goal: bring the per-cell action set up to Jupyter/VS Code conventions and give
the two dead ellipsis buttons real menus.

Primary files: `crates/zed_actions/src/lib.rs`,
`crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/notebook/cell.rs`,
`assets/keymaps/default-{linux,macos,windows}.json`.

## Tasks (implemented — each ⚠ = needs user confirmation)

### New actions + handlers

- [x] `DeleteCell` — removes the selected cell; refuses to delete the only
      remaining cell; selects a neighbour and returns to command mode; clears
      queued/pending execution state for the removed cell. ⚠ untested
- [x] `AddCellAbove` / `AddCellBelow` — generalised insertion; above inserts
      at the selected index, below reuses the add-code path. ⚠ untested
- [x] `RunCellsAbove` — runs all cells above the selected cell. ⚠ untested
- [x] `RunCellAndBelow` — runs the selected cell and everything below.
      ⚠ untested
- [x] `ConvertToCode` / `ConvertToMarkdown` — rebuilds the cell as the other
      type in place, preserving the current editor text. ⚠ untested

### Nav-mode keybindings (`NotebookEditor && notebook_mode == command`)

- [x] `a` add above, `b` add below, `x` and `d d` delete, `m` convert to
      markdown, `y` convert to code — added to all three keymaps. ⚠ untested

### Menus

- [x] Right-toolbar "More options" → popover menu (run above / run cell and
      below, add above/below, move up/down, convert, clear outputs, delete).
      ⚠ untested
- [x] Output "..." → per-cell output menu (Copy Output, Clear Output).
      ⚠ untested

### Cell hover controls

- [ ] Moved to `backlog.md` — actions/keybinds/menus already expose this; the
      VS Code-style per-cell hover toolbar is a later convenience pass.

## Notes / findings

- Refactored duplicated cell-subscription wiring into `wire_code_cell` /
  `wire_markdown_cell` and `build_code_cell` / `build_markdown_cell`.
- Dirty tracking: the workspace `Item::is_dirty` for `NotebookEditor`
  (`notebook_ui.rs:2200`) already returns
  `has_structural_changes() || has_content_changes()`, so the new cell
  mutations mark the tab dirty. Bug #9 concerns only the separate
  `NotebookItem::is_dirty` stub (left open).
- Nav-mode `a`/`b` insert a cell and enter edit mode (friendlier than
  Jupyter's stay-in-command-mode); revisit if surprising.

## Manual test checklist (for the user)

- [ ] `x` / `d d` deletes the selected cell; last cell can't be deleted.
- [ ] `a` / `b` insert a cell above / below and focus it.
- [ ] `m` / `y` convert the selected cell, keeping its text.
- [ ] "More options" (right toolbar) menu opens and each entry works.
- [ ] Output "..." menu: Copy Output copies text; Clear Output clears just
      that cell.
- [ ] Run-cells-above / run-cell-and-below (via the More options menu).

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
