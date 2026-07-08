# Phase 4 — Cell actions, Jupyter shortcuts, and the two dead menus (COMPLETE, archived 2026-07-08)

Goal: bring the per-cell action set up to Jupyter/VS Code conventions and give
the two dead ellipsis buttons real menus.

Primary files: `crates/zed_actions/src/lib.rs`,
`crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/notebook/cell.rs`,
`assets/keymaps/default-{linux,macos,windows}.json`.

## Tasks

### New actions + handlers

- [x] `DeleteCell` — removes the selected cell; refuses to delete the only
      remaining cell; selects a neighbour and returns to command mode; clears
      any pending/queued execution state for the removed cell.
- [x] `AddCellAbove` / `AddCellBelow` — generalised insertion
      (`insert_cell` + `index_below_selection`); above inserts at the selected
      index, below reuses the existing add-code path.
- [x] `RunCellsAbove` — runs all cells above the selected cell.
- [x] `RunCellAndBelow` — runs the selected cell and everything below.
- [x] `ConvertToCode` / `ConvertToMarkdown` — rebuilds the cell as the other
      type in place, preserving the current editor text.

### Nav-mode keybindings (context `NotebookEditor && notebook_mode == command`)

- [x] `a` → add cell above, `b` → add cell below (code cells).
- [x] `x` and `d d` (double-tap) → delete cell.
- [x] `m` → convert to markdown, `y` → convert to code.
      Added to all three keymaps (linux/macos/windows).

### Menus

- [x] Right-toolbar "More options" (`notebook_ui.rs`) → `PopoverMenu` +
      `ContextMenu`: run cells above / run cell and below, add above/below,
      move up/down, convert type, clear all outputs, delete cell. Entries
      dispatch the actions (so they show keybindings and reuse handlers).
- [x] Output "..." button (`cell.rs`) → per-cell output menu: Copy Output
      (concatenated text of stdout/plain/error outputs via new
      `CodeCell::outputs_as_text`) and Clear Output (this cell only).

### Cell hover controls

- [ ] Moved to `backlog.md` — the actions, keybinds, and both ellipsis menus
      already expose this functionality; the VS Code-style per-cell hover
      toolbar is a convenience layer for a later pass.

## Notes / findings

- Refactored the duplicated cell-subscription wiring into `wire_code_cell` /
  `wire_markdown_cell` and `build_code_cell` / `build_markdown_cell` so
  delete/convert reuse the same setup as add.
- Dirty tracking: the workspace `Item::is_dirty` for `NotebookEditor`
  (`notebook_ui.rs:2200`) already returns
  `has_structural_changes() || has_content_changes()`, so all the new cell
  mutations mark the tab dirty and prompt on close — bug #9 only concerns the
  separate `NotebookItem::is_dirty` stub (left open, note added to bugs.md).
- Nav-mode `a`/`b` insert a cell and enter edit mode (friendlier than
  Jupyter's stay-in-command-mode); revisit if it feels surprising.

## Verification

- `cargo check -p repl` clean, `./script/clippy -p repl` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
- Runtime behaviour (menus, keybinds, delete/convert) needs user confirmation.
