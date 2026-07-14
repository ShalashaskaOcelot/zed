# Phase 24 — Cell operations polish

Kind: **mixed** (small features + behaviour changes). Not yet started — this
is a plan. Bundles three related backlog items.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/zed_actions/src/lib.rs`, `assets/keymaps/*`.

## Tasks

- [x] Paste cell ABOVE: new `PasteCellAbove` action inserting the clipboard
      cell(s) above the primary cell (multi-aware like phase 22's paste, via the
      shared `paste_cells_at`). Keybind `ctrl-shift-v` / `cmd-shift-v` in
      command mode, plus a "Paste Cell Above" More-options menu entry.
- [x] Deleting the LAST remaining cell: instead of refusing, replace it with a
      fresh empty code cell (delete + insert as one undo Group), so delete
      always "does something". Applies to single delete, multi delete of all
      cells, and cut (cut routes through `delete_cell`).
- [ ] Smart arrows in EDIT mode: bind up/down in the cell-editor context to
      `NotebookMoveUp`/`NotebookMoveDown` (handlers already exist) so arrowing
      past the first/last line of a cell moves into the previous/next cell,
      Jupyter/VS Code style. Must NOT break completion-menu navigation — the
      editor's completion menu context must win; verify context precedence at
      runtime (deferred from phase 3 for exactly this risk).

## Risks / gaps

- The up/down binding in `NotebookEditor > NotebookCellEditor > Editor` may
  shadow completion-menu selection if precedence is wrong — test with an open
  completions dropdown; if it misbehaves, gate on `!menu` context or drop the
  binding.
- Last-cell replacement must leave selection/mode sane (select the fresh cell,
  command mode).

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: paste-above lands above; deleting the only cell leaves one fresh
  empty cell (undo restores the original); edit-mode arrows cross cell
  boundaries without breaking completions.
