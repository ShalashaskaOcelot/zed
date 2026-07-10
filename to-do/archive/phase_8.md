# Phase 8 — Cell-operation undo/redo (COMPLETE, archived 2026-07-09)

> ✅ STATUS: CONFIRMED by the user 2026-07-09 ("So undo now works").
> Kind: **new feature** — archived per the new-feature rule (confirmed present
> and working). Any later defect in a specific op (redo, move, convert) is a
> new `bugs.md` entry, not a reopen of this phase.
>
> Root cause of the original "undo not working" report: undo was bound only to
> `z` (Jupyter-style) but the user pressed `ctrl-z`. Fixed by adding
> `ctrl-z` / `ctrl-shift-z` bindings; `z` / `shift-z` retained.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/zed_actions/src/lib.rs`, `assets/keymaps/*`.

## Implemented

- [x] `CellEdit` enum (Inserted / Deleted / Moved / Converted) + `undo_stack`
      / `redo_stack`.
- [x] Structural mutations record onto the undo stack (add/paste/duplicate →
      Inserted; delete → Deleted with LIVE content + outputs; move → Moved;
      convert → Converted). New op clears the redo stack.
- [x] `UndoCellOp` / `RedoCellOp` rebuild cells from serialized nbformat via
      raw primitives (fresh entities + wiring, preserving ids).
- [x] Keybinds `z` / `shift-z` (command mode) + `ctrl-z` / `ctrl-shift-z`;
      also in the More options menu.
- [x] Removed leftover `println!` debug lines from move_cell_up/down.

## Scope notes

- Structural ops only; in-cell text editing is the cell editor's own undo.

## Manual test checklist (for the user)

- [ ] Delete a cell → undo restores it (same position, content, outputs).
- [ ] Move a cell → undo moves it back; redo re-applies.
- [ ] Convert a cell → undo restores the original type + text.
- [ ] Paste/duplicate → undo removes the added cell.
- [ ] A new structural op after undo clears the redo stack.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean, tests pass.
