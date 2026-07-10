# Phase 8 — Cell-operation undo/redo

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (commit 7b41fbb). Compiles,
> clippy-clean, unit tests pass. Do NOT archive until the user confirms.
> Kind: **new feature**.
>
> User note (2026-07-08): undo reported "not working" (Ctrl-Z did nothing).
> Two contributing factors under investigation: (1) undo was bound to `z`
> (Jupyter) not `ctrl-z` — ctrl-z/ctrl-shift-z now added too; (2) the notebook
> can get stuck in Edit mode so command-mode `z` types into a cell instead of
> firing (see bug #13 / #15).

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
