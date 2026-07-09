# Phase 8 — Cell-operation undo/redo

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (2026-07-08). Compiles,
> clippy-clean, unit tests pass. Do NOT archive until the user confirms.
>
> Kind: **new feature.**
>
> ## Implementation summary
> - `CellEdit` enum (Inserted / Deleted / Moved / Converted) with `undo_stack`
>   and `redo_stack` on `NotebookEditor`.
> - Structural mutations record onto the undo stack (and clear redo): add
>   (code/markdown/above/below), paste, duplicate → `Inserted`; delete →
>   `Deleted` (captures the cell's LIVE content + outputs so undo restores
>   them); move up/down → `Moved`; convert → `Converted` (before/after).
> - `UndoCellOp` / `RedoCellOp` apply the inverse/forward via raw primitives
>   (`raw_insert_cell` / `raw_remove_cell` / `raw_move_cell` /
>   `raw_replace_cell`) that rebuild cells from serialized nbformat (fresh
>   entities + wiring, never resurrected). Keybinds: `z` undo, `shift-z` redo
>   (command mode, all keymaps); also in the More options menu.
> - Restored cells preserve their original id so undo/redo round-trip cleanly.
> - Removed leftover `println!` debug lines from move_cell_up/down along the way.
>
> ## Scope notes
> - Covers STRUCTURAL ops only; in-cell text editing is undone by the cell's
>   own editor. Undoing an add/paste restores the cell as it was at
>   creation/paste time (subsequent text edits are the editor's own history).
>
> ## Manual test checklist (for the user)
> - [ ] Delete a cell → `z` restores it (same position, content, outputs).
> - [ ] Move a cell → `z` moves it back; `shift-z` redoes.
> - [ ] Convert a cell → `z` restores the original type + text.
> - [ ] Paste/duplicate → `z` removes the added cell.
> - [ ] A new structural op after undo clears the redo stack.

Goal: undo/redo for structural cell operations, so deleting/moving/converting/
pasting a cell can be reverted. Text edits *within* a cell already undo via the
cell's own editor; this phase covers the notebook-level structure.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`
(`cell_order`, `cell_map`, `insert_cell`, `delete_cell`, `move_cell_*`,
`convert_selected_cell`), `crates/zed_actions/src/lib.rs`,
`assets/keymaps/default-{linux,macos,windows}.json`.

## Design sketch

- Maintain an undo stack and redo stack of structural operations. Each entry
  captures enough to reverse the op:
  - Delete → (index, serialized cell) to reinsert.
  - Insert/Paste/Duplicate → (index) to remove.
  - Move → (from, to) to swap back.
  - Convert → (index, previous cell type + source) to rebuild.
- Prefer storing serialized nbformat cells + a rebuild path (reuse
  `to_nbformat_cell` / `Cell::load` / the `build_*` helpers) rather than
  holding onto dead cell entities, so restored cells get fresh, correctly
  wired subscriptions.
- Clear the redo stack on any new structural op. Decide whether text edits
  interleave with structural undo (likely keep them separate — structural
  undo only touches structure).

## Tasks

- [ ] Introduce a `NotebookEdit` op enum + undo/redo stacks on `NotebookEditor`.
- [ ] Record ops from `delete_cell`, `insert_cell` callers (add/paste/dup),
      `move_cell_up`/`move_cell_down`, `convert_selected_cell`.
- [ ] `UndoCellOp` / `RedoCellOp` actions + handlers that apply the inverse and
      restore selection/scroll sensibly.
- [ ] Keybindings in command mode: `z` undo, `shift-z` (or `y`) redo — note
      `y` is already convert-to-code, so use `shift-z` for redo. All three
      keymaps.
- [ ] Ensure restored cells are fully re-wired (subscriptions) and the
      `ListState` is spliced correctly.
- [ ] Interaction with dirty tracking: undoing back to the saved state ideally
      clears dirty (nice-to-have; `original_cell_order` comparison already
      drives structural dirtiness).

## Risks / notes

- The main risk is entity lifecycle: don't resurrect dropped cell entities;
  rebuild from serialized form so subscriptions and language wiring are fresh.
- Keep scope to structural ops; do NOT try to unify with per-editor text undo.

## Manual test checklist (for the user)

- [ ] Delete a cell, undo → cell returns at the same position with content.
- [ ] Move a cell, undo → returns to original position.
- [ ] Convert a cell, undo → original type + text restored.
- [ ] Paste/duplicate, undo → the added cell is removed.
- [ ] Redo re-applies; a new op clears the redo stack.
