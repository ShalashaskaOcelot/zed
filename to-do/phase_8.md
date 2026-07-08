# Phase 8 — Cell-operation undo/redo

> STATUS: PLANNED — not started. High priority: cell mutations (delete, move,
> convert, paste) are currently irreversible, which makes delete in particular
> risky. This is the safety net for phases 4 and 7.
>
> Kind: **new feature.**

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
