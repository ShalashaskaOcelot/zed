# Phase 7 — Cell clipboard operations (copy / cut / paste / duplicate)

> STATUS: PLANNED — not started. High priority: these are everyday notebook
> operations and a natural safety companion to phase 4's delete/insert.
>
> Kind: **new feature** (adds operations that don't exist), with ONE small
> **change** item: reconciling the `x` keybinding with Jupyter's convention.

Goal: standard cell clipboard editing, matching Jupyter/VS Code conventions,
built on the phase-4 insert/delete/build helpers.

Primary files: `crates/zed_actions/src/lib.rs`,
`crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/notebook/cell.rs`,
`assets/keymaps/default-{linux,macos,windows}.json`.

## Tasks

- [ ] New actions: `CopyCell`, `CutCell`, `PasteCell`, `DuplicateCell`.
- [ ] Clipboard format: serialize the cell(s) to nbformat JSON and put them on
      the system clipboard (via `ClipboardItem`), so paste works within a
      notebook and across notebooks/windows. Keep a lightweight in-memory
      fallback if clipboard round-tripping proves fiddly. Preserve cell type,
      source, and (for copy of an executed cell) drop outputs on paste to
      avoid stale results.
- [ ] Paste position: paste BELOW the selected cell by default; consider
      `shift`-paste for above (Jupyter uses `v` / `V`). Reuse `insert_cell` and
      the `build_code_cell` / `build_markdown_cell` helpers.
- [ ] Duplicate = copy + paste-below of the selected cell in one action.
- [ ] Cut = copy + delete (respect the "never delete the only cell" guard).
- [ ] Nav-mode keybindings (`NotebookEditor && notebook_mode == command`),
      Jupyter-style, in all three keymaps: `c` copy, `x` cut, `v` paste below,
      `shift-v` paste above, `d` duplicate (or reuse a sensible key).
- [ ] **Change item — reconcile `x`:** phase 4 bound `x` to DeleteCell to
      match one Jupyter convention, but Jupyter's `x` is actually *cut*
      (delete stays on `d d`). Rebind `x` → `CutCell` and keep `d d` →
      `DeleteCell`. (This is the one "change to existing behaviour" item here;
      keep it open until confirmed.)
- [ ] Command-palette entries come for free via the `actions!` registration.

## Risks / notes

- Clipboard serialization needs a stable nbformat cell representation; the
  notebook already round-trips cells via `to_nbformat_cell` / `Cell::load`,
  which is the natural basis.
- Undo/redo for these operations is phase 8; until then, cut/paste are
  irreversible (same caveat as delete). Note this to the user.

## Manual test checklist (for the user)

- [ ] Copy a cell, paste it below; content and type preserved, outputs not
      duplicated.
- [ ] Cut a cell (removed + on clipboard), paste it elsewhere.
- [ ] Duplicate a cell.
- [ ] Paste into a different notebook / window.
- [ ] `x` now cuts (not deletes); `d d` still deletes.
