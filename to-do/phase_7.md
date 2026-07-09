# Phase 7 — Cell clipboard operations (copy / cut / paste / duplicate)

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (2026-07-08). Compiles,
> clippy-clean, unit tests pass. Do NOT archive until the user confirms.
>
> Kind: **new feature** (adds operations that don't exist), with ONE small
> **change** item: `x` rebound from DeleteCell to CutCell (keep OPEN until the
> user confirms the rebinding took effect).
>
> ## Implementation summary
> - Actions `CopyCell`, `CutCell`, `PasteCell`, `DuplicateCell`.
> - Clipboard format: the cell serialized as nbformat JSON via `ClipboardItem`
>   (works within a notebook and across notebooks/windows). Paste parses the
>   clipboard text as an `nbformat::v4::Cell`; non-cell text is ignored.
> - Copy → clipboard; Cut → copy + delete (respects the last-cell guard);
>   Paste → insert below with a FRESH cell id (via `insert_nbformat_cell`);
>   Duplicate → insert a copy of the selected cell below (does not touch the
>   clipboard). Pasted/duplicated code cells keep their source but get a new
>   id and are fully wired.
> - Nav-mode keybinds (command mode, all three keymaps): `c` copy, `x` cut,
>   `v` paste (below), `d d` delete (unchanged). Also in the "More options"
>   menu.
>
> ## Deferred (backlog)
> - Paste ABOVE (`shift-v`) — only paste-below is wired for now.
>
> ## Manual test checklist (for the user)
> - [ ] Copy a cell (`c`), paste below (`v`): content + type preserved,
>       outputs not duplicated, new cell is independent (new id).
> - [ ] Cut (`x`): cell removed and on clipboard; paste elsewhere.
> - [ ] `d d` still deletes (does not cut).
> - [ ] Duplicate (via More options menu).
> - [ ] Paste into a different notebook / window.

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
