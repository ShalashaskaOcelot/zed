# Phase 66 — Notebook file surfaces: save-as and global search

Kind: **mixed** — one defect fix (search opens raw JSON) and one change to
existing behaviour (save-as path handling). Promoted from the backlog
2026-08-06 to restore the runway after phases 63 and 64 completed.

Goal: the two places where a notebook meets the workspace's generic file
machinery stop treating it as a text file — saving one produces an `.ipynb`,
and opening one from a global-search hit opens the NOTEBOOK.

## Item 1 — Save-as gives you a real `.ipynb`

Reported 2026-07-16. Two separate sub-problems; only the first is cheaply in
scope, so decide the second explicitly rather than letting it drag the phase.

- **Extension (in scope).** `Item::suggested_filename` already returns
  `Untitled.ipynb` for notebooks (`notebook_ui.rs:5603`, phase 33), so the
  dialog is pre-filled correctly — but a user who types a bare name gets a
  file with no extension, and that file no longer opens as a notebook. The
  path comes back through `Pane::save_item` → `NotebookEditor::save_as`
  (`notebook_ui.rs:5717`), which writes wherever it is told. Appending
  `.ipynb` when the chosen path has no extension (or a non-`.ipynb` one —
  decide which, and say why in a comment) belongs there, where the notebook's
  own knowledge lives.
- **File-type filter (investigate, then decide).** The dialog's filter can't
  be set from here: `Workspace::prompt_for_new_path`
  (`workspace.rs:3040`) calls `cx.prompt_for_new_path(&relative_to,
  suggested_name)`, and gpui's platform trait
  (`gpui/src/platform.rs:180`) takes NO filter argument on any platform. Adding
  one is a cross-platform gpui change in a file upstream actively develops —
  real merge-conflict surface for a cosmetic win. Decide: either do it properly
  (all three platforms + the fallback picker) or backlog it explicitly. Do NOT
  half-do it for one platform.

## Item 2 — A global-search hit opens the notebook, not raw JSON

Reported 2026-07-16 (DEFECT). `Ctrl-Shift-F` does search notebook content, but
clicking a result opens the `.ipynb` as raw JSON text.

Root cause (research 2026-07-16, re-verified 2026-08-06): a search excerpt is
opened by `Editor::open_buffers_in_workspace`
(`crates/editor/src/editor.rs:10055`), which ends in
`workspace.open_project_item::<Self>(…)` — the TYPE registry, hardcoded to the
text `Editor`. The PATH registry that maps `.ipynb` → `NotebookEditor` (what
the file tree uses via `open_path`) is never consulted.

- Quick win: when the buffer's file has a non-`Editor` path opener registered,
  route the open through `workspace.open_path` instead. That opens the real
  notebook and loses the jump to the matching line — acceptable for a first
  cut, and far better than JSON.
- **Constraint:** this must be done GENERICALLY. `editor` cannot depend on
  `repl`; the check has to ask the workspace whether a path opener exists,
  not "is this an ipynb".
- Out of scope (separate backlog items if wanted): jumping to the matching
  CELL, and making the search preview show cell content rather than JSON.

## Tasks

- [ ] Append `.ipynb` in `NotebookEditor::save_as` when the chosen path lacks
      it, so a bare typed name still yields a notebook; cover it with a test
      alongside `test_untitled_notebook_saves_via_save_as`
      (`notebook_ui.rs:6407`).
- [ ] Confirm the saved-with-appended-extension notebook re-opens as a
      notebook (tab title, kernel memory, external-change watch all keyed off
      the real path).
- [ ] Investigate the dialog filter as scoped above and either implement it
      across all platforms or move it to the backlog with the finding written
      down. Record the decision in this file.
- [ ] Add a generic "does this path have a non-Editor opener?" query to the
      workspace and use it in `open_buffers_in_workspace` to route through
      `open_path`.
- [ ] Make sure ordinary text results are completely unaffected — same code
      path, same excerpt jump, no extra work per result.
- [ ] `./script/clippy` clean; `cargo test -p repl -p editor -p workspace`
      passes.

## User tests (runtime)

- [ ] Save-as an untitled notebook and type a name with NO extension: the file
      is written as `<name>.ipynb`, the tab shows it, and reopening it from the
      file tree gives a notebook (not JSON).
- [ ] Ctrl-Shift-F for text that lives in a notebook cell, click the result:
      the notebook opens in the notebook editor.
- [ ] Ctrl-Shift-F for text in a normal file: unchanged — opens the editor at
      the matching line, including the split (`ctrl-enter`) variant.
- [ ] A notebook already open in a tab doesn't get a second tab from a search
      hit.
