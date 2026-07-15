# Phase 33 — Truly unsaved "New Jupyter Notebook"

Kind: **change to existing behaviour**. Not yet started — this is a plan.
Promoted from the backlog (user 2026-07-11, re-raised) to keep 5 phases in
rotation after phase 28 completed. The backlog flagged this as "bigger
plumbing — schedule as its own phase"; this is that phase.

Currently the "New Jupyter Notebook" command writes `Untitled-N.ipynb` into
the workspace immediately and opens that. It should behave like Ctrl-N: an
untitled, session-only notebook that only hits disk on manual save (with a
save-as flow on first save).

Scope guard: this applies ONLY to the command-palette command. Notebooks
created via the file browser's New File are correctly saved where they're
created, with the given name, and must stay that way.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (`NotebookItem` open
path, `NewNotebook` handler, `Item::save`/`save_as`/`can_save_as`).

## Tasks

- [ ] Support a path-less `NotebookItem`: the open path currently requires a
      `ProjectEntryId` and a saved `.ipynb`; allow constructing the editor
      from an in-memory nbformat template with no backing file (no disk write,
      no file watcher).
- [ ] `NewNotebook` opens such an untitled notebook (tab shows "Untitled",
      dirty from first edit) instead of writing `Untitled-N.ipynb`.
- [ ] First save routes through save-as (path picker); after that the notebook
      behaves like any opened one (watcher, conflict guard, kernel metadata).
- [ ] File-browser-created notebooks keep today's behaviour untouched.

## Risks / gaps

- The external-change watcher, save-conflict guard (bug #14/#21 logic), and
  kernel metadata adoption all assume a backing buffer/path — they must
  no-op cleanly for an untitled notebook and attach after the first save.
- Workspace serialization/reopen of an untitled notebook: decide whether it
  restores (like untitled buffers) or is simply dropped; dropping is
  acceptable for v1 if restoring is heavy.

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: command palette → New Jupyter Notebook → no file appears on
  disk; edit + save → save-as dialog → file lands where chosen; file-browser
  New File `.ipynb` still saves in place immediately.
