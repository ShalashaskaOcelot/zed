# Phase 9 — External file sync (watch & reload)

> STATUS: PLANNED — not started. Medium-high priority: data-integrity issue —
> the notebook does not notice when its .ipynb changes on disk (git checkout,
> another editor, formatter), so it can silently overwrite external changes on
> save.
>
> Kind: **new feature.**

Goal: detect external changes to the open .ipynb and reload (or prompt),
matching how Zed's text editors handle on-disk changes.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`
(`NotebookItem::try_open` has `// todo: watch for changes to the file` at the
buffer open, and `is_dirty`/reload plumbing), plus how other Zed items observe
buffer/file events.

## Tasks

- [ ] Observe the underlying buffer / project file for on-disk changes (Zed
      buffers already emit reload/conflict events; the notebook opens a buffer
      in `try_open` — subscribe to it or to the project's file events).
- [ ] On external change with NO local unsaved changes: reload the notebook
      (rebuild `cell_order`/`cell_map` from the new content, preserving
      selection where possible).
- [ ] On external change WITH local unsaved changes (conflict): surface it
      rather than silently clobbering — mark the item conflicted / prompt,
      consistent with the text-editor conflict UX.
- [ ] Confirm `is_dirty` / conflict state is reported to the workspace so the
      save flow warns before overwriting (ties into bugs.md #9 — the
      `NotebookItem::is_dirty` stub).

## Risks / notes

- Rebuilding the notebook view from new content must re-wire all cell
  subscriptions (reuse the `new()` / `wire_*` paths) and reset kernel routing
  state carefully.
- Scope this to detect + reload/prompt; full 3-way merge is out of scope.

## Manual test checklist (for the user)

- [ ] Edit the .ipynb externally (or `git checkout`) with no local changes →
      the open notebook reflects the new content.
- [ ] With local unsaved changes, an external change is flagged rather than
      lost.
