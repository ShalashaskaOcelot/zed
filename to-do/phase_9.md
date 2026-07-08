# Phase 9 — External file sync (watch & reload)

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (2026-07-08). Compiles,
> clippy-clean, unit tests pass. Runtime behaviour NOT yet confirmed. Do NOT
> archive until the user confirms.
>
> Kind: **new feature** — on confirmation that external edits reload (and
> unsaved changes are protected), archive; issues become new items.
>
> ## Implementation summary
> - `NotebookItem` now RETAINS the project `Buffer` for the .ipynb (previously
>   opened then dropped), so the project keeps watching the file and emits
>   `BufferEvent::Reloaded` when it changes on disk.
> - `NotebookEditor::new` subscribes to that buffer; on `Reloaded` it calls
>   `handle_external_change`.
> - `handle_external_change`: if the notebook has unsaved changes (`is_dirty`),
>   it keeps them and shows a toast (conflict); otherwise it rebuilds the cells
>   from the new content. It skips the rebuild when the disk content already
>   matches ours (e.g. our own save) by comparing serialized JSON.
> - `reload_cells_from_notebook` rebuilds cell_order/cell_map, WIRES cell
>   subscriptions (run/focus/cursor-follow) via `wire_*`, resets execution/queue
>   state, and refreshes language. `Item::reload` now uses it too (previously it
>   rebuilt cells WITHOUT wiring — a latent bug, now fixed).
> - Shared `parse_notebook_text` used by open, reload, and external-change.
>
> ## Manual test checklist (for the user)
> - [ ] Edit the .ipynb externally (or `git checkout`) with NO local changes →
>       the open notebook updates to the new content.
> - [ ] With unsaved changes in the notebook, an external change shows a toast
>       and keeps your changes (doesn't clobber them).
> - [ ] Saving from Zed does NOT cause a spurious reload/selection reset.
> - [ ] After a reload, run/focus/typing still work in the rebuilt cells.

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
