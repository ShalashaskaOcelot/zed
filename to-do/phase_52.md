# Phase 52 — Notebook session persistence (restore open notebooks on relaunch)

Kind: **new feature**. Promoted from the backlog (user 2026-07-16) to restore
5 phases in rotation after phase 51 completed, and scheduled as the next work
item (user 2026-07-21).

Notebooks don't participate in Zed's workspace session restore, so:
1. SAVED notebooks that were open are NOT reopened when the workspace restores
   on relaunch (text/code buffers are).
2. UNSAVED / untitled notebooks (from "New Jupyter Notebook") are LOST on
   quit — and because notebooks don't participate in the session, closing the
   last window prompts to save the notebook instead of silently keeping it the
   way an unsaved Ctrl-N text buffer is kept.

Goal: `NotebookEditor` implements `workspace::SerializableItem` so both cases
behave like text buffers — saved notebooks reopen by path; untitled ones
persist their nbformat JSON in the workspace DB and reopen as untitled.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (the
`SerializableItem` impl), `crates/repl/src/repl.rs` or the repl `init`
(register the item), plus a small DB module for the serialized contents.

## Key facts (code inspection 2026-07-21)

- `workspace::SerializableItem` (`crates/workspace/src/item.rs:407`) needs:
  `serialized_item_kind()`, `serialize(&mut self, workspace, item_id, closing,
  …) -> Option<Task<Result<()>>>`, `deserialize(project, workspace,
  workspace_id, item_id, …) -> Task<Result<Entity<Self>>>`, `cleanup(workspace_id,
  alive_items, …)`, and `should_serialize(&self, event) -> bool`.
- Registered once at startup via `workspace::register_serializable_item::<T>(cx)`
  (see `crates/editor/src/editor.rs:357`). Do the same from the repl crate's
  `init` so notebooks are restored.
- The `Editor` impl (`crates/editor/src/items.rs`) is the closest model: it
  serializes by ABS PATH for saved buffers and stores contents for untitled
  ones in a sqlez DB (`editor/src/persistence.rs`). Mirror that split; reuse
  the existing notebook open-by-path route (the `.ipynb` path → `NotebookEditor`
  path opener already used by the file tree) in `deserialize`.

## Tasks

- [ ] Register `NotebookEditor` as a serializable item from the repl `init`
      (`register_serializable_item::<NotebookEditor>`), gated the same way the
      notebook feature already is.
- [ ] Add a `repl`-crate persistence module (sqlez, modeled on
      `editor/src/persistence.rs`): a table keyed by `(workspace_id, item_id)`
      storing either an abs path (saved) or the serialized nbformat JSON
      (untitled), plus save/load/delete queries.
- [ ] Implement `serialize`: for a saved notebook, store its abs path; for an
      untitled one, store its current nbformat JSON (the same bytes the save
      path writes). Honor `closing`. `should_serialize` fires on the events
      that change identity/content (path set on save, structural/exec edits) —
      keep it cheap.
- [ ] Implement `deserialize`: saved → reopen by path via the existing
      `.ipynb` path opener (don't duplicate load logic); untitled → rebuild a
      `NotebookEditor` from the stored nbformat JSON as an untitled item.
- [ ] Implement `cleanup` to drop rows for items no longer alive (mirror the
      editor's cleanup), so the DB doesn't grow unbounded.
- [ ] Make an untitled notebook behave like an unsaved buffer on last-window
      close: it should be kept in the session (serialized) rather than forcing
      a save prompt. Verify the prompt-on-close path keys off session
      participation now that notebooks participate.

## Risks / gaps

- Untitled-notebook serialization must round-trip nbformat exactly (outputs,
  metadata, exec counts) — reuse the save serializer, don't hand-roll JSON.
- Kernel state is NOT persisted (kernels are process-backed): a restored
  notebook comes back with no running kernel, lazily selecting the remembered
  spec on first run (phase 6 behavior). Call this out; don't try to revive
  kernels.
- Deserialize runs early in workspace restore — the project/worktrees may not
  be fully ready; match how the editor defers/loads by path to avoid races.
- DB migrations: add the table with a versioned migration so existing users
  upgrade cleanly.

## Verification

- `cargo clippy -p repl` clean; `cargo test -p repl` passes; add a
  round-trip unit test for the untitled-notebook serialize→deserialize path if
  feasible without a full workspace harness.
- User test: open a saved notebook + an untitled one, quit and relaunch → both
  reopen (saved by path with content; untitled with its cells intact), no
  save-prompt on quit for the untitled one.
