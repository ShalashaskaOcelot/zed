# Phase 52 — Notebook session persistence (restore open notebooks on relaunch)

⚠️ **CORE CONFIRMED WORKING** (user 2026-07-21): saved-by-path restore, untitled
restore (with outputs), and silent keep-on-close (no save prompt) all verified.
ONE task remains before archiving — harden the deserialize path so it can never
abort the whole session when a restored notebook's file/worktree is gone (see
the open task + bug #50). Kept open deliberately: a persistence feature isn't
"done" until its restore is proven unable to lose a session.

Testing also surfaced separate, pre-existing bugs around saving OUTSIDE the
workspace — NOT phase 52 (they reproduce with no restart): tab title
(bug #47, fixed), reopen-as-raw-JSON (bug #48, fixed), external-delete not
reflected (bug #49, open), session loss (bug #50, open), and the Zed-wide
"external save shouldn't join the workspace" behaviour change (phase 53).

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

- [x] Register `NotebookEditor` as a serializable item from the repl `init`
      (`register_serializable_item::<NotebookEditor>`), gated the same way the
      notebook feature already is.
- [x] Add a `repl`-crate persistence module (`notebook/persistence.rs`, sqlez
      `NotebookDb` modeled on `editor/src/persistence.rs`): a `notebook_editors`
      table keyed by `(item_id, workspace_id)` storing an abs path (saved) OR
      the serialized nbformat JSON (untitled), with get/save queries and
      `delete_unloaded_items` cleanup. Added `db` as a repl dependency.
- [x] Implement `serialize`: saved → store abs path; untitled (`path.is_none()`)
      → store `serde_json::to_string(to_notebook())`. Returns `None` when
      there's nothing to restore. `should_serialize` returns true (the item's
      only `Event` is `()`; the close-time serialize captures final state).
- [x] Implement `deserialize`: saved → `<NotebookItem as project::ProjectItem>
      ::try_open` by path then `NotebookEditor::new` (reuses the real load/watch
      path); untitled → `parse_notebook_text` → `NotebookItem::untitled` →
      `NotebookEditor::new`.
- [x] Implement `cleanup` via `delete_unloaded_items(.., "notebook_editors",
      &NotebookDb::global(cx), ..)`.
- [x] Make an untitled notebook behave like an unsaved buffer on last-window
      close (kept in the session, no save prompt). CONFIRMED 2026-07-21: the
      user's untitled notebook was silently kept and restored on relaunch (with
      outputs) — no explicit change was needed beyond participating in the
      session.
- [ ] Harden deserialize robustness so a restored notebook can NEVER abort the
      workspace session restore — a saved notebook whose file/worktree is gone
      must fail gracefully and be skipped, not take down the whole session.
      Confirm the workspace isolates a failed `SerializableItem::deserialize`
      per-item (coordinate with bug #50, the session-loss investigation); harden
      here if it does not. This is the only reason the phase is still open: the
      restore path must be proven safe before persistence is called done.

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

- [x] `cargo clippy -p repl` clean; compiles.
- [ ] ⚠ untested — User test (saved): open a saved `.ipynb`, quit and relaunch
      → it reopens with its content.
- [ ] ⚠ untested — User test (untitled): New Jupyter Notebook, add/edit cells
      (don't save), quit and relaunch → it reopens as untitled with the cells
      intact. Note whether quitting still shows a save-prompt for it (the open
      task) or keeps it silently like an unsaved buffer.
- [ ] (deferred) `cargo test -p repl` round-trip unit test for the
      untitled-notebook serialize→deserialize path, if feasible without a full
      workspace harness.
