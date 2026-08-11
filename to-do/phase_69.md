# Phase 69 — Make notebooks behave like text files on disk

Kind: **mixed** — items 1 and 3 are changes to existing behaviour (if a loose
notebook still vanishes, or unsaved notebook edits still die on quit, the item
stays open), item 2 is a new affordance for notebooks.

Requested by the user 2026-08-11 after a session-restore discussion. Everything
here has ONE rationale: a notebook is a file-backed item like any other, and
three places where `Editor` handles a file's disk state, `NotebookEditor` does
not. The fix in each case is to do what the editor does — no new mechanism.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`, with `Editor`'s
equivalents in `crates/editor/src/items.rs` as the reference implementation.

## Deliberately OUT of scope (user decision, 2026-08-11)

**Do not change generic session-restore semantics.** The user's rule is to keep
the fork's changes inside notebook code as far as possible and let upstream own
the rest, so the following stay exactly as upstream wrote them even though the
user would prefer different behaviour:

- Restoring a path that no longer exists yields an EMPTY buffer with
  `DiskState::New`, not `Deleted` (`BufferStore::open_buffer`,
  `buffer_store.rs:665-682`). So a file deleted BETWEEN sessions comes back as a
  blank tab with no marker.
- A clean file whose path is gone is restored as that blank tab rather than
  dropped.

Consequence to accept knowingly: item 2 below makes the strikethrough appear for
a notebook deleted WHILE Zed is running (the live file-watch path), but NOT for
one deleted between sessions — because restore never sets `Deleted`. That is the
same behaviour `.md` has today, which is the point. Likewise a deleted saved
notebook restores as a blank one-cell notebook (empty content parses to one
cell), and saving it writes a blank notebook back to that path — exactly what
saving a phantom empty text buffer does. If upstream ever fixes the restore
side, all three items here sit on top of it unchanged, because they read the
same `DiskState`.

## Item 1 — A loose notebook must come back after a restart

`NotebookEditor::deserialize` requires `find_worktree(&abs_path)` to succeed and
returns an error otherwise (`notebook_ui.rs:6060`). A notebook opened from
outside every project root lives in an INVISIBLE single-file worktree, and only
VISIBLE worktrees are saved as the workspace's roots
(`Workspace::root_paths`, `workspace.rs:6975`) — so at restore time that
worktree does not exist, the deserialize errors, and `deserialize_to`'s
`log_err()` (`persistence/model.rs:388`) drops the tab silently.

`Editor` has the fallback already: when `find_worktree` returns `None` it calls
`project.open_local_buffer(&abs_path)` (`items.rs:1362-1390`), which does
`find_or_create_worktree(..., visible=false)` and recreates the invisible
worktree. Do the same, then hand the resulting `ProjectPath` to the existing
`NotebookItem::try_open` route so the notebook is opened and watched exactly as
a fresh open would be.

## Item 2 — The "deleted on disk" indicator

Deleting an open file on disk strikes its tab title through. It has never worked
for notebooks (`strikethrough` has never appeared in `notebook_ui.rs`), because
the indicator is NOT generic tab machinery — `TabContentParams`
(`workspace/item.rs:130`) carries no "deleted" flag and the default
`Item::tab_content` has no deleted handling, so every item type opts in itself.
(The image viewer is the other item that tracks the state and never renders it —
an upstream inconsistency, not a fork one. Out of scope here.)

The state is already correct and already reachable: `NotebookItem.buffer` holds
the worktree-tracked backing buffer, so `BufferStore::local_worktree_entry_changed`
(`buffer_store.rs:544-553`) flips its `File` to `DiskState::Deleted` on the
watch event. Nothing reads it. Four gaps, all in the notebook:

1. `tab_content` (`notebook_ui.rs:5757`) is a plain `Label` — needs
   `.strikethrough()` gated on the backing buffer's
   `file().disk_state().is_deleted()`, mirroring `Editor::tab_content`
   (`items.rs:776-782` and `:803`).
2. `has_deleted_file` is not overridden, so it defaults to `false`
   (`workspace/item.rs:304`) — which is also why the "This file has been deleted
   on disk" close prompt (`pane.rs:2243`) and the `close_on_file_delete` setting
   never fire for a notebook.
3. `to_item_events` is not implemented, so notebooks never emit
   `ItemEvent::UpdateTab`. The title still repaints on ordinary redraws (which is
   why the dirty dot works), but `workspace.update_item_dirty_state` and the
   auto-close path in `workspace/item.rs:914-947` never run.
4. `watch_backing_buffer` (`notebook_ui.rs:1533`) subscribes only to
   `BufferEvent::Reloaded`; deletion arrives as `BufferEvent::FileHandleChanged`
   (`buffer.rs:1706` → `:1734`).

## Item 3 — Hot exit for a saved notebook with unsaved changes

A file-backed notebook serializes only its PATH; contents are stored only when
it is untitled (`notebook_ui.rs:6007`). So quitting with unsaved notebook edits
loses them, where a text file would not. The editor's rule is simply different
and better:

| | `Editor` (`items.rs:1417-1487`) | `NotebookEditor` today |
|---|---|---|
| stores `contents` | whenever the buffer **is dirty**, saved or untitled | only when **untitled**, dirty or not |
| encode runs | cheap `snapshot()` on main, `snapshot.text()` in **background** | `to_notebook(cx)` AND `serde_json::to_string` both on **main** |
| stores `mtime` | always | never |
| `should_serialize` | 4 events (`Saved`/`DirtyChanged`/`BufferEdited`/`FileHandleChanged`) | `true` for every event |

Align all four. Notes that matter:

- **Move the JSON encode off the main thread.** `to_notebook(cx)` must stay on
  it (it reads entities), but `serde_json::to_string` must not: a notebook
  carrying base64 image outputs is megabytes, and item serialization is
  throttled to 200ms batches (`SERIALIZATION_THROTTLE_TIME`), i.e. up to ~5
  encodes/sec while typing. This is the ONE reason to be careful here; today it
  only bites untitled notebooks, which are rare and small.
- Gating on dirty makes notebooks do LESS work than today: a clean untitled
  notebook currently re-encodes itself on every event.
- **`(abs_path: None, contents: None)` currently ERRORS** ("empty serialized
  notebook", `notebook_ui.rs:6091`). Once contents are dirty-gated, an untitled
  notebook that was clean at quit hits that case and would lose its tab. The
  editor's third match arm creates a fresh empty buffer; do the same — a fresh
  untitled notebook.
- Storing `mtime` buys conflict detection for free (the editor's `did_reload`
  path, `items.rs:2182-2197`): "changed on disk while Zed was closed" is
  currently undetectable for notebooks.

## Tasks

- [ ] Give `NotebookEditor::deserialize` the loose-file fallback: when the path
      is in no worktree, create/find the invisible single-file worktree the way
      `project.open_local_buffer` does, then open through `try_open` as usual.
- [ ] Strike the tab title through when the backing buffer's file is deleted,
      matching `Editor::tab_content`.
- [ ] Override `has_deleted_file` so the close prompt and `close_on_file_delete`
      work for notebooks.
- [ ] Implement `to_item_events` for the notebook's `()` event so `UpdateTab` /
      `UpdateBreadcrumbs` fire, and check nothing double-fires now that the pane
      gets real events (the tab already repaints on redraw today).
- [ ] Subscribe `watch_backing_buffer` to `FileHandleChanged` as well as
      `Reloaded`, and notify so the tab restyles at the moment of deletion.
- [ ] Serialize notebook contents when the notebook is DIRTY rather than when it
      is untitled, store `mtime`, and narrow `should_serialize` to match the
      editor's event set.
- [ ] Do the `serde_json` encode on a background thread; only `to_notebook(cx)`
      stays on the main thread.
- [ ] Restore `(abs_path: None, contents: None)` as a fresh untitled notebook
      instead of erroring.
- [ ] Use the restored `mtime` to detect an external change the way the editor
      does, so reopening a notebook edited elsewhere while Zed was closed raises
      the existing conflict path rather than silently winning.
- [ ] Unit-test what can be tested headlessly: the dirty-gated serialize shape
      (dirty saved → contents + mtime; clean saved → path only; clean untitled →
      restores as a fresh notebook), and the deleted-tab predicate given a
      buffer whose file reports `DiskState::Deleted`.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] Open a notebook from OUTSIDE any project folder, quit, reopen: the tab
      comes back with the notebook in it (today it silently disappears).
- [ ] With a notebook open, delete the file in Explorer: the tab title goes
      struck through, the same as a `.md` does.
- [ ] With `close_on_file_delete` on, the same deletion closes the notebook tab
      if it has no unsaved changes; with unsaved changes it stays open and
      closing it prompts.
- [ ] Edit a saved notebook WITHOUT saving, quit Zed, reopen: the unsaved edits
      are still there and the notebook still shows as dirty.
- [ ] Edit a notebook in another program while Zed is closed, then reopen Zed
      with unsaved changes to that notebook: the existing external-change
      conflict handling fires rather than one side silently winning.
- [ ] A large notebook with image outputs stays responsive while typing — this
      is the one performance risk in the phase (the JSON encode now runs for
      every dirty notebook, not just untitled ones).
- [ ] Not regressed: untitled notebooks still restore with their cells, and an
      untitled notebook you never touched still comes back as an empty notebook
      rather than vanishing.
