# Phase 53 — Saving a file outside the workspace shouldn't add it to the workspace

Kind: **change to existing behaviour** (Zed-wide, user 2026-07-21). Added as a
6th phase at the user's explicit request. Applies to ALL file types, not just
notebooks — the user wants: a buffer saved to a path OUTSIDE the current
project stays open ad-hoc but does NOT get added to the project panel as a new
root. (If they wanted it in the workspace, they'd have saved it inside a
workspace folder.)

## Root cause (workflow investigation 2026-07-21, high confidence)

- The single save entry point `Pane::save_item` takes the save-as branch for any
  dirty singleton (`crates/workspace/src/pane.rs:2448`, generic — no notebook
  special-casing).
- Before calling the item's `save_as`, it resolves the picked path to a
  `ProjectPath` via `project.find_or_create_worktree(new_path, /*visible=*/true, cx)`
  (`pane.rs:2474`).
- `WorktreeStore::find_or_create_worktree` (`crates/project/src/worktree_store.rs:425-438`):
  if the path is inside an existing worktree it reuses it (no new root — the
  working in-workspace case); if it's OUTSIDE every worktree it calls
  `create_worktree(abs_path, /*visible=*/true)` → a NEW **visible** single-file
  worktree = a standalone root in the project panel (the bug).
- Contrast: OPENING an external file uses an INVISIBLE single-file worktree
  (that's why opening a random file doesn't add a panel root). Save-as is
  inconsistent in passing `visible=true`.

Note: this single-file-worktree-with-empty-relative-path is also the root of
bugs #47/#48 (already fixed in the notebook) and #49 (external deletes not
reflected). #49 (fs-watching of single-file worktrees) is tracked separately.

## Tasks

- [ ] Make save-as to a path outside all existing worktrees create the
      single-file worktree as **invisible** (so it doesn't become a project-panel
      root), matching how Zed opens external files. Likely: change the
      `visible` argument at `pane.rs:2474` — but do it CONDITIONALLY (only when
      the path is not inside an existing visible worktree), or push the decision
      into `find_or_create_worktree`, so that legitimately saving a new file
      INSIDE the project still works and stays visible.
- [ ] Verify the item stays fully functional after an external save: the buffer
      keeps its path, Ctrl-S re-saves without a dialog, and the tab shows the
      real name (for notebooks this now depends on the abs-path title fix,
      bug #47). Confirm an invisible worktree still supports save + external
      file-watching hookup that a visible one had.
- [ ] Confirm no regression to: saving a brand-new file INTO the project
      (should still appear), "save as" onto a path in another already-open
      worktree, and the multi-worktree case.

## Risks / gaps

- This is CORE, SHARED workspace code — a wrong change alters save-as for every
  file type. Test with both a plain text buffer AND a notebook, inside and
  outside the project.
- Some flows may rely on the external save becoming a visible root (e.g. a user
  who "saves as" specifically to start a new project folder). Decide whether
  invisible-by-default is right for all, or whether only truly out-of-tree
  single-FILE saves go invisible. Lean on how open-file already behaves.
- Upstreamability: this diverges from upstream Zed save-as; keep the change
  small and localized (one visibility decision) so merges stay clean, and note
  it in the upstream-merge playbook (phase 46) as a fork behavior change.
- An invisible worktree that later turns out to be wanted in the panel: Zed
  already promotes invisible worktrees to visible when appropriate (e.g. adding
  a folder) — verify that path still works.

## Verification

- `./script/clippy` clean; `cargo test -p workspace -p project` passes (or the
  scoped equivalents runnable in the env).
- User test: save a new buffer (text) and a notebook to the Desktop → each stays
  open and saveable but does NOT appear as a root in the project panel; saving a
  new file into a workspace folder DOES still appear.
