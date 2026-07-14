# Phase 25 — Kernel selection quality-of-life

Kind: **mixed**. Not yet started — this is a plan. Bundles two kernel-flow
backlog items.

Primary files: `crates/repl/src/repl_store.rs`,
`crates/repl/src/notebook/notebook_ui.rs`.

## Tasks

- [ ] Persist the per-notebook/worktree kernel choice across FULL Zed restarts
      (today it only survives within a session via `ReplStore`'s in-memory
      `selected_kernel_for_worktree`). Approach: persist the selection keyed by
      worktree (or notebook path) — either in the workspace's sqlite store
      (`db` crate, like other per-workspace state) or via the notebook's saved
      `kernelspec` metadata matched against discovered kernels on open (VS Code
      does the latter). Prefer the metadata match first (no new storage):
      on open, if `metadata.kernelspec.name` matches a discovered kernel,
      pre-select it; fall back to the session store.
- [ ] Auto-run the triggering cell(s) after a kernel is picked from the
      run-prompt: already works via `cells_awaiting_kernel_choice` — VERIFY it
      still does after the phase 17-22 changes and close the stale backlog
      item ("re-add the auto-run once the picker-dismiss lifecycle can be
      tracked cleanly" — the lifecycle IS now tracked cleanly). If anything is
      missing, fix it here.

## Risks / gaps

- Metadata-based matching must not override an explicit in-session selection.
- A stale metadata kernelspec (env deleted) must fall back gracefully to the
  picker rather than erroring.

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: select a kernel, quit Zed fully, reopen the notebook → the kernel
  shows as selected (and lazy-starts on first run) without re-picking.
