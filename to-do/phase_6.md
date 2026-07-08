# Phase 6 — Kernel selection persistence and lazy start

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (2026-07-08). Compiles,
> clippy-clean, unit tests pass. Runtime behaviour NOT yet confirmed. Do NOT
> archive until the user confirms. Ask periodically.
>
> Kind: **change to existing behaviour** — persistence and lazy start modify
> how kernel selection/startup already works. Keep each item OPEN until the
> user confirms the new behaviour actually takes effect; if it still behaves
> the old way, leave the item open and fix in place (do not re-file elsewhere).

Goal: stop forgetting the user's kernel choice, and stop auto-starting a
(usually wrong) global kernel on open. Match VS Code: no kernel runs until one
is explicitly chosen or a cell is run, and the choice sticks across reopen.

User report (2026-07-08): selected a project `.venv`, used it, closed the file,
reopened it — defaulted to the global Python. Also wants no auto-start on open.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/repl_store.rs`.

## Tasks (implemented — each ⚠ = needs user confirmation)

### Persistence

- [x] `change_kernel` now calls `ReplStore::set_active_kernelspec`, so an
      explicit selection is remembered for the worktree (fixes same-session
      reopen — the exact reported case). ⚠ untested
- [x] On open, `remembered_kernel_spec` resolves in order: explicit
      in-session selection → worktree selection (`selected_kernel`) → a spec
      matching the notebook's saved `metadata.kernelspec.name`. It does NOT
      fall back to the recommended/global kernel. ⚠ untested
- [x] Cross-session persistence: `launch_kernel_with_spec` already writes the
      chosen kernel into the notebook metadata, which is saved to the .ipynb
      on save and picked up by the metadata match above on next open.
      ⚠ untested (requires saving the notebook)

### Lazy start

- [x] Removed the auto-launch from `NotebookEditor::new`; the kernel stays
      `Shutdown` on open and the status bar shows the remembered kernel's name
      (or "Select Kernel"). ⚠ untested
- [x] First run: `execute_cell` on a `Shutdown`/`ErroredLaunch` kernel queues
      the cell and calls `launch_kernel`, which launches the remembered kernel
      if there is one, otherwise opens the kernel picker
      (`kernel_picker_handle.show`). The queued cell runs once a kernel is
      picked and ready (reuses phase-2 queueing). ⚠ untested

### Tests

- [x] Updated `test_run_cell_with_missing_interpreter_shows_error` for lazy
      start: asserts the kernel is `Shutdown` on open, then that running a cell
      launches the remembered (broken) kernel and surfaces the launch error.

## Manual test checklist (for the user)

- [ ] Open a notebook: no kernel starts; status bar shows the remembered
      kernel name (or "Select Kernel" if none).
- [ ] Pick `.venv`, run a cell, close the file, reopen it (same session):
      still uses `.venv`, not global Python.
- [ ] Save after selecting `.venv`, restart Zed, reopen: still `.venv`
      (cross-session, via saved metadata).
- [ ] Open a notebook with no remembered kernel and run a cell: the kernel
      picker opens; after picking, the cell runs.

## Known limitations / follow-ups (candidate backlog)

- If a cell is run with no remembered kernel, the picker opens but the cell
  already shows a running spinner while it waits; if the user dismisses the
  picker the cell stays queued. Consider a distinct "waiting for kernel"
  state.
- The remembered name is resolved once in `new()`; if kernelspecs are still
  loading at open time, a metadata-only match may miss and the bar shows
  "Select Kernel" until first run. Consider re-resolving after
  `refresh_kernelspecs` completes.
- Selection persistence is per-worktree in `ReplStore` (session-scoped) plus
  per-file via notebook metadata. Two notebooks in one worktree share the
  session selection until each is opened/selected; metadata makes it per-file
  across sessions.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
