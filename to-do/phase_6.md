# Phase 6 — Kernel selection persistence and lazy start

Goal: stop forgetting the user's kernel choice, and stop auto-starting a
(usually wrong) global kernel on open. Match VS Code: no kernel runs until one
is explicitly chosen or a cell is run, and the choice sticks across reopen.

User report (2026-07-08): selected a project `.venv`, used it, closed the
file (kernel still running), reopened the same file — it defaulted to the
global Python interpreter (missing the venv's packages) instead of the venv.
Also wants the kernel NOT to auto-start on open, but to prompt/start on first
run or explicit selection.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`
(`new`, `launch_kernel`, `change_kernel`, `execute_cell`),
`crates/repl/src/repl_store.rs` (`active_kernelspec`, `set_active_kernelspec`,
`selected_kernel_for_worktree`), notebook metadata handling
(`notebook_ui.rs:399-410`).

## Findings from discovery (why it forgets the venv)

- `ReplStore::active_kernelspec` returns `selected_kernel_for_worktree` first
  (`repl_store.rs:330`), but the notebook's `change_kernel`
  (`notebook_ui.rs:497`) never calls `set_active_kernelspec` — so picking a
  kernel in the notebook does NOT update the store. On reopen (same session)
  the store has no selection, so it falls back to the "recommended" =
  active toolchain = global Python.
- The notebook DOES write the chosen kernel into the .ipynb metadata
  (`notebook_ui.rs:399-410`), but `active_kernelspec` never consults notebook
  metadata on open, so even the saved-to-disk choice is ignored.
- `selected_kernel_for_worktree` is in-memory only (session-scoped) and keyed
  by worktree, not by notebook file — two notebooks in one worktree would
  share a selection.

## Tasks

### Persistence

- [ ] On explicit selection, persist the choice: `change_kernel` calls
      `ReplStore::set_active_kernelspec` (fixes same-session reopen).
- [ ] On open, prefer the notebook's saved `metadata.kernelspec` when it
      resolves to an available kernel spec, before falling back to the
      recommended/active-toolchain kernel (fixes cross-session reopen). Add a
      resolver (match saved kernelspec name/path/language against
      `kernel_specifications_for_worktree`).
- [ ] Decide persistence granularity: per-notebook is more correct than
      per-worktree. Options: (a) rely on notebook metadata as the source of
      truth per file; (b) key an in-memory map by ProjectPath. Prefer (a) for
      durability; use (b) only as a session cache. Document the decision.

### Lazy start

- [ ] Remove the auto-launch from `NotebookEditor::new` (drop the
      `editor.launch_kernel(...)` call); initial kernel state stays
      `Shutdown` and the status bar shows "Select Kernel"/idle-unstarted.
- [ ] First-run behaviour: if a kernel spec IS resolvable (saved metadata or
      prior selection), running a cell starts it (the phase-2 auto-start on
      run already handles `Shutdown`). If NO kernel is selected/resolvable,
      running a cell opens the kernel picker (via `kernel_picker_handle`) and
      queues the cell to run once a kernel is chosen and ready.
- [ ] Ensure the "no kernel yet" state reads clearly in the status bar and
      doesn't look like an error.

### Tests / verification

- [ ] Unit/integration: selecting a kernel then re-creating the editor for
      the same notebook resolves to that kernel, not the global default.
- [ ] Manual (user, Windows): pick `.venv`, run, close, reopen → still
      `.venv`; open a fresh notebook with no selection → no kernel starts
      until first run, which prompts the picker.

## Notes

- This supersedes the current unconditional default-to-`python3` fallback in
  `launch_kernel` (`notebook_ui.rs:345-364`) for the open path; keep a sane
  fallback only once a kernel is actually being launched.
- Interacts with phase-2 auto-start-on-run and queued executions — reuse that
  machinery rather than adding a parallel path.
