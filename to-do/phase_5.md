# Phase 5 — Create Python environments from the kernel picker

Goal: VS Code-style "create new environment" from the kernel selector: create
a `.venv` in the project, install ipykernel into it, and select it — without
leaving Zed.

Primary files: `crates/repl/src/components/kernel_options.rs` (picker),
`crates/repl/src/repl_editor.rs` (`install_ipykernel_and_assign` — the
template for the create flow), `crates/repl/src/repl_store.rs`
(`refresh_python_kernelspecs`), `crates/languages/src/python.rs` (pet-based
discovery; internal venv creation example at `python.rs:1740-1786`).

## Tasks

- [ ] Add a "Create Python Environment…" entry to the kernel picker (footer
      or under the "Python Environments" section header).
- [ ] Flow: pick a base interpreter (from the pet-discovered global pythons),
      run `python -m venv .venv` in the worktree root (or `uv venv` when uv
      is available — detection exists, `kernels/mod.rs:237-243`), stream
      progress via toasts (same UX as `install_ipykernel_and_assign`).
- [ ] Chain into the existing ipykernel auto-install
      (`repl_editor.rs:78-202`) and then `assign_kernelspec` so the new env
      becomes the active kernel immediately.
- [ ] Refresh kernelspecs afterwards (`refresh_python_kernelspecs`) so the
      new env appears in the picker with correct labels/recommended state.
- [ ] Handle failure modes: no base python found, `.venv` already exists
      (offer to use it), venv creation error (surface stderr in the toast).
- [ ] Windows check: `Scripts/` vs `bin/` layout (see `BINARY_DIR` handling
      in `python.rs:1789-1793`).

## Notes

- Conda environment creation is deliberately excluded (backlog) — different
  tooling and slower creation; venv covers the primary ask.
- pet re-discovery already finds new `.venv` dirs; no locator changes needed.
