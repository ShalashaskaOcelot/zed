# Phase 5 — Create Python environments from the kernel picker

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (2026-07-08). Compiles,
> clippy-clean, unit tests pass. Runtime behaviour NOT yet confirmed (needs a
> real Python on PATH). Do NOT archive until the user confirms.
>
> Kind: **new feature** — on confirmation that it creates a venv, installs
> ipykernel, and selects it, archive; any tweak/defect becomes a new item.

Goal: VS Code-style "create new environment" from the kernel selector: create
a `.venv` in the project, install ipykernel into it, and select it — without
leaving Zed.

Primary files: `crates/repl/src/components/kernel_options.rs` (picker footer +
`on_create_env` callback), `crates/repl/src/notebook/notebook_ui.rs`
(`create_python_environment`), `crates/repl/src/kernels/mod.rs`
(`PythonEnvKernelSpecification::from_python_path`).

## Tasks (implemented — ⚠ = needs user confirmation)

- [x] "Create Python Environment" button in the kernel picker footer (next to
      "Kernel Docs"), wired via a new `on_create_env` callback on
      `KernelSelector`/`KernelPickerDelegate`. ⚠ untested
- [x] Flow: `create_python_environment` runs `python3 -m venv .venv` (falls
      back to `python`) in the worktree root, then installs ipykernel into it,
      streaming progress via a workspace toast (same UX as
      `install_ipykernel_and_assign`). ⚠ untested
- [x] Selects the new env immediately via `change_kernel` using
      `PythonEnvKernelSpecification::from_python_path` (sets PATH + VIRTUAL_ENV
      like the pet-discovered specs). ⚠ untested
- [x] Refreshes kernelspecs afterward so the env also appears in the picker. ⚠ untested
- [x] Reuses an existing `.venv` if present (skips creation, still installs
      ipykernel + selects). ⚠ untested
- [x] Windows `Scripts/python.exe` vs `bin/python` handled via `cfg!(windows)`. ⚠ untested
- [x] Failure toasts: no project folder, no base Python on PATH, venv/pip
      errors (surface last stderr line). ⚠ untested

## Known limitations / follow-ups (candidate backlog)

- Base interpreter is chosen automatically (`python3` then `python` on PATH).
  No UI to pick a specific base interpreter yet — add a base-interpreter
  sub-picker if users need a non-default Python.
- Always targets `.venv` in the worktree root; no custom name/location.
- `uv venv` is not used even when uv is available (the notebook path uses
  `python -m venv` + `pip`); wire uv for speed as a follow-up.
- Conda creation remains out of scope (backlog).

## Manual test checklist (for the user)

- [ ] With no `.venv`: "Create Python Environment" creates one, installs
      ipykernel, and the kernel switches to `.venv` (toast shows progress).
- [ ] With an existing `.venv`: reuses it (installs ipykernel if missing) and
      selects it.
- [ ] No Python on PATH: a clear error toast appears.
- [ ] The new `.venv` also shows up in the picker list afterward.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
