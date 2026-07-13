# Phase 5 — Create Python environments from the kernel picker (archived 2026-07-12)

> ✅ STATUS: IMPLEMENTATION COMPLETE; core flows user-confirmed (create venv,
> reuse existing `.venv`, new env appears in the picker). The one outstanding
> user test (the no-Python-on-PATH error toast) is tracked in
> `to-do/awaiting_testing.md` under "Phase 5".
> Kind: **new feature**.

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

- [x] With no `.venv`: "Create Python Environment" creates one, installs
      ipykernel, and the kernel switches to `.venv`. ✅ CONFIRMED.
- [x] With an existing `.venv`: reuses it. ✅ CONFIRMED.
- [ ] No Python on PATH: a clear error toast appears. ⚠ PENDING (user will
      test later).
- [x] The new `.venv` also shows up in the picker list afterward. ✅ CONFIRMED.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
