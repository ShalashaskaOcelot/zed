# Phase 39 — Conda environment creation from the kernel picker

Kind: **new feature**. Not yet started — this is a plan. Promoted from the
backlog to keep 5 phases in rotation after phase 34 completed.

venv creation already exists ("Create Python Environment" in the kernel
picker, incl. the phase 34 location chooser). Conda is the other major Python
environment toolchain; users on conda-managed machines can't use the venv
flow (their base python may itself be a conda install, and team norms often
mandate conda envs).

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (the
`create_python_environment` / `create_python_environment_at` flow).

## Tasks

- [ ] Detect whether `conda` (or `mamba`/`micromamba`) is available on PATH;
      only offer the conda option when it is.
- [ ] Extend the create-environment prompt with a conda choice (keep the fast
      path: Enter still = workspace `.venv`). Ask for an environment name,
      then run `conda create -y -n <name> python ipykernel` (or the
      micromamba equivalent) with progress + error toasts, reusing the venv
      flow's toast/error patterns.
- [ ] Register the new env as a kernel spec the same way the venv flow does
      (resolve the env's python path via `conda env list`/known env dirs and
      build the spec from it), select it, and continue any queued run.

## Risks / gaps

- `conda create` is slow (solver + downloads) — the progress toast must make
  the wait obvious, and failure output (solver conflicts) must reach the user.
- Conda env python paths differ per platform (`envs/<name>/bin/python` vs
  `envs\<name>\python.exe`); resolve via conda itself rather than guessing.
- Don't break the existing venv fast path; conda is strictly additive.

## Verification

- clippy clean; `cargo test -p repl` passes.
- User test (conda machine): create a conda env from the picker, kernel spec
  appears and runs cells; venv fast path unchanged; no conda on PATH → no
  conda option shown.
