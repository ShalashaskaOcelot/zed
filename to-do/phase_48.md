# Phase 48 — Select a newly-created kernel immediately (queue runs while it builds)

Kind: **change to existing behaviour**. Promoted from the backlog (user
2026-07-16, from the phase-42 testing round). Planned but NOT yet
implemented — this touches the kernel-launch / run-queue state machine (the
crash-prone area of bug #48/#28/#30/#16), so it wants review before coding.

## Problem (user-observed, 2026-07-16)

When you choose "Create Python/Conda Environment" while a DIFFERENT kernel is
already selected, there's a gap while the env builds (venv + pip install, or
`conda create`). During that gap the notebook still points at the OLD kernel:
running a cell starts the OLD kernel and runs there, and the new kernel only
auto-switches in once it finishes building.

Desired (mirrors what already happens when creating from the "Select Kernel"
/ no-kernel state, which the user confirms works):
1. On choosing create, the notebook IMMEDIATELY shows the new env as its
   selected kernel.
2. The new env appears in the picker as a greyed/disabled entry (like a
   no-ipykernel entry) until it's ready.
3. Running a cell before the kernel is ready queues it as Pending (no error,
   no launch of the old kernel) — exactly the wait-for-kernel behaviour.
4. Once the env is built and the kernel is ready, it launches and the queued
   cells run.

## Relevant machinery (from investigation 2026-07-16)

- `execute_cell` dispositions (`notebook_ui.rs`): `Kernel::StartingKernel` and
  `Kernel::Restarting` → `Disposition::Queued { launch: false }` (cell pends,
  no launch). `Kernel::Shutdown` + remembered spec → `Queued { launch: true }`.
  So a "kernel is coming" state that queues runs already exists —
  `StartingKernel(Shared<Task<()>>)`.
- `change_kernel` → `promote_awaiting_cells` + `launch_kernel_with_spec`
  already runs cells queued before a kernel choice; the create flows call
  `change_kernel` on completion (via `finalize_env_creation`).
- The picker/top-strip reflect `self.kernel_specification`; greyed entries in
  the picker are rendered for specs with `has_ipykernel == false`
  (`components/kernel_options.rs`).
- WRINKLE: a `PythonEnvKernelSpecification` needs the interpreter `path`.
  For a **venv** the path is known upfront (`venv_dir/bin|Scripts/python`).
  For a **conda named env** the path is NOT known until after creation
  (resolved via `<frontend> run -n <name> python -c ...`). So a provisional
  "creating" spec can't always be a fully-formed `PythonEnvKernelSpecification`.

## Tasks

- [ ] Introduce a provisional "creating kernel" representation that carries
      just a display name + language (no interpreter path required), used only
      to (a) show the notebook's selected kernel immediately and (b) render a
      greyed picker entry. Options to decide at implementation: a dedicated
      `Kernel`/spec variant (`Kernel::CreatingKernel { name }` +
      a `KernelSpecification::Creating`), vs. a provisional
      `PythonEnvKernelSpecification` with `has_ipykernel=false` and a
      placeholder path (simpler for venv, awkward for conda). Prefer a real
      "creating" variant so conda's unknown path isn't faked.
- [ ] On create start (both `create_python_environment_at` and
      `create_conda_environment_named`): set the notebook's selected kernel to
      the provisional spec and put `self.kernel` into a state that queues runs
      (reuse the `StartingKernel` queuing path or add an equivalent). Do NOT
      write the provisional spec to notebook metadata / persistence (only the
      real resolved spec is persisted on launch).
- [ ] Route runs during creation through the existing Pending path (queued,
      spinner/pending as appropriate) so a cell run before readiness waits and
      then runs on the new kernel — no launch of the previously-selected
      kernel.
- [ ] On create success: transition from the provisional state to the real
      resolved spec and launch (existing `finalize_env_creation` →
      `change_kernel`), promoting the queued cells. On FAILURE: drop the
      provisional selection, restore a sensible state (previous kernel or
      "Select Kernel"), surface the existing error toast, and don't leave
      cells stuck Pending.
- [ ] Render the provisional kernel greyed/disabled in the picker and as the
      top-strip selection (with a "creating…"/"starting" affordance), matching
      the no-ipykernel styling.

## Risks / gaps

- This is the most bug-prone subsystem (kernel state × run queue × picker).
  Every transition (start, success, failure, user picks a different kernel
  mid-create, restart during create, close during create) must be handled or
  it strands Pending cells / leaks the create task.
- Must not regress the confirmed-good no-kernel-start path (which already does
  the right thing) — ideally this UNIFIES the two paths rather than forking.
- Provisional spec must never be persisted or matched against saved metadata.
- GUI-only verification (as with all notebook work).

## Verification

- `cargo check -p repl` + `./script/clippy` clean; `cargo test -p repl`.
- User test: with kernel A selected, Create Env → the new env shows as
  selected immediately (greyed in the picker); run a cell during creation →
  it goes Pending (kernel A does NOT start); when the env is ready it launches
  and the cell runs on it. Create FAILURE → selection reverts, cells don't
  hang. The no-kernel-start path still works.
