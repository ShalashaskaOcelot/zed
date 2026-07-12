# Phase 15 — Cell output & execution-state management

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Compiles, clippy-clean, tests
> pass.
> Kind: **mixed** — a change to existing behaviour (reset counters on restart:
> keep OPEN until confirmed to take effect) plus a new feature (clear-cell-
> outputs action: archive once confirmed present and working).

Goal: keep the per-cell execution counters and outputs coherent with the kernel
state, and give a first-class way to clear a single cell's outputs.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (`restart_kernel`),
`crates/repl/src/notebook/cell.rs` (execution_count, outputs), the actions
module + keymaps for the new action.

## Note on the counter (investigated 2026-07-11)

The number under the run button is `CodeCell.execution_count`, set in
`handle_message` from the kernel's `ExecuteInput.execution_count`. This is the
Jupyter execution number (`In [N]`): a SESSION-GLOBAL counter the kernel
increments once per execution — i.e. execution ORDER within the session, NOT a
per-cell run tally. Running each cell once in order shows 1,2,3,…; re-running a
cell relabels it with the next number. (It is now displayed as `[N]`.) The
reset-on-restart task below is therefore still valid and useful: it clears the
stale `In [N]` numbers so a fresh run-through after restart visibly starts at 1
instead of showing leftover numbers from the previous kernel session.

## Implemented

- [x] On kernel restart, each code cell's `In [N]` execution number is cleared
      (`CodeCell::reset_execution_count`, called in `restart_kernel`), so a
      fresh run-through visibly starts at [1] instead of showing the previous
      session's numbers. Interrupt does NOT clear them (an interrupted session
      keeps its history). Outputs are not cleared — only the stale numbers.
- [x] `notebook::ClearCellOutputs` action: clears the SELECTED cell's outputs
      (the per-output "..." menu's "Clear Output" remains for a single
      output). In the command palette and the More options menu.

## Deviation from the plan

- No default KEYBIND for ClearCellOutputs: there is no standard shortcut for
  this across Jupyter/VS Code, and the user's standing direction is to avoid
  inventing non-standard bindings. Menu + command palette only; a user can
  bind it themselves.

## Manual test checklist (for the user)

- [ ] Run several cells (numbers [1]..[N] appear), restart the kernel → all
      numbers disappear; running again starts at [1].
- [ ] Interrupt does NOT clear the numbers.
- [ ] Select a cell with outputs → "Clear Cell Outputs" (More options menu or
      command palette) clears only that cell's outputs.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; tests pass.
