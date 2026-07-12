# Phase 15 — Cell output & execution-state management

Kind: **mixed** — a change to existing behaviour (reset counters on restart)
plus a new feature (dedicated clear-outputs action). Not yet started — a plan.

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

## Tasks

- [ ] On kernel restart, reset each cell's execution counter (the `In [N]`
      number) and its outputs' counts to empty, so a fresh run-through is
      visually distinct from the previous session. Clear each cell's
      `execution_count` in `restart_kernel` (the kernel itself already restarts
      its counter at 1; this just drops the stale UI numbers).
- [ ] Add a dedicated `ClearCellOutputs` action + keybind that clears the
      SELECTED cell's outputs (the per-output "..." menu already offers "Clear
      Output" for one output; this does the whole cell). Crib from the inline
      REPL's `ClearCurrentOutput` (`repl_sessions_ui.rs`). Add to command mode
      keymaps + the More options menu.

## Risks / gaps

- Only reset counters on a genuine restart/shutdown, not on interrupt (an
  interrupted cell keeps its history).
- Keep the clear-outputs action's keybind clear of the clipboard/undo combos
  just standardised in phase 7/8.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: run cells, restart → counters reset to empty; select a cell and
  clear its outputs via the action/keybind.
