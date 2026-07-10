# Phase 15 — Cell output & execution-state management

Kind: **mixed** — a change to existing behaviour (reset counters on restart)
plus a new feature (dedicated clear-outputs action). Not yet started — a plan.

Goal: keep the per-cell execution counters and outputs coherent with the kernel
state, and give a first-class way to clear a single cell's outputs.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (`restart_kernel`),
`crates/repl/src/notebook/cell.rs` (execution_count, outputs), the actions
module + keymaps for the new action.

## Tasks

- [ ] On kernel restart, reset each cell's execution counter (the number under
      the run button) and its outputs' counts, so a fresh run-through is
      visually distinct from a re-run session. The count comes from
      `execute_input.execution_count`; clear it in `restart_kernel`.
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
