# Phase 11 — Run always returns to command mode

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING. Do NOT archive until the user
> confirms.
> Kind: **change to existing behaviour** — keep OPEN until the user confirms the
> mode actually switches; if it still lands in edit mode the change did not take.

Requested by the user (2026-07-09): "if I ctrl-enter or shift-enter to execute
[it] should always push to command mode. I've noticed that shift-enter will
stay in whatever mode you're in when you press it, so if you're in edit mode it
will move to the next cell in edit mode, which it shouldn't."

Primary file: `crates/repl/src/notebook/notebook_ui.rs`.

## Implemented

- [x] `run_current_cell` (Run — `ctrl-enter`/`cmd-enter`) now calls
      `enter_command_mode` after executing, for code AND markdown cells. It
      previously left a code cell in edit mode (cursor stayed in the editor).
- [x] `run_and_advance` (RunAndAdvance — `shift-enter`) continues to advance via
      `advance_in_command_mode` / `enter_command_mode`, both of which set
      `NotebookMode::Command` and focus the notebook (not the next cell's
      editor).

## Manual test checklist (for the user)

- [ ] In EDIT mode, `ctrl-enter` runs the cell and drops to command mode
      (cursor leaves the editor; single-key shortcuts work immediately).
- [ ] In EDIT mode, `shift-enter` runs, advances to the next cell, and lands in
      COMMAND mode (not edit mode).
- [ ] In COMMAND mode, both still behave the same (stay in command mode).

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean, tests pass.
