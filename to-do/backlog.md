# Backlog

Non-phased suggestions and to-do items that are NOT yet scheduled. Move an item
into a `phase_<n>.md` when it is scheduled (and delete it from here); never
implement directly from here. Completed and scheduled work is not tracked here —
see the phase files, `CHANGELOG.md`, and git history. Roughly ordered
high → low within each group.

## Medium priority

- Dedicated REPL / Notebook section in the GUI settings UI (user 2026-07-11):
  as notebook config grows (landing mode, and future options), surface a
  grouped settings page/section so they're discoverable and editable in one
  place rather than only via settings.json. (Depends on how Zed's settings UI
  registers sections.)

- Let "Create Python Environment" choose the location (user 2026-07-08 —
  for user-based rather than repo-based venvs in a central place). Default to
  the local workspace `.venv` so Enter/OK just creates it there, but allow
  picking a different directory. (Phase 5 follow-up.)

- Kernel autostart on notebook open, setting-gated (opt-in), now that
  lazy-start is the default and auto-start-on-run exists.

- Immediate interrupt of C-level blocking calls (e.g. `time.sleep`) on Windows
  (follow-up to bug #3). The event-based interrupt sets Python's interrupt flag
  but doesn't wake a blocking C call, so `time.sleep` only interrupts when it
  returns. jupyter's "signal" interrupt mode launches the kernel in a new
  process group and uses `GenerateConsoleCtrlEvent(CTRL_BREAK/ CTRL_C)` for
  prompt interruption — bigger launch change; normal Python loops already
  interrupt promptly.

## Low priority

- Notebook-level "Run all above / run all below" dedicated toolbar buttons
  (the "More options" menu already exposes both actions).
- Conda environment creation (venv creation exists; conda adds a second
  toolchain + `conda create` flow).
- Split cell / join cells (user: low priority, rarely used).
- Cell grouping (user: low priority, rarely used).
- Implement `open_notebook` (currently a `println!` stub, `notebook_ui.rs`).
- Implement `Item::pixel_position_of_cursor` so the workspace can track the
  notebook cursor (`notebook_ui.rs`).

## Cleanup (do once the feature stabilises)

- Remove `#![allow(unused, dead_code)]` from `notebook_ui.rs` and delete the
  large commented-out `NotebookControls` block.
