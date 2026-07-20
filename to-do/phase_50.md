# Phase 50 — Go to running cell + Follow running cell

⚠️ **AWAITING USER TESTING** — implementation complete (compiles; scoped
`cargo clippy -p repl -p zed_actions` clean). The two user tests below are
unconfirmed. Once confirmed the phase can be archived (it's a new feature, so
refinements/defects found become separate backlog/bug items, not a reopen).

Kind: **new feature**. Promoted from the backlog (both items, user 2026-07-16)
to restore 5 phases in rotation after phase 44 completed. Two tightly-related
notebook features that share a "which cell is currently running?" lookup:

1. A **Go to running cell** action/button that jumps to the executing cell.
2. A **Follow running cell** toggle that auto-scrolls the viewport to the
   running cell as it changes, so a Run All visibly "walks" down the notebook.

Primary file: `crates/repl/src/notebook/notebook_ui.rs` (cell lookup via
`Cell::is_executing()` in `cell.rs`).

## Tasks

- [x] Add a `running_cell_index(&self, cx) -> Option<usize>` helper on
      `NotebookEditor`: the first cell whose `is_executing()` is true (scan
      `cell_order`). Both features reuse it.
- [x] `notebook::GoToRunningCell` action (auto-appears in the command palette):
      when a cell is running, `set_selected_index(index, true, …)` (which
      top-aligns via `jump_to_cell`) + `enter_command_mode`. No-op (not an
      error) when nothing is running.
- [x] Sidebar button for Go to running cell via `render_notebook_control`
      (`IconName::Crosshair`), placed under the Run All group,
      `.disabled(!has_running_cell)` (mirrors the Interrupt button's disabled
      pattern in `render_notebook_controls`).
- [x] Persist a `follow_running_cell: bool` on `NotebookEditor` (default off).
      Added a toggle `IconButton` (`IconName::Eye`) in the right sidebar
      directly beside the Go to running cell button, using
      `.toggle_state(self.follow_running_cell)` +
      `notebook::ToggleFollowRunningCell` action. Turning it on jumps to the
      running cell immediately.
- [x] When the running cell changes and the toggle is ON, scroll the viewport
      to it WITHOUT stealing edit focus or forcing command mode — calls
      `cell_list.scroll_to_reveal_item(index)` (viewport only) from
      `advance_run_queue` as it begins the next running cell. Non-code cells
      are skipped there and the final advance (empty queue) doesn't scroll, so
      it never jumps back to the top / off the end.

## Risks / gaps

- The running-cell scroll in follow mode must not fight the user: viewport
  scroll only (`scroll_to_reveal_item`, NOT `_top_aligned` + select + focus
  like the explicit Go to action), and never change edit/command mode, so a
  user editing a later cell during a Run All isn't yanked away mid-keystroke.
- Single-cell runs vs Run All batches: `running_cell_index` returns the first
  executing cell, which is correct for both (only one runs at a time in the
  sequential queue).
- Don't scroll on the FINAL advance when the queue empties — guard on there
  actually being a next running cell, else the viewport lurches at batch end.

## Verification

- [x] `cargo clippy -p repl -p zed_actions` clean (the full `./script/clippy`
      can't run in the remote env — `--all-features` pulls in `alsa-sys`, whose
      build needs the system ALSA dev lib; unrelated to this change).
- [ ] ⚠ untested — User test: run a long cell → Go to running cell (palette +
      sidebar Crosshair button) reveals and selects it at the top; the button
      is greyed when nothing runs.
- [ ] ⚠ untested — User test: turn Follow (Eye button) on, Run All a
      multi-cell notebook → the viewport walks down with execution and parks on
      the errored cell on failure; editing a cell during the run is not
      interrupted by the auto-scroll; toggling Follow off stops the auto-scroll.
