# Phase 50 — Go to running cell + Follow running cell

Kind: **new feature**. Promoted from the backlog (both items, user 2026-07-16)
to restore 5 phases in rotation after phase 44 completed. Two tightly-related
notebook features that share a "which cell is currently running?" lookup:

1. A **Go to running cell** action/button that jumps to the executing cell.
2. A **Follow running cell** toggle that auto-scrolls the viewport to the
   running cell as it changes, so a Run All visibly "walks" down the notebook.

Primary file: `crates/repl/src/notebook/notebook_ui.rs` (cell lookup via
`Cell::is_executing()` in `cell.rs`).

## Tasks

- [ ] Add a `running_cell_index(&self, cx) -> Option<usize>` helper on
      `NotebookEditor`: the first cell whose `is_executing()` is true (scan
      `cell_order`). Both features reuse it.
- [ ] `notebook::GoToRunningCell` action (auto-appears in the command palette):
      when a cell is running, `set_selected_index(index, true, …)` +
      `enter_command_mode` + `scroll_to_reveal_item_top_aligned(index)` (land
      at the top of the cell, matching the rest of the notebook's navigation).
      No-op (not an error) when nothing is running.
- [ ] Sidebar button for Go to running cell via `render_notebook_control`,
      placed under Run All, `.disabled(running_cell_index(cx).is_none())`
      (mirrors the existing Interrupt button's disabled pattern in
      `render_notebook_controls`).
- [ ] Persist a `follow_running_cell: bool` on `NotebookEditor` (default off).
      Add a toggle `IconButton` in the right sidebar directly under the Go to
      running cell button, using `.toggle_state(self.follow_running_cell)`.
- [ ] When the running cell changes and the toggle is ON, scroll the viewport
      to it WITHOUT stealing edit focus or forcing command mode — call
      `cell_list.scroll_to_reveal_item(index)` (viewport only) from the point
      where `advance_run_queue` moves to / begins the next running cell. Must
      NOT scroll once the batch ends or errors past the last executed cell (do
      not jump back to the top / off the end).

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

- `./script/clippy` clean; `cargo test -p repl` passes.
- User test: run a long cell → Go to running cell (palette + sidebar button)
  reveals and selects it at the top; the button is greyed when nothing runs.
- User test: turn Follow on, Run All a multi-cell notebook → the viewport
  walks down with execution and parks on the errored cell on failure; editing
  a cell during the run is not interrupted by the auto-scroll; toggling Follow
  off stops the auto-scroll.
