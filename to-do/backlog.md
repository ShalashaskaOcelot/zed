# Backlog

Non-phased suggestions and to-do items. Move an item into a `phase_<n>.md`
when it is scheduled; never implement directly from here. Roughly ordered
high → low within each group.

Scheduled into phases (kept here only as a pointer):
- Copy / cut / paste / duplicate cell → **phase 7**.
- Undo/redo for cell operations → **phase 8**.
- Watch the .ipynb for external changes and reload → **phase 9**.

## Medium priority

- Cell hover controls: VS Code-style per-cell toolbar on the selected/hovered
  cell (run-above / run-below / delete / add-below), reusing the phase-4
  actions. The actions, keybinds, and the two ellipsis menus already expose
  this functionality; the hover toolbar is a convenience layer.
- Collapse/expand cell input and output (a `CellControlType` scaffold exists).
  Useful for cells with large outputs.
- Bind the "smart arrows" (`NotebookMoveUp`/`NotebookMoveDown`) to up/down in
  edit mode so arrow travel crosses cell boundaries at the first/last line
  (Jupyter/VS Code style). Deferred from phase 3: handlers exist but are
  unbound; binding up/down in the `NotebookEditor > Editor` context risks
  overriding completion-menu up/down navigation — needs runtime testing to get
  context precedence right.
- Kernel autostart on notebook open, setting-gated (opt-in), now that
  lazy-start is the default (phase 6) and auto-start-on-run (bug #2) exists.
- Surface kernel stderr in the UI on launch failure (the WSL path captures it;
  native now captures it on premature exit — extend to post-connect failures).
- Replace the fixed 500ms native-launch readiness sleep with a proper
  kernel_info/heartbeat handshake (follow-up to bug #6 if 10054 persists).
- Dedicated `ClearCellOutputs` action + keybind for a single cell (the
  per-output "..." menu already offers "Clear Output"; this adds a
  command/keybind). Crib from the inline REPL's `ClearCurrentOutput`
  (`repl_sessions_ui.rs:51`).
- Markdown cell rendered-preview toggle improvements (render on exit-edit).
- Decide behaviour of ctrl-c inside a focused cell editor (currently copy;
  interrupt is only bound in the outer `NotebookEditor` context — likely
  intentional). Consider `i i` double-tap interrupt like Jupyter.

## Low priority

- Notebook-level "Run all above / run all below" dedicated toolbar buttons
  (the phase-4 "More options" menu already exposes both actions).
- Conda environment creation (venv creation is phase 5; conda adds a second
  toolchain + `conda create` flow).
- Split cell / join cells (user: low priority, rarely used).
- Cell grouping (user: low priority, rarely used).
- Implement `open_notebook` (currently a `println!` stub,
  `notebook_ui.rs`).
- Implement `Item::pixel_position_of_cursor` so the workspace can track the
  notebook cursor (`notebook_ui.rs`).

## Cleanup (do once the feature stabilises)

- Remove `#![allow(unused, dead_code)]` from `notebook_ui.rs` and delete the
  large commented-out `NotebookControls` block.
- Remove leftover `println!` debug lines in `move_cell_up` / `move_cell_down`.
- Fix typo `"CellControlType::CollapseCelln"` (`cell.rs`).

## Done (implemented in an earlier phase — kept briefly for reference)

- Change cell type via keyboard `m`/`y` in nav mode → phase 4.
- Queue cell executions while the kernel is starting → phase 2.
- Run all above / run cell and below (via "More options" menu) → phase 4.
