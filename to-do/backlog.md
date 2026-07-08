# Backlog

Non-phased suggestions and to-do items. Move an item into a `phase_<n>.md`
when it is scheduled; never implement directly from here.

- Bind the "smart arrows" (`NotebookMoveUp`/`NotebookMoveDown`) to up/down in
  edit mode so arrow travel crosses cell boundaries at the first/last line
  (Jupyter/VS Code style). Deferred from phase 3: the handlers exist and are
  registered but unbound; binding up/down in the `NotebookEditor > Editor`
  context risks overriding completion-menu up/down navigation, which needs
  runtime testing to get the context precedence right.
- Cell hover controls: VS Code-style per-cell toolbar on the selected/hovered
  cell (run-above / run-below / delete / add-below), reusing the phase-4
  actions. Deferred out of phase 4 — the actions, keybinds, and the two
  ellipsis menus already expose this functionality; the hover toolbar is a
  convenience layer.
- Split cell / join cells (user: low priority, rarely used)
- Cell grouping (user: low priority, rarely used)
- Copy / cut / paste / duplicate cell actions (with clipboard format
  compatible enough for within-notebook use)
- Undo/redo for cell-level operations (delete, move, add, change type)
- Change cell type via keyboard: `M` (to markdown) / `Y` (to code) in nav
  mode — pairs with the cell-type-conversion action from phase 4
- Clear output for a single cell (only clear-all exists; the inline REPL has
  `ClearCurrentOutput` to crib from, `repl_sessions_ui.rs:51`)
- Queue cell executions while the kernel is starting instead of erroring
  ("the kernel is still starting") — parity with `session.rs:683-793`
- Conda environment creation (venv creation is phase 5; conda adds a second
  toolchain + `conda create` flow)
- Watch the .ipynb file for external changes and reload (TODO at
  `notebook_ui.rs:1519`)
- Implement `open_notebook` (currently a `println!` stub,
  `notebook_ui.rs:739-741`)
- Implement `Item::pixel_position_of_cursor` (`notebook_ui.rs:1740`) so the
  workspace can track the notebook cursor
- Remove `#![allow(unused, dead_code)]` from `notebook_ui.rs` once the
  feature stabilises, and delete the large commented-out `NotebookControls`
  block (~`notebook_ui.rs:1634`)
- Remove leftover `println!` debug lines in `move_cell_up` / `move_cell_down`
  (`notebook_ui.rs:744, 753`)
- Fix typo `"CellControlType::CollapseCelln"` (`cell.rs:71`)
- Decide behaviour of ctrl-c inside a focused cell editor (currently copy;
  interrupt is only bound in the outer `NotebookEditor` context — likely
  intentional, but worth a decision; consider `i i` double-tap interrupt like
  Jupyter)
- Kernel autostart on notebook open (setting-gated), once bug #2 lands
- Collapse/expand cell input and output (a `CellControlType` scaffold exists)
- Consider surfacing kernel stderr in the UI on launch failure (the WSL path
  captures it; native now captures it on premature exit — extend to
  post-connect failures)
- Replace the fixed 500ms native-launch readiness sleep with a proper
  kernel_info/heartbeat handshake (follow-up to bug #6 if 10054 persists)
- Markdown cell rendered-preview toggle improvements (render on exit-edit)
- Notebook-level "Run all above/below from toolbar" once per-cell variants
  (phase 4) prove out
