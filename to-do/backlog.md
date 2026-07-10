# Backlog

Non-phased suggestions and to-do items. Move an item into a `phase_<n>.md`
when it is scheduled; never implement directly from here. Roughly ordered
high → low within each group.

Scheduled into phases (kept here only as a pointer):
- Copy / cut / paste / duplicate cell → **phase 7**.
- Undo/redo for cell operations → **phase 8**.
- Watch the .ipynb for external changes and reload → **phase 9**.

Small follow-ups:
- Paste cell ABOVE (`shift-v`) — phase 7 wired only paste-below.

## Medium priority

- Let "Create Python Environment" choose the location (user 2026-07-08 —
  for user-based rather than repo-based venvs in a central place). Default to
  the local workspace `.venv` so Enter/OK just creates it there, but allow
  picking a different directory. (Phase 5 follow-up.)
- A convenient "reload from disk" affordance for the notebook (user 2026-07-08):
  when the .ipynb changed on disk, the conflict toast currently tells the user
  to close and reopen. `Item::reload` exists (wired in phase 9) but isn't
  surfaced as a button/command. Add a Reload button (e.g. on the conflict
  toast) or a "Reload Notebook" command that calls the existing reload.
- Create new `.ipynb` files (user 2026-07-08 — currently you must duplicate an
  existing notebook and empty it). Two parts, VS Code parity:
  1. When a `.ipynb` is created via the file browser (empty file), populate it
     with a minimal valid nbformat v4 template (one empty code cell) so it
     opens as a notebook instead of erroring on empty/invalid JSON.
  2. A "New Jupyter Notebook" command-palette action that opens an untitled,
     unsaved notebook in the editor.
- Cell hover controls: VS Code-style per-cell toolbar on the selected/hovered
  cell, reusing the phase-4 actions. USER PREFERENCE (2026-07-08): wants the
  run-above / run-cell-and-below (and likely delete / add) buttons shown
  individually in the focused cell's top-right, NOT tucked in the "More
  options" menu. The actions, keybinds, and menus already exist; this is the
  discoverable per-cell button layer.
- Deleting the LAST remaining cell: currently refused. Decide/implement the
  preferred behaviour — either clear its contents in place, or delete it and
  insert a fresh empty cell — so delete always "does something". (Spun off
  from phase 4; user asked.)
- `a`/`b` add-cell (and the + toolbar buttons) currently switch straight into
  EDIT mode. VS Code focuses the new cell but stays in command mode until you
  press Enter. Consider matching that (focus + command mode). (Spun off from
  phase 4.)
- Auto-run the triggering cell after a kernel is picked from the run-prompt.
  Dropped when fixing bugs.md #11 (the cell no longer queues on a no-kernel
  run to avoid the stuck-"Running" state); re-add the auto-run once the
  picker-dismiss lifecycle can be tracked cleanly.
- Reset the per-cell execution counter (the number under each cell's run
  button) when the kernel is restarted, so a fresh run-through is visually
  distinct from a session with re-runs. (User 2026-07-08.) The count comes
  from the kernel's `execute_input.execution_count` (`cell.rs`); on restart,
  clear each cell's `execution_count` (and probably its outputs' counts).
- Persist the per-notebook kernel choice across full Zed restarts (currently
  only within a session; user: "not a big deal"). Likely via the saved .ipynb
  metadata match or a persisted `ReplStore`. (Spun off from phase 6.)
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
- Immediate interrupt of C-level blocking calls (e.g. `time.sleep`) on Windows
  (follow-up to bug #3). The event-based interrupt sets Python's interrupt flag
  but doesn't wake a blocking C call, so `time.sleep` only interrupts when it
  returns. jupyter's "signal" interrupt mode launches the kernel in a new
  process group and uses `GenerateConsoleCtrlEvent(CTRL_BREAK/ CTRL_C)` for
  prompt interruption — bigger launch change; normal Python loops already
  interrupt promptly.
- Return keyboard focus to the notebook after toolbar-button / popover
  interactions so command-mode shortcuts keep working without clicking a cell
  (follow-up to bug #15 if the on_focus/Escape mitigations aren't enough).
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
- ~~Remove leftover `println!` debug lines in `move_cell_up` / `move_cell_down`~~
  (done in phase 8).
- Fix typo `"CellControlType::CollapseCelln"` (`cell.rs`).

## Done (implemented in an earlier phase — kept briefly for reference)

- Change cell type via keyboard `m`/`y` in nav mode → phase 4.
- Queue cell executions while the kernel is starting → phase 2.
- Run all above / run cell and below (via "More options" menu) → phase 4.
