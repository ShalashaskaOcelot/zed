# Backlog

Non-phased suggestions and to-do items. Move an item into a `phase_<n>.md`
when it is scheduled; never implement directly from here. Roughly ordered
high → low within each group.

Scheduled into phases (kept here only as a pointer):
- Copy / cut / paste / duplicate cell → **phase 7**.
- Undo/redo for cell operations → **phase 8**.
- Watch the .ipynb for external changes and reload → **phase 9**.
- Create new `.ipynb` + "New Jupyter Notebook" command → **phase 12**.
- Per-cell hover/selection toolbar → **phase 13**.
- Save-conflict guard + Reload affordance (with bug #14) → **phase 14**.
- Reset execution counter on restart + `ClearCellOutputs` action → **phase 15**.
- Better DataFrame/table output rendering → **phase 16**.
- Execution status & timing correctness (pending vs running, per-cell timing,
  batch status, restart-cancel) → **phase 17**.
- Move execution status/time into the cell + center gutter run button →
  **phase 18**.
- Configurable post-run landing mode (command/edit/remember) → **phase 19**.
- Per-cell scoped stop/interrupt → **phase 20**.
- Live elapsed-time counter while running → **phase 21**.
- Multi-select cells → **phase 22**.
- Collapse/expand cell input & output → **phase 23**.

Small follow-ups:
- Paste cell ABOVE (`shift-v`) — phase 7 wired only paste-below.

## Medium priority

- (Moved to **phase 20**) Per-cell stop/interrupt should be scoped to the cell,
  not global (user 2026-07-11): the gutter stop button dispatches the global
  `InterruptKernel`; stopping a queued cell should remove just that cell and
  leave the rest running.
- Dedicated REPL / Notebook section in the GUI settings UI (user 2026-07-11):
  as notebook config grows (landing mode, and future options), surface a
  grouped settings page/section so they're discoverable and editable in one
  place rather than only via settings.json. (Depends on how Zed's settings UI
  registers sections.)

- (Moved to **phase 22**) Multi-select cells (user 2026-07-11), VS Code /
  file-explorer / Excel style (shift+arrow range, shift+click range, ctrl+click
  toggle, ctrl+arrows no-op); actions operate on the whole selection.
- (Moved to **phase 21**) Live elapsed-time counter while a cell runs (user
  2026-07-11): a ticking timer during execution instead of only the total on
  finish.
- Further DataFrame/table polish (user 2026-07-11, "could be done better"):
  follow-up refinements to the phase-16 table rendering (spacing, column
  sizing/eliding, header styling, very wide frames, dark/light contrast).
- Selectable output text (user 2026-07-11): allow selecting a PORTION of a
  cell's output to copy, instead of only the whole output via the "..." menu's
  Copy Output. Outputs render as TerminalOutput / markdown / table elements
  that don't support text selection today.
- "Open output in new editor" (user 2026-07-11, larger item): for long /
  scrolling outputs, an affordance (like VS Code's "Open in text editor") that
  opens the full output in a regular editor buffer/tab for searching,
  selecting, and scrolling comfortably.

- "New Jupyter Notebook" should open a truly UNSAVED notebook (phase 12
  follow-up, user 2026-07-11, re-raised 2026-07-11). Currently the command
  writes `Untitled-N.ipynb` into the workspace immediately and opens that. It
  should behave like Ctrl-N: an untitled, session-only buffer that only hits
  disk on manual save (with a save-as flow on first save). This applies ONLY
  to the command-palette command — notebooks created via the file browser's
  New File are correctly saved where they're created, with the given name, and
  must stay that way. Needs project-item / editor routing for a notebook
  backed by a path-less buffer (the `NotebookItem` open path currently
  requires a `ProjectEntryId` and a saved `.ipynb`). Bigger plumbing —
  schedule as its own phase when picked up.

- Let "Create Python Environment" choose the location (user 2026-07-08 —
  for user-based rather than repo-based venvs in a central place). Default to
  the local workspace `.venv` so Enter/OK just creates it there, but allow
  picking a different directory. (Phase 5 follow-up.)
- (Moved to **phase 14**) A convenient "reload from disk" affordance for the
  notebook (user 2026-07-08): the conflict toast currently tells the user to
  close and reopen. `Item::reload` exists (wired in phase 9) but isn't surfaced
  as a button/command.
- (Moved to **phase 12**) Create new `.ipynb` files (user 2026-07-08 — currently
  you must duplicate an existing notebook and empty it). Two parts, VS Code
  parity: populate an empty file with a minimal nbformat v4 template; add a
  "New Jupyter Notebook" command.
- (Moved to **phase 13**) Cell hover controls: VS Code-style per-cell toolbar on
  the selected/hovered cell, reusing the phase-4 actions. USER PREFERENCE
  (2026-07-08): run-above / run-cell-and-below (and likely delete / add) buttons
  shown individually in the focused cell's top-right, NOT tucked in "More
  options".
- Deleting the LAST remaining cell: currently refused. Decide/implement the
  preferred behaviour — either clear its contents in place, or delete it and
  insert a fresh empty cell — so delete always "does something". (Spun off
  from phase 4; user asked.)
- (DONE via bug #13) `a`/`b` add-cell (and the + toolbar buttons) now stay in
  COMMAND mode (focus the new cell, press Enter to edit) instead of jumping into
  edit mode. Awaiting the same user confirmation as bug #13.
- Auto-run the triggering cell after a kernel is picked from the run-prompt.
  Dropped when fixing bugs.md #11 (the cell no longer queues on a no-kernel
  run to avoid the stuck-"Running" state); re-add the auto-run once the
  picker-dismiss lifecycle can be tracked cleanly.
- (Moved to **phase 15**) Reset the per-cell execution counter when the kernel
  is restarted, so a fresh run-through is visually distinct from a re-run
  session. (User 2026-07-08.)
- Persist the per-notebook kernel choice across full Zed restarts (currently
  only within a session; user: "not a big deal"). Likely via the saved .ipynb
  metadata match or a persisted `ReplStore`. (Spun off from phase 6.)
- (Moved to **phase 23**) Collapse/expand cell input and output (a
  `CellControlType` scaffold exists). Useful for cells with large outputs.
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
- (Moved to **phase 15**) Dedicated `ClearCellOutputs` action + keybind for a
  single cell (the per-output "..." menu already offers "Clear Output"; this
  adds a command/keybind). Crib from the inline REPL's `ClearCurrentOutput`.
- Markdown cell rendered-preview toggle improvements (render on exit-edit).
- (Resolved in phase 7 rebind) ctrl-c inside a focused cell editor now does
  editor text-copy; cell copy/cut/paste moved to ctrl-c/x/v in command mode;
  interrupt moved off ctrl-c onto the Jupyter-standard `i i` in command mode
  (plus the toolbar Stop button).

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
