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
  follow-up, user 2026-07-11). Currently the command writes `Untitled-N.ipynb`
  into the workspace immediately and opens that. It should behave like Ctrl-N:
  an untitled, session-only buffer that only hits disk on manual save (with a
  save-as flow on first save). This applies ONLY to the command-palette
  command — notebooks created via the file browser's New File are correctly
  saved where they're created, with the given name, and must stay that way.
  Needs project-item / editor routing for a notebook backed by a path-less
  buffer (the `NotebookItem` open path currently requires a `ProjectEntryId`
  and a saved `.ipynb`). Bigger plumbing — schedule as its own phase.

- Let "Create Python Environment" choose the location (user 2026-07-08 —
  for user-based rather than repo-based venvs in a central place). Default to
  the local workspace `.venv` so Enter/OK just creates it there, but allow
  picking a different directory. (Phase 5 follow-up.)

- Kernel autostart on notebook open, setting-gated (opt-in), now that
  lazy-start is the default and auto-start-on-run exists.

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

- Markdown cell rendered-preview toggle improvements (render on exit-edit).

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
