# Changelog

A concise record of completed notebook work, replacing the per-phase archive.
Each entry references the commit where the work was implemented (follow-up fix
commits are not all listed — see `git log`). Add a one-line entry here when a
phase is completed or a bug is confirmed fixed; then delete the phase file / bug
entry rather than archiving it.

## Completed phases

- Phase 1 — Discovery: mapped the notebook implementation and seeded the
  to-do system. (planning; no code commit)
- Phase 2 — Kernel lifecycle fixes (restart/relaunch, run-after-shutdown). `7bd5b5a`
- Phase 3 — Navigation and scrolling between cells. `2244bac`
- Phase 4 — Cell actions, Jupyter keyboard shortcuts, and the "More options" /
  output menus. `b018b45`
- Phase 5 — Create Python environments from the kernel picker. `b846cbe`
- Phase 6 — Kernel selection persistence and lazy start. `94be9f5`
- Phase 7 — Cell clipboard operations (copy / cut / paste / duplicate). `1b6d566`
- Phase 8 — Cell-operation undo/redo. `7b41fbb`
- Phase 9 — External file sync (watch the .ipynb and reload on change). `3ae7e5a`
- Phase 10 — Sequential multi-cell execution with stop-on-error. `fd17668`
- Phase 11 — Run always returns to command mode. `da77390`
- Phase 12 — Create & open new notebooks ("New Jupyter Notebook"). `5641245`
- Phase 13 — Per-cell hover/selection toolbar. `4177b51`
- Phase 14 — Notebook data safety: save-conflict guard + reload affordance. `28c153d`
- Phase 15 — Cell output & execution-state management (clear outputs, counter
  reset on restart). `28c153d`
- Phase 16 — Better DataFrame (table) output rendering. `34be0cd`
- Phase 17 — Execution status & queue correctness (pending vs running, batch
  supersede, per-cell timing). `5946ec5`
- Phase 18 — In-cell execution status display (VS Code style). `a983214`
- Phase 19 — Configurable post-run landing mode (shift-enter / ctrl-enter). `447889a`
- Phase 20 — Per-cell scoped stop / interrupt. `9564103`
- Phase 21 — Live elapsed-time counter while a cell runs. `bf7a79f`
- Phase 22 — Multi-select cells (shift/ctrl gestures; actions over the
  selection). `bd6790e`
- Phase 23 — Collapse / expand cell input & output (persisted to the .ipynb). `970219c`
- Phase 24 — Cell operations polish: paste-above, replace-emptied-notebook on
  delete, smart edit-mode arrows. `5b8efcb` `add918d` `013fa4c`
- Phase 25 — Kernel selection QoL: a reopened notebook pre-selects the kernel
  saved in its metadata (VS Code-compatible, lazy-start on first run); auto-run
  of cells queued behind the kernel picker verified working.
- Phase 26 — More multi-select gestures: ctrl/cmd-a select-all,
  shift-home/shift-end to first/last cell (command mode only).
- Phase 27 — Retain cell output through clipboard & undo: verified the whole
  snapshot pipeline (copy/cut/delete → paste/undo) carries outputs — bug #24's
  source-media fix supplied the missing serialization — and locked it in with a
  round-trip test.
- Phase 28 — Per-cell "last executed" timestamp: completion time shown next to
  the ✓/✕ + duration (setting `notebook_show_last_executed`, default on),
  persisted VS Code-compatibly in cell metadata.execution; loaded notebooks
  restore ✓ + duration + time from saved timestamps.
- Phase 30 — Notebook chrome rework: bottom kernel bar removed; kernel
  cluster (status + name = picker trigger) now a slim top-right strip above
  the cells; Restart/Interrupt moved into the right sidebar.
- Phase 32 — Notebook UX niceties: Esc in command mode collapses a
  multi-selection to the primary cell; output-menu actions return focus to
  the notebook (command mode); markdown preview render-on-blur verified
  already wired.
- Phase 31 — Kernel launch robustness: the fixed 500ms readiness sleep is
  replaced by a heartbeat handshake (echo = ready, 30s cap, early-exit watch
  with stderr); kernels that die after connecting report their last stderr
  lines instead of a bare exit status.
- Phase 29 — Output interaction: notebook outputs get the inline REPL's
  copy / open-in-buffer controls (open-in-buffer = selectable text for long
  outputs); DataFrame tables render in the measured font (fixes wrapped/
  truncated cells) and fill the output width proportionally when a long
  column exists (compact tables stay compact). In-place text selection
  re-phased as phase 37.
- Phase 34 — Notebook & kernel configuration: opt-in kernel autostart on open
  (`notebook_autostart_kernel`, default off, remembered kernels only, never
  after a failed launch); "Create Python Environment" can now choose the venv
  location (fast path still the workspace `.venv`); new "REPL & Notebooks"
  page in the GUI settings UI (landing mode dropdown, timestamps, autostart,
  REPL output limits).
- Phase 36 — Notebook code health: removed the file-wide allow(unused,
  dead_code) from notebook_ui.rs and every piece of dead code it hid
  (constants, remote_id field, two dead methods, stale imports, the
  commented-out NotebookControls block); OpenNotebook now opens the
  workspace file picker instead of println!; Item::pixel_position_of_cursor
  implemented (delegates to the selected cell's editor).

## Fixed bugs (confirmed)

- #1 — Restart kernel killed the kernel but the relaunch failed. `7bd5b5a`
- #2 — Running a cell with a dead/shutdown kernel didn't start the kernel. `7bd5b5a`
- #3 — Interrupt / stop button didn't interrupt a running cell (OS-level
  interrupt; C-blocking-call interrupt on Windows remains a backlog item). `0ad52d7`
- #4 — "More options" toolbar button opened nothing. `b018b45`
- #5 — Output "…" (ellipsis) button next to cell output did nothing. `b018b45`
- #8 — Clean kernel exit left stale RunningKernel state. `7bd5b5a`
- #10 — Kernel picker didn't accept Enter to select. `cfcbcd2`
- #11 — Kernel-select prompt: wrong cell state on dismiss vs. select. `759b681`
- #13 — After adding a cell with `a`/`b`, Enter sometimes wouldn't enter edit
  mode (add-cell now stays in command mode). `8026f54`
- #19 — Reloading didn't clear the conflict notification toast. `eb99d21`
- #21 — Zed's own metadata save raised a spurious "changed on disk" toast. `82d98a7`
- #24 — Rich outputs (tables/images/markdown/json) were dropped on save, so
  outputs didn't survive close/reopen. `7c3c1bd`
- #22 — Very fast cells showed a bare ✓ (or 0ms) with no execution time.
  `02f395c` `ec70129`
- #23 — Clicking a cell's gutter/margin didn't select the cell. `7f9e17a`
  `b066179`
- #25 — A cell added at the viewport bottom landed out of view behind the
  bottom bar. `ccc2164` `d63b87e`
- #26 — A cell that errored showed a completed ✓ instead of a red ✕. `02de430`
  `b3b7887`
- #28 — Cells flashed/stuck "Cancelled" around kernel selection (picker Escape
  and post-pick startup). `db1173b`
