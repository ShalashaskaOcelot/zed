# Changelog

A concise record of completed notebook work, replacing the per-phase archive.
Full detail lives in the git history (commit messages). Add a one-line entry
here when a phase is completed or a bug is confirmed fixed; then delete the
phase file / bug entry rather than archiving it.

## Completed phases

- Phase 1 — Discovery: mapped the notebook implementation and seeded the
  to-do system.
- Phase 2 — Kernel lifecycle fixes (restart/relaunch, dead-kernel handling).
- Phase 3 — Navigation and scrolling between cells.
- Phase 4 — Cell actions, Jupyter keyboard shortcuts, and the "More options" /
  output menus.
- Phase 5 — Create Python environments from the kernel picker.
- Phase 6 — Kernel selection persistence and lazy start.
- Phase 7 — Cell clipboard operations (copy / cut / paste / duplicate).
- Phase 8 — Cell-operation undo/redo.
- Phase 9 — External file sync (watch the .ipynb and reload on change).
- Phase 10 — Sequential multi-cell execution with stop-on-error.
- Phase 11 — Run always returns to command mode.
- Phase 12 — Create & open new notebooks ("New Jupyter Notebook").
- Phase 13 — Per-cell hover/selection toolbar.
- Phase 14 — Notebook data safety: save-conflict guard + reload affordance.
- Phase 15 — Cell output & execution-state management (clear outputs, counter
  reset on restart).
- Phase 16 — Better DataFrame (table) output rendering.
- Phase 17 — Execution status & queue correctness (pending vs running, batch
  supersede, per-cell timing).
- Phase 18 — In-cell execution status display (VS Code style).
- Phase 19 — Configurable post-run landing mode (shift-enter / ctrl-enter).
- Phase 20 — Per-cell scoped stop / interrupt.
- Phase 21 — Live elapsed-time counter while a cell runs.
- Phase 22 — Multi-select cells (shift/ctrl gestures; actions over the
  selection).
- Phase 23 — Collapse / expand cell input & output (persisted to the .ipynb).

## Fixed bugs (confirmed)

- #1 — Restart kernel killed the kernel but the relaunch failed.
- #2 — Running a cell with a dead/shutdown kernel didn't start the kernel.
- #3 — Interrupt / stop button didn't interrupt a running cell (OS-level
  interrupt; C-blocking-call interrupt on Windows remains a backlog item).
- #4 — "More options" toolbar button opened nothing.
- #5 — Output "…" (ellipsis) button next to cell output did nothing.
- #8 — Clean kernel exit left stale RunningKernel state.
- #10 — Kernel picker didn't accept Enter to select.
- #11 — Kernel-select prompt: wrong cell state on dismiss vs. select.
- #13 — After adding a cell with `a`/`b`, Enter sometimes wouldn't enter edit
  mode (add-cell now stays in command mode).
- #19 — Reloading didn't clear the conflict notification toast.
