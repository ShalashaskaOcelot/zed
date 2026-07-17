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
- Phase 40 — UI polish (2026-07-16 round): notebook output blocks span the
  cell's full width (`max_columns` now only sizes the inline REPL); slimmer
  notebook top strip.
- Phase 42 — Kernel environment validation: a remembered kernel whose
  interpreter vanished is never launched — the run drops the stale selection
  and prompts; opening the picker re-runs discovery and prunes registered
  kernelspecs pointing at deleted envs; the indicator stops showing ghosts.
- Phase 33 — Truly unsaved "New Jupyter Notebook": the command now opens an
  untitled, in-memory notebook (no `Untitled-N.ipynb` on disk); first save
  goes through the save-as prompt, which attaches the notebook to its file
  (watcher, dedup, kernel memory) — file-browser-created notebooks unchanged.
  Untitled notebooks are not restored across restarts (v1).
- Phase 35 — Prompt interrupt of C-blocking calls on Windows: interrupts now
  send a real console CTRL_C on the kernel's hidden console (CPython only
  wakes main-thread C blockers like `time.sleep` for real signals — the JPY
  event's `interrupt_main()` merely trips the bytecode flag), with the old
  event as fallback; `interrupt_mode: "message"` kernelspecs get a
  control-channel interrupt_request instead. The planned CTRL_BREAK approach
  was researched and rejected (kills handler-less kernels). `9acd68c254`
- Phase 39 — Conda environment creation from the kernel picker: when a conda
  frontend (conda/mamba/micromamba) is on PATH, the "Create Python
  Environment" prompt offers "Create Conda Env…", which prompts for a name
  (new `EnvNameModal`), runs `<frontend> create -y -n <name> python
  ipykernel`, resolves the interpreter via the frontend itself (not a guessed
  path), registers a kernelspec, and selects it. The venv fast path is
  unchanged; the shared finalize/kernelspec-registration tail is factored
  into `finalize_env_creation` + `sanitize_kernel_name`.
- Phase 38 — Cell structure operations: split cell at the cursor
  (`ctrl-shift--` in edit mode; top half keeps id/metadata, bottom gets a
  fresh identity, execution records cleared, one undo group), join cells
  (`shift-m`: selected+below or a contiguous multi-selection, same-type
  only, sources joined with a blank line, outputs cleared, one undo group),
  and sidebar "Run cells above" / "Run cell and below" buttons.
- Phase 37 — In-place selectable output text: stream/plain/error outputs
  (the terminal-rendered ones) support mouse selection directly in the
  notebook — drag/double-click on the output canvas drives the terminal's
  own selection machinery, highlight painted like the terminal's, ctrl/cmd-c
  copies the selected text (cell copy when nothing is selected; cut is
  always cell-level), click-away deselects. Idle outputs skip the terminal
  sync so unselected notebooks render as cheaply as before.
- Phase 43 — Release & distribution discovery: mapped Zed's Windows
  packaging (Inno Setup 6 via `script/bundle-windows.ps1`; the installer is
  ALREADY user-level — `PrivilegesRequired=lowest`, HKCU-only; unsigned
  builds work with `CI` unset), auto-update (custom `{version,url}` feed at
  `{server}/releases/{channel}/{version}/asset`; single choke point
  `get_release_asset`; no integrity checks; `dev` channel never updates),
  and fork divergence (111 commits / 37 files since 2026-07-08; dry-run
  upstream merge clean; hotspots ranked). Full findings: `8f9a5d3dc5`.
  Produced plan phases 44 (local installer build), 45 (Gitea-fed
  auto-update), 46 (upstream-merge playbook), 47 (Drone pipeline).
- Phase 48 — Select a newly-created kernel immediately: creating a venv/conda
  env while another kernel is selected now makes the new env the notebook's
  pending selection right away (top strip shows it, Starting) and HOLDS runs
  (via the awaiting-kernel path) instead of running them on the old kernel;
  on build success the new kernel launches and the held cells run on it, on
  failure the selection reverts and the cells return to Idle. An explicit
  kernel pick during the build supersedes it. Picker greyed-entry split to
  phase 49.

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
- #32 — Notebooks reported dirty immediately upon opening (redundant
  `set_text` bumped every cell buffer's version at load). `06d8d45`
- #30 — An explicit kernel pick in one notebook changed other notebooks'
  kernels; picks are now per-notebook. `d2e9c63`
- #31 — A stale (deleted) kernel was retried forever; a failed launch now
  re-prompts with the picker on the next run. `138bc0c`
- #35 — Closing a notebook left its kernel process running (entity cycle
  kept the editor alive; kernels now hold weak session handles). `74ec3fd`
- #36 — Custom-location venvs vanished from the picker after restart; they
  are now registered as per-user Jupyter kernelspecs at creation. `ffdde0e`
- #37 — Table outputs lacked "Open in Buffer" and the output menu's copy
  skipped them; both now use the table's markdown text. `f6bd2d6`
- #38 — A failed kernel launch marked the cell "Cancelled" instead of the
  red ✕ failed state. `fc5e84b`
- #39 — CLOSED, not a Zed defect (investigated with on-disk JSON): Zed
  writes/reads VS Code's exact dotted `metadata.execution` keys, but VS
  Code itself does not persist execution times to the file (it displays
  from internal workspace state), so times can't round-trip into VS Code,
  and a VS Code run carries Zed's older timestamps forward unchanged.
- #33 — Clearing outputs left the [N] execution number and ✓/✕ status; now
  the whole run record clears. `cbea20f`
- #48 — Running a cell with no kernel selected (the prompt path) crashed the
  app with a GPUI double-lease panic; the kernel picker open is now deferred
  via `window.defer` so `on_open`'s re-entrant update holds no outer lease.
  `ff16a00`
- #40 — Upward command-mode navigation onto an already-visible cell sometimes
  bottom-pinned it (a gpui `ListState::scroll_to_reveal_item` index-vs-pixel
  guard); the reveal now compares pixel offsets and only scrolls to reveal an
  off-screen edge. Confirmed 2026-07-16.
