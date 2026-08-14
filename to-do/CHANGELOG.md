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
- Phase 49 — Creating kernel shown in the picker: while an env is building
  (phase 48), the kernel picker shows a greyed, non-selectable "Creating
  <name>…" row as the current selection (checkmark on it, none on the old
  kernel); it's re-injected on the store-observer rebuild so a mid-build
  refresh doesn't drop it, and it's replaced by the real selectable kernel
  entry once the build completes. New `KernelPickerEntry::Creating` variant +
  `KernelSelector::with_creating`.
- Phase 44 — Local Windows installer build: made `script/bundle-windows.ps1`
  work off GitHub Actions — guard the `>> $env:GITHUB_ENV` append (unset on a
  plain machine/Drone runner, where the redirect aborted the script after a
  successful compile), fix the stale `-Install` launch path to the real
  `target/Zed-<arch>.exe`, and discover the VS 2022 install via `vswhere`
  (with a Community/Professional/Enterprise/BuildTools filesystem fallback)
  instead of hardcoding the Community edition. Added
  `docs/fork/windows-installer-build.md` documenting the reproducible
  user-level unsigned build. Windows-machine verification tracked in
  `awaiting_testing.md`.
- Phase 50 — Go to / Follow running cell: `notebook::GoToRunningCell` action +
  sidebar Crosshair button reveals, selects, and top-aligns the executing cell
  (greyed when idle); `notebook::ToggleFollowRunningCell` + sidebar Eye toggle
  viewport-scrolls to each cell as a batch run advances, without touching
  selection or edit/command mode. Both back onto a shared `running_cell_index`.
  User-confirmed working; the follow-scroll landing-position tweak became
  phase 51. `1516463c03`
- Phase 51 — Follow running cell pins near the top: new
  `ListState::scroll_to_item_near_top` anchors follow mode on an item boundary
  (the highest preceding item that fits within a viewport-scaled margin, else
  the running cell itself), so the running cell settles near the top with short
  items as context while a tall preceding markdown/output is simply not shown
  rather than pushing it down (immune to remeasurement above the anchor).
  Replaces the minimal reveal that landed each running cell on the bottom edge.
  User-confirmed working. `2e255cda3e`
- Phase 52 — Notebook session persistence: `NotebookEditor` implements
  `workspace::SerializableItem` (new `NotebookDb` sqlez module), so open
  notebooks restore with the workspace session — saved ones reopen by path,
  untitled ones round-trip their nbformat JSON and come back untitled with cells
  and outputs intact, and an untitled notebook is silently kept on close (no
  save prompt) like an unsaved buffer. User-confirmed working. `4f8238ff8f`.
  (Deserialize robustness against a vanished restore path is tracked as bug #50,
  not this phase.)

- Phase 49 — Kernel picker shows an env being created as a greyed,
  non-selectable "Creating <name>…" row at the top, checkmarked as the pending
  selection. Confirmed working 2026-07-30 (live refresh on completion filed as
  bug #58; visual polish backlogged).

- Phase 54 — Smooth mouse-wheel scrolling in the editor: new
  `editor.smooth_scrolling` setting (default on, VS Code's
  `editor.smoothScrolling` mapped on import); ScrollManager eases toward a
  target with frame-rate-independent exponential decay, successive ticks extend
  the pending target, and any other scroll (scrollbar drag, go-to-line,
  autoscroll) supersedes it. Trackpad left untouched. Keyboard scroll actions
  split out as phase 58 (vim reads the scroll position back synchronously).
  `01f09be`. Confirmed working on a mousewheel 2026-07-30. GUI settings entry
  backlogged.

- Phase 53 — Saving a file outside the project no longer adds it to the
  workspace: out-of-project save-as creates an INVISIBLE single-file worktree,
  so the file stays open and re-saveable without becoming a project-panel root.
  Applies to all file types. `a26a766`. Confirmed 2026-07-30.
- Phase 55 — Smooth mouse-wheel scrolling in the notebook cell list: opt-in on
  gpui's `ListState` (off by default, so no other list changes), driven by a
  pending pixel delta that decays to zero rather than an absolute ListOffset —
  so it needs no item-height math and cannot regress reveal accuracy. Absolute
  scrolls cancel a glide in flight. `e4bb183`. Confirmed 2026-07-30 (easing
  values still to be tuned — backlogged).
- Phase 57 — In-notebook Ctrl-F: `NotebookEditor` implements `SearchableItem`,
  delegating each primitive to the per-cell editors (Match = cell + range), so
  matches highlight across all cells, next/prev cycles them in document order,
  and activating one selects + reveals the owning cell. `2915bbd` (+ `45cb601`
  fixing an invalidation loop). Confirmed working 2026-07-30.

- Phase 59 — Truncate long cell output from the TOP (VS Code style): an opt-in
  pin-to-top mode on `TerminalOutput` keeps the viewport at the START of the
  content instead of following the tail, so a long output shows its head plus a
  muted notice naming the hidden line count. The terminal is still fed
  everything, so scrollback — and hence `full_text` / open-in-buffer — stays
  complete; only the viewport changes. The pin is re-applied after each append
  and after the canvas resize (a scroll is a queued event, and a resize reflows
  the grid and drops the display offset). Off by default, so the inline REPL
  keeps the console behaviour. Fixes bug #63. `2f2cd24`. Confirmed 2026-08-06
  (head shown, notice correct, open-in-buffer complete; also verified on a
  truncated traceback, so it composes with phase 60).
- Phase 60 — Cell error detection and "go to error" navigation: `GoToError`
  plus `NextError`/`PreviousError` (`f8`/`shift-f8`) over failed cells in
  document order with wrapping, and a clickable failure count in the pinned
  kernel strip. The reveal lands on the ERROR, not the cell top — new gpui
  `ListState::scroll_to_item_bottom_aligned` puts the item's bottom at the
  viewport bottom so a tall cell shows its traceback, degrading to the plain
  reveal when the cell already fits. Detection itself already existed
  (`CellExecutionStatus::Failed`); this phase is navigation and affordance
  only. `5e4ca4b`. Confirmed 2026-08-06: indicator, click-to-jump, the
  no-failures case and the caught-exception case all behave as designed. Two
  caveats carried forward rather than reopening the phase — `f8`/`shift-f8` did
  nothing for the user (→ bug #64), and the short-cell reveal wasn't
  specifically exercised.

- Phase 61 — Notebook runtime timers in the kernel strip: two opt-in timers
  (`repl.notebook_show_execution_time`, `repl.notebook_show_kernel_uptime`).
  The execution total is an ACCUMULATOR, not a sum of the cells' recorded
  durations — time is banked when a run ends, so it pauses between cells and a
  re-run notebook's stale per-cell durations can't inflate it; both timers are
  scoped to a kernel session and reset on launch/restart/switch, and a stopped
  kernel's uptime freezes rather than disappearing. `Cell::format_duration` was
  extracted into a shared `format_duration` (plus an hours tier) so the strip
  and the cell footers can't drift. The 100 ms tick is started from `render`
  and ends itself when no live timer is on screen. Confirmed 2026-08-06:
  restart zeroes both, Run All drives the exec total, it stops when the last
  cell finishes while uptime keeps running, a re-run resumes it, and an
  interrupt stops it. (Idle CPU cost not separately measured.)

- Phase 62 — Global kernel busy/idle indicator: the pinned kernel strip's
  status icon moved out of the kernel-selector button into its own cluster,
  spins (`with_rotate_animation`) while the kernel is Busy/Starting/Restarting/
  Shutting Down, and is labelled with the state so busy vs idle is readable
  from across the screen at any scroll position. The animation is derived from
  the current `KernelStatus` every render — no latched flag — so it always
  stops in a settled state, including the env-creation `Starting` override.
  Confirmed 2026-08-06 (Busy/Idle/Starting/Restarting all read correctly and
  the cluster stays pinned); the `Shutdown` label was then dropped at the
  user's request — the grey dot already says "not running".

- Phase 64 — Kernel picker and notebook control polish, five small items: the
  picker says "Searching for kernels…" while discovery is in flight (new
  `ReplStore::is_discovering_kernels`, backed by an in-flight flag on both
  refreshes) so "No matches" only appears once it is true; registered Jupyter
  kernelspecs show the INTERPRETER they launch (`argv[0]`, new
  `KernelSpecification::interpreter_path`) rather than nothing, since that is
  what identifies the venv; the "Creating…" row adopts the kernel rows' layout
  with a spinning icon; the sidebar's redundant kernel selector is gone (it
  shared a `PopoverMenuHandle` with the strip, so it opened the popover at the
  OTHER trigger); and every sidebar control now moves focus to the notebook
  unless focus is already inside it, so shortcuts work right after a click.
  AWAITING USER TESTING.

- Phase 63 — Wide notebook outputs and output-body interaction: a wide table
  (a pandas DataFrame arrives as HTML → markdown → `TableView`) was clipped,
  not merely unscrollable — its container already scrolled horizontally and its
  rows were already laid out at their natural width, but the bordered frame
  between them stretched to the container and its `overflow_hidden` cut off
  everything past the right edge. Sizing the frame to the rows
  (`min_w(total_width)`) makes the overflow real, and a tracked `ScrollHandle`
  gives it a horizontal-only scrollbar following the editor scrollbar setting.
  Plain text can never overflow (the terminal wraps at `max_columns`), which is
  why this lives on the table rather than on every cell's output block. Also:
  pressing anywhere in an output now selects its cell, on mouse-DOWN and
  without consuming the event, so drag-to-select-text still works.
  AWAITING USER TESTING.
- Phase 66 (CONFIRMED 2026-08-11) — Notebook file surfaces: save-as always yields an `.ipynb`
  (via a new `Item::adjust_save_as_path` applied before the worktree is
  created), and a global-search hit on a notebook opens the NOTEBOOK — the
  editor asks the project-item registry who claims a path instead of knowing
  about extensions. `PENDING`
- Phase 70 — Notebook keyboard and control corrections: cell navigation moved
  off the shared `menu::SelectNext`/`SelectPrevious` actions onto its own, which
  gives `ctrl-n`/`ctrl-p` back to New File / the file finder, and Run All now
  leaves edit mode. `PENDING`
- Phase 68 — Kernel strip states and the Run All timer reset: the strip has its
  own "Creating…" state instead of borrowing "Starting" while an environment
  builds, and `repl.notebook_reset_execution_time_on_run_all` makes Run All
  measure that one pass. `PENDING`
- Phase 69 — Notebooks behave like text files on disk: a loose notebook (one
  outside every project root) restores instead of vanishing, its tab strikes
  through when the file is deleted, and unsaved changes to a SAVED notebook
  survive a quit. `PENDING`
- Phase 67 — Page-wise follow mode: a second `repl.notebook_follow_mode`
  where the selection carries execution and the viewport moves a page at a
  time, clamped at the end of the notebook. `PENDING`

## Fixed bugs (confirmed)

- #78 — Three-digit execution counts (`[169]`) wrapped their closing bracket in
  the cell gutter; the gutter is wider and the count no longer wraps. Confirmed
  2026-08-11.

- #7 — Restart left stale `msg_id → CellId` entries, so output from a
  pre-restart execution could land on a cell after restarting. Confirmed
  2026-08-11.
- #66 — Clear Outputs was disabled for cells that ran but printed nothing;
  it now enables whenever there is any execution record and clears counts,
  times and ✓ markers. Confirmed 2026-08-11.
- #67 — A vertical wheel dragged wide tables sideways while the notebook
  scrolled (`restrict_scroll_to_axis`). Confirmed 2026-08-11.
- #68 — The wide table's horizontal scrollbar slid out from under the pointer
  when dragged; moved off the scrolling element onto a wrapper. Confirmed
  2026-08-11.
- #70 — Table columns didn't line up row to row; `flex_basis(0)` makes the
  split depend only on the per-column grow factors. Confirmed 2026-08-11.
- #72 — Kernel status stuck on "Starting" until a cell was run: a late
  `status: starting` broadcast overwrote the launched state. Confirmed
  2026-08-11. (Follow-up in the backlog: a RESTART should go straight to Idle
  rather than via Starting.)

- Stale unit test `test_last_session_restores_workspace_with_missing_paths`
  asserted the behaviour bug #50 deliberately reversed, so `cargo test -p
  workspace` had been RED since `cd2162f` (2026-07-23). Rewritten to assert
  the intended behaviour (missing roots dropped, workspace still restored).
  Confirmed by the suite itself — 220 pass. `PENDING`

- #63 — Long text output showed only its LAST ~32 lines: the terminal emulator
  behind plain output follows the tail like a console, so anything past
  `max_lines` scrolled off the top. Fixed by phase 59's pin-to-top mode plus a
  truncation notice. `2f2cd24`. Confirmed 2026-08-06.
- #20 — Kernel picker came up empty on a fresh app start: kernelspec/toolchain
  discovery is async and the picker's delegate captured a one-off snapshot, so
  an early-opened picker stayed empty forever. The picker now observes
  `ReplStore` and rebuilds its entries as discovery completes. Confirmed
  2026-08-06 — the list is empty only for the moment discovery is still
  running, then fills in. (The remaining gap, that "still searching" is
  indistinguishable from "no matches", is backlogged as its own item.)
- #53 — Notebook text output wrapped at a fixed 128 columns, well short of the
  block's right edge. Phase 40 widened the output CONTAINER but the terminal
  inside it was still sized from `max_columns`: the canvas sync took
  `terminal_size()`'s width and adopted only the ORIGIN from the element's real
  bounds. It now adopts the real laid-out width too; the inline REPL is
  unaffected because its container is already capped at `max_columns`.
  `e826b05`. Confirmed 2026-07-31.
- #56 — A notebook opened outside the workspace couldn't start any kernel
  ("The directory name is invalid", os error 267). The kernel's cwd came from
  the notebook's worktree, and a single-file worktree's `abs_path()` is the FILE
  — so a file was passed as a directory. Uses the file's parent directory now.
  `c199237`. Confirmed 2026-07-31. (The related limitation — workspace venvs not
  offered to such notebooks — remains open as bug #55.)
- #61 — A cell's stream output was split into one block per flush, each with its
  own copy button and selection boundary, so a print loop produced dozens of
  blocks and no drag could select across them. Consecutive stream messages now
  append to the trailing block, as Jupyter and the inline REPL already did.
  `4deb715`. Confirmed 2026-07-31.
- #62 — Edit-mode cursor movement across a cell boundary snapped the viewport,
  yanking the newly-entered cell to the top even when the target line was
  already visible. Edit-mode crossings now reveal only the destination line
  (and only when off-screen); command-mode cell navigation still reveals the
  whole cell top-aligned. `7f3906f`. Confirmed 2026-07-31.

- #60 — The notebook viewport chased the cursor: clicking an already-visible
  line jumped it several lines, a click-drag near an edge scrolled under the
  held pointer until the whole cell was selected, and arrow keys could walk the
  cursor out of view after which the follow never recovered. Two causes: the
  editor asks its container to reveal `cursor_row-3..+4` whenever a selection is
  pending (narrowed to the cursor's own line for notebook cells via
  `Editor::set_minimal_container_autoscroll`, default off elsewhere), and the
  notebook's own follow used a margin plus a position taken from PAINT — absent
  once the cursor left the viewport. The follow is now derived from
  `Editor::last_bounds()` (layout, not paint) with no margin.
  `79f116f` `324c165` `a12c539`. Confirmed 2026-07-30.

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
- #47 — Saving a notebook outside the workspace failed with "no such worktree"
  (while still writing the file) and left the tab titled "Untitled": phase 53's
  invisible worktree is held only weakly, and `Pane::save_item`'s strong handle
  dropped before the save completed, so the post-write `open_buffer` failed and
  the `?` skipped recording the new path. The worktree is now held across the
  save. `a2009b4`. Confirmed 2026-07-30.
- #54 — Clicking the notebook sidebar's kernel selector aborted the app with a
  GPUI double-lease panic (the click listener held a lease and toggled the
  picker inline, whose on_open updates the same notebook); the toggle is now
  deferred with `window.defer`. `a2009b4`. Confirmed 2026-07-30.
- #46 — Interrupting the Rust kernel made the next run prompt for a kernel
  instead of relaunching the selected one. Confirmed 2026-07-30.
- #51 — Run All (and the other notebook control buttons) silently did nothing
  until the notebook had focus: the buttons used `window.dispatch_action`, which
  routes to the FOCUSED element, so opening a notebook from the project panel
  left the action going to the panel. The buttons now call their own
  NotebookEditor's methods via `cx.listener`. `a925b8b`. Confirmed 2026-07-30.
- #45 — End and "Go to running cell" landed short in large notebooks: the
  cumulative-height reveal derived the scroll offset from the summed height of
  cells above the target, which is wrong when those cells are unmeasured
  (0 px) or grew outputs off-screen. Routed the two far jumps through the
  index-anchored primitives instead — End uses `scroll_to_end`, Go to running
  cell uses `scroll_to_item_near_top` — immune to heights above the target.
  `3e96779`. Confirmed 2026-07-23.
