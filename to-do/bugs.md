# Bugs

Status values: `open` | `fix attempted - untested` | `fixed - confirmed`.
Never attempt a further fix while a bug is `fix attempted - untested`.
Move to `to-do/archive/` only when `fixed - confirmed`.

---

(Bug #1 "Restart kernel kills but relaunch fails" — fixed & confirmed
2026-07-08, moved to `archive/bugs-fixed.md`.)

(Bug #2 "Running a cell with a dead kernel does not start it" — fixed &
confirmed 2026-07-10, moved to `archive/bugs-fixed.md`.)

(Bug #3 "Interrupt does not stop a running cell" — fixed & confirmed 2026-07-10
via the OS-level interrupt; a pure-Python loop interrupts immediately. Immediate
interrupt of C-blocking calls like `time.sleep` on Windows remains a backlog
item. Moved to `archive/bugs-fixed.md`.)

(Bug #4 "More options button opens nothing" — fixed & confirmed 2026-07-08
via phase 4's popover menu; moved to `archive/bugs-fixed.md`.)

(Bug #5 "Output ... button does nothing" — fixed & confirmed 2026-07-08 via
phase 4's output menu; moved to `archive/bugs-fixed.md`.)

## 6. Native kernel launch is flaky on Windows (os error 10054)

- **Status:** fix attempted - untested
- **Symptom:** "Kernel Error: cell could not be executed — the kernel failed
  to launch: handling failed for recv task: control recv: Codec Error: An
  existing connection was forcibly closed by the remote host. (os error
  10054)"
- **Analysis:** Message chain fully traced: control-socket read failure
  (`kernels/mod.rs:144-147`) → recv task bails → `kernel_errored`
  (`mod.rs:190-197`) → `Kernel::ErroredLaunch` → error rendered by
  `execute_cell`/`show_kernel_error`. Root cause: the native launch path
  (`native_kernel.rs:112-255`) connects to the kernel's sockets immediately
  after spawn with no readiness wait and no premature-exit check — the WSL
  path (`wsl_kernel.rs:290-323`) does both (2s wait + `try_status()` +
  stderr capture). Also aggravated by the restart races in bug #1.
- **Fix attempted:** Native kernel launch now waits 500ms after spawning and
  checks for premature process exit, reporting the kernel's stderr in the
  error message (mirrors the WSL path). Connection files are unique per
  launch so a dying old kernel can no longer delete the new kernel's file.
  The stale-message-task fix under bug #1 also removes the main source of
  spurious `ErroredLaunch` states after restarts. A fixed sleep is a partial
  measure — a kernel_info/heartbeat readiness handshake is a possible
  follow-up if 10054 persists (noted in backlog).
- **Tested:** no — needs user confirmation on Windows

## 7. Restart does not clear per-execution state

- **Status:** fix attempted - untested
- **Symptom:** (found in code review, not user-reported) After a restart,
  stale `msg_id → CellId` entries linger in `execution_requests`; incoming
  messages could be routed to old cells.
- **Analysis:** `restart_kernel` doesn't clear `self.execution_requests`;
  `change_kernel` does (`notebook_ui.rs:486`).
- **Fix attempted:** `restart_kernel` now clears `execution_requests` (and
  stops executing-cell spinners); `kernel_errored`/`kernel_exited` clear it
  too.
- **Tested:** no — needs user confirmation

(Bug #8 "Clean kernel exit leaves stale RunningKernel state" — fixed &
confirmed 2026-07-08, moved to `archive/bugs-fixed.md`.)

## 9. Notebook never reports itself dirty

- **Status:** open
- **Symptom:** (found in code review) Structural/metadata changes (add/move
  cells, outputs) don't mark the notebook modified, so closing may not
  prompt to save.
- **Analysis:** `NotebookItem::is_dirty` (the ProjectItem impl) is hardcoded
  `false` with a TODO (`notebook_ui.rs:1921-1924`). NOTE (phase 4): the
  workspace `Item::is_dirty` for `NotebookEditor` (`notebook_ui.rs:2200`)
  already returns `has_structural_changes() || has_content_changes()`, so
  cell add/delete/move/convert and edits DO mark the tab dirty and prompt on
  close. The remaining `NotebookItem::is_dirty` stub may be dead/irrelevant —
  confirm whether anything consults it before "fixing" it.
- **Fix attempted:** none
- **Tested:** n/a

(Bug #10 "Kernel picker does not accept Enter" — fixed & confirmed 2026-07-08
via the NotebookCellEditor keymap scoping; moved to `archive/bugs-fixed.md`.)

(Bug #11 "Kernel-select prompt: cell state on dismiss vs. select" — fixed &
confirmed 2026-07-08, moved to `archive/bugs-fixed.md`.)

## 12. "Clear all outputs" sometimes needed several presses

- **Status:** open (not reproduced)
- **Symptom:** (user 2026-07-08) One instance where "Clear all outputs" had to
  be pressed ~5 times before it worked. Not reproducible so far.
- **Analysis:** none yet. Low-priority note; investigate only if it recurs.
- **Fix attempted:** none
- **Tested:** n/a

(Bug #13 "After adding a cell with `a`/`b`, Enter won't enter edit mode" —
fixed & confirmed 2026-07-10 (`a`/`b` now stay in command mode); moved to
`archive/bugs-fixed.md`. The broader full-focus-loss edge remains as bug #15.)

## 15. Notebook keyboard shortcuts get stuck (focus/mode desync)

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-08) Intermittently, command-mode shortcuts
  (delete, add, convert, copy/paste, undo) stop firing. In the stuck state
  there is NO cursor and NO characters appear (so it is NOT edit mode), and
  pressing Escape does not recover it. Recovery required selecting another
  cell, pressing Escape, navigating back, then the shortcut worked. Also seen:
  `b` sometimes doesn't even select/focus the new cell (stays on the old one),
  other times it does. So it's a focus/mode desync, not a specific action bug —
  it's very likely the underlying cause of the "undo/copy/paste not working"
  reports too (the `z`/`c`/`v` keys simply don't dispatch when stuck).
- **Analysis:** command-mode keybindings require BOTH `notebook_mode == command`
  AND the notebook (or a descendant) focused. These can desync: `select_cell_by_id`
  sets `Edit` on any editor focus, and focus can be lost entirely (e.g. after
  clicking toolbar buttons / popovers) with nothing bringing it back — Escape
  is only bound in the notebook contexts, so if focus is fully off the notebook
  it can't recover. Needs runtime debugging to pin the exact focus-loss
  trigger(s).
- **Fix attempted (2026-07-08), partial/mitigations:**
  1. `cx.on_focus` on the notebook root handle → force `notebook_mode = Command`
     whenever the notebook itself gains focus, keeping mode synced to focus.
  2. `a`/`b`/+ now stay in command mode (bug #13) so adding a cell no longer
     drops you into edit unexpectedly.
  3. Bound Escape → `EnterCommandMode` in the base `NotebookEditor` context
     (not just edit mode) as a recovery path — works whenever the notebook is
     still in the focus chain.
  These may not fully fix the "focus fully lost" case (where nothing in the
  notebook is focused); that likely needs returning focus to the notebook after
  toolbar-button / popover interactions. Kept OPEN.
- **Update (user 2026-07-09):** "Your fix on focus seems to have worked, mark
  that down but keep it open as an ongoing observation." The on_focus mode-sync
  mitigation appears effective; keeping the bug OPEN as an ongoing observation
  in case the "focus fully lost" edge recurs. If it does, capture the exact
  action that dropped focus (the reporter wasn't sure of the trigger this time).
- **Tested:** partially — the on_focus fix seems to work; kept open as an
  ongoing observation for the rarer full-focus-loss case.

## 14. Saving overwrites external changes without warning

- **Status:** open
- **Symptom:** (user 2026-07-08) If the .ipynb changed on disk while open in
  Zed and you have unsaved notebook changes, saving from Zed silently
  overwrites the on-disk (external) changes with no warning/confirmation.
- **Analysis:** Phase 9 surfaces the conflict on the RELOAD side (a toast when
  the file changes on disk under unsaved changes) but the SAVE side
  (`Item::save`, `notebook_ui.rs`) unconditionally `fs.atomic_write`s. It needs
  a save-time conflict check: before writing, compare the on-disk content (or
  the retained buffer's `has_conflict` / mtime) against what we loaded, and if
  it changed, prompt to overwrite / cancel / diff. Requires a confirm dialog,
  so it's its own chunk of work. Data-loss risk → medium-high.
- **Fix attempted (2026-07-11, phase 14):** a `disk_changed_externally` flag is
  set when the external-change conflict is detected (the phase-9 toast path);
  while set, `Item::save` shows an Overwrite/Cancel prompt and only writes on
  Overwrite. Flag clears on reload or confirmed overwrite. A Reload button on
  the conflict toast + a "Reload Notebook" command were added alongside.
- **Tested:** no — needs user confirmation (see phase 14 checklist)

## 16. Queued cell output misrouted when a run spans kernel selection/launch

- **Status:** open (needs runtime instrumentation)
- **Symptom:** (user 2026-07-10) With no kernel selected, the user ran cells 1
  and 2 (which brought up the kernel picker), selected a kernel, then ran cell
  3. Cells 1 and 2 ran, cell 2 errored — but cell 2's error output appeared in
  **cell 3's** output area, not cell 2's. Does NOT reproduce when the kernel is
  already selected before running; only when the queued cells straddle the
  kernel selection/launch boundary. User's hunch: "3 was queued before 1 had
  completed."
- **Analysis so far (what has been RULED OUT):**
  - Routing is by `parent_header.msg_id → execution_requests[msg_id] → cell`
    (`notebook_ui.rs` `route`), and each `ExecuteRequest.into()` gets a UNIQUE
    `msg_id` (jupyter-protocol `JupyterMessage::new` → `Uuid::new_v4()`), so a
    simple msg_id collision is not the cause.
  - The queued-cell flush (`launch_kernel` ready branch:
    `for cell_id in take(pending_executions) { execute_cell(cell_id) }`) sends
    in order and records `execution_requests.insert(msg_id, cell_id)` per
    iteration, so the recorded mapping looks correct on paper.
  - `promote_awaiting_cells` preserves order; the manually-run cell 3 is
    appended after, so the drain order is [1, 2, 3].
  - Not the batch/`run_queue` path (these were individual runs, so
    `active_run_cell` is None).
  - NOTE anomaly: `JupyterMessage::new` also generates a fresh `session` id PER
    message when there is no parent, so every execute request uses a DIFFERENT
    session. Unusual (normally one session per client); shouldn't cross-wire by
    msg_id, but worth checking whether iopub attribution is affected.
- **Leading remaining hypotheses:** (a) a race between `route()` handling
  incoming iopub messages and `execution_requests` being mutated during the
  rapid back-to-back flush; (b) the kernel attaching the wrong parent under the
  per-message-session quirk; (c) a display/index issue where the output lands
  on the cell at a stale index. Needs targeted logging of the actual
  `msg_id ↔ cell_id` map and the misrouted error's `parent_header` at runtime.
- **Fix attempted:** none directly (the obvious cause is ruled out; a
  speculative fix would be premature). HOWEVER, phase 17 (2026-07-11) made two
  changes to the same span-kernel-selection path that plausibly resolve or at
  least reshape this: (1) the focus-grab that dismissed the kernel picker and
  DROPPED awaiting cells is gone — cells queued before/after selection now
  survive in order; (2) outputs are now cleared and attributed at each cell's
  own `execute_input`, so an output can no longer land on a cell whose state
  was stale from the queue-time clear. Debug logging was added around
  promote/dismiss/flush (`log::debug`, scope `repl`) to capture the queue
  contents if it recurs.
- **Tested:** no — re-test the original repro (queue 2 cells pre-selection,
  1 post, second cell errors) after phase 17 lands.

## 17. One-off shift-enter focus jump on a brand-new notebook

- **Status:** open (not reproduced)
- **Symptom:** (user 2026-07-11) On the FIRST run in a freshly-created notebook:
  added an import line to the first cell, pressed shift-enter to execute. It
  advanced down, created a new cell (expected, since it was the last cell), then
  jumped focus BACK to the first cell. Could not recreate; a second attempt
  behaved normally.
- **Analysis:** likely a focus/selection race specific to the just-created
  single-cell notebook — `run_and_advance`'s last-cell branch does
  `add_code_block` (which selects the new cell in command mode via
  `add_code_cell_at` → `enter_command_mode`) and then `enter_command_mode`
  again; combined with the async execute/notify and the `cx.on_focus` mode-sync
  handler, selection may momentarily bounce. Overlaps with the phase-11 /
  bug #15 focus work. Watch for recurrence; capture the cell count and whether
  a kernel was attached if it happens again.
- **Fix attempted:** none
- **Tested:** n/a

## 18. Executing a notebook doesn't mark it dirty (external save discards runs)

- **Status:** open
- **Symptom:** (user 2026-07-11) Opened the same .ipynb in Zed and VS Code.
  In Zed, executed the first 3 cells but did NOT change any cell content. Edited
  the file in VS Code and saved. Zed auto-reloaded and LOST the execution
  state, because with no content edits Zed didn't consider itself dirty and so
  treated the disk version as authoritative. A notebook whose execution
  outputs / counts have changed should count as modified (VS Code stores
  execution metadata in the JSON, so the in-memory and on-disk versions do
  differ).
- **Analysis:** `is_dirty` = `has_structural_changes() || has_content_changes()`
  — neither accounts for outputs / execution_count having changed since load.
  The external-change handler (`handle_external_change`) auto-reloads when not
  dirty. Relates to phase 14 (save-conflict) and phase 9 (reload).
- **Fix attempted (2026-07-11):** new `execution_state_changed` flag on the
  notebook, OR'd into `Item::is_dirty`. Set whenever execution state mutates:
  submitting/queueing a run, kernel messages routed to a cell (outputs /
  counts), clear-cell-outputs, clear-all-outputs, and the restart counter
  reset. Cleared by `mark_as_saved` (save persists the state) and
  `reload_cells_from_notebook` (state now matches disk). Result: an
  executed-but-unedited notebook counts as dirty → an external save now shows
  the conflict toast instead of silently auto-reloading over the run results,
  and the tab dirty-dot/save-prompt reflect execution state too.
- **Refined (same day, from self-review before user testing):** (1) the flag
  is only set for kernel messages that change savable content (stream/display/
  result/input/error) — Status busy/idle broadcasts no longer re-dirty a
  just-saved notebook; (2) new `last_saved_disk_text` records exactly what we
  last wrote/loaded, and `handle_external_change` recognizes our OWN save by
  comparing against it BEFORE the dirty check — otherwise saving during a
  long-running cell (outputs arriving between the write and the file-watcher
  event) raised a spurious conflict toast for our own save.
- **Tested:** dirty tracking CONFIRMED by user 2026-07-11 (bug fixed). One
  follow-up defect noted → see bug #19 (an Overwrite save didn't dismiss the
  conflict toast). Otherwise confirmed.

(Bug #19 "Reloading doesn't clear the conflict toast" — fixed & confirmed
2026-07-12 (reload paths AND the overwrite-save path); moved to
`archive/bugs-fixed.md`.)

## 20. Kernel picker shows no kernels on a fresh app start

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-11) On a freshly-started app, the first
  Execute-All prompts for a kernel but the picker is EMPTY ("No matches"),
  even though two global Pythons (3.11.15, 3.11.14) and a workspace `.venv`
  exist. Escaping and running again does NOT prompt — it just starts (a kernel
  got resolved by then). So the kernel list simply hadn't loaded when the
  picker first opened.
- **Analysis:** the picker's entries are a static snapshot built at render
  time from `ReplStore.kernel_specifications` + discovered python toolchains
  (`build_grouped_entries` in `components/kernel_options.rs`). Both are
  populated ASYNCHRONOUSLY (`refresh_kernelspecs` / `refresh_python_kernelspecs`
  — pet toolchain discovery). `ensure_kernelspecs` kicks the refresh once, but
  if the picker opens before it completes the delegate captures an empty list
  and does NOT live-update when specs arrive (RenderOnce snapshot; the open
  Picker entity's delegate is fixed).
- **Fix attempted (2026-07-12):** the picker now observes `ReplStore`
  (`cx.observe_in` wired when the picker entity is built): whenever the store
  updates (async kernelspec/toolchain discovery completing), the delegate's
  entries are rebuilt from `build_grouped_entries` and `picker.refresh`
  re-applies the current query — so kernels stream into an already-open picker
  instead of it staying empty. (User note 2026-07-12: "Sometimes kernels do
  load in instantly" — intermittent, consistent with the async-discovery race.)
- **Tested:** no — needs user confirmation on a fresh app start

## 21. Own metadata save raises a spurious "changed on disk" toast

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-12) Collapsing a cell's code and then saving
  reproduces a rogue "This notebook changed on disk…" conflict toast, even
  though the only change was Zed's own metadata save. The save does NOT prompt
  to overwrite (correct — `disk_changed_externally` is false, no real external
  change), and saving does not dismiss the toast either. User's diagnosis: the
  file-change system detects Zed's own save as an external change and triggers
  the toast, while the save path correctly recognizes it as our own write.
- **Analysis:** `handle_external_change` guarded against our own save by
  comparing the reloaded buffer text against `last_saved_disk_text` with a
  plain `trim()` string equality. That guard is fragile: the buffer the file
  watcher reloads after our `fs.atomic_write` can differ byte-for-byte from
  what we wrote — final-newline handling, CRLF vs LF normalization, or JSON
  map key ordering (`metadata`/`jupyter.additional` are maps) — while being
  semantically identical. When the byte compare failed and the notebook was
  (or briefly appeared) dirty, the conflict toast fired for our own write.
- **Fix attempted (2026-07-14):** `handle_external_change` now compares CONTENT
  rather than text. It parses the on-disk notebook and compares it to the
  in-memory `to_notebook()` as `serde_json::Value`s (order-independent, so map
  key ordering and formatting differences are ignored). On a content match it
  treats the change as our own save: clears `disk_changed_externally`, updates
  `last_saved_disk_text`, and dismisses any stale conflict toast. The
  byte-identical `last_saved_disk_text` check is kept as a fast path; the
  conflict toast now only appears for a genuine content divergence while the
  notebook has unsaved edits.
- **Tested:** no — needs user confirmation (collapse a cell, save, confirm no
  toast; then have VS Code edit the file under unsaved Zed changes and confirm
  the toast still appears for a real external change)

## 22. Very fast cells show a ✓ but no execution time

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-12, screenshot) Cells that finish almost instantly
  (e.g. two cells run via Execute All) show the completed ✓ but no duration
  next to it. User's theory: they complete in <1ms.
- **Analysis:** The precise `execution_start_time` is set only in
  `begin_running`, which fires on the iopub `execute_input`. For a very fast
  cell the shell `ExecuteReply` (which calls `finish_execution`) can arrive
  BEFORE that `execute_input` — the two travel on separate channels — so
  `begin_running` is skipped (correctly, to avoid resurrecting a finished cell),
  leaving `execution_start_time` unset. `finish_execution` then finds no start
  time and records no duration, so the cell shows a ✓ with no time.
- **Fix attempted (2026-07-14):** added a `submitted_at` timing anchor recorded
  when the execute request is dispatched to a running kernel
  (`CodeCell::record_submitted`, called from the `Disposition::Sent` path).
  `finish_execution` now uses `execution_start_time.or(submitted_at)`, so a fast
  cell reports the send→reply duration (near-exact for a sub-millisecond cell)
  instead of nothing. `begin_running` still overwrites with the precise start
  when its `execute_input` arrives first; `mark_pending` / `cancel_execution` /
  `show_kernel_error` clear the anchor.
- **Tested:** no — needs user confirmation (run several instant cells; each
  should show a small ms duration next to the ✓)

## 23. Clicking a cell's gutter/margin doesn't select the cell

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-12, phase 22 feedback) Clicking the gutter/margin
  area of a cell (where the execute button sits, but not on the button itself)
  does NOT select that cell. To select a single cell with the mouse the user
  had to click the cell text (entering edit mode) and then press Esc. A plain
  click on the cell's border/gutter area should select it.
- **Analysis:** the shared capture-phase mouse-down handler on each cell root
  (`selection_modifiers`) only acted on clicks with a selection modifier held
  (shift → range, ctrl/cmd → toggle), emitting `ModifiedClick`. A plain click
  fell through with no handler, so clicking anywhere that wasn't the editor did
  nothing. Only the editor (via its focus → `FocusedIn` → `select_cell_by_id`)
  selected a cell, and that also forced edit mode.
- **Fix attempted (2026-07-14):** a plain left click on a cell root now emits a
  new `CellEvent::PlainClick`, handled by `handle_plain_click`, which selects
  just that cell (collapsing any multi-selection) and enters command mode. It
  does NOT stop propagation, so a click that lands on the editor still focuses
  it and enters edit mode (the editor's focus event fires after and wins);
  clicks on the gutter/margins/output select the cell in command mode.
- **Tested:** no — needs user confirmation (click a cell's gutter/margin →
  selects it in command mode without entering edit; click the editor text →
  still enters edit mode; shift/ctrl-click ranges still work)
