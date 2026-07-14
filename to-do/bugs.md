# Bugs

Status values: `open` | `fix attempted - untested` | `fixed - confirmed`.
Never attempt a further fix while a bug is `fix attempted - untested`.
When `fixed - confirmed`, add a one-line entry to `CHANGELOG.md` and DELETE the
bug's entry here (there is no archive dir; the CHANGELOG + commit is the record).

---

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

## 12. "Clear all outputs" sometimes needed several presses

- **Status:** open (not reproduced)
- **Symptom:** (user 2026-07-08) One instance where "Clear all outputs" had to
  be pressed ~5 times before it worked. Not reproducible so far.
- **Analysis:** none yet. Low-priority note; investigate only if it recurs.
- **Fix attempted:** none
- **Tested:** n/a

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

## 25. Adding a cell at the viewport bottom doesn't scroll it into view

- **Status:** open (first fix failed; root cause now confirmed in code)
- **Symptom:** (user 2026-07-12, phase 22 feedback) Adding a cell below the
  bottom-most cell while scrolled to the very bottom of the notebook inserts
  the new cell off-screen — the viewport does not scroll down to reveal it.
  The bottom status bar may also be obscuring the viewport's lower edge.
- **First fix FAILED (user 2026-07-14):** removing the `.h_full()` from the
  content row (candidate "layout overlap") did not help — new cells still land
  behind the bottom bar.
- **Root cause (confirmed in gpui code, 2026-07-14):** candidate #2 was right.
  `ListState::splice` inserts the new item as `ListItem::Unmeasured` with no
  size hint, so it contributes ZERO height to the list's sum-tree.
  `scroll_to_reveal_item(ix)` — called synchronously right after the splice —
  computes `bottom` from those heights, so the goal scroll puts the new item's
  TOP exactly at the viewport's bottom edge: zero pixels of it visible, i.e.
  "behind the bottom bar". The item only gets measured during the NEXT layout
  pass (it is within the list's 1000px overdraw), by which time the reveal has
  already run short. Additionally, `Window::on_next_frame` callbacks run at the
  START of the next frame tick — BEFORE that frame's layout — so a single-frame
  deferral still sees height 0; it takes two hops (frame N's draw measures the
  item; a callback at frame N+1's start reveals with the real height).
- **Fix plan (second attempt):** give `insert_cell` access to `window`/`cx`
  and, after the synchronous best-effort reveal, schedule a two-frame deferred
  re-reveal (`window.on_next_frame` twice) that runs after the new item has
  been measured, then notify the view so the corrected scroll paints. Covers
  every insertion path (add above/below, paste, duplicate, undo/redo).
- **Tested:** n/a — second fix not yet implemented

## 26. A cell that errors shows a completed ✓ instead of a failure marker

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-14) Ran a cell that failed with an ImportError
  ("cannot open shared object file"); the traceback rendered, but the cell got
  the completed ✓ tick rather than a failure ✕.
- **Analysis:** deliberate-but-wrong current behaviour, not a regression: the
  phase-17 state machine only distinguishes Finished from Cancelled. The
  KeyboardInterrupt→Cancelled fix (051c3c6) covered INTERRUPTED cells; a real
  error's `ExecuteReply(status: Error)` still calls `finish_execution` → ✓
  (`handle_message` even has a comment "Real errors keep the finished ✓").
  The user expects VS Code semantics: an errored cell gets a failure marker.
- **Fix attempted (2026-07-14):** added a `Failed` variant to
  `CellExecutionStatus`. `ExecuteReply(Error)` → new `fail_execution()`
  (shares `complete_execution` with finish: records the duration, terminal;
  does NOT override Cancelled — an interrupted cell's reply also reports Error
  after the KeyboardInterrupt iopub message, and must stay a muted ✕).
  Failed renders as a red ✕ + duration; Failed added to `begin_running`'s
  monotonic guard so a late `execute_input` can't revive a failed fast cell.
  Stop-on-error batch handling unchanged (it already keyed off the reply).
- **Tested:** no — needs user confirmation (run a cell that raises → red ✕ +
  time, traceback below; interrupt a running cell → still the muted ✕
  "Cancelled"; successful cells still ✓)

## 27. One-off "changed on disk" toast on save (post-#21 fix)

- **Status:** open (not reproduced)
- **Symptom:** (user 2026-07-14) With the notebook open ONLY in Zed, one save
  raised the "notebook changed on disk — reload or overwrite" toast. Closing
  and reopening Zed cleared it and it did not recur. No other info available.
- **Analysis:** none yet. Bug #21's content-compare guard is confirmed working
  for the metadata-save repro, so this is a different (or racier) path — e.g.
  outputs arriving between serialize and the watcher event making the
  in-memory notebook genuinely differ from the just-written disk state, which
  would fail the value-equality guard and hit the is_dirty branch. WATCH ITEM:
  if it recurs, capture what was running/dirty at the time of save.
- **Fix attempted:** none
- **Tested:** n/a

## 28. Cells flash/stick "Cancelled" around kernel selection

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-14) Two related wrongs around the kernel picker:
  1. Escaping the picker (no selection) leaves the triggering/queued cells
     showing "Cancelled" — they should return to their idle state (arguably
     they were never really queued to a kernel at all).
  2. Even when a kernel IS selected: between pressing Enter and the kernel
     starting, all batch cells briefly show "Cancelled", then jump back to
     Pending/Running as the queue engages. (Open Zed → Run All → pick kernel →
     everything flashes Cancelled → kernel starts → statuses correct.)
- **Analysis (confirmed in code):**
  - The flash (2): `change_kernel` calls `stop_executing_cells(cx)`
    UNCONDITIONALLY, which cancels every Pending/Running cell — including the
    whole batch that is waiting on this very kernel choice. The batch's
    `run_queue` is deliberately preserved (the `cells_awaiting_kernel_choice`
    guard covers `cancel_run_queue`) but the cells' STATUSES are wiped to
    Cancelled, and each only re-Pends when the queue reaches it after the
    kernel starts — exactly the observed flash.
  - The Escape case (1): the picker's dismiss callback (`clear_awaiting_cells`)
    marks awaiting cells and the run queue Cancelled via `cancel_execution` /
    `cancel_run_queue`. For cells that were never submitted to any kernel,
    Idle (no marker) is the right end state, not Cancelled.
  - (On selection the dismiss callback is a no-op — `change_kernel` runs first
    and drains `cells_awaiting_kernel_choice`, so the `if !empty` guard is
    false. The flash comes solely from `stop_executing_cells`.)
- **Fix attempted (2026-07-14):** (a) `change_kernel` now scopes
  `stop_executing_cells` under the same `cells_awaiting_kernel_choice.is_empty()`
  guard as `cancel_run_queue` — a selection that satisfies a waiting run keeps
  every batch cell Pending straight through kernel startup (no Cancelled
  flash); a deliberate mid-run kernel SWITCH still cancels in-flight work.
  (b) dismissing the picker without selecting now returns the awaiting cells
  AND the queued batch to IDLE (new `CodeCell::reset_execution_status` +
  `abandon_run_queue`) instead of marking them Cancelled — nothing was ever
  submitted, so no marker. Kernel restart/interrupt/error paths still use
  `cancel_run_queue` (Cancelled) — no frozen queues (`abandon_run_queue` also
  clears `active_run_cell`/`resume_run_queue_on_idle`). The user's alternative
  (a queue-level status that engages at kernel start) wasn't needed once the
  statuses stay correct.
- **Tested:** no — needs user confirmation (Run All with no kernel → Escape the
  picker → cells show NO status marker, not Cancelled; Run All → pick a kernel →
  cells stay Pending through startup with no Cancelled flash, then run; Restart
  Kernel mid-batch still cancels the queue)
