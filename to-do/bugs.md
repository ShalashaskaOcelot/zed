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

## 29. Notebook kernel matching never works for WSL-authored notebooks

- **Status:** open (low priority; pre-existing, surfaced by the phase-25 review)
- **Symptom:** (adversarial code review, 2026-07-14) A notebook authored in
  JupyterLab inside WSL saves `metadata.kernelspec.name = "python3"`, but the
  saved-kernel matching (both `remembered_kernel_spec`'s fallback from phase 6
  and phase 25's pre-selection) can never match the WSL kernel — and a
  same-named Windows-local kernel can silently steal the match instead.
- **Analysis:** `KernelSpecification::name()` (`kernels/mod.rs`) returns the
  DISPLAY name (e.g. "Python 3 (ipykernel)") for the `WslRemote` variant only;
  every other variant returns the real kernelspec name. So `"python3"` never
  equals the WSL spec's `name()`. Notebooks round-trip fine WITHIN Zed (we
  write `spec.name()` back into the metadata on launch), but externally-authored
  WSL notebooks miss, and a Windows-local kernel dir named `python3` matches
  first. Fix direction: make `WslRemote`'s `name()` return the kernelspec dir
  name like the other variants — but audit the display sites first (labels use
  `name()` too), or match on both name fields.
- **Fix attempted:** none
- **Tested:** n/a

## 30. Explicit kernel pick in one notebook changes other notebooks' kernels

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-14) "Selecting a different kernel in one notebook
  still changes the kernel for other notebooks also." Expected: per-notebook.
- **Analysis:** the adversarial-review fix made METADATA adoption per-notebook,
  but an EXPLICIT pick still goes through `change_kernel` →
  `ReplStore::set_active_kernelspec`, which is keyed BY WORKTREE — and
  `remembered_kernel_spec` consults that store selection BEFORE the notebook's
  own metadata. So a pick in notebook A wins over B's saved kernel.
- **Fix attempted (2026-07-14):** explicit picks are per-notebook: the store
  gains `selected_kernel_for_notebook` keyed by the notebook's absolute path;
  `change_kernel` writes that (never the worktree entry), and
  `remembered_kernel_spec` resolves own-spec → per-notebook memory → own
  metadata. Cross-session persistence flows through the notebook's own
  kernelspec metadata (written at launch). The kernel picker rendered from a
  notebook now shows THAT notebook's kernel as selected (new
  `KernelSelector::with_selected` override) instead of the worktree selection,
  which now belongs solely to the inline `.py` REPL.
- **Tested:** no — the core repro (same-session cross-notebook contamination)
  has not been retested yet. **Finding (user 2026-07-16):** closed Zed with
  different kernels selected on two notebooks; on relaunch NEITHER showed a
  kernel selected. Cross-session persistence flows through the notebook's own
  kernelspec metadata, which is only written at LAUNCH and only persisted when
  the notebook is SAVED — and with lazy start the indicator stays empty until
  the first run even when a kernel IS remembered. So the on-launch blank
  indicator may be expected; what matters is whether the first RUN uses the
  remembered kernel without prompting, and whether a pick in one notebook
  still leaks into the other within a session. Refreshed test recipe in
  `awaiting_testing.md` covers both.

## 31. A stale (deleted) kernel is retried forever instead of re-prompting

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-14) Delete the kernel env while the notebook is
  closed but Zed stays running; reopen the notebook and run → "Kernel error:
  cell could not be executed" (launch failure), and every subsequent run
  retries the same dead kernel. (Deleting the env with Zed CLOSED works —
  fresh discovery doesn't list it and the picker correctly appears.)
- **Analysis:** the deleted env's spec survives in the store's cached
  discovery list for the session, so metadata matching adopts it. After the
  launch fails (`Kernel::ErroredLaunch`), `execute_cell`'s disposition for
  ErroredLaunch-with-remembered-kernel is `Queued { launch: true }` — an
  endless retry loop with the same broken spec, and the picker (which would
  offer alternatives and refresh discovery) is never shown.
- **Fix attempted (2026-07-14):** after a failed launch the next run PROMPTS:
  `Kernel::ErroredLaunch` is split out of the `Shutdown` disposition arm and
  always yields `Prompt`, and the Prompt branch now opens the picker DIRECTLY
  (previously it went through `launch_kernel`, whose remembered-spec fallback
  would have relaunched the very spec that just failed). An explicit pick
  replaces the broken selection; dismissing leaves the cells idle (bug #28
  behavior).
- **Tested:** no — test BLOCKED (user 2026-07-16): deleting the venv while Zed
  was running failed with "Failed to delete 1 of 1 file" — the notebook's
  kernel process keeps running after the notebook is closed and holds a lock
  on the env's files (filed as bug #35). Retest this once #35 is fixed (or
  work around by killing python.exe manually before deleting).

## 34. The same notebook file can end up open in multiple tabs

- **Status:** open (not reproduced)
- **Symptom:** (user 2026-07-16) `Untitled.ipynb` was somehow open in THREE
  tabs at once (screenshot). Opening the file again correctly returned to one
  of the already-open tabs, and after closing two of the three the bug could
  not be recreated. Opening the same file must never create a second
  independent view of it.
- **Analysis:** unknown trigger. Workspace item dedup is per project-path, so
  candidate holes: the "New Jupyter Notebook" flow creating `Untitled.ipynb`
  and opening it without going through the dedup path (three "new notebook"
  invocations before the first tab registered?), a race between two open
  requests resolving the same path, or external-reload re-adding an item.
  Likely interacts with phase 33 (truly-unsaved new notebooks), which will
  rework the Untitled flow anyway. WATCH ITEM: if it recurs, note how each of
  the duplicate tabs was opened (new-notebook command vs file open).
- **Fix attempted:** none
- **Tested:** n/a

## 35. Closing a notebook leaves its kernel process running

- **Status:** open
- **Symptom:** (user 2026-07-16) After closing a notebook tab, the kernel
  process is still alive: attempting to delete the notebook's venv fails with
  "Failed to delete 1 of 1 file" because the running kernel holds a lock on
  the env's files (python.exe in use on Windows). Only restarting Zed killed
  the kernel. This also blocks testing bug #31.
- **Analysis:** closing the notebook editor drops the UI but the
  `RunningKernel` (owned via the session/store) is not shut down — nothing in
  the notebook item's close/release path sends a shutdown request or kills the
  child process. Expected: closing the last view of a notebook shuts its
  kernel down (like Jupyter closing a session); at minimum the process must
  die with the item so env files aren't locked.
- **Fix attempted:** none
- **Tested:** n/a

## 36. Custom-location venv disappears from the kernel picker after restart

- **Status:** open
- **Symptom:** (user 2026-07-16) Created a venv at a custom location
  (`~/Dev/venvs/test_venv`) via kernel picker → "Create Python Environment" →
  "Choose Location…". It worked, ran notebooks, pip-installed fine. After
  restarting Zed the venv no longer appears in the kernel picker (only the
  workspace `.venv` and the global Pythons show).
- **Analysis:** the created env is registered as a kernel spec for the CURRENT
  session only (added to the store's spec list in memory). Discovery on a
  fresh start only finds workspace-local envs (`.venv` etc. via toolchain
  discovery) and installed jupyter kernelspecs — a venv outside the worktree
  is invisible to both. Fix direction: persist created-env locations (e.g.
  register a real jupyter kernelspec via `python -m ipykernel install --user
  --name <name>` at creation time, or persist known env paths and re-add them
  during discovery).
- **Fix attempted:** none
- **Tested:** n/a

## 37. Table (DataFrame) outputs lack "Open in Buffer"; old menu Copy skips them

- **Status:** open
- **Symptom:** (user 2026-07-16) On DataFrame/table outputs the new hover
  controls show Copy Output (works, nice markdown) but NOT "Open in Buffer";
  plain text outputs show and support both. Also the output "…" menu's
  long-standing Copy Output entry copies NOTHING for table outputs.
- **Analysis:** the phase-29 controls derive their availability from the
  output's clipboard/text representation; `TableView` implements the new
  markdown copy for the hover button but evidently isn't wired into the
  buffer-open path, and the older menu copy path uses a different
  (text-only) accessor that returns nothing for tables. Give tables the same
  markdown text representation on ALL paths: open-in-buffer opens the
  markdown table, menu copy copies it.
- **Fix attempted:** none
- **Tested:** n/a

## 38. Failed kernel launch marks the cell "Cancelled" instead of errored

- **Status:** open
- **Symptom:** (user 2026-07-16) With ipykernel removed from the env, running
  a cell correctly reports the launch failure (kernel stderr shown: "No module
  named ipykernel_launcher") but the cell's status shows "Cancelled" rather
  than a failed/error state.
- **Analysis:** when the launch fails, queued cells are cleared via the
  cancelled path (same as picker-dismiss, bug #28) instead of an errored path.
  `kernel_errored` should mark cells that were awaiting that launch as failed
  (red ✕), reserving "Cancelled" for user-initiated dismissal/interrupt.
- **Fix attempted:** none
- **Tested:** n/a

## 39. Last-executed timestamps don't interop with VS Code

- **Status:** open
- **Symptom:** (user 2026-07-16) Phase 28 timestamps round-trip within Zed,
  but: (a) a notebook run in Zed shows NO execution times when opened in VS
  Code; (b) running a cell in VS Code (while closed in Zed) and then opening
  in Zed shows Zed's STALE old time (11:59:44) instead of VS Code's newer run
  (12:02:49), surviving even a Zed restart.
- **Analysis:** Zed writes/reads its own cell-metadata key; VS Code's Jupyter
  extension stores its execution summary differently (and doesn't read ours).
  Needs investigation of what VS Code actually persists per cell
  (`metadata.execution` shell/iopub timestamps) and then: write both formats
  on save, and prefer the NEWEST available record on load so an external run
  updates the display. (b) is the more important half — showing a stale time
  as if current is misleading.
- **Fix attempted:** none
- **Tested:** n/a
