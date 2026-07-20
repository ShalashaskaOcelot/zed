# Bugs

Status values: `open` | `fix attempted - untested` | `fixed - confirmed`.
Never attempt a further fix while a bug is `fix attempted - untested`.
When `fixed - confirmed`, add a one-line entry to `CHANGELOG.md` and DELETE the
bug's entry here (there is no archive dir; the CHANGELOG + commit is the record).

---

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

## 34. The same notebook file can end up open in multiple tabs

- **Status:** fix attempted - untested (plausible cause found by inspection;
  original trigger was never reproduced)
- **Symptom:** (user 2026-07-16) `Untitled.ipynb` was somehow open in THREE
  tabs at once (screenshot). Opening the file again correctly returned to one
  of the already-open tabs, and after closing two of the three the bug could
  not be recreated. Opening the same file must never create a second
  independent view of it.
- **Analysis (code inspection):** the pane's already-open dedup compares the
  incoming path's CURRENT `ProjectEntryId` against the ids each open tab
  advertises. `NotebookItem` captured its entry id ONCE at open and never
  refreshed it — but notebook save rewrites the .ipynb via `fs.atomic_write`
  (temp file + rename), which can replace the worktree entry under a NEW id.
  After such a save the open tab advertises a stale id, the next open of the
  same path resolves the new id, nothing matches, and a duplicate tab opens
  (each further save/open can repeat this → 3 tabs). Also explains why
  opening "again" dedups fine most of the time (no id churn between opens).
- **Fix attempted (2026-07-16):** `NotebookItem::entry_id` now resolves the
  entry id LIVE from the project by path (falling back to the cached id if
  the project/entry is gone), so the dedup comparison always sees the current
  id. The original trigger was never reproduced, so this is a
  best-explanation fix — treat a recurrence as this fix having failed.
- **Tested:** no — can only be soak-tested: work normally (create/save/reopen
  Untitled notebooks); if the same file never opens twice again over a few
  sessions, call it fixed.


## 41. Text/table output doesn't use the output block's full width

- **Status:** open (pre-existing; NOT a regression — never fully solved.
  Confirmed by user 2026-07-16.)
- **Symptom:** (user 2026-07-16, phase 40 round 2) The output block is now the
  cell's full width (copy/open-in-buffer controls sit at the right edge), but
  the CONTENT only reaches roughly the middle: text output "ends in the
  middle", and DataFrame/table output shows stretched columns whose cell text
  is truncated. So the box is full width but the content isn't.
- **Analysis (code inspection, 2026-07-16):** two distinct causes.
  1. TEXT (stream/plain/error → `TerminalOutput`): the display-only terminal
     grid is a FIXED size — `terminal_size()` builds it at
     `max_columns` (default 128) × `max_lines` (32); content beyond scrolls
     into history. `with_renderable_cells` reads that fixed-width grid, so the
     painted text can never exceed 128 columns regardless of how wide the box
     is (on a wide monitor 128 cols ≈ half). Phase 40 widened the CONTAINER
     (`outputs.rs` dropped the `max_width` cap for notebooks) but not the grid.
     Fix direction: size the terminal grid's COLUMN count to the output box's
     pixel width (keep rows at `max_lines` so the vertical-scroll cap is
     unchanged), resizing in the canvas prepaint where the real bounds are
     known and re-rendering once when the column count changes. Risk: alacritty
     reflow interacts with the fixed 32-row screen + scrollback; needs GUI
     iteration (can't be verified headless).
  2. TABLE (rich HTML → `TableView`, `outputs/table.rs`): columns stretch to
     fill the full width but cell text is truncated instead of shown/scrolled.
     Separate component; likely a column-measurement/`text_ellipsis` issue.
- **Fix attempted:** none yet (deliberately not a 3rd blind attempt — this is
  GUI-dependent and phase 40 already failed twice on it). Needs a focused,
  build-and-test pass with the user.
- **Tested:** n/a

## 42. Output text selection isn't cleared when clicking into another cell to edit

- **Status:** open (phase 37 follow-up; minor)
- **Symptom:** (user 2026-07-16) After drag-selecting output text, clicking
  another cell in COMMAND mode (its gutter/border) or empty space DOES clear
  the highlight, but clicking directly into another cell to enter EDIT mode
  leaves the previous output's selection highlighted.
- **Analysis:** the deselect is driven by a window-level mousedown handler on
  each output (`outputs/plain.rs`): a click outside the output's bounds while
  it has a selection clears it. Clicking into another cell's editor is outside
  those bounds, so it should hit that branch — but the editor likely consumes
  the mousedown (capture/stop-propagation) before the global bubble handler
  runs, or the ensuing focus/edit-mode re-render drops the handler first.
  Needs GUI debugging to confirm which. Cosmetic only (copy/cut still correct;
  the stale highlight clears on the next click).
- **Fix attempted:** none
- **Tested:** n/a

## 43. Embedded (non-notebook) terminal text is too large

- **Status:** open (needs the user to pin when it changed)
- **Symptom:** (user 2026-07-16) The built-in Zed terminal (e.g. the pwsh
  panel, screenshot) started rendering with very large text "a few commits
  ago". User prefers small terminal text. This is the EMBEDDED terminal, not
  notebook output.
- **Analysis (code inspection, 2026-07-16):** unrelated to the REPL's
  `terminal_size` (that only sizes notebook/inline-REPL output, in
  `crates/repl/`). The embedded terminal's font comes from
  `TerminalElement::rem_size` = `ThemeSettings::buffer_font_size * 1.125`
  (a fixed UI-scale factor, `terminal_view/src/terminal_element.rs:839-857`),
  unless overridden by the `terminal.font_size` setting
  (`terminal/src/terminal_settings.rs`). NOTE: the fork's own history since
  the 2026-07-08 fork point does NOT touch `crates/terminal_view`, terminal
  font handling, or the font-settings defaults (`git log 950ec7943f..HEAD`
  over those paths shows only phase 37's unrelated `clear_selection`), so
  this most likely arrived via an upstream sync or a `buffer_font_size` /
  theme / zoom change rather than fork work.
- **Workaround (shared with user):** set an explicit size in settings.json,
  e.g. `"terminal": { "font_size": 12 }`, or lower `buffer_font_size`
  (the terminal scales off it when `terminal.font_size` is unset).
- **Next step:** have the user note the rough commit/date it changed (or
  whether it followed an upstream merge / settings edit) so the cause can be
  bisected; confirm whether `terminal.font_size` is set in their config.
- **Fix attempted:** none
- **Tested:** n/a

## 44. Rust (evcxr) kernel stuck Busy on a second "Run All" — all cells left Pending

- **Status:** open (needs a rust kernel + runtime instrumentation to confirm)
- **Symptom:** (user 2026-07-16, seen on TWO devices) Using the evcxr Rust
  kernel (NOT python): run the whole notebook to completion, then — without
  restarting the kernel — hit Run All again. The kernel goes/stays "Busy" and
  every cell just sits Pending; nothing executes. Restarting the Rust kernel
  fixes it, but that wipes all kernel state. Not reported with ipykernel.
- **Analysis (hypothesis, code inspection):** the batch runner
  (`advance_run_queue`, `notebook_ui.rs`) submits the next cell only once the
  current cell's `ExecuteReply` arrives and `active_run_cell` is cleared (in
  `route`); `advance_run_queue` early-returns while `active_run_cell.is_some()`.
  If evcxr's status/reply messages differ from ipykernel's — e.g. the last
  cell of the first batch never yields a clean `ExecuteReply` / `status: idle`
  that Zed recognises — then after the first Run All completes, `active_run_cell`
  (and/or the kernel's `execution_state`, which drives the "Busy" indicator)
  is left stuck. The second Run All then queues everything but
  `advance_run_queue` never advances (active cell still "set"), so all cells
  stay Pending and the strip shows Busy. evcxr is known to diverge from the
  ipykernel status/heartbeat conventions, which fits the rust-only report.
- **Investigation needed:** with a rust kernel, log the shell `ExecuteReply`
  and iopub `status` (busy/idle) `parent_header.msg_id`s across a full Run All,
  and check `active_run_cell` / `execution_state` after it "completes". Confirm
  whether evcxr sends idle/reply for the final cell and whether Zed clears
  `active_run_cell`. Likely fix: make batch/kernel-idle detection robust to a
  missing/late final reply (e.g. clear `active_run_cell` on `status: idle` as a
  fallback, or reconcile the queue when the kernel returns to idle).
- **Fix attempted:** none
- **Tested:** n/a
