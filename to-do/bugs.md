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

## 47. Notebook tab title stays "Untitled" after saving it outside the workspace

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-21) Save a notebook (untitled or otherwise) to a
  path OUTSIDE the current workspace (e.g. the Desktop). The tab keeps showing
  "Untitled" even though the file is saved and named (Ctrl-S shows no dialog and
  there's no dirty dot — the path IS attached; only the title is wrong). Saving
  INSIDE the workspace shows the correct name. Reproduces without any restart,
  so unrelated to session restore (phase 52).
- **Analysis (workflow investigation, high confidence):** saving to a path in no
  existing worktree makes the shared save flow
  `project.find_or_create_worktree(new_path, /*visible=*/true)` create a NEW
  single-file worktree rooted AT the file; its worktree-relative path is
  `RelPath::empty_arc()` (`worktree_store.rs:436`). `tab_content_text`
  (`notebook_ui.rs:4856`) derived the label from `project_path.path.file_name()`,
  which is `None` on the empty relative path → "Untitled". (The plain text
  editor is unaffected: it titles from the buffer's file, which falls back to
  the worktree root name.)
- **Fix attempted (1):** derive the tab label from the ABSOLUTE `path` (always
  set on save/open) instead of the relative path (`notebook_ui.rs`
  tab_content_text). Commit `ae75857da4`. NOT SUFFICIENT — see below.
- **Real root cause (found 2026-07-30 from the user's "no such worktree"
  report):** the title fix could never take effect because the SAVE ITSELF was
  failing partway. Phase 53 changed out-of-project save-as to create an
  INVISIBLE worktree, and `WorktreeStore::add` keeps an invisible worktree with
  only a WEAK handle (`worktree_store.rs`, `push_strong_handle`). In
  `Pane::save_item` the strong `worktree` binding lived inside the `if let`
  block, so it was dropped BEFORE `save_task.await` — killing the just-created
  worktree mid-save. The notebook's `save_as` writes the file first
  (`fs.atomic_write` — which is why the file DID appear on the Desktop) and then
  calls `project.open_buffer(path)`, which failed with "no such worktree"; the
  `?` then skipped the block that sets `item.path` / `item.project_path`, so the
  item stayed untitled and detached from the file it had just written.
- **Fix attempted (2, 2026-07-30):** hold the created worktree alive across the
  save in `Pane::save_item` (bind it outside the `if let` and drop it after
  `save_task.await`, by which point the saved buffer's `File` holds it).
- **Tested:** no — see `awaiting_testing.md`. This should fix the "Failed to
  save / no such worktree" dialog AND the "Untitled" title together.

## 49. Files opened/saved outside the workspace aren't removed from the panel when deleted externally

- **Status:** open (low priority — user-facing symptom resolved by phase 53;
  only a latent watching gap remains)
- **Symptom:** (user 2026-07-21) A notebook saved outside the workspace shows as
  a standalone root in the panel (see phase for the "don't add external saves to
  the workspace" change). Deleting that file from the OS file manager does not
  remove it from Zed's panel: one entry got renamed to the Windows Recycle Bin
  artifact `$RVIBBRG.ipynb` (a rename event was seen) and another
  (`test2.ipynb`) just stayed. Files INSIDE the workspace disappear from the
  panel immediately on external delete.
- **Analysis (hypothesis):** these external files live in single-file worktrees.
  Single-file-worktree file-watching / entry-removal appears not to handle
  deletion (or a Recycle-Bin rename) the way a normal directory worktree does.
  Likely shared Zed behavior, not notebook-specific. Needs investigation of
  single-file worktree fs-event handling in `crates/worktree`. May become moot
  for the notebook flow once external saves no longer create visible worktrees
  (the Zed-wide behavior change), but the underlying watching gap is separate.
- **Update (post phase 53):** the reported SYMPTOM is resolved — external saves
  no longer add a visible single-file-worktree root to the panel, and
  externally-OPENED files were always in an INVISIBLE worktree (not shown in the
  panel), so there is no longer a stuck panel entry to fail to remove. What
  remains is a latent gap (a single-file worktree may not react to its root file
  being deleted/renamed on disk), but it has no current user-visible symptom in
  the save/open flow. Downgraded to low priority; fix only if a concrete symptom
  reappears (would be a `crates/worktree` fs-event investigation).
- **Fix attempted:** none (symptom addressed indirectly by phase 53)
- **Tested:** n/a

## 50. Whole workspace/session lost after deleting externally-saved files

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-21) After saving files outside the workspace (which
  were added as single-file worktree roots — see #49) and then deleting them
  externally, closing and reopening Zed restored NO session at all — the open
  workspace was lost. User couldn't test with unsaved work open. User suspects
  the stale external entries corrupted the workspace session rather than a
  phase-52 issue (phase 52 restore had been working).
- **Root cause (code inspection, high confidence):** the whole-workspace
  restore gate is all-or-nothing on path existence.
  `WorkspaceDb::all_paths_exist_with_a_directory` (`persistence.rs:2001-2014`)
  loops the workspace's serialized root paths and `return false` the moment ONE
  path's `fs.metadata` is `None` (deleted). The restore callers
  (`recent_project_workspaces_ungrouped:2044`, `last_workspace`/its helper
  `:2155`, `last_session_workspace_locations:2204`) then DROP the entire
  workspace when it returns false. The user's workspace location was
  `[rust-data-analysis (dir), Desktop/test.ipynb (file), Desktop/test2.ipynb
  (file)]` — the external files were added to the location by save-as (the
  phase-53 bug). Deleting them made one path missing → the whole workspace
  (including the real folder) was excluded from restore = total session loss.
  NOT a phase-52 issue: this gate runs on the workspace LOCATION before any item
  is deserialized, so `NotebookEditor::deserialize` is not implicated (and
  per-item deserialize errors are isolated, like the editor's). Phase 53
  (external saves no longer add roots) removes the main TRIGGER going forward,
  but the underlying all-or-nothing gate remains a robustness gap for any
  multi-root workspace where one root later disappears (e.g. an unplugged drive).
- **Fix direction (Zed-core, needs care — several callers):** instead of
  rejecting the whole workspace when a path is missing, FILTER the missing paths
  out and restore the workspace from the survivors as long as at least one
  directory remains; only skip entirely when nothing usable survives. Must
  thread the filtered path set through the callers (they currently build
  `RecentWorkspace`/`SessionWorkspace` with the full `paths`) without breaking
  workspace identity/dedup (note the existing `identity_paths` vs `paths`
  split). Add a unit test: a location with one missing file + one present dir
  restores with just the dir.
- **Fix attempted:** in `last_session_workspace_locations` (`persistence.rs`),
  replaced the all-or-nothing `paths.is_empty() || all_paths_exist_with_a_directory`
  gate with a new `existing_paths` filter: each restored session workspace keeps
  only its surviving root paths, and is ALWAYS restored (even with zero surviving
  roots, as an empty location) so its unsaved, DB-stored items are recovered
  rather than discarded with a deleted folder. Covers both facets the user
  raised: (1) a multi-root workspace with one dead root restores the survivors;
  (2) unsaved items survive even when every folder is gone. Added unit test
  `test_session_restore_drops_missing_roots_keeps_survivors`. The recent-projects
  UI list (`recent_project_workspaces_ungrouped`) intentionally still hides
  fully-dead projects; `garbage_collect_workspaces`'s 7-day deletion is mitigated
  because a restored workspace rejoins the current session (so it's not GC'd).
- **Tested:** unit test passes; runtime untested — see `awaiting_testing.md`.

## 52. Pane nav buttons (new / split / zoom) flicker at times

- **Status:** open
- **Symptom:** (user 2026-07-30, third screenshot) The pane toolbar buttons at
  the top-right — new file, split pane, zoom-in — randomly flicker/redraw at
  times while a notebook is open.
- **Analysis (hypothesis, unconfirmed):** likely re-render churn — the notebook
  emits frequent `cx.notify()` (kernel status ticks, execution-state changes,
  follow-scroll) which can drive the surrounding pane toolbar to repaint. Could
  also be pre-existing Zed behaviour unrelated to the notebook. Needs runtime
  investigation: does it flicker only during kernel activity / execution, or at
  idle too? Does it happen with a non-notebook item?
- **Fix attempted:** none
- **Tested:** n/a

## 54. Crash: clicking the sidebar kernel selector double-leases the notebook

- **Status:** fix attempted - untested
- **Symptom:** (user 2026-07-30, full backtrace supplied) Clicking the kernel
  selector button at the bottom of the notebook's right control sidebar aborts
  the app: `cannot update repl::notebook::notebook_ui::NotebookEditor while it
  is already being updated`, then `panic in a function that cannot unwind` →
  `thread caused non-unwinding panic. aborting.`
- **Root cause (confirmed from the trace):** the button's `cx.listener` already
  holds a lease on `NotebookEditor`, and it called
  `kernel_picker_handle.toggle(window, cx)` INLINE. `toggle` → `show`
  synchronously fires the picker's `with_on_open` callback
  (`notebook_ui.rs`, render_kernel_strip), which does `view.update(cx, ...)` on
  the same entity → double lease → abort. PRE-EXISTING (the button dates from
  phase 24); not introduced by the phase-57/#51 work. Identical in shape to the
  earlier run-with-no-kernel double-lease, which was fixed with `window.defer`.
- **Fix attempted (2026-07-30):** defer the toggle via `window.defer` so it runs
  with no entity lease held (NOT `cx.defer_in`, which re-wraps the closure in
  another `NotebookEditor` update and reintroduces the nesting). Applied the
  same treatment to `launch_kernel`'s `show` — currently unreachable (that
  branch requires no remembered kernel, which routes elsewhere) but the same
  latent hazard.
- **Tested:** no — see `awaiting_testing.md`.

## 55. Workspace venvs unavailable to notebooks opened from outside the workspace

- **Status:** open
- **Symptom:** (user 2026-07-30) With a workspace open that contains a `.venv`,
  open a notebook from OUTSIDE that workspace (e.g. dragged in from the
  Desktop). The kernel picker offers only "global" interpreters — the
  workspace's `.venv` is missing, even though that workspace is open.
- **Expected:** any venv in the currently-open workspace should be selectable
  for any open notebook, regardless of where the notebook file lives.
- **Analysis (to investigate):** kernel/toolchain discovery appears to be scoped
  to the notebook's OWN worktree. A notebook opened from outside the project
  gets its own (invisible, single-file) worktree whose root is the file's
  directory, so a `.venv` in the real project worktree is out of scope. Look at
  how `repl::kernels` enumerates Python toolchains (the `Preparing Python kernel
  for toolchain` path) and which worktree/project it queries; it likely needs to
  consider ALL visible project worktrees rather than just the notebook's.
- **Fix attempted:** none
- **Tested:** n/a

## 59. External notebooks are not restored on restart (reopen as empty "Untitled")

- **Status:** open
- **Symptom:** (user 2026-07-30) Save notebooks to a path OUTSIDE the workspace
  (e.g. the Desktop), then quit and reopen Zed. The external notebooks are NOT
  restored — instead two EMPTY "Untitled" notebooks open in their place. The
  startup log shows no attempt to load the external paths at all.
- **Analysis (hypothesis, needs confirming):** likely a direct consequence of
  phase 53. Session restore persists the workspace's VISIBLE roots; an
  out-of-project file now lives in an INVISIBLE single-file worktree, which is
  not persisted as a root. On restart the item's stored worktree id resolves to
  nothing, so the notebook deserializes with no path — i.e. an untitled
  notebook. Before phase 53 these files became visible roots, so they did come
  back (but as unwanted panel roots — the very thing phase 53 removed).
  The goal is BOTH: restored as open tabs, still not panel roots. That means the
  item's serialization must carry the ABSOLUTE PATH rather than relying on a
  worktree id that no longer survives, and restore must re-create the invisible
  worktree for it. Check `NotebookEditor`'s `SerializableItem` impl
  (`serialize`/`deserialize`, `notebook/persistence.rs`) and how the workspace
  restores items whose worktree is gone. NOTE: plain TEXT files saved outside
  the project probably have the same problem — worth testing both.
- **Fix attempted:** none
- **Tested:** n/a

## 57. Notebook scrollbar overlaps the cell's left margin gutter

- **Status:** open
- **Symptom:** (user 2026-07-30, screenshot) The cell-list scrollbar (phase 56)
  is drawn over the cell's left margin, so that margin can no longer be clicked
  to select the cell in command mode. Two related notes: (a) clicking the OTHER
  margins doesn't select the cell either (user expected it would — may be
  pre-existing/by design, confirm); (b) clicking far down the scrollbar track
  does not jump the view there the way a normal scrollbar does.
- **Analysis (to investigate):** the scrollbar is attached to the cell-list
  column, whose right edge apparently sits over the cell's margin rather than
  outside it. Either inset the list content by the scrollbar's reserved width,
  or move the scrollbar outside the cell's margin. The track-click-to-jump
  behaviour is a `ui::Scrollbars` feature (`ScrollbarStyle::Editor` reserves the
  track) — check whether the notebook's list needs a reserved track for the
  track-click hit area to exist.
- **Fix attempted:** none (user: "happy with the shape, it's just weirdly
  positioned")
- **Tested:** n/a

## 58. Kernel picker's "Creating…" row doesn't refresh when the build finishes

- **Status:** open
- **Symptom:** (user 2026-07-30, phase 49 testing) While an environment is being
  created the picker correctly shows a greyed "Creating <name>…" row. When the
  build COMPLETES, that row does not update in place — it stays as "Creating…"
  until the picker is closed and reopened, at which point the real kernel entry
  appears.
- **Analysis (to investigate):** phase 49's design intended the row to be
  replaced live ("Leaving the picker open across completion updates it
  correctly"). The picker's list is presumably built once when opened and not
  re-rendered on the notebook's completion notification. Look at how
  `KernelPickerDelegate` gets its entries and whether the notebook's
  build-completion path notifies/refreshes the open picker (vs only updating the
  notebook itself).
- **Fix attempted:** none
- **Tested:** n/a

## 63. Long text output shows only its last ~32 lines (the head is cut off)

- **Status:** open — needs a decision on the wanted behaviour before fixing.
- **Symptom:** (user 2026-07-30, screenshots) A cell whose output is a large
  text repr (a JSON-ish API response) displays only the TAIL in Zed — the user
  sees the `meta`/warnings section and the beginning (`jsonapi`, `links`,
  `data`) is simply absent. VS Code, opening the SAME .ipynb (so literally the
  same saved output bytes), shows it from the start with its own "Output is
  truncated. View as a scrollable element…" notice. Round-trips both ways, so
  it is purely a Zed DISPLAY issue, not parsing or serialization — and the data
  is definitely present (`hubs.get('data')` still works in Zed).
- **Root cause (CONFIRMED):** plain/stream output is rendered by
  `TerminalOutput`, a real terminal emulator sized to
  `ReplSettings::max_lines` rows (`outputs/plain.rs`, `terminal_size`).
  `max_lines` defaults to **32** (`assets/settings/default.json`, clamped
  [4, 256]). Appending more than 32 lines scrolls the earlier ones off the
  viewport exactly as a console would, so what remains visible is the LAST 32
  lines. The content is not lost — the emulator keeps 10,000 lines of scrollback
  (`DEFAULT_SCROLL_HISTORY_LINES`) and the full text is retained separately in
  `TerminalOutput::full_buffer` for "open in buffer" — but the notebook renders
  a fixed-height viewport with no way to scroll inside the output block.
- **Immediate workaround for the user:** raise `"max_lines"` in settings (up to
  256). That setting was designed for the INLINE REPL, where a short window is
  reasonable; it is a poor default for notebook cells.
- **Decision (user 2026-07-30):** truncate like VS Code — show the HEAD with a
  truncation notice, keeping `max_lines` at 32 ("actually a fine default"). The
  existing open-in-buffer covers seeing everything. A "view as a scrollable
  element" affordance may come later, but outputs must NOT capture the
  scrollwheel by default. Scheduled as **phase 59**.
- **Also noted (user 2026-07-30):** outputs CAN already be scrolled by dragging
  a selection, "but it doesn't work properly" — highlighting and dragging up
  selects everything and moves the view. Not addressed by phase 59; file
  separately if it still grates once truncation lands.
- **Fix attempted:** none yet — see phase 59.
- **Tested:** n/a
