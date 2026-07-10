# Bugs

Status values: `open` | `fix attempted - untested` | `fixed - confirmed`.
Never attempt a further fix while a bug is `fix attempted - untested`.
Move to `to-do/archive/` only when `fixed - confirmed`.

---

(Bug #1 "Restart kernel kills but relaunch fails" — fixed & confirmed
2026-07-08, moved to `archive/bugs-fixed.md`.)

## 2. Running a cell with a dead/shutdown kernel does not start the kernel

- **Status:** fix attempted - untested
- **Symptom:** After the kernel is killed, running a cell just errors
  ("the kernel is shut down" / "failed to launch") instead of starting the
  selected kernel.
- **Analysis:** `execute_cell` (`notebook_ui.rs:518-567`) has no relaunch
  branch for `Shutdown` / `ErroredLaunch`; it only renders an error output.
  It also does not queue executions while `StartingKernel` (the REPL does,
  `session.rs:683-793`).
- **Fix attempted:** `execute_cell` now queues the execution and relaunches
  the selected kernel when the kernel is `Shutdown` or `ErroredLaunch`, and
  queues (without relaunching) while `StartingKernel`/`Restarting`. Queued
  cells run in order once the kernel is up; if the launch fails they show the
  launch error instead of spinning. Covered by updated test
  `test_run_cell_with_missing_interpreter_shows_error`.
- **Tested:** no — automated test passes; needs user confirmation on Windows

## 3. Interrupt / stop button does not interrupt a running cell

- **Status:** fix attempted - untested
- **Symptom:** (Round 1) Stop button always disabled. (Round 2, user
  2026-07-08) Button is now enabled when idle, but pressing it does not stop
  a running task.
- **Analysis:** Round 1 was pure UI enablement (Busy-only gate). Round 2 is
  the real mechanism: the notebook sent a message-based `interrupt_request`
  over the control channel, but ipykernel does NOT honor message-based
  interrupts by default — it expects an OS-level interrupt (SIGINT on Unix, a
  Windows interrupt event). We know the control channel itself works because
  `ShutdownRequest` (restart) travels the same channel and succeeds, so the
  message is delivered but ignored. The user is on Windows, where ipykernel's
  parent poller waits on a `JPY_INTERRUPT_EVENT` handle.
- **Fix attempted:**
  - Round 1: enable the stop button whenever the kernel is connected (Idle or
    Busy); log send failures; warn when no kernel is running.
  - Round 2: OS-level interrupt. Added `Child::spawn_interruptible` and
    `Child::interrupt` in `util::process`. Unix sends `SIGINT` to the kernel's
    process group (`killpg`). Windows creates an inheritable auto-reset event,
    passes it to the kernel via `JPY_INTERRUPT_EVENT`, and signals it with
    `SetEvent` — matching how jupyter_client interrupts kernels on Windows.
    `RunningKernel::interrupt` defaults to the old message-based path;
    `NativeRunningKernel` overrides it to use the OS interrupt (with the
    message send as a fallback). Notebook and REPL both call
    `kernel.interrupt()`.
- **Round 2 result (user 2026-07-08):** the interrupt IS delivered — clicking
  interrupt during `time.sleep(10)` raised `KeyboardInterrupt` and stopped the
  batch (the cells below did not run). BUT it did not fire until the sleep
  finished (~10s), rather than immediately. This is the KNOWN Windows/ipykernel
  limitation: the event-based interrupt (`JPY_INTERRUPT_EVENT` →
  `interrupt_main` → `PyErr_SetInterrupt`) sets the interrupt flag but does not
  wake a C-level blocking call like `time.sleep`; CPython only raises the
  exception once control returns to the interpreter. A pure-Python busy loop
  should be interrupted promptly. So the mechanism works; only immediate
  interruption of C-blocking calls is limited.
- **Status:** the core interrupt now works; keep this open only until the user
  confirms a pure-Python loop interrupts PROMPTLY. Immediate interruption of
  C-blocking calls on Windows would require launching the kernel in a new
  process group and using `GenerateConsoleCtrlEvent` (jupyter's "signal"
  interrupt mode) — filed in backlog.
- **Tested:** partially — delivered + batch-cancel confirmed; prompt interrupt
  of normal Python code still to confirm.

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

## 13. After adding a cell with `a`/`b`, Enter sometimes won't enter edit mode

- **Status:** open (intermittent, hard to reproduce)
- **Symptom:** (user 2026-07-08) Pressed `b` to add a cell; the cell was
  created and focused but Enter did nothing (repeatedly), and esc→enter,
  refocusing from an adjacent cell, etc. didn't help. Clicking directly in the
  cell text area fixed it and it then behaved. On another attempt, `b` created
  the cell AND went straight into edit mode. Inconsistent.
- **Analysis:** Likely a focus/mode race in the add-cell path
  (`add_code_cell_at` → `focus_cell_editor_in_edit_mode` sets
  `NotebookMode::Edit` and focuses the new editor). When it lands in a bad
  state, `notebook_mode`/focus and the actual focused element disagree, so the
  command-mode `enter → EnterEditMode` binding either isn't active or
  `enter_edit_mode` focuses an editor that isn't the one showing. Overlaps
  with the backlog item to make `a`/`b` focus in COMMAND mode (which would
  sidestep this by not auto-entering edit mode).
- **Fix attempted (2026-07-08):** `a`/`b` (and the + toolbar buttons) now
  insert the new cell and stay in COMMAND mode (select it, focus the notebook
  handle) instead of jumping into edit mode. Note: the user separately hit a
  broader "no cursor, no chars, esc doesn't help" stuck state that is NOT edit
  mode — that is the focus/mode desync in bug #15.
- **Tested:** no — needs user confirmation that `a`/`b` now land in command
  mode and shortcuts keep working after adding a cell.

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
- **Tested:** no — needs user confirmation + more repro detail on the exact
  action that drops focus.

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
- **Fix attempted:** none
- **Tested:** n/a
