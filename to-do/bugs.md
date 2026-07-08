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
- **Tested:** no — Unix path type-checks locally; Windows path type-checks and
  clippy-checks against the `x86_64-pc-windows-msvc` target but the actual
  interrupt behaviour needs user confirmation on Windows (interrupt a
  long-running cell, e.g. `import time; time.sleep(30)`).

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

## 10. Kernel picker does not accept Enter to select

- **Status:** open
- **Symptom:** (user 2026-07-08) In the kernel selector, arrow keys navigate
  the list, but pressing Enter does not confirm the highlighted kernel — the
  user perceives a newline being entered in the search box instead.
- **Analysis:** The picker's query editor is single-line
  (`Editor::single_line` via the erased-editor factory) and
  `KernelPickerDelegate::confirm` (`kernel_options.rs:300`) looks correct
  (calls `on_select` + emits `DismissEvent`). Enter is globally bound to
  `menu::Confirm`. So Enter should reach `Picker::confirm` → delegate. The
  failure is most likely a focus / key-context interaction specific to this
  picker being hosted in a `PopoverMenu` and/or opened via
  `kernel_picker_handle.show()` (phase 6 lazy-start) — arrows reach it but
  Confirm does not. NEEDS RUNTIME DEBUGGING; not safe to guess-fix shared
  picker infra.
- **ROOT CAUSE (2026-07-08):** the user's diagnosis was right — the focused
  query editor ate Enter as a newline. The kernel picker's popover is rendered
  inside the `NotebookEditor` element tree, so its query editor matched the
  notebook's OWN keymap context `"NotebookEditor > Editor"`, which binds
  `enter → editor::Newline`. That binding matches at the editor node (deeper)
  and beat the picker's `"Picker"` context `enter → menu::Confirm`, so Enter
  inserted a newline instead of confirming. (Zed's GitBranchSelector avoids
  this by scoping its editor bindings through `> Picker > Editor`.)
- **Fix attempted:** gave the notebook's CELL editors a distinct
  `NotebookCellEditor` key context (added to the div wrapping each cell's
  editor in `cell.rs`) and changed the keymap context from
  `"NotebookEditor > Editor"` to `"NotebookEditor > NotebookCellEditor >
  Editor"` in all three keymaps. The picker's query editor is not inside a
  `NotebookCellEditor`, so it no longer matches — Enter now resolves to the
  picker's `menu::Confirm`.
- **Tested:** no — needs user confirmation (Enter selects the kernel; cell
  editors still get enter=newline / ctrl-enter=run / escape=command mode).

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
  sidestep this by not auto-entering edit mode). Needs reliable repro.
- **Fix attempted:** none
- **Tested:** n/a
