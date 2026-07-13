# Fixed bugs (confirmed)

Confirmed-fixed bugs moved out of `to-do/bugs.md`.

---

## 4. "More options" toolbar button opens nothing

- **Status:** fixed - confirmed (user 2026-07-08: "More options menu seems to
  be working").
- **Fix:** phase 4 wired the right-toolbar Ellipsis button to a `PopoverMenu` +
  `ContextMenu` of cell actions.

## 5. Output "..." (ellipsis) button next to cell output does nothing

- **Status:** fixed - confirmed (user 2026-07-08: "output menu working").
- **Fix:** phase 4 wired the per-output Ellipsis to a menu (Copy Output /
  Clear Output).

## 10. Kernel picker does not accept Enter to select

- **Status:** fixed - confirmed (user 2026-07-08: "the kernel picker now
  accepts enter as an option").
- **Root cause:** the picker's popover renders inside the NotebookEditor
  element tree, so its single-line query editor matched the notebook's
  `"NotebookEditor > Editor"` keymap context (`enter -> editor::Newline`),
  which beat the picker's `"Picker"` context `enter -> menu::Confirm`.
- **Fix:** gave notebook CELL editors a distinct `NotebookCellEditor` key
  context and scoped the notebook editor bindings to
  `"NotebookEditor > NotebookCellEditor > Editor"`, so the picker's query
  editor no longer matches and Enter resolves to Confirm. (commit cfcbcd2)

## 11. Kernel-select prompt: cell state on dismiss vs. select

- **Status:** fixed - confirmed (user 2026-07-08: "queued items now run
  successfully on kernel selection and pressing esc does not cause it to keep
  'running'").
- **Symptom:** Running a cell with no kernel opened the picker; Esc left the
  cell stuck "Running", and (after the first fix) selecting a kernel didn't
  run the queued cell.
- **Fix:** `execute_cell` holds a no-kernel run in `cells_awaiting_kernel_choice`
  without a spinner and opens the picker; `change_kernel` promotes those cells
  to run once the kernel is ready; a new `on_dismiss` picker callback clears
  them on Esc. (commits b846cbe / 7e77a96 area)

## 1. Restart kernel kills the kernel but the relaunch fails

- **Status:** fixed - confirmed (user, 2026-07-08: "Restart now working fine")
- **Symptom:** Restart button / ctrl-shift-r killed the kernel; it never came
  back. Running a cell afterwards reported the kernel was not running.
- **Root cause / fix:** The kernel's message-handling tasks were detached, so
  a killed kernel's stale recv task errored asynchronously AFTER the kill and
  overwrote the replacement kernel's state with `ErroredLaunch`. Fixed by
  returning the supervisor task from `start_kernel_tasks` and storing it on
  each running kernel so dropping/killing a kernel cancels its message
  handling. `restart_kernel` was also reworked to the REPL sequence
  (`ShutdownRequest{restart:true}`, grace period, awaited forced kill,
  relaunch), clears `execution_requests`, stops cell spinners, and ignores
  errors that arrive while restarting/shutting down. Connection files are now
  unique per launch. (commit 7bd5b5a)

## 8. Clean kernel exit leaves stale RunningKernel state

- **Status:** fixed - confirmed (user, 2026-07-08: "exit() works perfectly")
- **Symptom:** A kernel that exited on its own (e.g. `exit()`) left the UI
  showing a running kernel.
- **Fix:** Added `KernelSession::kernel_exited`, called by the native and WSL
  process watchers on a successful exit, transitioning the notebook (and REPL
  session) to `Shutdown` and stopping cell spinners. (commit 7bd5b5a)

## 2. Running a cell with a dead/shutdown kernel does not start the kernel

- **Status:** fixed - confirmed (user, 2026-07-10: "bug 2 is fixed")
- **Symptom:** After the kernel was killed, running a cell just errored instead
  of starting the selected kernel.
- **Fix:** `execute_cell` now queues the execution and relaunches the selected
  kernel when it is `Shutdown` / `ErroredLaunch`, and queues (without
  relaunching) while `StartingKernel` / `Restarting`. Queued cells run in order
  once the kernel is up; a failed launch shows the launch error instead of
  spinning.

## 3. Interrupt / stop button does not interrupt a running cell

- **Status:** fixed - confirmed (user, 2026-07-10: tested with a Python loop
  instead of `time.sleep`, "killed immediately")
- **Symptom:** Stop button did not stop a running task.
- **Fix:** OS-level interrupt. `Child::spawn_interruptible` / `Child::interrupt`
  in `util::process`: Unix sends `SIGINT` to the kernel's process group;
  Windows creates an inheritable auto-reset event, passes it via
  `JPY_INTERRUPT_EVENT`, and signals it with `SetEvent` (matching
  jupyter_client). `NativeRunningKernel::interrupt` uses the OS interrupt with
  the message-based path as a fallback. A pure-Python loop interrupts promptly.
- **Follow-up (backlog, not a bug):** immediate interruption of a C-level
  blocking call (e.g. `time.sleep`) on Windows is delayed until the call
  returns — a known ipykernel/Windows event-interrupt limitation; would need
  `GenerateConsoleCtrlEvent` (jupyter's "signal" interrupt mode).

## 13. After adding a cell with `a`/`b`, Enter sometimes won't enter edit mode

- **Status:** fixed - confirmed (user, 2026-07-10: "13 looks good")
- **Symptom:** `b` created and focused a cell but Enter did nothing; the
  notebook could land in a focus/mode desync after add.
- **Fix:** `a`/`b` (and the + toolbar buttons) now insert the new cell and stay
  in COMMAND mode (select it, focus the notebook handle) instead of jumping into
  edit mode, so `enter → EnterEditMode` reliably fires. (The broader
  full-focus-loss edge remains tracked as bug #15.)

## 19. Reloading doesn't clear the conflict notification toast

- **Status:** fixed - confirmed (user 2026-07-12: reload clears it; "Overwrite
  now successfully clears the toast" / "Overwrite dismisses toast correctly")
- **Symptom:** (user 2026-07-11) After the "notebook changed on disk but you
  have unsaved changes" toast appears, reloading via the COMMAND PALETTE
  leaves the toast sitting in the corner. (Clarified: the toast's own Reload
  button DOES clear it — clicking a toast action auto-dismisses that toast —
  only reloads from outside the toast leave it up.) Any reload of that file
  should dismiss the toast, since the conflict is resolved.
- **Analysis:** the toast is shown via `workspace.show_toast` with a
  `NotificationId` (`NotebookConflictToast`); nothing dismissed it on reload.
- **Fix attempted (2026-07-11):** `reload_cells_from_notebook` — the shared
  sink for every reload path (command palette, toast button, external-change
  auto-reload) — now calls `workspace.dismiss_toast` for the conflict toast id
  (the marker type was hoisted to module scope so show and dismiss share it).
- **Reload CONFIRMED by user 2026-07-11.** Follow-up (same day): the SAVE side
  (choosing Overwrite in the conflict prompt) also re-aligns with disk but did
  NOT dismiss the toast. Factored the dismissal into `dismiss_conflict_toast`
  and call it from the confirmed-overwrite save branch too.
- **Tested:** confirmed — reload paths (2026-07-11) and the overwrite-save
  dismissal (2026-07-12).
