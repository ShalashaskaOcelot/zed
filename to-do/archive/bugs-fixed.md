# Fixed bugs (confirmed)

Confirmed-fixed bugs moved out of `to-do/bugs.md`.

---

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
