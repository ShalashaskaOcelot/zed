# Fixed bugs (confirmed)

Confirmed-fixed bugs moved out of `to-do/bugs.md`.

---

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
