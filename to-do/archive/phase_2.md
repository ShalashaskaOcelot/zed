# Phase 2 — Kernel lifecycle fixes (COMPLETE, archived 2026-07-08)

Goal: make the kernel lifecycle dependable: restart actually restarts,
running a cell revives a dead kernel, interrupt works, launches stop failing
with 10054. This phase is bug-driven (bugs #1, #2, #3, #6, #7, #8) — the
bugs remain in `bugs.md` as `fix attempted - untested` until the user
confirms them on Windows; only the implementation work is archived here.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/kernels/native_kernel.rs`, `crates/repl/src/kernels/mod.rs`.
Reference implementation for sequencing: `crates/repl/src/session.rs:873-913`.

## Tasks

- [x] Rework `restart_kernel` (bug #1): follows the REPL sequence — sends
      `ShutdownRequest { restart: true }`, 1s grace, awaits the forced kill,
      then relaunches. Shows `Restarting` throughout. Clears
      `execution_requests` and stops cell spinners (bug #7). A restart while
      a launch is already in flight is a no-op instead of racing it.
- [x] Make connection files unique per launch (bug #1): connection file
      names now include a per-launch UUID (native and WSL), so an old
      kernel's `Drop` can no longer delete the new kernel's file.
- [x] Add native-launch readiness handling (bug #6): after spawning, waits
      500ms, checks `try_status()` for premature exit and reports the
      kernel's stderr in the error (mirrors `wsl_kernel.rs`). The
      kernel_info/heartbeat handshake variant was NOT done — moved to
      backlog as a follow-up if 10054 persists.
- [x] Auto-start kernel on run (bug #2): `execute_cell` queues the cell and
      relaunches when the kernel is `Shutdown`/`ErroredLaunch`; queues while
      `StartingKernel`/`Restarting`; queued cells run in order on launch or
      show the launch error on failure. Test updated
      (`test_run_cell_with_missing_interpreter_shows_error`).
- [x] Fix interrupt (bug #3): stop button enabled whenever connected
      (Idle or Busy); send failures logged; no-kernel case logs a warning.
- [x] Handle clean kernel exit (bug #8): new `KernelSession::kernel_exited`
      called by native/WSL process watchers on successful exit; notebook and
      REPL transition to `Shutdown`.
- [x] Manual test checklist for the user (Windows) — delivered (see bugs.md
      statuses and the session summary): restart while idle, restart while
      busy, run cell after kill, interrupt a long-running cell, repeated
      restarts in quick succession, run `exit()` in a cell.

## Additional root-cause fix found during implementation

The kernel's message-handling tasks (`start_kernel_tasks` supervisor) were
detached, so a killed kernel's recv task errored asynchronously AFTER the
kill and called `kernel_errored`, clobbering the replacement kernel's state
with `ErroredLaunch`. This is the most likely reason restart appeared to
"kill but never restart". Fixed by returning the supervisor task from
`start_kernel_tasks` and storing it on each running kernel (native, WSL,
SSH; remote already owned its tasks), so dropping/killing a kernel cancels
its message handling. `NotebookEditor::kernel_errored` additionally ignores
errors that arrive while `Restarting`/`ShuttingDown`.

## Verification

- `cargo check -p repl` clean, `./script/clippy -p repl` clean.
- `cargo test -p repl`: 37 passed, 0 failed.
