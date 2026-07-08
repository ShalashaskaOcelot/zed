# Phase 2 — Kernel lifecycle fixes

Goal: make the kernel lifecycle dependable: restart actually restarts,
running a cell revives a dead kernel, interrupt works, launches stop failing
with 10054. This phase is bug-driven (bugs #1, #2, #3, #6, #7, #8) — update
`bugs.md` statuses as fixes land.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/kernels/native_kernel.rs`, `crates/repl/src/kernels/mod.rs`.
Reference implementation for sequencing: `crates/repl/src/session.rs:873-913`.

## Tasks

- [ ] Rework `restart_kernel` (bug #1): follow the REPL sequence — send
      `ShutdownRequest { restart: true }` to a running kernel, bounded wait,
      await the forced kill, and only then relaunch. Show `Restarting` state
      throughout. Clear `execution_requests` on restart (bug #7).
- [ ] Make connection files unique per launch (bug #1): include a launch
      nonce/counter in the `kernel-zed-{entity_id}.json` filename so an old
      kernel's `Drop` cannot delete the new kernel's connection file.
- [ ] Add native-launch readiness handling (bug #6): after spawning, check
      `try_status()` for premature exit and capture stderr for the error
      message; wait briefly before connecting sockets (mirror
      `wsl_kernel.rs:290-323`). Prefer a kernel_info/heartbeat handshake over
      a fixed sleep if practical.
- [ ] Auto-start kernel on run (bug #2): in `execute_cell`, when the kernel
      is `Shutdown` or `ErroredLaunch` and a kernel spec is selected, launch
      it and queue the execution to run once the kernel is ready, instead of
      rendering an error output.
- [ ] Fix interrupt (bug #3): enable the stop button whenever the kernel is
      connected (`Idle` or `Busy`) rather than `Busy` only; surface
      `try_send` failures with `.log_err()` instead of `.ok()`; make the
      handler give feedback (toast or status) when there is nothing to
      interrupt.
- [ ] Handle clean kernel exit (bug #8): transition kernel state (e.g. to
      `Shutdown`) when the process exits with success status.
- [ ] Manual test checklist for the user (Windows): restart while idle,
      restart while busy, run cell after kill, interrupt a long-running cell,
      repeated restarts in quick succession.

## Notes

- Do not archive this phase until every box is ticked or explicitly moved to
  backlog/a later phase. Bugs remain `fix attempted - untested` until the
  user confirms on their machine (all these are runtime/Windows behaviours).
