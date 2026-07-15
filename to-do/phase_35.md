# Phase 35 — Prompt interrupt of C-blocking calls on Windows

Kind: **change to existing behaviour**. Not yet started — this is a plan.
Promoted from the backlog (follow-up to bug #3) to keep 5 phases in rotation
after phase 32 completed. This is the last substantial backlog item.

Today the event-based Windows interrupt sets Python's interrupt flag, which a
pure-Python loop honors promptly — but a C-level blocking call (e.g.
`time.sleep(60)`) only notices when it returns. Jupyter's "signal" interrupt
mode fixes this by launching the kernel in a NEW PROCESS GROUP and sending
`GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT)` to interrupt promptly.

Primary files: `crates/repl/src/kernels/native_kernel.rs` (launch:
CREATE_NEW_PROCESS_GROUP creation flag), `crates/repl/src/kernels/mod.rs`
(interrupt path), Windows-only code paths.

## Tasks

- [ ] Launch native kernels on Windows in a new process group
      (`CREATE_NEW_PROCESS_GROUP` creation flag) with
      `interrupt_mode: "signal"` semantics.
- [ ] Interrupt via `GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)` (falling
      back to the existing event-based interrupt for kernels that request
      `interrupt_mode: "message"` or when the console event fails).
- [ ] Verify a `time.sleep(60)` cell interrupts immediately; keep the WSL and
      unix paths untouched (unix already signals SIGINT).

## Risks / gaps

- Console control events have process-group subtleties (the child must not
  share the parent's console group, or the CTRL_BREAK hits Zed too).
- Some kernels register their own CTRL_BREAK handlers; respect
  `interrupt_mode` from the kernelspec when present.

## Verification

- `cargo check -p repl` + clippy clean; tests pass (Windows CI/user machine
  for runtime confirmation).
- User test (Windows): interrupt a `time.sleep(60)` cell → it stops
  immediately with KeyboardInterrupt, not after the sleep finishes.
