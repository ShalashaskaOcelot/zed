# Phase 35 — Prompt interrupt of C-blocking calls on Windows

Kind: **change to existing behaviour**. ⚠️ AWAITING USER TESTING.
Promoted from the backlog (follow-up to bug #3) to keep 5 phases in rotation
after phase 32 completed.

## Findings that changed the plan (research 2026-07-16)

The originally planned mechanism (`CREATE_NEW_PROCESS_GROUP` +
`GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT)`) was researched against primary
sources and REJECTED:

- The real root cause is in CPython (`Modules/signalmodule.c`, verified on
  3.9→main): the hidden "sigint event" that wakes main-thread C blockers
  (`time.sleep`, `input()`) is set ONLY in `signal_handler()` — the handler
  for REAL OS signals. `_thread.interrupt_main()` — what ipykernel's poller
  calls when our `JPY_INTERRUPT_EVENT` fires — goes through
  `PyErr_SetInterruptEx` → `trip_signal`, which only trips the
  between-bytecodes flag. Hence the observed symptom on every
  ipykernel/Python version: pure-Python loops interrupt, C blockers don't.
- Jupyter's `interrupt_mode: "signal"` on Windows IS the event mechanism
  (jupyter_client `win_interrupt.py`/`local_provisioner.py`) — CTRL_BREAK is
  not used by the ecosystem for kernel interrupts.
- CTRL_BREAK would TERMINATE kernels: ipykernel installs no SIGBREAK
  handler, and Python's default SIGBREAK disposition kills the process.
- `CREATE_NEW_PROCESS_GROUP` would be actively harmful: it starts the child
  with Ctrl+C DISABLED, breaking the mechanism that does work.
- `GenerateConsoleCtrlEvent` only reaches the CALLER's console; Zed is a
  console-less GUI app, so it must attach to the kernel's hidden console
  first (kernels get their own via `CREATE_NO_WINDOW`).
- ipykernel's control-channel `interrupt_request` is explicitly UNSUPPORTED
  on Windows ("Interrupt message not supported on Windows") — the old
  message fallback was a no-op for Python kernels.

Chosen mechanism: deliver a REAL `CTRL_C_EVENT` on the kernel's own hidden
console — `AttachConsole(kernel_pid)` → `GenerateConsoleCtrlEvent(CTRL_C, 0)`
→ `FreeConsole`, serialized process-globally, with Zed permanently ignoring
CTRL_C (`SetConsoleCtrlHandler(NULL, TRUE)`) so the broadcast can't kill Zed.
A real console event runs CPython's `signal_handler` → sets the sigint event
→ `time.sleep` wakes immediately, exactly like a terminal Ctrl+C. Scope
containment mirrors Unix `killpg`: the hidden console holds only the kernel's
process tree. `AttachConsole`'s failure modes are the safety gate (fails if
Zed owns a console or the kernel has none) → fall back to the JPY event.

## Tasks

- [x] Windows `Child::interrupt`: prefer a real console CTRL_C via the
      attach/generate/free dance (`util::process::windows_interrupt::send_ctrl_c`);
      fall back to the existing `JPY_INTERRUPT_EVENT` when the console path
      fails. (`CREATE_NEW_PROCESS_GROUP`/CTRL_BREAK dropped — see findings.)
- [x] Respect `interrupt_mode: "message"` from the kernelspec: such kernels
      get a control-channel `interrupt_request` INSTEAD of any OS-level
      signal (`NativeRunningKernel::interrupt`); signal-mode kernels keep the
      message send only as a last-resort fallback.
- [x] Windows test: `ping.exe` (which ignores the JPY event) exits after
      `send_ctrl_c` — proves the console path specifically, on a
      `CREATE_NO_WINDOW` child console so the test runner's console is never
      touched.
- [x] WSL and unix paths untouched (unix still `killpg(SIGINT)`; WSL kernel
      keeps its own interrupt).

## Manual user verification (Windows)

- [ ] ⚠ untested — Interrupt a `time.sleep(60)` cell: it stops IMMEDIATELY
      with KeyboardInterrupt (not after the sleep finishes). Also worth
      trying: an `input()` call and a pure-Python `while True: pass` loop
      (the latter already worked and must keep working).
- [ ] ⚠ untested — Zed itself is unaffected: interrupting repeatedly (also
      two notebooks with different kernels back-to-back) never closes or
      hangs Zed, and the kernel survives (interrupt, not restart — the cell
      shows the red ✕ KeyboardInterrupt, kernel runs the next cell fine).

## Risks / gaps

- Kernels with no console (a kernelspec pointing at a GUI-subsystem binary
  like pythonw) fall back to the event path — same behaviour as before this
  phase, not a regression.
- A cell using top-level `await` runs on ipykernel's async path where SIGINT
  is converted to coroutine cancellation; a C-blocking call inside such a
  cell still can't be promptly interrupted — that's ipykernel's design and
  is identical in a terminal.
- Non-Python signal-mode kernels that install no console handler would be
  terminated by a real CTRL_C (default console handler). This mirrors Unix
  SIGINT-default-terminate semantics for signal-mode kernels; kernels that
  can't handle signals must declare `interrupt_mode: "message"`, which we
  now honor.

## Verification

- `cargo check -p repl -p util` (Linux + `--target x86_64-pc-windows-msvc`),
  `./script/clippy -p util -p repl`, Windows-target clippy on util, and
  `cargo test -p util -p repl` all clean.
- The two ⚠ user tests above on the user's Windows machine.
