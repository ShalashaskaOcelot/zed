# Phase 31 — Kernel launch robustness

Kind: **mixed** (a behaviour change + hardening). Not yet started — this is a
plan. Promoted from the backlog to keep 5 phases in rotation after phase 25
completed. Bundles the two remaining kernel-launch backlog items (both are
follow-ups to bug #6, which is still `fix attempted - untested` on Windows).

Primary files: `crates/repl/src/kernels/native_kernel.rs` (launch path),
`crates/repl/src/kernels/mod.rs`, `wsl_kernel.rs` (reference implementation
for stderr capture).

## Tasks

- [ ] Replace the fixed 500ms native-launch readiness sleep with a proper
      readiness handshake: after spawning, poll a `kernel_info_request` (or
      heartbeat) with a timeout until the kernel answers, instead of hoping
      500ms is enough. Keeps slow machines from hitting os error 10054 and
      fast machines from waiting longer than needed.
- [ ] Surface kernel stderr in the UI on POST-CONNECT launch failures: the WSL
      path captures stderr, and the native path now captures it on premature
      exit — extend to failures after the sockets connect (e.g. the kernel
      dies during the handshake), so the user sees the kernel's own error
      instead of a bare socket error.

## Risks / gaps

- The handshake must respect the existing unique-connection-file-per-launch
  and restart flows (bug #1/#6 fixes) — don't reintroduce a race by retrying
  against a stale socket.
- Keep a hard timeout (a few seconds) with a clear error, or a broken kernel
  would spin forever in "Starting".

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test (Windows especially): kernels start reliably from a fresh app
  start; a deliberately broken kernel (e.g. corrupt env) reports its stderr
  in the cell error instead of a socket error.
