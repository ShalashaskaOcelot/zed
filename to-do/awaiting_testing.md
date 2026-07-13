# Awaiting user testing

Outstanding MANUAL TEST items from phases whose implementation is complete
(the phase files themselves are archived). Nothing here needs implementation —
these are tasks for the user to verify at runtime.

Workflow: tick an item the moment the user confirms it. If a test FAILS, file
it in `bugs.md` (or as a backlog/phase item per the kind rules), annotate the
line with the bug number, and tick it here (the follow-up is tracked
elsewhere). Delete a section once all its boxes are ticked.

## Phase 5 — Create Python environments from the kernel picker

- [ ] No Python on PATH: clicking "Create Python Environment" shows a clear
      error toast (instead of failing silently). (Needs a machine/session
      where `python3`/`python` isn't on PATH.)

## Phase 17 — Execution status & queue correctness

- [ ] Interrupt mid-batch: the interrupted (running) cell now shows the muted
      ✕ Cancelled (traceback still visible) instead of ✓ + time.
      (Change made 2026-07-12, commit 051c3c6.)
- [ ] Rerun during a run no longer hangs: pressing Run All while cells are
      running interrupts the old run and the new batch actually starts
      (deadlock fix 051c3c6 — previously everything could sit stuck if the
      kernel wasn't busy at that instant).
- [ ] Re-run after a run finishes: cells can be re-queued/run again without a
      kernel restart (regression fix 93f8a96).
- [ ] Ongoing observation: no cells get stuck showing "Running" (channel-race
      fix 5fcfca1 — user monitoring, nothing seen since).

