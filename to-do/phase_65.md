# Phase 65 — Rust (evcxr) build chatter must not look like errors

Kind: **change to existing behaviour**. Promoted from the backlog 2026-08-06
to replace the parked release-engineering phases. Reported by the user
2026-07-20.

Goal: running a Rust cell shows its build/diagnostic chatter as ordinary
output, not as a wall of red `ERROR:` entries, while REAL Rust errors stay
clearly errors.

## Why this happens

`evcxr_jupyter` writes EVERYTHING to stderr — `Compiling {crate}` progress
lines, warnings, notes — because it reserves stdout for the user program's
actual output. Every Rust cell therefore floods the output block with
error-styled entries that are not errors.

## Key facts (code inspection 2026-08-06)

- The notebook's output path does NOT branch on the stream NAME: an iopub
  `StreamContent` message is appended to the cell's `TerminalOutput` via
  `apply_terminal_text` regardless of `name: stdout|stderr`
  (`outputs.rs:682`). Real errors arrive as `ErrorOutput` (rendered by
  `ErrorView` with ename/evalue/traceback) or as an `ExecuteReply` with
  `status: "error"`.
- So the red `ERROR:` lines the user sees are NOT simply "stderr rendered red"
  — something else is producing them. Candidates: evcxr genuinely sending
  `ErrorOutput` messages for compile progress, `ErrorView`'s own formatting, or
  the kernel-process stderr path (`kernels/native_kernel.rs:288-315`, which
  logs stderr at `Level::Error` and keeps a tail for death diagnostics)
  surfacing into the UI. **This must be established from real messages before
  anything is changed** — a fix aimed at the wrong layer will either do nothing
  or swallow real errors.

## FINDING (2026-08-06) — the flood is in the LOG, not in cell output

Established from evcxr's own source plus this repo's code; no guessing was
needed after all.

**What evcxr sends** (`evcxr_jupyter/src/core.rs`): build chatter goes out as
ordinary `stream` messages — `pass_output_line` sends `{"name": "stdout" |
"stderr", "text": …}` — and errors go out ONLY via `emit_errors`, as `error`
messages with a hardcoded `ename: "Error"`. It does NOT send progress or
warnings as errors. So the premise "evcxr's stderr arrives as error messages"
is false.

**What Zed does with them.** Cell output never styled stderr as an error:
`StreamContent` is appended to the cell's `TerminalOutput` regardless of the
stream name, and only `ErrorOutput` / an `ExecuteReply` with `status: error`
produce the red treatment and the failed-cell marker. Now pinned by
`test_push_message_stderr_is_not_an_error`.

**Where `ERROR:` really came from.** The kernel PROCESS reader in
`native_kernel.rs` tagged every stderr line `log::Level::Error` and logged it
as `ERROR kernel: …`. For evcxr — which puts all build output on stderr — a
perfectly successful run wrote a screenful of ERROR lines into Zed's log. That
matches "floods with `ERROR: compiling {crate}`" exactly, including the shape
of the text. Fixed: stderr is logged at INFO as `kernel stderr: …`, and the
death-diagnostics tail now keys off the STREAM the line arrived on rather than
its log level (so a dead kernel still quotes its last words). Genuine failures
still log at ERROR from the process-exit path.

## OPEN QUESTION for the user (blocks the rest)

Where were the `ERROR:` entries you saw — Zed's **log** (`zed: open log`), or
inside a **cell's output block**? Everything above fixes the log. If they were
in cell output, then something is producing `ErrorOutput` messages that this
analysis says shouldn't exist, and a screenshot of one Rust cell (plus
`zed: open log` around that run) would pin it down. Do not add a
pattern-matching filter before that is answered — a filter aimed at the wrong
layer either does nothing or swallows real errors.

## Tasks

- [x] Capture what evcxr actually sends for a plain Rust cell (log every
      `JupyterMessageContent` variant + `stream.name` for one run, or read
      evcxr's source) and record the finding in this file. Everything below
      depends on it.
- [x] Gate the error styling on the message TYPE / reply status rather than on
      anything stderr-shaped, so ordinary build chatter renders as normal
      output. (Cell output already did; the process logger did not, and now
      does.)
- [ ] If (and only if) evcxr really does send errors for progress lines, add a
      narrowly-scoped, kernel-language-aware filter for known-benign evcxr
      patterns — documented as an evcxr workaround, not a general stderr rule,
      and never applied to other kernels. **Evidence says NOT needed**; resolve
      this line (do it or drop it) once the open question above is answered.
- [ ] Make sure a genuine Rust error (type error, panic) still renders as an
      error with its traceback, and that the cell still goes red / counts
      toward phase 60's failure indicator. (True by construction — errors key
      off `ErrorOutput` / `ExecuteReply`, untouched by this phase — but it
      needs one runtime check on a machine with evcxr.)
- [x] Unit-test the classification (message shape → error vs normal output) so
      the rule is pinned down without needing a Rust kernel in CI.
- [x] `./script/clippy` clean and `cargo test -p repl` passes (53).

## User tests (runtime)

- [ ] Run a Rust cell that compiles cleanly: the compile progress appears as
      ordinary (non-red) output, or not at all — no `ERROR:` entries.
- [ ] Run a Rust cell with a genuine compile error: it is shown as an error,
      the cell goes red, and the kernel-strip failure count picks it up.
- [ ] Run a Rust cell that emits a `warning:` — it reads as a warning/normal
      output, not an error.
- [ ] Python cells are unaffected: a Python `print` to stderr still looks
      exactly as it did, and Python exceptions still render as errors.
