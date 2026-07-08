# Bugs

Status values: `open` | `fix attempted - untested` | `fixed - confirmed`.
Never attempt a further fix while a bug is `fix attempted - untested`.
Move to `to-do/archive/` only when `fixed - confirmed`.

---

## 1. Restart kernel kills the kernel but the relaunch fails

- **Status:** open
- **Symptom:** Restart button / ctrl-shift-r kills the kernel; it never comes
  back. Running a cell afterwards reports the kernel is not running.
- **Analysis:** Restart IS shutdown + relaunch (`notebook_ui.rs:491-502`),
  but: `force_shutdown().detach()` is fire-and-forget with no await/delay;
  the connection file path is keyed only on `entity_id`
  (`native_kernel.rs:144`) so the relaunch reuses the same path while the old
  kernel's `Drop` deletes it (`native_kernel.rs:300-305`); ephemeral-port
  TOCTOU (`native_kernel.rs:79-91`). The REPL's restart (`session.rs:873-913`)
  sequences this correctly (ShutdownRequest{restart:true}, wait, await forced
  kill, relaunch) — notebook should match.
- **Fix attempted:** none
- **Tested:** n/a

## 2. Running a cell with a dead/shutdown kernel does not start the kernel

- **Status:** open
- **Symptom:** After the kernel is killed, running a cell just errors
  ("the kernel is shut down" / "failed to launch") instead of starting the
  selected kernel.
- **Analysis:** `execute_cell` (`notebook_ui.rs:518-567`) has no relaunch
  branch for `Shutdown` / `ErroredLaunch`; it only renders an error output.
  It also does not queue executions while `StartingKernel` (the REPL does,
  `session.rs:683-793`).
- **Fix attempted:** none
- **Tested:** n/a

## 3. Interrupt is greyed out / ctrl-c does nothing

- **Status:** open
- **Symptom:** Stop button always disabled; ctrl-c has no effect.
- **Analysis:** Interrupt is implemented (protocol-level `InterruptRequest`
  on the control channel, `kernels/mod.rs:156-160`; handler
  `notebook_ui.rs:504-516`). The button is enabled ONLY when
  `KernelStatus::Busy` (`notebook_ui.rs:1253-1256`); a kernel that errored or
  died reports `Error`, so the button greys out exactly when the user wants
  it. The ctrl-c handler early-returns unless `Kernel::RunningKernel` and
  swallows send failures with `try_send(...).ok()`.
- **Fix attempted:** none
- **Tested:** n/a

## 4. "More options" toolbar button opens nothing

- **Status:** open
- **Symptom:** The Ellipsis button at the bottom of the right toolbar does
  nothing when clicked.
- **Analysis:** Dead stub — tooltip only, no `on_click`, no popover
  (`notebook_ui.rs:1124-1127`). No menu content was ever defined.
- **Fix attempted:** none
- **Tested:** n/a

## 5. Output "..." (ellipsis) button next to cell output does nothing

- **Status:** open
- **Symptom:** Clicking the three-dot button next to a cell's output does
  nothing.
- **Analysis:** Bare `IconButton::new("control", IconName::Ellipsis)` with no
  handler, no tooltip, no menu (`cell.rs:924-939`). The per-output-type copy
  buttons in `outputs.rs:200-221` are separate and DO work.
- **Fix attempted:** none
- **Tested:** n/a

## 6. Native kernel launch is flaky on Windows (os error 10054)

- **Status:** open
- **Symptom:** "Kernel Error: cell could not be executed — the kernel failed
  to launch: handling failed for recv task: control recv: Codec Error: An
  existing connection was forcibly closed by the remote host. (os error
  10054)"
- **Analysis:** Message chain fully traced: control-socket read failure
  (`kernels/mod.rs:144-147`) → recv task bails → `kernel_errored`
  (`mod.rs:190-197`) → `Kernel::ErroredLaunch` → error rendered by
  `execute_cell`/`show_kernel_error`. Root cause: the native launch path
  (`native_kernel.rs:112-255`) connects to the kernel's sockets immediately
  after spawn with no readiness wait and no premature-exit check — the WSL
  path (`wsl_kernel.rs:290-323`) does both (2s wait + `try_status()` +
  stderr capture). Also aggravated by the restart races in bug #1.
- **Fix attempted:** none
- **Tested:** n/a

## 7. Restart does not clear per-execution state

- **Status:** open
- **Symptom:** (found in code review, not user-reported) After a restart,
  stale `msg_id → CellId` entries linger in `execution_requests`; incoming
  messages could be routed to old cells.
- **Analysis:** `restart_kernel` doesn't clear `self.execution_requests`;
  `change_kernel` does (`notebook_ui.rs:486`).
- **Fix attempted:** none
- **Tested:** n/a

## 8. Clean kernel exit leaves stale RunningKernel state

- **Status:** open
- **Symptom:** (found in code review) If the kernel process exits with a
  success status, the UI keeps showing a running kernel.
- **Analysis:** The process-exit watcher only reports failed exits
  (`native_kernel.rs:220-242`); a zero-status exit returns silently without
  a state transition.
- **Fix attempted:** none
- **Tested:** n/a

## 9. Notebook never reports itself dirty

- **Status:** open
- **Symptom:** (found in code review) Structural/metadata changes (add/move
  cells, outputs) don't mark the notebook modified, so closing may not
  prompt to save.
- **Analysis:** `NotebookItem::is_dirty` is hardcoded `false` with a TODO
  (`notebook_ui.rs:1595-1598`).
- **Fix attempted:** none
- **Tested:** n/a
