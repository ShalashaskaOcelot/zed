# Phase 61 — Notebook runtime timers in the kernel strip

Kind: **new feature**. Promoted from the backlog at the user's request
(2026-08-06); originally raised 2026-07-16.

Goal: settings-gated timer(s) shown next to the kernel name/status in the
TOP-RIGHT kernel strip (`render_kernel_strip`, `notebook_ui.rs:4438`) — NOT the
right sidebar. Two independent timers, each behind its own boolean setting:

1. **Total execution time** — a run-scoped tally of time spent EXECUTING.
   Counts up only while a cell is actually running and PAUSES between cells, so
   the time spent writing the next cell is not counted.
2. **Kernel uptime** — wall-clock since the kernel started, running until the
   kernel is stopped or restarted.

## Design decisions (settled here so implementation is mechanical)

### Execution total: accumulate, never sum

The naive "sum every cell's `execution_duration`" is wrong for the reason the
user flagged: completed cells keep last run's durations, so the total is
non-zero the instant a second run starts, and cell status alone can't bound
"this run" when cells are run one at a time. Instead keep an accumulator on
`NotebookEditor`:

- `execution_time_total: Duration` — time already banked.
- `execution_time_started_at: Option<Instant>` — set while something runs.

A single `sync_execution_clock` reconciles the two against "is any cell
`is_executing()`" (`running_cell_index`, `notebook_ui.rs:2266`): running with no
start → start it; not running with a start → bank `start.elapsed()` and clear.
Displayed value is `total + started_at.map(elapsed)`. This makes the pause
between cells free and needs no per-cell bookkeeping.

**Reset semantics:** reset to zero when the kernel starts, restarts, or is
changed — the timer measures "work this kernel has done", which is the only
boundary that is meaningful for the user's actual question ("how long has this
notebook been computing for"). Deliberately NOT reset per Run All: running
cells one-by-one is a normal working pattern and resetting under it would make
the number useless. `Clear Outputs` does not touch it either.

### Kernel uptime: freeze on stop

- `kernel_started_at: Option<Instant>` set when the kernel reaches a live state
  (`KernelStatus::is_alive()`, `kernels/mod.rs:770`) from a non-live one.
- On shutdown/exit the elapsed value is FROZEN (moved into
  `kernel_uptime_frozen: Option<Duration>`) and stays visible until a kernel
  starts again; a restart resets it immediately (restart = new process).

### Ticking

One repeating task on the notebook (`_runtime_timer`), mirroring the cell
timer's pattern (`cell.rs:1180`): 100 ms tick, `cx.notify()`, and the task ENDS
ITSELF once neither timer needs updating (nothing executing, and no live kernel
or uptime disabled). Started wherever a run begins (`execute_cell`) and when a
kernel comes up. Never leave a permanent 100 ms notify loop running per open
notebook.

### Formatting

Extract the cell timer's `Cell::format_duration` (`cell.rs:1431`) into a shared
free function (e.g. `notebook::format_duration`) and use it for both the cell
footer and the strip so they can never drift. Add an HOURS tier
(`Xh Ym SS.Ss`), which the cell formatter lacks today.

**Discrepancy to flag, not silently resolve:** the backlog entry asked for 2 dp
on seconds (`1.83s`, `2h 32m 43.36s`); the shipped cell timer uses 1 dp
(`1.8s`, `3m 04.2s`). Keep 1 dp so the existing in-cell display does not change
under the user, and note it for them — switching both to 2 dp is a one-line
follow-up if that is what they want.

### Settings

Two booleans, both **default false** (opt-in, matching how the backlog framed
them):

- `repl.notebook_show_execution_time`
- `repl.notebook_show_kernel_uptime`

Each needs the full four-place wiring (see `notebook_show_last_executed` for the
exact pattern): `settings_content.rs:1333` (Option field + doc comment),
`repl_settings.rs` (field + `from_settings` default), `page_data.rs:7406` (a
`SettingsPageItem` on the REPL & Notebooks page), and a line in `docs/src/repl.md`.

## Tasks

- [ ] Extract `Cell::format_duration` into a shared `format_duration` in the
      notebook module, add the hours tier, and point the cell footer at it.
      Unit-test the tiers (ms / s / m / h) — pure function, cheap to test.
- [ ] Add the two settings end to end (settings_content, repl_settings,
      settings UI page item, `docs/src/repl.md`).
- [ ] Add `execution_time_total` / `execution_time_started_at` +
      `sync_execution_clock` to `NotebookEditor`; reconcile on tick and at run
      start/finish so the banked total is not tick-quantised.
- [ ] Reset the execution total on kernel start / restart / change; make sure
      an interrupted or failed cell still banks the time it did spend running.
- [ ] Add `kernel_started_at` / `kernel_uptime_frozen`, set on kernel-alive
      transition, freeze on stop/exit, reset on restart.
- [ ] Add the `_runtime_timer` repeating task with the self-terminating
      condition; verify it stops (no busy loop) once idle.
- [ ] Render both timers in `render_kernel_strip`, each gated on its setting,
      as muted small labels left of the kernel selector. Icons: keep it text-
      only to avoid crowding the strip; tooltips name which timer is which.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] With both settings off (default) the strip is unchanged.
- [ ] `notebook_show_execution_time` on: Run All a notebook with a few slow
      cells — the number climbs while a cell runs, HOLDS between cells, and
      ends equal to roughly the sum of the cells' own durations.
- [ ] Run cells one at a time with pauses in between: the total accumulates
      across the runs and does not jump when a previously-run cell is re-run.
- [ ] Restarting the kernel zeroes the execution total.
- [ ] `notebook_show_kernel_uptime` on: counts from kernel start, keeps
      counting while idle, freezes on Stop/Shutdown, resets on Restart.
- [ ] Neither timer keeps the CPU busy when the notebook sits idle with the
      kernel stopped.
