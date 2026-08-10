# Phase 68 — Kernel strip states and the Run All timer reset

Kind: **mixed** — item 1 is a change to existing behaviour (if the strip still
says "Starting" during a build, it stays open), item 2 is a new opt-in setting.

Promoted from the backlog 2026-08-10 to restore the runway after phase 66
completed. Both items were raised by the user on 2026-08-06 and are small and
self-contained; they share the kernel strip / notebook timer code, so they are
one phase rather than two.

Primary file: `crates/repl/src/notebook/notebook_ui.rs`.

## Item 1 — The strip should say "creating", not "starting"

Reported 2026-08-06 with a screenshot. While an environment is being built the
kernel picker's row reads "Creating environment…" but the TOP-RIGHT strip reads
"Starting", which is misleading: nothing is starting yet, a build is running and
the kernel launch only follows it.

Phase 48 deliberately mapped `creating_kernel_name` onto `KernelStatus::Starting`
for the strip (`render_kernel_strip`) so the new env showed up immediately. This
wants its own state instead, with its own label — and, since phase 62, it gets
the spinning icon for free. Keep the two distinct so the sequence reads
properly: creating → starting → idle.

Note this is only the STRIP. The picker's own row already says the right thing,
and its failure to refresh when the build finishes is bug #58 — a different
defect, still open and untouched by this phase.

## Item 2 — "Run All resets the tally"

The user confirmed phase 61's execution timer and asked for this specific
option. It is only meaningful when `repl.notebook_show_execution_time` is on:
with it, Run All zeroes the tally first, so the number reads as the wall-clock
cost of that ONE end-to-end pass rather than everything run since the kernel
started.

**Where the reset goes (do not get this wrong):** reset
`execution_time_banked` / `execution_time_started_at` in `run_cells` (the
`RunAll` action) BEFORE it calls `run_cell_batch` — NOT inside `run_cell_batch`,
which is shared with Run Above, Run Below and multi-select run. None of those
mean "the whole notebook", and resetting there would silently redefine the
number for all of them.

There is a competing model in the backlog (per-cell accounting, so a re-run
replaces that cell's contribution rather than adding to it). It is parked and
the user is unsure it is worth building. If it is ever built it largely subsumes
this option — at that point replace the setting PAIR with one enum
(`session` | `notebook`) rather than stacking a third boolean. Leave a comment
on the new setting saying so, so the next person doesn't stack.

## Tasks

- [ ] Give the kernel strip its own "creating" state rather than borrowing
      `KernelStatus::Starting`, with its own label and the existing spinner.
      Make sure the real Starting state still shows when the build finishes and
      the kernel actually launches.
- [ ] Check the other things keyed off the strip's status (the failure count,
      the timers, the tooltip) still read correctly in the new state — a build
      is not a running kernel and must not, say, start the uptime clock.
- [ ] Add `repl.notebook_reset_execution_time_on_run_all` (default `false`) end
      to end: `settings_content`, `repl_settings`, the settings UI page item,
      `assets/settings/default.json`, `docs/src/repl.md`.
- [ ] Reset the tally in `run_cells` only, before `run_cell_batch`, and comment
      why it cannot live in the shared path.
- [ ] Unit-test the reset: with the setting on, a Run All zeroes a non-zero
      banked total; with it off, the total carries over; and Run Above / Run
      Below / a multi-cell selection never reset regardless of the setting.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] Create an environment: the strip reads "Creating…" (spinning) for the
      whole build, then switches to "Starting" when the kernel actually
      launches, then "Idle". The picker row is unchanged.
- [ ] With `notebook_show_execution_time` on and the new setting ON: run some
      cells individually, note the tally, then Run All — the tally restarts from
      zero and ends as the cost of that pass.
- [ ] Same but with the new setting OFF (the default): Run All adds to the
      existing tally exactly as it does today.
- [ ] Run Above / Run Below / running a multi-cell selection never reset the
      tally, whichever way the setting is set.
