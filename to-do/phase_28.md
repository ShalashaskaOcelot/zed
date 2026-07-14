# Phase 28 — Per-cell "last executed time"

Kind: **new feature**. Not yet started — this is a plan. Promoted from the
backlog (user 2026-07-12, refined 2026-07-14).

Show WHEN a cell was last executed, alongside the existing ✓ + duration.

Primary files: `crates/repl/src/notebook/cell.rs` (record/read/display the
timestamp), `crates/repl/src/repl_settings.rs` +
`crates/settings_content/src/settings_content.rs` (the setting).

## Tasks

- [ ] Record the run-COMPLETION time when a cell finishes (in
      `finish_execution`), stored VS Code / Jupyter compatibly in cell
      `metadata.execution` — the standard keys `shell.execute_reply` /
      `iopub.status.idle` (and optionally `iopub.execute_input` for the start),
      as ISO 8601 strings. Write them on `to_nbformat_cell` and read them on
      `load` so timings round-trip BOTH ways (a notebook run in VS Code shows
      its times in Zed, and vice versa).
- [ ] Display a PROPER TIMESTAMP (e.g. `14:32:05` / a short date-time), NOT a
      relative "2m ago", near the ✓ + duration in `execution_status_element`.
- [ ] Gate the display behind a setting (default off or on — decide during
      implementation; VS Code shows it), e.g. `notebook_show_last_executed`.

## Risks / gaps

- `metadata.execution` must match VS Code's shape exactly for seamless interop
  (verify against a notebook VS Code has run).
- Timezone/format: store UTC ISO 8601 in metadata; format to local for display.
- Don't let the extra metadata churn trigger spurious dirty/conflict state
  (interacts with the phase-14/bug-21 save-conflict logic).

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: run a cell → the completion timestamp shows; save/reopen → it
  persists; open a VS Code-run notebook → its times display; run in Zed, open
  in VS Code → VS Code shows the times.
