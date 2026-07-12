# Phase 16 — Better DataFrame (table) output rendering

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING.
> Kind: **change to existing behaviour** (restyle/harden the existing
> TableView) — keep open until the user confirms tables actually render nicer.

Context (investigated 2026-07-11): pandas DataFrames render as plain text by
default. The "setting that prints them slightly better" is a PANDAS option —
`pd.set_option('display.html.table_schema', True)` — which makes pandas emit
`application/vnd.dataresource+json`; Zed ranks that mime type highest and
renders it with the native `TableView` widget
(`crates/repl/src/outputs/table.rs`). Per the user, improve THAT path (the
TableView), not the plain-text default.

## Implemented

- [x] Restyled `TableView`: outer rounded+bordered container; header row with
      a distinct background and semibold text; horizontal hairline row
      separators + alternating row striping instead of a full per-cell border
      grid; numbers/dates stay right-aligned; nulls render as a dimmed "—".
- [x] Row cap for large frames: only the first 300 rows are rendered, with a
      "Showing first 300 of N rows" footer (the full data is still on the
      clipboard via Copy Output, which was already markdown).

## Not done (candidate follow-ups)

- Auto-enable `display.html.table_schema` in Python kernels on connect, so
  users don't have to set the pandas option per notebook. (Would inject a
  startup snippet into the kernel; needs care to not clobber user config.)
- Document the pandas option in `docs/src/repl.md` (it is currently only in
  the module's rustdoc).
- Column-count cap / horizontal virtualization for very wide frames.

## Manual test checklist (for the user)

- [ ] In a notebook: `import pandas as pd`,
      `pd.set_option('display.html.table_schema', True)`, then display a
      DataFrame → renders as a styled table (rounded border, header band,
      striped rows) instead of monospace text.
- [ ] A frame with >300 rows shows the truncation footer and stays responsive.
- [ ] Numeric columns right-aligned; missing values show a dimmed "—".
- [ ] Copy Output still yields the full table as markdown.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; tests pass.
