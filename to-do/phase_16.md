# Phase 16 — Better DataFrame (table) output rendering

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING.
> Kind: **change to existing behaviour** (restyle/harden the existing
> TableView) — keep open until the user confirms tables actually render nicer.

Context (investigated 2026-07-11): Zed has three DataFrame display paths,
ranked best→worst: (1) native `TableView` grid, used when pandas emits
table-schema JSON (`pd.set_option('display.html.table_schema', True)` — a
pandas option, not a Zed setting); (2) the DEFAULT: pandas' `text/html` repr
converted to a markdown table; (3) plain text, clipped by the Zed settings
`repl.max_lines` / `repl.max_columns` (the only Zed settings.json options that
touch output size — there is no DataFrame-specific Zed setting). User
direction (2026-07-11): make DataFrames "look/feel more like VS Code".

## Implemented

- [x] Restyled `TableView`: outer rounded+bordered container; header row with
      a distinct background and semibold text; horizontal hairline row
      separators + alternating row striping instead of a full per-cell border
      grid; numbers/dates stay right-aligned; nulls render as a dimmed "—".
- [x] Row cap for large frames: only the first 300 rows are rendered, with a
      "Showing first 300 of N rows" footer (the full data is still on the
      clipboard via Copy Output, which was already markdown).
- [x] VS Code-style DEFAULT display (round 2): HTML outputs that are
      essentially a single table (pandas' default `text/html` DataFrame repr)
      now render with the native `TableView` grid instead of a markdown text
      table — zero config needed. `table_from_markdown` parses the converted
      markdown into a `TabularDataResource`: blank index header stays blank,
      duplicate headers deduped, numeric columns detected (right-aligned,
      tolerating pandas' "..." truncation markers), pandas' trailing
      "N rows × M columns" summary tolerated. Mixed (non-table) HTML falls
      back to markdown rendering as before. Unit-tested.

## Not done (candidate follow-ups)

- Auto-enable `display.html.table_schema` in Python kernels on connect, so
  users don't have to set the pandas option per notebook. (Would inject a
  startup snippet into the kernel; needs care to not clobber user config.)
- Document the pandas option in `docs/src/repl.md` (it is currently only in
  the module's rustdoc).
- Column-count cap / horizontal virtualization for very wide frames.

## Manual test checklist (for the user)

- [ ] With NO pandas config at all: display a DataFrame → renders as a styled
      native table (rounded border, header band, striped rows), not markdown
      text. Index column header is blank; numeric columns right-aligned.
- [ ] A truncated frame (pandas shows "..." rows) still renders as a table.
- [ ] A frame with >300 rows shows the truncation footer and stays responsive.
- [ ] Non-table HTML output (e.g. `display(HTML("<h1>hi</h1>"))`) still
      renders as markdown, not a broken table.
- [ ] (Optional) with `pd.set_option('display.html.table_schema', True)`: the
      table-schema path renders with the same styling; missing values show a
      dimmed "—".
- [ ] Copy Output still yields the table as markdown.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean; tests pass.
