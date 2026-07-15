# Phase 32 — Notebook UX niceties

Kind: **mixed** (small behaviour changes). Not yet started — this is a plan.
Promoted from the backlog to keep 5 phases in rotation after phase 27
completed. Bundles three small interaction items.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/notebook/cell.rs`.

## Tasks

- [ ] Esc in command mode with a multi-cell selection collapses it back to a
      single-cell selection on the primary/focused cell (user 2026-07-14,
      phase 26 follow-up). Esc's existing mode/focus recovery behaviour stays.
- [ ] Markdown cell rendered-preview toggle improvements: render the preview
      when EXITING edit mode (Esc / focus loss), not only via the explicit
      toggle, so a markdown cell doesn't linger as raw source.
- [ ] Return keyboard focus to the notebook after toolbar-button / popover
      interactions so command-mode shortcuts keep working without clicking a
      cell (bug #15 follow-up — the remaining "focus fully lost" edge).

## Risks / gaps

- The focus-return item overlaps bug #15's ongoing observation; if the
  on_focus mitigations already cover a case, don't double-handle it.
- Esc collapse must not interfere with Esc-to-command-mode from edit mode
  (different context) or Esc dismissing pickers/popovers (those should win).

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: multi-select then Esc → single selection on the primary; markdown
  cells re-render on exit-edit; after clicking a toolbar button, command-mode
  keys still work without re-clicking a cell.
