# Phase 23 — Collapse / expand cell input & output

> ✅ STATUS: COMPLETE & CONFIRMED (2026-07-12). Kind: **new feature** — the
> chevron input-collapse and output-collapse both work and input-collapse state
> survives reopen, so this is archived. Remaining manual-test lines moved to
> `awaiting_testing.md` (Phase 23 section); "output collapse survives reopen" is
> BLOCKED by bug #24 (outputs are dropped on save) and retests once #24 is
> fixed. User decision (2026-07-12): collapse state PERSISTS to the .ipynb.

From the backlog (useful for cells with large outputs).

Goal: let the user collapse a cell's input (the code editor) and/or its output
independently, VS Code / Jupyter style, to tame long notebooks.

Primary files: `crates/repl/src/notebook/cell.rs` (a `CellControlType`
collapse/expand scaffold already exists — `CollapseCell` / `ExpandCell`, plus
the pre-existing typo `CollapseCelln` to fix), `notebook_ui.rs` for any actions.

## Implemented

- [x] Per-cell `source_collapsed` / `outputs_collapsed` on `CodeCell`,
      initialized from `metadata.jupyter.source_hidden` / `outputs_hidden` on
      load and written back on save (`metadata_with_visibility`) — VS Code /
      Jupyter-compatible round-tripping. Toggling marks the notebook dirty
      (`CellEvent::MetadataChanged`).
- [x] Input collapse affordance: a chevron button (leftmost in the per-cell
      toolbar) toggles it; collapsed input renders as a muted one-line summary
      (first source line + ⋯) that expands on click. The cell stays runnable.
- [x] Output collapse: "Collapse/Expand Output" entry in the output "…" menu;
      collapsed output renders as a muted "Output collapsed ⋯" row that
      expands on click.
- [x] Fixed the `CellControlType::CollapseCelln` typo.

## Not done (kept out of scope)

- Dedicated `CollapseCell`/`ExpandCell` palette actions + keybinds and a
  notebook-level collapse-all — backlog if wanted.
- Markdown cells don't collapse (code cells only).

## Manual test checklist (for the user)

- [ ] Toolbar chevron collapses the input to a one-line summary; clicking the
      summary (or the chevron) expands it; the cell still runs while collapsed.
- [ ] Output "…" menu collapses/expands the output; clicking the collapsed
      row expands it.
- [ ] Collapse state survives save + reopen (and round-trips with VS Code).
- [ ] Toggling collapse marks the notebook dirty (save persists it).

## Verification (automated)

- `cargo check -p repl` + clippy clean; 42 repl tests pass.
