# Phase 23 — Collapse / expand cell input & output

Kind: **new feature**. Not yet started — this is a plan. From the backlog
(useful for cells with large outputs).

Goal: let the user collapse a cell's input (the code editor) and/or its output
independently, VS Code / Jupyter style, to tame long notebooks.

Primary files: `crates/repl/src/notebook/cell.rs` (a `CellControlType`
collapse/expand scaffold already exists — `CollapseCell` / `ExpandCell`, plus
the pre-existing typo `CollapseCelln` to fix), `notebook_ui.rs` for any actions.

## Tasks

- [ ] Track per-cell collapsed state for input and output separately (two
      bools on the cell).
- [ ] A collapse/expand affordance (chevron in the gutter or cell chrome) that
      toggles input collapse; likewise for output (near the output "…" menu).
- [ ] When input is collapsed, show a compact placeholder (e.g. first line +
      an expand chevron) instead of the full editor; same idea for output.
- [ ] Optional: actions `CollapseCell` / `ExpandCell` (already scaffolded) +
      command palette / keybind; and a notebook-level collapse-all/expand-all.
- [ ] Persist collapse state in the notebook metadata if round-tripping is
      desired (nbformat supports `jupyter.source_hidden` / `outputs_hidden`
      cell metadata) — otherwise session-only.
- [ ] Fix the `CellControlType::CollapseCelln` typo while here.

## Risks / gaps

- Collapsed editors must not lose content or break focus/execution.
- Decide session-only vs persisted (nbformat metadata) up front.

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- User test: collapse/expand input and output independently; run still works on
  a collapsed cell; (if persisted) state survives save/reload.
