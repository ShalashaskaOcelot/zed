# Phase 41 — Cell grouping

Kind: **new feature**. Not yet started — this is a plan. Promoted from the
backlog (the LAST backlog item) to keep 5 phases in rotation after phase 33
completed. User priority: LOW ("rarely used") — if fresher backlog items
arrive before this starts, consider re-backlogging it in favour of a phase
built from those.

Group contiguous cells so they can be collapsed/expanded and acted on as a
unit — the notebook equivalent of code folding for sections (typically headed
by a markdown cell).

Primary files: `crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/notebook/cell.rs`.

## Tasks

- [ ] Define grouping semantics: a markdown HEADING cell implicitly heads a
      group that runs until the next heading of equal/higher level (JupyterLab
      "collapsible headings" model — no new metadata needed, works with
      existing notebooks).
- [ ] Collapse/expand a group from the heading cell (gutter chevron +
      keyboard shortcut in command mode); collapsed groups render the heading
      plus a "N cells hidden" placeholder.
- [ ] Persist collapsed state in cell metadata the way JupyterLab does
      (`jupyter.outputs_hidden`-style key; match JupyterLab's actual key so
      state round-trips between clients).
- [ ] Run-group action: run all cells in the group (reuses the sequential
      run queue from phase 10).

## Risks / gaps

- Selection/navigation must skip hidden cells without corrupting
  `selected_indices`/`cell_order` bookkeeping (multi-select, move up/down).
- Cell operations (delete/paste/undo) on a collapsed group need defined
  behaviour — simplest v1: auto-expand before structural edits.

## Verification

- clippy clean; `cargo test -p repl` passes.
- User test: headings show a chevron; collapse hides the section with a
  placeholder; state survives save/reopen (and JupyterLab agrees); run-group
  runs exactly the section's cells in order.
