# Phase 12 — Create & open new notebooks (COMPLETE, archived 2026-07-11)

> ✅ STATUS: CONFIRMED by the user 2026-07-11 ("Create notebook both ways
> works"). Kind: **new feature** — archived. Two follow-ups filed as new items
> (not reopens): (1) the "New Jupyter Notebook" command should open a truly
> UNSAVED buffer like Ctrl-N rather than writing `Untitled.ipynb` to disk →
> backlog; (2) a one-off shift-enter focus jump on a brand-new notebook →
> bug #17.

Goal: VS Code parity for getting a notebook to exist. Today you must duplicate
an existing `.ipynb` and empty it; an empty/new `.ipynb` fails to open because
it is not valid nbformat JSON.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (open/parse path,
`parse_notebook_text`), the project item registration for `.ipynb`, and
wherever notebook open is routed. A command-palette action needs an entry in
`crates/zed_actions` (or the repl actions module) + a workspace handler.

## Implemented

- [x] An empty/whitespace `.ipynb` now opens with a one-cell template.
      `parse_notebook_text`'s empty branch returns `empty_notebook()` (a minimal
      nbformat v4 notebook with one empty code cell) instead of a zero-cell
      notebook, so an empty file opens as a usable notebook, not a blank pane.
- [x] "New Jupyter Notebook" command (`notebook::NewNotebook`, registered as a
      workspace action). Creates a unique `Untitled-N.ipynb` in the first
      visible worktree, seeds it with the template, and opens it. Gated on the
      notebook feature flag; toasts if no folder is open.
- [x] Template round-trips: unit test `test_empty_notebook_template_round_trips`
      asserts an empty file yields one code cell and the serialized template
      re-parses.

## Deviation from the original plan (candidate backlog)

- The command creates a REAL file (`Untitled-N.ipynb`) on disk rather than a
  truly untitled/unsaved buffer. Untitled notebooks would need project-item
  routing for buffers without a path (bigger plumbing). Creating a file in the
  worktree reuses the whole existing open/save path and still removes the
  "duplicate an existing notebook" pain. Truly-untitled is a possible follow-up.

## Manual test checklist (for the user)

- [ ] Create an empty file named `something.ipynb` (file browser → New File);
      opening it shows a one-cell notebook, not an error/blank pane.
- [ ] Run "New Jupyter Notebook" from the command palette → a new
      `Untitled.ipynb` opens as a notebook; run it again → `Untitled-1.ipynb`.
- [ ] With no folder open, "New Jupyter Notebook" shows a toast instead of
      failing silently.

## Verification (automated)

- `cargo check -p repl` + `./script/clippy -p repl` clean.
- `cargo test -p repl notebook`: 2 passed (incl. the new round-trip test).
