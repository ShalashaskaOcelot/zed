# Phase 12 — Create & open new notebooks

Kind: **new feature**. Not yet started — this is a plan.

Goal: VS Code parity for getting a notebook to exist. Today you must duplicate
an existing `.ipynb` and empty it; an empty/new `.ipynb` fails to open because
it is not valid nbformat JSON.

Primary files: `crates/repl/src/notebook/notebook_ui.rs` (open/parse path,
`parse_notebook_text`), the project item registration for `.ipynb`, and
wherever notebook open is routed. A command-palette action needs an entry in
`crates/zed_actions` (or the repl actions module) + a workspace handler.

## Tasks

- [ ] When a `.ipynb` opened from the file browser is empty (or whitespace),
      populate it with a minimal valid nbformat v4 template (one empty code
      cell) so it opens as a notebook instead of erroring on empty/invalid JSON.
      Decide whether to write the template to disk on open or hold it in the
      buffer until first save (prefer: fill the buffer, mark dirty, so an
      untouched file isn't rewritten).
- [ ] Add a "New Jupyter Notebook" command-palette action that opens an
      untitled, unsaved notebook (minimal nbformat v4 template) in the editor,
      routed like other "new file" actions.
- [ ] Ensure the template round-trips: it saves as valid nbformat and reopens.

## Risks / gaps

- Empty-file detection must not clobber a file that is mid-write by another
  process; only treat truly empty/whitespace content as "new".
- Untitled notebooks need a language/kernel selection flow (reuse the existing
  lazy-start + kernel picker; no kernel until first run).

## Verification

- `cargo check -p repl` + `./script/clippy -p repl` clean; add a unit test that
  `parse_notebook_text` accepts the generated template.
- User test: create an empty `.ipynb` and open it; run "New Jupyter Notebook".
