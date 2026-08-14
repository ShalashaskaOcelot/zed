# Phase 71 — Line numbers per notebook cell

Kind: **new feature** — nothing shows line numbers in a notebook cell today.
Once the user confirms line numbers appear and both toggles work, archive it;
refinements (placement, styling, which gutter decorations show) become new
backlog or bug items rather than reopening this.

Promoted from the backlog 2026-08-12 to restore the runway after phase 70's
implementation completed. Requested by the user 2026-08-06; the binding and
implementation research below was done then and still holds.

Primary files: `crates/repl/src/notebook/cell.rs`,
`crates/repl/src/notebook/notebook_ui.rs`, `crates/repl/src/repl_settings.rs`,
`crates/settings_content/src/settings_content.rs`,
`crates/settings_ui/src/page_data.rs`, the three keymaps.

## What the user asked for

Two settings, BOTH defaulting to off — one for the whole notebook, one per cell
— plus two command-mode keybindings following Jupyter's own convention:
`l` toggles line numbers on the FOCUSED cell, `shift-l` toggles the
notebook-wide setting.

## Research already done (2026-08-06) — do not redo

**Binding check.** Both keys are free in
`NotebookEditor && notebook_mode == command`: the only `l` binding anywhere is
`menu::SelectNext` under the `Prompt` context, which cannot be active there, and
`shift-l` is unbound. This deliberately avoids `ctrl-l`, which is taken
(`editor::SelectLine` on Linux, `editor::ScrollCursorCenter` on macOS): bound in
the command-mode context it would have worked, but in the plain `NotebookEditor`
context the editor's binding would shadow it while editing a cell and it would
silently do nothing — the trap behind bug #64. **Bind both in the COMMAND-MODE
context only.**

**Gutter.** Cell editors call `editor.set_show_gutter(false, cx)`
(`cell.rs:529`), so line numbers need the gutter turned back ON — which also
brings breakpoints, code actions, runnables and git-diff markers with it. Those
need suppressing, or the gutter needs a line-numbers-only mode.

**The override already exists.** `Editor::show_line_numbers: Option<bool>` with
`line_numbers_enabled()` falling back to `EditorSettings::gutter.line_numbers`
(`editor/src/config.rs`) is exactly the shape needed for "cell overrides
notebook overrides global".

**Layout.** The notebook draws its OWN gutter left of each cell (the accent bar
+ run button, `GUTTER_WIDTH`), so two adjacent gutters need a look: line numbers
must not push cell content around or double the left margin.

## Note from phase 70 (2026-08-12)

Phase 70 established that a notebook must not answer a SHARED action for its own
behaviour. That does not bite here — `l`/`shift-l` map to new notebook-owned
actions — but the same rule applies: give the toggles their own actions, never
reuse something the menu layer also dispatches.

## Tasks

- [ ] Add the notebook-wide setting (`repl.notebook_show_line_numbers`, default
      false) through the full chain: `repl_settings.rs`, `settings_content.rs`,
      `assets/settings/default.json`, `settings_ui/page_data.rs`, and
      `docs/src/repl.md`.
- [ ] Add per-cell state (not a setting file entry — it is per cell, in memory,
      and should follow the cell through moves/splits where that is cheap).
      Resolution order: cell override → notebook setting → off.
- [ ] Turn the cell editor's gutter back on when line numbers are enabled, with
      breakpoints / code actions / runnables / git markers suppressed so only
      numbers show.
- [ ] Check the layout against the notebook's own gutter: cell content must not
      shift horizontally when numbers are toggled on, and the two gutters must
      not read as one double-width margin.
- [ ] Add `notebook::ToggleCellLineNumbers` and
      `notebook::ToggleNotebookLineNumbers`, bound to `l` and `shift-l` in the
      command-mode context of all three keymaps.
- [ ] Unit-test the resolution order (cell override beats notebook setting beats
      global default) and that a toggle flips only its own level.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] With both settings off (the default), cells look exactly as they do now.
- [ ] `shift-l` in command mode turns numbers on for the whole notebook; the
      setting also works from the settings UI and `settings.json`.
- [ ] `l` toggles the focused cell only, and overrides the notebook-wide setting
      in both directions (on when the notebook is off, and off when it is on).
- [ ] Numbers appear for both code and markdown cells while editing, and the
      cell's text does not jump sideways when they are toggled.
- [ ] No breakpoint dots, runnable arrows or git markers appear in the cell
      gutter alongside the numbers.
