# Phase 4 — Cell actions, Jupyter shortcuts, and the two dead menus

Goal: bring the per-cell action set up to Jupyter/VS Code conventions and
give the two dead ellipsis buttons real menus (the menus need the actions to
exist first, hence one phase).

Primary files: `crates/zed_actions/src/lib.rs` (action definitions),
`crates/repl/src/notebook/notebook_ui.rs`,
`crates/repl/src/notebook/cell.rs`,
`assets/keymaps/default-{linux,macos,windows}.json`.

## Tasks

### New actions + handlers

- [ ] `DeleteCell` — remove selected cell (guard: never delete the last
      remaining cell; select a sensible neighbour after). **Top priority in
      this phase** (user, 2026-07-08): confirmed there is currently NO way to
      delete a cell anywhere — no action exists, so it isn't in the command
      palette either; once added a cell can only be removed by editing the
      raw .ipynb externally.
- [ ] `AddCellAbove` / `AddCellBelow` — `insert_cell_at_current_position`
      currently hardcodes insert-below (`notebook_ui.rs:763`); parameterise.
- [ ] `RunCellsAbove` — run all cells above the selected cell.
- [ ] `RunCellAndBelow` — run selected cell and all below.
- [ ] `ConvertCellToCode` / `ConvertCellToMarkdown` — change cell type in
      place, preserving source text.

### Nav-mode keybindings (context `NotebookEditor && notebook_mode == command`)

- [ ] `a` → add cell above, `b` → add cell below (code cells; markdown via
      conversion)
- [ ] `x` and `d d` (double-tap) → delete cell
- [ ] `m` → convert to markdown, `y` → convert to code
- [ ] Audit remaining VS Code/Jupyter nav-mode shortcuts and bind any that
      map cleanly to existing actions (e.g. `z` undo-cell-delete is backlog —
      needs cell undo)

### Menus

- [ ] Wire the right-toolbar "More options" button
      (`notebook_ui.rs:1124-1127`) to a `PopoverMenu` (pattern: the kernel
      picker's `PopoverMenuHandle`). Contents: run cells above / run cell
      and below, delete cell, add cell above/below, convert cell type,
      clear all outputs.
- [ ] Wire the output "..." button (`cell.rs:924-939`) to a per-output menu:
      copy output (reuse the working copy logic in `outputs.rs:200-221`),
      clear this cell's output. Extras (save output, scrollable output
      toggle) go to backlog if non-trivial.

### Cell hover controls

- [ ] Show run-above / run-below / delete / add-below affordances on the
      selected cell (VS Code-style cell toolbar), reusing the actions above.
      Skip split/group (user doesn't use them — backlog).

## Notes

- Existing notebook actions (add code/markdown cell, run, move cell, restart,
  etc.) already show up in the command palette because they are registered
  `actions!` handled via `.on_action` — any NEW action added in this phase
  gets command-palette presence for free. There is no hidden delete-cell
  action to surface; it genuinely does not exist yet.
- Keybindings are attached to actions, not buttons: adding a keybind for
  something that today is button-only (e.g. clear-all-outputs, which has no
  binding) is just a keymap JSON entry dispatching the same action the button
  dispatches — no new code needed. Note add code/markdown cell DO already
  have bindings (`ctrl-m` / `ctrl-shift-m`; `cmd-` on macOS).
- Cell mutations must mark the notebook dirty once bug #9 is fixed — if
  bug #9 isn't fixed yet, fixing it alongside `DeleteCell` is in scope here
  (it's a bug, so allowed regardless of phasing).
- Every new action must appear in all three keymap files, in both the
  `NotebookEditor` and (where sensible) `NotebookEditor > Editor` contexts,
  matching the existing duplication pattern.
