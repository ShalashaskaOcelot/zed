# Phase 70 — Notebook keyboard and control corrections

Kind: **change to existing behaviour** — both items change something that works
today into something else, so if the old behaviour survives, the item stays open
and gets fixed in place rather than refiled.

Promoted from the backlog 2026-08-12 to restore the runway after phase 68
completed. Both were raised by the user on 2026-08-11/12, both are small, and
both are about a notebook overriding something the rest of the editor owns.

Primary files: `assets/keymaps/default-linux.json`,
`assets/keymaps/default-windows.json`, `crates/repl/src/notebook/notebook_ui.rs`.

## Item 1 — `ctrl-n` / `ctrl-p` should belong to the workspace, not the notebook

Reported 2026-08-11: in a notebook, `ctrl-n` moves down a cell instead of making
a new file. `ctrl-p` has the same fault (previous cell instead of the file
finder) for the same reason.

**Explicitly NOT a bug** (settled with the user 2026-08-11): every piece is doing
what it was written to do, and the behaviour is emergent rather than broken.
Upstream binds `ctrl-n` → `menu::SelectNext` **context-free** at the top of each
keymap, for menus and pickers. Upstream's own notebook keymap binds `down` to
that same action for cell navigation (confirmed present in `origin/main` — not a
fork addition). The notebook registers a handler for it. gpui resolves the
deepest matching context first, and a context-free binding matches at EVERY
depth, so inside a notebook it beats `Workspace`'s
`ctrl-n → workspace::NewFile`. Nothing malfunctions; the outcome is a
consequence of reusing a shared action name. That is why this is a phased
change and not a bug fix.

**macOS is deliberately excluded.** There `ctrl-n`/`ctrl-p` are standard
emacs-style line navigation and `cmd-n` is New File, so there is no conflict —
nulling them would remove behaviour a macOS user expects.

## Item 2 — Run All should leave edit mode

Reported 2026-08-11. Phase 64 made the sidebar controls focus the notebook, but
deliberately did NOT pull you out of a cell you were editing ("clicking a
control WHILE editing a cell should NOT throw you out of the cell"). The user
now wants Run All specifically to exit edit mode as well: running the whole
notebook is not an editing action, and staying in the cell afterwards leaves the
keyboard talking to a cell while the whole notebook executes.

**Scope this narrowly.** Run All only. The no-yanking behaviour was a deliberate
choice for the other controls and stays as it is unless the user says otherwise.

## Tasks

- [ ] Add `"ctrl-n": null` and `"ctrl-p": null` to the base `NotebookEditor`
      context in the Linux and Windows keymaps, with a comment saying why (the
      context-free menu bindings, not an arbitrary preference). Leave
      `default-macos.json` untouched.
- [ ] Check nothing else in the notebook relied on those two keys, and that
      arrow-key cell navigation is unaffected.
- [ ] Look for OTHER context-free menu bindings shadowed the same way inside a
      notebook (the top-of-keymap block binds several), and list what you find
      in this file — fix only the ones that shadow a workspace command the user
      would miss, and say which you left alone.
- [ ] Make the Run All control leave edit mode as well as focusing the notebook,
      without changing the other controls.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] In a notebook, `ctrl-n` makes a new file and `ctrl-p` opens the file
      finder, as they do everywhere else. Arrow keys still move between cells.
- [ ] Neither key does anything odd while EDITING a cell (they should behave as
      they do in any editor).
- [ ] Start editing a cell, then click Run All: the notebook leaves edit mode,
      the run starts, and a keyboard shortcut afterwards acts on the notebook
      rather than the cell.
- [ ] The other sidebar controls still do NOT throw you out of a cell you are
      editing.
