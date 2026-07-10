# Phase 7 — Cell clipboard operations (copy / cut / paste / duplicate)

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING of the rebind. The clipboard
> logic itself is CONFIRMED working — the user verified copy/paste worked with
> the old single-key `c`/`v` bindings ("Copy and paste using c and v did work").
> The only open item is the keybinding change below.
> Kind: **new feature** (clipboard, confirmed working) + **change** (rebind to
> standard Ctrl-C / Ctrl-X / Ctrl-V — awaiting confirmation).

Primary files: `crates/zed_actions/src/lib.rs`,
`crates/repl/src/notebook/notebook_ui.rs`, `assets/keymaps/*`.

## Implemented

- [x] Actions `CopyCell`, `CutCell`, `PasteCell`, `DuplicateCell`.
- [x] Clipboard format: cell serialized to nbformat JSON via `ClipboardItem`
      (works within a notebook and across notebooks/windows). Paste parses an
      `nbformat::v4::Cell`; non-cell text is ignored.
- [x] Copy → clipboard; Cut → copy + delete (respects the last-cell guard);
      Paste → insert below with a FRESH id; Duplicate → copy of selected below.
- [x] **Confirmed working** via the old `c`/`v`/`x` command-mode keys.

## Rebind to standard combos (awaiting confirmation)

The single-key `c`/`x`/`v` bindings were unintuitive (and `x` clashed
conceptually with delete). Rebound to the OS-standard clipboard combos, in the
command-mode context only (edit mode keeps text copy/cut/paste):

- [x] `ctrl-c` / `cmd-c` → CopyCell (was `c`).
- [x] `ctrl-x` / `cmd-x` → CutCell (was `x`).
- [x] `ctrl-v` / `cmd-v` → PasteCell (was `v`).
- [x] `d d` still deletes (unchanged).
- [x] Freed the base-context `ctrl-c`/`cmd-c` (was InterruptKernel — it was
      shadowed everywhere anyway); interrupt moved to Jupyter-standard `i i` in
      command mode, and remains on the toolbar Stop button.

## Deferred (backlog)

- Paste ABOVE (`shift-v`) — only paste-below is wired.

## Manual test checklist (for the user)

- [ ] In command mode: `ctrl-c`/`cmd-c` copies, `ctrl-v`/`cmd-v` pastes below.
- [ ] `ctrl-x`/`cmd-x` cuts (cell removed + on clipboard); paste elsewhere.
- [ ] In edit mode, `ctrl-c`/`ctrl-x`/`ctrl-v` still operate on TEXT, not cells.
- [ ] `d d` still deletes (does not cut).
- [ ] `i i` interrupts a running cell (replaces the old `ctrl-c` interrupt).
- [ ] Paste into a different notebook / window.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean, tests pass.
