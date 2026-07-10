# Phase 7 — Cell clipboard operations (copy / cut / paste / duplicate)

> ⚠️ STATUS: IMPLEMENTED, AWAITING USER TESTING (commit 1b6d566). Compiles,
> clippy-clean, unit tests pass. Do NOT archive until the user confirms.
> Kind: **new feature** (+ one **change**: `x` rebound from delete to cut).
>
> User note (2026-07-08): copy/paste reported "not working" — under
> investigation; likely the notebook getting stuck in Edit mode (see bug #13 /
> #15), where the `c`/`v` command-mode keys type into a cell instead of firing.

Primary files: `crates/zed_actions/src/lib.rs`,
`crates/repl/src/notebook/notebook_ui.rs`, `assets/keymaps/*`.

## Implemented

- [x] Actions `CopyCell`, `CutCell`, `PasteCell`, `DuplicateCell`.
- [x] Clipboard format: cell serialized to nbformat JSON via `ClipboardItem`
      (works within a notebook and across notebooks/windows). Paste parses an
      `nbformat::v4::Cell`; non-cell text is ignored.
- [x] Copy → clipboard; Cut → copy + delete (respects the last-cell guard);
      Paste → insert below with a FRESH id; Duplicate → copy of selected below.
- [x] Nav-mode keybinds (command mode, all keymaps): `c` copy, `x` cut
      (rebound from delete), `v` paste; `d d` still deletes. Also in the More
      options menu + command palette.

## Deferred (backlog)

- Paste ABOVE (`shift-v`) — only paste-below is wired.

## Manual test checklist (for the user)

- [ ] Copy a cell (`c`), paste below (`v`): content + type preserved, outputs
      not duplicated, new cell independent.
- [ ] Cut (`x`): cell removed and on clipboard; paste elsewhere.
- [ ] `d d` still deletes (does not cut).
- [ ] Duplicate (More options menu).
- [ ] Paste into a different notebook / window.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean, tests pass.
