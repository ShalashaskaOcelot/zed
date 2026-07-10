# Phase 9 — External file sync (watch & reload) (COMPLETE, archived 2026-07-10)

> ✅ STATUS: CONFIRMED by the user 2026-07-10 ("9 is good"). Kind: **new
> feature**. The external-change popup appears and reload works. The save-side
> conflict warning was never part of this phase's delivery; it is carried
> forward as bug #14 / **phase 14** (save-conflict guard + reload affordance).

Primary files: `crates/repl/src/notebook/notebook_ui.rs`.

## Implemented

- [x] `NotebookItem` retains the project `Buffer` so the file stays watched
      and emits `BufferEvent::Reloaded` on external change.
- [x] On `Reloaded`, rebuild from the new content unless there are unsaved
      changes (then keep them + show a conflict toast). Skips when disk content
      matches ours (own save). ✅ CONFIRMED by user.
- [x] `reload_cells_from_notebook` rebuilds AND wires cell subscriptions;
      `Item::reload` now uses it too (fixing a latent no-wiring bug).
- [x] Shared `parse_notebook_text` for open / reload / external-change.
- [x] Conflict toast reworded to be accurate (no misleading "Reload").

## Remaining (tracked elsewhere)

- Save-side conflict warning → **bug #14** (saving over external changes with
  no warning). Needs a save-time conflict check + confirm dialog.
- A convenient Reload button/command → backlog.

## Manual test checklist (for the user)

- [x] External edit with NO local changes → notebook updates. ✅ CONFIRMED.
- [ ] External edit WITH unsaved changes → toast, changes kept (does NOT warn
      on subsequent SAVE yet — bug #14).
- [ ] Saving from Zed does not cause a spurious reload/selection reset.
- [ ] After a reload, run/focus/typing still work in the rebuilt cells.

## Verification (automated)

- `cargo check -p repl` clean, `./script/clippy -p repl` clean, tests pass.
