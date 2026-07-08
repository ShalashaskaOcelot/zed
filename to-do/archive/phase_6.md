# Phase 6 — Kernel selection persistence and lazy start (COMPLETE, archived 2026-07-08)

Kind: change to existing behaviour. Confirmed by the user 2026-07-08: the
selected kernel is remembered and its name shows between closing and reopening
a file; running with no selection prompts the picker; the kernel no longer
auto-starts on open. The core changes took effect, so the phase is archived.

## Delivered (confirmed)

- `change_kernel` persists the selection via `ReplStore::set_active_kernelspec`
  → reopening the notebook in the same session reuses it and shows the name.
- Lazy start: no kernel launched on open; the kernel starts on first run
  (remembered) or the picker is prompted (no selection).
- `remembered_kernel_spec` resolves selection/metadata without falling back to
  the global/recommended kernel.

## Follow-ups spun off (NOT reopened here)

- Cross-session persistence does NOT survive fully quitting and relaunching
  Zed (the in-session `ReplStore` selection is lost; the on-disk metadata match
  didn't restore it). User: "not a big deal." → `backlog.md` (enhancement:
  persist the per-notebook kernel choice across Zed restarts, likely via the
  saved .ipynb metadata match or a persisted store).

## Related bug found during testing

- Dismissing the run-prompt picker (Esc) left the cell stuck "Running" →
  `bugs.md` #11 (fix attempted - untested).

## Verification

- Confirmed working at runtime by the user (2026-07-08).
- `cargo test -p repl`: 37 passed; clippy clean.
