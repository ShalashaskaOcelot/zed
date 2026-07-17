# Phase 49 — Show the kernel being created as a greyed entry in the picker

Kind: **new feature** (UI polish). Split from phase 48. Depends on phase 48
(the `creating_kernel_name` state).

Phase 48 makes a newly-created env the notebook's selected kernel immediately
(shown in the top strip, runs queued). This phase surfaces it in the kernel
PICKER dropdown too: while the env is building, it appears as a
greyed/disabled entry (like a no-ipykernel entry) and as the current
selection, so opening the picker mid-build shows what's coming rather than the
old selection.

Primary files: `crates/repl/src/components/kernel_options.rs` (entry building
+ render), `crates/repl/src/notebook/notebook_ui.rs` (pass the creating state
into the selector).

## Tasks

- [ ] Thread the notebook's `creating_kernel_name` (phase 48) into the kernel
      selector — e.g. a `with_creating(Option<String>)` builder on
      `KernelSelector`, analogous to `with_selected`.
- [ ] In `build_grouped_entries` / the delegate, add a disabled entry for the
      creating env (matching the greyed no-ipykernel styling) and mark it as
      the selected entry so the checkmark/selection reflects it.
- [ ] Ensure the entry is not selectable (can't be picked/launched while
      building) and disappears once the real kernel is selected (phase 48's
      success path clears `creating_kernel_name` and change_kernel selects the
      real spec, which then appears normally).

## Risks / gaps

- The picker is a `RenderOnce` snapshot rebuilt on store updates (bug #20
  area) — make sure the creating entry appears/updates without needing a
  store refresh, and doesn't linger after creation completes.
- Keep it purely additive to phase 48; if phase 48's state changes shape,
  update the thread-through here.

## Verification

- `cargo check -p repl` + `./script/clippy` clean; `cargo test -p repl`.
- User test: start creating an env, open the kernel picker during the build →
  the new env shows as a greyed, non-selectable entry and as the current
  selection; after the build it becomes a normal, selectable entry.
