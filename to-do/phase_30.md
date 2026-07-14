# Phase 30 — Notebook chrome rework (drop the bottom kernel bar)

Kind: **change to existing behaviour** (the layout change is the deliverable —
keep items OPEN until the user confirms the new layout works for them). Not yet
started — this is a plan. Promoted from the backlog (user 2026-07-14) to keep 5
phases in rotation after phase 26 completed.

The bottom bar's kernel selector duplicates the sidebar's. Remove the bar, keep
every function reachable, and give the running-kernel status a better home.

Primary files: `crates/repl/src/notebook/notebook_ui.rs`
(`render_kernel_status_bar`, `render_notebook_controls`, the outer `render`).

## Tasks

- [ ] Remove the bottom kernel status bar from the notebook layout.
- [ ] Move the Restart Kernel and Interrupt Kernel buttons into the right
      sidebar (`render_notebook_controls`) so they aren't lost with the bar.
- [ ] Relocate the running-kernel name + status (and the kernel-selector
      trigger with them) to a slim TOP-RIGHT strip: a small top margin so the
      cells start slightly lower, with the kernel cluster right-aligned in
      that strip — the user's preferred option (a light take on VS Code's
      notebook top bar, WITHOUT moving the sidebar's actions up).
      - Alternatives kept in reserve if (a) reads badly at runtime:
        (b) kernel name/status right-aligned in the pane toolbar / tab-bar
        area (zero notebook space cost, further from the cells);
        (c) a minimal floating bottom-right status chip (no full bar).
- [ ] Make sure everything the bottom bar offered survives: kernel picker
      (sidebar + new top-right trigger), kernel status indicator, restart,
      interrupt. Nothing reachable only via the removed bar.

## Risks / gaps

- The kernel picker's `PopoverMenuHandle` and the `cells_awaiting_kernel_choice`
  flow (bug #28 fixes) hang off the current bar's selector — rewire the handle
  to the new trigger(s) without breaking the run-prompt flow.
- The top strip must not steal vertical space when a notebook has no kernel
  attached (decide: always show vs. only when a kernel is selected/running).
- Bug #25's reveal math assumed the bar's height; removing it changes the
  viewport height — retest add-at-bottom after this lands.

## Verification

- `cargo check -p repl` + clippy clean; tests pass.
- User test: bottom bar gone; restart/interrupt available in the sidebar;
  kernel name/status visible top-right; picker opens from the new trigger and
  the run-prompt flow (run with no kernel) still works end-to-end.
