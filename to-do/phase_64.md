# Phase 64 — Kernel picker and notebook control polish

Kind: **mixed** — mostly changes to existing behaviour, one small new
affordance (the discovery state). Promoted from the backlog 2026-08-06 to
replace the parked release-engineering phases. Five small, independent items
that all sit on the kernel picker / notebook control surface.

## Items and key facts

1. **"Searching for kernels…" instead of a bare "No matches"** (came out of
   confirming bug #20). Kernelspec + python-toolchain discovery is async, so a
   picker opened right after launch is legitimately empty for a moment and
   currently shows the generic empty-match text — which reads as "you have no
   Python interpreters". The picker already OBSERVES `ReplStore` and rebuilds
   its entries as discovery lands (bug #20's fix), so this is purely about what
   is displayed while empty. Needs an in-flight flag on the store's refreshes
   (`ReplStore::refresh_kernelspecs`, `refresh_python_kernelspecs` —
   `repl_store.rs:165,261`, both return `Task`s) that the delegate can read.
   Once discovery settles, "No matches" becomes trustworthy.
2. **Show the env path under Jupyter-kernel entries** the way Python
   Environment entries already show theirs (`build_grouped_entries`,
   `components/kernel_options.rs:49`). A registered venv kernelspec currently
   gives no clue which directory it points at, so two same-named kernels are
   indistinguishable. The path is in the kernelspec's `argv[0]`/its
   `kernel.json` location — pick whichever is honest for each variant and say
   which in the code comment.
3. **"Creating <name>…" row polish** (`KernelPickerEntry::Creating`,
   `kernel_options.rs:429-449`): the row works but looks unfinished. Visual
   only. NOT in scope: the row failing to refresh when the build completes —
   that is bug #58 and stays there.
4. **Remove the redundant kernel selector from the right control sidebar.**
   There are two picker triggers (top-right strip and sidebar bottom) sharing
   one `PopoverMenuHandle`, so clicking the SIDEBAR one opens the popover
   anchored at the TOP-RIGHT trigger — visibly odd and proof of the redundancy.
   Drop the sidebar trigger; the strip already shows kernel name + status (and
   after phase 62, the animated status). This is a subset of the larger
   "top control bar vs right sidebar" backlog item but stands alone.
5. **Notebook control buttons should focus the notebook.** Bug #51 made the
   sidebar buttons act on their own notebook regardless of focus, but focus
   itself stays where it was (e.g. the project panel), so keyboard shortcuts
   still don't reach the notebook afterwards. Clicking Run All / Restart /
   Interrupt / etc. should also focus that notebook, exactly as clicking a cell
   or a per-cell run button does — focus the notebook's focus handle in the
   control-button listeners (`render_notebook_controls`,
   `notebook_ui.rs:4259`).

## Tasks

- [ ] Add an in-flight/`is_discovering` signal to `ReplStore`'s kernelspec and
      python-toolchain refreshes, and render a "Searching for kernels…" state
      in the picker while it is set (empty + discovering ⇒ searching; empty +
      settled ⇒ the existing no-matches text).
- [ ] Show the environment/kernel path as the secondary line on Jupyter
      kernelspec entries, matching how Python Environment entries do it.
- [ ] Tidy the "Creating <name>…" row's presentation (spacing, spinner/label
      alignment, muted styling consistent with the other entries).
- [ ] Remove the kernel-selector trigger from the right control sidebar,
      leaving the top-right strip as the single picker entry point.
- [ ] Focus the notebook from every control-button listener in
      `render_notebook_controls`.
- [ ] `./script/clippy` clean and `cargo test -p repl` passes.

## User tests (runtime)

- [ ] Launch the app and open the kernel picker immediately: it says it is
      searching, then fills in. With a machine that genuinely has no
      interpreter, it settles on "No matches".
- [ ] Jupyter kernelspec entries show their path; two similarly-named kernels
      are now distinguishable.
- [ ] The sidebar no longer has a kernel selector; the top-right one still
      opens the picker normally.
- [ ] Open a notebook from the project panel WITHOUT clicking into it, press
      Run All in the sidebar, then use a keyboard shortcut (e.g. `escape`,
      arrow keys, `shift-enter`) — it acts on the notebook, because the button
      moved focus there.
- [ ] Creating an env still works and the row looks tidier (its live-refresh
      defect remains bug #58).
