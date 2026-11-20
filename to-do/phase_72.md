# Phase 72 — Merge upstream into the fork

Kind: **mixed** — the merge itself is a change to existing behaviour (the fork
must still build and the notebook must still work), and adopting upstream's new
notebook features is effectively new behaviour arriving in the fork.

Promoted from `backlog.md`'s PARKED section 2026-11-20 at the user's explicit
request ("I want to merge upstream changes"), which is the only thing that
unparks an item. Supersedes the old phase-46 summary; the 2026-07-16 research in
it is folded in below, with everything re-verified against today's upstream.

## Measured state (verified 2026-11-20, not from memory)

| | |
|---|---|
| Fork point | `950ec7943f`, 2026-07-08 |
| Upstream tip | `10d1b2a564`, 2026-11-20 |
| Upstream commits to absorb | **2057** |
| Fork commits since the fork point | 208 |
| Conflicted files (dry run) | **19** |
| Toolchain | ours 1.95.0 → upstream **1.98.1** |

`git merge-tree` dry run names every conflict:

```
.rules (modify/delete — deleted upstream)   crates/repl/src/notebook/cell.rs
Cargo.lock                                  crates/repl/src/notebook/notebook_ui.rs
assets/keymaps/default-linux.json           crates/repl/src/outputs/plain.rs
assets/keymaps/default-macos.json           crates/repl/src/outputs/table.rs
crates/editor/src/editor.rs                 crates/repl/src/session.rs
crates/editor/src/element/mouse.rs          crates/ui/src/components/scrollbar.rs
crates/gpui/src/elements/list.rs            crates/util/src/process.rs
crates/repl/src/components/kernel_options.rs crates/workspace/src/pane.rs
crates/repl/src/kernels/ssh_kernel.rs       crates/workspace/src/workspace.rs
                                            crates/zed_actions/src/lib.rs
```

Note `default-windows.json` and `assets/settings/default.json` auto-merge
cleanly — the fork's additions there are additive blocks, as predicted.

## The thing that changes the fork's position: upstream now develops notebooks

Phase 46's research recorded `crates/repl` as the tail risk — "upstream has not
touched it since the fork point, but a real upstream notebook push would conflict
massively." That risk has now landed: **18 upstream commits touch `crates/repl`**,
and four of them build things this fork already built independently.

| Upstream commit | What it does | Fork's own version |
|---|---|---|
| `55ea5baf15` | Open notebooks from single-file worktrees | phase 69 + bug #79 (the invisible-worktree fix) |
| `b535bec7bf` | Add a delete-cell action | the fork's `DeleteCell` — and **bug #81** is open against it |
| `d1548bd013` | Add-cell controls in empty notebooks | the fork's add-cell affordances |
| `bc538def45` | Save notebooks through the project, not the local fs | the fork's save path |
| `9f03ff88e4` | Fix language injections in notebook cells | — (pure gain) |

**Resolution policy (user, 2026-11-20): LEAN UPSTREAM.** Compare the two
implementations for each. Where they are equivalent, or upstream's is better,
take UPSTREAM's — even when the difference is cosmetic, because every line of
fork-specific code is carried forever and widens the next merge. Keep the fork's
version ONLY where it is genuinely better, and say what makes it better. Record
the decision and the reason for each one in this file. Two specific notes:

- **Bug #81 (`d d` does nothing)** may be resolved or reshaped by upstream's own
  delete-cell action. Re-test it after the merge BEFORE investigating further —
  the open diagnostic (does `i i` work?) may be moot.
- The fork's phase-69 loose-notebook work was hard-won (an invisible worktree is
  held only weakly; see bug #79). Compare carefully: if upstream's approach is
  cleaner, taking it removes fork-specific code we would otherwise carry forever.

## The dependency jump is the hard part, and it is NOT in the conflict list

`b1962f7cd8` (upstream, 2026-11-10) replaced the whole Jupyter stack:

| | before | after |
|---|---|---|
| `jupyter-protocol` | 1.4.0 | 2.0.3 |
| `jupyter-websocket-client` | 1.1.0 | 3.0.1 |
| `nbformat` | 1.2.0 | **3.0.0** |
| `runtimelib` | 1.4.0 | **replaced by `jupyter-zmq-client` 3.0.0** |
| `async-tungstenite` | 0.31 | 0.33 |

Upstream's own migration was ~136/-107 lines across 12 repl files. The fork has
**far more** code against those APIs than upstream does — `nbformat::` appears 53
times in `notebook_ui.rs` and 20 in `cell.rs`, and `runtimelib::` is spread
across every `kernels/*.rs` plus `outputs/table.rs`. None of that surfaces as a
merge conflict; it surfaces as a wall of compile errors AFTER the merge resolves.
**Budget for this as the single largest piece of the phase.**

It also kills `async-std` (RUSTSEC-2025-0052, unmaintained) from the tree.

## Decisions taken at conflict resolution (2026-11-20)

All 19 conflicts resolved. Policy was LEAN UPSTREAM; these are the calls made
and why, including every place the fork's version was kept.

**Took upstream:**
- `Cargo.lock` wholesale; the three keymaps gained upstream's new
  `"backspace": "notebook::DeleteCell"` binding on top of the fork's block.
- `editor/element/mouse.rs` delta filtering — upstream's
  `scroll_manager.filter_scroll_delta(delta, touch_phase)` replaces the fork's
  `ongoing_scroll.filter(delta)`. Upstream moved `OngoingScroll` into gpui and
  it now locks the delta to the dominant axis IN PLACE, so the fork's axis
  plumbing (`scroll_smoothly(.., axis, ..)`, `update_ongoing_scroll`) became
  redundant and was dropped. The fork's smooth glide itself is KEPT: upstream
  has no smooth scrolling at all, so taking theirs wholesale would have deleted
  a feature rather than swapped an equivalent one.
- `gpui/elements/list.rs`: upstream replaced `logical_scroll_top` +
  `pending_scroll` with a single `scroll_anchor: Option<ScrollAnchor>` whose
  `set_scroll_anchor` does the pending-rebasing the fork did by hand. Ported the
  fork's calls onto it and deleted the fork's `rebase_pending_scroll`.
- `notebook/cell.rs`, `session.rs`, `ssh_kernel.rs`, `outputs/table.rs`:
  `runtimelib` → `jupyter_protocol` imports.
- `notebook_ui.rs`: upstream's cell-constructor signature and its new
  `KeyContext::new_with_defaults(cx.platform_info().os)` API.

**Kept the fork's (with the reason):**
- `workspace/pane.rs` save-as: the fork creates an INVISIBLE single-file
  worktree; upstream passes `visible=true`, which would add a new root to the
  project panel for every out-of-project save-as. The already-merged code below
  the hunk also depends on the fork's `created_worktree` hold.
- `notebook_ui.rs` `delete_cell`: the fork's handles multi-cell selections, undo
  as one cell-op, and never leaves an empty notebook; upstream's deletes one
  cell. **Caveat: bug #81 says the fork's does not fire on `d d`.** Re-test it
  first — if it is still dead, upstream's simpler version is the fallback.
- `notebook_ui.rs` interrupt: the fork sends a real OS interrupt (SIGINT /
  Windows CTRL_C, see `util/process.rs`) and cancels the run queue; upstream
  sends a Jupyter `InterruptRequest` message, which only works for kernels
  declaring `interrupt_mode: message`.
- `GUTTER_WIDTH` 38.0, not upstream's 19.0 — bug #78 (three-digit execution
  counts wrapped).
- The whole `NotebookItem` path model (`path: Option<PathBuf>`,
  `project_path: Option<ProjectPath>`). Upstream's is non-optional, which has no
  room for an untitled notebook — a fork feature (phase 33).

**Deliberately DEFERRED, not rejected** (each is a genuine upstream improvement
that the fork's different data model blocks; they need their own phase):
- Upstream's `save_impl` — saving through the PROJECT rather than the local
  filesystem (`bc538def45`). Matters for remote projects. Blocked on the
  optional-path model above; adopting it means reworking untitled notebooks.
- Upstream's `render_empty_state` / add-cell controls for empty notebooks
  (`d1548bd013`).
- Upstream's two new tests, `test_open_single_file_notebook` and
  `test_save_goes_through_the_project`, which sit in a conflicted region with
  the fork's test suite.

## Tasks

- [x] Branch `upstream-merge-2026-11` off the current work branch. Do NOT merge
      into `dev` until it builds and the smoke test passes.
- [x] `git merge upstream/main`. **Merge, never rebase** — the history is
      published and the phasing system references SHAs.
- [x] Resolve the mechanical conflicts first: `Cargo.lock` (take upstream's
      wholesale, then `cargo check` regenerates), the two keymaps (re-apply the
      fork's additive notebook blocks on top of upstream's file).
- [x] Resolve `.rules` — moved to `CLAUDE.md`, a name upstream does not use, so it can never conflict again: upstream DELETED it in `a57e2fdc38` ("Unifying agent
      instructions to AGENTS.md") and now ships `AGENTS.md` + `.agents/`. The
      fork's phasing rules live in `.rules`. Decide where they go — likely fold
      into the fork's `CLAUDE.md` or a fork-owned `AGENTS.md` section — and keep
      the rules themselves intact whatever the filename.
- [x] Resolve the generic-code conflicts: `gpui/src/elements/list.rs`,
      `util/src/process.rs`, `workspace/{pane,workspace}.rs`,
      `editor/src/{editor.rs,element/mouse.rs}`, `ui/src/components/scrollbar.rs`,
      `zed_actions/src/lib.rs`. These are the fork's additive blocks; re-apply
      them onto upstream's version rather than keeping ours wholesale.
- [x] Resolve the `crates/repl` conflicts, taking a DECISION on each of the four
      overlapping features above and recording it in this file.
- [x] Migrate the fork's own code to the new Jupyter APIs (nbformat 3.0,
      `jupyter-zmq-client`). Use upstream's `b1962f7cd8` as the worked example
      for every pattern.
- [ ] `rust-toolchain.toml` arrives as 1.98.1 — confirm the build works on it and
      that `./script/clippy` is clean under the newer lint set (three minor
      versions of new lints will likely fire).
- [ ] `cargo tree -d | grep async-tungstenite` returns nothing — i.e. ONE copy in
      the graph (see the dependency note in `backlog.md`).
- [x] `cargo check` clean for gpui, workspace, editor and repl.
- [x] `cargo test -p repl --lib`: **65 passed, 0 failed** (2026-11-20).
- [ ] `cargo test -p gpui -p workspace`.
- [ ] `./script/clippy` clean. NOTE: the cloud container cannot build the FULL
      workspace — `alsa-sys` needs libasound2-dev, which is absent and cannot be
      installed there. That is environmental, not a merge problem (alsa-sys was
      already a dependency before the merge), but it means clippy and the
      `zed` binary have to be built on the user's machine.
- [ ] Toolchain moved to 1.98.1; expect new lints from three Rust releases.
- [ ] Re-test bug #81 and re-check every item in `awaiting_testing.md` that
      touches code upstream changed — the merge may have fixed or broken some.

## User tests (runtime) — the merge is NOT done until these pass

A textually clean merge can still leave the notebook broken, so this is
mandatory, not optional:

- [ ] Zed launches and a notebook opens.
- [ ] Select a kernel, run a cell, see output. Restart the kernel. Interrupt a run.
- [ ] Run All over a multi-cell notebook with outputs; follow mode behaves.
- [ ] A DataFrame renders in a table output (the Jupyter-dep migration touches
      `outputs/table.rs` directly).
- [ ] Save, close, reopen a notebook — including an unsaved one and one outside
      any project root (the two restore paths from phases 66/69).
- [ ] The fork's own notebook features still work: cell add/delete/move/convert,
      command vs edit mode, the kernel strip, the timers.
