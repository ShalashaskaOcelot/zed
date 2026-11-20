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

Each of these needs a DECISION at resolution time, not a default. For each:
compare the two implementations, keep the better one, and record which was kept
and why. Two specific notes:

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

## Tasks

- [ ] Branch `upstream-merge-2026-11` off the current work branch. Do NOT merge
      into `dev` until it builds and the smoke test passes.
- [ ] `git merge upstream/main`. **Merge, never rebase** — the history is
      published and the phasing system references SHAs.
- [ ] Resolve the mechanical conflicts first: `Cargo.lock` (take upstream's
      wholesale, then `cargo check` regenerates), the two keymaps (re-apply the
      fork's additive notebook blocks on top of upstream's file).
- [ ] Resolve `.rules`: upstream DELETED it in `a57e2fdc38` ("Unifying agent
      instructions to AGENTS.md") and now ships `AGENTS.md` + `.agents/`. The
      fork's phasing rules live in `.rules`. Decide where they go — likely fold
      into the fork's `CLAUDE.md` or a fork-owned `AGENTS.md` section — and keep
      the rules themselves intact whatever the filename.
- [ ] Resolve the generic-code conflicts: `gpui/src/elements/list.rs`,
      `util/src/process.rs`, `workspace/{pane,workspace}.rs`,
      `editor/src/{editor.rs,element/mouse.rs}`, `ui/src/components/scrollbar.rs`,
      `zed_actions/src/lib.rs`. These are the fork's additive blocks; re-apply
      them onto upstream's version rather than keeping ours wholesale.
- [ ] Resolve the `crates/repl` conflicts, taking a DECISION on each of the four
      overlapping features above and recording it in this file.
- [ ] Migrate the fork's own code to the new Jupyter APIs (nbformat 3.0,
      `jupyter-zmq-client`). Use upstream's `b1962f7cd8` as the worked example
      for every pattern.
- [ ] `rust-toolchain.toml` arrives as 1.98.1 — confirm the build works on it and
      that `./script/clippy` is clean under the newer lint set (three minor
      versions of new lints will likely fire).
- [ ] `cargo tree -d | grep async-tungstenite` returns nothing — i.e. ONE copy in
      the graph (see the dependency note in `backlog.md`).
- [ ] Full `cargo test -p repl -p gpui -p workspace` passes.
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
