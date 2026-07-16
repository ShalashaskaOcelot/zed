# Phase 46 — Upstream-merge workflow (routine zed-industries → fork syncs)

Kind: **new feature** (process + docs; planned by phase 43, 2026-07-16).
Goal: merging upstream `zed-industries/zed` into this fork becomes a routine,
documented, low-drama operation with a written playbook — instead of an
ad-hoc event. Merge, not rebase: the fork's history is published and the
phasing system references commit SHAs.

Key discovery facts (details in phase 43's findings, commit `8f9a5d3dc5`):
- `origin/main` is already an upstream MIRROR kept current via GitHub's
  fork-sync button; `origin/dev` is the fork mainline. A dry-run merge of
  164 upstream commits was CLEAN on 2026-07-16 — weekly merges are cheap
  today; the cost compounds with delay.
- Upstream velocity ~150 commits/week. Hotspots ranked: `Cargo.lock`
  (constant churn, trivial resolution), `crates/settings_ui/src/page_data.rs`
  (fork +177 lines in a file upstream actively develops),
  `assets/settings/default.json`, three keymap files,
  `crates/settings_content/src/settings_content.rs`,
  `crates/zed_actions/src/lib.rs`, `crates/gpui/src/elements/list.rs`
  (semantic-breakage risk even on clean merges), `crates/util/src/process.rs`.
  `crates/repl/**`: upstream rarely touches it (zero commits since the fork
  point) but a real upstream notebook push would conflict massively — the
  tail risk, not the routine one.
- The working clones are SHALLOW; merge-base math needs full history.

## Tasks

- [ ] Write `docs/fork/upstream-merge.md` — the playbook:
      1. One-time setup: `git fetch --unshallow origin`, add the upstream
         remote (`git remote add upstream https://github.com/zed-industries/zed.git`),
         or rely on GitHub fork-sync updating `origin/main` (document both;
         fork-sync is the current practice and needs no extra remote).
      2. Cadence: WEEKLY, plus an immediate sync before starting any phase
         that touches shared files (settings, keymaps, gpui).
      3. Procedure: sync `origin/main` → `git merge origin/main` into `dev`
         (never rebase) → resolve → verify → push. Include the dry-run
         preview command (`git merge-tree --write-tree HEAD origin/main`)
         to size conflicts before committing to the merge.
      4. Per-file resolution rules: `Cargo.lock` → take upstream wholesale,
         then `cargo check` regenerates fork lines (chrono);
         keymaps/`default.json` → fork's notebook blocks are additive,
         re-apply them on conflict; `page_data.rs`/`settings_content.rs` →
         merge by hand, fork sections are the "Notebook" settings block;
         `zed_actions/lib.rs` → fork's `notebook::` action block is
         additive; `.rules` → fork sections live at the bottom, keep both;
         `crates/repl/**` → fork wins, then re-apply upstream API
         migrations by hand; `crates/zed/Cargo.toml` version field → keep
         upstream's (phase 45's release scheme re-applies the fork patch
         offset at release time).
      5. Post-merge verification checklist (semantic breakage happens even
         on clean merges — gpui/list.rs especially): `cargo build -p repl`,
         `./script/clippy`, `cargo test -p repl -p gpui`, then a manual
         notebook smoke test (open, run cells, kernel picker, save).
      6. What to do when a merge goes badly: abort criteria
         (`git merge --abort`), and the escalation path (park it, file the
         breakage as a bug, merge a shorter range).
- [ ] Add `docs/fork/merge-checklist.md` — the short tick-list version of
      the playbook for actually running a merge (copy-paste commands, boxes
      to tick), referencing the playbook for the "why".
- [ ] Do ONE supervised merge following the playbook (upstream is ~1 day
      ahead of the mirror already) and fix anything the playbook got wrong
      — the playbook is only done when it has survived a real merge.

Manual user verification:

- [ ] ⚠ untested — User reviews the playbook (cadence + who runs it: agent
      sessions, the user, or both) and runs/observes the first routine
      merge; the post-merge notebook smoke test passes on their machine.

## Risks / gaps

- GitHub fork-sync of `origin/main` silently fast-forwards only when
  possible; if the user ever commits to `main` by accident the mirror
  breaks — the playbook must state `main` is READ-ONLY upstream mirror.
- A textually-clean merge can still break the notebook at runtime (gpui
  refactors, settings-system reshapes) — hence the mandatory smoke test.
- Upstreaming the two generic fork changes (gpui
  `scroll_to_reveal_item_top_aligned`, util::process `spawn_interruptible`)
  would permanently shrink the conflict surface — tracked in backlog, not
  this phase.

## Verification

- Playbook + checklist exist and reflect what the first real merge actually
  required; that merge is pushed with the full post-merge checklist green.
