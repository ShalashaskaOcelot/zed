# Backlog

Non-phased suggestions and to-do items that are NOT yet scheduled. Move an item
into a `phase_<n>.md` when it is scheduled (and delete it from here); never
implement directly from here. Completed and scheduled work is not tracked here —
see the phase files, `CHANGELOG.md`, and git history. Roughly ordered
high → low within each group.

## PARKED — do NOT auto-promote (user 2026-08-06)

Items in this section are excluded from the "keep 5 phases in rotation" rule.
They must NOT be promoted into a phase to replenish the runway — not even if
that leaves the count at 2. Promote one ONLY when the user asks for it by name.
Everything here is the fork's release-engineering track, which the user has
explicitly parked for a while.

The three entries below were full phase files (45, 46, 47), all with ZERO tasks
implemented, moved back here on 2026-08-06. The summaries keep what matters;
the complete task-by-task detail is in the deleted files — see commit `4174e8b`
(and the phase-43 discovery findings in commit `8f9a5d3dc5`) if one is revived.

- **Fork auto-update from the Gitea releases feed** (was phase 45; change to
  existing behaviour). Installed fork builds should update from the user's
  Gitea — never zed.dev — or have updates cleanly disabled with an
  explanation. Key facts: ONE choke point builds and parses the update check,
  `AutoUpdater::get_release_asset` (`crates/auto_update/src/auto_update.rs`),
  and remote-server downloads flow through it too; Gitea's
  `/api/v1/repos/{owner}/{repo}/releases` is GitHub-compatible with the
  existing `http_client::github` structs (which require `tag_name`,
  `prerelease`, `assets`, `tarball_url`, `zipball_url` to be present); the
  updater does no checksum/signature verification, so unsigned fork installers
  update fine over OS-trusted TLS; on Windows the release asset must BE the
  Inno installer (run `/verysilent /update=true`). Release scheme decided:
  channel `stable`, tags `v<version>` with the PATCH bumped from a high offset
  (upstream `0.196.x` → fork `0.196.100`…) because the comparator STRIPS
  semver pre-release suffixes, so `-fork.1` cannot work. Gate the whole thing
  on a compile-time `ZED_FORK_UPDATE_GITEA_BASE`; unset ⇒ upstream behaviour
  untouched. Depends on an installer to update (phase 44, built) and on
  releases actually being published (the Drone item below).
- **Upstream-merge workflow** (was phase 46; new feature — process + docs).
  Make merging `zed-industries/zed` into the fork routine and documented
  (`docs/fork/upstream-merge.md` + a short checklist), merge NEVER rebase
  (published history, and the phasing system references SHAs). Key facts:
  `origin/main` is an upstream MIRROR kept current by GitHub's fork-sync and
  must stay read-only; `origin/dev` is the fork mainline; upstream velocity is
  ~150 commits/week and a 164-commit dry-run merge was CLEAN on 2026-07-16, so
  weekly merges are cheap and the cost compounds with delay. Conflict hotspots,
  ranked: `Cargo.lock` (churn, trivial — take upstream, `cargo check`
  regenerates), `crates/settings_ui/src/page_data.rs`, `assets/settings/default.json`,
  the three keymap files, `crates/settings_content/src/settings_content.rs`,
  `crates/zed_actions/src/lib.rs`, `crates/gpui/src/elements/list.rs` (semantic
  breakage risk even on a clean merge), `crates/util/src/process.rs`; the fork's
  additions to those are additive blocks that get re-applied. `crates/repl/**`
  is the tail risk: upstream has not touched it since the fork point, but a real
  upstream notebook push would conflict massively. Working clones are SHALLOW —
  merge-base math needs `git fetch --unshallow`. A textually-clean merge can
  still break the notebook, so the playbook must mandate a post-merge smoke test
  (`cargo build -p repl`, clippy, `cargo test -p repl -p gpui`, then open a
  notebook and run cells). The playbook is only "done" once it has survived one
  supervised real merge.
- **Drone pipeline: tag → Windows installer → Gitea release** (was phase 47;
  new feature). A pushed release tag should produce a Drone build on the user's
  Windows runner that runs `script/bundle-windows.ps1` and attaches the
  installer to that tag's Gitea release. Key facts: everything Zed's CI adds on
  top of a local bundle is either Zed-Industries-only (signing, Sentry,
  telemetry seeds, winget) or replaceable, and with `CI` unset the bundle script
  skips it cleanly; `.github/workflows/*` are xtask-generated and stay
  GitHub-only, so `.drone.yml` is a NEW fork-owned file that upstream can never
  conflict with; the runner must be an EXEC runner on Windows (the toolchain
  makes Windows containers impractical) with VS 2022 Build Tools (MSVC x64 +
  Spectre + CMake), Windows 11 SDK 10.0.26100, Inno Setup 6, rustup, long paths
  enabled, no `RUSTFLAGS`, ~100 GB disk (200+ if `target/` persists) and a
  RAM-heavy final link. Cheapest thing to verify FIRST: whether a Gitea
  pull-mirror sync actually fires Drone tag events on the user's versions — if
  not, fall back to a Drone cron pipeline that polls for new `v*` tags. Token
  scoped to the one repo; exclude `target/` from AV scanning or builds crawl.

## Medium priority

- Line numbers per notebook cell (user 2026-08-06). Two settings, BOTH default
  off — a global one for the whole notebook and a per-cell one — plus two
  command-mode keybindings, settled on Jupyter's own convention (user
  2026-08-06): `l` toggles line numbers on the FOCUSED cell, `shift-l` toggles
  the notebook-wide setting.
  **Binding check (done 2026-08-06):** both are free in
  `NotebookEditor && notebook_mode == command` — the only `l` binding anywhere
  is `menu::SelectNext` under the `Prompt` context, which cannot be active
  there, and `shift-l` is unbound. This deliberately avoids `ctrl-l`, which is
  taken (`editor::SelectLine` on Linux, `editor::ScrollCursorCenter` on macOS):
  bound in the command-mode context it would have worked, but in the plain
  `NotebookEditor` context the editor's binding would shadow it while editing a
  cell and it would silently do nothing — the trap behind bug #64. Bind both in
  the COMMAND-MODE context only.
  **Implementation note:** cell editors currently call
  `editor.set_show_gutter(false, cx)` (`cell.rs:529`), so line numbers need the
  gutter turned back ON, which also brings breakpoints / code actions /
  runnables / git-diff markers with it — those need suppressing, or the gutter
  needs a line-numbers-only mode. There is already a per-editor override,
  `Editor::show_line_numbers: Option<bool>` with `line_numbers_enabled()`
  falling back to `EditorSettings::gutter.line_numbers` (`editor/src/config.rs`),
  which is exactly the shape needed for "cell overrides notebook overrides
  global". Also note the notebook draws its OWN gutter to the left of each cell
  (the accent bar + run button, `GUTTER_WIDTH`), so the layout of two adjacent
  gutters needs a look — line numbers should not push the cell content around
  or double the left margin.
- Show a "creating environment" status in the kernel strip, not just in the
  picker (user 2026-08-06, screenshot). While an env is being built the picker
  row says "Creating environment…" but the TOP-RIGHT strip says "Starting",
  which is misleading — nothing is starting yet, a build is running and the
  kernel launch only follows it. Phase 48 deliberately maps
  `creating_kernel_name` onto `KernelStatus::Starting` for the strip
  (`render_kernel_strip`); this wants its own state instead, with its own label
  (and, after phase 62, the spinning icon it already gets for free). Small:
  a `Creating` arm in the strip's status derivation rather than the current
  `if creating { Starting }`. Keep it distinct from the real Starting state so
  the two are legible in sequence: creating → starting → idle.
- Exec timer: "Run All resets the tally" sub-option (user 2026-08-06, after
  confirming phase 61 — user wants this one). Its own setting, only meaningful
  when `repl.notebook_show_execution_time` is on: hitting Run All zeroes the
  tally first, so the number is the wall-clock cost of that ONE end-to-end pass
  regardless of what was run before. Cheap and self-contained: reset
  `execution_time_banked` / `execution_time_started_at` in `run_cells` (the
  RunAll action) BEFORE it calls `run_cell_batch` — NOT inside
  `run_cell_batch`, which is shared with Run Above / Run Below / multi-select
  run, none of which mean "the whole notebook". See the low-priority per-cell
  item for the alternative model this competes with.

## Low priority

- Exec timer: per-cell accounting so a re-run REPLACES that cell's contribution
  (user 2026-08-06). Instead of one accumulator, hold
  `HashMap<CellId, Duration>` of the last duration MEASURED THIS SESSION per
  cell and display the sum (+ the live cell's elapsed). Re-running a cell
  overwrites its entry instead of adding to it, so the total stays "what it
  currently costs to produce this notebook" after editing and re-running part
  of it. The user also wanted deleted cells to stop counting: that falls out
  for FREE if the sum is computed by walking `cell_order` and looking each id
  up (prune-on-read) — no delete/undo/cut bookkeeping needed. Cheaper to build
  than the user expected, but it redefines the number: today (and with the
  Run-All-reset option) it means "time this session spent computing"; this
  makes it "cost to reproduce the notebook as it stands". The two agree right
  after a Run All, which is why this largely subsumes the Run-All-reset option
  — if this is ever built, replace the setting pair with ONE enum
  (`session` | `notebook`) rather than stacking a third boolean. User is
  unsure it is worth it; parked here until the simpler option has been lived
  with.
- Arch Linux distribution (user 2026-07-16, explicitly deferred — "long
  finger"): proper pacman-managed install, i.e. a self-hosted pacman repo
  the user's machines can pull from, or an AUR package (paru-manageable).
  No AppImages. Needs the Linux bundle (`script/bundle-linux`) plus
  PKGBUILD/repo tooling, and updates handed to pacman (build with
  `ZED_UPDATE_EXPLANATION` so in-app auto-update stays off for the pacman
  build). Was gated behind the Windows work, which is now PARKED — so this is
  parked in practice too until the user revives that track.
- Upstream the fork's two generic changes as PRs to zed-industries/zed to
  permanently shrink the merge-conflict surface (phase 43 discovery):
  gpui `scroll_to_reveal_item_top_aligned` (`crates/gpui/src/elements/list.rs`)
  and `util::process` `spawn_interruptible`/Windows interrupt plumbing
  (`crates/util/src/process.rs`).

- Notebook-aware copy/paste with a separate notebook clipboard (user 2026-07-30):
  copying cells should paste as FULL CELLS (structure, types, outputs) into
  another notebook, but as PLAIN CODE — no JSON — into a normal text file or
  outside Zed. VS Code does this with two clipboards: an internal notebook
  clipboard holding the rich cell data, and the system clipboard holding just
  the code. Mirror that: keep the existing rich snapshot for notebook→notebook
  (phase 7/27 machinery) and additionally put a plain-text rendering on the
  SYSTEM clipboard. **Improvement over VS Code the user specifically wants:**
  when producing that plain-text form, prefix every line of a MARKDOWN cell with
  `# ` so pasting a mix of code and markdown cells into a `.py` yields valid
  Python with the prose as comments. (VS Code leaves markdown uncommented, so
  any non-heading markdown becomes a syntax error.)
- Setting to swap the right control sidebar for a VS Code-style TOP control bar
  (user 2026-07-30). Requirements the user gave:
  * A top version of the control bar, which ABSORBS the existing top-right
    kernel cluster (so there aren't two kernel indicators in the same place) and
    makes the sidebar's kernel selector redundant — remove it in this mode
    (note: that sidebar selector is also the one that crashes, bug #54).
  * Remove the right sidebar entirely in this mode.
  * With the sidebar gone, the cell-list scrollbar moves to the true right edge
    and renders properly (relates to bug #57 — the scrollbar currently overlaps
    the cell margin because the sidebar occupies that space).
  * A setting to choose between the two layouts.

- Accurate notebook scrollbar via full measurement (user 2026-07-30). The
  scrollbar thumb is wrong on open and SHRINKS as you scroll: a notebook that
  looked ~3 viewports long read as ~20+ once scrolled to the end.
  **Cause (confirmed):** the cell list is virtualized and an UNMEASURED item
  contributes `px(0.)` to the height tree (`gpui/src/elements/list.rs`,
  `ListItem::Unmeasured` summary). Total content height is therefore massively
  underestimated until items are laid out, and grows as you scroll — so the
  thumb shrinks. This is the SAME root cause as bug #45 (End landing short), and
  the user's diagnosis of that bug was right: End was effectively jumping to the
  end of the MEASURED content, then further each press as more got measured.
  (#45 itself is fixed — End now uses the index-anchored `scroll_to_end`, which
  walks backwards from the last item and so is immune to heights. This item is
  only about the SCROLLBAR being honest.)
  **Lever that already exists:** `ListState::measure_all()` — its own doc says
  "useful for ensuring that the scrollbar size is correct instead of based on
  only rendered elements". Already used by csv_preview, settings_ui and pickers.
  Applying it to the notebook's `cell_list` should give a correct thumb.
  **Cost / the setting the user suggested:** `measure_all` LAYS OUT every item
  on first layout (it does not paint them all), so it is O(cells) work on open —
  noticeable for a large notebook with heavy outputs, but much cheaper than full
  rendering. Do it as the user proposed: measure everything by DEFAULT, with a
  setting (e.g. `notebook_dynamic_render`, default off) to fall back to
  measure-as-you-scroll on low-power machines. Verify open time on the user's
  large Rust notebooks before settling the default.
- Alternative "follow running cell" modes (user 2026-07-31). Follow mode
  currently pins each cell near the TOP of the viewport as it starts
  (`follow_scroll_to`, called from `advance_run_queue` in `notebook_ui.rs`),
  which means you watch the cell's SOURCE — the part you already wrote and the
  least informative part while it runs. What you actually want to watch is the
  execution status and the output being produced.
  **The anchor is the STATUS LINE, not the output (user 2026-07-31, with
  screenshots).** This is not a head-vs-tail-of-output question — the user
  ruled that out. What the framing has to deliver is:
  * that there IS output accumulating (the notebook is visibly working), and
  * the cell's status/runtime footer — the `Running… 10.2s` / `✓ 51.9s` line
    that sits between the source and the output — so you can tell a cell that
    is still running with no output yet from one that has completed or failed.
  The wanted framing is the bottom of the source, the status footer, and the
  first chunk of output all on screen at once. Today's top-aligned framing
  shows only source, giving no signal that the notebook is even still alive.
  Implement by anchoring the reveal on the status footer rather than the cell
  top (`follow_scroll_to`), leaving a little source above it and as much output
  below as fits. Decide what to do when a cell is taller than the viewport —
  the footer is the priority, source above it is the first thing to sacrifice.
  Same underlying complaint as the reveal-target note in the cell-error
  navigation item above; whatever is chosen here should inform that, since both
  are "scroll so the USEFUL part is on screen". The user notes the proposed
  failure indicator in the top kernel strip would partly cover the
  "is-it-still-running" gap, but not fully — this is still worth doing.
  **Priority: explicitly deferred by the user (2026-07-31) — not needed now.**
  **PAGE-WISE FOLLOW is now PHASE 67** (promoted 2026-08-06 with the user's
  answers: added as a SECOND mode rather than replacing this one, selection
  follows execution, entering edit mode turns follow off, a taller-than-viewport
  cell pins its top and overflows). What remains in THIS item is only the
  status-footer anchor described above — a different framing question, still
  deferred.
- Add `smooth_scrolling` to the GUI settings UI (user 2026-07-30). Phase 54
  added the setting to `default.json`, the schema and the docs, but NOT to the
  settings UI — so it's JSON-only today. It belongs in the existing **Editor →
  "Scrolling"** section (`crates/settings_ui/src/page_data.rs:1811`), right
  alongside its siblings `scroll_sensitivity`, `mouse_wheel_zoom` and
  `fast_scroll_sensitivity`, which all already have entries there — so its
  absence is a genuine gap, not an upstream convention. It is a GENERAL editor
  setting (it affects every file, not just notebooks), so it does not belong on
  the REPL & Notebooks page. Small: one `SettingsPageItem` with a
  `json_path: Some("smooth_scrolling")` + pick/place pair following the
  `mouse_wheel_zoom` boolean exactly. NOTE: `page_data.rs` is a known
  upstream-merge hotspot (phase 46) — keep the addition minimal and adjacent to
  the related entries.
