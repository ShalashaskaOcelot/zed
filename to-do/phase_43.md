# Phase 43 — Release & distribution discovery (installer, updates, pipeline)

Kind: **discovery / planning** (user-requested 2026-07-16). NO implementation
in this phase — no pipeline code, no installer setup. The deliverables are
findings plus NEW PHASE FILES, each outlining one implementation chunk. Added
as a 6th phase at the user's explicit request.

Target setup (user's infrastructure):
- GitHub fork stays the upstream link (direct fork of zed-industries/zed).
- Self-hosted Gitea mirrors the repo; self-hosted Drone builds it.
- Release artifacts (Windows installer first) attach to Gitea release tags.
- Wanted: a USER-level Windows installer (no admin/system-level install).

## Tasks

- [x] Discovery — how Zed builds its release distributables, WINDOWS in
      particular (findings below).
- [x] Discovery — how Zed's auto-update works today and what must change to
      serve updates from Gitea (findings below).
- [ ] Plan — write `phase_44.md`: build the user-level Windows installer for
      THIS fork locally (reproduce Zed's packaging steps by hand, minus
      signing; document every prerequisite tool/version).
- [ ] Plan — write `phase_45.md`: fork auto-update — point the updater at
      the Gitea releases feed (or document disabling auto-update + manual
      install flow if the updater is too coupled to zed.dev for a first cut).
- [ ] Plan — write `phase_46.md`: upstream-merge workflow — how to pull
      zed-industries/zed into the fork routinely (cadence, `git merge` vs
      rebase for a long-lived fork with a phasing system in-tree, conflict
      hotspots to expect — `crates/repl/**` and `to-do/**` are ours; a
      documented conflict-resolution playbook + a "merge checklist" file).
- [ ] Plan — write `phase_47.md`: Drone pipeline — mirror GitHub → Gitea,
      `.drone.yml` (or starlark) that builds release commits/tags on a
      Windows-capable runner (document the runner requirement — Zed's
      Windows build needs MSVC toolchain), produces the installer from
      phase 44's steps, and publishes artifacts to the Gitea release for
      that tag.

## Findings — Windows release build

- **Installer tech is Inno Setup 6** (no WiX/NSIS anywhere). One source:
  `crates/zed/resources/windows/zed.iss`, compiled by
  `script/bundle-windows.ps1` (`BuildInstaller`, lines 249–361) via the
  hardcoded path `C:\Program Files (x86)\Inno Setup 6\ISCC.exe`. Output:
  `target/Zed-<arch>.exe`.
- **The installer is ALREADY user-level** — nothing to change for scope:
  `PrivilegesRequired=lowest` (`zed.iss:39`), `DefaultDirName={autopf}`
  resolves to `%LOCALAPPDATA%\Programs\<AppName>` under lowest privileges,
  EVERY registry key is HKCU (file associations, PATH via `HKCU\Environment`,
  `zed://` scheme), shortcuts are per-user, and the Win11 Explorer
  context-menu APPX is added per-user (`Add-AppxPackage`). No HKLM, no UAC.
- **Full bundle pipeline** (`script/bundle-windows.ps1:363-383`): stage
  `crates/zed/resources/windows/*` into `inno\<arch>\` → generate licenses
  (cargo-about 0.8.2) → `cargo build --release -p zed -p cli
  -p auto_update_helper --target <arch>-pc-windows-msvc` →
  `explorer_command_injector` DLL (channel-feature) → `remote_server`
  (zipped separately) → makeAppx (SDK 10.0.26100 hardcoded path) → download
  AMD AGS 6.3.0 + ConPTY v1.23 from GitHub → ISCC. Channel comes from
  `crates/zed/RELEASE_CHANNEL` (in-repo: `dev`; `BuildInstaller` has a
  `dev` branding branch, so local dev bundles work out of the box).
- **Signing is entirely CI-gated**: with `$env:CI` unset, Azure Trusted
  Signing, Sentry upload, and the env-var check are all skipped — a plain
  local run yields an unsigned but complete installer. Zed-Industries-only
  pieces a fork drops: Azure Trusted Signing (secrets + `sign.ps1`),
  `ZED_CLIENT_CHECKSUM_SEED` / `ZED_MINIDUMP_ENDPOINT` telemetry keys,
  Sentry symbol upload, `self-32vcpu-windows-2022` runner, winget publish,
  and the APPX publisher-hash suffix (`..._japxn1gcva8rg`) which is derived
  from Zed's signing cert.
- **Toolchain prerequisites** (docs/src/development/windows.md + scripts):
  VS 2022 (script hardcodes the *Community* `Launch-VsDevShell.ps1` path)
  with MSVC x64 build tools, **Spectre-mitigated libs**, CMake, Windows 11
  SDK **10.0.26100**; rustup (1.95.0 pinned by rust-toolchain.toml; the
  script `rustup target add`s the msvc target); Inno Setup 6; long paths
  enabled (registry + `git core.longpaths`); `RUSTFLAGS` must NOT be set;
  outbound HTTPS to github.com during bundling. Disk ~45–100 GB per clean
  build; thin-LTO + codegen-units=1 makes the final link RAM-heavy.
- **Known script warts for non-GitHub runs** (verified): line 354 appends
  `SETUP_PATH=...` to `$env:GITHUB_ENV` unconditionally (unset outside
  GitHub Actions → likely aborts the script after a successful compile);
  line 389's `-Install` path references a stale filename
  (`ZedEditorUserSetup-x64-*.exe` vs the actual `Zed-x86_64.exe`); the
  AppMutex names passed to Inno (`Zed-Stable-Instance-Mutex` etc.) don't
  match what the app actually creates (`Zed-Editor-<Channel>-Instance-Mutex`
  per `crates/release_channel/src/lib.rs:31-38` +
  `crates/zed/src/zed/windows_only_instance.rs:34`) — an upstream bug that
  neuters Inno's running-instance detection (updates still work because
  `/update=true` bypasses the mutex check and `CloseApplications=force`).
- **Release CI shape** (for phase 47): `release.yml` triggers on `v*` tags;
  channel/tag agreement is enforced (`v{version}` = stable,
  `v{version}-pre` = preview) by `script/determine-release-channel`;
  workflow YAML is GENERATED by `cargo xtask workflows` — hand-edits fail CI.

## Findings — auto-update

- **Single choke point**: `AutoUpdater::get_release_asset`
  (`crates/auto_update/src/auto_update.rs:642-703`) builds
  `{server}/releases/{channel}/{version}/asset?os=&arch=&asset=zed` and
  expects the **custom JSON** `{"version": "x.y.z", "url": "<download>"}` —
  NOT GitHub/Gitea release JSON. The same endpoint serves remote-server
  binaries (`asset=zed-remote-server`, used by SSH remoting).
- **The base URL is runtime-configurable**: `server_url` setting (default
  `https://zed.dev`) or `ZED_SERVER_URL` env; any non-zed.dev base passes
  through verbatim (`crates/http_client/src/http_client.rs:295-305`).
  Caveat: `server_url` also repoints sign-in/collab/release-notes, so a
  scoped code change (phase 45) beats repointing the whole setting.
- **Windows apply flow**: the downloaded "asset" IS the Inno installer, run
  `/verysilent /update=true`; it stages into `{app}\install\` and writes
  `{app}\updates\versions.txt`; `{app}\tools\auto_update_helper.exe` swaps
  files after Zed exits and relaunches. A plain zip cannot replace it.
- **No integrity checks**: the updater verifies no checksums or signatures
  anywhere — unsigned fork builds update fine; TLS is the only requirement
  (rustls with native roots: an internal CA in the Windows cert store works).
- **Channel semantics**: `dev` channel NEVER polls or manually updates;
  nightly compares commit SHAs (updates whenever different); stable/preview
  do plain semver `fetched > installed` with pre-release/build metadata
  STRIPPED (so `-fork.N` style pre-release tags cannot distinguish fork
  releases). `auto_update: false` only stops background polling — the manual
  check still installs. The official full off-switch is
  `ZED_UPDATE_EXPLANATION` (compile-time or runtime env), which disables
  polling AND turns manual checks into an explanation dialog while keeping
  remote-server downloads functional.
- **Gitea options** (phase 45): (A) zero-code shim serving Zed's JSON shape
  in front of Gitea, point `server_url` at it; (B) recommended — patch
  `get_release_asset` to call Gitea's GitHub-compatible
  `/api/v1/repos/{owner}/{repo}/releases` directly (the existing
  `http_client::github::GithubRelease` structs match Gitea's JSON) and keep
  zed.dev for everything else. Private-repo assets are a blocker either way
  (downloads send no auth header).

## Findings — fork divergence (for phase 46)

- `origin/main` = upstream mirror (GitHub fork-sync, last 2026-07-15);
  `origin/dev` = fork mainline; fork point `950ec7943f` (2026-07-08);
  **111 fork commits, 37 files, +7,062/−1,024**; upstream moves ~150
  commits/week. A dry-run `git merge-tree HEAD origin/main` today (164
  upstream commits ahead) merges **clean** — weekly merges are currently
  cheap.
- Conflict hotspots ranked: `Cargo.lock` (13% of upstream commits touch it;
  trivial — take upstream, rebuild), `crates/settings_ui/src/page_data.rs`
  (+177 fork lines, upstream actively edits), `assets/settings/default.json`,
  the three keymap files, `crates/settings_content/src/settings_content.rs`,
  `crates/zed_actions/src/lib.rs`, `crates/gpui/src/elements/list.rs` (+102,
  semantic-breakage risk), `crates/util/src/process.rs` (+161).
  `crates/repl/**` is low-frequency but catastrophic-magnitude (upstream
  hasn't touched it since the fork point; fork owns `notebook_ui.rs` at
  2.4× upstream size).
- The local clone is SHALLOW — unshallow before establishing the merge
  workflow. Generic fork changes worth upstreaming to shrink the permanent
  diff: gpui `scroll_to_reveal_item_top_aligned`, util::process
  `spawn_interruptible` (added to backlog).

## Constraints / notes

- Plans must respect that GitHub remains the fork's upstream link; Gitea is
  a MIRROR (one-way GitHub → Gitea is simplest for Drone triggering).
- No implementation here — each plan phase must be executable standalone
  later, with its own verification section.
- Windows expertise caveat: this repo's agents build on Linux; anything that
  can only be verified on Windows must be marked as user-verified steps in
  the implementation phases. User confirmed 2026-07-16 that Windows testing
  falls on them (they test on Windows + Linux anyway).
- Arch Linux distribution (pacman repo / AUR, paru-managed, no AppImages)
  was explicitly DEFERRED by the user 2026-07-16 — recorded in backlog, not
  planned into 44–47.

## Verification

- Findings recorded in this file (then summarized into `CHANGELOG.md` when
  the phase completes); phases 44-47 exist with concrete, actionable task
  lists; user reviews and re-orders/re-scopes them before any is started.
