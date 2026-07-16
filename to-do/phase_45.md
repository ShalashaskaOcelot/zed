# Phase 45 — Fork auto-update from the Gitea releases feed

Kind: **change to existing behaviour** (planned by phase 43, 2026-07-16).
Goal: installed fork builds update themselves from the USER'S Gitea releases
— never from zed.dev — or, failing that, updates are cleanly disabled with a
useful explanation. Depends on phase 44 (an installer to update) and phase 47
(releases actually published on Gitea) for end-to-end verification; the code
is implementable and unit-testable before either.

Key discovery facts (details in phase 43's findings, commit `8f9a5d3dc5`):
- ONE choke point builds and parses the update check:
  `AutoUpdater::get_release_asset` (`crates/auto_update/src/auto_update.rs:642-703`),
  expecting custom JSON `{"version","url"}`. Remote-server (SSH remoting)
  downloads flow through the same function.
- Gitea's `/api/v1/repos/{owner}/{repo}/releases` is GitHub-compatible; the
  existing `http_client::github::GithubRelease`/`GithubReleaseAsset` structs
  match it (note: deserialization REQUIRES `tag_name`, `prerelease`,
  `assets`, `tarball_url`, `zipball_url` to be present).
- The updater does NO checksum/signature verification — unsigned fork
  installers update fine; the Gitea host just needs OS-trusted TLS.
- Windows apply: the release asset must BE the Inno installer (run
  `/verysilent /update=true`); `auto_update_helper.exe` finishes the swap.
- `dev` channel never updates; stable/preview compare plain semver with
  pre-release/build metadata STRIPPED; `ZED_UPDATE_EXPLANATION`
  (compile-time or runtime env) is the official full off-switch.

Approach decision (recommended: **Option B**, scoped code change): patch
`get_release_asset` to call Gitea directly and keep `server_url`/zed.dev
untouched for everything else (sign-in, collab, docs links). Option A (a
shim server speaking Zed's JSON in front of Gitea + repointing `server_url`)
needs zero Zed code but repoints ALL zed.dev services and adds a service to
operate — keep it as fallback only.

## Tasks

- [ ] Decide + document the fork release scheme in `docs/fork/releases.md`:
      builds ship as channel `stable` (RELEASE_CHANNEL file set at release
      time, exactly like upstream's release branches); fork releases are
      Gitea tags `v<version>` where `<version>` is the `zed` crate version
      with the PATCH bumped per fork release from a high offset (e.g.
      upstream `0.196.x` → fork `0.196.100`, `0.196.101`, …) so fork
      releases order correctly among themselves and never collide with
      upstream patch numbers. (Semver pre-release suffixes like `-fork.1`
      do NOT work — the comparator strips them.)
- [ ] Add fork update-feed configuration to `crates/auto_update`: a
      compile-time env (`ZED_FORK_UPDATE_GITEA_BASE`, e.g.
      `https://gitea.example.com/owner/repo` — baked by the release build,
      like `ZED_UPDATE_EXPLANATION`) selecting the Gitea path; unset ⇒
      current upstream behaviour so plain dev builds are unaffected.
- [ ] Implement the Gitea branch of `get_release_asset`: GET
      `{base}/api/v1/repos/{owner}/{repo}/releases?limit=…` (or
      `/releases/latest`), tolerate missing optional fields (own minimal
      structs if `GithubRelease`'s required fields prove brittle), filter
      out prereleases/drafts, map `tag_name` (strip leading `v`) →
      `ReleaseAsset.version`, pick the asset named
      `{asset}-{os}-{arch}.<ext>` (define + document the exact asset-name
      convention; Windows app asset = the installer exe) →
      `ReleaseAsset.url`. Both app and remote-server flows go through it —
      remote-server assets 404 until we publish them, which must surface as
      a clear error, not a hang.
- [ ] Point `release_notes_url` at the Gitea release page
      (`{base}/releases/tag/v{version}`) and make
      `view_release_notes_locally` degrade to the browser URL when the
      zed.dev release-notes API is absent (it already error-toasts; make the
      fallback one click, don't build a Gitea-markdown fetcher yet).
- [ ] Unit tests alongside the existing `auto_update` tests: Gitea JSON →
      version/url mapping, `v`-prefix stripping, prerelease filtering,
      asset-name selection, and unset-env ⇒ upstream path untouched.
- [ ] Document the interim state in `docs/fork/releases.md`: until phase 47
      publishes real releases, local/dev builds simply never update (dev
      channel), and manual install of a newer phase-44 installer is the
      update path.

Manual user verification (needs a Gitea release published — after phase 47,
or hand-upload a phase-44 installer to a test tag):

- [ ] ⚠ untested — A fork build configured with the Gitea base sees a newer
      release, downloads, silently installs, and "Restart to update" lands
      in the new version (Windows).
- [ ] ⚠ untested — With no newer release, the manual `auto update: check`
      reports up-to-date; release-notes button opens the Gitea release page.

## Risks / gaps

- The Gitea repo must be PUBLIC (or asset downloads proxied): the updater
  sends no auth header on downloads.
- Version-bump-per-release means `crates/zed/Cargo.toml`'s version field
  diverges from upstream between merges — a guaranteed-but-trivial merge
  conflict; the playbook (phase 46) must say "keep upstream's version, fork
  patch offset is re-applied at release time by the pipeline" (phase 47
  automates the bump at tag time rather than committing it, if possible).
- SSH remoting: remote-server binaries would come from the Gitea feed too;
  the fork doesn't publish them (yet) — acceptable, but the failure must be
  legible. If the user never uses SSH remoting this stays theoretical.
- WSL-sandbox bootstrap has a separate hardcoded `cloud.zed.dev` URL
  (`crates/sandbox/src/windows_wsl.rs:116`) — out of scope unless the user
  uses the WSL sandbox flow.

## Verification

- `cargo test -p auto_update` green; `./script/clippy` clean; unset-env
  builds behave identically to upstream (tests assert it).
- The two ⚠ user tests above once a Gitea release exists.
