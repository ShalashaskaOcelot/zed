# Phase 44 — Build the user-level Windows installer locally

Kind: **new feature** (fork release engineering; planned by phase 43,
2026-07-16). Goal: the user can produce a working, unsigned, USER-level
Windows installer of THIS fork on their own Windows machine by following a
documented, reproducible procedure — no CI, no signing, no Zed-Industries
services. This is the foundation phase 47 (Drone) later automates.

Key discovery facts this plan builds on (details in phase 43's findings,
commit `8f9a5d3dc5`):
- Zed's installer is ALREADY user-level (`PrivilegesRequired=lowest`,
  `{autopf}` → `%LOCALAPPDATA%\Programs`, HKCU-only registry) — scope needs
  NO changes.
- `script/bundle-windows.ps1` run WITHOUT `$env:CI` skips signing, Sentry,
  and the env-var check, producing `target/Zed-x86_64.exe`.
- `crates/zed/RELEASE_CHANNEL` is `dev` in-repo and the script has a `dev`
  branding branch ("Zed Dev", own AppId/icon), so a first local bundle
  works with zero channel changes. Note: dev-channel builds NEVER
  auto-update (fine until phase 45 decides the fork's channel strategy).

## Tasks

Implementation (doable on Linux, user-verified on Windows):

- [ ] Fix `script/bundle-windows.ps1` for non-GitHub-Actions runs: guard the
      `>> $env:GITHUB_ENV` append (line 354) so it only writes when the
      variable is set (on a plain machine/Drone it is unset and the
      redirect can abort the script AFTER a successful compile).
- [ ] Fix the stale `-Install` filename (line 389): launch the actual
      output `target/Zed-$Architecture.exe`, not
      `ZedEditorUserSetup-x64-<version>.exe`.
- [ ] Make the VS dev-shell path tolerant (line 43): probe
      `Community`/`Professional`/`BuildTools` editions (e.g. via
      `vswhere.exe -property installationPath`) instead of hardcoding
      Community, so a Build-Tools-only machine (likely for Drone later)
      works unmodified.
- [ ] Write `docs/fork/windows-installer-build.md`: prerequisites with exact
      versions and the hardcoded paths the script expects — VS 2022 with
      MSVC x64 build tools + Spectre-mitigated libs + CMake + Windows 11
      SDK 10.0.26100 (makeAppx path is hardcoded to `10.0.26100.0`),
      rustup (1.95.0 auto-pinned), Inno Setup 6 at
      `C:\Program Files (x86)\Inno Setup 6\ISCC.exe`, long-paths enabled
      (registry + `git config core.longpaths true`), `RUSTFLAGS` must be
      unset, outbound HTTPS to github.com (AGS SDK + ConPTY downloads),
      ~45–100 GB free disk; then the invocation
      (`./script/bundle-windows.ps1` from the repo root in pwsh) and what
      "done" looks like (`target/Zed-x86_64.exe`).

Manual user verification (Windows laptop):

- [ ] ⚠ untested — Follow the doc from scratch: bundle completes and
      produces `target/Zed-x86_64.exe`. Note the wall-clock time and disk
      used (feeds phase 47's runner sizing).
- [ ] ⚠ untested — Run the installer WITHOUT admin rights: no UAC prompt;
      installs under `%LOCALAPPDATA%\Programs\Zed Dev`; Start-menu entry,
      optional desktop icon, `zed` on the user PATH work; SmartScreen's
      unsigned-installer warning is the expected cost (note what it looks
      like for future reference).
- [ ] ⚠ untested — The installed fork build opens and runs a Jupyter
      notebook end-to-end (the whole point of the fork); uninstall from
      per-user Apps & Features cleans up.

## Risks / gaps

- SmartScreen/AV on unsigned installers: a browser-downloaded unsigned exe
  carries Mark-of-the-Web and WILL warn; corporate endpoint policy on the
  work laptop may block it outright — if so, self-signing with an
  internal-CA cert becomes a follow-up item (not planned here).
- The `dev` channel's `Zed Dev` branding/AppId is shared with any real
  upstream dev build the user might also install; acceptable for now,
  revisit when phase 45 picks the fork's channel.
- The AppMutex passed to Inno doesn't match the app's real mutex name
  (upstream bug, see phase 43 findings) — running-instance detection during
  MANUAL reinstalls won't fire. Harmless for this phase; fix only if it
  bites (fold into phase 45's installer touches if needed).
- aarch64 Windows is out of scope (x86_64 only).

## Verification

- `./script/clippy` clean (script changes are PowerShell — no Rust impact);
  the three script fixes reviewed by diff.
- The three ⚠ user tests above pass on the user's Windows machine.
