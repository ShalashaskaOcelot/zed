# Phase 47 — Drone pipeline: tag → Windows installer → Gitea release

Kind: **new feature** (planned by phase 43, 2026-07-16). Goal: pushing a
release tag results — without manual steps — in a Drone build on the user's
Windows-capable runner that produces the phase-44 installer and attaches it
to the Gitea release for that tag. Depends on phase 44 (build procedure +
script fixes); phase 45 (auto-update) consumes what this publishes.

Key discovery facts (details in phase 43's findings, commit `8f9a5d3dc5`):
- Everything Zed's CI adds on top of phase 44's local build is either
  Zed-Industries-only (signing, Sentry, telemetry seeds, winget) or
  replaceable (artifact upload). The bundle script itself is the whole
  build; with `CI` unset it skips all of that cleanly.
- Upstream's trigger model: tag `v{version}` (stable) / `v{version}-pre`
  (preview) must agree with `crates/zed/RELEASE_CHANNEL`;
  `.github/workflows/*` are xtask-generated and stay GitHub-only — the
  Drone pipeline is a NEW, fork-owned file, not a port of those workflows.
- Runner requirements (phase 43 CI findings): Windows Server 2022 / Win 11,
  VS 2022 Build Tools (MSVC x64 + Spectre libs + CMake) + Windows 11 SDK
  10.0.26100 + Inno Setup 6 + rustup, long paths enabled, no `RUSTFLAGS`,
  outbound HTTPS to github.com + crates.io, ~100 GB build disk (200+ if the
  target dir persists between builds), RAM-heavy final link (thin-LTO,
  codegen-units=1). Zed uses a 32-vCPU class machine with 60-min timeouts —
  expect meaningfully longer walls on smaller hardware.

## Tasks

- [ ] Verify the trigger chain on the user's infra FIRST (cheapest thing to
      get wrong): configure the Gitea pull-mirror of the GitHub fork to
      sync tags, and confirm whether mirror syncs fire Drone tag events on
      the user's Gitea/Drone versions. If they don't, fall back to a Drone
      cron pipeline that polls for new `v*` tags and triggers itself.
      Document the outcome in `docs/fork/release-pipeline.md`.
- [ ] Write the Windows runner provisioning checklist in
      `docs/fork/release-pipeline.md` (from the phase-43 runner-spec table:
      exact VS components, SDK 26100, Inno Setup 6 path, rustup, long
      paths, disk sizing, drone-runner-exec as a Windows service — exec
      runner, not Docker: the toolchain stack makes Windows containers
      impractical).
- [ ] Add `.drone.yml` (fork-owned; upstream never conflicts): pipeline
      `type: exec`, `platform: windows`, trigger on `v*` tag events;
      steps: checkout at tag → set `RELEASE_CHANNEL` file to `stable` +
      apply the phase-45 version scheme (fork patch offset) → pwsh
      `./script/bundle-windows.ps1` with `CI` unset,
      `ZED_FORK_UPDATE_GITEA_BASE` set (phase 45), and `GITHUB_ENV`
      pointed at a scratch file → publish `target/Zed-x86_64.exe` (asset
      name per phase 45's convention) to the Gitea release for the tag via
      Gitea's release API (`drone-gitea-release` plugin equivalent or a
      `curl`/`tea` step with a Drone secret token; create the release if
      the tag push didn't).
- [ ] Add a build-health guard step: fail fast with a clear message when
      disk is low (reuse `script/exit-ci-if-dev-drive-is-full.ps1` or a
      simpler check) and clear the target dir above a threshold suited to
      the runner's disk (`script/clear-target-dir-if-larger-than.ps1`).
- [ ] Document the end-to-end release runbook in
      `docs/fork/release-pipeline.md`: bump/tag commands, what Drone does,
      where artifacts land, how to re-run a failed build, and the manual
      fallback (phase 44 local build + hand-upload to the Gitea release).

Manual user verification (all on the user's infra):

- [ ] ⚠ untested — Push a test tag `v*` on GitHub → Gitea mirror picks it
      up → Drone builds → the Gitea release for that tag holds a working
      installer (install it per phase 44's checks).
- [ ] ⚠ untested — A second tagged release: an installed fork build from
      the first release sees and applies the update (this is phase 45's
      end-to-end test — coordinate).

## Risks / gaps

- Gitea mirror-sync → Drone tag-event behaviour is version-dependent and
  unverifiable from this repo; the cron fallback keeps the phase shippable
  either way.
- The runner is a single point of failure and holds a Gitea token — scope
  the token to the one repo, and note the runner must be excluded from
  aggressive AV scanning of `target/` or builds crawl.
- Build duration on non-32-vCPU hardware is unknown until the first run
  (phase 44's timing note feeds this); Drone's default timeouts may need
  raising.
- Unsigned artifacts: SmartScreen warnings on manual downloads are accepted
  (phase 44); silent auto-updates avoid MotW so they're unaffected.

## Verification

- `.drone.yml` + `docs/fork/release-pipeline.md` exist; the two ⚠ user
  tests pass on the user's Gitea/Drone/runner setup.
