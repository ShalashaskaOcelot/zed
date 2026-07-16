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

- [ ] Discovery — how Zed builds its release distributables, WINDOWS in
      particular: walk `.github/workflows/` (release jobs), `script/`
      (bundle-*/package scripts), and any installer sources in the repo
      (WiX/Inno/NSIS configs, `crates/zed/resources`, signing steps). Answer:
      what produces the Windows installer, from which configs/files, what
      makes an install user-level vs system-level (install dir, registry
      scope, shortcuts), and which steps are Zed-Industries-only (signing
      certs, notarization, auto-update feeds) that a fork must replace or
      drop. Record findings IN THIS FILE as they land.
- [ ] Discovery — how Zed's auto-update works today (feed URL, channels,
      version JSON, binary delta vs full download; `crates/auto_update*`):
      what must change so updates come from the USER'S source (Gitea
      releases) instead of zed.dev, and what a minimal fork-update channel
      looks like (e.g. check Gitea's releases API for a newer tag).
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

## Constraints / notes

- Plans must respect that GitHub remains the fork's upstream link; Gitea is
  a MIRROR (one-way GitHub → Gitea is simplest for Drone triggering).
- No implementation here — each plan phase must be executable standalone
  later, with its own verification section.
- Windows expertise caveat: this repo's agents build on Linux; anything that
  can only be verified on Windows must be marked as user-verified steps in
  the implementation phases.

## Verification

- Findings recorded in this file (then summarized into `CHANGELOG.md` when
  the phase completes); phases 44-47 exist with concrete, actionable task
  lists; user reviews and re-orders/re-scopes them before any is started.
