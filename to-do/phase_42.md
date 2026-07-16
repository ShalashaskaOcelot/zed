# Phase 42 — Kernel environment validation (stale-env handling)

Kind: **change to existing behaviour**. Added as a 6th phase at the user's
explicit request (2026-07-16): after deleting a venv while Zed ran, the
picker/top-right kept showing the deleted kernel as selected and the first
run after recreating the same-named env still failed. No constant polling —
existence is verified at exactly two moments:

## Tasks

- [ ] Validate on RUN: before launching the selected/remembered kernel,
      check its interpreter still exists (local kinds only: PythonEnv /
      Jupyter argv path; skip remote/WSL/SSH). If missing, drop the stale
      selection (own field + per-notebook store memory) and open the kernel
      picker exactly as if no kernel were selected — no doomed launch, no
      error flash. Refresh discovery so a recreated same-name env resolves
      to its NEW interpreter instead of the cached dead spec.
- [ ] Validate on PICKER OPEN: opening the kernel picker kicks a kernelspec/
      toolchain re-discovery (entries already live-update via the store
      observer from bug #20), and prunes/deselects a selected spec whose
      interpreter no longer exists — the checkmark and top-right indicator
      must not point at a ghost env.
- [ ] The top-right indicator reflects the deselection: a vanished env shows
      "Select Kernel" (grey), not the dead env's name.

## Risks / gaps

- The existence check must be cheap (a stat on run/open); no background
  polling.
- Same-name recreation: the cached spec's interpreter path may exist again
  but belong to a NEW env (user saw a failed first run + delayed recovery) —
  refresh-on-prompt should hand back the rediscovered spec, not the cached
  one.

## Verification

- clippy clean; `cargo test -p repl` passes.
- User test: delete a selected venv while Zed runs (notebook closed, kernel
  dead) → reopening the notebook and running opens the picker (no error
  first); the picker shows no ghost entry/checkmark; recreating the same
  `.venv` and running works on the FIRST try.
- Clean bug #30-persistence retest (2026-07-16 attempt was confounded by a
  deleted env): two notebooks, different kernels, RUN + SAVE both, restart →
  each first run uses its own saved kernel without prompting.
