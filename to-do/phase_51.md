# Phase 51 — Follow running cell: pin near the top, not the bottom edge

⚠️ **AWAITING USER TESTING** — implementation complete (compiles; scoped
`cargo clippy` clean). The user test below is unconfirmed.

Kind: **change to existing behaviour** — the follow-mode scroll (phase 50) is
being changed, so this item stays OPEN until the user confirms the new landing
position actually takes effect; if it still lands the running cell on the
bottom edge, the change didn't take and it's fixed in place (not archived).

Refinement of phase 50's "Follow running cell": the user found that following a
Run All was awkward because each newly-running cell was revealed on the BOTTOM
edge of the viewport (the minimal-reveal behaviour of
`ListState::scroll_to_reveal_item`). Wanted: the running cell held NEAR the top
of the viewport with a small margin of the preceding cell for context — but
NOT flush to the very top, and crucially without a tall preceding output
(pinned above) pushing the running cell down and out of view.

Primary files: `crates/gpui/src/elements/list.rs` (new scroll primitive),
`crates/repl/src/notebook/notebook_ui.rs` (follow-mode call sites).

## Tasks

- [x] Add `ListState::scroll_to_item_near_top(ix, margin)` in gpui: scroll so
      `ix`'s top sits `margin` px below the viewport top. The margin is a fixed
      offset applied ABOVE `ix` (converted back to a `ListOffset` via the
      sum-tree cursor), so a tall preceding item only ever shows its last
      `margin` px and can never push `ix` out of view. Margin clamped to ≤ ⅓ of
      the viewport (and to 0 before the list is measured) so it degrades on
      short viewports.
- [x] Add a `follow_scroll_to(index)` helper on `NotebookEditor` that computes
      the margin from the viewport (`viewport_height * 0.12`, clamped to
      40–96 px — "near the top" on large screens without dominating small ones)
      and calls the new primitive.
- [x] Route both follow-mode scroll sites through it: the `advance_run_queue`
      hook (batch walk-down) and the immediate jump when the Follow toggle is
      switched on.

## Risks / gaps

- Distinct from the explicit Go to running cell action, which intentionally
  top-aligns (flush) + selects + focuses. Follow stays viewport-only and near
  (not flush) top, so the two don't feel identical.
- Near the END of the notebook there isn't enough content below to hold the
  cell at the near-top anchor; the list clamps to max scroll and the cell lands
  wherever the bottom allows (still visible). Acceptable — matches how any
  list behaves at its tail.
- Margin is a taste value (40–96 px); easy to retune if the user wants
  more/less context above.

## Verification

- [x] `cargo clippy -p repl -p zed_actions` clean (full `./script/clippy` can't
      run in the remote env — `--all-features` needs the system ALSA dev lib).
- [ ] ⚠ untested — User test: Follow on, Run All a notebook where cells have
      substantial output → each running cell settles NEAR the top (a sliver of
      the previous cell visible above), not jammed on the bottom edge, and a
      big previous output does not push the running cell off-screen; walk
      reaches the last cell without lurching.
