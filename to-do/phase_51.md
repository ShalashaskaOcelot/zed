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

- [x] Add `ListState::scroll_to_item_near_top(ix, margin)` in gpui. Anchors on
      an item BOUNDARY: the highest preceding item whose cumulative height (down
      to `ix`'s top) still fits within `margin`, else `ix` itself. The list
      paints from the anchor down, so anchoring on a boundary makes `ix`'s
      position immune to remeasurement of everything above the anchor — a tall
      preceding item (big output OR large markdown cells) is simply not shown
      rather than pushing `ix` down as it lays out. Margin clamped to ≤ ⅓ of the
      viewport (and to 0 before measurement, pinning `ix` flush to the top).
      (First cut anchored a partial slice INTO the item above; that drifted the
      running cell down when a tall markdown/output above it measured taller —
      user-found. Boundary anchoring fixes it.)
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
- Consequence of boundary anchoring: context is whole preceding items that fit
  within the margin (a heading, a short output), not a fixed-height sliver. A
  preceding item taller than the margin shows nothing above — the running cell
  pins to the top. This is the intended trade: robustness over always showing a
  sliver of a big cell.

## Verification

- [x] `cargo clippy` clean on touched crates (full `./script/clippy` can't run
      in the remote env — `--all-features` needs the system ALSA dev lib).
- [ ] ⚠ untested — User test: Follow on, Run All a notebook whose cells have
      substantial output AND large markdown cells → each running cell settles
      NEAR the top (whole short preceding items shown as context; large
      preceding cells NOT shown and NOT pushing it down), never jammed on the
      bottom edge; walk reaches the last cell without lurching.
