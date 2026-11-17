# RimWorld's wall rendering, and how Agon uses it

Reference: RimWorld's decompiled source (the REA zip; the same classes are public
at github.com/josh-m/RW-Decompile under `Verse/`). The decompiled source is **not**
vendored into this repo — it is copyrighted and marked personal-use-only, and this
repo is public. What follows is a description of the algorithm and how Agon
implements it; the implementation here is written from scratch.

Classes that matter: `Graphic_Linked`, `Graphic_LinkedCornerFiller`,
`MaterialAtlasPool`, `LinkGrid`, `LinkDirections`, `LinkFlags`.

## 1. Linking — `Graphic_Linked`

Each tile tests its four cardinal neighbours in a fixed order and accumulates a bit
per linked neighbour:

    Up = 1, Right = 2, Down = 4, Left = 8

The resulting 0–15 mask indexes a 16-cell atlas. Two tiles link when the
neighbour's `LinkFlags` intersect the tile's own (`ShouldLinkWith`), so a wall joins
walls, doors, natural rock, and — when the `MapEdge` flag is set — off-map cells,
which is why a wall run closes against the map edge instead of ending in a stub.

## 2. Atlas UVs — `MaterialAtlasPool`

The 16 sub-tiles sit in a 4×4 grid. Cell *i*:

    offset = (i % 4 * 0.25 + 0.03125,  i / 4 * 0.25 + 0.03125)
    scale  = 0.1875

so each cell samples only the middle 75% of its quarter. The culled 1/32 border on
every side is the point: authored art runs out past the sampled edge, which both
stops neighbouring cells bleeding in under filtering and lets each tile's arms meet
its neighbour's with no seam.

## 3. Corner filling — `Graphic_LinkedCornerFiller`

This is the subclass walls actually draw with, and the piece Agon was missing.

After the base tile, for each **diagonal** neighbour that links, *and* whose **two
flanking cardinals** also link, the tile prints an extra cover quad:

- size 0.5 × 0.5 tiles
- offset along the normalized diagonal by `DistCenterCorner − CoverSizeCornerCorner × 0.5`
  = |(.5,.5)| − |(.5,.5)|×.5 ≈ 0.354, i.e. (0.25, 0.25) per axis — so the quad spans
  from the tile's centre exactly to its corner
- altitude +0.09, so it wins the depth test
- all four UVs the same point, giving a flat patch of the material's colour

Diagonal order is SW, NW, NE, SE, and the guard pairs follow it: (W,S), (W,N),
(E,N), (E,S).

The net effect: all four tiles around a shared corner each cover their own quadrant,
and together they close the notch the inset tile art leaves there. Without it, every
2×2 wall block, every double-thick wall and every shared room corner shows a hole.

The flanking test is what stops a patch hanging off the end of a wall — a lone
diagonal neighbour, as at the inside of an L or where two walls touch only at a
corner, is not a corner to fill.

## How Agon implements it

**Unity** (`unity/Assets/Agon/`) — the faithful version. `LinkGrid.LinksAt` is the
mask, `LinkedAtlas.UvRect` the atlas math, `WallMeshBuilder.ShouldFillCorner` the
corner rule, and `WallMeshBuilder` emits the quads. `WallAtlasGenerator` draws the
16-cell sheet at runtime so it works with no art. Covered by `CornerFillerTests`
and `WallLinkingTests`.

**HTML** (`src/Main.dc.html`, round 19) — the same rules against CSS sprites. The
16 `wl*` sprites per material are the atlas; the `wcf*` sprites are the corner
patch. It differs in one way: rather than four quarter-quads, one 24px patch is
drawn per shared corner from the tile whose south-east corner it is, which produces
the same covered area with a quarter of the elements. Note that current worlds
contain no 2×2 wall blocks at all (verified: 5 seeds × 5000 ticks, walls stay
single-thickness), so the corner filler is future-proofing there rather than a
visible change.
