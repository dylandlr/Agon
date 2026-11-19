# Agon — Unity port

The colony sim as a Unity project. The HTML build in `../src` stays the reference
implementation and is still the one that runs today; this tree is the port, and it
grows system by system against the HTML build's behaviour.

**Nothing here has been compiled.** The container this was authored in has no Unity
and no .NET, and `unity.com` is not reachable from it, so the Editor could not be
installed. Every `.cs` file parses clean under a C# grammar, and the simulation
rules are covered by the tests below, but the first real compile happens when you
open the project. Expect to fix a few things on that first open.

## Open it

```sh
# Unity 6000.0 LTS (ProjectSettings/ProjectVersion.txt pins 6000.0.58f1; any 6000.0.x is fine)
unity open unity/
```

Then: new empty scene → empty GameObject → add the **Agon Game** component → Play.
It builds the map, a camp, pawns and the camera on its own. No scene setup, no
prefabs, and no art assets are needed — the wall atlas is generated at runtime by
`WallAtlasGenerator`, so walls render on a blank project.

## Run the tests

Window → General → Test Runner → EditMode → Run All, or:

```sh
unity test unity/ --mode EditMode --report-format junit --output test-results.xml
```

29 of the assertions in `WallLinkingTests` and `CornerFillerTests` were additionally
checked against an independent re-implementation of the same rules before being
written down, so they encode the intended behaviour rather than whatever the code
happens to do.

## Layout

```
Assets/Agon/
  Core/         Agon.Core      — simulation, no rendering
    Map/        Map, TerrainGrid, EdificeGrid, LinkGrid, LinkDirections
    Ticks/      TickManager, TickerType, TimeSpeed
    Defs/       WallMaterial and its properties
    Needs/      Need, Need_Food, Need_Water, Need_Rest, NeedsTracker
    Pawns/      Pawn
  Rendering/    Agon.Rendering — references Core
                LinkedAtlas, WallMeshBuilder, WallAtlasGenerator, WallRenderer
  Bootstrap/    Agon.Bootstrap — references both; AgonGame
  Tests/        Agon.Tests.EditMode
```

Assembly definitions enforce the direction: Core knows nothing about rendering, so
the simulation can be tested and later run headless. `LinkDirections` and
`LinkFlags` live in Core rather than Rendering because what links with what is a
simulation fact, not a drawing one.

## How the walls work

Ported from RimWorld's own decompiled source, which is what "look correct" means
here. The write-up with the exact source references is in
`../docs/walls-rimworld-linked-atlas.md`.

**Linking** (`Verse.Graphic_Linked`). Each wall tests its four cardinal neighbours
in the order Up=1, Right=2, Down=4, Left=8. The resulting 0–15 mask indexes a 4×4
atlas of 16 sub-tiles — `LinkGrid.LinksAt` and `LinkedAtlas.UvRect`. A neighbour
links when its `LinkFlags` intersect the tile's own, so walls join walls, doors,
natural rock and the map edge.

**Atlas UVs** (`Verse.MaterialAtlasPool`). Cell *i* sits at
`(i % 4 * 0.25 + 0.03125, i / 4 * 0.25 + 0.03125)` and samples 0.1875 — the middle
75% of its quarter. The culled border is what stops neighbouring cells bleeding
into each other when the GPU filters, and it is where each tile's arms run out to
meet its neighbour's.

**Corner filling** (`Verse.Graphic_LinkedCornerFiller`). This is the piece the HTML
build was missing and the reason 2×2 wall blocks showed a hole at their centre. For
each diagonal neighbour that links, *and* whose two flanking cardinals also link,
the tile draws an extra half-size quad over its own quadrant facing that diagonal,
sampling a solid point of the texture. All four tiles around a shared corner do it,
so their quadrants together cover the notch. The flanking test is what keeps a
patch from hanging off the end of a wall: a lone diagonal neighbour, as at the
inside of an L, is not a corner to fill.

`WallMeshBuilder.ShouldFillCorner` is that rule on its own, which is why it is
`public static` — the tests drive it directly rather than inspecting a mesh.

## What is here and what is not

Ported: map grids, link grid, tick manager with the three ticker rates, wall
materials, the full wall rendering path, needs (food/water/rest) with the
cold-makes-hunger-fall-faster coupling, pawn health, and a bootstrap that puts a
camp on screen.

Also ported: A* pathfinding, path following with terrain cost, and the job system —
pawns now act on their needs. `JobGiver` picks the most urgent need, `Job` records
what to do, `JobDriver` carries it out over ticks (walk there, then work), and
`JobTracker` holds the one job a pawn has and asks for another when it ends.
`ThingGrid` holds what there is to use — beds, a larder, a water butt — with
reservation, so two pawns are never sent to the same bed.

Also ported: worldgen. `WorldGenerator` builds a map from a seed and a biome —
elevation, water, ground cover, rock and cliffs, the camp site, and scattered
boulders — and it is deterministic on every machine, because `Noise` is implemented
here rather than taken from `Mathf.PerlinNoise`, which Unity does not guarantee
across platforms or versions. A world you cannot reproduce from its seed cannot be
debugged from a bug report.

Not ported yet: emergent tech and staged building, migration and ruins, animals and
ecology, weather beyond temperature, rivals, the glossary, and saving.

## How the pathfinding and jobs work

**A\*** over the grid, eight-directional, with two rules that matter:

- **No corner cutting.** A diagonal step needs both of the cells it passes between to
  be walkable, or pawns slip through the join between two wall corners and walk out
  of sealed rooms.
- **Octile heuristic.** Straight-line distance underestimates grid movement badly
  enough that A\* explores most of the map; octile is the exact cost of an
  unobstructed eight-way walk, so it stays admissible and still prunes.

The open set is a binary heap; closed cells use a generation-stamped array rather
than a HashSet, so repeat searches on one map allocate nothing after the first.

**Jobs** follow RimWorld's Job / JobDriver split: the job is a plain record (what,
where, to what), and the behaviour lives in the driver, so a job can be cancelled or
saved without carrying a state machine. Every driver is walk-then-work, which is the
shape almost all of them have.

## How worldgen works

Passes run in order, and the order matters: elevation first, because every later
pass reads it; then water, ground cover, rock and cliffs, the camp site, and
boulders.

Two choices are worth knowing about:

- **The waterline is a quantile, not a height.** Taking the elevation under which
  the biome's stated share of the map lies is what makes "10% water" true on every
  seed, rather than some seeds coming out as swamp and others bone dry.
- **The camp is levelled and cleared, but not stamped bare.** It takes the same
  ground cover as the rest of the map, with only forest held back because trees
  block building. The HTML build ran its terrain decision only *outside* the camp,
  so every camp tile fell through to bare soil and left a dirt scar that regrew at
  about four tiles a year. `WorldGenTests` has a test whose whole job is to stop
  that coming back.

The camp site is chosen as the flattest patch that is dry and not stone, which is
also why the camera points at it rather than at the middle of the map.

## Notes for the next round

- `WallRenderer` draws with `Graphics.DrawMesh` each frame. Once wall counts grow,
  move to a `MeshRenderer` per material, or to `Graphics.RenderMeshIndirect`.
- `JobGiver` re-offers an unreachable target every tick, so a pawn that cannot get
  to the only store retries forever rather than giving up on it for a while. It is
  correct but wasteful; a short cooldown on a failed target is the fix.
- `AgonGame.GenerateTerrain` is a placeholder. The HTML build's worldgen (ramps,
  pads, caves, rivers, biomes) is the real target and is the largest single piece
  left.
- The runtime-generated atlas is deliberately plain. Replacing it with painted art
  needs nothing but a texture laid out 4×4 in the index order above.
