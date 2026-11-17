# Agon p14 — round notes

(Rounds 1–18 notes were lost with the old cloud workspace on 2026-11-16; r18 source was
recovered verbatim from the sent build via pack/recover.js. Summary of r11–r18: ramps/pads/
caves worldgen, grass spread, emergent-default tech with staged rooms + annex + windows,
wall materials wattle/timber/stone with top-down autotiled walls + south faces, survival
economy (water store, providers, pack-eat-first, windbreak/huddle, predator fear), sea-animal
swim fixes, Tech glossary tab. Survival at tick 2500, 5 runs/seed temperate 3/5/7 ended
r17 at 6.6/8.8/10.6 of 12.)

## Round 19 — walls per RimWorld's decompiled linking code (2026-11-16)

Reference: Verse.Graphic_Linked / Graphic_LinkedCornerFiller / MaterialAtlasPool from
decompiled RimWorld (user's REA zip; mirror github.com/josh-m/RW-Decompile). Write-up in
project doc claude/research/walls-rimworld-linked-atlas.md.

Verified:
- Existing 16-sprite cardinal autotile matches RimWorld's Graphic_Linked exactly
  (bits Up=1 Right=2 Down=4 Left=8 over a 16-cell atlas).
- Corner filler added per Graphic_LinkedCornerFiller: where a diagonal neighbour links and
  both flanking cardinals link, a flat 24px patch of the material covers the corner notch
  ('wcf*' sprites + loop after the wall SCENERY loop). Verified on a synthetic 2x2 board:
  hole without patch, solid mass with. NOTE: zero 2x2 wall blocks occur in current worlds
  (5 seeds, 5000 ticks — WALLS is static at 65 and single-thickness everywhere), so this is
  future-proofing for larger shared-corner compounds, not a visible change today.
- doorParts rework: door now lives in the wall band (ink band y7..41 flush with wall
  outline). Closed: planked leaf, lit south face strip, knob. Open: two retracted stubs the
  leaf slides into (RimWorld doors retract the same way). Both axes. Verified on the board
  and in-game (seed 3 camp, tick 213): wall runs no longer break at doors.
- Error sweep: seeds 3 / 5-desert / 7-tundra / 21-sea, 1500 ticks each, zero page errors.

Not done / open:
- Interior walls still push shadow sprites (wsh) same as exterior.
- Window size unchanged.
- Larder never fills; thirst loop (colonists walk to camp with water in skin) untraced.
- Branch question settled (2026-11-17, by evidence, not assumption): colony-sim is a
  strict superset of game/p14-worldgen-forest, so there is nothing to merge. An
  exhaustive token set-difference of the two Main.dc.html files found 96 tokens
  present in worldgen-forest and absent here, every one of them accounted for:
  SVG path data and hex colours from sprite art that rounds 11-18 redrew; the
  capH-based "cap + face" wall layout that the autotile replaced; the hardcoded
  'itlm'/'wasdqe' key strings, since refactored into the remappable KB table; and
  the word "plantains" in one flora note, restored in this commit. Zero feature
  identifiers and zero system names — nothing from that branch's headline work
  (camp spring, passable forest, thirst fixes, tech-gated camp, soil model) is
  missing; those markers all appear MORE here (spring 14 -> 21, soil 23 -> 42).
  The earlier "merge round pending" note was wrong. colony-sim is canonical.


## Round 20 — hearth light, camp dirt (2026-11-17)

Verified:
- **Hearth light no longer outlives the hearth.** The night-darkness overlay's mask
  punched its lit circles at the hearth position unconditionally, so with the fire
  gone (`CAMPON.fire` false) a bright pool stayed where the hearth had been. The
  fire sprite and its glow rings were already gated; the mask was not. Now the
  hearth hole and the shadows it casts are built only when the fire exists, which
  also skips most of the light-mask work when it doesn't. Measured at night, seed 3:
  fire on -> 4 mask circles, fire gone -> 0, night overlay still drawn in both.
- **The camp is no longer a permanent dirt scar.** Worldgen ran the terrain decision
  only for tiles outside the camp (`if (!inCamp(x,y)) {...}`), so every camp tile
  fell through to the `let t = 0` default and became soil. Grass could only creep
  back in from the perimeter: measured at seed 3, 174 of 174 camp cells were soil at
  spawn and still 140 after 8 in-game years, about four tiles reclaimed a year.
  Camp tiles now take the same coverage-model terrain as the rest of the map
  (`b > GTH ? grass : soil`), with forest excluded since trees would block building.
  Seed 3 goes 174 -> 4 soil at spawn and reaches 0 by year 8.
  Checked with real biome switching via `setWorld` (an earlier run that passed
  `AGON_P.biome` silently stayed temperate and produced identical numbers for every
  biome — the result is read from component state, not AGON_P): temperate 201 grass
  / 10 soil, desert 210 soil / 0 grass, tundra 221 soil, savanna 197 soil / 5 grass,
  jungle 159 grass, taiga 205 grass. Zero unbuildable camp tiles in any of them.

Not done, carried forward:
- **Rock and mountain textures need work.** Not started this round.
- **Unity port: pathfinding and the job system.** Pawns in the Unity build have
  needs but no way to act on them; that is the next system, then worldgen.
- Interior walls still push shadow sprites (`wsh`) the same as exterior ones.
- Window size unchanged.
- Larder never fills; the thirst loop (colonists walking to camp with water already
  in the skin) is still untraced.
