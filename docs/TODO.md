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


## Round 21 — mute button, colony migration (2026-11-17)

Verified:
- **Mute button.** The audio system already had an `aOn` switch, but only inside the
  Volume settings. There is now a button beside the speed controls that toggles the
  same pref, so the settings panel and the button always agree. It starts the audio
  context on unmute, since browsers only allow that from a gesture. Measured through
  the real WebAudio graph: master gain 0.603 -> 0.003 on mute -> 0.597 on unmute,
  with the label, title and aria-pressed flipping each time.

- **The colony can now move between local areas.** This was the "agents migrating
  with no support" gap, and it was worse than missing polish: every change of local
  cell ran `setWorld`, which sets `tick: 0` and takes `agents` and `store` straight
  from the freshly generated world. Walking to the next valley silently destroyed the
  colony and handed back a different one with a reset clock. New `migrate(cell)`
  carries the living agonists, what they hold, the stores and the techs; `setWorld`
  applies them through `applyCarry` instead of the fresh world's. Left behind:
  buildings, beds and plots, because a house cannot be carried.
  - Journey cost: days scale with distance (region hops weighted x8), bounded at 30
    days. Rations are eaten on the way (about a fifth of a unit per agonist per day),
    water skins arrive empty, and hunger, thirst and energy take a hit that floors
    rather than zeroes — and never *raises* a need, so arriving cannot feed someone
    who set out starving (that was a real bug in the first cut: a colonist at sat 0
    arrived at 0.2).
  - All three `settle` buttons now route through it when a colony is alive; an empty
    world still settles outright. The button says what it will cost, e.g.
    "Move the colony here (17 agonists, 3 days walk)".
  - Measured, seed 3: local step, tick 300 -> 400, 16 agonists with their names and
    3 techs intact, food 60 -> 54, sat 1 -> 0.88. Region trek, tick 300 -> 900 (12
    days), 18 agonists, 11 techs, food 99 -> 56, sat 1 -> 0.45. A non-migrating
    `setWorld` still resets exactly as before: tick 0, different colonists, 0 techs.
  - Uncapped, a cross-planet move billed 521 days, over a year in which nothing
    simulates and every need bottoms out. Capped to 30 days: tick 900 -> 2400.
  - A refused target (open sea, or the cell already occupied) leaves `setWorld`
    unchanged, so the carry is dropped afterwards rather than leaking into whatever
    world change comes next.

Not done, carried forward:
- **RimWorld-inspired UI overhaul.** Asked for, not started. The dock is a fixed
  460px bar centred on the map; a RimWorld-style layout wants a bottom-left
  inspector, bottom-right minimap, and an architect-style build menu. Worth its own
  round and a look at reference shots first.
- **Rock and mountain textures.** Still queued, still needs direction on what is
  wrong with them.
- **Unity port: pathfinding and the job system**, then worldgen.
- Migration leaves animals, bushes and plots fresh each time; a returning colony
  finds no trace of its old camp. Ruins would be the richer behaviour.
- Interior walls still push shadow sprites (`wsh`) the same as exterior ones.
- Larder never fills; the thirst loop is still untraced.


## Round 22 — ruins, and the Unity port gets pathfinding and jobs (2026-11-17)

Verified:
- **Ruins** (see the r22 commit for the detail): a camp a colony walks away from stays
  standing, decaying over about a decade, and sits alongside a rebuild rather than
  replacing it. Returning to your own camp is a head start: measured at seed 3, 59 of
  65 walls and all 80 floors still there a year on, rebuilding to 65 and 5 doors.

- **Unity port: A\* pathfinding and the job system.** Pawns now act on their needs
  instead of merely having them.
  - `PathFinder` — eight-directional A\* with a binary heap and generation-stamped
    closed set, octile heuristic, and no corner cutting (a diagonal needs both
    flanking cells open, or pawns walk out of sealed rooms along the diagonals).
  - `PathFollower` — walks the path paying each cell's terrain cost, and recomputes
    when something blocks the way after the path was found.
  - `Job` / `JobDriver` / `JobGiver` / `JobTracker` — RimWorld's split: the job is a
    plain record, the driver carries it out over ticks as walk-then-work, the giver
    picks by which need has fallen furthest past its own threshold, and the tracker
    holds the one current job. Drivers for Eat, Drink, Sleep, Goto, Wait.
  - `ThingGrid` — beds, larder, water butt, hearth, with stock and reservation, so
    two pawns are never sent to the same bed and an empty store is skipped. Falls
    back to drinking from open water, standing on the bank.
  - `AgonGame` places those things and has a `DebugReport()` giving a line per pawn.
  - Still not compiled (no Unity, no .NET, unity.com unreachable). All 26 .cs files
    parse clean under a C# grammar, and both subsystems were re-implemented
    independently in JS and run: `tests/verify_astar.mjs` (15 checks) and
    `tests/verify_jobs.mjs` (24 checks), both green.
  - The cross-check earned its keep three times: two of my A\* fixtures were wrong
    maps rather than wrong code (one "doorway" map was in fact sealed), and the sleep
    test's loop exited while the pawn was still in bed, so its "and they get up
    afterwards" assertion would have failed in Unity. All fixed before commit.

Not done, carried forward:
- **RimWorld-inspired UI overhaul.** Still wants reference shots and a direction
  before any code: the dock is a fixed 460px centred bar, and a RimWorld layout wants
  a bottom-left inspector, bottom-right minimap and an architect-style build menu.
- **Rock and mountain textures.** Still needs a direction on what is wrong with them.
- Unity: worldgen is next, then the emergent tech and staged building.
- Unity: `JobGiver` re-offers an unreachable target every tick; a cooldown on a failed
  target would stop the retry loop.
- Interior walls still push shadow sprites (`wsh`) the same as exterior ones.
- Larder never fills in the HTML build; the thirst loop is still untraced.


## Round 23 — Unity worldgen, and a look at the rock art (2026-11-17)

Verified:
- **Unity worldgen.** `WorldGenerator` builds a map from a seed and a biome:
  elevation (fractal plus a ridge, so maps have a spine), water, ground cover, rock
  and impassable cliffs, the camp site, and scattered boulders. `Noise` is
  implemented rather than taken from `Mathf.PerlinNoise`, which Unity does not
  guarantee across platforms or versions — a world that cannot be reproduced from
  its seed cannot be debugged from a bug report. `BiomeDef` holds the six biomes as
  rows of numbers, so a new biome is a row rather than a branch in the generator.
  - The waterline is a quantile of the elevation field, not a fixed height, so a
    biome's stated water share holds on every seed. Measured within 0.02 of target
    for temperate, jungle, desert and savanna.
  - The camp is levelled and cleared but **not** stamped bare — it takes the same
    cover as the rest of the map, forest excluded. There is a test whose only job is
    to stop the HTML build's dirt-scar regression reappearing here.
  - Checked over 48 worlds (6 biomes x 8 seeds): every camp site is free of water,
    rock, forest and boulders, walkable throughout, on flat ground and inland.
  - `AgonGame` now generates its world instead of scattering placeholder terrain,
    builds the camp at the chosen site, and points the camera there.
  - Still not compiled. All 30 .cs parse clean; worldgen was re-implemented
    independently in JS and run (`tests/verify_worldgen.mjs`, 15 checks, green), as
    were the A* and job loops from round 22.

Rock textures — a look, not a fix:
- Screenshot sent (`agon-rock-terrain.png`). The rock/mountain mass reads as a flat
  brown silhouette next to boulders that are faceted and lit and look right.
- `rockTex` does exist and does emit — 1848 texture ellipses and 5 rock clip-groups
  reach the DOM. An earlier probe of mine reported zero and was simply wrong: the
  SVG is percent-encoded and the probe searched for raw tags. So the texture is
  there and too faint to read at normal zoom, rather than missing. Waiting on Dyl to
  confirm the mass in the screenshot is the thing he means before changing anything.

Not done, carried forward:
- **RimWorld-inspired UI overhaul.** Still wants reference shots.
- **Rock and mountain textures.** Diagnosis above; awaiting confirmation.
- Unity: emergent tech and staged building next, then migration and ruins.
- Unity: `JobGiver` re-offers an unreachable target every tick; a cooldown would fix it.
- Interior walls still push shadow sprites (`wsh`) the same as exterior ones.
- Larder never fills in the HTML build; the thirst loop is still untraced.


## Round 24 — the camp stops being a cookie-cutter hole in the map (2026-11-17)

Dyl: "it inserts itself on top of a local cell and it carved out parts of the
existing map and environment." Confirmed as the camp.

Verified:
- **Cause found.** `if (inCamp(x, y)) t = b > GTH ? 1 : 0;` forbade forest across the
  entire camp blob, and trees render on forest terrain — so the tree-line traced the
  camp's superellipse outline exactly. Screenshot at seed 13 jungle, tick 120, shows
  a smooth rounded clearing with a wide empty apron on every side.
- **Fix.** The clearing now follows what the colony actually builds: a distance field
  from `CAMPPTS` (the same point list `CAMPBOX` is built from, annex included), with
  clear ground kept only within `CAMPCLR` = 2.6 tiles of a structure, and the edge
  jittered by +/-1.2 tiles so it stops following a curve. Before/after at identical
  seed, tick, lighting and camera: trees now encroach from three sides and the edge
  is ragged.
- Invariant holds across six biomes: `buildTilesOnForest` is 0 everywhere, so nothing
  the colony builds ever lands on a tree. Forest inside camp is 0/4/15/3/0/0 for
  temperate3 / temperate5 / taiga9 / jungle13 / tundra7 / desert11 — trees where the
  biome has them, none where it does not.
- Survival unharmed: 11/10/7/12/11/9 alive at tick 2500 across those six; the
  four-config sweep is clean at 15-17 alive, no page errors.

A failed attempt first, recorded because the measurement is the useful part:
- I assumed the carve was the **elevation** stamp (`if (inCamp) e = CAMPE`) and built
  a feathered depth field to ease it out. Three things were wrong and each was caught
  by measuring rather than by looking:
  1. The first depth function returned 1 for all 192 camp tiles, because `CAMPBOX` is
     15x10 and the whole superellipse falls inside it (`tilesOnlyInBlob: 0`). The
     feather never ran.
  2. Rebuilt on distance-to-structure, it still returned 1 for 150 of 192, because
     the floors cover nearly the whole box. There is no yard inside the camp to
     feather across.
  3. Extending the feather outward past `inCamp` did run — and made things worse:
     against a matched baseline, earth moved went 39.9 -> 42.7, tiles touched 353 ->
     595, and the rim metric was **identical to four decimal places in all five
     worlds**, meaning the discontinuity I was chasing was never the camp's. Reverted
     the whole elevation line of work.
- Lesson for the next one of these: the complaint was visual, and one screenshot of
  the right place would have pointed at the terrain rule in a minute. I measured
  elevation for an hour first.

Not done, carried forward:
- The clearing is less cookie-cutter but its apron is still fairly wide, and the east
  edge still reads smooth on seed 13. Worth another pass if Dyl wants it tighter.
- **RimWorld-inspired UI overhaul.** Still wants reference shots.
- **Rock and mountain textures.** Diagnosis stands (texture emits but is too faint);
  awaiting confirmation the mass in the screenshot is what he meant.
- Unity: emergent tech and staged building next; worldgen there already sites the
  camp on flat ground and does not stamp it bare.
- Larder never fills; the thirst loop is still untraced.

## Round 25 — RimWorld-faithful UI (layout A)

Dyl picked layout A from the three mockups: inspector bottom-left, minimap
bottom-right, alerts down the right edge. This round moves the real HUD to match it.

Verified (measured against the live build at 1440x900):
- **The minimap is a real one, not a second one.** Before building anything I found
  the existing minimap at template line ~229 after a key collision gave it away
  (`miniDots` returned 74 markers, not the 16 mine would have, and a 2.5px dot width
  I had not written). My duplicate was deleted whole. What is new on it is
  `miniViewBox` — a white rectangle showing which part of the world the camera is
  on — and `miniGo`, click-to-look-there, both on the panel that was already there.
  Measured: box at `left: 45.65%; top: 3.85%; width: 47.44%; height: 96.15%` on seed
  7, and a click moved `panX` from -1367.42 to -1574.20.
- **The minimap is up by default.** It was `Map: { hide: true }` in `_pp`, which is
  why bottom-right was empty in the first attempt. Not an `sc-if`: the `</sc-if>`
  sitting above the panel belongs to the Agon menu button, which cost me a detour.
- **The clock and the speed controls moved into the minimap panel**, as in the mock:
  day/night bar, `Year 1 - Summer 9  23C`, weather and wind, the days-to-next-season
  bar, then play / 1x / 2x / 4x and the mute button. The panel widened 132 -> 188px
  (`mW` 114 -> 170). The speed row was *moved*, not copied: the dock keeps only its
  summary line and the rewind scrubber.
- **Alerts moved from the left column to the right edge** (`right: 16px; top: 64px`,
  right-aligned). The resource readout stays top-left, which is where RimWorld puts
  it. Checked with 2 and 3 alerts across four worlds: none overflow.
- **The inspector moved from a right-hand aside to bottom-left**
  (`left: 16px; bottom: 56px; width: 360px; max-height: 520px`), measured at
  y 324-844, clear of the taskbar at y 850 and of the minimap. It had to move: the
  alerts now occupy the right edge it used to sit on.
- **Bottom-left is now a single slot.** Inspector, Tools and Log hide each other, in
  both the click handler and the keyboard handler — the old code only paired
  Tools<->Log, and the string appears twice in the file, which is worth remembering.
- The tab row moved up 30px (`top: calc(100% - 46px)`) so it stops overlapping the
  dock strip, which it did in the first attempt.
- Sweeps clean: `tests/final3.py` 15-18 alive at tick 1500 on temperate / desert /
  tundra / sea, no page errors; `tests/mute.py` still toggles masterGain
  0.603 -> 0.003 -> 0.597 from the button's new home.

Rough:
- The dock's bounding box reads as off-screen in the geometry check (866-992 against
  a 900px viewport). That is its own `translateY(calc(100% - 34px))` park, by design,
  not a regression — noted so the next sweep does not chase it.
- The season bar and its "1 days to autumn" label are tight against each other at
  188px wide. Legible, not pretty.

Found, not fixed — Dyl's call, because it is balance and not UI:
- **Colonists are born 437 years old.** `age` is in ticks and advances correctly
  (873883 -> 874783 over 900 ticks), and the inspector's "% of lifespan" is right.
  The scale is what is wrong: `LIFE = 2000000` ticks against `YEAR = 2000` is a
  1000-year lifespan, and the adult draw is `(0.1 + r*0.4) * LIFE`, i.e. 100-500
  years. A human-scale `LIFE` would be about 150000. Changing it moves aging, death,
  fertility and the `trained` ramp, so it is not something to slip into a UI round.

Not done, carried forward:
- Layout A's colonist bar is drawn top-left in the mock; the real one is top-centre.
  Left as is — RimWorld centres its colonist bar, so the mock was the less faithful
  of the two.
- The clearing apron is still fairly wide; seed 13's east edge still reads smooth.
- **Rock and mountain textures.** Diagnosis stands; awaiting confirmation.
- Unity: emergent tech and staged building next, then migration and ruins.
- Unity `JobGiver` re-offers unreachable targets every tick.
- Interior walls still push shadow sprites; larder never fills; thirst loop untraced.

## Round 26 — rivals, caves, animals, and the beginnings of a language

Six asks in one message. Five are done, one question is answered below, and one thing I
found along the way is left for Dyl to decide.

Verified:
- **Rivals no longer start with tools the colony has to earn.** `RTECH` was a hardcoded
  `['shaping','binding','hafting','footwork']`, substituted for the colony's real tech list
  wherever a rival fought or crafted (`techs: a.fac ? RTECH : techs`). It is now the same
  starting set the colony gets — `['firecraft']` by default — and rivals still do not
  research, so that is where they stay. Confirmed in the build and at runtime.
- **The brick look on rock and cave walls is gone.** The cause was two
  `repeating-linear-gradient`s on every rock tile: horizontal lines every 9-10px over the
  whole tile and vertical ticks every 6-7px on the south face. Regular spacing in both axes
  on an unvarying grid is mortar coursing, which is exactly what it read as. Replaced with
  bedding planes that only appear on sedimentary and metamorphic rock, at an offset and
  spacing drawn per tile so neighbours never line up, and irregular fracture marks on
  igneous and granite. Checked at 3.2x zoom against a cliff face.
- **Caves lead somewhere now.** Measured before: 2-6 caves per map, largest 27 tiles, max
  depth 8-12. The passages were dying early — walker life was `max(6, target * 0.55)` — and
  the rock budget was 30% of host tiles. Now `max(16, target * 0.85)`, budget 50%, and a
  larger target. After: seed 3's largest cave went 15 -> 52 tiles and max depth 9 -> 17;
  seed 21 went to 26 deep and 140 cave tiles. Jungle is the exception and barely moved
  (31 tiles, depth 6) because that map has almost no host rock to dig into.
- **Animals are at true relative size.** The game already had `REAL`, a table of real
  head-and-body lengths in metres, but only 54 of 103 species were in it — every one of the
  49 missing was aquatic — and the conversion had two faults: a hard 11px floor that
  flattened everything under half a metre (a 0.12m vole and a 0.5m hawk both landed at
  ~14px), and raw division by the sprite-span table, which let a short-spanning sprite
  out-draw a longer animal (pika, 0.2m, drew at 19px; hawk, 0.5m, at 15px). Added the 49
  missing lengths and replaced the floor with `px = 40.5 * metres^0.55`, with the span
  divisor clamped to [0.75, 0.95]. All 103 species now have a real length, and across every
  pair of species one of which is more than 30% longer than the other, exactly one still
  draws smaller — muskox 80px against bison 79px, a one-pixel tie.
- **Signs can be read as words.** Every message already carried a `text` beside its glyphs;
  nothing surfaced it. There is now a Signs/Words button beside the sound button, and in
  Words mode the bubble reads "Tired, wants a bed" instead of three pictures.
- **The colony coins its own signs, and grammar is one of them.** This is the new system.
  The lexicon starts at five signs and five only — the ones a body can make on itself:
  empty belly, dry mouth, closed eye, wound, fright. Every other glyph is uncoined, and an
  uncoined glyph shows as a question mark, so you see that something was meant without
  knowing what. A sign is coined when two colonists are standing together, one of them
  means something the colony has no sign for, and the roll goes their way; wit and memory
  make it likelier. `arrow` is deliberately in that pool, because it is not a noun — it is
  the relation that lets two signs be put in order. Until it is coined, every compound
  message truncates to its first sign, so colonists can say "cold" but not "cold -> fire".
  Measured on seed 3: 5 signs at tick 0, 10 by tick 401 (arrow among them), 13 by 1601,
  and a plateau there — the colony coins signs for what it actually lives through, and
  never coins ones for experiences it has not had. Each coining writes a line to the log.
- **Damage numbers and stamina in the world.** `harm` now records what it took off, and the
  number rises off the victim and fades over six ticks; anything under one point is not
  drawn, so no more "-0". A stamina bar sits under each health bar, blue down to yellow
  under 50% and red under 25%. Energy was already a numbered row in the inspector.
- Sweeps clean throughout: `tests/final3.py` 11-19 alive at tick 1500 across four worlds,
  no page errors; `tests/mute.py` unchanged; the layout-A geometry check still passes.

The question Dyl asked — **why the UI is not RimWorld level yet** — is answered in the
reply, not here, but the two measurable parts are: the HUD is a set of panels that each
solve their own problem rather than one information hierarchy, and almost nothing is
clickable as a *thing* (the map, the bars and the resource rows are read-only; RimWorld's
are all entry points).

Found along the way, not fixed:
- **`setWorld` takes positional arguments** — `setWorld(size, form, biome, seed, nRivals, gp,
  cell)` — and an object argument hits `if (!SIZES[kk]) return;` and silently does nothing.
  My first cave measurement passed an object and got identical numbers for every seed and
  biome, which is what gave it away. Any future sweep must use the positional form; the
  silent return makes a wrong call look like a working one.
- **Something strips keys from the store.** I added `start0` to the store literal, confirmed
  it in source and in the build, and it never appeared in `state.store` at runtime — while
  `forced`, declared in the same object literal, did. The live store's key order
  (`...insight, bad, spoiled, lots, lk, know, forced...`) does not match the literal's
  either, so the state store is not the object that literal builds. I worked around it by
  computing the rival tech list from globals instead, and I have not explained it. Worth
  understanding before anything else is added to the store.
- **Caves still dead-end rather than descending.** This round made the passages much longer,
  but a cave is still one level: `CDEPTH` is distance from the mouth, not a floor below.
  Real "deeper" would be linked levels with their own maps. That is a feature, not a tune.
- Jungle maps have too little host rock for caves to grow; worth a biome-specific floor.
- Rivals get a flat insulation of 6 from their floors where the colony gets its wall
  material's value (wattle is 8). That is a rival *disadvantage*, not a head start, so I
  left it; say if you want it levelled too.

Carried forward: camp clearing apron width; Unity emergent tech, staged building, migration
and ruins; Unity `JobGiver` cooldown on unreachable targets; interior walls pushing shadow
sprites; larder never fills; thirst loop untraced; colonists still born 437 years old
(`LIFE = 2000000` against `YEAR = 2000`, r25).

## Round 27 — the default start has no house

Verified:
- **The default scenario is now a band with nothing built.** `SCENARIOS.community` — twelve
  settlers in a walled house with a hearth, a bench and a stocked table — was the fallback in
  five places, and it also handed over the `shelter` tech at tick 0 (`SC.house === 'full' ?
  ['shelter'] : []`). A new `band` scenario is the default: same twelve people, no walls, no
  shelter tech. `community` is still there and still selectable.
- **"No house" was deleting the blueprint, not just leaving it unbuilt.** This is the real
  find. `gen(r, house)` skips its entire room-construction block when `house === 'none'`, so
  `CAMP.W`, `CAMP.F`, `CAMP.D` and `CAMP.beds` all came out empty. `applyCamp` rebuilds the
  live `WALLS`/`FLOORS`/`BEDS` from those arrays every tick, so a band that discovered shelter
  would turn `room0` on and still have nowhere to put it — `WALLS.size` sat at exactly 15 from
  tick 1 to tick 12000, and those 15 were the rival camp's, which is what made it look static.
  The bug was already live for the `loner` and `nomads` scenarios; making no-house the default
  is what surfaced it. The blueprint is now always generated in full, and `CAMPON` alone — which
  `applyCamp` derives from discovered techs each tick — decides what is standing.
  Measured on seed 3: tick 1, twelve colonists, `campon: []`, `shelterT: null`; tick 1001, only
  `fire` on; shelter discovered at tick 1504; by tick 3001 `room0/room1/room2` are on and walls
  have gone 15 -> 65, floors 10 -> 80. The house is built rather than given.
  (I first patched `legacy()` at 1210-1214 and measured no change — that is the fallback
  generator, only reached when `gen` fails 400 times. The live one is `gen` at 1217.)
- **The store key-stripping from r26 is explained.** Line 4616 rebuilds the store every tick as
  an explicit field whitelist: `{ shelterT, water, wallMat, norms, mats, food, items, known,
  techs, insight, bad, spoiled, lots, lk, know, forced, rec, plots0, cult }`. Anything outside
  that list is dropped on the first tick, which is exactly why `start0` disappeared while
  `forced`, declared in the same object literal, survived; the whitelist's key order matches the
  runtime store's order exactly. **Adding a field to the store literal does nothing unless it is
  also added at 4616.** Worth knowing before anything else goes in there.
- `tests/final3.py` on the new default: 13-18 alive at tick 1500 across temperate, desert,
  tundra and sea, no page errors.

Measured after the fact, and it corrected me:
- **The unbuilt start is not the cause of the population collapse.** I flagged seed 3 falling to
  2 alive at tick 6000 and assumed the shelterless start did it. The band-versus-community sweep
  says otherwise. Alive at tick 12000 across four seeds and biomes:

      band       3 temperate 4   11 temperate 3   5 desert 5   7 tundra 6     mean 4.5
      community  3 temperate 9   11 temperate 7   5 desert 0   7 tundra 5     mean 5.25

  Band is marginally worse, and the only total extinction in the set is *community* on desert
  seed 5. Both scenarios fall from 12-15 at tick 2000 to about 5 by tick 6000. So there is a
  **pre-existing survival collapse that has nothing to do with the house**, and it is a larger
  problem than the scenario default. Do not tune the band's starting food to chase it; the
  walled start with a stocked table collapses the same way.
- **Shelter arrives far too late in tundra.** Time to `shelterT` for the band: desert 308,
  temperate 2412 and 2564, **tundra 8820** — still unwalled at tick 6000, in the coldest biome
  there is. Community reaches it in 716 there, so this is about discovery order under cold
  pressure, not about the blueprint. Worth fixing before the collapse above, since freezing
  without walls probably feeds it.
- `larder never fills` turns out to be about `store.food`, not a larder — the word only appears
  in scenario blurbs. `storeCapN` is at least 16, so the cap is not the blocker; the deposit
  sites are 5714 (carcass) and 5724 (berries, needs `a.inv.berries >= 3`) against consumption at
  5645. Not traced to a cause yet.

Carried forward: camp clearing apron width; Unity emergent tech, staged building, migration and
ruins; Unity `JobGiver` cooldown on unreachable targets; interior walls pushing shadow sprites;
thirst loop untraced; colonists born 437 years old (`LIFE = 2000000` against `YEAR = 2000`,
r25); jungle maps too rock-poor for caves; caves still one level rather than descending.

## Round 28 — shelter can be discovered by living, not only by crafting

This round answers the first half of Dyl's ask. The second half — placing rooms when they are
earned rather than reading extents fixed at worldgen — is scoped below and deliberately not
started, because it cannot be half-done without leaving the camp system broken.

Verified:
- **`LORE.shelter` had seven patterns and all seven were crafting.** `['craft','wood','wood']`
  and friends, `min: 2`. So every colony in every biome arrived at shelter from the workbench,
  and a band that spent a winter shivering in the open learned nothing from it. Four experiential
  patterns are added beside them:

      ['ev', 'cold', 2000]       being genuinely cold (a.temp < -0.5) and not under a roof
      ['ev', 'stormed', 1800]    rain or storm, out in it, and not under rock either
      ['ev', 'sheltered', 25]    standing inside a cave (terrain 6 or 7)
      ['ev', 'sawwall', 60]      within 3 tiles of a wall that is not the colony's own

  Each is a tick counter on `a.work`, so it is carried by the colonist and lost when they die —
  the same rule the craft signatures already follow. `sawwall` uses `!inCamp(x, y)` so that
  standing next to your own house teaches nothing, which is the whole point of that route.
- **The tundra outlier is gone and the spread is now biome-shaped.** Time to `shelterT` on the
  band start, measured over 4000 ticks:

      before:  temperate 2564 / 2412,  desert 308,   tundra 8820
      after:   tundra 192,  desert 520,  temperate 912 / 1684,  jungle 960

  Tundra now learns shelter fastest, which is what being cold all the time should teach you.
  Nothing has a page error; 6-11 colonists alive at tick 4000 across five worlds.
- Evidence accrues differently per biome, which was the point: `sheltered` fires in tundra (53)
  and jungle (84) and nowhere else; `sawwall` in temperate seed 11 (52) and tundra (20) and
  nowhere else; `cold` and `stormed` accrue everywhere.
- `tests/final3.py` 9-17 alive at tick 1500 across four worlds, no page errors.
  `tests/mute.py` unchanged.

Honest about what I could not establish:
- **I cannot say which route actually triggered any given discovery.** `ev` is summed over
  *living* colonists only, so a reading taken at tick 4000 does not describe the colony that
  existed at tick 192 — the people whose cold ticks or cave ticks tipped it may be dead by then.
  An early run of this showed exactly that trap: temperate reported `sheltered: 0, sawwall: 0`
  at tick 4000 while its `shelterT` had clearly moved when those two thresholds changed. To
  attribute a route properly, `loreHave` would have to log which patterns were satisfied at the
  moment the tech was derived. Worth doing before tuning these numbers further.
- Three tuning passes were needed and the first two were wrong in opposite directions: thresholds
  of 240/90/60/40 put shelter at tick 64 everywhere (the counters are colony-wide sums over a
  dozen people, so they reach hundreds within a few hundred ticks), and 2000/1800/120/500 left
  the cave and rival-wall routes dead at zero. The numbers committed are the third pass.

Scoped, not started — incremental floorplan (r29 if Dyl wants it):
- `applyCamp` iterates `CAMPW0`, `CAMPF0` and `CAMPK`, all built at worldgen. They are `const`
  bindings holding Sets, so their contents can be extended at runtime. `ROOMX`/`NROOM` have only
  3 and 7 readers and `state.rooms` is unused, so the slots are free.
- Design: `growRoom(committed, seed)` chooses the next room's extent at the moment it is earned,
  scored on buildability (no water, rock or cliff), flatness from `elevAt`, compactness against
  the rooms already standing, and the leeward side via `windAt`; it then adds its tiles to
  `CAMPW0`/`CAMPF0`/`CAMPK` and registers the extent. `state.rooms` persists the extents and
  `campGrp`/`applyCamp` read those instead of `ROOMX`. Room 0 stays from worldgen because it
  carries the hearth and all the furniture placement.
- **The hard part is rehydration.** The engine is rebuilt on load and on area change, so the
  runtime tiles must be reconstructed from `state.rooms` deterministically every time it comes
  up, or a reload or a migration loses rooms silently — and `campSig`'s early-return would hide
  it. That is the piece that needs a round to itself.

Carried forward: the scenario-independent population collapse (~12-15 down to about 5 by tick
6000, measured r27 — still the biggest open problem); `store.food` never accumulating; camp
clearing apron width; Unity emergent tech, staged building, migration and ruins; Unity
`JobGiver` cooldown; interior walls pushing shadow sprites; thirst loop untraced; colonists
born 437 years old (`LIFE = 2000000` against `YEAR = 2000`); jungle maps too rock-poor for
caves; caves still one level rather than descending.

## Round 29 — three real bugs fixed, one diagnosis overturned, and a large correction to my own measurements

### The correction first, because it matters more than anything else here

**`window.AGON_P` has no `seed` key and no `biome` key.** It is a bag of about fifty behaviour
flags (`hung`, `tire`, `eatFirst`, `storeCap`, `noStartFire`, ...). Passing
`{seed:21,biome:'desert'}` sets keys nothing reads, so the run is the default world every time:
`wseed 33403`, temperate, same camp, same 48 bushes. Verified by reading `state.wseed` and
`state.biome` back from six runs that should have differed — all identical.

World choice goes through `setWorld`, which is **positional**:
`setWorld(size, form, biome, seed, nRivals, gp, cell)`. That genuinely works (wseed 9076 / 5497 /
40190 / 24245 / 18239 for five worlds).

So the per-seed and per-biome structure of several earlier claims is invalid:

- **r28's headline table is void.** "Time to shelter: tundra 192, desert 520, temperate 912/1684,
  jungle 960" was five runs of one temperate world. The four new experiential routes to shelter
  are still in and still sound as a mechanism, but **the claim that the tundra outlier was fixed
  and that the spread follows the biome is unsupported** and needs re-measuring through `setWorld`.
- **r27's band-versus-community sweep** was one world sampled eight times, not four worlds twice.
  Its conclusion (the collapse is not caused by the unbuilt start) may still hold but was not
  established the way I said it was.
- **The r28/r29 analysis doc** is written as "seed 21". It is the default world. Its *within-world*
  findings stand -- the decision funnel, parley rates, who attacks, tech loss, the lexicon freeze,
  relDecay -- because none of those depend on which world it was. Its biome framing does not.

I had already caught `AGON_P.biome` doing nothing in an earlier round and then kept using
`AGON_P.seed`, assuming the seed worked even though the biome did not. It does not.

### Fixed, verified

- **Rival colonists had unseeded personalities, so no run was reproducible from its world seed.**
  `mkAgent` seeds each agent's personality generator from `500 + i * 7`, but for `i > 99` it added
  `Math.floor(Math.random() * 99999)` -- and rivals are created with ids `100 + k * 10 + j`, so
  every rival in every run got an unseeded brain. Found by fingerprinting two identical runs: they
  diverged on the **first tick**, in exactly four agents, all rivals, needs differing in the fourth
  decimal with positions identical. `mkAgent` already receives the seeded generator as `r`; it now
  uses it. Two runs are now byte-identical at 1, 5, 20, 100, 500, 1000 and 2000 ticks.
- **A stray debug global was shipping in builds.** `window.__TILLED = (window.__TILLED || 0) + 1`
  sat in the tilling path of every release build. Removed.
- **The only `Math.random()` inside the tick** was the tool-wear roll beside it; now `r()`.
- **A test-only `AGON_P.paused` flag.** The harness used to poll every 50 ms until `window.__agon`
  existed, so the game auto-advanced a load-dependent number of ticks before pausing and no two
  runs started from the same tick. Runs now start at tick 0 every time.
- **`tests/final3.py` rewritten** to select worlds through positional `setWorld`, with the trap
  documented at the top of the file. It now sweeps five genuinely different worlds: 9-12 colonists
  alive at tick 1500 across temperate, desert, tundra, jungle, no page errors.

### Diagnosis made, then overturned by its own A/B

I traced the food economy on a reproducible run and found what looked like a poverty trap. A
colonist could only bank food while holding **3+ berries** *and* standing within **1 tile** of one
of the **two** table cells *and* with goal `haul` or `wander`. Measured co-occurrence of the first
two conditions: 47 agent-ticks in 0-500, 150 in 500-1000, then **0 in 1000-2000, 0 in 2000-4000,
0 in 4000-6000** -- because as the near bushes thinned the band ranged further and agent-ticks spent
beside the table fell from 1047 to 62. Meanwhile `store.food` ran 6 -> 30 -> 33 -> 19 -> 0 and
`spoiled` froze at exactly 29 for 7000 ticks, which is what nothing-enters-storage looks like.

So I widened it: 2+ berries, within 2 tiles, and crates and the hearth counted as drop-off points.
Then I A/B'd it over five real worlds at 6000 ticks:

                 pop   deposits   food>0   meanSat   techs   extinct
      before     2.6      27      34.4%     0.402     7.6     2 of 5
      after      2.2      26      39.8%     0.386     6.4     2 of 5

**Deposits were already happening about 27 times per 6000 ticks in both arms.** The narrow window
was not the binding constraint, and the change costs satiation and techs for a small gain in the
fraction of time the store is non-empty. Reverted. The "both conditions never co-occur" measurement
was real but it was taken on one world, and it does not generalise.

What the A/B did establish, and it is worse than anything reported before: **mean population is 2.6
of a starting 12 at tick 6000, with 2 of 5 worlds extinct.** Every earlier survival figure came from
one world.

### Open, with a named hypothesis

- **Determinism still breaks between tick 2200 and 2500.** Not `Math.random` and not `Date.now` --
  I trapped both across 4000 advances and neither is called once. The first thing to differ is
  ecology: animal count 52 vs 44, bush count 86 vs 79, with techs still equal. Hypothesis:
  `stepPlanetEco` runs on `tick % 100 === 0 && this._eco`, and `this._eco` is created lazily by
  `ecoFor`, which the render path also calls -- so whether it exists when the tick-0 step fires
  depends on whether a frame rendered first, which is wall-clock dependent. A one-step offset in a
  planet ecology would surface exactly this way, thousands of ticks later. Worth confirming by
  forcing `_eco` to exist before tick 0.
- **Re-measure r28's shelter routes through `setWorld`** before trusting any biome claim about them.
- The population collapse is still the dominant problem and is now measured properly: 2.6 of 12.

Carried forward: incremental floorplan (scoped in r28); relDecay 0.01 erasing relationship memory;
decideEvery 6 discarding 69% of contact; 0 kills in 20,000 ticks; the colony never raids; lexicon
needs a second growth path; camp clearing apron; Unity emergent tech, staged building, migration
and ruins; Unity `JobGiver` cooldown; interior walls pushing shadow sprites; colonists born 437
years old; caves one level only.

## Round 30 — three view toggles, and an honest audit of the walls

Verified:
- **Shadows on/off.** One gate at the point every shadow is collected, so it covers scenery,
  colonists, corpses, animals and bushes together. Measured: 1371 shadow elements in the DOM with
  it on, 640 with it off, 1371 again on. (The 640 that remain are wall ink and other dark fills,
  not shadows.)
- **Names instead of numbers.** Colonists already carry `a.name`; the in-world label was hardcoded
  to `{{a.id}}`. It now reads `{{a.nameLabel}}`, which is the name when the toggle is on and the
  id otherwise. Measured: 29 id labels and 55 name strings on screen by default, 0 ids and 84
  names with it on.
- **Action captions.** When a colonist's `act` changes to something worth announcing, the word
  rises over their head and fades over 10 ticks -- `eating`, `gathering`, `taking shelter`,
  `fighting` and so on, from a table of 15. Only on a change, or every pawn would carry a
  permanent caption; `move` and `idle` are deliberately not in the table. Measured: 10 captions
  on screen at once shortly after switching it on.
- All three sit in the Map panel beside Sound and Signs. They render as plain text when off,
  which is the existing `segBtn` inactive style, the same as Sound and Signs.
- `tests/final3.py` across five worlds: 5-13 colonists alive at tick 1500, no page errors.

### The walls: the algorithm was already right, and I should say so

I audited the HTML renderer against `docs/walls-rimworld-linked-atlas.md` rather than assuming
there was something to fix, and most of it already matched:

- The link mask is `Up=1, Right=2, Down=4, Left=8`, which is `Graphic_Linked`'s order exactly.
- All 16 mask variants exist for each of the three materials, in both window and no-window forms,
  plus the corner-filler sprite. The atlas is complete.
- The corner-filler guard is `lk(1,0) && lk(0,1) && lk(1,1)` -- the SE diagonal plus both flanking
  cardinals, which is `Graphic_LinkedCornerFiller`'s rule. Iterating every wall tile and testing
  only its SE corner covers every four-tile meeting exactly once, so it is equivalent to
  RimWorld's four-diagonal version without the duplicate quads.

**The one genuine gap was the link flags.** `ShouldLinkWith` in RimWorld is a `LinkFlags`
intersection, and a wall's flags take in natural rock and the map edge as well as walls and doors
-- which is why a RimWorld wall raised against a mountain joins the rock face instead of showing
an end cap. Agon linked walls and doors only. Added `linksAt`, which also links Mountain (5),
Cliff (12), boulders and out-of-bounds. Hills (3) deliberately does not link: it is walkable
ground here, not a rock face.

**And here is the part worth knowing: it barely matters in practice.** Measured across the five
worlds, 307 camp walls in total:

- rock-adjacent wall sides: **1**
- map-edge wall sides: **0**
- corners meeting the fill rule: **0**, and 0 corner sprites in the DOM

The camp is always sited on cleared flat ground away from rock, and its floorplan is entirely
single-thickness walls, so there is never a 2x2 block to notch. The corner filler ported in r19 is
correct but is dead code for the standard camp; I confirmed the rule still fires by adding a
synthetic 2x2 block and checking it. So the link-flags fix is correct rather than visible, and the
remaining differences from RimWorld are art (grain texture, the lit south strip), not wall logic.

Not done:
- Walls are still SVG sprites rather than a sampled 4x4 texture atlas. That is a rendering
  technique difference, not a behaviour one, and nothing in the audit suggests it would look
  closer for the effort.
- If RimWorld-style thick walls or rock-abutting camps are wanted, the floorplan has to produce
  them -- which is the incremental-floorplan work scoped in r28.

Carried forward: incremental floorplan; the population collapse (mean 2.6 of 12 at tick 6000,
2 of 5 worlds extinct, r29); determinism breaking between tick 2200 and 2500 with the
`stepPlanetEco`/`ecoFor` lazy-creation hypothesis; re-measuring r28's shelter routes through
`setWorld`; relDecay 0.01; decideEvery 6; 0 kills in 20,000 ticks; the colony never raids; lexicon
second growth path; Unity port items; colonists born 437 years old.

## Round 31 — the house is built by colonists, tile by tile

Dyl, mid-round: "rooms should be built like how its done in rimworld by pawns". So the timer goes.

### What changed

Before this, `stage = floor((tick - shelterT) / BUILD_T)` revealed one whole room every 120 ticks,
free, with no labour and no materials. Now:

- **`store.built`** is a set of the camp tiles that have actually been raised (whitelisted in the
  tick's store rebuild at 4635, or it would be dropped -- see r29).
- **`applyCamp`'s `tileOn`** consults it: a house wall or floor exists when somebody built it. The
  room group still decides whether a room is *open* to build; `built` decides whether each tile is
  standing. So you watch a wall go up tile by tile rather than a room appear whole.
- **`campSig` includes `BUILT.size`**, or the cached early-return would freeze the build.
- **`stage` now counts finished rooms, not ticks**: the next room opens when the one before it is
  fully standing, so the house grows at the pace the colony can build.
- **`campTodo(built)`** is the queue, exposed from engine scope (CAMPW0/CAMPF0 live there). Walls
  before floors -- a room with its floor down and no walls keeps nothing out.
- **A `build` work type**, a `build` act that walks to a queued tile, spends one unit of wood or
  stone (by `wallMatFor`), and raises it. One tile per visit.
- **`BUILDKEEP = 10`**: construction never takes the last logs.

### Verified

On one world: shelter discovered at tick 868, 43 tiles standing by 1500 (walls 15 -> 55), queue
empty at 120 tiles by tick 12000, and `store.mats.wood` drawn 46 -> 1 -> 0 as they spend it. The
house is earned.

Two things the measurements taught me, both worth more than the feature:

- **Building used to take the last log.** In the first version wood sat at 0 for thousands of
  ticks while the walls went up, so nobody could craft clothing. The tundra runs died with
  `walls 15, built 0, techs 2` -- they froze before even discovering shelter, against 73/0/13 on
  the old timer. Hence `BUILDKEEP`.
- **Build work had to rank with clothing, not with chores.** Offered only in the candidate list at
  priority 3 it never won, because the food-security rule returns `gather` on almost every tick in
  a colony this hungry: `built` stayed 0 for 12000 ticks. It is now an early return gated on the
  colonist being fed (`sat > 0.45`) and the reserve being intact.

### The measurement problem, which is the real finding

I ran a 5-world A/B and read it as a regression: mean population 3.6 -> 2.6, extinctions 1/5 -> 3/5.
Then a follow-up run of what I believed was a changed build gave 2.0 and 3/5 -- except the edit had
silently failed, so **it was the same code as the run that gave 2.4 and 4/5**. Identical code, one
world set, materially different answers.

So I measured the noise. 15 runs of the unchanged build:

    BEFORE  n=15  pop mean 3.00  sd 2.94  median 2.0  extinct 5/15 (33%)
            pops: 4 4 0 2 0 6 9 0 1 0 6 7 0 5 1

**The standard deviation equals the mean.** Standard error at n=15 is 0.76, so a difference has to
clear about 2.1 to mean anything. Every 5-run comparison in this session -- including r29's deposit
A/B -- is inside the noise and should be treated as unmeasured, not as evidence. The run-to-run
spread comes from the r29 determinism bug (divergence after ~2200 ticks, not `Math.random`, not
`Date.now`), which is now the thing blocking every gameplay decision.

The AFTER arm at the same n is noted in the next entry.

Carried forward, reordered by what blocks what:
1. **The determinism bug** -- until identical runs give identical results, no balance change can be
   evaluated. Hypothesis in r29: `stepPlanetEco` fires on `tick % 100 === 0 && this._eco`, and
   `_eco` is created lazily by `ecoFor`, which the render path also calls.
2. **The population collapse** (mean 3.0 of a starting 12 at tick 6000, a third extinct).
3. Runtime room extents, now much easier with build state per tile.
4. relDecay 0.01; decideEvery 6; 0 kills in 20,000 ticks; the colony never raids; lexicon second
   growth path; re-measure r28's shelter routes through `setWorld`; Unity port; colonists born 437
   years old.

## Round 31b — the build loop lands at no survival cost

The AFTER arm of r31 at n=15, against the n=15 baseline in the previous entry:

                        mean   sd    median  extinct      sat    built
    BEFORE (free timer) 3.00   2.94   2.0     5/15 (33%)  0.171   0.0
    walls + floors      1.07   2.41   0.0    12/15 (80%)  0.039  34.6
    walls only          3.33   3.05   3.0     5/15 (33%)  0.253  16.3

Charging for both walls and floors was lethal: 80% of colonies dead by tick 6000 against 33%,
z = 2.6 on the extinction rate, p < 0.01 -- outside the noise floor, unlike every 5-run comparison
earlier in this session.

The framing that matters: **this did not create a problem, it removed a crutch.** The old timer
handed the colony about 63 walls for free regardless of what it could afford, which was a survival
subsidy masking the collapse measured since r27. Charging for shelter simply exposed that the
colony cannot pay for it.

The fix was not to lower numbers until it passed. Floors are trodden ground, not construction:
queueing them doubled the work (120 tiles against 65 walls) and the material bill for no shelter
benefit, since a room keeps the weather out when its walls close. Only walls are built now; an
opened room's floor is simply there.

Result: identical extinction rate to getting the walls free (5/15 both), satiation better
(0.253 against 0.171), and mean population 3.33 against 3.00 -- which is inside the noise, so the
claim is **no detectable regression**, not an improvement. `tests/final3.py` across five worlds:
4-14 colonists alive at tick 1500, no page errors.

So the house is earned rather than given, and it costs the colony nothing it was not already
paying. 16.3 tiles standing on average at tick 6000, against 65 for a finished house -- most
colonies are still mid-build when they fail, which is the collapse problem, not the build loop.

## r32 — determinism hunt (NOT FIXED; narrowed hard)

Dyl asked to fix determinism before roofs. Determinism is **not fixed**. Roofs not started.
Harness saved in `tests/determinism/` so this resumes cold.

### The bug, measured precisely
Two runs of identical code on the same world diverge. Uninstrumented (`uninstrumented.py`,
no rng wrapper, no instrumentation of any kind): 5 fresh pages at 3000 ticks gave 4 distinct
state hashes, population 11/12/12/12/12.

`first_diverging_tick.py` (6 pages, order-stable per-tick projection) pins it exactly:
- first differing tick is **always 1093**, never earlier, never later
- pages fall into **exactly two outcome clusters**, split ~50/50
So it is one binary branch at one fixed tick, not drift.

### What is ruled out (with the evidence)
- **Environment reads in the sim.** 0 occurrences of Math.random / Date.now / new Date /
  performance.now / requestAnimationFrame / setTimeout / getElementById / querySelector /
  getBoundingClientRect / offsetWidth / innerWidth / devicePixelRatio / navigator / document /
  localStorage / WeakMap / WeakRef across all 1436 lines of `advance` (4643-6078), and across
  ENGINE scope (1-3600). art.js has 2 Math.random but both are inside `mockDecide`, reachable
  only when LLM mode is on (off by default) — still a real determinism bug for LLM mode.
- **Start state.** `exhaust_fields.py`: at tick 1092, of **259** own fields across the component
  and the engine, exactly **2** vary across 6 pages — `__host` (the DOM node) and `_rebakeMs`
  (a `performance.now()` duration that is written at 3989 and never read anywhere). All 257
  sim-relevant fields are identical, as is the whole of `state` (full sorted deep diff: 0 fields).
- **RNG alignment.** 8 pages at waits of 400/700/1500 ms: identical state hash, identical engine
  sizes, identical all-12-cache fingerprint, and identical next 4 rng outputs. Stream is aligned.
- **The out-of-state caches.** `_Q` (per-agent Q-tables, full q-vectors), `_PV`, `_TX`, `_away`,
  `_burnT` hashed per tick for 1100 ticks across 4 pages: zero divergence.
- **Render.** Patching `__agon.render = () => null` for the whole run: still diverges (5 pages,
  3 distinct). Render also never touches `this.rng` (0 occurrences past 6093). `advance` calls
  `setState` exactly once, as its last statement.
- **The satellite painter.** Killing `_satW` + stubbing `_sat.prep`: diverges *more* (3 distinct
  vs 1), so not the cause.
- **Transcendental JIT instability.** `micro_transcendental.py`: sin cos tan exp log pow (both
  forms) ** atan atan2 hypot sqrt cbrt, 400 inputs each, digested cold then after 300k warmup
  iterations — **all stable**, byte-identical. So this is NOT last-bit float drift.

### The one decisive asymmetry
`jit_flags.py`: with `--js-flags=--no-opt`, 4 pages are **byte-identical**. With default JIT,
4 pages give **4 distinct** hashes. Since the transcendentals are provably stable, `--no-opt`
is *masking* something timing-related rather than fixing float math. Instrumenting the run also
pins the outcome to one cluster, which points the same way.

### Where it has to be
Identical `state`, identical all 257 enumerable fields, identical caches, aligned rng — yet the
tick-1093 branch flips. The remaining input is therefore **closure-scope state not reachable by
property enumeration**: a `let`/`const` inside `ENGINE()` (1195) or inside a bundle module, which
`Object.keys` cannot see. Next round: instrument at source level (counter/log inside ENGINE,
not a wrapper) and bisect tick 1093's own execution.

### Fixed this round
- **`setWorld` leaked 8 of 12 per-world caches.** It reset `_Q/_K/_KC/_away` only; `_burnT`,
  `_TX`, `_PV`, `_eco`, `_busy`, `_dec`, `_calls`, `_culture` survived a world change, so a new
  world inherited the old one's learning, toxin rolls, burn clocks and provisioning roster.
  Measured as a tick-1 divergence between two runs of the same world in one page. Now all 12
  are cleared. No behaviour change on a fresh page (they start empty) — verified: 3 pages at
  1000 ticks identical, alive=14 built=8 techs=9 walls=23, no page errors.

### Found, NOT fixed (needs a decision from Dyl)
- **`engineFor` memoizes engines: `ENGINES[id] || (ENGINES[id] = ENGINE(...))` (3709).**
  Re-entering a world returns the *same* engine object with every closure mutation from the
  previous visit intact (campSig, grass spread into TER, CAMPON, scenery). So "new game, same
  seed" resumes a dirty world instead of a fresh one. I did not clear it: the memo is presumably
  what lets you walk back to an already-visited macro cell and find it as you left it, so
  clearing it trades world persistence for determinism. **Which way?**
- **`this._culture` is read at 5224 but written nowhere in the codebase.** The cultural-inheritance
  path for a new colonist's Q-table (inherit the colony's accumulated learning rather than only
  the best living teacher's) is dead code and has never run.
- **`mockDecide` (art.js 1918, 1931) uses `Math.random`** for goal choice and reply delay, so
  LLM-mock mode is nondeterministic by construction. Left alone: it is off by default and the
  fix needs a seeded rng threaded into art.js.

### Not done
- **Roofs.** Deliberately not started: the point of fixing determinism first was to make the roof
  change measurable, and it still isn't. Say the word and I will either keep hunting at source
  level or build roofs accepting that A/B results below ~2.1 population remain inside the noise.

## r33 — determinism FIXED (root cause: random comparator in Array.prototype.sort)

### Root cause
`moveTo`'s sidestep fallback shuffled the 8 neighbour offsets with
`[...8 dirs].sort(() => r() - 0.5)` (old src line 5200). **V8's `Array.prototype.sort` calls the
comparator a different number of times depending on whether the calling frame is optimized.**
Measured directly (`tests/determinism/sortcmp` logic, mulberry32 rng, 8 elements):

- unoptimized (cold): **14** comparator calls
- optimized (hot): **8** comparator calls
- and the resulting permutation differs in **200 of 201** seeds tested

The comparator draws from the colony rng, so the moment TurboFan tiered up the frame containing
that sort, the run consumed a different number of draws, the rng stream desynchronised, and every
later tick diverged. Whether tier-up happened before or after the first time a colonist got boxed
in decided which way a run went — hence a *fixed* first-divergence tick with *two* stable outcomes.

### Why it took so long to see
Every way of observing it moved it, because observation changes tier-up timing:
- `--no-opt` → deterministic (always cold, always 14 draws)
- `--always-turbofan` / `--always-opt` → deterministic (always hot, always 8)
- an `Error().stack` label on each draw → pinned to the cold outcome
- enabling the existing `mvDbg` flag → pinned to the cold outcome
- inserting 13 no-op `ph()` statements into `advance` → pinned to the hot outcome
- heavy per-tick full-state dumps → pinned to one outcome
Only a *counter* on the rng was light enough to leave the bug intact, which is what finally
measured it: at tick 1093 the two clusters took **129 vs 115** draws, a 14-draw gap — one
8-element shuffle's worth.

### How it was localised (in order, all evidence in tests/determinism/)
1. `uninstrumented.py` — real, 5 pages / 3000 ticks gave 4 distinct hashes.
2. `first_diverging_tick.py` — first difference always tick 1093, two stable clusters.
3. `exact2` (full-precision `String(number)`, no rounding) — ticks 1..1092 **bit-identical**, so
   no float drift and nothing to quantize; a non-state input had to be read at 1093.
4. `classify.py` — classified each page, then diffed all 262 component+engine fields between a
   genuine canonical and a genuine deviant page at tick 1092: only `__host` (a DOM node) and
   `_rebakeMs` (a `performance.now()` duration that is written and never read) differed.
5. `draws3` — draw counts 129 vs 115 at tick 1093. A branch, not arithmetic.
6. Symptom check: `hap` differed by exactly 0.003 = the `grief ? 0.15 : 0` term in `hapT` times the
   0.02 EMA rate, i.e. downstream of one colonist stepping to y=11 instead of y=12. A movement flip.
7. 14 draws ≈ one 8-element random-comparator sort → micro-tested the sort → root cause.

Ruled out along the way, each with evidence, not inspection: clock/DOM/`Math.random` reads anywhere
in the tick path (0 occurrences across all 1436 lines of `advance`, all of ENGINE scope, and art.js
outside the LLM mock); render (stubbing it still diverged, and it never touches the rng); the
satellite painter (killing it diverged *more*); all 12 out-of-state caches; transcendental JIT
instability (13 functions stable cold vs hot at magnitudes 1 to 1e15); FMA contraction (10 patterns
stable; x86_64 anyway); every engine rng stream (all fresh-per-call except `rs`, used once at
worldgen); `MARS` (round-trips through state, `marInit()` returns a fresh object); `bfs`'s reusable
stamped buffers (stamps strictly increase, cannot collide).

### The fix
Fisher-Yates in place of the sort: exactly `ds.length - 1` draws, every time, tier-independent,
and unbiased (a random comparator is not a fair shuffle anyway). One site; it was the only
randomised or side-effecting sort comparator in the codebase (checked all 18 `.sort(` call sites
in Main.dc.html and art.js).

### Verified
`tests/determinism/verify_fixed.py` — fresh pages, no instrumentation, exact full-precision hash:
- 6000 ticks, default world: 3/3 identical (alive 6, built 50)
- 9000 ticks, default world: 3/3 identical (alive 6, built 50)
- 4000 ticks, seed 21 temperate: 3/3 identical
- 4000 ticks, seed 7 tundra: 3/3 identical
- no page errors in any run
`jit_flags.py` — `--no-opt` and default now produce the **same** hash (`1shorr9`); before the fix
`--no-opt` gave one value and default gave 4 distinct ones. That is the direct confirmation.

### Consequences to know about
- **Every A/B baseline from r1-r32 is void.** The draw sequence changed, so all previous numbers
  (the n=15 noise floor of mean 3.00 sd 2.94 included) were measured on a sim that silently forked
  mid-run. The noise floor has to be re-measured before any future A/B means anything — and it
  should now be much tighter, since identical inputs finally give identical output.
- A debug hook stayed in `advance`: `window.AGON_TR = <tick>` labels every rng draw taken during
  that one tick with its call site. Off by default (one comparison per tick). Determinism was
  verified with it present.
- Still open from r32, unchanged: `engineFor` memoizes engines so re-entering a world resumes a
  dirty one (needs Dyl's call — world persistence vs determinism); `this._culture` read but never
  written; `mockDecide` uses `Math.random` so LLM-mock mode is still nondeterministic.
- Seeds 21 and 7 finish 4000 ticks with `built=0` — pawn construction never starts there. Separate
  issue (shelter discovery), not new.

### Not done
- **Roofs.** Determinism was the gate and it is now open: a roof change can finally be A/B'd
  against a stable baseline. Next round.

## r34 — noise floor re-measured; roofs designed, NOT implemented

No source changes. Roofs are a proposal awaiting a go, as asked.

### Determinism gate (passed)
tests/determinism/verify_fixed.py, exit 0: 6000 and 9000 ticks on the default world and 4000
ticks on seed 21 temperate and seed 7 tundra, three fresh pages each, exact full-precision
state hash, no page errors. Repeat-run variance is exactly zero.

### The noise floor
Old floor (mean 3.00, sd 2.94, n=15) is void twice: the draw sequence changed in r33, and it
mixed run-to-run nondeterminism with world-to-world variance. Those are different things, and
the first is now zero.

New panel, tests/noise/panel.py: 12 biomes x seeds 3 and 21 = 24 worlds, fresh page each,
24 distinct world hashes. At 6000 ticks: alive mean 2.50 sd 3.44 (min 0, max 13, median 0),
walls 36.25/20.94, built 21.21/19.70, techs 9.38/4.85. Extinct 13/24 = 54%. No page errors.
Raw rows in /tmp/mar/panel6000.json (ephemeral; re-run the panel to regenerate).

Comparison method for every future A/B: same 24 worlds under both arms, compared as PAIRED
per-seed differences, never two independent means.
- Unpaired MDE on this panel: 2.91 colonists at 80% power. Useless against a mean of 2.50.
- Paired MDE = 2.93 * sd_d / sqrt(24): rho 0.9 -> 0.92, 0.7 -> 1.59, 0.5 -> 2.06, 0 -> 2.91.
- rho is NOT estimable without a real A/B, so the first roof A/B must report sd_d. Expect it
  lower than intuition: any change to the draw count makes the arms diverge outright from the
  tick it first fires. Pairing still removes biome and terrain variance, the big component.
- Halving the MDE needs 4x the panel = 96 worlds, about 60 minutes.

Surviving population is a BAD primary endpoint: median 0, 54% on the floor. Panel now also
records `survived` (time to extinction) and `peak`; use time-to-extinction as primary next
round. Its baseline needs one panel re-run (~15 min) as the roof round's first step.

### The finding that reframes roofs
Shelter is one predicate, FLOORS.has(tile), used at 8 sites in the tick (5479, 5495, 5497,
5587, 5663, 5667, 5694, 5915). Floors are NOT gated on walls being built — source comment:
"a floor is there as soon as the room is opened". So full wall-material insulation (wattle 8,
timber 10, stone 12) arrives the instant a room opens, with no wall standing.

**Built walls currently buy nothing thermally. The r31 build loop has no survival payoff.**

Panel agrees: built>0 worlds had alive mean 1.94 and 9/16 extinct; built==0 worlds had 3.62
and 4/8. Small n and confounded by biome, so not proof that building hurts — but no evidence
it helps, and the code says why it wouldn't.

### Roof proposal (full version in claude/research/roofs-design-and-noise-floor.md)
A. Roof as the last course (low risk): ROOF set + CAMPR0 blueprint, campTodo returns a room's
   walls then its roof, shelter moves off FLOORS. **Not as a hard swap** — only 5/24 worlds
   finish their walls by 6000 ticks (8 never start, 11 partial), so an all-or-nothing gate
   would strand 19/24. Grade insulation by roof coverage instead; safer AND more realistic
   than RimWorld's 75% cliff.
B. Spans, supports, collapse (medium): roof buildable only within span(material) of a support,
   BFS from supports like CDEPTH; losing the last support collapses it, reusing the existing
   cave roof-fall (rubble + harm blunt/blast).
C. Enclosed-room thermal envelope (high): flood-fill the room, relax toward ambient at a rate
   set by wall ins x perimeter + roof ins x area + door leakage, hearth as interior source.
   Replaces the flat additive warmth, so every downstream balance number moves.

Recommendation: A with graded coverage, paired against the panel; then B; C only if B holds.

RimWorld reference (from the wiki, never decompiled source): 6-tile support chaining through
roof, thin-roof collapse 15-30 crush + rubble, constructed roofs cost NO materials and take 65
ticks, room needs >=75% coverage to decouple from outdoor temperature. The three places to be
more honest: charge materials, span by material, insulation by coverage.

### Decision needed
engineFor memoizes engines (ENGINES[id] || (ENGINES[id] = ENGINE(...))), so re-entering a world
resumes a mutated one — "new game, same seed" is not a fresh world. Clearing it trades world
persistence (walking back to a visited macro cell) for determinism across re-entry. Which way?

### Still open, unchanged
this._culture read but never written (cultural inheritance of learning is dead code);
mockDecide uses Math.random so LLM-mock mode is nondeterministic; seeds 21 and 7 reach 4000
ticks with built=0 because pawn construction never starts there.

## r35 — roofs implemented (model A, graded) and measured: NULL, and the experiment was inconclusive

### Shipped
Roof model A behind `AGON_P.roof`, off by default. A room's interior floor tiles get a roof
blueprint (CAMPR0); campTodo hands pawns a room's walls first and its roof only once those walls
stand; roof tiles are raised one at a time from the same material store. Roof keys are floor
keys and CAMPW0/CAMPF0 are disjoint, so wall and roof keys share one BUILT set without colliding.
The eight sites that treated "on a floor" as "under a roof" now call shelt(). Insulation scales
with how much of the room is roofed (floored at 25% once covered). Walls alone cut wind chill in
proportion to wall coverage, so building pays off before the roof goes on. Shelter-seeking aims
at a roof once any exists, else at a floor — without that the seek test and its target disagree
and a cold colonist loops.

### Gates (both passed)
- Determinism: verify_fixed.py all four cases byte-identical over 3 fresh pages, exit 0.
- **Inertness**: with the flag off, every number reproduces r34 exactly — 6/50 at 6000, 6/50 at
  9000, 10/0 on seed 21 temperate, 3/0 on seed 7 tundra. So the A/B below measures the flag, not
  the rewrite that carried it.
- Zero page errors in all 48 A/B runs.

### The A/B (24 worlds, 6000 ticks, paired, tests/noise/ab.py)
| metric | A | B | B-A | sd_d | SE | t | rho | MDE | better/worse/same |
|---|---|---|---|---|---|---|---|---|---|
| survived | 4896.96 | 5051.21 | +154.25 | 1116.39 | 227.88 | 0.68 | 0.61 | 667.69 | 10/8/6 |
| alive | 2.50 | 2.46 | -0.04 | 3.88 | 0.79 | -0.05 | 0.45 | 2.32 | 9/7/8 |
| peak | 13.46 | 13.33 | -0.12 | 0.95 | 0.19 | -0.65 | 0.71 | 0.57 | 6/6/12 |
| built | 21.21 | 27.92 | +6.71 | 29.77 | 6.08 | 1.10 | 0.46 | 17.80 | 10/7/7 |
| techs | 9.38 | 9.04 | -0.33 | 3.96 | 0.81 | -0.41 | 0.65 | 2.37 | 7/10/7 |

Extinct 13/24 in both arms. McNemar discordants: B saved 5, B killed 5 — a perfect null.
**Nothing is significant; every |t| is far below 2.07.** `peak` is the one well-measured metric
(sd_d 0.95, rho 0.71, MDE 0.57) and it is flat, so that is a real null rather than a wide error bar.

Pairing worked better than I predicted: rho 0.45-0.71, not ~0. Record this — it is the first real
estimate and it means future A/Bs on this panel can be powered honestly. sd_d for `survived` is
1116 ticks, so this panel resolves about 670 ticks; for `alive`, 2.32 colonists.

### Why the result does not mean "roofs do not matter"
**Only 1 of 24 worlds ever completed a roof.** 9 built more than A but never finished, 14 built no
more at all. The feature essentially never fired, so this A/B did not test roofs — it tested a
build queue colonists do not finish. The one world that did complete a roof went alive 4 -> 0,
which at n=1 is not evidence of anything.

The upstream blocker predates roofs and was already visible in r34: only 5 of 24 worlds finish
their *walls* by 6000 ticks, so at most 5 could even begin roofing. Roofing roughly doubles the
tile count (todo reaches 61 in some worlds), which makes the ask harder still.

Several worlds show *negative* built diffs (alpine s3 44->0, savanna s3 43->0, jungle s3 38->2).
Checked: those worlds also died much earlier in B (alpine s3 survived 3721 -> 2055), so this looks
like ordinary trajectory divergence after the arms split, not roofs suppressing wall-building.
Not proven though — if the next round wants certainty, instrument BUILDQ occupancy directly.

### Next round should NOT be more roof tuning
The bottleneck is construction throughput, not roof design. Before roofs can be evaluated at all,
colonies have to finish what they start. Candidates, in rough order of cheapness:
- raise the build horizon: re-run at 12000 ticks and see how many worlds finish walls
- cut the roof's material cost (thatch cheaper than wall material, or charge every other tile)
- revisit BUILDKEEP = 10 (the reserve may starve building in material-poor biomes)
- revisit build priority against the food-security early return
Target to measure against: worlds completing their walls by 6000 ticks, currently 5/24.

### Unchanged open items
engineFor memoization decision still owed; this._culture read but never written; mockDecide uses
Math.random so LLM-mock mode is nondeterministic.

## r36 — found the real construction blocker. Two levers measured; neither adopted yet.

Defaults unchanged: both new flags (AGON_P.keep, AGON_P.fibre) are off, and AGON_P.bdbg
instrumentation is off. Nothing shipped as default this round.

### Panel-scale determinism and inertness
The keep A/B's arm A reproduced the r34 panel on **24/24 worlds** (alive mean 2.50, built 21.21,
identical per world on alive+built+techs). That is the strongest determinism/inertness check run
so far — stronger than verify_fixed, which is only 4 cases.

### Answering the r35 candidate list
1. **Longer horizon: closed.** 12000 ticks lifts wall completion only 5/24 -> 8/24. Not the fix.
2. **BUILDKEEP: real but minor.** keep=3 moved completion 5/24 -> 8/24 (5 gained, 2 lost, exact
   p=0.45, unresolved). Survival dead flat: alive diff exactly 0.00, t=0.00, MDE 1.01 (rho 0.89,
   much tighter pairing than the roof A/B's 0.45-0.71). The r31 fear that building takes the last
   log and freezes the colony **does not materialise at keep=3** — useful negative.
3. **Build priority: NOT the blocker**, despite what one world suggested. The single-world probe
   showed 1835 buildable ticks against 237 agent-ticks of building and I nearly acted on it; the
   24-world panel says otherwise. Mean share of ticks: materials at/below reserve **41.4%**,
   nobody fed/watered 20.0%, queue empty 17.2%, buildable 17.9%, no shelter tech 3.6%.
4. Roof cost: not tested; superseded by the finding below.

### The actual blocker (design defect, not tuning)
`wallMatFor` returns **wattle** unless the colony has carpentry (-> timber) or BOTH shaping and
leverage (-> stone). `BUILDMAT` maps **both wattle and timber to wood**. So in a treeless biome a
colony must discover two techs before placing one wall tile, while holding stone and fiber it
cannot spend. Of the nine material-blocked worlds, five average **0.0-0.6 wood**, so BUILDKEEP is
irrelevant there — the resource never arrives.

Evidence: tundra s21, highlands s3, alpine s21 have **no wall material chosen at all** while
holding 12/42/25 stone and 37/49/7 fiber. temperate s21 and badlands s21 chose timber holding
0 and 1 wood. Worlds that build (temperate s3) chose stone and had both.

This also explains r35: roofs looked unevaluable because most colonies cannot build *walls*.

### Fiber-costed wattle (AGON_P.fibre) — works, but taxes technology
Wattle is woven withies, so fiber is the more honest cost. Paired A/B, 24 worlds, 6000 ticks:

| metric | A | B | B-A | t | rho | MDE |
|---|---|---|---|---|---|---|
| survived | 4896.96 | 4774.96 | -122.00 | -0.70 | 0.80 | 511.94 |
| alive | 2.50 | 2.12 | -0.38 | -0.86 | 0.78 | 1.28 |
| peak | 13.46 | 13.46 | 0.00 | 0.00 | 0.98 | 0.18 |
| built | 21.21 | 27.21 | +6.00 | 1.75 | 0.61 | 10.02 |
| techs | 9.38 | 8.54 | **-0.83** | **-1.79** | 0.90 | 1.36 |

**Never placed a tile: 8/24 -> 4/24** — halved, which is the mechanism working. Finished walls
5/24 -> 8/24. Extinct 13/24 both, McNemar 1/1.

But `techs` is the largest |t| in the table and points the wrong way (p ~ 0.09, 2 better / 6
worse). Fiber is contested between walls, clothing and tools, so building now costs technology.
Split by wood availability:

| group | n | techs | built | alive | zero-tile |
|---|---|---|---|---|---|
| wood-poor (avg < 6) | 6 | -1.50 | +15.17 | -0.17 | 5 -> 3 |
| wood-ok (avg >= 6) | 18 | -0.61 | +2.94 | -0.44 | 3 -> 1 |

I expected the tax to fall only on wood-rich worlds (unnecessary diversion) — it does not, it is
worse in the poor ones. So blanket substitution is not the answer.

### Proposal for next round: material fallback, not substitution
Make the wall cost pick what is actually to hand — spend wood when the colony has it, else fiber,
else stone where the techs allow — rather than binding wattle to one resource. That leaves the 18
wood-rich worlds exactly as they are today (no tech tax, no change) and unblocks the 5 dry ones.
People build from what is at hand; this is more realistic than either current behaviour.

Open question for Dyl: in a wood-poor world, fiber-built walls still cost about 1.5 techs. Is
shelter worth that trade, or should wall material be cheaper per tile so it competes less with
tools? That is a design call, not a measurement.

### Also noted
- temperate s3 spends 1737 agent-ticks choosing to build and places 39 tiles (44.5 agent-ticks
  per tile, against 3-6 everywhere else). Some pathological re-targeting. Not chased.
- Five worlds are blocked by food/water, not materials (volcanic s3 unfed 81%, jungle s3 73%,
  highlands s21 72%, alpine s3 72%). No material fix helps them.
- The git remote reports dylandlr/Agon has moved; pushes follow a redirect. Worth updating origin.

### Unchanged open items
engineFor memoization decision still owed; this._culture read but never written; mockDecide uses
Math.random so LLM-mock mode is nondeterministic.
