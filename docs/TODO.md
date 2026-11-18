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
