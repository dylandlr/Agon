# Design revamp: UI, assets, emergent building, animation
Brief from Dyl (2026-11-24): UI revamp; asset revamp (walls, doors); house/shelter building
should be emergent — from real agent innovation, planning, and structural/spatial reasoning;
look into animations. Grounded in the t=6000 desert s21 capture and the current code.

## A. UI — critique of what the capture shows, and the staged fix

What works: the world rendering has identity (rain, hearth light pool, biome palette); the
roster strip and resource key are information-dense; the map card is a sensible dock.

What fails, concretely:
1. **The alerts lie.** "4 colonists starving / Food stock low (0)" over a thriving camp —
   store.food is 0 while every pocket holds berries (the planner deliberately keeps pockets
   full). An alert that cries wolf all run teaches the observer to ignore the one that
   matters. Fix: alert on the same quantity the colonists plan with (held food per head),
   with hysteresis. (U2, small round)
2. **Rain falls indoors.** Weather streaks render over roofed interiors. Clip precipitation
   by the roof mask that already exists (ROOF set). (U3, small)
3. **Debug chrome in the player's face.** A01-A20 id pills (names exist via syncNames —
   show names, ids on a toggle), the Generator button, "Tick 6000 · paused" — dev affordances
   belong behind a dev toggle. (U3)
4. **Four competing panel systems.** Roster strip, resource list (15 rows of key-color
   squares), map card, bottom tab dock — different paddings, alignments, transparencies. One
   chrome system: unified panel style, one type scale (keep the AGON display face for
   identity; one UI face elsewhere), icons for resources instead of color squares. (U1, the
   big visual round — render candidates per the design-guide discipline before committing)
5. **The stories are invisible.** This is an observation game whose log shows 5 lines. The
   r52-r58 mechanics generate narratives (truces, feuds, defections, epitaphs) that nothing
   surfaces. A chronicle panel + story-beat toasts ("the camps swear peace") is the highest
   UX leverage in the whole brief. (U4 — pairs with the existing chron() event system)
6. Controls: Resume/speed live inside the map card; move play controls to persistent chrome.

Recommended order: U2+U3 (quick, honest wins) -> U4 chronicle -> U1 full chrome (taste round,
candidates rendered side by side for Dyl).

## B. Emergent building — the centerpiece (an arc, not a round)

Today: worldgen computes a fixed blueprint (CAMPW0/CAMPF0/CAMPK); campTodo() hands agents a
queue; agents execute. The two-room rectangle in every capture is this blueprint. Nothing
about placement is decided by an agent.

The engine already holds the spatial-reasoning primitives emergence needs: lee/windward
checks, covAt coverage, campGrp room/enclosure detection, elevation, water adjacency, the
warmth model that REWARDS enclosure+roof, ruins with weathering, and (since r55/r59) the
social graph to decide who shares a dwelling.

Staged, each stage one flag + panel vs the blueprint baseline:
- **EB1 — need-driven placement.** Kill the global queue under the flag. A cold colonist
  claims a sleep spot and builds a windbreak on its windward side; a wet one adds cover.
  Structures grow from individual needs at individual sites. Endpoints: time-to-first-shelter,
  cold/heat deaths, warmth-hours vs blueprint arm.
- **EB2 — enclosure reasoning.** Agents perceive almost-rooms (flood-fill they already have)
  and close gaps; completing an enclosure is the plan, not following a drawing. Walls meet
  walls because closing is valuable, not because a blueprint says so.
- **EB3 — innovation and diffusion.** Structure ideas enter the insight/tech system (first
  hearth-room, doors, windows, multi-room). Agents copy configurations they have SEEN work
  (the fog/knowledge system carries this) — camp architecture can then diverge by colony and
  by what its members lived through. This is where "real innovation" lives.
- **EB4 — households.** Warmth bonds (r55) and pair bonds (r59) decide who builds and sleeps
  together: houses become social units, and the r55 clustering question gets its substrate.
Risks, named: this retires campTodo/applyCamp/campSig as the construction driver (the largest
refactor since r52a); build-order economics (BUILDKEEP) and roof coverage assume rooms; panel
construction metrics must switch to outcome metrics (warmth-hours, deaths) because walls/built
counts stop meaning one thing. Blueprint path stays as the flag-off arm throughout.

## C. Assets (serves B)
Organic layouts need walls that read at any shape: corner/junction autotiling (what TILESX/
wallMatFor support today is rectangles), per-material identity at a glance (wattle lashings,
timber grain, stone coursing), door orientation variants, and ruin states that look ruined
(weathering age exists in s.ruins; render should show it). Render candidates as SVG montages
against the real tiles before adopting any (design-guide step 5).

## D. Animation
Exists today: pfx particle bursts (eat/drink), EMBERS, bub thought-bubbles, weather streaks,
an fps overlay for budget checks. Missing, in value order:
1. **Movement tweening** — agents hop tile-to-tile instantly; interpolating position is the
   single highest-value motion change in the game.
2. Work loops: build/gather/craft gestures (even 2-frame).
3. Hearth flicker + light-pool breathing; water shimmer.
4. Story-beat emphasis: a slow zoom/flash on chronicle events (ties to U4).
Budget rule: every motion round proves itself on the fps overlay before shipping.

## Recommended sequence
1. U2+U3 (honest alerts, indoor rain, names) — one small round, immediate quality.
2. EB1 probe + flag — the centerpiece starts; C's autotiling rides along as needed.
3. U4 chronicle — makes every shipped social mechanic visible.
4. D1 movement tween; then U1 chrome with rendered candidates; then EB2-EB4.
