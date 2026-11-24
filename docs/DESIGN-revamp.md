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

## E. Emergence spec expansion (Dyl, 2026-11-24 03:15) — mapped against the engine

What the brief asks vs what exists:
- **Emergent tech/tool creation**: EXISTS at core — insight-driven discovery (store.techs,
  "X discovers:", lore.front), material-property crafting (MATS hard/sharp/flex), tools used
  by work (toolOf, fishHow; tilling already uses digging stick/hoe). GAPS: no UI surfaces the
  discovery tree or who knows what (U-tech panel), and tool ARRIVAL ordering is not tuned to
  survival need.
- **Four pillars first (gathering, hunting, farming, carpentry)**: partially emergent now;
  needs an early-game ordering audit — probe when each practice's first tool/tech lands per
  biome, then tune insight weights so survival pressure pulls its pillar first (hungry
  colonies discover hunting/gathering tech before ornament). One flag, panel-measured.
- **Combat ↔ weapon tools**: ALREADY SYNCED mechanically (gear.weapon reach/thrown/ranged
  drive MOVES; per-move animations fire — verified live). GAP: impact emphasis (hit flash/
  knockback) and weapon visibility in hand during moves.
- **Emergent roles**: stScore (r54) measures what someone IS by deeds; archetypes set initial
  leanings. GAP: role LABELS surfaced in UI from deeds (the hunter, the builder — epithets
  from work tallies, same PLEX pattern), not assigned.
- **Personality development from events/past/psyche**: genuinely NEW. pers[] is static after
  creation. Design: slow pers drift from lived events via the existing mind/belief system —
  surviving an attack nudges bravery (either way, by temperament), social success nudges
  sociability, loss nudges empathy/patience; bounded drift (+-0.15 lifetime), every change
  event-logged so the observer can trace a biography. One flag, panel + biography endpoint.
- **Emergent housing + UI**: the EB1-EB4 arc above; the UI half is a dwelling inspector
  (who lives here, built from what, warmth record) and chronicle lines for building events.
Sequencing: pillars audit -> pers drift -> tech-tree UI panel -> EB1. Each round keeps the
one-flag panel discipline.

## Chrome decision (r-ui-chrome-pick)

**Pick: Candidate A "Field Notes", with B's ember-amber #d8a43b as A's accent.**

Evidence: both candidates were styled live on the real app (token overrides on the Main :root
tokens + translucent-panel selectors; stylesheets in docs/chrome-candidates/). A's one stated
risk was light chrome over bright biomes. Stress test: tundra s3 at tick 1660, mid-storm — pale
snow ground plus storm desaturation, the brightest/lowest-contrast world the sim produces.
Result (scratchpad frames/chrome_A_snow.png): panel/world separation holds because the paper
panels carry their own borders and shadows rather than relying on value contrast; ink-on-paper
text contrast is biome-independent; alert stack still dominates. Weakest element is the bottom
tick bar — gets a stronger border in the full round. B's single advantage (zero contrast risk)
is closed by this test; its best idea (ember-amber) is adopted as A's accent for story beats,
headers, and the signal color. A wins on identity: a real re-direction (Dyl asked for one), and
the observer's-journal frame matches what Agon is — you watch, you don't command.

Full round scope: typography (serif label system), icons, alert styling on paper, chronicle
panel (U4) in the same language, bright-biome border pass, sweep of hardcoded-dark corners.

## Dark ground circles (Dyl asked)

They are fertile-soil patches, not shadows: the `rich` terrain decal path fills tiles where
FERTN(x,y) > FRICH (founding-camp rect excluded) with darker(SOILB[0], 0.78) — information
(where farming would take), but at 0.78 they read as orphaned shadow blobs. Differential test
(shadows toggled off, circles persist) proved they are terrain, not the shadow layer.
Fix shipped: darken factor 0.78 → 0.9, so they stay visible as soil, not shade.

## Appearance genetics audit (beards / hair / clothing styles)

What already exists (art.js `ge` genome): skin 3 additive loci; hair pigment dark-dominant with
red recessive (k4); curl incomplete-dominance (k5); baldness sex-modified (k6); 8 face measures;
beard density (k15); BEARDNAMES 8 styles; ap[] inherited in mkChild with ±0.12 noise. Clothing
is already material-true: wornParts spec() renders "tpl|mat" tinted by MATS[mat].color, so a
hide coat looks like hide. The gap is not genetics — it is **style-as-culture**: grooming and
garment style variants (which beard cut, how a coat is worn) currently never vary socially.
Proposed round: style variants as norms in store.norms, spreading via the existing homophily/
warmth graph (r55 relc), inherited household-first — measured like any other emergence round
(do styles cluster by camp/kin more than chance?), opt-out flag, 24-world panel.

## EB0 — the shelter ladder (emergence audit + program, r62)

Dyl's requirement: building must run on the agents' own thinking, learning and emergence,
with varied shelter forms, solo primitive labor, and openings (windows) as learned
technology. Audit of what is and isn't emergent today:

**Already emergent** (verified in code and measured):
- WHEN building happens: the shelter tech is discovered by doing (tinker + lived events:
  cold, storms, taking refuge, seeing rival walls — four ev routes added so a shivering
  band learns walls without a workbench).
- WHAT it is made of: wallMatFor(techs, mats) picks the vernacular material from known
  techs + what the colony actually holds; insulation physically differs by material
  (REALINS), and "the walls are rebuilt in stone" is a logged colony event.
- WHO builds and WHEN they down tools: agents pick the build goal themselves (needs,
  warmth, mats thresholds), pace is real labor (BUILDT), and now solo (BSOLO r62):
  one segment belongs to one agent; claims free on completion/orphan/death. Cooperation
  (shared segments, faster walls) is reserved as a FUTURE learned practice.
- Openings: windows are now a carpentry-level refinement — before carpentry a colony
  window tile is a crude lashed opening (dark gap behind stick bars), after it a framed
  window. (Named tech nodes 'openings'/'glazing' with their own LORE patterns are the
  proper ladder — queued as a flagged sim round since new discoverable techs shift runs.)

**Seeded, not emergent** (the honest gap):
- The floor plan. Every world generates its own house (1-3 rooms, sizes, mirror, door and
  window placement) from the WORLD SEED at generation time — variety exists, but agents
  never decide it, and experience never changes it. A colony that loses two members to
  cold builds the same plan as one that didn't.

**EB1 program — layout emergence (flagged sim rounds, in order):**
1. EB1a shack-first: before 'shelter' matures, a 1-room makeshift shack from whatever is
   held (planks/logs lean-to against rock or trees; reuses tent/windbreak warmth plumbing)
   sized by band count; the full plan only opens with carpentry+markmaking.
2. EB1b plan from need: room count/sizes derived from living colony state (heads, children,
   stores overflow -> storage room) re-planned at season boundaries, not from the seed.
3. EB1c site reasoning: plan placement/shape bends to terrain the agents know (lee of rock,
   near water-but-not-flood, cave-mouth annexes), using their explored map, not the
   generator's omniscience.
Each: opt-out flag, 24-world panel, measured against the then-current reference.
