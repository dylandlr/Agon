# Agon technology — refined design (r62 ideation session)

**How might we** make technology in Agon arrive, spread, and die the way knowledge does in
a real band of people — earned by practice, paced by understanding, carried in heads and
things — while staying legible to an observer and measurable by panel?

Decided with Dyl (2026-11-24): discovery = **patterns + insight spend**; prereqs = **hard
DAG with the gaps fixed**; shelter = **windbreak → shack → house ladder** (EB1a); loss &
spread = **investigate before choosing** a mix of death-loss / rust / explicit teaching.

## Audit that framed the session (what is true today)

- Timber walls already require carpentry (`wallMatFor`), stone requires shaping, wattle is
  the floor material; wooden furniture is carpentry's unlock. Dyl's gating instinct was
  already satisfied here.
- `shelter` is already a discovered tech ("people sleep under the sky" until then) — but
  it has **no prerequisites**: a band with no shaping and no binding can raise a house.
- **Two half-systems coexist**: the TECH table declares an insight-research economy
  (per-node costs, "Emergent mode researches them with insight") that is dead code; the
  live system is the lore pattern-matcher (do things → per-person know + lived events →
  colony derivation). `store.insight` accrues and is never consumed.
- Windows ride free inside shelter's definition; the r62 carpentry render ladder is a
  patch, not a model.

## The refined model

### 1. Discovery: reachable, then paid (r63 "TECHPAY", flag `notechpay`)

A tech **fires** when three things are true, in order:

1. **Reqs known** (hard DAG).
2. **Reachable**: its LORE patterns are met — the colony has lived the evidence
   (unchanged from today; this is the emergent heart and stays untouched).
3. **Paid**: the colony's insight pool has covered the node's cost. Insight drips into a
   reachable tech while agents keep practicing its domain; when paid, the tech fires,
   credited to the agent whose knowledge completed it.

Insight sources (revives the dead field): tinkering discoveries (novel combination found),
colony-first events, truce trades (+0.5 exists today), reading records (post-markmaking).
Pace comes from the existing per-node costs (tier 0 ≈ 6–14 … tier 3 ≈ 70–120), tuned on
2 worlds before the panel so tier-2 lands around year 2–3.

What this buys: a visible "almost there" state (the Knowledge panel's half-carved tokens
get a true fill value = paid/cost), a pacing knob that isn't a rate hack, and one coherent
system instead of a live one and a dead one.

### 2. Prerequisite fixes (r64 "DAGFIX", flag `nodagfix` — new discoverables move runs)

- `shelter.req = ['binding']` — walls are lashed and woven before they are anything else.
- `storage.req = ['binding']` — "woven baskets, lashed crates" says so itself.
- New node `openings` (tier 1, req `['shelter']`, low cost): **without it, window tiles in
  the plan render as solid wall — windows do not exist**. With it: the crude lashed
  opening (r62's barred-hole art becomes this unlock). Carpentry refines the look to the
  framed window (r62 ladder, rewired to `openings`+`carpentry`); a later `glazing`
  (tier 3, req kilning) adds glass.
- Complexity flagged honestly: the light-occlusion layers treat WINDOWS as light-passing
  statically; gating openings means occlusion must join campSig so solid-wall colonies
  are darker indoors until the tech lands. Scoped into r64, measured.
- Bedding keeps `req: []` (a pile of fur is nobody's technology).

### 3. Shelter ladder (EB1a, as specced in DESIGN-revamp EB0)

Windbreak/lean-to (binding only, brush+hide, reuses tent warmth plumbing) → `shelter`
unlocks the rough one-room shack sized to the band → carpentry+markmaking open the full
generated plan. Solo labor (BSOLO) and build duration (BUILDT) apply at every rung.

### 4. Loss & spread — investigation before commitment (Dyl: "a mix of all three?")

Questions the next measurement round answers before any of it ships:
- How often does last-carrier loss actually fire across the 24-world panel today, and is
  anything ever re-derived after being lost? (Panel script gains counters.)
- Rust prototype (sandbox flag): unpracticed, unanchored techs drain their *paid* insight
  back toward reachable — rust integrates with TECHPAY as an economy drain, never a
  binary forget. Measure whether it produces tension or just annoyance.
- Explicit teaching as part of the social arc (status r54 + warmth r55): a teach act
  copies patterns and grants insight; belongs with the relationship systems, measured
  there.
Decision gate: after the measurements, pick the mix with Dyl.

## Assumptions to validate

- [ ] Insight flow at current rates can pay tier-2 costs by year 2–3 — tune on 2 worlds
      pre-panel, else costs/sources adjust.
- [ ] Pattern-reachable + payment never deadlocks a survival-critical tech (firecraft,
      shelter) on harsh worlds — panel watches time-to-shelter per biome.
- [ ] `openings` gating doesn't swing warmth (windows are wall tiles for coverage — no
      delta expected) and the indoor-light change reads as intended, not as a bug.

## Not doing (and why)

- **Per-person-only mastery** (no colony tech list): most emergent on paper, but it
  invalidates every panel metric, the whole reference chain, and the Knowledge panel
  design in one stroke. Revisit only if the paid-pool model proves too colony-flavored.
- **Soft prerequisites**: surprising paths aren't worth an observer being unable to
  reason about why a colony knows what it knows.
- **A research-points screen**: insight is never a spendable player resource; nobody
  commands research. It is colony understanding, visible only as progress on carved
  tokens.
- **Hotfixing shelter reqs outside a flagged round**: any new gate or node changes runs;
  everything here ships through the standard flag + 24-world panel + McNemar gate.

## Round order

1. r63 TECHPAY (economy; flag `notechpay`; panel chained on then-current reference)
2. r64 DAGFIX (`shelter`/`storage` reqs + `openings` node + light scope; flag `nodagfix`)
3. EB1a shack-first (after TECHPAY so shack gates read the real economy)
4. Loss/spread measurement round → decision gate → rust and/or teaching rounds
5. Knowledge panel (redesign Phase 3) consumes paid/cost fill for its carved tokens
