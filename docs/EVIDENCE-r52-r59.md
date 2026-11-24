# Evidence dossier: the r52–r59 arc (shipped build = walls · roofs · order · workwarm · shade · mig · cool · pers2 · fission · treaty)

Every ship/park decision below cites its instrument. Panels: fixed 24-world set (12 biomes ×
seeds {3,21}), paired per-seed differences, |t| ≥ 2.069 significant at df=23, byte-exact state
hash as the inertness gate. Raw arm data: /tmp/mar/ab_*.json (per-world rows incl. hashes).

## Ships, with panel statistics

| Round | Mechanic | alive Δ (t) | Extinctions | Decisive evidence |
|---|---|---|---|---|
| r52 | migration (distress-triggered camp move) | +0.17 (0.46) | 7=7 | kill gate passed; fired 4/24; forest s21 +8 at 12k horizon persists |
| r53 | cooling at water | **+4.42 (3.72)** | **7→1** | largest win in project history; saved all 7 blocked worlds; survives Bonferroni over 19 panels (p≈0.0011 < 0.0026) |
| r56 | personality → behavior+language | −0.46 (−0.55) | 1=1 | clean null; 258–446 voiced lines/world; stable per-agent character |
| r57 | fission (signed grudges + defection) | −0.21 (−1.23) | 1=1 | behavior identical in 22/24 worlds; cost = 5 legible lives where feuds exist |
| r58 | treaties (truce ≥60, torn ≤−20) | **+0.00 exactly** | 1=1 | desert +11 cancels temperate pair −5s; built +9.7 (t=1.57); all secondaries positive |

## Parks, with the measurement that killed each

| Candidate | Killed by |
|---|---|
| r54 status-drift | one-directional negative lean (techs t=−1.92); clustering was scripted, not emergent |
| r55 affil-drift | clustering did not emerge (dispersion unchanged, topShare 0.19–0.32); work displaced (built desert 103→62) |
| r51 fresh (prior session, upheld) | extinctions 7→10; false beliefs load-bearing for sedentism |
| M5 group-tactics round | unnecessary: 92% of fight-ticks already multi-agent (probe, 4 worlds) |
| r59 pair bonds | dormant: births never run outside the sandbox (s.births unset) — built, inert, awaiting the births call |

## Emergence measurements (claims from structure, never anecdote)

- **Leadership (r54):** follow events 1.4–3.6k/world; concentration topShare 0.48–0.95; leaders are the top providers (gather 51–96, haul 45–90); succession free (jungle s3 cycled 8 leaders).
- **Homophily sorting (r55):** bonded pairs more personality-similar than the population in 4/4 worlds — simBonded 0.44–0.51 vs simAll 0.30–0.43 — from a purely local accrual rule.
- **Stress-gated feuds (r57 probe):** desert s21: 11 cold edges (coldest −1.51) at t=3000 healing to 2 by t=6000; every fed world: zero.
- **War/peace episodes (r58):** grassland s3 cycles swear(1621)/break(3117)/swear(3412)/break(5084)/swear(5724); wetland s3 swears at t=242 and holds 5,759 ticks; desert s21 swears at t=298 → 5→16 alive.

## The desert s21 chain — one world, every mechanic, both endings

pers2 greed-hoarding drains the larder → hunger → impulse attacks (3 fatal wounds the callous
don't tend) → grudges (11 cold edges) → fission defections (waves at t=1937/3137, exiles die
at Redhollow) → colony 16→9→5. **With treaties (shipped): truce sworn at t=298 defuses the
spiral before it starts → 16 alive, trades 368→1316.** Both endings emerge from identical
seeds; only the mechanic set differs. Diagnostics: nopers2 arm = 16 alive, zero deaths.

## Audit corrections made along the way (the roadmap's table vs reality)

1. "Relations EXISTS (pairwise)" → was one global colony↔rivals scalar; pairwise built in r55.
2. "Fission trigger: local resources exhausted" → ecologically true (jungle food within 12
   collapses to 1–2), epistemically unusable (the belief map never learns it).
3. "Group tactics ABSENT" → 92% of fight time was already multi-agent via allyR joins.
4. "Mating EXISTS" → code exists, dormant (sandbox-only toggle).

## Bugs the discipline caught (all mine, all recorded)

TDZ crashes in the advance preamble ×2 (store r52b, log r57) · gather-distance trigger
survivorship-starved (0/48 windows) · relc prune floor above one accrual step (bonds
guillotined each tick) · similarity kernel flattened by 8-dim averaging · literal flag read
forking from the const at ship flip (caught by 8-way inertness) · s.log is a 5-entry window
(event counts from it are worthless) · panel 'walls'/'built' ruin-contaminated post-migration.
