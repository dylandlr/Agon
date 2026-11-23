# Social systems roadmap: migration, leaders, tribes, treaties

Written 2026-11-23 against dfe1358, from Dyl's list: migration, leaders, social hierarchies,
personalities, group aggregation and splitting, hostility, friendship, treaties, emergent nomadic
and sedentary tribes, groups, relations, tactics, sorting, mating, tech.

Two laws carry over from the construction arc and are not negotiable:
- **Everything emerges.** No scripted roles. A "leader" is whoever others in fact follow; a
  "treaty" is a standing of accumulated parleys; a "tribe" is a cluster in the relation graph.
  We build the mechanics that make these possible and observe whether they arise.
- **One mechanic per round**, behind a flag, byte-exact inert when off, panel A/B before any
  default flip. The 24-world panel and paired stats remain the instrument.

## What already exists (audit of dfe1358)

| Asked for | State | Where |
|---|---|---|
| Personalities | **EXISTS** — `pers[8]` per agent, read by combat deliberation (aggression, kindness, bravery, greed) and friendliness | agent init 3345; 5562, 6107 |
| Relations | **EXISTS** (pairwise) — `rel.score` with decay, trades, grudges (`foeId`), trespass/intrusion tracking, tuned by TUNE_DEFAULT | 1171, 5541+ |
| Hostility | **EXISTS** (individual) — anger model + deliberate weighing of attack/raid/defend/flee/parley with power ratio (`dom`), impulse, grudge | 5529–5572 |
| Friendship | **PARTIAL** — friendly encounters raise rel.score via personality match; no structure above the pair | 6107 |
| Parley/trade | **EXISTS** (pairwise) — talk-it-out + food gift, rel.score +5 | 5501–5507 |
| Mating | **EXISTS** — newborns from two parents: recombined genes, mutations, inherited traits, mother's tongue and house | 3361–3372 |
| Tech | **EXISTS** — emergent insight-driven techs; no diffusion between factions | store.techs |
| Groups | **PARTIAL** — factions fixed at worldgen (player colony + N rivals); `fac` is an integer, not a structure | 3466 |
| Leaders / hierarchy | **ABSENT** — `dom` is a per-encounter power ratio, not standing status |
| Group split / merge | **ABSENT** — factions never fission or fuse |
| Human migration | **ABSENT** — CAMPC is fixed for the whole run; only animals migrate |
| Nomadic vs sedentary | **ABSENT** — requires camp relocation; the "nomads" scenario is just a start condition |
| Group tactics | **ABSENT** — raids and defense are individual decisions (allyR exists as an ingredient) |
| Sorting | **ABSENT** as a mechanic — should emerge from relations + splitting (homophily), then be measured, not scripted |

## The dependency spine

Migration is the keystone: splitting needs somewhere for the splinter to GO, nomadism IS
repeated migration, and the 5 food/water-blocked panel worlds (open since r36) are exactly the
worlds a relocating colony could survive. Leadership is the second key: splits, relocations and
tactics all need "who do the others follow", and status must be derived from what agents already
do, never assigned.

## Round sequence (each = one flag, one A/B)

- **M1 — camp relocation (migration).** When food security fails for a sustained window and
  known resources within reach are exhausted, the colony re-sites camp at the best scanned spot
  (water adjacency, forage density, building material). Walls/floors are left as ruins; the camp
  blueprint re-anchors. Panel endpoints: survival/extinctions in the 5 blocked worlds,
  relocation count, time-to-extinction. Nomadism then emerges in poor biomes (many moves) and
  sedentism in rich ones (zero moves) — measure the split, don't code it.
- **M2 — standing status (leadership).** A derived status score from what already happens:
  food brought in, wounds tended, fights won, dom encounters, age/experience. High-status agents
  get follow-weight: others copy their work pick more often, and relocation targets are chosen
  by the highest-status agent's knowledge. Endpoint: status concentration (Gini over
  follow-events), plus the usual survival panel. A "leader" is then a fact in the data.
- **M3 — fission/fusion (groups, sorting).** Cluster the rel.score graph; when a colony is
  large and inter-clique relations are cold, the second-status agent leads their clique out
  (M1 relocation to a new site, new fac). Small bands with warm cross-band relations merge.
  Add a personality-similarity term to rel growth and measure assortment (sorting emerges).
  Endpoints: split/merge counts, post-split survival, assortment index vs the no-homophily arm.
- **M4 — treaties.** Faction-level standing from accumulated pairwise parleys/trades: above a
  threshold, a truce damps attack/raid bias between those facs (tech and goods can diffuse
  along the same channel — tech spread is the treaty payoff). Killings tear it (grudge
  escalation already exists). Endpoints: war/peace episode lengths, deaths to violence,
  cross-fac trades, tech diffusion events.
- **M5 — group tactics.** Raiding parties and coordinated defense: an alarm recruits nearby
  same-fac agents (allyR), target selection shared via the existing message/lexicon machinery.
  Endpoints: raid success asymmetry, defender survival, versus individual-combat baseline.
- **M6 — pair bonds (mating structure).** Mating already works; add durable pair preference
  via rel.score (high-rel pairs preferentially re-mate, share beds/food). Endpoints: pair
  stability, birth rate, infant survival. Small round.

Suggested order: M1 → M2 → M3 → M4 → M5, with M6 anywhere. M1 first because it attacks the
oldest open item (food/water-blocked worlds) while laying the keystone.

## Honesty budget

Each M-round gets the construction arc's treatment: flag, inertness, panel A/B, exact tests on
discordants, TODO entry, and a kill if it measures harmful (r44/r45 precedent). Emergence
claims get the eos standard: a "tribe", "leader" or "treaty" is claimed only from measured
structure (clusters, follow concentration, truce episodes), never from anecdote; single worlds
prove nothing in either direction.
