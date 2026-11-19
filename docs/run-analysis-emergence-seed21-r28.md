# What one 20,000-tick run says about emergent behaviour and social strategy

2026-11-19. Seed 21, the `band` start (12 colonists, nothing built, one rival camp of 4), build
`17b4f9c` (r28). 20,000 ticks is about ten years. Source: an instrumented run capturing a
snapshot every 250 ticks, the relations chronicle, and a counter at every early return inside
`conflictThink`. The debug counters were reverted before anything shipped.

Written to separate what actually emerges from what is authored and revealed, because the
shelter system taught me that lesson the hard way: a staged reveal of a fixed plan looks exactly
like emergence from the outside.

## The method has a flaw worth fixing first

The harness polls every 50 ms until `window.__agon` exists, then pauses. The game auto-advances
a load-dependent number of ticks before that pause lands, so **runs are not reproducible across
scripts**. A first 30,000-tick run of this same seed and build went extinct at tick 15,001 with
every social counter at zero. The run analysed here troughs at 2 colonists and recovers to 7,
with 74 trades and 7 attacks. Same seed, same code.

So nothing below should be read as "what seed 21 does". It is one trajectory, and the spread
between two trajectories of one seed is the difference between extinction and recovery. The
harness now records its start tick; it should be made to pause deterministically before any of
these numbers are used to tune anything.

## The social layer is real decision-making, and it almost never runs

Every non-downed agent reaches `conflictThink` every tick. Over the run that is 198,957
agent-visits, and here is where they go:

| gate | visits | |
|---|---|---|
| entered | 198,957 | |
| asleep, or a child | 89,488 | 45% |
| no foe within `engageR` (7) | 105,542 | 53% |
| **foe in range** | **5,929** | **3.0%** |
| on cooldown | 2,802 | |
| already busy with an intent | 764 | |
| past the range checks | 361 | |
| discarded by `decideEvery` | 250 | 69% of those 361 |
| **an actual decision** | **111** | |

111 social decisions in 20,000 ticks. What happens at that point is genuine: five utilities are
scored from the agent's own personality, its anger, the power ratio of allies within 6 tiles
against the foe's, the relations score and any standing grudge, and the argmax wins. Nothing
about the outcome is written down in advance. The choices made were **parley 88, ignore 12,
attack 7, flee 4**.

The `decideEvery: 6` gate is the expensive one. Contact is already rare — 3% of visits have a
foe in range — and the gate then throws away 69% of what survives the range checks.

## Parley dominates, and it mostly fails

79% of all decisions (88 of 111) were parley. Its success is decided by the *target*, not the
speaker: `0.35 + 0.6 * ((t.pers[1] + t.pers[6]) / 2) - 0.7 * t.ang`, so an overture to a
disagreeable or angry stranger is refused. The chronicle records **8 successes against 18
rebuffs, a 31% success rate**. A success moves berries from the richer party to the poorer and
raises relations; a rebuff raises the anger of both.

The dominant social behaviour in this world is therefore a repeated, mostly-failing attempt at
contact between two groups — which is a more interesting outcome than either war or trade, and
no part of it is scripted.

## Every single attack came from the rivals

All 7 attacks were initiated by rival agents, and **6 of the 7 give the reason "defending home
ground"**. The colony never once chose to attack in 20,000 ticks.

This is a coherent emergent asymmetry rather than a bias in the code. Trespass anger (`intr`)
only accrues when a foe is within 9 tiles of your *own* home. The rival camp sits at (11,11) and
the colony at (55,16) on an 80-wide map, and colonists forage west. So the colony is structurally
the intruder, and the rivals are structurally the defenders. Nobody wrote that; the map produced it.

## Violence is real and never lethal

7 clashes, 16 hits, 3 brawls, 1 recorded "spare", and **0 kills in 20,000 ticks**. `yieldHea`
is 0.326, and the attack branch breaks off when the attacker drops below it and spares a target
that falls below it with a 60% chance. Conflict is a bruising ritual with no selection pressure
behind it.

## Relations accumulate and then decay to nothing

`rel.score` peaks at 20.03 and 17 of 81 snapshots sit above |0.5|, so the score is not dead — I
nearly concluded it was, from a 4,000-tick sampling interval that kept landing on near-zero
crossings. But `relDecay` is 0.01 per tick, which pulls the score toward zero with a half-life of
about 69 ticks: a +5 trade is worth 0.03 after 500 ticks. The `rs = rel.score / 100` term that
feeds every utility is therefore ~0 almost always.

**The two groups have no long-run memory of each other.** `trades` reaches 74 and `clashes` 7,
but those counters feed nothing back into behaviour. Alliance and feud cannot form, which is why
ten years of contact produce no trajectory — just a flat sequence of independent encounters.

There are also two separate trade paths: parley inside `conflictThink`, and a "chance meetings
can turn into trade" path elsewhere. That is why `trades` reaches 74 while the chronicle logs
only 26 parleys.

## Knowledge is genuinely held by people, and genuinely lost

This is the strongest emergent system in the build. 16 techs appeared over the run. `footwork`
was lost twice, at ticks 4,001 and 11,001. `herbalism` and `cultivation` were both acquired at
tick 17,251 and both gone by 17,751.

The mechanism earns it: `loreDerive` recomputes the whole tech set every tick from `uni`, the
union of the signatures held by *living* colonists. When the people who hold a piece of evidence
die, the tech derived from it stops being derivable. Nothing marks it as lost; it simply stops
being true.

## The lexicon stops growing after the first year

5 signs at tick 1, 12 by tick 1,001, and then nothing for the remaining 19,000 ticks. `arrow` —
the grammar sign, without which no message can put two signs in order — was coined at tick 251,
so compound sentences unlocked early and cheaply.

About 9 of the 21 glyphs are never coined at all. A sign is only coined when two adults stand
together and one of them means something the colony has no sign for, so the vocabulary tracks
novel experience. After the first year the colony's routine stops producing any, and the language
freezes. Growing it further needs either new kinds of experience or a second mechanism — compounding
signs it already has into new ones.

## The colony lives in permanent crisis, and that colours everything above

Satiation never exceeds 0.49 and sits between 0.17 and 0.35 for most of the run. Stress ran
0.86–0.99 through the first 6,000 ticks and sanity bottomed at 0.008 at tick 6,001. Population
went 12 → 15 → **2 at tick 15,001** → 7. Lifetime work totals across every colonist: gather 884,
haul 411, craft 103, fish 71, **hunt 10**.

Hunting is effectively absent — 10 hunts against 884 gathers — and food never accumulates. Every
emergent system described above is being observed in a colony that is starving, and the
population trough of 2 out of 15 is the same scenario-independent collapse measured in r27.

## What is not emergent

- **The house.** Still a staged reveal of a floorplan fixed at worldgen, on a 120-tick timer, with
  no labour and no materials. r28's notes cover this.
- **Rival camps.** Fixed at worldgen. They never grow, never rebuild, never move.
- **The chronicle's prose.** The reason strings ("defending home ground", "sees an easy target")
  are authored and selected by condition. The *selection* is emergent; the wording is not. Worth
  remembering when reading a log and feeling like you are watching a story.
- **`state.log` keeps almost nothing.** Only 5 of the last 600 entries survived, so deaths have no
  recorded cause. The chronicle is the only durable social record.

## What this changes

1. **`relDecay` 0.01 is the highest-leverage number in the social model.** At that rate the
   relationship has no memory and neither alliance nor feud can form. Either lower it by an order
   of magnitude or give the score a floor set by cumulative kills and trades, so history sticks.
2. **`decideEvery: 6` is too expensive given how rare contact is.** 3% of visits are in range and
   the gate discards 69% of those. Dropping it to 2, or bypassing it when a foe is within 3 tiles,
   would multiply social events several times over without touching the decision model.
3. **0 kills in 20,000 ticks means conflict has no stakes.** If that is deliberate, it should be
   stated; if not, `yieldHea` and the spare rule are where to look.
4. **The colony never raids.** `raid` needs `a.sat < 0.29` and a target carrying 2+ berries. With
   satiation sitting at ~0.2 for most of the run, this should have fired far more than once —
   worth checking whether rivals ever carry berries within `engageR`.
5. **The lexicon needs a second growth path** once lived experience stops being novel.
6. **Fix the starvation first.** Hunt 10 against gather 884, satiation pinned near 0.2, a trough of
   2 survivors. Every social and cultural reading here is taken from a colony in collapse, and
   fixing that will change all of these numbers.
7. **Make the harness pause deterministically** before any of this is used for tuning. Two
   trajectories of one seed differed by extinction versus recovery.
