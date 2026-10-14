# lifesim: emergent early societies

Phase 1 implementation of the project design: a deterministic JAX world, a
single agent learning to survive with recurrent PPO, and an event-sourced
recorder with exact rewind.

## Layout

```
lifesim/
  config.py     EnvConfig: world, ecology, body and reward parameters
  genotype.py   Heritable traits: sampling, recombination, mutation
  env.py        Pure reset/step/observe/action_mask, simultaneous resolution
  recorder.py   Event log, snapshots, action log, exact rewind
  policy.py     GRU actor-critic with action masking
  ppo.py        Phase 1 recurrent PPO: one agent per world, shared network
  ppo_multi.py  Phase 2 multi-agent PPO: one network per individual
  baselines.py  Random policy and a scripted forager (global-knowledge reference)
scripts/
  train_phase1.py   Train single-agent survival
  train_phase2.py   Train a population of individuals sharing worlds (--resume)
  benchmark.py      Steady-state throughput and SU estimate for a target run
  continuous_world.py  One world never reset, lives continuing (--learn, --map-seed, --archetypes)
  experiment.py     The thesis experiments, gate and report
  hierarchy.py      Dominance, linearity, steepness, stability, leadership, rank outcomes
  culture.py        Between-world differences, traditions, persistence, dialects
hprc/               Grace setup, benchmark, single-run and E0 array job scripts
  evaluate.py       Random vs scripted vs trained survival + record an episode
tests/
  test_env.py   Determinism, masking, contests, conservation, rewind, hunting,
                predators, violence, disease, cold, weather, trauma, huddling
```

## Quick start

```bash
pip install -e ".[dev]"          # add "jax[cuda12]" on a GPU machine
pytest -q
python scripts/train_phase1.py --timesteps 4000000 --out runs/phase1
python scripts/evaluate.py --params runs/phase1/params.msgpack
```

## What exists now

The world is 64 x 64 by default. Every mechanic is a dial in `EnvConfig`, so
experiments change configuration, never code.

**Environment.** Terrain (grass, forest, rock, water) from smooth noise.
Rock costs more to cross and can cause falls and injuries. Plants regrow by
biome, season and weather. Markov weather alternates clear and rain spells
with extreme events (drought, cold snaps, heat waves) at a rate set by
`climate_volatility`, the H3 dial. Ambient temperature follows season,
day/night and weather; forest buffers it toward comfort.

**Physiology.** Satiety, hydration, energy, health, body temperature
(relaxing toward ambient at a rate set by the insulation gene), injury,
disease, aging. Heat raises thirst through sweating; drought raises thirst
directly. Injury slows movement, can make moves fail, weakens strikes, and
bleeds when severe. Disease (from raw meat, water, or contact with sick
neighbors) raises hunger, drains energy and damages health, then leaves
immunity that wanes. Strength declines with age after 60% of lifespan.
Resting and huddling with neighbors keep agents warmer.

**Psychology.** Stress rises with damage, nearby predators and critical
needs, blocks healing when high, and costs energy. Trauma shifts the
pain-aversion phenotype away from the genotype, by an amount set by the
temperament-drift gene, then slowly recovers. Agents feel their own stress,
temperature, injury and sickness; they never observe their genes.

**Animals.** Small prey flee and can be killed by one strike. Large prey
have armor: damage below a threshold per tick does nothing, so killing one
takes several hunters striking in the same tick. Predators chase agents in
range, attack lone agents, are deterred when the target has a neighbor, and
rest after attacking. Kills drop meat, which feeds more than plants, spoils,
and can carry disease. Animal populations regrow with plant growth.

**Actions.** See Materials and fire for the current 42 generic actions.
Strikes hit whatever occupies the target cell, with priority predator, large
prey, small prey, agent. Striking another agent is possible and is recorded
as violence. Invalid actions are masked.

**Simultaneous resolution.** All agents act in the same tick. Contested
gathering has one strength-weighted random winner per cell; damage from
simultaneous strikes on a cell adds up, which is what makes group hunting
work. Agents are never processed in index order.

**Recorder.** Events include gathers, contests, eating, drinking, strikes,
kills by animal type, predator attacks, violence, falls, infections,
recoveries, weather changes, and deaths with cause and causal parent.
`rewind(rec, t)` replays logged actions from the nearest snapshot exactly.

**Balance (64 worlds, one agent each, one year).** Random agents are all dead
by about t=1000, mostly of thirst. The scripted forager, which sees the whole
map, keeps 61% alive to year end; winter cold is its main killer. That is
intended: winter without fire, shelter or company is the hardest test for a
solitary agent, and it is the pressure that should make grouping valuable.

## Results so far

Held-out evaluation, 128 worlds never seen in training, one agent each,
flat (MLP) encoder after 5M steps:

| Policy | Alive t=500 | t=1000 | Full year |
| --- | --- | --- | --- |
| Random | 3% | 0% | 0% |
| Scripted (sees whole map) | 95% | 84% | 52% |
| Trained (local view) | 81% | 60% | 27% |

**Finding: over-drinking.** The trained agent drinks about every 6 ticks.
Drinking when hydrated has no immediate cost, but each drink carries a small
infection risk, and illness arrives long after the drink. The agent cannot
connect them, which plausibly explains its disease deaths (24 vs 8 for the
scripted forager). This is a credit-assignment gap that a transmitted hygiene
norm could close, so it is kept as a finding rather than patched.

**Encoders.** `--encoder conv` reads the 11 x 11 view as an image with a
3-layer convolutional encoder; `--encoder mlp` flattens it. On CPU the
convolution runs as patch extraction plus matmul, because XLA's native
convolution gradients ran over 100 times slower there.

## Phase 2: social world

**New primitives (33 actions total).** Release drops one unit of carried
food (meat first) on the current cell; any adjacent agent is recorded as
having seen a gift. Touch (8 directions) gives both parties a small reward
scaled by their social-reward gene. Four tokens with no built-in meaning:
emitting one costs the tick, and listeners see which token each nearby agent
emitted on the previous tick. Strikes on agents existed already.

**Appearance.** A 3-number heritable look per agent, visible to others, so
kin can resemble each other once reproduction exists (Phase 3).

**Relationship memory.** Each agent keeps a decayed record about every other
agent: struck me, gave food next to me, touched me, and seen being violent to
someone else (third-party witnessing, the raw material for reputation). It is
memory capacity, not a norm: nothing in the world reacts to it except other
agents' learned policies.

**Social observation.** In each visible cell holding another agent, the
observer sees its own memory of that agent, the agent's appearance, and the
token it emitted last tick. 22 channels over an 11 x 11 view.

**Per-agent learning (ppo_multi.py).** A population of N individuals, each
with a fixed genotype, an appearance and its own network, lives in B replica
worlds in parallel: one individual, one brain, B times the experience. Each
agent's update is clipped separately and scaled by its own learning rate
(lr0 x the lr_scale plasticity gene). `--trait-gamma` makes each agent's
discount its patience gene; it is off by default because myopic agents simply
die until evolution exists to select against them. Dead agents' steps are
masked out of learning.

**Running E0 and the scarcity dial.**

```bash
python scripts/train_phase2.py --agents 8 --worlds 16 --ticks 400000 --out runs/e0_base
python scripts/train_phase2.py --agents 8 --worlds 16 --ticks 400000 --scarcity 0.4 --out runs/e0_scarce
```

Compare `per_1000_agent_ticks.strikes_on_agents` over training. The
prediction (H1) is more violence under scarcity. Untrained agents strike each
other at random, so only the trend after learning counts.

**Speed warning.** About 1,000 agent-steps per second on one CPU core.
Meaningful Phase 2 runs need tens of millions of agent-steps: use a GPU.

## What agents perceive

Every agent sees only an 11 x 11 window around itself by day (5 cells each
way; about 12% of a 32 x 32 map) and 2 cells at night except where firelight
reaches. Beyond that it knows only its own body, the time, the weather, and the
messages of the 3 nearest speakers. Learning respects the same limits:
imitation uses only what the observer could see, and witnessing violence needs
light. The playback and the scripted forager see everything; they are tools for
researchers, not agents.

## Time, daylight and darkness

One tick is 30 minutes: 48 ticks per day. By default a year is compressed
to 40 days (10-day seasons, 1,920 ticks, one episode); `days_per_year=365`
gives a real year of 17,520 ticks with seasons of 92, 91, 91 and 91 days.
Seasons always change at midnight, and the physics and the clock share the
same boundaries. `env.clock(t, cfg)` gives the calendar, e.g.
`Y1 Summer D3 06:30`, and every logged event carries it.

**Why compress by default.** Physiology, weather and growth are per tick, so
they are the same either way; only the seasons stretch. With a real year:
- Preparing for winter means acting on consequences up to ~9,000 ticks
  away. With discount 0.997 the effective horizon is ~330 ticks (7 days), so
  winter is invisible to the returns; even 0.9999 struggles at that range.
- Each training run sees 9 times fewer winters for the same compute.
- Lifespans (max_age 20,000 ticks, about 417 days) would cover barely one
  year, not a lifetime of seasons.
A real year is the right test once multi-horizon value heads, culture or
evolution can carry seasonal knowledge; it is a stretch condition, not the
training default.

- **Daylight** follows the season: 12 hours in spring and autumn, 15 at
  midsummer, 9 at midwinter; solar noon is 12:00.
- **Temperature** peaks at 15:00 and bottoms out at 03:00.
- **Darkness:** at night agents see food, animals and other agents only within
  2 cells, or where a fire's light reaches (2 cells around it). Terrain and the
  map edge stay visible. Messages are heard in the dark, so speech and fire
  both gain value at night without any rule saying so.

**Simultaneous events.** Everything in a tick happens at once: every agent
acts, speaks, and the world updates. The world resolves a tick in a fixed
order (weather, movement, strikes, animals, gathering, eating, social, fire,
bodies, deaths), and each logged event carries its `phase`; same-tick events
are listed in that order, which is not an order in real time. An agent still
takes one action plus one message per tick, and every action lasts one tick;
parallel and multi-tick actions are a stretch goal.

**Balance under this calendar** (scripted forager, 64 worlds): 97% alive at the
end of spring, 91% summer, 86% autumn, 69% at year end.

## Materials and fire: physics, not technology

The world has physics; the agents have no technology. Nothing in the code
mentions making fire, cooking or torches. Those exist only as sequences of
generic actions that agents may or may not discover.

**Materials** (plant, raw meat, cooked meat, wood, stone) differ only in
physical properties: food value, pathogen load, hardness, fuel value. Agents
see each as a separate observation channel and are never told what any of
them is for. Forest cells hold dead wood; rock cells hold unlimited stone.

**Generic actions (46 total).** Select a material into the hand (5); grasp,
eat, release and place (8 directions) all act on whatever is in hand; strikes
use the hand item as a tool.

**Large prey use damage reduction:** damage taken = total^2 / (total + armor).
One bare-handed hunter wounds a large prey animal slowly (~0.26 a tick); two
striking together do ~0.72. As of the latest runs agents still kill no large
prey: they kill small prey but have not learned to eat meat, which takes three
steps (select raw meat, pick up, eat).

**Physics rules.**
- A hard hand item makes strikes stronger (stone: x1.5).
- Hard struck on hard (stone in hand on rock ground or a loose stone) can
  spark; a spark on a cell with fuel (wood, dry plants) can ignite it, more
  easily in dry weather.
- Lightning starts fires in summer, drought and heat waves (about 2.7 per
  world-year); fire spreads to neighbors with fuel and burns down as fuel runs
  out; rain slows it.
- Heat within two cells warms bodies, cooks raw meat lying there (cooked meat
  is more nourishing and carries no infection), and keeps predators from
  attacking. Standing in flames burns. Animals caught in flames leave cooked
  carcasses. Cooked meat left in flames chars away.
- Grasping wood while standing on a burning cell yields a burning branch that
  burns for 60 ticks; placing or dropping it lights fuel where it lands.

**The ladder agents could climb, none of it coded:** use natural fire (warmth,
carcasses) -> feed it wood -> place raw meat by it -> carry a burning branch
-> strike stone on rock beside wood to make fire.

**Chance baseline** (`scripts/fire_baseline.py --immortal`): random agents
kept alive for a year in 16 worlds, 256,000 agent-ticks: near a fire's heat
0.5% of the time; fed a fire once and pulled a burning branch once; never
cooked deliberately, ate cooked meat or started a fire. Learned rates must be
compared with these. Wildfires stay bounded: on average 4% of forest burns per
year, at most 26% in one world.

The multi-agent trainer logs each ladder step per 1,000 agent-ticks.

## Social learning

Agents learn from watching each other (`lifesim/social.py`). At each step an
agent may watch one visible neighbor and learn to predict that neighbor's
action from the neighbor's own observation. This imitation loss is added to
PPO, scaled by the observer's heritable social-learning gene and
`--imitation` (0 turns it off).

- **Who is watched** (`--imitation-bias`): any visible agent (`uniform`),
  healthy-looking agents (`success`, default), or similar-looking agents (`kin`).
- **What is copied:** by default only actions that turned out better than the
  demonstrator expected (positive advantage), weighted by how much better.
  `--imitate-all` copies everything watched. Observers cannot really see
  advantages; this stands in for seeing that an action paid off.
- **Only what the observer could see:** the demonstrator's observation is cut
  down to cells within the observer's own sight (day or night radius), and the
  demonstrator's private state (hunger, inventory, what it hears) is blanked;
  only time and weather, which everyone perceives, remain.
- **No memory transfer:** the watched action is predicted with the hidden
  state reset, since the observer did not live the demonstrator's history.
- Cost: one extra forward pass per agent per sample in the update.

This is the channel through which a discovery such as fire can spread. Fire
left burning is the only other way.

## Language and knowledge transfer

Agents speak on a channel separate from their actions. Every tick each agent
chooses an action (42, all generic) and a message: 3 symbols from a 16-symbol
vocabulary, symbol 0 = silence. No symbol has a built-in meaning; speaking
costs 0.0005 energy so noise is not free. Each listener hears the 3 nearest
speakers in view: their symbols, where they stand, and its own memory of them.
`env.step(state, actions, cfg, messages)`; omit `messages` for silence.

Social learning depends on language in three ways:
- **Speech is copied with actions.** Imitation predicts the watched agent's
  action and its message, so words can spread culturally.
- **Speaking draws attention.** An agent that speaks while acting is more
  likely to be watched (`speech_attention`).
- **Speaking makes a demonstration count more** (`teach_bonus`). Knowledge
  flows faster through agents who talk while doing, so teaching can emerge
  without being coded.

PPO treats the action and the message as one joint choice.

**Speech attention and the teaching bonus are off by default.** In a matched
pair of runs (614k agent-steps each), with them on, speech climbed from 136 to
309 per 1,000 agent-ticks; with them off it stayed at 106 to 133, with similar
survival, violence and gifts. Watching and copying talkers more made talking
spread on its own. Use `--speech-attention 2 --teach-bonus 1` to study that
effect deliberately.

## The core experiment: group reward, counters, comparison

- **Prosociality weight alpha** (`--alpha`): each agent trains on
  (1 - alpha) x its own reward + alpha x the group's mean reward. 0 is selfish,
  1 a fully shared team reward.
- **Counters** logged per window: predator attacks deterred by a neighbor
  (`deterred_per_1000`), the share of winter-night ticks spent next to another
  agent (`winter_huddle_share`), large-prey kills, gifts, strikes, and
  inequality of hunting credit across agents (`hunting_gini`, the free-riding
  check for H2).
- **Comparison:** `python scripts/compare_runs.py runs/core/*` averages each run's
  final 20% of windows and reports mean and 95% interval per (alpha, food)
  condition across seeds.

## Baselines in the same setting

`python scripts/eval_baselines.py --agents 8 --grid 32 [--scarcity 0.4]` runs the
random policy and the scripted forager (full map knowledge, one per agent) for
whole years in exactly the training setting, and reports the training log's
measures: lifespan, share alive at episode end, worlds reaching year end, global
reward per world-day, deaths by cause. Results go to `runs/baselines/`.

Training logs the same two headline measures for every window:
`share_alive_at_episode_end` and `global_reward_per_world_day` (all agents' own
rewards before alpha mixing, so it compares fairly across alpha).

## Reward modes

`--reward-mode acts` (default) is the original design: the base reward pays for
staying alive, avoiding pain, novelty and touch, and the moral sentiments react
to acts (with habituation).

`--reward-mode survival` makes survival the only reward, felt through each
agent's dispositions:
- **Own survival:** +0.01 per tick alive, -1 on dying, plus the change in felt
  well-being, `gamma * W(after) - W(before)`, where W comes from the body
  (health weighted by the pain-aversion trait, food, water, energy, warmth).
  Because these changes telescope, any loop of actions that ends where it
  started earns nothing (potential-based shaping, Ng, Harada & Russell 1999).
- **Others' survival:** for agents it can see, the same signal weighted by
  empathy, kin care (by similarity), reciprocity (toward those who fed it) and,
  negatively, vengefulness (toward those who struck it). Envy and guilt compare
  well-being and only cost reward.
- **Nothing rewards an act:** touch, novelty, giving and striking pay only
  through what they do to someone's survival.
- **Others are felt through their real changes only** (harm, recovery, death),
  as plain differences that sum to the other's net change in well-being.
  Merely having someone alive nearby earns nothing.
- **Others are felt through impressions.** Each agent keeps an impression of how
  every other agent looked when last seen: wounds, injuries, sickness and
  thinness (`visible_condition`; hunger, thirst and stress are internal and
  cannot be seen). Impressions update only on sight (day view, or night radius
  and fire-lit cells), and agents feel changes to them, weighted by
  disposition. Harm out of sight is felt on the next sighting if it left marks;
  harm healed unseen is never felt; a death is felt on seeing the body, which
  is visible in a new observation channel (`CH_BODIES`). Envy and guilt compare
  impressions with how one looks oneself. Two earlier designs failed in
  training: feeling only visible agents let agents look away from suffering,
  and feeling everyone regardless of sight flooded each agent with others'
  fortunes it could not influence. Impressions are also where hearsay and
  noticing others' moods can later plug in.

**Scars and lasting injuries.** Wounds (blows, predator attacks, burns, falls;
a fall on rock costs 0.05 health and adds 0.1 injury) leave scars, 0.5 per unit
of health lost (about 0.07 per bare-handed blow), that
never fade and do no harm but are visible to others. Injury above 0.5 leaves a
lasting part (half the excess, at most 0.35): injury heals down to that floor,
so weaker strikes and costlier, riskier movement stay for life, but a lasting
injury never reaches the bleeding level. Both appear in observations (one's own
and others') and in how others look, so a wound that healed out of sight is
still noticed through its scars.

**Height (2.5D), cliffs and falls.** Every cell has a height in [0, 1] from
the elevation map that places water (lowest) and rock (highest). Agents see
ground height relative to their own (`CH_HEIGHT`) and each cell's steepness
(`CH_CLIFF`), both visible at night like other terrain shape.
- **Climbing** costs energy: `move energy x (1 + 2 x rise / steepest drop)`;
  descending costs nothing extra.
- **Altitude** is colder: up to 6 C at the highest point.
- **Steepness** (`cliff`) is each cell's drop to its lowest neighbor relative to
  the map's steepest drop, squared, since danger rises sharply with steepness.
  Ordinary ground averages about 0.1; rock edges are the steep ones.
- **Falls** can happen on any move: `fall_prob x hazard x (1 + injury)`,
  doubled in the dark, where `hazard = 0.2 on rock + cliff`. Flat ground never
  trips; gentle slopes cause rare, minor stumbles (about 0.2% of moves).
  Severity has a long tail, `hazard x 0.35 x an exponential draw` of health,
  plus 1.5x that in injury. Over 4,000 forced falls onto rock each:

| Fall from | Killed outright | Lasting injury | Injury above 0.3 (days to heal) | Minor |
| --- | --- | --- | --- | --- |
| Flat rock | 0% | 0.8% | 5% | 61% |
| Medium edge (0.5) | 1.4% | 25% | 44% | 24% |
| Steepest edge (1.0) | 8.9% | 41% | 59% | 16% |

Badly injured survivors can still bleed out later. Falls are their own cause of
death ("fall") and scar by health lost, like any wound.

**Company acts on the body, in both modes.** Loneliness raises stress by
0.004 per tick x the sociability trait x 0.5^(companions in view), so the first
companion matters most; company also speeds recovery from stress. Healing
slows as stress rises and stops at 0.7. An average agent alone settles near
stress 0.4 and heals at about 43% speed; isolation alone never kills. Calm is
part of felt well-being.

`shaping_gamma` must equal the training discount; `train_phase2.py` sets it.

## Running the experiments (Colab or HPRC)

`scripts/experiment.py` runs the thesis experiments with the "society" preset
(`lifesim/presets.py`: 16 slots, 8 founders, reproduction, survival reward, 32 x 32 world,
192-tick rollouts, caveman slate, living world, world model), identically on any machine.
The "tribe" preset (64 slots, 24 founders, 48 x 48) is for culture at a larger scale, at about
4.5x the compute.

- `validate`: the preset; without sleep-consolidated learning; the blank slate; without the world
  model; each followed by a 10-year continuous world.
- `inheritance`: after `validate`, the caveman run's agents live on with learning, newborns
  inheriting a parent's learned brain or its inborn one (see below).
- `core`: reward sharing alpha {0, 0.5, 1} x food {1.0, 0.4} x seeds {0, 1, 2}, 100M agent-steps
  each (the first A100 run plateaued around 60-100M).
- `archetypes`: retraining with one archetype among the founders (`--which`, default tyrant,saint).
- `placed`: one archetype among the validated agents in a continuous world where they keep learning
  (`--which`, default all 17; 2 seeds each and a control with none).
- `culture`: replicate worlds on one map with different histories, with (`social`) and without
  (`solitary`) learning by watching (`--replicates`, default 4 each); see "Hierarchy and culture".
- `gate`: did anyone born in the validated caveman agents' continuous world grow up (15 days)?
  `core`, `archetypes`, `inheritance` and `stress` refuse to start until it passes
  (`--skip-gate` overrides).
- `report`: training results by condition and how each continuous world went.

Everything resumes. Every run records a code version (a hash of `lifesim/*.py`,
`scripts/train_phase2.py` and `scripts/continuous_world.py`); `--resume` refuses to continue a run
with different code, and `experiment.py` stops before doing anything if a run it would reuse
was made by other code. Keep one `--root` per code version: the notebook and the HPRC jobs use
`runs/exp_<code version>`.

**Training worlds run for 10 years** (`--episode-years`, preset 10), or until everyone in them
has died, and a world cut off by the clock is bootstrapped from the value of the state it was
cut off in, not treated as death (`ppo_multi.gae`; Pardo et al., 2018, "Time limits in
reinforcement learning"). Before this, every training world restarted after one year and the
learner treated the restart as the end of everything: nothing after day 40 was worth anything,
so neither surviving the end of winter nor a child who would pay off later counted. On the A100
(code db0d7c52adc0) the scripted band persisted (3 of 3 seeds, about 40 births in 160 days), but
the trained agents of the one run made with that code (`caveman_no_world_model`) mated 0 times in
8 lives in their continuous world and died out by day 55. `scripts/benchmark.py
--preset society --worlds 32 64 128 256` finds the fastest world count that fits.
HPRC: `hprc/README.md`. Colab: `agon_eos_colab_a100.ipynb`.

## Viability: can this world be lived in at all?

`scripts/viability.py` runs the experiment's world with hand-scripted agents that do what a
competent forager band would: flee predators when alone and stand together otherwise, drink and
eat before needs run low, sleep at night, take cover and huddle when cold or hot, mothers stay
with nursing children, fed couples mate. No fire, tools, hunting or sharing. `--knowledge full`
routes from the true map (an upper bound); `--knowledge own` uses only the caveman slate's given
skills and what each agent has seen, which is what trained agents can do. If such a band cannot
keep its numbers up, no training can, and the world is what needs fixing.

It found three faults that made every continuous world die out whatever agents learned (all code
up to and including version 228e12e6f030):

| Fault | Effect | Fix |
|---|---|---|
| Children could gain weight only as fat (0.01 kg a tick) | every child born grew up at ~70% of ideal weight, below the 85% needed to conceive; once the founders passed menopause (10-27 days in) no one could have children | lean mass grows with the frame while fed (a starving child is stunted) |
| Heat damage from a body state equal to holding 28 C; resting added warmth even in heat | resting in shade killed in an ordinary summer | damage from ~35 C; curling up warms only in the cold |
| Illness 200 ticks, 0.004 health a tick, no healing while sick | 0.8 health per illness: fatal to anyone not at full health | 96 ticks (~0.4 for a healthy adult) |

Before: every scripted band died out within 18-82 days. After: 3 of 3 lasted 4
years with both `full` and `own` knowledge, and at food 0.4; the 64-slot tribe also lasted. Nearly
all deaths are now from old age, and numbers sit at the slot limit: neither food at 0.4 nor the
seasons bind a competent band, so the earlier "scarcity made no difference" result is expected.
What remains hard is learning that competence.

## Minerals, ores and metals

15 materials beyond plants, meat, wood and stone (`lifesim/minerals.py`), each a physical thing
with properties, never a recipe. They lie in deposits where geology puts them (flint, pyrite,
obsidian, coal, copper, tin and iron ore in rock; clay and bog iron in marsh; salt and ochre in
desert), are finite, and are picked up, carried, held, dropped, placed and struck like anything else.

| Property | What it does, through existing mechanics |
|---|---|
| hardness | a blow with it in hand lands harder; hard on hard can spark |
| sharpness (flint, obsidian, metals) | blows land harder still; gathering meat or wood with an edge in hand takes up to 2x |
| spark (pyrite, iron) | struck against something hard (flint), sparks are ~5x as likely: the classic fire kit |
| fuel (coal) | burns 4x as long as wood and leaves the fire coal-hot |
| heat | clay in any fire becomes fired clay; ores in a coal-hot fire become copper, tin, iron (slower); copper and tin heated together become bronze |
| preserve (salt) | carried with meat, slows its spoiling by 75% |

Ochre (a pigment) and fired clay (a vessel) do nothing yet: there is no way to make a mark or to
carry water. Agents never see names: each cell's minerals are seen as how much lies there and
its colour and lustre (5 observation channels), and they feel what they carry. The playback draws
minerals as pebbles in their own colour, lists them per tile with what each does, and logs
"heat made copper at ..." as an event caused by the fire (and whoever fed it).
`EnvConfig.mineral_abundance` scales every deposit (0 = none).

This version also fixes a bug in social learning: what an observer learns from watching
should keep only what everyone perceives (time, weather) from the demonstrator's body vector,
but the slice was computed for an older, shorter body vector, so since the sleep features and
conditions were added it kept part of the demonstrator's private state (conditions, inventory)
and dropped the time and weather.

## Animals and bugs

Each animal group now holds several species (`lifesim/fauna.py`), each with the body and habits
of its kind, all through mechanics rather than scripts:

| Group | Species and what they do |
|---|---|
| small prey | rabbit (flees fast); fowl (runs two cells, half of blows miss); fish (in water, struck from the bank; most blows miss without a point); tortoise (slow, shelled) |
| large prey | deer (bounds two cells, loose herds); aurochs (herds; struck, may trample); boar (struck, often gores); horse (herds, runs two cells, wary); mammoth (huge, herds, tramples; feeds a band for days) |
| predators | wolf (night hunter in packs: each extra wolf beside the prey cancels one defender); big cat (ambush: notices late, strikes hard, not deterred); bear (attacks only when struck or startled close by, mauls back, hibernates in winter); snake (barely moves, venomous bite); crocodile (lies in water and takes whoever stands at the edge) |

Predators also hunt animals: any takes small prey on its cell, and wolves, big cats, bears and
crocodiles take large prey (never a mammoth), leaving a carcass for scavengers. A slot's species is
fixed, so each world keeps the same mix (slot 0 of each group is the classic rabbit, deer and
wolf, which take the group's config values). 6 predators per 1,000 cells (was 1) so that each
kind is present: on a 32 x 32 map, 2 wolves, 2 big cats, a bear and a crocodile. The scripted
competent band still persists with them (viability.py: 2 of 2, 3 years).

Bugs are fields: mosquitoes (marsh and water edges in warm seasons, worse at night, driven off by
smoke) carry fever; flies on meat and bodies speed rot and sickness; bees keep hives of honey in
woods and sting whoever robs them unless a fire's smoke calms them; termite mounds in grass and
desert hold grubs that a stick (wood in hand) draws out four times as well as fingers. Honey
(dense food, never spoils) and grubs are new materials.

Agents see animals as they look, not by name: per cell the size of the biggest animal, whether it
has fangs or tusks, and its colour; plus mosquitoes, flies, hives and mounds (7 observation
channels). The playback draws each species in its own colours, lists them per tile with their
habits, and logs kills, attacks, goring, stings and predators' catches by species.

**Food web view.** The "Food web" toggle (beside "Social web") draws who has fed on whom so far in
the recording: food at the bottom, prey in the middle, people and predators at the top, arrows
from food to eater with width by how often, coloured by kind (hunted, gathered, attacks on
people, prey that fought back), circles sized by how many are alive. Hover a line or a circle for
counts and habits. On the map, rings mark the last day's kills, catches and attacks. A diagram,
not a map overlay, because feeding is rare and scattered in space; the rings keep the "where".

## Technology tree and skills by personality

`lifesim/techtree.py` names 25 building blocks of know-how that can be observed in the event log
(foraging, hunting small then large then big game, fishing, standing together, killing a predator,
striking with a tool, using an edge, carrying and keeping fire, fire from fire, fire by striking,
a fire kit, cooking, salting meat, sharing food, robbing a hive, smoking the bees, fishing for
termites with a stick, firing clay, a coal fire, smelting, bronze, metal tools), each with its
physical prerequisites (no smelting without a coal-hot fire). It is a measuring lens: agents are
never shown it, and nothing in the world gates on it. A block is discovered the first time anyone
does it, established once several individuals or days do it, lost if it then goes unused for 40
days. The recorder logs what was in hand for blows, gathering, fire feeding and ignition, so tool
use is observable.

`scripts/tech_tree.py RUN` prints the tree (first by whom and when, spread, status) and relates
each skill to inborn personality across lives: Spearman correlation of each life's rate of each
block with each temperament trait, permutation p-values and Benjamini-Hochberg correction, then
the strongest correlations whether or not they pass. Personality is never given a skill; any
association has to emerge, and correlation is not cause (traits can go with longer lives or with a
lineage). A continuous world writes `techtree.txt` and `techtree.json` at its end, linked from
`index.html`; the inspector shows each individual's own skill tree.

## Mutations

At every birth each trait mutates with probability 1% (`EnvConfig.mutation_rate`), jumping 2.5
population standard deviations up or down (`mutation_size`), on top of the ordinary variation
between siblings. Inborn conditions that neither parent carried are new mutations too. Both are
logged as a "born with a mutation" event (and in `story.txt`), recorded in `lives.json`, and
tagged in the inspector's trait list.

## Looks, and the size of a tile

Eye colour (dark brown to blue) and hair texture (straight, wavy, curly, tightly coiled) follow
the heritable appearance genes, so kin resemble each other. Hair is natural: nothing in this world
cuts, ties or styles it, so it grows with age, and adult men are bearded. Both show in the
inspector and on the figure.

A tile's size is nominal, derived from rules the world already has: one move per 30-minute tick at
4 km/h makes a cell about 2 km across, and the altitude cooling (6 C over the map's relief) at
6.5 C per km makes the relief about 920 m. The tile inspector shows length x width x the steepest
drop to a neighbour, the ground's height above the lowest point, and an enlarged render of the
tile. Close-range rules (striking or huddling with someone in the next cell) do not fit a 2 km
cell; no rule uses the scale.

## Threat knobs and injected shocks

Baseline knobs for a whole run (`train_phase2.py`, recorded in history.json and used by every
script that rebuilds the run's world): `--scarcity` (x plant abundance), `--predators` (x
predator numbers and respawning; 0 = a harmless predator), `--disease` (x every chance of
infection), `--hazards` (x extreme weather, lightning and falls). `export_playback.py` and
`viability.py` take `--food --predators --disease --hazards` too.

Shocks inject a crisis into a continuous world (or viability.py): `--shock KIND:DAY:DAYS:STRENGTH`,
repeatable. `famine:40:10:0.2` cuts standing plants and regrowth to 0.2x for 10 days from day 40;
`plague:40:10:10` makes infection 10x as likely; `predators:40:10:3` makes predators 3x as
aggressive; `cold:40:10:10` / `heat:40:10:10` shift the air 10 C; `drought:40:10:2` doubles thirst
and stops rain. Agents are told nothing: they meet a shock as any hard season. A shock is a world
event in the log, and the hardships it makes likelier (hunger in a famine, infection in a plague,
attacks, freezing, overheating, thirst) name it as a cause, so `story.txt` shows what each shock
did. `experiment.py stress` runs one world per shock against the unshocked control and reports
numbers before, at the worst and 40 days after, and the deaths during the shock.

**Several agents can share a cell.** Nothing stops it: a cell is a patch of ground about one
30-minute walk across, and huddling, groups standing against predators and contact infection all
count neighbours within one cell. There is no crowding cost.

## Map overlays (playback)

Like a colony sim's data views, the playback can dim the ground and paint one quantity over it.
Use the Overlay buttons, or press O to cycle through them (Shift+O goes back). Hover a cell for its value.

| Overlay | What it shows |
| --- | --- |
| Temperature | air temperature where one stands, including a fire's warmth (from `air_map`) |
| Food | plants, meat, honey and grubs |
| Minerals | deposits, coloured by mineral |
| Danger | how far each predator notices, mosquito density, and the last day's deaths and attacks |
| Elevation | height, contour lines and cliffs |
| Trails | where agents stood in the last day |
| What one knows | the selected agent's own map of the world |

While an overlay is on, clouds and night shading are left off.

**Look.** The map is drawn top-down in the manner of colony sims such as RimWorld (the shapes are our
own): flat colours, a dark ink outline and a soft shadow on every figure. The ground uses muted colours,
blended at the borders between kinds of ground, so that living things, fire and danger stand out.
Trees, boulders, cave mouths and reeds belong to the ground. People have a rounded body, a round head and
natural hair from their genes, with a coloured ring at the feet showing who they are (kin share colours).
People on the map are small top-down pawns built from the same body numbers as the inspector's
figure (`bodyLook`). Height, build, weight and strength set their size and breadth. Skin, hair colour,
hair texture and length, eye colour and an adult man's beard come from the same genes and years.
Pregnancy, scars, a lasting injury and what is in hand show as well. So a person looks like their
inspector figure without being a copy of it, and children are smaller because they are shorter. The
pixel look's sprites follow the same rules. Each species of animal has its own silhouette. Food plants show as leafy
patches that bear berries when full. The sidebar and the inspector use the same figures. **Look: Painted / Pixel** under the map switches the whole page between this style and the earlier
pixel-art one (map, figures, fonts). The choice is remembered in this browser. Overlays only display data: they change nothing in the
simulation or the code version.

## Playback sources: random, untrained, trained

`scripts/export_playback.py` records a playback from one of three kinds of agent:
`--random` (no brain: uniform over possible actions; a statistical baseline, no decision
records), `--decisions` (untrained brains: freshly initialized networks, with a world model
unless `--no-world-model`, that really decide, so the Decisions and Ideas tabs show what an
unlearned mind computes; the baseline for a trained run's records) and `--run DIR` (a trained
run). `--slate caveman` and `--world living` apply to the first two. In the inspector, the
**Tile** button opens the tile the agent stands on, and **Back** returns to the agent. A tile
lists the sounds and words said on it and within earshot (each symbol drawn as the agents' own
3x3 mark, with who heard it) and which words this place hears more than anywhere else.

## Cause and effect

Every event the recorder logs carries its causes (`parents`, with a plain-words `because`),
taken from the simulation's mechanics rather than guessed from timing: a blow causes the
victim's wound; striking back links to the blow being answered (within 2 days); a death links
to what its cause names (the last blow for violence, the infection for disease, the slide into
thirst or hunger for those, a mother's death for an orphan whose hunger began after it); an
infection to the sick neighbour, the water or the raw meat it came from; taking food to whoever
left it; a birth to the pregnancy and the mating; lameness to the wound that never healed,
chronic cough to the illness survived, "haunted" to the brush with death; fire use to the fire
used; freezing or overheating to a cold snap or heat wave when there was one. The body crossing
into danger (starving, parched, freezing, overheating) is logged as its own event so needs
deaths have a beginning.

A continuous world logs every event (recorded chunk or not) to `events.jsonl.gz` and, at the
end, writes `story.txt` (linked from `index.html`). `scripts/causal_story.py` tells any run,
playback page or event log as stories, each a chain of causes and effects, with branches where
one cause had several effects:

    python scripts/causal_story.py runs/exp/validate/caveman --agent 5 --from-day 40 --to-day 80

    Story 8: Y1 Spring D2 08:00 to Y1 Spring D5 05:00; 8 events; #3, #6
    Y1 Spring D2 08:00 #6 struck #3 -> (#6's blow) 08:00 #3 was struck by #6 -> ...
      ↳ after Y1 Spring D2 08:00 #3 was struck by #6: -> (striking back at #6) Y1 Spring D4 07:30 #3 struck #6 -> ...

It ends with every death traced back to its first recorded cause. Routine stories (a single
blow and nothing after, thirst then a drink) are left out unless `--all`. A reused slot's later
lives are written #5.2, #5.3. The playback inspector's event lists show each event's causes too.

## World model and ideas

With `--world-model` (on in the presets), each agent's network has a model of its own world: its
belief state (the GRU's memory) is predicted forward by a dynamics network given an action, and
read as the reward it expects and its own state (food, water, energy, health, body temperature,
stress, sleepiness). It is trained only on the agent's own experience, and like all learning only
once slept on: it can imagine only what its life has taught it.

Before each choice the agent imagines its 6 likeliest actions 6 ticks ahead (after the first step
it follows its habits among the actions possible now), scores each by expected rewards plus the
value of where it ends up, and its choice leans toward the best-looking future by a learned
`trust` that starts at 0. Trust grows only if acting on ideas pays off (and turns negative if they
mislead). Nothing about what is a good idea is given.

The playback inspector has an **Ideas** tab: the best imagined future (the action, what it would
do after, and the change it expects in its own body), every option with its imagined value, how
much its ideas moved its choice, and a list of new ideas (a best imagined action its habits would
clearly not have chosen) and whether it acted on them. Training logs `world_model`: errors
predicting its own state, beliefs and rewards, and mean trust. `--no-world-model` is the ablation.
Cost: see `scripts/benchmark.py --preset society` vs `--no-world-model`.

## History with learning: Lamarckian or Darwinian

`continuous_world.py --learn` keeps every individual learning from its own life in the
continuous world, with the trainer's own update (PPO, watching others, the world model, sleep
consolidation) every `--learn-every` ticks (96 = 2 days). Training is then the species' deep past
(the slot networks are what ancestral lives taught each lineage); the continuous world is history.

`--inherit` sets a newborn's brain, from one parent (mother or father at random) plus small
mutations: `parent` (Lamarckian: the parent's brain now, with all it learned) or `innate`
(Darwinian: the brain the parent was born with). Under `innate`, what a parent learns can pass on
only by being watched. A newborn's optimiser state starts fresh, and buffered experience from
the slot's previous occupant is dropped (that brain is gone). `experiment.py inheritance` compares
the two, 3 seeds each; births.json records which brain each child got.

## Slates, given skills and rare altruism

`--slate` sets what agents start with:

- `blank`: primitive actions only (move, grasp, eat, drink, strike, touch, ...).
- `caveman` (the "society" preset): plus four given survival skills, chosen like any action:
  `seek_water` (walk to water the agent remembers, then drink), `seek_food` (walk to remembered
  plants, gather, eat; never raw meat), `flee` (step away from the nearest predator in sight) and
  `seek_shelter` (walk to a forest or cave it has seen, then rest there: cover from cold and heat).
  Each tick a skill picks one step from the agent's own cognitive map and sight, never the true
  map, routing around water and away from steep ground; with nothing known it explores unseen
  ground. When to use them, and whether to drop them for someone else, stays the agent's choice:
  everything social is learned.
- A "civilized" slate (human knowledge through an LLM) is not part of Agon; see Paragon.

**Rare selflessness:** a temperament that weighs others' visible fortunes like one's own, kin
or not, so such an agent can give its last food or its life for a stranger. Each founder or
newborn has it with probability `--saint-rate` (default one in a million); it is not inherited.
At that rate a whole training run may contain no such agent; raise the rate to study it.

**Exploration:** `--ent-coef` (default 0.005) sets how strongly learning keeps choices varied.

**Social web (playback):** the "Social web" toggle draws the relationships agents already have,
read from their own memories and true kinship (it changes nothing in the world): gold dashed for
kin, green for bonds (touch, food left near each other), red for blows, with groups, average
steps between group members, the most connected agent and who strikes most without being struck.

## Archetypes and emergent roles

**Injected archetypes** (`--archetypes tyrant:1,saint:1`, profiles in `lifesim/archetypes.py`;
`archetypes.table()` lists them) set the inborn traits of the first founders, each meant to test
one question:

| Archetype | Temperament | Tests |
| --- | --- | --- |
| tyrant | strong, vengeful, no empathy or guilt, envious | does a remorseless founder come to dominate? |
| saint | selfless, empathetic, forgiving | does selflessness survive and spread? |
| outcast | unsociable, pain-averse, curious | does a loner stay one? |
| trickster | very curious, no guilt, reciprocal | does curiosity without guilt pay? |
| matriarch | a woman: kin care, empathy, resilience | does a carer anchor a lineage? |
| chief | strong, very sociable, far-sighted, firm | do others follow her or him (leadership)? |
| sage | fast learner, from others above all, long-lived | does a social learner become a store of know-how? |
| conformist | copies others, seeks company, avoids the new | does copying spread what is already done? |
| innovator | curious and plastic, ignores others | does individual learning find what copying misses? |
| warrior | big, fast, fearless, retaliates | does physical dominance become rank? |
| nurturer | cares for young and hurt, either sex | does care raise more children to adulthood? |
| hoarder | envious, no guilt or gratitude, frugal | does envy without guilt pay, or isolate? |
| coward | dreads pain, never retaliates, weak | is avoiding every fight viable? |
| enforcer | answers in kind both ways, feels unfairness | does a punisher of harm make others gentler? |
| peacemaker | empathetic, never retaliates, sociable | does a non-retaliating carer lower violence? |
| charmer | strong drive for intimacy and company | does it change who has children? |
| wanderer | restless explorer, fast, hardy, solitary | does an explorer find new ground, and do others follow? |

They set feelings and capacities, never behaviour. Traits the learner reads (learning rate,
social learning, discount) are set on the trainer's population too. Two interventions:
`experiment.py archetypes --which tyrant,saint` retrains with the archetype among the founders
(3 seeds each; control: core `a0.0_f1.0_s*`), about an A100-hour each; `experiment.py placed
--which all` places one archetype among the validated caveman agents in a continuous world where
they keep learning (2 seeds each, control: the same world with none), far cheaper. The report
shows where each archetype stood: dominance, leadership, children.

**Emergent roles:** the continuous world records every life (`lives.json`: who it was, how long
it lived, how it died, how often it did each of 21 behaviours). `scripts/archetypes.py` groups
lives by what they did (k-means on standardised rates), names each group after its most
distinctive behaviours (fighter, giver, forager, carer, loner...), shows each group's average
temperament, and where each injected archetype actually landed. The experiment report runs it.

**Inspecting agents (playback):** click an agent on the map or in the list. The inspector shows,
as of the tick being viewed: who it is (sex, age, generation, mother, archetype, what it is doing
and saying), its body (health, size, strength, needs, sleepiness, temperature, injury, lasting
injury, scars, sickness, stress, pregnancy, what it holds and carries), every inborn trait with
its range, its relationships right now (kin, bonds, blows each way) and its history in this
recording. A character viewer switches between **Anatomy** (a figure coloured by condition;
hover or tap a part, brain, eyes, mouth, immune system, heart and lungs, stomach, womb, arms,
legs or skin, to see what the body tracks there) and **Loadout** (what it looks like, what is in
its hand, a burning branch, and what it wears: nothing yet, as the world has no clothing), with
an **Inventory** of what it carries out of its capacity. The parts show the body's real state
(needs, health, sight, strength, scars and so on); injury itself is still one whole-body value,
not placed on parts.

**Decisions (inspector history):** for trained agents the playback records, every tick, what the
network actually computed: the likelihood it gave its top options, what the chosen action leaned
on most (gradient x input of the chosen action, summed over 13 groups of what it perceives:
the land, food in sight, others, animals, predators, fire and materials, the dead, its own needs,
its own condition, what it carries, time and weather, voices, its memory of places), its
expectation of its future (the value estimate, and whether it is rising or falling) and what it
felt (the reward, and the part felt for others). "Now" shows the decision in motion; the list
shows earlier ones. There is no inner monologue: these are the real quantities behind each choice.

**Social web cards:** hovering a line shows what lies between the two agents: what kind of tie it
is (kin, bonded, rivals, one-sided blows), kinship, bond, blows each way, distance, both
temperaments and anything notable (fighting despite close kinship, a strongly vengeful striker).
The inspector heading is the agent's number until the agents themselves give it a name; mother,
children and relationships link to those agents' own records.

**Links everywhere:** every agent mentioned in the playback (event list, histories, tile and
agent views) is a blue link with the agent's own colour swatch that opens its record; places
are purple links that open the tile inspector. **Trait cards:** hovering a trait shows what it
means, how it acts on this agent right now (for example strength now = inborn x age x
condition, with the numbers), and how heritable it is. **Tile inspector:** click an empty tile
(or shift-click any tile, or a place link): ground, height, steepness, water beside it; what is
on it now (plants, meat, wood, fire, prey, predators, agents); what each living agent remembers
of it (how fresh the memory is, and how much food it believes is there against what actually
is), from snapshots every 6 hours of recording; and its history (who stood there, what happened
there).

**Measuring survival:** training's lifespan counts only agents who died within a one-year
episode, so it is capped and leans toward the unlucky; prefer the share alive at year end and
how long the continuous world lasts. Deaths are also logged by season (`deaths_by_season`;
the report's `winter` column), since winter was the first A100 run's main killer.

## Hierarchy and culture: measured, never given

Agents are never told about rank or custom; these analyses read them from what happened. The
continuous world writes `trace.npz` (per tick: positions, alive, actions, speech, and whether a
predator, large prey or fire was in view of each speaker) beside `lives.json` and the events.

**Hierarchy** (`scripts/hierarchy.py <world>`):
- *Dominance* comes from decided encounters: blows struck, and supplants (one moves next to
  another, who moves away at once while the first stays).
- *Per 40-day window:*
  - normalised David's scores (de Vries et al. 2006);
  - steepness;
  - Landau's linearity h', corrected for unknown and tied relations, with a randomisation p (de Vries 1995);
  - despotism: the share of blows struck by the top individual.
- *Across windows:* rank stability.
- *Rank against outcomes:* eating, matings, children, food taken from others' leavings, and
  lifespan, by Spearman correlation with a permutation p.
- *Leadership:* who moves first and is followed by those within 3 cells. It is set against
  dominance and time lived, because rank by force and leadership can come apart.

**Culture** (`scripts/culture.py <worlds> --against <control worlds>`):
- *Design.* The `culture` experiment gives every world the same map and founders, with
  different histories and innate inheritance, so learning cannot pass down through the brain.
  A control has no learning by watching.
- *Differentiation.* PERMANOVA (Anderson 2001) on each life's behaviour profile, with worlds as
  groups. It reports pseudo-F, R², a permutation p, and the same test with inherited traits
  regressed out ("beyond genes"). It also reports how often a life's world can be told from its
  behaviour.
- *Traditions.* Behaviours customary (at least half of lives) in some worlds and absent
  (at most 5%) in others (Whiten et al. 1999).
- *Persistence.* Does a world's distinctive profile among its earlier lives recur among those
  born later? Own world against other worlds, with a permutation p.
- *Dialects.* Message divergence between worlds beyond a world's own drift between its first
  and second half (Jensen-Shannon, at equal sample sizes), vocabulary overlap, and each world's
  most typical message near a predator, large prey or fire.

A candidate culture is one that differs between social worlds, persists past generations, holds
beyond genes, and is weaker or absent in the solitary control.

## Attachment: survival felt toward particular others

Not a reward for love: the felt form of a survival fact. Agent i's attachment to j (0 to 1)
grows with evidence that j keeps i alive: food j left that i took when hungry, warmth huddled beside j when
cold, a predator attack held off with j beside, calm time together in view, intimacy, a living
child they share, and milk (an infant toward its mother). It falls by 30% with each blow from j
and fades by half over 20 days apart. It acts only through survival:

- j's visible condition counts in i's reward (care weight + attachment), through the same
  impressions empathy and kin care use: j's wounds and death are felt as i's loss (grief).
- An attached companion in view calms i more; attached ones out of sight leave i less secure
  (a little stress). Stress feeds healing, fertility and well-being.
- Agents see their own attachment to each visible agent (one more social feature).

How fast that evidence counts depends on fit, which is also survival: a partner of like
temperament (empathy, vengefulness, reciprocity, kin care, sociability, novelty-seeking) is
predictable, and a fit partner (healthy, well-fed, strong) is worth more to stay beside. Evidence is
scaled by (0.5 + 0.5 x similarity) x (0.5 + 0.5 x the other's fitness). Two kinds of evidence count
whatever the fit: danger faced together (both in view of each other with a predator near) and
teamwork (striking the same animal in the same tick). Shared emotion enters here, as shared
threat; there is no separate emotion-matching term.

So mating, staying together and defending one another can come to follow particular partners,
without any rule about them. The playback's social web draws attachment in pink (thicker =
stronger), its hover cards give it both ways, and the inspector lists it per relationship.

## Conditions: traits good and bad, inborn or acquired, with rarities

17 discrete conditions (lifesim/conditions.py) on top of the continuous genome. Each acts only on
a physical quantity the world already has; none prescribes behaviour.

| Kind | Conditions (base chance) | How it arrives |
|---|---|---|
| inborn | keen eyes 3%, poor eyesight 5%, robust 4%, frail 4%, iron stomach 3%, long-lived 2%, slow clotting 1%, cold-hardy 4%, infertile 2% | at birth; a child of a parent who has one gets it with its inherit chance (40-50%; infertile 0%), else the base chance |
| age | stiff joints (past 45, 3%/yr), failing sight (past 50, 4%/yr) | a chance each tick, scaled to the yearly rate |
| event | lame, chronic cough (10%), haunted (30%), hardened (20%) | lasting injury above 15%; surviving an illness; health falling through 15%; 48+ ticks of critical need survived |
| use | enduring, heavy-handed | 1,500 moves walked; 200 blows struck |

Chronic cough and haunted can fade (2% and 1% a day). Rarity labels: common 10%+, uncommon 4%+,
rare 1.5%+, very rare below. Agents observe their own conditions. The playback lists them in the
inspector (hover for what each does, how common it is, how it arrived), mentions them on the body
part they act on, and logs each one as an event; lives.json and the continuous world's index.html
record every life's conditions and the age each arrived.

## The living world (biomes, weather, climate)

`--world living` (the society preset's default; `classic` keeps the old world exactly):

- **Biomes** from elevation, moisture and a north-south gradient (north and high ground colder):
  grass, forest, rock and water as before, plus **desert** (6 C hotter, little food, thirst 1.5x),
  **tundra** (6 C colder, sparse food, slow going), **marsh** (rich food, drinkable where you
  stand but twice the disease of open water, slow going) and **caves** in rock outcrops (air held
  near comfort, no rain or snow inside, no food).
- **Dynamic weather:** clouds form over water and marsh (more in summer), drift with a wind that
  shifts direction, and rain where thick (about 4% of the land at a time, somewhere a third of
  the time); fronts sweep in. Rain is local: it waters plants, damps fire, chills whoever is out
  in it and can be drunk. Below freezing it falls as snow, which lies, slows walking, chills and
  stops growth until it melts. Lightning strikes under storm clouds.
- **Climate:** a 3-year warm-cool cycle (+-3 C) plus random drift, added to every temperature;
  warm phases bring more droughts and heat waves.
- Agents perceive rain and snow in their view (two new observation channels).
- The playback draws all of it: biome colours (legend under the map), cloud shadows, falling
  rain and snow, lying snow; the clock shows temperature, wind and the climate phase; the tile
  inspector describes each biome and the weather on that tile.

**Full playback:** `scripts/full_playback.py --root runs/exp` finds the longest-lived continuous
world among finished runs and lives it again with the same seed, recording every 5-day chunk
into `<run>/full_playback/` (open `index.html`; every page links to the previous and next five
days). The Colab notebook's step 9 plays it in the notebook.

**The downed and grazing** (on in the living world; `--no-downed`, `--no-grazing` for ablations;
both off in the base config):

- **Downed** (`EnvConfig.downed`, RimWorld's): past health 0.2 or injury 0.8 one cannot stand. One
  lies where one fell and can only rest, sleep, call out, drink if water is beside one, and pick up and
  eat what is on one's own cell. Others see them lying as they see a sleeper, and decide whether to
  bring food (drop it on their cell), keep watch, or finish them; predators find them easy. Events
  "downed" (caused by the last wound) and "stood_up".
- **Grazing** (`EnvConfig.grazing`): rabbits, fowl, tortoises, deer, aurochs, boar, horses and
  mammoths eat the plants where they stand (shared when a cell is crowded; a large grazer eats about
  seven times what a small one does) and drift toward the richest ground beside them; one that finds
  too little loses condition and can starve. Empty slots refill faster the better fed the rest of its
  kind are, with a trickle of newcomers when none are (`immigration`). **Predators get hungry:** a kill
  on their cell or a carcass sates them for a while; a hunter long unfed weakens and dies; predator
  births follow how much prey is about. People and herds now compete for the same plants, and a
  crash in prey is a lean time for wolves. RimWorld's wild animals do not breed at all (they wander in
  at the map edge); this is closer to a real food web.

## Caves: the underground layer

`--world living --underground` (training; `EnvConfig.underground`, off by default, so every
earlier design is unchanged). The cave mouths of the living world open onto a second layer.

- **Geography.** From each mouth, tunnels wind under the land (never under open water) and open
  into ragged chambers, some holding a pool. Each cave system's **formation** comes from the land
  at its mouth: **karst** (limestone, wet, many pools, stalagmites; a quarter of its chambers are
  quartz **geodes**, now and then with gold; **sinkholes** open extra mouths onto grass or forest),
  **lava tube** on the highest ground (long straight tubes; basalt, obsidian, sulfur; warm air;
  **hot springs** bubble up beside it: warm, drinkable, sulfur at the edge), **ice cave** in the cold
  north (frozen pools, air near freezing all year), **salt cave** in desert (salt, gypsum, no water).
- **New substances** (only underground, so the surface is as before): limestone (burns to
  quicklime in coal heat), quartz, sulfur (burns), gypsum, saltpetre (from bat guano; cures meat
  as salt does), galena (even a campfire smelts it to lead), lead, gold, basalt. 25 minerals now,
  32 materials, so 76 actions.
- **Delve** (action 75): on a mouth, go down or come back up. An infant on its mother's cell goes
  with her. Below, one walks only along passages.
- **Darkness.** Below it is always dark: one feels the cells next to one and sees what a burning
  branch lights (3 cells, anyone's). Awake in the dark, fear builds (stress) and steps go astray
  (35%: a random way; into rock is no step) or fail (25%). A torch removes all of it. Setting a
  burning branch down below puts it out: no fire underground.
- **Life below.** The air holds near the season's mean pulled most of the way to the formation's
  own (comfort; warmer in lava tubes; near freezing in ice caves); no rain, snow, plants or weather.
  Only pools give water. The walls hold the formation's minerals and unlimited stone. Whatever is
  dropped, placed or killed below lies on the cave floor (meat rots at half the rate in the cool),
  and food left there can still be a gift.
- **Two layers, kept apart.** Agents above and below do not see, hear, touch, strike, mate with,
  infect, imitate or warm each other. Animals live on both: they notice the other layer only by
  scent within 2 cells, chasers follow a quarry down a mouth, fleeing prey may bolt down, **bats**
  roost below by day and fly at night (leaving saltpetre), bears den below in winter.
- Agents observe the cave instead of the surface while below, and one more own feature says
  they are underground (obs 7,066 at the default sizes, with the larger inventory).
- **Measured:** events `went_underground` (with whether a torch was carried) and `came_up`; every
  event underground is tagged `below`. Tech-tree blocks *Going underground*, *Lighting the way*,
  *Mining*, *Burning lime*. Traces store `below`, and the hierarchy analysis keeps the layers apart.
- **Playback:** **Layer: Surface / Caves** under the map (key C). The cave map draws the tunnels by
  formation, pools, mouths in daylight, ore and crystal in the walls, what lies on the floor, the
  animals and agents below, torchlight, and a **?** over anyone groping in the dark. On the surface
  map someone underground shows as a dashed ring. Selecting an agent switches to its layer, and the
  view follows it down and up.

## Time scale: 1-minute ticks (RimWorld-like)

RimWorld does not run in real time: at 1x its clock runs about 86 times faster than real time (60
ticks a second, 1.44 game seconds each; an hour in ~42 s, a day in 16 min 40 s). It feels real-time
because each act is short and animated at a human pace. To follow social decisions the same way, the
presets now use **1-minute ticks** (`tick_minutes=1`; `--tick-minutes` on the trainer, the continuous
world, the exporter, viability and the benchmark; 30 is the calibration).

- `config.with_tick_minutes` keeps every rule the same per unit of **game time**: amounts per tick
  (hunger, thirst, healing, sleep pressure, regrowth, spoilage, stress, rewards...) scale with the
  tick, chances per tick become 1 - (1 - p)^k, per-tick decays become d^k, and durations counted in
  ticks (illness, torches, fuel, memory and attachment half-lives, weather spells, episodes, lifespans)
  stretch. Per **act** nothing changes: a blow, a bite, a drink, a step, a mating is the same act, so a
  finer tick means more and shorter acts, as in a colony sim. Hard-coded rates in `env.py` go through
  `_k(cfg)`; the recorder's causal windows and every analysis read the world's own day length.
- **Space** follows from one step per tick at walking pace: a cell is 67 m at 1-minute ticks, so a
  32 x 32 map is a ~2 km village area, not a region. View radius 5 is ~335 m.
- **Learning:** gamma, gae_lambda, the patience gene and consolidation are given per 30 minutes and
  converted (`MAConfig.tick_scale`). A day (1,440 steps) no longer fits in a rollout, so sleep
  consolidation switches to a **pending change** (`consolidation="auto"` picks it whenever ticks are shorter than 30 minutes): each update computes the
  day's gradient at the unchanged weights (weights only change in sleep, so it is exact), adds it to a
  per-agent pending change that fades with a half-life of a day, and applies the share earned by deep
  sleep (or a death). One step per sleep, not several PPO epochs. The long-term store works as before.
- **Costs, bluntly:** a run of a given number of steps lives 30 times fewer game days (a 100M-step
  society run: ~135 days per world instead of ~4,000; about two lifetimes). The continuous world takes
  30 times more steps per day; its chunk playbacks keep the first day of each chunk (`--playback-days`).
  On a fixed grid of smaller cells food, water, others and predators are all closer in game time, so the
  ecology needs re-checking (`scripts/viability.py --tick-minutes 1`). `--tick-minutes 2` halves the cost.

## Playback: pacing, animation and colony-sim UI

- **Speeds.** With short ticks the speeds are RimWorld's: 1x = 1.44 game minutes a second (a day in
  16 min 40 s), 3x, 6x and 15x (keys 1-4, or the buttons on the map). With 30-minute ticks they are
  1, 3, 6 and 15 ticks a second (RimWorld's rate would be one decision every 21 s).
- **Animation.** People glide from cell to cell over each tick and bob as they step; a blow lunges
  toward its target and a touch leans; the struck flash red; floating notes rise over people as things
  happen (+food, +water, gift, hit!, bitten!, ♥, born, †, ▼, ▲...); night falls and lifts over an
  hour with a warm tint at dusk and dawn, fires and torches lighting pools; names under people; corner
  brackets on the one selected.
- **UI.** A portrait bar above the map (health and calm bars, pulsing red when someone is in danger;
  click to inspect); letters down the map's side for what matters (deaths, births, attacks, blows,
  conception, gifts, fire, going underground; click to jump there); a clock and speed controls on the
  map. **Slow on danger** drops to 1x at a death, a predator attack, a blow or a birth.
- **Pixel look**, studied from Mudborne: 16-pixel tiles drawn at exactly twice their size (the map
  is 32 px a cell in both looks); everything - people, animals, trees, rocks, the edges of patches and
  banks - outlined in one ink so it reads at a glance (a lifted slate teal `#203036`, as Mudborne's, never
  black: black made everything harsh); grass and woods as a dense weave of
  short blades in close tones; trees as tall pillars with a light lid and a ragged hem (now and then a
  round crown, ferns at the foot, berries in the crowns when the woods bear food); bare ground flat with
  faint brickwork, the grass hanging over its edge in ink-tipped drips; dark deep water with reeds and
  sparkles, a bank face and foam where land lies north of it; people with round heads and rosy cheeks;
  a cream ring round the one selected; night graded toward violet. The painted look keeps its inked style.
  **Colour discipline** (measured from Mudborne's own tileset): the ground is muted (saturation about
  0.15-0.25, grass and woods leaning blue-green, bare ground leaning mauve); water is the one saturated
  surface (turquoise); strong colour is kept for the living and for food. **Shadows** of everything that
  moves go to one surface, opaque, laid on the ground once at 22% in the ink colour, so overlapping
  shadows never darken each other.
- **Standing in the ground** (Mudborne's ground-aware sprites): wading in marsh shows a band of murky
  water over the shins with ripples when moving; deep snow a band of snow over the feet, deeper with
  depth; each footfall throws a puff of dust (grass, sand, rock, frozen ground), a splash (marsh) or
  snow; footprints stay in snow for about two game hours. Animals likewise. **Cadence:** every ambient
  tile (swaying grass, water sparkles) shares one world frame at 1 Hz in the order rest, lean, rest,
  lean the other way; people breathe (a half-pixel rise) at under 1 Hz; only fire and torches flicker fast.
- **The ground answers back.** People and animals stand *in* the grass: a few blades are drawn over
  their feet, swaying with the world's frame and pushed the way they walk. Tall grass lies flat where
  someone stands; a bush shakes when someone brushes through its cell. Footprints stay in snow (about 2
  game hours), wet sand (1 hour) and marsh mud (1.5 hours, filling with water), set along the way they
  went. Ground that keeps being trodden (people, and large animals at half weight, over the last two
  game days; about six hours of treading wears it) turns to bare, flattened earth, so paths and camps
  show. Drinking sends rings across the water; gathering flicks up bits of leaf and stem.
- **Animals at real size**, on the people's scale (175 cm = 33 sprite pixels), from each species'
  shoulder height and head-and-body length: wolf 80/130 cm, boar 90/150, deer 120/200, cave lion
  120/210, cave bear 130/250, wild horse 135/210, aurochs 175/290, woolly mammoth 300/440, crocodile
  about 4 m long; body depth, leg length and leg thickness follow per-species shares. Small animals
  (rabbit, fowl, tortoise, fish, bat, snake) are drawn no smaller than a legibility floor (a 14-pixel
  body), so they are larger than life; everything from a wolf up is true to scale.
- **Depth and line of sight.** Trees, animals and people are drawn as one scene sorted by where each
  stands (each row of trees on its own strip), so a tree hides whoever stands behind it. Where a tree or
  a big animal (mammoth, aurochs, horse, bear) hides someone, the hidden part of them shows through as a
  light outline and faint fill in their own colour, and only there (RuneScape's character silhouettes;
  Hades and Diablo III do the same). For the selected person the trees in front also thin out in a
  small circle round them (RuneScape's canopy cutout; Stardew Valley fades what the player walks behind).
- **Legibility** (RimWorld, Oxygen Not Included): a one-pixel white outline round the person under
  the pointer; a tell above anyone at 60% stress or more (a sweat drop; a dark scribble from 80%); the
  inspector lists **what stresses it** over the last game hour by source (hurt, danger, unmet needs,
  loneliness, missing someone, sleep debt, darkness: `StepInfo.stress_from`, exported per agent as
  `sw`), and names ties under **Relationships** (mother, child, mate when they conceived together in
  the recording, close kin, kin, friend for attachment of 0.3 or more without blows, rival for blows
  without attachment). The downed lie with blood beside them and a red cross above; "is down" is a
  letter.
  **People are measured bodies** drawn in a colony-sim manner (a large head on a simple figure, a
  two-dot face, a thin ink line), at twice the world's pixel density, sized from what the simulation
  tracks and kept within human bounds (`bodySprite`): figure height from height in cm (175 cm = 33
  sprite pixels; bounded 0.33-1.15 of that); head share of stature from growth curves (1/4 at birth,
  1/5 at 2, 1/6 at 6, 1/7 at 12, 1/7.5 adult), drawn larger by one fixed rule for everyone (0.15 +
  1.15 x real, bounded 0.30-0.44), so differences between people stay true; leg share from growth
  (0.38 at birth to 0.47 adult); shoulder breadth 0.25 of stature x sqrt(build) x body mass (BMI from
  weight and height, bounded 14-38) x a little for strength, hips wider than shoulders in women past
  13; a belly when pregnant; wounds with injury and pale lines with scars; ribs below 0.88 of ideal weight
  and a rounder belly above 1.1; light on the shoulders when strong; hair greying from 40 (up to 80% by 70);
  skin paler and yellower when sick, bluer when cold, flushed when overheated; what is in hand at the hand. Nothing is drawn that the
  world lacks: no clothes, no anatomical detail. Who someone is shows as a thin ring at the feet in
  their colour (a marker, not clothing). They face the way they last stepped, struck, touched or placed
  (front, back, and profile both ways), walk in a four-frame cycle while a tick plays (arms swinging
  against the legs), and raise or thrust an arm to strike.
  **What they are doing shows in the body**, read from each tick's action and outcome, in two frames
  while the tick plays (held still when paused): eating brings a hand to the mouth; drinking is a
  crouch with the arms down to the water; picking up or putting down is a stoop with a hand to the
  ground; touching (and mating) reaches an arm out; striking raises or thrusts a fist; walking is the
  four-frame step. Arms that cross the body are inked so they read against it. The colonist bar shows
  each person's own sprite as a head-and-shoulders bust (the whole head, never cut off), and the
  inspector shows their map sprite front, side and back, as it is drawn now.
- **Animals** are drawn at the same density as people, each species its own build and marks (antlers,
  horns, tusks, manes, spots, a mammoth's trunk), facing the way they went, gliding between cells and
  stepping as they go (bats flap, fish swim, snakes wave). An animal's previous cell is the nearest of its
  kind a tick before.
- **Ground detail:** grass curls round the corners of bare patches; ink closes the corners where water
  meets land; fallen logs, stumps and mushrooms in woods; stones in grass; bones and dry shrubs on bare
  ground; lily pads (some in flower) by the shore. **Ambient life:** tall grass swaying, rain rings on
  water where it rains, smoke rising from fires, warm light added round fires and torches (stronger as it
  darkens), and fireflies over woods and marsh from dusk.
- **Camera.** Zoom 1x, 1.5x, 2x, 3x (wheel about the pointer, the -/+ buttons, or the - and = keys); drag
  to pan; **Follow** (F) keeps the selected person centred. Per tile the art has Mudborne's density (16
  pixels); at 3x a tile is also about Mudborne's size on screen.

## Felt needs and the cognitive map

- **Convex needs:** each need (food, water, energy) adds `need_weight` (0.6) x
  (1 - deficit^2) to felt well-being. A small deficit is barely felt; a large one
  strongly (homeostatic RL, Keramati and Gutkin 2014). Drinking when parched
  (water 0.1 to 0.6) feels about 4.6 times what it did under the old linear
  term; topping up from 0.7 feels the same. Felt well-being enters reward only as
  potential-based shaping, so this changes how soon thirst is felt, not what is
  optimal.
- **Cognitive map:** each agent keeps its own map of what it has seen (water,
  plants, how recently), updated only for cells in sight and fading with a
  10-day half-life. It is read as an 8 x 8 grid of blocks centred on the agent,
  spanning twice the world's width, so the whole world is in it from anywhere
  (256 extra observation numbers). A newborn starts with an empty map.

## Sleep

Sleep is built in, as in people; how agents handle it is up to them.

- **Sleepiness** builds while awake (0.0125 a tick, 1.5 times that at night: a
  body clock), about 0.45 after 16 waking hours. Sleep (`SLEEP`) removes it,
  faster once sleep is deep (after 2 unbroken hours) and at night; about 8 hours
  a night keeps up. Past 0.95 (about 30 hours awake) an agent collapses asleep
  whatever it chose and cannot wake by choice until sleepiness is below 0.6.
- **The cost is awareness.** Asleep, an agent sees and hears only the cells next
  to it, cannot act (only sleep on, or `REST` to wake), and is woken by being
  struck, attacked, burned, touched or falling. Others can see who is asleep.
- **What sleep pays for,** only through the body: extra energy, lower stress and
  1.5 times the healing while asleep. Going without it adds stress (from 0.7),
  narrows sight by up to 2 cells (from 0.5) and makes falls likelier.
- **Felt:** rest from sleepiness is a fourth need in felt well-being, on the
  same convex curve as food, water and energy, so sleep feels good when tired.
- **Sleep is when the day is learned** (complementary learning systems: a fast
  store of the day, slow learning while asleep; `ppo_multi.consolidation_plan`,
  `learn_core`, `after_sleep`):
  - *The day waits for sleep.* Experience not yet slept on is carried from one
    update to the next for up to `memory_span` ticks (48, a day). Each step,
    and what was learned by watching others then, is learned **exactly once**:
    in full once 8 ticks (4 hours) of deep sleep have followed it, or, if it
    leaves the store first, in proportion to the sleep it got. Going without
    sleep costs memory. A life that ends in death counts in full: with one
    network per slot, learning from deaths stands in for selection.
  - *Surprise replays more.* Consolidated steps are reweighted by the size of
    their prediction error, |TD error|^0.5, half and half with uniform weight
    (`surprise_priority`); the total weight is unchanged (prioritised replay;
    Mattar and Daw 2018 on hippocampal replay).
  - *Old memories are interleaved.* Each agent keeps a long-term store of 256
    moments (what it saw, the memory state it had, what it chose). Each update
    adds a KL term (`retention_coef` 0.1) pulling its choices there back toward
    what it chose then, so new days do not overwrite old skills; a sixteenth of
    the store is replaced by newly consolidated moments each update, so old
    memories fade over about 16 updates (McClelland et al. 1995; Kumaran et al. 2016).
  - The world model learns only from consolidated steps too.
  - Logged: `learned_share` (of the steps settled in an update, the share
    learned rather than forgotten), and under `sleep_memory` the share still
    waiting, the retention KL and how full the long-term store is.
  - `--no-sleep-learning` turns all of it off (plain PPO per rollout). Left out
    on purpose: synaptic down-scaling and training the policy on dreams.
- **Sleep keeps memories of places.** A new memory in the cognitive map fades
  by half in a day; deep sleep consolidates it, after which it fades by half in
  10 days.
- Logged: share asleep by night and by day, sleepers woken, collapses; in the
  continuous world also how often sleepers lie beside someone or by a fire.

## Reproduction

Off by default; `--reproduction` (with `--agents` as slots and `--founders`
alive at the start) or `config.with_reproduction()` turns it on, at 1 day per
biological year (adult about day 15, lifespan about 70 days).

**Time scale.** `--days-per-year` (calendar: seasons, climate) and `--days-per-bio-year`
(age: growth, fertility, pregnancy, lifespan) set it; `config.with_time_scale()`. The default
(40, 1) compresses both so winters and generations fit in a run. The `realtime` preset is
(365, 365): 365-day years, 9-month (274-day) pregnancies, adults at 15 calendar years. Its cost:
a child is fertile after 262,800 ticks, while a 100M agent-step run of 16 agents in 32 worlds
lives about 195,000 ticks per world, so no one born in training grows up and a 400-day
continuous world is 1.1 years. It suits realism checks within one generation, not persistence or
culture across generations. `experiment.py --preset realtime` runs it; the gate scales
adulthood with it.

- **Sexes** alternate among founders; newborns are 50/50. Sex is visible.
- **Mating** is an action (`MATE`). Conception needs a woman and a man, both
  adults, adjacent, both choosing it: one now, the other now or within the last
  `mate_window` ticks (4, i.e. 2 hours; 1 restores the same-tick rule) while awake.
  An open offer is visible to others as *courting*, and lasts only while its maker stays
  with it (resting beside or touching the other): any other act withdraws it at once, so no
  one is ever mated without having chosen it themselves within the window. Coercion is not
  modelled, on purpose. Then a 15% chance if she
  is 15-45, not pregnant, at least 85% of ideal weight, not sick and stress
  below 0.7, and he is not sick, and a slot is free.
- **Pregnancy** lasts 0.75 biological years (36 ticks); she needs 30% more
  food and water. Childbirth kills the mother 1% of the time (2% if weak).
- **Birth:** the child appears at the mother's cell in a free slot (never used
  first, else a dead agent's). Traits come from both parents through
  `genotype.inherit` (heritable with random variation); appearance blends
  theirs; others' memories of the slot's last occupant are cleared.
- **Infants** under 1 year cannot walk or forage. Until 2, a child within one
  cell of its mother is nursed: its food and water are topped up and she pays
  60% of what it takes.
- **Genetic survival:** in survival mode each agent's reward adds its living
  relatives' survival signal (+0.01 alive, -1 dying) times relatedness (true
  kinship, seen or not: 0.5 for parent, child or sibling, 0.25 for
  grandchildren). Mating itself is never rewarded directly.
- **Intimacy calms:** mating (with or without conception) lowers both
  partners' stress by `intimacy_relief` (0.2) times an inherited `libido`
  trait (log-normal, heritability 0.4). Repeats with the same partner feel
  less (the same habituation as touch and reciprocity: halves every 48 ticks).
  The benefit reaches reward only through the body: calmer agents heal faster,
  feel better and stay fertile (stress at 0.7 or more blocks conception).
  Stress cannot drop below zero, so there is nothing to farm.
- **Brains:** training still gives each slot one network, so a newborn in
  training uses its slot's network. In `continuous_world.py` (evaluation only)
  a newborn gets one parent's network with small mutations. True per-individual
  birth brains that evolve are not built yet.

## Gifts and reciprocity

A gift is food (plant, raw or cooked meat) that one agent drops and a different
agent takes within 20 ticks. Dropping alone, taking back one's own drop, or
dropping wood or stone is not giving. The recipient remembers who gave to it.
Reciprocity rewards repaying someone who gave you food.

**Feelings fade with repetition (habituation).** Repaying, touching and striking
back are never limited, but each repetition with the same partner feels e^-h as
good, where h rises by one per felt act and halves every day apart. An ordinary
act feels full again within about a day; a whole day of nonstop repetition pays
at most about 1.6 single acts and builds tolerance that takes days to fade.
Feelings that only cost reward (empathy, kin care, envy, guilt) need no fading. An earlier rule counted any drop next to a neighbor as a gift and paid
every time; a 30M-step run learned to drop and re-pick items beside each other,
inflating gifts to 1 tick in 8 with no food changing hands. Global reward in
the logs now excludes the moral sentiments, so a loophole there cannot inflate
the headline measure.

## Names, and whether agents invent them

Each agent has a real, randomly assigned name (`lifesim/names.py`, drawn from
100 names across many languages), saved as `names.json` with every run and
shown in logs and the playback. **Agents never perceive these names**: nothing
in the state or observations carries them.

Whether agents come to refer to one another is measured
(`lifesim/analysis.py`). For every utterance we note the nearest other agent:
- **Naming index:** information (in bits) between messages and that agent,
  minus its value under shuffled targets; about 0 when messages say nothing
  about who is near.
- **Invented names:** a message that is the most common thing at least two
  different speakers say near the same agent, and that is used near that agent
  in at least half its uses.

The tests check that the analysis recovers names when speakers share them and
finds none in random speech. The playback shows the result in its Language
panel, along with how much messages say about predators, large prey and fire.

## Does the system support emergent life?

`python scripts/validate_system.py` (writes `VALIDATION.md`) checks this
without any training. Hand-scripted agents act out each behavior to show it is
possible with generic actions alone and actually pays off; an audit shows the
reward does not pay for it. Current result: 9 of 9 checks pass.

- Two hunters kill a large animal in 5 ticks; one takes 13.
- Wood placed on rock and struck with a stone catches fire (here after 8
  strikes); heat cooks meat dropped beside it, cooked meat is eaten with no
  infection, and a fire warms its surroundings by 15 C.
- A companion keeps an agent above the cold-damage threshold at 0 C (a small
  effect: body temperature -0.67 together vs -0.71 alone).
- A predator next to a lone agent attacks every tick; with a companion beside
  it, never.
- An observer that only watches, seeing only what it could see, learns a
  demonstrated action (probability 0.09 before, 1.00 after).
- Invented names and meaningful messages are detected when present and not in
  random speech.
- The reward pays for staying alive, avoiding pain, novelty, touch and the moral
  sentiments; nothing rewards hunting, giving, fire, cooking, speech, huddling or
  company in general. Two inborn sentiments reward answering a past act in kind
  (reciprocity and vengefulness).

What the system does not yet do: reproduction (only inheritance, aging and
lifespans exist), and parallel or multi-tick actions.

## Development: branches, pull requests, CI

The code lives at [github.com/dylandlr/agon](https://github.com/dylandlr/agon). Work goes on a branch
and into `main` by pull request; every push to a pull request runs CI (`.github/workflows/ci.yml`):

- **Syntax:** every Python file byte-compiles, and the playback page's script parses (`tools/check_playback.py`).
- **Tests:** the whole suite on CPU JAX with pinned versions (`requirements-ci.txt`), in four shards run
  side by side (`tools/ci_shard.py`), each with `pytest -n auto`.

A pull request that changes `lifesim/` or the training scripts changes `code_version` (lifesim/presets.py):
runs from before it cannot be resumed or mixed with runs after it. The pull request template asks
which runs must be redone. Training runs and recordings (`runs/`) are never committed.

## Checking everything

- `pytest -q`: unit tests for every mechanism.
- `bash scripts/full_check.sh`: every pipeline end to end at small sizes,
  every feature on (Phase 1 training and evaluation; Phase 2 with alpha,
  scarcity, social learning, messages, fire, morals and names; checkpoint and
  resume; playback with naming analysis; run comparison; fire baseline;
  benchmark). Prints `ALL PIPELINES PASSED`.

## Running on your own computer

1. **Set up once.** macOS, Linux or WSL2: `bash setup_local.sh`. Native
   Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File setup_local.ps1`.
   Each creates `.venv`, installs JAX (with CUDA if Linux has an NVIDIA GPU),
   Flax 0.12.10 or newer and Optax, then checks that a network runs.
2. **Confirm everything works:** `python scripts/run_local.py --quick`
   (about 10 minutes on a CPU): checks, tests, validation, benchmark, a tiny
   training run, learning curves and a playback.
3. **Train for real:** `python scripts/run_local.py --steps train plot playback`.
   Defaults: 8 agents, 32 x 32; 32 worlds with the conv encoder on a GPU, 16
   worlds with the mlp encoder on a CPU. Rerun the same command to resume.

GPU support: NVIDIA on Linux, or on Windows through WSL2. Native Windows and
Macs run on the CPU. Outputs land in `runs/<run-name>/`: `history.json`,
`curves.png`, `playback.html` (opened in your browser automatically).

## Running on GPUs (Texas A&M HPRC)

The `hprc/` folder targets the Grace cluster (A100 GPUs). Everything also
runs anywhere JAX finds a CUDA GPU: `pip install -e ".[gpu]"`.

1. **Account.** Apply at https://hprc.tamu.edu/apply. Students get a Basic
   allocation of 20,000 SUs per fiscal year, which needs approval from a
   faculty PI (your advisor). Larger Startup allocations (up to 200,000 SUs)
   are requested by the PI.
2. **Budget.** On Grace an A100 costs 72 SUs per hour plus the effective
   cores; with 8 cores and 40 GB that is about 80 SUs per hour, so a Basic
   allocation is roughly 250 A100-hours. Check any job script's charge with
   `maxconfig`.
3. **Setup (login node).** Copy the repo to `$SCRATCH/lifesim`, then
   `bash hprc/setup_env.sh`.
4. **Benchmark first.** `sbatch hprc/benchmark.slurm` measures agent-steps
   per second at 16, 64 and 256 worlds and prints hours and SUs for a 50M
   agent-step run. Use the fastest world count in the training scripts.
5. **Train.** `sbatch hprc/train_phase2.slurm` for one run, or
   `sbatch hprc/core_array.slurm` for the core experiment: 3 alphas x 2 food levels x 3 seeds (18 runs of 100M agent-steps),
   six GPUs in parallel. All scripts pass `--resume`: resubmit after a time
   limit and the run continues from its last checkpoint.

The module line (`module load GCCcore Python`) and the 24-hour time limit are
assumptions; confirm with `module spider Python` and `sinfo -p gpu`.

**Checkpoints** are written atomically every `--ckpt-every` updates to
`<out>/checkpoint.pkl`, with history in `<out>/history.json`.

**Convolutions** use the native GPU kernel on GPU and the patch path on CPU
(`conv_impl="auto"`); both share one parameter layout, so checkpoints move
between CPU and GPU.

## Known limits and next steps

- **Discount factor 0.997.** Threats unfold over 200 to 500 ticks; 0.99 sees
  about 100 ahead. At 2M steps 0.997 beat 0.99 (mean life 1,421 vs 1,256,
  full-year survival 31% vs 23%), but one seed each and noisy, so unconfirmed.
- **Convolutional encoder is the default.** At 820k steps it led the flat
  encoder (1,263 vs 1,163 ticks mean life) after trailing early; one seed.
- **Genotype is not observed** by the agent, matching the design: agents do
  not know their genes; genes act through the body and the reward.
- **Speed.** About 5,000 steps/sec on one CPU core with 64 environments of
  64 x 64.
  Resets are about 50 times costlier than steps, so finished environments
  draw from a pool of worlds generated once per update. Use a GPU and more
  environments for real runs.
- **Phase 1 gate:** agents reliably survive a full seasonal cycle and the
  simulation is fast enough. See the evaluation output for current status.

Next (Phase 2): multiple agents per world with the full primitive set
(release, strike, touch, cache access, tokens, pointing), relationship
memory in observations, per-agent parameters, and the E0 scarcity replication.

- **Observation bug fixed (this version).** Since the observation grew to 11
  channels, the code removing an agent from its own "other agents" channel
  subtracted from the small-prey channel instead. Agents counted themselves as
  company and saw a phantom -1 prey on their own cell. Earlier results ran
  with this bug, and old checkpoints do not match the new action and
  observation sizes anyway.
- **Replay includes speech.** `record_episode` logs messages
  (`messages.npy`) as well as actions, so runs with speech replay exactly.
- **One extra compile per run.** The second `update` call retraces once
  (array types settle after the first step), so expect two compiles at the
  start of every run. Harmless, but worth fixing at the root later.
- **Founder pretraining (deferred idea).** Train one survival brain in the
  Phase 2 world and start every individual from it plus noise, so social
  training begins with competent agents. Not implemented.
- **Imagination uses today's options.** The world model predicts beliefs, rewards and the agent's
  own state, not which actions will be possible; imagined habits choose among the actions possible
  now. Its own-state predictions are unbounded and are clamped to valid ranges in the playback.
- **Training brains belong to slots.** In training (B parallel worlds sharing one network per
  slot) a newborn takes over its slot's network, whoever trained it. Only the continuous world
  with `--learn` has true individual brains and real inheritance; read training as the lineage's
  deep past.
- **The slot count binds before food does.** A competent band fills its slots and stays there
  (viability.py); food at 0.4 does not limit it. Pressure that makes cooperation pay needs either
  scarcer food or more slots than the land can feed.
