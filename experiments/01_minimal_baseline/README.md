# 01 · Minimal baseline: does it learn to survive, and does cooperation appear?

| | |
| --- | --- |
| **Status** | Running: `s0`, 20M agent-steps, started 2026-10-20 |
| **Branch** | `minimal-baseline` (from `ecology` at `94d678a`) |
| **Code version** | `32b65fbbdc97` (`lifesim.presets.code_version()`; unchanged by this experiment) |
| **Hardware** | GTX 1660 Ti Max-Q (6 GB; about 4 GB usable under WSL2), JAX 0.11.2 + CUDA 12 |
| **Owner** | Preston |

## Question

The full society design (the `society30` preset) layers reproduction, sleep-gated learning, a
world model, imitation, given skills, a living ecology and disposition-weighted social rewards on
top of the learner at once, and no multi-agent configuration has yet been shown to train. Before
adding any of that, we test the smallest setting where the research question still means something:

1. **Does it learn?** Do independent recurrent PPO agents, rewarded only for their own survival,
   learn to live clearly longer than random agents, and approach or exceed a scripted forager?
2. **Does cooperation appear?** With nothing in the reward for acts toward others, do behaviors
   that need or benefit from others (group hunting, standing together against predators,
   huddling in winter) show up because they keep the learner alive?

If (1) fails here, it will fail in the full design too, and the problem is in the learner or the
world's balance, not the social layers. If (1) passes, the later experiments add one layer at a
time and attribute changes to it.

## Setup

Everything is defined in `train.py` (`minimal_env_config`, `minimal_ma_config`), with a comment on
each departure from the defaults.

### World
- **Classic world** on a **32 × 32** map: grass, forest, rock and water terrain, Markov
  weather (rain, drought, cold snaps, heat waves), seasons, day and night. At 30-minute ticks a
  cell is 2 km.
- **8 agents**, one network each, starting as a band within 3 cells of each other near water.
  No reproduction: the same 8 individuals per episode.
- **Episodes** last one compressed year: 40 days, 1,920 ticks, with winter in the last 10 days.
  A world restarts at year end or when everyone has died.
- **Kept** (everything that makes another agent useful): food, water and cold; disease
  (raw meat, water, contact); falls and injury; small prey; large prey that need two or more
  hunters striking in the same tick; predators that won't attack an agent with a neighbor;
  huddling for warmth; fire physics (sparks, lightning, cooking); 4 × 32-symbol messages, which
  cost energy; touch; gifts (dropping food); striking other agents.
- **Turned off** to simplify:

| Setting | Value | Why |
| --- | --- | --- |
| `sleep_pressure` | 0 | No sleep need, so SLEEP is never available. Sleep also gates learning in the default trainer, which this experiment doesn't use. |
| `mineral_abundance` | 0 | No minerals or metals; fewer things to select, carry and place. |
| `fauna_diversity` | False | One species per animal group (small prey, large prey, predator). |
| `hive_density`, `mound_density` | 0 | No bees or termites: no honey, grubs or stings. |
| `byproducts` | False | No hides, bones, feathers, leaf litter or fiber. |
| living-world features | off | No biomes, moving weather, plant ecology, camp fouling, caves or "downed" state. |

### Reward: own survival only
`reward_mode="survival"`: +0.01 per tick alive, −1 on death, plus `γ·W(after) − W(before)`,
where W is felt well-being from the agent's own body (health weighted by pain aversion, food,
water, energy, warmth, calm). This is potential-based shaping (Ng, Harada & Russell 1999): any
loop of actions that ends where it started earns about zero, so the shaping can't be farmed.

Every channel through which another agent's fortunes reach one's own reward is zeroed:

| Setting | Value | What it would have done |
| --- | --- | --- |
| `care_empathy`, `care_kin`, `care_reciprocity`, `care_revenge`, `care_sadism`, `care_attachment` | 0 | Feel visible others' gains and losses, weighted by inherited dispositions |
| `care_inequity` | 0 | Envy and guilt over well-being gaps |
| `saint_rate` | 0 | The rare agent born selfless toward everyone |
| `heartache_empathy`, `heartache_grief` | 0 | Seeing wounds or losing an attached companion raises stress, and calm is part of well-being |
| `attach_absence`, `attach_calm` | 0 | Stress from missing attached companions, and calm from their company |

**Kept on purpose:** loneliness raises stress and company eases it, for agents in general rather
than particular ones. This is the body's response to company, one of the physical reasons staying
together can pay off, alongside predator deterrence and huddling. It is world physics, not a reward
for social acts. Verified: over 300 random ticks the social reward term is exactly 0.

### Learner
Plain recurrent PPO, one GRU actor-critic per individual, all running in parallel over 8
replica worlds (`lifesim/ppo_multi.py`):

| Setting | Value | Why |
| --- | --- | --- |
| `sleep_learning` | False | Plain PPO on every rollout. Required: without sleep, the default trainer would wait forever for sleep before learning. |
| `imitation_coef` | 0 | No social learning; isolates individual RL. |
| `world_model` | False | No imagined rollouts. |
| `per_agent_lr` | False | Same learning rate for everyone; the default scales it per agent by a gene. |
| `alpha` | 0 | Selfish: each agent trains on its own reward. |
| `gamma` | 0.997 | Default; a horizon of about 330 ticks (about 7 days). |
| `encoder` | mlp | Fastest on this GPU (1,763 vs 1,352 agent-steps/s for conv). |
| `num_worlds` | 8 | 16 runs out of the GPU's memory; 4 → 8 gave only 1.16×, so more would add little. |
| rollout, epochs, minibatches, lr, clip, entropy | 128, 4, 4, 3e-4, 0.2, 0.005 | Defaults. The learning rate is constant, so a run can be extended with `--resume`. |

**Budget:** 20M agent-steps is 2.5M per individual network, about 40 world-years per replica,
roughly 3.5 hours. It's checkpointed every 50 updates (about 410k steps); stop early or extend
with `--resume`.

## Decisions log

| Decision | Choice | Why | Made by |
| --- | --- | --- | --- |
| Agents per world | 8 | More encounters, so more chances to cooperate; the world's animal densities are tuned for 8 on 32 × 32; comparable with the project's baselines | Preston |
| Reward | Own survival only | Nothing pays for social acts, so any cooperation is learned | Preston |
| Social stress channels | Off (except loneliness and buffering) | They would bring others' suffering into one's reward through stress | Claude, agreed |
| Make the world easier first? | **No** | The random (206 ticks) to scripted (831 ticks) gap is wide enough to show learning; the shaping reward is dense; retuning the balance now would mean revalidating it later and would part ways with the world used in the partner's experiments. The difficulty dials (`--scarcity`, predators, disease) are the fallback if nothing is learned. | Claude, agreed |
| Code location | `experiments/01_minimal_baseline/` | One self-contained folder; doesn't touch `code_version` | Preston + Claude |
| Training length | 20M agent-steps | Phase 1 needed about 5M on one shared network; here each of the 8 networks gets one eighth of the data | Claude |

## Reference policies

Same world, 32 worlds never seen in training (seed 12345), one year each. From
`results/baselines.json`:

| Policy | Mean lifespan (ticks) | Alive at year end | Worlds with a survivor at year end | Survival score per world-day | Time near others | Large prey kills | Main causes of death |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 206 | 0% | 0% | 1.25 | 0.73 | 0 | thirst 143, violence 40 |
| Scripted forager | 831 | 18.4% | 40.6% | 2.52 | 0.94 | 0 | thirst 96, injury 43, falls 28 |

- **Random** picks a uniformly random valid action each tick. It is the floor.
- **Scripted forager** sees the whole map and follows fixed priorities: flee predators, drink
  when below 50% water, eat, rest at night, gather, seek forest when cold. It never cooperates and
  is not optimal: it walks across rock, ignores injury, and waits too long to head for water. It is
  a reference point, not a ceiling.

**Reading the cooperation measures:** "time near others" is already high for random agents (0.73),
because the band starts together. On its own it is not evidence of cooperation; it counts only if
it rises above random and agents stay together after their starting area runs out of food. The
cleanest signal is **large prey kills**, which need two hunters striking the same animal in the same
tick, and which neither reference policy ever achieves.

## Success criteria

- **Learns (main result):** mean lifespan rises clearly and keeps rising, well above random's
  206, and thirst's share of deaths falls. Reaching the scripted forager's 831 ticks or 18% alive
  at year end is a strong result.
- **Cooperation (exploratory):** any of these, sustained over training:
  - large prey kills above 0
  - predator attacks deterred by a neighbor, per 1,000 agent-ticks, above random's 0.80
  - winter huddling share rising
  - time near others above 0.73 later in episodes
- **Failure modes to watch:** entropy collapsing early (premature commitment to one action); lifespan
  flat after about 5M steps; deaths dominated by violence (agents learn to hit each other).

## How to run

```bash
source .venv/bin/activate                                          # from the repo root
python experiments/01_minimal_baseline/train.py --baselines        # writes results/baselines.json (~3 min)
python experiments/01_minimal_baseline/train.py --run-name s0      # 20M agent-steps (~3.5 h on the 1660 Ti)
python experiments/01_minimal_baseline/train.py --run-name s0 --resume --agent-steps 3e7   # extend
tail -f experiments/01_minimal_baseline/runs/s0/train.log          # progress: one JSON line per 10 updates
```

Seeds: `--seed 1 --run-name s1`, and so on. A run refuses to resume if the code version, agents,
grid, worlds, rollout length, encoder, alpha, scarcity or seed has changed.

## Files

| Path | In git | Contents |
| --- | --- | --- |
| `README.md` | yes | This document: design, decisions, results |
| `train.py` | yes | Config and training loop; `--baselines` for the reference policies |
| `results/baselines.json` | yes | Random and scripted measures |
| `results/<run>/history.json` | yes | Arguments and one row of measures per 10 updates (small) |
| `runs/<run>/checkpoint.pkl`, `train.log` | no | Network weights and optimizer state (large), raw log |

## Results

*Pending: `s0` is training.*

## Next: 02 · Reproduction toward a steady-state society

Goal: collaborative bands that last across generations. Open design questions, decided when 02
starts:

- **Births:** `with_reproduction`. `n_agents` becomes the number of slots, with fewer founders.
  `society30` uses 64 slots and 8 founders; 64 likely won't fit this GPU, so benchmark 16–32.
- **Kin term in the reward:** in survival mode with reproduction, relatives' survival is added to
  each agent's reward, weighted by relatedness (`env.py`, the `genetic` term). It is the first social
  term to enter the reward, and whether to keep it is a research decision (kin selection as a given
  versus something learned).
- **Generations within training:** with 1 day per biological year, adults are fertile from about
  day 15, but 1-year (40-day) episodes cut lives short. Longer episodes (`society30` uses 10 years)
  or `scripts/continuous_world.py` will be needed for persistence.
- **Viability first:** `scripts/viability.py` checks whether a scripted band can keep its numbers
  up in the chosen setting. Run it before training.
- **Measures:** births, the share of founders' descendants reaching adulthood, population over
  time, and the persistence gate (`scripts/experiment.py gate`).
