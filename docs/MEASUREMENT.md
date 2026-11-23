# Measurement: what every run records, and how to read it

The single source of truth for run artifacts and run diagnosis. If something here disagrees with
another doc, this file is right and the other one is stale.

Sections 1 to 7 describe what is *recorded* and what each number would mean. Section 0 is the
exception: it holds findings that have actually been measured, with the measurement that produced
each one. Nothing enters section 0 on reasoning alone.

---

## 0. Confirmed findings

Every line here was measured. Anything derived by reading code and not yet checked against a run
belongs in the open-questions list at the end, not here.

### The return identity holds exactly

`G = c_a/(1−γ)·(1−γ^(T−1)) − c_d·γ^(T−1) − Φ₀`

Verified on five seeds, one agent (so `social` and `genetic` are both zero — `genetic` is
`(relatedness * ~eye) @ signal`, which excludes self), resting until death. Measured versus
predicted agreed to **1e-5**, max |diff| 0.00001:

| seed | T | Φ₀ | measured | predicted |
|---|---|---|---|---|
| 0 | 269 | 2.296 | −0.8993 | −0.8993 |
| 1 | 200 | 2.119 | −1.1686 | −1.1685 |
| 2 | 261 | 2.032 | −0.6832 | −0.6832 |
| 3 | 221 | 2.303 | −1.2074 | −1.2074 |
| 4 | 210 | 2.559 | −1.5382 | −1.5382 |

So the potential-based shaping telescopes to exactly `−Φ₀` and contributes nothing policy-dependent
to the return. Observed `Φ₀` reaches **2.559**, so `pain_aversion` exceeds 1 — an earlier claim here
that `Φ ≤ 2.2` was wrong.

Lifetimes for a do-nothing policy land at **T = 200–269** (single agent) and episodes under a random
policy end at t = 221–547 with every agent dead. Both bracket the independent figure that an agent
dies of thirst roughly 230 ticks after its last drink.

### Saturation is not the current failure

With the identity confirmed, the marginal value of one more tick is `dG/dT = 0.013·γ^(T−1)`:

| lifetime | `dG/dT` |
|---|---|
| 230 ticks (where agents die today) | 6.5e-3 |
| 1,920 (one year) | 4.0e-5 |
| 4,800 (2.5 years) | 7.3e-9 |

Agents dying in the low hundreds of ticks sit at the **steepest** part of the objective. Saturation
cannot explain a failure that happens where the gradient is strongest. It remains a real problem for
the emergence track once lifetimes pass ~1,000 ticks, and §3 has the derivation.

### The critic is blind to lethal danger

From `intel_f5550cf9ebbd_society30_w4/validate/caveman_dangerous_predators/diagnose/diagnose.json`
(Nov 7, code version `f5550cf9ebbd`). The probe sweeps `body_temp` over
`[0.0, −0.3, −0.5, −0.7, −0.9]`, where `−0.3` is `cold_onset` and `−0.7` is `cold_threshold`, the
point damage begins. Outside shelter at night, n=153:

| body_temp | V | P(seek_shelter) |
|---|---|---|
| 0.0 comfortable | −1.4657 | 0.02942 |
| −0.7 damage | −1.4725 | 0.02953 |
| −0.9 lethal | −1.4746 | 0.02955 |

V falls **0.0089** from comfortable to lethal — **0.2% of the 4.333 return range** — while
`P(seek_shelter)` rises 0.00013 and stays near 3%. `P(rest/sleep)` and `P(move)` move the wrong way.
By `diagnose.py`'s own rubric both failure modes fire: the value does not fall with cold, so credit
assignment never reached it; and the shelter probability does not rise, so there is no remedy either.

V sits near −1.4 in every context, which is about the mean return of a short-lived agent — it
predicts the average and almost nothing conditional. The one cold death on record knew shelter was
2 cells away (`d_known: 2`) and stayed out.

### Trained performance is reflex-level

Same file, `survival` section, in agent-years: **policy 9.81**, `reflex_cold` **9.89**,
`reflex_dusk` 8.78, random 3.96. A hand-coded reflex beats the trained policy. Deaths under the
policy: **thirst 10 of 24 (42%)**, predation 4, starvation 3, disease 3. Thirst kills ~230 ticks
after the last drink; GAE carries 99% of its credit within 85 ticks, so the dominant death cause is
the one just beyond reach.

Fire ladder: `near_fire 0, fed_fire 0, torch_lit 0, ignited 0` across 13,459 ticks while carrying
2,226 wood, and 226 wood through 1,332 cold ticks. Not one rung, ever.

### The architecture can represent survival value

Supervised probe: train the real network to predict an agent's remaining lifetime from one
observation. Train MSE falls 0.3834 → 0.0045 on unit-variance targets, i.e. **train R² ≈ 0.995**.
So `feat_dense(924 → 64)` does **not** structurally prevent body state from reaching the critic, and
the capacity-bottleneck hypothesis is dead. The encoder is also not degenerate: 0 of 128 hidden dims
constant, condition number 1.3e3, participation-ratio effective rank 34/128.

Held-out R² under a grouped split oscillated near 0, but that is overfitting at **48 agent-lifetimes
(38 in train)**, not a capacity limit — the independent unit for this task is the lifetime, not the
agent-tick. Real runs see ~6,800 lifetimes per learner at 1e8 steps.

**Two probe designs that produced garbage, recorded so they are not repeated.** A temporal
train/test split measures the monotone drift of hydration, not decodability (hydration R² −5.7).
A random split leaks, because consecutive agent-ticks are near-identical observations with
near-identical labels — it inflated the raw-observation reference from a true **−553** to a fake
**+0.914**. Split by whole agent-lifetime.

### Still open

Why the critic is blind. Capacity is eliminated, so credit assignment leads — but by elimination,
not by direct evidence. The instrument that settles it is `value.explained_variance` from a real run
on the emergence config; offline CPU probing has reached its limit.

---

## 1. The invariant

**Every run writes `history.json`, on every logged row, atomically.**

`write_history()` in `scripts/train_phase2.py` dumps to `history.json.tmp` and `os.replace`s it
over `history.json`. `os.replace` is atomic on POSIX, so the file on disk is always either the
previous complete version or the new complete version, never half a file. It is called:

- after every row is appended (`--log-every`, default below),
- at every checkpoint (`--ckpt-every`),
- once more after the training loop ends.

This matters because **most runs so far did not finish.** A Colab timeout or a Slurm time limit
kills the process between checkpoints. Before this change `history.json` was written only at
`--ckpt-every`, so an interrupted run lost every row logged since its last checkpoint. Now it
loses nothing.

Do not weaken this. `history.json` is a few hundred KB for a full run — the write cost is
irrelevant next to a training update, and it is the only artifact that survives an interrupted run
and still says what the run learned.

---

## 2. Where runs live

A run directory is `<root>/<step>/<design>/`, e.g. `intel_<code_version>_<preset>/validate/caveman/`.

| File | Written | Size (society30) | Needed for |
|---|---|---|---|
| `history.json` | every logged row | ~100–400 KB | **all diagnosis** |
| `checkpoint.pkl` | every `--ckpt-every` | ~1.3 GB | resuming |
| `params.msgpack` | **after the loop ends only** | ~104 MB | playback, eval, analysis |
| `population.pkl` | **after the loop ends only** | ~8 KB | genotypes of the final population |
| `<design>.log` | streamed | ~100 KB | stdout, one JSON row per line |
| `diagnose/` | `scripts/diagnose.py` | ~55 MB | needs/deaths sections, rollouts |

`params.msgpack` is 64 independent per-agent parameter copies: 404,971 params × 64 =
25,918,144 floats = 103,672,576 bytes + msgpack framing. If you see ~104 MB, that is why.

**A run directory with `checkpoint.pkl` but no `params.msgpack` did not finish.** That is the
fastest way to tell. It is still fully diagnosable from `history.json`, and resumable with
`--resume`.

### Storage locations

- **Colab**: written to Google Drive. The run roots live under Drive folder
  `1T-mpxtiyLkPy0134B3sRq13RXHBm2Stf`, as `intel_<code_version>_<preset>[_w<worlds>]`.
- **HPRC Grace**: `$SCRATCH/runs/exp_<code_version>_<preset>`. See `hprc/README.md`.
- **Local**: wherever `--out` points.

### Two traps when reopening an old run

1. **`code_version` must match.** `--resume` refuses a checkpoint from different code unless you
   pass `--allow-code-change`. This guard is correct; do not routinely override it.
2. **`population.pkl` is tied to the `Genotype` field count at the time it was written.** Adding a
   trait changes the NamedTuple's arity, so newer code cannot unpickle an older `population.pkl`.
   To analyse an old run, check out the commit whose `code_version` matches that run. Drive run
   roots carry the version in their name precisely so this is possible.

---

## 3. Why these metrics and not others

The objective is not what it looks like. Per tick, in `reward_mode="survival"`:

```
r_t = c_a·1[alive] − c_d·1[died] + (γ·Φ(s_{t+1}) − Φ(s_t)) + social_t + genetic_t
```

with `c_a = alive_reward = 0.01`, `c_d = death_penalty = 1.0`, `γ = shaping_gamma = MAConfig.gamma
= 0.997`, and `Φ = wellbeing()`, a bounded survival proxy over health, satiety, hydration, energy,
sleepiness, warmth and calm.

The shaping term is potential-based (Ng, Harada & Russell 1999), so over a life of length `T` it
telescopes, and `Φ = 0` once dead:

```
Σ_{t<T−1} γ^t(γΦ_{t+1} − Φ_t) + γ^{T−1}(−Φ_{T−1}) = [γ^{T−1}Φ_{T−1} − Φ_0] − γ^{T−1}Φ_{T−1} = −Φ_0
```

Exactly `−Φ_0`, independent of the trajectory. So the whole return reduces to:

```
G = (c_a/(1−γ))·(1 − γ^{T−1}) − c_d·γ^{T−1} − Φ_0 = 3.333 − 4.333·γ^{T−1} − Φ_0
```

**The objective is an affine, strictly decreasing function of one scalar: `E[γ^{T−1}]`.** Every
hand-tuned feeling in `Φ` contributes zero to the return; shaping only densifies the local
gradient. Nothing in the reward distinguishes thriving from barely surviving — only when you die.

At 30-minute ticks (48/day, `days_per_year=40` → 1920 ticks/year):

| survive to | `γ^{T−1}` | `G` | share of the objective's range |
|---|---|---|---|
| 1 day (48 ticks) | 0.868 | −0.43 | 13% |
| 5 days (240) | 0.489 | 1.21 | 51% |
| 10 days (480) | 0.236 | 2.31 | 76% |
| 20 days (960) | 0.056 | 3.09 | 94% |
| 1 year (1920) | 0.0031 | 3.32 | 99.7% |
| 2.5 years (4800) | 5.6e−7 | 3.333 | 100% |

Half the objective's total range is earned by surviving ~232 ticks (4.8 days); 90% by 16 days; 99%
by 32 days. An episode is `episode_years × 1920` ticks (48,000 for society30 = 1000 days), so
**99% of everything the learner can be paid for is decided in the first ~3% of the episode.**
Food storage, shelter, seasonal preparation and culture all pay off on horizons where the
remaining 1% lives.

This is why `mean_age_at_death` and `value.explained_variance` are the two numbers to read first.
The first says where on that curve the population sits; the second says whether the critic can
carry any consequence at all beyond GAE's reach.

### GAE's reach

Credit decays as `(γλ)^k = (0.997 × 0.95)^k = 0.9472^k`: half-life **12.8 ticks**, 99% of the mass
within ~85 ticks (1.8 days). `--steps` (default 128) is not the binding truncation; `gae_lambda`
is. Any action whose consequence arrives later than ~2 days reaches the policy **only** through
`V`. Hence `value.explained_variance`: if it sits near 0, delayed survival behaviour cannot be
learned no matter what the reward says.

---

## 4. `history.json` fields

Shape: `{"args": {...}, "history": [row, row, ...]}`. `args` is `vars(args)` — the full
command line including `code_version`, `seed`, `preset`. One row per `--log-every` updates.

### Read these first

| Field | Meaning | What to look for |
|---|---|---|
| `mean_age_at_death` | mean lifetime in **ticks** of agents who died this window | the headline. Compare against the table in §3. Plateau = the objective has saturated |
| `value.explained_variance` | 1 − Var(target−V)/Var(target), on-policy, pre-update | **the credit-assignment discriminator.** >0.5 healthy; ~0 the critic predicts nothing; <0 worse than the mean |
| `deaths` | count by cause, this window (`E.CAUSE_NAMES`, cause 0 omitted) | thirst/starvation dominant = the within-horizon needs loop is not learned |
| `entropy` | mean policy entropy over learned steps | max is ln(84) = 4.43 nats. →0 exploration died; stuck at 4.4 the policy never committed |
| `survival_score_per_world_day` | `0.01·alive_ticks − 1·deaths` per world-day | mode-independent; the only score comparable across reward modes and `alpha` |

### Everything else

- **Progress**: `update`, `ticks`, `agent_steps`, `agent_sps`.
- **Deaths in detail**: `deaths_by_season` (spring/summer/autumn/winter), `cold_deaths`
  (`open_adult`/`open_child`/`sheltered_adult`/`sheltered_child` — distinguishes "caught out" from
  "froze even under cover", which is what "learned winter" actually means).
- **Episode outcomes**: `episodes_ended`, `share_alive_at_episode_end`,
  `share_of_worlds_reaching_year_end`.
- **Critic**: `value.explained_variance`, `value.rmse`, `value.target_sd`. `rmse` near `target_sd`
  is the same statement as `ev` near 0, in reward units.
- **Optimisation health**: `per_output.{action,how,said,written}.{clip_fraction, approx_kl}`.
  Clip fraction far above ~0.2 or a large KL means steps are too big.
- **Entropy control**: `entropy`, `entropy_coef` (the bonus *actually applied* — at the
  `--entropy-boost-max` ceiling and still falling means the floor cannot hold).
- **Sleep/consolidation**: `learned_share`, `sleep_memory.{waiting_share, retention_kl,
  long_term_filled}`.
- **World model**: `world_model.{own_state_error, belief_error, reward_error, trust_in_ideas}`.
  `trust_in_ideas` starts at 0 and is learned; if it stays ~0 the imagination rollouts are being
  ignored.
- **Social learning**: `imitation_loss`, `watched_fraction`.
- **Behaviour rates** (per 1000 agent-ticks alive): `skills_per_1000` (`seek_water`, `seek_food`,
  `flee`, `seek_shelter` — the four macros that collapse navigation into one action),
  `per_1000_agent_ticks` (`strikes_on_agents`, `gifts`, `touches`, `messages`),
  `fire_ladder_per_1000` (use → maintain → cook → carry → make), `meat_eaten_per_1000`,
  `large_prey_hits_per_1000`, `moral_reward_per_1000`, `deterred_per_1000`, `matings_per_1000`,
  `woken_per_1000`, `collapses_per_1000`.
- **Shares**: `asleep_share_night`, `asleep_share_day`, `in_company_share`,
  `winter_huddle_share`, `scarred_share`, `lasting_injury_share`.
- **Population**: `births`, `conceptions`, `conceptions_blocked_by_slot_cap`, `kills`,
  `hunting_gini` (free-riding on large kills), `lightning_fires`.
- **Evolution**: `mean_sharing` — mean heritable reward-sharing gene over the living. Only
  meaningful with `--alpha-heritable`; drift toward 0 or 1 means selection is acting on it, a flat
  ~0.5 means it is not.

### Trap: a `0.0` that means "nothing learned yet", not "zero"

`entropy`, `entropy_coef`, `imitation_loss`, `retention_kl` and all three `world_model` errors are
means weighted by `w`, the share of each step learned after sleep consolidation. Until steps have
been slept on, `w` is zero everywhere and **every one of those fields reads exactly `0.0`.**

A verified example, the first row of a fresh run:

```
entropy = 0.0   entropy_coef = 0.0   imitation_loss = 0.0
world_model = {own_state_error: 0.0, belief_error: 0.0, reward_error: 0.0, trust_in_ideas: 0.0}
learned_share = 0.0   sleep_memory.waiting_share = 1.0
```

`entropy = 0.0` here does **not** mean the policy collapsed — it means no step had been learned
yet. **Always read `learned_share` and `sleep_memory.waiting_share` before believing any of those
fields.** If `learned_share` is 0, they carry no information. Discard early rows until
`learned_share` rises, or run with `--no-sleep-learning` when you want the metrics unweighted.

`value.explained_variance`, `value.rmse` and `value.target_sd` are **not** affected: they are
masked by `traj.valid` (steps actually lived), not by `w`. They are meaningful from the first row,
which is another reason to read them first.

### Reading rules

- **Judge on ground truth, never on training reward.** `mean_age_at_death`, cause-of-death mix and
  `survival_score_per_world_day` are comparable across designs. Training reward is not — its terms
  change with `--reward-mode`, `--alpha` and the caring weights.
- **One run is one sample.** Designs trained on the same `--seed` and map are paired, which helps,
  but a few percent is noise. Say so rather than reporting a difference as real.
- `--alpha` and the `care_*` weights both mix other agents' fortunes into the signal. Vary them
  separately or the attribution is lost.

---

## 5. Diagnosis routes

### First: is there a `history.json` at all?

A design folder holding only a `<design>.log` of one or two KB, with no `history.json`, means the
run **died before its first logged update**. Several designs with logs of *identical* size, created
a few minutes apart, means every one of them died the same way — the harness moved to the next
design each time. Read the log; it is a traceback, not a training log.

The failure seen on Colab, 2026-11-22, on all five designs of `intel_d23dd987b2ec_society30_w4`:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9.21GiB
  with allocator GPU_0_bfc on device 0. [executable_name='jit_update']
Can't reduce memory use below 9.25GiB by rematerialization; only reduced to 11.12GiB,
  down from 12.02GiB originally
```

**Cause: no `--agent-chunk`, on a GPU too small for the whole 64-agent update at once.** At
society30 the rollout observation buffer alone is `steps x agents x worlds x obs_dim x 4 B` =
`128 x 64 x 4 x 8668 x 4` = 1.06 GiB, and backprop through 128 GRU steps multiplies that. With
`agent_chunk = 0` the loss is computed over all 64 agents simultaneously and the peak reaches
~12 GiB, which does not fit a T4 (15 GB) or an L4 (22 GB, shared).

**Fix: `--agent-chunk 8`** (4 or 2 if it still OOMs). It computes the loss that many agents at a
time: **identical gradients**, about 30% slower, a large drop in peak memory. The notebook passes
it as `CHUNK`. Only set it to 0 on an A100/H100 — `nvidia-smi` in notebook cell 2 says which.

Note the shape of this failure: it costs ~5 minutes of compile per design before dying, so five
designs burn 20+ minutes of GPU and produce nothing. Run **one** design first and confirm
`history.json` has rows before queueing the rest.

**`benchmark.py` cannot size a chunked run.** It has no `--agent-chunk` flag, so it always measures
the unchunked update — the configuration that OOMs. When you train with `--agent-chunk`, the
benchmark's `RECOMMENDED --worlds` is measured under a different and much larger memory profile
than the run will use, so treat it as a lower bound and set `WORLDS` yourself. Either add the flag
to `benchmark.py` or skip it: on a config that has already OOMed, start from the same `--worlds`
that failed, add `--agent-chunk`, confirm it trains, and raise `--worlds` only on the next run.
Changing one thing at a time is what tells you the chunking is what fixed it.

Do not reach for this explanation when the log is large and rows exist — then the run trained, and
the table below applies.

| Symptom in `history.json` | Reading | Next probe |
|---|---|---|
| `mean_age_at_death` plateaus at 1k–2k ticks | objective saturation (§3): nothing left to pay for | raise `c_d` relative to `c_a/(1−γ)`, or raise `γ` (`--gamma`, or `--trait-gamma` to let the `patience` gene ∈ [0.95, 0.999] carry it) |
| `value.explained_variance` ≈ 0 and flat | the critic carries nothing; delayed behaviour is unlearnable | the critic's input path: the 8 reward-relevant scalars reach it only via `924 → 64` `feat_dense`, sharing that bottleneck with 532 heard/read dims |
| `deaths` dominated by thirst/starvation | the needs loop fails *inside* the GAE window — worse than saturation | check `skills_per_1000.seek_water` / `seek_food`: near zero means the macros are not being fired at all |
| `entropy` → 0 early | exploration collapsed | `--entropy-target` (the floor; off by default at 0) |
| `entropy` pinned at ~4.4 | no commitment; policy is ~uniform over 84 actions | look at `value.explained_variance` and `per_output.action.approx_kl` before touching entropy |
| `cold_deaths.sheltered_*` ≫ `open_*` | shelter is found but insufficient | an environment-calibration question, not a learning one |
| high `clip_fraction`, large `approx_kl` | steps too large | `--lr`, or `--no-agent-lr` to remove per-agent `lr_scale` |

### Structural facts worth knowing before blaming the learner

- **64 independent learners**, 404,971 params each, each fed 1/64 of the data: 100M agent-steps is
  1.56M steps per learner, ~3.9 steps per parameter.
- **Network weights belong to the slot, not the individual.** `Population` is
  `(genotype, appearance)` only. A newborn gets a GRU reset and inherits the slot's existing
  weights, not its parents'. So **selection acts on the 27 genes and never on the policy.** The
  only weight-level channel between agents is the imitation loss (`--imitation`, default 0.5).
- **No exploration bonus in `reward_mode="survival"`.** `novelty_reward` is applied only in the
  other reward branch. Exploration rests entirely on the entropy bonus and imitation.

---

## 5b. The group-mean denominator

`mix_reward`'s `living` argument picks what the group mean divides by, and the two answer different
questions.

**Default: every slot.** `(1 - alpha) r_i + alpha (1/N) sum_j r_j` over all N, the proposal's
formula. Dead and unborn slots carry reward 0 and so dilute the mean. Keep this for the alpha
experiment: hypothesis H2 says a fully shared reward suffers from credit assignment, and that `1/N`
dilution *is* the effect H2 predicts. Dividing by the living instead weakens it and biases the
experiment toward refuting its own hypothesis.

**`--share-living`: the agents alive at the start of the tick.** Right where slots far outnumber the
living — 64 slots seeded with 8 founders — because there the shared share of the signal would
otherwise shrink with the population for a reason the design never intended. A declared deviation
from the proposal, not a default.

The two coincide whenever every slot is alive, which is why only `tests/test_sharing.py` changes
when the default flips: `mix_reward` is called only when `alpha` or `alpha_heritable` is set, and
every other test trains at `alpha = 0`.

## 5c. Two experiments, two configs

`core` and the emergence track measure different things and need different configs. This is not
about the Oct 2026 proposal being authoritative — it is dated and the system has moved past it
deliberately. It is that an 8-agent cooperation measurement and a 64-slot multi-generation run
cannot share one config without conflating them.

| | `core` (cooperation) | emergence (`validate`, Master's track) |
|---|---|---|
| Agents | 8 fixed | 64 slots, 8 founders, reproduction |
| Episode | 1 year, 1,920 ticks | 25 years, 48,000 ticks |
| Why | its 7 metrics need no birth | a founder is fertile at 15 bio years x 40 days = **600 days**, so one generation needs ~15 years |
| Denominator | every slot (`1/N`) | `--share-living` |
| Peak update memory | ~1.5 GiB | ~12 GiB, needs `--agent-chunk` |
| Steps per learner at 1e8 | 12.5M | 1.56M |

`core` pins `--agents 8 --founders 8 --episode-years 1` per item. Running it on the emergence config
produced three failures at once: the Colab OOM (64 copies of a 404,971-parameter network, ~12 GiB),
sample starvation (1e8 spread over 64 learners, not 8), and objective saturation (gamma 0.997 gives a
333-tick horizon against a 48,000-tick episode).

**The emergence config is the harder and more interesting one, and its problems are real, not
artifacts.** Saturation at 48,000 ticks is a property of that track and has to be solved there, not
configured away — see §3 for the derivation.

### Current spec versus the proposal

The proposal is from Oct 2026. Everything below grew on purpose; the table exists so the writeup can
state the current system accurately, not as a list of debt.

| Item | Proposal | Now |
|---|---|---|
| Observation | 3,098 (11×11×24 + 194) | **8,668** (11×11×64 + 668 + 256 map) |
| Vocabulary | 16 symbols, ≤3/tick | 32 symbols, `MSG_LEN` 4, plus written inscriptions |
| Actions | 42 | **84** primitives × 60 aim × 32⁸ symbols |
| Heritable traits | 14 (6 moral) | 27 |
| Cognitive map | — | `8×8×4`, two decay rates |
| Episodes | 1 compressed year | 25 years (reproduction needs it) |

On the observation, the growth is not uniformly expensive: conv weights are shared over space, so the
window's +4,840 numbers cost only **+5,760** parameters (`3·3·24·16 → 3·3·64·16`), while the body's
+730 numbers cost **+46,720** (`194·64 → 924·64`) by landing in a dense layer. 11% of the network,
not 180%. The live question is that 532 of `feat_dense`'s 924 inputs are heard/read symbols sharing a
64-unit bottleneck with the 8 scalars that determine reward.

## 6. Spaces, for reference

- **Observation** `ℝ^8668` = `11×11×64` egocentric window (7744) ⊕ 668 body/social/heard/read
  ⊕ `8×8×4` cognitive map (256). 532 of the 668 body dims are heard (402) and read (130) symbols.
- **Action**, factored, independent categoricals: `{0..83}` primitive × `{0..59}` aim ×
  `{0..31}^8` symbols, so `84 · 60 · 32^8 ≈ 5.5e15`. Uniform entropy
  `ln84 + ln60 + 8·ln32 = 36.2` nats, 77% of it in the symbol heads — hence `msg_ent_coef`
  (0.0005) an order of magnitude below `ent_coef` (0.005), plus `silence_bias` on symbol 0.
  38 of the 84 primitives are `select_<material>`.
- **Model**, per agent: conv(16, 32s2, 32s2) → `288 → 128`; feats `924 → 64`; concat `192 → 128`
  → GRU(128) → policy/value/message heads, plus the world model (dynamics, reward and 8-dim needs
  heads) and a learned `trust` scalar gating imagination over 6 candidates × 6 ticks.

---

## 7. Reproducing a diagnosis

```bash
# Train, logging every update, checkpointing every 50
PYTHONPATH=. python scripts/train_phase2.py --preset society30 --worlds 128 \
    --log-every 1 --ckpt-every 50 --out runs/my_run

# Read the curves without loading any weights
python -c "
import json; h = json.load(open('runs/my_run/history.json'))['history']
for r in h[::max(1, len(h)//20)]:
    print(r['agent_steps'], r['mean_age_at_death'], r['value']['explained_variance'],
          round(r['entropy'], 2), r['deaths'])
"
```

`history.json` needs no GPU, no JAX and no matching `code_version` to read — it is plain JSON.
Start every diagnosis there, and only load weights once the curves say which hypothesis to test.
