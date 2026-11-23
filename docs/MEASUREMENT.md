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

### The policy is a habit, not a contingency — and that follows from the constant critic

From `decision_check.json` in the same run. Each rule is `times_taken / opportunities`:

| behaviour | random | untrained | **trained** |
|---|---|---|---|
| drink **when thirsty** | 0.061 | 0.048 | **0.082** |
| drink **when full** | 0.076 | 0.052 | **0.234** |
| eat when full | 0.000 | 0.015 | **0.115** |
| flee from predator | 0.037 | 0.044 | **0.020** |
| get away from predator | 0.167 | 0.165 | **0.067** |
| alive share at end | 0.208 | 0.292 | 0.500 |

The trained policy is **2.9x more likely to drink when full than when thirsty** (0.234 vs 0.082):
its drinking is anti-correlated with need. It flees predators **less than random** and gets away
from them 2.5x worse than random. Training made lookahead behaviour actively worse.

**The mechanism.** With `V(s)` constant at about -1.4 (measured: 0.2% movement across the full cold
sweep), the advantage collapses:

```
A(s,a) = r + gamma*V(s') - V(s)  ~=  r + (gamma-1)*V  =  r + const
```

So the advantage carries **no state information**, and the policy can only learn a
state-independent ranking over actions — a habit. That predicts every row above. Drinking and
eating carry immediate `Phi` gain averaged over states, so they are reinforced unconditionally.
Fleeing has no immediate reward; its whole value is a future cost avoided, which needs `V` to see
the predator, so it decays below random. `alive_share_end` still rises to 0.50 because drinking a
lot helps on average: the agent survives better while being less intelligent.

Three independent measurements fit one mechanism: a near-constant critic, a habit-shaped policy,
and specifically degraded lookahead. The October proposal already noted the compulsive drinking and
attributed it to episode noise. It is not noise.

### The credit path is 14× shorter than the window it already runs in

Measured on `gae` itself, not derived from it — `tests/test_credit_reach.py` puts a death on the
last tick of the window, holds the value at zero, and reads the advantage at the first tick. A run
has two truncations of the credit path, and only the shorter one matters:

| truncation | ticks |
|---|---|
| rollout window: `--steps` (128) plus the carried memory tail (`memory_span` = one game day, 48) | **176** |
| credit half-life at `gae_lambda` 0.95: `(0.997 × 0.95)^k` | **12.8** |

So `gae_lambda` throws away 93% of a window the run already computes and pays for. What the first
tick of the window sees of a −1.0 death on its last tick:

| `gae_lambda` | γλ | half-life | reach | sees of a −1.0 death |
|---|---|---|---|---|
| **0.95 (every run so far)** | 0.94715 | 12.8 | 12.8, λ-bound | **−0.00007** |
| 0.99 | 0.98703 | 53.1 | 53.1, λ-bound | −0.10181 |
| 0.995 | 0.99201 | 86.5 | 86.5, λ-bound | −0.24586 |
| 1.0 | 0.99700 | 230.7 | 176, window-bound | −0.59109 |

Thirst kills roughly 230 ticks after the last drink (§0, the return identity). At 0.95 a death
has to cross **18 windows** of chained bootstrapping to reach the drink that would have prevented
it, and every one of those hops is a regression onto a `V` that is itself near-constant. At 1.0 it
crosses **1.3**. `gae_lambda` is given per 30-minute tick and raised to `tick_scale`; every run so
far is at the 30-minute calibration, so `tick_scale` is 1.0 and the effective λ is the configured
one. `--gae-lambda` now exposes it, defaulting to 0.95 so nothing changed silently.

### Raising λ removes signal from the critic's target rather than adding it

**This refutes the reasoning this document gave for candidate A.** Measured by
`scripts/probe_target_info.py --seeds 8`: roll a random policy until every agent dies, record the
real reward and the real network's value each tick, compute GAE over the true 176-tick windows
bootstrapping from `V` at each cut, and ask how much of the resulting target is explained by
remaining lifetime. The network is freshly initialised on purpose — the trained critic is
near-constant, so a near-constant `V` is the regime the run is stuck in, and the probe reports its
own `V` spread (0.0928) to check that rather than assume it.

| `gae_lambda` | R² target↔lifetime | R² target↔Φ | sd(target) |
|---|---|---|---|
| **0.95 (shipped)** | **0.2692** | 0.2364 | 0.5812 |
| 0.99 | 0.2587 | 0.1905 | 1.0051 |
| 0.995 | 0.2285 | 0.1611 | 1.1223 |
| 1.0 | 0.1795 | 0.1209 | 1.2811 |

Monotone, and not seed noise: paired over 8 seeds, λ=0.95 minus λ=1.0 is **+0.0897** (t = 2.90,
df 7, p = 0.023; Wilcoxon p = 0.039; Spearman λ↔R² ρ = −0.44, p = 0.012), 6 of 8 seeds favouring
0.95. Meanwhile `sd(target)` rises 2.2×. **Raising λ buys variance, not signal.**

The mechanism is the shaping term. `R²(target, Φ)` falls with λ on the same monotone curve, and the
per-step reward contains `γΦ(s′) − Φ(s)` — which is why the verified return identity ends in `−Φ₀`:
it telescopes. Φ is a hand-designed function of the eight body scalars, i.e. a proxy for nearness to
death, and it is inherently a one-step quantity. Averaging over 176 ticks dilutes it into the noise
of a longer sum. 49% of agent-windows do contain the agent's death, so "most windows have no death
to carry" is *not* the explanation, and was checked.

Where the earlier argument went wrong: the return identity is proven for a **whole-life** return
from reset, with Φ₀ the initial potential. A windowed **mid-life** GAE target also carries `Φ(s_t)`
and a `γ^k·V(cut)` bootstrap, so it is not affine in remaining lifetime and "MC variance is lifetime
variance" does not transfer to it.

What this does **not** establish: that λ=1.0 fails in training. The probe measures the target at
initialisation. As `V` becomes informative its bootstrap carries real information, and that could
reverse the ordering. One random policy, one untrained critic, 8 agents, 8 seeds.

### Candidate C's labels: the trunk learns the trend, never the level

Measured by `scripts/probe_aux_tasks.py --seeds 6`. C would extend the world model's existing
`decode` head, which predicts the change in the eight own-state features at `DECODE_INDEX` one step
ahead. So the question is which labels survive a split that holds out **whole agent-lifetimes**.
Instrument: the GRU state `h_t` carried properly through a random-policy roll — a 128-dim random
nonlinear map of the agent's whole history, i.e. a reservoir — ridge-regressed with the penalty
tuned inside train, averaged over 5 fixed splits that every label shares.

| label | between-life var | R² all | R² within-life | R² from current needs |
|---|---|---|---|---|
| remaining lifetime | 0.19 | −2.139 ±0.31 | **+0.479 ±0.11** | −1.741 ±0.31 |
| needs change, h=1 (mean of 8) | 0.01 | 0.120 | **0.218** | 0.001 |
| needs change, h=24 | 0.05 | −0.335 | 0.180 | −0.215 |
| needs change, h=48 | 0.13 | −0.718 | 0.108 | −0.471 |
| needs change, h=200 | 0.53 | −1.699 | 0.076 | −1.881 |

Three things, all of which change what C should be:

**The level never generalizes; the trend does.** Overall R² is negative everywhere while within-life
R² is solidly positive. The trunk tracks an agent's *progression* and cannot place its *absolute
level* on a lifetime it has not seen. That is a complete explanation of the earlier probe's train
R² 0.995 against held-out R² below 0.3: it was fitting a per-lifetime offset that no per-tick
observation supplies. An MSE head on an absolute label chases that offset, which is pure gradient
noise — it is *how* a head memorizes.

**Longer horizons are monotonically worse**, 0.218 → 0.180 → 0.108 → 0.076. C was specified as the
existing head "at a longer horizon"; h=1, the horizon already implemented, is the best of the four.
The horizon extension is contraindicated.

**Per-scalar at h=24, the choice of feature matters more than the horizon.** `body_temp` reaches
within-life 0.528 where the current needs give −0.082, so the trunk supplies something real and
non-trivial. `energy` (0.374) and `sleepiness` (0.281) are beaten or matched by the current needs
alone (0.958, 0.484), so a head on them teaches the trunk nothing. And `hydration` is **0.024** —
the cause of 42% of deaths is the label that does not work.

That hydration figure survived three challenges, each of which could have overturned it:

| challenge | result |
|---|---|
| unpaired splits — every label got a different split | real bug; fixing it moved hydration 0.193 → **0.024**. All figures here are paired, 5 shared splits. |
| the head is action-conditioned (`decode(dyn(z, a))`) and the probe regressed from `h_t` alone | adding the realized action counts over the window lifts hydration only 0.024 → **0.074**, and *lowers* the 8-feature mean (0.180 → 0.156) because 84 features cost more variance than they buy |
| the label may simply not vary, making R² noise | it varies: within-life sd **0.103**, *larger* than `body_temp`'s 0.088 — comparable variance, 20× worse predictability |
| the random-init trunk simply cannot represent it, where a trained encoder could | `scripts/probe_trained_trunk.py`: training the real network on the label reaches train R² **0.816** and held-out R² **0.059** — it memorizes. `body_temp`, the positive control, reaches held-out **0.873**. |

So it is not variance, not missing actions, and **not a representation limit either** — a trained
encoder fits the training lifetimes and generalizes nothing. The information is not in the
observation at tick *t*, because hydration change over the next 24 ticks depends on whether the
agent *will* choose `drink` while on water during them, and under a random policy that is a future
exogenous event no observation can contain. `body_temp` generalizes (0.873) because it is driven by
ambient conditions already visible at *t*.

This changes what would settle it. An earlier version of this section said to re-measure with a
trained *trunk*; that has now been done and it does not help. What is needed is a trained
**policy**, where drinking becomes a function of state and the label stops being exogenous. That is
a GPU run, not another local probe.

One caveat on the figure: `probe_trained_trunk.py` prints the *best* held-out R² over epochs, which
is a max over 8 noisy values and so optimistic. The final epoch is 0.059 and the mean is about
0.04. The conclusion does not turn on which is quoted, but the max should not be read as the
estimate.

Limits: a random policy, so death timing is largely exogenous and lifetime predictability is
understated relative to a trained agent; a frozen random trunk, so these are lower bounds; 48
lifetimes from 6 seeds; and within-life R² uses each held-out lifetime's own mean, so it measures
within-life predictability rather than a predictor that could be deployed as-is.

### The target is not empty, and the critic still does not fit it

At the shipped λ the target already carries R² 0.27 of remaining lifetime, yet `value.
explained_variance` is ~0 in training (−0.4334 at update 1 of a local CPU run). So some signal *is*
delivered and the critic is not extracting it. That sharpens the root cause: the failure is not
only in what credit assignment delivers, it is in the critic fitting a noisy target — sd 0.58 with
roughly a quarter of it structured. Representational capacity stays ruled out (train R² 0.995), so
what is left is optimization: one gradient step per sleep, weighted by consolidation, against a
target whose signal-to-noise is about 1:3.

### Still open

**Root cause: the critic, confirmed with a mechanism** (above), not merely by elimination — but the
mechanism is narrower than "credit assignment" as this section first put it. The target handed to
the critic is not empty: it carries R² 0.27 of remaining lifetime at the shipped λ, and no λ
improves that. So the failure is the critic not fitting a noisy target, with representational
capacity ruled out (train R² 0.995). What is left is optimization, and `value.explained_variance`
on a live run is still the number that quantifies it — read against the scale in §5d, not against 1.

The candidates are in **§5d**, re-ranked after the λ probe: **C** (auxiliary prediction heads) is
now first, D second, A demoted after being measured against, B last. None is applied. Capacity is
still not among them.

**The λ experiment is the cautionary one.** It was the best-argued candidate here and the argument
was wrong — a whole-life return identity applied to a windowed mid-life target. Before writing code
for C, measure its premise the way A's was measured.

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

Do not weaken this. It is the only artifact that survives an interrupted run and still says what
the run learned.

### But pick `--log-every` from the update count, not out of habit

The whole file is rewritten on every logged row, so **total write I/O grows with the square of the
row count**: `row_size x rows^2 / 2`. A row is about 1.5 KB. That is harmless when updates are few
and ruinous when they are many, and the update count depends entirely on batch size:

| worlds | agent-steps/update | updates for 1e8 | rows at `--log-every 1` | total write I/O |
|---|---|---|---|---|
| 128 | 1,048,576 | 95 | 95 | ~7 MB |
| 4 | 32,768 | **3,052** | **3,052** | **~7 GB** |

7 GB of repeated rewrites to a mounted Google Drive is far slower than the byte count suggests,
because the cost is per operation. So on a small-batch run, set `--log-every` so you land near
100-400 rows: `--log-every 10` at `--worlds 4` gives 305 rows, ~450 KB and ~90 MB of I/O, which is
still denser than any curve needs.

Durability is not weakened by this. Set `--log-every` equal to `--ckpt-every` and a crash loses the
same number of updates of rows as it loses of training.

The fix if many-row runs are ever wanted: append each row to a `history.jsonl` (O(1) per row) and
rewrite `history.json` only at checkpoints. Not done.

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
within ~85 ticks (1.8 days). The rollout window is 176 ticks — `--steps` (128) plus the carried
memory tail — so `--steps` is **not** the binding truncation; `gae_lambda` is, by 14×. Any action
whose consequence arrives later than ~2 days reaches the policy **only** through `V`. Hence
`value.explained_variance`: if it sits near 0, delayed survival behaviour cannot be learned no
matter what the reward says. §0 measures the reach on `gae` itself and §5d is the fix spec.

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

**Trap: the Python traceback blames the wrong line.** In that log it ends at
`m = jax.tree_util.tree_map(np.asarray, m)` — the metric copy in `train_phase2.py`, which looks
like the logging is at fault. It is not. JAX executes asynchronously, so the host-side traceback
lands at the first place a lazy array is *read*, not at the op that failed to allocate. The
allocation that failed is named in the error itself: `executable_name='jit_update'`, the training
update. Read the `executable_name`, not the traceback, when attributing an OOM.

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

## 5d. The targeted fix: four candidates, one measurement

Every candidate targets the same confirmed defect — `V` is near-constant, so `A(s,a) ≈ r + const`
and the policy can only learn a state-independent habit. They differ in how they get
state-dependent signal into `V`. None is applied yet.

All four are judged on the same number: **`value.explained_variance`** in `history.json`, now
logged every update. **Judge it against the right scale.** Explained variance is
`1 − E[(target−V)²]/Var(target)`, and the target is noisy: at the shipped λ it carries R² 0.27 of
remaining lifetime at sd 0.58 (§0). Lifetime is only one function of state, so 0.27 is a lower
bound on what a perfect critic could reach, not a ceiling — but it is nowhere near 1, and an
earlier version of this section implied a fix should drive explained variance near 1. It should not.
A working fix moves it clearly and durably above 0. The behavioural confirmations are the second
gate — a policy that drinks 2.9× more when full than when thirsty, and flees predators less than
random, must reverse — because explained variance can rise on a critic that predicts lifetime
without the policy using it.

### A. Raise `gae_lambda` — ~~recommended first~~ **measured against, demoted**

This was recommended first. `scripts/probe_target_info.py` then measured what raising λ does to the
critic's target, and it is the opposite of what the recommendation assumed: R²(target, lifetime)
falls monotonically from 0.269 at λ=0.95 to 0.179 at λ=1.0 (paired p = 0.023) while `sd(target)`
rises 2.2×. λ buys variance, not signal. §0 has the table, the mechanism (the one-step shaping term
`γΦ(s′) − Φ(s)` is what carries the state-dependent content, and a long λ dilutes it) and the error
in the original argument (the return identity is a whole-life result and does not transfer to a
windowed mid-life target).

What survives is only the arithmetic: at λ=0.95 the credit half-life is 12.8 ticks inside a 176-tick
window, so a consequence 230 ticks away reaches `V` through 18 windows of chained bootstrapping.
That is still true, still bad, and no longer evidence that raising λ is the remedy — a longer credit
path is worth nothing if what travels down it is noise.

**Do not run A on the old grounds.** It stays on the list because the probe sees only
initialisation: it cannot say what happens once `V` is informative enough for its bootstrap to
carry real information, and that is exactly the regime a long λ is supposed to exploit. If it is
run, it is as a cheap one-number experiment with an explicit prior that it may well do nothing, not
as the targeted fix. `--gae-lambda` and the paired-root machinery are in place either way.

- **Cost**: zero. The flag exists; no new parameters, no new compute, no new loss term.
- **Blast radius**: learning dynamics of every run. Nothing structural. Fully reversible.
- **Residual risk**: at λ=1 the *policy* advantage also becomes Monte-Carlo, which does raise
  policy-gradient variance for real — and the probe now measures that rise directly.
- **If run anyway**: 1.0, because it is the only value where the 176-tick window rather than λ is
  the binding truncation, and 0.995 still discards half of it. If 1.0 destabilizes the policy, 0.99
  is the fallback and still a 4× improvement.

#### Running A without destroying the comparison

`--gae-lambda` is forwarded by `experiment.py` to `train_phase2.py`, and a design's own `extra`
overrides it because `extra` comes last on the command line.

A run's λ is **not** part of `--root` and **not** part of `code_version` (which hashes
`lifesim/*.py`, `train_phase2.py` and `continuous_world.py`, not the arguments). So the baseline and
the λ run land in the same `<root>/validate/<design>/` folder and the second silently resumes or
overwrites the first — and `experiment.py`'s existing guard will not catch it, because that guard
only refuses to mix *code versions* in one root. **Use a separate `--root` per λ.**

The notebook now does this for you. Cell 12 has one knob, `LAM`, and derives the root from it:
λ=0.95 keeps the baseline's existing path, and anything else appends `_lam1p0`, `_lam0p99` and so
on. So the paired runs are: cell 12 with `LAM = 0.95`, cell 16; then cell 12 with `LAM = 1.0`,
cell 16 again. Cell 12 also derives `LOG_EVERY` from the update count — 3,051 updates at 4 worlds
against 95 at 128, so a fixed `--log-every 1` is right at one world count and gigabytes of
quadratic rewriting at the other (§1). Cell 4 refuses an old zip that has no `--gae-lambda`.

`experiment.py` also refuses it now, which covers the paths the notebook does not: `hprc/*.slurm`
and anything launched by hand pass `--root` directly. `stale()` compares a run's `gae_lambda`
against the one being asked for, on the same condition as the code-version check — only where
weights exist and would really be reused. A run made before the flag existed has no such key and
counts as 0.95, which is what it actually used. A code mismatch still reports as a code mismatch
and not also as a λ mismatch.

### B. Lengthen the rollout window (`--steps`)

Raises the ceiling that A would then bind against: 176 → 304 at `--steps 256`.

Only worth doing **after** A, and only if A's measurement shows the window has become the binding
constraint. Memory scales linearly in `--steps`, and the Colab profile already OOMs at 128 with
`--agent-chunk 8`, so this is the expensive candidate on current hardware.

### C. Auxiliary prediction heads — now the leading candidate

Predict needs at longer horizons, or remaining lifetime, as a supervised head. Labels are free from
the rollout. This is the proposal's own §5 "consequence models", and in the literature it is UNREAL
/ general value functions (Sutton & Barto ch17, auxiliary tasks).

- **Cost**: a head, a loss term, a coefficient to tune. Changes `policy.py` and `ppo_multi.py`.
- **Why it moved up.** This section used to say C "shapes the representation; it does not change the
  critic's target", and that C without A was likely insufficient. The first half is still true and
  is now the argument *for* it. The probe showed the credit path is not what puts lifetime
  information into the target — no λ does — so there is no version of A that supplies the signal C
  would supply. A supervised head bypasses the credit path, the bootstrap and the noise in one step.

**C survives its premise check, but not as specified.** `scripts/probe_aux_tasks.py` measured which
labels generalize across held-out lifetimes (§0), and the answer reshapes the candidate:

1. **Predict a level-free target.** Within-life R² is positive everywhere and overall R² is negative
   everywhere: the trunk learns an agent's progression and cannot place the absolute level on an
   unseen lifetime. A head on an absolute label chases a per-lifetime offset no observation
   supplies, which is how the earlier probe got train R² 0.995 and held out below 0.3. Center within
   the life, or predict a difference or a monotone progress fraction — the label is known in
   hindsight on a completed rollout, which is all a supervised head needs.
2. **Do not extend the horizon.** Within-life R² falls monotonically with h: 0.218 at h=1, down to
   0.076 at h=200. C was specified as the existing `decode` head at a longer horizon. That is the
   wrong direction, and h=1 is already implemented.
3. **Pick the feature on the measurement, not on intuition.** `body_temp` is the one own-state
   feature where the trunk clearly beats the current needs (within-life 0.528 against −0.082).
   `energy` and `sleepiness` are matched or beaten by the current needs alone, so a head on them
   adds no pressure.

   **`hydration` is out of C's spec.** It scores 0.024 from the random trunk and survived four
   challenges (§0): unpaired splits, action-conditioning, label-has-no-variance, and — the one that
   would have put it back in — a trained encoder, which reaches train R² 0.816 and held-out 0.059
   while the `body_temp` control reaches 0.873. Under a random policy the label is partly
   exogenous: hydration change over 24 ticks depends on whether the agent *will* choose `drink`
   while on water, which no observation at *t* contains. So a hydration head would be trained on
   noise. It can only be reconsidered after a run with a trained policy, where drinking becomes
   state-determined — and that is the single measurement most likely to change C's value, because
   thirst is 42% of deaths.
4. **The label worth adding is progression toward death** (within-life R² 0.479, against −1.741 from
   the current needs). It is the strongest signal measured, the existing head does not predict it,
   and it is the thing `V` actually needs. That — not a longer horizon on the needs head — is what
   C should be.

- **Why it is still not applied.** It now has a measured premise and a shape, but a coefficient to
  tune and two files to change. The spec is the next artifact; the code is not.

### D. A critic input path not shared with the 532 language dimensions

`feat_dense` puts 532 heard/read symbols and the 8 scalars that determine reward through the same
64-unit bottleneck (§6). The rationale would be gradient interference, not capacity.

**Last, and not yet justified.** Capacity was measured fine (train R² 0.995; encoder effective rank
34/128, condition number 1.3e3). Interference is unmeasured, so this candidate rests on a
hypothesis nothing in §0 supports. It needs its own measurement before it needs an implementation.

### E. Φ is carrying the signal — so the Φ experiment points the other way

Not a fix, but the measurement reorders a decision already on the list. `R²(target, Φ)` is 0.236 at
the shipped λ and falls with λ on the same curve as lifetime, so the one-step shaping term is the
main state-dependent content of the critic's target. Potential-based shaping provably leaves the
optimal policy unchanged and telescopes out of the full return (Ng, Harada & Russell 1999) — it is
affecting learning dynamics only, and right now it is most of what the critic has to go on.

An earlier decision was to keep Φ and "measure it separately", with the implied experiment being to
disable it. That experiment now has a predicted answer and it is the uninteresting direction:
removing the main source of state-dependent signal should make things worse. The informative
experiment is the opposite one — whether Φ can be made a *better* proxy for nearness to death, or
whether C's supervised head makes Φ redundant. Neither is specified.

### Why this order

**C first**, because it is the only candidate whose premise is measured rather than argued: the
signal is absent from the target at every λ, the labels are free, and the architecture already fits
them to R² 0.995. **D second**, because the sharpened root cause — signal present at R² 0.27,
explained variance ~0, capacity ruled out — now points at optimization and interference, which is
exactly D's hypothesis; it needs its own measurement first, and it is no longer "nothing in §0
supports it". **A** is cheap enough to run as a side experiment but has been measured against and
must not be presented as the fix. **B** still waits on A.

One change at a time, or the result cannot be attributed — which is the whole reason the diagnosis
took this long. And the reason A was demoted is worth keeping in view: it was the best-argued
candidate on this list and the argument was wrong. Measure the premise of C before writing its
coefficient, not after.

---

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
