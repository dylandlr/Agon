# Design audit: does Agon's learning machinery fit the intelligent-agent goal?

Date: 2026-10-23. Branch `ecology`. Evidence: the `intel_9961b53f23dd_society30_w4` run
(100M agent-steps, caveman slate, world model on), its decision report over five continuous-world
chunks (156 sampled moments read in full), the decision check against random and untrained agents,
and the code in `lifesim/policy.py`, `lifesim/ppo_multi.py`, `lifesim/env.py`.

## Verdict in one paragraph

The agents learned a weak foraging-and-sleep policy that survives a little better than chance
(10% of founders alive after 10 days against 2% for random agents). They did not learn to act on
their needs reliably (only 4 of 33 sampled hungry moments chose eat or seek food; 17 of 17 sampled
drinks happened at full water), and the world model's "ideas" had no effect (every option imagined
within a few thousandths of the same return; trust never left zero at the population level). The
recurrent cell is not the cause. The causes are, in order: too little experience per brain, two
scale defects in how ideas reach the policy (fixed today), and an evaluation that until yesterday
could not have told a thinking agent from a lucky one.

## 1. The learning problem is starved, by design

Every individual owns its own network (`ppo_multi.py`: "one network per individual"). The run's
"100M agent-steps" counts all 64 slots in 4 worlds every tick, dead slots included (64 x 4 x
390,528 ticks = 1.0e8). About 8 were alive at a time, so alive experience was about 12.5M steps
in all, spread over mostly the 8 founder brains: about 1.5M steps each, lived as 2,289 lives
averaging 114 days. A slot filled late in training saw far fewer. A 128-unit GRU over an 11x11x64 window plus about 30 body features,
trained by PPO from scratch, is not going to learn planning-grade behaviour from 1M on-policy
steps. Imitation (`imitation_coef` 0.5) shares behaviour but not gradients, and the imitation
network is run without memory, so it transfers reflexes, not strategies.

This is a design choice with a purpose (individual brains make individual lives, inheritance and
culture possible), but it is the first thing to question when asking whether the agents can be
intelligent. Options, least to most invasive:

- Shared trunk, individual heads. One encoder + GRU for the population, with each agent's
  genotype (and an identity embedding) as input and its own recurrent state; per-agent policy and
  value heads, or a per-agent low-rank adapter. Individuality stays in the state, genes and heads;
  the costly perception learning is pooled 64x.
- Lineage sharing. Founders share one brain per lineage; children inherit (the continuous world's
  `inherit_brain` already does parent/innate inheritance).
- Keep per-individual brains but pretrain the trunk on the shared data for the first part of a
  run, then split. Cheap to try with the current code.

The shared-trunk variant is the one to run first, as an ablation against the current design on
the same seeds. If it closes most of the gap, the per-individual design is the bottleneck.

## 2. The GRU is not the problem; neither is depth

An LSTM in place of the GRU changes the gate arithmetic and adds a cell state. With hidden size
128 and the data budget above, there is no reason to expect a difference that survives three
seeds; the GRU's known weakness is very long dependencies, and the horizon that matters here
(getting to water before thirst kills, days away at 30-minute ticks) is handled by the critic and
the discount (gamma 0.997, about a week), not by the recurrent memory. Replace the cell only if an
ablation shows memory failures (e.g. an agent forgetting where water was within a day). A
transformer over the history would be a larger change for the same uncertain gain, and would
cost far more per step across 64 brains.

Where more capacity would help is the encoder, and only once data is pooled (section 1). The
conv encoder is three 3x3 layers to 32 channels; it is small for 64 input channels, several of
which (the room, trees by kind, animal bonds) were added after it was sized.

## 3. The world model's ideas: two scale defects, fixed today

The design is sound: imagine the k most likely actions for h ticks in a learned dynamics model
over the belief, score each by expected reward plus the critic's value, and nudge the policy
toward the best by a learned trust. The report showed it inert. Two reasons, both of scale:

- `bias = trust * (q - mean)`. Imagined returns differ by hundredths (a few ticks of a slowly
  changing life). Moving a logit by one unit needed trust near 100, and trust's own gradient
  carried the same factor, so it could not get there. Fixed: standardise `(q - mean)` over the
  valid candidates; trust 1 now means one standard idea's worth of logit.
- The own-state head predicted the next level of each need. The belief already holds the level,
  so an action's effect was a sliver of the target and the dynamics had little reason to depend on
  the action. Fixed: it predicts the change over the tick, x10, so what the action did is the
  whole signal. (The playback's expected levels are now the current level plus the summed changes;
  `tests/test_world_model.py` passes.)

Still open: the dynamics is trained only on the action actually taken, so counterfactuals are
never checked against reality. Add a metric to the run log: the standard deviation of imagined
returns across candidates divided by their mean absolute value (how much the model distinguishes
options), and a probe: after an imagined `eat`, is the predicted food change larger than after an
imagined `rest`? Report trust per agent, not the population mean, which hid values from -0.14 to
+0.07.

## 4. Reward: right shape, two weak spots

`reward_mode="survival"`: +0.01 per tick alive, -1 on death, plus the potential-based change in
felt well-being (Ng, Harada & Russell 1999), so no loop of acts pays. Dispositions (pain
aversion, empathy, envy, patience as the discount) are heritable weights on it. That is the
correct substrate for "the reward function is the agent's own and can be gamed" and for
disposition-dependent risk appetite.

Weak spots. (a) Acting on a full need costs nothing but the tick: drinking at full water is free,
so there is no gradient against the waste the report shows. A small energy or time cost per act,
or a need curve that goes flat near full (it is `1 - deficit^2`, which still rewards the last
drop), would give one. (b) Danger is felt only as realised health loss and death. A predator
beside you costs nothing until it bites, so "flee when a threat is near" has to be learned from
rare, high-variance deaths. Stress already rises with danger (`STRESS_SOURCES` has "danger") and
stress is in well-being, so the channel exists; check its magnitude against a bite.

## 5. Evaluation: what could not be seen until this week

- The flee rule counted tame, resting and other-layer predators as threats; none can attack. The
  0.53x "flee" result was not a measurement. Fixed: only live threats; also "ended farther from
  it", since walking away is fleeing too.
- `rest_at_night` counted sleeping agents, whose choices are constrained. Fixed: awake only.
- The 1.9x drink and 5.2x eat lifts rested on one seed's crisis (seed 1 supplied 536 of 544
  thirsty ticks). Report per-seed rates and a bootstrap interval, and raise the days or seeds
  until each rule has thousands of ticks in every seed.
- No waste measure existed. Added: drink-when-full and eat-when-full rates (lower is better).
- No ablation has been run: one training seed, world model always on. Minimum matrix: world model
  on/off x shared-trunk/per-individual x 3 seeds, 100M steps each (about 4.7 h per run on an A100
  with 4 worlds).
- The persistence gate (400 days from 8 founders) tests a band too small for the question; the
  one continuous world died in year 3 with one conception from 56 matings, which is within the
  design odds (0.039 per mating). Run 3 seeds with 16 founders before reading anything into it.

## 6. On LLMs and "more deep learning"

An LLM in the decision loop would import human priors about fire, cooking, kinship and law,
which is exactly what Agon is meant to measure the absence or appearance of. (Paragon, the
designed-society arm, is shelved for now, 2026-10-23.) The one place for an LLM is analysis:
naming what agents did, summarising a life, proposing hypotheses from the event log. Within Agon, the
deep-learning changes worth making are the pooled trunk (section 1), a sized-up encoder once data
is pooled, and a dynamics model trained with some counterfactual signal (section 3). Nothing here
calls for a different family of model.

## The stall, measured (2026-10-23, evening)

Per sixth of the validate run (306 windows): deaths 1,025, 325, 280, 177, 209, 276; policy
entropy 0.116 then 0.07-0.08 (of 4.4 nats over 84 actions); mean age at death 1,346, 2,964,
3,679, 5,912, 5,567, 3,988 ticks (births per sixth 0, 5, 21, 19, 22, 28: infants pull the last
figure down). By cause, thirst 345 -> 73 in the first sixth then 30-70; cold 271, 123, 85, 49,
54, 79. Winter behaviour was learned, partly: huddling on winter nights 0.30, 0.40, 0.55, 0.63,
0.61, 0.64; seek_shelter 18 -> 89 per 1,000 ticks. The scripted band (shelter + huddle, no fire)
has no cold deaths; the trained agents do it about two thirds of the time and stopped improving
there. Reading: the policies went near-deterministic a sixth of the way in, kept what they had,
and explored no further. Two things hold them: a small fixed entropy bonus (0.005) and the
retention loss (0.1) toward their own past behaviour. Added: an entropy floor
(`--entropy-target`, coef = ent_coef x exp(gain x (target - entropy))), and the ablations
`caveman_entropy_floor` and `caveman_no_retention` in `experiment.py validate`; a stall panel in
the notebook. Not the world and not the brains.

## Note on animals (2026-10-23)

Animals now have a temperament, are neutral to strangers once tame, and aim their anger: a
grudge against who struck them or who struck their person beside them. When they did not see
the blow (struck in their sleep) or several struck at once, they blame the one they trust least,
by bond. None of this is scripted behaviour; it is state the animal carries.

## Order of work

1. Ablation: shared trunk vs per-individual brains, world model on/off, 3 seeds. Decides the
   architecture question with data instead of opinion.
2. Idea-discrimination metric and per-agent trust in the run log; rerun the decision check with
   the fixed rules on the existing checkpoint.
3. Cost for acting on a full need; check the danger-to-stress magnitude.
4. Persistence: 3 seeds x 16 founders, 400 days.
5. Group-level state (faction, shared belief) only after agents reliably keep themselves alive;
   there is nothing for a social response to emerge from before that.
