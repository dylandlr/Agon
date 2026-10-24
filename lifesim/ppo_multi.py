"""Phase 2 trainer: multi-agent recurrent PPO with one network per individual.

Setup: a population of N individuals, each with a fixed genotype, appearance
and its own policy parameters (stacked along a leading axis of size N). The
same N individuals live in B replica worlds in parallel, so each individual
gathers B times more experience per tick while staying one individual with
one brain and one set of genes.

Per-agent learning: gradients are clipped per agent, Adam runs element-wise,
and each agent's update is scaled by its own learning rate,
lr_i = lr0 * lr_scale_i (the plasticity gene). Optionally the discount factor
comes from the patience gene.

Dead agents stay dead until their world resets; their steps are masked out of
the loss. An agent's death is terminal for its own returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from . import env as E
from . import genotype as gt
from . import social
from .config import EnvConfig
from .policy import DECODE_INDEX, make_network, sample_outputs, split_logits


@dataclass(frozen=True)
class MAConfig:
    num_worlds: int = 16
    num_steps: int = 128
    total_ticks: int = 200_000      # world-ticks per replica (x num_worlds for env steps)
    lr: float = 3e-4
    gamma: float = 0.997
    trait_gamma: bool = False       # use each agent's patience gene as its discount
    per_agent_lr: bool = True       # scale updates by the lr_scale gene
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.005
    # An entropy floor (0 = off): the entropy bonus grows as a policy's entropy falls under the
    # target and shrinks above it, coef = ent_coef * exp(gain * (target - entropy)), per agent and
    # step. The validate run collapsed to 0.07 nats (of 4.4 possible) a sixth of the way in and
    # learned nothing after; a floor keeps each brain trying things without making it dither.
    entropy_target: float = 0.0
    entropy_gain: float = 10.0
    # Symbols said (4 x 32: up to 13.9 nats) and written: far more entropy than the 84 actions (~4.4).
    # At full weight the entropy bonus alone pushes agents to talk (randomly) more and more;
    # a smaller weight makes speech earn its place.
    msg_ent_coef: float = 0.0005
    # Where speech and writing branch off the brain (policy.ActorCritic.msg_net): "branch",
    # "stopgrad" or "separate"; msg_frozen keeps them at their initial weights (a control: what
    # speaker-side measures show with no learning at all).
    msg_net: str = "branch"
    msg_frozen: bool = False
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4        # split over worlds
    # Memory: 0 computes every agent's loss at once (fastest). k > 0 computes it k agents at a time
    # and recomputes activations in the backward pass: the same gradients, with peak memory about
    # k/N of the all-at-once pass, for ~30% more compute (a 64-agent society30 update asks ~19 GB
    # all at once, beyond a T4 or a small CPU box).
    agent_chunk: int = 0
    hidden: int = 128
    encoder: str = "conv"
    imitation_coef: float = 0.5     # 0 disables social learning
    imitation_bias: str = "success" # uniform | success | kin
    imitation_bias_strength: float = 5.0
    imitate_positive_only: bool = True
    alpha: float = 0.0              # prosociality: train on (1 - alpha) * own + alpha * group mean reward
    # Off by default: with them on, speech more than doubled over 614k steps (133 vs 309 per
    # 1,000 agent-ticks) with no other benefit; talkers got copied, so talking spread.
    # Keep them as an experimental condition for language, not a default.
    speech_attention: float = 0.0   # speaking demonstrators are more likely to be watched
    teach_bonus: float = 0.0        # and their demonstrations count (1 + bonus) times as much
    # Sleep consolidates learning (complementary learning systems: a fast store of the day, slow
    # learning while asleep). See consolidation_plan() and learn_core():
    # - Day store: experience waits, across updates, for the deep sleep that follows it, for up to
    #   memory_span ticks. Each step is learned exactly once: in full once consolidation_ticks of
    #   deep sleep have followed it, or, if it leaves the store first, in proportion to the sleep it
    #   got (sleep deprivation costs memory). A life that ends in death counts in full: with one
    #   network per slot, learning from deaths stands in for selection.
    # - Replay favours surprise: consolidated steps are reweighted by |TD error|^surprise_power,
    #   mixed with uniform weight by surprise_priority (total weight unchanged).
    # - Old memories are interleaved: a long-term store of reservoir_size moments per agent (what
    #   it saw, its memory state, what it chose then); each sleep pulls the policy toward what it
    #   did there (KL, retention_coef), so new days do not overwrite old skills; a share
    #   reservoir_refresh of the store is replaced by newly consolidated moments at each update.
    # sleep_learning=False turns all of it off (plain PPO on each rollout).
    #
    # When a day does not fit in one rollout (fine ticks: a 1-minute day is 1,440 steps), the
    # day's experience cannot be kept for its night. Then the day is kept as what it would teach
    # (consolidation "pending", chosen by "auto" whenever ticks are shorter than the 30-minute
    # calibration, i.e. tick_scale < 1): each update computes, at the unchanged weights,
    # the gradient of what it lived (the weights only change in sleep, so the gradient is exact
    # when sleep comes), adds it to a pending change per agent that fades with a half-life of a day
    # (what is never slept on is forgotten), and applies the share earned by deep sleep since the
    # last update (consolidation_ticks of deep sleep, or a death, earns all of it; averaged over
    # the worlds the agent lives in). One gradient step per sleep, not several epochs: the day's
    # data are not kept. The long-term store and its retention pull work as before.
    consolidation: str = "auto"     # auto | window | pending
    day_ticks: int = 48             # ticks in a game day (set from the world by the trainer)
    # Tick length as a share of the 30-minute calibration (1/30 at 1-minute ticks). gamma, the
    # patience gene, gae_lambda and consolidation_ticks are given per 30 minutes and converted.
    tick_scale: float = 1.0
    sleep_learning: bool = True
    consolidation_ticks: float = 8.0
    memory_span: int = 48
    surprise_priority: float = 0.5
    surprise_power: float = 0.5
    reservoir_size: int = 256
    retention_coef: float = 0.1
    reservoir_refresh: float = 1.0 / 16
    # World model and ideas (policy.py): a dynamics network over the agent's own beliefs, trained
    # on its own experience (and, like everything, only once slept on); imagination nudges choices
    # by a learned trust that starts at 0.
    world_model: bool = False
    wm_coef: float = 1.0
    # One trunk for the population: the encoder, the GRU and the world model (perception and the
    # physics of the world, the same for everyone) learn from everyone's experience; the policy,
    # value, message heads and trust stay each agent's own. Done by tying: every agent keeps a
    # copy, started identical, and the trunk's update is the mean over the agents who learned
    # this round, so the copies never part. Checkpoints, inheritance and playback see the same
    # shapes as before. The ablation against per-individual brains (claude/research/design-audit).
    share_trunk: bool = False
    seed: int = 0

    @property
    def num_updates(self) -> int:
        return self.total_ticks // self.num_steps


class Transition(NamedTuple):   # all fields (N, B, ...)
    reset: jnp.ndarray
    done: jnp.ndarray
    valid: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray       # (N, B, 4) log-probs of action, aim, said, written (where each acted)
    obs: jnp.ndarray
    mask: jnp.ndarray
    msg: jnp.ndarray            # (N, B, OUT_SLOTS) [said..., written..., aim]
    spoke: jnp.ndarray          # (N, B) bool
    pos: jnp.ndarray            # (N, B, 2) where each agent stood (for who-sees-whom)
    health: jnp.ndarray         # (N, B) visible health (for the success bias)
    night: jnp.ndarray          # (N, B) bool: dark at this tick (limits what observers see)
    died: jnp.ndarray           # (N, B) bool: the agent died at this tick
    sleep_q: jnp.ndarray        # (N, B) depth of sleep this tick, 0 awake .. 1 deep
    learn_w: jnp.ndarray        # (N, B) how much of this experience is learned (set after the rollout)
    boot: jnp.ndarray           # (N, B) value of the state after a time-limit cut-off (0 if none or died):
                                # running out of clock is not death, so the future still counts
    stage: jnp.ndarray          # (N, B) int: 0 new; 1 carried over, not yet learned (learn now, finally);
                                # 2 carried over and already learned (context for memory and returns only)


class Reservoir(NamedTuple):    # long-term memory: (N, R, ...) per agent
    obs: jnp.ndarray
    mask: jnp.ndarray
    carry: jnp.ndarray          # recurrent state going into that moment
    reset: jnp.ndarray
    probs: jnp.ndarray          # (N, R, A) action probabilities when it was stored
    filled: jnp.ndarray         # (N, R) bool


class Memory(NamedTuple):
    tail: Transition            # (C, N, B, ...) the last memory_span ticks of the previous update
    tail_carry: jnp.ndarray     # (N, B, H) recurrent state at the start of the tail
    reservoir: Reservoir
    pending: object = None      # pending consolidation: a per-agent change not yet slept on (params-like)


def consolidation_plan(done, died, sleep_q, stage, consolidation_ticks, span):
    """(T, ...) arrays over a window (carried tail first, then new steps) -> (weight, finalized,
    consolidated, next_stage).

    weight: how much each step is learned in this update (exactly once over its life in memory).
    A new step consolidated by the end of the window (enough deep sleep after it, or its life
    ended in death) is learned in full; a new step in the window's last `span` ticks that is not
    yet consolidated waits (weight 0, next_stage 1); an older one is learned in proportion to the
    sleep it got. A carried step that waited (stage 1) is learned now in proportion to its sleep;
    one already learned (stage 2) only gives context. next_stage is for the last `span` steps."""
    graded = consolidation_weights(done, died, sleep_q, consolidation_ticks)
    T = done.shape[0]
    t = jnp.arange(T).reshape((T,) + (1,) * (done.ndim - 1))
    consolidated = graded >= 1.0 - 1e-6
    in_tail = t >= T - span
    new_w = jnp.where(consolidated, 1.0, jnp.where(in_tail, 0.0, graded))
    weight = jnp.select([stage == 0, stage == 1], [new_w, graded], 0.0)
    finalized = (stage == 1) | ((stage == 0) & (consolidated | ~in_tail))
    next_stage = jnp.where((stage == 0) & ~finalized, 1, 2)[T - span:] if span > 0 else stage[:0]
    return weight, finalized, consolidated, next_stage


def gae(reward, value, done, boot, last_val, gam, lam):
    """Generalized advantage estimates over (T, ...) arrays. done ends a life or a world; boot is
    the value to continue from when it ended by the clock rather than by death (0 otherwise)."""
    def step(c, x):
        adv, next_val = c
        r, v, d, b = x
        delta = r + gam * (next_val * (1 - d) + b) - v
        adv = delta + gam * lam * (1 - d) * adv
        return (adv, v), adv

    _, adv = jax.lax.scan(step, (jnp.zeros_like(last_val), last_val), (reward, value, done.astype(jnp.float32), boot),
                          reverse=True)
    return adv


def consolidation_weights(done, died, sleep_q, consolidation_ticks):
    """(T, ...) arrays -> (T, ...) learning weight per transition: the deep sleep that follows it
    within the same life (sleep at the tick itself included), divided by consolidation_ticks and
    capped at 1. A life that ends in death within the window counts in full."""
    def step(c, x):
        slept_after, dies = c
        d, dd, q = x
        slept_after = jnp.where(d, 0.0, slept_after) + q
        dies = jnp.where(d, dd, dies)
        return (slept_after, dies), jnp.where(dies, 1.0, jnp.minimum(slept_after / consolidation_ticks, 1.0))

    init = (jnp.zeros(done.shape[1:]), jnp.zeros(done.shape[1:], bool))
    return jax.lax.scan(step, init, (done, died, sleep_q), reverse=True)[1]


class Population(NamedTuple):
    genotype: gt.Genotype       # (N,) per trait
    appearance: jnp.ndarray     # (N, 3)


def _per_agent_clip(grads, max_norm):
    sq = sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1)
             for g in jax.tree_util.tree_leaves(grads))
    scale = jnp.minimum(1.0, max_norm / (jnp.sqrt(sq) + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), grads)


def _scale_per_agent(updates, lrs):
    return jax.tree_util.tree_map(lambda u: u * lrs.reshape((-1,) + (1,) * (u.ndim - 1)), updates)


# The modules of policy.ActorCritic that make up the trunk (MAConfig.share_trunk)
TRUNK = ("conv1", "conv2", "conv3", "win_dense", "feat_dense", "in_dense", "mid_dense", "gru",
         "dyn1", "dyn2", "rew_hidden", "rew_out", "dec_hidden", "dec_out")


def _is_trunk(path):
    return any(getattr(k, "key", None) in TRUNK for k in path)


def _tie_trunk(tree, weights=None):
    """Trunk leaves: the mean over the agent axis (weighted; by default over the agents whose
    leaf is not all zero, i.e. who learned this round), broadcast back to every agent. Other
    leaves unchanged."""
    def one(path, x):
        if not _is_trunk(path) or x.ndim < 1:
            return x
        w = weights
        if w is None:
            w = (jnp.abs(x).reshape(x.shape[0], -1).max(1) > 0).astype(x.dtype)
        w = w.reshape((-1,) + (1,) * (x.ndim - 1))
        mean = (x * w).sum(0, keepdims=True) / jnp.maximum(w.sum(0, keepdims=True), 1.0)
        return jnp.broadcast_to(mean, x.shape)
    return jax.tree_util.tree_map_with_path(one, tree)


def _copy_trunk_from_first(params):
    """Every agent's trunk set to agent 0's: identical copies to start the tie."""
    return jax.tree_util.tree_map_with_path(
        lambda path, x: jnp.broadcast_to(x[:1], x.shape) if _is_trunk(path) and x.ndim >= 1 else x, params)


def _swap_axes_nb(x):
    """(B, N, ...) <-> (N, B, ...)"""
    return jnp.swapaxes(x, 0, 1)


def make_train(env_cfg: EnvConfig, cfg: MAConfig):
    n, b = env_cfg.n_agents, cfg.num_worlds
    net = make_network(env_cfg, cfg.hidden, cfg.encoder, messages=True, world_model=cfg.world_model,
                       msg_net=cfg.msg_net)
    A, M, L, V = E.N_ACTIONS, E.EXTRA_LOGITS, E.MSG_LEN, E.VOCAB
    n_win = env_cfg.view_size ** 2 * E.N_OBS_CHANNELS
    dec_idx = jnp.asarray(DECODE_INDEX) + n_win               # own-state features the model predicts

    def full_mask(mask):
        """Action mask plus all-True entries for the aim and symbol logits."""
        return jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (M,), bool)], axis=-1)

    speak_on = 0.0 if (env_cfg.channel == "none" or cfg.msg_frozen) else 1.0
    write_on = 0.0 if (not env_cfg.writing or cfg.msg_frozen) else 1.0

    def live_masks(action, out):
        """Where each output had an effect: the aim only on a strike; the symbols said after the first
        only when the first was not silence; the written symbols only on WRITE. A factor with no
        effect is left out of the ratio, the entropy bonus and imitation (it would only add noise)."""
        f32 = jnp.float32
        striking = ((action >= E.STRIKE_N) & (action < E.STRIKE_N + 8)).astype(f32)
        said = jnp.concatenate([jnp.ones(out.shape[:-1] + (1,), f32),
                                jnp.broadcast_to((out[..., :1] > 0).astype(f32), out.shape[:-1] + (L - 1,))], -1) * speak_on
        written = jnp.broadcast_to(((action == E.WRITE).astype(f32) * write_on)[..., None], out.shape[:-1] + (L,))
        return striking, said, written

    def factor_logps(logits, action, out):
        """Log-probs of the four outputs (action, aim, said, written), each only where it had an
        effect (live_masks), and the pieces their entropies need."""
        act, aim_l, sym = split_logits(logits)
        la, lk, ls = jax.nn.log_softmax(act), jax.nn.log_softmax(aim_l), jax.nn.log_softmax(sym)
        striking, said, written = live_masks(action, out)
        tok = jnp.take_along_axis(ls, out[..., :2 * L, None], -1)[..., 0]                # (..., 2L)
        lp = jnp.stack([jnp.take_along_axis(la, action[..., None], -1)[..., 0],
                        jnp.take_along_axis(lk, out[..., 2 * L:2 * L + 1], -1)[..., 0] * striking,
                        (tok[..., :L] * said).sum(-1), (tok[..., L:] * written).sum(-1)], -1)
        live = jnp.stack([jnp.ones_like(striking), striking, said[..., 0], written[..., 0]], -1)
        return lp, (la, lk, ls, live, said, written)
    adam = optax.adam(1.0, eps=1e-5)          # direction only; per-agent lr applied after

    def apply_agent(p, carry, obs, reset, mask):
        """One agent's network over its B worlds. obs (T, B, D)."""
        return net.apply(p, carry, (obs, reset, full_mask(mask)))

    v_apply = jax.vmap(apply_agent)            # over agents: params, carry (N,B,H), x (N,T,B,..)

    def reset_worlds(key, pop: Population):
        return jax.vmap(lambda k: E.reset(k, env_cfg, pop.genotype, pop.appearance))(
            jax.random.split(key, b))

    def observe(states):
        obs = jax.vmap(lambda s: E.observe(s, env_cfg))(states)         # (B, N, D)
        mask = jax.vmap(lambda s: E.action_mask(s, env_cfg))(states)    # (B, N, A)
        return _swap_axes_nb(obs), _swap_axes_nb(mask)

    def init(key):
        k_pop, k_app, k_net, k_env, k_run = jax.random.split(key, 5)
        geno = gt.sample(k_pop, n)
        if env_cfg.archetypes:          # the learner reads some traits (learning rate, social learning)
            from . import archetypes as arch
            geno, _ = arch.apply(geno, jnp.zeros(n, jnp.int32), env_cfg.archetypes,
                                 env_cfg.n_founders if env_cfg.n_founders > 0 else n)
        pop = Population(geno, jax.random.uniform(k_app, (n, E.N_APPEARANCE)))
        states = reset_worlds(k_env, pop)
        obs, mask = observe(states)
        carry = jnp.zeros((n, b, cfg.hidden))
        dummy = (jnp.zeros((1, b, obs.shape[-1])), jnp.zeros((1, b), bool),
                 jnp.ones((1, b, A + M), bool))
        params = jax.vmap(lambda k: net.init(k, jnp.zeros((b, cfg.hidden)), dummy))(
            jax.random.split(k_net, n))
        if cfg.share_trunk:
            params = _copy_trunk_from_first(params)
        opt_state = adam.init(params)
        reset_flag = jnp.ones((n, b), bool)
        return (params, opt_state, states, obs, mask, reset_flag, carry, k_run, empty_memory(b, obs.shape[-1])), pop

    PENDING = cfg.sleep_learning and (cfg.consolidation == "pending"
                                      or (cfg.consolidation == "auto" and cfg.tick_scale < 1.0))
    C = min(cfg.memory_span, cfg.num_steps) if cfg.sleep_learning and not PENDING else 0   # carried steps
    R = cfg.reservoir_size if cfg.sleep_learning else 0

    def param_zeros(nb, d):
        """A zero change for every agent's weights (the pending store)."""
        dummy = (jnp.zeros((1, nb, d)), jnp.zeros((1, nb), bool), jnp.ones((1, nb, A + M), bool))
        shapes = jax.eval_shape(lambda k: jax.vmap(lambda kk: net.init(kk, jnp.zeros((nb, cfg.hidden)), dummy))(
            jax.random.split(k, n)), jax.random.PRNGKey(0))
        return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), shapes)

    def empty_memory(nb, d):
        """No tail yet (C context-only steps that reset memory) and an empty long-term store."""
        z = lambda *shape, dt=jnp.float32: jnp.zeros(shape, dt)
        tail = Transition(jnp.ones((C, n, nb), bool), z(C, n, nb, dt=bool), z(C, n, nb, dt=bool),
                          z(C, n, nb, dt=jnp.int32), z(C, n, nb), z(C, n, nb), z(C, n, nb, 4), z(C, n, nb, d),
                          z(C, n, nb, A, dt=bool), z(C, n, nb, E.OUT_SLOTS, dt=jnp.int32), z(C, n, nb, dt=bool),
                          z(C, n, nb, 2, dt=jnp.int32), z(C, n, nb), z(C, n, nb, dt=bool), z(C, n, nb, dt=bool),
                          z(C, n, nb), z(C, n, nb), z(C, n, nb), jnp.full((C, n, nb), 2, jnp.int32))
        res = Reservoir(z(n, R, d), z(n, R, A, dt=bool), z(n, R, cfg.hidden), z(n, R, dt=bool), z(n, R, A),
                        z(n, R, dt=bool))
        return Memory(tail, z(n, nb, cfg.hidden), res, param_zeros(nb, d) if PENDING else None)

    def lrs_of(pop: Population):
        scale = pop.genotype.lr_scale if cfg.per_agent_lr else jnp.ones(n)
        return cfg.lr * scale

    def scaled(updates, lrs):
        """Each agent's update at its own learning rate; a shared trunk at the population's mean
        rate (so the tied copies move together)."""
        if not cfg.share_trunk:
            return _scale_per_agent(updates, lrs)
        mean_lr = jnp.full_like(lrs, lrs.mean())
        return jax.tree_util.tree_map_with_path(
            lambda path, u: u * (mean_lr if _is_trunk(path) else lrs).reshape((-1,) + (1,) * (u.ndim - 1)), updates)

    def tied(grads):
        return _tie_trunk(grads) if cfg.share_trunk else grads

    def gammas_of(pop: Population):
        g = pop.genotype.patience if cfg.trait_gamma else jnp.full(n, cfg.gamma)
        return (g ** cfg.tick_scale)[:, None]                                               # (N, 1)

    def update(runner, pop: Population):
        params, opt_state, states, obs, mask, reset_flag, carry, key, memory = runner
        init_carry = carry
        key, k_pool = jax.random.split(key)
        pool = reset_worlds(k_pool, pop)

        # ---------------------------------------------------- rollout
        def env_step(c, _):
            params, states, obs, mask, reset_flag, carry, key = c
            key, k_act, k_pick = jax.random.split(key, 3)
            carry, logits, value = v_apply(params, carry, obs[:, None], reset_flag[:, None],
                                           mask[:, None])
            logits, value = logits[:, 0], value[:, 0]                   # (N, B, A), (N, B)
            action, msg = sample_outputs(k_act, logits)                    # msg: [said, written, aim]
            log_prob = factor_logps(logits, action, msg)[0]                  # (N, B, 4)
            alive_before = _swap_axes_nb(states.alive)                  # (N, B)
            ages_before = _swap_axes_nb(states.age)
            nxt, reward, done, info = jax.vmap(lambda s, a, m: E.step(s, a, env_cfg, m))(
                states, _swap_axes_nb(action), _swap_axes_nb(msg))
            own_reward = reward                                          # before mixing: comparable across alpha
            # Prosociality weight alpha: each agent's training reward mixes in the group mean
            reward = (1 - cfg.alpha) * reward + cfg.alpha * reward.mean(axis=1, keepdims=True)
            idx = jax.random.randint(k_pick, (b,), 0, b)
            fresh = jax.tree_util.tree_map(lambda x: x[idx], pool)
            fresh = fresh._replace(key=jax.vmap(lambda k: jax.random.fold_in(k, 1))(fresh.key))
            sel = lambda f, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s)
            states2 = jax.tree_util.tree_map(sel, fresh, nxt)
            obs2, mask2 = observe(states2)
            died = _swap_axes_nb(info.died)
            agent_done = died | done[None, :]
            # A world cut off by the clock (not by everyone dying): bootstrap from the value of the
            # state it was cut off in (Pardo et al. 2018), computed only on ticks where it happens.
            cut = done[None, :] & _swap_axes_nb(nxt.alive) & ~died               # (N, B)

            def cut_value(_):
                o_t, m_t = observe(nxt)
                _, _, v_t = v_apply(params, carry, o_t[:, None], jnp.zeros_like(reset_flag)[:, None], m_t[:, None])
                return v_t[:, 0]
            boot = jnp.where(cut, jax.lax.cond(jnp.any(cut), cut_value, lambda _: jnp.zeros_like(value), None), 0.0)
            tr = Transition(reset_flag, agent_done, alive_before, action, value,
                            _swap_axes_nb(reward), log_prob, obs, mask, msg,
                            _swap_axes_nb(info.spoke) & alive_before,
                            _swap_axes_nb(E.layered_pos(states.pos, states.below)), _swap_axes_nb(states.health),   # layers apart
                            jnp.broadcast_to(jax.vmap(lambda t: E.is_night(t, env_cfg))(states.t)[None, :], (n, b)),
                            died, _swap_axes_nb(info.slept) * jnp.minimum(
                                _swap_axes_nb(states.sleep_depth) + 1, env_cfg.sleep_deep_ticks) / env_cfg.sleep_deep_ticks,
                            jnp.ones((n, b)), boot, jnp.zeros((n, b), jnp.int32))
            night_b = jax.vmap(lambda t: E.is_night(t, env_cfg))(states.t)          # (B,)
            stats = {
                "alive": alive_before.sum(),
                "deaths": died.sum(),
                "age_at_death": jnp.where(died, ages_before + 1, 0).sum(),
                "cause": jnp.stack([(died & (_swap_axes_nb(info.death_cause) == c)).sum()
                                    for c in range(len(E.CAUSE_NAMES))]),
                "deaths_by_season": jnp.stack([(died & (info.season[None, :] == q)).sum() for q in range(4)]),
                "hits": info.hits.sum(),
                "gifts": jnp.any(info.gifts, axis=-1).sum(),
                "touches": jnp.any(info.touches, axis=-1).sum(),
                "messages": info.spoke.sum(),
                "moral_reward": info.moral_reward.sum(),
                "deterred": info.deterred.sum(),
                "company": info.company.sum(),
                "births": info.born.sum(),
                "conceptions": info.conceived.sum(),
                "no_room": info.no_room.sum(),
                "matings": info.mated.sum(),
                "asleep_night": (info.slept & night_b[:, None]).sum(),
                "asleep_day": (info.slept & ~night_b[:, None]).sum(),
                "alive_night": (alive_before & night_b[None, :]).sum(),
                "woken": info.woken.sum(),
                "collapses": info.collapsed_now.sum(),
                "skill_water": (info.skill == 1).sum(),
                "skill_food": (info.skill == 2).sum(),
                "skill_flee": (info.skill == 3).sum(),
                "skill_shelter": (info.skill == 4).sum(),
                "scarred": ((nxt.scars > 0.1) & nxt.alive).sum(),
                "lasting_injured": ((nxt.lasting_injury > 0.02) & nxt.alive).sum(),
                "meat_eaten": info.meat_eaten.sum(),
                "large_prey_hits": info.large_prey_hit.sum(),
                "winter_night_ticks": (info.winter_night[:, None] & _swap_axes_nb(alive_before)).sum(),
                "winter_huddle_ticks": (info.winter_night[:, None] & info.huddling).sum(),
                "large_kills_by_agent": info.kills[:, :, 1].sum(axis=0),   # (N,) for free-riding
                "kills": info.kills.sum(axis=(0, 1)),
                "world_ends": done.sum(),
                # Group survival and global reward (the proposal's headline measures)
                # Global reward: the base reward only (staying alive, death, pain, novelty, touch),
                # before alpha mixing and without the moral sentiments, as the proposal defines it
                "own_reward": (own_reward - info.moral_reward).sum(),
                "world_ticks": jnp.asarray(b),
                "survivors_at_end": (nxt.alive.sum(axis=-1) * done).sum(),
                "full_years": (done & jnp.any(nxt.alive, axis=-1)).sum(),
                # Fire ladder: use, maintain, cook, carry, make
                "near_fire": info.near_fire.sum(),
                "fed_fire": info.fed_fire.sum(),
                "meat_to_heat": info.meat_to_heat.sum(),
                "ate_cooked": info.ate_cooked.sum(),
                "pulled_burning_wood": info.torch_lit.sum(),
                "fires_by_spark": (info.ignited == 1).sum(),
                "fires_by_burning_wood": (info.ignited == 2).sum(),
                "lightning_fires": info.lightning.sum(),
                "burning_cells": info.burning_cells.sum(),
            }
            # Fresh recurrent memory at world resets and for newborns (a slot's new occupant)
            new_reset = jnp.broadcast_to(done[None, :], (n, b)) | _swap_axes_nb(info.born)
            return (params, states2, obs2, mask2, new_reset, carry, key), (tr, stats)

        c0 = (params, states, obs, mask, reset_flag, carry, key)
        (params, states, obs, mask, reset_flag, carry, key), (traj, stats) = jax.lax.scan(
            env_step, c0, None, cfg.num_steps)
        _, _, last_val = v_apply(params, carry, obs[:, None], reset_flag[:, None], mask[:, None])
        params, opt_state, key, extra, memory = learn_core(params, opt_state, traj, init_carry, last_val[:, 0], key,
                                                           pop, memory)
        runner = (params, opt_state, states, obs, mask, reset_flag, carry, key, memory)
        metrics = jax.tree_util.tree_map(lambda x: x.sum(0), stats)
        metrics.update(extra)
        return runner, metrics

    def learn_core(params, opt_state, traj, init_carry, last_val, key, pop: Population, memory=None):
        """PPO with social learning (and the world model) on one stretch of experience: traj fields
        (T, N, B, ...), the recurrent state at its start (N, B, H) and the value after it (N, B).
        With sleep learning, `memory` carries the steps still waiting for sleep and the long-term
        store; the update learns from [carried tail | traj] and returns the memory for next time."""
        if memory is None:
            memory = empty_memory(traj.reward.shape[2], traj.obs.shape[-1])
        if C > 0:
            traj = jax.tree_util.tree_map(lambda a, b_: jnp.concatenate([a, b_.astype(a.dtype)], 0), memory.tail, traj)
            init_carry = memory.tail_carry
        Tw = traj.reward.shape[0]
        # ---------------------------------------------------- GAE (per agent)
        gam = gammas_of(pop)

        adv = gae(traj.reward, traj.value, traj.done, traj.boot, last_val, gam, cfg.gae_lambda ** cfg.tick_scale)
        targets = adv + traj.value

        # ---------------------------------------------------- sleep consolidation
        valid_f = traj.valid.astype(jnp.float32)
        if PENDING:
            # Everything lived is in the day's pending change; sleep decides how much of it is kept
            next_v = jnp.concatenate([traj.value[1:], last_val[None]], 0)
            delta = traj.reward + gam * (next_v * (1 - traj.done.astype(jnp.float32)) + traj.boot) - traj.value
            pr = jnp.abs(delta) ** cfg.surprise_power
            mean_pr = (pr * valid_f).sum((0, 2), keepdims=True) / jnp.maximum(valid_f.sum((0, 2), keepdims=True), 1e-8)
            prio = (1 - cfg.surprise_priority) + cfg.surprise_priority * pr / jnp.maximum(mean_pr, 1e-8)
            learn_w = jax.lax.stop_gradient(prio)
            graded_w = jnp.ones_like(traj.reward)
            finalized = jnp.ones_like(traj.reward, bool)
            next_stage = traj.stage[:0]
        elif cfg.sleep_learning:
            graded_w, finalized, _, next_stage = consolidation_plan(
                traj.done, traj.died, traj.sleep_q, traj.stage, cfg.consolidation_ticks / cfg.tick_scale, C)
            # Replay favours surprise: |TD error| among the steps learned now, mean weight kept
            next_v = jnp.concatenate([traj.value[1:], last_val[None]], 0)
            delta = traj.reward + gam * (next_v * (1 - traj.done.astype(jnp.float32)) + traj.boot) - traj.value
            pr = jnp.abs(delta) ** cfg.surprise_power
            on = graded_w * valid_f
            mean_pr = (pr * on).sum((0, 2), keepdims=True) / jnp.maximum(on.sum((0, 2), keepdims=True), 1e-8)
            prio = (1 - cfg.surprise_priority) + cfg.surprise_priority * pr / jnp.maximum(mean_pr, 1e-8)
            learn_w = jax.lax.stop_gradient(graded_w * prio)
        else:
            graded_w = learn_w = jnp.ones_like(traj.reward)
            finalized = jnp.ones_like(traj.reward, bool)
            next_stage = traj.stage[:0]
        traj = traj._replace(learn_w=learn_w)

        # ---------------------------------------------------- social learning: who watches whom
        key, k_demo = jax.random.split(key)
        demo, demo_w = social.select_demonstrators(
            k_demo, traj.pos, traj.valid, traj.health, adv, pop.appearance, env_cfg.view_radius,
            cfg.imitation_bias, cfg.imitation_bias_strength, cfg.imitate_positive_only,
            spoke=traj.spoke, speech_attention=cfg.speech_attention, teach_bonus=cfg.teach_bonus)
        # Learn only from what the observer itself could see of the demonstrator
        demo_obs = social.restrict_to_observer_view(
            social.gather_demo(traj.obs, demo), social.gather_demo(traj.pos, demo), traj.pos,
            traj.night, env_cfg)
        demo_act = social.gather_demo(traj.action, demo)
        demo_mask = social.gather_demo(traj.mask, demo)
        demo_msg = social.gather_demo(traj.msg, demo)
        social_w = pop.genotype.social_learning
        demo_w = demo_w * learn_w                # what one watched is also learned only once slept on

        # ---------------------------------------------------- loss (one agent)
        res = memory.reservoir

        def retention(p, rs):
            """KL from what the agent chose at its stored moments to what it would choose now."""
            if R == 0:
                return jnp.float32(0.0)
            full = jnp.concatenate([rs.mask, jnp.ones(rs.mask.shape[:-1] + (M,), bool)], -1)
            _, lg, _ = net.apply(p, rs.carry, (rs.obs[None], rs.reset[None], full[None]))
            la = jax.nn.log_softmax(lg[0, :, :A])
            old = rs.probs
            kl = jnp.where(rs.mask & (old > 0), old * (jnp.log(jnp.maximum(old, 1e-12)) - la), 0.0).sum(-1)
            f = rs.filled.astype(jnp.float32)
            return (kl * f).sum() / jnp.maximum(f.sum(), 1.0)

        def agent_loss(p, ic, tr, adv, tg, d_obs, d_act, d_mask, d_msg, d_w, sl, rs):
            """tr fields (T, B, ...) for one agent; d_* = what it watched; sl = its gene; rs = its
            long-term store."""
            terms = net.apply(p, ic, (tr.obs, tr.reset, full_mask(tr.mask)), tr.action, method=net.learn_terms)
            logits, value = terms["logits"], terms["value"]
            w = tr.valid.astype(jnp.float32) * tr.learn_w
            # Normalised by the new steps of this update (not the carried ones), so how much is
            # learned still scales with how much was slept on
            denom = jnp.maximum((tr.valid & (tr.stage == 0)).astype(jnp.float32).sum(), 1.0)
            lp, (logp_all, lk, ls, live, said, written) = factor_logps(logits, tr.action, tr.msg)
            wsum = jnp.maximum(w.sum(), 1e-8)
            mean = (adv * w).sum() / wsum
            std = jnp.sqrt((((adv - mean) ** 2) * w).sum() / wsum) + 1e-8
            an = (adv - mean) / std
            # Each output clipped on its own ratio (action, aim, said, written): a message that moved
            # a lot does not freeze the action's update, nor the other way round
            ratio = jnp.exp(lp - tr.log_prob)                                        # (T, B, 4)
            surr = jnp.minimum(ratio * an[..., None],
                               jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * an[..., None])
            pg = -(surr * live).sum(-1)
            lw = live * w[..., None]
            lws = jnp.maximum(lw.sum((0, 1)), 1e-8)
            clipfrac = ((jnp.abs(ratio - 1) > cfg.clip_eps) * lw).sum((0, 1)) / lws    # per output
            approx_kl = (((ratio - 1) - (lp - tr.log_prob)) * lw).sum((0, 1)) / lws
            v_clip = tr.value + jnp.clip(value - tr.value, -cfg.clip_eps, cfg.clip_eps)
            vf = 0.5 * jnp.maximum((value - tg) ** 2, (v_clip - tg) ** 2)
            ent = -(jnp.exp(logp_all) * jnp.where(tr.mask, logp_all, 0.0)).sum(-1)
            ent_aim = -(jnp.exp(lk) * lk).sum(-1) * live[..., 1]
            h_sym = -(jnp.exp(ls) * ls).sum(-1)                                        # (T, B, 2L)
            speaks = jax.lax.stop_gradient(1.0 - jnp.exp(ls[..., 0, 0]))                # chance the first is not silence
            ent_said = (h_sym[..., 0] + speaks * h_sym[..., 1:L].sum(-1)) * said[..., 0]
            ent_written = (h_sym[..., L:] * written).sum(-1)
            if cfg.entropy_target > 0:          # the floor: a bigger bonus the further under the target
                ec = cfg.ent_coef * jnp.exp(jnp.clip(cfg.entropy_gain * (cfg.entropy_target - jax.lax.stop_gradient(ent)), -4.0, 4.0))
            else:
                ec = cfg.ent_coef
            per_step = pg + cfg.vf_coef * vf - ec * (ent + ent_aim) \
                - cfg.msg_ent_coef * (ent_said + ent_written)
            loss = (per_step * w).sum() / denom
            # Imitation: predict the watched agent's action from ITS observation, without
            # memory (hidden state reset each step: the observer did not live its history)
            # Words are copied along with actions: language can spread culturally
            _, d_logits, _ = net.apply(p, jnp.zeros_like(ic),
                                       (d_obs, jnp.ones(d_act.shape, bool), full_mask(d_mask)))
            nll = -factor_logps(d_logits, d_act, d_msg)[0].sum(-1)                   # each output only where it acted
            imit = (nll * d_w).sum() / jnp.maximum((d_w > 0).sum(), 1.0)
            loss = loss + cfg.imitation_coef * sl * imit
            wm = (jnp.float32(0.0),) * 3
            if cfg.world_model:
                # World model: from its belief and what it did, predict its next belief, the reward
                # that came, and its own state next tick (only within one life; learned once slept on)
                pair = (tr.valid[:-1] & tr.valid[1:] & ~tr.done[:-1] & ~tr.reset[1:]).astype(jnp.float32) \
                    * tr.learn_w[:-1]
                np_ = jnp.maximum(pair.sum(), 1.0)
                lat = ((terms["z"][:-1] - jax.lax.stop_gradient(terms["h"][1:])) ** 2).mean(-1)
                # the own-state head learns the CHANGE over the tick, scaled (policy.own_scale): what the
                # action did, not the level the belief already carries
                own = ((terms["own"][:-1] - net.own_scale * (tr.obs[1:][..., dec_idx] - tr.obs[:-1][..., dec_idx])) ** 2).mean(-1)
                rew = (terms["r"] - tr.reward) ** 2
                wm = ((lat * pair).sum() / np_, (rew * w).sum() / denom, (own * pair).sum() / np_)
                loss = loss + cfg.wm_coef * (wm[0] + wm[1] + wm[2])
            ret = retention(p, rs)
            loss = loss + cfg.retention_coef * ret
            return loss, ((ent * w).sum() / denom, imit, wm, ret, clipfrac, approx_kl)

        def total_loss(params, ic, tr, adv, tg, d_obs, d_act, d_mask, d_msg, d_w):
            # Agent axis is 0 for params/ic/genes/the store, 1 for (T, N, B, ...) data
            if cfg.agent_chunk > 0:
                lead = lambda x: jnp.moveaxis(x, 1, 0)
                xs = (params, ic) + tuple(jax.tree_util.tree_map(lead, v)
                                          for v in (tr, adv, tg, d_obs, d_act, d_mask, d_msg, d_w)) + (social_w, res)
                losses, (ents, imits, wm, ret, cf, kl) = jax.lax.map(lambda a: jax.checkpoint(agent_loss)(*a), xs,
                                                                     batch_size=cfg.agent_chunk)
            else:
                losses, (ents, imits, wm, ret, cf, kl) = jax.vmap(agent_loss, in_axes=(0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0))(
                    params, ic, tr, adv, tg, d_obs, d_act, d_mask, d_msg, d_w, social_w, res)
            return losses.sum(), (ents.mean(), imits.mean(), tuple(x.mean() for x in wm), ret.mean(),
                                  cf.mean(0), kl.mean(0))

        lrs = lrs_of(pop)

        def epoch(c, _):
            params, opt_state, key = c
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, b)
            mb = b // cfg.num_minibatches

            def split(x, world_axis):
                x = jnp.take(x, perm, axis=world_axis)
                shape = x.shape[:world_axis] + (cfg.num_minibatches, mb) + x.shape[world_axis + 1:]
                return jnp.moveaxis(x.reshape(shape), world_axis, 0)

            batches = (jax.tree_util.tree_map(lambda x: split(x, 2), traj),
                       split(adv, 2), split(targets, 2), split(init_carry, 1),
                       split(demo_obs, 2), split(demo_act, 2), split(demo_mask, 2),
                       split(demo_msg, 2), split(demo_w, 2))

            def mb_step(c, bt):
                params, opt_state = c
                tr, a, tg, ic, d_obs, d_act, d_mask, d_msg, d_w = bt
                grads, ent = jax.grad(total_loss, has_aux=True)(params, ic, tr, a, tg,
                                                                d_obs, d_act, d_mask, d_msg, d_w)
                grads = tied(_per_agent_clip(grads, cfg.max_grad_norm))
                updates, opt_state = adam.update(grads, opt_state, params)
                params = optax.apply_updates(params, scaled(updates, lrs))
                return (params, opt_state), ent

            (params, opt_state), ents = jax.lax.scan(mb_step, (params, opt_state), batches)
            return (params, opt_state, key), ents

        if PENDING:
            # The day's gradient at the unchanged weights, averaged over the minibatches of worlds
            nmb = cfg.num_minibatches
            mbw = b // nmb

            def chunk(x, world_axis):
                shape = x.shape[:world_axis] + (nmb, mbw) + x.shape[world_axis + 1:]
                return jnp.moveaxis(x.reshape(shape), world_axis, 0)
            batches = (jax.tree_util.tree_map(lambda x: chunk(x, 2), traj), chunk(adv, 2), chunk(targets, 2),
                       chunk(init_carry, 1), chunk(demo_obs, 2), chunk(demo_act, 2), chunk(demo_mask, 2),
                       chunk(demo_msg, 2), chunk(demo_w, 2))

            def mb_grad(acc, bt):
                tr, a_, tg, ic, d_obs, d_act, d_mask, d_msg, d_w = bt
                g, aux = jax.grad(total_loss, has_aux=True)(params, ic, tr, a_, tg, d_obs, d_act, d_mask, d_msg, d_w)
                return jax.tree_util.tree_map(lambda u, v: u + v / nmb, acc, g), aux
            grads, ents = jax.lax.scan(mb_grad, jax.tree_util.tree_map(jnp.zeros_like, params), batches)
            fade = 0.5 ** (traj.reward.shape[0] / max(cfg.day_ticks, 1))            # unslept: half gone in a day
            pending = jax.tree_util.tree_map(lambda q, g: fade * q + g, memory.pending, grads)
            # Share earned by deep sleep in this stretch (or a death), per world, averaged
            slept = (traj.sleep_q * valid_f).sum(0)                                   # (N, B)
            earned = jnp.where(jnp.any(traj.died, 0), 1.0, jnp.minimum(slept / (cfg.consolidation_ticks / cfg.tick_scale), 1.0))
            share = earned.mean(1)                                                    # (N,)
            sh = lambda x: share.reshape((-1,) + (1,) * (x.ndim - 1))
            step = tied(_per_agent_clip(jax.tree_util.tree_map(lambda q: q * sh(q), pending), cfg.max_grad_norm))
            updates, new_opt = adam.update(step, opt_state, params)
            on = share > 0
            if cfg.share_trunk:                       # a tied trunk moves for everyone whenever anyone slept
                on_all = jnp.broadcast_to(jnp.any(on), on.shape)
            keep = lambda new, old: jax.tree_util.tree_map_with_path(
                lambda path, a_, b_: jnp.where((on_all if cfg.share_trunk and _is_trunk(path) else on)
                                               .reshape((-1,) + (1,) * (a_.ndim - 1)), a_, b_)
                if (a_.ndim >= 1 and a_.shape[:1] == (n,)) else a_, new, old)
            params = keep(optax.apply_updates(params, scaled(updates, lrs)), params)
            opt_state = keep(new_opt, opt_state)
            memory = memory._replace(pending=jax.tree_util.tree_map(lambda q: q * (1 - sh(q)), pending))
            graded_w = jnp.broadcast_to(earned[None], traj.reward.shape)
        else:
            (params, opt_state, key), ents = jax.lax.scan(epoch, (params, opt_state, key), None,
                                                          cfg.update_epochs)
        fin = finalized.astype(jnp.float32) * valid_f
        metrics = {"entropy": ents[0].mean(), "imitation_loss": ents[1].mean(),
                   "watched_fraction": (demo_w > 0).mean(),
                   # Of the steps whose fate was settled this update, the share learned (the rest
                   # was forgotten for want of sleep); and the share of new steps still waiting
                   "learned_share": (graded_w * fin).sum() / jnp.maximum(fin.sum(), 1),
                   "waiting_share": ((traj.stage == 0) & ~finalized & traj.valid).sum()
                   / jnp.maximum(((traj.stage == 0) & traj.valid).sum(), 1),
                   "retention_kl": ents[3].mean(), "long_term_filled": res.filled.mean(),
                   # per output (action, aim, said, written): share of updates clipped, and the KL moved
                   "clip_fraction": ents[4].reshape(-1, 4).mean(0), "approx_kl": ents[5].reshape(-1, 4).mean(0)}
        if cfg.world_model:
            metrics.update(wm_latent=ents[2][0].mean(), wm_reward=ents[2][1].mean(), wm_own=ents[2][2].mean(),
                           trust=params["params"]["trust"].mean())
        if C > 0 or R > 0:
            memory = after_sleep(params, traj, init_carry, graded_w * valid_f * fin, next_stage, res, key)._replace(
                pending=memory.pending)
        return params, opt_state, key, metrics, memory

    def after_sleep(params, traj, carry0, learned, next_stage, res, key):
        """With the new weights: the tail to carry over (its recurrent state recomputed) and the
        long-term store, a share of it replaced by moments learned in this update."""
        sw = lambda x: jnp.swapaxes(x, 0, 1)                            # (T, N, ...) <-> (N, T, ...)

        def one(p, c0, obs, rst, mask):                                 # (B, H), (T, B, ...)
            _, out = net.apply(p, c0, (obs, rst, full_mask(mask)), method=net.trace)
            return out["h"], out["logits"]
        if cfg.agent_chunk > 0:                                         # a few agents at a time (memory)
            h, logits = jax.lax.map(lambda a: one(*a), (params, carry0, sw(traj.obs), sw(traj.reset), sw(traj.mask)),
                                    batch_size=cfg.agent_chunk)
        else:
            h, logits = jax.vmap(one)(params, carry0, sw(traj.obs), sw(traj.reset), sw(traj.mask))   # (N, T, B, ...)
        Tw = traj.reward.shape[0]
        carry_in = jnp.concatenate([carry0[:, None], h[:, :-1]], 1)     # (N, T, B, H): state going into each step
        tail = jax.tree_util.tree_map(lambda x: x[Tw - C:], traj)._replace(stage=next_stage) if C > 0 else None
        tail_carry = carry_in[:, Tw - C] if C > 0 else carry0
        if R > 0:
            k_pick, k_where = jax.random.split(jax.random.fold_in(key, 17))
            k_new = max(1, int(round(R * cfg.reservoir_refresh)))
            cand = sw(learned).reshape(n, -1)                           # (N, T*B)
            flat = lambda x: x.reshape((n, -1) + x.shape[3:])           # (N, T, B, ...) -> (N, T*B, ...)

            def pick(k, c):
                return jax.random.choice(k, c.shape[0], (k_new,), p=(c > 0) + 1e-9)
            idx = jax.vmap(pick)(jax.random.split(k_pick, n), cand)     # (N, k_new)
            ok = jnp.take_along_axis(cand, idx, 1) > 0
            g = lambda x: jnp.take_along_axis(x, idx.reshape(idx.shape + (1,) * (x.ndim - 2)), 1)
            obs_n = g(flat(sw(traj.obs)))
            mask_n = g(flat(sw(traj.mask)))
            carry_n = g(flat(carry_in))
            reset_n = g(flat(sw(traj.reset)))
            la = jax.nn.log_softmax(jnp.where(g(flat(sw(traj.mask))), g(flat(logits))[..., :A], -1e9))
            probs_n = jnp.where(mask_n, jnp.exp(la), 0.0)
            where = jax.vmap(lambda k: jax.random.choice(k, R, (k_new,), replace=False))(jax.random.split(k_where, n))

            def put(old, new):
                upd = jax.vmap(lambda o, w, v, m: o.at[w].set(jnp.where(m.reshape(m.shape + (1,) * (v.ndim - 1)), v, o[w])))
                return upd(old, where, new.astype(old.dtype), ok)
            res = Reservoir(put(res.obs, obs_n), put(res.mask, mask_n), put(res.carry, carry_n),
                            put(res.reset, reset_n), put(res.probs, probs_n), put(res.filled, ok))
        if C == 0:
            tail = jax.tree_util.tree_map(lambda x: x[:0], traj)
        return Memory(tail, tail_carry, res)

    def learn_outside(params, opt_state, traj, init_carry, end_obs, end_mask, end_reset, end_carry, key,
                      pop: Population, memory=None):
        """Learn from experience gathered outside the trainer (the continuous world): traj fields
        (T, N, B, ...) with value and log_prob left empty. Both are recomputed here from the same
        weights that acted, which gives exactly what acting computed: the weights have not changed
        since, and a newborn's steps begin with a reset, so its memory matches too."""
        sw = lambda x: jnp.swapaxes(x, 0, 1)                            # (T, N, ...) <-> (N, T, ...)
        _, logits, value = v_apply(params, init_carry, sw(traj.obs), sw(traj.reset), sw(traj.mask))
        log_prob = factor_logps(sw(logits), traj.action, traj.msg)[0]
        traj = traj._replace(value=sw(value), log_prob=log_prob)
        _, _, last_val = v_apply(params, end_carry, end_obs[:, None], end_reset[:, None], end_mask[:, None])
        return learn_core(params, opt_state, traj, init_carry, last_val[:, 0], key, pop, memory)

    def act(params, carry, obs, mask, reset, key):
        """One tick of acting, as in the rollout: obs (N, B, D), mask (N, B, A), reset (N, B).
        Returns carry, action, the packed outputs [said, written, aim], their log-probs (N, B, 4), value."""
        carry, logits, value = v_apply(params, carry, obs[:, None], reset[:, None], mask[:, None])
        logits, value = logits[:, 0], value[:, 0]
        action, msg = sample_outputs(key, logits)
        return carry, action, msg, factor_logps(logits, action, msg)[0], value

    update = jax.jit(update)
    update.act = jax.jit(act)
    update.learn_core = jax.jit(learn_core)
    update.learn_outside = jax.jit(learn_outside)
    update.opt_init = adam.init
    update.empty_memory = empty_memory
    update.net = net
    return net, init, update
