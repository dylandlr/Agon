"""Recurrent PPO for Phase 1 (one agent per environment, many environments).

Structure follows the PureJaxRL recurrent PPO: rollouts, GAE and updates are
all jitted. The Phase 2+ version replaces the single shared network with
stacked per-agent parameters and per-agent learning rates from the genotype.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from . import env as E
from .config import EnvConfig
from .policy import make_network


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 64
    num_steps: int = 128          # rollout length per update
    total_timesteps: int = 2_000_000
    lr: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    num_minibatches: int = 4
    hidden: int = 128
    encoder: str = "conv"
    seed: int = 0

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)


class Transition(NamedTuple):
    reset: jnp.ndarray    # hidden state was reset before this step
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray


class EpisodeStats(NamedTuple):
    ret: jnp.ndarray      # running return per env
    length: jnp.ndarray   # running length per env


# ------------------------------------------------------- env wrappers
def _single_reset(key, cfg: EnvConfig):
    s = E.reset(key, cfg)
    return s, E.observe(s, cfg)[0], E.action_mask(s, cfg)[0]


def _single_step(state, action, fresh, cfg: EnvConfig):
    """Step one env; on done, swap in `fresh` (a pre-generated world)."""
    s2, r, done, info = E.step(state, action[None], cfg)
    s_new = fresh
    s_next = jax.tree_util.tree_map(lambda a, b: jnp.where(done, a, b), s_new, s2)
    return (s_next, E.observe(s_next, cfg)[0], E.action_mask(s_next, cfg)[0],
            r[0], done, info.death_cause[0])


def make_train(env_cfg: EnvConfig, cfg: PPOConfig):
    assert env_cfg.n_agents == 1, "Phase 1 trainer is single-agent per env"
    n_actions = E.N_ACTIONS
    net = make_network(env_cfg, cfg.hidden, cfg.encoder)
    v_reset = jax.vmap(_single_reset, in_axes=(0, None))
    v_step = jax.vmap(_single_step, in_axes=(0, 0, 0, None))

    def init(key):
        k_net, k_env, k_run = jax.random.split(key, 3)
        env_state, obs, mask = v_reset(jax.random.split(k_env, cfg.num_envs), env_cfg)
        carry = jnp.zeros((cfg.num_envs, cfg.hidden))
        dummy = (jnp.zeros((1, cfg.num_envs, obs.shape[-1])),
                 jnp.zeros((1, cfg.num_envs), dtype=bool),
                 jnp.ones((1, cfg.num_envs, n_actions), dtype=bool))
        params = net.init(k_net, carry, dummy)
        tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                         optax.adam(cfg.lr, eps=1e-5))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        stats = EpisodeStats(jnp.zeros(cfg.num_envs), jnp.zeros(cfg.num_envs))
        reset_flag = jnp.ones(cfg.num_envs, dtype=bool)
        return (ts, env_state, obs, mask, reset_flag, carry, stats, k_run)

    def _fresh_worlds(key):
        """World pool: resets are ~50x costlier than steps, so generate one
        batch per update and let finished envs draw from it."""
        return jax.vmap(E.reset, in_axes=(0, None))(
            jax.random.split(key, cfg.num_envs), env_cfg)

    def update(runner):
        ts, env_state, obs, mask, reset_flag, carry, stats, key = runner
        init_carry = carry
        key, k_pool = jax.random.split(key)
        pool = _fresh_worlds(k_pool)

        # ---------------------------------------------------- rollout
        def env_step(c, _):
            ts, env_state, obs, mask, reset_flag, carry, stats, key = c
            key, k_act, k_env = jax.random.split(key, 3)
            carry, logits, value = ts.apply_fn(
                ts.params, carry, (obs[None], reset_flag[None], mask[None]))
            logits, value = logits[0], value[0]
            action = jax.random.categorical(k_act, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(cfg.num_envs), action]
            k_idx, k_new = jax.random.split(k_env)
            idx = jax.random.randint(k_idx, (cfg.num_envs,), 0, cfg.num_envs)
            fresh = jax.tree_util.tree_map(lambda x: x[idx], pool)
            fresh = fresh._replace(key=jax.random.split(k_new, cfg.num_envs))
            env_state2, obs2, mask2, reward, done, cause = v_step(
                env_state, action, fresh, env_cfg)
            tr = Transition(reset_flag, done, action, value, reward, log_prob, obs, mask)
            ret, length = stats.ret + reward, stats.length + 1
            ep_info = {"done": done, "return": ret, "length": length, "cause": cause}
            stats = EpisodeStats(jnp.where(done, 0.0, ret), jnp.where(done, 0.0, length))
            return (ts, env_state2, obs2, mask2, done, carry, stats, key), (tr, ep_info)

        runner = (ts, env_state, obs, mask, reset_flag, carry, stats, key)
        runner, (traj, ep_info) = jax.lax.scan(env_step, runner, None, cfg.num_steps)
        ts, env_state, obs, mask, reset_flag, carry, stats, key = runner

        # ---------------------------------------------------- GAE
        _, _, last_val = ts.apply_fn(ts.params, carry,
                                     (obs[None], reset_flag[None], mask[None]))
        last_val = last_val[0]

        def gae_step(c, tr):
            gae, next_val = c
            delta = tr.reward + cfg.gamma * next_val * (1 - tr.done) - tr.value
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - tr.done) * gae
            return (gae, tr.value), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val),
                              traj, reverse=True)
        targets = adv + traj.value

        # ---------------------------------------------------- update
        def loss_fn(params, init_c, tr, adv, targets):
            _, logits, value = ts.apply_fn(params, init_c, (tr.obs, tr.reset, tr.mask))
            logp_all = jax.nn.log_softmax(logits)
            logp = jnp.take_along_axis(logp_all, tr.action[..., None], -1)[..., 0]
            ratio = jnp.exp(logp - tr.log_prob)
            adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.minimum(ratio * adv_n,
                              jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_n).mean()
            v_clip = tr.value + jnp.clip(value - tr.value, -cfg.clip_eps, cfg.clip_eps)
            vf = 0.5 * jnp.maximum((value - targets) ** 2, (v_clip - targets) ** 2).mean()
            probs = jnp.exp(logp_all)
            ent = -(probs * jnp.where(tr.mask, logp_all, 0.0)).sum(-1).mean()
            return pg + cfg.vf_coef * vf - cfg.ent_coef * ent, (pg, vf, ent)

        def epoch(c, _):
            ts, key = c
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, cfg.num_envs)
            mb = cfg.num_envs // cfg.num_minibatches
            shuffle = lambda x: jnp.take(x, perm, axis=1).reshape(
                x.shape[0], cfg.num_minibatches, mb, *x.shape[2:]).swapaxes(0, 1)
            batches = (jax.tree_util.tree_map(shuffle, traj), shuffle(adv), shuffle(targets),
                       jnp.take(init_carry, perm, axis=0).reshape(cfg.num_minibatches, mb, -1))

            def mb_step(ts, b):
                tr, a, tg, ic = b
                grads, aux = jax.grad(loss_fn, has_aux=True)(ts.params, ic, tr, a, tg)
                return ts.apply_gradients(grads=grads), aux

            ts, aux = jax.lax.scan(mb_step, ts, (batches[0], batches[1], batches[2], batches[3]))
            return (ts, key), aux

        (ts, key), aux = jax.lax.scan(epoch, (ts, key), None, cfg.update_epochs)
        runner = (ts, env_state, obs, mask, reset_flag, carry, stats, key)
        d = ep_info["done"]
        n_done = d.sum()
        metrics = {
            "episodes": n_done,
            "mean_life": jnp.where(n_done > 0, (ep_info["length"] * d).sum() / jnp.maximum(n_done, 1), jnp.nan),
            "mean_return": jnp.where(n_done > 0, (ep_info["return"] * d).sum() / jnp.maximum(n_done, 1), jnp.nan),
            "deaths_by_cause": jnp.stack([((ep_info["cause"] == c) & d).sum()
                                          for c in range(len(E.CAUSE_NAMES))]),
            "timeouts": (d & (ep_info["cause"] == E.CAUSE_NONE)).sum(),
            "entropy": aux[2].mean(),
            "value_loss": aux[1].mean(),
        }
        return runner, metrics

    return net, init, jax.jit(update)
