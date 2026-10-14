"""Evaluate a trained Phase 1 policy against the random baseline and record
one episode (events, actions, snapshots) for replay.

    python scripts/evaluate.py --params runs/phase1/params.msgpack --episodes 64
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lifesim import baselines as B
from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.policy import make_network
from lifesim.recorder import random_policy, record_episode


def make_policy(params, net):
    first = {"flag": True}

    @jax.jit
    def act(params, key, obs, mask, carry, reset):
        carry, logits, _ = net.apply(params, carry, (obs[None], reset[None], mask[None]))
        return jax.random.categorical(key, logits[0]), carry

    def policy_fn(key, obs, mask, carry):
        reset = jnp.full((obs.shape[0],), first["flag"])
        first["flag"] = False
        return act(params, key, obs, mask, carry, reset)

    return net, policy_fn


def batch_survival(params, net, env_cfg, n, seed, kind="trained"):
    """Survival curve over n parallel single-agent worlds.
    kind: "random", "scripted" (global-knowledge reference) or "trained"."""
    hidden = net.hidden

    @jax.jit
    def run(key):
        keys = jax.random.split(key, n + 1)
        s = jax.vmap(E.reset, in_axes=(0, None))(keys[1:], env_cfg)

        def body(c, t):
            s, carry, k = c
            k, ka = jax.random.split(k)
            obs = jax.vmap(E.observe, in_axes=(0, None))(s, env_cfg)[:, 0]
            mask = jax.vmap(E.action_mask, in_axes=(0, None))(s, env_cfg)[:, 0]
            if kind == "random":
                a = jax.random.categorical(ka, jnp.where(mask, 0.0, -1e9))
            elif kind == "scripted":
                a = jax.vmap(lambda st, mk: B.scripted_forager(st, mk[None], env_cfg))(s, mask)
            else:
                carry, logits, _ = net.apply(params, carry,
                                             (obs[None], jnp.full((n,), t == 0)[None], mask[None]))
                a = jax.random.categorical(ka, logits[0])
            s2, _, _, info = jax.vmap(E.step, in_axes=(0, 0, None))(s, a[:, None], env_cfg)
            return (s2, carry, k), (s2.alive[:, 0], info.death_cause[:, 0])

        c0 = (s, jnp.zeros((n, hidden)), keys[0])
        _, (alive, cause) = jax.lax.scan(body, c0, jnp.arange(env_cfg.max_steps))
        return alive, cause

    alive, cause = run(jax.random.PRNGKey(seed))
    alive, cause = np.asarray(alive), np.asarray(cause)
    counts = collections.Counter(E.CAUSE_NAMES[c] for c in cause[cause > 0].tolist())
    return alive.mean(axis=1), dict(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="runs/phase1/params.msgpack")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="conv")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=1920)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--record-to", type=str, default="runs/phase1/recording")
    args = ap.parse_args()

    env_cfg = EnvConfig(grid_size=args.grid, n_agents=1, max_steps=args.max_steps)
    net = make_network(env_cfg, args.hidden, args.encoder)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, args.hidden)),
                        (jnp.zeros((1, 1, E.obs_dim(env_cfg))), jnp.zeros((1, 1), bool),
                         jnp.ones((1, 1, E.N_ACTIONS), bool)))
    params = serialization.from_bytes(template, Path(args.params).read_bytes())

    checkpoints = [t for t in (250, 500, 1000, 1500) if t < args.max_steps] + [args.max_steps]
    report = {}
    for name in ["random", "scripted", "trained"]:
        curve, causes = batch_survival(params, net, env_cfg, args.episodes, args.seed, name)
        report[name] = {"alive_fraction": {t: round(float(curve[t - 1]), 3) for t in checkpoints},
                        "death_causes": causes}
    print(json.dumps(report, indent=1))

    _, policy_fn = make_policy(params, net)
    rec = record_episode(policy_fn, jnp.zeros((1, args.hidden)), seed=args.seed, cfg=env_cfg)
    rec.save(args.record_to)
    kinds = collections.Counter(ev["type"] for ev in rec.events)
    print(f"recorded {len(rec.actions)} ticks, {len(rec.events)} events {dict(kinds)} -> {args.record_to}")


if __name__ == "__main__":
    main()
