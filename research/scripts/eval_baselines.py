"""Random and scripted baselines in the same multi-agent setting as training.

    python scripts/eval_baselines.py --agents 8 --grid 32 --worlds 32
    python scripts/eval_baselines.py --agents 8 --grid 32 --scarcity 0.4

Every agent in a world follows the same baseline policy for whole episodes (one
year, or until all die). Reports the measures the training log uses, so the
numbers sit directly beside a trained run:

  random   a uniformly random valid action each tick: the lower bound
  scripted a hand-coded forager with full map knowledge, one per agent: an
           upper reference for individual survival (it never cooperates)

Writes runs/baselines/<setting>.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import baselines as B, env as E
from lifesim.config import EnvConfig


def evaluate(policy: str, env_cfg: EnvConfig, worlds: int, seed: int) -> dict:
    n = env_cfg.n_agents
    keys = jax.random.split(jax.random.PRNGKey(seed), worlds)
    states = jax.vmap(lambda k: E.reset(k, env_cfg))(keys)

    def act(key, s):
        mask = E.action_mask(s, env_cfg)
        if policy == "random":
            return B.random_action(key, mask)
        return jax.vmap(lambda i: B.scripted_forager(s, mask, env_cfg, i))(jnp.arange(n))

    def world_step(carry, key):
        s = carry
        a = act(key, s)
        alive_before = s.alive
        nxt, reward, done, info = E.step(s, a, env_cfg)
        live = alive_before.astype(jnp.float32)
        died = info.died
        out = {
            "alive_ticks": live.sum(),
            # A world with nobody left does not count: training restarts such worlds at once
            "live_world_ticks": jnp.any(alive_before).astype(jnp.float32),
            "own_reward": ((reward - info.moral_reward) * live).sum(),   # base reward, as in training
            "deaths": died.sum(),
            "age_at_death": jnp.where(died, s.age + 1, 0).sum(),
            "cause": jnp.stack([(died & (info.death_cause == c)).sum() for c in range(len(E.CAUSE_NAMES))]),
            "hits": info.hits.sum(),
            "gifts": jnp.any(info.gifts, axis=-1).sum(),
            "deterred": info.deterred.sum(),
            "company": info.company.sum(),
            "large_kills": info.kills[:, 1].sum(),
        }
        return nxt, out

    @jax.jit
    def run_world(s, key):
        last, out = jax.lax.scan(world_step, s, jax.random.split(key, env_cfg.max_steps))
        tot = jax.tree_util.tree_map(lambda x: x.sum(0), out)
        tot["survivors"] = last.alive.sum()
        tot["reached_year_end"] = jnp.any(last.alive)
        return tot

    t0 = time.time()
    res = jax.vmap(run_world)(states, jax.random.split(jax.random.PRNGKey(seed + 1), worlds))
    res = jax.tree_util.tree_map(np.asarray, res)
    alive_ticks = max(float(res["alive_ticks"].sum()), 1.0)
    deaths = float(res["deaths"].sum())
    per_k = lambda x: round(1000 * float(np.sum(x)) / alive_ticks, 3)
    return {
        "policy": policy,
        "worlds": worlds,
        "seconds": round(time.time() - t0, 1),
        "mean_age_at_death": round(float(res["age_at_death"].sum()) / deaths, 1) if deaths else None,
        "share_alive_at_episode_end": round(float(res["survivors"].sum()) / (worlds * n), 3),
        "share_of_worlds_reaching_year_end": round(float(res["reached_year_end"].mean()), 3),
        "survival_score_per_world_day": round((env_cfg.alive_reward * alive_ticks - env_cfg.death_penalty * deaths)
                                              / max(float(res["live_world_ticks"].sum()), 1.0) * env_cfg.day_length, 4),
        "global_reward_per_world_day": round(float(res["own_reward"].sum())
                                             / max(float(res["live_world_ticks"].sum()), 1.0)
                                             * env_cfg.day_length, 4),
        "deaths": {E.CAUSE_NAMES[c]: int(k) for c, k in enumerate(res["cause"].sum(0)) if c and k},
        "per_1000_agent_ticks": {"strikes_on_agents": per_k(res["hits"]), "gifts": per_k(res["gifts"]),
                                 "deterred": per_k(res["deterred"])},
        "in_company_share": round(float(res["company"].sum()) / alive_ticks, 3),
        "large_prey_kills": int(res["large_kills"].sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--scarcity", type=float, default=1.0, help="plant abundance, as in training")
    ap.add_argument("--max-steps", type=int, default=None, help="episode length (default: one year)")
    ap.add_argument("--seed", type=int, default=12345, help="evaluation worlds, unseen in training")
    ap.add_argument("--policies", nargs="+", default=["random", "scripted"], choices=["random", "scripted"])
    ap.add_argument("--out-dir", default="runs/baselines")
    args = ap.parse_args()

    cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, plant_abundance=args.scarcity)
    if args.max_steps:
        cfg = dataclasses.replace(cfg, max_steps=args.max_steps)
    results = {"setting": {"agents": args.agents, "grid": args.grid, "scarcity": args.scarcity,
                           "episode_ticks": cfg.max_steps, "worlds": args.worlds, "seed": args.seed},
               "baselines": {}}
    print(f"{args.agents} agents, {args.grid}x{args.grid}, food {args.scarcity}, {args.worlds} worlds x "
          f"{cfg.max_steps} ticks (compiling first)", flush=True)
    for policy in args.policies:
        r = evaluate(policy, cfg, args.worlds, args.seed)
        results["baselines"][policy] = r
        print(f"{policy:>9}: lifespan {r['mean_age_at_death']}  alive at end {r['share_alive_at_episode_end']:.0%}  "
              f"worlds reaching year end {r['share_of_worlds_reaching_year_end']:.0%}  "
              f"global reward/world-day {r['global_reward_per_world_day']}  deaths {r['deaths']}  ({r['seconds']}s)",
              flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"a{args.agents}_g{args.grid}_f{args.scarcity:g}_t{cfg.max_steps}.json"
    json.dump(results, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
