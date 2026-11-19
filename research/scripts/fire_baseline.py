"""Calibration and chance baseline for fire. Runs random agents for one year in
many worlds and reports (1) how often lightning starts fires and how far they
spread, and (2) how often each fire-ladder step happens by pure chance. Any
learned 'discovery' has to beat these chance rates.

    python scripts/fire_baseline.py --worlds 32 --agents 8 --grid 32
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=1920)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--immortal", action="store_true",
                    help="refill needs each tick, so chance rates are measured on living agents")
    ap.add_argument("--lightning", type=float, default=None)
    ap.add_argument("--spread", type=float, default=None)
    args = ap.parse_args()
    over = {}
    if args.lightning is not None:
        over["lightning_prob"] = args.lightning
    if args.spread is not None:
        over["spread_prob"] = args.spread
    cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, max_steps=args.ticks, **over)

    def run(key):
        k0, k1 = jax.random.split(key)
        s = E.reset(k0, cfg)
        forest = (s.terrain == E.FOREST).sum()

        def body(c, _):
            s, k, ever = c
            k, ka = jax.random.split(k)
            if args.immortal:
                one = jnp.ones_like(s.health)
                s = s._replace(health=one, satiety=one, hydration=one, energy=one,
                               alive=jnp.ones_like(s.alive), body_temp=jnp.zeros_like(s.body_temp))
            m = E.action_mask(s, cfg)
            a = jax.random.categorical(ka, jnp.where(m, 0.0, -1e9))
            s, _, _, info = E.step(s, a, cfg)
            ever = ever | (s.fire > 0)
            out = {"alive": s.alive.sum(), "burning": info.burning_cells,
                   "lightning": info.lightning, "near_fire": info.near_fire.sum(),
                   "fed_fire": info.fed_fire.sum(), "meat_to_heat": info.meat_to_heat.sum(),
                   "ate_cooked": info.ate_cooked.sum(), "pulled_burning_wood": info.torch_lit.sum(),
                   "fires_by_spark": (info.ignited == 1).sum(),
                   "fires_by_burning_wood": (info.ignited == 2).sum(),
                   "burn_deaths": (info.died & (info.death_cause == E.CAUSE_BURNS)).sum()}
            return (s, k, ever), out

        (s, _, ever), out = jax.lax.scan(body, (s, k1, jnp.zeros((cfg.grid_size,) * 2, bool)),
                                         None, args.ticks)
        return out, (ever & (s.terrain == E.FOREST)).sum() / jnp.maximum(forest, 1)

    out, burned_frac = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(args.seed), args.worlds))
    out = jax.tree_util.tree_map(np.asarray, out)
    alive_ticks = out["alive"].sum()
    per_k = lambda x: round(1000 * float(x.sum()) / alive_ticks, 4)
    report = {
        "immortal": args.immortal, "lightning_prob": cfg.lightning_prob, "spread_prob": cfg.spread_prob,
        "worlds": args.worlds, "grid": args.grid, "agents": args.agents, "ticks": args.ticks,
        "lightning_fires_per_world_year": round(float(out["lightning"].sum()) / args.worlds, 2),
        "max_burning_cells_any_world": int(out["burning"].max()),
        "mean_forest_fraction_ever_burned": round(float(np.mean(burned_frac)), 3),
        "max_forest_fraction_ever_burned": round(float(np.max(burned_frac)), 3),
        "burn_deaths_total": int(out["burn_deaths"].sum()),
        "random_agents_per_1000_agent_ticks": {k: per_k(out[k]) for k in
            ["near_fire", "fed_fire", "meat_to_heat", "ate_cooked", "pulled_burning_wood",
             "fires_by_spark", "fires_by_burning_wood"]},
        "random_agents_totals": {k: int(out[k].sum()) for k in
            ["fed_fire", "meat_to_heat", "ate_cooked", "pulled_burning_wood",
             "fires_by_spark", "fires_by_burning_wood"]},
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
