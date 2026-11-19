"""Phase 1: train a single agent to survive. Gate: agents reliably survive a
full seasonal cycle (a year = 2000 ticks at default settings).

    python scripts/train_phase1.py --timesteps 5000000 --out runs/phase1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.ppo import PPOConfig, make_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=1920)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="conv")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="runs/phase1")
    args = ap.parse_args()

    env_cfg = EnvConfig(grid_size=args.grid, n_agents=1, max_steps=args.max_steps)
    cfg = PPOConfig(num_envs=args.num_envs, total_timesteps=args.timesteps, seed=args.seed,
                    ent_coef=args.ent_coef, encoder=args.encoder,
                    gamma=args.gamma)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net, init, update = make_train(env_cfg, cfg)
    runner = init(jax.random.PRNGKey(cfg.seed))
    history, window = [], []
    t0 = time.time()
    for u in range(cfg.num_updates):
        runner, m = update(runner)
        m = {k: (np.asarray(v) if k == "deaths_by_cause" else float(v)) for k, v in m.items()}
        if m["episodes"] > 0:
            window.append(m)
        if (u + 1) % args.log_every == 0 or u == cfg.num_updates - 1:
            steps = (u + 1) * cfg.num_envs * cfg.num_steps
            eps = sum(w["episodes"] for w in window)
            life = (sum(w["mean_life"] * w["episodes"] for w in window) / eps) if eps else math.nan
            causes = sum(w["deaths_by_cause"] for w in window) if window else np.zeros(1)
            deaths = {E.CAUSE_NAMES[c]: int(n) for c, n in enumerate(np.atleast_1d(causes))
                      if c > 0 and n > 0}
            row = {"update": u + 1, "steps": steps, "episodes": eps, "mean_life": life,
                   "survived_full_episode": sum(w["timeouts"] for w in window),
                   "deaths": deaths,
                   "entropy": m["entropy"], "sps": steps / (time.time() - t0)}
            history.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}),
                  flush=True)
            window = []
    with open(out / "params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(runner[0].params))
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
