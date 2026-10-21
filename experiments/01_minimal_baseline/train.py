"""Minimal multi-agent setup: does recurrent PPO learn to survive, and does cooperation appear?

The full society design layers many mechanics on the learner at once. This script trains the
smallest setting where the core question still means something: a few individuals with their own
networks, a classic 32 x 32 world (food, water, weather, cold, disease, prey, large prey that
need group hunting, predators deterred by company) and survival as the only reward, felt only
through one's own body. Nothing pays for acts toward others, and others' fortunes never enter
one's own reward, so any cooperation that appears is learned because it keeps the learner alive.

    python experiments/01_minimal_baseline/train.py --baselines            # random and scripted references
    python experiments/01_minimal_baseline/train.py --agent-steps 2e7 --run-name s0
    python experiments/01_minimal_baseline/train.py --agent-steps 3e7 --run-name s0 --resume   # extend

Outputs stay in this experiment's folder: checkpoints in runs/<run-name>/ (git-ignored),
history.json and baselines.json in results/ (committed). Nothing here is in lifesim/ or
train_phase2.py, so it does not change code_version: runs made with the full design elsewhere
stay resumable. See README.md for the experiment's design and decisions.
"""
from __future__ import annotations

import os

# Allocate GPU memory on demand: JAX's default reservation of 75% fails on small GPUs under WSL.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from lifesim import checkpoint as ckpt  # noqa: E402
from lifesim import env as E  # noqa: E402
from lifesim.config import EnvConfig  # noqa: E402
from lifesim.ppo_multi import MAConfig, make_train  # noqa: E402
from lifesim.presets import code_version  # noqa: E402
from train_phase2 import gini  # noqa: E402


def minimal_env_config(agents: int = 8, grid: int = 32, scarcity: float = 1.0, gamma: float = 0.997) -> EnvConfig:
    """The classic world (no biomes, moving weather, ecology or caves), without reproduction,
    with the extra layers below turned off. Everything left is a defaults-calibrated mechanic."""
    return EnvConfig(
        grid_size=grid, n_agents=agents, plant_abundance=scarcity,
        max_steps=EnvConfig().year_length,         # one 40-day year per episode, as in training
        # Reward: own survival only. Survival mode is +alive, -death and the potential-based
        # change in one's own well-being; the care weights below are how others' visible
        # fortunes would enter it, and all are zero. shaping_gamma must equal the discount.
        reward_mode="survival", shaping_gamma=gamma,
        care_empathy=0.0, care_kin=0.0, care_reciprocity=0.0, care_revenge=0.0, care_sadism=0.0,
        care_inequity=0.0, care_attachment=0.0, saint_rate=0.0,
        # Others' suffering must not reach one's own reward through stress either: heartache
        # (seeing wounds, grief) and missing attached companions raise stress, and calm is part
        # of well-being. Loneliness and social buffering stay: they are the body's response to
        # company in general, part of what makes staying together worth something.
        heartache_empathy=0.0, heartache_grief=0.0, attach_absence=0.0, attach_calm=0.0,
        # Simpler world: no sleep need (sleepiness stays 0, SLEEP is never available), no
        # minerals, one species per animal group, no bees or termites, no hides, bones or fiber.
        sleep_pressure=0.0, mineral_abundance=0.0, fauna_diversity=False,
        hive_density=0.0, mound_density=0.0, byproducts=False,
    )


def minimal_ma_config(env_cfg: EnvConfig, worlds: int, steps: int, total_ticks: int, seed: int,
                      encoder: str, gamma: float, alpha: float) -> MAConfig:
    """Plain recurrent PPO with one network per individual: no learning during sleep (there is
    no sleep, and that store would wait for it forever), no imitation, no world model, and the
    same learning rate for everyone (the lr_scale gene is not used)."""
    return MAConfig(num_worlds=worlds, num_steps=steps, total_ticks=total_ticks, seed=seed,
                    encoder=encoder, gamma=gamma, alpha=alpha, per_agent_lr=False,
                    num_minibatches=min(4, worlds), imitation_coef=0.0, sleep_learning=False,
                    world_model=False, day_ticks=env_cfg.day_length, tick_scale=1.0,
                    memory_span=env_cfg.day_length)


def row_from(acc, m, u, cfg: MAConfig, env_cfg: EnvConfig, agents: int, sps: float) -> dict:
    """One log line: survival first (did it learn), then the cooperation counters."""
    alive = max(float(acc["alive"]), 1.0)
    deaths = float(acc["deaths"])
    per_k = lambda x: round(1000 * float(x) / alive, 3)
    ends = int(acc["world_ends"])
    return {
        "update": u + 1, "agent_steps": (u + 1) * cfg.num_steps * cfg.num_worlds * agents,
        "mean_age_at_death": round(float(acc["age_at_death"]) / deaths, 1) if deaths else None,
        "share_alive_at_episode_end": round(float(acc["survivors_at_end"]) / (ends * agents), 3) if ends else None,
        "share_of_worlds_reaching_year_end": round(float(acc["full_years"]) / ends, 3) if ends else None,
        "survival_score_per_world_day": round((env_cfg.alive_reward * float(acc["alive"])
                                               - env_cfg.death_penalty * deaths)
                                              / max(float(acc["world_ticks"]), 1.0) * env_cfg.day_length, 4),
        "deaths": {E.CAUSE_NAMES[c]: int(k) for c, k in enumerate(acc["cause"]) if c and k},
        "in_company_share": round(float(acc["company"]) / alive, 3),
        "winter_huddle_share": round(float(acc["winter_huddle_ticks"]) / max(float(acc["winter_night_ticks"]), 1.0), 3),
        "per_1000_agent_ticks": {"deterred": per_k(acc["deterred"]), "large_prey_hits": per_k(acc["large_prey_hits"]),
                                 "meat_eaten": per_k(acc["meat_eaten"]), "strikes_on_agents": per_k(acc["hits"]),
                                 "gifts": per_k(acc["gifts"]), "touches": per_k(acc["touches"]),
                                 "messages": per_k(acc["messages"])},
        "kills": {"small_prey": int(acc["kills"][0]), "large_prey": int(acc["kills"][1]),
                  "predator": int(acc["kills"][2])},
        "hunting_gini": round(gini(acc["large_kills_by_agent"]), 3),
        "entropy": round(float(m["entropy"]), 3),
        "agent_sps": round(sps, 1),
    }


def run_baselines(env_cfg: EnvConfig, worlds: int, seed: int, results: Path):
    """Random (floor) and scripted forager (individual ceiling: full map knowledge, never
    cooperates) in exactly this world, with the same measures as the training log."""
    from eval_baselines import evaluate
    found = {}
    for policy in ("random", "scripted"):
        r = evaluate(policy, env_cfg, worlds, seed)
        found[policy] = r
        print(json.dumps(r), flush=True)
    results.mkdir(parents=True, exist_ok=True)
    json.dump({"code_version": code_version(), "worlds": worlds, "seed": seed, "baselines": found},
              open(results / "baselines.json", "w"), indent=1)
    print("wrote", results / "baselines.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--worlds", type=int, default=8, help="parallel worlds (8 fits a 6 GB GPU under WSL)")
    ap.add_argument("--steps", type=int, default=128, help="ticks per PPO update")
    ap.add_argument("--agent-steps", type=float, default=2e7,
                    help="training budget (the learning rate is constant, so --resume with more extends a run)")
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="mlp")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--alpha", type=float, default=0.0, help="0 own reward, 1 group-mean reward")
    ap.add_argument("--scarcity", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--run-name", default="s0", help="runs/<name>/ for checkpoints, results/<name>/ for history")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--baselines", action="store_true", help="evaluate random and scripted policies, then exit")
    ap.add_argument("--baseline-worlds", type=int, default=32)
    ap.add_argument("--baseline-seed", type=int, default=12345, help="evaluation worlds, unseen in training")
    args = ap.parse_args()
    args.code_version = code_version()

    env_cfg = minimal_env_config(args.agents, args.grid, args.scarcity, args.gamma)
    if args.baselines:
        return run_baselines(env_cfg, args.baseline_worlds, args.baseline_seed, HERE / "results")

    total_ticks = int(args.agent_steps // (args.worlds * args.agents))
    cfg = minimal_ma_config(env_cfg, args.worlds, args.steps, total_ticks, args.seed, args.encoder,
                            args.gamma, args.alpha)
    out, res = HERE / "runs" / args.run_name, HERE / "results" / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    net, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(cfg.seed))
    history, acc, start = [], None, 0
    ck_path, hist_path = out / "checkpoint.pkl", res / "history.json"
    if args.resume and ck_path.exists():
        old = json.load(open(hist_path))["args"]
        for k in ("code_version", "agents", "grid", "worlds", "steps", "encoder", "alpha", "scarcity", "seed"):
            if old.get(k) != getattr(args, k):
                raise SystemExit(f"{args.run_name}: --{k} was {old.get(k)!r}, now {getattr(args, k)!r}; "
                                 "use a new --run-name")
        state = ckpt.load(ck_path)
        runner, pop, start = state["runner"], state["pop"], int(state["update"])
        history = json.load(open(hist_path))["history"]
        print(f"resumed from update {start}", flush=True)
    elif ck_path.exists():
        raise SystemExit(f"{out} already has a checkpoint: pass --resume, or use a new --run-name")
    print(f"backend={jax.default_backend()} updates={cfg.num_updates} "
          f"({cfg.num_updates * cfg.num_steps * cfg.num_worlds * args.agents:,} agent-steps); "
          f"first update compiles for a few minutes", flush=True)
    t0, u0 = time.time(), start
    for u in range(start, cfg.num_updates):
        runner, m = update(runner, pop)
        m = jax.tree_util.tree_map(np.asarray, m)
        acc = m if acc is None else jax.tree_util.tree_map(lambda a, x: a + x, acc, m)
        if (u + 1) % args.log_every == 0 or u == cfg.num_updates - 1:
            sps = (u + 1 - u0) * cfg.num_steps * cfg.num_worlds * args.agents / (time.time() - t0)
            row = row_from(acc, m, u, cfg, env_cfg, args.agents, sps)
            history.append(row)
            print(json.dumps(row), flush=True)
            acc = None
        if (u + 1) % args.ckpt_every == 0 or u == cfg.num_updates - 1:
            ckpt.save(ck_path, {"runner": runner, "pop": pop, "update": u + 1})
            json.dump({"args": vars(args), "history": history}, open(hist_path, "w"), indent=1)
    print(f"saved: checkpoint in {out}, history in {hist_path}")


if __name__ == "__main__":
    main()
