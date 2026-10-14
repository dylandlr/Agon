"""Measure steady-state Phase 2 throughput on the current device, excluding
compile time, and estimate wall time and HPRC Service Units for a target run.

    python scripts/benchmark.py --agents 8 --worlds 64 --target-agent-steps 50000000
    python scripts/benchmark.py --preset society --worlds 16 32 64 128 --target-agent-steps 1e8

With several --worlds values it tries each (smallest first), reports out-of-memory instead of
failing, and recommends the fastest that fit.
"""
from __future__ import annotations

import argparse
import time

import jax

from lifesim.config import EnvConfig, with_reproduction
from lifesim.ppo_multi import MAConfig, make_train
from lifesim.presets import PRESETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--worlds", type=int, nargs="+", default=[64])
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--founders", type=int, default=0)
    ap.add_argument("--reproduction", action="store_true")
    ap.add_argument("--reward-mode", default="acts")
    ap.add_argument("--world", default="classic")
    ap.add_argument("--slate", default="blank")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="conv")
    ap.add_argument("--world-model", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--target-agent-steps", type=float, default=5e7)
    ap.add_argument("--su-per-hour", type=float, default=80.0,
                    help="Grace: 72 per A100 plus effective cores (8 cores -> 80)")
    pre, _ = ap.parse_known_args()
    if pre.preset:
        ap.set_defaults(**PRESETS[pre.preset])
    args = ap.parse_args()

    env_cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, reward_mode=args.reward_mode,
                        skills=args.slate == "caveman")
    if args.world == "living":
        from lifesim.config import living_world
        env_cfg = living_world(env_cfg)
    from lifesim.config import with_time_scale
    env_cfg = with_time_scale(env_cfg, getattr(args, "days_per_year", 40), getattr(args, "days_per_bio_year", 1.0))
    if args.reproduction:
        env_cfg = with_reproduction(env_cfg, n_founders=args.founders or None)
    from lifesim.config import with_tick_minutes
    env_cfg = with_tick_minutes(env_cfg, getattr(args, "tick_minutes", 30.0))
    results = {}
    for worlds in sorted(args.worlds):
        try:
            results[worlds] = run_one(env_cfg, args, worlds)
        except Exception as e:                     # XLA reports running out of GPU memory as an error
            msg = str(e).splitlines()[0][:120]
            print(f"--- {worlds} worlds: failed ({'out of memory' if 'memory' in msg.lower() or 'RESOURCE' in msg else msg})")
            break                                  # more worlds will not fit either
    if results:
        best = max(results, key=results.get)
        hours = args.target_agent_steps / results[best] / 3600
        print(f"RECOMMENDED: --worlds {best}  ({results[best]:,.0f} agent-steps/s; "
              f"{args.target_agent_steps:,.0f} agent-steps in {hours:.1f} h)")


def run_one(env_cfg, args, worlds):
    cfg = MAConfig(num_worlds=worlds, num_steps=args.steps, total_ticks=10 ** 9,
                   encoder=args.encoder, num_minibatches=min(4, worlds), world_model=args.world_model,
                   day_ticks=env_cfg.day_length, tick_scale=env_cfg.tick_minutes / 30.0,
                   memory_span=env_cfg.day_length)
    _, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    t = time.time()
    for _ in range(2):          # the second call can retrace once (weak types settle)
        runner, m = update(runner, pop)
    jax.block_until_ready(m["alive"])
    compile_s = time.time() - t
    t = time.time()
    for _ in range(args.updates):
        runner, m = update(runner, pop)
    jax.block_until_ready(m["alive"])
    per_update = (time.time() - t) / args.updates
    sps = args.steps * worlds * args.agents / per_update
    hours = args.target_agent_steps / sps / 3600
    print(f"--- {worlds} worlds on {jax.devices()[0]}")
    print(f"compile: {compile_s:.1f}s  per update: {per_update:.2f}s  agent-steps/s: {sps:,.0f}")
    print(f"{args.target_agent_steps:,.0f} agent-steps -> {hours:.1f} h -> ~{hours * args.su_per_hour:,.0f} SUs", flush=True)
    return sps


if __name__ == "__main__":
    main()
