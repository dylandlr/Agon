"""Are the decisions intelligent? Scores need-appropriate choices, against chance.

Three kinds of agent live the same seeded worlds for the same time:
  random     uniform over the actions the world allows (no brain): the chance level
  untrained  freshly initialised networks (the baseline a training run must beat)
  trained    a run's own agents (--run), when given

Each tick of each awake agent is checked against a plain rule of survival, and we count how often
the agent chose the act that rule asks for:

  drink_when_thirsty   hydration < 0.3     -> drink, or seek_water
  eat_when_hungry      satiety   < 0.3     -> eat, or seek_food
  rest_at_night        night               -> sleep or rest (asleep: staying so counts)
  flee_from_predator   a predator within 3 cells -> flee

plus survival: share alive at the end and deaths by cause. "Lift" is the agent's rate over the random
agents' rate: 1.0 is chance, higher means the choice follows the need. A rule with too few ticks
behind it is reported as n/a, not as a number.

Nothing here rewards the rules: it only reads the agents' choices and the world's state. A network
that merely does well on these four rules is not proven clever, but one that does not beat chance on
them is not.

  python scripts/decision_check.py                       # untrained vs random, 3 seeds
  python scripts/decision_check.py --run runs/exp_x/validate/caveman
  python scripts/decision_check.py --days 20 --seeds 5 --out runs/decision_check.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifesim import env as E                                           # noqa: E402
from lifesim.config import EnvConfig, living_world, with_reproduction, with_tick_minutes, with_time_scale  # noqa: E402
from lifesim.policy import make_network, sample_outputs                # noqa: E402
from lifesim.threats import with_threats                               # noqa: E402

LOW = 0.3                    # a need is "low" below this (needs run 0 = empty .. 1 = full)
NEAR = 3                     # a predator this close (cells, either axis) is a threat
MIN_TICKS = 30              # fewer checked ticks than this: not reported as a number
RULES = {
    "drink_when_thirsty": ({E.DRINK, E.SEEK_WATER}, "hydration < 0.3"),
    "eat_when_hungry": ({E.EAT, E.SEEK_FOOD}, "satiety < 0.3"),
    "rest_at_night": ({E.SLEEP, E.REST}, "night"),
    "flee_from_predator": ({E.FLEE}, "predator within 3 cells"),
}


def build_cfg(args, run: Path | None):
    """The world the agents are judged in: the run's own, or the standard living world."""
    if run is not None and (run / "history.json").exists():
        h = json.load(open(run / "history.json"))["args"]
        cfg = EnvConfig(grid_size=h["grid"], n_agents=h["agents"], skills=h.get("slate") == "caveman",
                        reward_mode=h.get("reward_mode", "acts"), plant_abundance=h.get("scarcity", 1.0))
        cfg = with_time_scale(cfg, h.get("days_per_year", 40), h.get("days_per_bio_year", 1.0))
        if h.get("world") == "living":
            cfg = living_world(cfg, underground=bool(h.get("underground", False)), downed=bool(h.get("downed", False)),
                               grazing=bool(h.get("grazing", False)), clothing=bool(h.get("clothing", False)),
                               ecology=bool(h.get("ecology", False)), fouling=bool(h.get("fouling", False)),
                               excretion=bool(h.get("excretion", True)))
        cfg = dataclasses.replace(cfg, language_mode=h.get("language", "emergent"), writing=bool(h.get("writing", False)))
        if h.get("reproduction"):
            cfg = with_reproduction(cfg, n_founders=h.get("founders") or None)
        cfg = with_threats(cfg, food=h.get("food", 1.0), predators=h.get("predators", 1.0),
                           disease=h.get("disease", 1.0), hazards=h.get("hazards", 1.0))
        return with_tick_minutes(cfg, h.get("tick_minutes", 30.0)), h
    cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, skills=True)
    cfg = living_world(cfg, underground=args.underground, excretion=True)
    cfg = with_threats(cfg)
    return with_tick_minutes(cfg, args.tick_minutes), {}


def _policy(net, hidden):
    A, M = E.N_ACTIONS, E.EXTRA_LOGITS

    @jax.jit
    def act(params, carry, obs, mask, reset, k):
        full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (M,), bool)], -1)
        carry, logits, _ = jax.vmap(lambda p, c, o, m: net.apply(
            p, c, (o[None, None], reset[None, None], m[None, None])))(params, carry, obs, full)
        a, msg = sample_outputs(k, logits[:, 0, 0])
        return carry, a, msg
    return act


def live(cfg, kind, seed, days, hidden, encoder, run=None, world_model=True):
    """One world, one kind of agent: per rule (checked ticks, matching choices), and the outcome."""
    key = jax.random.PRNGKey(seed)
    k_env, key = jax.random.split(key)
    n = cfg.n_agents
    state = E.reset(k_env, cfg)
    if kind != "random":
        if kind == "trained":
            import export_playback as X
            net, params, pop = X.load_population(run, cfg, hidden, encoder)
            state = E.reset(k_env, cfg, jax.tree_util.tree_map(jnp.asarray, pop.genotype), jnp.asarray(pop.appearance))
        else:
            net = make_network(cfg, hidden, encoder, messages=True, world_model=world_model)
            dummy = (jnp.zeros((1, 1, E.obs_dim(cfg))), jnp.zeros((1, 1), bool),
                     jnp.ones((1, 1, E.N_ACTIONS + E.EXTRA_LOGITS), bool))
            key, k_net = jax.random.split(key)
            params = jax.vmap(lambda k: net.init(k, jnp.zeros((1, hidden)), dummy))(jax.random.split(k_net, n))
        act = _policy(net, hidden)
        carry = jnp.zeros((n, 1, hidden))
    step = jax.jit(lambda s, a, m: E.step(s, a, cfg, m))
    observe = jax.jit(lambda s: (E.observe(s, cfg), E.action_mask(s, cfg)))
    night = jax.jit(lambda t: E.is_night(t, cfg))
    seen = {r: [0, 0] for r in RULES}
    deaths = {}
    for t in range(int(days * cfg.day_length)):
        obs, mask = observe(state)
        key, k = jax.random.split(key)
        if kind == "random":
            ka, km = jax.random.split(k)
            a = jax.random.categorical(ka, jnp.where(mask, 0.0, -1e9))
            msg = jnp.zeros((n, E.MSG_LEN), jnp.int32)
        else:
            carry, a, msg = act(params, carry, obs, mask, jnp.asarray(t == 0), k)
        alive = np.asarray(state.alive) & ~np.asarray(state.asleep)
        awake_or_not = np.asarray(state.alive)
        an = np.asarray(a)
        hyd, sat = np.asarray(state.hydration), np.asarray(state.satiety)
        pp = np.asarray(state.pos)
        pr = np.asarray(state.predators.pos)[np.asarray(state.predators.alive)]
        near = np.zeros(n, bool)
        if len(pr):
            near = (np.abs(pp[:, None, :] - pr[None, :, :]).max(-1) <= NEAR).any(1)
        conds = {"drink_when_thirsty": alive & (hyd < LOW), "eat_when_hungry": alive & (sat < LOW),
                 "rest_at_night": awake_or_not & bool(night(state.t)), "flee_from_predator": alive & near}
        for r, (acts, _) in RULES.items():
            m = conds[r]
            seen[r][0] += int(m.sum())
            seen[r][1] += int(np.isin(an[m], list(acts)).sum())
        state, _, _, info = step(state, a, msg)
        for c in np.asarray(info.death_cause)[np.asarray(info.died)]:
            name = E.CAUSE_NAMES[int(c)]
            deaths[name] = deaths.get(name, 0) + 1
    return {"rules": seen, "alive_end": int(np.asarray(state.alive).sum()), "agents": n, "deaths": deaths}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, default=None, help="a train_phase2 output folder: adds its trained agents")
    ap.add_argument("--days", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="mlp")
    ap.add_argument("--tick-minutes", type=float, default=30.0)
    ap.add_argument("--underground", action="store_true")
    ap.add_argument("--out", type=str, default=None, help="write the numbers as JSON")
    args = ap.parse_args()

    run = Path(args.run) if args.run else None
    cfg, h = build_cfg(args, run)
    hidden, encoder = (h.get("hidden", args.hidden), h.get("encoder", args.encoder)) if h else (args.hidden, args.encoder)
    kinds = ["random", "untrained"] + (["trained"] if run else [])
    results = {k: [] for k in kinds}
    for seed in range(args.seeds):
        for kind in kinds:
            results[kind].append(live(cfg, kind, 100 + seed, args.days, hidden, encoder, run,
                                      world_model=bool(h.get("world_model", True))))
            print(f"seed {seed} {kind}: alive at end {results[kind][-1]['alive_end']}/{results[kind][-1]['agents']}", flush=True)

    def rate(kind, rule):
        n = sum(r["rules"][rule][0] for r in results[kind])
        k = sum(r["rules"][rule][1] for r in results[kind])
        return (k / n if n >= MIN_TICKS else None), n

    table = {kind: {r: dict(zip(("rate", "ticks"), rate(kind, r))) for r in RULES} for kind in kinds}
    for kind in kinds:
        table[kind]["alive_share_end"] = float(np.mean([r["alive_end"] / r["agents"] for r in results[kind]]))
    lines = [f"{args.days:g} days x {args.seeds} seeds, {cfg.n_agents} agents, grid {cfg.grid_size}, "
             f"{cfg.tick_minutes:g}-minute ticks. Rate = share of checked ticks where the agent chose the act the rule asks for.",
             "", "| rule | when | " + " | ".join(f"{k} (ticks)" for k in kinds) + " | lift over chance |", "|---|---|" + "---|" * (len(kinds) + 1)]
    for r, (_, when) in RULES.items():
        cells, base = [], table["random"][r]["rate"]
        for k in kinds:
            v = table[k][r]
            cells.append("n/a" if v["rate"] is None else f"{v['rate']:.1%} ({v['ticks']})")
        best = table[kinds[-1]][r]["rate"]
        lift = "n/a" if (best is None or not base) else f"{best / base:.2f}x"
        lines.append(f"| {r} | {when} | " + " | ".join(cells) + f" | {lift} |")
    lines.append("| alive at end | | " + " | ".join(f"{table[k]['alive_share_end']:.0%}" for k in kinds) + " | |")
    print("\n" + "\n".join(lines))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"table": table, "runs": results, "cfg": {
            "agents": cfg.n_agents, "grid": cfg.grid_size, "tick_minutes": cfg.tick_minutes}}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
