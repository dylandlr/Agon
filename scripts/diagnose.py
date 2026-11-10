"""Why do they die in the cold, how well are they learning, and what can they do? Read from a trained run.

    python scripts/diagnose.py --run runs/exp/validate/caveman --days 80 --seeds 3 --out runs/exp/diagnose
    python scripts/diagnose.py --run R --sections cold,probe        # only some sections
    python scripts/diagnose.py --run R --reuse --out DIR            # redo the report from saved rollouts
    python scripts/diagnose.py --untrained --days 6 --seeds 1       # smoke test with fresh networks

Nothing here trains or changes the world's rules. Agents live the run's own world; we record what
they do and what happens to them, and ask the brain a few questions.

Sections (--sections, default all):

  survival  Deaths by cause per agent-year for three kinds of agent (see "reflex" below) and for
            random agents. Cold deaths split into caught outside shelter / frozen inside it.
  cold      Forensics for each cold death (policy agents). Exposure starts when body temperature
            first falls below -0.3 and stays under it until death. At that moment: were they in
            shelter (forest, cave, underground); if not, how far was the nearest shelter, and how far
            the nearest shelter they had ever stood on or near ("visited")? Each death lands in one
            bin: froze inside shelter / knew shelter within 12 cells and stayed out / shelter within
            12 cells that they had never visited / no shelter within 12 cells.
  probe     Does the brain perceive cold and act on it? At intervals, for every awake agent, set its
            body temperature to 0, -0.3, -0.5, -0.7, -0.9 (nothing else changes, its memory is as
            it was) and read the critic's value and the action probabilities. If the value does not
            fall with cold, credit assignment did not reach it. If the value falls but the shelter
            and rest probabilities do not rise, it sees the danger and has no remedy. One tick of
            counterfactual, so it understates sensitivity built up over hours.
  reflex    The same agents with one rule laid over their choices: when cold (body < -0.3), or from two
            hours before dark through the night, an agent that is not in shelter takes SEEK_SHELTER.
            Nothing else is overridden. If cold deaths vanish, the world is survivable and the
            missing piece is the decision (or its timing); if they remain, shelter is not enough.
  fire      The fire ladder, per 1000 awake agent-ticks, policy against random agents (chance), and
            during cold (body < -0.3): carrying wood, standing on wood, near a fire, feeding one,
            lighting a torch, starting one.
  caps      What they do and cannot do: each family of actions and each outcome per 1000 awake
            ticks, policy against random, with "never" and "at or below chance" flagged.
  needs     While hydration or satiety is below 0.3 (the decision check's line): thirst by distance to water, hunger
            by the chain food takes (holding something edible / empty-handed on good forage / on poor forage; with
            ecology almost every cell has some food, so distance to food says nothing), and what it chooses in each,
            policy against random. Separates "beside the resource and does not take it" (decision) from "far and does
            not go" (navigation / memory) from "none within reach" (placement).
  deaths    Every cause of death (trained agents): age, asleep, in shelter, at night, child; and for thirst and
            starvation how it happened (thirst: reached water while low and died anyway / knew it but never reached
            it / never been near it / none within 12 cells; starvation: held food and died anyway / stood on good
            forage but never held food / never on good forage), for predation whether it was an ambush, asleep, or
            a predator it saw and did not flee.
  learn     From history.json: is learning still moving (last third against the third before),
            per-output clipping, KL, imitation, watching, world-model errors and trust. Entropy is
            the true mean entropy only for runs made after the logging fix (commit ab25481);
            earlier runs logged about a tenth of it.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifesim import env as E                                           # noqa: E402
from lifesim.policy import make_network                # noqa: E402
import decision_check as D                                             # noqa: E402

COLD = -0.3                  # exposure starts below this body temperature; damage starts at cfg.cold_threshold
PROBE_TEMPS = [0.0, -0.3, -0.5, -0.7, -0.9]
RMAX = 12                    # how far (cells) we look for shelter
SECTIONS = ["survival", "cold", "probe", "reflex", "fire", "caps", "learn", "needs", "deaths"]
LOWNEED = 0.3                # a need is low (the decision check's line) below this
NEARPRED = 3                 # a predator this close (cells) is a threat
DEATH_CAUSES = ("thirst", "starvation", "predation", "violence", "disease", "injury", "exhaustion", "heat", "cold", "fall", "burns", "childbirth", "age")
VARIANTS = ["policy", "reflex_cold", "reflex_dusk"]
FAMILIES = {
    "move": list(range(E.MOVE_N, E.MOVE_NW + 1)), "rest": [E.REST], "sleep": [E.SLEEP],
    "eat": [E.EAT], "drink": [E.DRINK], "grasp": [E.GRASP], "release": [E.RELEASE],
    "strike": list(range(E.STRIKE_N, E.STRIKE_N + 8)), "touch": list(range(E.TOUCH_N, E.TOUCH_N + 8)),
    "select material": list(range(E.SELECT_0, E.PLACE_N)),
    "place": list(range(E.PLACE_N, E.PLACE_N + 8)), "mate": [E.MATE],
    "seek_water": [E.SEEK_WATER], "seek_food": [E.SEEK_FOOD], "flee": [E.FLEE], "seek_shelter": [E.SEEK_SHELTER],
    "delve": [E.DELVE], "relieve": [E.RELIEVE], "write": [E.WRITE],
}
# per-tick StepInfo flags counted per awake agent (name -> info field)
FLAGS = {"near_fire": "near_fire", "fed_fire": "fed_fire", "meat_to_heat": "meat_to_heat",
         "ate_cooked": "ate_cooked", "torch_lit": "torch_lit", "meat_eaten": "meat_eaten",
         "large_prey_hit": "large_prey_hit", "mated": "mated", "spoke": "spoke", "huddling": "huddling",
         "bound a spear": "bound", "slept on a bed": "bedded", "tamed an animal": "tamed",
         "gift taken": None, "relieved": "relieved"}


# ------------------------------------------------------------------ setup
def setup(args):
    run = Path(args.run) if args.run else None
    cfg, h = D.build_cfg(args, run)
    hidden, encoder = (h.get("hidden", args.hidden), h.get("encoder", args.encoder)) if h else (args.hidden, args.encoder)
    wm = bool(h.get("world_model", True)) if h else True
    return cfg, h, hidden, encoder, wm, run


def build_brains(cfg, hidden, encoder, wm, run, key):
    """(net, params, reset_kwargs) for the run's trained agents, or fresh networks when run is None."""
    n = cfg.n_agents
    if run is not None:
        import export_playback as X
        net, params, pop = X.load_population(run, cfg, hidden, encoder)
        kw = (jax.tree_util.tree_map(jnp.asarray, pop.genotype), jnp.asarray(pop.appearance))
        return net, params, kw
    net = make_network(cfg, hidden, encoder, messages=True, world_model=wm)
    dummy = (jnp.zeros((1, 1, E.obs_dim(cfg))), jnp.zeros((1, 1), bool),
             jnp.ones((1, 1, E.N_ACTIONS + E.EXTRA_LOGITS), bool))
    params = jax.vmap(lambda k: net.init(k, jnp.zeros((1, hidden)), dummy))(jax.random.split(key, n))
    return net, params, None


def cover_cells(terrain):
    return (terrain == E.FOREST) | (terrain == E.CAVE)


def make_dist_fn(G, n):
    """Chebyshev distance (0..RMAX, RMAX+1 = none within reach) from each agent to the nearest True cell
    of its own (N, G, G) mask, by repeated one-cell dilation."""
    idx = jnp.arange(n)

    def dilate(m):
        p = jnp.pad(m, ((0, 0), (1, 1), (1, 1)))
        out = m
        for dx in range(3):
            for dy in range(3):
                out = out | p[:, dx:dx + G, dy:dy + G]
        return out

    @jax.jit
    def dist(mask, pos):
        reach = mask
        d = jnp.where(reach[idx, pos[:, 0], pos[:, 1]], 0, RMAX + 1)
        for r in range(1, RMAX + 1):
            reach = dilate(reach)
            d = jnp.where((d > RMAX) & reach[idx, pos[:, 0], pos[:, 1]], r, d)
        return d
    return dist


def field_dist(mask, G):
    """(G, G) Chebyshev distance (0..RMAX, RMAX+1 = none within reach) from every cell to the nearest True cell of one
    shared mask (water, or food now)."""
    reach = mask
    d = jnp.where(mask, 0, RMAX + 1)
    for r in range(1, RMAX + 1):
        p = jnp.pad(reach, 1)
        new = reach
        for dx in range(3):
            for dy in range(3):
                new = new | p[dx:dx + G, dy:dy + G]
        reach = new
        d = jnp.where((d > RMAX) & reach, r, d)
    return d


# ------------------------------------------------------------------ one world
def rollout(cfg, brains, kind, variant, seed, days, probe_every, hidden):
    """kind: "random" or "policy". variant: "policy" | "reflex_cold" | "reflex_dusk" (laid over a policy)."""
    n, G = cfg.n_agents, cfg.grid_size
    key = jax.random.PRNGKey(seed)
    k_env, key = jax.random.split(key)
    if kind == "policy":
        net, params, kw = brains
        state = E.reset(k_env, cfg, *kw) if kw is not None else E.reset(k_env, cfg)
        act = D._policy(net, hidden)
        carry = jnp.zeros((n, 1, hidden))

        @jax.jit
        def probe_fn(params, carry, state, mask):
            """Per probe temperature: critic value and action probabilities, with only body_temp changed."""
            full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (E.EXTRA_LOGITS,), bool)], -1)
            outs = []
            for b in PROBE_TEMPS:
                s2 = state._replace(body_temp=jnp.full((n,), b))
                obs = E.observe(s2, cfg)
                _, logits, value = jax.vmap(lambda p, c, o, m: net.apply(
                    p, c, (o[None, None], jnp.zeros((1, 1), bool), m[None, None])))(params, carry, obs, full)
                pa = jax.nn.softmax(logits[:, 0, 0, :E.N_ACTIONS], axis=-1)
                fam = jnp.stack([pa[:, jnp.asarray(ix)].sum(-1) for ix in
                                 (FAMILIES["seek_shelter"], FAMILIES["rest"] + FAMILIES["sleep"], FAMILIES["move"])], -1)
                outs.append(jnp.concatenate([value[:, 0, 0][:, None], fam], -1))
            return jnp.stack(outs)                                  # (n_temps, N, 4)
    else:
        state = E.reset(k_env, cfg)
        params = None
    step = jax.jit(lambda s, a, m: E.step(s, a, cfg, m))
    observe = jax.jit(lambda s: (E.observe(s, cfg), E.action_mask(s, cfg)))
    dist_fn = make_dist_fn(G, n)
    terr0 = np.asarray(state.terrain)
    cover0 = jnp.asarray(cover_cells(state.terrain))
    ticks = int(days * cfg.day_length)
    daylen = cfg.day_length
    tick_h = 24.0 / daylen

    @jax.jit
    def ctx(s):
        terr_here = s.terrain[s.pos[:, 0], s.pos[:, 1]]
        in_cover = cover_cells(terr_here) | s.below
        known = s.visited & cover0[None]
        d_true = dist_fn(jnp.broadcast_to(cover0[None], (n, G, G)), s.pos)
        d_known = dist_fn(known, s.pos)
        hour = E.hour_of_day(s.t, cfg)
        dl = E.daylight_hours(s.t, cfg)
        night = E.is_night(s.t, cfg)
        dusk = night | (hour >= 12.0 + dl / 2 - 2.0)
        return in_cover, d_true, d_known, night, dusk, E.season_of(s.t, cfg)

    water0 = jnp.asarray(E.drinkable(state.terrain) | (state.terrain == E.MARSH))   # where one can drink (adjacent) or stand in it

    @jax.jit
    def ctx2(s):
        """Needs context: distance to water (true, and to water the agent has stood near), whether it is at water, how good
        the forage is where it stands (share of the full rate; with ecology almost every cell has some, so distance to
        food says nothing), and whether it holds something edible."""
        r_, c_ = s.pos[:, 0], s.pos[:, 1]
        dw = field_dist(water0, G)[r_, c_]
        dwk = dist_fn(s.visited & water0[None], s.pos)
        fg = E.forage_grid(s, cfg)[r_, c_] / cfg.plant_max
        held = s.inv[jnp.arange(n), s.hand]
        holding = E.EDIBLE[s.hand] & (held > 0.05)
        return dw, dwk, fg, E.adjacent_to_water(s.terrain, s.pos, cfg), holding

    def predator_distance(s):
        """Nearest predator that can attack now (alive, awake, not tame), Chebyshev, 99 = none (as decision_check)."""
        P = s.predators
        pp = np.asarray(s.pos) + np.stack([np.zeros(n, np.int32), np.asarray(s.below).astype(np.int32) * E.LAYER_OFF], -1)
        lp = np.asarray(P.pos) + np.stack([np.zeros(len(P.alive), np.int32), np.asarray(P.below).astype(np.int32) * E.LAYER_OFF], -1)
        hunting = np.asarray(P.alive) & (np.asarray(P.rest) <= 0)
        dist = np.abs(pp[:, None, :] - lp[None, :, :]).max(-1).astype(float)
        ok = hunting[None, :] & (np.asarray(P.bond).T < cfg.tame_calm if cfg.taming else True)
        return np.minimum(np.where(ok, dist, 99.0).min(1), 99).astype(np.int8)

    rec = collections.defaultdict(list)
    deaths, probes = [], []
    counts = collections.Counter()
    start_alive = int(np.asarray(state.alive).sum())
    awake_ticks = 0
    for t in range(ticks):
        obs, mask = observe(state)
        key, k = jax.random.split(key)
        in_cover, d_true, d_known, night, dusk, season = ctx(state)
        if kind == "policy":
            carry_prev = carry
            carry, a, msg = act(params, carry, obs, mask, jnp.asarray(t == 0), k)
        else:
            ka, _ = jax.random.split(k)
            a = jax.random.categorical(ka, jnp.where(mask, 0.0, -1e9))
            msg = jnp.zeros((n, E.MSG_LEN), jnp.int32)
        alive = np.asarray(state.alive)
        asleep = np.asarray(state.asleep)
        awake = alive & ~asleep
        bt = np.asarray(state.body_temp)
        if variant != "policy" and kind == "policy":
            trig = awake & ((bt < COLD) if variant == "reflex_cold" else bool(dusk))
            go = trig & ~np.asarray(in_cover) & np.asarray(mask[:, E.SEEK_SHELTER])
            a = jnp.where(jnp.asarray(go), E.SEEK_SHELTER, a)
            counts["reflex_overrides"] += int(go.sum())
        an = np.asarray(a)
        if probe_every and kind == "policy" and variant == "policy" and t % probe_every == 0 and awake.any():
            pr = np.asarray(probe_fn(params, carry_prev, state, mask))
            probes.append({"t": t, "res": pr, "awake": awake, "cover": np.asarray(in_cover), "night": bool(night)})
        inv_wood = np.asarray(state.inv[:, E.WOOD])
        wood_here = np.asarray(state.wood[state.pos[:, 0], state.pos[:, 1]])
        age = np.asarray(state.age)
        birth_age = np.asarray(state.birth_age)
        state_prev_pos = np.asarray(state.pos)
        dw, dwk, fg, atw, holding = ctx2(state)
        rec["hyd"].append(np.asarray(state.hydration).astype(np.float16))
        rec["sat"].append(np.asarray(state.satiety).astype(np.float16))
        rec["dwater"].append(np.asarray(dw).astype(np.int8))
        rec["dwater_known"].append(np.asarray(dwk).astype(np.int8))
        rec["forage"].append(np.asarray(fg).astype(np.float16))
        rec["holding"].append(np.asarray(holding))
        rec["atwater"].append(np.asarray(atw))
        rec["dpred"].append(predator_distance(state))
        state, _, _, info = step(state, a, msg)
        # record
        rec["alive"].append(alive)
        rec["asleep"].append(asleep)
        rec["bt"].append(bt.astype(np.float32))
        rec["act"].append(an.astype(np.int16))
        rec["cover"].append(np.asarray(in_cover))
        rec["dtrue"].append(np.asarray(d_true).astype(np.int8))
        rec["dknown"].append(np.asarray(d_known).astype(np.int8))
        rec["night"].append(bool(night))
        rec["season"].append(int(season))
        for nm, f in (("near_fire", info.near_fire), ("fed_fire", info.fed_fire), ("torch_lit", info.torch_lit),
                      ("huddling", info.huddling)):
            rec[nm].append(np.asarray(f))
        rec["ignited"].append(np.asarray(info.ignited) > 0)
        rec["wood_inv"].append(inv_wood > 0.5)
        rec["wood_here"].append(wood_here > 0.5)
        o = np.asarray(info.outcome)
        for c in np.unique(o[awake]):
            counts[f"outcome:{E.OUTCOME_NAMES[int(c)]}"] += int((o[awake] == c).sum())
        for nm, fld in FLAGS.items():
            if fld is None:
                continue
            v = np.asarray(getattr(info, fld))
            v = (v >= 0) if v.dtype.kind in "iu" else v.astype(bool)          # tamed holds a species or -1
            counts[f"flag:{nm}"] += int((v & awake).sum())
        counts["flag:gift taken"] += int(np.asarray(info.gifts).sum())
        counts["flag:born"] += int(np.asarray(info.born).sum())
        counts["flag:conceived"] += int(np.asarray(info.conceived).sum())
        kills = np.asarray(info.kills)
        counts["kills:small prey"] += int(kills[:, 0].sum())
        counts["kills:large prey"] += int(kills[:, 1].sum())
        counts["kills:predator"] += int(kills[:, 2].sum())
        awake_ticks += int(awake.sum())
        for i in np.nonzero(np.asarray(info.died))[0]:
            deaths.append({"t": t, "agent": int(i), "cause": E.CAUSE_NAMES[int(np.asarray(info.death_cause)[i])],
                           "age": int(age[i]), "birth_age": int(birth_age[i]), "season": int(season),
                           "night": bool(night), "in_cover": bool(np.asarray(in_cover)[i]),
                           "asleep": bool(asleep[i])})
    stack = {k: np.stack(v) for k, v in rec.items() if k not in ("night", "season")}
    stack["night"] = np.array(rec["night"])
    stack["season"] = np.array(rec["season"])
    return {"kind": kind, "variant": variant, "seed": seed, "ticks": ticks, "start_alive": start_alive,
            "awake_ticks": awake_ticks, "alive_ticks": int(stack["alive"].sum()), "deaths": deaths, "counts": dict(counts),
            "rec": stack, "probes": probes, "alive_end": int(np.asarray(state.alive).sum())}


# ------------------------------------------------------------------ analysis
def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x, n):
    return "n/a" if not n else f"{x / n:.0%}"


def child_of(cfg, d):
    return (d["age"] + d["birth_age"]) < cfg.adult_years * cfg.ticks_per_bio_year


def sec_survival(cfg, R):
    yr = cfg.year_length
    kinds = [("policy", "random"), ("policy", "policy"), ("reflex_cold", "policy"), ("reflex_dusk", "policy")]
    rows, data = [], {}
    for variant, kind in kinds:
        rs = [r for r in R if r["variant"] == variant and r["kind"] == kind]
        if not rs:
            continue
        variant = "random" if kind == "random" else variant
        ay = sum(r["alive_ticks"] for r in rs) / yr
        causes = collections.Counter(d["cause"] for r in rs for d in r["deaths"])
        cold = [d for r in rs for d in r["deaths"] if d["cause"] == "cold"]
        cold_open = sum(1 for d in cold if not d["in_cover"])
        total = sum(causes.values())
        data[variant] = {"agent_years": ay, "deaths": dict(causes), "cold_open": cold_open, "cold_covered": len(cold) - cold_open}
        rows.append([variant, f"{ay:.1f}", total, f"{total / max(ay, 1e-9):.2f}", causes.get("cold", 0),
                     f"{causes.get('cold', 0) / max(ay, 1e-9):.2f}", f"{cold_open} / {len(cold) - cold_open}",
                     sum(r["alive_end"] for r in rs), sum(r["start_alive"] for r in rs),
                     ", ".join(f"{k} {v}" for k, v in causes.most_common(5) if k != "cold")])
    print("\n## Survival: deaths per agent-year, by agent kind\n")
    print(md(rows, ["agents", "agent-years", "deaths", "per agent-year", "cold deaths", "cold per agent-year",
                    "cold: open / sheltered", "alive at end", "at start", "other leading causes"]))
    print("\nreflex_cold and reflex_dusk are the trained agents with the shelter rule laid over them (see the docstring). "
          "If their cold deaths fall far below policy's, the world can be survived this way and the gap is the agents' decision.")
    return data


def sec_cold(cfg, R):
    out = []
    for r in R:
        if r["variant"] != "policy" or r["kind"] != "policy":
            continue
        rec, dr = r["rec"], r["deaths"]
        for d in dr:
            if d["cause"] != "cold":
                continue
            i, t = d["agent"], d["t"]
            bt = rec["bt"][:t + 1, i]
            alive = rec["alive"][:t + 1, i]
            # exposure: the last stretch ending at death with body temp below COLD
            e = t
            while e > 0 and alive[e - 1] and bt[e - 1] < COLD:
                e -= 1
            sl = slice(e, t + 1)
            cov = rec["cover"][sl, i]
            hours = (t - e + 1) * 24.0 / cfg.day_length
            acts = collections.Counter(int(a) for a in rec["act"][sl, i][~rec["asleep"][sl, i]])
            out.append({"seed": r["seed"], "agent": i, "t": t, "child": bool(child_of(cfg, d)), "season": d["season"],
                        "exposure_hours": hours, "frac_in_cover": float(cov.mean()), "died_in_cover": d["in_cover"],
                        "died_asleep": d["asleep"], "night_share": float(rec["night"][sl].mean()),
                        "asleep_share": float(rec["asleep"][sl, i].mean()),
                        "d_true": int(rec["dtrue"][e, i]), "d_known": int(rec["dknown"][e, i]),
                        "huddled_share": float(rec["huddling"][sl, i].mean()),
                        "top_acts": [(E.ACTION_NAMES[a], c) for a, c in acts.most_common(4)]})
    print("\n## Cold forensics: how each cold death happened (trained agents)\n")
    if not out:
        print("No cold deaths in these rollouts. Raise --days or --seeds, or the run's world has no cold deaths.")
        return {"n": 0}

    def bin_of(x):
        if x["frac_in_cover"] >= 0.5:
            return "froze inside shelter"
        if x["d_known"] <= RMAX:
            return f"knew shelter within {RMAX} cells, stayed out"
        if x["d_true"] <= RMAX:
            return f"shelter within {RMAX} cells they had never visited"
        return f"no shelter within {RMAX} cells"
    bins = collections.Counter(bin_of(x) for x in out)
    n = len(out)
    print(md([[k, v, pct(v, n)] for k, v in bins.most_common()], ["how", "deaths", "share"]))
    ad = [x for x in out if not x["child"]]
    ch = [x for x in out if x["child"]]
    med = lambda v: "n/a" if not v else f"{float(np.median(v)):.1f}"
    print(f"\n{n} cold deaths ({len(ad)} adults, {len(ch)} children) in {sum(1 for r in R if r['variant']=='policy' and r['kind']=='policy')} rollouts. "
          f"Winter: {sum(1 for x in out if x['season'] == 3)}; died asleep: {sum(1 for x in out if x['died_asleep'])}.")
    print(md([
        ["exposure before death, hours (median)", med([x["exposure_hours"] for x in out])],
        ["share of the exposure spent in shelter (median)", med([x["frac_in_cover"] for x in out])],
        ["share of the exposure at night (median)", med([x["night_share"] for x in out])],
        ["share of the exposure asleep (median)", med([x["asleep_share"] for x in out])],
        ["share of the exposure huddled with a neighbour (median)", med([x["huddled_share"] for x in out])],
        ["distance to any shelter when exposure began, cells (median)", med([x["d_true"] for x in out])],
        ["distance to shelter they had visited, cells (median)", med([x["d_known"] for x in out])]], ["", ""]))
    top = collections.Counter()
    for x in out:
        for name, c in x["top_acts"]:
            top[name] += c
    print("\nWhat they were doing while exposed (awake ticks):",
          ", ".join(f"{k} {v}" for k, v in top.most_common(8)))
    print("\nReading it: 'froze inside shelter' says shelter alone is not enough (cold snap, snow, no huddle, poor insulation, a child). "
          "'knew shelter, stayed out' is a decision failure. 'never visited' is exploration or memory. 'no shelter within reach' is placement or a long walk.")
    return {"n": n, "bins": dict(bins), "rows": out}


def sec_probe(cfg, R):
    print("\n## Probe: does the brain perceive cold, and act on it? (body temperature set by hand, nothing else changed)\n")
    P = [p for r in R if r["variant"] == "policy" and r["kind"] == "policy" for p in r["probes"]]
    if not P:
        print("No probes taken.")
        return {}
    groups = {"outside shelter, night": lambda p: ~p["cover"] if p["night"] else np.zeros_like(p["cover"]),
              "outside shelter, day": lambda p: np.zeros_like(p["cover"]) if p["night"] else ~p["cover"],
              "inside shelter, night": lambda p: p["cover"] if p["night"] else np.zeros_like(p["cover"]),
              "inside shelter, day": lambda p: np.zeros_like(p["cover"]) if p["night"] else p["cover"]}
    result = {}
    for gname, sel in groups.items():
        rows, cnt = [], 0
        acc = np.zeros((len(PROBE_TEMPS), 4))
        for p in P:
            m = p["awake"] & sel(p)
            if m.any():
                acc += p["res"][:, m, :].sum(1)
                cnt += int(m.sum())
        if not cnt:
            continue
        acc /= cnt
        for j, b in enumerate(PROBE_TEMPS):
            rows.append([f"{b:+.1f}", f"{acc[j, 0]:+.3f}", f"{acc[j, 1]:.1%}", f"{acc[j, 2]:.1%}", f"{acc[j, 3]:.1%}"])
        print(f"\n**{gname}** ({cnt} agent-moments)\n")
        print(md(rows, ["body temperature", "critic value", "P(seek_shelter)", "P(rest or sleep)", "P(move)"]))
        result[gname] = {"n": cnt, "table": acc.tolist()}
    print("\nValue should fall as temperature drops (damage starts at -0.7). Outside shelter at night, P(seek_shelter) should rise "
          "as it drops. Flat value: the critic never learned cold is costly. Value falls, probabilities flat: it sees the danger and has no habit of the remedy.")
    return result


def sec_fire(cfg, R):
    print("\n## The fire ladder, per 1000 awake agent-ticks\n")

    def rates(rs, cold_only):
        tot = 0
        c = collections.Counter()
        for r in rs:
            rec = r["rec"]
            awake = rec["alive"] & ~rec["asleep"]
            m = awake & (rec["bt"] < COLD) if cold_only else awake
            tot += int(m.sum())
            for k in ("wood_inv", "wood_here", "near_fire", "fed_fire", "torch_lit", "ignited"):
                c[k] += int((rec[k] & m).sum())
        return tot, c
    names = [("wood_inv", "carrying wood"), ("wood_here", "standing on wood"), ("near_fire", "near a fire"),
             ("fed_fire", "fed a fire"), ("torch_lit", "picked up burning wood"), ("ignited", "started a fire")]
    pol = [r for r in R if r["variant"] == "policy" and r["kind"] == "policy"]
    rnd = [r for r in R if r["kind"] == "random"]
    res = {}
    rows = []
    cols = [("trained", pol, False), ("random", rnd, False), ("trained, cold", pol, True), ("random, cold", rnd, True)]
    calc = {nm: rates(rs, co) for nm, rs, co in cols if rs}
    for key, label in names:
        row = [label]
        for nm, _, _ in cols:
            if nm in calc:
                tot, c = calc[nm]
                row.append(f"{1000 * c[key] / tot:.1f}" if tot else "n/a")
            else:
                row.append("n/a")
        rows.append(row)
    rows.append(["ticks behind the rate"] + [str(calc[nm][0]) if nm in calc else "n/a" for nm, _, _ in cols])
    print(md(rows, ["per 1000 awake ticks"] + [c[0] for c in cols]))
    res = {nm: {"ticks": calc[nm][0], **dict(calc[nm][1])} for nm in calc}
    print("\nLadder: wood on the ground -> picked up -> a fire exists (lightning, or one they start) -> near it -> feed it. "
          "A rung at or below random's rate was never learned. 'Cold' rows count only ticks with body temperature below -0.3: "
          "do they reach for fire when it matters?")
    return res


def sec_caps(cfg, R):
    print("\n## What they do, against chance (per 1000 awake agent-ticks)\n")
    pol = [r for r in R if r["variant"] == "policy" and r["kind"] == "policy"]
    rnd = [r for r in R if r["kind"] == "random"]

    def fam_rates(rs):
        tot = sum(r["awake_ticks"] for r in rs)
        c = collections.Counter()
        for r in rs:
            rec = r["rec"]
            awake = rec["alive"] & ~rec["asleep"]
            a = rec["act"][awake]
            for f, ix in FAMILIES.items():
                c[f] += int(np.isin(a, ix).sum())
        return tot, c
    tp, cp = fam_rates(pol)
    tr, cr = fam_rates(rnd) if rnd else (0, collections.Counter())
    rows = []
    for f in FAMILIES:
        a = 1000 * cp[f] / tp if tp else None
        b = 1000 * cr[f] / tr if tr else None
        flag = "NEVER" if tp and cp[f] == 0 else ("<= chance" if (a is not None and b is not None and a <= b) else "")
        rows.append([f, "n/a" if a is None else f"{a:.1f}", "n/a" if b is None else f"{b:.1f}", flag])
    print(md(rows, ["action family", "trained", "random", "flag"]))

    def count_rates(rs):
        tot = sum(r["awake_ticks"] for r in rs)
        c = collections.Counter()
        for r in rs:
            c.update(r["counts"])
        return tot, c
    tp, cp = count_rates(pol)
    tr, cr = count_rates(rnd) if rnd else (0, collections.Counter())
    keys = sorted(set(cp) | set(cr))
    rows = []
    for k in keys:
        if k in ("reflex_overrides",):
            continue
        a = 1000 * cp[k] / tp if tp else None
        b = 1000 * cr[k] / tr if tr else None
        flag = "NEVER" if tp and cp[k] == 0 else ("<= chance" if (a is not None and b is not None and a <= b and b > 0) else "")
        rows.append([k, cp[k], "n/a" if a is None else f"{a:.2f}", "n/a" if b is None else f"{b:.2f}", flag])
    print("\n" + md(rows, ["outcome or event", "trained total", "trained per 1000", "random per 1000", "flag"]))
    return {"families": {f: cp.get(f, 0) for f in FAMILIES}}


def sec_learn(run):
    print("\n## Is learning still moving? (from history.json)\n")
    if run is None or not (run / "history.json").exists():
        print("No history.json: give --run.")
        return {}
    H = json.load(open(run / "history.json"))["history"]
    if len(H) < 6:
        print("Fewer than 6 log rows.")
        return {}

    def get(row, path):
        v = row
        for p in path.split("."):
            if not isinstance(v, dict) or p not in v:
                return None
            v = v[p]
        return v
    metrics = [("survival_score_per_world_day", "survival score per world-day (higher is better)", +1),
               ("mean_age_at_death", "mean age at death, ticks (higher is better)", +1),
               ("entropy", "policy entropy (see note on logging)", 0),
               ("entropy_coef", "entropy bonus actually applied", 0),
               ("learned_share", "share of steps learned from (after sleep)", 0),
               ("watched_fraction", "watching others (share)", 0),
               ("imitation_loss", "imitation loss (lower: closer to who it watched)", -1),
               ("per_output.action.clip_fraction", "action: share of updates clipped", 0),
               ("per_output.action.approx_kl", "action: KL moved per update", 0),
               ("winter_huddle_share", "huddling on winter nights", +1),
               ("world_model.reward_error", "world model: reward error (lower is better)", -1),
               ("world_model.own_state_error", "world model: own-state error (lower is better)", -1),
               ("world_model.trust_in_ideas", "trust in its ideas (0 = ignored)", 0),
               ("sleep_memory.retention_kl", "retention KL (drift from stored past behaviour)", 0),
               ("hunting_gini", "hunting inequality", 0)]
    thirds = np.array_split(np.arange(len(H)), 3)
    rows = []
    out = {}
    for path, label, good in metrics:
        vals = [get(r, path) for r in H]
        if all(v is None for v in vals):
            continue
        t = [np.mean([vals[i] for i in ix if vals[i] is not None]) if any(vals[i] is not None for i in ix) else None for ix in thirds]
        if None in t:
            continue
        d = (t[2] - t[1]) / (abs(t[1]) + 1e-9)
        verdict = ""
        if good:
            verdict = "still improving" if good * d > 0.05 else ("worse" if good * d < -0.05 else "flat")
        rows.append([label, f"{t[0]:.4g}", f"{t[1]:.4g}", f"{t[2]:.4g}", f"{d:+.0%}", verdict])
        out[path] = t
    print(md(rows, ["metric", "first third", "middle third", "last third", "last vs middle", "verdict"]))
    last = H[-1]
    print(f"\nTotal: {last.get('agent_steps', 0) / 1e6:.0f}M agent-steps. Deaths in the last log row by cause: {last.get('deaths')}. "
          f"Cold deaths (open/sheltered, adult/child): {last.get('cold_deaths')}.")
    print("Note: entropy and entropy_coef in runs made before commit ab25481 were logged about 10x too low; compare them only within such runs.")
    return out


def _pooled(R, variant, kind, keys):
    """Per-tick arrays of the given rec keys, pooled over rollouts of one kind: dict of flattened arrays over (tick, agent)."""
    rs = [r for r in R if r["variant"] == variant and r["kind"] == kind and "dwater" in r["rec"]]
    if not rs:
        return None
    out = {k: np.concatenate([r["rec"][k].reshape(-1) if r["rec"][k].ndim == 2 else
                              np.repeat(r["rec"][k], r["rec"]["alive"].shape[1]) for r in rs]) for k in keys}
    return out


def sec_needs(cfg, R):
    """Where are they, and what do they choose, while a need is low? Policy against random agents. Thirst is split by
    distance to water; hunger by the chain food takes (hold something edible -> eat; on good forage -> grasp it)."""
    keys = ["alive", "asleep", "act", "hyd", "sat", "dwater", "forage", "holding", "atwater"]
    pol, rnd = _pooled(R, "policy", "policy", keys), _pooled(R, "policy", "random", keys)
    print("\n## While a need is low (below %.1f): where they are and what they choose\n" % LOWNEED)
    if pol is None:
        print("No rollouts with needs context (rerun without --reuse).")
        return {}
    FOOD_OK = 0.3                      # forage at least this share of the full rate counts as good ground
    spec = {
        "thirst": dict(level="hyd", title="hydration", cols=[("drink", [E.DRINK]), ("seek_water", [E.SEEK_WATER]),
                                                           ("a move", FAMILIES["move"]), ("rest/sleep", FAMILIES["rest"] + FAMILIES["sleep"])],
                       buckets=lambda P: [
                           ("at water (can drink)", P["atwater"]),
                           ("1 to 5 cells from water", ~P["atwater"] & (P["dwater"] >= 1) & (P["dwater"] <= 5)),
                           (f"6 to {RMAX} cells", (P["dwater"] >= 6) & (P["dwater"] <= RMAX)),
                           (f"no water within {RMAX}", P["dwater"] > RMAX)]),
        "hunger": dict(level="sat", title="satiety", cols=[("eat", [E.EAT]), ("grasp", [E.GRASP]), ("seek_food", [E.SEEK_FOOD]),
                                                         ("a move", FAMILIES["move"]), ("rest/sleep", FAMILIES["rest"] + FAMILIES["sleep"])],
                       buckets=lambda P: [
                           ("holding something edible", P["holding"]),
                           (f"empty-handed, on good forage (>= {FOOD_OK:.0%} of full)", ~P["holding"] & (P["forage"].astype(np.float32) >= FOOD_OK)),
                           (f"empty-handed, on poor forage (< {FOOD_OK:.0%})", ~P["holding"] & (P["forage"].astype(np.float32) < FOOD_OK))]),
    }
    result = {}
    for need, sp in spec.items():
        rows, data = [], []
        for who, P in (("policy", pol), ("random", rnd)):
            if P is None:
                continue
            m = P["alive"] & ~P["asleep"] & (P[sp["level"]].astype(np.float32) < LOWNEED)
            tot = int(m.sum())
            for name, bm in sp["buckets"](P):
                b = m & bm
                nb = int(b.sum())
                a = P["act"][b].astype(int)
                rates = [(float(np.isin(a, ix).mean()) if nb else None) for _, ix in sp["cols"]]
                rows.append([who, name, nb, pct(nb, tot)] + [pct(int(np.isin(a, ix).sum()), nb) for _, ix in sp["cols"]])
                data.append({"who": who, "bucket": name, "ticks": nb, "share": nb / tot if tot else None,
                             "rates": dict(zip([c for c, _ in sp["cols"]], rates))})
        print(f"### {need}: awake ticks with {sp['title']} < {LOWNEED}\n")
        print(md(rows, ["agents", "where", "ticks", "share of low-need ticks"] + ["chose " + c for c, _ in sp["cols"]]))
        print()
        result[need] = data
    print("Reading it: ticks beside water (or holding food) where they do not drink (eat) are a decision failure; ticks far from "
          "water, or empty-handed on good forage with no grasp, are navigation or the gather step; none within reach is placement. "
          "Distances are on the surface map (agents underground are measured from the cell above).")
    return result


def sec_deaths(cfg, R):
    """Forensics for every cause of death (policy agents): who, when, in what state, and for thirst, starvation and
    predation how it happened."""
    pol = [r for r in R if r["variant"] == "policy" and r["kind"] == "policy"]
    ds = [(r, d) for r in pol for d in r["deaths"]]
    print("\n## Deaths by cause, all of them (trained agents)\n")
    if not ds:
        print("No deaths in these rollouts.")
        return {"n": 0}
    yr = cfg.year_length
    ay = sum(r["alive_ticks"] for r in pol) / yr
    rows, med = [], (lambda v: "n/a" if not len(v) else f"{float(np.median(v)):.1f}")
    for cause, c in collections.Counter(d["cause"] for _, d in ds).most_common():
        dd = [d for _, d in ds if d["cause"] == cause]
        rows.append([cause, c, f"{c / max(ay, 1e-9):.2f}", med([d["age"] / cfg.day_length for d in dd]),
                     pct(sum(d["asleep"] for d in dd), c), pct(sum(d["in_cover"] for d in dd), c),
                     pct(sum(d["night"] for d in dd), c), pct(sum(child_of(cfg, d) for d in dd), c)])
    print(md(rows, ["cause", "deaths", "per agent-year", "median days lived in the sim", "died asleep", "in shelter", "at night", "children"]))
    out = {"n": len(ds), "agent_years": ay}

    def stretch(rec, i, t, ok):
        """First tick e of the unbroken stretch up to t (inclusive) in which agent i was alive with ok[:, i] true."""
        e = t
        while e > 0 and rec["alive"][e - 1, i] and ok[e - 1, i]:
            e -= 1
        return e

    for cause, need, dist_key, known_key, acts, label in (
            ("thirst", "hyd", "dwater", "dwater_known", [E.DRINK], "water"),
            ("starvation", "sat", None, None, [E.EAT], "food")):
        rows_, bins = [], collections.Counter()
        for r, d in ds:
            if d["cause"] != cause or "dwater" not in r["rec"]:
                continue
            rec, i, t = r["rec"], d["agent"], d["t"]
            low = rec[need].astype(np.float32) < LOWNEED
            e = stretch(rec, i, t, low)
            sl = slice(e, t + 1)
            awake = ~rec["asleep"][sl, i]
            hours = (t - e + 1) * 24.0 / cfg.day_length
            acts_c = collections.Counter(int(a) for a in rec["act"][sl, i][awake])
            if cause == "thirst":
                near = rec["atwater"][sl, i]
                reached = bool(near.any())
                at_ticks = int((near & awake).sum())
                took = int(np.isin(rec["act"][sl, i][near & awake], acts).sum())
                d0 = int(rec[dist_key][e, i])
                dk = int(rec[known_key][e, i])
                if reached:
                    bins["reached water while low, died anyway"] += 1
                elif d0 > RMAX:
                    bins[f"no water within {RMAX} cells when it began"] += 1
                elif dk <= RMAX:
                    bins[f"knew water within {RMAX} cells, never reached it"] += 1
                else:
                    bins[f"water within {RMAX} cells, never reached it (never been near it)"] += 1
                rows_.append({"hours": hours, "reached": reached, "at_ticks": at_ticks, "took": took, "d_at_onset": d0,
                              "asleep_share": float(1 - awake.mean()), "top": [(E.ACTION_NAMES[a], c) for a, c in acts_c.most_common(3)]})
            else:
                hold = rec["holding"][sl, i]
                good = rec["forage"][sl, i].astype(np.float32) >= 0.3
                at_ticks = int((hold & awake).sum())                      # ticks it held food while starving
                took = int(np.isin(rec["act"][sl, i][hold & awake], acts).sum())
                if hold.any():
                    bins["held food while starving, died anyway"] += 1
                elif good.any():
                    bins["stood on good forage while starving, never held food"] += 1
                else:
                    bins["never on good forage while starving"] += 1
                rows_.append({"hours": hours, "reached": bool(hold.any()), "at_ticks": at_ticks, "took": took,
                              "d_at_onset": None, "asleep_share": float(1 - awake.mean()),
                              "top": [(E.ACTION_NAMES[a], c) for a, c in acts_c.most_common(3)]})
        if not rows_:
            continue
        n = len(rows_)
        print(f"\n### {cause}: {n} deaths\n")
        print(md([[k, v, pct(v, n)] for k, v in bins.most_common()], ["how", "deaths", "share"]))
        print()
        trs = [["hours from the need first low to death (median)", med([x["hours"] for x in rows_])]]
        if cause == "thirst":
            trs.append(["distance to water when it began, cells (median)", med([x["d_at_onset"] for x in rows_])])
        trs += [[f"awake ticks {'next to water' if cause == 'thirst' else 'holding food'} while low (median)", med([x["at_ticks"] for x in rows_])],
                [f"of those, ticks choosing {'/'.join(E.ACTION_NAMES[a] for a in acts)} (median)", med([x["took"] for x in rows_])],
                ["share of the stretch asleep (median)", med([x["asleep_share"] for x in rows_])]]
        print(md(trs, ["", ""]))
        top = collections.Counter()
        for x in rows_:
            for name, c in x["top"]:
                top[name] += c
        print("\nWhat they did while low (awake ticks): " + ", ".join(f"{k} {v}" for k, v in top.most_common(6)))
        out[cause] = {"n": n, "bins": dict(bins), "rows": rows_}

    pr = []
    for r, d in ds:
        if d["cause"] not in ("predation", "violence") or "dpred" not in r["rec"]:
            continue
        rec, i, t = r["rec"], d["agent"], d["t"]
        close = rec["dpred"][:, i] <= NEARPRED if d["cause"] == "predation" else np.zeros(len(rec["dpred"]), bool)
        e = t
        while e > 0 and rec["alive"][e - 1, i] and close[e - 1]:
            e -= 1
        warn = t - e                                      # ticks it had a predator within reach before the end
        sl = slice(e, t)
        a = rec["act"][sl, i].astype(int)
        pr.append({"cause": d["cause"], "warn": warn, "asleep": d["asleep"], "in_cover": d["in_cover"],
                   "fled": int(np.isin(a, [E.FLEE]).sum()), "moved": int(np.isin(a, FAMILIES["move"]).sum()),
                   "stayed": int(np.isin(a, FAMILIES["rest"] + FAMILIES["sleep"]).sum())})
    pp = [x for x in pr if x["cause"] == "predation"]
    if pp:
        def pbin(x):
            if x["asleep"]:
                return "died asleep"
            if x["warn"] <= 2:
                return "no predator within 3 cells until the last 2 ticks (ambush)"
            return "predator within 3 cells for 3+ ticks, never chose FLEE" if x["fled"] == 0 else "chose FLEE and was caught"
        print(f"\n### predation: {len(pp)} deaths\n")
        bins = collections.Counter(pbin(x) for x in pp)
        print(md([[k, v, pct(v, len(pp))] for k, v in bins.most_common()], ["how", "deaths", "share"]))
        print()
        print(md([["ticks with a predator within 3 cells before the end (median)", med([x["warn"] for x in pp])],
                  ["ticks choosing FLEE in that stretch (median)", med([x["fled"] for x in pp])],
                  ["ticks choosing a move (median)", med([x["moved"] for x in pp])],
                  ["ticks choosing rest/sleep (median)", med([x["stayed"] for x in pp])],
                  ["in shelter at death", pct(sum(x["in_cover"] for x in pp), len(pp))]], ["", ""]))
        out["predation"] = {"n": len(pp), "bins": dict(bins), "rows": pp}
    print("\nReading it: for thirst and starvation, 'reached it while low, died anyway' means the resource was there and the "
          "need was not met (check the drink/eat choices beside it); 'knew it, never reached' is a decision or a walk too long "
          "for the need; 'never been near it' is exploration; 'none within reach' is placement. For predation, 'ambush' is not "
          "avoidable by choice; 'never chose FLEE' is a decision failure.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--untrained", action="store_true", help="no run: fresh networks stand in for the trained agents (smoke test)")
    ap.add_argument("--sections", type=str, default=",".join(SECTIONS))
    ap.add_argument("--days", type=float, default=80.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--probe-every", type=int, default=24, help="ticks between probes (0 = none)")
    ap.add_argument("--agents", type=int, default=16)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="mlp")
    ap.add_argument("--tick-minutes", type=float, default=30.0)
    ap.add_argument("--underground", action="store_true")
    ap.add_argument("--out", type=str, default="diagnose_out")
    ap.add_argument("--reuse", action="store_true", help="read rollouts.pkl from --out instead of running")
    args = ap.parse_args()
    secs = [s for s in args.sections.split(",") if s]
    bad = [s for s in secs if s not in SECTIONS]
    if bad:
        sys.exit(f"unknown sections {bad}; choose from {SECTIONS}")
    cfg, h, hidden, encoder, wm, run = setup(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pk = out / "rollouts.pkl"
    need_roll = any(s in secs for s in ["survival", "cold", "probe", "reflex", "fire", "caps", "needs", "deaths"])
    R = []
    if need_roll and args.reuse and pk.exists():
        R = pickle.load(open(pk, "rb"))
        print(f"read {len(R)} rollouts from {pk}")
    elif need_roll:
        if run is None and not args.untrained:
            sys.exit("give --run (a trained run) or --untrained")
        brains = build_brains(cfg, hidden, encoder, wm, run, jax.random.PRNGKey(0))
        variants = ["policy"] + (["reflex_cold", "reflex_dusk"] if ("reflex" in secs or "survival" in secs) else [])
        for seed in range(args.seeds):
            for variant in variants:
                r = rollout(cfg, brains, "policy", variant, 100 + seed, args.days,
                            args.probe_every if "probe" in secs else 0, hidden)
                R.append(r)
                print(f"seed {seed} {variant}: alive at end {r['alive_end']}/{r['start_alive']}, deaths {len(r['deaths'])}, "
                      f"cold {sum(1 for d in r['deaths'] if d['cause'] == 'cold')}", flush=True)
            if any(s in secs for s in ["survival", "fire", "caps", "needs"]):
                r = rollout(cfg, brains, "random", "policy", 100 + seed, args.days, 0, hidden)
                R.append(r)
                print(f"seed {seed} random: alive at end {r['alive_end']}/{r['start_alive']}, deaths {len(r['deaths'])}", flush=True)
        pickle.dump(R, open(pk, "wb"))
    print(f"\n# Diagnosis: {args.days:g} days x {args.seeds} seeds, {cfg.n_agents} agents, grid {cfg.grid_size}, "
          f"{cfg.tick_minutes:g}-minute ticks, year = {cfg.days_per_year} days, cold damage below {cfg.cold_threshold}")
    result = {}
    if "survival" in secs or "reflex" in secs:
        result["survival"] = sec_survival(cfg, R)
    if "cold" in secs:
        r = sec_cold(cfg, R)
        result["cold"] = {k: v for k, v in r.items() if k != "rows"}
        result["cold_rows"] = r.get("rows", [])
    if "probe" in secs:
        result["probe"] = sec_probe(cfg, R)
    if "fire" in secs:
        result["fire"] = sec_fire(cfg, R)
    if "caps" in secs:
        result["caps"] = sec_caps(cfg, R)
    if "needs" in secs:
        result["needs"] = sec_needs(cfg, R)
    if "deaths" in secs:
        result["deaths"] = sec_deaths(cfg, R)
    if "learn" in secs:
        result["learn"] = sec_learn(run)
    (out / "diagnose.json").write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\nwrote", out / "diagnose.json")


if __name__ == "__main__":
    main()
