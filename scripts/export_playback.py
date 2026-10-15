"""Record a playback of one world and write a self-contained pixel-art player.

Runs a population of trained agents (from train_phase2 outputs) or random
agents in a single world, records every tick (agents, animals, fire, food,
weather, messages, timestamped events) and embeds it in
scripts/playback_template.html.

    python scripts/export_playback.py --run runs/moral_run --days 2 --out runs/moral_run/playback.html
    python scripts/export_playback.py --random --agents 6 --grid 32 --days 2
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lifesim import env as E
from lifesim.analysis import language_metrics
from lifesim.config import EnvConfig
from lifesim.names import spoken_form
from lifesim.policy import make_network
from lifesim.recorder import _stamp_events, extract_events

TEMPLATE = Path(__file__).with_name("playback_template.html")
BODY_MODEL = Path(__file__).with_name("body_model.js")


def page_html(data) -> str:
    """The playback page with its data and the body model inlined (one self-contained file)."""
    return (TEMPLATE.read_text().replace("/*__BODY_MODEL__*/", BODY_MODEL.read_text())
            .replace("/*__PLAYBACK_DATA__*/null", json.dumps(data, separators=(",", ":"))))
from lifesim import conditions as _cond  # noqa: E402
COND_NAMES = _cond.NAMES
from lifesim import minerals as _min  # noqa: E402
from lifesim import techtree as _tech  # noqa: E402
from lifesim import fauna as _fa  # noqa: E402
from lifesim.policy import DECODE_NAMES  # noqa: E402

# ---------------------------------------------------------------- the decision behind each action
# What a network "thinks" is not readable directly; these are the parts that are real: the chance
# it gave each option, its expectation of how its future will go (the value estimate), and which
# parts of what it perceived pushed the chosen action most (gradient x input, summed by group).
MIND_GROUPS = ["the land (terrain, slopes, edges)", "food in sight", "others in sight", "animals",
               "predators", "fire and materials", "the dead", "its own needs (food, water, energy, sleep)",
               "its own condition (health, injury, stress, warmth)", "what it carries", "time and weather",
               "voices it heard", "its memory of places"]


def obs_groups(cfg):
    """(obs_dim,) group index of every observation number, following observe()'s layout."""
    v, C = cfg.view_size, E.N_OBS_CHANNELS
    chan = np.zeros(C, int)
    chan[:E.N_TERRAIN] = 0
    chan[E.N_TERRAIN + 0] = 1; chan[E.N_TERRAIN + 1] = 1; chan[E.N_TERRAIN + 7] = 1          # plants, meat, cooked
    chan[E.N_TERRAIN + 2] = 2; chan[E.CH_SNOW + 1:] = 2                                        # agents, social features
    chan[E.CH_RAIN] = 10; chan[E.CH_SNOW] = 10                                                 # rain, snow: weather
    chan[E.N_TERRAIN + 3] = 3; chan[E.N_TERRAIN + 4] = 3                                      # small, large animals
    chan[E.N_TERRAIN + 5] = 4                                                                  # predators
    chan[E.N_TERRAIN + 6] = 0; chan[E.CH_CLIFF] = 0; chan[E.CH_HEIGHT] = 0                     # edge, cliff, height
    chan[E.N_TERRAIN + 8:E.N_TERRAIN + 12] = 5                                                 # wood, stones, fire, heat
    chan[E.CH_BODIES] = 6
    chan[E.CH_MINERALS:E.CH_MINERALS + E.N_MINERAL_CHANNELS] = 5                              # minerals (how they look)
    chan[E.CH_FAUNA:E.CH_FAUNA + E.N_FAUNA_CHANNELS] = 3                                       # animals' looks, bugs
    groups = [np.tile(chan, v * v)]
    own = np.full(E.OWN_FEATURES, 8)                                                           # body, underground, conditions
    own[[0, 1, 2, 19, 20]] = 7                                                                 # satiety, water, energy, sleepiness, asleep
    groups += [own, np.full(2 * E.N_MAT + 1, 9), np.full(5 + E.N_WEATHER, 10),
               np.full(E.HEARD_FEATURES, 11), np.full(E.map_features(cfg), 12)]
    g = np.concatenate(groups)
    assert len(g) == E.obs_dim(cfg), (len(g), E.obs_dim(cfg))
    return g


def make_thinker(net, cfg):
    """A jitted step that acts like the evaluation policy and also returns, per agent: action
    probabilities, the value estimate, attribution of the chosen action to each MIND_GROUP and,
    with a world model, its ideas (the actions it imagined, how each looked, and the best one
    played out: what it would do after, and what it expects of its own body)."""
    A, M = E.N_ACTIONS, E.MSG_LEN * E.VOCAB
    seg = jnp.asarray(obs_groups(cfg))
    n_groups = len(MIND_GROUPS)

    @jax.jit
    def think(params, carry, obs, mask, reset, k):
        full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (M,), bool)], -1)
        apply = lambda p, c, o, m: net.apply(p, c, (o[None, None], reset[None, None], m[None, None]))
        carry2, logits, value = jax.vmap(apply)(params, carry, obs, full)
        logits, value = logits[:, 0, 0], value[:, 0, 0]
        ka, km = jax.random.split(k)
        a = jax.random.categorical(ka, logits[:, :A])
        msg = jax.random.categorical(km, logits[:, A:].reshape(-1, E.MSG_LEN, E.VOCAB))
        probs = jax.nn.softmax(logits[:, :A], axis=-1)

        def lean(p, c, o, m, ai):
            g = jax.grad(lambda o_: apply(p, c, o_, m)[1][0, 0, ai])(o)
            return jax.ops.segment_sum(jnp.abs(g * o), seg, num_segments=n_groups)

        why = jax.vmap(lean)(params, carry, obs, full, a)
        why = why / jnp.maximum(why.sum(-1, keepdims=True), 1e-9)
        ideas = None
        if net.world_model:
            x = lambda o, m: (o[None, None], reset[None, None], m[None, None])
            tr = jax.vmap(lambda p, c, o, m: net.apply(p, c, x(o, m), method=net.trace)[1])(params, carry, obs, full)
            cand, q = tr["cand"][:, 0, 0], tr["q"][:, 0, 0]                          # (N, k)
            ok = jnp.take_along_axis(mask, cand, -1)
            habit = jnp.take_along_axis(jax.nn.softmax(tr["prior"][:, 0, 0, :A], -1), cand, -1)
            best = jnp.take_along_axis(cand, jnp.argmax(jnp.where(ok, q, -1e9), -1)[:, None], -1)[:, 0]
            acts, states, rews, now = jax.vmap(lambda p, c, o, m, b: net.apply(
                p, c, x(o, m), b[None, None], method=net.idea))(params, carry, obs, full, best)
            ideas = {"cand": cand, "q": q, "ok": ok, "habit": habit, "chosen": jnp.take_along_axis(probs, cand, -1),
                     "best": best, "plan": acts[:, 0, 0], "expect": states[:, 0, 0], "now": now[:, 0, 0],
                     "rewards": rews[:, 0, 0], "trust": params["params"]["trust"]}
        return carry2, a, msg, probs, value, why, ideas

    return think


def idea_record(ideas, i):
    """One agent's ideas this tick, JSON-ready: each action it imagined (its habit's probability,
    the final probability after its ideas, the imagined return), the best one played out (the
    actions after it and the change it expects in its own body by the end), and its trust."""
    d = {k: np.asarray(v[i]) for k, v in ideas.items()}
    options = [[E.ACTION_NAMES[int(c)], round(float(h), 3), round(float(p), 3), round(float(q), 3)]
               for c, h, p, q, ok in zip(d["cand"], d["habit"], d["chosen"], d["q"], d["ok"]) if ok]
    if not options:
        return None
    best = E.ACTION_NAMES[int(d["best"])]
    habit_best = max(options, key=lambda o: o[1])[0]
    lo = np.array([0, 0, 0, 0, -1.5, 0, 0.0])                   # the range each own-state value can take
    hi = np.array([1, 1, 1, 1, 1.5, 1, 1.0])
    change = np.clip(d["expect"][-1], lo, hi) - np.clip(d["now"], lo, hi)
    return {"options": options, "best": best, "habit": habit_best,
            # a new idea: the best imagined action is one its habits clearly would not have chosen
            "new": bool(best != habit_best and next(o[1] for o in options if o[0] == best)
                        < min(0.1, 0.5 * max(o[1] for o in options))),
            "plan": [E.ACTION_NAMES[int(a)] for a in d["plan"]],
            "expects": {n: round(float(v), 3) for n, v in zip(DECODE_NAMES, change) if abs(v) >= 0.01},
            "trust": round(float(d["trust"]), 4)}


def mind_records(probs, value, why, reward, social, ideas=None):
    """Per-agent JSON-ready summaries of one decision."""
    probs, value, why = np.asarray(probs), np.asarray(value), np.asarray(why)
    reward, social = np.asarray(reward), np.asarray(social)
    out = []
    for i in range(len(value)):
        top = np.argsort(-probs[i])[:3]
        lean = np.argsort(-why[i])[:3]
        rec = {"top": [[E.ACTION_NAMES[k], round(float(probs[i, k]), 3)] for k in top],
               "value": round(float(value[i]), 3),
               "why": [[MIND_GROUPS[g], round(float(why[i, g]), 2)] for g in lean if why[i, g] > 0.02],
               "felt": round(float(reward[i]), 4), "felt_others": round(float(social[i]), 4)}
        if ideas is not None:
            idea = idea_record(ideas, i)
            if idea:
                rec["idea"] = idea
        out.append(rec)
    return out


def world_static(state, cfg):
    """What does not change over a recording: ground height and steepness (0-9 per cell), the
    constants trait explanations need, and how heritable each trait is."""
    from lifesim import genotype as gt
    h = np.asarray(state.height)
    h = (h - h.min()) / max(float(h.max() - h.min()), 1e-6)
    cl = np.clip(np.asarray(state.cliff), 0, 1)
    digits = lambda x: "".join(str(int(round(9 * v))) for v in x.reshape(-1))
    return {"height_map": digits(h), "cliff_map": digits(cl), "heritability": gt.HERITABILITY,
            "terrain_names": E.TERRAIN_NAMES, "mat_names": E.MAT_NAMES, "n_base_mat": E.N_BASE_MAT, "living": bool(cfg.biomes or cfg.dynamic_weather),
            "conditions": _cond.table(), "minerals": _min.table(),
            "fauna": [dict(row, notice=sp["notice"], attacks=sp["attack"][0] > 0) for row, sp in zip(_fa.table(), _fa.SPECIES)],
            "bugs": _fa.BUGS,
            **({"caves": cave_static(state, cfg)} if cfg.underground else {}),
            "hives": [[int(r), int(c)] for r, c in zip(*np.nonzero(np.asarray(state.hive)))],
            "mounds": [[int(r), int(c)] for r, c in zip(*np.nonzero(np.asarray(state.mound)))],
            "techtree": [{"id": b, "name": nm, "requires": req, "about": about} for b, nm, req, about, _ in _tech.NODES],
            "height_m": [round(float(v) * cfg.relief_m) for v in np.asarray(state.height).reshape(-1)],
            "consts": {"base_height_cm": cfg.base_height_cm, "ideal_bmi": cfg.ideal_bmi,
                       "comfort_temp": cfg.comfort_temp, "air_min": AIR_MIN, "view_radius": cfg.view_radius,
                       "tick_minutes": cfg.tick_minutes, "day_length": cfg.day_length, "stress_sources": list(E.STRESS_SOURCES),
                       "cell_km": cfg.cell_km, "relief_m": cfg.relief_m,
                       "max_bio_years": cfg.max_bio_years, "care_empathy": cfg.care_empathy,
                       "care_kin": cfg.care_kin, "care_reciprocity": cfg.care_reciprocity,
                       "care_revenge": cfg.care_revenge, "loneliness_stress": cfg.loneliness_stress,
                       "intimacy_relief": cfg.intimacy_relief, "map_half_life": cfg.map_half_life,
                       "map_fresh_half_life": cfg.map_fresh_half_life}}


def cave_static(state, cfg):
    """The cave layer for the playback's cave map: per cell 0 rock, 1 passage, 2 pool ('under'),
    the formation (digits, minerals.KINDS), the mouths, hot springs and the cave air by formation."""
    u = np.asarray(state.under).reshape(-1)
    w = np.asarray(state.under_water).reshape(-1)
    mouths, ok = E.cave_mouths(state, cfg)
    return {"under": "".join("2" if wi else ("1" if ui else "0") for ui, wi in zip(u, w)),
            "kind": "".join(str(int(k)) for k in np.asarray(state.under_kind).reshape(-1)),
            "kinds": _min.KINDS, "mouths": [[int(r), int(c)] for (r, c), o in zip(np.asarray(mouths), np.asarray(ok)) if o],
            "springs": [[int(r), int(c)] for r, c in zip(*np.nonzero(np.asarray(state.spring)))],
            "torch_light": cfg.torch_light}


def knowledge_snapshot(t, state):
    """What each agent remembers of every cell, as digit strings (0 = never seen or forgotten):
    'rec' how fresh the memory is, 'plants' how much food it remembers there."""
    mm = np.asarray(state.mind_map)
    q = lambda x: "".join(str(int(v)) for v in np.clip(np.round(9 * x), 0, 9).reshape(-1))
    return {"t": int(t), "rec": [q(mm[i, ..., 2]) for i in range(mm.shape[0])],
            "plants": [q(mm[i, ..., 1] * (mm[i, ..., 2] > 0.05)) for i in range(mm.shape[0])]}


def weather_record(t, state, cfg):
    """Air temperature at mid-map grassland, the climate's offset and, in the living world,
    cloud cover and snow per cell (digits 0-9) and the wind."""
    rec = {"air_c": round(float(E.ambient_temperature(jnp.int32(t), state.weather, jnp.int32(E.GRASS), cfg, 0.3,
                                                       jnp.float32((cfg.grid_size - 1) / 2),
                                                       E.climate_offset(jnp.int32(t), state.climate, cfg))), 1),
           "climate": round(float(E.climate_offset(jnp.int32(t), state.climate, cfg)), 2)}
    if cfg.dynamic_weather:
        dig = lambda x: "".join(str(int(v)) for v in np.clip(np.round(9 * np.asarray(x)), 0, 9).reshape(-1))
        rec.update(clouds=dig(state.clouds), snow=dig(state.snow), wind=[int(v) for v in np.asarray(state.wind)])
    rec.update(overlay_maps(t, state, cfg))
    return rec


AIR_MIN = -30          # air_map: one character per cell, chr(48 + degrees - AIR_MIN), -30..46 C


def overlay_maps(t, state, cfg):
    """Per-cell fields for the playback's map overlays: air temperature where one stands, with a
    fire's warmth (air_map), and how thick the mosquitoes are (bugs_map, digits 0-9)."""
    g = cfg.grid_size
    tt = jnp.int32(t)
    rows = jnp.broadcast_to(jnp.arange(g)[:, None], (g, g)).astype(jnp.float32)
    heat = E.fire_heat(state.fire, cfg)
    air = E.ambient_temperature(tt, state.weather, state.terrain, cfg, state.height, rows,
                                E.climate_offset(tt, state.climate, cfg)) + heat
    a = np.clip(np.round(np.asarray(air)) - AIR_MIN, 0, 76).astype(int).reshape(-1)
    bugs = E.mosquito_density(state, cfg, E.season_of(tt, cfg), E.is_night(tt, cfg), heat)
    return {"air_map": "".join(chr(48 + int(v)) for v in a),
            "bugs_map": "".join(str(int(v)) for v in np.clip(np.round(9 * np.asarray(bugs)), 0, 9).reshape(-1))}


def build_frame(t, prev, state, a, info, evs, env_cfg, n_agents, minds=None):
    """One recorded tick: every agent's state and action, animals, fire, food and events."""
    p = jax.tree_util.tree_map(np.asarray, prev)
    s = jax.tree_util.tree_map(np.asarray, state)
    bod = jax.tree_util.tree_map(np.asarray, E.body(state, env_cfg))
    acts, outs, msgs = np.asarray(a), np.asarray(info.outcome), np.asarray(info.message)
    r_view = env_cfg.view_radius
    sf = np.asarray(info.stress_from)                 # stress gained this tick by source, x 1e4 (E.STRESS_SOURCES)

    def agent_record(i):
        spoke = bool(msgs[i, 0] > 0)
        hearers = [j for j in range(n_agents) if j != i and s.alive[j] and
                   max(abs(int(s.pos[j, 0]) - int(s.pos[i, 0])), abs(int(s.pos[j, 1]) - int(s.pos[i, 1]))) <= r_view]
        return {"r": int(s.pos[i, 0]), "c": int(s.pos[i, 1]), "alive": bool(s.alive[i]),
                "was_alive": bool(p.alive[i]),
                "health": round(float(s.health[i]), 2), "hand": E.MAT_NAMES[int(s.hand[i])],
                "torch": bool(s.torch[i] > 0), "msg": [int(x) for x in msgs[i]],
                "said": spoken_form(msgs[i]) if spoke else "", "heard_by": hearers if spoke else [],
                "action": E.ACTION_NAMES[int(acts[i])], "act": E.OUTCOME_NAMES[int(outs[i])],
                "age": round(float(bod["years"][i]), 1), "height": round(float(bod["height_cm"][i])),
                "weight": round(float(bod["weight_kg"][i]), 1), "strength": round(float(bod["strength"][i]), 2),
                "speed": round(float(bod["speed"][i]), 2), "ideal_kg": round(float(bod["ideal_kg"][i]), 1),
                "pain_now": round(float(s.pain_aversion[i]), 2),
                "conditions": [n for n, v in zip(COND_NAMES, s.conditions[i]) if v > 0.5],
                "food": round(float(s.satiety[i]), 2), "water": round(float(s.hydration[i]), 2),
                "energy": round(float(s.energy[i]), 2), "temp": round(float(s.body_temp[i]), 2),
                "injury": round(float(s.injury[i]), 2), "sick": bool(s.infection[i] > 0),
                "scars": round(float(s.scars[i]), 2), "lasting_injury": round(float(s.lasting_injury[i]), 2),
                "stress": round(float(s.stress[i]), 2), "clothing": round(float(s.clothing[i]), 2),
                "sw": [int(round(1e4 * float(v))) for v in sf[i]] if sf[i].max() > 5e-5 else 0,
                "asleep": bool(s.asleep[i]), "sleepiness": round(float(s.sleepiness[i]), 2),
                "existed": bool(s.existed[i]), "sex": "female" if int(s.sex[i]) == E.FEMALE else "male",
                "pregnant": bool(s.gestation[i] > 0), "courting": bool(s.courting[i] > 0), "mother": int(s.mother[i]),
                "below": bool(s.below[i]), "dark": bool(np.asarray(info.dark)[i]),
                "downed": bool(np.asarray(info.downed)[i]),
                "carrying": {E.MAT_NAMES[m]: round(float(s.inv[i, m]), 1) for m in range(E.N_MAT) if s.inv[i, m] > 0.05}}
    nz = lambda grid, thr=0.3: [[int(r), int(c)] for r, c in np.argwhere(grid > thr)]
    # The social web, read from the agents' own memories (nothing here feeds back into the world):
    # kinship, bonds (touching and food left near each other) and blows struck each way.
    mem = s.memory
    ties = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if not (s.alive[i] and s.alive[j]):
                continue
            kin = float(s.relatedness[i, j])
            bond = float(np.log1p(mem[i, j, E.MEM_TOUCHED_ME] + mem[j, i, E.MEM_TOUCHED_ME]
                                  + mem[i, j, E.MEM_GAVE_NEAR_ME] + mem[j, i, E.MEM_GAVE_NEAR_ME]))
            i_hit_j, j_hit_i = float(np.log1p(mem[j, i, E.MEM_STRUCK_ME])), float(np.log1p(mem[i, j, E.MEM_STRUCK_ME]))
            att_ij, att_ji = float(s.attachment[i, j]), float(s.attachment[j, i])        # i to j, j to i
            if max(kin, bond, i_hit_j, j_hit_i, att_ij, att_ji) > 0.05:
                ties.append([i, j, round(kin, 2), round(bond, 2), round(i_hit_j, 2), round(j_hit_i, 2),
                             round(att_ij, 2), round(att_ji, 2)])
    return {
        "t": int(t), "time": E.clock(t, env_cfg)["stamp"], "night": bool(E.is_night(jnp.int32(t), env_cfg)),
        "weather": E.WEATHER_NAMES[int(s.weather)],
        "agents": [agent_record(i) for i in range(n_agents)],
        # animals: [row, column, species (lifesim/fauna.py), underground (0/1)]
        "small": [[int(r), int(c), int(k), int(b)] for (r, c), al, k, b in zip(s.small.pos, s.small.alive, s.small.species, s.small.below) if al],
        "large": [[int(r), int(c), int(k), int(b)] for (r, c), al, k, b in zip(s.large.pos, s.large.alive, s.large.species, s.large.below) if al],
        "pred": [[int(r), int(c), int(k), int(b)] for (r, c), al, k, b in zip(s.predators.pos, s.predators.alive, s.predators.species, s.predators.below) if al],
        # what lies on the cave floor (with caves): [row, column, material index (E.MAT_NAMES), amount]
        **({"cave_floor": [[int(r), int(c), int(k), round(float(s.cave_floor[r, c, k]), 1)]
                           for r, c, k in zip(*np.nonzero(s.cave_floor > 0.05))]} if env_cfg.underground else {}),
        "honey": nz(s.honey * s.hive, 0.1), "grubs": nz(s.grubs * s.mound, 0.1),
        "fire": nz(s.fire, 0.0), "meat": nz(s.meat), "cooked": nz(s.cooked), "wood": nz(s.wood, 0.8),
        # minerals lying on the ground: [row, column, mineral index, amount]
        "minerals": [[int(r), int(c), int(k), round(float(s.minerals[r, c, k]), 1)]
                     for r, c, k in zip(*np.nonzero(s.minerals > 0.05))],
        "plants": "".join(str(min(3, int(round(float(v))))) for v in s.plants.reshape(-1)),
        "events": [ev["id"] for ev in evs],
        "ties": ties,
        **weather_record(t, state, env_cfg),
        **({"minds": minds} if minds is not None else {}),
    }


def load_population(run: Path, env_cfg: EnvConfig, hidden: int, encoder: str, world_model: bool | None = None):
    """Final outputs if training finished, else the latest checkpoint. world_model: read from the
    run's history when not given."""
    if world_model is None:
        hist = run / "history.json"
        world_model = bool(json.load(open(hist))["args"].get("world_model")) if hist.exists() else False
    net = make_network(env_cfg, hidden, encoder, messages=True, world_model=world_model)
    if not (run / "params.msgpack").exists():
        from lifesim import checkpoint as ckpt
        state = ckpt.load(run / "checkpoint.pkl")
        print(f"using checkpoint at update {int(state['update'])}")
        return net, state["runner"][0], state["pop"]
    with open(run / "population.pkl", "rb") as f:
        pop = pickle.load(f)
    n = env_cfg.n_agents
    dummy = (jnp.zeros((1, 1, E.obs_dim(env_cfg))), jnp.zeros((1, 1), bool),
             jnp.ones((1, 1, E.N_ACTIONS + E.MSG_LEN * E.VOCAB), bool))
    template = jax.vmap(lambda k: net.init(k, jnp.zeros((1, hidden)), dummy))(
        jax.random.split(jax.random.PRNGKey(0), n))
    params = serialization.from_bytes(template, (run / "params.msgpack").read_bytes())
    params = jax.tree_util.tree_map(jnp.asarray, params)        # device arrays, as from a checkpoint
    return net, params, pop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None, help="train_phase2 output folder")
    ap.add_argument("--random", action="store_true",
                    help="random agents: uniform over possible actions, no brain (a statistical baseline)")
    ap.add_argument("--decisions", action="store_true",
                    help="untrained brains: freshly initialized networks that really decide (Decisions and "
                         "Ideas tabs filled), before any learning; the baseline for a trained run's records")
    ap.add_argument("--world-model", action=argparse.BooleanOptionalAction, default=True,
                    help="with --decisions: give the untrained brains a world model (ideas)")
    ap.add_argument("--slate", choices=["blank", "caveman"], default="blank",
                    help="for --random and --decisions: caveman adds the given survival skills")
    ap.add_argument("--underground", action="store_true",
                    help="for --random and --decisions with --world living: caves under the land")
    ap.add_argument("--tick-minutes", type=float, default=None,
                    help="game minutes per tick (default: the run's own, else 30; 1 = RimWorld-like)")
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="mlp")
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--skip-days", type=float, default=0.0, help="run this long before recording")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="runs/playback.html")
    ap.add_argument("--world", choices=["classic", "living"], default="classic", help="for --random and --decisions")
    for knob, what in (("food", "plant abundance (scarcity)"), ("predators", "predator numbers (0 = harmless)"),
                       ("disease", "infection chances"), ("hazards", "extreme weather, lightning, falls")):
        ap.add_argument(f"--{knob}", type=float, default=None,
                        help=f"x {what}; default: the run's own setting, or 1 for --random/--decisions")
    args = ap.parse_args()
    if args.decisions:
        args.random = False
    if not (args.run or args.random or args.decisions):
        ap.error("give --run, --random or --decisions")

    if args.run and not args.random and not args.decisions and (Path(args.run) / "history.json").exists():
        hist = json.load(open(Path(args.run) / "history.json"))["args"]
        args.agents, args.grid = hist["agents"], hist["grid"]
        args.encoder = hist.get("encoder", args.encoder)
    env_cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, skills=args.slate == "caveman")
    if args.run and not args.random and not args.decisions and (Path(args.run) / "history.json").exists():
        h = json.load(open(Path(args.run) / "history.json"))["args"]
        from lifesim.config import living_world, with_reproduction, with_time_scale
        env_cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, skills=h.get("slate") == "caveman",
                            reward_mode=h.get("reward_mode", "acts"), plant_abundance=h.get("scarcity", 1.0))
        env_cfg = with_time_scale(env_cfg, h.get("days_per_year", 40), h.get("days_per_bio_year", 1.0))
        if h.get("world") == "living":
            env_cfg = living_world(env_cfg, underground=bool(h.get("underground", False)),
                                   downed=bool(h.get("downed", False)), grazing=bool(h.get("grazing", False)),
                                   clothing=bool(h.get("clothing", False)))
        if h.get("reproduction"):
            env_cfg = with_reproduction(env_cfg, n_founders=h.get("founders") or None)
    elif args.world == "living":
        from lifesim.config import living_world
        env_cfg = living_world(env_cfg, underground=args.underground)
    from lifesim.threats import with_threats
    run_knobs = {}
    if args.run and not args.random and not args.decisions and (Path(args.run) / "history.json").exists():
        run_knobs = json.load(open(Path(args.run) / "history.json"))["args"]
    knob = lambda k, d=1.0: getattr(args, k) if getattr(args, k) is not None else run_knobs.get(k, d)
    env_cfg = with_threats(env_cfg, food=args.food if args.food is not None else 1.0, predators=knob("predators"),
                           disease=knob("disease"), hazards=knob("hazards"))
    from lifesim.config import with_tick_minutes
    env_cfg = with_tick_minutes(env_cfg, args.tick_minutes if args.tick_minutes is not None
                                else run_knobs.get("tick_minutes", 30.0))
    key = jax.random.PRNGKey(args.seed)
    k_env, key = jax.random.split(key)

    if args.random:
        state = E.reset(k_env, env_cfg)
        policy = None
    else:
        if args.decisions:                        # untrained brains: each agent its own fresh network
            net = make_network(env_cfg, args.hidden, args.encoder, messages=True, world_model=args.world_model)
            dummy = (jnp.zeros((1, 1, E.obs_dim(env_cfg))), jnp.zeros((1, 1), bool),
                     jnp.ones((1, 1, E.N_ACTIONS + E.MSG_LEN * E.VOCAB), bool))
            key, k_net = jax.random.split(key)
            params = jax.vmap(lambda k: net.init(k, jnp.zeros((1, args.hidden)), dummy))(
                jax.random.split(k_net, args.agents))
            state = E.reset(k_env, env_cfg)
        else:
            net, params, pop = load_population(Path(args.run), env_cfg, args.hidden, args.encoder)
            state = E.reset(k_env, env_cfg, jax.tree_util.tree_map(jnp.asarray, pop.genotype),
                            jnp.asarray(pop.appearance))
        A, M = E.N_ACTIONS, E.MSG_LEN * E.VOCAB

        @jax.jit
        def policy(params, carry, obs, mask, reset, k):
            full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (M,), bool)], -1)
            carry, logits, _ = jax.vmap(lambda p, c, o, m: net.apply(
                p, c, (o[None, None], reset[None, None], m[None, None])))(params, carry, obs, full)
            logits = logits[:, 0, 0]
            ka, km = jax.random.split(k)
            a = jax.random.categorical(ka, logits[:, :A])
            msg = jax.random.categorical(km, logits[:, A:].reshape(-1, E.MSG_LEN, E.VOCAB))
            return carry, a, msg

        carry = jnp.zeros((args.agents, 1, args.hidden))
        think = make_thinker(net, env_cfg)

    step = jax.jit(lambda s, a, m: E.step(s, a, env_cfg, m))
    observe = jax.jit(lambda s: (E.observe(s, env_cfg), E.action_mask(s, env_cfg)))
    skip = int(args.skip_days * env_cfg.day_length)
    total = skip + int(args.days * env_cfg.day_length)
    frames, events, last_event, knowledge = [], [], {}, []
    g = env_cfg.grid_size
    for t in range(total):
        obs, mask = observe(state)
        key, k = jax.random.split(key)
        if policy is None:
            ka, km = jax.random.split(k)
            a = jax.random.categorical(ka, jnp.where(mask, 0.0, -1e9))
            msg = jax.random.randint(km, (args.agents, E.MSG_LEN), 0, E.VOCAB) * \
                (jax.random.uniform(km, (args.agents, 1)) < 0.1)
        else:
            if t >= skip:
                carry, a, msg, probs, value, why, ideas = think(params, carry, obs, mask, jnp.asarray(t == 0), k)
            else:
                carry, a, msg = policy(params, carry, obs, mask, jnp.asarray(t == 0), k)
        prev = state
        state, reward, _, info = step(state, a, msg)
        if t < skip:
            continue
        evs = _stamp_events(extract_events(info, np.asarray(a), last_event, state, prev, env_cfg), env_cfg)
        events.extend(evs)
        minds = None if policy is None else mind_records(probs, value, why, reward, info.moral_reward, ideas)
        frames.append(build_frame(t, prev, state, a, info, evs, env_cfg, args.agents, minds))
        if (t - skip) % 12 == 0:
            knowledge.append(knowledge_snapshot(t, state))
    morals = ["empathy", "kin_care", "envy", "guilt", "reciprocity", "vengefulness"]
    geno = jax.tree_util.tree_map(np.asarray, state.genotype)
    from lifesim import techtree
    skills = [dict() for _ in range(args.agents)]           # tech blocks each did in this recording
    for ev in events:
        if 0 <= ev.get("agent", -1) < args.agents:
            for b in techtree.blocks_of(ev):
                skills[ev["agent"]][b] = skills[ev["agent"]].get(b, 0) + 1
    data = {
        "grid": g, "terrain": "".join(str(int(v)) for v in np.asarray(state.terrain).reshape(-1)),
        "source": "random agents (no brain)" if args.random else
                  f"untrained brains{' with a world model' if args.world_model else ''} (no learning yet)"
                  if args.decisions else f"trained run: {Path(args.run).name}",
        "view_radius": env_cfg.view_radius, "inventory_cap": env_cfg.inventory_cap,
        "agents": [{"id": i, "appearance": [round(float(x), 3) for x in np.asarray(state.appearance)[i]],
                    "morals": {m: round(float(getattr(geno, m)[i]), 2) for m in morals},
                    "traits": {t: round(float(getattr(geno, t)[i]), 3) for t in geno._fields},
                    "mutations": {t: round(float(v), 1) for t, v in zip(geno._fields, np.asarray(state.mutations)[i]) if v},
                    "skills": skills[i]}
                   for i in range(args.agents)],
        "frames": frames, "events": events, "knowledge": knowledge, **world_static(state, env_cfg),
    }
    data["language"] = language_metrics(data)
    html = page_html(data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    lang = data["language"]
    print(f"wrote {out}: {len(frames)} ticks, {len(events)} events, {out.stat().st_size // 1024} KB; "
          f"{lang['utterances']} utterances, naming index {lang['naming']['index_bits']} bits, "
          f"{len(lang['names_given'])} individuals named by the agents")


if __name__ == "__main__":
    main()
