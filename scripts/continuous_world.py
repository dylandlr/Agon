"""Run one world continuously (no resets) with trained agents, recorded in chunks.

    python scripts/continuous_world.py --run runs/status_run --days 40 --chunk-days 5
    python scripts/continuous_world.py --run runs/libido_run --days 0      # ongoing: until extinction

    python scripts/continuous_world.py --run R --days 400 --learn --inherit innate   # Darwinian

The world is never reset: seasons, weather, food, animals and the agents' own bodies carry over
from chunk to chunk. Each chunk writes a playback page (chunk_XX.html) and a summary;
summary.json collects them all. With no births, the run stops early if everyone dies.

Learning (--learn). Without it, agents act with their trained weights and never learn. With it,
every individual keeps learning from its own life with the trainer's own update (PPO, watching
others, the world model, and learning only what sleep has consolidated), every --learn-every ticks.
Training is then the species' deep past (the slot networks: what the ancestors' lives taught
the lineage) and this world is history.

Inheritance (--inherit). A newborn's brain comes from one parent (mother or father at random),
with small random mutations:
- parent (Lamarckian): the parent's brain as it is now, with everything that parent learned.
- innate (Darwinian): the brain that parent was born with. What a parent learned in life does not
  pass down through the brain; it can only pass on by being watched (social learning). Only
  meaningful with --learn (without learning, a brain never changes after birth).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_playback import (page_html, build_frame, knowledge_snapshot, load_population, stamp_choice,  # noqa: E402
                             make_thinker, mind_records, world_static)

from lifesim import env as E  # noqa: E402
from lifesim import archetypes  # noqa: E402
from lifesim import conditions as C  # noqa: E402
from lifesim import threats  # noqa: E402
from lifesim import techtree  # noqa: E402
from lifesim.threats import with_threats  # noqa: E402
from lifesim.analysis import language_metrics  # noqa: E402
from lifesim.config import EnvConfig, living_world, with_reproduction, with_tick_minutes, with_time_scale  # noqa: E402
from lifesim.ppo_multi import MAConfig, Population, Transition, make_train  # noqa: E402
from lifesim.recorder import _stamp_events, extract_events  # noqa: E402

MORALS = ["empathy", "kin_care", "envy", "guilt", "reciprocity", "vengefulness"]
TRAITS = MORALS + ["selflessness", "libido", "pain_aversion", "social_reward", "novelty", "strength",
                   "patience", "social_learning", "lr_scale", "speed"]
# What each life did, counted per tick alive (scripts/archetypes.py clusters these into roles)
BEHAVIOURS = ["strikes_on_agents", "struck", "gifts", "touches", "speech", "kills_small", "kills_large",
              "predator_kills", "fire_fed", "drinks", "eats", "gathers", "moves", "rests", "sleeps",
              "company", "matings", "nursing", "skill_water", "skill_food", "skill_flee", "skill_shelter"]


class LifeLog:
    """One record per life: who it was (slot, archetype, traits, parents), how long it lived and how
    it died, and how often it did each behaviour. A slot's next occupant starts a new record."""

    def __init__(self, n, archetype_of, cfg=None):
        self.n, self.archetype_of, self.cfg = n, archetype_of, cfg or EnvConfig()
        self.open, self.done = {}, []

    def start(self, i, t, state, generation, mother=-1, father=-1):
        g = state.genotype
        self.open[i] = {"slot": int(i), "born_t": int(t), "generation": int(generation),
                        "archetype": self.archetype_of.get(i) if t == 0 else None,
                        "sex": "female" if int(state.sex[i]) == E.FEMALE else "male",
                        "mother": int(mother), "father": int(father),
                        "traits": {k: round(float(getattr(g, k)[i]), 3) for k in TRAITS},
                        "ticks": 0, "counts": dict.fromkeys(BEHAVIOURS, 0),
                        "skills": {},                     # tech blocks done in this life (lifesim/techtree.py)
                        "mutations": {k: round(float(v), 1) for k, v in
                                      zip(state.genotype._fields, np.asarray(state.mutations)[i]) if v},
                        # conditions present at birth (inborn), then each one gained later, with when
                        "conditions": [{"name": C.NAMES[k], "t": int(t), "age": 0.0, "how": "born with"}
                                       for k in np.flatnonzero(np.asarray(state.conditions)[i] > 0.5)]}

    def add(self, prev, state, info, t):
        p_alive = np.asarray(prev.alive)
        out = np.asarray(info.outcome)
        hits, gifts, touches = np.asarray(info.hits), np.asarray(info.gifts), np.asarray(info.touches)
        kills, sk = np.asarray(info.kills), np.asarray(info.skill)
        flags = {"matings": np.asarray(info.mated), "nursing": np.asarray(info.nursed),
                 "sleeps": np.asarray(info.slept), "company": np.asarray(info.company),
                 "speech": np.asarray(info.spoke), "fire_fed": np.asarray(info.fed_fire)}
        for i in np.flatnonzero(p_alive):
            life = self.open.get(int(i))
            if life is None:
                continue
            c = life["counts"]
            life["ticks"] += 1
            c["strikes_on_agents"] += int(hits[i].any())
            c["struck"] += int(bool(np.asarray(info.struck_by_agent)[i]))
            c["gifts"] += int(gifts[i].any())
            c["touches"] += int(touches[i].any())
            c["kills_small"] += int(kills[i, 0])
            c["kills_large"] += int(kills[i, 1])
            c["predator_kills"] += int(kills[i, 2]) if kills.shape[1] > 2 else 0
            c["drinks"] += int(out[i] == E.OUT_DRINK)
            c["eats"] += int(out[i] == E.OUT_EAT)
            c["gathers"] += int(out[i] == E.OUT_GATHER)
            c["moves"] += int(out[i] == E.OUT_MOVE)
            c["rests"] += int(out[i] == E.OUT_REST)
            for k, v in flags.items():
                c[k] += int(bool(v[i]))
            c["skill_water"] += int(sk[i] == 1)
            c["skill_food"] += int(sk[i] == 2)
            c["skill_flee"] += int(sk[i] == 3)
            c["skill_shelter"] += int(sk[i] == 4)
        if hasattr(info, "new_conditions"):
            newc = np.asarray(info.new_conditions)
            for i, k in zip(*np.nonzero(newc)):
                life = self.open.get(int(i))
                if life is not None:
                    kind = C.CONDITIONS[int(k)][1]
                    life["conditions"].append({"name": C.NAMES[int(k)], "t": int(t),
                                               "age": round(float(np.asarray(state.age)[i] + np.asarray(state.birth_age)[i])
                                                            / self.cfg.ticks_per_bio_year, 1),
                                               "how": {"age": "with age", "event": "after what happened",
                                                       "use": "earned by use"}.get(kind, kind)})
        for i in np.flatnonzero(np.asarray(info.died)):
            self.close(int(i), t, E.CAUSE_NAMES[int(np.asarray(info.death_cause)[i])])

    def add_events(self, evs):
        """Count the tech blocks each living individual did."""
        for ev in evs:
            life = self.open.get(ev.get("agent", -1))
            if life is not None:
                for b in techtree.blocks_of(ev):
                    life["skills"][b] = life["skills"].get(b, 0) + 1

    def close(self, i, t, cause):
        life = self.open.pop(i, None)
        if life is not None:
            life.update(died_t=int(t), cause=cause)
            self.done.append(life)

    def records(self):
        """Finished lives plus the living (died_t None)."""
        return self.done + [dict(l, died_t=None, cause=None) for l in self.open.values()]


def inherit_brain(params, innate, opt_state, child, src, mode, mutation, key, n):
    """The child's brain in slot `child`: parent `src`'s current brain (mode "parent") or the brain
    it was born with ("innate"), plus mutation (x each weight tensor's spread). The child's inborn
    brain is recorded, and its optimiser moments start fresh. Returns params, innate, opt_state."""
    base = innate if mode == "innate" else params
    leaves, tree = jax.tree_util.tree_flatten(params)
    bases = jax.tree_util.tree_leaves(base)
    ks = jax.random.split(key, len(leaves))
    leaves = [x.at[child].set(y[src] + mutation * jnp.std(y[src]) * jax.random.normal(k, y[src].shape))
              for x, y, k in zip(leaves, bases, ks)]
    params = jax.tree_util.tree_unflatten(tree, leaves)
    innate = jax.tree_util.tree_map(lambda i, p: i.at[child].set(p[child]), innate, params)
    if opt_state is not None:
        opt_state = jax.tree_util.tree_map(
            lambda v: v.at[child].set(0.0) if getattr(v, "ndim", 0) and v.shape[0] == n else v, opt_state)
    return params, innate, opt_state


def drinkable(terrain):
    water = terrain == E.WATER
    g = terrain.shape[0]
    pad = np.pad(water, 1)
    near = np.zeros_like(water)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            near |= pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g]
    return near & ~water


class Chunk:
    """Running tallies for one chunk."""

    def __init__(self, n):
        self.n = n
        self.c = collections.Counter()
        self.deaths = collections.Counter()
        self.season_deaths = collections.Counter()
        self.weather = collections.Counter()
        self.seasons = collections.Counter()
        self.drink_before = []
        self.eat_before = []
        self.needs = collections.defaultdict(list)
        self.min_temp = 9.0

    def add(self, prev, state, info, env_cfg, drink_cells, t):
        g = env_cfg.grid_size
        p = jax.tree_util.tree_map(np.asarray, prev)
        s = jax.tree_util.tree_map(np.asarray, state)
        out = np.asarray(info.outcome)
        alive0 = p.alive
        if env_cfg.dynamic_weather:                     # local weather: how much of the map is under rain
            wet = float((s.clouds > env_cfg.rain_threshold).mean())
            self.weather["rain over 30%+ of the map" if wet >= 0.3 else
                         "rain in places" if wet > 0.02 else "dry everywhere"] += 1
        else:
            self.weather[E.WEATHER_NAMES[int(s.weather)]] += 1
        self.seasons[E.clock(t, env_cfg)["season"]] += 1
        self.c["agent_ticks"] += int(alive0.sum())
        for i in np.where(np.asarray(info.died))[0]:
            self.deaths[E.CAUSE_NAMES[int(np.asarray(info.death_cause)[i])]] += 1
            self.season_deaths[E.clock(t, env_cfg)["season"]] += 1
        for i in np.where(alive0)[0]:
            r, c_ = s.pos[i]
            if out[i] == E.OUT_DRINK:
                self.drink_before.append(float(p.hydration[i]))
            if out[i] == E.OUT_EAT:
                self.eat_before.append(float(p.satiety[i]))
            if p.hydration[i] < 0.3 and drink_cells[p.pos[i][0], p.pos[i][1]] and out[i] != E.OUT_DRINK:
                self.c["thirsty_at_water_not_drinking"] += 1
            if p.hydration[i] < 0.3:
                self.c["thirsty_ticks"] += 1
            if min(r, c_, g - 1 - r, g - 1 - c_) <= 1:
                self.c["edge_ticks"] += 1
            for k, v in (("food", s.satiety), ("water", s.hydration), ("energy", s.energy),
                         ("health", s.health), ("stress", s.stress)):
                self.needs[k].append(float(v[i]))
            self.min_temp = min(self.min_temp, float(s.body_temp[i]))
        pos = s.pos[alive0 & s.alive]
        if len(pos) > 1:
            d = np.max(np.abs(pos[:, None] - pos[None]), -1)
            self.c["pairs_in_view"] += int(((d <= env_cfg.view_radius).sum() - len(pos)) // 2)
        self.c["ticks"] += 1
        self.c["company"] += int(np.asarray(info.company).sum())
        self.c["strikes_on_agents"] += int(np.asarray(info.hits).sum())
        self.c["gifts"] += int(np.asarray(info.gifts).sum())
        self.c["touches"] += int(np.asarray(info.touches).sum())
        self.c["speech"] += int(np.asarray(info.spoke).sum())
        self.c["deterred"] += int(np.asarray(info.deterred).sum())
        self.c["predator_attacks"] += int(np.asarray(info.attacked_by_predator).sum())
        self.c["winter_night_ticks"] += int((np.asarray(info.winter_night) & alive0).sum())
        self.c["winter_huddle_ticks"] += int(np.asarray(info.huddling & info.winter_night).sum())
        kills = np.asarray(info.kills)
        self.c["small_prey_killed"] += int(kills[:, 0].sum())
        self.c["large_prey_killed"] += int(kills[:, 1].sum())
        self.c["meat_eaten"] += int(np.asarray(info.meat_eaten).sum())
        self.c["near_fire"] += int(np.asarray(info.near_fire).sum())
        self.c["fed_fire"] += int(np.asarray(info.fed_fire).sum())
        self.c["ate_cooked"] += int(np.asarray(info.ate_cooked).sum())
        self.c["fires_made"] += int((np.asarray(info.ignited) > 0).sum())
        self.c["lightning_fires"] += int(np.asarray(info.lightning).sum())
        night = bool(E.is_night(jnp.int32(t), env_cfg))
        slept = np.asarray(info.slept)
        self.c["alive_night" if night else "alive_day"] += int(alive0.sum())
        self.c["asleep_night" if night else "asleep_day"] += int(slept.sum())
        if slept.any():
            d = np.max(np.abs(s.pos[:, None] - s.pos[None]), -1)
            beside = ((d <= 1) & alive0[None, :] & ~np.eye(self.n, dtype=bool)).any(1)
            self.c["asleep_beside_someone"] += int((slept & beside).sum())
            self.c["asleep_by_fire"] += int((slept & np.asarray(info.near_fire)).sum())
        sk = np.asarray(info.skill)
        for k, name in ((1, "skill_water"), (2, "skill_food"), (3, "skill_flee"), (4, "skill_shelter")):
            self.c[name] += int((sk == k).sum())
        for k in ("woken", "collapsed_now"):
            self.c[k] += int(np.asarray(getattr(info, k)).sum())
        for k in ("mated", "conceived", "born", "nursed"):
            self.c[k] += int(np.asarray(getattr(info, k)).sum())

    def summary(self, state, start_alive):
        s = jax.tree_util.tree_map(np.asarray, state)
        c, at = self.c, max(self.c["agent_ticks"], 1)
        mean = lambda xs: round(float(np.mean(xs)), 2) if xs else None
        share = lambda xs, f: round(sum(f(x) for x in xs) / len(xs), 2) if xs else None
        alive = s.alive
        return {
            "seasons": dict(self.seasons), "weather": dict(self.weather),
            "alive_start": int(start_alive), "alive_end": int(alive.sum()), "deaths": dict(self.deaths),
            "deaths_by_season": dict(self.season_deaths),
            "mean_needs_of_living": {k: mean(v) for k, v in self.needs.items()},
            "coldest_body_temp": round(self.min_temp, 2) if self.min_temp < 9 else None,
            "drinks": len(self.drink_before), "drinks_when_thirsty_share": share(self.drink_before, lambda x: x < 0.5),
            "eats": len(self.eat_before), "eats_when_hungry_share": share(self.eat_before, lambda x: x < 0.5),
            "thirsty_share": round(c["thirsty_ticks"] / at, 3),
            "thirsty_at_water_not_drinking": c["thirsty_at_water_not_drinking"],
            "edge_share": round(c["edge_ticks"] / at, 2),
            "pairs_in_view_per_tick": round(c["pairs_in_view"] / max(c["ticks"], 1), 2),
            "company_share": round(c["company"] / at, 2),
            "per_1000_agent_ticks": {k: round(1000 * c[k] / at, 2) for k in
                                     ("strikes_on_agents", "gifts", "touches", "speech", "deterred", "predator_attacks")},
            "winter_huddle_share": round(c["winter_huddle_ticks"] / c["winter_night_ticks"], 2) if c["winter_night_ticks"] else None,
            "hunting": {"small_prey_killed": c["small_prey_killed"], "large_prey_killed": c["large_prey_killed"],
                        "meat_eaten": c["meat_eaten"]},
            "fire": {k: c[k] for k in ("near_fire", "fed_fire", "ate_cooked", "fires_made", "lightning_fires")},
            "reproduction": {"matings": c["mated"] // 2, "conceptions": c["conceived"], "births": c["born"],
                             "nursing_ticks": c["nursed"], "pregnant_now": int(((s.gestation > 0) & alive).sum()),
                             "children_alive": int(((s.mother >= 0) & alive).sum())},
            "sleep": {"asleep_share_night": round(c["asleep_night"] / max(c["alive_night"], 1), 2),
                      "asleep_share_day": round(c["asleep_day"] / max(c["alive_day"], 1), 2),
                      "beside_someone_share": round(c["asleep_beside_someone"] / max(c["asleep_night"] + c["asleep_day"], 1), 2),
                      "by_fire_share": round(c["asleep_by_fire"] / max(c["asleep_night"] + c["asleep_day"], 1), 2),
                      "woken": c["woken"], "collapses": c["collapsed_now"]},
            "skills_per_1000_agent_ticks": {k: round(1000 * c["skill_" + k] / at, 1) for k in ("water", "food", "flee")},
            "selfless_alive": int(((s.genotype.selflessness > 0) & alive).sum()),
            "living_scarred": int(((s.scars > 0.1) & alive).sum()),
            "living_lasting_injury": int(((s.lasting_injury > 0.02) & alive).sum()),
            "world": {"plant_stock": round(float(s.plants.sum()), 1), "small_prey": int(s.small.alive.sum()),
                      "large_prey": int(s.large.alive.sum()), "predators": int(s.predators.alive.sum()),
                      "burning_cells": int((s.fire > 0).sum())},
        }


def lives_table(lives, day=48):
    """The run log of every life: who, when, how it ended, what it carried."""
    if not lives:
        return ""
    rows = []
    for l in sorted(lives, key=lambda l: (l["born_t"], l["slot"])):
        died = f"{(l['died_t'] + 1) / day:.1f} ({l['cause']})" if l.get("died_t") is not None else "alive"
        conds = "; ".join(f"{c['name']} ({c['how']}{', age ' + str(c['age']) if c['how'] != 'born with' else ''})"
                          for c in l.get("conditions", [])) or ""
        n_kids = sum(1 for m in lives if m["mother"] == l["slot"] and m["born_t"] > l["born_t"]
                     and (l.get("died_t") is None or m["born_t"] <= l["died_t"]))
        rows.append(f"<tr><td>#{l['slot']}</td><td>{l['born_t'] / day:.1f}</td><td>{died}</td>"
                    f"<td>{l['ticks'] / day:.1f}</td><td>{l['sex']}</td><td>{l['generation']}</td>"
                    f"<td>{l.get('archetype') or ''}</td><td>{n_kids or ''}</td><td>{conds}</td></tr>")
    return (f"<h2>Every life ({len(lives)})</h2><p>Times in days since the world began. A slot number is "
            "reused when its occupant dies, so each row is one life. Conditions: inborn ones, then each one "
            "gained, with the age it arrived.</p><div class=\"wrap\"><table><thead><tr><th>Slot</th><th>Born</th>"
            "<th>Died (cause)</th><th>Lived</th><th>Sex</th><th>Gen</th><th>Archetype</th><th>Children</th>"
            "<th>Conditions</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


class Trace:
    """Per tick, for the hierarchy and culture analyses (scripts/hierarchy.py, scripts/culture.py):
    where everyone is, what they did, what they said and what was around them when they said it
    (a predator, large prey or fire within view). Who is who comes from lives.json (slot, born_t,
    died_t). Saved as trace.npz after every chunk."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pos, self.alive, self.action, self.msg, self.ctx, self.spoke, self.below = [], [], [], [], [], [], []

    def add(self, state, action, msg, info):
        pos = np.asarray(state.pos)
        alive = np.asarray(state.alive)
        spoke = np.asarray(info.spoke) & alive
        self.pos.append(pos.astype(np.uint8))
        below = np.asarray(state.below)
        self.below.append(below)
        self.alive.append(alive)
        self.action.append(np.asarray(action).astype(np.uint8))
        self.spoke.append(spoke)
        self.msg.append(np.where(spoke[:, None], np.asarray(msg), 0).astype(np.uint8))
        r = self.cfg.view_radius
        lay = lambda p, b: p + np.stack([np.zeros_like(b, int), 4096 * np.asarray(b, int)], -1)   # layers apart
        lpos = lay(pos.astype(int), below)
        near = lambda cells, ok: (np.abs(lpos[:, None, :] - cells[None]).max(-1) <= r)[:, ok].any(1) \
            if ok.any() else np.zeros(len(pos), bool)
        fire = np.argwhere(np.asarray(state.fire) > 0)
        an = lambda a: lay(np.asarray(a.pos).astype(int), np.asarray(a.below))
        self.ctx.append(np.stack([near(an(state.predators), np.asarray(state.predators.alive)),
                                  near(an(state.large), np.asarray(state.large.alive)),
                                  near(fire, np.ones(len(fire), bool)) if len(fire) else np.zeros(len(pos), bool)],
                                 -1))

    def save(self, path, args):
        np.savez_compressed(path, pos=np.stack(self.pos), below=np.stack(self.below), alive=np.stack(self.alive),
                            action=np.stack(self.action),
                            msg=np.stack(self.msg), spoke=np.stack(self.spoke), ctx=np.stack(self.ctx),
                            context_names=np.array(["predator", "large_prey", "fire"]),
                            map_seed=np.int64(-1 if args.map_seed is None else args.map_seed),
                            seed=np.int64(args.seed), view_radius=np.int64(self.cfg.view_radius),
                            day_length=np.int64(self.cfg.day_length))


def newborn_memory(memory, c, src, inherit):
    """A newborn in slot c: the steps still carried for the slot were its previous occupant's
    (context no longer: invalid), and its long-term store goes with its brain: the parent's store
    with the parent's learned brain, an empty one with the brain the parent was born with."""
    tail = memory.tail._replace(valid=memory.tail.valid.at[:, c].set(False),
                                spoke=memory.tail.spoke.at[:, c].set(False),
                                stage=memory.tail.stage.at[:, c].set(2))
    res = memory.reservoir
    if inherit == "parent":
        res = jax.tree_util.tree_map(lambda x: x.at[c].set(x[src]), res)
    else:
        res = res._replace(filled=res.filled.at[c].set(False))
    pending = memory.pending          # what the previous occupant had not yet slept on is not the newborn's
    if pending is not None:
        pending = jax.tree_util.tree_map(lambda x: x.at[c].set(0.0), pending)
    return memory._replace(tail=tail, tail_carry=memory.tail_carry.at[c].set(0.0), reservoir=res, pending=pending)


def write_index(out_dir, summaries, lives=None, day=48):
    """index.html: one row per chunk (dates, population, births, deaths, sleep) with a link to its
    playback page where one was recorded."""
    rows = []
    for s in summaries:
        page = f"chunk_{s['chunk']:03d}.html"
        link = f'<a href="{page}">{s["from"]} to {s["to"]}</a>' if (out_dir / page).exists() else f"{s['from']} to {s['to']}"
        deaths = ", ".join(f"{k} {v}" for k, v in sorted(s["deaths"].items(), key=lambda kv: -kv[1])) or "none"
        if s.get("shocks"):
            deaths += "<br><b>shock:</b> " + "; ".join(s["shocks"])
        pop = s.get("population", {})
        rows.append(f"<tr><td>{s['chunk']}</td><td>{link}</td><td>{s['alive_start']} to {s['alive_end']}</td>"
                    f"<td>{s.get('reproduction', {}).get('births', 0)}</td><td>{pop.get('max_generation', '')}</td>"
                    f"<td>{deaths}</td></tr>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Continuous world</title><style>
:root {{ --bg: #e9ecdf; --ink: #22261c; --line: #c3c7b4; --link: #2b86c4; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #1d2236; --ink: #e6e8f2; --line: #3a4060; --link: #6cc6f0; }} }}
body {{ margin: 0; padding: 24px 16px; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; }} h1 {{ font: 600 24px/1.2 "Courier New", monospace; margin: 0 0 6px; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
a {{ color: var(--link); }} .wrap {{ overflow-x: auto; }}
</style></head><body><main><h1>Continuous world</h1>
<p>{len(summaries)} chunks of 5 days, one world never reset. Each linked period opens its playback; every page links to the previous and next.{' <a href="story.txt">Cause and effect: the whole run as stories</a>.' if (out_dir / "story.txt").exists() else ''}{' <a href="techtree.txt">Technology tree and skills by personality</a>.' if (out_dir / "techtree.txt").exists() else ''}</p>
<div class="wrap"><table><thead><tr><th>#</th><th>Period</th><th>Alive</th><th>Births</th><th>Generation</th><th>Deaths</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>{lives_table(lives or [], day)}</main></body></html>"""
    (out_dir / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="train_phase2 output folder")
    ap.add_argument("--days", type=float, default=40.0, help="total length (40 = one year); 0 = keep going until everyone is dead")
    ap.add_argument("--chunk-days", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-playback", action="store_true", help="summaries only")
    ap.add_argument("--playback-every", type=int, default=1, help="write a playback page for every k-th chunk")
    ap.add_argument("--playback-days", type=float, default=None,
                    help="days of each recorded chunk kept in its playback (default: all of it at 30-minute "
                         "ticks; the first day with finer ticks, which record every minute)")
    ap.add_argument("--mutation", type=float, default=0.01, help="newborn brain noise, x each weight tensor's spread")
    ap.add_argument("--shock", action="append", default=[], metavar="KIND:DAY:DAYS:STRENGTH",
                    help="inject a crisis, e.g. famine:40:10:0.2 (food at 0.2x for 10 days from day 40); kinds: "
                         "famine, plague, predators, cold, heat, drought (lifesim/threats.py). Repeatable.")
    ap.add_argument("--learn", action="store_true", help="individuals keep learning from their own lives")
    ap.add_argument("--learn-every", type=int, default=None,
                    help="ticks between learning updates (default: 2 days at 30-minute ticks; with finer ticks "
                         "the run's rollout length, the day then being held as a pending change for sleep)")
    ap.add_argument("--inherit", choices=["parent", "innate"], default="parent",
                    help="newborn brain: the parent's as it is now (parent) or as the parent was born (innate)")
    ap.add_argument("--map-seed", type=int, default=None,
                    help="world (terrain, resources, founders' places) from this seed, events from --seed: "
                         "replicate worlds with the same ecology and different histories")
    ap.add_argument("--archetypes", default=None,
                    help="place archetypes among the founders, e.g. chief:1 (overrides the run's own; "
                         "lifesim/archetypes.py); with --learn their feelings shape what they learn")
    ap.add_argument("--imitation", type=float, default=None,
                    help="social learning weight while living (default: the run's; 0 = no learning by watching)")
    ap.add_argument("--no-trace", action="store_true", help="skip trace.npz (positions, actions, speech per tick)")
    args = ap.parse_args()
    if args.inherit == "innate" and not args.learn:
        raise SystemExit("--inherit innate needs --learn: without learning, brains never change after birth")

    hist = json.load(open(Path(args.run) / "history.json"))["args"]
    n, grid, encoder = hist["agents"], hist["grid"], hist.get("encoder", "mlp")
    env_cfg = EnvConfig(grid_size=grid, n_agents=n, plant_abundance=hist.get("scarcity", 1.0),
                        skills=hist.get("slate", "blank") == "caveman", saint_rate=hist.get("saint_rate", 1e-6),
                        archetypes=archetypes.parse(args.archetypes if args.archetypes is not None
                                                    else hist.get("archetypes", "")),
                        reward_mode=hist.get("reward_mode", "acts"))
    if hist.get("world") == "living":
        env_cfg = living_world(env_cfg, underground=bool(hist.get("underground", False)),
                               downed=bool(hist.get("downed", False)), grazing=bool(hist.get("grazing", False)),
                               clothing=bool(hist.get("clothing", False)), ecology=bool(hist.get("ecology", False)),
                               fouling=bool(hist.get("fouling", False)),
                               excretion=bool(hist.get("excretion", True)))   # runs from before the flag had it on
        import dataclasses as _dc
        env_cfg = _dc.replace(env_cfg, language_mode=hist.get("language", "emergent"), writing=bool(hist.get("writing", False)))
    if hist.get("reproduction"):
        env_cfg = with_reproduction(with_time_scale(env_cfg, hist.get("days_per_year", 40),
                                                    hist.get("days_per_bio_year", 1.0)),
                                    n_founders=hist.get("founders") or None)
    else:
        env_cfg = with_time_scale(env_cfg, hist.get("days_per_year", 40), hist.get("days_per_bio_year", 1.0))
    env_cfg = with_threats(env_cfg, predators=hist.get("predators", 1.0), disease=hist.get("disease", 1.0),
                           hazards=hist.get("hazards", 1.0))
    env_cfg = with_tick_minutes(env_cfg, hist.get("tick_minutes", 30.0))
    if args.learn_every is None:
        args.learn_every = 2 * env_cfg.day_length if env_cfg.tick_minutes >= 30 else int(hist.get("steps", 192))
    shocks = [threats.parse_shock(x) for x in args.shock]
    base_cfg = env_cfg
    out_dir = Path(args.out_dir or Path(args.run) / "continuous")
    out_dir.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    k_env, key = jax.random.split(key)
    net, params, pop = load_population(Path(args.run), env_cfg, args.hidden, encoder,
                                       bool(hist.get("world_model")))
    learner = opt_state = None
    if args.learn:
        lcfg = MAConfig(num_worlds=1, num_minibatches=1, num_steps=args.learn_every, hidden=args.hidden,
                        encoder=encoder, gamma=hist.get("gamma", 0.997), alpha=hist.get("alpha", 0.0),
                        imitation_coef=args.imitation if args.imitation is not None else hist.get("imitation", 0.5),
                        imitation_bias=hist.get("imitation_bias", "success"),
                        imitate_positive_only=not hist.get("imitate_all", False),
                        ent_coef=hist.get("ent_coef", 0.005), msg_ent_coef=hist.get("msg_ent_coef", 0.0005),
                        sleep_learning=not hist.get("no_sleep_learning", False),
                        world_model=bool(hist.get("world_model")), per_agent_lr=not hist.get("no_agent_lr", False),
                        memory_span=min(env_cfg.day_length, args.learn_every),
                        day_ticks=env_cfg.day_length, tick_scale=env_cfg.tick_minutes / 30.0)
        _, _, learner = make_train(env_cfg, lcfg)
        opt_state = learner.opt_init(params)
    innate = params                                  # the brain each individual was born with
    if args.map_seed is not None:
        k_env = jax.random.PRNGKey(args.map_seed)
    state = E.reset(k_env, env_cfg, jax.tree_util.tree_map(jnp.asarray, pop.genotype), jnp.asarray(pop.appearance))
    if args.map_seed is not None:                   # same world, own history
        state = state._replace(key=jax.random.fold_in(jax.random.PRNGKey(args.seed), 7))
    A, M = E.N_ACTIONS, E.MSG_LEN * E.VOCAB

    @jax.jit
    def policy(params, carry, obs, mask, reset, k):
        full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (M,), bool)], -1)
        carry, logits, _ = jax.vmap(lambda p, c, o, m: net.apply(
            p, c, (o[None, None], reset[None, None], m[None, None])))(params, carry, obs, full)
        logits = logits[:, 0, 0]
        ka, km = jax.random.split(k)
        a = jax.random.categorical(ka, logits[:, :A])
        p = jnp.take_along_axis(jax.nn.softmax(logits[:, :A], -1), a[:, None], -1)[:, 0]   # how likely it rated its own choice
        return carry, a, jax.random.categorical(km, logits[:, A:].reshape(-1, E.MSG_LEN, E.VOCAB)), p

    world_fns = {}

    def fns_for(which):
        """The world's step and observation under the shocks in force (compiled once per combination)."""
        if which not in world_fns:
            wc = threats.cfg_during(base_cfg, shocks, which)
            world_fns[which] = (jax.jit(lambda s, a, m: E.step(s, a, wc, m)),
                                jax.jit(lambda s: (E.observe(s, wc), E.action_mask(s, wc))))
        return world_fns[which]

    in_force = ()
    shock_ids = {}
    step, observe = fns_for(in_force)
    think = make_thinker(net, env_cfg)
    carry = jnp.zeros((n, 1, args.hidden))
    drink_cells = drinkable(np.asarray(state.terrain))
    chunk_ticks = int(args.chunk_days * env_cfg.day_length)
    total = int(args.days * env_cfg.day_length) if args.days > 0 else 2 ** 62

    def agent_meta(state):
        """Who occupies each slot now (newborns replace the dead, so this is refreshed per chunk)."""
        geno = jax.tree_util.tree_map(np.asarray, state.genotype)
        return [{"id": i, "appearance": [round(float(x), 3) for x in np.asarray(state.appearance)[i]],
                 "morals": {m: round(float(getattr(geno, m)[i]), 2) for m in MORALS},
                 "libido": round(float(geno.libido[i]), 2), "generation": int(generation[i]),
                 "traits": {t: round(float(getattr(geno, t)[i]), 3) for t in geno._fields},
                 "mutations": {t: round(float(v), 1) for t, v in zip(geno._fields, np.asarray(state.mutations)[i]) if v},
                 "skills": dict(lives.open[i]["skills"]) if i in lives.open else {},
                 "archetype": archetype_of.get(i) if founder_alive_since_start(i) else None}
                for i in range(n)]

    generation = np.where(np.asarray(state.existed), 0, -1)
    archetype_of = dict(enumerate(archetypes.slots(env_cfg.archetypes)))
    replaced = set()                          # slots whose founder has died (a newborn there is someone else)
    founder_alive_since_start = lambda i: i not in replaced
    lives = LifeLog(n, archetype_of, env_cfg)
    for i in np.flatnonzero(np.asarray(state.existed)):
        lives.start(int(i), 0, state, 0)
    summaries, last_event, births_log = [], {}, []
    event_log = gzip.open(out_dir / "events.jsonl.gz", "wt")       # scripts/causal_story.py reads it
    buf, buf_carry, reset_next = [], carry, np.ones(n, bool)
    trace = None if args.no_trace else Trace(env_cfg)
    memory = learner.empty_memory(1, E.obs_dim(env_cfg)) if args.learn else None
    learn_log = collections.Counter()

    def learn_now(params, opt_state, key, memory):
        """One learning update from the buffered experience."""
        traj = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *buf)
        o, m = observe(state)
        key, k = jax.random.split(key)
        params, opt_state, _, met, memory = learner.learn_outside(
            params, opt_state, traj, buf_carry, o[:, None], m[:, None], jnp.asarray(reset_next)[:, None], carry, k,
            Population(state.genotype, state.appearance), memory)
        learn_log["updates"] += 1
        for name, v in met.items():
            learn_log[name] += float(v)
        return params, opt_state, key, memory

    pb_days = args.playback_days if args.playback_days is not None else (
        args.chunk_days if env_cfg.tick_minutes >= 30 else 1.0)
    pb_ticks = int(pb_days * env_cfg.day_length)
    t = 0
    while t < total:
        chunk = Chunk(n)
        start_alive = int(np.asarray(state.alive).sum())
        frames, events, knowledge = [], [], []
        meta = agent_meta(state)
        record = not args.no_playback and len(summaries) % max(1, args.playback_every) == 0
        t0 = t
        for _ in range(min(chunk_ticks, total - t)):
            shock_events = []
            now_in_force = threats.active(shocks, t / env_cfg.day_length)
            if now_in_force != in_force:
                for i in set(now_in_force) - set(in_force):
                    sh = shocks[i]
                    shock_ids[i] = f"{t}:world:shock:{sh['kind']}"
                    shock_events.append({"id": shock_ids[i], "t": t, "agent": -1, "type": "shock", "kind": sh["kind"],
                                         "strength": sh["strength"], "days": sh["days"],
                                         "about": threats.describe(sh), "parents": [], "because": []})
                    if sh["kind"] == "famine":          # standing food is cut to the famine's share at once
                        state = state._replace(plants=state.plants * sh["strength"])
                for i in set(in_force) - set(now_in_force):
                    shock_events.append({"id": f"{t}:world:shock_end:{shocks[i]['kind']}", "t": t, "agent": -1,
                                         "type": "shock_end", "kind": shocks[i]["kind"],
                                         "parents": [shock_ids.get(i)], "because": ["it ran its course"]})
                in_force = now_in_force
                step, observe = fns_for(in_force)
            obs, mask = observe(state)
            key, k = jax.random.split(key)
            rec_now = record and (t - t0) < pb_ticks
            if rec_now:
                carry, a, msg, probs, value, why, ideas = think(params, carry, obs, mask, jnp.asarray(t == 0), k)
                chosen_p = np.asarray(probs)[np.arange(n), np.asarray(a)]
            else:
                carry, a, msg, chosen_p = policy(params, carry, obs, mask, jnp.asarray(t == 0), k)
            prev = state
            state, reward, _, info = step(state, a, msg)
            if args.learn:
                r = reward if not hist.get("alpha") else (1 - hist["alpha"]) * reward + hist["alpha"] * reward.mean()
                col = lambda x: jnp.asarray(x)[:, None]
                buf.append(Transition(
                    col(reset_next | (t == 0)), col(info.died), col(prev.alive), col(a), col(jnp.zeros(n)), col(r),
                    col(jnp.zeros(n)), obs[:, None], mask[:, None], msg[:, None], col(info.spoke & prev.alive),
                    E.layered_pos(prev.pos, prev.below)[:, None], col(prev.health), col(jnp.full(n, E.is_night(prev.t, env_cfg))), col(info.died),
                    col(info.slept * jnp.minimum(prev.sleep_depth + 1, env_cfg.sleep_deep_ticks) / env_cfg.sleep_deep_ticks),
                    col(jnp.ones(n)), col(jnp.zeros(n)), col(jnp.zeros(n, jnp.int32))))
            reset_next = np.zeros(n, bool)
            born = np.asarray(info.born)
            if born.any():
                # A newborn's brain: one parent's (mother or father at random), with small mutations;
                # the parent's current brain (parent) or the one it was born with (innate).
                for c in np.flatnonzero(born):
                    mom = int(np.asarray(state.mother)[c])
                    dad = int(np.asarray(prev.father)[mom])
                    key, kp, km = jax.random.split(key, 3)
                    src = mom if (dad < 0 or bool(jax.random.bernoulli(kp))) else dad
                    params, innate, opt_state = inherit_brain(params, innate, opt_state, c, src, args.inherit,
                                                              args.mutation, km, n)
                    carry = carry.at[c].set(0.0)
                    reset_next[c] = True
                    if args.learn:      # the slot's earlier buffered steps were someone else's (that brain is gone)
                        buf = [tr._replace(valid=tr.valid.at[c].set(False), spoke=tr.spoke.at[c].set(False))
                               for tr in buf]
                        memory = newborn_memory(memory, c, src, args.inherit)
                    generation[c] = 1 + max(generation[mom], generation[dad] if dad >= 0 else 0)
                    births_log.append({"t": t, "time": E.clock(t, env_cfg)["stamp"], "child": int(c),
                                       "mother": mom, "father": dad, "brain_from": src,
                                       "brain": "parent's inborn brain" if args.inherit == "innate"
                                                else "parent's learned brain",
                                       "generation": int(generation[c]),
                                       "libido": round(float(np.asarray(state.genotype.libido)[c]), 3)})
            if args.learn and len(buf) >= args.learn_every:
                params, opt_state, key, memory = learn_now(params, opt_state, key, memory)
                buf, buf_carry = [], carry
            chunk.add(prev, state, info, env_cfg, drink_cells, t)
            if trace is not None:
                trace.add(state, a, msg, info)
            lives.add(prev, state, info, t)
            replaced.update(int(i) for i in np.flatnonzero(np.asarray(info.died)))
            for c in np.flatnonzero(born):
                lives.start(int(c), t, state, int(generation[c]), int(np.asarray(state.mother)[c]),
                            int(np.asarray(prev.father)[int(np.asarray(state.mother)[c])]))
            # every event, recorded chunk or not, goes to the run's causal log (events.jsonl.gz)
            evs = stamp_choice(_stamp_events(shock_events + extract_events(info, np.asarray(a), last_event, state, prev, env_cfg),
                                             env_cfg), chosen_p)
            for i in in_force:                          # hardships a shock makes likelier name it as a cause
                sh = shocks[i]
                for ev in evs:
                    if ev["type"] in threats.SHOCK_EFFECTS[sh["kind"]]:
                        ev["parents"].append(shock_ids[i])
                        ev["because"].append(f"the {sh['kind']}")
            for ev in evs:
                event_log.write(json.dumps(ev, separators=(",", ":")) + "\n")
            lives.add_events(evs)
            if rec_now:
                events.extend(evs)
                frames.append(build_frame(t, prev, state, a, info, evs, env_cfg, n,
                                          mind_records(probs, value, why, reward, info.moral_reward, ideas, a)))
                if (t - t0) % max(1, env_cfg.day_length // 4) == 0:
                    knowledge.append(knowledge_snapshot(t, state))
            t += 1
        idx = len(summaries)
        summ = {"chunk": idx, "from": E.clock(t0, env_cfg)["stamp"], "to": E.clock(t - 1, env_cfg)["stamp"],
                **chunk.summary(state, start_alive)}
        hit = [threats.describe(s) for s in shocks
               if s["day"] < t / env_cfg.day_length and s["day"] + s["days"] > t0 / env_cfg.day_length]
        if hit:
            summ["shocks"] = hit
        if record:
            data = {"grid": grid, "terrain": "".join(str(int(v)) for v in np.asarray(state.terrain).reshape(-1)),
                    "source": f"continuous world, chunk {idx}: {summ['from']} to {summ['to']}",
                    "view_radius": env_cfg.view_radius, "inventory_cap": env_cfg.inventory_cap, "agents": meta, "frames": frames, "events": events,
                    "knowledge": knowledge, **world_static(state, env_cfg)}
            data["language"] = language_metrics(data)
            summ["language"] = {"utterances": data["language"]["utterances"],
                                "distinct": data["language"]["distinct_messages"],
                                "context_bits": data["language"]["context_bits"],
                                "naming_index_bits": data["language"]["naming"]["index_bits"],
                                "names_given": len(data["language"]["names_given"])}
            data["nav"] = {"index": "index.html", "prev": f"chunk_{idx - 1:03d}.html" if idx > 0 else None,
                           "next": f"chunk_{idx + 1:03d}.html", "chunk": idx}
            (out_dir / f"chunk_{idx:03d}.html").write_text(
                page_html(data))
        summ["births_so_far"] = len(births_log)
        if args.learn:
            u = max(learn_log["updates"], 1)
            summ["learning"] = {"updates": int(learn_log["updates"]), "inherit": args.inherit,
                                **{k: round(v / u, 5) for k, v in learn_log.items()
                                   if k in ("entropy", "imitation_loss", "learned_share", "wm_own", "wm_latent",
                                            "wm_reward", "trust", "waiting_share", "retention_kl",
                                            "long_term_filled")}}
            learn_log.clear()
        alive = np.asarray(state.alive)
        lib = np.asarray(state.genotype.libido)
        summ["population"] = {"alive": int(alive.sum()), "pregnant": int((np.asarray(state.gestation) > 0).sum()),
                              "max_generation": int(generation[alive].max()) if alive.any() else None,
                              "mean_libido_alive": round(float(lib[alive].mean()), 3) if alive.any() else None}
        summaries.append(summ)
        json.dump(summaries, open(out_dir / "summary.json", "w"), indent=1)
        json.dump(births_log, open(out_dir / "births.json", "w"), indent=1)
        json.dump(lives.records(), open(out_dir / "lives.json", "w"))
        write_index(out_dir, summaries, lives.records(), env_cfg.day_length)
        if trace is not None:
            trace.save(out_dir / "trace.npz", args)
        event_log.flush()
        print(json.dumps(summ), flush=True)
        if summ["alive_end"] == 0:
            print(f"everyone has died by {summ['to']}; stopping (no births)", flush=True)
            break
    event_log.close()
    # the whole run as cause-and-effect stories (scripts/causal_story.py)
    from causal_story import load_events, tell
    lines, summary, n_stories = tell(load_events(out_dir / "events.jsonl.gz"), day_length=env_cfg.day_length)
    (out_dir / "story.txt").write_text("\n".join([f"Cause and effect: {out_dir}", f"{n_stories:,} stories", ""]
                                                 + lines + summary) + "\n")
    # the technology tree, and which personalities did which skills (scripts/tech_tree.py)
    from tech_tree import report as tech_report
    tlines, tres = tech_report(out_dir, perms=499, day_length=env_cfg.day_length)
    (out_dir / "techtree.txt").write_text("\n".join(tlines) + "\n")
    (out_dir / "techtree.json").write_text(json.dumps(tres, indent=1))
    write_index(out_dir, summaries, lives.records(), env_cfg.day_length)
    print(f"cause and effect: {out_dir / 'story.txt'} ({n_stories:,} stories); technology: {out_dir / 'techtree.txt'}",
          flush=True)


if __name__ == "__main__":
    main()
