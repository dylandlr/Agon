"""Event-sourced recording: an append-only event log, periodic full-state
snapshots, and the action log needed for exact replay.

Rewind = load the nearest earlier snapshot, then replay logged actions forward.
Actions are always logged (not regenerated), so replay stays exact even when
the policy is stochastic or runs on nondeterministic hardware.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import env as E
from . import minerals as _mn
from . import fauna as _fa
from .config import EnvConfig

# policy_fn(key, obs (N, D), mask (N, A), carry) -> (actions (N,), new_carry)
#   or ((actions (N,), messages (N, MSG_LEN)), new_carry) for policies that speak
PolicyFn = Callable


@dataclass
class Recording:
    cfg: EnvConfig
    seed: int
    snapshot_every: int
    snapshots: dict = field(default_factory=dict)   # tick -> State (numpy leaves)
    actions: list = field(default_factory=list)     # per tick: (N,) int array
    messages: list = field(default_factory=list)    # per tick: (N, MSG_LEN) int array
    names: list = field(default_factory=list)       # researcher-facing names; agents never see them
    events: list = field(default_factory=list)      # list of event dicts

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "events.jsonl", "w") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")
        np.save(path / "actions.npy", np.stack(self.actions) if self.actions else np.zeros((0,)))
        np.save(path / "messages.npy", np.stack(self.messages) if self.messages else np.zeros((0,)))
        json.dump(self.names, open(path / "names.json", "w"))
        with open(path / "snapshots.pkl", "wb") as f:
            pickle.dump({"cfg": self.cfg, "seed": self.seed,
                         "snapshot_every": self.snapshot_every,
                         "snapshots": self.snapshots}, f)


def _to_numpy(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _to_jax(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


ANIMAL_NAMES = ["small_prey", "large_prey", "predator"]

# Everything in one tick happens simultaneously. Within a tick the world resolves
# in this fixed order, so each event carries its phase: a timeline lists
# same-tick events in resolution order, which is not an order in real time.
PHASES = ["weather", "movement", "strikes", "animals", "gathering", "eating", "social",
          "fire", "bodies", "deaths"]
EVENT_PHASE = {
    "weather": "weather", "lightning_fire": "fire", "move_failed": "movement", "fall": "movement",
    "strike_agent": "strikes", "strike_animal": "strikes", "kill": "strikes", "ignite": "fire",
    "struck_by_agent": "strikes", "predator_attack": "animals", "gather": "gathering",
    "gather_lost": "gathering", "pulled_burning_wood": "gathering", "eat": "eating",
    "drink": "eating", "ate_cooked": "eating", "release": "social", "touch": "social",
    "fed_fire": "social", "meat_to_heat": "social", "infected": "bodies", "recovered": "bodies",
    "burned": "bodies", "death": "deaths", "took_food": "gathering", "born": "deaths",
    "shock": "weather", "shock_end": "weather", "made": "fire", "preyed": "animals",
    "hurt_by_prey": "strikes", "stung": "gathering",
}


def _stamp_events(events: list, cfg, names: list | None = None) -> list:
    for ev in events:
        if names and ev["agent"] >= 0:
            ev["name"] = names[ev["agent"]]
        ev["time"] = E.clock(ev["t"], cfg)["stamp"]
        ev["phase"] = EVENT_PHASE.get(ev["type"], "bodies")
    events.sort(key=lambda ev: (ev["t"], PHASES.index(ev["phase"]), ev["agent"]))
    return events


# How long a cause stays a candidate for a later effect (30-minute ticks; 48 = one day; scaled to
# the world's tick length in extract_events)
RETALIATION_WINDOW = 96       # striking back someone who struck you
GIFT_WINDOW = 10 ** 9         # food left and taken: linked whenever the taking happens
FIRE_WINDOW = 480             # a fire (or heat) still burning nearby
COOKED_WINDOW = 96            # meat put on the heat, then cooked meat eaten nearby
NEED_LOW = 0.15               # satiety or hydration below this: "starving" / "parched"
WOUNDS = ("struck_by_agent", "predator_attack", "fall", "burned")


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def extract_events(info: E.StepInfo, actions: np.ndarray, prev_last_event: dict,
                   state=None, prev=None, cfg: EnvConfig | None = None) -> list:
    """Convert one step's StepInfo into event records, with their causes.

    Each event carries `parents` (ids of earlier or same-tick events it causally depends on) and
    `because` (one plain-words relation per parent). Links follow the simulation's own mechanics,
    not guesses from timing alone: a blow causes the victim's wound; a death links to what the
    cause of death names (the last blow for violence, the infection for disease, the slide into
    thirst for thirst, a mother's death for an orphan who starved); an infection to the sick
    neighbour, the water or the raw meat it came from; striking back to the blow being answered;
    taking food to the one who left it; a birth to the conception and the mating; a lasting
    condition to the wound, illness or hardship that brought it; fire use to the fire it used.

    `prev_last_event` is the recorder's memory across ticks (pass the same dict every tick).
    With `state` (after the step), `prev` (before) and `cfg`, it also records the body crossing
    into danger (starving, parched, freezing, overheating), and infection by contact and orphaned
    deaths can be traced; without them those links are left out.
    """
    info = _to_numpy(info)
    t = int(info.t)
    events = []
    mem = prev_last_event
    life = mem.setdefault("life", {})                       # slot -> life number (1 = founder)
    A = mem.setdefault("agent", {})                         # slot -> {kind: (id, t)} for the current life
    st = _to_numpy(state) if state is not None else None
    pv = _to_numpy(prev) if prev is not None else None
    n = len(info.outcome)
    per = 30.0 / cfg.tick_minutes if cfg is not None else 1.0       # this world's ticks per calibration tick
    retaliation, fire_window, cooked_window, day = (int(round(w * per)) for w in (RETALIATION_WINDOW, FIRE_WINDOW,
                                                                              COOKED_WINDOW, 48))

    def last(i, *kinds, within=10 ** 9):
        best = None
        for k in kinds:
            e = A.get(i, {}).get(k)
            if e and t - e[1] <= within and (best is None or e[1] > best[1]):
                best = e
        return best

    def emit(i, kind, parents=(), **extra):
        ev = {"id": f"{t}:{i}:{kind}", "t": t, "agent": int(i), "type": kind, "life": life.setdefault(int(i), 1),
              "pos": [int(info.pos[i, 0]), int(info.pos[i, 1])], **extra}
        if hasattr(info, "below") and bool(np.asarray(info.below)[i]):
            ev["below"] = True                              # it happened underground
        ps = [(p, why) for p, why in parents if p]
        ev["parents"] = [p for p, _ in ps]
        ev["because"] = [why for _, why in ps]
        events.append(ev)
        mem[i] = ev["id"]
        A.setdefault(int(i), {})[kind] = (ev["id"], t)
        return ev

    def pid(e):
        return e[0] if e else None

    if bool(info.lightning):
        ev = {"id": f"{t}:world:lightning", "t": t, "agent": -1, "type": "lightning_fire", "parents": [], "because": []}
        events.append(ev)
        mem.setdefault("fires", []).append((ev["id"], t, None))
    if bool(info.weather_changed):
        ev = {"id": f"{t}:world:weather", "t": t, "agent": -1, "type": "weather",
              "weather": E.WEATHER_NAMES[int(info.weather)], "parents": [], "because": []}
        events.append(ev)
        mem["weather"] = (ev["id"], t, E.WEATHER_NAMES[int(info.weather)])

    def fire_near(p, within=None):
        within = fire_window if within is None else within
        best = None
        for fid, ft, fpos in reversed(mem.get("fires", [])):
            if t - ft > within:
                break
            if fpos is None or _cheb(fpos, p) <= 2:
                best = fid
                break
        return best

    hits, gifts = np.asarray(info.hits), np.asarray(info.gifts)
    same_tick = {}                                          # (agent, kind) -> id, this tick
    hardness = np.asarray(E.HARDNESS)

    def tool(i, any_kind=False):
        """What agent i held when it acted (a hard thing, or anything with any_kind), for the tech tree."""
        if pv is None:
            return {}
        h = int(pv.hand[i])
        if pv.inv[i, h] <= 0.05 or (not any_kind and hardness[h] <= 0.2):
            return {}
        return {"tool": E.MAT_NAMES[h]}

    def struck_on(i):
        """The spark-throwing mineral on the cell agent i struck, if any."""
        if pv is None or actions is None:
            return {}
        a = int(np.asarray(actions)[i])
        if not (E.STRIKE_N <= a < E.STRIKE_N + 8):
            return {}
        d = np.asarray(E.NEIGHBORS8)[a - E.STRIKE_N]
        r, c = int(pv.pos[i, 0] + d[0]), int(pv.pos[i, 1] + d[1])
        g = pv.minerals.shape[0]
        if not (0 <= r < g and 0 <= c < g):
            return {}
        from . import minerals as mn
        here = [mn.NAMES[k] for k in range(mn.M) if pv.minerals[r, c, k] > 0.1 and mn.SPARK[k] > 0.5]
        return {"struck_on": here[0]} if here else {}
    for i in range(n):
        if bool(info.born[i]):
            life[i] = life.get(i, 1) + (1 if i in A or i in life else 0)
            A.pop(i, None)
            mem.pop(i, None)
            mother = int(st.mother[i]) if st is not None else -1
            gb = (f"{t}:{mother}:gave_birth", "born to") if mother >= 0 else (None, "")
            ev = emit(i, "born", [gb], mother=mother)
            # new in its line: de novo mutations of traits, and inborn conditions neither parent had
            if st is not None and mother >= 0:
                from . import conditions as cond
                from . import genotype as gt
                jumps = {name: round(float(v), 1) for name, v in zip(gt.Genotype._fields, st.mutations[i]) if v != 0}
                new_conds = [cond.NAMES[k] for k in cond.INBORN if st.conditions[i, k] > 0.5
                             and st.conditions[mother, k] < 0.5 and st.father_cond[mother, k] < 0.5]
                if jumps or new_conds:
                    emit(i, "mutation", [(ev["id"], "a new mutation")], traits=jumps, conditions=new_conds)
    for i in range(n):
        code = int(info.outcome[i])
        p_i = [int(info.pos[i, 0]), int(info.pos[i, 1])]
        if code == E.OUT_GATHER:
            what = {"what": E.MAT_NAMES[int(st.hand[i])].replace("_", " ")} if st is not None else {}
            if what.get("what") == "honey" and pv is not None and cfg is not None and bool(pv.hive[p_i[0], p_i[1]]):
                calm = float(np.asarray(E.fire_heat(pv.fire, cfg))[p_i[0], p_i[1]]) > 0
                what.update(from_hive=True, smoke=calm)
            if hasattr(info, "gathered_kind") and int(np.asarray(info.gathered_kind)[i]) >= 0:
                kk = int(np.asarray(info.gathered_kind)[i])
                what["plant"] = ("greens", "fruit", "nuts", "roots", "dropped food")[kk]
            emit(i, "gather", amount=round(float(info.gathered[i]), 4), **tool(i), **what)
        elif code == E.OUT_STRIKE_AGENT:
            victims = np.flatnonzero(hits[i]).tolist()
            answers = [(pid(A.get(i, {}).get(("struck_by", v))), f"striking back at #{v}") for v in victims
                       if A.get(i, {}).get(("struck_by", v)) and t - A[i][("struck_by", v)][1] <= retaliation]
            aimed = {}
            if hasattr(info, "aim") and int(np.asarray(info.aim)[i]) >= 0:     # where it aimed; whether it landed
                aimed = {"part": E.PARTS[int(np.asarray(info.aim)[i])], "landed": bool(np.asarray(info.blow_landed)[i])}
            ev = emit(i, "strike_agent", answers, victims=victims, **tool(i), **aimed)
            same_tick[(i, "strike_agent")] = ev["id"]
        elif code == E.OUT_RELEASE:
            ev = emit(i, "release", near=np.flatnonzero(gifts[i]).tolist())
            A[i]["release_food"] = (ev["id"], t)
        elif code == E.OUT_TOUCH and info.touches[i].any():
            emit(i, "touch", targets=np.flatnonzero(info.touches[i]).tolist())
        elif code == E.OUT_DRINK:
            ev = emit(i, "drink", [(pid(last(i, "parched")), "relief from thirst")]
                      if A.get(i, {}).get("_parched") else [])
            same_tick[(i, "drink")] = ev["id"]
        elif code == E.OUT_EAT:
            ate = {"what": E.MAT_NAMES[int(pv.hand[i])].replace("_", " ")} if pv is not None else {}
            ev = emit(i, "eat", [(pid(last(i, "took_food")), "eating food someone left")] if last(i, "took_food", within=day) else
                      [(pid(last(i, "starving")), "relief from hunger")] if A.get(i, {}).get("_starving") else [], **ate)
            same_tick[(i, "eat")] = ev["id"]
        elif code in (E.OUT_GATHER_LOST, E.OUT_STRIKE_ANIMAL, E.OUT_MOVE_FAILED):
            ev = emit(i, E.OUTCOME_NAMES[code], **(tool(i) if code == E.OUT_STRIKE_ANIMAL else {}))
            same_tick[(i, E.OUTCOME_NAMES[code])] = ev["id"]
        for j in np.flatnonzero(gifts[:, i]):                   # took food that j had left
            emit(i, "took_food", [(pid(A.get(int(j), {}).get("release_food")), f"food #{int(j)} left")], giver=int(j))
        for k, name in enumerate(ANIMAL_NAMES):
            if bool(info.kills[i, k]):
                sp = int(np.asarray(info.kill_species)[i, k]) if hasattr(info, "kill_species") else -1
                emit(i, "kill", [(same_tick.get((i, "strike_animal")), "the blow that killed it")], animal=name,
                     **({"species": _fa.NAMES[sp]} if sp >= 0 else {}), **tool(i))
        if bool(info.attacked_by_predator[i]):
            sp = int(np.asarray(info.attacker_species)[i]) if hasattr(info, "attacker_species") else -1
            emit(i, "predator_attack", health_loss=round(float(info.health_loss[i]), 4),
                 **({"species": _fa.NAMES[sp]} if sp >= 0 else {}),
                 **({"venom": True} if hasattr(info, "venomed") and bool(np.asarray(info.venomed)[i]) else {}))
        if hasattr(info, "gored") and float(np.asarray(info.gored)[i]) > 0:
            sp = int(np.asarray(info.gored_by)[i])
            emit(i, "hurt_by_prey", [(same_tick.get((i, "strike_animal")), "it struck the animal")],
                 species=_fa.NAMES[sp] if sp >= 0 else "an animal")
        if hasattr(info, "stung") and bool(np.asarray(info.stung)[i]):
            emit(i, "stung", [(pid(last(i, "gather", within=0)), "robbing a hive")])
        if bool(np.asarray(info.deterred)[i]):
            emit(i, "held_off_predator")
    # the receiving side of blows (after all strikes this tick are known)
    for i in range(n):
        if bool(info.struck_by_agent[i]):
            by = np.flatnonzero(hits[:, i]).tolist()
            ev = emit(i, "struck_by_agent", [(same_tick.get((j, "strike_agent")), f"#{j}'s blow") for j in by], by=by)
            for j in by:
                A[i][("struck_by", j)] = (ev["id"], t)
    for i in range(n):
        p_i = [int(info.pos[i, 0]), int(info.pos[i, 1])]
        if bool(info.fell[i]):
            emit(i, "fall")
        if bool(info.newly_infected[i]):
            ps = []
            if same_tick.get((i, "drink")):
                ps.append((same_tick[(i, "drink")], "the water it drank"))
            if same_tick.get((i, "eat")) and bool(info.meat_eaten[i]) and not bool(info.ate_cooked[i]):
                ps.append((same_tick[(i, "eat")], "the raw meat it ate"))
            if pv is not None:
                for j in range(n):
                    if j != i and pv.alive[j] and pv.infection[j] > 0 and _cheb(pv.pos[j], pv.pos[i]) <= 1:
                        ps.append((pid(A.get(j, {}).get("infected")), f"close contact with sick #{j}"))
            mosq = float(np.asarray(info.mosquitoes)[i]) if hasattr(info, "mosquitoes") else 0.0
            emit(i, "infected", ps, **({"mosquitoes": round(mosq, 2)} if mosq > 0.2 else {}))
        if bool(info.recovered[i]):
            emit(i, "recovered", [(pid(last(i, "infected")), "got over the illness")])
        if int(info.ignited[i]) > 0:
            ev = emit(i, "ignite", method=["", "spark", "burning_wood"][int(info.ignited[i])], **tool(i), **struck_on(i))
            mem.setdefault("fires", []).append((ev["id"], t, p_i))
        for flag, kind, why in [(info.torch_lit, "pulled_burning_wood", "the fire it took a branch from"),
                                (info.fed_fire, "fed_fire", "the fire it fed"),
                                (info.meat_to_heat, "meat_to_heat", "the fire's heat"),
                                (info.burned, "burned", "the fire it stood in")]:
            if bool(flag[i]):
                ev = emit(i, kind, [(fire_near(p_i), why)], **(tool(i, any_kind=True) if kind == "fed_fire" else {}))
                if kind == "meat_to_heat":
                    mem.setdefault("cooking", []).append((ev["id"], t, p_i))
                if kind == "fed_fire":                       # what it fed the fire with may smelt something
                    mem.setdefault("fires", []).append((ev["id"], t, p_i))
        if bool(info.ate_cooked[i]):
            cook = next((cid for cid, ct, cp in reversed(mem.get("cooking", []))
                         if t - ct <= cooked_window and _cheb(cp, p_i) <= 2), None)
            emit(i, "ate_cooked", [(cook, "meat put on the heat")])
        if bool(info.collapsed_now[i]):
            emit(i, "collapsed_asleep")
        elif bool(info.fell_asleep[i]):
            emit(i, "fell_asleep")
        if bool(info.woken[i]):
            emit(i, "woken", [(pid(last(i, "struck_by_agent", "predator_attack", "burned", within=0)), "being disturbed")])
        if bool(info.mated[i]):
            ev = emit(i, "mated")
            same_tick[(i, "mated")] = ev["id"]
    for i in range(n):
        if bool(info.conceived[i]):
            dad = int(st.father[i]) if st is not None else -1
            emit(i, "conceived", [(same_tick.get((i, "mated")), "mating"),
                                  (same_tick.get((dad, "mated")), f"#{dad}, the father")], father=dad)
        if bool(info.gave_birth[i]):
            emit(i, "gave_birth", [(pid(last(i, "conceived")), "the pregnancy")],
                 child=[int(c) for c in np.flatnonzero(info.born) if st is None or int(st.mother[c]) == i])
    # carrying salt with meat (once per life): meat spoils slower
    if st is not None:
        salt = E.MIN0 + _mn.IDX["salt"]
        for i in np.flatnonzero(np.asarray(st.alive)):
            a = A.setdefault(int(i), {})
            if not a.get("_salted") and st.inv[i, salt] >= 0.2 and st.inv[i, E.RAW_MEAT] > 0.05:
                emit(int(i), "carried_salt_with_meat")
                A[int(i)]["_salted"] = True
    # the body crossing into danger (needs state)
    if st is not None and cfg is not None and pv is not None:
        for i in range(n):
            if not pv.alive[i]:
                continue
            a = A.setdefault(i, {})
            for name, now, flag, why in [
                    ("starving", st.satiety[i] < NEED_LOW, "_starving", None),
                    ("parched", st.hydration[i] < NEED_LOW, "_parched", None),
                    ("freezing", st.body_temp[i] < cfg.cold_threshold, "_freezing", "cold"),
                    ("overheating", st.body_temp[i] > cfg.heat_threshold, "_overheating", "heat")]:
                if bool(now) and not a.get(flag):
                    ps = []
                    w = mem.get("weather")
                    if why == "cold" and w and w[2] == "cold_snap":
                        ps.append((w[0], "a cold snap"))
                    if why == "heat" and w and w[2] == "heat_wave":
                        ps.append((w[0], "a heat wave"))
                    if why is None and last(i, "infected", within=cfg.disease_duration):
                        ps.append((pid(last(i, "infected")), "illness (it saps food and water)"))
                    mom = int(st.mother[i])
                    if why is None and mom >= 0 and not st.alive[mom] and A.get(i, {}).get("born"):
                        d = mem.get(("death_of_life", mom))
                        if d and d[1] >= A[i]["born"][1]:
                            ps.append((d[0], "its mother's death (no milk)"))
                    extra = {"air": round(float(info.ambient_temp[i]), 1)} if why else {}
                    emit(i, name, ps, **extra)
                    a[flag] = True
                elif not bool(now):
                    a[flag] = False                         # out of danger: a new crossing is a new event
    for i in range(n):
        new = np.flatnonzero(np.asarray(info.new_conditions)[i]) if hasattr(info, "new_conditions") else []
        for k in new:
            from . import conditions as cond
            name = cond.NAMES[int(k)]
            ps = []
            if name == "lame":
                ps = [(pid(last(i, *WOUNDS)), "a wound that never fully healed")]
            elif name == "chronic cough":
                ps = [(pid(last(i, "recovered", "infected")), "the illness it survived")]
            elif name == "haunted":
                ps = [(pid(last(i, *WOUNDS, "infected", "starving", "parched", "freezing", "overheating")),
                       "nearly dying")]
            elif name == "hardened":
                ps = [(pid(last(i, "starving", "parched", "freezing", "overheating")), "long hardship survived")]
            emit(i, "condition", ps, condition=name)
    for i in range(n):
        if bool(info.died[i]):
            cause = E.CAUSE_NAMES[int(info.death_cause[i])]
            kinds = {"violence": ("struck_by_agent",), "predation": ("predator_attack",), "disease": ("infected",),
                     "fall": ("fall",), "burns": ("burned",), "starvation": ("starving",), "thirst": ("parched",),
                     "cold": ("freezing",), "heat": ("overheating",), "injury": WOUNDS,
                     "exhaustion": ("collapsed_asleep",), "childbirth": ("gave_birth", "conceived")}.get(cause, ())
            ps = [(pid(last(i, *kinds)), cause)] if kinds else []
            if not kinds or not ps[0][0]:
                ps = [(mem.get(i), "the last thing that happened to it")]
            prev_down = mem.get("downed")
            finished = prev_down is not None and i < len(prev_down) and bool(prev_down[i]) \
                and cause in ("violence", "predation")              # killed where it lay (RimWorld: finished off)
            ev = emit(i, "death", ps, cause=cause, **({"finished_off": True} if finished else {}))
            mem[("death_of_life", i)] = (ev["id"], t)
    # predators taking animals (the food chain beyond the agents)
    if hasattr(info, "small_preyed_by"):
        for grp, by, spc, pos in (("small_prey", info.small_preyed_by, info.small_species, info.small_pos),
                                  ("large_prey", info.large_preyed_by, info.large_species, info.large_pos)):
            by, spc, pos = np.asarray(by), np.asarray(spc), np.asarray(pos)
            for k in np.flatnonzero(by >= 0):
                events.append({"id": f"{t}:world:preyed:{grp}:{k}", "t": t, "agent": -1, "type": "preyed",
                               "predator": _fa.NAMES[int(by[k])], "prey": _fa.NAMES[int(spc[k])], "group": grp,
                               "pos": [int(pos[k, 0]), int(pos[k, 1])], "parents": [], "because": []})
    # heat made something where it lay: fired clay, copper, tin, iron, bronze (each whole unit)
    if st is not None and pv is not None and hasattr(info, "made"):
        from . import minerals as mn
        for r, c, k in zip(*np.nonzero(np.asarray(info.made) > 0)):
            if np.floor(st.minerals[r, c, k] + 1e-6) > np.floor(pv.minerals[r, c, k] + 1e-6):
                p = [int(r), int(c)]
                parent = fire_near(p)
                events.append({"id": f"{t}:world:made:{r}:{c}:{k}", "t": t, "agent": -1, "type": "made",
                               "product": mn.NAMES[int(k)], "pos": p, "parents": [parent] if parent else [],
                               "because": ["the fire's heat"] if parent else []})
    # going underground and coming back up (cfg.underground)
    if hasattr(info, "delved"):
        for i in np.flatnonzero(np.asarray(info.delved)):
            down = bool(np.asarray(info.below)[i])
            emit(i, "went_underground" if down else "came_up",
                 **({"torch": bool(st.torch[i] > 0)} if st is not None else {}))
    if hasattr(info, "bound"):                              # a point bound to a stick with fiber: a spear
        for i in np.flatnonzero(np.asarray(info.bound)):
            emit(i, "made", [(pid(last(i, "gather")), "what it had gathered")], product="spear", by_hand=True)
    if hasattr(info, "bedded"):                             # lay down to sleep on leaves or a hide
        now_bed = np.asarray(info.bedded)
        was_bed = mem.get("bedded", np.zeros_like(now_bed))
        if len(was_bed) < len(now_bed):
            was_bed = np.concatenate([was_bed, np.zeros(len(now_bed) - len(was_bed), bool)])
        for i in np.flatnonzero(now_bed & ~was_bed):
            emit(i, "bedded")
        mem["bedded"] = now_bed
    if hasattr(info, "downed"):                             # fell and cannot stand; got back up
        now_down = np.asarray(info.downed)
        was_down = mem.get("downed", np.zeros_like(now_down))
        if len(was_down) < len(now_down):                 # the world grew its slots (lifesim/grow.py)
            was_down = np.concatenate([was_down, np.zeros(len(now_down) - len(was_down), bool)])
        for i in np.flatnonzero(now_down & ~was_down):
            shock = st is not None and float(st.health[i]) >= cfg.downed_health and float(st.injury[i]) < cfg.downed_injury
            emit(i, "downed", [(pid(last(i, "struck_by_agent", "predator_attack", "hurt_by_prey", "fall", "stung",
                                        within=max(2, int(round(2 * per))))), "the wound")],
                 **({"pain_shock": True} if shock else {}))
        for i in np.flatnonzero(was_down & ~now_down):
            if st is not None and bool(st.alive[i]):        # up again (not dead)
                emit(i, "stood_up")
        mem["downed"] = now_down
    if hasattr(info, "soiled"):                             # relieved itself, by choice or not
        for i in np.flatnonzero(np.asarray(info.relieved) | np.asarray(info.soiled)):
            r_, c_ = int(info.pos[i, 0]), int(info.pos[i, 1])
            near_water = st is not None and bool(np.asarray(E.adjacent_to_water(st.terrain, np.array([[r_, c_]]), cfg))[0]) \
                if cfg is not None else False
            emit(i, "soiled" if bool(np.asarray(info.soiled)[i]) else "relieved", beside_water=near_water)
    for k in ("fires", "cooking"):                          # only recent ones can still be causes
        if len(mem.get(k, [])) > 200:
            mem[k] = mem[k][-200:]
    return events


def record_episode(policy_fn: PolicyFn, carry, seed: int, cfg: EnvConfig,
                   snapshot_every: int = 100, max_steps: int | None = None,
                   names: list | None = None) -> Recording:
    """Run one episode on the host, logging events, actions and snapshots."""
    key = jax.random.PRNGKey(seed)
    k_env, k_pol = jax.random.split(key)
    state = E.reset(k_env, cfg)
    step_fn = jax.jit(lambda s, a, m: E.step(s, a, cfg, m))
    obs_fn = jax.jit(lambda s: (E.observe(s, cfg), E.action_mask(s, cfg)))
    rec = Recording(cfg=cfg, seed=seed, snapshot_every=snapshot_every,
                    names=list(names) if names else [])   # no names unless the caller has them
    last_event: dict = {}
    for t in range(max_steps or cfg.max_steps):
        if t % snapshot_every == 0:
            rec.snapshots[t] = _to_numpy(state)
        obs, mask = obs_fn(state)
        k_pol, k = jax.random.split(k_pol)
        out, carry = policy_fn(k, obs, mask, carry)
        actions, messages = out if isinstance(out, tuple) else (out, jnp.zeros((cfg.n_agents, E.MSG_LEN), jnp.int32))
        state_before = state
        state, _, done, info = step_fn(state, actions, messages)
        rec.actions.append(np.asarray(actions))
        rec.messages.append(np.asarray(messages))
        prev_state = state_before
        rec.events.extend(_stamp_events(extract_events(info, np.asarray(actions), last_event, state, prev_state, cfg),
                                        cfg, rec.names))
        if bool(done):
            break
    rec.snapshots[int(state.t)] = _to_numpy(state)
    return rec


def rewind(rec: Recording, tick: int) -> E.State:
    """Exact world state at `tick`: nearest earlier snapshot + replayed actions."""
    base = max(t for t in rec.snapshots if t <= tick)
    state = _to_jax(rec.snapshots[base])
    step_fn = jax.jit(lambda s, a, m: E.step(s, a, rec.cfg, m))
    for t in range(base, tick):
        msg = rec.messages[t] if rec.messages else np.zeros((rec.cfg.n_agents, E.MSG_LEN), np.int32)
        state, *_ = step_fn(state, jnp.asarray(rec.actions[t]), jnp.asarray(msg))
    return state


def random_policy(key, obs, mask, carry):
    """Uniform over valid actions. Useful as a baseline and for tests."""
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(key, logits), carry
