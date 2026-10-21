"""The technology tree: building blocks of know-how, as a lens for measuring what emerges.

Agents never see this. It names what they can be observed doing, from the event log, and how
each block physically depends on others: no one can smelt ore without a coal-hot fire, or keep a
fire without one to keep. A block is "discovered" the first time anyone does it, "established"
once it is done by several individuals or on several separate days (it did not die with its
discoverer), and "lost" if an established block then goes unused for a long stretch.

Each block's `match` says which logged events count as doing it (lifesim/recorder.py logs them,
including what was in hand). Prerequisites are physical, not a curriculum: a block can appear
without its prerequisites ever having been logged (fire started from a lightning blaze someone
kept going, say), and the report says so.
"""
from __future__ import annotations

import collections

# id, name, prerequisites, what it is, matcher(event) -> bool
_EDGE = {"flint", "obsidian", "copper", "bronze", "iron", "fired_clay"}
_METAL = {"copper", "bronze", "iron", "tin"}
_KIT = {"pyrite", "iron"}

NODES = [
    ("forage", "Foraging", [], "picking up plants and eating them",
     lambda e: e["type"] == "gather"),
    ("hunt_small", "Hunting small game", [], "killing a small animal",
     lambda e: e["type"] == "kill" and e.get("animal") == "small_prey"),
    ("hunt_large", "Hunting large game", ["hunt_small"], "killing a large animal (armoured: easier in a group)",
     lambda e: e["type"] == "kill" and e.get("animal") == "large_prey"),
    ("defence", "Standing together", [], "a predator held off by agents standing together",
     lambda e: e["type"] == "held_off_predator"),
    ("kill_predator", "Killing a predator", ["defence"], "killing a predator",
     lambda e: e["type"] == "kill" and e.get("animal") == "predator"),
    ("tool_use", "Striking with a tool", [], "a blow struck with something hard in hand",
     lambda e: e["type"] in ("strike_animal", "strike_agent", "kill") and e.get("tool") not in (None, "")),
    ("edge_use", "Using an edge", ["tool_use"], "striking or cutting with flint, obsidian, metal or shards",
     lambda e: e.get("tool") in _EDGE and e["type"] in ("strike_animal", "strike_agent", "kill", "gather")),
    ("carry_fire", "Carrying fire", [], "pulling a burning branch from a fire",
     lambda e: e["type"] == "pulled_burning_wood"),
    ("keep_fire", "Keeping a fire", [], "feeding a burning fire",
     lambda e: e["type"] == "fed_fire"),
    ("fire_from_fire", "Starting fire from fire", ["carry_fire"], "lighting fuel with a burning branch",
     lambda e: e["type"] == "ignite" and e.get("method") == "burning_wood"),
    ("fire_by_spark", "Making fire by striking", ["tool_use"], "lighting dry fuel with sparks from a blow",
     lambda e: e["type"] == "ignite" and e.get("method") == "spark"),
    ("fire_kit", "A fire kit", ["fire_by_spark"], "sparks from pyrite or iron struck on something hard",
     lambda e: e["type"] == "ignite" and e.get("method") == "spark" and (e.get("tool") in _KIT or e.get("struck_on") in _KIT)),
    ("cook", "Cooking", ["keep_fire"], "putting meat by a fire, or eating cooked meat",
     lambda e: e["type"] in ("meat_to_heat", "ate_cooked")),
    ("preserve", "Salting meat", [], "carrying salt with meat (it spoils slower)",
     lambda e: e["type"] == "carried_salt_with_meat"),
    ("share_food", "Sharing food", [], "taking food someone left",
     lambda e: e["type"] == "took_food"),
    ("fishing", "Fishing", [], "catching a fish from the bank",
     lambda e: e["type"] == "kill" and e.get("species") == "fish"),
    ("big_game", "Big game", ["hunt_large"], "killing an aurochs, a mammoth or a bear",
     lambda e: e["type"] == "kill" and e.get("species") in ("aurochs", "mammoth", "bear")),
    ("honey", "Robbing a hive", [], "taking honey from a bees' hive",
     lambda e: e["type"] == "gather" and e.get("from_hive")),
    ("smoking_bees", "Smoking the bees", ["honey", "keep_fire"], "taking honey with a fire's smoke calming the bees",
     lambda e: e["type"] == "gather" and e.get("from_hive") and e.get("smoke")),
    ("termite_fishing", "Fishing for termites", ["forage"], "drawing grubs from a mound with a stick",
     lambda e: e["type"] == "gather" and e.get("what") == "grubs" and e.get("tool") == "wood"),
    ("ceramics", "Firing clay", ["keep_fire"], "clay hardened in a fire",
     lambda e: e["type"] == "made" and e.get("product") == "fired clay"),
    ("coal_fire", "A coal fire", ["keep_fire"], "feeding a fire with coal",
     lambda e: e["type"] == "fed_fire" and e.get("tool") == "coal"),
    ("smelting", "Smelting", ["coal_fire"], "ore turned to metal in coal heat",
     lambda e: e["type"] == "made" and e.get("product") in ("copper", "tin", "iron", "lead")),
    ("alloy", "Bronze", ["smelting"], "copper and tin heated together into bronze",
     lambda e: e["type"] == "made" and e.get("product") == "bronze"),
    ("metal_tools", "Metal tools", ["smelting", "tool_use"], "striking or cutting with metal",
     lambda e: e.get("tool") in _METAL and e["type"] in ("strike_animal", "strike_agent", "kill", "gather")),
    # underground (cfg.underground)
    ("caving", "Going underground", [], "going down a cave mouth",
     lambda e: e["type"] == "went_underground"),
    ("torch_caving", "Lighting the way", ["carry_fire", "caving"], "going underground with a burning branch",
     lambda e: e["type"] == "went_underground" and e.get("torch")),
    ("mining", "Mining", ["caving"], "taking ore or crystal from a cave wall",
     lambda e: e["type"] == "gather" and e.get("below") and e.get("what") not in (None, "stone", "raw meat",
                                                                                "cooked meat", "wood", "plant", "honey", "grubs")),
    ("lime", "Burning lime", ["coal_fire", "mining"], "limestone burned to quicklime in coal heat",
     lambda e: e["type"] == "made" and e.get("product") == "quicklime"),
]
IDS = [n[0] for n in NODES]
BY_ID = {n[0]: n for n in NODES}


def blocks_of(event) -> list:
    """The tech blocks an event shows being done."""
    return [n[0] for n in NODES if n[4](event)]


def tree(events, day_length=48, established_people=3, established_days=3, lost_days=40) -> dict:
    """Per block: when first done and by whom, how many individuals and days, last done, status."""
    t_end = max((e["t"] for e in events), default=0)
    first, last = {}, {}
    people, days = collections.defaultdict(set), collections.defaultdict(set)
    count = collections.Counter()
    for e in sorted(events, key=lambda e: e["t"]):
        for b in blocks_of(e):
            who = (e["agent"], e.get("life", 1)) if e.get("agent", -1) >= 0 else ("world", e.get("pos") and tuple(e["pos"]))
            if b not in first:
                first[b] = {"t": e["t"], "time": e.get("time"), "agent": e.get("agent"), "life": e.get("life")}
            last[b] = e["t"]
            people[b].add(who)
            days[b].add(e["t"] // day_length)
            count[b] += 1
    out = {}
    for b in IDS:
        if b not in first:
            out[b] = {"status": "undiscovered", "requires": BY_ID[b][2]}
            continue
        est = len(people[b]) >= established_people or len(days[b]) >= established_days
        lost = est and (t_end - last[b]) > lost_days * day_length
        missing = [p for p in BY_ID[b][2] if p not in first or first[p]["t"] > first[b]["t"]]
        out[b] = {"status": "lost" if lost else "established" if est else "discovered",
                  "first": first[b], "individuals": len(people[b]), "days": len(days[b]), "times": count[b],
                  "last_t": last[b], "requires": BY_ID[b][2], "before_its_prerequisites": missing}
    return out


def _n(k, word):
    return f"{k} {word}{'' if k == 1 else 's'}"


def render(t: dict) -> list:
    """The tree as text lines, prerequisites first."""
    lines, done = [], set()

    def show(b, depth):
        n, x = BY_ID[b], t[b]
        mark = {"undiscovered": "  ", "discovered": "o ", "established": "* ", "lost": "x "}[x["status"]]
        if x["status"] == "undiscovered":
            info = "not yet"
        else:
            f = x["first"]
            who = f"#{f['agent']}" + (f".{f['life']}" if (f.get("life") or 1) > 1 else "") if (f.get("agent") or -1) >= 0 else "the world"
            info = (f"{x['status']}: first {f.get('time') or 'tick ' + str(f['t'])} by {who}; "
                    f"{_n(x['individuals'], 'individual')}, {_n(x['times'], 'time')} on {_n(x['days'], 'day')}")
            if x["before_its_prerequisites"]:
                info += f" (before {', '.join(x['before_its_prerequisites'])})"
        lines.append(f"{'    ' * depth}{mark}{n[1]}: {info}")
        done.add(b)
        for c in IDS:
            if b in BY_ID[c][2] and BY_ID[c][2][-1] == b and c not in done:
                show(c, depth + 1)
    for b in IDS:
        if not BY_ID[b][2] and b not in done:
            show(b, 0)
    for b in IDS:                                          # anything not reached through its last prerequisite
        if b not in done:
            show(b, 0)
    return lines
