"""Cause and effect, as a plain list of chains: what led to what in a run.

    python scripts/causal_story.py runs/exp/validate/caveman/continuous     # a continuous world
    python scripts/causal_story.py runs/exp/validate/caveman                # same (finds continuous/)
    python scripts/causal_story.py some_playback.html                       # one recorded page
    python scripts/causal_story.py DIR --agent 5                            # stories involving slot 5
    python scripts/causal_story.py DIR --from-day 40 --to-day 80 --out story.txt

Every event the recorder logs carries its causes (lifesim/recorder.py: extract_events), taken
from the simulation's own mechanics: a blow causes the victim's wound, a wound that never healed
causes lameness, a death links to what its cause names (the last blow for violence, the slide
into thirst for thirst, the infection for disease, a mother's death for an orphan who starved),
an infection to the sick neighbour, water or raw meat it came from, striking back to the blow
being answered, taking food to whoever left it, a birth to the pregnancy and the mating, fire use
to the fire. This script walks those links and prints every chain from a first cause to its
final effect:

    D2 03:00 #3 struck #5 -> (#3's blow) #5 was struck by #3 -> (a wound that never fully healed)
    #5 developed lame -> ...

Events that are part of no chain (most eating, walking, sleeping) are left out. A slot number is
reused when its occupant dies, so a later life in slot 5 is written #5.2, the third #5.3.
Routine chains (thirst relieved by a drink, an illness recovered from) are shown with --all.
At the end: every death, traced back to its first cause, tallied.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

PHRASE = {
    "gather": "picked up food", "gather_lost": "lost a scramble for food", "eat": "ate", "drink": "drank",
    "strike_animal": "struck an animal", "kill": "killed", "move_failed": "stumbled",
    "release": "left food on the ground", "predator_attack": "was attacked by a predator",
    "fall": "fell on rock", "infected": "fell ill", "recovered": "recovered", "ignite": "started a fire",
    "pulled_burning_wood": "pulled a burning branch from a fire", "fed_fire": "fed a fire",
    "meat_to_heat": "put meat by the fire", "ate_cooked": "ate cooked meat", "burned": "was burned",
    "fell_asleep": "fell asleep", "collapsed_asleep": "collapsed asleep from exhaustion", "woken": "was woken",
    "mated": "mated", "conceived": "conceived a child", "gave_birth": "gave birth", "born": "was born",
    "starving": "began starving", "parched": "became parched", "freezing": "began freezing",
    "overheating": "began overheating", "took_food": "took food", "touch": "touched",
    "held_off_predator": "held off a predator with others beside it", "carried_salt_with_meat": "carried salt with its meat",
    "mutation": "a mutation",
    "strike_agent": "a blow", "struck_by_agent": "being struck", "condition": "a lasting condition",
    "death": "a death", "weather": "the weather", "lightning_fire": "lightning"
}
def load_events(src: Path) -> list:
    """Events from a continuous world folder (events.jsonl.gz), a run folder, a playback page or
    an events.jsonl file."""
    if src.is_dir():
        for cand in (src / "events.jsonl.gz", src / "continuous" / "events.jsonl.gz", src / "events.jsonl"):
            if cand.exists():
                return load_events(cand)
        pages = sorted(src.glob("chunk_*.html"))
        if pages:
            evs = {}
            for p in pages:
                for e in load_events(p):
                    evs[e["id"]] = e
            return list(evs.values())
        raise SystemExit(f"no events.jsonl.gz, events.jsonl or chunk pages in {src}")
    if src.suffix == ".gz":
        with gzip.open(src, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    if src.suffix == ".jsonl":
        return [json.loads(line) for line in open(src) if line.strip()]
    if src.suffix == ".html":
        m = re.search(r"const DATA = (\{.*?\});\n", src.read_text(), re.S)
        if not m:
            raise SystemExit(f"{src} has no playback data")
        return json.loads(m.group(1))["events"]
    raise SystemExit(f"cannot read events from {src}")


class Teller:
    """Describes events in words, with lives told apart (#5, #5.2, ...)."""

    def __init__(self, events):
        self.lives = collections.defaultdict(list)                  # slot -> [(t, life)]
        for e in sorted(events, key=lambda e: e["t"]):
            if e.get("agent", -1) >= 0 and e.get("life"):
                lv = self.lives[e["agent"]]
                if not lv or lv[-1][1] != e["life"]:
                    lv.append((e["t"], e["life"]))

    def who(self, k, t, life=None):
        if life is None:
            life = 1
            for tt, lf in self.lives.get(k, []):
                if tt <= t:
                    life = lf
                else:
                    break
        return f"#{k}" if life <= 1 else f"#{k}.{life}"

    def say(self, e):
        t = e["t"]
        if e.get("agent", -1) < 0:
            if e["type"] == "shock":
                return f"a shock began: {e.get('about', e.get('kind'))}"
            if e["type"] == "shock_end":
                return f"the {e.get('kind')} ended"
            if e["type"] == "preyed":
                return f"a {e.get('predator')} took a {e.get('prey')}"
            if e["type"] == "made":
                return f"heat made {e.get('product')} at row {e['pos'][0]}, column {e['pos'][1]}"
            return f"weather turned {e.get('weather', '').replace('_', ' ')}" if e["type"] == "weather" \
                else "lightning started a fire"
        w = lambda k: self.who(k, t)
        a = self.who(e["agent"], t, e.get("life"))
        ty = e["type"]
        if ty == "strike_agent":
            return f"{a} struck {' and '.join(map(w, e.get('victims', []))) or 'someone'}"
        if ty == "struck_by_agent":
            return f"{a} was struck by {' and '.join(map(w, e.get('by', []))) or 'someone'}"
        if ty == "touch":
            return f"{a} touched {' and '.join(map(w, e.get('targets', []))) or 'someone'}"
        if ty == "gather" and e.get("what"):
            return f"{a} picked up {e['what']}" + (f" with {e['tool'].replace('_', ' ')}" if e.get("tool") else "")
        if ty == "took_food":
            return f"{a} took food {w(e['giver'])} had left" if "giver" in e else f"{a} took food"
        if ty == "kill":
            return f"{a} killed a {e.get('species') or e.get('animal', 'animal').replace('_', ' ')}"
        if ty == "predator_attack":
            return f"{a} was attacked by a {e.get('species', 'predator')}" + (" (venomous)" if e.get("venom") else "")
        if ty == "hurt_by_prey":
            return f"{a} was hurt by the {e.get('species', 'animal')} it struck"
        if ty == "stung":
            return f"{a} was stung by bees"
        if ty == "eat" and e.get("what"):
            return f"{a} ate {e['what']}"
        if ty == "death":
            return f"{a} died of {e.get('cause', '?')}"
        if ty == "condition":
            return f"{a} developed {e.get('condition')}"
        if ty == "mutation":
            parts = [f"{k.replace('_', ' ')} {'+' if v > 0 else ''}{v} SD" for k, v in (e.get("traits") or {}).items()]
            parts += [f"{c} (neither parent had it)" for c in e.get("conditions") or []]
            return f"{a} was born with a mutation: {', '.join(parts)}"
        if ty == "gave_birth":
            kids = e.get("child") or []
            return f"{a} gave birth" + (f" to {' and '.join(self.who(c, t + 1) for c in kids)}" if kids else "")
        if ty == "born":
            return f"{a} was born" + (f" to {w(e['mother'])}" if e.get("mother", -1) >= 0 else "")
        if ty == "conceived":
            return f"{a} conceived a child" + (f" with {w(e['father'])}" if e.get("father", -1) >= 0 else "")
        if ty in ("freezing", "overheating") and "air" in e:
            return f"{a} {PHRASE[ty]} (air {e['air']} C)"
        if ty == "ignite":
            return f"{a} started a fire with {'a spark' if e.get('method') == 'spark' else 'burning wood'}"
        return f"{a} {PHRASE.get(ty, ty.replace('_', ' '))}"


def short_time(e):
    """'Y1 Spring D2 03:00' -> 'Y1 Spring D2 03:00' (kept whole: years and seasons matter)."""
    return e.get("time") or f"tick {e['t']}"


def stories(events):
    """Group linked events into stories (connected pieces of the cause graph). Within a story each
    event hangs from its main cause (its first recorded parent); other causes are noted inline.
    Returns [(events in time order, roots)], events by id, and each event's consequences in time order."""
    by_id = {e["id"]: e for e in events}
    main_parent, kids = {}, collections.defaultdict(list)
    for e in events:
        ps = [p for p in e.get("parents") or [] if p in by_id]
        if ps:
            main_parent[e["id"]] = ps[0]
            kids[ps[0]].append(e["id"])
    linked = set(main_parent) | set(kids)
    up = {i: i for i in linked}                     # connected pieces, counting every link

    def find(i):
        while up[i] != i:
            up[i] = up[up[i]]
            i = up[i]
        return i
    for e in events:
        for p in e.get("parents") or []:
            if p in up and e["id"] in up:
                up[find(e["id"])] = find(p)
    groups = collections.defaultdict(list)
    for i in linked:
        groups[find(i)].append(by_id[i])
    for k in kids:
        kids[k].sort(key=lambda i: (by_id[i]["t"], i))
    out = []
    for g in groups.values():
        g.sort(key=lambda e: (e["t"], e["id"]))
        out.append((g, [e for e in g if e["id"] not in main_parent]))
    out.sort(key=lambda x: (x[0][0]["t"], x[0][0]["id"]))
    return out, by_id, kids


# A story is told if it contains one of these, a feud (a blow answered), or 4+ linked events;
# otherwise it is routine (a single blow and nothing after, thirst then a drink): use --all
MAJOR = {"death", "born", "conceived", "condition", "mutation", "kill", "ignite", "fed_fire", "meat_to_heat",
         "ate_cooked", "pulled_burning_wood", "took_food", "lightning_fire", "shock", "made", "hurt_by_prey", "stung"}


def render_story(g, roots, by_id, kids, teller, indent="  "):
    """A story as lines: each first cause starts a chain that follows its earliest consequence;
    other consequences branch off on their own lines ("↳ after ...")."""
    lines = []

    def step_text(e, prev):
        ps = [(p, w) for p, w in zip(e.get("parents") or [], (e.get("because") or []) + [""] * 9) if p in by_id]
        why = ps[0][1] if ps else ""
        also = [w or teller.say(by_id[p]) for p, w in ps[1:]]
        day, prev_day = e.get("time", "").rsplit(" ", 1)[0], prev.get("time", "").rsplit(" ", 1)[0]
        when = e.get("time", "") if day != prev_day else e.get("time", "").rsplit(" ", 1)[-1]
        txt = f"{when} {teller.say(e)}"
        if also:
            txt += f" [also: {'; '.join(also)}]"
        return f"({why}) {txt}" if why else txt

    todo = [(r, None) for r in roots]
    while todo:
        node, branch_from = todo.pop(0)
        if branch_from is None:
            parts = [f"{short_time(node)} {teller.say(node)}"]
        else:
            parts = [f"↳ after {short_time(branch_from)} {teller.say(branch_from)}:", step_text(node, branch_from)]
        cur = node
        while kids.get(cur["id"]):
            nxt = [by_id[i] for i in kids[cur["id"]]]
            todo.extend((other, cur) for other in nxt[1:])
            parts.append(step_text(nxt[0], cur))
            cur = nxt[0]
        lines.append((indent if branch_from else "") + " -> ".join(parts))
    return lines


def involves(e, k):
    return e.get("agent") == k or any(k in (e.get(f) or []) for f in ("victims", "by", "targets", "child")) \
        or e.get("giver") == k or e.get("mother") == k or e.get("father") == k


def tell(events, agent=None, from_day=None, to_day=None, routine=False, day_length=48, max_chains=None):
    """The story as lines (one block per story), a summary of deaths, and how many stories."""
    teller = Teller(events)
    groups, by_id, kids = stories(events)
    lines, shown = [], 0
    for g, roots in groups:
        types = {e["type"] for e in g}
        feud = any("striking back" in w for e in g for w in e.get("because") or [])
        if not routine and not (types & MAJOR or feud or len(g) >= 4):
            continue
        if agent is not None and not any(involves(e, agent) for e in g):
            continue
        start, end = g[0]["t"] / day_length, g[-1]["t"] / day_length
        if (from_day is not None and end < from_day) or (to_day is not None and start > to_day):
            continue
        who = sorted({e["agent"] for e in g if e.get("agent", -1) >= 0})
        span = short_time(g[0]) + (f" to {short_time(g[-1])}" if g[-1]["t"] != g[0]["t"] else "")
        shown += 1
        lines.append(f"Story {shown}: {span}; {len(g)} events; "
                     f"{', '.join(teller.who(k, g[-1]['t']) for k in who) or 'the world'}")
        lines.extend(render_story(g, roots, by_id, kids, teller))
        lines.append("")
        if max_chains and shown >= max_chains:
            break
    # every death, traced back along its first cause to where it began
    begins, deaths = collections.Counter(), collections.Counter()
    for e in events:
        if e["type"] != "death" or (agent is not None and e["agent"] != agent):
            continue
        deaths[e.get("cause", "?")] += 1
        cur, steps = e, 0
        while cur.get("parents") and cur["parents"][0] in by_id and steps < 500:
            cur, steps = by_id[cur["parents"][0]], steps + 1
        begin = "nothing recorded before it" if cur is e else \
            (f"weather ({cur.get('weather', '')})" if cur["type"] == "weather" else
             PHRASE.get(cur["type"], cur["type"].replace("_", " ")))
        begins[(e.get("cause", "?"), begin)] += 1
    summary = []
    if deaths:
        summary.append("Deaths, traced back to their first recorded cause:")
        for cause, total in deaths.most_common():
            parts = ", ".join(f"{n} began with: {b}" for (c, b), n in begins.most_common() if c == cause)
            summary.append(f"  {cause}: {total} ({parts})")
    return lines, summary, shown


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="continuous world folder, run folder, playback .html or events.jsonl(.gz)")
    ap.add_argument("--agent", type=int, default=None, help="only stories involving this slot")
    ap.add_argument("--from-day", type=float, default=None)
    ap.add_argument("--to-day", type=float, default=None)
    ap.add_argument("--all", action="store_true", help="include routine stories (a blow and nothing after, thirst then a drink, ...)")
    ap.add_argument("--max", type=int, default=None, help="at most this many stories")
    ap.add_argument("--out", default=None, help="write here instead of printing")
    args = ap.parse_args()
    events = load_events(Path(args.source))
    lines, summary, n_stories = tell(events, args.agent, args.from_day, args.to_day, args.all, max_chains=args.max)
    t0 = min((e["t"] for e in events), default=0)
    t1 = max((e["t"] for e in events), default=0)
    head = [f"Cause and effect: {args.source}",
            f"{len(events):,} events from tick {t0:,} to {t1:,}; {n_stories:,} stories"
            + (f" involving #{args.agent}" if args.agent is not None else ""), ""]
    text = "\n".join(head + lines + summary) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}: {n_stories:,} stories")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
