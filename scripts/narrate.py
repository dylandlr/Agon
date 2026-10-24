"""A run's events as a story to read: narrative.txt from the recorder's own log, nothing invented.

    python scripts/narrate.py runs/x/validate/caveman/continuous            # a continuous world
    python scripts/narrate.py runs/x/validate/caveman --out story.txt       # same (finds continuous/)
    python scripts/narrate.py DIR --from-day 30 --to-day 60 --agent 3
    python scripts/narrate.py DIR --minds                  # every decision and thought, from minds.jsonl.gz

With --minds (the continuous world writes minds.jsonl.gz unless --no-minds), each tick also gives,
for every awake agent, the body it was in, the options it weighed with the policy's probabilities,
what it chose and how likely it rated that, what the choice leaned on, and its ideas (the imagined
future and what it expected of its body); the effect follows in the same tick's events. That is
the whole trace: thought, decision, effect, cause.

Every sentence comes from an event in events.jsonl (lifesim/recorder.py) and the causes it
carries, tick by tick: who gathered, drank, ate, slept, touched whom, and the notable (fights,
sickness, animals, weather, hunger and cold, mating, births, deaths) with its causes. Nothing is
left out; people doing the same thing at the same moment are named together. People are named for reading, by slot and life (the names are not the
agents' own; their own naming, if any, is in summary.json): founders in the order of their slots,
a later life in a slot gets the next unused name. Sex comes from lives.json when it is there.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

NAMES = ["Ash", "Bren", "Cael", "Dov", "Esk", "Fenn", "Gant", "Hael", "Ilse", "Jory", "Kesh", "Lir",
         "Mab", "Nol", "Oren", "Pim", "Quill", "Rook", "Sel", "Tam", "Ulf", "Vey", "Wren", "Yara", "Zed"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
PLANTS = {"roots": "roots", "berries": "berries", "nuts": "nuts", "grubs": "grubs", "honey": "honey",
          "fruit": "fruit", "leaves": "leaves", "mushrooms": "mushrooms", "tubers": "tubers"}
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]


def load_events(where: Path) -> list[dict]:
    if where.is_dir() and (where / "continuous").is_dir():
        where = where / "continuous"
    for name in ("events.jsonl", "events.jsonl.gz"):
        f = where / name if where.is_dir() else where
        if f.exists():
            op = gzip.open if f.suffix == ".gz" else open
            with op(f, "rt") as fh:
                return [json.loads(line) for line in fh if line.strip()]
    raise SystemExit(f"no events.jsonl in {where}")


def load_minds(folder: Path):
    """minds.jsonl.gz -> {t: {agent: record}}, or None when there is none."""
    f = folder / "minds.jsonl.gz"
    if not f.exists():
        f = folder / "minds.jsonl"
        if not f.exists():
            return None
    out = collections.defaultdict(dict)
    op = gzip.open if f.suffix == ".gz" else open
    with op(f, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["t"]][r["agent"]] = r
    return out


def parse_time(stamp: str):
    """'Y1 Spring D7 10:30' -> (year, season index, day, 'HH:MM')."""
    m = re.match(r"Y(\d+) (\w+) D(\d+) (\d\d:\d\d)", stamp)
    return int(m.group(1)), SEASONS.index(m.group(2)), int(m.group(3)), m.group(4)


class People:
    """Names by (slot, life); pronouns by sex when lives.json gives it."""

    def __init__(self, lives):
        self.names, self.sex, self.next = {}, {}, 0
        for life in lives or []:
            key = (life.get("slot", life.get("agent")), life.get("life", 1))
            if life.get("sex") in ("m", "f", "male", "female"):
                self.sex[key] = life["sex"][0]

    def name(self, slot, life=1):
        key = (slot, life)
        if key not in self.names:
            self.names[key] = NAMES[self.next % len(NAMES)] + ("" if self.next < len(NAMES) else f" {self.next // len(NAMES) + 1}")
            self.next += 1
        return self.names[key]

    def he(self, slot, life=1):
        return {"m": "he", "f": "she"}.get(self.sex.get((slot, life)), "they")

    def his(self, slot, life=1):
        return {"m": "his", "f": "her"}.get(self.sex.get((slot, life)), "their")

    def him(self, slot, life=1):
        return {"m": "him", "f": "her"}.get(self.sex.get((slot, life)), "them")

    def was(self, slot, life=1):
        return "were" if self.he(slot, life) == "they" else "was"


def clock_phrase(hhmm: str) -> str:
    h = int(hhmm[:2])
    if h < 4:
        return "in the small hours"
    if h < 7:
        return "at dawn"
    if h < 12:
        return "in the morning"
    if h < 14:
        return "at midday"
    if h < 18:
        return "in the afternoon"
    if h < 21:
        return "in the evening"
    return "at night"


def joined(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def plural(n, one, many=None):
    return f"{n} {one if n == 1 else (many or one + 's')}"


class Narrator:
    def __init__(self, events, lives=None, only_agent=None, minds=None):
        self.ev = events
        self.minds = minds
        self.by_id = {e["id"]: e for e in events}
        self.people = People(lives)
        self.only = only_agent
        self.stung = collections.Counter()
        self.sick_since = {}
        self.down_since = {}
        self.dead = {}
        self.pregnant = {}                 # slot -> father slot, until a birth or death
        self.last_attack = {}              # slot -> species of the last predator that caught them
        self.lives = {}                     # slot -> current life number seen
        self.weather = "clear"
        self.snap_since = None
        # founders: the agents alive at t = 0 (life 1), in slot order
        self.founders = sorted({e["agent"] for e in events if e.get("agent", -1) >= 0 and e.get("life", 1) == 1})
        for s in self.founders:
            self.people.name(s, 1)

    # ------------------------------------------------------------ helpers
    def who(self, e, slot=None, life=None):
        slot = e["agent"] if slot is None else slot
        life = e.get("life", 1) if life is None else life
        self.lives[slot] = max(self.lives.get(slot, 1), life)
        return self.people.name(slot, life)

    def other(self, slot):
        """Another agent by slot, at its current life (events name others by slot only)."""
        return self.people.name(slot, self.lives.get(slot, 1))

    def cause_text(self, e):
        return ", ".join(e.get("because") or [])

    def root_weather(self, e):
        for p in e.get("parents") or []:
            w = self.by_id.get(p)
            if w and w["type"] == "weather":
                return w
        return None

    # ------------------------------------------------------------ one event -> a sentence or None
    def tell(self, e) -> str | None:
        t, k = e["type"], e.get("agent", -1)
        if self.only is not None and k != self.only and self.only not in (e.get("victims") or []) \
                and self.only not in (e.get("targets") or []) and e.get("giver") != self.only and e.get("body") != self.only:
            return None
        P = self.people
        if k >= 0:
            n = self.who(e)
            he, his, him = P.he(k, e.get("life", 1)), P.his(k, e.get("life", 1)), P.him(k, e.get("life", 1))
        when = clock_phrase(e["time"][-5:])
        if t == "stung":
            self.stung[n] += 1
            c = self.stung[n]
            if c == 1:
                return f"{n} robbed a hive {when} and was stung for it."
            if c == 2:
                return f"{n} went back to the hive and was stung again."
            if c == 3:
                return f"{n} robbed a hive a third time. The bees were consistent."
            return f"{n} was stung robbing a hive, the {ORD[min(c - 1, 9)]} time."
        if t == "strike_agent":
            v = joined(self.other(x) for x in e.get("victims") or [])
            tech = (e.get("technique") or "blow").replace("useless hold (a head cannot grip)", "hold with the head, which cannot grip")
            back = " striking back" if any("striking back" in b for b in e.get("because") or []) else ""
            if "(broken)" in tech:
                return f"{n} got {v} in a {tech.replace(' (broken)', '')}{back}; {v} broke it."
            if "(moved them)" in tech:
                return f"{n} hit {v} with a {tech.replace(' (moved them)', '')} and shoved them back."
            if e.get("landed"):
                return f"{n} landed a {tech} on {v}{back}."
            return f"{n} tried a {tech} on {v}{back} and missed."
        if t == "struck_by_agent":
            return None                                   # told from the striker's side
        if t == "predator_attack":
            loss = e.get("health_loss", 0)
            sev = "a bad wound" if loss >= 0.2 else "a bite"
            self.last_attack[k] = e.get("species", "predator")
            return f"A {e.get('species', 'predator')} caught {n} {when}: {sev}, {loss:.0%} of {his} health."
        if t == "downed":
            why = self.cause_text(e)
            self.down_since[k] = e["t"]
            return f"{n} went down" + (f" from {why}" if why else "") + "."
        if t == "stood_up":
            since = self.down_since.pop(k, None)
            hrs = (e["t"] - since) / 2 if since is not None else None
            return f"{n} got up again" + (" soon after" if hrs is not None and hrs < 1 else (f" after {hrs:.0f} hours" if hrs else "")) + "."
        if t == "held_off_predator":
            return f"{n} stood {his} ground and held a predator off."
        if t == "strike_animal":
            return f"{n} struck at an animal."
        if t == "infected":
            why = self.cause_text(e).replace("the water it drank", "bad water")
            why = re.sub(r"sick #(\d+)", lambda m: "sick " + self.other(int(m.group(1))), why)
            self.sick_since[k] = e["t"]
            if why == "source not traced":
                return f"{n} fell sick; the source is not traced."
            return f"{n} fell sick" + (f" from {why}" if why else "") + "."
        if t == "recovered":
            since = self.sick_since.pop(k, None)
            days = (e["t"] - since) / 48 if since is not None else None
            return f"{n} got over the illness" + (f" after {days:.0f} days" if days and days >= 1 else "") + "."
        if t == "condition":
            c = e.get("condition")
            why = self.cause_text(e)
            if c == "haunted":
                return f"Nearly dying left {n} haunted."
            if c == "hardened":
                return f"Long hardship had hardened {n}."
            return f"{n} became {c}" + (f" ({why})" if why else "") + "."
        if t == "weather":
            w = e.get("weather")
            self.weather = w
            if w == "cold_snap":
                self.snap_since = e["t"]
                return "A cold snap set in."
            if w == "heat_wave":
                return "A heat wave began."
            if w == "drought":
                return "The rains failed: drought."
            if w == "clear":
                return "The weather broke and cleared."
            return f"The weather turned: {w}."
        if t == "freezing":
            air = e.get("air")
            why = self.cause_text(e)
            deg = None if air is None else int(round(air))
            return f"{n} was freezing {when}" + (f", the air at {deg} {'degree' if abs(deg) == 1 else 'degrees'}" if deg is not None else "") + "."
        if t == "starving":
            why = self.cause_text(e)
            return f"{n} was starving" + (f" ({why})" if why else "") + "."
        if t == "parched":
            why = self.cause_text(e)
            return f"{n} was parched" + (f" ({why})" if why else "") + "."
        if t == "fall":
            why = self.cause_text(e)
            return f"{n} fell" + (f" ({why})" if why else "") + "."
        if t == "gather_lost":
            return f"{n} lost what {he} {P.was(k, e.get('life', 1))} gathering."
        if t == "move_failed":
            return None
        if t == "took_food":
            g = self.other(e["giver"]) if e.get("giver") is not None else "someone"
            return f"{n} took food that {g} had left on the ground."
        if t == "tamed":
            an = e.get("animal", "animal")
            art = "An" if an[:1] in "aeiou" else "A"
            return f"{art} {an} that had been eating what {n} left behind came to be at ease with {him}."
        if t == "conceived":
            f = self.other(e["father"]) if e.get("father") is not None else None
            self.pregnant[k] = e.get("father")
            return f"{n} conceived" + (f"; the father was {f}" if f else "") + "."
        if t == "born":
            m = self.other(e["mother"]) if e.get("mother") is not None else None
            if e.get("mother") is not None:
                self.pregnant.pop(e["mother"], None)
            return f"{n} was born" + (f" to {m}" if m else "") + "."
        if t == "death":
            cause = e.get("cause", "unknown")
            self.dead[k] = (cause, e["time"])
            w = None
            for p in e.get("parents") or []:
                pe = self.by_id.get(p)
                if pe and pe["type"] == "freezing":
                    w = self.root_weather(pe)
            verbs = {"cold": "froze to death", "starvation": "starved to death", "thirst": "died of thirst",
                     "disease": "died of the sickness", "predation": "was killed by a predator",
                     "violence": "was killed", "injury": "died of {his} wounds", "age": "died of old age",
                     "burns": "burned to death", "fall": "died of a fall", "childbirth": "died in childbirth",
                     "heat": "died of the heat", "exhaustion": "died of exhaustion"}
            v = verbs.get(cause, f"died of {cause}").format(his=his)
            if cause == "predation" and k in self.last_attack:
                sp = self.last_attack[k]
                v = f"was killed by the {sp}"
            carrying = ""
            if k in self.pregnant:
                fa = self.pregnant.pop(k)
                carrying = f", carrying {self.other(fa) + chr(39) + 's' if fa is not None and fa >= 0 else 'a'} child,"
            extra = ""
            if w is not None and self.snap_since is not None:
                extra = f", {plural(round((e['t'] - w['t']) / 48), 'day')} into the cold snap"
            return f"{n}{carrying} {v} {when}{extra}."
        if t == "bones":
            b = e.get("body")
            return f"What was left of {self.other(b)} was bones now." if b is not None else None
        return None

    # ------------------------------------------------------------ routine, tick by tick
    ROUTINE_VERB = {"drink": "drank", "eat": "ate", "fell_asleep": "lay down to sleep", "woken": "woke",
                    "bedded": "bedded down on leaves", "release": "set down what {they} carried"}

    def routine_clauses(self, evs) -> list[str]:
        """The routine of one tick as clauses: people doing the same thing are named together."""
        groups = collections.OrderedDict()
        for e in evs:
            t = e["type"]
            if t == "gather":
                what = e.get("plant") or e.get("what") or "food"
                key = ("gather", {"dropped food": "picked up food others had dropped", "honey": "took honey",
                                  "wood": "gathered dead wood", "leaves": "gathered leaves", "clay": "dug clay",
                                  "nuts": "gathered nuts", "fruit": "picked fruit", "roots": "dug roots"}.get(what, f"gathered {what}"))
            elif t == "touch":
                key = ("touch", tuple(e.get("targets") or []))
            elif t == "mated":
                key = ("mated", e["t"])
            else:
                key = (t, None)
            groups.setdefault(key, []).append(e)
        out = []
        for (t, extra), es in groups.items():
            names = [self.who(e) for e in es]
            if t == "mated":
                if len(names) >= 2:
                    out.append(f"{joined(names)} lay together")
                continue
            if t == "touch":
                tg = joined(self.other(x) for x in extra) or "someone"
                out.append(f"{joined(names)} reached out to {tg}")
                continue
            if t == "gather":
                out.append(f"{joined(names)} {extra}")
                continue
            verb = self.ROUTINE_VERB.get(t, t)
            if len(names) == 1:
                k = es[0]["agent"]
                verb = verb.format(they=self.people.he(k, es[0].get("life", 1)))
            else:
                verb = verb.format(they="they")
            out.append(f"{joined(names)} {verb}")
        return out

    def thought(self, r) -> str:
        """One agent's decision this tick, as a line: the body, the options, the choice, the lean, the idea."""
        k = r["agent"]
        n = self.who({"agent": k, "life": self.lives.get(k, 1)})
        b = r["body"]
        state = f"food {b['food']:.2f} water {b['water']:.2f} energy {b['energy']:.2f} warmth {b['temp']:+.2f} health {b['health']:.2f}"
        flags = [w for w, c in (("hungry", b["food"] < 0.3), ("thirsty", b["water"] < 0.3), ("tired", b["energy"] < 0.2),
                                ("cold", b["temp"] <= -0.5), ("hot", b["temp"] >= 0.5), ("hurt", b["health"] < 0.4),
                                ("in pain", b["pain"] > 0.5), ("stressed", b["stress"] > 0.7)) if c]
        if r.get("asleep"):
            return f"    {n} slept on ({state})."
        opts = ", ".join(f"{a} {p:.0%}" for a, p in r.get("top", []))
        p = r.get("p")
        why = ", ".join(f"{g} {w:.0%}" for g, w in r.get("why", []))
        line = f"    {n}" + (f", {' and '.join(flags)}," if flags else "") + f" ({state}) weighed {opts}; chose {r['action']}" \
            + (f" ({p:.0%})" if p is not None else "") + f"; expected {r.get('value', 0):+.2f}"
        if why:
            line += f"; leaned on {why}"
        idea = r.get("idea")
        if idea:
            exp = ", ".join(f"{k_} {v:+.2f}" for k_, v in idea.get("expects", {}).items()) or "no change"
            line += f". Imagined: {idea['best']}" + (" (a new idea)" if idea.get("new") else "") \
                + f", then {' > '.join(idea.get('plan', [])[1:4])}; expecting {exp}; trust {idea.get('trust', 0):+.3f}"
        rolled = r.get("rolled")
        if rolled:
            parts = []
            for k_, v in rolled.items():
                if isinstance(v, list):                                   # an older record: [p, draw]
                    parts.append(f"P({k_}) = {v[0]:.1%}: {'it happened' if v[1] < v[0] else 'it did not'}")
                    continue
                why = v.get("why")
                why_s = ""
                if isinstance(why, dict):
                    bits = []
                    for wk, wv in why.items():
                        if isinstance(wv, bool):
                            if wv:
                                bits.append(wk)
                        elif wk.endswith(" x"):
                            if abs(wv - 1.0) > 1e-6:
                                bits.append(f"{wk[:-2]} x{wv:g}")
                        else:
                            bits.append(f"{wk} {wv:g}")
                    why_s = " (" + ", ".join(bits) + ")" if bits else ""
                elif isinstance(why, str):
                    why_s = f" ({why})"
                src = v.get("from")
                src_s = " from " + ", ".join(f"{a_} {b_:.1%}" for a_, b_ in src.items()) if src else ""
                out = "it happened" if v.get("happened") else ("not possible" if v.get("p", 0) == 0 else "it did not")
                parts.append(f"P({k_}) = {v.get('p', 0):.1%}{src_s}{why_s}: {out}")
            line += ". Odds: " + "; ".join(parts)
        return line + "."

    # ------------------------------------------------------------ the story
    def story(self, from_day=None, to_day=None) -> str:
        P = self.people
        out = []
        say = out.append
        founders = [P.name(s, 1) for s in self.founders]
        first, last = self.ev[0]["time"], self.ev[-1]["time"]
        say("THE BAND")
        say("")
        say(f"{len(founders)} of them at the start: {joined(founders)}. Names are for reading; the record calls them "
            + joined(f"#{s}" for s in self.founders) + ".")
        say(f"From {first} to {last}. Everything below is from the record, tick by tick (a tick is half an hour).")
        routine = {"gather", "drink", "eat", "fell_asleep", "bedded", "touch", "woken", "release", "mated"}
        seasons = collections.OrderedDict()
        def place(stamp, t, e=None):
            y, s, d, hhmm = parse_time(stamp)
            day_index = (y - 1) * 40 + s * 10 + d          # only for --from/--to; year length taken as 40 days
            if from_day is not None and day_index < from_day:
                return
            if to_day is not None and day_index > to_day:
                return
            bucket = seasons.setdefault((y, s), collections.OrderedDict()).setdefault(d, collections.OrderedDict()) \
                .setdefault(t, [])
            if e is not None:
                bucket.append(e)
        for e in self.ev:
            place(e["time"], e["t"], e)
        if self.minds is not None:
            for t, recs in self.minds.items():
                place(next(iter(recs.values()))["time"], t)
            for (y, s), days in seasons.items():
                for d in days:
                    days[d] = collections.OrderedDict(sorted(days[d].items()))
        for (y, s), days in seasons.items():
            say("")
            say(f"YEAR {y}, {SEASONS[s].upper()}")
            tally = collections.Counter()
            for d, ticks in days.items():
                say("")
                say(f"Day {d}.")
                for t, evs in ticks.items():
                    hhmm = (evs[0]["time"] if evs else next(iter(self.minds[t].values()))["time"])[-5:]
                    notable = [ln for ln in (self.tell(e) for e in evs if e["type"] not in routine) if ln]
                    rout = self.routine_clauses([e for e in evs if e["type"] in routine
                                                 and (self.only is None or e["agent"] == self.only
                                                      or self.only in (e.get("targets") or []))])
                    for e in evs:
                        tally[e["type"]] += 1
                    thoughts = []
                    if self.minds is not None:
                        for k, r in sorted(self.minds.get(t, {}).items()):
                            if self.only is None or k == self.only:
                                thoughts.append(self.thought(r))
                    if not notable and not rout and not thoughts:
                        continue
                    line = f"{hhmm}  "
                    if rout:
                        text = "; ".join(rout)
                        line += text[0].upper() + text[1:] + "."
                    if notable:
                        line += (" " if rout else "") + " ".join(notable)
                    if thoughts:
                        say(f"{hhmm}")
                        out.extend(thoughts)
                        if rout or notable:
                            say("  -> " + line[7:])
                    else:
                        say(line)
            say("")
            bits = []
            if tally["gather"]:
                bits.append(f"{tally['gather']} gatherings")
            for k, w in (("drink", "drinks"), ("eat", "meals"), ("fell_asleep", "nights' sleep"), ("touch", "touches"),
                         ("strike_agent", "blows struck"), ("infected", "infections"), ("predator_attack", "predator attacks"),
                         ("death", "deaths")):
                if tally[k]:
                    bits.append(f"{tally[k]} {w}")
            say(f"(The season in numbers: {joined(bits)}.)")
        say("")
        say("AFTER")
        say("")
        if self.dead:
            tal = collections.Counter(c for c, _ in self.dead.values())
            say("The dead: " + joined(f"{P.name(sl, self.lives.get(sl, 1))} ({c}, {tm})" for sl, (c, tm) in self.dead.items()) + ".")
            say("By cause: " + joined(f"{c} {k}" for k, c in tal.most_common()) + ".")
        alive = [P.name(sl, self.lives.get(sl, 1)) for sl in self.founders if sl not in self.dead]
        say(("Still alive at the end: " + joined(alive) + ".") if alive else "No one was left.")
        return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("where", help="a run's continuous/ folder, the run folder, or an events.jsonl")
    ap.add_argument("--out", default=None, help="write the story here (default: narrative.txt next to the events, "
                                                 "narrative_minds.txt with --minds; story.txt is causal_story's)")
    ap.add_argument("--from-day", type=int, default=None)
    ap.add_argument("--to-day", type=int, default=None)
    ap.add_argument("--agent", type=int, default=None, help="only what touches this slot")
    ap.add_argument("--minds", action="store_true", help="every decision and thought too (needs minds.jsonl.gz)")
    args = ap.parse_args()
    where = Path(args.where)
    events = load_events(where)
    folder = (where / "continuous" if (where / "continuous").is_dir() else where) if where.is_dir() else where.parent
    lives = json.load(open(folder / "lives.json")) if (folder / "lives.json").exists() else None
    minds = load_minds(folder) if args.minds else None
    if args.minds and minds is None:
        raise SystemExit(f"no minds.jsonl.gz in {folder}: run the continuous world without --no-minds")
    text = Narrator(events, lives, args.agent, minds).story(args.from_day, args.to_day)
    out = Path(args.out) if args.out else folder / ("narrative_minds.txt" if args.minds else "narrative.txt")
    out.write_text(text)
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
