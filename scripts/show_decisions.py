"""Read the decisions out of playback pages, as plain text.

A playback made from a policy (trained or untrained) records, for every agent every tick: the options
it weighed with the network's own probabilities, what it chose, what it leaned on, and, with a world
model, the ideas it imagined. This script reads them out of the page (a few KB of text instead of a
28 MB page), so they can be read anywhere, including from a notebook cell or a Drive folder.

  python scripts/show_decisions.py runs/x/continuous/chunk_000.html
  python scripts/show_decisions.py chunk_004.html --agent 3 --max 12 --out decisions.txt
  python scripts/show_decisions.py chunk_000.html chunk_004.html --tags thirsty,cold,predator

Three parts:
  1. Overview: how many decisions, how confident the policy was, how often it took an act it rated
     under 5% likely (a slip or an exploration).
  2. What agents chose when a need was pressing: the actual choices, by situation.
  3. Moments: a spread of pressing situations, each with the options, probabilities, what the choice
     leaned on, and the imagined plan with the expected change to its body.

Situations: thirsty (water < 0.3), hungry (food < 0.3), tired (energy < 0.2), cold (body temperature
<= -0.5), hurt (health < 0.4), predator (one within 3 cells), mating (chose to mate).
Pages made by scripted agents (no policy) have no decisions: the script says so.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

TAGS = ["thirsty", "hungry", "tired", "cold", "hurt", "predator", "mating"]


def load(path: Path) -> dict:
    text = path.read_text()
    marker = "const DATA = "
    i = text.index(marker) + len(marker)
    data, _ = json.JSONDecoder().raw_decode(text[i:])
    return data


def situations(a: dict, pred: list) -> list[str]:
    """Which pressing situations an agent is in this tick."""
    tags = []
    if a.get("water", 1.0) < 0.3:
        tags.append("thirsty")
    if a.get("food", 1.0) < 0.3:
        tags.append("hungry")
    if a.get("energy", 1.0) < 0.2:
        tags.append("tired")
    if a.get("temp", 0.0) <= -0.5:
        tags.append("cold")
    if a.get("health", 1.0) < 0.4:
        tags.append("hurt")
    # a predator is a threat if it is in the agent's layer and not tame to anyone (a tame one stands off).
    # The page does not record a predator's rest, so a sated one still counts: a slight over-count.
    if any(max(abs(p[0] - a["r"]), abs(p[1] - a["c"])) <= 3 and (len(p) < 4 or bool(p[3]) == bool(a.get("below", False))) and (len(p) < 5 or p[4] < 0)
           for p in pred):
        tags.append("predator")
    if a.get("action") == "mate":
        tags.append("mating")
    return tags


def line_for(frame: dict, i: int, a: dict, m: dict, tags: list[str]) -> str:
    out = [f"[{frame['time']}] #{i} at ({a['r']},{a['c']}) {'/'.join(tags) or 'ordinary'}   "
           f"food {a.get('food', 0):.2f} water {a.get('water', 0):.2f} energy {a.get('energy', 0):.2f} "
           f"temp {a.get('temp', 0):+.2f} health {a.get('health', 0):.2f}"]
    p = m.get("p")
    out.append(f"    chose {a.get('action')}" + (f" (the policy gave it {p:.0%})" if p is not None else "")
               + "; weighed " + ", ".join(f"{n} {q:.0%}" for n, q in m["top"]))
    why = ", ".join(f"{n} {q:.0%}" for n, q in m.get("why", []))
    out.append(f"    value {m['value']:+.2f}" + (f"; leaned on: {why}" if why else ""))
    idea = m.get("idea")
    if idea:
        opts = ", ".join(f"{n} (habit {h:.0%} -> {f:.0%}, imagined {q:+.2f})" for n, h, f, q in idea["options"][:4])
        exp = ", ".join(f"{k} {v:+.2f}" for k, v in idea["expects"].items()) or "no change"
        out.append(f"    ideas: {opts}")
        out.append(f"    best idea {idea['best']} (habit {idea['habit']})" + (" NEW" if idea.get("new") else "")
                   + f"; plan {' > '.join(idea['plan'])}; expects {exp}; trust {idea['trust']:+.4f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pages", nargs="+", help="playback .html files")
    ap.add_argument("--agent", type=int, default=None, help="only this agent")
    ap.add_argument("--tags", default=",".join(TAGS), help="situations to list moments for")
    ap.add_argument("--max", type=int, default=6, help="moments per situation per page")
    ap.add_argument("--out", type=str, default=None, help="also write the report to this file")
    args = ap.parse_args()
    want = [t for t in args.tags.split(",") if t in TAGS]

    lines: list[str] = []
    say = lines.append
    for page in map(Path, args.pages):
        data = load(page)
        frames = data["frames"]
        say(f"=== {page.name}: {data.get('source', '?')}; {len(frames)} frames, {frames[0]['time']} to {frames[-1]['time']}")
        if not any(f.get("minds") for f in frames):
            say("    no decisions recorded: these agents had no policy (scripted or random).\n")
            continue
        n = conf = slips = taken_top = 0
        chose = collections.defaultdict(collections.Counter)
        moments = collections.defaultdict(list)
        for f in frames:
            minds = f.get("minds")
            if not minds:
                continue
            for i, (a, m) in enumerate(zip(f["agents"], minds)):
                if not a.get("alive") or a.get("asleep") or (args.agent is not None and i != args.agent):
                    continue
                n += 1
                conf += m["top"][0][1]
                taken_top += a.get("action") == m["top"][0][0]
                slips += (m.get("p") is not None and m["p"] < 0.05)
                tags = situations(a, f.get("pred", []))
                chose["any"][a.get("action")] += 1
                for t in tags:
                    chose[t][a.get("action")] += 1
                    if t in want:
                        moments[t].append((f, i, a, m, tags))
        if not n:
            say("    no awake agents to read.\n")
            continue
        say(f"\n1. Overview ({n:,} decisions by awake agents{'' if args.agent is None else f', agent #{args.agent}'})")
        say(f"   mean probability of its top choice: {conf / n:.0%}; took its top choice: {taken_top / n:.0%}; "
            f"took an act it rated under 5%: {slips / n:.1%}")
        say("   most chosen: " + ", ".join(f"{k} {v / n:.0%}" for k, v in chose['any'].most_common(6)))
        say("\n2. What was chosen in each situation")
        for t in TAGS:
            if t in chose:
                tot = sum(chose[t].values())
                say(f"   {t:9s} ({tot:,} decisions): " + ", ".join(f"{k} {v / tot:.0%}" for k, v in chose[t].most_common(5)))
        say("\n3. Moments")
        for t in want:
            ms = moments.get(t, [])
            if not ms:
                continue
            say(f"\n  -- {t}: {len(ms):,} decisions; {min(args.max, len(ms))} shown, spread over the page")
            step = max(1, len(ms) // args.max)
            for f, i, a, m, tags in ms[::step][:args.max]:
                say(line_for(f, i, a, m, tags))
        say("")
    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
