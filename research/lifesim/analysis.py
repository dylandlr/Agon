"""Analysis of recorded behaviour: language, context and the emergence of names.

Works on playback data (scripts/export_playback.py), which holds per-tick
positions, messages, animals and fires. Agents never see the researcher-facing
names; a "name" here is something the agents invent: a message that different
speakers use around the same individual, and only around that individual.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


def mutual_information(xs, ys) -> float:
    """Plug-in mutual information in bits between two paired sequences."""
    n = len(xs)
    if n == 0:
        return 0.0
    pxy, px, py = Counter(zip(xs, ys)), Counter(xs), Counter(ys)
    return sum(c / n * math.log2(c * n / (px[x] * py[y])) for (x, y), c in pxy.items())


def mi_above_chance(xs, ys, shuffles: int = 50, seed: int = 0) -> float:
    """Mutual information minus its mean when ys are shuffled, in bits.

    The plug-in estimate is biased upward when there are many distinct values
    and few samples (it reads noise as information); subtracting the shuffled
    baseline removes most of that bias. About 0 means no real relation."""
    if len(xs) < 2:
        return 0.0
    rng = random.Random(seed)
    null = []
    for _ in range(shuffles):
        sh = list(ys)
        rng.shuffle(sh)
        null.append(mutual_information(xs, sh))
    return mutual_information(xs, ys) - sum(null) / len(null)


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def utterances(data: dict, radius: int | None = None) -> list[dict]:
    """One record per spoken message: speaker, message, its context and the
    nearest other living agent within view (the likeliest addressee or topic)."""
    radius = radius or data.get("view_radius", 5)
    out = []
    for f in data["frames"]:
        agents = f["agents"]
        for i, a in enumerate(agents):
            if not a["alive"] or a["msg"][0] == 0:
                continue
            me = (a["r"], a["c"])
            others = [(j, _cheb(me, (b["r"], b["c"]))) for j, b in enumerate(agents)
                      if j != i and b["alive"] and _cheb(me, (b["r"], b["c"])) <= radius]
            near = lambda cells: any(_cheb(me, tuple(p)) <= radius for p in cells)
            out.append({"t": f["t"], "speaker": i, "msg": tuple(a["msg"]),
                        "target": min(others, key=lambda x: x[1])[0] if others else None,
                        "predator": near(f["pred"]), "large_prey": near(f["large"]), "fire": near(f["fire"])})
    return out


def naming_index(records: list[dict], shuffles: int = 50, seed: int = 0, min_uses: int = 3,
                 specificity: float = 0.5) -> dict:
    """Do messages pick out individuals?

    index: MI(message; nearest agent) minus its mean under shuffled targets, in
    bits; about 0 when messages say nothing about who is near.
    names: messages that (a) are the most common thing at least two different
    speakers say near the same agent and (b) are used near that agent in at least
    `specificity` of all their uses.
    """
    recs = [r for r in records if r["target"] is not None]
    if len(recs) < 10:
        return {"utterances_with_target": len(recs), "index_bits": 0.0, "names": []}
    msgs, targets = [r["msg"] for r in recs], [r["target"] for r in recs]
    index = mi_above_chance(msgs, targets, shuffles, seed)
    by_target_speaker = defaultdict(Counter)
    for r in recs:
        by_target_speaker[(r["target"], r["speaker"])][r["msg"]] += 1
    msg_total = Counter(msgs)
    msg_target = Counter(zip(msgs, targets))
    names = []
    for target in sorted(set(targets)):
        modal = Counter()
        speakers = defaultdict(list)
        for (t, s), c in by_target_speaker.items():
            if t == target and sum(c.values()) >= min_uses:
                m = c.most_common(1)[0][0]
                modal[m] += 1
                speakers[m].append(s)
        for m, k in modal.items():
            spec = msg_target[(m, target)] / msg_total[m]
            if k >= 2 and spec >= specificity:
                names.append({"target": target, "message": list(m), "speakers": sorted(speakers[m]),
                              "specificity": round(spec, 2)})
    return {"utterances_with_target": len(recs), "index_bits": round(index, 4), "names": names}


def language_metrics(data: dict) -> dict:
    recs = utterances(data)
    alive_ticks = sum(a["alive"] for f in data["frames"] for a in f["agents"])
    msgs = [r["msg"] for r in recs]
    # Bias-corrected: raw information is inflated when nearly every message is unique
    ctx = {k: round(mi_above_chance(msgs, [r[k] for r in recs]), 4)
           for k in ("predator", "large_prey", "fire")}
    naming = naming_index(recs)
    from .names import names_over_time, spoken_form
    for nm in naming["names"]:
        nm["spoken"] = spoken_form(nm["message"])
    return {"utterances": len(recs), "speech_rate": round(len(recs) / max(alive_ticks, 1), 3),
            "distinct_messages": len(set(msgs)), "context_bits": ctx, "naming": naming,
            "names_given": names_over_time(recs)}
