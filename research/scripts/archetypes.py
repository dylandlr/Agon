"""Emergent archetypes: the roles agents actually fall into, found by clustering what each life did.

    python scripts/archetypes.py runs/exp/core               # every lives.json under these folders
    python scripts/archetypes.py runs/exp/validate/caveman --k 6

Each life recorded by continuous_world.py (lives.json) becomes a profile: how often it did each
behaviour per 1,000 ticks alive. Profiles are standardised and grouped with k-means; each group is
named after the behaviours it does most compared with everyone else. Then:
- injected archetypes (--archetypes in training) are checked against the group they landed in:
  did a designed "tyrant" end up fighting, or providing?
- each group's average temperament shows which feelings go with which roles.
Lives shorter than --min-ticks are left out (too short to show a role).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROLE_WORDS = {
    "strikes_on_agents": "fighter", "struck": "victim", "gifts": "giver", "touches": "toucher",
    "speech": "talker", "kills_small": "hunter", "kills_large": "big-game hunter",
    "predator_kills": "defender", "fire_fed": "fire-keeper", "drinks": "drinker", "eats": "eater",
    "gathers": "gatherer", "moves": "wanderer", "rests": "rester", "sleeps": "sleeper",
    "company": "companion", "matings": "mate-seeker", "nursing": "carer",
    "skill_water": "water-seeker", "skill_food": "forager", "skill_flee": "fleer", "skill_shelter": "shelterer",
}
TEMPERAMENT = ["empathy", "kin_care", "vengefulness", "guilt", "envy", "reciprocity", "selflessness",
               "social_reward", "novelty", "pain_aversion", "strength"]


def load(paths, min_ticks):
    lives = []
    for p in paths:
        p = Path(p)
        files = [p] if p.name == "lives.json" else sorted(p.rglob("lives.json"))
        for f in files:
            for life in json.load(open(f)):
                if life["ticks"] >= min_ticks:
                    life["run"] = str(f.parent.parent.name if f.parent.name == "continuous" else f.parent)
                    lives.append(life)
    return lives


def kmeans(x, k, restarts=20, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(restarts):
        centres = x[rng.choice(len(x), k, replace=False)]
        for _ in range(iters):
            label = np.argmin(((x[:, None] - centres[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[label == j].mean(0) if (label == j).any() else centres[j] for j in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((x - centres[label]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, label, centres)
    return best[1], best[2]


def analyse(lives, k):
    behaviours = list(lives[0]["counts"])
    rates = np.array([[1000 * l["counts"][b] / l["ticks"] for b in behaviours] for l in lives])
    keep = rates.std(0) > 1e-9                                  # behaviours nobody varies on say nothing
    behaviours = [b for b, ok in zip(behaviours, keep) if ok]
    rates = rates[:, keep]
    z = (rates - rates.mean(0)) / rates.std(0)
    k = min(k, len(lives))
    label, centres = kmeans(z, k)
    groups = []
    for j in np.argsort(-np.bincount(label, minlength=k)):
        members = [lives[i] for i in np.flatnonzero(label == j)]
        if not members:
            continue
        top = np.argsort(-centres[j])[:3]
        name = " / ".join(ROLE_WORDS.get(behaviours[t], behaviours[t]) for t in top if centres[j][t] > 0.3) \
            or "ordinary"
        temperament = {t: round(float(np.mean([m["traits"].get(t, np.nan) for m in members])), 2)
                       for t in TEMPERAMENT if any(t in m["traits"] for m in members)}
        groups.append({"group": len(groups), "role": name, "lives": len(members),
                       "share": round(len(members) / len(lives), 3),
                       "distinctive": {behaviours[t]: round(float(centres[j][t]), 2) for t in top},
                       "mean_ticks_lived": round(float(np.mean([m["ticks"] for m in members])), 1),
                       "temperament": temperament, "_label": int(j)})
    by_label = {g["_label"]: g for g in groups}
    injected = [{"run": l["run"], "slot": l["slot"], "archetype": l["archetype"],
                 "landed_in": by_label[int(label[i])]["role"], "ticks_lived": l["ticks"],
                 "cause_of_death": l.get("cause")}
                for i, l in enumerate(lives) if l.get("archetype")]
    for g in groups:
        g.pop("_label")
    return {"lives": len(lives), "groups": groups, "injected": injected}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--k", type=int, default=5, help="number of roles to look for")
    ap.add_argument("--min-ticks", type=int, default=96, help="shortest life counted (96 = 2 days)")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args()
    lives = load(args.paths, args.min_ticks)
    if len(lives) < 2:
        sys.exit(f"only {len(lives)} lives of at least {args.min_ticks} ticks found; nothing to group")
    res = analyse(lives, args.k)
    print(f"{res['lives']} lives grouped into {len(res['groups'])} roles by what they did")
    print(f"{'role':<38}{'lives':>6}{'share':>7}{'lived':>7}  temperament (mean)")
    for g in res["groups"]:
        t = g["temperament"]
        print(f"{g['role']:<38}{g['lives']:>6}{g['share']:>7.0%}{g['mean_ticks_lived']:>7.0f}  "
              + ", ".join(f"{k} {v}" for k, v in t.items() if k in ("empathy", "vengefulness", "selflessness",
                                                                   "social_reward", "kin_care")))
    if res["injected"]:
        print("\ninjected archetypes and the role each actually took")
        for r in res["injected"]:
            print(f"  {r['archetype']:<10} ({r['run']}, slot {r['slot']}): {r['landed_in']}; "
                  f"lived {r['ticks_lived']} ticks{'; died of ' + r['cause_of_death'] if r['cause_of_death'] else ''}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
