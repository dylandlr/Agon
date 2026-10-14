"""Compare finished training runs, grouped by every setting except the seed.

    python scripts/compare_runs.py runs/core/*            # any train_phase2 output folders

Each run's final 20% of logged windows is averaged; runs that share all settings
but the seed form one condition, reported as mean and 95% interval across seeds.
Runs that differ in anything else (alpha, food, social learning, grid size, ...)
stay in separate rows, so they are never averaged together by accident.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

METRICS = {
    "lifespan": lambda r: r["mean_age_at_death"],
    "alive_at_end": lambda r: r.get("share_alive_at_episode_end"),
    "survival_score": lambda r: r.get("survival_score_per_world_day"),
    "global_reward": lambda r: r.get("global_reward_per_world_day"),
    "large_prey_kills": lambda r: r["kills"]["large_prey"],
    "deterred_per_1000": lambda r: r.get("deterred_per_1000"),
    "winter_huddle": lambda r: r.get("winter_huddle_share"),
    "gifts_per_1000": lambda r: r["per_1000_agent_ticks"]["gifts"],
    "strikes_per_1000": lambda r: r["per_1000_agent_ticks"]["strikes_on_agents"],
    "hunting_gini": lambda r: r.get("hunting_gini"),
    "winter_death_share": lambda r: (r["deaths_by_season"]["winter"] / max(1, sum(r["deaths_by_season"].values()))
                                     if "deaths_by_season" in r else None),
    "thirst_death_share": lambda r: (r["deaths"].get("thirst", 0) / max(1, sum(r["deaths"].values()))
                                     if "deaths" in r else None),
    "births": lambda r: r.get("births"),
    "matings_per_1000": lambda r: r.get("matings_per_1000"),
    "company": lambda r: r.get("in_company_share"),
    "asleep_night": lambda r: r.get("asleep_share_night"),
    "asleep_day": lambda r: r.get("asleep_share_day"),
    "learned_share": lambda r: r.get("learned_share"),
}
# Settings that do not define a condition
IGNORE = {"seed", "out", "resume", "log_every", "ckpt_every", "ticks", "allow_code_change", "preset"}


def summarize(folder: Path):
    h = json.load(open(folder / "history.json"))
    rows = h["history"][-max(1, len(h["history"]) // 5):]
    vals = {}
    for k, f in METRICS.items():
        xs = [f(r) for r in rows if f(r) is not None]
        vals[k] = sum(xs) / len(xs) if xs else float("nan")
    cond = tuple(sorted((k, v) for k, v in h["args"].items() if k not in IGNORE))
    return cond, vals


def main(paths):
    groups = defaultdict(list)
    for p in paths:
        if (Path(p) / "history.json").exists():
            cond, vals = summarize(Path(p))
            groups[cond].append(vals)
    if not groups:
        sys.exit("no runs with history.json found")
    # Show only the settings that differ between conditions
    keys = sorted({k for c in groups for k, _ in c})
    varying = [k for k in keys if len({dict(c).get(k) for c in groups}) > 1] or ["alpha", "scarcity"]
    head = " ".join(f"{k:>10}" for k in varying)
    print(f"{head} {'seeds':>5}  " + "  ".join(f"{m:>18}" for m in METRICS))
    for cond, runs in sorted(groups.items(), key=lambda kv: [str(dict(kv[0]).get(k)) for k in varying]):
        d = dict(cond)
        cells = []
        for m in METRICS:
            xs = [r[m] for r in runs if not math.isnan(r[m])]
            if not xs:
                cells.append(f"{'n/a':>18}")
                continue
            mean = sum(xs) / len(xs)
            if len(xs) > 1:
                ci = 1.96 * (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 / len(xs) ** 0.5
                cells.append(f"{mean:>10.3f} +/- {ci:<4.2f}")
            else:
                cells.append(f"{mean:>18.3f}")
        print(" ".join(f"{str(d.get(k)):>10}" for k in varying) + f" {len(runs):>5}  " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
