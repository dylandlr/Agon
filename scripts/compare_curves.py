"""Compare learning curves of training runs at matched agent-step counts.

    python scripts/compare_curves.py runs/fixed_run runs/long_run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = [
    ("lifespan", lambda r: r["mean_age_at_death"]),
    ("company", lambda r: r.get("in_company_share")),
    ("thirst", lambda r: r["deaths"].get("thirst", 0)),
    ("starve", lambda r: r["deaths"].get("starvation", 0)),
    ("violent", lambda r: r["deaths"].get("violence", 0)),
    ("strikes/k", lambda r: r["per_1000_agent_ticks"]["strikes_on_agents"]),
    ("gifts/k", lambda r: r["per_1000_agent_ticks"]["gifts"]),
    ("touch/k", lambda r: r["per_1000_agent_ticks"]["touches"]),
    ("speech/k", lambda r: r["per_1000_agent_ticks"].get("messages")),
    ("deter/k", lambda r: r.get("deterred_per_1000")),
    ("small", lambda r: r["kills"]["small_prey"]),
    ("large", lambda r: r["kills"]["large_prey"]),
    ("watched", lambda r: r.get("watched_fraction")),
]


def load(path):
    return {r["agent_steps"]: r for r in json.load(open(Path(path) / "history.json"))["history"]}


def fmt(v):
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else f"{v:.0f}" if isinstance(v, (int, float)) else str(v)


def main(paths):
    runs = {Path(p).name: load(p) for p in paths}
    steps = sorted(set.union(*[set(r) for r in runs.values()]))
    print(f"{'steps':>9} {'run':>12} " + " ".join(f"{n:>9}" for n, _ in COLUMNS))
    for st in steps:
        for name, hist in runs.items():
            if st in hist:
                r = hist[st]
                print(f"{st:>9} {name:>12} " + " ".join(f"{fmt(f(r)):>9}" for _, f in COLUMNS))


if __name__ == "__main__":
    main(sys.argv[1:])
