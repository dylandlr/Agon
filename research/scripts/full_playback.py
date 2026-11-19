"""Record the full playback of the longest-lived continuous world among finished runs.

    python scripts/full_playback.py --root runs/exp                 # longest-lived of all runs
    python scripts/full_playback.py --run runs/exp/core/a0.0_f1.0_s1   # a chosen run

Experiment runs record a playback page only every few chunks, to save space. This finds the run
whose continuous world lasted longest (then most births, then latest generation), and lives that
world again with the same seed, recording every 5-day chunk, into <run>/full_playback/. Open
index.html there: every page links to the previous and next five days.

The same trained agents, world and seed give the same history up to floating-point differences
between machines, so on another GPU the replay can drift from the original summary over time.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def longest(root: Path):
    best = None
    for f in root.glob("*/*/continuous/summary.json"):
        S = json.load(open(f))
        if not S:
            continue
        last = S[-1]
        score = (len(S), last.get("births_so_far", 0), last.get("population", {}).get("max_generation") or 0)
        if best is None or score > best[0]:
            best = (score, f.parent.parent)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="experiment folder (searches */*/continuous)")
    ap.add_argument("--run", default=None, help="a specific trained run instead")
    ap.add_argument("--days", type=float, default=None, help="how long to record (default: as long as it lasted, up to 400)")
    ap.add_argument("--seed", type=int, default=21, help="the seed the experiment used (21)")
    args = ap.parse_args()
    if args.run:
        run = Path(args.run)
        S = json.load(open(run / "continuous" / "summary.json")) if (run / "continuous" / "summary.json").exists() else []
    else:
        found = longest(Path(args.root))
        if not found:
            raise SystemExit(f"no continuous worlds under {args.root}")
        (chunks, births, gen), run = found
        S = json.load(open(run / "continuous" / "summary.json"))
        print(f"longest-lived world: {run} ({5 * chunks} days, {births} births, generation {gen})", flush=True)
    days = args.days or (min(400, 5 * len(S) + 5) if S else 400)
    out = run / "full_playback"
    cmd = [sys.executable, str(HERE / "continuous_world.py"), "--run", str(run), "--days", str(days),
           "--chunk-days", "5", "--playback-every", "1", "--seed", str(args.seed), "--out-dir", str(out)]
    print("$ " + " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code == 0:
        print(f"full playback: {out / 'index.html'}", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
