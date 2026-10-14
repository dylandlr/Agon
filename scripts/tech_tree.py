"""The technology tree of a run, and which personalities do which skills.

    python scripts/tech_tree.py runs/exp/validate/caveman              # finds continuous/
    python scripts/tech_tree.py runs/exp/inheritance/innate_s21 --json tech.json

Part 1, the tree: every building block of know-how (lifesim/techtree.py) with whether it was ever
done, first by whom and when, how many individuals and days it spread to, and whether it was
established (several individuals or days) or lost again. Blocks are a measuring lens: agents
are never shown them.

Part 2, skills by personality: for every life of at least --min-days, how often it did each
block (per 1,000 ticks alive), against each inborn personality trait. Spearman rank correlation,
a two-sided permutation p-value (--perms shuffles), and Benjamini-Hochberg false-discovery
correction across all block x trait pairs. Reported: pairs with q below --q, then the strongest
correlations regardless, so an empty result is visible as such. Correlation, not cause: a trait
can go with a skill because both go with living longer, or with a lineage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from lifesim import techtree  # noqa: E402
from causal_story import load_events  # noqa: E402


def _ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def spearman(x, y):
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def perm_p(x, y, rho, perms, rng):
    if rho == 0.0:
        return 1.0
    y = np.asarray(y, float)
    hits = sum(abs(spearman(x, rng.permutation(y))) >= abs(rho) - 1e-12 for _ in range(perms))
    return (hits + 1) / (perms + 1)


def bh(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    q = np.empty(len(p))
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * len(p) / rank)
        q[i] = prev
    return q


def personality(lives, min_ticks, perms, q_max, seed=0):
    lives = [l for l in lives if l["ticks"] >= min_ticks]
    if len(lives) < 8:
        return [f"Only {len(lives)} lives of at least {min_ticks} ticks: too few to relate skills to personality."], []
    traits = sorted({t for l in lives for t in l["traits"]})
    rows, rng = [], np.random.default_rng(seed)
    for b in techtree.IDS:
        rate = np.array([1000.0 * l.get("skills", {}).get(b, 0) / l["ticks"] for l in lives])
        if (rate > 0).sum() < 3:
            continue
        for t in traits:
            x = np.array([l["traits"].get(t, np.nan) for l in lives])
            ok = ~np.isnan(x)
            if ok.sum() < 8 or np.ptp(x[ok]) == 0:
                continue
            rho = spearman(x[ok], rate[ok])
            rows.append([b, t, rho, perm_p(x[ok], rate[ok], rho, perms, rng), int(ok.sum())])
    if not rows:
        return ["No skill was done by 3 or more lives: nothing to relate to personality yet."], []
    q = bh([r[3] for r in rows])
    for r, qq in zip(rows, q):
        r.append(float(qq))
    name = lambda b: techtree.BY_ID[b][1]
    out = [f"{len(lives)} lives of at least {min_ticks} ticks; {len(rows)} skill x trait pairs tested "
           f"(Spearman rho, permutation p with {perms} shuffles, Benjamini-Hochberg q)."]
    sig = sorted([r for r in rows if r[5] < q_max], key=lambda r: r[5])
    out.append(f"Associations with q < {q_max}: {len(sig)}")
    out += [f"  {name(b)} ~ {t.replace('_', ' ')}: rho {rho:+.2f}, p {p:.3f}, q {qq:.3f} (n {n})"
            for b, t, rho, p, n, qq in sig]
    out.append("Strongest correlations (whether or not they pass):")
    out += [f"  {name(b)} ~ {t.replace('_', ' ')}: rho {rho:+.2f}, p {p:.3f}, q {qq:.3f} (n {n})"
            for b, t, rho, p, n, qq in sorted(rows, key=lambda r: -abs(r[2]))[:8]]
    return out, rows


def report(folder: Path, min_days=2.0, perms=999, q_max=0.1, day_length=48):
    folder = folder / "continuous" if (folder / "continuous").exists() else folder
    events = load_events(folder)
    tree = techtree.tree(events, day_length)
    lines = ["Technology tree (* established, o discovered, x lost, blank not yet):"] + techtree.render(tree)
    lives_f = folder / "lives.json"
    rows = []
    if lives_f.exists():
        lines += ["", "Skills by personality:"]
        text, rows = personality(json.load(open(lives_f)), int(min_days * day_length), perms, q_max)
        lines += text
    return lines, {"tree": tree, "skill_personality": [dict(zip(["skill", "trait", "rho", "p", "n", "q"], r))
                                                       for r in rows]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a run folder or its continuous/ folder")
    ap.add_argument("--min-days", type=float, default=2.0, help="shortest life counted")
    ap.add_argument("--perms", type=int, default=999)
    ap.add_argument("--q", type=float, default=0.1, help="false-discovery threshold")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()
    lines, res = report(Path(args.folder), args.min_days, args.perms, args.q)
    print("\n".join(lines))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
