"""Hierarchy in a continuous world: who dominates whom, whether it forms a ranking, whether rank
lasts and pays, and who leads.

    python scripts/hierarchy.py runs/exp_<v>/validate/caveman            # finds continuous/
    python scripts/hierarchy.py <world folder> --window-days 40 --json h.json

Nothing here feeds back into the world; agents are never told about rank.

Dominance, from two kinds of decided encounters (the standard ones in primatology):
- blows: i strikes j (events.jsonl.gz, strike_agent).
- supplants (needs trace.npz): i moves next to j, and on the next tick j moves away while i stays
  (approach-retreat).
Per window (default one 40-day year) among individuals with at least one encounter:
- Normalised David's scores (de Vries et al. 2006, dyadic win proportions corrected for the
  number of encounters): each individual's rank, 0..N-1.
- Steepness: the slope of normalised David's scores against rank (0 = all equal, 1 = steepest).
- Linearity: Landau's h' corrected for unknown and tied relationships (de Vries 1995), with a
  p-value from 10,000 randomised dominance relations (h' expected about 3/(N+1) by chance).
- Despotism: the share of all blows struck by the top-ranked individual.
Across windows: stability, Spearman's rho of the scores of individuals present in consecutive
windows. Over the whole run: rank against eating, matings, children, food taken from others' leavings
and lifespan (Spearman, permutation p).

Leadership (needs trace.npz): i leads j when i moves one way and j, within 3 cells, moves the same
way in the next 2 ticks (not at the same time). Per individual, net leadership = (led - followed)
/ (led + followed), against dominance and age. A ranking can exist without leaders, and leaders
without force: the two are reported apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from society_data import World, perm_p, spearman  # noqa: E402


# ------------------------------------------------------------------ dominance statistics

def layered(trace):
    """(T, N, 2) positions, with those underground (trace 'below', when the world had caves) moved
    far off, so nobody above is ever beside, supplanting or following someone below."""
    pos = trace["pos"].astype(int)
    if "below" in trace:
        pos = pos.copy()
        pos[..., 1] += 4096 * trace["below"].astype(int)
    return pos

def david_scores(W):
    """Normalised David's scores from a win matrix W[i, j] (i beat j), dyadic-corrected."""
    W = np.asarray(W, float)
    n = len(W)
    tot = W + W.T
    P = np.where(tot > 0, W / np.maximum(tot, 1e-12), 0.0)
    P = np.where(tot > 0, P - (P - 0.5) / (tot + 1), 0.0)            # de Vries et al. 2006
    w = P.sum(1)
    l = P.sum(0)
    w2 = P @ w
    l2 = P.T @ l
    ds = w + w2 - l - l2
    return (ds + n * (n - 1) / 2) / n


def steepness(nds):
    """Slope (absolute) of normalised David's scores sorted high to low against rank 1..N."""
    y = np.sort(np.asarray(nds))[::-1]
    if len(y) < 3:
        return float("nan")
    x = np.arange(1, len(y) + 1)
    return float(abs(np.polyfit(x, y, 1)[0]))


def landau_h(W, rng=None, reps=10000):
    """(h', p): Landau's linearity corrected for unknown and tied relationships (de Vries 1995)."""
    W = np.asarray(W, float)
    n = len(W)
    if n < 3:
        return float("nan"), float("nan")
    iu = np.triu_indices(n, 1)
    a, b = W[iu], W.T[iu]
    known = (a + b) > 0
    # dominance per dyad: 1 to the winner, 0.5 each when tied, 0 each when unknown
    d_ij = np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0)) * known
    d_ji = np.where(b > a, 1.0, np.where(a == b, 0.5, 0.0)) * known
    V = np.zeros(n)
    np.add.at(V, iu[0], d_ij)
    np.add.at(V, iu[1], d_ji)
    k = 12.0 / (n ** 3 - n)
    h = k * ((V - (n - 1) / 2) ** 2).sum()
    u = (~known).sum()
    h_corr = h + 6.0 * u / (n ** 3 - n)
    rng = rng or np.random.default_rng(0)
    # randomised relations: every dyad's dominance to one or the other at random (de Vries 1995)
    draws = rng.random((reps, len(iu[0]))) < 0.5
    Vr = np.zeros((reps, n))
    np.add.at(Vr, (slice(None), iu[0]), draws.astype(float))
    np.add.at(Vr, (slice(None), iu[1]), (~draws).astype(float))
    hr = k * ((Vr - (n - 1) / 2) ** 2).sum(1)
    p = float(((hr >= h_corr - 1e-12).sum() + 1) / (reps + 1))
    return float(h_corr), p


# ------------------------------------------------------------------ encounters
def encounters(world: World, supplants=True):
    """List of (t, winner life id, loser life id, kind)."""
    out = []
    for e in world.events:
        if e.get("type") == "strike_agent":
            a = world.life_at(e["agent"], e["t"])
            for v in e.get("victims", []):
                b = world.life_at(v, e["t"])
                if a is not None and b is not None and a["id"] != b["id"]:
                    out.append((e["t"], a["id"], b["id"], "blow"))
    if supplants and world.trace is not None:
        pos = layered(world.trace)
        ids = world.life_ids_grid()
        T, N = ids.shape
        for t in range(1, T - 1):
            moved = (pos[t] != pos[t - 1]).any(-1) & (ids[t] >= 0)
            stay_next = ~(pos[t + 1] != pos[t]).any(-1)
            d_now = np.abs(pos[t][:, None] - pos[t][None]).max(-1)
            d_before = np.abs(pos[t - 1][:, None] - pos[t - 1][None]).max(-1)
            d_next = np.abs(pos[t + 1][:, None] - pos[t + 1][None]).max(-1)
            j_moved = (pos[t + 1] != pos[t]).any(-1)
            cand = moved[:, None] & (d_now <= 1) & (d_before > 1) & stay_next[:, None] & j_moved[None, :] \
                & (d_next > 1) & (ids[t][:, None] >= 0) & (ids[t][None, :] >= 0) & (ids[t + 1][None, :] == ids[t][None, :])
            np.fill_diagonal(cand, False)
            for i, j in zip(*np.nonzero(cand)):
                out.append((t, int(ids[t, i]), int(ids[t, j]), "supplant"))
    return out


def win_matrix(enc, members):
    idx = {m: k for k, m in enumerate(members)}
    W = np.zeros((len(members), len(members)))
    for _, a, b, _ in enc:
        if a in idx and b in idx:
            W[idx[a], idx[b]] += 1
    return W


def leadership(world: World, radius=3, lag=2):
    """Per life id: (times it led, times it followed)."""
    if world.trace is None:
        return {}
    pos = layered(world.trace)
    ids = world.life_ids_grid()
    T, N = ids.shape
    step = np.zeros((T, N, 2), int)
    step[1:] = pos[1:] - pos[:-1]
    moving = (step != 0).any(-1)
    led, followed = defaultdict(int), defaultdict(int)
    for t in range(1, T - lag):
        here = ids[t] >= 0
        if not (moving[t] & here).any():
            continue
        near = np.abs(pos[t][:, None] - pos[t][None]).max(-1) <= radius
        same_now = moving[t][None, :] & (step[t][:, None, :] == step[t][None, :, :]).all(-1)
        later = np.zeros((N, N), bool)
        for k in range(1, lag + 1):
            later |= (step[t + k][None, :, :] == step[t][:, None, :]).all(-1) & moving[t + k][None, :] \
                & (ids[t + k] == ids[t])[None, :]
        m = (moving[t] & here)[:, None] & here[None, :] & near & ~same_now & later
        np.fill_diagonal(m, False)
        for i, j in zip(*np.nonzero(m)):
            led[int(ids[t, i])] += 1
            followed[int(ids[t, j])] += 1
    return {k: (led.get(k, 0), followed.get(k, 0)) for k in set(led) | set(followed)}


# ------------------------------------------------------------------ report
def analyse(world: World, window_days=40.0, day_length=None, min_encounters=1, seed=0):
    rng = np.random.default_rng(seed)
    enc = encounters(world)
    lives = {l["id"]: l for l in world.lives}
    day_length = day_length or world.day_length
    W_ticks = int(window_days * day_length)
    windows = []
    n_win = max(1, int(np.ceil((world.t_end + 1) / W_ticks)))
    for k in range(n_win):
        lo, hi = k * W_ticks, (k + 1) * W_ticks
        e = [x for x in enc if lo <= x[0] < hi]
        members = sorted({x[1] for x in e} | {x[2] for x in e})
        if len(members) < 3:
            windows.append({"window": k, "from_day": lo / day_length, "individuals": len(members),
                            "encounters": len(e)})
            continue
        W = win_matrix(e, members)
        nds = david_scores(W)
        h, p = landau_h(W, rng)
        blows = [x for x in e if x[3] == "blow"]
        top = members[int(np.argmax(nds))]
        windows.append({"window": k, "from_day": lo / day_length, "individuals": len(members),
                        "encounters": len(e), "blows": len(blows), "supplants": len(e) - len(blows),
                        "steepness": round(steepness(nds), 3), "linearity_h": round(h, 3), "linearity_p": round(p, 4),
                        "despotism": round(sum(1 for x in blows if x[1] == top) / len(blows), 3) if blows else None,
                        "scores": {int(m): round(float(s), 3) for m, s in zip(members, nds)}})
    stab = []
    for a, b in zip(windows, windows[1:]):
        if "scores" in a and "scores" in b:
            both = sorted(set(a["scores"]) & set(b["scores"]))
            if len(both) >= 4:
                stab.append(spearman([a["scores"][m] for m in both], [b["scores"][m] for m in both]))
    # whole run: rank against outcomes
    members = sorted({x[1] for x in enc} | {x[2] for x in enc})
    outcomes = {}
    whole = {}
    if len(members) >= 5:
        nds = david_scores(win_matrix(enc, members))
        whole = {int(m): float(s) for m, s in zip(members, nds)}
        kids = world.children()
        taken = defaultdict(int)
        for e in world.events:
            if e.get("type") == "took_food":
                l = world.life_at(e["agent"], e["t"])
                if l is not None:
                    taken[l["id"]] += 1
        per_k = lambda l, x: 1000.0 * x / max(l["ticks"], 1)
        meas = {"eats per 1000 ticks": lambda l: per_k(l, l["counts"]["eats"]),
                "matings per 1000 ticks": lambda l: per_k(l, l["counts"]["matings"]),
                "children": lambda l: kids[l["id"]],
                "food taken from others' leavings per 1000 ticks": lambda l: per_k(l, taken[l["id"]]),
                "lifespan (ticks)": lambda l: l["ticks"]}
        x = np.array([whole[m] for m in members])
        for name, f in meas.items():
            y = np.array([f(lives[m]) for m in members], float)
            if np.ptp(y) > 0:
                rho, p = perm_p(x, y, spearman, 999, seed)
                outcomes[name] = {"rho": round(rho, 3), "p": round(p, 4), "n": len(members)}
    lead = leadership(world)
    lead_rows = {}
    for m, (a, b) in lead.items():
        if a + b >= 5:
            lead_rows[m] = round((a - b) / (a + b), 3)
    lead_vs = {}
    common = sorted(set(lead_rows) & set(whole))
    if len(common) >= 5:
        rho, p = perm_p(np.array([whole[m] for m in common]), np.array([lead_rows[m] for m in common]), spearman, 999, seed)
        lead_vs["dominance"] = {"rho": round(rho, 3), "p": round(p, 4), "n": len(common)}
    if len(lead_rows) >= 5:
        ages = np.array([lives[m]["ticks"] for m in lead_rows], float)
        rho, p = perm_p(ages, np.array(list(lead_rows.values())), spearman, 999, seed)
        lead_vs["time lived"] = {"rho": round(rho, 3), "p": round(p, 4), "n": len(lead_rows)}
    return {"world": world.name, "encounters": len(enc), "blows": sum(1 for x in enc if x[3] == "blow"),
            "supplants": sum(1 for x in enc if x[3] == "supplant"), "windows": windows,
            "stability": [round(s, 3) for s in stab], "rank_outcomes": outcomes,
            "leadership": {"net_by_life": {int(k): v for k, v in lead_rows.items()}, "against": lead_vs},
            "has_trace": world.trace is not None}


def render(r):
    out = [f"Hierarchy: {r['world']}",
           f"{r['encounters']} decided encounters: {r['blows']} blows, {r['supplants']} supplants"
           + ("" if r["has_trace"] else " (no trace.npz: blows only, no leadership)")]
    for w in r["windows"]:
        head = f"  day {w['from_day']:.0f}+: {w['individuals']} individuals, {w['encounters']} encounters"
        if "steepness" not in w:
            out.append(head + " (too few to rank)")
            continue
        out.append(head + f"; steepness {w['steepness']}, linearity h' {w['linearity_h']} (p {w['linearity_p']}), "
                   f"top individual struck {w['despotism'] if w['despotism'] is not None else 'n/a'} of blows")
    if r["stability"]:
        out.append(f"Rank stability between consecutive windows (Spearman): {', '.join(map(str, r['stability']))}")
    if r["rank_outcomes"]:
        out.append("Rank against outcomes, whole run (Spearman, permutation p):")
        out += [f"  {k}: rho {v['rho']:+.2f}, p {v['p']} (n {v['n']})" for k, v in r["rank_outcomes"].items()]
    lv = r["leadership"]
    if lv["net_by_life"]:
        top = sorted(lv["net_by_life"].items(), key=lambda kv: -kv[1])[:5]
        out.append(f"Leadership: {len(lv['net_by_life'])} individuals with 5+ lead/follow moves; most followed: "
                   + ", ".join(f"life {k} ({v:+.2f})" for k, v in top))
        out += [f"  net leadership against {k}: rho {v['rho']:+.2f}, p {v['p']} (n {v['n']})" for k, v in lv["against"].items()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="+", help="continuous world folders (or run folders containing continuous/)")
    ap.add_argument("--window-days", type=float, default=40.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = []
    for f in args.folder:
        r = analyse(World(f), args.window_days)
        res.append(r)
        print("\n".join(render(r)) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
