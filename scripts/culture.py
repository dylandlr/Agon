"""Culture across worlds: behaviour that differs between groups living in the same ecology,
persists across generations, and is not explained by genes; and dialects of the agents' speech.

    python scripts/culture.py runs/exp_<v>/culture/social_* --json culture.json
    python scripts/culture.py runs/exp_<v>/culture/social_* --against runs/exp_<v>/culture/solitary_*

The test follows the field studies of animal cultures (Whiten et al. 1999; Whitehead and Rendell
2015): with ecology held fixed (the `culture` experiment gives every world the same map and
founders, and differs only in what happens), behaviour that differs between groups, is passed on
to those born later, and remains after accounting for genes, is a candidate tradition. With
--against, the same is computed for worlds whose agents cannot learn by watching: differences
that persist there are not social.

Per life (alive at least --min-days): a behaviour profile, rates per 1,000 ticks of each counted
behaviour and technology block, and from trace.npz the share of ticks spent on each kind of action.
1. Differentiation: a permutational ANOVA (PERMANOVA, Anderson 2001) on standardised profiles with
   worlds as groups: pseudo-F, R^2 (share of variation between worlds) and a permutation p; the same
   on profiles with inherited traits regressed out ("beyond genes"); and how often a life is assigned
   to its own world by the nearest world mean (leave-one-out), against chance.
2. Traditions: behaviours customary in some worlds (done by at least half of the lives there) and
   absent in others (done by at most 5%).
3. Persistence: does a world's distinctive profile among its earlier lives (founders and the first
   born) recur among its later-born lives? Mean correlation of each world's early and late
   deviation from the all-world mean, own world minus other worlds, with a permutation p.
4. Dialects (from trace.npz): how far worlds' messages differ (Jensen-Shannon divergence, at equal
   sample sizes) beyond how far one world's messages differ between its first and second half;
   vocabulary overlap; each world's most typical message where a predator, large prey or fire is near.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from society_data import World  # noqa: E402

# Action groups for the action profile (lifesim/env.py layout)
def action_groups():
    from lifesim import env as E
    g = {"wait": [E.REST], "move": list(range(E.MOVE_N, E.MOVE_N + 8)), "grasp": [E.GRASP], "eat": [E.EAT],
         "drink": [E.DRINK], "strike": list(range(E.STRIKE_N, E.STRIKE_N + 8)),
         "select": list(range(E.SELECT_0, E.SELECT_0 + E.N_MAT)), "place": list(range(E.PLACE_N, E.PLACE_N + 8)),
         "mate": [E.MATE], "sleep": [E.SLEEP],
         "seek_water": [E.SEEK_WATER], "seek_food": [E.SEEK_FOOD], "flee": [E.FLEE], "seek_shelter": [E.SEEK_SHELTER],
         "delve": [E.DELVE]}
    for name in ("RELEASE", "TOUCH_N"):
        if hasattr(E, name):
            base = getattr(E, name)
            g["release" if name == "RELEASE" else "touch"] = [base] if name == "RELEASE" else list(range(base, base + 8))
    return g


def profiles(worlds, min_ticks):
    """(X, feature names, world index, life records) over lives alive at least min_ticks."""
    groups = action_groups() if any(w.trace is not None for w in worlds) else {}
    lut = {}
    for name, acts in groups.items():
        for a in acts:
            lut[a] = name
    counts = sorted({k for w in worlds for l in w.lives for k in l["counts"]})
    skills = sorted({k for w in worlds for l in w.lives for k in l.get("skills", {})})
    acts = sorted(groups)
    names = [f"rate:{k}" for k in counts] + [f"skill:{k}" for k in skills] + [f"share:{k}" for k in acts]
    rows, labels, recs = [], [], []
    for wi, w in enumerate(worlds):
        share = {}
        if w.trace is not None and groups:
            ids = w.life_ids_grid()
            a = w.trace["action"]
            for lid in np.unique(ids[ids >= 0]):
                mask = ids == lid
                c = Counter(lut.get(int(x), "other") for x in a[mask])
                tot = max(mask.sum(), 1)
                share[int(lid)] = {k: c.get(k, 0) / tot for k in acts}
        for l in w.lives:
            if l["ticks"] < min_ticks:
                continue
            per_k = lambda x: 1000.0 * x / l["ticks"]
            row = [per_k(l["counts"].get(k, 0)) for k in counts] + [per_k(l.get("skills", {}).get(k, 0)) for k in skills]
            sh = share.get(l["id"], {})
            row += [sh.get(k, np.nan) for k in acts]
            rows.append(row)
            labels.append(wi)
            recs.append(l)
    X = np.array(rows, float) if rows else np.zeros((0, len(names)))
    keep = ~np.isnan(X).any(0) & (np.nanstd(X, 0) > 0) if len(X) else np.zeros(len(names), bool)
    return X[:, keep], [n for n, k in zip(names, keep) if k], np.array(labels), recs


def standardise(X):
    return (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)


def permanova(Z, labels, perms=999, seed=0):
    """(pseudo-F, R^2, p) for Euclidean distances on Z (Anderson 2001)."""
    def stat(lab):
        groups = np.unique(lab)
        ss_t = ((Z - Z.mean(0)) ** 2).sum()
        ss_w = sum(((Z[lab == g] - Z[lab == g].mean(0)) ** 2).sum() for g in groups)
        a, n = len(groups), len(Z)
        ss_b = ss_t - ss_w
        return (ss_b / (a - 1)) / max(ss_w / (n - a), 1e-12), ss_b / max(ss_t, 1e-12)
    f, r2 = stat(labels)
    rng = np.random.default_rng(seed)
    hits = sum(stat(rng.permutation(labels))[0] >= f - 1e-12 for _ in range(perms))
    return float(f), float(r2), (hits + 1) / (perms + 1)


def residualise(Z, T):
    """Z with the inherited traits T regressed out (pooled least squares, with intercept)."""
    A = np.column_stack([np.ones(len(T)), standardise(T)])
    beta, *_ = np.linalg.lstsq(A, Z, rcond=None)
    return Z - A @ beta


def assignment(Z, labels, perms=199, seed=0):
    """Leave-one-out nearest-world-mean accuracy, chance, permutation p."""
    def acc(lab):
        ok = 0
        for i in range(len(Z)):
            best, bd = None, np.inf
            for g in np.unique(lab):
                m = (lab == g)
                m[i] = False
                if m.sum() == 0:
                    continue
                d = ((Z[i] - Z[m].mean(0)) ** 2).sum()
                if d < bd:
                    best, bd = g, d
            ok += best == lab[i]
        return ok / len(Z)
    a = acc(labels)
    rng = np.random.default_rng(seed)
    null = [acc(rng.permutation(labels)) for _ in range(perms)]
    chance = max(np.bincount(labels)) / len(labels)
    return float(a), float(chance), (sum(x >= a - 1e-12 for x in null) + 1) / (perms + 1)


def traditions(X, names, labels, worlds, customary=0.5, absent=0.05):
    out = []
    for k, name in enumerate(names):
        if not name.startswith(("rate:", "skill:")):
            continue
        share_doing = [float((X[labels == w, k] > 0).mean()) if (labels == w).any() else np.nan
                       for w in range(len(worlds))]
        cust = [worlds[w].name for w, s in enumerate(share_doing) if s >= customary]
        none = [worlds[w].name for w, s in enumerate(share_doing) if s <= absent]
        if cust and none:
            out.append({"behaviour": name.split(":", 1)[1], "customary_in": cust, "absent_in": none,
                        "share_doing": {worlds[w].name: round(s, 2) for w, s in enumerate(share_doing)}})
    return out


def persistence(Z, labels, recs, perms=999, seed=0):
    """Own-world minus other-world correlation of early and late deviations; permutation p."""
    born = [r["born_t"] for r in recs if r["born_t"] > 0]
    if len(born) < 6:
        return None
    cut = float(np.median(born))
    late = np.array([r["born_t"] > cut for r in recs])
    W = np.unique(labels)

    def stat(lab_late):
        dev = {}
        for part, sel, lab in (("early", ~late, labels), ("late", late, lab_late)):
            mean_all = Z[sel].mean(0)
            for w in W:
                m = sel & (lab == w)
                if m.sum() >= 2:
                    dev[(part, w)] = Z[m].mean(0) - mean_all
        own, other = [], []
        for w in W:
            for v in W:
                if ("early", w) in dev and ("late", v) in dev:
                    c = np.corrcoef(dev[("early", w)], dev[("late", v)])[0, 1]
                    (own if w == v else other).append(c)
        if not own or not other:
            return np.nan
        return float(np.mean(own) - np.mean(other))

    obs = stat(labels)
    if np.isnan(obs):
        return None
    rng = np.random.default_rng(seed)
    lab_late = labels.copy()
    null = []
    for _ in range(perms):
        lp = labels.copy()
        lp[late] = rng.permutation(labels[late])
        null.append(stat(lp))
    null = np.array([x for x in null if not np.isnan(x)])
    return {"own_minus_other": round(obs, 3), "p": round(float(((null >= obs - 1e-12).sum() + 1) / (len(null) + 1)), 4),
            "late_lives": int(late.sum()), "early_lives": int((~late).sum())}


# ------------------------------------------------------------------ dialects
def utterances(world):
    tr = world.trace
    if tr is None:
        return []
    t, i = np.nonzero(tr["spoke"])
    msg = tr["msg"][t, i]
    ctx = tr["ctx"][t, i]
    names = list(tr["context_names"])
    return [(int(tt), tuple(int(x) for x in m), {n: bool(c) for n, c in zip(names, cc)})
            for tt, m, cc in zip(t, msg, ctx)]


def jsd(a: Counter, b: Counter):
    keys = set(a) | set(b)
    p = np.array([a.get(k, 0) for k in keys], float)
    q = np.array([b.get(k, 0) for k in keys], float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda x, y: (x[x > 0] * np.log2(x[x > 0] / y[x > 0])).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _sampled_jsd(xs, ys, n, rng, draws=20):
    """Mean JSD between equal-size samples (n each) of two message lists: divergence estimates
    grow as samples shrink, so comparisons are made at one sample size."""
    if min(len(xs), len(ys)) < n or n < 5:
        return np.nan
    pick = lambda zs: Counter(zs[k] for k in rng.choice(len(zs), n, replace=False))
    return float(np.mean([jsd(pick(xs), pick(ys)) for _ in range(draws)]))


def dialects(worlds, seed=0, min_uses=3):
    rng = np.random.default_rng(seed)
    U = [utterances(w) for w in worlds]
    if sum(len(u) for u in U) == 0:
        return None
    msgs = [[m for _, m, _ in u] for u in U]
    halves = []
    for u in U:
        if len(u) >= 20:
            mid = np.median([t for t, _, _ in u])
            halves.append(([m for t, m, _ in u if t <= mid], [m for t, m, _ in u if t > mid]))
        else:
            halves.append(None)
    within = [_sampled_jsd(a, b, min(len(a), len(b)), rng) for a, b in [h for h in halves if h]]
    n_between = min((min(len(a), len(b)) for a, b in [h for h in halves if h]), default=0)
    between = []
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            between.append(_sampled_jsd(msgs[i], msgs[j], min(n_between, len(msgs[i]), len(msgs[j])), rng))
    vocab = [{m for m, c in Counter(ms).items() if c >= min_uses} for ms in msgs]
    jac = [len(vocab[i] & vocab[j]) / max(len(vocab[i] | vocab[j]), 1)
           for i in range(len(worlds)) for j in range(i + 1, len(worlds))]
    words = {}
    for w, u in zip(worlds, U):
        base = Counter(m for _, m, _ in u)
        tot = sum(base.values())
        words[w.name] = {}
        for ctx in ("predator", "large_prey", "fire"):
            sub = Counter(m for _, m, c in u if c.get(ctx))
            n = sum(sub.values())
            if n < 5:
                continue
            # most typical: highest ratio of use near it to use overall, among messages used 3+ times near it
            cand = [(sub[m] / n) / (base[m] / tot) for m in sub if sub[m] >= min_uses]
            if cand:
                m = max((m for m in sub if sub[m] >= min_uses), key=lambda m: (sub[m] / n) / (base[m] / tot))
                words[w.name][ctx] = {"message": list(m), "lift": round((sub[m] / n) / (base[m] / tot), 2), "uses": sub[m]}
    wb = [x for x in between if not np.isnan(x)]
    ww = [x for x in within if not np.isnan(x)]
    return {"utterances": [len(u) for u in U],
            "between_worlds_jsd_bits": round(float(np.mean(wb)), 4) if wb else None,
            "within_world_jsd_bits": round(float(np.mean(ww)), 4) if ww else None,
            "dialect_index_bits": round(float(np.mean(wb) - np.mean(ww)), 4) if wb and ww else None,
            "vocabulary_overlap_jaccard": round(float(np.mean(jac)), 3) if jac else None,
            "typical_words": words}


def analyse(folders, min_days=5.0, day_length=None, perms=999, seed=0):
    worlds = [World(f) for f in folders]
    seeds = {int(w.trace["map_seed"]) for w in worlds if w.trace is not None}
    day_length = day_length or worlds[0].day_length
    X, names, labels, recs = profiles(worlds, min_days * day_length)
    out = {"worlds": [w.name for w in worlds], "lives": int(len(X)), "features": len(names),
           "same_map": len(seeds) == 1 and -1 not in seeds,
           "lives_per_world": np.bincount(labels, minlength=len(worlds)).tolist() if len(labels) else []}
    if len(worlds) < 2 or len(X) < 2 * len(worlds) or min(out["lives_per_world"]) < 2:
        out["note"] = "too few worlds or lives to compare"
        return out
    Z = standardise(X)
    f, r2, p = permanova(Z, labels, perms, seed)
    out["differentiation"] = {"pseudo_F": round(f, 3), "R2": round(r2, 3), "p": round(p, 4)}
    traits = sorted({k for r in recs for k in r.get("traits", {})})
    T = np.array([[r["traits"].get(k, 0.0) for k in traits] for r in recs], float)
    T = T[:, np.std(T, 0) > 0] if len(T) else T
    if T.shape[1] and len(Z) > T.shape[1] + len(worlds) + 2:
        f2, r22, p2 = permanova(residualise(Z, T), labels, perms, seed)
        out["beyond_genes"] = {"pseudo_F": round(f2, 3), "R2": round(r22, 3), "p": round(p2, 4)}
    a, chance, pa = assignment(Z, labels, 199, seed)
    out["assignment"] = {"accuracy": round(a, 3), "chance": round(chance, 3), "p": round(pa, 4)}
    out["traditions"] = traditions(X, names, labels, worlds)
    out["persistence"] = persistence(Z, labels, recs, perms, seed)
    out["dialects"] = dialects(worlds, seed)
    return out


def render(r, title="Culture"):
    out = [f"{title}: {len(r['worlds'])} worlds ({', '.join(r['worlds'])}), {r['lives']} lives, {r['features']} behaviour measures"]
    if not r.get("same_map"):
        out.append("  WARNING: the worlds do not share one map (or have no trace): differences may be ecological")
    if r.get("note"):
        return out + ["  " + r["note"]]
    d = r["differentiation"]
    out.append(f"  Worlds differ: pseudo-F {d['pseudo_F']}, R^2 {d['R2']}, p {d['p']}")
    if "beyond_genes" in r:
        g = r["beyond_genes"]
        out.append(f"  ... with inherited traits regressed out: pseudo-F {g['pseudo_F']}, R^2 {g['R2']}, p {g['p']}")
    a = r["assignment"]
    out.append(f"  A life's world told from its behaviour: {a['accuracy']:.0%} (chance {a['chance']:.0%}), p {a['p']}")
    if r["traditions"]:
        out.append("  Traditions (customary in some worlds, absent in others):")
        out += [f"    {t['behaviour']}: customary in {', '.join(t['customary_in'])}; absent in {', '.join(t['absent_in'])}"
                for t in r["traditions"][:15]]
    else:
        out.append("  Traditions: none (no behaviour customary in one world and absent in another)")
    p = r["persistence"]
    out.append("  Persistence across generations: " + (
        f"own-world early/late similarity minus other worlds' {p['own_minus_other']:+.3f}, p {p['p']} "
        f"({p['early_lives']} earlier, {p['late_lives']} later-born lives)" if p else "too few births to test"))
    dl = r["dialects"]
    if dl:
        out.append(f"  Dialects: between-world divergence {dl['between_worlds_jsd_bits']} bits vs within-world "
                   f"{dl['within_world_jsd_bits']} (dialect index {dl['dialect_index_bits']}); vocabulary overlap "
                   f"{dl['vocabulary_overlap_jaccard']}")
        for w, ws in dl["typical_words"].items():
            if ws:
                out.append(f"    {w}: " + "; ".join(f"{c} -> {v['message']} (x{v['lift']}, {v['uses']} uses)" for c, v in ws.items()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="+", help="continuous world folders (the same map, different histories)")
    ap.add_argument("--against", nargs="*", default=None, help="control worlds (e.g. no learning by watching)")
    ap.add_argument("--min-days", type=float, default=5.0)
    ap.add_argument("--perms", type=int, default=999)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {"worlds": analyse(args.folder, args.min_days, perms=args.perms)}
    print("\n".join(render(res["worlds"])))
    if args.against:
        res["control"] = analyse(args.against, args.min_days, perms=args.perms)
        print("\n" + "\n".join(render(res["control"], "Control")))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
