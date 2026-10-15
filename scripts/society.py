"""Society in a continuous world: whether people form bands, keep places, share, hunt together,
divide work, and punish, and how all of that changes over the generations.

    python scripts/society.py runs/exp_<v>/validate/caveman            # finds continuous/
    python scripts/society.py <world folder> --window-days 40 --json society.json

Nothing here feeds back into the world. Each measure is set against a null model of the same
people doing the same things without the social structure in question, so "more than chance" is
always a number with a p-value, not an impression. Needs trace.npz (positions and actions per
tick) and events.jsonl.gz; who is who comes from lives.json (scripts/society_data.py).

Bands (fission-fusion; Aureli et al. 2008, Whitehead 2008):
- Together: alive, on the same layer, within --radius cells (Chebyshev; a cell is about a
  half-hour walk across). Groups at a tick are the connected clusters of people together.
- Typical group size (Jarman 1974): the size of the group the average person is in, and the share
  of people alone; the fission-fusion rate is how often a person's companions change per day.
- Association: the simple ratio index (SRI) of each pair, the share of the hours both were alive
  that they were together; social differentiation is the coefficient of variation of the SRIs
  (near 0: everyone mixes evenly; above about 0.5: well-differentiated, as in many primate and
  cetacean societies). Null: for each pair, one track shifted in time by every whole number of days
  (keeping each person's daily rhythm and the places they go, breaking who they go with). The
  association ratio (observed over that null) above 1 is company sought, not company from sharing a
  water hole. Its p (rank among the lags) is approximate: when people drift far over a window (no
  home range), neighbouring lags are alike and the p is about twice too liberal; compare replicate
  worlds for firm conclusions. Preferred pairs: above their own null's 95th percentile and at least
  0.2, against the 5% expected by chance (conservative, given the 0.2 floor).
- Bands: the connected components of preferred companions (pairs with an SRI above their own null's
  95th percentile and at least 0.2). Band stability: for each band, its best overlap (Jaccard, among
  people alive in both) with a band in the next window.
Places:
- Home range: the number of cells holding 95% of a person's hourly positions in a window.
- Camps: where people slept (the SLEEP action). Fidelity is the share of nights slept within one
  cell of a place the same person slept before in the window; co-sleeping is the share of nights
  slept within one cell of someone else.
- Between bands: the share of meetings (people of different bands together) with contact aggression,
  against the same share within bands.
Sharing (took_food: someone takes food another left):
- Rate per person-year; the share given to kin (pedigree relatedness r >= 0.25) against the kin
  share of the giver's time spent together (a gift that only follows company has the same share);
  reciprocity: the correlation of who gave to whom with who gave back (dyads; node-permutation p).
Hunting together: a kill of large prey is joint when someone else struck an animal within two
cells in the three ticks before. The joint share, and how often food left by a hunter within a
day of their kill is taken by another.
Division of labour (Gorelick et al. 2004): tasks are gathering, hunting, tending fire, nursing and
leaving food for others. DOL = I(individual; task) / H(task), from 0 (everyone does the same mix)
to 1 (each does one task only), against the null of each person doing as many acts drawn from the
population's mix (which corrects small-sample bias). By sex: the female share of each task over the
female share of person-time, with a permutation p on sex labels.
Punishment (Fehr and Gaechter 2002, in the field Boehm 1999): after an act of contact aggression,
is the aggressor the target of third-party aggression within a day more often than if the times
they received aggression were shifted at random (each person's own record moved round in time, keeping its rate and
clustering)? Also the share of those third parties who are kin of the victim. Acts of aggression
the actor's own policy rated under 5% likely are left out as exploratory (where events record it).
Inequality: Gini of eating rate and of children among adults who lived at least a year.
The trajectory gives the main numbers per window, with the generations alive, so a society's
course over the generations reads in one table.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from society_data import World, deliberate, perm_p, spearman  # noqa: F401

SLEEP_ACTION = 70           # lifesim/env.py SLEEP; checked against the env in tests/test_society.py
TASKS = ["gathering", "hunting", "tending fire", "nursing", "leaving food"]


# ------------------------------------------------------------------ positions and groups
def layered(trace):
    pos = trace["pos"].astype(int)
    if "below" in trace:
        pos = pos.copy()
        pos[..., 1] += 4096 * trace["below"].astype(int)
    return pos


def components(adj):
    """Connected components of (T, N, N) boolean graphs: (T, N) labels (the smallest member index)."""
    T, N, _ = adj.shape
    lab = np.broadcast_to(np.arange(N), (T, N)).copy()
    adj = adj | np.eye(N, dtype=bool)[None]
    for _ in range(N):
        new = np.where(adj, lab[:, None, :], N).min(-1)
        if (new == lab).all():
            break
        lab = new
    return lab


def together(pos, alive, radius):
    """(T, N, N) bool: both alive and within radius (Chebyshev), not oneself."""
    d = np.abs(pos[:, :, None, :] - pos[:, None, :, :]).max(-1)
    m = (d <= radius) & alive[:, :, None] & alive[:, None, :]
    idx = np.arange(pos.shape[1])
    m[:, idx, idx] = False
    return m


def group_stats(tog, alive):
    """Per tick: typical group size, share alone, number of groups (of 2+)."""
    lab = components(tog)
    T, N = alive.shape
    tgs, alone, ngroups = np.full(T, np.nan), np.full(T, np.nan), np.zeros(T, int)
    for t in range(T):
        a = alive[t]
        if not a.any():
            continue
        l = lab[t][a]
        _, inv, cnt = np.unique(l, return_inverse=True, return_counts=True)
        size = cnt[inv]
        tgs[t] = size.mean()
        alone[t] = (size == 1).mean()
        ngroups[t] = int((cnt >= 2).sum())
    return tgs, alone, ngroups, lab


def year_ticks(world: World):
    """Ticks in the world's year (its run's days_per_year, else 40)."""
    for d in (world.dir, world.dir.parent):
        h = Path(d) / "history.json"
        if h.exists():
            return world.day_length * int(json.load(open(h)).get("args", {}).get("days_per_year", 40))
    return world.day_length * 40


# ------------------------------------------------------------------ kinship
def relatedness(world: World):
    """Life id -> life id -> r (2 x kinship coefficient) from the pedigree; founders unrelated."""
    lives = sorted(world.lives, key=lambda l: l["born_t"])
    parents = {}
    for l in lives:
        ps = []
        if l["born_t"] > 0:
            for role in ("mother", "father"):
                if l.get(role, -1) >= 0:
                    p = world.life_at(l[role], l["born_t"])
                    if p is not None and p["id"] != l["id"]:
                        ps.append(p["id"])
        parents[l["id"]] = ps
    order = [l["id"] for l in lives]
    pos = {k: n for n, k in enumerate(order)}
    phi = {}

    def kin(a, b):
        if a == b:
            ps = parents[a]
            return 0.5 * (1 + (kin(ps[0], ps[1]) if len(ps) == 2 else 0.0))
        if pos[a] < pos[b]:
            a, b = b, a                                   # a is the younger: recurse through a's parents
        key = (a, b)
        if key not in phi:
            ps = parents[a]
            phi[key] = sum(kin(p, b) for p in ps) / 2 if ps else 0.0
        return phi[key]

    import sys
    sys.setrecursionlimit(max(10000, 4 * len(order)))
    return lambda a, b: 2 * kin(a, b)


# ------------------------------------------------------------------ the measures
def bands(world: World, lo, hi, radius=1, every=None, nulls=199, seed=0):
    tr = world.trace
    day = world.day_length
    every = every or max(1, day // 24)                    # hourly samples
    ts = np.arange(lo, min(hi, tr["alive"].shape[0]), every)
    if len(ts) < 4:
        return None
    ids = world.life_ids_grid()[ts]
    pos = layered(tr)[ts]
    alive = ids >= 0
    tog = together(pos, alive, radius)
    tgs, alone, ngroups, lab = group_stats(tog, alive)
    # who: life ids present in the window, by slot track (a slot is re-used: split by life id)
    present = sorted({int(x) for x in np.unique(ids) if x >= 0})
    k_of = {l: k for k, l in enumerate(present)}
    L = len(present)
    if L < 3:
        return None
    # per life: (T,) mask and positions (a life is one slot)
    occ = np.zeros((len(ts), L), bool)
    lpos = np.zeros((len(ts), L, 2), int)
    for s in range(ids.shape[1]):
        for t_ in np.flatnonzero(ids[:, s] >= 0):
            k = k_of[int(ids[t_, s])]
            occ[t_, k] = True
            lpos[t_, k] = pos[t_, s]

    def sri(P, O):
        tg = together(P, O, radius).sum(0).astype(float)
        both = (O[:, :, None] & O[:, None, :]).sum(0).astype(float)
        return np.where(both >= 4, tg / np.maximum(both, 1), np.nan), both

    obs, both = sri(lpos, occ)
    iu = np.triu_indices(L, 1)
    ok = ~np.isnan(obs[iu])
    vals = obs[iu][ok]
    cv = float(np.std(vals) / np.mean(vals)) if len(vals) and np.mean(vals) > 0 else 0.0
    # null: each pair with one of them shifted in time by whole days (every lag of 1 .. days-1, each
    # way): who goes where, and when in the day, is kept; going with this person is broken
    per_day = max(1, day // every)
    n_days = len(ts) // per_day
    lags = list(range(1, n_days))
    if len(lags) > nulls:
        lags = sorted(np.random.default_rng(seed).choice(lags, nulls, replace=False).tolist())
    null_vals, null_cv, null_mean = [], [], []
    for lag in lags:
        Pr, Or = np.roll(lpos, -lag * per_day, axis=0), np.roll(occ, -lag * per_day, axis=0)
        near = (np.abs(lpos[:, :, None, :] - Pr[:, None, :, :]).max(-1) <= radius) & occ[:, :, None] & Or[:, None, :]
        bb = (occ[:, :, None] & Or[:, None, :]).sum(0).astype(float)
        s2 = np.where(bb >= 4, near.sum(0) / np.maximum(bb, 1), np.nan)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            s2 = np.nanmean(np.stack([s2, s2.T]), 0)                # both ways round
        null_vals.append(s2[iu])
        v = s2[iu][ok]
        v = v[~np.isnan(v)]
        null_mean.append(float(v.mean()) if len(v) else 0.0)
        null_cv.append(float(v.std() / v.mean()) if len(v) and v.mean() > 0 else 0.0)
    if not lags:                                                     # under two days: no null to test against
        null_vals, null_mean, null_cv = [np.full(len(iu[0]), np.nan)], [np.nan], [np.nan]
    null_vals = np.array(null_vals)
    q95 = np.nanpercentile(null_vals, 95, axis=0) if lags else np.full(len(iu[0]), np.nan)
    pref = np.zeros((L, L), bool)
    hit = ok & (obs[iu] > np.nan_to_num(q95, nan=1.0)) & (obs[iu] >= 0.2)
    pref[iu[0][hit], iu[1][hit]] = True
    pref = pref | pref.T
    blab = components(pref[None])[0]
    groups = {}
    for k, b in enumerate(blab):
        groups.setdefault(int(b), []).append(present[k])
    band_list = sorted((sorted(g) for g in groups.values() if len(g) >= 2), key=len, reverse=True)
    mean_sri = float(vals.mean()) if len(vals) else 0.0
    p_greg = (1 + sum(m >= mean_sri for m in null_mean)) / (1 + len(lags)) if lags else None
    # preferred pairs: how many pairs beat their own null's 95th percentile, against the 5% expected
    # by chance (binomial tail); the CV is descriptive (a time-shift null keeps camps, so its CV is
    # no fair yardstick)
    n_pairs = int(ok.sum())
    n_pref = int(hit.sum()) if lags else 0                       # above the null's 95th percentile and >= 0.2
    from math import comb
    p_diff = sum(comb(n_pairs, k) * 0.05 ** k * 0.95 ** (n_pairs - k) for k in range(n_pref, n_pairs + 1)) if lags and n_pairs else None
    # fission-fusion: how often one's companions change, per day
    changes, n = 0, 0
    for t_ in range(1, len(ts)):
        for s in range(ids.shape[1]):
            if ids[t_, s] >= 0 and ids[t_ - 1, s] == ids[t_, s]:
                now = set(ids[t_][lab[t_] == lab[t_, s]]) - {ids[t_, s]}
                before = set(ids[t_ - 1][lab[t_ - 1] == lab[t_ - 1, s]]) - {ids[t_ - 1, s]}
                changes += int(now != before)
                n += 1
    return {"samples": int(len(ts)), "individuals": L,
            "typical_group_size": round(float(np.nanmean(tgs)), 2), "alone": round(float(np.nanmean(alone)), 3),
            "groups_of_2plus": round(float(ngroups.mean()), 2),
            "companion_changes_per_day": round(changes / max(n, 1) * per_day, 2),
            "mean_association": round(mean_sri, 3), "null_association": round(float(np.nanmean(null_mean)), 3) if lags else None,
            "association_ratio": round(mean_sri / float(np.nanmean(null_mean)), 2) if lags and np.nanmean(null_mean) > 0 else None,
            "gregarious_p": None if p_greg is None else round(p_greg, 4), "null_lags": len(lags),
            "differentiation_cv": round(cv, 3),
            "preferred_pairs": n_pref, "pairs": n_pairs, "preferred_p": None if p_diff is None else round(min(1.0, p_diff), 6),
            "bands": band_list, "in_a_band": round(sum(len(b) for b in band_list) / L, 3),
            "_band_of": {m: n for n, b in enumerate(band_list) for m in b}}


def places(world: World, lo, hi, band_of, radius=1):
    tr = world.trace
    day = world.day_length
    hi = min(hi, tr["alive"].shape[0])
    ids = world.life_ids_grid()[lo:hi]
    pos = layered(tr)[lo:hi]
    act = tr["action"][lo:hi]
    every = max(1, day // 24)
    ranges = []
    for l in np.unique(ids[::every]):
        if l < 0:
            continue
        m = ids[::every] == l
        cells = pos[::every][m]
        if len(cells) < 12:
            continue
        _, cnt = np.unique(cells[:, 0] * 8192 + cells[:, 1], return_counts=True)
        c = np.sort(cnt)[::-1].cumsum() / cnt.sum()
        ranges.append(int(np.searchsorted(c, 0.95) + 1))
    # sleeping places, one per person-night (the first sleeping tick of each day)
    nights = defaultdict(dict)                             # day -> life -> cell
    for t_ in range(hi - lo):
        d = (lo + t_) // day
        for s in np.flatnonzero((ids[t_] >= 0) & (act[t_] == SLEEP_ACTION)):
            nights[d].setdefault(int(ids[t_, s]), tuple(pos[t_, s]))
    fid, cosleep, seen = [], [], defaultdict(list)
    for d in sorted(nights):
        here = nights[d]
        for l, cell in here.items():
            if seen[l]:
                fid.append(any(max(abs(cell[0] - a), abs(cell[1] - b)) <= 1 for a, b in seen[l]))
            seen[l].append(cell)
            cosleep.append(any(max(abs(cell[0] - c2[0]), abs(cell[1] - c2[1])) <= 1 for l2, c2 in here.items() if l2 != l))
    # meetings between bands, and contact aggression in them
    blows = defaultdict(int)
    for e in world.events:
        if e.get("type") == "strike_agent" and lo <= e["t"] < hi and deliberate(e):
            a = world.life_at(e["agent"], e["t"])
            for v in e.get("victims", []):
                b = world.life_at(v, e["t"])
                if a and b:
                    blows[(e["t"], a["id"], b["id"])] += 1
    tog = together(pos[::every], ids[::every] >= 0, radius)
    inter = intra = 0
    upper = np.triu(np.ones(tog.shape[1:], bool), 1)[None]
    for t_, i, j in zip(*np.nonzero(tog & upper)):
        a, b = int(ids[::every][t_, i]), int(ids[::every][t_, j])
        if a in band_of and b in band_of:
            if band_of[a] == band_of[b]:
                intra += 1
            else:
                inter += 1
    bi = bx = 0
    for (t_, a, b) in blows:
        if a in band_of and b in band_of:
            if band_of[a] == band_of[b]:
                bi += 1
            else:
                bx += 1
    return {"home_range_cells_median": int(np.median(ranges)) if ranges else None,
            "sleep_site_fidelity": round(float(np.mean(fid)), 3) if fid else None,
            "co_sleeping": round(float(np.mean(cosleep)), 3) if cosleep else None,
            "between_band_meeting_hours": inter, "within_band_hours": intra,
            "aggression_per_100_hours_between_bands": round(100 * bx / inter, 2) if inter else None,
            "aggression_per_100_hours_within_bands": round(100 * bi / intra, 2) if intra else None}


def sharing(world: World, lo, hi, r_of, assoc=None, perms=999, seed=0):
    gifts = []
    for e in world.events:
        if e.get("type") == "took_food" and lo <= e["t"] < hi and e.get("giver", -1) >= 0:
            b = world.life_at(e["agent"], e["t"])
            a = world.life_at(e["giver"], e["t"])
            if a and b and a["id"] != b["id"]:
                gifts.append((a["id"], b["id"]))
    person_years = sum(max(0, min(hi, (l["died_t"] if l.get("died_t") is not None else world.t_end)) - max(lo, l["born_t"]))
                       for l in world.lives) / year_ticks(world)
    out = {"gifts": len(gifts), "gifts_per_person_year": round(len(gifts) / max(person_years, 1e-9), 2)}
    if not gifts:
        return out
    to_kin = np.mean([r_of(a, b) >= 0.25 for a, b in gifts])
    out["share_to_kin"] = round(float(to_kin), 3)
    if assoc:
        exp = []
        for a, _ in gifts:
            row = assoc.get(a, {})
            tot = sum(row.values())
            if tot > 0:
                exp.append(sum(v for b, v in row.items() if r_of(a, b) >= 0.25) / tot)
        if exp:
            out["kin_share_of_company"] = round(float(np.mean(exp)), 3)
    who = sorted({x for g in gifts for x in g})
    if len(who) >= 4:
        k = {w: n for n, w in enumerate(who)}
        G = np.zeros((len(who), len(who)))
        for a, b in gifts:
            G[k[a], k[b]] += 1
        iu = np.triu_indices(len(who), 1)
        x, y = G[iu], G.T[iu]
        if np.ptp(x) > 0 and np.ptp(y) > 0:
            rho = float(np.corrcoef(x, y)[0, 1])
            rng = np.random.default_rng(seed)
            hits = 0
            for _ in range(perms):
                p = rng.permutation(len(who))
                Gp = G[np.ix_(p, p)]
                hits += np.corrcoef(G[iu], Gp.T[iu])[0, 1] >= rho - 1e-12
            out["reciprocity_r"] = round(rho, 3)
            out["reciprocity_p"] = round((hits + 1) / (perms + 1), 4)
    return out


def hunting(world: World, lo, hi):
    strikes = defaultdict(list)                            # t -> [(life id, pos)]
    kills, hunters_food = [], []
    for e in world.events:
        if not (lo <= e["t"] < hi):
            continue
        if e.get("type") in ("strike_animal", "kill"):
            l = world.life_at(e["agent"], e["t"])
            if l:
                strikes[e["t"]].append((l["id"], e.get("pos", [0, 0])))
        if e.get("type") == "kill" and e.get("animal") == "large_prey":
            l = world.life_at(e["agent"], e["t"])
            if l:
                kills.append((e["t"], l["id"], e.get("pos", [0, 0])))
    joint = []
    for t, who, p in kills:
        mates = {w for tt in range(t - 3, t + 1) for w, q in strikes.get(tt, [])
                 if w != who and max(abs(q[0] - p[0]), abs(q[1] - p[1])) <= 2}
        joint.append(len(mates))
    taken = 0
    for e in world.events:
        if e.get("type") == "took_food" and lo <= e["t"] < hi:
            g = world.life_at(e.get("giver", -1), e["t"]) if e.get("giver", -1) >= 0 else None
            if g and any(k[1] == g["id"] and 0 <= e["t"] - k[0] <= world.day_length for k in kills):
                taken += 1
    return {"large_kills": len(kills),
            "joint_share": round(float(np.mean([j > 0 for j in joint])), 3) if joint else None,
            "hunters_per_kill": round(1 + float(np.mean(joint)), 2) if joint else None,
            "food_taken_from_hunters_within_a_day": taken}


def task_counts(world: World, lo, hi, min_acts=10):
    counts = defaultdict(lambda: np.zeros(len(TASKS)))
    kind = {"gather": 0, "strike_animal": 1, "kill": 1, "fed_fire": 2, "ignite": 2, "release": 4}
    for e in world.events:
        if lo <= e["t"] < hi and e.get("type") in kind:
            if e["type"] == "release" and not e.get("near"):
                continue                                  # dropped with nobody near: not left for someone
            l = world.life_at(e["agent"], e["t"])
            if l:
                counts[l["id"]][kind[e["type"]]] += 1
    for l in world.lives:                                 # nursing is counted per life, not as events
        n = l.get("counts", {}).get("nursing", 0)
        if n and l["born_t"] < hi and (l.get("died_t") is None or l["died_t"] >= lo):
            counts[l["id"]][3] += n
    return {k: v for k, v in counts.items() if v.sum() >= min_acts}


def dol_index(M):
    """Gorelick's division of labour: I(individual; task) / H(task) from an (individuals x tasks) count matrix."""
    P = M / M.sum()
    pi, pt = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    I = float((P[nz] * np.log(P[nz] / (pi @ pt)[nz])).sum())
    Ht = float(-(pt[pt > 0] * np.log(pt[pt > 0])).sum())
    return I / Ht if Ht > 1e-9 else 0.0


def labour(world: World, lo, hi, nulls=999, seed=0):
    tc = task_counts(world, lo, hi)
    if len(tc) < 3:
        return {"individuals": len(tc)}
    ids = sorted(tc)
    M = np.array([tc[k] for k in ids])
    M = M[:, M.sum(0) > 0]
    if M.shape[1] < 2:
        return {"individuals": len(ids), "tasks_done": [TASKS[j] for j in range(len(TASKS)) if np.array([tc[k] for k in ids])[:, j].sum() > 0],
                "dol": None, "note": "only one kind of work was done"}
    obs = dol_index(M)
    rng = np.random.default_rng(seed)
    mix = M.sum(0) / M.sum()
    null = [dol_index(np.array([rng.multinomial(int(r.sum()), mix) for r in M])) for _ in range(nulls)]
    out = {"individuals": len(ids), "dol": round(obs, 3), "null_dol": round(float(np.mean(null)), 3),
           "dol_p": round((1 + sum(x >= obs for x in null)) / (1 + nulls), 4)}
    sex = {l["id"]: l.get("sex") for l in world.lives}
    fem = np.array([sex.get(k) == "female" for k in ids])
    if 0 < fem.sum() < len(ids):
        full = np.array([tc[k] for k in ids])
        by = {}
        for j, name in enumerate(TASKS):
            col = full[:, j]
            if col.sum() < 10:
                continue
            stat = lambda f, c=col: c[f].sum() / c.sum() - f.mean()
            o = stat(fem)
            hits = sum(abs(stat(rng.permutation(fem))) >= abs(o) - 1e-12 for _ in range(nulls))
            by[name] = {"female_share_minus_expected": round(float(o), 3), "p": round((hits + 1) / (nulls + 1), 4)}
        out["by_sex"] = by
    return out


def punishment(world: World, lo, hi, r_of, perms=999, seed=0):
    day = world.day_length
    blows = []
    for e in world.events:
        if e.get("type") == "strike_agent" and lo <= e["t"] < hi and deliberate(e):
            a = world.life_at(e["agent"], e["t"])
            for v in e.get("victims", []):
                b = world.life_at(v, e["t"])
                if a and b and a["id"] != b["id"]:
                    blows.append((e["t"], a["id"], b["id"]))
    if len(blows) < 5:
        return {"aggression": len(blows)}
    by_victim = defaultdict(list)                          # life -> [(t, striker)]
    for t, a, b in blows:
        by_victim[b].append((t, a))

    def struck_by_third(a, t, excl):
        return [s for tt, s in by_victim[a] if t < tt <= t + day and s not in excl]

    after = [struck_by_third(a, t, {b}) for t, a, b in blows]
    p_obs = float(np.mean([len(x) > 0 for x in after]))
    # null: each person's record of being struck is shifted round in time (circularly within the
    # window) by a random amount, keeping how often and how clustered it is, and breaking only its
    # timing relative to their own aggression; p is the share of shifts doing as well as what happened
    rng = np.random.default_rng(seed)
    span = max(1, hi - lo)
    null = []
    for _ in range(perms):
        sh = {v: int(rng.integers(span)) for v in by_victim}
        moved = {v: [((tt - lo + sh[v]) % span + lo, s) for tt, s in xs] for v, xs in by_victim.items()}
        null.append(np.mean([any(t < tt <= t + day and s != b for tt, s in moved.get(a, [])) for t, a, b in blows]))
    base = float(np.mean(null))
    p = (1 + sum(x >= p_obs - 1e-12 for x in null)) / (1 + perms)
    kin_share = [np.mean([r_of(s, b) >= 0.25 for s in x]) for x, (t, a, b) in zip(after, blows) if x]
    return {"aggression": len(blows), "third_party_aggression_within_a_day": round(p_obs, 3), "baseline": round(base, 3),
            "punishment_ratio": round(p_obs / base, 2) if base > 0 else None, "p": round(p, 4),
            "punishers_kin_of_victim": round(float(np.mean(kin_share)), 3) if kin_share else None}


def gini(x):
    x = np.sort(np.asarray(x, float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def inequality(world: World, lo, hi):
    year = year_ticks(world)
    kids = world.children()
    adults = [l for l in world.lives if l["ticks"] >= year and l["born_t"] < hi
              and (l.get("died_t") is None or l["died_t"] >= lo)]
    if len(adults) < 4:
        return {"adults": len(adults)}
    eat = [1000 * l["counts"].get("eats", 0) / max(l["ticks"], 1) for l in adults]
    return {"adults": len(adults), "gini_eating": round(gini(eat), 3), "gini_children": round(gini([kids[l["id"]] for l in adults]), 3)}


# ------------------------------------------------------------------ report
def analyse(world: World, window_days=40.0, radius=1, nulls=199, seed=0):
    if world.trace is None:
        raise SystemExit(f"no trace.npz in {world.dir}: rerun the continuous world with this code")
    day = world.day_length
    W = int(window_days * day)
    T = world.trace["alive"].shape[0]
    r_of = relatedness(world)
    windows = []
    for k in range(max(1, int(np.ceil(T / W)))):
        lo, hi = k * W, min(T, (k + 1) * W)
        gens = [l.get("generation", 0) for l in world.lives if l["born_t"] < hi and (l.get("died_t") is None or l["died_t"] >= lo)]
        b = bands(world, lo, hi, radius=radius, nulls=nulls, seed=seed + k)
        band_of = b.pop("_band_of") if b else {}
        assoc = None
        if b:                                              # time together by pair, for the gift null
            ids = world.life_ids_grid()[lo:hi:max(1, day // 24)]
            pos = layered(world.trace)[lo:hi:max(1, day // 24)]
            tg = together(pos, ids >= 0, radius)
            assoc = defaultdict(lambda: defaultdict(int))
            for t_, i, j in zip(*np.nonzero(tg)):
                assoc[int(ids[t_, i])][int(ids[t_, j])] += 1
        windows.append({"window": k, "from_day": round(lo / day, 1), "generations_alive": sorted(set(gens)),
                        "bands": b, "places": places(world, lo, hi, band_of, radius) if b else None,
                        "sharing": sharing(world, lo, hi, r_of, assoc, seed=seed), "hunting": hunting(world, lo, hi),
                        "labour": labour(world, lo, hi, seed=seed), "punishment": punishment(world, lo, hi, r_of, seed=seed),
                        "inequality": inequality(world, lo, hi)})
    stab = []
    for a, b in zip(windows, windows[1:]):
        if a["bands"] and b["bands"] and a["bands"]["bands"] and b["bands"]["bands"]:
            nxt = [set(x) for x in b["bands"]["bands"]]
            alive_next = {m for x in nxt for m in x} | {l["id"] for l in world.lives if l["born_t"] < (b["window"] + 1) * W}
            for band in a["bands"]["bands"]:
                s = set(band) & alive_next
                if len(s) >= 2:
                    stab.append(max(len(s & x) / len(s | x) for x in nxt))
    return {"world": world.name, "window_days": window_days, "radius": radius, "windows": windows,
            "band_stability": round(float(np.mean(stab)), 3) if stab else None}


def _f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(r):
    out = [f"Society: {r['world']} (windows of {r['window_days']:.0f} days; together = within {r['radius']} cell)"]
    head = ("  from day  gens  grp size  alone  bands  in band  diff CV (pref, p)     gifts/py  to kin  recip   "
            "joint hunts  DOL (null, p)       punish x (p)   gini eat")
    out.append(head)
    for w in r["windows"]:
        b, s, h, lab, pu, ineq = w["bands"] or {}, w["sharing"], w["hunting"], w["labour"], w["punishment"], w["inequality"]
        g = w["generations_alive"]
        out.append(f"  {w['from_day']:>8.0f}  {(f'{min(g)}-{max(g)}' if g else '-'):>4}  "
                   f"{_f(b.get('typical_group_size')):>8}  {_f(b.get('alone')):>5}  {len(b.get('bands', [])):>5}  {_f(b.get('in_a_band')):>7}  "
                   f"{_f(b.get('differentiation_cv')):>5} ({b.get('preferred_pairs', '-')}/{b.get('pairs', '-')}, {_f(b.get('preferred_p'), 3)})  "
                   f"{_f(s.get('gifts_per_person_year')):>8}  {_f(s.get('share_to_kin')):>6}  {_f(s.get('reciprocity_r')):>5}   "
                   f"{_f(h.get('joint_share')):>5} of {h.get('large_kills', 0):<4} {_f(lab.get('dol')):>5} ({_f(lab.get('null_dol'))}, {_f(lab.get('dol_p'), 3)})  "
                   f"{_f(pu.get('punishment_ratio')):>5} ({_f(pu.get('p'), 3)})  {_f(ineq.get('gini_eating')):>8}")
    if r["band_stability"] is not None:
        out.append(f"Band stability between windows (mean best Jaccard): {r['band_stability']}")
    last = next((w for w in reversed(r["windows"]) if w["bands"]), None)
    if last:
        b, p = last["bands"], last["places"] or {}
        out.append(f"Last window: {b['individuals']} people, association {b['mean_association']} against {b['null_association']} "
                   f"by chance (p {b['gregarious_p']}); bands of {', '.join(str(len(x)) for x in b['bands']) or 'none'}; "
                   f"companions change {b['companion_changes_per_day']} times a day; home range {p.get('home_range_cells_median')} cells; "
                   f"sleep-site fidelity {p.get('sleep_site_fidelity')}, co-sleeping {p.get('co_sleeping')}; acts of aggression per 100 hours together "
                   f"within bands {p.get('aggression_per_100_hours_within_bands')}, between {p.get('aggression_per_100_hours_between_bands')}.")
        lab = last["labour"]
        if lab.get("by_sex"):
            out.append("By sex (female share of the task minus her share of the people): "
                       + "; ".join(f"{k} {v['female_share_minus_expected']:+.2f} (p {v['p']})" for k, v in lab["by_sex"].items()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="+", help="continuous world folders (or run folders containing continuous/)")
    ap.add_argument("--window-days", type=float, default=40.0)
    ap.add_argument("--radius", type=int, default=1, help="cells (Chebyshev) within which people count as together")
    ap.add_argument("--nulls", type=int, default=199, help="time-shift null samples for association")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = []
    for f in args.folder:
        r = analyse(World(f, need_trace=True), args.window_days, args.radius, args.nulls)
        res.append(r)
        print("\n".join(render(r)) + "\n")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
