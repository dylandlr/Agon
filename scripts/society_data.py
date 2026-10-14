"""Shared loading for the society analyses (hierarchy.py, culture.py): who was who, when.

A continuous world folder holds lives.json (one record per life: slot, born_t, died_t, counts,
traits, parents as slots), events.jsonl.gz (what happened, by slot) and trace.npz (per tick:
positions, alive, actions, speech and what was around each speaker). A slot is reused after a
death, so an event or a tick is tied to a life by its slot and time.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def folder_of(src) -> Path:
    src = Path(src)
    return src / "continuous" if (src / "continuous").exists() and not (src / "lives.json").exists() else src


def day_length_of(folder, trace=None) -> int:
    """Ticks per day of a recorded world: from its trace, else its run's history (tick_minutes),
    else the 30-minute calibration's 48."""
    if trace is not None and "day_length" in trace:
        return int(trace["day_length"])
    for d in (Path(folder), Path(folder).parent):
        h = d / "history.json"
        if h.exists():
            m = json.load(open(h)).get("args", {}).get("tick_minutes", 30)
            return int(round(1440 / float(m)))
    return 48


class World:
    def __init__(self, src, need_trace=False):
        self.dir = folder_of(src)
        self.name = self.dir.parent.name if self.dir.name == "continuous" else self.dir.name
        f = self.dir / "lives.json"
        if not f.exists():
            raise SystemExit(f"no lives.json in {self.dir}")
        self.lives = json.load(open(f))
        for k, l in enumerate(self.lives):
            l["id"] = k
        self.by_slot = {}
        for l in self.lives:
            self.by_slot.setdefault(l["slot"], []).append(l)
        for v in self.by_slot.values():
            v.sort(key=lambda l: l["born_t"])
        ev = self.dir / "events.jsonl.gz"
        self.events = [json.loads(x) for x in gzip.open(ev, "rt")] if ev.exists() else []
        tr = self.dir / "trace.npz"
        self.trace = dict(np.load(tr)) if tr.exists() else None
        if need_trace and self.trace is None:
            raise SystemExit(f"no trace.npz in {self.dir}: rerun the continuous world with this code")
        self.day_length = day_length_of(self.dir, self.trace)
        self.t_end = max([l["died_t"] for l in self.lives if l.get("died_t") is not None]
                         + [l["born_t"] + l["ticks"] for l in self.lives] + [0])

    def life_at(self, slot, t):
        """The life occupying `slot` at tick t, or None."""
        for l in self.by_slot.get(int(slot), []):
            if l["born_t"] <= t and (l.get("died_t") is None or t <= l["died_t"]):
                return l
        return None

    def life_ids_grid(self):
        """(T, N) life id per tick and slot (-1: nobody alive there), from the trace."""
        T, N = self.trace["alive"].shape
        ids = np.full((T, N), -1, int)
        for l in self.lives:
            end = T - 1 if l.get("died_t") is None else min(l["died_t"], T - 1)
            if l["born_t"] < T:
                ids[l["born_t"]:end + 1, l["slot"]] = l["id"]
        return np.where(self.trace["alive"], ids, -1)

    def children(self):
        """life id -> number of children born (parents are recorded as slots at the birth)."""
        n = {l["id"]: 0 for l in self.lives}
        for c in self.lives:
            if c["born_t"] > 0:
                for role in ("mother", "father"):
                    if c.get(role, -1) >= 0:
                        p = self.life_at(c[role], c["born_t"])
                        if p is not None:
                            n[p["id"]] += 1
        return n


def rankdata(x):
    """Ranks with ties averaged (as scipy.stats.rankdata, without needing scipy)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals))
    np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def perm_p(x, y, stat, perms=999, seed=0):
    """Two-sided permutation p-value of stat(x, y) against y shuffled."""
    rng = np.random.default_rng(seed)
    obs = stat(x, y)
    hits = sum(abs(stat(x, rng.permutation(y))) >= abs(obs) - 1e-12 for _ in range(perms))
    return obs, (hits + 1) / (perms + 1)
