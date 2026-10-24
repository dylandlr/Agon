#!/usr/bin/env python
"""Proof of concept: can language develop, and can we measure it?

Two parts.

  tell     Does speech carry knowledge that matters? Scripted agents (the viability foragers, using only
           the given skills and what each remembers) on a band camped out of sight of water; one scout
           knows where water is. In "tell" every agent who remembers water says where it is ("water
           east far") each tick; in "silent" nobody speaks. Same seeds. With --language grounded the
           listeners understand; with emergent the same words mean nothing (a control: it must equal
           silent). Measures: ticks until each naive agent first drinks, thirst deaths, mean water.

  measure  Did trained agents develop a language? From an exported playback (scripts/export_playback.py,
           with --run): how much words tell about the speaker's situation (mutual information between
           the words said and whether water is in sight, its direction, a predator in sight), how
           consistent each word is across speakers, how often a water phrase points where water
           is (grounded mode), and whether listeners turn toward what they were told (the share of
           their next moves that go the way the phrase pointed, against their moves otherwise).

  python scripts/language_poc.py tell --seeds 12 --days 4
  python scripts/language_poc.py measure runs/eval/grounded.html
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lifesim import env as E  # noqa: E402
from lifesim.config import EnvConfig, living_world, with_tick_minutes  # noqa: E402

DIR_OF = {tuple(int(v) for v in E.MOVE_DELTAS[1 + k]): k for k in range(8)}


def direction_word(dr, dc):
    return E.W_DIR0 + DIR_OF[(int(np.sign(dr)), int(np.sign(dc)))]


# ------------------------------------------------------------------ part 1: telling
def tell_cfg(language: str, agents: int, grid: int):
    cfg = living_world(EnvConfig(grid_size=grid, n_agents=agents, skills=True, founder_spread=2,
                                 camp_water_radius=-1), downed=False)
    cfg = with_tick_minutes(cfg, 30.0)
    return dataclasses.replace(cfg, language_mode=language, climate_volatility=0.0, lightning_prob=0.0)


def out_of_sight_of_water(cfg, key, tries=80):
    """A world whose band starts with no water in sight (else there is nothing to tell)."""
    for k in range(tries):
        s = E.reset(jax.random.fold_in(key, k), cfg)
        wet = np.asarray(E.drinkable(s.terrain))
        if not wet.any():
            continue
        pos = np.asarray(s.pos)
        cells = np.argwhere(wet)
        d = np.max(np.abs(cells[None, :, :] - pos[:, None, :]), axis=-1).min()
        if d > cfg.view_radius + 1 and d < 3 * cfg.view_radius:
            return s
    raise SystemExit("no suitable world found")


def seed_scout(s, cfg, scout=0):
    """The scout remembers every water cell within 3 view radii (it has been there)."""
    wet = np.asarray(E.drinkable(s.terrain))
    p = np.asarray(s.pos[scout])
    rr, cc = np.nonzero(wet)
    near = np.max(np.abs(np.stack([rr, cc], -1) - p), axis=-1) <= 3 * cfg.view_radius
    mm = s.mind_map
    for r, c in zip(rr[near], cc[near]):
        mm = mm.at[scout, r, c].set(jnp.array([1.0, 0.0, 1.0, 1.0]))
    return s._replace(mind_map=mm)


def speech_of(s, cfg):
    """Each agent that remembers water says where the nearest remembered water is."""
    mm = np.asarray(s.mind_map)
    pos = np.asarray(s.pos)
    msgs = np.zeros((cfg.n_agents, E.MSG_LEN), np.int32)
    for i in range(cfg.n_agents):
        if not bool(s.alive[i]):
            continue
        known = np.argwhere((mm[i, ..., 0] > 0.5) & (mm[i, ..., 2] > 0.05))
        if not len(known):
            continue
        d = np.max(np.abs(known - pos[i]), axis=-1)
        r, c = known[np.argmin(d)]
        dist = int(d.min())
        if dist <= 1:
            msgs[i, :2] = [E.W_WATER, E.W_HERE]
        else:
            msgs[i, :3] = [E.W_WATER, direction_word(r - pos[i, 0], c - pos[i, 1]),
                           E.W_FAR if dist > cfg.view_radius else E.W_NEAR]
    return jnp.asarray(msgs)


def run_tell(seed, days, speak, cfg):
    import viability as V
    act = V.scripted_policy(cfg, "own")
    step = jax.jit(lambda st, a, m: E.step(st, a, cfg, m))
    s = seed_scout(out_of_sight_of_water(cfg, jax.random.PRNGKey(seed)), cfg)
    n, T = cfg.n_agents, int(days * cfg.day_length)
    first = np.full(n, T, float)
    water, said = [], 0
    for t in range(T):
        a = act(s)
        m = speech_of(s, cfg) if speak else jnp.zeros((n, E.MSG_LEN), jnp.int32)
        said += int((np.asarray(m)[:, 0] > 0).sum())
        s, _, _, info = step(s, a, m)
        drank = (np.asarray(info.outcome) == E.OUT_DRINK) & (first == T)
        first[drank] = t
        water.append(float(np.asarray(s.hydration)[np.asarray(s.alive)].mean()) if bool(s.alive.any()) else 0.0)
    naive = first[1:]
    return {"seed": seed, "first_drink_ticks": naive.tolist(), "median_first_drink": float(np.median(naive)),
            "never_drank": int((naive >= T).sum()), "dead": int((~np.asarray(s.alive) & np.asarray(s.existed)).sum()),
            "mean_water": round(float(np.mean(water)), 3), "phrases_said": said}


def cmd_tell(args):
    rows = {}
    for cond, language, speak in (("silent", "grounded", False), ("tell", "grounded", True),
                                  ("tell, not understood", "emergent", True)):
        cfg = tell_cfg(language, args.agents, args.grid)
        rows[cond] = [run_tell(s, args.days, speak, cfg) for s in range(args.seeds)]
        r = rows[cond]
        print(f"{cond:22s} median ticks to first drink {np.median([x['median_first_drink'] for x in r]):6.1f}  "
              f"never drank {sum(x['never_drank'] for x in r):3d}/{(args.agents - 1) * args.seeds}  "
              f"dead {sum(x['dead'] for x in r):3d}  mean water {np.mean([x['mean_water'] for x in r]):.3f}", flush=True)
    a = np.array([x["median_first_drink"] for x in rows["silent"]])
    b = np.array([x["median_first_drink"] for x in rows["tell"]])
    diff = a - b
    # paired sign-flip permutation test on per-seed medians
    rng = np.random.default_rng(0)
    perm = np.array([np.mean(diff * rng.choice([-1, 1], len(diff))) for _ in range(20000)])
    p = float((np.abs(perm) >= abs(diff.mean())).mean())
    print(f"telling saves {diff.mean():.1f} ticks ({diff.mean() * 30 / 60:.1f} h) to first drink on average; "
          f"paired permutation p = {p:.4f} over {len(diff)} seeds")
    if args.out:
        json.dump({"args": vars(args), "rows": rows, "p": p}, open(args.out, "w"), indent=1)


# ------------------------------------------------------------------ part 2: measuring a trained language
def load_playback(path):
    text = Path(path).read_text()
    if path.endswith(".json"):
        return json.loads(text)
    i = text.index("const DATA = ") + len("const DATA = ")
    return json.loads(text[i:text.index(";\n(function", i)])


def mutual_information(xs, ys):
    """I(X; Y) in bits from paired samples, with a Miller-Madow bias correction."""
    n = len(xs)
    if n == 0:
        return 0.0
    cx, cy, cxy = collections.Counter(xs), collections.Counter(ys), collections.Counter(zip(xs, ys))
    h = lambda c: -sum(v / n * math.log2(v / n) for v in c.values())
    mi = h(cx) + h(cy) - h(cxy)
    bias = (len(cx) + len(cy) - len(cxy) - 1) / (2 * n * math.log(2))
    return max(0.0, mi - bias)


def cmd_measure(args):
    d = load_playback(args.playback)
    G, F = d["grid"], d["frames"]
    terr = d["terrain"]
    lex = (d.get("consts") or {}).get("lexicon")
    grounded = (d.get("consts") or {}).get("language_mode") == "grounded"
    view = d.get("view_radius", 5)
    wet = np.array([[terr[r * G + c] in "36" for c in range(G)] for r in range(G)])   # water, marsh
    wcells = np.argwhere(wet)
    word = lambda m: (lex[m] if lex else str(m))

    def water_dir(r, c):                   # direction to the nearest water within sight, or "none"
        if not len(wcells):
            return "none"
        dd = np.max(np.abs(wcells - [r, c]), axis=-1)
        k = int(np.argmin(dd))
        if dd[k] > view:
            return "none"
        if dd[k] <= 1:
            return "here"
        return E.ACTION_NAMES[E.MOVE_N + DIR_OF[(int(np.sign(wcells[k][0] - r)), int(np.sign(wcells[k][1] - c)))]][5:]

    first_words, contexts = [], collections.defaultdict(list)
    per_word_speakers = collections.defaultdict(set)
    told, true_told = 0, 0
    follow = collections.Counter()
    for t, f in enumerate(F):
        pred = [(r, c) for r, c, *_ in f.get("pred", [])]
        for k, a in enumerate(f["agents"]):
            if not a.get("alive") or not a.get("said"):
                continue
            msg = [m for m in a["msg"] if m > 0]
            if not msg:
                continue
            w = tuple(msg)
            first_words.append(w)
            wd = water_dir(a["r"], a["c"])
            contexts["water in sight (direction)"].append(wd)
            contexts["water in sight (yes/no)"].append(wd != "none")
            contexts["predator in sight"].append(any(max(abs(r - a["r"]), abs(c - a["c"])) <= view for r, c in pred))
            contexts["thirsty"].append(a.get("water", 1) < 0.5)
            for m in msg:
                per_word_speakers[m].add(k)
            # grounded: is a water phrase true (water within 3 cells of the place it points to)?
            if grounded and E.W_WATER in msg:
                dirs = [m for m in msg if E.W_DIR0 <= m < E.W_DIR0 + 8]
                if dirs or E.W_HERE in msg:
                    dist = 2 * view if E.W_FAR in msg else max(2, view // 2)
                    dlt = E.MOVE_DELTAS[1 + dirs[0] - E.W_DIR0] if dirs else np.zeros(2, int)
                    pr = np.clip(np.array([a["r"], a["c"]]) + np.asarray(dlt) * (dist if dirs else 0), 0, G - 1)
                    told += 1
                    true_told += bool(len(wcells) and np.max(np.abs(wcells - pr), axis=-1).min() <= 3)
                    # listeners: does their next move go the way the phrase pointed?
                    if dirs and t + 1 < len(F):
                        for j in a.get("heard_by", []):
                            p0, p1 = f["agents"][j], F[t + 1]["agents"][j]
                            mv = (np.sign(p1["r"] - p0["r"]), np.sign(p1["c"] - p0["c"]))
                            if mv == (0, 0):
                                follow["stayed"] += 1
                            elif mv == (int(np.sign(dlt[0])), int(np.sign(dlt[1]))):
                                follow["went the way told"] += 1
                            else:
                                follow["went elsewhere"] += 1
    n = len(first_words)
    print(f"{args.playback}: {len(F)} ticks, {n} utterances, language {'grounded' if grounded else 'emergent'}")
    if not n:
        print("nobody spoke")
        return
    common = collections.Counter(first_words).most_common(8)
    print("most said:", ", ".join(f"{' '.join(word(m) for m in w)} x{c}" for w, c in common))
    # each message's leading word against the speaker's situation, with a shuffled-situation null
    # (with few utterances raw MI is inflated: only the excess over chance means anything)
    lead = [w[0] for w in first_words]
    rng = np.random.default_rng(0)
    print("what the leading word tells about the speaker's situation (bits over chance; p from 500 shuffles):")
    for name, ys in contexts.items():
        mi = mutual_information(lead, ys)
        null = [mutual_information(lead, list(rng.permutation(np.array(ys, dtype=object)))) for _ in range(500)]
        p = (1 + sum(x >= mi for x in null)) / 501
        print(f"  {name:28s} {mi - float(np.mean(null)):+.3f}  (raw {mi:.3f}, p = {p:.3f})")
    shared = [m for m, sp in per_word_speakers.items() if len(sp) >= 2]
    print(f"words used by two or more speakers: {len(shared)} of {len(per_word_speakers)}")
    if grounded:
        print(f"water phrases pointing to water (within 3 cells): {true_told}/{told}"
              + (f" = {true_told / told:.0%}" if told else ""))
        tot = sum(follow.values())
        if tot:
            chance = 1 / 9
            print(f"listeners' next move after a water phrase with a direction: "
                  + ", ".join(f"{k} {v / tot:.0%}" for k, v in follow.items())
                  + f"  (one of 9 moves by chance: {chance:.0%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tell")
    t.add_argument("--seeds", type=int, default=12)
    t.add_argument("--days", type=float, default=4)
    t.add_argument("--agents", type=int, default=6)
    t.add_argument("--grid", type=int, default=32)
    t.add_argument("--out", default="")
    m = sub.add_parser("measure")
    m.add_argument("playback")
    args = ap.parse_args()
    (cmd_tell if args.cmd == "tell" else cmd_measure)(args)


if __name__ == "__main__":
    main()
