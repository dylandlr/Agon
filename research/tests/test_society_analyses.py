"""Hierarchy and culture analyses on worlds built by hand, where the answer is known."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import culture  # noqa: E402
import hierarchy as H  # noqa: E402
from society_data import World  # noqa: E402


def _life(slot, born=0, ticks=400, counts=None, traits=None, died=None, mother=-1, father=-1, skills=None):
    c = dict.fromkeys(["eats", "matings", "gifts", "strikes_on_agents", "touches", "moves"], 0)
    c.update(counts or {})
    return {"slot": slot, "born_t": born, "ticks": ticks, "counts": c, "traits": traits or {"strength": 1.0},
            "died_t": died, "mother": mother, "father": father, "skills": skills or {}, "generation": 0}


def _write(folder, lives, events=(), trace=None):
    folder.mkdir(parents=True, exist_ok=True)
    json.dump(lives, open(folder / "lives.json", "w"))
    with gzip.open(folder / "events.jsonl.gz", "wt") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    if trace is not None:
        np.savez_compressed(folder / "trace.npz", **trace)
    return folder


def _trace(T, N, pos=None, spoke=None, msg=None, ctx=None, map_seed=21):
    return dict(pos=(np.zeros((T, N, 2)) if pos is None else pos).astype(np.uint8), alive=np.ones((T, N), bool),
                action=np.zeros((T, N), np.uint8), msg=np.zeros((T, N, 3), np.uint8) if msg is None else msg,
                spoke=np.zeros((T, N), bool) if spoke is None else spoke,
                ctx=np.zeros((T, N, 3), bool) if ctx is None else ctx,
                context_names=np.array(["predator", "large_prey", "fire"]), map_seed=np.int64(map_seed),
                seed=np.int64(0), view_radius=np.int64(5), day_length=np.int64(48))


# ------------------------------------------------------------------ hierarchy
def test_a_strict_pecking_order_is_linear_steep_and_ranked_in_order(tmp_path):
    n = 6
    events = [{"type": "strike_agent", "agent": i, "victims": [j], "t": 10 + k}
              for i in range(n) for j in range(i + 1, n) for k in range(3)]
    w = World(_write(tmp_path / "w", [_life(i) for i in range(n)], events))
    r = H.analyse(w, window_days=40)
    win = r["windows"][0]
    scores = [win["scores"][i] for i in range(n)]
    assert scores == sorted(scores, reverse=True)                   # 0 on top, 5 at the bottom
    assert win["linearity_h"] > 0.99 and win["linearity_p"] < 0.05
    assert win["steepness"] > 0.5


def test_random_fights_are_not_a_linear_order():
    rng = np.random.default_rng(1)
    n = 8
    W = rng.integers(0, 3, (n, n)).astype(float)
    np.fill_diagonal(W, 0)
    h, p = H.landau_h(W)
    assert p > 0.05


def test_supplants_and_leading_are_read_from_movement(tmp_path):
    T, N = 6, 2
    pos = np.zeros((T, N, 2))
    pos[:, 0] = [[5, 2], [5, 3], [5, 4], [5, 4], [5, 4], [5, 4]]     # 0 walks up beside 1 and stays
    pos[:, 1] = [[5, 5], [5, 5], [5, 5], [5, 6], [5, 6], [5, 6]]     # 1 steps away at once
    w = World(_write(tmp_path / "s", [_life(0, ticks=T), _life(1, ticks=T)], (), _trace(T, N, pos)))
    enc = H.encounters(w)
    assert ("supplant" in [e[3] for e in enc]) and all(e[1] == 0 for e in enc if e[3] == "supplant")
    # leading: 0 steps east every tick; 1, beside it, does the same one tick later
    T = 12
    pos = np.zeros((T, N, 2))
    pos[:, 0, 0] = pos[:, 1, 0] = 5
    pos[:, 0, 1] = np.arange(T) + 2
    pos[:, 1, 1] = np.concatenate([[1], np.arange(T - 1) + 1])
    w = World(_write(tmp_path / "l", [_life(0, ticks=T), _life(1, ticks=T)], (), _trace(T, N, pos)))
    lead = H.leadership(w)
    assert lead[0][0] > 0 and lead[1][1] > 0 and lead[0][1] == 0


# ------------------------------------------------------------------ culture
def _culture_worlds(tmp_path, distinct, n_lives=12, seed=0, dialect=False):
    rng = np.random.default_rng(seed)
    folders = []
    for w in range(3):
        lives = []
        for k in range(n_lives):
            gifts = rng.poisson(40 if (distinct and w == 0) else 0.0)
            touches = rng.poisson(30 + (25 * w if distinct else 0))
            born = 0 if k < n_lives // 2 else 200 + 10 * k
            lives.append(_life(k, born=born, counts={"gifts": gifts, "touches": touches, "eats": rng.poisson(50),
                                                    "moves": rng.poisson(100)},
                               traits={"strength": float(rng.normal(1, 0.1)), "empathy": float(rng.random())}))
        tr = None
        if dialect:
            T, N = 400, n_lives
            spoke = rng.random((T, N)) < 0.2
            ctx = np.zeros((T, N, 3), bool)
            ctx[:, :, 0] = rng.random((T, N)) < 0.5
            msg = rng.integers(0, 16, (T, N, 3)).astype(np.uint8)
            word = np.array([w + 1] * 3, np.uint8)                   # each world's own word for "predator"
            msg[ctx[:, :, 0]] = word
            tr = _trace(T, N, spoke=spoke, msg=msg, ctx=ctx)
        folders.append(_write(tmp_path / f"world{w}", lives, (), tr))
    return folders


def test_worlds_with_their_own_customs_are_told_apart(tmp_path):
    r = culture.analyse(_culture_worlds(tmp_path, True), min_days=1, perms=199)
    assert r["differentiation"]["p"] < 0.05 and r["assignment"]["accuracy"] > 0.6
    assert any(t["behaviour"] == "gifts" and t["customary_in"] == ["world0"] for t in r["traditions"])
    assert r["persistence"]["own_minus_other"] > 0.3 and r["persistence"]["p"] < 0.1


def test_worlds_without_customs_are_not(tmp_path):
    r = culture.analyse(_culture_worlds(tmp_path, False, seed=3), min_days=1, perms=199)
    assert r["differentiation"]["p"] > 0.05 and not r["traditions"]


def test_dialects_show_as_divergence_beyond_drift_and_own_words(tmp_path):
    r = culture.analyse(_culture_worlds(tmp_path, False, dialect=True, seed=4), min_days=1, perms=99)
    d = r["dialects"]
    assert d["dialect_index_bits"] > 0.05
    words = {w: v["predator"]["message"] for w, v in d["typical_words"].items()}
    assert words == {"world0": [1, 1, 1], "world1": [2, 2, 2], "world2": [3, 3, 3]}


def test_the_continuous_world_trace_records_what_the_analyses_read(tmp_path):
    import argparse
    import jax
    import jax.numpy as jnp
    from lifesim import env as E
    from lifesim.config import EnvConfig
    from continuous_world import Trace
    cfg = EnvConfig(grid_size=12, n_agents=3)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    tr = Trace(cfg)
    for t in range(4):
        a = jnp.array([E.MOVE_E, E.REST, E.REST])
        msg = jnp.ones((3, E.MSG_LEN), jnp.int32)
        s, _, _, info = E.step(s, a, cfg, msg)
        tr.add(s, a, msg, info)
    tr.save(tmp_path / "trace.npz", argparse.Namespace(map_seed=21, seed=3))
    d = np.load(tmp_path / "trace.npz")
    assert d["pos"].shape == (4, 3, 2) and d["action"][0, 0] == E.MOVE_E and int(d["map_seed"]) == 21
    assert (d["msg"][~d["spoke"]] == 0).all() and d["ctx"].shape == (4, 3, 3)
