"""The society report (scripts/society.py) finds structure planted in a synthetic world, and does
not find it where there is none."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import society as S  # noqa: E402
from society_data import World  # noqa: E402

DAY = 48
DAYS = 24
T = DAY * DAYS
N = 12


def _lives(sex=None):
    return [{"slot": k, "born_t": 0, "died_t": None, "ticks": T, "generation": 0, "mother": -1, "father": -1,
             "sex": (sex[k] if sex else ("female" if k % 2 else "male")),
             "counts": {"eats": 100 + k, "nursing": 0}} for k in range(N)]


def _write(folder, pos, events, lives, action=None):
    folder.mkdir(parents=True, exist_ok=True)
    json.dump(lives, open(folder / "lives.json", "w"))
    with gzip.open(folder / "events.jsonl.gz", "wt") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    act = np.zeros((T, N), np.uint8) if action is None else action
    np.savez_compressed(folder / "trace.npz", pos=pos.astype(np.uint8), alive=np.ones((T, N), bool), action=act,
                        below=np.zeros((T, N), bool), day_length=np.int64(DAY))
    return World(folder, need_trace=True)


def _walk(rng, start, steps, lo=2, hi=60):
    p = np.array(start, int)
    out = []
    for _ in range(steps):
        p = np.clip(p + rng.integers(-1, 2, 2), lo, hi)
        out.append(p.copy())
    return np.array(out)


def _banded(rng):
    """Two bands of six, each wandering as one (members in their band's cell or beside it), sleeping
    at the same camp every night."""
    pos = np.zeros((T, N, 2), int)
    act = np.zeros((T, N), np.uint8)
    for band, start in ((0, (12, 12)), (1, (45, 45))):
        centre = _walk(rng, start, T, lo=4, hi=58)
        camp = np.array(start)
        for k in range(band * 6, band * 6 + 6):
            off = rng.integers(-1, 2, (T, 2))
            pos[:, k] = centre + off
            for d in range(DAYS):                         # nights at camp
                pos[d * DAY + 40:(d + 1) * DAY, k] = camp + (k % 2, 0)
                act[d * DAY + 40:(d + 1) * DAY, k] = S.SLEEP_ACTION
    return pos, act


def _loners(rng):
    pos = np.zeros((T, N, 2), int)
    for k in range(N):
        pos[:, k] = _walk(rng, rng.integers(5, 55, 2), T)
    return pos


def test_sleep_action_matches_the_env():
    from lifesim import env as E
    assert S.SLEEP_ACTION == E.SLEEP


def test_planted_bands_are_found_and_loners_form_none(tmp_path):
    rng = np.random.default_rng(0)
    pos, act = _banded(rng)
    w = _write(tmp_path / "banded", pos, [], _lives(), act)
    b = S.bands(w, 0, T, nulls=49)
    assert sorted(len(x) for x in b["bands"]) == [6, 6]
    assert b["gregarious_p"] < 0.05 and b["preferred_p"] < 0.01 and b["association_ratio"] > 3
    assert b["typical_group_size"] > 4 and b["alone"] < 0.2
    p = S.places(w, 0, T, b["_band_of"])
    assert p["sleep_site_fidelity"] > 0.9 and p["co_sleeping"] > 0.9
    assert p["between_band_meeting_hours"] < p["within_band_hours"] / 10
    lone = _write(tmp_path / "loners", _loners(np.random.default_rng(1)), [], _lives())
    b2 = S.bands(lone, 0, T, nulls=49)
    assert b2["preferred_p"] > 0.05 and b2["typical_group_size"] < 2 and b2["in_a_band"] < 0.5 and b2["bands"] == []


def _ev(t, k, kind, **kw):
    return {"id": f"{t}:{k}:{kind}", "t": t, "agent": k, "type": kind, "pos": kw.pop("pos", [10, 10]), **kw}


def test_division_of_labour_against_its_null(tmp_path):
    rng = np.random.default_rng(2)
    pos = _loners(rng)
    split = [_ev(t, k, "gather" if k < 6 else "strike_animal") for k in range(N) for t in range(0, 300, 10)]
    w = _write(tmp_path / "split", pos, split, _lives())
    lab = S.labour(w, 0, T, nulls=199)
    assert lab["dol"] > 0.9 and lab["dol_p"] < 0.01
    mixed = [_ev(t, k, ["gather", "strike_animal"][int(rng.integers(2))]) for k in range(N) for t in range(0, 300, 10)]
    w2 = _write(tmp_path / "mixed", pos, mixed, _lives())
    lab2 = S.labour(w2, 0, T, nulls=199)
    assert lab2["dol"] < 0.15 and lab2["dol_p"] > 0.05
    # men hunt, women gather: found by sex
    sexed = [_ev(t, k, "strike_animal" if k % 2 == 0 else "gather") for k in range(N) for t in range(0, 300, 10)]
    w3 = _write(tmp_path / "sexed", pos, sexed, _lives())
    by = S.labour(w3, 0, T, nulls=199)["by_sex"]
    assert by["gathering"]["female_share_minus_expected"] > 0.4 and by["gathering"]["p"] < 0.05


def test_sharing_reciprocity_and_kin(tmp_path):
    lives = _lives()
    for k in (6, 7):                                      # 6 and 7 are children of 0 and 1
        lives[k]["born_t"], lives[k]["mother"], lives[k]["father"] = 5, 1, 0
    pairs = [(0, 6), (6, 0), (1, 7), (7, 1), (2, 3), (3, 2), (4, 5), (5, 4)]
    evs = [_ev(t, b, "took_food", giver=a) for t in range(10, 400, 20) for a, b in pairs]
    w = _write(tmp_path / "gifts", _loners(np.random.default_rng(3)), evs, lives)
    r_of = S.relatedness(w)
    assert r_of(0, 6) == pytest.approx(0.5) and r_of(6, 7) == pytest.approx(0.5) and r_of(2, 3) == 0.0
    s = S.sharing(w, 0, T, r_of, perms=199)
    assert s["reciprocity_r"] > 0.9 and s["reciprocity_p"] < 0.05 and s["share_to_kin"] == pytest.approx(0.5)


def test_half_siblings_and_grandparents():
    class W:                                               # just a pedigree
        def __init__(self):
            self.lives = [{"id": i, "slot": i, "born_t": b, "died_t": None, "mother": m, "father": f}
                          for i, (b, m, f) in enumerate([(0, -1, -1), (0, -1, -1), (0, -1, -1), (5, 1, 0), (6, 2, 0), (20, 3, 2)])]

        def life_at(self, slot, t):
            return self.lives[slot]
    r = S.relatedness(W())
    assert r(3, 4) == pytest.approx(0.25)                  # half siblings (father 0)
    assert r(5, 0) == pytest.approx(0.25)                  # grandchild through 3
    assert r(5, 3) == pytest.approx(0.5)                   # mother (2, its father, is unrelated to 3)


def test_punishment_and_joint_hunts(tmp_path):
    rng = np.random.default_rng(4)
    evs = []
    for n, t in enumerate(range(20, 500, 25)):
        a, b, c = n % 4, 4 + n % 4, 8 + n % 4
        evs.append(_ev(t, a, "strike_agent", victims=[b]))
        evs.append(_ev(t + 3, c, "strike_agent", victims=[a]))      # a third party answers
    for t in range(30, 500, 60):                                     # joint hunts of large prey
        evs.append(_ev(t - 1, 9, "strike_animal", pos=[20, 20]))
        evs.append(_ev(t, 10, "kill", animal="large_prey", pos=[20, 21]))
    w = _write(tmp_path / "punish", _loners(rng), evs, _lives())
    p = S.punishment(w, 0, T, S.relatedness(w))
    assert p["punishment_ratio"] > 3 and p["p"] < 0.01
    h = S.hunting(w, 0, T)
    assert h["joint_share"] == 1.0 and h["hunters_per_kill"] == 2.0


def test_report_runs_end_to_end(tmp_path):
    pos, act = _banded(np.random.default_rng(5))
    w = _write(tmp_path / "all", pos, [_ev(10, 0, "took_food", giver=1)], _lives(), act)
    r = S.analyse(w, window_days=12, nulls=19)
    assert len(r["windows"]) == 2 and r["band_stability"] is not None and r["band_stability"] > 0.8
    assert any("Society" in line for line in S.render(r))


def test_punishment_test_is_calibrated_on_random_blows(tmp_path):
    """Blows at random: the test should cry punishment about 5% of the time, and the ratio sit near 1."""
    ps, ratios = [], []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        evs = []
        for t in sorted(rng.choice(np.arange(10, T - 10), 120, replace=False)):
            a, b = rng.choice(N, 2, replace=False)
            evs.append(_ev(int(t), int(a), "strike_agent", victims=[int(b)]))
        w = _write(tmp_path / f"random{seed}", _loners(rng), evs, _lives())
        p = S.punishment(w, 0, T, lambda a, b: 0.0, perms=199, seed=seed)
        ps.append(p["p"]); ratios.append(p["punishment_ratio"])
    assert np.mean(np.array(ps) < 0.05) <= 0.15 and 0.85 < np.mean(ratios) < 1.15
