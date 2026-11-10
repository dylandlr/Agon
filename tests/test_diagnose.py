"""The shelter-distance helper in scripts/diagnose.py: Chebyshev distance by repeated dilation."""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import diagnose as DG  # noqa: E402


def test_distance_to_nearest_true_cell():
    G = 40
    mask = np.zeros((2, G, G), bool)
    mask[0, 5, 5] = True                      # agent 0's only shelter
    mask[1, 30, 30] = True
    pos = jnp.array([[5, 5], [3, 9]])
    d = np.asarray(DG.make_dist_fn(G, 2)(jnp.asarray(mask), pos))
    assert d[0] == 0                          # standing on it
    assert d[1] == DG.RMAX + 1                # (30,30) is 27 away: beyond the look-out radius


def test_chebyshev_not_manhattan():
    G = 20
    mask = np.zeros((1, G, G), bool)
    mask[0, 10, 10] = True
    d = np.asarray(DG.make_dist_fn(G, 1)(jnp.asarray(mask), jnp.array([[7, 13]])))
    assert d[0] == 3                          # diagonal steps count as one


def test_families_cover_distinct_actions():
    seen = set()
    for ix in DG.FAMILIES.values():
        assert not (seen & set(ix))
        seen |= set(ix)


def _fake_run(in_cover, dknown, dtrue):
    from types import SimpleNamespace
    T, N = 100, 2
    bt = np.zeros((T, N), np.float32)
    bt[40:, 0] = -0.6                                    # agent 0 is cold from tick 40 until it dies at 90
    cover = np.zeros((T, N), bool)
    cover[40:, 0] = in_cover
    rec = {"alive": np.ones((T, N), bool), "asleep": np.zeros((T, N), bool), "bt": bt, "act": np.zeros((T, N), np.int16),
           "cover": cover, "dtrue": np.full((T, N), dtrue, np.int8), "dknown": np.full((T, N), dknown, np.int8),
           "night": np.zeros(T, bool), "season": np.full(T, 3), "huddling": np.zeros((T, N), bool)}
    death = {"t": 90, "agent": 0, "cause": "cold", "age": 5000, "birth_age": 0, "season": 3, "night": False,
             "in_cover": in_cover, "asleep": False}
    run = {"kind": "policy", "variant": "policy", "seed": 0, "rec": rec, "deaths": [death], "probes": [],
           "start_alive": 2, "alive_end": 1, "alive_ticks": 200, "awake_ticks": 200, "counts": {}}
    cfg = SimpleNamespace(day_length=48, adult_years=15.0, ticks_per_bio_year=48.0, year_length=1920)
    return cfg, [run]


def test_cold_death_bins():
    for in_cover, dk, dt, want in ((False, 5, 5, "knew shelter"), (False, 13, 8, "never visited"),
                                   (False, 13, 13, "no shelter"), (True, 0, 0, "froze inside")):
        cfg, R = _fake_run(in_cover, dk, dt)
        out = DG.sec_cold(cfg, R)
        assert out["n"] == 1 and any(want in k for k in out["bins"]), (want, out["bins"])
        x = out["rows"][0]
        assert abs(x["exposure_hours"] - 25.5) < 1e-6      # ticks 40..90 at 30 minutes each


def test_survival_counts_cold_by_shelter():
    cfg, R = _fake_run(False, 5, 5)
    d = DG.sec_survival(cfg, R)
    assert d["policy"]["cold_open"] == 1 and d["policy"]["cold_covered"] == 0


def test_learning_panel_reads_history(tmp_path, capsys):
    import json
    rows = [{"agent_steps": 1e6 * (i + 1), "survival_score_per_world_day": 1.0 + (0.5 if i < 6 else 0.0) * (i / 6),
             "mean_age_at_death": 100 + i, "deaths": {"cold": 5}, "cold_deaths": {"open_adult": 3},
             "entropy": 1.5, "world_model": {"trust_in_ideas": 0.001, "reward_error": 0.1, "own_state_error": 0.2},
             "per_output": {"action": {"clip_fraction": 0.1, "approx_kl": 0.01}}} for i in range(12)]
    (tmp_path / "history.json").write_text(json.dumps({"args": {}, "history": rows}))
    out = DG.sec_learn(tmp_path)
    text = capsys.readouterr().out
    assert "survival score per world-day" in text and "flat" in text      # the score stopped rising in the last third
    assert "survival_score_per_world_day" in out


def _fake_needs_run(T=100, N=2, *, kind="policy", hyd_low_from=40, atwater=False, dwater=5, dwater_known=13,
                    act=0, death=None, dpred=None, extra=None):
    from types import SimpleNamespace
    hyd = np.ones((T, N), np.float16)
    hyd[hyd_low_from:, 0] = 0.1
    rec = {"alive": np.ones((T, N), bool), "asleep": np.zeros((T, N), bool), "act": np.full((T, N), act, np.int16),
           "cover": np.zeros((T, N), bool), "hyd": hyd, "sat": np.ones((T, N), np.float16),
           "dwater": np.full((T, N), dwater, np.int8), "dwater_known": np.full((T, N), dwater_known, np.int8),
           "forage": np.full((T, N), 0.5, np.float16), "holding": np.zeros((T, N), bool),
           "atwater": np.full((T, N), atwater, bool),
           "dpred": np.full((T, N), 99, np.int8) if dpred is None else dpred,
           "night": np.zeros(T, bool), "season": np.zeros(T, int)}
    rec.update(extra or {})
    run = {"kind": kind, "variant": "policy", "seed": 0, "rec": rec, "deaths": [death] if death else [], "probes": [],
           "start_alive": N, "alive_end": N - bool(death), "alive_ticks": 200, "awake_ticks": 200, "counts": {}}
    cfg = SimpleNamespace(day_length=48, adult_years=15.0, ticks_per_bio_year=48.0, year_length=1920)
    return cfg, [run]


def _death(cause, t=90):
    return {"t": t, "agent": 0, "cause": cause, "age": 5000, "birth_age": 0, "season": 1, "night": False,
            "in_cover": False, "asleep": False}


def test_thirst_death_bins():
    for atw, dw, dk, want in ((True, 1, 13, "reached water while low"), (False, 13, 13, "no water within"),
                              (False, 6, 13, "never been near it"), (False, 6, 5, "knew water within")):
        cfg, R = _fake_needs_run(atwater=atw, dwater=dw, dwater_known=dk, death=_death("thirst"))
        out = DG.sec_deaths(cfg, R)
        assert any(want in k for k in out["thirst"]["bins"]), (want, out["thirst"]["bins"])
        assert abs(out["thirst"]["rows"][0]["hours"] - 25.5) < 1e-6      # ticks 40..90 at 30 minutes each


def test_predation_death_bins():
    from lifesim import env as E
    T = 100
    for dp_from, act, want in ((89, E.REST, "ambush"), (60, E.REST, "never chose FLEE"), (60, E.FLEE, "chose FLEE and was caught")):
        dpred = np.full((T, 2), 99, np.int8)
        dpred[dp_from:, 0] = 2
        cfg, R = _fake_needs_run(death=_death("predation"), dpred=dpred, act=act)
        out = DG.sec_deaths(cfg, R)
        assert any(want in k for k in out["predation"]["bins"]), (want, out["predation"]["bins"])


def test_needs_table_separates_beside_water_from_far(capsys):
    from lifesim import env as E
    # policy: thirsty from tick 40; beside water it drinks, far from water it rests. random never drinks.
    cfg, Rp = _fake_needs_run(atwater=True, dwater=1, act=E.DRINK)
    cfg, Rr = _fake_needs_run(kind="random", atwater=True, dwater=1, act=E.REST)
    out = DG.sec_needs(cfg, Rp + Rr)
    text = capsys.readouterr().out
    pol = {d["bucket"]: d for d in out["thirst"] if d["who"] == "policy"}
    rnd = {d["bucket"]: d for d in out["thirst"] if d["who"] == "random"}
    assert pol["at water (can drink)"]["ticks"] == 60 and pol["at water (can drink)"]["rates"]["drink"] == 1.0
    assert rnd["at water (can drink)"]["rates"]["drink"] == 0.0
    assert "while a need is low" in text.lower()


def test_starvation_death_bins_follow_the_food_chain():
    from lifesim import env as E
    T = 100
    for hold, forage, want in ((True, 0.5, "held food while starving"), (False, 0.5, "stood on good forage"),
                               (False, 0.1, "never on good forage")):
        cfg, R = _fake_needs_run(death=_death("starvation"), hyd_low_from=100)
        rec = R[0]["rec"]
        rec["sat"] = np.ones((T, 2), np.float16)
        rec["sat"][40:, 0] = 0.1
        rec["holding"][:, 0] = hold
        rec["forage"][:, 0] = forage
        out = DG.sec_deaths(cfg, R)
        assert any(want in k for k in out["starvation"]["bins"]), (want, out["starvation"]["bins"])


def test_hunger_table_splits_holding_from_empty_handed(capsys):
    from lifesim import env as E
    cfg, Rp = _fake_needs_run(act=E.EAT, hyd_low_from=100)
    rec = Rp[0]["rec"]
    rec["sat"] = np.ones((100, 2), np.float16)
    rec["sat"][40:, 0] = 0.1
    rec["holding"][:, 0] = True                              # holding food and choosing EAT
    cfg, Rr = _fake_needs_run(kind="random", act=E.REST, hyd_low_from=100)
    Rr[0]["rec"]["sat"] = rec["sat"].copy()
    out = DG.sec_needs(cfg, Rp + Rr)
    capsys.readouterr()
    pol = {d["bucket"]: d for d in out["hunger"] if d["who"] == "policy"}
    assert pol["holding something edible"]["ticks"] == 60 and pol["holding something edible"]["rates"]["eat"] == 1.0
