"""Archetypes (injected temperaments), lifetime records and the emergent-role analysis."""
import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

from lifesim import archetypes as arch
from lifesim import env as E
from lifesim.config import EnvConfig, with_reproduction


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_and_unknown_names():
    assert arch.parse("tyrant:1, saint:2") == (("tyrant", 1), ("saint", 2))
    assert arch.parse("") == () and arch.parse(None) == ()
    with pytest.raises(ValueError):
        arch.parse("wizard:1")
    assert arch.slots((("tyrant", 1), ("saint", 2))) == ["tyrant", "saint", "saint"]


def test_archetypes_set_founders_traits_and_only_theirs():
    cfg = with_reproduction(EnvConfig(grid_size=16, n_agents=6, archetypes=arch.parse("tyrant:1,matriarch:1,saint:1")),
                            n_founders=4)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    g = s.genotype
    assert float(g.vengefulness[0]) == 1.0 and float(g.empathy[0]) == 0.0 and np.isclose(float(g.strength[0]), 1.6)
    assert float(g.kin_care[1]) == 1.0 and int(s.sex[1]) == E.FEMALE              # the matriarch is a woman
    assert float(g.selflessness[2]) == 1.0
    plain = E.reset(jax.random.PRNGKey(0), with_reproduction(EnvConfig(grid_size=16, n_agents=6), n_founders=4))
    assert float(g.empathy[3]) == float(plain.genotype.empathy[3])                # others keep their draw
    with pytest.raises(ValueError):
        E.reset(jax.random.PRNGKey(0), with_reproduction(
            EnvConfig(grid_size=16, n_agents=6, archetypes=arch.parse("tyrant:5")), n_founders=4))


def test_deaths_are_logged_by_season():
    cfg = EnvConfig(grid_size=16, n_agents=2)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    _, _, _, info = E.step(s, jax.numpy.zeros(2, jax.numpy.int32), cfg)
    assert int(info.season) == int(E.season_of(s.t, cfg))


def test_role_analysis_finds_groups_and_places_injected_archetypes():
    ana = _script("archetypes")
    rng = np.random.default_rng(0)
    behaviours = ["strikes_on_agents", "gifts", "gathers", "company"]

    def life(kind, i, archetype=None):
        base = {"fighter": [80, 1, 20, 300], "giver": [1, 60, 40, 400], "forager": [2, 2, 150, 100]}[kind]
        counts = {b: max(0, int(v + rng.normal(0, 3))) for b, v in zip(behaviours, base)}
        return {"slot": i, "ticks": 1000, "counts": counts, "archetype": archetype, "run": "x",
                "traits": {"empathy": 0.1 if kind == "fighter" else 0.8, "vengefulness": 0.9 if kind == "fighter" else 0.1}}

    lives = [life(k, i) for i, k in enumerate(["fighter"] * 8 + ["giver"] * 8 + ["forager"] * 8)]
    lives.append(life("fighter", 99, archetype="tyrant"))
    res = ana.analyse(lives, 3)
    roles = {g["role"].split(" / ")[0] for g in res["groups"]}
    assert {"fighter", "giver", "gatherer"} <= roles
    assert res["injected"][0]["archetype"] == "tyrant" and res["injected"][0]["landed_in"].startswith("fighter")
    fighters = next(g for g in res["groups"] if g["role"].startswith("fighter"))
    assert fighters["temperament"]["vengefulness"] > 0.5


def test_experiment_has_the_archetype_intervention():
    ex = _script("experiment")
    names = [n for n, _ in ex.items("archetypes")]
    assert len(names) == 6 and names[0] == "tyrant_s0" and names[-1] == "saint_s2"


def test_every_archetype_stays_in_range_and_says_what_it_tests():
    from lifesim import genotype as gt
    assert len(arch.PROFILES) >= 17 and set(arch.ABOUT) == set(arch.PROFILES)
    for name, prof in arch.PROFILES.items():
        for trait, v in prof.items():
            assert trait in gt._FIELDS, (name, trait)
            lo, hi = gt._BOUNDS.get(trait, (0.25, 4.0))
            assert lo <= v <= hi, (name, trait, v)


def test_learner_traits_of_an_archetype_reach_the_trainer():
    """A sage learns more from watching: social_learning is read by the learner, not the world,
    so the trainer's population must carry it too."""
    from lifesim.ppo_multi import MAConfig, make_train
    cfg = with_reproduction(EnvConfig(grid_size=12, n_agents=4, archetypes=arch.parse("sage:1")), n_founders=2)
    _, init, _ = make_train(cfg, MAConfig(num_worlds=1, num_steps=4, encoder="mlp", hidden=8, num_minibatches=1))
    _, pop = init(jax.random.PRNGKey(0))
    assert float(pop.genotype.social_learning[0]) == 1.0 and np.isclose(float(pop.genotype.lr_scale[0]), 1.8)


def test_experiment_places_archetypes_and_builds_replicate_worlds():
    import argparse
    ex = _script("experiment")
    a = argparse.Namespace(which="chief,warrior", replicates=3)
    placed = [n for n, _ in ex.items("placed", a)]
    assert placed == ["none_s21", "chief_s21", "warrior_s21", "none_s22", "chief_s22", "warrior_s22"]
    cul = ex.items("culture", a)
    assert [n for n, _ in cul] == ["social_r0", "social_r1", "social_r2", "solitary_r0", "solitary_r1", "solitary_r2"]
    assert all("--map-seed" in x and "21" in x for _, x in cul)
    assert ["--imitation", "0"] == cul[-1][1][-2:] and "--imitation" not in cul[0][1]
    with pytest.raises(SystemExit):
        ex.items("placed", argparse.Namespace(which="nobody", replicates=1))
