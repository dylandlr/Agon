"""Trees (lifesim/trees.py): kinds by climate, in stands; they grow, fall when old (leaving wood),
burn back, bear fruit and nuts by kind; agents see only height and leaf tone."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import trees as T
from lifesim.config import EnvConfig, living_world


def test_cold_ground_grows_conifers_and_warm_ground_olives():
    g = 24
    forest = jnp.ones((g, g), bool)
    cold, _ = T.plant(jax.random.PRNGKey(0), forest, jnp.full((g, g), 1.0), jnp.zeros((g, g)))
    warm, _ = T.plant(jax.random.PRNGKey(0), forest, jnp.full((g, g), 18.0), jnp.zeros((g, g)))
    wet, _ = T.plant(jax.random.PRNGKey(0), forest, jnp.full((g, g), 10.0), jnp.ones((g, g)))
    share = lambda k, names: np.isin(np.asarray(k), [T.NAMES.index(n) for n in names]).mean()
    assert share(cold, ("spruce", "pine", "juniper", "birch")) > 0.7
    assert share(warm, ("wild olive",)) > 0.5
    assert share(wet, ("willow",)) > 0.3
    assert len(np.unique(np.asarray(cold))) > 1                       # stands, not one kind everywhere


def test_a_stand_grows_falls_when_old_and_burns_back():
    oak = jnp.array([[T.NAMES.index("oak")]], jnp.int32)
    h = [float(T.height(oak, jnp.array([[a]]))[0, 0]) for a in (5.0, 60.0, 300.0)]
    assert h[0] < h[1] < h[2] <= 35.0 and h[1] > 0.5 * 35.0
    life = T.SPECIES[T.NAMES.index("oak")]["life"]
    age, fell = T.age_step(oak, jnp.array([[life - 0.001]]), 0.01, jnp.zeros((1, 1), bool))
    assert bool(fell[0, 0]) and float(age[0, 0]) == 0.0
    age, fell = T.age_step(oak, jnp.array([[100.0]]), 0.01, jnp.ones((1, 1), bool))
    assert float(age[0, 0]) == 0.0 and not bool(fell[0, 0])


def test_fruit_and_nut_trees_bear_when_grown():
    apple, hazel, pine = (jnp.array([[T.NAMES.index(n)]], jnp.int32) for n in ("wild apple", "hazel", "pine"))
    old, young = jnp.array([[90.0]]), jnp.array([[1.0]])
    assert float(T.crop(apple, old)[0, 0, 1]) > 1.5 > float(T.crop(apple, young)[0, 0, 1])
    assert float(T.crop(hazel, old)[0, 0, 2]) > 1.5 and float(T.crop(pine, old)[0, 0, 1]) == 0.5


def test_agents_see_a_stands_height_and_leaves_and_the_world_wires_it():
    cfg = living_world(EnvConfig(grid_size=20, n_agents=1))
    s = E.reset(jax.random.PRNGKey(2), cfg)
    k = np.asarray(s.tree_kind)
    assert ((k >= 0) == (np.asarray(s.terrain) == E.FOREST)).all()
    looks = np.asarray(T.looks(s.tree_kind, s.tree_age))
    assert (looks[k < 0] == 0).all() and (looks[k >= 0, 0] > 0).any()
    s2, _, _, _ = E.step(s, jnp.array([E.REST]), cfg)
    assert np.allclose(np.asarray(s2.tree_age)[k >= 0] - np.asarray(s.tree_age)[k >= 0],
                       cfg.tick_minutes / 1440.0 / cfg.days_per_year, atol=1e-3)
