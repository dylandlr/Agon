"""The dead: what one carries falls where one dies; the body decomposes, faster in warmth, giving off
rot (sickness to those beside it, flies, foul ground) until only bones remain; earth or stone heaped
on it, or fire, ends that. Nothing tells anyone to bury."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import minerals as mn
from lifesim.config import EnvConfig, living_world

CFG = living_world(EnvConfig(grid_size=16, n_agents=2, climate_volatility=0.0, lightning_prob=0.0,
                             season_temp=(20.0,) * 4, day_temp_swing=0.0, small_prey_per_1000=0.0,
                             large_prey_per_1000=0.0, predators_per_1000=0.0, predator_respawn=0.0,
                             small_prey_respawn=0.0, large_prey_respawn=0.0))
KEYS = jax.random.split(jax.random.PRNGKey(0), 300)


def _two(cfg=CFG, apart=False):
    """Agent 0 dead on a grass cell, agent 1 alive beside it (or far off)."""
    s = E.reset(jax.random.PRNGKey(1), cfg)
    t = np.asarray(s.terrain)
    cells = [(r, c) for r in range(2, 14) for c in range(2, 13) if t[r, c] == E.GRASS and t[r, c + 1] == E.GRASS]
    (r, c) = cells[0]
    other = (r, c + 1) if not apart else next(p for p in cells if max(abs(p[0] - r), abs(p[1] - c)) > 4)
    s = s._replace(pos=jnp.asarray([(r, c), other], jnp.int32), alive=jnp.array([False, True]),
                   existed=jnp.array([True, True]), infection=jnp.zeros(2, jnp.int32),
                   immunity=jnp.zeros(2), conditions=jnp.zeros_like(s.conditions))
    return s, (r, c)


def test_what_one_carries_falls_where_one_dies():
    s, (r, c) = _two()
    s = s._replace(alive=jnp.array([True, True]), health=jnp.array([-1.0, 1.0]),
                   inv=s.inv.at[0, E.WOOD].set(2.0).at[0, E.MIN0 + mn.IDX["flint"]].set(1.0))
    wood0, flint0 = float(s.wood[r, c]), float(s.minerals[r, c, mn.IDX["flint"]])
    s2, _, _, info = E.step(s, jnp.array([E.REST, E.REST]), CFG)
    assert bool(info.died[0]) and float(s2.inv[0].sum()) == 0.0
    assert float(s2.wood[r, c]) >= wood0 + 2.0 - 1e-5 and abs(float(s2.minerals[r, c, mn.IDX["flint"]]) - flint0 - 1.0) < 1e-5


def test_a_body_rots_faster_warm_and_hardly_at_all_frozen():
    rates = {}
    for temp in (30.0, 20.0, -10.0):
        cfg = dataclasses.replace(CFG, season_temp=(temp,) * 4)
        s, _ = _two(cfg, apart=True)
        s2, _, _, _ = E.step(s, jnp.array([E.REST, E.REST]), cfg)
        rates[temp] = float(s2.rot[0])
    assert abs(rates[30.0] / rates[20.0] - 2.0) < 0.2                 # twice as fast 10 degrees warmer
    assert rates[-10.0] < 0.1 * rates[20.0]


def test_a_rotting_body_sickens_those_beside_it_unless_covered():
    s, (r, c) = _two()
    s = s._replace(rot=jnp.array([0.5, 0.0]))
    hot = dataclasses.replace(CFG, corpse_infection=0.3)          # (the default is far rarer: a mechanism test)

    def sick(st):
        out = jax.vmap(lambda k: E.step(st._replace(key=k), jnp.array([E.REST, E.REST]), hot))(KEYS)[0]
        return (np.asarray(out.infection[:, 1]) > 0).mean()
    rotting = sick(s)
    covered = sick(s._replace(stones=s.stones.at[r, c].set(CFG.bury_cover)))
    fresh = sick(s._replace(rot=jnp.array([0.02, 0.0])))
    assert float(E.corpse_hazard(s, CFG)[0]) == 1.0
    assert rotting > 0.1 and covered < 0.25 * rotting and fresh < 0.25 * rotting, (rotting, covered, fresh)
    assert float(E.corpse_hazard(s._replace(stones=s.stones.at[r, c].set(CFG.bury_cover)), CFG)[0]) == 0.0


def test_fire_burns_a_body_to_bones_and_rot_ends_in_bones():
    s, (r, c) = _two(apart=True)
    s = s._replace(rot=jnp.array([0.3, 0.0]), fire=s.fire.at[r, c].set(10.0))
    bone0 = float(s.minerals[r, c, mn.IDX["bone"]])
    s2, _, _, _ = E.step(s, jnp.array([E.REST, E.REST]), CFG)
    assert float(s2.rot[0]) == 1.0 and float(s2.minerals[r, c, mn.IDX["bone"]]) >= bone0 + CFG.body_bones - 1e-5
    assert float(E.corpse_hazard(s2, CFG)[0]) == 0.0 and float(E.body_look(s2, CFG)[0]) == 0.0
