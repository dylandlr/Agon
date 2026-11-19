"""Room (EnvConfig.cell_room): each cell holds so much body. A step into a cell without room fails;
those lying down take twice the room; animals take room by kind; the observation shows how full."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig, living_world

CFG = living_world(EnvConfig(grid_size=16, n_agents=6, climate_volatility=0.0, lightning_prob=0.0,
                             season_temp=(18.0,) * 4, day_temp_swing=0.0, small_prey_per_1000=0.0,
                             large_prey_per_1000=0.0, predators_per_1000=0.0, predator_respawn=0.0,
                             small_prey_respawn=0.0, large_prey_respawn=0.0))
EAST = E.MOVE_N + 2                                        # MOVE_DELTAS[3] = (0, +1)


def _row(n_in_cell, asleep=()):
    """n_in_cell agents in cell B, agent 5 in cell A just west of it; the rest far off."""
    s = E.reset(jax.random.PRNGKey(3), CFG)
    t = np.asarray(s.terrain)
    a = next((r, c) for r in range(2, 14) for c in range(2, 12) if (t[r, c:c + 3] == E.GRASS).all())
    b = (a[0], a[1] + 1)
    far = [(r, c) for r in range(16) for c in range(16) if t[r, c] != E.WATER and max(abs(r - a[0]), abs(c - a[1])) > 4]
    pos = [b if i < n_in_cell else far[i] for i in range(5)] + [a]
    off = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    s = s._replace(pos=jnp.asarray(pos, jnp.int32), small=off(s.small), large=off(s.large), predators=off(s.predators),
                   genotype=s.genotype._replace(height=jnp.ones(6), build=jnp.ones(6)),
                   asleep=jnp.array([i in asleep for i in range(6)]),
                   birth_age=jnp.full(6, int(25 * CFG.ticks_per_bio_year), jnp.int32))
    return s, a, b


def _step_east(s, cfg=CFG):
    acts = jnp.full(6, E.REST, jnp.int32).at[5].set(EAST)
    return E.step(s, acts, cfg)


def test_a_full_cell_turns_a_step_away_and_one_with_room_does_not():
    s, a, b = _row(4)                                         # four standing adults: the cell is full
    s2, _, _, info = _step_east(s)
    assert tuple(np.asarray(s2.pos[5])) == a and bool(info.crowded[5])
    assert int(info.outcome[5]) == E.OUT_MOVE_FAILED
    s, a, b = _row(3)
    s2, _, _, info = _step_east(s)
    assert tuple(np.asarray(s2.pos[5])) == b and not bool(info.crowded[5])
    off = dataclasses.replace(CFG, cell_room=0.0)            # no limit
    s, a, b = _row(4)
    assert tuple(np.asarray(_step_east(s, off)[0].pos[5])) == b


def test_lying_down_takes_twice_the_room():
    s, a, b = _row(2, asleep=(0, 1))                          # two asleep: as full as four standing
    room = E.agent_room(s, CFG, E.body(s, CFG)["ideal_kg"])
    assert abs(float(room[0]) - 2 * float(room[2])) < 1e-5
    s2, _, _, info = _step_east(s)
    assert bool(info.crowded[5])


def test_two_cannot_squeeze_into_the_last_place():
    s, a, b = _row(3)                                         # room for one more
    c = (a[0], a[1] + 2)                                      # agent 4 east of the cell, stepping west into it
    s = s._replace(pos=s.pos.at[4].set(jnp.asarray(c, jnp.int32)))
    west = E.MOVE_N + 6
    acts = jnp.full(6, E.REST, jnp.int32).at[5].set(EAST).at[4].set(west)
    infos = jax.vmap(lambda k: E.step(s._replace(key=k), acts, CFG)[3])(jax.random.split(jax.random.PRNGKey(0), 40))
    crowded = np.asarray(infos.crowded)[:, [4, 5]]
    assert (crowded.sum(1) == 1).all()                        # exactly one gets in, either one
    assert 0 < crowded[:, 0].mean() < 1


def test_animals_take_room_and_it_is_seen():
    s, a, b = _row(2)
    s = s._replace(large=s.large._replace(alive=s.large.alive.at[0].set(True), species=s.large.species.at[0].set(
        E.fa.IDX["deer"]), pos=s.large.pos.at[0].set(jnp.asarray(b, jnp.int32)), below=s.large.below.at[0].set(False)))
    used = E.room_used(s, CFG, E.body(s, CFG)["ideal_kg"])
    assert abs(float(used[b]) - (2.0 * float(E.agent_room(s, CFG, E.body(s, CFG)["ideal_kg"])[0]) + 1.2)) < 1e-4
    o = np.asarray(E.observe(s, CFG))[5, :CFG.view_size ** 2 * E.N_OBS_CHANNELS].reshape(CFG.view_size, CFG.view_size, -1)
    r = CFG.view_radius
    assert abs(o[r, r + 1, E.N_TERRAIN + 2] - float(used[b]) / CFG.cell_room) < 1e-4    # how full the cell east is
