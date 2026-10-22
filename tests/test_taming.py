"""Taming (EnvConfig.taming) is emergent: no act tames. An animal comes to be at ease with a person
who feeds it and does it no harm; then it neither flees nor hunts them, a tame predator stands guard
beside them, and one closer still follows them. A blow breaks it; some species never take to it."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import fauna as fa
from test_fauna import CFG, KEYS, _world, put

AWAKE = dataclasses.replace(CFG, animal_sleep=0.0)      # (no sleep bouts: every test at any hour)


def _cells(land, k):
    """k land cells in a row, west to east."""
    r, c = next((r, c) for r, c in land if all((r, c + j) in land for j in range(k)))
    return [(r, c + j) for j in range(k)]


def _far(land, cell):
    return next(p for p in land if max(abs(p[0] - cell[0]), abs(p[1] - cell[1])) > 10)


def _with_bond(an, i, agent, value):
    return an._replace(bond=an.bond.at[i, agent].set(value))


def test_a_hungry_wolf_fed_meat_comes_to_be_at_ease_with_who_dropped_it():
    s, land = _world()
    (cell,) = _cells(land, 1)
    far = _far(land, cell)
    s = s._replace(pos=jnp.asarray([far, far], jnp.int32), predators=put(s.predators, 0, "wolf", cell))
    step = jax.jit(lambda st: E.step(st, jnp.full(2, E.REST, jnp.int32), AWAKE))
    bonds, tamed_at = [], None
    for k in range(6):
        # agent 0 left meat on the wolf's cell a moment ago; the wolf is hungry again
        st = s._replace(meat=s.meat.at[cell].set(2.0), drop_owner=s.drop_owner.at[cell].set(0),
                        drop_time=s.drop_time.at[cell].set(s.t), predators=s.predators._replace(rest=s.predators.rest.at[0].set(0)))
        s, _, _, info = step(st)
        bonds.append(float(s.predators.bond[0, 0]))
        if tamed_at is None and int(info.tamed[0]) == fa.IDX["wolf"]:
            tamed_at = k
    assert abs(bonds[0] - AWAKE.tame_feed) < 0.01            # a wolf takes to people fastest (tameable 1)
    assert all(b2 > b1 for b1, b2 in zip(bonds, bonds[1:]))  # each meal more
    assert float(s.predators.bond[0, 1]) == 0.0              # nothing with the one who gave nothing
    assert tamed_at == 3                                     # 0.2, 0.36, 0.49, 0.59: at ease on the 4th meal


def test_a_snake_never_takes_to_people():
    s, land = _world()
    (cell,) = _cells(land, 1)
    far = _far(land, cell)
    s = s._replace(pos=jnp.asarray([far, far], jnp.int32), predators=put(s.predators, 0, "snake", cell),
                   meat=s.meat.at[cell].set(2.0), drop_owner=s.drop_owner.at[cell].set(0),
                   drop_time=s.drop_time.at[cell].set(s.t))
    s2, _, _, _ = E.step(s, jnp.full(2, E.REST, jnp.int32), AWAKE)
    assert float(s2.predators.bond[0, 0]) == 0.0


def test_a_wolf_at_ease_with_someone_does_not_attack_them():
    s, land = _world()
    a, w = _cells(land, 2)
    s = s._replace(pos=jnp.asarray([a, _far(land, a)], jnp.int32), predators=put(s.predators, 0, "wolf", w))

    def attacks(st):
        info = jax.vmap(lambda k: E.step(st._replace(key=k), jnp.full(2, E.REST, jnp.int32), AWAKE))(KEYS)[3]
        return np.asarray(info.attacked_by_predator[:, 0]).mean()
    wild = attacks(s)
    tame = attacks(s._replace(predators=_with_bond(s.predators, 0, 0, 0.9)))
    assert wild > 0.1 and tame == 0.0, (wild, tame)


def test_a_tame_wolf_beside_its_person_stands_off_a_wild_one():
    s, land = _world()
    w_wild, a, w_tame = _cells(land, 3)
    s = s._replace(pos=jnp.asarray([a, _far(land, a)], jnp.int32),
                   predators=put(put(s.predators, 0, "wolf", w_wild), 1, "wolf", w_tame))

    def attacked(st):
        info = jax.vmap(lambda k: E.step(st._replace(key=k), jnp.full(2, E.REST, jnp.int32), AWAKE))(KEYS)[3]
        return np.asarray(info.attacked_by_predator[:, 0]).mean()
    # two wild wolves: a pack, the lone person is attacked; the second at ease with the person: a guard
    pack = attacked(s)
    guarded = attacked(s._replace(predators=_with_bond(s.predators, 1, 0, 0.9)))
    assert pack > 0.1 and guarded < 0.25 * pack, (pack, guarded)


def test_a_rabbit_at_ease_does_not_run_and_one_close_follows():
    s, land = _world()
    a, _, r = _cells(land, 3)                                 # a person, an empty cell, a rabbit (it notices at 2)
    s = s._replace(pos=jnp.asarray([a, _far(land, a)], jnp.int32), small=put(s.small, 0, "rabbit", r))

    def east(st):
        out = jax.vmap(lambda k: E.step(st._replace(key=k), jnp.full(2, E.REST, jnp.int32), AWAKE))(KEYS)[0]
        return (np.asarray(out.small.pos[:, 0, 1]) - r[1]).mean()
    alone = east(s._replace(pos=jnp.asarray([_far(land, a)] * 2, jnp.int32)))   # no one near: it grazes
    wild = east(s)
    calm = east(s._replace(small=_with_bond(s.small, 0, 0, 0.55)))
    follow = east(s._replace(small=_with_bond(s.small, 0, 0, 0.9)))
    assert wild > alone + 0.2, (wild, alone)                 # wild: it runs (east, away)
    assert abs(calm - alone) < 0.1, (calm, alone)            # at ease: it goes about its grazing
    assert follow < alone - 0.3, (follow, alone)             # close: it comes west, to the person


def test_a_blow_breaks_the_bond():
    s, land = _world()
    a, r = _cells(land, 2)
    s = s._replace(pos=jnp.asarray([a, _far(land, a)], jnp.int32),
                   large=_with_bond(_with_bond(put(s.large, 0, "deer", r), 0, 0, 0.9), 0, 1, 0.9))
    east = E.STRIKE_N + 2                                     # strike east (MOVE_DELTAS[3] = (0, +1))
    assert tuple(np.asarray(E.MOVE_DELTAS[3])) == (0, 1)
    s2, _, _, _ = E.step(s, jnp.asarray([east, E.REST], jnp.int32), AWAKE)
    assert bool(s2.large.alive[0])                            # (a deer survives one blow)
    b = np.asarray(s2.large.bond[0])
    assert b[0] < 0.2 and 0.4 < b[1] < 0.9, b                # the striker most, everyone some


def test_bonds_fade_apart_and_reset_with_a_new_animal():
    s, land = _world()
    (cell,) = _cells(land, 1)
    far = _far(land, cell)
    s = s._replace(pos=jnp.asarray([far, far], jnp.int32), large=_with_bond(put(s.large, 0, "deer", cell), 0, 0, 0.8))
    s2, _, _, _ = E.step(s, jnp.full(2, E.REST, jnp.int32), AWAKE)
    half = AWAKE.tame_half_life_days * AWAKE.day_length
    assert abs(float(s2.large.bond[0, 0]) - 0.8 * 0.5 ** (1 / half)) < 1e-4
    dead = s2._replace(large=s2.large._replace(alive=s2.large.alive.at[0].set(False)))
    reborn = E._respawn(jax.random.PRNGKey(0), dead.large, dead.terrain, 1.0, AWAKE, E.species_params(AWAKE)["hp"])
    assert float(reborn.bond[0].max()) == 0.0
