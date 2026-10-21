"""Animals and bugs: each species does what its kind does."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import fauna as fa
from lifesim.config import EnvConfig, living_world

CFG = living_world(EnvConfig(grid_size=24, n_agents=2, climate_volatility=0.0, lightning_prob=0.0,
                             season_temp=(18.0,) * 4, day_temp_swing=0.0))
SP = E.species_params(CFG)
KEYS = jax.random.split(jax.random.PRNGKey(0), 300)


def _world():
    s = E.reset(jax.random.PRNGKey(4), CFG)
    off = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    t = np.asarray(s.terrain)
    land = [(r, c) for r in range(3, 20) for c in range(3, 20)
            if (t[r - 1:r + 2, c - 1:c + 2] != E.WATER).all()]
    return s._replace(small=off(s.small), large=off(s.large), predators=off(s.predators),
                      conditions=jnp.zeros_like(s.conditions), t=jnp.int32(24),
                      birth_age=jnp.full(2, int(25 * CFG.ticks_per_bio_year), jnp.int32)), land


def put(an, i, name, cell, **kw):
    k = fa.IDX[name]
    an = an._replace(pos=an.pos.at[i].set(jnp.asarray(cell, jnp.int32)), alive=an.alive.at[i].set(True),
                     hp=an.hp.at[i].set(float(SP["hp"][k])), species=an.species.at[i].set(k),
                     rest=an.rest.at[i].set(0), anger=an.anger.at[i].set(kw.get("anger", 0)))
    return an


def _many(s, actions):
    """The same tick from 300 different random keys."""
    return jax.vmap(lambda k: E.step(s._replace(key=k), jnp.asarray(actions, jnp.int32), CFG))(KEYS)


def test_every_species_lives_in_its_habitat_and_slot_zero_keeps_the_classic_animal():
    s = E.reset(jax.random.PRNGKey(1), living_world(EnvConfig(grid_size=48, n_agents=2)))
    t = np.asarray(s.terrain)
    for an, first in ((s.small, "rabbit"), (s.large, "deer"), (s.predators, "wolf")):
        assert fa.NAMES[int(an.species[0])] == first
        for (r, c), k in zip(np.asarray(an.pos), np.asarray(an.species)):
            assert fa.SPECIES[k]["habitat"].count(E.TERRAIN_NAMES[t[r, c]]) == 1
    names = {fa.NAMES[int(k)] for an in (s.small, s.large, s.predators) for k in an.species}
    assert {"fish", "aurochs", "mammoth", "big cat", "bear"} <= names


def test_a_fish_dodges_bare_hands_far_more_than_a_spear_point():
    s, land = _world()
    t = np.asarray(s.terrain)
    shore = next((r, c, dr, dc) for r in range(1, 23) for c in range(1, 23) if t[r, c] != E.WATER
                 for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)) if t[r + dr, c + dc] == E.WATER)
    r, c, dr, dc = shore
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray((r, c))), small=put(s.small, 0, "fish", (r + dr, c + dc)))
    d = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)].index((dr, dc))
    hits = lambda st: float((np.asarray(_many(st, [E.STRIKE_N + d, E.REST])[0].small.hp[:, 0]) < 0.3).mean())
    bare = hits(s)
    spear = hits(s._replace(inv=s.inv.at[0, E.MIN0 + 1].set(1.0), hand=s.hand.at[0].set(E.MIN0 + 1)))   # obsidian
    assert spear > 1.5 * bare


def test_struck_boars_fight_back_and_deer_do_not():
    s, land = _world()
    a = land[0]
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(a)))
    tgt = (a[0], a[1] + 1)
    gored = lambda name: float((np.asarray(_many(s._replace(large=put(s.large, 0, name, tgt)),
                                                 [E.STRIKE_N + 2, E.REST])[3].gored[:, 0]) > 0).mean())
    assert 0.4 < gored("boar") < 0.8 and gored("deer") == 0.0


def test_a_bear_leaves_people_alone_unless_provoked_and_sleeps_through_winter():
    s, land = _world()
    a = land[0]
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(a)).at[1].set(jnp.asarray(land[-1])))
    near = (a[0], a[1] + 1)
    calm = _many(s._replace(predators=put(s.predators, 0, "bear", near)), [E.REST, E.REST])[3]
    angry = _many(s._replace(predators=put(s.predators, 0, "bear", near, anger=30)), [E.REST, E.REST])[3]
    assert float(calm.attacked_by_predator[:, 0].mean()) < 0.06 < 0.3 < float(angry.attacked_by_predator[:, 0].mean())
    winter = s._replace(t=jnp.int32(int(3.2 * CFG.year_length / 4)), predators=put(s.predators, 0, "bear", near, anger=30))
    assert float(_many(winter, [E.REST, E.REST])[3].attacked_by_predator[:, 0].mean()) == 0.0


def test_a_wolf_pack_overcomes_two_people_standing_together():
    s, land = _world()
    a = land[0]
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(a)).at[1].set(jnp.asarray((a[0] + 1, a[1]))), t=jnp.int32(0))  # night
    one = s._replace(predators=put(s.predators, 0, "wolf", (a[0], a[1] + 1)))
    pack = s._replace(predators=put(put(s.predators, 0, "wolf", (a[0], a[1] + 1)), 1, "wolf", (a[0] - 1, a[1])))
    att = lambda st: float(_many(st, [E.REST, E.REST])[3].attacked_by_predator.any(axis=1).mean())
    assert att(one) == 0.0 and att(pack) > 0.1


def test_bees_sting_robbers_unless_smoke_calms_them():
    s, land = _world()
    a = land[0]
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(a)), hive=s.hive.at[a].set(True), honey=s.honey.at[a].set(2.0),
                   hand=s.hand.at[0].set(E.HONEY))
    _, _, _, info = E.step(s, jnp.array([E.GRASP, E.REST]), CFG)
    assert bool(info.stung[0])
    smoky = s._replace(fire=s.fire.at[a[0], a[1] + 1].set(30.0), wood=s.wood.at[a[0], a[1] + 1].set(1.0))
    s2, _, _, info = E.step(smoky, jnp.array([E.GRASP, E.REST]), CFG)
    assert not bool(info.stung[0]) and float(s2.inv[0, E.HONEY]) > 0


def test_a_stick_draws_far_more_grubs_than_fingers():
    s, land = _world()
    a = land[0]
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(a)), mound=s.mound.at[a].set(True), grubs=s.grubs.at[a].set(3.0),
                   hand=s.hand.at[0].set(E.GRUBS))
    took = lambda st: float(E.step(st, jnp.array([E.GRASP, E.REST]), CFG)[0].inv[0, E.GRUBS])
    fingers = took(s)
    stick = took(s._replace(inv=s.inv.at[0, E.WOOD].set(1.0), hand=s.hand.at[0].set(E.WOOD)))
    assert stick > 3 * fingers > 0


def test_mosquitoes_haunt_warm_marshes_and_flee_smoke():
    s, _ = _world()
    marsh = np.argwhere(np.asarray(s.terrain) == E.MARSH)
    assert len(marsh)
    heat = jnp.zeros_like(s.fire)
    d = E.mosquito_density(s, CFG, jnp.int32(1), jnp.bool_(True), heat)
    assert float(d[tuple(marsh[0])]) > 0.5
    assert float(E.mosquito_density(s, CFG, jnp.int32(3), jnp.bool_(True), heat).max()) == 0.0     # winter
    assert float(E.mosquito_density(s, CFG, jnp.int32(1), jnp.bool_(True), heat + 1.0).max()) == 0.0  # smoke


def test_agents_see_an_animals_size_and_fangs_not_its_name():
    s, land = _world()
    a = land[0]
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(a)), large=put(s.large, 0, "mammoth", (a[0], a[1] + 1)),
                   predators=put(s.predators, 0, "wolf", (a[0] + 1, a[1])))
    v, r = CFG.view_size, CFG.view_radius
    o = np.asarray(E.observe(s, CFG))[0, :v * v * E.N_OBS_CHANNELS].reshape(v, v, -1)
    mam, wolf = o[r, r + 1, E.CH_FAUNA:E.CH_FAUNA + 3], o[r + 1, r, E.CH_FAUNA:E.CH_FAUNA + 3]
    assert mam[0] == 1.0 and wolf[1] == 1.0 and wolf[0] < mam[0]


def test_wolves_stalk_deer_and_deer_run_from_them():
    """cfg.animal_hunting with stalk_large: a hunting predator closes on prey it can take, and the
    prey moves away; without hunting, neither notices the other. People are far off."""
    import dataclasses
    cfg_on = dataclasses.replace(CFG, stalk_large=True)
    s, land = _world()
    r, c = next((r, c) for r, c in land if c + 3 < 20 and all((r, c + k) in land for k in range(4)))
    far = next(p for p in land if max(abs(p[0] - r), abs(p[1] - c)) > 10)
    s = s._replace(pos=jnp.asarray([far, far], jnp.int32),
                   predators=put(s.predators, 0, "wolf", (r, c)), large=put(s.large, 0, "deer", (r, c + 2)))   # a deer notices at 2

    def gaps(cfg):
        out = jax.vmap(lambda k: E.step(s._replace(key=k), jnp.full(2, E.REST, jnp.int32), cfg))(KEYS[:200])[0]
        wolf_c, deer_c = np.asarray(out.predators.pos[:, 0, 1]), np.asarray(out.large.pos[:, 0, 1])
        return (wolf_c - c).mean(), (deer_c - (c + 2)).mean()
    wolf_move, deer_move = gaps(cfg_on)
    off_w, off_d = gaps(dataclasses.replace(CFG, animal_hunting=False))
    assert wolf_move > off_w + 0.15 and wolf_move > 0.2    # the wolf comes east
    assert deer_move > off_d + 0.1 and deer_move > 0.1     # the deer runs east (at its flight rate)
    assert abs(off_w) < 0.1                                # without hunting the wolf only wanders (the deer grazes)


def test_a_wolf_prefers_a_deer_to_a_nearer_person():
    import dataclasses
    s, land = _world()
    r, c = next((r, c) for r, c in land if 2 <= c - 2 and c + 2 < 20 and (r, c - 2) in land and (r, c + 2) in land)
    far = next(p for p in land if max(abs(p[0] - r), abs(p[1] - c)) > 10)
    s = s._replace(pos=jnp.asarray([(r, c - 1), far], jnp.int32),       # a person 1 cell west, a deer 2 east
                   predators=put(s.predators, 0, "wolf", (r, c)), large=put(s.large, 0, "deer", (r, c + 2)))
    def drift(cfg):
        out = jax.vmap(lambda k: E.step(s._replace(key=k), jnp.full(2, E.REST, jnp.int32), cfg))(KEYS[:200])[0]
        return (np.asarray(out.predators.pos[:, 0, 1]) - c).mean()
    on = dataclasses.replace(CFG, stalk_large=True)
    assert drift(on) > 0.2                                               # toward the deer (east)
    assert drift(dataclasses.replace(on, prey_preference=0.0)) < 0.05   # no preference: the nearer person
