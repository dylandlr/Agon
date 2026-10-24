"""Byproducts of the living world: leaves, fiber, feathers, hides, bones, and the spear they allow.
Each does only what its physics allows; nothing tells an agent what any of it is for."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import fauna as fa
from lifesim import minerals as mn
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=16, n_agents=1, climate_volatility=0.0, season_temp=(18.0,) * 4, day_temp_swing=0.0,
                lightning_prob=0.0)
SP = E.species_params(CFG)
M = lambda mat: mat - E.MIN0                                   # index into the minerals grid


def _world(cfg=CFG):
    s = E.reset(jax.random.PRNGKey(5), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    t = np.asarray(s.terrain)
    land = [(r, c) for r in range(2, 14) for c in range(2, 13) if (t[r, c - 1:c + 2] != E.WATER).all()]
    west, mid = (land[0][0], land[0][1] - 1), land[0]
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   minerals=jnp.zeros_like(s.minerals), plants=jnp.zeros_like(s.plants), wood=jnp.zeros_like(s.wood),
                   conditions=jnp.zeros_like(s.conditions), pos=jnp.asarray([west], jnp.int32), clothing=jnp.zeros(1),
                   birth_age=jnp.full(1, int(25 * cfg.ticks_per_bio_year), jnp.int32))
    return s, west, mid


def _hold(s, mat, amount=1.0):
    return s._replace(inv=jnp.zeros_like(s.inv).at[0, mat].set(amount), hand=jnp.array([mat], jnp.int32))


def _put(an, name, cell, hp):
    k = fa.IDX[name]
    return an._replace(pos=an.pos.at[0].set(jnp.asarray(cell, jnp.int32)), alive=an.alive.at[0].set(True),
                       hp=an.hp.at[0].set(hp), species=an.species.at[0].set(k))


def test_a_killed_deer_leaves_hide_and_bone_and_a_fowl_feathers():
    s, west, mid = _world()
    s = _hold(s, E.STONE)._replace(large=_put(s.large, "deer", mid, 0.01))
    s1, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2]), CFG)
    assert bool(info.kills[0, 1])
    m = np.asarray(s1.minerals)[mid]
    assert m[M(E.HIDE)] == fa.BYPRODUCTS["deer"][0] and m[M(E.BONE)] == fa.BYPRODUCTS["deer"][1]
    s = _hold(_world()[0], E.STONE)._replace(small=_put(s.small, "fowl", mid, 0.01))
    keys = jax.random.split(jax.random.PRNGKey(3), 64)                  # fowl dodge half the blows
    s1, _, _, info = jax.vmap(lambda k: E.step(s._replace(key=k), jnp.array([E.STRIKE_N + 2]), CFG))(keys)
    hit = np.asarray(info.kills[:, 0, 0])
    assert hit.any() and (np.asarray(s1.minerals)[hit][:, mid[0], mid[1], M(E.FEATHER)] >= fa.BYPRODUCTS["fowl"][2]).all()


def test_fiber_placed_on_a_stick_and_a_point_binds_a_spear():
    s, west, mid = _world()
    flint = E.MIN0 + mn.IDX["flint"]
    ready = s._replace(wood=s.wood.at[mid].set(1.0), minerals=s.minerals.at[mid + (M(flint),)].set(1.0))
    s1, _, _, info = E.step(_hold(ready, E.FIBER), jnp.array([E.PLACE_N + 2]), CFG)       # place east, onto mid
    m = np.asarray(s1.minerals)[mid]
    assert bool(info.bound[0]) and m[M(E.SPEAR)] == 1.0 and m[M(flint)] == 0.0 and m[M(E.FIBER)] == 0.0
    assert float(s1.wood[mid]) < 1e-3
    # no point there (a stone is not an edge), or no stick: nothing binds
    for st in (s._replace(wood=s.wood.at[mid].set(1.0), stones=s.stones.at[mid].set(1.0)),
               s._replace(minerals=s.minerals.at[mid + (M(flint),)].set(1.0))):
        _, _, _, info = E.step(_hold(st, E.FIBER), jnp.array([E.PLACE_N + 2]), CFG)
        assert not bool(info.bound[0])


def test_a_spear_keeps_a_struck_boar_off():
    s, west, mid = _world()
    s = s._replace(large=_put(s.large, "boar", mid, 50.0))
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    gored = lambda st: float((jax.vmap(lambda k: E.step(st._replace(key=k), jnp.array([E.STRIKE_N + 2]), CFG))(keys)[3]
                              .gored[:, 0] > 0).mean())
    with_bone, with_spear = gored(_hold(s, E.BONE)), gored(_hold(s, E.SPEAR))
    assert with_spear < 0.7 * with_bone and with_bone > 0.2


def test_carried_hides_keep_warm_and_leaves_make_a_bed():
    cold = EnvConfig(**{**CFG.__dict__, "season_temp": (-10.0,) * 4})
    s, west, mid = _world(cold)
    run = lambda st, a: [float((st := E.step(st, jnp.array([a]), cold)[0]).body_temp[0]) for _ in range(6)][-1]
    bare, cloaked = run(s, E.REST), run(_hold(s, E.HIDE, 3.0), E.REST)
    assert cloaked > bare
    s = s._replace(sleepiness=jnp.full(1, 0.8))
    floor, bed = run(s, E.SLEEP), run(s._replace(minerals=s.minerals.at[west + (M(E.LEAVES),)].set(3.0)), E.SLEEP)
    assert bed > floor
    _, _, _, info = E.step(s._replace(minerals=s.minerals.at[west + (M(E.LEAVES),)].set(3.0)), jnp.array([E.SLEEP]), cold)
    assert bool(info.bedded[0])


def test_leaves_fall_under_trees_most_in_autumn_and_rot():
    s = E.reset(jax.random.PRNGKey(2), CFG)
    s = s._replace(minerals=jnp.zeros_like(s.minerals))
    forest = np.asarray(s.terrain) == E.FOREST
    assert forest.any()
    step = jax.jit(lambda st: E.step(st, jnp.zeros(1, jnp.int32), CFG)[0])
    a = s
    for _ in range(20):
        a = step(a)
    leaves = np.asarray(a.minerals[..., M(E.LEAVES)])
    assert leaves[forest].mean() > 0 and leaves[~forest].max() == 0
    autumn_t = CFG.season_starts[1] * CFG.day_length                       # first tick of autumn
    b = s._replace(t=jnp.int32(autumn_t))
    for _ in range(20):
        b = step(b)
    assert np.asarray(b.minerals[..., M(E.LEAVES)])[forest].mean() > 3 * leaves[forest].mean()
    full = s._replace(minerals=s.minerals.at[..., M(E.LEAVES)].set(1.0), terrain=jnp.full_like(s.terrain, E.GRASS))
    assert float(step(full).minerals[0, 0, M(E.LEAVES)]) < 1.0


def test_leaves_are_tinder_for_a_spark():
    s, west, mid = _world()
    s = s._replace(plants=s.plants.at[mid].set(0.3), weather=jnp.int32(E.CLEAR),
                   stones=s.stones.at[mid].set(2.0))
    keys = jax.random.split(jax.random.PRNGKey(1), 400)
    fires = lambda st: float((jax.vmap(lambda k: E.step(st._replace(key=k), jnp.array([E.STRIKE_N + 2]), CFG))(keys)[3]
                              .ignited[:, 0] > 0).mean())
    dry = s._replace(minerals=s.minerals.at[mid + (M(E.LEAVES),)].set(1.0))
    assert fires(_hold(dry, E.MIN0 + mn.IDX["pyrite"])) > 1.5 * fires(_hold(s, E.MIN0 + mn.IDX["pyrite"]))


def test_byproducts_are_organic_never_deposits_and_have_sizes():
    for name in mn.ORGANIC:
        i = mn.IDX[name]
        assert mn.IS_ORGANIC[i] and sum(mn.DEPOSIT_CHANCE[i]) == 0 and mn.SIZE[i] > 0
    assert mn.SIZE[mn.IDX["leaves"]] < 0.5 <= mn.SIZE[mn.IDX["hide"]]
