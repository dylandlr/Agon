"""Honest sleep, the plant ecology (EnvConfig.ecology) and camp fouling (EnvConfig.fouling)."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import ecology as eco
from lifesim import env as E
from lifesim.config import EnvConfig, living_world, with_tick_minutes

CFG = dataclasses.replace(living_world(EnvConfig(grid_size=16, n_agents=2), downed=False, excretion=True),
                          climate_volatility=0.0, lightning_prob=0.0, founder_spread=0)
STEP = jax.jit(lambda s, a: E.step(s, a, CFG))


def _calm(s):
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    return s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators))


def _cell(s, terrain):
    return tuple(int(v) for v in np.argwhere(np.asarray(s.terrain) == terrain)[0])


def _day_night():
    day = next(t for t in range(48) if not bool(E.is_night(jnp.int32(t), CFG)))
    night = next(t for t in range(48) if bool(E.is_night(jnp.int32(t), CFG)))
    return day, night


# ---------------------------------------------------------------- honest sleep
def test_sleep_needs_sleep_pressure_lower_at_night():
    s = _calm(E.reset(jax.random.PRNGKey(0), CFG))
    day, night = _day_night()
    rested = s._replace(t=jnp.int32(day), sleepiness=jnp.zeros(2))
    assert not bool(E.action_mask(rested, CFG)[0, E.SLEEP])
    out, *_ = STEP(rested, jnp.array([E.SLEEP, E.REST]))
    assert not bool(out.asleep[0]) and int(out.sleep_depth[0]) == 0          # lying down is only rest
    some = s._replace(sleepiness=jnp.full(2, 0.2))
    assert not bool(E.action_mask(some._replace(t=jnp.int32(day)), CFG)[0, E.SLEEP])
    assert bool(E.action_mask(some._replace(t=jnp.int32(night)), CFG)[0, E.SLEEP])   # the body clock
    tired = s._replace(t=jnp.int32(day), sleepiness=jnp.full(2, CFG.sleep_onset))
    out, *_ = STEP(tired, jnp.array([E.SLEEP, E.REST]))
    assert bool(out.asleep[0])


def test_no_free_sleep_benefits():
    s = _calm(E.reset(jax.random.PRNGKey(1), CFG))
    day, _ = _day_night()
    s = s._replace(t=jnp.int32(day), sleepiness=jnp.zeros(2), stress=jnp.full(2, 0.4), health=jnp.full(2, 0.6))
    a, *_ = STEP(s, jnp.array([E.SLEEP, E.REST]))
    assert np.isclose(float(a.stress[0]), float(a.stress[1])) and np.isclose(float(a.health[0]), float(a.health[1]))


# ---------------------------------------------------------------- plant ecology
def test_flora_follows_biome_and_is_per_km2():
    s = E.reset(jax.random.PRNGKey(2), CFG)
    assert s.flora.shape == (16, 16, eco.N_KINDS)
    water = np.asarray(s.terrain) == E.WATER
    assert float(np.asarray(s.flora)[water].sum()) == 0.0
    assert np.allclose(np.asarray(s.plants), np.asarray(s.flora.sum(-1)), atol=1e-4)
    # the same land at 1-minute ticks: cells 900x smaller in area hold 900x less, the same per km^2
    one = with_tick_minutes(CFG, 1.0)
    terr, patch = s.terrain, s.flora_patch
    k30, k1 = eco.capacity(terr, patch, CFG), eco.capacity(terr, patch, one)
    assert np.allclose(np.asarray(k30) / eco.cell_km2(CFG), np.asarray(k1) / eco.cell_km2(one), rtol=1e-4)
    assert np.isclose(eco.unit_kcal(CFG), eco.unit_kcal(one))


def test_gathering_pays_by_density_and_a_patch_runs_out():
    cfg = CFG
    rich = jnp.array([[0.0, 0.0, 0.0, 50.0]])
    thin = jnp.array([[0.0, 0.0, 0.0, 0.5]])
    y_rich = float(eco.gather_yield(rich, jnp.array([False]), cfg)[0, eco.ROOTS])
    y_thin = float(eco.gather_yield(thin, jnp.array([False]), cfg)[0, eco.ROOTS])
    stick = float(eco.gather_yield(rich, jnp.array([True]), cfg)[0, eco.ROOTS])
    assert y_rich > 3 * y_thin and np.isclose(stick, eco.DIG_STICK_BONUS * y_rich)
    # one act is half an hour of work: well under the old one unit per grasp
    assert y_rich < 0.25
    # gathering one patch over and over (a day's foraging is several acts): each act gets less
    f = jnp.array([[0.0, 0.0, 0.0, 60.0]])
    got = []
    step_ = jax.jit(lambda f: eco.gather_yield(f, jnp.array([False]), cfg)[0, eco.ROOTS])
    for _ in range(1200):
        y = step_(f)
        got.append(float(y))
        f = f.at[0, eco.ROOTS].add(-y)
    assert got[-1] < 0.5 * got[0]
    # in the world: an act takes what it yields from the growing roots, into the hand
    s = _calm(E.reset(jax.random.PRNGKey(3), CFG))
    r, c = _cell(s, E.GRASS)
    s = s._replace(pos=s.pos.at[0].set(jnp.array([r, c])).at[1].set(jnp.array([(r + 5) % 16, c])),
                   flora=s.flora.at[r, c].set(jnp.array([0.0, 0.0, 0.0, 60.0])), satiety=jnp.ones(2), hydration=jnp.ones(2))
    s = s._replace(plants=s.flora.sum(-1))
    out, *_ = STEP(s, jnp.array([E.GRASP, E.REST]))
    took = float(out.inv[0, E.PLANT])
    expect = float(eco.gather_yield(s.flora[r, c][None], jnp.array([False]), CFG)[0, eco.ROOTS])
    assert np.isclose(took, expect, rtol=1e-3)
    assert float(out.flora[r, c, eco.ROOTS]) < 60.0 - 0.9 * took + 0.05   # (it also grew a little)


def test_regrowth_is_logistic_seasonal_and_fire_spares_roots():
    s = E.reset(jax.random.PRNGKey(4), CFG)
    terr, patch = s.terrain, s.flora_patch
    K = eco.capacity(terr, patch, CFG)
    one = jnp.ones(terr.shape)
    no_fire = jnp.zeros(terr.shape, bool)
    grow = lambda f, season, burn=no_fire: eco.grow(f, terr, patch, season, one, burn, CFG)
    gr = np.asarray(terr) == E.GRASS
    # greens: a half-full patch grows faster than a bare one (it grows from what is there)
    half, bare = K * 0.5, K * 0.0
    dh = np.asarray(grow(half, 0) - half)[..., eco.GREENS][gr]
    db = np.asarray(grow(bare, 0) - bare)[..., eco.GREENS][gr]
    assert (dh > db).all()
    # fruit ripens in summer, not spring; greens die back in winter
    f0 = K * 0.2
    assert float((grow(f0, 1) - f0)[..., eco.FRUIT].sum()) > 0
    assert float((grow(f0, 0) - f0)[..., eco.FRUIT].sum()) < 0
    assert float((grow(f0, 3) - f0)[..., eco.GREENS].sum()) < 0
    # nuts fall in autumn
    assert float((grow(f0, 2) - f0)[..., eco.NUTS].sum()) > 0
    # fire: everything above ground burns, roots survive
    burnt = grow(K * 0.5, 1, jnp.ones(terr.shape, bool))
    assert float(burnt[..., :3].sum()) == 0.0 and float(burnt[..., eco.ROOTS].sum()) > 0
    # never more than the patch holds
    assert bool(jnp.all(grow(K, 1) <= K + 1e-6))


def test_dropped_food_is_loose_and_kept_apart_from_what_grows():
    s = _calm(E.reset(jax.random.PRNGKey(5), CFG))
    r, c = _cell(s, E.GRASS)
    s = s._replace(pos=s.pos.at[0].set(jnp.array([r, c])), inv=s.inv.at[0, E.PLANT].set(1.0),
                   hand=s.hand.at[0].set(E.PLANT))
    flora_before = float(s.flora[r, c].sum())
    out, *_ = STEP(s, jnp.array([E.RELEASE, E.REST]))
    assert float(E.loose_plants(out, CFG)[r, c]) > 0.9
    back, *_ = STEP(out, jnp.array([E.GRASP, E.REST]))                    # loose food is picked up whole
    assert float(back.inv[0, E.PLANT]) > 0.9
    assert abs(float(back.flora[r, c].sum()) - flora_before) < 0.2          # the growing plants were not taken


def test_grazers_eat_greens_not_roots():
    from lifesim import fauna as fa
    s = E.reset(jax.random.PRNGKey(6), CFG)
    r, c = _cell(s, E.GRASS)
    L = s.large._replace(alive=jnp.zeros_like(s.large.alive).at[0].set(True), species=s.large.species.at[0].set(fa.IDX["deer"]),
                         pos=s.large.pos.at[0].set(jnp.array([r, c])), below=jnp.zeros_like(s.large.below))
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    s = s._replace(large=L, small=dead(s.small), predators=dead(s.predators),
                   flora=s.flora.at[r, c].set(jnp.array([1.0, 0.0, 0.0, 5.0])))
    s = s._replace(plants=s.flora.sum(-1))
    out, *_ = STEP(s, jnp.array([E.REST, E.REST]))
    rr, cc = (int(v) for v in np.asarray(out.large.pos[0]))
    if (rr, cc) == (r, c):
        assert float(out.flora[r, c, eco.GREENS]) < 1.0
    assert float(out.flora[r, c, eco.ROOTS]) >= 5.0 - 1e-3


# ---------------------------------------------------------------- camp fouling
def test_without_excretion_refuse_builds_where_people_stay_and_fades():
    cfg = dataclasses.replace(CFG, excretion=False)
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    s = _calm(E.reset(jax.random.PRNGKey(7), cfg))
    r, c = _cell(s, E.GRASS)
    s = s._replace(pos=s.pos.at[0].set(jnp.array([r, c])).at[1].set(jnp.array([r, c])),
                   satiety=jnp.ones(2), hydration=jnp.ones(2))
    for _ in range(10):
        s, *_ = step(s, jnp.array([E.REST, E.REST]))
    built = float(s.foul[r, c])
    assert np.isclose(built, 2 * 10 * (CFG.tick_minutes / 1440) / eco.cell_km2(CFG), rtol=0.05)
    gone = s._replace(alive=jnp.zeros(2, bool))
    for _ in range(5):
        gone, *_ = step(gone, jnp.array([E.REST, E.REST]))
    assert float(gone.foul[r, c]) < built


def test_drinking_beside_waste_makes_people_ill():
    s = _calm(E.reset(jax.random.PRNGKey(8), CFG))
    w = np.argwhere(np.asarray(s.terrain) == E.WATER)
    spot = next((int(a), int(b) + 1) for a, b in w if b + 1 < 16 and int(s.terrain[a, b + 1]) != E.WATER)
    base = s._replace(pos=s.pos.at[0].set(jnp.array(spot)).at[1].set(jnp.array([(spot[0] + 6) % 16, spot[1]])),
                      immunity=jnp.zeros(2), hydration=jnp.full(2, 0.3))
    dirty = base._replace(foul=base.foul.at[spot].set(1e6))
    run = jax.jit(jax.vmap(lambda k, st: E.step(st._replace(key=k), jnp.array([E.DRINK, E.REST]), CFG)[0].infection[0] > 0,
                           in_axes=(0, None)))
    keys = jax.random.split(jax.random.PRNGKey(9), 400)
    clean_ill, dirty_ill = int(run(keys, base).sum()), int(run(keys, dirty).sum())
    assert dirty_ill > 4 * max(clean_ill, 1)


def test_a_patch_feeds_everyone_gathering_it_but_loose_food_goes_to_one():
    s = _calm(E.reset(jax.random.PRNGKey(10), CFG))
    r, c = _cell(s, E.GRASS)
    both = jnp.array([[r, c], [r, c]], jnp.int32)
    s = s._replace(pos=both, flora=s.flora.at[r, c].set(jnp.array([0.0, 0.0, 0.0, 60.0])), satiety=jnp.ones(2),
                   hydration=jnp.ones(2))
    s = s._replace(plants=s.flora.sum(-1))
    out, _, _, info = STEP(s, jnp.array([E.GRASP, E.GRASP]))
    assert float(out.inv[0, E.PLANT]) > 0 and float(out.inv[1, E.PLANT]) > 0
    assert int(info.outcome[0]) == int(info.outcome[1]) == E.OUT_GATHER


# ---------------------------------------------------------------- excretion
def test_urge_builds_relief_leaves_waste_and_holding_too_long_soils():
    s = _calm(E.reset(jax.random.PRNGKey(11), CFG))
    r, c = _cell(s, E.GRASS)
    s = s._replace(pos=s.pos.at[0].set(jnp.array([r, c])).at[1].set(jnp.array([(r + 6) % 16, c])),
                   satiety=jnp.ones(2), hydration=jnp.ones(2), foul=jnp.zeros_like(s.foul))
    assert not bool(E.action_mask(s, CFG)[0, E.RELIEVE])                  # nothing to go with
    out, *_ = STEP(s, jnp.array([E.REST, E.REST]))
    assert float(out.urge[0]) > 0                                        # it builds
    full = s._replace(urge=jnp.full(2, 0.6))
    assert bool(E.action_mask(full, CFG)[0, E.RELIEVE])
    out, _, _, info = STEP(full, jnp.array([E.RELIEVE, E.REST]))
    assert float(out.urge[0]) == 0.0 and bool(info.relieved[0]) and int(info.outcome[0]) == E.OUT_RELIEVE
    assert float(out.foul[r, c]) > 0                                     # waste where it went
    # held past full: it happens anyway, stressfully
    burst = s._replace(urge=jnp.full(2, 0.999), stress=jnp.zeros(2))
    out, _, _, info = STEP(burst, jnp.array([E.REST, E.REST]))
    assert bool(info.soiled[0]) and float(out.urge[0]) == 0.0 and float(out.stress[0]) > 0.05
    assert float(info.stress_from[0, E.STRESS_SOURCES.index("needing to relieve oneself")]) > 0


def test_urge_is_seen_and_off_when_excretion_is_off():
    s = _calm(E.reset(jax.random.PRNGKey(12), CFG))._replace(urge=jnp.array([0.42, 0.0]))
    obs = np.asarray(E.observe(s, CFG))
    body0 = CFG.view_size ** 2 * E.N_OBS_CHANNELS
    assert np.isclose(obs[0, body0 + 22], 0.42)
    off = dataclasses.replace(CFG, excretion=False)
    assert not bool(E.action_mask(s._replace(urge=jnp.full(2, 0.9)), off)[0, E.RELIEVE])


# ---------------------------------------------------------------- grounded language and writing
LANG = dataclasses.replace(CFG, language_mode="grounded", writing=True)
LSTEP = jax.jit(lambda s, a, m: E.step(s, a, LANG, m))


def _phrase(*words):
    m = jnp.zeros(E.MSG_LEN, jnp.int32)
    return m.at[:len(words)].set(jnp.array([E.LEXICON.index(w) for w in words]))


def _two(s, a=(5, 5), b=(5, 7)):
    return _calm(s)._replace(pos=jnp.array([a, b], jnp.int32), mind_map=jnp.zeros_like(s.mind_map))


def test_a_phrase_about_water_puts_the_place_in_listeners_maps():
    s = _two(E.reset(jax.random.PRNGKey(20), LANG))
    msg = jnp.stack([_phrase("water", "east", "far"), _phrase()])
    out, *_ = LSTEP(s, jnp.array([E.REST, E.REST]), msg)
    far = 2 * LANG.view_radius
    place = (5, min(15, 5 + far))
    assert float(out.mind_map[1, place[0], place[1], 0]) == 1.0 and float(out.mind_map[1, place[0], place[1], 2]) > 0.5
    # "no food far west" clears what one remembered there (out of the listener's sight: sight beats hearsay)
    s2 = s._replace(mind_map=s.mind_map.at[1, 5, 0, 1].set(1.0).at[1, 5, 0, 2].set(1.0))
    out, *_ = LSTEP(s2, jnp.array([E.REST, E.REST]), jnp.stack([_phrase("no", "food", "west", "far"), _phrase()]))
    assert float(out.mind_map[1, 5, 0, 1]) == 0.0
    # emergent mode: the same symbols mean nothing
    emer = jax.jit(lambda st, a, m: E.step(st, a, CFG, m))
    out, *_ = emer(s, jnp.array([E.REST, E.REST]), msg)
    assert float(out.mind_map[1, place[0], place[1], 0]) == 0.0


FLINT = E.MIN0 + E.mn.IDX["flint"]


def _with_tool(s, k=0, mat=None, amount=1.0):
    mat = FLINT if mat is None else mat
    return s._replace(inv=s.inv.at[k, mat].set(amount), hand=s.hand.at[k].set(mat))


def _board(s, r, c, wood=1.0):                                        # a piece of wood lying here, nothing else
    return s._replace(wood=s.wood.at[r, c].set(wood), stones=s.stones.at[r, c].set(0.0),
                      terrain=s.terrain.at[r, c].set(E.GRASS))


def test_writing_needs_a_tool_and_something_to_write_on():
    s = _two(E.reset(jax.random.PRNGKey(22), LANG), a=(5, 5), b=(12, 12))
    bare = s._replace(wood=s.wood.at[5, 5].set(0.0), stones=s.stones.at[5, 5].set(0.0), terrain=s.terrain.at[5, 5].set(E.GRASS))
    can = lambda st: bool(E.action_mask(st, LANG)[0, E.WRITE])
    assert not can(_board(s, 5, 5))                                  # wood to write on, nothing to write with
    assert not can(_with_tool(bare))                                 # an edge, but bare earth takes no mark
    assert can(_with_tool(_board(s, 5, 5)))                          # flint on wood: a sign
    assert not can(_with_tool(_board(s, 5, 5), mat=E.STONE))         # a plain stone has no edge
    rock = _with_tool(bare._replace(terrain=bare.terrain.at[5, 5].set(E.ROCK)))
    assert can(rock)                                                 # flint is hard enough to cut rock
    copper = E.MIN0 + E.mn.IDX["copper"]
    assert can(_with_tool(_board(s, 5, 5), mat=copper)) and not can(_with_tool(rock, mat=copper))   # soft metal: wood, not rock
    ochre = E.MIN0 + E.mn.IDX["ochre"]
    assert can(_with_tool(rock, mat=ochre))                          # pigment draws on rock
    forest = bare._replace(terrain=bare.terrain.at[5, 5].set(E.FOREST))
    assert can(_with_tool(forest)) and not can(_with_tool(forest, mat=ochre))   # bark is cut, not painted
    # the WRITE action without the means does nothing
    out, _, _, info = LSTEP(_with_tool(bare), jnp.array([E.WRITE, E.REST]), jnp.stack([_phrase("water", "here"), _phrase()]))
    assert not bool(jnp.any(out.marks[5, 5] > 0))


def test_a_sign_goes_with_its_wood():
    s = _with_tool(_board(_two(E.reset(jax.random.PRNGKey(23), LANG), a=(5, 5), b=(12, 12)), 5, 5))
    out, _, _, info = LSTEP(s, jnp.array([E.WRITE, E.REST]), jnp.stack([_phrase("water", "east"), _phrase()]))
    assert int(info.outcome[0]) == E.OUT_WRITE and int(out.mark_kind[5, 5]) == E.SURF_WOOD
    taken, *_ = LSTEP(out._replace(wood=out.wood.at[5, 5].set(0.0)), jnp.array([E.REST, E.REST]), jnp.zeros((2, E.MSG_LEN), jnp.int32))
    assert not bool(jnp.any(taken.marks[5, 5] > 0)) and int(taken.mark_kind[5, 5]) == 0


def test_writing_leaves_an_inscription_that_others_read_and_that_fades():
    s = _with_tool(_board(_two(E.reset(jax.random.PRNGKey(21), LANG), a=(5, 5), b=(12, 12)), 5, 5))
    msg = jnp.stack([_phrase("food", "south", "far"), _phrase()])
    out, _, _, info = LSTEP(s, jnp.array([E.WRITE, E.REST]), msg)
    assert int(info.outcome[0]) == E.OUT_WRITE and list(np.asarray(out.marks[5, 5])) == list(np.asarray(msg[0]))
    # someone who later stands beside it reads it: the place enters their map, and it is in their senses
    reader = out._replace(pos=jnp.array([[12, 12], [5, 6]], jnp.int32), last_msg=jnp.zeros_like(out.last_msg))
    seen = np.asarray(E.read_marks(reader, LANG))[1]
    assert seen[:E.MSG_LEN * E.VOCAB].reshape(E.MSG_LEN, E.VOCAB)[0, E.LEXICON.index("food")] == 1.0
    after, *_ = LSTEP(reader, jnp.array([E.REST, E.REST]), jnp.zeros((2, E.MSG_LEN), jnp.int32))
    far = 2 * LANG.view_radius                                           # out of the reader's sight
    assert float(after.mind_map[1, min(15, 5 + far), 5, 1]) == 1.0
    # cut in wood, it is gone after mark_days
    old = out._replace(t=out.t + int(LANG.mark_days * LANG.day_length) + 2)
    gone, *_ = LSTEP(old, jnp.array([E.REST, E.REST]), jnp.zeros((2, E.MSG_LEN), jnp.int32))
    assert not bool(jnp.any(gone.marks[5, 5] > 0))
