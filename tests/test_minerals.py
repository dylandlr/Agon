"""Minerals, ores and metals: each does only what its physics allows."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import minerals as mn
from lifesim.config import EnvConfig, living_world

CFG = EnvConfig(grid_size=16, n_agents=1, climate_volatility=0.0, season_temp=(18.0,) * 4, day_temp_swing=0.0,
                lightning_prob=0.0)
MI = lambda name: E.MIN0 + mn.IDX[name]                       # material index of a mineral


def _world(cfg=CFG):
    s = E.reset(jax.random.PRNGKey(5), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    t = np.asarray(s.terrain)
    land = [(r, c) for r in range(2, 14) for c in range(2, 13) if (t[r, c - 1:c + 2] != E.WATER).all()]
    west, mid, east = (land[0][0], land[0][1] - 1), land[0], (land[0][0], land[0][1] + 1)
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   minerals=jnp.zeros_like(s.minerals), plants=jnp.zeros_like(s.plants), wood=jnp.zeros_like(s.wood),
                   conditions=jnp.zeros_like(s.conditions), pos=jnp.asarray([west], jnp.int32),
                   birth_age=jnp.full(1, int(25 * cfg.ticks_per_bio_year), jnp.int32))
    return s, west, mid, east


def _hold(s, mat, amount=1.0):
    return s._replace(inv=jnp.zeros_like(s.inv).at[0, mat].set(amount), hand=jnp.array([mat], jnp.int32))


def test_deposits_follow_geology():
    cfg = living_world(EnvConfig(grid_size=48, n_agents=2))
    s = E.reset(jax.random.PRNGKey(1), cfg)
    t, m = np.asarray(s.terrain), np.asarray(s.minerals)
    assert m[t == E.WATER].sum() == 0                                     # nothing lies in water
    for name in ("copper", "tin", "bronze", "iron", "fired clay"):
        assert m[..., mn.IDX[name]].sum() == 0                            # made, never found
    clay = m[..., mn.IDX["clay"]] > 0
    assert set(np.unique(t[clay])) <= {E.MARSH, E.GRASS}
    assert (m.sum(-1) > 0).sum() > 10
    none = E.reset(jax.random.PRNGKey(1), cfg.__class__(**{**cfg.__dict__, "mineral_abundance": 0.0}))
    assert float(none.minerals.sum()) == 0.0


def test_pyrite_struck_on_flint_lights_dry_grass_far_more_often_than_stone_on_stone():
    s, west, mid, _ = _world()
    s = s._replace(plants=s.plants.at[mid].set(2.0), weather=jnp.int32(E.CLEAR))
    strike = jnp.array([E.STRIKE_N + 2])                                   # east, onto mid

    def fires(st):
        keys = jax.random.split(jax.random.PRNGKey(0), 400)
        _, _, _, info = jax.vmap(lambda k: E.step(st._replace(key=k), strike, CFG))(keys)
        return float((info.ignited[:, 0] > 0).mean())
    kit = fires(_hold(s, MI("pyrite"))._replace(minerals=s.minerals.at[mid + (mn.IDX["flint"],)].set(2.0)))
    plain = fires(_hold(s, E.STONE)._replace(stones=s.stones.at[mid].set(2.0)))
    soft = fires(_hold(s, MI("clay"))._replace(stones=s.stones.at[mid].set(2.0)))
    assert kit > 3 * plain > 0 and soft == 0.0


def test_an_edge_hits_harder_and_cuts_more_meat():
    s, west, mid, _ = _world()
    s = s._replace(large=s.large._replace(pos=s.large.pos.at[0].set(jnp.asarray(mid)), alive=s.large.alive.at[0].set(True),
                                          hp=s.large.hp.at[0].set(3.0)))
    hit = lambda st: float(3.0 - E.step(st, jnp.array([E.STRIKE_N + 2]), CFG)[0].large.hp[0])
    bare, stone, blade = hit(s), hit(_hold(s, E.STONE)), hit(_hold(s, MI("obsidian")))
    assert blade > stone > bare
    meat = s._replace(meat=s.meat.at[west].set(5.0), large=s.large._replace(alive=jnp.zeros_like(s.large.alive)))
    took = lambda st: float(E.step(st, jnp.array([E.GRASP]), CFG)[0].inv[0, E.RAW_MEAT])
    bare_hands = took(meat._replace(inv=jnp.zeros_like(meat.inv)))
    assert took(_hold(meat, MI("flint"))) > 1.4 * took(_hold(meat, E.STONE)) > 1.4 * bare_hands * 0.8


def test_heat_fires_clay_but_only_coal_heat_smelts_ore_and_makes_bronze():
    s, _, mid, _ = _world()
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST]), CFG))

    def burn(st, ticks=60):
        for _ in range(ticks):
            st, _, _, _ = step(st)
        return np.asarray(st.minerals[mid])
    lay = lambda st, **kw: st._replace(minerals=st.minerals.at[mid].set(
        jnp.zeros(mn.M).at[jnp.array([mn.IDX[k.replace("_", " ")] for k in kw])].set(jnp.array(list(kw.values()), jnp.float32))))
    wood_fire = lay(s, clay=1.0, copper_ore=1.0)._replace(wood=s.wood.at[mid].set(2.0), fire=s.fire.at[mid].set(1.0))
    m = burn(wood_fire)
    assert m[mn.IDX["fired clay"]] > 0.9 and m[mn.IDX["copper"]] == 0.0    # a wood fire fires clay, smelts nothing
    coal_fire = lay(s, copper_ore=3.0, tin_ore=1.0, coal=2.0)._replace(fire=s.fire.at[mid].set(1.0))
    m = burn(coal_fire, 200)
    assert m[mn.IDX["coal"]] == 0.0 and m[mn.IDX["bronze"]] > 1.0          # coal heat: ore to metal, metals to bronze
    assert m[mn.IDX["copper ore"]] < 3.0


def test_salt_slows_spoiling_of_carried_meat():
    s, *_ = _world()
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST]), CFG))
    plain = _hold(s, E.RAW_MEAT, 2.0)
    salted = plain._replace(inv=plain.inv.at[0, MI("salt")].set(1.0))
    for _ in range(100):
        plain, *_ = step(plain._replace(satiety=jnp.ones(1), hydration=jnp.ones(1)))
        salted, *_ = step(salted._replace(satiety=jnp.ones(1), hydration=jnp.ones(1)))
    lost = lambda st: 2.0 - float(st.inv[0, E.RAW_MEAT])
    assert lost(salted) < 0.5 * lost(plain)


def test_minerals_are_seen_by_colour_not_name():
    s, west, mid, _ = _world()
    s = s._replace(minerals=s.minerals.at[mid + (mn.IDX["copper ore"],)].set(2.0), t=jnp.int32(24))
    o = np.asarray(E.observe(s, CFG))[0, :CFG.view_size ** 2 * E.N_OBS_CHANNELS].reshape(CFG.view_size, CFG.view_size, -1)
    r = CFG.view_radius
    seen = o[r, r + 1, E.CH_MINERALS:E.CH_MINERALS + E.N_MINERAL_CHANNELS]
    assert seen[0] > 0 and np.allclose(seen[1:4], mn.COLOR[mn.IDX["copper ore"]], atol=1e-5)
    assert o[r, r, E.CH_MINERALS] == 0.0                                    # nothing on its own cell


def test_watching_keeps_only_what_everyone_perceives():
    """Social learning blanks the demonstrator's private state and keeps time and weather; the
    slice must follow the body vector's real layout."""
    from lifesim import social
    s, *_ = _world()
    o = E.observe(s, CFG)[0]
    n_win = CFG.view_size ** 2 * E.N_OBS_CHANNELS
    kept = social.restrict_to_observer_view(o[None], s.pos, s.pos, jnp.array([False]), CFG)[0]
    body, kept_body = np.asarray(o[n_win:]), np.asarray(kept[n_win:])
    shared = slice(E.SHARED_START, E.SHARED_START + 5 + E.N_WEATHER)
    np.testing.assert_allclose(kept_body[shared], body[shared])
    assert np.abs(body[shared]).sum() > 0                                   # time of day is there to keep
    assert np.abs(kept_body[:E.SHARED_START]).sum() == 0                    # hunger, inventory: private
