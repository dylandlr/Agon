"""Conditions: inborn (inherited), with age, from events, earned by use; each with real effects."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import conditions as C
from lifesim import env as E
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=16, n_agents=2, climate_volatility=0.0, lightning_prob=0.0,
                season_temp=(18.0,) * 4, day_temp_swing=0.0)


def with_cond(s, i, name, on=True):
    return s._replace(conditions=s.conditions.at[i, C.IDX[name]].set(1.0 if on else 0.0))


def _world():
    s = E.reset(jax.random.PRNGKey(3), CFG)
    t = np.asarray(s.terrain)
    p = next((r, c) for r in range(2, 13) for c in range(2, 10) if (t[r, c:c + 2] != E.WATER).all())
    return s._replace(pos=jnp.asarray([p, (p[0], p[1] + 1)], jnp.int32), conditions=jnp.zeros_like(s.conditions))


def test_inborn_rates_and_inheritance():
    many = E.draw_inborn(jax.random.PRNGKey(0), 20000, None, None)
    rates = np.asarray(many).mean(0)
    for i in C.INBORN:
        assert abs(rates[i] - C.CONDITIONS[i][3]) < 0.01
    assert rates[C.IDX["lame"]] == 0.0                                       # not inborn
    parent = jnp.zeros((20000, C.K)).at[:, C.IDX["keen eyes"]].set(1.0)
    kids = np.asarray(E.draw_inborn(jax.random.PRNGKey(1), 20000, parent, jnp.zeros_like(parent))).mean(0)
    assert abs(kids[C.IDX["keen eyes"]] - 0.5) < 0.02                         # a parent's trait: 50%


def test_effects_reach_the_body():
    s = _world()
    base = E.body(s, CFG)
    strong = E.body(with_cond(s, 0, "robust"), CFG)
    assert np.isclose(float(strong["strength"][0]), 1.2 * float(base["strength"][0]), rtol=1e-4)
    stiff = E.body(with_cond(s, 0, "stiff joints"), CFG)
    assert float(stiff["speed"][0]) < float(base["speed"][0])
    assert int(E.sight_radius(with_cond(s, 0, "poor eyesight"), CFG)[0]) == CFG.view_radius - 1
    assert int(E.dark_radius(with_cond(s, 0, "keen eyes"), CFG)[0]) == CFG.night_view_radius + 1


def test_events_and_use_bring_conditions():
    s = _world()
    step = jax.jit(lambda st, a: E.step(st, a, CFG))
    crippled, _, _, info = step(s._replace(injury=jnp.array([0.95, 0.0])), jnp.zeros(2, jnp.int32))
    assert bool(info.new_conditions[0, C.IDX["lame"]]) and float(crippled.conditions[0, C.IDX["lame"]]) == 1.0
    walker, _, _, info = step(s._replace(life_moves=jnp.array([1499, 0], jnp.int32)), jnp.array([E.MOVE_N, E.REST]))
    moved = int(walker.life_moves[0]) == 1500
    assert (not moved) or bool(info.new_conditions[0, C.IDX["enduring"]])


def test_age_brings_stiff_joints_eventually():
    import dataclasses
    from lifesim.config import with_reproduction
    n = 16
    cfg = with_reproduction(dataclasses.replace(CFG, n_agents=n), n_founders=n)
    s = E.reset(jax.random.PRNGKey(3), cfg)
    young = s._replace(birth_age=jnp.full(n, int(30 * cfg.ticks_per_bio_year), jnp.int32))
    old = s._replace(birth_age=jnp.full(n, int(46 * cfg.ticks_per_bio_year), jnp.int32))
    def life(start):
        def body(st, _):
            st = st._replace(satiety=jnp.ones(n), hydration=jnp.ones(n), energy=jnp.ones(n), health=jnp.ones(n),
                             conditions=st.conditions.at[:, C.IDX["infertile"]].set(0.0))
            st2, _, _, info = E.step(st, jnp.zeros(n, jnp.int32), cfg)
            return st2, info.new_conditions[:, C.IDX["stiff joints"]]
        return np.asarray(jax.jit(lambda st: jax.lax.scan(body, st, None, int(10 * cfg.ticks_per_bio_year)))(start)[1])
    assert life(old).any()                                                # past 45: 3% a year, 16 agents, 10 years
    assert not life(young).any()                                          # 30 to 40: never


def test_playback_observation_groups_cover_every_number():
    """The decision view attributes each observation number to a group; the layout must match observe()."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from export_playback import obs_groups
    from lifesim.config import EnvConfig, living_world
    for cfg in (EnvConfig(grid_size=16, n_agents=4), living_world(EnvConfig(grid_size=16, n_agents=4))):
        assert len(obs_groups(cfg)) == E.obs_dim(cfg)
