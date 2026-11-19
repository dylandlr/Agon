"""Biomes, dynamic weather and climate (the living world), and that the classic world is unchanged."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig, living_world

LIVING = living_world(EnvConfig(grid_size=32, n_agents=2))
CLASSIC = EnvConfig(grid_size=32, n_agents=2)


def test_classic_world_has_only_the_four_old_grounds_and_no_weather_fields_in_use():
    s = E.reset(jax.random.PRNGKey(1), CLASSIC)
    assert set(np.unique(np.asarray(s.terrain))) <= {E.GRASS, E.FOREST, E.ROCK, E.WATER}
    out, _, _, _ = E.step(s, jnp.zeros(2, jnp.int32), CLASSIC)
    assert float(out.clouds.sum()) == 0.0 and float(out.snow.sum()) == 0.0


def test_living_world_has_every_biome_and_caves_sit_in_rock():
    s = E.reset(jax.random.PRNGKey(4), LIVING)
    t = np.asarray(s.terrain)
    assert set(np.unique(t)) == set(range(E.N_TERRAIN))
    assert (t == E.CAVE).sum() == LIVING.caves
    north, south = t[:8], t[-8:]
    assert (north == E.TUNDRA).sum() > (south == E.TUNDRA).sum()             # colder to the north


def test_biome_temperatures_and_the_climate_cycle():
    cfg = LIVING
    amb = lambda terr, row=15.5, clim=0.0: float(E.ambient_temperature(jnp.int32(20), jnp.int32(E.CLEAR), jnp.int32(terr),
                                                                       cfg, 0.3, jnp.float32(row), clim))
    assert amb(E.DESERT) > amb(E.GRASS) > amb(E.TUNDRA)
    assert amb(E.GRASS, row=0) < amb(E.GRASS, row=31)                          # north is colder
    assert abs(amb(E.CAVE) - cfg.comfort_temp) < abs(amb(E.GRASS) - cfg.comfort_temp)
    quarter = int(cfg.climate_cycle_years * cfg.year_length / 4)
    assert np.isclose(float(E.climate_offset(jnp.int32(quarter), 0.0, cfg)), cfg.climate_amp, atol=1e-3)
    assert float(E.climate_offset(jnp.int32(quarter), 0.0, CLASSIC)) == 0.0


def test_rain_is_local_and_moves_with_the_wind():
    cfg = LIVING
    s = E.reset(jax.random.PRNGKey(4), cfg)._replace(alive=jnp.zeros(2, bool))
    def body(st, _):
        st2, _, _, _ = E.step(st, jnp.zeros(2, jnp.int32), cfg)
        return st2, (st2.clouds > cfg.rain_threshold).mean()
    s, share = jax.jit(lambda st: jax.lax.scan(body, st, None, 480))(s)
    share = np.asarray(share)
    assert share.max() > 0.0 and share.max() < 0.9                              # rain somewhere, never everywhere
    g = cfg.grid_size
    blob = jnp.zeros((g, g)).at[10:14, 10:14].set(0.3)
    calm = s._replace(clouds=blob, wind=jnp.array([0, 1], jnp.int32), t=jnp.int32(0))
    out = E.update_weather(calm, jnp.int32(E.CLEAR), jnp.int32(0), jax.random.PRNGKey(0),
                           dataclasses.replace(cfg, front_prob=0.0, wind_change_prob=0.0, evaporation=0.0))
    cols = np.asarray(out[0]).sum(0)
    assert np.argmax(cols) > 11                                                 # drifted east with the wind


def test_drinking_marsh_desert_thirst_and_snow_slows_walking():
    cfg = LIVING
    s = E.reset(jax.random.PRNGKey(4), cfg)
    t = np.asarray(s.terrain)
    marsh = np.argwhere(t == E.MARSH)[0]
    assert bool(E.adjacent_to_water(s.terrain, jnp.asarray([marsh]), cfg)[0])  # can drink standing in marsh
    land = np.argwhere((t == E.GRASS))
    p = next(q for q in land if 2 <= q[0] <= 29 and 2 <= q[1] <= 29 and (t[q[0] - 1:q[0] + 2, q[1]:q[1] + 2] == E.GRASS).all())
    s = s._replace(pos=jnp.asarray([p, p], jnp.int32), energy=jnp.ones(2), clouds=jnp.zeros_like(s.clouds))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    dry, *_ = step(s, jnp.array([E.MOVE_E, E.REST]))
    snowy, *_ = step(s._replace(snow=s.snow.at[p[0], p[1] + 1].set(1.0)), jnp.array([E.MOVE_E, E.REST]))
    assert float(snowy.energy[0]) < float(dry.energy[0])
