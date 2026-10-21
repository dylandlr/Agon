"""Convex felt needs and the cognitive map."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=32, n_agents=2, climate_volatility=0.0, lightning_prob=0.0)


def _w(h, cfg=CFG):
    one = jnp.ones(1)
    return float(E.wellbeing(one, one, jnp.full(1, h), one, jnp.zeros(1), jnp.zeros(1), one, cfg)[0])


def test_thirst_is_felt_mildly_when_slight_and_strongly_when_severe():
    slight = _w(1.0) - _w(0.8)
    severe = _w(0.3) - _w(0.1)
    assert severe > 3 * slight                              # same 0.2 drop, felt far more when parched
    assert _w(0.0) > 0.0                                    # living never feels worse than death (0)


def _state():
    s = E.reset(jax.random.PRNGKey(3), CFG)
    return s._replace(pos=jnp.array([[16, 16], [2, 2]], jnp.int32), t=jnp.int32(24))


def test_map_holds_only_what_was_seen_and_fades():
    s = _state()
    m = E.update_mind_map(jnp.zeros_like(s.mind_map), s.pos, s.alive, s.terrain, s.plants,
                          jnp.full(2, CFG.view_radius), CFG)
    rec = np.asarray(m[0, ..., 2])
    assert rec[16, 16] == 1.0 and rec[16, 21] == 1.0 and rec[16, 22] == 0.0 and rec[2, 2] == 0.0
    water = np.asarray(s.terrain == E.WATER)
    seen = rec > 0
    assert np.array_equal(np.asarray(m[0, ..., 0]) > 0, water & seen)          # only water it saw
    away = jnp.array([[30, 30], [2, 2]], jnp.int32)
    fade = lambda mm, ticks, sleep=False: [mm := E.update_mind_map(
        mm, away, s.alive, s.terrain, s.plants, jnp.full(2, CFG.view_radius), CFG,
        consolidate=jnp.full(2, sleep)) for _ in range(ticks)][-1]
    unslept = fade(m, CFG.map_fresh_half_life)
    assert np.isclose(float(unslept[0, 16, 16, 2]), 0.5, atol=1e-3)             # a new memory fades in a day
    slept = fade(fade(m, 1, sleep=True), CFG.map_half_life - 1)
    assert float(slept[0, 16, 16, 3]) == 1.0                                     # consolidated in sleep
    assert np.isclose(float(slept[0, 16, 16, 2]), 0.5, atol=0.02)               # then lasts ten days
    assert float(slept[0, 16, 16, 0]) == float(m[0, 16, 16, 0])                  # still knows what was there


def test_map_is_read_relative_to_self_and_is_part_of_the_observation():
    s = _state()
    s = s._replace(mind_map=E.update_mind_map(s.mind_map, s.pos, s.alive, s.terrain, s.plants,
                                              jnp.full(2, CFG.view_radius), CFG))
    p = CFG.map_cells
    obs = np.asarray(E.observe(s, CFG))
    assert obs.shape[1] == E.obs_dim(CFG)
    tail = obs[:, -E.map_features(CFG):].reshape(2, p, p, E.N_MAP_CHANNELS)
    assert np.allclose(tail, np.asarray(E.read_mind_map(s.mind_map, s.pos, CFG)).reshape(2, p, p, E.N_MAP_CHANNELS))
    c = p // 2
    assert tail[0, c - 1:c + 1, c - 1:c + 1, 2].max() == 1.0                  # what is around it is at the centre
    assert tail[0, 0, :, 3].max() == 0.0 and tail[1, 0, 0, 3] == 0.0           # beyond the world's edge


def test_step_updates_the_map():
    s = _state()
    out, _, _, _ = jax.jit(lambda st: E.step(st, jnp.full(2, E.REST, jnp.int32), CFG))(s)
    assert float(out.mind_map[0, 16, 16, 2]) == 1.0 and float(out.mind_map[1, 2, 2, 2]) == 1.0
    assert float(out.mind_map[0, 2, 2, 2]) == 0.0
