"""Pain shock and finishing off the downed (EnvConfig.pain_shock, downed_hit_mult, coup_prob).
RimWorld's model: pain past a threshold drops one; the downed cannot defend themselves."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=12, n_agents=2, max_steps=1000, climate_volatility=0.0, season_temp=(18.0,) * 4,
                day_temp_swing=0.0, lightning_prob=0.0, downed=True, coup_prob=0.0)


def _world(cfg=CFG):
    s = E.reset(jax.random.PRNGKey(5), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   conditions=jnp.zeros_like(s.conditions), pain_aversion=jnp.ones(2),
                   birth_age=jnp.full(2, int(25 * cfg.ticks_per_bio_year), jnp.int32))
    t = np.asarray(s.terrain)
    west, mid = next(((r, c - 1), (r, c)) for r in range(12) for c in range(1, 11)
                     if t[r, c] != E.WATER and t[r, c - 1] != E.WATER)
    return s._replace(pos=jnp.asarray([west, mid], jnp.int32))


def test_pain_past_the_threshold_drops_one_and_the_sensitive_drop_sooner():
    s = _world()
    down = lambda st: np.asarray(E.downed_of(st, CFG)).tolist()
    assert down(s._replace(pain=jnp.array([0.9, 0.5]))) == [True, False]
    assert down(s._replace(pain=jnp.array([0.5, 0.5]), pain_aversion=jnp.array([4.0, 1.0]))) == [True, False]
    # standing injury aches too: 0.6 x injury
    assert down(s._replace(injury=jnp.array([0.7, 0.0]), pain=jnp.array([0.4, 0.0]))) == [True, False]
    off = dataclasses.replace(CFG, pain_shock=0.0)
    assert not np.asarray(E.downed_of(s._replace(pain=jnp.array([1.5, 1.5])), off)).any()


def test_blows_hurt_and_the_shock_passes_as_the_pain_fades():
    s = _world()
    s1, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), CFG)
    assert float(s1.pain[1]) > 0.2 and float(s1.pain[0]) == 0.0
    shocked = s._replace(pain=jnp.array([0.0, 1.0]))
    assert bool(E.downed_of(shocked, CFG)[1])
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST]), CFG)[0])
    st, up_at = shocked, None
    for i in range(24):
        st = step(st)
        if up_at is None and not bool(E.downed_of(st, CFG)[1]):
            up_at = i
    assert up_at is not None and up_at >= 1 and bool(st.alive[1])       # down for a while, then up again


def test_blows_on_the_downed_land_square():
    s = _world()
    hurt = lambda st: float(st.health[1] - E.step(st, jnp.array([E.STRIKE_N + 2, E.REST]), CFG)[0].health[1])
    standing = hurt(s._replace(pain=jnp.array([0.0, 0.0])))
    down = hurt(s._replace(health=jnp.array([1.0, 1.0]), pain=jnp.array([0.0, 1.0])))
    assert np.isclose(down, CFG.downed_hit_mult * standing, rtol=0.05)


def test_a_blow_on_the_downed_can_end_it():
    cfg = dataclasses.replace(CFG, coup_prob=1.0)
    s = _world(cfg)
    s1, _, _, info = E.step(s._replace(pain=jnp.array([0.0, 1.0])), jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    assert not bool(s1.alive[1]) and int(info.death_cause[1]) == E.CAUSE_VIOLENCE
    s1, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)       # standing: no coup
    assert bool(s1.alive[1])


def test_one_feels_ones_own_pain_against_ones_tolerance_and_others_see_the_wince():
    s = _world()._replace(pain=jnp.array([0.4, 0.0]))
    obs = np.asarray(E.observe(s, CFG))
    v, r, c = CFG.view_size, CFG.view_radius, E.N_OBS_CHANNELS
    felt = obs[:, v * v * c + 23]                                         # last of the own-body features
    assert np.isclose(felt[0], 0.4 / 0.8) and felt[1] == 0.0
    wimp = np.asarray(E.observe(s._replace(pain_aversion=jnp.array([4.0, 1.0])), CFG))[0, v * v * c + 23]
    assert np.isclose(wimp, 1.0)                                          # the same hurt is shock to a wimp
    win = obs[1, :v * v * c].reshape(v, v, c)                             # agent 1 sees agent 0, one cell west
    assert np.isclose(win[r, r - 1, c - 1], 0.5) and win[r, r, c - 1] == 0.0


def test_the_world_model_learns_to_expect_pain():
    from lifesim.policy import DECODE_INDEX, DECODE_NAMES
    assert DECODE_NAMES[DECODE_INDEX.index(23)] == "pain"
