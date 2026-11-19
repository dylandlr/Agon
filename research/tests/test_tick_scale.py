"""Tick length (config.with_tick_minutes): finer ticks keep every rate in game time, and the
learner keeps consolidating in sleep when a day no longer fits in a rollout (pending change)."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig, with_reproduction, with_tick_minutes
from lifesim.ppo_multi import MAConfig, make_train


def _calm(cfg):
    return dataclasses.replace(cfg, climate_volatility=0.0, lightning_prob=0.0, season_temp=(18.0,) * 4,
                               day_temp_swing=0.0, small_prey_per_1000=0.0, large_prey_per_1000=0.0,
                               predators_per_1000=0.0, predator_respawn=0.0, small_prey_respawn=0.0,
                               large_prey_respawn=0.0)   # an empty slot refills: no animal may appear


def test_scale_and_calendar():
    base = EnvConfig(grid_size=16, n_agents=2)
    fine = with_tick_minutes(base, 1)
    assert fine.day_length == 1440 and fine.year_length == 40 * 1440
    assert abs(fine.cell_km - 4.0 / 60) < 1e-9 and abs(base.cell_km - 2.0) < 1e-9
    assert fine.hunger_rate * 1440 == base.hunger_rate * 48 or abs(fine.hunger_rate * 1440 - base.hunger_rate * 48) < 1e-9
    assert abs((1 - fine.contact_infection) ** 30 - (1 - base.contact_infection)) < 1e-9
    assert fine.disease_duration == base.disease_duration * 30 and fine.sleep_deep_ticks == 120
    assert abs(fine.shaping_gamma ** 30 - base.shaping_gamma) < 1e-9
    assert E.clock(1440 + 61, fine)["stamp"] == "Y1 Spring D2 01:01"
    rep = with_reproduction(fine)                     # either order gives the same lifespan in days
    assert rep.max_age == with_tick_minutes(with_reproduction(base), 1).max_age == 1440 * 70


def test_a_day_of_hunger_thirst_and_tiredness_is_the_same_at_any_tick():
    out = {}
    for minutes in (30, 2):
        cfg = with_tick_minutes(_calm(EnvConfig(grid_size=16, n_agents=1)), minutes)
        s = E.reset(jax.random.PRNGKey(0), cfg)
        off = lambda an: an._replace(alive=jnp.zeros_like(an.alive))   # a slot is kept even at 0 per 1,000
        s = s._replace(satiety=jnp.ones(1), hydration=jnp.ones(1), energy=jnp.ones(1), t=jnp.int32(cfg.day_length // 3),
                       small=off(s.small), large=off(s.large), predators=off(s.predators))
        step = lambda st, _: (E.step(st, jnp.array([E.REST]), cfg)[0], None)
        s, _ = jax.lax.scan(step, s, None, cfg.day_length // 2)          # half a day, awake and resting
        out[minutes] = np.array([float(s.satiety[0]), float(s.hydration[0]), float(s.sleepiness[0]),
                                 float(s.health[0])])
    assert np.allclose(out[30], out[2], atol=0.02), out


def test_sleep_consolidates_a_pending_change_when_the_day_does_not_fit():
    env_cfg = with_tick_minutes(_calm(EnvConfig(grid_size=12, n_agents=2, reward_mode="survival")), 2)
    cfg = MAConfig(num_worlds=2, num_steps=32, total_ticks=10 ** 6, hidden=16, encoder="mlp", num_minibatches=1,
                   day_ticks=env_cfg.day_length, tick_scale=env_cfg.tick_minutes / 30.0, reservoir_size=8)
    _, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    p0 = runner[0]
    # awake by day: nothing is learned yet, the day waits as a pending change
    states = runner[2]._replace(t=jnp.full((2,), env_cfg.day_length // 2, jnp.int32))
    runner = runner[:2] + (states,) + runner[3:]
    runner, m = update(runner, pop)
    same = jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, b: bool(jnp.all(a == b)), p0, runner[0]))
    pend = sum(float(jnp.abs(x).sum()) for x in jax.tree_util.tree_leaves(runner[8].pending))
    share = float(m["learned_share"])
    assert pend > 0 and share < 0.2                  # (untrained agents doze off now and then)
    assert same == (share == 0.0)
    # then sleep, unable to wake: the pending change is applied
    st = runner[2]
    st = st._replace(asleep=jnp.ones_like(st.asleep), collapsed=jnp.ones_like(st.collapsed),
                     sleepiness=jnp.ones_like(st.sleepiness), sleep_depth=jnp.full_like(st.sleep_depth, 10 ** 4))
    runner = runner[:2] + (st,) + runner[3:]
    runner, m = update(runner, pop)
    changed = not jax.tree_util.tree_all(jax.tree_util.tree_map(lambda a, b: bool(jnp.all(a == b)), p0, runner[0]))
    assert changed and float(m["learned_share"]) > 0.2
