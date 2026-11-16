"""Staged conditions (EnvConfig.staged_conditions): hypothermia, heat illness, dehydration and starvation
are graded, persistent conditions that do the damage over time, instead of a flat drain past a line."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig

BASE = EnvConfig(grid_size=12, n_agents=2, max_steps=1000, climate_volatility=0.0, season_temp=(18.0,) * 4,
                 day_temp_swing=0.0, lightning_prob=0.0)
STAGED = dataclasses.replace(BASE, staged_conditions=True)


def _world(cfg):
    s = E.reset(jax.random.PRNGKey(5), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   conditions=jnp.zeros_like(s.conditions), pain_aversion=jnp.ones(2),
                   birth_age=jnp.full(2, int(25 * cfg.ticks_per_bio_year), jnp.int32))
    t = np.asarray(s.terrain)
    west, mid = next(((r, c - 1), (r, c)) for r in range(12) for c in range(1, 11)
                     if t[r, c] != E.WATER and t[r, c - 1] != E.WATER)
    return s._replace(pos=jnp.asarray([west, mid], jnp.int32))


def _run(cfg, s, ticks, body=None, satiety=None, hydration=None):
    """Hold the given body temperature and needs for agent 0 each tick (agent 1 is left alone);
    return the state after each tick."""
    out = []
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST]), cfg, jnp.zeros((2, E.OUT_SLOTS), jnp.int32)))
    for _ in range(ticks):
        r = {"satiety": jnp.ones(2), "hydration": jnp.ones(2)}   # needs held full unless a test overrides them
        if body is not None:
            r["body_temp"] = s.body_temp.at[0].set(body)
        if satiety is not None:
            r["satiety"] = r["satiety"].at[0].set(satiety)
        if hydration is not None:
            r["hydration"] = r["hydration"].at[0].set(hydration)
        s = s._replace(**r)
        s, _, _, info = step(s)
        out.append((s, info))
    return out


def test_off_by_default_no_severity_and_the_flat_drain_is_unchanged():
    assert BASE.staged_conditions is False
    s = _world(BASE)
    run = _run(BASE, s, 5, body=-0.5)
    assert float(np.abs(np.asarray(run[-1][0].ailment)).max()) == 0.0
    assert float(run[-1][0].health[0]) >= 0.99                 # -0.5 is above the flat line (-0.7): no damage
    run = _run(BASE, s, 5, body=-0.9)
    assert float(run[-1][0].health[0]) < 0.99                  # past it: flat damage from the first tick


def test_cold_is_graded_it_builds_slowly_and_recovers_slower_than_it_builds():
    s = _world(STAGED)
    run = _run(STAGED, s, 60, body=-0.5)                       # halfway between onset (-0.3) and the old line (-0.7)
    sev = [float(r[0].ailment[0, E.AIL_COLD]) for r in run]
    assert all(b >= a for a, b in zip(sev, sev[1:]))           # only builds while it stays cold
    assert 0.05 < sev[9] < sev[-1] < 0.51                      # a lag, then toward depth 0.5, never past it
    assert float(run[-1][0].ailment[1, E.AIL_COLD]) == 0.0     # the other agent, at ease, has none
    built = sev[-1]
    warm = _run(STAGED, run[-1][0], 20, body=0.0)              # warmed: the condition lingers, then fades
    fall = [float(r[0].ailment[0, E.AIL_COLD]) for r in warm]
    assert fall[-1] < built and fall[-1] > 0.4 * built         # recovery is slower than the build was
    assert (built - fall[-1]) < (sev[19] - sev[0])             # (20 ticks of fall < 20 ticks of rise)


def test_deep_cold_freezes_legs_and_arms_and_hurts_before_it_kills():
    s = _world(STAGED)
    run = _run(STAGED, s, 400, body=-1.0)
    first_pain = next(i for i, r in enumerate(run) if float(r[0].pain[0]) > 0)
    parts = np.asarray(run[min(first_pain + 30, len(run) - 1)][0].part_damage[0])
    assert parts[E.PART_LEGS] > 0 and parts[E.PART_ARMS] > 0   # frostbite on the extremities
    assert parts[E.PART_HEAD] == 0 and parts[E.PART_NECK] == 0
    died = [i for i, r in enumerate(run) if not bool(r[0].alive[0])]
    assert died, "held at -1.0 the agent should die"
    assert first_pain < died[0]                                # it hurt first
    cause = int(run[died[0]][1].death_cause[0])
    assert E.CAUSE_NAMES[cause] == "cold"                      # and the cause of death is still cold
    assert bool(run[-1][0].alive[1])                           # the agent held at ease lives


def test_severity_scales_the_damage_of_every_cause_and_names_its_cause():
    for ail, kw, name in [(E.AIL_HEAT, dict(body=1.2), "heat"), (E.AIL_THIRST, dict(hydration=0.0), "thirst"),
                          (E.AIL_HUNGER, dict(satiety=0.0), "starvation")]:
        s = _world(STAGED)
        run = _run(STAGED, s, 700, **kw)
        sev = [float(r[0].ailment[0, ail]) for r in run]
        assert sev[0] < sev[20] < sev[200]                     # builds over ticks, not instantly
        h = [float(r[0].health[0]) for r in run]
        assert h[5] > 0.95 and h[-1] < h[5]                    # a lag before real damage, then it falls
        died = [i for i, r in enumerate(run) if not bool(r[0].alive[0])]
        assert died and E.CAUSE_NAMES[int(run[died[0]][1].death_cause[0])] == name
