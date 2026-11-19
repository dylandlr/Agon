"""Sleep: a built-in need with real costs (unawareness) and real benefits (energy, calm, healing)."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=16, n_agents=2, climate_volatility=0.0, lightning_prob=0.0,
                season_temp=(18.0,) * 4, day_temp_swing=0.0, founder_spread=0)
STEP = jax.jit(lambda s, a: E.step(s, a, CFG))


def _world(t=24):
    s = E.reset(jax.random.PRNGKey(2), CFG)
    terr = np.asarray(s.terrain)
    spot = next((r, c) for r in range(2, 13) for c in range(2, 12) if (terr[r, c:c + 2] != E.WATER).all())
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    return s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                      pos=jnp.asarray([spot, (spot[0], spot[1] + 1)], jnp.int32), t=jnp.int32(t),
                      satiety=jnp.ones(2), hydration=jnp.ones(2), energy=jnp.full(2, 0.5))


def act(s, a0, a1=E.REST):
    return STEP(s, jnp.asarray([a0, a1], jnp.int32))


def _day_and_night():
    s = _world()
    day = next(t for t in range(48) if not bool(E.is_night(jnp.int32(t), CFG)))
    night = next(t for t in range(48) if bool(E.is_night(jnp.int32(t), CFG)))
    return s, day, night


def test_sleepiness_builds_awake_faster_at_night_and_sleep_removes_it():
    s, day, night = _day_and_night()
    d, *_ = act(s._replace(t=jnp.int32(day)), E.REST)
    n, *_ = act(s._replace(t=jnp.int32(night)), E.REST)
    assert np.isclose(float(d.sleepiness[0]), CFG.sleep_pressure)
    assert np.isclose(float(n.sleepiness[0]), CFG.sleep_pressure * CFG.sleep_night_mult)
    tired = s._replace(t=jnp.int32(night), sleepiness=jnp.full(2, 0.6))
    out = tired
    drops = []
    for _ in range(6):
        prev = float(out.sleepiness[0])
        out, *_ = act(out, E.SLEEP)
        drops.append(prev - float(out.sleepiness[0]))
    assert bool(out.asleep[0]) and drops[0] < drops[-1]                 # sleep deepens
    by_day, *_ = act(tired._replace(t=jnp.int32(day)), E.SLEEP)
    assert 0.6 - float(by_day.sleepiness[0]) < drops[0]                 # day sleep is shallower


def test_asleep_one_can_only_sleep_on_or_wake():
    s = _world()._replace(sleepiness=jnp.full(2, 0.4))
    s, *_ = act(s, E.SLEEP)
    mask = np.asarray(E.action_mask(s, CFG))[0]
    assert bool(s.asleep[0]) and set(np.flatnonzero(mask)) == {E.REST, E.SLEEP}
    still, *_ = act(s, E.MOVE_N)                                         # not possible asleep: sleeps on
    assert bool(still.asleep[0]) and tuple(np.asarray(still.pos[0])) == tuple(np.asarray(s.pos[0]))
    up, *_ = act(s, E.REST)
    assert not bool(up.asleep[0])


def test_exhaustion_forces_sleep_until_rested_enough():
    s = _world()._replace(sleepiness=jnp.full(2, CFG.sleep_forced))
    s, _, _, info = act(s, E.MOVE_N)
    assert bool(s.asleep[0]) and bool(s.collapsed[0]) and bool(info.collapsed_now[0])
    assert set(np.flatnonzero(np.asarray(E.action_mask(s, CFG))[0])) == {E.SLEEP}
    for _ in range(40):
        if float(s.sleepiness[0]) < CFG.sleep_wake_block:
            break
        s, *_ = act(s, E.REST)
    assert bool(E.action_mask(s, CFG)[0, E.REST])                        # can choose to wake now


def test_asleep_one_sees_and_hears_only_what_is_adjacent():
    s = _world()
    s = s._replace(pos=s.pos.at[1].set(s.pos[0] + jnp.array([0, 3])))
    s = s._replace(asleep=jnp.array([True, False]), last_msg=s.last_msg.at[1].set(jnp.zeros(E.MSG_LEN, jnp.int32).at[:2].set(jnp.array([3, 4]))))
    v, c, r = CFG.view_size, E.N_OBS_CHANNELS, CFG.view_radius
    win = np.asarray(E.observe(s, CFG)[0, :v * v * c]).reshape(v, v, c)
    assert win[r - 1:r + 2, r - 1:r + 2].sum() > 0
    far = np.ones((v, v), bool); far[r - 1:r + 2, r - 1:r + 2] = False
    assert win[far].sum() == 0
    assert float(E.heard_messages(s, CFG)[0].sum()) == 0.0                # the voice 3 cells away is not heard
    awake = s._replace(asleep=jnp.zeros(2, bool))
    assert float(E.heard_messages(awake, CFG)[0].sum()) > 0.0


def test_being_struck_or_touched_wakes_a_sleeper():
    s = _world()._replace(sleepiness=jnp.full(2, 0.5), asleep=jnp.array([True, False]))
    east, west = E.MOVE_E - E.MOVE_N, E.MOVE_W - E.MOVE_N
    out, _, _, info = act(s, E.SLEEP, E.TOUCH_N + west)
    assert bool(info.woken[0]) and not bool(out.asleep[0])
    out, _, _, info = act(s, E.SLEEP, E.STRIKE_N + west)
    assert bool(info.woken[0]) and not bool(out.asleep[0])
    calm, _, _, info = act(s, E.SLEEP, E.REST)
    assert bool(calm.asleep[0]) and not bool(info.woken[0])


def test_sleep_restores_energy_calms_and_heals():
    s = _world()._replace(sleepiness=jnp.full(2, 0.8), stress=jnp.full(2, 0.3), health=jnp.full(2, 0.6),
                          sleep_depth=jnp.full(2, CFG.sleep_deep_ticks, jnp.int32), asleep=jnp.ones(2, bool))
    out, *_ = act(s, E.SLEEP, E.REST)                  # 0 sleeps on, 1 wakes and rests
    assert float(out.energy[0]) > float(out.energy[1])
    assert float(out.stress[0]) < float(out.stress[1])
    assert float(out.health[0]) > float(out.health[1])


def test_going_without_sleep_stresses_and_narrows_sight():
    s = _world()
    rested, *_ = act(s._replace(sleepiness=jnp.full(2, 0.2)), E.REST, E.REST)
    worn, *_ = act(s._replace(sleepiness=jnp.full(2, 0.92)), E.REST, E.REST)
    assert float(worn.stress[0]) > float(rested.stress[0])
    reach = np.asarray(E.sight_radius(s._replace(sleepiness=jnp.array([0.2, 0.92])), CFG))
    assert reach[0] == CFG.view_radius and reach[1] < CFG.view_radius


def test_sleepiness_is_felt_and_sleep_feels_good_when_tired():
    one, z = jnp.ones(1), jnp.zeros(1)
    w = lambda sl: float(E.wellbeing(one, one, one, one, z, z, one, CFG, jnp.full(1, sl))[0])
    assert w(0.0) > w(0.5) > w(0.9)
    assert (w(0.5) - w(0.9)) > 2 * (w(0.0) - w(0.2))                    # convex, like the other needs


def test_learning_counts_only_what_sleep_consolidates():
    from lifesim.ppo_multi import consolidation_weights
    T = 12
    done = np.zeros(T, bool)
    died = np.zeros(T, bool)
    q = np.zeros(T)
    q[6:10] = 1.0                                   # four ticks of deep sleep after a morning awake
    w = np.asarray(consolidation_weights(jnp.asarray(done), jnp.asarray(died), jnp.asarray(q), 8.0))
    assert np.allclose(w[:6], 0.5)                  # half consolidated by 4 of the 8 ticks needed
    assert np.allclose(w[10:], 0.0)                 # after the last sleep: not (yet) learned
    full = np.asarray(consolidation_weights(jnp.asarray(done), jnp.asarray(died), jnp.asarray(q * 2), 8.0))
    assert np.allclose(full[:6], 1.0)
    done2, died2 = done.copy(), died.copy()
    done2[3], died2[3] = True, True                 # a life ended in death at tick 3: learned in full
    w2 = np.asarray(consolidation_weights(jnp.asarray(done2), jnp.asarray(died2), jnp.asarray(q), 8.0))
    assert np.allclose(w2[:4], 1.0) and np.allclose(w2[4:6], 0.5)
