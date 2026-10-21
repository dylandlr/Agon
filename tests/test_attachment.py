"""Attachment: survival felt toward particular others, grown from evidence they keep one alive."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=16, n_agents=2, climate_volatility=0.0, lightning_prob=0.0, reward_mode="survival",
                season_temp=(18.0,) * 4, day_temp_swing=0.0)


def _world(gap=1):
    s = E.reset(jax.random.PRNGKey(2), CFG)
    t = np.asarray(s.terrain)
    p = next((r, c) for r in range(2, 13) for c in range(2, 10) if (t[r, c:c + gap + 1] != E.WATER).all())
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    return s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                      pos=jnp.asarray([p, (p[0], p[1] + gap)], jnp.int32), t=jnp.int32(24),
                      satiety=jnp.ones(2), hydration=jnp.ones(2), energy=jnp.ones(2))


def run(s, actions, ticks):
    def body(st, _):
        st = st._replace(satiety=jnp.ones(2), hydration=jnp.ones(2), energy=jnp.ones(2), sleepiness=jnp.zeros(2))
        st2, r, _, _ = E.step(st, jnp.asarray(actions, jnp.int32), CFG)
        return st2, r
    return jax.jit(lambda st: jax.lax.scan(body, st, None, ticks))(s)


def test_grows_from_shared_survival_and_needs_no_reward_of_its_own():
    together, _ = run(_world(1), [E.REST, E.REST], 200)
    cold, _ = run(_world(1)._replace(body_temp=jnp.full(2, -0.5)), [E.REST, E.REST], 1)
    assert float(together.attachment[0, 1]) > 0.02 and float(together.attachment[1, 0]) > 0.02
    assert float(cold.attachment[0, 1]) > CFG.attach_company                    # warmth when cold counts more
    apart, _ = run(_world(12), [E.REST, E.REST], 200)
    assert float(apart.attachment.max()) == 0.0                                  # strangers out of sight: nothing


def test_blows_cut_it_and_time_apart_fades_it():
    s = _world(1)._replace(attachment=jnp.full((2, 2), 0.5) * (1 - jnp.eye(2)))
    west = E.MOVE_W - E.MOVE_N
    struck, _ = run(s, [E.REST, E.STRIKE_N + west], 1)
    assert float(struck.attachment[0, 1]) < 0.5 * (1 - CFG.attach_harm) + 0.01  # 1 struck 0
    far = _world(12)._replace(attachment=jnp.full((2, 2), 0.5) * (1 - jnp.eye(2)))
    later, _ = run(far, [E.REST, E.REST], CFG.attach_half_life)
    assert np.isclose(float(later.attachment[0, 1]), 0.25, atol=0.01)


def test_their_harm_is_felt_and_their_company_calms():
    s = _world(1)
    look = s.impression
    hurt = s._replace(health=s.health.at[1].set(0.3), impression=look)
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST], jnp.int32), CFG))
    _, r_stranger, _, _ = step(hurt)
    _, r_attached, _, _ = step(hurt._replace(attachment=jnp.array([[0.0, 1.0], [0.0, 0.0]])))
    assert float(r_attached[0]) < float(r_stranger[0]) - 0.1                    # their wound is felt as one's own loss
    tense = s._replace(stress=jnp.full(2, 0.6))
    calm_with, _, _, _ = step(tense._replace(attachment=jnp.array([[0.0, 1.0], [0.0, 0.0]])))
    calm_without, _, _, _ = step(tense)
    assert float(calm_with.stress[0]) < float(calm_without.stress[0])


def test_alike_and_fit_partners_bond_faster_and_danger_shared_binds():
    s = _world(1)
    g = s.genotype
    alike = s._replace(genotype=g._replace(empathy=jnp.full(2, 0.5), vengefulness=jnp.full(2, 0.2),
                                           reciprocity=jnp.full(2, 0.6), kin_care=jnp.full(2, 0.5),
                                           social_reward=jnp.ones(2), novelty=jnp.ones(2)))
    unlike = s._replace(genotype=g._replace(empathy=jnp.array([0.0, 1.0]), vengefulness=jnp.array([1.0, 0.0]),
                                            reciprocity=jnp.array([0.0, 1.0]), kin_care=jnp.array([0.0, 1.0]),
                                            social_reward=jnp.array([0.4, 2.5]), novelty=jnp.array([0.4, 2.5])))
    a, _ = run(alike, [E.REST, E.REST], 100)
    u, _ = run(unlike, [E.REST, E.REST], 100)
    assert float(a.attachment[0, 1]) > 1.5 * float(u.attachment[0, 1])
    weak = alike._replace(health=jnp.array([1.0, 0.3]))
    w, _ = run(weak, [E.REST, E.REST], 1)
    f, _ = run(alike, [E.REST, E.REST], 1)
    assert float(w.attachment[0, 1]) < float(f.attachment[0, 1])            # a frail partner is a worse bet
    pred = s.predators._replace(alive=s.predators.alive.at[0].set(True),
                                pos=s.predators.pos.at[0].set(s.pos[0] + jnp.array([2, 0])))
    scared, _, _, _ = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST], jnp.int32), CFG))(alike._replace(predators=pred))
    calm, _, _, _ = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST], jnp.int32), CFG))(alike)
    assert float(scared.attachment[0, 1]) > float(calm.attachment[0, 1]) + 0.01   # danger faced together
