"""Running out of clock is not death: advantages bootstrap from the cut-off state's value."""
import jax.numpy as jnp
import numpy as np

from lifesim.ppo_multi import gae


def test_a_time_limit_bootstraps_and_a_death_does_not():
    T = 3
    r = jnp.zeros((T, 1))
    v = jnp.ones((T, 1))
    done = jnp.array([[0.0], [0.0], [1.0]])
    gam, lam = 0.9, 1.0
    died = gae(r, v, done, jnp.zeros((T, 1)), jnp.zeros(1), gam, lam)
    cut = gae(r, v, done, jnp.array([[0.0], [0.0], [1.0]]), jnp.zeros(1), gam, lam)
    # Death: the return from the last step is 0, so its advantage is -V
    assert np.isclose(float(died[-1, 0]), -1.0)
    # Cut by the clock with V(cut-off state) = 1: the return continues, advantage gam - 1
    assert np.isclose(float(cut[-1, 0]), gam - 1.0)
    # Earlier steps see the same difference, discounted
    assert float(cut[0, 0]) > float(died[0, 0])


def test_ordinary_steps_match_textbook_gae():
    r = jnp.array([[1.0], [0.0]])
    v = jnp.array([[0.5], [0.2]])
    adv = gae(r, v, jnp.zeros((2, 1)), jnp.zeros((2, 1)), jnp.array([0.4]), 0.9, 0.8)
    d1 = 0.0 + 0.9 * 0.4 - 0.2
    d0 = 1.0 + 0.9 * 0.2 - 0.5
    assert np.allclose(np.asarray(adv[:, 0]), [d0 + 0.9 * 0.8 * d1, d1])
