"""Shared AND selfish reward: per-agent heritable alpha, a local group, and a mean over the living only."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import genotype as gt
from lifesim.config import EnvConfig
from lifesim.ppo_multi import MAConfig, make_train, mix_reward


def _args(reward, alive, pos=None):
    r = jnp.asarray([reward], jnp.float32)
    a = jnp.asarray([alive], bool)
    p = jnp.zeros((1, len(reward), 2), jnp.int32) if pos is None else jnp.asarray([pos], jnp.int32)
    return r, a, p


def test_mean_is_over_the_living_not_every_slot():
    r, a, p = _args([1.0, 3.0, 0.0, 0.0], [True, True, False, False])
    out = np.asarray(mix_reward(r, a, p, 1.0, 0))[0]
    assert np.allclose(out[:2], 2.0)                         # (1 + 3) / 2, not (1 + 3) / 4 = 1
    assert np.allclose(out[2:], 0.0)                         # the dead keep their own (zero) reward


def test_alpha_zero_is_selfish_alpha_one_is_the_group_and_alpha_can_differ_per_agent():
    r, a, p = _args([1.0, 3.0], [True, True])
    assert np.allclose(mix_reward(r, a, p, 0.0, 0)[0], [1.0, 3.0])
    assert np.allclose(mix_reward(r, a, p, 1.0, 0)[0], [2.0, 2.0])
    mixed = np.asarray(mix_reward(r, a, p, jnp.asarray([[0.0, 0.5]]), 0))[0]
    assert np.allclose(mixed, [1.0, 0.5 * 3.0 + 0.5 * 2.0])  # the selfish one keeps 1; the half-sharer gets 2.5


def test_local_group_only_shares_with_those_nearby():
    # agents 0 and 1 stand together, agent 2 is far away
    r, a, p = _args([1.0, 3.0, 10.0], [True, True, True], [[0, 0], [0, 2], [0, 30]])
    out = np.asarray(mix_reward(r, a, p, 1.0, 6))[0]
    assert np.allclose(out, [2.0, 2.0, 10.0])                # the lone one shares only with itself
    out = np.asarray(mix_reward(r, a, p, 1.0, 0))[0]
    assert np.allclose(out, [14 / 3] * 3)                    # the whole world: everyone gets the same mean


def test_a_dead_neighbour_is_not_part_of_the_group():
    r, a, p = _args([1.0, 3.0, 99.0], [True, True, False], [[0, 0], [0, 1], [0, 1]])
    assert np.allclose(np.asarray(mix_reward(r, a, p, 1.0, 6))[0][:2], [2.0, 2.0])


def test_adding_the_sharing_gene_did_not_change_any_other_trait_of_a_seed():
    key = jax.random.PRNGKey(3)
    g = gt.sample(key, 8)
    old_keys = jax.random.split(key, len(gt._FIELDS) - 1)        # how every other trait was drawn before
    expect = jax.random.uniform(old_keys[gt._FIELDS.index("patience")], (8,), minval=0.95, maxval=0.999)
    assert np.allclose(np.asarray(g.patience), np.asarray(expect))
    assert np.all((np.asarray(g.sharing) >= 0) & (np.asarray(g.sharing) <= 1))


def test_sharing_is_inherited_and_mutates_inside_its_range():
    key = jax.random.PRNGKey(4)
    pop = gt.sample(key, 32)
    m = pop._replace(sharing=jnp.ones(32))
    f = pop._replace(sharing=jnp.ones(32))
    child = gt.inherit(jax.random.PRNGKey(5), m, f, pop)
    assert float(child.sharing.mean()) > float(pop.sharing.mean())        # pulled toward the parents'
    c2, jumps = gt.mutate(jax.random.PRNGKey(6), child, pop, rate=1.0, size=2.5)
    assert float(c2.sharing.min()) >= 0 and float(c2.sharing.max()) <= 1
    assert jumps.shape[-1] == len(gt._FIELDS)


def test_training_runs_with_heritable_local_sharing_and_logs_it():
    env_cfg = EnvConfig(grid_size=12, n_agents=4)
    cfg = MAConfig(num_worlds=2, num_steps=16, total_ticks=32, num_minibatches=2, encoder="mlp", hidden=32,
                   alpha_heritable=True, share_radius=4)
    _, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    runner, m = update(runner, pop)
    n = float(m["sharing_n"])
    assert n > 0 and 0.0 <= float(m["sharing_sum"]) / n <= 1.0
    assert np.isfinite(float(m["entropy"])) and np.isfinite(float(m["own_reward"]))
