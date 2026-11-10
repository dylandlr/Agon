"""MAConfig.entropy_target: an entropy floor keeps a policy from collapsing to a few acts."""
import jax
import numpy as np

from lifesim.config import EnvConfig
from lifesim.ppo_multi import MAConfig, make_train

ENV = EnvConfig(grid_size=12, n_agents=3)


def _final_entropy(target, updates=10):
    cfg = MAConfig(num_worlds=2, num_steps=32, total_ticks=10 ** 6, num_minibatches=2, encoder="mlp",
                   hidden=32, sleep_learning=False, update_epochs=4, lr=3e-3, ent_coef=0.0005,
                   entropy_target=target)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    ents = []
    for _ in range(updates):
        runner, m = update(runner, pop)
        ents.append(float(m["entropy"]))
    return ents


def test_floor_holds_entropy_up():
    free = _final_entropy(0.0)
    floored = _final_entropy(1.5)
    assert all(np.isfinite(floored))
    # a hot learning rate and a tiny bonus drive the free policy down; the floor holds it higher
    assert np.mean(floored[-3:]) > np.mean(free[-3:]) + 0.1, (free, floored)
