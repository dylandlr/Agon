"""MAConfig.share_trunk: the trunk (encoder, memory, world model) is one for the population, kept
as tied copies; the heads stay each agent's own."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim.config import EnvConfig
from lifesim.ppo_multi import TRUNK, MAConfig, make_train

ENV = EnvConfig(grid_size=12, n_agents=3)


def _spread(params, names):
    """Largest difference between agents' copies over the named modules."""
    out = 0.0
    for k in names:
        if k in params["params"]:
            for x in jax.tree_util.tree_leaves(params["params"][k]):
                out = max(out, float(jnp.abs(x - x[:1]).max()))
    return out


def test_tied_trunk_stays_identical_while_heads_part():
    cfg = MAConfig(num_worlds=2, num_steps=16, total_ticks=10 ** 6, num_minibatches=2, encoder="mlp",
                   hidden=32, world_model=True, sleep_learning=False, update_epochs=2, share_trunk=True)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    assert _spread(runner[0], TRUNK) == 0.0                     # started identical
    for _ in range(4):
        runner, m = update(runner, pop)
    params = runner[0]
    assert _spread(params, TRUNK) < 1e-6                        # still one trunk
    assert _spread(params, ("pol_out", "val_out")) > 1e-6       # heads are their own
    assert np.isfinite(float(m["wm_own"]))


def test_untied_trunks_part():
    cfg = MAConfig(num_worlds=2, num_steps=16, total_ticks=10 ** 6, num_minibatches=2, encoder="mlp",
                   hidden=32, world_model=False, sleep_learning=False, update_epochs=2, share_trunk=False)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    assert _spread(runner[0], TRUNK) > 1e-3                     # their own from the start


def test_tied_trunk_holds_with_sleep_consolidation():
    """The pending-gradient path (learning applied on sleep) keeps the copies together too."""
    cfg = MAConfig(num_worlds=2, num_steps=16, total_ticks=10 ** 6, num_minibatches=2, encoder="mlp",
                   hidden=32, world_model=False, sleep_learning=True, consolidation="pending",
                   update_epochs=1, share_trunk=True)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    for _ in range(6):
        runner, m = update(runner, pop)
    assert _spread(runner[0], TRUNK) < 1e-6
