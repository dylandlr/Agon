"""World model and ideas: it learns to predict its own next state, imagination only nudges choices
as far as the learned trust allows, and the learner is identical whether experience comes from
the trainer's own rollout or from outside (the continuous world)."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.policy import DECODE_INDEX, make_network
from lifesim.ppo_multi import MAConfig, make_train

ENV = EnvConfig(grid_size=12, n_agents=3)


def _x(cfg, t=3, b=2, seed=0):
    D, A = E.obs_dim(cfg), E.N_ACTIONS + E.EXTRA_LOGITS
    obs = jax.random.normal(jax.random.PRNGKey(seed), (t, b, D))
    return obs, jnp.zeros((t, b), bool), jnp.ones((t, b, A), bool)


def test_zero_trust_means_ideas_change_nothing():
    """At birth trust is 0: the policy is exactly its habits. With trust, the best imagined
    action gains probability."""
    net = make_network(ENV, 32, "mlp", messages=True, world_model=True)
    x = _x(ENV)
    p = net.init(jax.random.PRNGKey(1), jnp.zeros((2, 32)), x)
    _, out = net.apply(p, jnp.zeros((2, 32)), x, method=net.trace)
    np.testing.assert_allclose(np.asarray(out["logits"]), np.asarray(out["prior"]), atol=1e-6)
    p2 = jax.tree_util.tree_map(lambda v: v, p)
    p2["params"]["trust"] = jnp.float32(5.0)
    _, out2 = net.apply(p2, jnp.zeros((2, 32)), x, method=net.trace)
    best = np.take_along_axis(np.asarray(out2["cand"]), np.asarray(out2["q"]).argmax(-1)[..., None], -1)[..., 0]
    up = np.take_along_axis(np.asarray(out2["logits"] - out2["prior"]), best[..., None], -1)
    assert (up > 0).all()


def test_world_model_learns_to_predict_its_own_state():
    cfg = MAConfig(num_worlds=4, num_steps=32, total_ticks=10 ** 6, num_minibatches=2, encoder="mlp",
                   hidden=32, world_model=True, sleep_learning=False, update_epochs=4)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    first = None
    for _ in range(12):
        runner, m = update(runner, pop)
        if first is None:
            first = float(m["wm_own"])
    for k in ("wm_latent", "wm_reward", "wm_own", "trust"):
        assert np.isfinite(float(m[k]))
    assert float(m["wm_own"]) < 0.7 * first           # predicting its own next needs gets better


def test_learner_on_outside_experience_matches_learning_with_acting_values():
    """learn_outside recomputes values and log-probs from the weights that acted; that must equal
    learning from the values recorded while acting (as the trainer does). This is what lets the
    continuous world keep learning with the trainer's own update."""
    from lifesim.ppo_multi import Transition
    # (sleep learning off: in 8 waking ticks nothing would be consolidated, so nothing learned)
    cfg = MAConfig(num_worlds=1, num_steps=8, total_ticks=10 ** 6, num_minibatches=1, encoder="mlp",
                   hidden=32, world_model=True, update_epochs=2, sleep_learning=False)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(3))
    params, opt_state, states, obs, mask, reset, carry, key, _ = runner
    s = jax.tree_util.tree_map(lambda x: x[0], states)                 # the one world
    init_carry, rows = carry, []
    for t in range(8):
        o, m = E.observe(s, ENV)[:, None], E.action_mask(s, ENV)[:, None]
        key, k = jax.random.split(key)
        carry, a, msg, logp, v = update.act(params, carry, o, m, reset, k)
        s2, r, done, info = E.step(s, a[:, 0], ENV, msg[:, 0])
        rows.append(Transition(reset, info.died[:, None], s.alive[:, None], a, v, r[:, None], logp, o, m, msg,
                               (info.spoke & s.alive)[:, None], s.pos[:, None], s.health[:, None],
                               jnp.broadcast_to(E.is_night(s.t, ENV), (3, 1)), info.died[:, None],
                               info.slept.astype(jnp.float32)[:, None], jnp.ones((3, 1)), jnp.zeros((3, 1)), jnp.zeros((3, 1), jnp.int32)))
        reset = info.born[:, None]
        s = s2
    traj = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *rows)
    o, m = E.observe(s, ENV)[:, None], E.action_mask(s, ENV)[:, None]
    _, _, _, _, last_v = update.act(params, carry, o, m, reset, key)
    a = update.learn_core(params, opt_state, traj, init_carry, last_v, jax.random.PRNGKey(9), pop)
    blank = traj._replace(value=jnp.zeros_like(traj.value), log_prob=jnp.zeros_like(traj.log_prob))
    b = update.learn_outside(params, opt_state, blank, init_carry, o, m, reset, carry, jax.random.PRNGKey(9), pop)
    for x, y in zip(jax.tree_util.tree_leaves(a[0]), jax.tree_util.tree_leaves(b[0])):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)
    assert any(not np.allclose(x, y) for x, y in zip(jax.tree_util.tree_leaves(params),
                                                     jax.tree_util.tree_leaves(b[0])))


def test_inheritance_lamarckian_vs_darwinian():
    """parent: the child starts from what the parent has learned; innate: from the brain the parent
    was born with, so learning does not pass down through the brain."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from continuous_world import inherit_brain
    born = {"w": jnp.zeros((3, 4)), "b": jnp.zeros((3,))}
    learned = {"w": born["w"].at[0].set(1.0), "b": born["b"].at[0].set(2.0)}     # parent 0 learned something
    opt = ({"w": jnp.ones((3, 4)), "b": jnp.ones((3,))}, jnp.int32(7))
    p, inn, o = inherit_brain(learned, born, opt, 2, 0, "parent", 0.0, jax.random.PRNGKey(0), 3)
    assert float(p["w"][2].mean()) == 1.0 and float(inn["w"][2].mean()) == 1.0
    assert float(o[0]["w"][2].sum()) == 0.0 and int(o[1]) == 7                  # fresh moments, shared count kept
    p, inn, _ = inherit_brain(learned, born, opt, 2, 0, "innate", 0.0, jax.random.PRNGKey(0), 3)
    assert float(p["w"][2].mean()) == 0.0 and float(p["b"][2]) == 0.0           # not what the parent learned
    assert float(p["w"][0].mean()) == 1.0                                        # the parent keeps its own
    varied = {"w": born["w"].at[0].set(jnp.arange(4.0)), "b": born["b"]}
    p, _, _ = inherit_brain(varied, born, None, 2, 0, "parent", 0.1, jax.random.PRNGKey(0), 3)
    diff = np.asarray(p["w"][2] - varied["w"][0])
    assert 0 < np.abs(diff).max() < 1.0                                          # small mutations
