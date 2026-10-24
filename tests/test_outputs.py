"""What a brain puts out besides its action: the aim, the symbols said and the symbols written, from
their own branch (policy.ActorCritic.msg_net), with the Lowe et al. (2019) controls."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.policy import make_network, sample_outputs, split_logits
from lifesim.ppo_multi import MAConfig, make_train

ENV = dataclasses.replace(EnvConfig(grid_size=12, n_agents=3), writing=True)


def test_channel_controls_silence_or_shuffle_what_is_heard():
    s = E.reset(jax.random.PRNGKey(0), ENV)
    out = jnp.array([[5, 1, 2, 3, 0, 0, 0, 0, 2], [0] * 9, [7, 7, 7, 7, 0, 0, 0, 0, 2]])
    heard = {}
    for ch in ("full", "none", "shuffled"):
        cfg = dataclasses.replace(ENV, channel=ch)
        heard[ch] = np.asarray(E.step(s, jnp.zeros(3, jnp.int32), cfg, out)[0].last_msg)
    assert heard["full"].tolist() == [[5, 1, 2, 3], [0] * 4, [7] * 4]
    assert (heard["none"] == 0).all()
    assert sorted(map(tuple, heard["shuffled"])) == sorted(map(tuple, heard["full"]))    # same messages, other mouths


def test_what_is_written_is_chosen_apart_from_what_is_said():
    from lifesim import minerals as mn
    s = E.reset(jax.random.PRNGKey(0), ENV)
    flint = E.MIN0 + mn.IDX["flint"]
    r, c = (int(x) for x in s.pos[0])
    s = s._replace(inv=s.inv.at[0, flint].set(1.0), hand=s.hand.at[0].set(flint), wood=s.wood.at[r, c].set(1.0),
                   asleep=jnp.zeros_like(s.asleep))
    assert bool(E.action_mask(s, ENV)[0, E.WRITE])
    out = jnp.array([[5, 1, 2, 3, 9, 9, 9, 9, 2], [0] * 9, [0] * 9])
    s1 = E.step(s, jnp.array([E.WRITE, E.REST, E.REST]), ENV, out)[0]
    assert np.asarray(s1.marks[r, c]).tolist() == [9, 9, 9, 9]                 # the written half, not what was said
    assert np.asarray(s1.last_msg[0]).tolist() == [5, 1, 2, 3]


def test_brains_start_mostly_silent_and_sample_every_output():
    net = make_network(ENV, 32, "mlp", messages=True)
    x = (jnp.zeros((1, 1, E.obs_dim(ENV))), jnp.zeros((1, 1), bool), jnp.ones((1, 1, E.N_ACTIONS + E.EXTRA_LOGITS), bool))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 32)), x)
    _, logits, _ = net.apply(params, jnp.zeros((1, 32)), x)
    act, aim, sym = split_logits(logits)
    assert act.shape[-1] == E.N_ACTIONS and aim.shape[-1] == E.N_HOW and sym.shape[-2:] == (E.OUT_TOKENS, E.VOCAB)
    speak = 1 - float(jax.nn.softmax(sym[0, 0, 0])[0])
    assert 0.06 < speak < 0.12                                   # ~9% speech at the start
    a, out = sample_outputs(jax.random.PRNGKey(1), logits[0, 0])
    assert out.shape == (E.OUT_SLOTS,) and 0 <= int(out[-1]) < E.N_HOW


def _one_update(**kw):
    cfg = MAConfig(num_worlds=2, num_steps=16, total_ticks=32, hidden=32, update_epochs=1, num_minibatches=1,
                   sleep_learning=False, **kw)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    p0 = runner[0]
    runner, m = update(runner, pop)
    moved = lambda k: float(sum(jnp.abs(a - b).sum() for a, b in zip(jax.tree_util.tree_leaves(p0["params"][k]),
                                                                     jax.tree_util.tree_leaves(runner[0]["params"][k]))))
    return moved, m, update, runner


def test_a_frozen_message_head_never_learns_and_stopgrad_keeps_messages_out_of_the_trunk():
    moved_f, _, _, _ = _one_update(msg_frozen=True)
    moved_s, _, _, _ = _one_update(msg_net="stopgrad")
    moved_b, m, _, _ = _one_update()
    assert moved_f("msg_head") == 0.0 and moved_b("msg_head") > 0
    assert abs(moved_s("mid_dense") - moved_f("mid_dense")) < 1e-5 * max(1.0, moved_f("mid_dense"))
    assert np.asarray(m["clip_fraction"]).shape == (4,)


def test_an_output_with_no_effect_carries_no_log_probability():
    _, _, update, runner = _one_update()
    params, _, states, obs, mask, reset, carry, key, _ = runner
    _, action, out, logp, _ = update.act(params, carry, obs, mask, reset, jax.random.PRNGKey(4))
    action, out, logp = np.asarray(action), np.asarray(out), np.asarray(logp)
    striking = (action >= E.STRIKE_N) & (action < E.STRIKE_N + 8)
    assert (logp[..., 1][~striking] == 0).all()                    # no strike: the aim did nothing
    assert (logp[..., 3][action != E.WRITE] == 0).all()            # no WRITE: the written symbols did nothing
