"""Sleep consolidation: the day waits for sleep, each step is learned exactly once, unslept
experience is partly forgotten, and the long-term store holds what the agent chose then."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.ppo_multi import MAConfig, consolidation_plan, make_train


def _plan(sleep, stage, died=None, span=3, need=2.0):
    T = len(sleep)
    col = lambda x, dt=jnp.float32: jnp.asarray(x, dt)[:, None]
    died = died or [False] * T
    return [np.asarray(x)[:, 0] for x in consolidation_plan(
        col(died, bool), col(died, bool), col(sleep), col(stage, jnp.int32), need, span)]


def test_a_step_is_learned_in_full_once_slept_on_and_waits_otherwise():
    # 6 new steps; deep sleep of 2 ticks at steps 2-3; the last 3 ticks form the carried tail
    w, fin, cons, nxt = _plan([0, 0, 1, 1, 0, 0], [0] * 6)
    assert list(w[:3]) == [1, 1, 1]                      # steps 0-2 have 2 ticks of deep sleep after them
    # step 3 (1 tick after it: half consolidated) and 4-5 (none) are in the tail: they wait
    assert list(fin) == [True, True, True, False, False, False] and list(w[3:]) == [0, 0, 0]
    assert list(nxt) == [1, 1, 1]
    # A step that leaves the store before its sleep is learned in proportion to the sleep it got
    w, fin, _, _ = _plan([0, 0, 0, 0, 0, 1], [0] * 6, span=2)
    assert np.allclose(w[:4], [0.5] * 4) and fin[:4].all()


def test_carried_steps_are_learned_once_then_only_give_context():
    # Window = 3 carried steps (two waited, one already learned) + 3 new; sleep comes in the new part
    w, fin, _, _ = _plan([0, 0, 0, 1, 1, 0], [1, 1, 2, 0, 0, 0])
    assert list(w[:3]) == [1, 1, 0]                     # waited -> learned now; already learned -> 0
    assert fin[0] and fin[1]
    # Dying settles everything before it in full: with one network per slot, deaths teach
    w, fin, _, _ = _plan([0, 0, 0, 0], [0] * 4, died=[False, False, True, False], span=2)
    assert list(w[:3]) == [1, 1, 1]


ENV = EnvConfig(grid_size=12, n_agents=3)


def test_the_trainer_carries_its_day_and_fills_the_long_term_store():
    cfg = MAConfig(num_worlds=2, num_steps=16, total_ticks=10 ** 6, num_minibatches=1, encoder="mlp",
                   hidden=16, update_epochs=1, memory_span=8, reservoir_size=8, reservoir_refresh=0.5,
                   consolidation_ticks=1.0)
    net, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    for _ in range(3):
        runner, m = update(runner, pop)
    before = np.asarray(runner[-1].reservoir.probs)
    runner, m = update(runner, pop)
    mem = runner[-1]
    assert mem.tail.reward.shape[:3] == (8, 3, 2) and mem.tail_carry.shape == (3, 2, 16)
    for k in ("learned_share", "waiting_share", "retention_kl", "long_term_filled"):
        assert np.isfinite(float(m[k])), k
    res = mem.reservoir
    assert np.asarray(res.filled).any(), "nothing consolidated in 4 updates"
    # A moment stored in the last update holds what the agent's brain chose there, as replayed
    # from the recurrent state it had going into it (stored after learning, with the new weights)
    fresh = np.asarray(res.filled) & ~np.isclose(np.asarray(res.probs), before).all(-1)
    params = runner[0]
    A, M = E.N_ACTIONS, E.EXTRA_LOGITS
    full = jnp.concatenate([res.mask, jnp.ones(res.mask.shape[:-1] + (M,), bool)], -1)
    _, lg, _ = jax.vmap(lambda p, c, o, r, mk: net.apply(p, c, (o[None], r[None], mk[None])))(
        params, res.carry, res.obs, res.reset, full)
    probs = jax.nn.softmax(jnp.where(res.mask, lg[:, 0, :, :A], -1e9))
    assert fresh.any()
    np.testing.assert_allclose(np.asarray(probs)[fresh], np.asarray(res.probs)[fresh], atol=1e-4)
    # Older moments were stored by an earlier brain: replaying them now differs (that is what
    # the retention term pulls back toward)
    old = np.asarray(res.filled) & ~fresh
    if old.any():
        assert not np.allclose(np.asarray(probs)[old], np.asarray(res.probs)[old], atol=1e-6)


def test_without_sleep_learning_nothing_is_carried():
    cfg = MAConfig(num_worlds=2, num_steps=8, total_ticks=10 ** 6, num_minibatches=1, encoder="mlp",
                   hidden=16, update_epochs=1, sleep_learning=False)
    _, init, update = make_train(ENV, cfg)
    runner, pop = init(jax.random.PRNGKey(1))
    runner, m = update(runner, pop)
    assert runner[-1].tail.reward.shape[0] == 0 and runner[-1].reservoir.filled.shape[1] == 0
    assert float(m["learned_share"]) == 1.0
