import jax
import jax.numpy as jnp
import numpy as np

from lifesim import social
from lifesim.config import EnvConfig
from lifesim.ppo_multi import MAConfig, make_train


def _setup(t=1, adv=(1.0, 1.0, 1.0)):
    # Agents 0 and 1 adjacent, agent 2 far away; one world
    pos = jnp.array([[0, 0], [0, 1], [20, 20]])[None, :, None, :].repeat(t, 0)   # (T, N, B, 2)
    valid = jnp.ones((t, 3, 1), bool)
    health = jnp.ones((t, 3, 1))
    a = jnp.array(adv)[None, :, None].repeat(t, 0)
    app = jnp.array([[0.0, 0, 0], [0.0, 0, 0], [1.0, 1, 1]])
    return pos, valid, health, a, app


def test_only_visible_agents_are_watched_and_never_oneself():
    pos, valid, health, adv, app = _setup()
    demo, w = social.select_demonstrators(jax.random.PRNGKey(0), pos, valid, health, adv, app, 5)
    assert int(demo[0, 0, 0]) == 1 and int(demo[0, 1, 0]) == 0
    assert float(w[0, 2, 0]) == 0.0            # agent 2 sees nobody
    assert float(w[0, 0, 0]) > 0.0


def test_only_actions_that_paid_off_are_copied_by_default():
    pos, valid, health, adv, app = _setup(adv=(1.0, -0.5, 1.0))
    demo, w = social.select_demonstrators(jax.random.PRNGKey(0), pos, valid, health, adv, app, 5)
    assert float(w[0, 0, 0]) == 0.0            # agent 1's action did worse than expected
    demo, w = social.select_demonstrators(jax.random.PRNGKey(0), pos, valid, health, adv, app, 5,
                                          positive_only=False)
    assert float(w[0, 0, 0]) == 1.0


def test_kin_bias_prefers_similar_looking_agents():
    t = 400
    pos = jnp.array([[0, 0], [0, 1], [1, 0]])[None, :, None, :].repeat(t, 0)    # all three adjacent
    valid, health = jnp.ones((t, 3, 1), bool), jnp.ones((t, 3, 1))
    adv = jnp.ones((t, 3, 1))
    app = jnp.array([[0.0, 0, 0], [0.0, 0, 0], [1.0, 1, 1]])                    # 1 looks like 0
    demo, _ = social.select_demonstrators(jax.random.PRNGKey(1), pos, valid, health, adv, app, 5,
                                          bias="kin", bias_strength=5.0)
    share_kin = float(jnp.mean(demo[:, 0, 0] == 1))
    demo_u, _ = social.select_demonstrators(jax.random.PRNGKey(1), pos, valid, health, adv, app, 5,
                                            bias="uniform")
    assert share_kin > 0.95 and 0.35 < float(jnp.mean(demo_u[:, 0, 0] == 1)) < 0.65


def test_gather_demo_reads_the_watched_agents_data():
    x = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)[..., None] * 1.0   # (T, N, B, 1)
    demo = jnp.full((2, 3, 4), 2, jnp.int32)
    out = social.gather_demo(x, demo)
    np.testing.assert_array_equal(np.asarray(out[:, 0]), np.asarray(x[:, 2]))


def test_training_runs_with_social_learning():
    env_cfg = EnvConfig(grid_size=12, n_agents=3)
    cfg = MAConfig(num_worlds=4, num_steps=16, total_ticks=32, num_minibatches=2,
                   encoder="mlp", hidden=32, imitation_coef=1.0, imitate_positive_only=False)
    _, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    for _ in range(2):
        runner, m = update(runner, pop)
    assert np.isfinite(float(m["imitation_loss"])) and float(m["watched_fraction"]) > 0


def test_speaking_demonstrators_draw_attention_and_count_more():
    t = 400
    pos = jnp.array([[0, 0], [0, 1], [1, 0]])[None, :, None, :].repeat(t, 0)
    valid, health, adv = jnp.ones((t, 3, 1), bool), jnp.ones((t, 3, 1)), jnp.ones((t, 3, 1))
    app = jnp.zeros((3, 3))
    spoke = jnp.zeros((t, 3, 1), bool).at[:, 2].set(True)          # only agent 2 talks
    demo, w = social.select_demonstrators(jax.random.PRNGKey(2), pos, valid, health, adv, app, 5,
                                          bias="uniform", spoke=spoke, speech_attention=3.0,
                                          teach_bonus=1.0)
    watched_speaker = demo[:, 0, 0] == 2
    assert float(jnp.mean(watched_speaker)) > 0.9
    assert float(jnp.mean(w[:, 0, 0][watched_speaker])) > 1.5 * float(jnp.mean(w[:, 0, 0][~watched_speaker]))



def test_observers_learn_only_from_what_they_can_see():
    from lifesim import env as E
    from lifesim.config import EnvConfig
    cfg = EnvConfig(grid_size=32, n_agents=2)
    v, r, c = cfg.view_size, cfg.view_radius, E.N_OBS_CHANNELS
    n_win = v * v * c
    demo_obs = jnp.ones((1, E.obs_dim(cfg)))
    demo_pos, obs_pos = jnp.array([[10, 13]]), jnp.array([[10, 10]])   # demonstrator 3 cells east
    out = np.asarray(social.restrict_to_observer_view(demo_obs, demo_pos, obs_pos, jnp.array([False]), cfg))[0]
    win = out[:n_win].reshape(v, v, c)
    # Demonstrator's window spans columns 8..18; the observer sees columns 5..15
    assert win[r, r - 5].sum() > 0 and win[r, r + 2].sum() > 0         # columns 8 and 15: seen
    assert win[r, r + 3].sum() == 0 and win[r, r + 5].sum() == 0       # columns 16 and 18: beyond sight
    body = out[n_win:]
    shared = slice(E.SHARED_START, E.SHARED_START + 5 + E.N_WEATHER)   # after own body, inventory, hand, torch
    assert body[shared].sum() == 10 and body.sum() == 10              # only time and weather survive
    dark = np.asarray(social.restrict_to_observer_view(demo_obs, demo_pos, obs_pos, jnp.array([True]), cfg))[0]
    assert dark[:n_win].sum() < out[:n_win].sum()                      # less is visible at night
