"""Social learning: who an agent watches, and how much each observation counts.

Each agent, at each step, may watch ONE other agent within its view and learn
to predict that agent's action in that agent's situation (behavioral cloning
on the demonstrator's observation). The imitation loss is added to PPO, scaled
by the observer's heritable social_learning gene.

Copying biases (which visible agent is watched):
  "uniform"  any visible agent, equally
  "success"  agents that look healthy are watched more (a success/prestige cue)
  "kin"      agents that look similar (appearance vector) are watched more

Advantage weighting: by default only actions that turned out better than the
demonstrator expected (positive advantage) are imitated, weighted by how much
better. Observers cannot see advantages directly; this stands in for seeing
that an action paid off (the demonstrator ate, got warm, made a kill).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

BIASES = ("uniform", "success", "kin")


def select_demonstrators(key, pos, valid, health, adv, appearance, view_radius: int,
                         bias: str = "success", bias_strength: float = 5.0,
                         positive_only: bool = True, spoke=None,
                         speech_attention: float = 0.0, teach_bonus: float = 0.0):
    """Choose one demonstrator per (step, observer, world).

    pos (T, N, B, 2), valid (T, N, B) bool, health (T, N, B), adv (T, N, B),
    appearance (N, 3), spoke (T, N, B) bool or None.

    Language: an agent that speaks while acting draws attention (speech_attention
    adds to its chance of being watched) and its demonstration counts more
    (weight x (1 + teach_bonus)). Returns demo (T, N, B) int32 index of the watched agent
    (meaningless where weight is 0) and weight (T, N, B) float32.
    """
    assert bias in BIASES, bias
    t, n, b = valid.shape
    p = jnp.moveaxis(pos, 2, 1)                               # (T, B, N, 2)
    v = jnp.moveaxis(valid, 2, 1)                             # (T, B, N)
    h = jnp.moveaxis(health, 2, 1)
    a = jnp.moveaxis(adv, 2, 1)
    dist = jnp.max(jnp.abs(p[:, :, :, None, :] - p[:, :, None, :, :]), axis=-1)  # (T, B, i, j)
    visible = (dist <= view_radius) & ~jnp.eye(n, dtype=bool) & v[..., :, None] & v[..., None, :]
    if positive_only:
        visible = visible & (a[..., None, :] > 0)
    if bias == "success":
        score = bias_strength * jnp.broadcast_to(h[..., None, :], visible.shape)
    elif bias == "kin":
        app_d = jnp.linalg.norm(appearance[:, None, :] - appearance[None, :, :], axis=-1)  # (N, N)
        score = -bias_strength * jnp.broadcast_to(app_d, visible.shape)
    else:
        score = jnp.zeros(visible.shape)
    if spoke is not None:
        sp = jnp.moveaxis(spoke, 2, 1).astype(jnp.float32)         # (T, B, N)
        score = score + speech_attention * sp[..., None, :]
    logits = jnp.where(visible, score, -1e9)
    demo = jax.random.categorical(key, logits, axis=-1)       # (T, B, N)
    any_visible = jnp.any(visible, axis=-1)
    demo_adv = jnp.take_along_axis(a, demo, axis=-1)          # adv[j] for the chosen j
    if positive_only:
        pos_adv = jnp.where(any_visible, demo_adv, 0.0)
        scale = jnp.sum(pos_adv) / jnp.maximum(jnp.sum(any_visible), 1)
        weight = jnp.where(any_visible, pos_adv / (scale + 1e-8), 0.0)
    else:
        weight = any_visible.astype(jnp.float32)
    if spoke is not None:
        weight = weight * (1 + teach_bonus * jnp.take_along_axis(sp, demo, axis=-1))
    return (jnp.moveaxis(demo, 1, 2).astype(jnp.int32),
            jnp.moveaxis(weight, 1, 2).astype(jnp.float32))


def restrict_to_observer_view(demo_obs, demo_pos, obs_pos, night, cfg):
    """What an observer could actually see of a demonstrator's situation.

    The demonstrator's observation covers cells up to view_radius around IT, which
    may lie beyond the observer's own sight, and includes private state (hunger,
    inventory, what it hears). Cells outside the observer's own range (day or
    night radius) are blanked, and so is the private state; time and weather,
    which everyone perceives, are kept.

    demo_obs (..., D), demo_pos and obs_pos (..., 2), night (...) bool.
    """
    from . import env as E
    r, v, c = cfg.view_radius, cfg.view_size, E.N_OBS_CHANNELS
    n_win = v * v * c
    offs = jnp.arange(-r, r + 1)
    rel = (demo_pos - obs_pos)[..., None, None, :] + jnp.stack(jnp.meshgrid(offs, offs, indexing="ij"), -1)
    sight = jnp.where(night, cfg.night_view_radius, r)[..., None, None]
    seen = jnp.max(jnp.abs(rel), axis=-1) <= sight                       # (..., V, V)
    win = demo_obs[..., :n_win].reshape(demo_obs.shape[:-1] + (v, v, c)) * seen[..., None]
    body = demo_obs[..., n_win:]
    shared_start = E.SHARED_START                                       # after own body, inventory, hand, torch
    shared_end = shared_start + 5 + E.N_WEATHER                          # time of day and year, weather
    keep = jnp.zeros(body.shape[-1]).at[shared_start:shared_end].set(1.0)
    return jnp.concatenate([win.reshape(demo_obs.shape[:-1] + (n_win,)), body * keep], axis=-1)


def gather_demo(x, demo):
    """x (T, N, B, ...) agent-major data; demo (T, N, B) -> x[t, demo[t,i,b], b, ...]."""
    idx = demo.reshape(demo.shape + (1,) * (x.ndim - 3))
    return jnp.take_along_axis(x, idx, axis=1)
