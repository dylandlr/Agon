"""Reference policies for single-agent worlds.

`scripted_forager` uses global knowledge of the map (all water, food, forest
and predators), so it is a reference ceiling for what a competent solitary
agent can achieve, not a fair competitor to a learner with a local view.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import env as E
from .config import EnvConfig

_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def random_action(key, mask):
    """mask: (A,) -> scalar action, uniform over valid actions."""
    return jax.random.categorical(key, jnp.where(mask, 0.0, -1e9))


def _step_toward(pos, target, move_mask, g, away=False):
    rr, cc = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    d = jnp.maximum(jnp.abs(rr - pos[0]), jnp.abs(cc - pos[1]))
    tgt = jnp.argmin(jnp.where(target, d, 10 ** 6).reshape(-1))
    tr, tc = tgt // g, tgt % g
    nxt = pos[None, :] + E.MOVE_DELTAS[1:]
    nd = jnp.maximum(jnp.abs(nxt[:, 0] - tr), jnp.abs(nxt[:, 1] - tc))
    nd = jnp.where(away, -nd, nd)
    return 1 + jnp.argmin(jnp.where(move_mask, nd, 10 ** 6))


def scripted_forager(state: E.State, mask, cfg: EnvConfig, i=0):
    """Action for agent i (default 0). Priorities, highest last: forage, seek forest when
    too cold or hot, seek water when thirsty, gather, rest at night, eat,
    drink, flee nearby predators."""
    g = cfg.grid_size
    m, p = mask[i], state.pos[i]
    water = state.terrain == E.WATER
    wp = jnp.pad(water, 1)
    near_water = jnp.zeros((g, g), bool)
    for dr, dc in _OFFSETS:
        near_water = near_water | wp[1 + dr:1 + dr + g, 1 + dc:1 + dc + g]
    near_water = near_water & ~water
    forest = state.terrain == E.FOREST
    food = (state.plants > 0.5) | (state.meat > 0.5)
    pred = jnp.zeros((g, g), bool).at[state.predators.pos[:, 0], state.predators.pos[:, 1]].max(
        state.predators.alive)
    pred_dist = jnp.min(jnp.where(state.predators.alive,
                                  jnp.max(jnp.abs(state.predators.pos - p), -1), 99))
    thirsty = state.hydration[i] < 0.5
    hungry = state.satiety[i] < 0.6
    cold = state.body_temp[i] < -0.35
    hot = state.body_temp[i] > 0.3
    night = E.is_night(state.t, cfg)
    in_forest = forest[p[0], p[1]]
    carried = state.inv[i].sum()
    hand = state.hand[i]
    r0, c0 = p[0], p[1]
    # Food to pick up here, best first; the scripted agent must select before grasping
    food_here = jnp.where(state.cooked[r0, c0] > 0.1, E.COOKED_MEAT,
                          jnp.where(state.meat[r0, c0] > 0.1, E.RAW_MEAT,
                                    jnp.where(state.plants[r0, c0] > 0.1, E.PLANT, -1)))
    a_grasp = jnp.where(hand == food_here, E.GRASP, E.SELECT_0 + jnp.maximum(food_here, 0))
    inv = state.inv[i]
    best_food = jnp.where(inv[E.COOKED_MEAT] > 0.05, E.COOKED_MEAT,
                          jnp.where(inv[E.RAW_MEAT] > 0.05, E.RAW_MEAT, E.PLANT))
    has_food = (inv[E.COOKED_MEAT] + inv[E.RAW_MEAT] + inv[E.PLANT]) > 0.05
    a_eat = jnp.where(hand == best_food, E.EAT, E.SELECT_0 + best_food)
    moves = m[1:9]

    a = _step_toward(p, food, moves, g)
    a = jnp.where((cold | hot) & ~in_forest, _step_toward(p, forest, moves, g), a)
    a = jnp.where(thirsty, _step_toward(p, near_water, moves, g), a)
    a = jnp.where((food_here >= 0) & (carried < 3) & ~thirsty, a_grasp, a)
    a = jnp.where(night & (state.energy[i] < 0.95) & ~thirsty & (in_forest | ~cold), E.REST, a)
    a = jnp.where(state.energy[i] < 0.2, E.REST, a)
    a = jnp.where(hungry & has_food, a_eat, a)
    a = jnp.where(thirsty & m[E.DRINK], E.DRINK, a)
    a = jnp.where(pred_dist <= 2, _step_toward(p, pred, moves, g, away=True), a)
    return a
