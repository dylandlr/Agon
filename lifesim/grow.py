"""Growing a world's slot count while it runs, so a continuous world has no population ceiling.

JAX compiles a world for fixed array sizes, so the number of slots (EnvConfig.n_agents) is fixed for
a compiled step. A continuous world that is nearly full instead grows: every structure sized by the
slot count (the world state, brains, optimiser moments, the learner's memory, recurrent state,
buffered experience) is copied into one of the new size, and the step is compiled again. Brains do
not depend on the slot count (an agent's observation is the same size in a world of any size).

Agent axes are found by comparison with a structure freshly built at the new size: an axis is an
agent axis where the old size is n_old and the new one is n_new. Everything else must match. New
slots take the fresh structure's values (unborn, empty, never seen) and are filled at birth.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def headroom(n: int) -> int:
    """Grow when fewer slots than this are free (not alive, not held for a pregnancy)."""
    return max(2, n // 8)


def free_slots(state) -> int:
    alive = np.asarray(state.alive)
    return int((~alive).sum() - ((np.asarray(state.gestation) > 0) & alive).sum())


def grow_like(old, fresh, n_old: int, n_new: int):
    """`old` copied into `fresh` (the same structure built for n_new slots) along its agent axes."""
    def one(o, f):
        o, f = jnp.asarray(o), jnp.asarray(f)
        if o.shape == f.shape and n_old not in o.shape:
            return o
        if o.ndim != f.ndim:
            raise ValueError(f"cannot grow {o.shape} into {f.shape}")
        idx = []
        for a, (so, sf) in enumerate(zip(o.shape, f.shape)):
            if so == n_old and sf == n_new:
                idx.append(slice(0, n_old))
            elif so == sf:
                idx.append(slice(None))
            else:
                raise ValueError(f"axis {a} of {o.shape} -> {f.shape} is not an agent axis")
        if all(s == slice(None) for s in idx):
            return o
        return f.astype(o.dtype).at[tuple(idx)].set(o)
    return jax.tree_util.tree_map(one, old, fresh)


def pad_agents(tree, n_old: int, n_new: int, axis: int = 0, fill: str = "zeros"):
    """Leaves with n_old along `axis` padded to n_new: with zeros, or copies of the first row
    (brains for empty slots: replaced by a parent's at birth)."""
    def one(x):
        x = jnp.asarray(x)
        if x.ndim <= axis or x.shape[axis] != n_old:
            return x
        extra = list(x.shape)
        extra[axis] = n_new - n_old
        if fill == "first":
            row = jnp.take(x, jnp.array([0]), axis=axis)
            pad = jnp.repeat(row, n_new - n_old, axis=axis)
        else:
            pad = jnp.zeros(extra, x.dtype)
        return jnp.concatenate([x, pad], axis=axis)
    return jax.tree_util.tree_map(one, tree)


def grow_state(state, cfg_old, cfg_new, key):
    """The world state with cfg_new.n_agents slots; the new ones unborn."""
    from lifesim import env as E
    n_old, n_new = cfg_old.n_agents, cfg_new.n_agents
    fresh = E.reset(key, cfg_new)
    grown = grow_like(state, fresh, n_old, n_new)
    # who dropped food here is stored as agent + n * below: re-encode for the new n
    d = state.drop_owner
    grown = grown._replace(drop_owner=jnp.where(d >= 0, d % n_old + n_new * (d // n_old), d))
    return grown


def bigger(cfg, factor: int = 2):
    return dataclasses.replace(cfg, n_agents=cfg.n_agents * factor)
