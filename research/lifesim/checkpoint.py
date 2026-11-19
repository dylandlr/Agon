"""Atomic checkpoint save/load for training state (pytrees of arrays).
Cluster jobs hit wall-time limits; every trainer saves periodically and can
resume from the last checkpoint with --resume."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def save(path: str | Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, obj), f)
    os.replace(tmp, path)                    # atomic: never leaves a half-written file


def load(path: str | Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, obj)
