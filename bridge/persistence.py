"""Atomic array-only saves; never unpickles gameplay saves."""
import json
import os
from pathlib import Path
import re
import tempfile

import jax
import jax.numpy as jnp
import numpy as np


def slot_path(directory, slot):
    if not isinstance(slot, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', slot):
        raise ValueError('save slot must contain 1..64 letters, digits, underscores or hyphens')
    directory = Path(directory).resolve()
    target = directory / (slot + '.npz')
    if target.resolve().parent != directory:
        raise ValueError('save path escapes save directory')
    return target


def save(path, tree, identity):
    arrays = [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]
    if any(a.dtype.hasobject for a in arrays):
        raise ValueError('object arrays are not supported')
    metadata = {'format': 1, 'identity': identity,
                'arrays': [{'shape': list(a.shape), 'dtype': str(a.dtype)} for a in arrays]}
    # Bytes retain bfloat16 and other JAX dtypes without relying on NumPy dtype registration in npz.
    payload = {f'a{i}': np.frombuffer(a.tobytes(), dtype=np.uint8) for i, a in enumerate(arrays)}
    payload['metadata'] = np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as f:
            temp_path = f.name
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def load(path, template, identity):
    leaves, tree_def = jax.tree_util.tree_flatten(template)
    templates = [np.asarray(x) for x in leaves]
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data['metadata'].tobytes())
        expected = [{'shape': list(a.shape), 'dtype': str(a.dtype)} for a in templates]
        if metadata.get('format') != 1 or metadata.get('identity') != identity:
            raise ValueError('save uses a different engine, configuration or policy mode')
        if metadata.get('arrays') != expected or set(data.files) != {'metadata', *(f'a{i}' for i in range(len(leaves)))}:
            raise ValueError('save array schema does not match this session')
        restored = []
        for i, a in enumerate(templates):
            raw = data[f'a{i}']
            if raw.dtype != np.uint8 or raw.ndim != 1 or raw.nbytes != a.nbytes:
                raise ValueError('save contains an invalid array payload')
            restored.append(jnp.asarray(np.frombuffer(raw.tobytes(), dtype=a.dtype).reshape(a.shape)))
    return jax.tree_util.tree_unflatten(tree_def, restored)
