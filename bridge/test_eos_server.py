import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eos_server import Session, E, source_version
import persistence


def test_stdio_errors_do_not_corrupt_protocol():
    inputs = ['{broken', json.dumps({'version': 99, 'command': 'hello'}),
              json.dumps({'version': 1, 'id': 'hello', 'command': 'hello'}),
              json.dumps({'version': 1, 'command': 'close'})]
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('eos_server.py'))],
                            input='\n'.join(inputs)+'\n', text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert [r['ok'] for r in responses] == [False, False, True, True]
    assert responses[2]['id'] == 'hello'
    assert responses[2]['result']['actions'] == E.N_ACTIONS


def test_missing_trained_run_never_falls_back():
    s = Session()
    with pytest.raises(ValueError, match='--run'):
        s.reset({'mode': 'trained'})
    assert s.state is None


def test_incompatible_run_rejected_before_population_loading(tmp_path):
    (tmp_path / 'bridge.json').write_text(json.dumps({'code_version': 'old-version'}))
    (tmp_path / 'history.json').write_text(json.dumps({'args': {'code_version': 'old-version'}}))
    s = Session(tmp_path)
    with pytest.raises(ValueError, match='original code'):
        s.reset({'mode': 'trained'})
    assert s.state is None
    assert source_version() == '043e8b3ebb99'


def test_manual_step_matches_direct_eos_and_invalid_requests_preserve_state():
    session = Session()
    initial = session.reset({'mode': 'manual', 'seed': 31,
                             'config': {'grid_size': 12, 'n_agents': 2}})
    before = session.state
    mask = np.asarray(E.action_mask(before, session.cfg))
    actions = [int(np.flatnonzero(row)[0]) for row in mask]
    expected, *_ = jax.jit(lambda s, a, m: E.step(s, a, session.cfg, m))(
        before, jnp.asarray(actions, dtype=jnp.int32), jnp.zeros((2, E.OUT_SLOTS), jnp.int32))
    actual = session.step({'actions': actions})
    for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(session.state)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert actual['tick'] == initial['tick'] + 1
    assert actual['policy'] == 'manual'
    state, key = session.state, np.asarray(session.key).copy()
    for bad in ({'count': 17}, {'actions': [E.N_ACTIONS, 0]}, {'actions': [True, 0]}):
        with pytest.raises(ValueError): session.step(bad)
        assert session.state is state
        np.testing.assert_array_equal(session.key, key)
    # Snapshot rows are flattened in Eos order, independent of Unity enum values.
    assert actual['fields']['terrain'] == np.asarray(session.state.terrain).reshape(-1).tolist()
    assert actual['agents'][0]['row'] == int(session.state.pos[0, 0])
    assert actual['agents'][0]['column'] == int(session.state.pos[0, 1])


def assert_tree_equal(a, b):
    a_leaves, a_def = jax.tree_util.tree_flatten(a)
    b_leaves, b_def = jax.tree_util.tree_flatten(b)
    assert a_def == b_def
    for x, y in zip(a_leaves, b_leaves):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_save_load_continuation_matches_uninterrupted_random_world(tmp_path):
    request = {'mode': 'random', 'seed': 21, 'config': {'grid_size': 12, 'n_agents': 2}}
    original = Session(save_dir=tmp_path)
    original.reset(request)
    original.step({'count': 2})
    original.save({'slot': 'continuation'})
    saved = original.snapshot()
    original.step({'count': 3})
    resumed = Session(save_dir=tmp_path)
    resumed.reset({**request, 'seed': 99})
    assert resumed.load({'slot': 'continuation'}) == saved
    resumed.step({'count': 3})
    assert_tree_equal((original.state, original.key), (resumed.state, resumed.key))
    before = resumed.state
    for slot in ('../escape', '', 'bad/name'):
        with pytest.raises(ValueError): resumed.save({'slot': slot})
    (tmp_path / 'broken.npz').write_bytes(b'not a save')
    with pytest.raises(ValueError): resumed.load({'slot': 'broken'})
    assert resumed.state is before
    other = Session(save_dir=tmp_path)
    other.reset({**request, 'mode': 'manual'})
    with pytest.raises(ValueError, match='different engine'):
        other.load({'slot': 'continuation'})


def test_array_save_restores_recurrent_carry_and_parameter_dtypes(tmp_path):
    # Exercises the storage path for policy trees without claiming trained inference.
    tree = (jnp.arange(4, dtype=jnp.uint32), jnp.ones((2, 1, 8)),
            {'weights': jnp.arange(16, dtype=jnp.bfloat16).reshape(4, 4)})
    identity = {'mode': 'test'}
    path = tmp_path / 'arrays.npz'
    persistence.save(path, tree, identity)
    restored = persistence.load(path, jax.tree_util.tree_map(jnp.zeros_like, tree), identity)
    assert_tree_equal(tree, restored)
