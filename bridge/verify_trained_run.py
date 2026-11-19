"""Opt-in parity check against a local, trusted trained Eos run.

Compares bridge snapshots with independent direct Eos inference/stepping, then
checks recurrent-policy continuation through save/load. No training is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def verify(engine, run, steps):
    engine, run = Path(engine).resolve(), Path(run).resolve()
    sys.path[:0] = [str(engine), str(engine / 'scripts')]
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lifesim import env as E
    from lifesim.config import EnvConfig
    from lifesim.policy import sample_outputs
    from export_playback import load_population

    manifest = json.loads((run / 'bridge.json').read_text())
    digest = hashlib.sha256()
    for path in [*sorted((engine / 'lifesim').glob('*.py')),
                 engine / 'scripts/train_phase2.py', engine / 'scripts/continuous_world.py']:
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    version = digest.hexdigest()[:12]
    history = json.loads((run / 'history.json').read_text())
    if manifest['code_version'] != version or history['args']['code_version'] != version:
        raise ValueError('Refusing to load a population from different source')
    cfg = EnvConfig(**manifest['env_config'])
    net, params, pop = load_population(run, cfg, manifest['hidden'], manifest['encoder'], manifest['world_model'])
    key, env_key = jax.random.split(jax.random.PRNGKey(7))
    state = E.reset(env_key, cfg, jax.tree_util.tree_map(jnp.asarray, pop.genotype), jnp.asarray(pop.appearance))
    carry = jnp.zeros((cfg.n_agents, 1, manifest['hidden']))

    @jax.jit
    def direct_tick(state, carry, key):
        key, choice = jax.random.split(key)
        obs, mask = E.observe(state, cfg), E.action_mask(state, cfg)
        full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (E.EXTRA_LOGITS,), bool)], -1)
        reset = state.t == 0
        carry, logits, _ = jax.vmap(lambda p, c, o, m: net.apply(
            p, c, (o[None, None], reset[None, None], m[None, None])))(params, carry, obs, full)
        actions, messages = sample_outputs(choice, logits[:, 0, 0])
        state, _, _, _ = E.step(state, actions, cfg, messages)
        return state, carry, key

    with tempfile.TemporaryDirectory(prefix='agon-trained-') as saves, ThreadPoolExecutor(max_workers=1) as io:
        process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('eos_server.py')),
                                    '--engine-root', str(engine), '--run', str(run), '--save-dir', saves],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        request_id = 0
        def request(command, **kwargs):
            nonlocal request_id
            request_id += 1
            process.stdin.write(json.dumps(dict(version=1, id=request_id, command=command, **kwargs)) + '\n')
            process.stdin.flush()
            response = json.loads(io.submit(process.stdout.readline).result(timeout=180))
            assert response['id'] == request_id and response['version'] == 1
            if not response['ok']:
                raise RuntimeError(response['error'])
            return response['result']
        try:
            snapshot = request('reset', mode='trained', seed=7)
            for tick in range(steps + 1):
                direct = jax.device_get(state)
                assert snapshot['policy'] == 'trained' and snapshot['engine_version'] == version
                assert snapshot['tick'] == int(direct.t)
                for name, field in snapshot['fields'].items():
                    np.testing.assert_array_equal(field, np.asarray(getattr(direct, name)).reshape(-1))
                for agent in snapshot['agents']:
                    i = agent['slot']
                    assert (agent['row'], agent['column']) == tuple(direct.pos[i])
                    assert agent['alive'] == bool(direct.alive[i])
                    assert agent['existed'] == bool(direct.existed[i])
                    assert agent['age_ticks'] == int(direct.age[i])
                    for name in ('satiety', 'hydration', 'energy', 'health', 'body_temp', 'stress', 'injury'):
                        assert agent[name] == float(getattr(direct, name)[i])
                    np.testing.assert_array_equal(agent['inventory'], direct.inv[i])
                print(json.dumps({'direct_parity_tick': tick}), flush=True)
                if tick < steps:
                    state, carry, key = direct_tick(state, carry, key)
                    snapshot = request('step', count=1)
            request('save', slot='parity')
            expected = request('step', count=2)
            request('load', slot='parity')
            assert request('step', count=2) == expected, 'Trained continuation differs'
            return {'passed': True, 'engine': version, 'direct_ticks': steps,
                    'trained_continuation': True,
                    'weights_sha256': hashlib.sha256((run / 'params.msgpack').read_bytes()).hexdigest()}
        finally:
            process.kill()
            process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=3)
    options = parser.parse_args()
    if not 1 <= options.steps <= 100:
        parser.error('--steps must be 1..100')
    print(json.dumps(verify(options.engine_root, options.run, options.steps)))
