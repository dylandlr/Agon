"""Local Eos authority for Unity. NDJSON stdout is reserved for protocol responses.

Run with the research environment. Trained mode requires --run, explicitly
selected at process launch, and its bridge.json with exact env_config, hidden,
encoder and world_model. No checkpoint path is accepted over the protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import persistence

ROOT = Path(__file__).resolve().parents[1]
# Select source before importing any Eos types (historical populations have
# different NamedTuple schemas). This path is a launch option, never a request.
ENGINE_ROOT = ROOT / 'research'
if __name__ == '__main__':
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--engine-root', type=Path, default=ENGINE_ROOT)
    ENGINE_ROOT = bootstrap.parse_known_args()[0].engine_root.resolve()
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(0, str(ENGINE_ROOT / 'scripts'))

import jax
import jax.numpy as jnp
import numpy as np
from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.policy import sample_outputs

VERSION = 1


def source_version():
    """Training fingerprint, canonicalizing only checkout CRLF back to Git LF."""
    h = hashlib.sha256()
    paths = sorted((ENGINE_ROOT / 'lifesim').glob('*.py'))
    paths += [ENGINE_ROOT / 'scripts' / n for n in ('train_phase2.py', 'continuous_world.py')]
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()[:12]


class Session:
    def __init__(self, run=None, save_dir=None):
        self.run = Path(run).resolve() if run else None
        self.state = None
        self.mode = None
        self.net = self.params = self.carry = None
        self.save_dir = Path(save_dir) if save_dir else ROOT / 'node_modules' / '.cache' / 'eos-saves'
        self.policy_identity = None

    def reset(self, request):
        mode = request.get('mode')
        if mode not in ('random', 'trained', 'manual'):
            raise ValueError('mode must explicitly be random, manual or trained')
        seed = request.get('seed', 7)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be a uint32')
        net = params = carry = pop = None
        if mode == 'trained':
            if self.run is None:
                raise ValueError('trained mode requires a local run selected with --run')
            manifest = json.loads((self.run / 'bridge.json').read_text(encoding='utf-8'))
            history = json.loads((self.run / 'history.json').read_text(encoding='utf-8'))
            version = source_version()
            if manifest.get('code_version') != version or history.get('args', {}).get('code_version') != version:
                raise ValueError('run code_version does not match this Eos engine; use its original code before loading weights')
            if 'config' in request:
                raise ValueError('trained mode uses the exact run configuration')
            config = manifest['env_config']
            # Require a complete export rather than guessing missing training dials.
            if set(config) != {f.name for f in dataclasses.fields(EnvConfig)}:
                raise ValueError('bridge.json must contain every EnvConfig field')
            cfg = EnvConfig(**config)
            from export_playback import load_population
            net, params, pop = load_population(self.run, cfg, manifest['hidden'],
                                               manifest['encoder'], manifest['world_model'])
            carry = jnp.zeros((cfg.n_agents, 1, manifest['hidden']))
        else:
            config = request.get('config', {})
            if not isinstance(config, dict):
                raise ValueError('config must be an object')
            # Diagnostic modes deliberately expose only bounded test dimensions.
            if set(config) - {'grid_size', 'n_agents'}:
                raise ValueError('diagnostic config accepts only grid_size and n_agents')
            cfg = EnvConfig(**{'grid_size': 24, 'n_agents': 4, **config})
        if type(cfg.grid_size) is not int or not 12 <= cfg.grid_size <= 128:
            raise ValueError('grid_size must be 12..128')
        if type(cfg.n_agents) is not int or not 1 <= cfg.n_agents <= 64:
            raise ValueError('n_agents must be 1..64')
        key, env_key = jax.random.split(jax.random.PRNGKey(seed))
        state = E.reset(env_key, cfg, None if pop is None else
                        jax.tree_util.tree_map(jnp.asarray, pop.genotype),
                        None if pop is None else jnp.asarray(pop.appearance))
        jax.block_until_ready(state)
        self.cfg, self.key, self.state, self.mode = cfg, key, state, mode
        self.net, self.params, self.carry = net, params, carry
        self.policy_identity = None if mode != 'trained' else {
            'hidden': manifest['hidden'], 'encoder': manifest['encoder'],
            'world_model': manifest['world_model'], 'msg_net': history['args'].get('msg_net', 'branch')}
        self.config_id = hashlib.sha256(json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()).hexdigest()
        self._step = jax.jit(lambda s, a, m: E.step(s, a, cfg, m))
        self._mask = jax.jit(lambda s: E.action_mask(s, cfg))
        if mode == 'trained':
            @jax.jit
            def infer(params, carry, state, key):
                obs = E.observe(state, cfg)
                mask = E.action_mask(state, cfg)
                full = jnp.concatenate([mask, jnp.ones(mask.shape[:-1] + (E.EXTRA_LOGITS,), bool)], -1)
                reset = state.t == 0
                next_carry, logits, _ = jax.vmap(lambda p, c, o, m: net.apply(
                    p, c, (o[None, None], reset[None, None], m[None, None])))(params, carry, obs, full)
                actions, messages = sample_outputs(key, logits[:, 0, 0])
                return next_carry, actions, messages
            self._infer = infer
        return self.snapshot()

    def step(self, request):
        self.require_state()
        count = request.get('count', 1)
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError('count must be 1..16')
        forced = request.get('actions')
        if self.mode == 'manual':
            if count != 1 or not isinstance(forced, list) or len(forced) != self.cfg.n_agents:
                raise ValueError('manual mode requires one step and one action per agent')
            if any(type(a) is not int or not 0 <= a < E.N_ACTIONS for a in forced):
                raise ValueError('action outside action enum')
            mask = np.asarray(self._mask(self.state))
            if not mask[np.arange(self.cfg.n_agents), forced].all():
                raise ValueError('action masked by Eos')
        elif forced is not None:
            raise ValueError('actions can only be supplied in explicit manual mode')
        # Commit the entire bounded request only after device execution succeeds.
        state, key, carry = self.state, self.key, self.carry
        for _ in range(count):
            key, choice_key = jax.random.split(key)
            if self.mode == 'trained':
                carry, actions, messages = self._infer(self.params, carry, state, choice_key)
            else:
                actions = jnp.asarray(forced, dtype=jnp.int32) if forced is not None else \
                    jax.random.categorical(choice_key, jnp.where(self._mask(state), 0.0, -1e9)).astype(jnp.int32)
                messages = jnp.zeros((self.cfg.n_agents, E.OUT_SLOTS), dtype=jnp.int32)
            state, _, _, _ = self._step(state, actions, messages)
        jax.block_until_ready(state)
        self.state, self.key, self.carry = state, key, carry
        return self.snapshot()

    def require_state(self):
        if self.state is None:
            raise ValueError('reset is required before this command')

    def save_identity(self):
        return {'engine': source_version(), 'config': self.config_id, 'mode': self.mode,
                'policy_architecture': self.policy_identity}

    def save(self, request):
        self.require_state()
        path = persistence.slot_path(self.save_dir, request.get('slot'))
        persistence.save(path, (self.state, self.key, self.carry, self.params), self.save_identity())
        return self.snapshot()

    def load(self, request):
        self.require_state()
        path = persistence.slot_path(self.save_dir, request.get('slot'))
        restored = persistence.load(path, (self.state, self.key, self.carry, self.params), self.save_identity())
        jax.block_until_ready(restored)
        self.state, self.key, self.carry, self.params = restored
        return self.snapshot()

    def snapshot(self):
        self.require_state()
        s = jax.device_get(self.state)
        fields = ('terrain', 'height', 'plants', 'meat', 'cooked', 'wood', 'stones', 'fire', 'snow')
        agents = []
        for i in range(self.cfg.n_agents):
            agents.append({'slot': i, 'row': int(s.pos[i, 0]), 'column': int(s.pos[i, 1]),
                           'alive': bool(s.alive[i]), 'existed': bool(s.existed[i]), 'age_ticks': int(s.age[i]),
                           **{k: float(getattr(s, k)[i]) for k in
                              ('satiety', 'hydration', 'energy', 'health', 'body_temp', 'stress', 'injury')},
                           'inventory': np.asarray(s.inv[i]).tolist()})
        return {'tick': int(s.t), 'policy': self.mode, 'config_id': self.config_id,
                'engine_version': source_version(), 'learning': 'fixed-policy inference',
                'tick_minutes': self.cfg.tick_minutes, 'grid_size': self.cfg.grid_size,
                'weather': int(s.weather), 'agents': agents,
                'fields': {k: np.asarray(getattr(s, k)).reshape(-1).tolist() for k in fields},
                'coordinates': 'row-major; Unity x=column, y=grid_size-1-row',
                'terrain_ids': {'grass': E.GRASS, 'forest': E.FOREST, 'rock': E.ROCK, 'water': E.WATER}}

    def handle(self, request):
        if not isinstance(request, dict) or request.get('version') != VERSION:
            raise ValueError('expected object with version=1')
        command = request.get('command')
        if command == 'hello':
            return {'version': VERSION, 'commands': ['hello', 'reset', 'step', 'snapshot', 'save', 'load', 'close'],
                    'actions': E.N_ACTIONS, 'trained_run_selected': self.run is not None}
        if command == 'reset': return self.reset(request)
        if command == 'step': return self.step(request)
        if command == 'snapshot': return self.snapshot()
        if command == 'save': return self.save(request)
        if command == 'load': return self.load(request)
        if command == 'close': return {'closed': True}
        raise ValueError('unknown command')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--engine-root', type=Path, default=ENGINE_ROOT)
    args = parser.parse_args()
    session = Session(args.run, args.save_dir)
    for line in sys.stdin:
        request = None
        try:
            if len(line) > 65536:
                raise ValueError('request exceeds 64 KiB')
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = session.handle(request)
            response = {'version': VERSION, 'id': request.get('id'), 'ok': True, 'result': result}
        except Exception as error:
            response = {'version': VERSION, 'id': request.get('id') if isinstance(request, dict) else None,
                        'ok': False, 'error': {'type': type(error).__name__, 'message': str(error)}}
        print(json.dumps(response, allow_nan=False), flush=True)
        if response['ok'] and request.get('command') == 'close':
            break


if __name__ == '__main__':
    main()
