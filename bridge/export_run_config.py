"""Export exact training configuration using a trusted, matching source checkout.

Runs only the configuration-building portion of that checkout's train_phase2
main, with saved arguments. It never starts training or opens a pickle.
"""
import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def export(engine, run):
    engine, run = Path(engine).resolve(), Path(run).resolve()
    source = engine / 'scripts' / 'train_phase2.py'
    digest = hashlib.sha256()
    for path in [*sorted((engine / 'lifesim').glob('*.py')), source,
                 engine / 'scripts' / 'continuous_world.py']:
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    version = digest.hexdigest()[:12]
    args = json.loads((run / 'history.json').read_text())['args']
    if args.get('code_version') != version:
        raise ValueError('History and selected engine fingerprints differ')
    sys.path[:0] = [str(engine), str(engine / 'scripts')]
    spec = importlib.util.spec_from_file_location('source_train_phase2', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, n in enumerate(main.body) if assigned(n, 'base'))
    end = next(i for i, n in enumerate(main.body) if assigned(n, 'out'))
    block = main.body[start:end]
    if not any(assigned(n, 'env_cfg') for n in block) or not any(assigned(n, 'cfg') for n in block):
        raise ValueError('Unrecognized training configuration layout')
    namespace = {**vars(module), 'args': argparse.Namespace(**args)}
    exec(compile(ast.Module(body=block, type_ignores=[]), str(source), 'exec'), namespace)
    cfg = namespace['cfg']
    manifest = {'code_version': version, 'env_config': dataclasses.asdict(namespace['env_cfg']),
                'hidden': cfg.hidden, 'encoder': cfg.encoder, 'world_model': cfg.world_model}
    # Exclusive creation preserves any previously reviewed configuration.
    with (run / 'bridge.json').open('x', encoding='utf-8') as out:
        json.dump(manifest, out, indent=2)
    return {'code_version': version, 'fields': len(manifest['env_config']), 'hidden': cfg.hidden}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(export(options.engine_root, options.run)))
