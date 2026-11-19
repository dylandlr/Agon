"""Map training fingerprints to local Git history without checkout/newline changes."""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import subprocess


def scan(versions, refs):
    git = lambda *args: subprocess.check_output(['git', *args])
    commits = git('rev-list', *refs).decode().split()
    cache, matches = {}, {}
    for commit in commits:
        entries = {}
        for line in git('ls-tree', '-r', commit, 'lifesim/', 'scripts/train_phase2.py',
                        'scripts/continuous_world.py').decode().splitlines():
            meta, path = line.split('\t')
            entries[path] = meta.split()[2]
        paths = sorted(p for p in entries if p.startswith('lifesim/') and p.count('/') == 1 and p.endswith('.py'))
        if not paths:
            continue
        paths += [p for p in ['scripts/train_phase2.py', 'scripts/continuous_world.py'] if p in entries]
        h = hashlib.sha256()
        for path in paths:
            sha = entries[path]
            if sha not in cache:
                cache[sha] = git('cat-file', 'blob', sha)
            h.update(PurePosixPath(path).name.encode())
            h.update(cache[sha])
        version = h.hexdigest()[:12]
        if version in versions:
            matches.setdefault(version, []).append(commit)
    return {'commits_checked': len(commits), 'matches': matches,
            'unmatched': sorted(set(versions) - set(matches))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('versions', nargs='+')
    args = parser.parse_args()
    print(json.dumps(scan(args.versions, ['--all']), indent=2))
