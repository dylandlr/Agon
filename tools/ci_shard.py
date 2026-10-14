"""Split the test files into N shards of similar size (by bytes, a rough proxy for run time),
deterministically, and print shard K's files: `pytest $(python tools/ci_shard.py 4 1)`."""
import sys
from pathlib import Path

n, k = int(sys.argv[1]), int(sys.argv[2])
files = sorted(Path("tests").glob("test_*.py"), key=lambda p: (-p.stat().st_size, p.name))
load, shards = [0] * n, [[] for _ in range(n)]
for f in files:                                   # largest first, onto the lightest shard
    i = load.index(min(load))
    shards[i].append(str(f))
    load[i] += f.stat().st_size
print(" ".join(shards[k - 1]))
