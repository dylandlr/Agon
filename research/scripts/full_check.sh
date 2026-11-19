#!/bin/bash
# End-to-end check: every pipeline, every feature on, small sizes. Stops at the first failure.
set -euo pipefail
export PYTHONPATH=.
OUT=${1:-/tmp/full_check}
rm -rf "$OUT" && mkdir -p "$OUT"
step() { echo; echo "=== $1"; }

step "Phase 1: train one agent, evaluate against random and scripted, record an episode"
python scripts/train_phase1.py --timesteps 16384 --num-envs 8 --grid 16 --max-steps 300 --log-every 4 --encoder mlp --out "$OUT/p1"
python scripts/evaluate.py --params "$OUT/p1/params.msgpack" --encoder mlp --grid 16 --max-steps 300 --episodes 8 --record-to "$OUT/p1/rec"

step "Phase 2: alpha 0.5, scarce food, social learning, messages, fire, morals, names"
python scripts/train_phase2.py --agents 4 --worlds 4 --grid 24 --steps 32 --ticks 128 --encoder mlp \
  --alpha 0.5 --scarcity 0.6 --imitation 0.5 --log-every 2 --ckpt-every 2 --out "$OUT/p2"

step "Phase 2: resume from checkpoint and continue"
python scripts/train_phase2.py --agents 4 --worlds 4 --grid 24 --steps 32 --ticks 192 --encoder mlp \
  --alpha 0.5 --scarcity 0.6 --imitation 0.5 --log-every 2 --ckpt-every 2 --out "$OUT/p2" --resume

step "Playback with names and language analysis"
python scripts/export_playback.py --run "$OUT/p2" --days 1 --out "$OUT/p2/playback.html"

step "Compare runs (core experiment table)"
python scripts/compare_runs.py "$OUT/p2"

step "Fire chance baseline"
python scripts/fire_baseline.py --worlds 2 --agents 4 --grid 16 --ticks 300 --immortal

step "Throughput benchmark"
python scripts/benchmark.py --agents 4 --worlds 4 --grid 24 --steps 32 --encoder mlp --updates 2 --target-agent-steps 1000000

step "System validation: emergent behaviors are possible, pay off, and are not rewarded"
python scripts/validate_system.py

echo; echo "ALL PIPELINES PASSED"
