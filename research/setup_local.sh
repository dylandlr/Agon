#!/usr/bin/env bash
# Set up Agon: Eos on macOS or Linux (including WSL2 on Windows).
# Usage:  bash setup_local.sh        then:  source .venv/bin/activate && python scripts/run_local.py --quick
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install -q -U pip
if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU found: installing JAX with CUDA 12"
  python -m pip install -q -U "jax[cuda12]"
else
  echo "No NVIDIA GPU on Linux: installing CPU JAX (Macs use the CPU for now)"
  python -m pip install -q -U jax
fi
python -m pip install -q -U "flax>=0.12.10" "optax>=0.2.8" pytest matplotlib
python -m pip install -q --no-deps -e .
python scripts/run_local.py --steps check
echo
echo "Ready. Next:  source .venv/bin/activate && python scripts/run_local.py --quick"
