#!/bin/bash
# One-time setup on a Grace LOGIN node (compute nodes may lack internet).
# Puts code, environment and runs on $SCRATCH (home quota is small).
set -euo pipefail
module purge
module load GCCcore Python        # if this fails: `module spider Python`, pick a 3.12+ version
python3 --version
# The code is developed and tested on jax 0.11.2, which needs Python >= 3.12. Stop here rather than let pip
# quietly resolve an older jax/flax that was never tested. `module spider Python` lists what Grace offers.
python3 -c 'import sys; assert sys.version_info >= (3, 12), "need Python >= 3.12, got %s" % sys.version.split()[0]'
mkdir -p "$SCRATCH/venvs" "$SCRATCH/runs" "$SCRATCH/lifesim/logs"   # Slurm writes job logs to logs/
python3 -m venv "$SCRATCH/venvs/lifesim"
source "$SCRATCH/venvs/lifesim/bin/activate"
pip install --upgrade pip
pip install -r "$SCRATCH/lifesim/requirements-ci.txt"      # the tested pins: jax, flax, optax, numpy, pytest
pip install "jax[cuda12]==0.11.2"                          # same jax version, plus the CUDA 12 libraries it ships with
pip install -e "$SCRATCH/lifesim" --no-deps                # the code itself; dependencies are already pinned above
python -c "import jax; print('jax', jax.__version__)"   # devices are only visible on a GPU node (see below)
echo "Done. On a GPU node, jax.devices() must list a CUDA device, not CPU (a silent CPU fallback is the failure to look for):"
echo "  srun --partition=gpu --gres=gpu:a100:1 --time=00:10:00 --pty bash -c 'source $SCRATCH/venvs/lifesim/bin/activate; python -c \"import jax; print(jax.devices())\"'"
echo "Then: sbatch hprc/benchmark.slurm"
