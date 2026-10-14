#!/bin/bash
# One-time setup on a Grace LOGIN node (compute nodes may lack internet).
# Puts code, environment and runs on $SCRATCH (home quota is small).
set -euo pipefail
module purge
module load GCCcore Python        # if this fails: `module spider Python`, pick a 3.10+ version
python3 --version
mkdir -p "$SCRATCH/venvs" "$SCRATCH/runs" "$SCRATCH/lifesim/logs"   # Slurm writes job logs to logs/
python3 -m venv "$SCRATCH/venvs/lifesim"
source "$SCRATCH/venvs/lifesim/bin/activate"
pip install --upgrade pip
pip install -e "$SCRATCH/lifesim[dev,gpu]"   # jax[cuda12] ships its own CUDA libraries
python -c "import jax; print('jax', jax.__version__)"
echo "Done. Verify on a GPU node with: sbatch hprc/benchmark.slurm"
