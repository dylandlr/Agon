#!/bin/bash
# Finds the fastest number of parallel worlds that fits on one A100 for the "society" preset.
# Read the RECOMMENDED line in logs/lifesim-bench-<job>.out and pass it as WORLDS below.
#SBATCH --job-name=lifesim-bench
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
##SBATCH --account=YOUR_PROJECT_ACCOUNT    # only needed if not your default (myproject -l)
module purge
module load GCCcore Python
source "$SCRATCH/venvs/lifesim/bin/activate"
cd "$SCRATCH/lifesim"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p logs
nvidia-smi
python scripts/benchmark.py --preset ${PRESET:-society} --worlds 32 64 128 256 --target-agent-steps 1e8
