#!/bin/bash
# Validation (4 array tasks): the society preset (caveman slate, world model), the same without
# sleep-consolidated learning, the blank slate, and the society preset without the world model;
# each trained, then lived for 10 years in a continuous world.
#   sbatch --export=ALL,WORLDS=128 hprc/validate.slurm          # WORLDS from the benchmark
#   sbatch --export=ALL,WORLDS=128,STEPS=2e8 hprc/validate.slurm  # a bigger budget
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>
#SBATCH --job-name=lifesim-validate
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
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
# One results folder per code version (experiment.py refuses to reuse runs made by other code)
ROOT="$SCRATCH/runs/exp_$(python -c 'from lifesim.presets import code_version; print(code_version())')"
python scripts/experiment.py validate --root "$ROOT" --worlds ${WORLDS:-64} \
  --agent-steps ${STEPS:-1e8} --playback-every 4 --only $SLURM_ARRAY_TASK_ID
