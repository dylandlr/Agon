#!/bin/bash
# Placed archetypes (36 array tasks: 2 seeds x (no archetype + 17 archetypes)): the validated caveman
# agents live and learn in a continuous world with one founder given an archetype's temperament.
# Needs `validate` task 0 (caveman) finished and the persistence gate passed.
#   sbatch hprc/placed.slurm            # fewer: WHICH=chief,warrior sbatch --array=0-35 hprc/placed.slurm
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>
#SBATCH --job-name=lifesim-placed
#SBATCH --array=0-35
#SBATCH --time=08:00:00
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
python scripts/experiment.py placed --root "$ROOT" --playback-every 8 --which "${WHICH:-all}" \
  --only $SLURM_ARRAY_TASK_ID
