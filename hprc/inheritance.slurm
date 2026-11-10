#!/bin/bash
# Inheritance (6 array tasks): the validated caveman run's agents live on in a continuous world where
# every individual keeps learning; newborns start from a parent's learned brain (parent) or the brain
# the parent was born with (innate), 3 seeds each. Needs `validate` task 0 (caveman) finished.
#   sbatch hprc/inheritance.slurm
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>_<preset>
#SBATCH --job-name=lifesim-inheritance
#SBATCH --array=0-5
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
ROOT="$SCRATCH/runs/exp_$(python -c 'from lifesim.presets import code_version; print(code_version())')_${PRESET:-society}"
python scripts/experiment.py inheritance --root "$ROOT" --preset ${PRESET:-society} --playback-every 8 \
  --only $SLURM_ARRAY_TASK_ID
