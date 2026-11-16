#!/bin/bash
# Culture (8 array tasks): replicate continuous worlds on one map with different histories, the validated
# caveman agents learning as they live, innate inheritance; 4 with learning by watching (social), 4
# without (solitary). Needs `validate` task 0 (caveman) finished and the persistence gate passed.
#   sbatch hprc/culture.slurm
# Then: python scripts/culture.py $ROOT/culture/social_* --against $ROOT/culture/solitary_*
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>_<preset>
#SBATCH --job-name=lifesim-culture
#SBATCH --array=0-7
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
python scripts/experiment.py culture --root "$ROOT" --preset ${PRESET:-society} --playback-every 8 --replicates 4 \
  --only $SLURM_ARRAY_TASK_ID
