#!/bin/bash
# Validation: one array task per design in scripts/experiment.py (--only K). Pass the tasks you want:
#   sbatch --array=0 --export=ALL,PRESET=society30,WORLDS=128 hprc/validate.slurm       # the baseline only (the default)
#   sbatch --array=0,9,10 --export=ALL,PRESET=society30,WORLDS=128 hprc/validate.slurm  # baseline, staged conditions, heritable sharing
#   sbatch --array=0,9,10 --export=ALL,PRESET=society30,WORLDS=128,STEPS=2e8 hprc/validate.slurm   # a bigger budget
# Designs: 0 caveman, 1 entropy_floor, 2 no_retention, 3 shared_trunk, 4 no_world_model, 5 no_sleep_learning,
# 6 blank, 7 entropy_floor_high, 8 floor_no_retention, 9 staged_conditions, 10 heritable_sharing.
# Every design trains on the same default seed and map, so designs are paired (one sample each).
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>_<preset>
#SBATCH --job-name=lifesim-validate
#SBATCH --array=0
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
ROOT="$SCRATCH/runs/exp_$(python -c 'from lifesim.presets import code_version; print(code_version())')_${PRESET:-society}"
python scripts/experiment.py validate --root "$ROOT" --preset ${PRESET:-society} --worlds ${WORLDS:-64} \
  --agent-steps ${STEPS:-1e8} --playback-every 4 --only $SLURM_ARRAY_TASK_ID
