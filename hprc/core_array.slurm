#!/bin/bash
# Core experiment: reward sharing alpha {0, 0.5, 1} x food {1.0, 0.4} x seeds {0, 1, 2} = 18 array
# tasks of the society preset, one A100 each; each trains (100M agent-steps by default), then lives 10 years in a continuous world.
# Run validate.slurm first and check it. Resubmitting resumes unfinished tasks.
#   sbatch --export=ALL,WORLDS=128 hprc/core_array.slurm
#   sbatch --export=ALL,WORLDS=128 --array=3,7 hprc/core_array.slurm    # just some tasks
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>_<preset>
#SBATCH --job-name=lifesim-core
#SBATCH --array=0-17
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
python scripts/experiment.py core --root "$ROOT" --preset ${PRESET:-society} --worlds ${WORLDS:-64} \
  --agent-steps ${STEPS:-1e8} --playback-every 8 --only $SLURM_ARRAY_TASK_ID
