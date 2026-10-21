#!/bin/bash
# Archetype intervention: one tyrant or one saint among the founders, 3 seeds each (6 array tasks),
# alpha 0 and normal food; the control is the core experiment's a0.0_f1.0_s0..2.
#   sbatch --export=ALL,WORLDS=128 hprc/archetypes.slurm
# Then: python scripts/experiment.py report --root $SCRATCH/runs/exp_<code version>
#SBATCH --job-name=lifesim-archetypes
#SBATCH --array=0-5
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
python scripts/experiment.py archetypes --root "$ROOT" --worlds ${WORLDS:-64} \
  --agent-steps ${STEPS:-1e8} --playback-every 8 --only $SLURM_ARRAY_TASK_ID
