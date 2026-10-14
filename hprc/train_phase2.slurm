#!/bin/bash
# One training run of the society preset. Resubmit the same command to continue after the time limit.
#   sbatch --export=ALL,NAME=my_run,WORLDS=128,TICKS=200000 hprc/train_phase2.slurm
#SBATCH --job-name=lifesim-train
#SBATCH --time=24:00:00
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
python scripts/train_phase2.py --preset society --worlds ${WORLDS:-64} --ticks ${TICKS:-100000} \
  --out "$SCRATCH/runs/${NAME:-single}" --resume --log-every 10 --ckpt-every 10
