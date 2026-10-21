# Running on HPRC (Grace, A100)

Every job runs from `$SCRATCH/lifesim`, so submit from there (`cd $SCRATCH/lifesim`). Job logs go to
`logs/`. All runs land in `$SCRATCH/runs/exp/`; Colab runs of the same commands are interchangeable.

## 1. Once: code and environment

On your own machine:

    scp lifesim_code.zip <netid>@grace.hprc.tamu.edu:$SCRATCH/       # or use the portal's file upload

On a Grace login node:

    cd $SCRATCH && mkdir -p lifesim && cd lifesim && unzip -o ../lifesim_code.zip
    bash hprc/setup_env.sh

To update the code later, unzip the new zip over the folder. Every run records a code version;
a run started with older code refuses to resume with new code, so versions never mix.

## 2. Benchmark (1 hour limit, usually ~10 minutes)

    sbatch hprc/benchmark.slurm

The last line of `logs/lifesim-bench-<job>.out` reads like
`RECOMMENDED: --worlds 128 (... agent-steps/s ...)`. Use that number as `WORLDS` below.

## 3. Validation (3 tasks)

    sbatch --export=ALL,WORLDS=128 hprc/validate.slurm

Trains three versions, 100M agent-steps each by default (`STEPS=2e8` for more), then lives each
for 10 years in a continuous world: `caveman` (the society preset: given survival skills),
`caveman_no_sleep_learning` (learning does not wait for sleep) and `blank` (no given skills).
Watch progress: `tail -f logs/lifesim-validate-<job>_0.out`. When all three finish:

    python scripts/experiment.py report --root $SCRATCH/runs/exp

Check before the core experiment: thirst no longer the main killer, more sleep at night than by
day, births happening, and the continuous worlds lasting.

## 4. Core experiment (18 tasks)

    sbatch --export=ALL,WORLDS=128 hprc/core_array.slurm

Reward sharing alpha {0, 0.5, 1} x food {1.0, 0.4} x seeds {0, 1, 2}, 100M agent-steps each
(`STEPS=` to change), each followed by a 10-year continuous world. Resubmitting resumes; to rerun
only some tasks: `--array=3,7`. Then `python scripts/experiment.py report --root $SCRATCH/runs/exp`.

## 4b. Optional: archetype intervention (6 tasks)

    sbatch --export=ALL,WORLDS=128 hprc/archetypes.slurm

One tyrant or one saint among the founders (alpha 0, normal food, 3 seeds each). Compare with the
core experiment's `a0.0_f1.0_s*` as the control:

    python scripts/compare_runs.py $SCRATCH/runs/exp/core/a0.0_f1.0_s* $SCRATCH/runs/exp/archetypes/*

The report also groups every recorded life into the roles agents actually fell into, and shows
which role each injected archetype took (`scripts/archetypes.py`).

## 4c. Full playback of the longest-lived world

    python scripts/full_playback.py --root $SCRATCH/runs/exp

Lives the longest-lasting continuous world again with every 5-day chunk recorded, into
`<run>/full_playback/` (open `index.html`). Run it on a GPU node for long worlds
(`srun --gres=gpu:a100:1 --time=02:00:00 --pty bash` first).

## 5. Bring results home

Without checkpoints (large) but with histories, summaries, births and playback pages:

    cd $SCRATCH/runs && zip -r exp_results.zip exp -x '*checkpoint.pkl' '*params.msgpack'

then download `exp_results.zip` with `scp` or the HPRC portal.

## Useful

- Queue: `squeue -u $USER`; cancel: `scancel <job>`; account balance: `myproject -l`.
- If a job runs out of GPU memory, resubmit with a smaller `WORLDS`.
- `hprc/train_phase2.slurm` runs a single training run with any name, for one-off experiments.
