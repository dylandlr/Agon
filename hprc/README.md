# Running on HPRC (Grace, A100)

Every job runs from `$SCRATCH/lifesim`, so submit from there (`cd $SCRATCH/lifesim`). Job logs go to
`logs/`. Runs land in `$SCRATCH/runs/exp_<code version>_<preset>/` (`$ROOT` below); Colab runs of the same
commands are interchangeable.

## 1. Once: code and environment

On your own machine:

    git archive --format=zip -o lifesim_code.zip HEAD                 # the code of the commit you are on
    scp lifesim_code.zip <netid>@grace.hprc.tamu.edu:$SCRATCH/       # or use the portal's file upload

On a Grace login node:

    cd $SCRATCH && mkdir -p lifesim && cd lifesim && unzip -o ../lifesim_code.zip
    bash hprc/setup_env.sh

To update the code later, unzip the new zip over the folder (unzip -o overwrites; files deleted in
the repository stay behind, which is harmless). Every run records a code version;
a run started with older code refuses to resume with new code, so versions never mix.

## 2. Pick the preset

Every job takes `PRESET` (default `society`), and each preset's results go to their own folder,
`$SCRATCH/runs/exp_<code version>_<preset>`, so presets never mix.

- `society30`: 32 slots, 8 founders, a 32 x 32 map, half-hour ticks, with reproduction, the
  world model and the caveman slate. Recommended for the first emergence runs.
  - 100M agent-steps is about 50 game years per training world, and a 400-day continuous world is
    19,200 ticks per world.
  - The scripted band (the upper bound on what agents could do) kept its numbers up in 2 of 3
    4-year worlds, with 17-32 births to mothers born in the world.
- `society`: the same people on a 250 x 250 map at 1-minute ticks. Thirty times the compute per
  game day: 100M agent-steps is about 2 game years per training world, and a 400-day continuous
  world is 576,000 ticks.

## 3. Benchmark (1 hour limit, usually ~10 minutes)

    sbatch --export=ALL,PRESET=society30 hprc/benchmark.slurm

The last line of `logs/lifesim-bench-<job>.out` reads like
`RECOMMENDED: --worlds 128 (... agent-steps/s ...)`. Use that number as `WORLDS` below. Benchmark
each preset you will use: memory per world differs.

## 4. Validation (4 tasks), then the gate

    sbatch --export=ALL,PRESET=society30,WORLDS=128 hprc/validate.slurm

This trains four versions, 100M agent-steps each by default (`STEPS=2e8` for more), then lives
each for 400 days in a continuous world:
- `caveman`: the preset;
- `caveman_no_sleep_learning`: learning does not wait for sleep;
- `blank`: no given skills;
- `caveman_no_world_model`.

Watch progress with `tail -f logs/lifesim-validate-<job>_0.out`. When all four finish:

    ROOT=$SCRATCH/runs/exp_$(python -c 'from lifesim.presets import code_version; print(code_version())')_society30
    python scripts/experiment.py gate --root $ROOT --preset society30
    python scripts/experiment.py report --root $ROOT

The gate passes when people born in the caveman world grow up and the world outlasts its
founders. The core experiment, archetypes, inheritance and stress refuse to start until it does.
The report has:
- training curves by condition;
- every continuous world (population, births, generations, sleep, deaths);
- society in each world (scripts/society.py: groups, company sought beyond chance, bands, gifts
  and reciprocity, joint hunts, division of labour, punishment; every year in `society.json`);
- hierarchy;
- the roles agents fell into.

## 5. Core experiment (18 tasks)

    sbatch --export=ALL,PRESET=society30,WORLDS=128 hprc/core_array.slurm

Reward sharing alpha {0, 0.5, 1} x food {1.0, 0.4} x seeds {0, 1, 2}, 100M agent-steps each
(`STEPS=` to change), each followed by a 400-day continuous world. Resubmitting resumes; to rerun
only some tasks: `--array=3,7`. Then `python scripts/experiment.py report --root $ROOT`.

## 5b. Optional: archetype intervention (6 tasks)

    sbatch --export=ALL,PRESET=society30,WORLDS=128 hprc/archetypes.slurm

One tyrant or one saint among the founders (alpha 0, normal food, 3 seeds each). Compare with the
core experiment's `a0.0_f1.0_s*` as the control:

    python scripts/compare_runs.py $ROOT/core/a0.0_f1.0_s* $ROOT/archetypes/*

The report also groups every recorded life into the roles agents actually fell into, and shows
which role each injected archetype took (`scripts/archetypes.py`).

## 5c. Full playback of the longest-lived world

    python scripts/full_playback.py --root $ROOT

Lives the longest-lasting continuous world again with every 5-day chunk recorded, into
`<run>/full_playback/` (open `index.html`). Run it on a GPU node for long worlds
(`srun --gres=gpu:a100:1 --time=02:00:00 --pty bash` first).

## 6. Bring results home

Without checkpoints (large) but with histories, summaries, births and playback pages:

    cd $SCRATCH/runs && zip -r exp_results.zip $(basename $ROOT) -x '*checkpoint.pkl' '*params.msgpack'

then download `exp_results.zip` with `scp` or the HPRC portal.

## Useful

- Queue: `squeue -u $USER`; cancel: `scancel <job>`; account balance: `myproject -l`.
- If a job runs out of GPU memory, resubmit with a smaller `WORLDS`.
- `hprc/train_phase2.slurm` runs a single training run with any name, for one-off experiments.
