# Experiments

Each folder here is one research question: its design, its code, its decisions and its results,
in one place. The rest of the repo is the shared toolkit these experiments build on.

## How this fits the repo

| Folder | What it is | Changed by experiments? |
| --- | --- | --- |
| `lifesim/` | The simulator and learners (world, policy, PPO). Shared by everyone. | Rarely, and on purpose: any change alters `code_version`, so runs made before it cannot be resumed or mixed with runs after |
| `scripts/` | The project's general tools: training (`train_phase2.py`), evaluation, playback, analyses, the thesis pipeline (`experiment.py`) | No; experiments import from them |
| `tests/` | Unit tests for every mechanic | Only alongside a `lifesim/` change |
| `hprc/`, `colab/` | Cluster and Colab job scripts for the full-scale pipeline | No |
| **`experiments/NN_name/`** | **One question: README, its own config and training script, results** | Owned by the experiment |

An experiment configures the world and learner in its own script (calling `EnvConfig`,
`MAConfig` and `make_train` directly) instead of editing shared code. This keeps `code_version`
stable and makes the experiment readable on its own. If an experiment needs a new mechanic or a
fix in `lifesim/`, that is a separate change with tests, and the README records the new code
version.

## Layout of an experiment

```
experiments/NN_short_name/
  README.md          question, setup, decisions log, references, success criteria, results
  train.py           the configuration (one function each for world and learner) and the run
  results/           small outputs, committed: baselines.json, <run>/history.json, figures
  runs/              large outputs, git-ignored: checkpoints, raw logs, recordings
```

Start a new one by copying `TEMPLATE.md` to `NN_name/README.md`. Number experiments in the order
they are started. A later experiment that builds on an earlier one imports its config functions
and changes only what it is testing, so the difference between the two is visible in code.

## Conventions

- **Results:** a run is `--run-name sK` for seed K. Report several seeds before drawing conclusions.
- **Decisions:** every choice that could change the result goes in the README's decisions log,
  with why and who decided, at the time it is made.
- **Commits:** commit the README, the script and `results/`. Never commit `runs/`; it is ignored
  repo-wide.
- **Code version:** record `lifesim.presets.code_version()` in the README when a run starts. The
  training scripts refuse to resume across a version change.
- **Status:** keep the index below and each README's status line current.

## Index

| # | Experiment | Question | Status |
| --- | --- | --- | --- |
| 01 | [Minimal baseline](01_minimal_baseline/README.md) | Does independent recurrent PPO learn to survive with an own-survival reward, and does cooperation appear? | Running (`s0`) |
| 02 | Reproduction | Can bands with births reach a steady-state, collaborative society? | Planned |
