"""Tune the learning rate and the action entropy bonus with Optuna, against survival.

    python scripts/tune.py --out runs/tune --trials 12 --seeds 2 -- --preset society30 --worlds 8 --ticks 64000

Everything after "--" goes to scripts/train_phase2.py unchanged (preset, worlds, ticks, agent-chunk...).
Each trial trains one run per seed with --lr and --ent-coef set from the trial, and scores it by
survival_score_per_world_day (0.01 per agent-tick alive, -1 per death, per world-day: the trainer's
mode-independent survival measure) averaged over the second half of its logged rows, then averaged
over seeds. Every trial uses the same seeds, so trials are compared on the same worlds.

The study lives in <out>/study.journal: rerun the same command to continue it after a disconnect.

Read the result with care. A short run rewards whatever learns fastest early, not what does best at
the full budget; a seed's luck moves the score as much as a setting does (two seeds is the minimum
for that reason). Fix the chosen values before the experiments and keep them fixed: settings chosen
by looking at experiment outcomes are a selection pressure of their own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

TRAIN = str(Path(__file__).resolve().parent / "train_phase2.py")


def score_of(log_lines):
    rows = []
    for line in log_lines:
        line = line.strip()
        if line.startswith("{") and "survival_score_per_world_day" in line:
            try:
                rows.append(json.loads(line)["survival_score_per_world_day"])
            except (ValueError, KeyError):
                pass
    if not rows:
        return None, 0
    tail = rows[len(rows) // 2:]
    return sum(tail) / len(tail), len(rows)


def run_one(cmd, log_path, env):
    lines = []
    with open(log_path, "w") as log, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                      text=True, env=env) as p:
        for line in p.stdout:
            log.write(line)
            lines.append(line)
            if line.startswith("{"):
                row = json.loads(line)
                print(f"    update {row['update']}: survival {row['survival_score_per_world_day']}"
                      f"  ({row.get('agent_sps')} agent-steps/s)", flush=True)
            elif "Error" in line or "Traceback" in line:
                print("    " + line.rstrip(), flush=True)
    return p.returncode, lines


def main():
    argv = sys.argv[1:]
    passthrough = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=2, help="seeds per trial (the same seeds for every trial)")
    ap.add_argument("--lr", type=float, nargs=2, default=(3e-5, 1e-3), help="log-uniform range")
    ap.add_argument("--ent-coef", type=float, nargs=2, default=(1e-3, 2e-2),
                    help="log-uniform range for the action entropy bonus (default 0.005)")
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Lock by creating a file, not a symlink: Google Drive's Colab mount cannot make symlinks
    journal = str(out / "study.journal")
    storage = JournalStorage(JournalFileBackend(journal, lock_obj=JournalFileOpenLock(journal)))
    study = optuna.create_study(study_name="agon", storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0))
    if not study.trials:                       # start from the current defaults
        study.enqueue_trial({"lr": 3e-4, "ent_coef": 0.005})
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)

    def objective(trial):
        lr = trial.suggest_float("lr", *args.lr, log=True)
        ent = trial.suggest_float("ent_coef", *args.ent_coef, log=True)
        print(f"\n=== trial {trial.number}: lr {lr:.2e}, ent_coef {ent:.4f}", flush=True)
        scores = []
        for seed in range(args.seeds):
            run_dir = out / f"trial_{trial.number}" / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, TRAIN, *passthrough, "--lr", repr(lr), "--ent-coef", repr(ent),
                   "--seed", str(seed), "--out", str(run_dir), "--ckpt-every", "1000000"]
            print(f"  seed {seed}", flush=True)
            code, lines = run_one(cmd, run_dir / "train.log", env)
            if code != 0:
                raise RuntimeError(f"trial {trial.number} seed {seed} failed (exit {code}); see {run_dir}/train.log")
            s, n = score_of(lines)
            if s is None:
                raise RuntimeError(f"trial {trial.number} seed {seed} logged no rows; raise --ticks or lower --log-every")
            scores.append(s)
            trial.set_user_attr(f"seed_{seed}", s)
            print(f"  seed {seed}: {s:.4f} (mean of the last {n - n // 2} of {n} rows)", flush=True)
        mean = sum(scores) / len(scores)
        spread = math.sqrt(sum((x - mean) ** 2 for x in scores) / max(len(scores) - 1, 1))
        trial.set_user_attr("seed_sd", spread)
        print(f"  trial {trial.number}: {mean:.4f} (sd over seeds {spread:.4f})", flush=True)
        return mean

    left = args.trials - len([t for t in study.trials if t.state.is_finished()])
    if left > 0:
        study.optimize(objective, n_trials=left, catch=(RuntimeError,))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    done.sort(key=lambda t: -t.value)
    print("\nrank  score    sd      lr        ent_coef")
    for i, t in enumerate(done):
        print(f"{i + 1:>4}  {t.value:.4f}  {t.user_attrs.get('seed_sd', 0):.4f}  {t.params['lr']:.2e}  "
              f"{t.params['ent_coef']:.4f}")
    if done:
        best = done[0]
        json.dump({"best": best.params, "score": best.value, "seed_sd": best.user_attrs.get("seed_sd"),
                   "trials": [{"params": t.params, "score": t.value, **t.user_attrs} for t in done],
                   "train_args": passthrough}, open(out / "best.json", "w"), indent=1)
        print(f"\nbest: lr {best.params['lr']:.2e}, ent_coef {best.params['ent_coef']:.4f} -> {out / 'best.json'}")
        print("A difference smaller than about two seed sds is noise, not a better setting.")


if __name__ == "__main__":
    main()
