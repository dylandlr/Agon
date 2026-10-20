"""Run Agon: Eos on your own computer: the same steps as the Colab notebook.

    python scripts/run_local.py                      # everything: check, tests, validate, benchmark, train, plot, playback
    python scripts/run_local.py --quick              # tiny settings: confirms the whole pipeline works in ~10 minutes
    python scripts/run_local.py --steps train plot playback --ticks 12800 --run-name longer
    python scripts/run_local.py --steps playback --run-name baseline_alpha0_s0

Training resumes automatically: rerun the same command with the same --run-name.
Outputs go to runs/<run-name>/ : checkpoint, history, curves.png, playback.html.
"""
from __future__ import annotations

import os

# JAX reserves 75% of GPU memory in the first process that touches the GPU. This script
# touches it (device check) and then launches training as subprocesses, which would be left
# with only a quarter of the memory. Allocate on demand instead; subprocesses inherit this.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["check", "test", "validate", "benchmark", "train", "baselines", "plot", "playback"]


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def run(cmd, **kw):
    """Run a command from the repo root with this Python; fail loudly."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env, **kw)
    if r.returncode != 0:
        sys.exit(f"step failed (exit code {r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


# ------------------------------------------------------------------ steps
def step_check(args):
    banner("1. Environment check")
    import importlib.metadata as md
    versions = {p: md.version(p) for p in ("jax", "flax", "optax")}
    print("versions:", versions)
    ver = lambda v: tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    if ver(versions["flax"]) < (0, 12, 10):
        sys.exit("Flax is too old for current JAX. Run: pip install -U \"flax>=0.12.10\" \"optax>=0.2.8\"")
    import jax
    backend = jax.default_backend()
    print("JAX backend:", backend, "devices:", jax.devices())
    if backend == "cpu":
        print("No GPU in use. Training will work but slowly; the script uses the faster-on-CPU 'mlp' encoder.\n"
              "  NVIDIA on Linux:   pip install -U \"jax[cuda12]\"\n"
              "  NVIDIA on Windows: use WSL2 (native Windows JAX has no GPU support)\n"
              "  Mac:               CPU only for now")
    sys.path.insert(0, str(ROOT))
    import jax.numpy as jnp
    from lifesim import env as E
    from lifesim.config import EnvConfig
    from lifesim.policy import make_network
    cfg = EnvConfig(grid_size=32, n_agents=2)
    net = make_network(cfg, 64, args.encoder, messages=True)
    x = (jnp.zeros((1, 2, E.obs_dim(cfg))), jnp.zeros((1, 2), bool),
         jnp.ones((1, 2, E.N_ACTIONS + E.EXTRA_LOGITS), bool))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((2, 64)), x)
    _, logits, _ = net.apply(params, jnp.zeros((2, 64)), x)
    print(f"network OK: logits {tuple(logits.shape)}, encoder {args.encoder}, convolution path {net.conv_impl}")


def step_test(args):
    banner("2. Unit tests" + ("" if args.full_tests else " (fast subset; --full-tests for all)"))
    targets = ["tests"] if args.full_tests else ["tests/test_features.py", "tests/test_social.py"]
    run([sys.executable, "-m", "pytest", "-q", *targets])


def step_validate(args):
    banner("3. System validation: emergent behaviors are possible, pay off, and are not rewarded")
    run([sys.executable, "scripts/validate_system.py"])


def step_benchmark(args):
    banner("4. Throughput benchmark")
    for worlds in args.benchmark_worlds:
        print(f"--- {worlds} worlds", flush=True)
        run([sys.executable, "scripts/benchmark.py", "--agents", args.agents, "--worlds", worlds, "--grid", args.grid,
             "--encoder", args.encoder, "--updates", 2 if args.quick else 5,
             "--target-agent-steps", args.ticks * args.worlds * args.agents])


def step_train(args):
    banner(f"5. Training: {args.run_name}")
    out = args.runs_dir / args.run_name
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "scripts/train_phase2.py", "--agents", args.agents, "--worlds", args.worlds,
           "--grid", args.grid, "--ticks", args.ticks, "--steps", args.rollout, "--alpha", args.alpha, "--reward-mode", args.reward_mode,
           "--scarcity", args.scarcity, "--encoder", args.encoder, "--seed", args.seed,
           "--log-every", args.log_every, "--ckpt-every", args.log_every, "--out", out, "--resume"]
    print("$", " ".join(str(c) for c in cmd), flush=True)
    print(f"total: {args.ticks * args.worlds * args.agents:,} agent-steps. The first update compiles (a minute or two).")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    t0 = time.time()
    with open(f"{out}.log", "a") as log:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if line.startswith("{"):
                r = json.loads(line)
                p = r["per_1000_agent_ticks"]
                print(f"[{(time.time() - t0) / 60:5.1f} min] steps {r['agent_steps']:>10,}  "
                      f"lifespan {r['mean_age_at_death']}  deaths {sum(r['deaths'].values()):>4}  "
                      f"company {r.get('in_company_share')}  strikes/k {p['strikes_on_agents']}  "
                      f"gifts/k {p['gifts']}  speech/k {p['messages']}  large prey {r['kills']['large_prey']}  "
                      f"{r['agent_sps']:.0f}/s", flush=True)
            else:
                print(line, end="", flush=True)
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"training failed (exit code {proc.returncode}); see {out}.log")


def step_baselines(args):
    banner("6. Random and scripted baselines in the same setting")
    run([sys.executable, "scripts/eval_baselines.py", "--agents", args.agents, "--grid", args.grid,
         "--scarcity", args.scarcity, "--worlds", 4 if args.quick else 32,
         *(["--max-steps", 300] if args.quick else []), "--out-dir", args.runs_dir / "baselines"])


def baseline_file(args):
    ticks = 300 if args.quick else 1920
    f = args.runs_dir / "baselines" / f"a{args.agents}_g{args.grid}_f{args.scarcity:g}_t{ticks}.json"
    return json.load(open(f))["baselines"] if f.exists() else {}


def step_plot(args):
    banner("7. Learning curves")
    hist_path = args.runs_dir / args.run_name / "history.json"
    if not hist_path.exists():
        sys.exit(f"no history at {hist_path}: train first")
    h = json.load(open(hist_path))["history"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (pip install matplotlib); printing the table instead")
        for r in h:
            print(r["agent_steps"], r["mean_age_at_death"], r["deaths"])
        return
    panels = [
        ("Mean age at death (ticks)", lambda r: r["mean_age_at_death"]),
        ("Share alive at episode end", lambda r: r.get("share_alive_at_episode_end")),
        ("Global reward per world-day", lambda r: r.get("global_reward_per_world_day")),
        ("Time with someone in view", lambda r: r.get("in_company_share")),
        ("Strikes on agents per 1,000", lambda r: r["per_1000_agent_ticks"]["strikes_on_agents"]),
        ("Gifts per 1,000", lambda r: r["per_1000_agent_ticks"]["gifts"]),
        ("Speech per 1,000", lambda r: r["per_1000_agent_ticks"]["messages"]),
        ("Large prey killed per window", lambda r: r["kills"]["large_prey"]),
        ("Predator attacks deterred per 1,000", lambda r: r.get("deterred_per_1000")),
        ("Meat eaten per 1,000", lambda r: r.get("meat_eaten_per_1000")),
    ]
    xs = [r["agent_steps"] for r in h]
    base = baseline_file(args)
    base_key = {"Mean age at death (ticks)": "mean_age_at_death",
                "Share alive at episode end": "share_alive_at_episode_end",
                "Global reward per world-day": "global_reward_per_world_day"}
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, (title, f) in zip(axes.flat, panels):
        pts = [(x, f(r)) for x, r in zip(xs, h) if f(r) is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label="trained")
        for pol, style in (("random", ":"), ("scripted", "--")):
            k = base_key.get(title)
            if k and pol in base and base[pol].get(k) is not None:
                ax.axhline(base[pol][k], ls=style, color="gray", label=f"{pol} baseline")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("agent-steps")
    axes.flat[0].legend(fontsize=8)
    causes = sorted({c for r in h for c in r["deaths"]})
    axes.flat[-2].axis("off")
    ax = axes.flat[-1]
    ax.stackplot(xs, *[[r["deaths"].get(c, 0) for r in h] for c in causes], labels=causes)
    ax.set_title("Deaths by cause per window", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    png = args.runs_dir / args.run_name / "curves.png"
    plt.savefig(png, dpi=110)
    print("saved", png)
    with_deaths = [r for r in h if r["mean_age_at_death"] is not None] or h
    first, last = with_deaths[0], with_deaths[-1]
    print(f"lifespan {first['mean_age_at_death']} -> {last['mean_age_at_death']} ticks; "
          f"deaths per window {sum(first['deaths'].values())} -> {sum(last['deaths'].values())}")
    if not args.no_browser:
        webbrowser.open(png.as_uri())


def step_playback(args):
    banner("8. Playback and language analysis")
    run_dir = args.runs_dir / args.run_name
    if not (run_dir / "history.json").exists():
        sys.exit(f"no trained run at {run_dir}: train first")
    page = run_dir / "playback.html"
    run([sys.executable, "scripts/export_playback.py", "--run", run_dir, "--encoder", args.encoder,
         "--days", args.days, "--skip-days", args.skip_days, "--out", page])
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", page.read_text(), re.S).group(1))
    L = data["language"]
    print(f"messages: {L['utterances']} ({L['speech_rate']:.0%} of living agent-ticks), {L['distinct_messages']} distinct")
    print("information about context, bits above chance:", L["context_bits"])
    print("naming index, bits above chance:", L["naming"]["index_bits"])
    print("names the agents invented:", [(g["agent"], g["name"]) for g in L["names_given"]] or "none yet")
    print("open in a browser:", page)
    if not args.no_browser:
        webbrowser.open(page.resolve().as_uri())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS + ["all"], default=["all"])
    ap.add_argument("--quick", action="store_true", help="tiny settings to confirm everything works")
    ap.add_argument("--full-tests", action="store_true", help="run all tests (slower)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=None, help="parallel worlds (default: 64 on GPU, 16 on CPU)")
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=None, help="world-ticks per replica")
    ap.add_argument("--rollout", type=int, default=128, help="ticks per PPO update")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--reward-mode", choices=["acts", "survival"], default="survival")
    ap.add_argument("--scarcity", type=float, default=1.0)
    ap.add_argument("--encoder", choices=["conv", "mlp"], default=None, help="default: conv on GPU, mlp on CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--benchmark-worlds", type=int, nargs="+", default=None)
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--skip-days", type=float, default=0.0)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    import jax
    gpu = jax.default_backend() != "cpu"
    args.encoder = args.encoder or ("conv" if gpu else "mlp")
    if args.quick:
        args.agents, args.grid, args.rollout, args.log_every = 4, 24, 32, 2
        args.worlds = args.worlds or 4
        args.ticks = args.ticks or 256
        args.benchmark_worlds = args.benchmark_worlds or [4]
        args.run_name = args.run_name or "quick_check"
    args.worlds = args.worlds or (32 if gpu else 16)
    args.ticks = args.ticks or (12800 if gpu else 2560)
    args.benchmark_worlds = args.benchmark_worlds or ([16, 32, 64] if gpu else [16])
    args.run_name = args.run_name or f"baseline_alpha{args.alpha:g}_food{args.scarcity:g}_s{args.seed}"
    steps = STEPS if "all" in args.steps else [s for s in STEPS if s in args.steps]
    print(f"device: {'GPU' if gpu else 'CPU'} | steps: {', '.join(steps)} | run: {args.run_name} | "
          f"{args.agents} agents x {args.worlds} worlds, {args.grid}x{args.grid}, {args.ticks} ticks, encoder {args.encoder}")
    for s in steps:
        globals()[f"step_{s}"](args)
    banner("Done")


if __name__ == "__main__":
    main()
