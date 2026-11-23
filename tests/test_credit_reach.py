"""How far back GAE actually carries a death, and that `--gae-lambda` is the knob that sets it.

The measured failure is a near-constant critic: V moves 0.2% of the return range from comfortable
to lethal cold, so the advantage carries no state information and the policy can only learn a
state-independent habit. The mechanism is the reach of the credit path. Credit decays as
`(gamma*lambda)^k`, and at the shipped `gae_lambda = 0.95` with `gamma = 0.997` that is a half-life
of 12.8 ticks -- inside a rollout window of `--steps` (128) plus the carried memory tail
(`memory_span`, a game day of 48), so 176 ticks. Thirst kills roughly 230 ticks after the last
drink. So lambda, not the window, throws the signal away, and it throws away 93% of the window the
run already pays for.

These tests pin that claim on `gae` itself rather than on arithmetic about it, and pin the flag that
changes it. They do NOT assert that raising lambda fixes the run -- that is what
`value.explained_variance` on a live run decides.
"""
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lifesim.ppo_multi import MAConfig, gae

GAMMA, WINDOW = 0.997, 128 + 48          # --steps plus the carried memory tail


def _death_at_the_end_of_the_window(lam, T=WINDOW, gamma=GAMMA):
    """One agent that earns nothing and then dies on the last tick of the window.

    Value is zero everywhere, so the only signal in the window is the terminal penalty. The
    advantage at t=0 is then exactly how much of that death the start of the window can see.
    """
    shape = (T, 1, 1)
    reward = jnp.zeros(shape).at[T - 1].set(-1.0)
    done = jnp.zeros(shape).at[T - 1].set(1.0)
    adv = gae(reward, jnp.zeros(shape), done, jnp.zeros(shape), jnp.zeros((1, 1)),
              jnp.full((1, 1), gamma), lam)
    return float(np.asarray(adv)[0, 0, 0])


def test_at_the_shipped_lambda_a_death_is_invisible_from_the_start_of_the_window():
    """0.95 leaves 7e-5 of a 1.0 penalty at t=0: the window is 176 ticks, the reach is 12.8."""
    seen = _death_at_the_end_of_the_window(0.95)
    assert abs(seen) < 1e-3, f"expected the death to vanish, {seen} survived"


def test_lambda_one_carries_the_death_across_the_whole_window():
    """At 1.0 only gamma discounts, so t=0 still sees 0.997**175 = 0.59 of the penalty."""
    seen = _death_at_the_end_of_the_window(1.0)
    assert seen < -0.5, f"expected most of the death to reach t=0, got {seen}"
    assert abs(seen - -(GAMMA ** (WINDOW - 1))) < 1e-4       # exactly -(gamma*lambda)^(T-1)


def test_the_reach_rises_monotonically_with_lambda():
    seen = [abs(_death_at_the_end_of_the_window(L)) for L in (0.95, 0.99, 0.995, 1.0)]
    assert seen == sorted(seen), seen
    assert seen[0] < 1e-3 < seen[-1]


def test_lambda_binds_before_the_window_does_at_every_value_below_one():
    """Which of the two truncations is the binding one, as a half-life against the window."""
    for lam, expect in ((0.95, "lambda"), (0.99, "lambda"), (0.995, "lambda"), (1.0, "window")):
        gl = GAMMA * lam
        half_life = float("inf") if gl >= 1.0 else np.log(0.5) / np.log(gl)
        assert ("window" if half_life >= WINDOW else "lambda") == expect, (lam, half_life)


def test_the_trainer_exposes_gae_lambda_and_still_defaults_to_the_shipped_value():
    """The flag must change nothing until it is passed: 0.95 is what every run so far used."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_phase2.py").read_text()
    flag = re.search(r'add_argument\("--gae-lambda",\s*type=float,\s*default=([0-9.]+)', src)
    assert flag, "train_phase2.py does not expose --gae-lambda"
    assert float(flag.group(1)) == MAConfig().gae_lambda == 0.95
    assert re.search(r"gae_lambda=args\.gae_lambda", src), "the flag is parsed but never reaches MAConfig"


def test_experiment_py_forwards_gae_lambda_to_the_trainer():
    """The Colab and HPRC paths both go through experiment.py, so a flag it does not forward is
    unreachable from the only places runs actually happen."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "experiment.py").read_text()
    flag = re.search(r'add_argument\("--gae-lambda",\s*type=float,\s*default=([0-9.]+)', src)
    assert flag, "experiment.py does not expose --gae-lambda"
    assert float(flag.group(1)) == 0.95
    assert re.search(r'"--gae-lambda",\s*str\(args\.gae_lambda\)', src), \
        "experiment.py parses --gae-lambda but never passes it to train_phase2.py"


def test_every_default_agrees_so_an_unflagged_run_matches_every_run_so_far():
    """Three places carry the default. If they drift, a run without the flag silently stops being
    comparable to the Nov 7 baseline, which is the only thing the fix will be measured against."""
    root = Path(__file__).resolve().parents[1] / "scripts"
    defaults = {"MAConfig": MAConfig().gae_lambda}
    for script in ("train_phase2.py", "experiment.py"):
        m = re.search(r'add_argument\("--gae-lambda",\s*type=float,\s*default=([0-9.]+)',
                      (root / script).read_text())
        defaults[script] = float(m.group(1))
    assert set(defaults.values()) == {0.95}, defaults


def test_the_experiment_parser_still_builds():
    """A regex over the source cannot catch a duplicate registration; building the parser can."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    p = subprocess.run([sys.executable, str(root / "scripts" / "experiment.py"), "--help"],
                       capture_output=True, text=True, cwd=root, timeout=600,
                       env={**__import__("os").environ, "PYTHONPATH": str(root)})
    assert p.returncode == 0, p.stderr[-2000:]
    assert "--gae-lambda" in p.stdout


def test_tick_scale_converts_lambda_like_gamma():
    """`gae_lambda` is given per 30-minute tick; the learner raises it to tick_scale, as for gamma.

    Every run so far is at the 30-minute calibration (tick_scale 1.0), so the effective lambda is
    the configured one. At 1-minute ticks it would not be, and the reach in TICKS changes while the
    reach in game time does not -- which is the point of the conversion.
    """
    assert MAConfig().tick_scale == 1.0
    assert MAConfig().gae_lambda ** MAConfig().tick_scale == 0.95
    fine = MAConfig(tick_scale=1.0 / 30)
    assert fine.gae_lambda ** fine.tick_scale > 0.998          # per-tick decay is far slower
