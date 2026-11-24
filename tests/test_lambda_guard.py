"""experiment.py must refuse to resume a run that used another GAE lambda.

Lambda is in neither `code_version` (which hashes lifesim/*.py, train_phase2.py and
continuous_world.py -- code, not arguments) nor the run path. So without this guard a lambda run
launched into the baseline's `--root` resumes the baseline's checkpoint in the same design folder,
and the comparison the run exists to make is void with nothing in the output saying so. The
notebook avoids it by deriving the root from LAM, but hprc/*.slurm and any run launched by hand
pass `--root` directly and bypass that entirely.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _experiment():
    """Import scripts/experiment.py, which is not inside a package."""
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("agon_experiment", ROOT / "scripts" / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return _experiment()


def _run(tmp_path, ex, *, lam, code=None, trained=True, haz=None):
    """A run directory as experiment.py would find it. lam=None omits the key, as old runs do."""
    d = tmp_path / "validate" / "caveman"
    d.mkdir(parents=True)
    args = {"code_version": code if code is not None else ex.code_version()}
    if lam is not None:
        args["gae_lambda"] = lam
    if haz is not None:
        args["hazard_coef"] = haz
    (d / "history.json").write_text(json.dumps({"args": args, "history": []}))
    if trained:
        (d / "params.msgpack").write_bytes(b"")
    return SimpleNamespace(root=str(tmp_path))


TODO = [("caveman", [])]


def test_the_same_lambda_is_not_stale(tmp_path, ex):
    args = _run(tmp_path, ex, lam=0.95)
    args.gae_lambda = 0.95
    assert ex.stale(args, "validate", TODO) == []


def test_a_different_lambda_is_refused(tmp_path, ex):
    args = _run(tmp_path, ex, lam=0.95)
    args.gae_lambda = 1.0
    bad = ex.stale(args, "validate", TODO)
    assert len(bad) == 1 and "gae_lambda 0.95" in bad[0] and "wants 1.0" in bad[0]


def test_a_run_from_before_the_flag_counts_as_0_95(tmp_path, ex):
    """Those runs all used the shipped default, so they are comparable to a 0.95 run and not to
    any other. Treating a missing key as unknown would either block valid resumes or allow bad
    ones; treating it as 0.95 is what actually happened."""
    args = _run(tmp_path, ex, lam=None)
    args.gae_lambda = 0.95
    assert ex.stale(args, "validate", TODO) == []


def test_a_run_from_before_the_flag_is_refused_for_a_new_lambda(tmp_path, ex):
    args = _run(tmp_path, ex, lam=None)
    args.gae_lambda = 1.0
    assert len(ex.stale(args, "validate", TODO)) == 1


def test_an_untrained_directory_is_not_stale(tmp_path, ex):
    """The guard is about reuse and resumption, so a folder with no weights is free to be used."""
    args = _run(tmp_path, ex, lam=0.95, trained=False)
    args.gae_lambda = 1.0
    assert ex.stale(args, "validate", TODO) == []


def test_a_code_mismatch_is_reported_as_code_and_not_also_as_lambda(tmp_path, ex):
    """One run, one reason. A wrong code version makes lambda moot."""
    args = _run(tmp_path, ex, lam=1.0, code="deadbeefcafe")
    args.gae_lambda = 0.95
    bad = ex.stale(args, "validate", TODO)
    assert len(bad) == 1 and "code deadbeefcafe" in bad[0] and "gae_lambda" not in bad[0]


def test_args_without_gae_lambda_default_to_0_95(tmp_path, ex):
    """stale() is called with the real parsed args, but it must not crash on a caller that
    predates the flag."""
    args = _run(tmp_path, ex, lam=0.95)
    assert ex.stale(args, "validate", TODO) == []


# ---------------------------------------------------------------- hazard_coef (candidate C)
# --hazard-coef shipped in 97e65aa with help text telling the user to keep a separate --root per
# value, and with nothing enforcing it. So the baseline-versus-hazard A/B -- the whole point of the
# flag -- was as silently corruptible as the lambda A/B this file was written for. These pin the
# guard that closes it.

def test_the_same_hazard_coef_is_not_stale(tmp_path, ex):
    args = _run(tmp_path, ex, lam=0.95, haz=0.5)
    args.gae_lambda, args.hazard_coef = 0.95, 0.5
    assert ex.stale(args, "validate", TODO) == []


def test_a_hazard_run_refuses_to_resume_a_baseline(tmp_path, ex):
    """The failure this exists to prevent: --hazard-coef 0.5 pointed at the baseline's root would
    have resumed the baseline's checkpoint and reported the result as the hazard condition."""
    args = _run(tmp_path, ex, lam=0.95, haz=0.0)
    args.gae_lambda, args.hazard_coef = 0.95, 0.5
    bad = ex.stale(args, "validate", TODO)
    assert len(bad) == 1 and "hazard_coef 0.0" in bad[0] and "wants 0.5" in bad[0]


def test_a_baseline_refuses_to_resume_a_hazard_run(tmp_path, ex):
    """And the reverse direction, which is just as wrong and just as quiet."""
    args = _run(tmp_path, ex, lam=0.95, haz=0.5)
    args.gae_lambda, args.hazard_coef = 0.95, 0.0
    bad = ex.stale(args, "validate", TODO)
    assert len(bad) == 1 and "hazard_coef 0.5" in bad[0] and "wants 0.0" in bad[0]


def test_a_run_from_before_the_hazard_flag_counts_as_off(tmp_path, ex):
    """Every run made before the flag existed had no head, so 0.0 is what actually happened."""
    args = _run(tmp_path, ex, lam=0.95, haz=None)
    args.gae_lambda, args.hazard_coef = 0.95, 0.0
    assert ex.stale(args, "validate", TODO) == []
    args.hazard_coef = 0.5
    assert len(ex.stale(args, "validate", TODO)) == 1


def test_code_version_still_takes_precedence(tmp_path, ex):
    """One reason per run in the message, and the code mismatch is the one that matters first."""
    args = _run(tmp_path, ex, lam=0.95, haz=0.0, code="deadbeefcafe")
    args.gae_lambda, args.hazard_coef = 0.95, 0.5
    bad = ex.stale(args, "validate", TODO)
    assert len(bad) == 1 and "code deadbeefcafe" in bad[0] and "hazard_coef" not in bad[0]


def test_the_refusal_message_names_hazard_coef(tmp_path, ex):
    src = (ROOT / "scripts" / "experiment.py").read_text()
    assert "hazard_coef" in src.split("raise SystemExit(\"These runs would be reused")[1][:600], \
        "the SystemExit text must tell the user hazard_coef is part of the check"
