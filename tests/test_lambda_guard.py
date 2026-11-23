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


def _run(tmp_path, ex, *, lam, code=None, trained=True):
    """A run directory as experiment.py would find it. lam=None omits the key, as old runs do."""
    d = tmp_path / "validate" / "caveman"
    d.mkdir(parents=True)
    args = {"code_version": code if code is not None else ex.code_version()}
    if lam is not None:
        args["gae_lambda"] = lam
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
