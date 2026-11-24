"""The two silent bugs the probes shipped with, pinned so they cannot come back.

Both were found while measuring candidate D and both had already contaminated published figures in
docs/MEASUREMENT.md, so they are worth a test rather than a comment.

1. THE ANNIHILATED INTERCEPT. The probes built their design matrix as
   `np.concatenate([X, ones], 1)` and THEN scaled the whole thing by `X[tr].std(0) + 1e-9`. The
   ones column has zero train variance, so it became (1-1)/1e-9 = exactly 0: every such ridge was
   fitted with no intercept against an uncentred target. The cost is about (mean(y))^2 / var(y) of
   R^2 -- nothing when the target happens to be centred within each life, and catastrophic when it
   is not, which is exactly the contrast those probes were reading.

2. SILENT AGENTS. `E.step(state, actions, cfg, messages=None)` defaults to silence and gates a
   whole utterance on its first symbol. Every probe written before probe_bottleneck.py left
   `messages` at the default, so the 402 heard and 130 read dimensions were identically zero and
   nothing could have been measured about them.

The `standardize` helper is duplicated in three standalone probes on purpose (they are
instruments, each readable on its own), so the numerical test extracts each shipped copy by AST
and exercises it, rather than testing one import and hoping the other two match.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBES = ["probe_aux_tasks.py", "probe_label_form.py", "probe_bottleneck.py"]


def _standardize_of(name):
    """Exec just the `standardize` function out of a probe, with numpy in scope."""
    src = (ROOT / "scripts" / name).read_text()
    tree = ast.parse(src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "standardize"),
              None)
    assert fn is not None, f"{name} has no standardize()"
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), name, "exec"), ns)
    return ns["standardize"]


@pytest.mark.parametrize("name", PROBES)
def test_the_intercept_column_survives_scaling(name):
    """The last column must be 1.0 everywhere -- train rows and held-out rows alike.

    This is the whole bug in one assertion. With the old ordering this column was all zeros, and
    nothing else about the fit looked wrong.
    """
    f = _standardize_of(name)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    tr = np.arange(35)
    Z = f(X, tr)
    assert Z.shape == (50, 5), Z.shape
    assert np.allclose(Z[:, -1], 1.0), "intercept column is not 1.0"


@pytest.mark.parametrize("name", PROBES)
def test_a_constant_column_is_zeroed_not_divided_by_an_epsilon(name):
    """`feats` has hundreds of constant columns. Dividing their spread by 1e-9 amplifies rounding
    noise ~1e9 into the Gram matrix; they carry no information, so they must be zeroed."""
    f = _standardize_of(name)
    X = np.ones((40, 3))
    X[:, 1] = np.linspace(0, 1, 40)
    Z = f(X, np.arange(30))
    assert np.allclose(Z[:, 0], 0.0) and np.allclose(Z[:, 2], 0.0)
    assert Z[:, 1].std() > 0.1, "the informative column must survive"
    assert np.isfinite(Z).all()


@pytest.mark.parametrize("name", PROBES)
def test_an_uncentred_target_is_recovered(name):
    """y with a mean far from zero and a single informative feature. Without an intercept this
    reads R^2 of about -(mean^2/var), which for these labels is in the thousands."""
    f = _standardize_of(name)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6))
    y = 300.0 + 5.0 * X[:, 0] + rng.normal(size=400)
    tr, te = np.arange(300), np.arange(300, 400)
    Z = f(X, tr)
    w = np.linalg.solve(Z[tr].T @ Z[tr] + np.eye(Z.shape[1]), Z[tr].T @ y[tr])
    r2 = 1 - ((y[te] - Z[te] @ w) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
    assert r2 > 0.9, f"{name}: uncentred target recovered at R^2 {r2:.3f}"


def _code_only(name):
    """The probe's source with every comment and string literal removed.

    Needed because these files now DOCUMENT the forbidden pattern in prose, and a plain substring
    search over the raw text flags the explanation as the offence.
    """
    import io
    import tokenize
    src = (ROOT / "scripts" / name).read_text()
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


@pytest.mark.parametrize("name", PROBES)
def test_the_dangerous_ordering_is_gone(name):
    """Append-then-scale is the bug. Scale-then-append is the fix. Forbid the old shape in CODE,
    because it fails silently and shows up only in the figures."""
    code = _code_only(name)
    assert not re.search(r"std \( 0 \) \+ 1e-9", code), "epsilon-padded std is back"
    # Exactly one place may build an intercept column, and it must be standardize().
    assert len(re.findall(r"np \. ones \( \( len \( X \) , 1 \) \)", code)) <= 1, \
        "a ones column is appended outside standardize(); scale first, then append"
    assert "def standardize" in code, f"{name} no longer routes through standardize()"


def test_the_probe_that_measures_language_actually_lets_agents_speak():
    """probe_bottleneck must pass `messages` to E.step by default, or its language block is all
    zeros and M1/M2/M3 measure nothing. `--silent` reproduces what the earlier probes did."""
    src = (ROOT / "scripts" / "probe_bottleneck.py").read_text()
    assert "sample_outputs" in src, "no messages are ever sampled"
    assert re.search(r'add_argument\("--silent"', src), "the silent control is missing"
    assert re.search(r"SPEAK\s*=\s*not args\.silent", src), "speaking must be the default"


def test_the_feature_blocks_tile_feats_exactly():
    """The block boundaries are arithmetic over env constants, so a change to any of HEARD, READ,
    N_MAT, N_WEATHER or the map silently moves them. The probe asserts this at run time; assert it
    here too so the suite catches it without a 40-minute rollout."""
    import lifesim.config as C
    import lifesim.env as E
    cfg = C.EnvConfig(n_agents=2, grid_size=32, reward_mode="survival")
    nwin = cfg.view_size ** 2 * E.N_OBS_CHANNELS
    lang0 = E.OWN_FEATURES + 2 * E.N_MAT + 1 + 5 + E.N_WEATHER
    lang1 = lang0 + E.HEARD_FEATURES + E.READ_FEATURES
    assert lang1 == E.N_BODY_FEATURES, (lang1, E.N_BODY_FEATURES)
    nfeat = E.obs_dim(cfg) - nwin
    assert nfeat == E.N_BODY_FEATURES + E.map_features(cfg)
    from lifesim.policy import DECODE_INDEX
    assert max(DECODE_INDEX) < E.OWN_FEATURES, "a reward scalar fell outside the own-body block"
