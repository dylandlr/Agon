"""Candidate C's auxiliary head: the "dies within k ticks" label, and the head's wiring.

The label shape was chosen by measurement, not argument (`scripts/probe_label_form.py`): from a
frozen random trunk the hazard reaches held-out AUC 0.907 at k=48, the best of k in {12, 24, 48,
96}. The reason it is the label, though, is that it is the only one of the four formulations
computable inside a 176-tick rollout window: `fraction of life elapsed` needs the life's total T
and the centred variants need the life's mean, whatever they score.

(The earlier version of this docstring justified the choice with `fraction of life elapsed` at
R^2 -2.424. That figure was an artifact of a missing intercept in the probe's ridge -- see
docs/MEASUREMENT.md 0 -- and the label in fact scores +0.621. The choice is unchanged, because
in-window computability is what decides it; the published reason was half wrong.)

The risky part is the label, so it is tested against hand-computed cases rather than against
itself. The rest pins that the head is genuinely optional: with `hazard_coef` 0 the parameters are
not created, `__call__` keeps its arity so no existing caller breaks, and the history row has no
`hazard` key, so an unflagged run stays comparable to the Nov 7 baseline field for field.
"""
import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import lifesim.config as C
import lifesim.env as E
from lifesim.policy import make_network
from lifesim.ppo_multi import MAConfig, hazard_labels

ROOT = Path(__file__).resolve().parents[1]


def _lv(died, k):
    y, v = hazard_labels(jnp.asarray(died, bool)[:, None], k)
    return (np.asarray(y).ravel().astype(int).tolist(),
            np.asarray(v).ravel().astype(int).tolist())


def test_the_label_marks_the_k_ticks_before_a_death():
    """Dies at tick 3, k=2: ticks 2 and 3 have remaining < 2. Tick 1 has remaining exactly 2, which
    is NOT within 2 -- the label matches the probe's `remaining < k`, so the boundary must agree."""
    label, valid = _lv([0, 0, 0, 1, 0], 2)
    assert label == [0, 0, 1, 1, 0]
    assert valid == [1, 1, 1, 1, 0]


def test_a_tick_is_valid_only_where_the_window_settles_it():
    """No death anywhere: "does not die within 3" is proven only where 3 whole ticks remain.
    Without this the last k-1 ticks of every window would be taught a 0 the data does not support."""
    label, valid = _lv([0] * 6, 3)
    assert label == [0] * 6
    assert valid == [1, 1, 1, 1, 0, 0]


def test_nothing_is_supervised_when_k_exceeds_the_window():
    """At k=48 in a 16-tick window with no death, no label is settled, so the head learns nothing
    and `hazard_rate` reads 0. That is correct, and it is how a short smoke run behaves -- do not
    read those zeros as the head being broken."""
    label, valid = _lv([0] * 16, 48)
    assert label == [0] * 16
    assert valid == [0] * 16


def test_each_occupant_of_a_refilled_slot_gets_its_own_label():
    """A slot whose agent dies and is refilled by a birth mid-window. The recursion finds the first
    death at or after t, which belongs to whoever holds the slot at t, because a death is what ends
    an occupancy -- so the newborn is never taught the dead agent's fate."""
    label, valid = _lv([0, 1, 0, 0, 1, 0], 2)
    assert label == [1, 1, 0, 1, 1, 0]
    assert valid == [1, 1, 1, 1, 1, 0]


def test_a_death_on_the_first_tick_is_labelled_without_needing_room():
    label, valid = _lv([1, 0, 0], 5)
    assert label[0] == 1 and valid[0] == 1


def test_the_label_is_per_slot_not_shared():
    """Two slots, different death ticks: the scan must not mix them."""
    died = np.zeros((6, 2), bool)
    died[1, 0] = True
    died[4, 1] = True
    y, v = hazard_labels(jnp.asarray(died), 2)
    assert np.asarray(y)[:, 0].astype(int).tolist() == [1, 1, 0, 0, 0, 0]
    assert np.asarray(y)[:, 1].astype(int).tolist() == [0, 0, 0, 1, 1, 0]


def test_the_head_is_off_by_default_and_creates_no_parameters():
    assert MAConfig().hazard_coef == 0.0
    assert MAConfig().hazard_k == 48
    cfg = C.EnvConfig(n_agents=2, grid_size=32, reward_mode="survival")
    A, D = E.N_ACTIONS + E.EXTRA_LOGITS, E.obs_dim(cfg)
    x = (jnp.zeros((1, 2, D)), jnp.ones((1, 2), bool), jnp.ones((1, 2, A), bool))
    import jax
    sizes = {}
    for hz in (False, True):
        net = make_network(cfg, hidden=32, encoder="conv", messages=True, world_model=True, hazard=hz)
        p = net.init(jax.random.PRNGKey(0), jnp.zeros((2, 32)), x)
        sizes[hz] = sum(v.size for v in jax.tree_util.tree_leaves(p))
        assert ("haz_hidden" in str(p)) is hz
        terms = net.apply(p, jnp.zeros((2, 32)), x, jnp.zeros((1, 2), jnp.int32),
                          method=net.learn_terms)
        assert ("hazard" in terms) is hz, (hz, sorted(terms))
    assert sizes[True] > sizes[False], sizes


def test_turning_the_head_on_does_not_change_the_call_signature():
    """ppo_multi, continuous_world, diagnose and the probes all unpack a 3-tuple from __call__.
    The head is reachable from learn_terms and trace, so none of them has to change."""
    import jax
    cfg = C.EnvConfig(n_agents=2, grid_size=32, reward_mode="survival")
    A, D = E.N_ACTIONS + E.EXTRA_LOGITS, E.obs_dim(cfg)
    x = (jnp.zeros((1, 2, D)), jnp.ones((1, 2), bool), jnp.ones((1, 2, A), bool))
    net = make_network(cfg, hidden=32, encoder="conv", messages=True, world_model=True, hazard=True)
    p = net.init(jax.random.PRNGKey(0), jnp.zeros((2, 32)), x)
    out = net.apply(p, jnp.zeros((2, 32)), x)
    assert len(out) == 3
    _, logits, value = out
    assert logits.shape[:2] == (1, 2) and value.shape == (1, 2)


@pytest.mark.parametrize("script,flag", [("train_phase2.py", "--hazard-coef"),
                                         ("train_phase2.py", "--hazard-k"),
                                         ("experiment.py", "--hazard-coef")])
def test_the_flags_exist_and_default_to_off(script, flag):
    src = (ROOT / "scripts" / script).read_text()
    m = re.search(rf'add_argument\("{flag}",\s*type=(?:float|int),\s*default=([0-9.]+)', src)
    assert m, f"{script} does not expose {flag}"
    if flag == "--hazard-coef":
        assert float(m.group(1)) == 0.0
    else:
        assert int(float(m.group(1))) == 48


def test_the_coefficient_reaches_the_learner_and_the_row_is_conditional():
    src = (ROOT / "scripts" / "train_phase2.py").read_text()
    assert re.search(r"hazard_coef=args\.hazard_coef", src), "flag never reaches MAConfig"
    assert re.search(r"hazard_k=args\.hazard_k", src), "horizon never reaches MAConfig"
    # the row must omit the block entirely when off, so an unflagged run stays field-comparable
    assert 'if args.hazard_coef > 0 else {}' in src
    exp = (ROOT / "scripts" / "experiment.py").read_text()
    assert re.search(r'"--hazard-coef",\s*str\(args\.hazard_coef\)', exp), \
        "experiment.py parses --hazard-coef but never forwards it"
