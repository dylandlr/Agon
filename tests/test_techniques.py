"""How a blow is delivered (EnvConfig.techniques): limb x mode x part, and what pushes and holds do.
Agent 0 stands west of agent 1 and acts east (STRIKE_N + 2)."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.recorder import technique_name
from test_body import CFG, _world

EAST = E.STRIKE_N + 2
STEP_EAST = E.MOVE_N + 2


def _how(part, limb, mode, n=2):
    o = np.zeros((n, E.OUT_SLOTS), np.int32)
    o[:, -1] = E.how_of(part, limb, mode)
    return jnp.asarray(o)


def _many(s, how, actions=(EAST, E.REST), k=400, cfg=CFG):
    keys = jax.random.split(jax.random.PRNGKey(11), k)
    return jax.vmap(lambda kk: E.step(s._replace(key=kk), jnp.array(actions), cfg, how))(keys)


def test_the_joint_index_keeps_a_hand_strike_at_part_k_as_k():
    for part in range(E.N_PARTS):
        assert E.how_of(part) == part
        for limb in range(E.N_LIMBS):
            for mode in range(E.N_MODES):
                assert tuple(int(x) for x in E.how_parts(E.how_of(part, limb, mode))) == (part, limb, mode)
    assert E.N_HOW == 60


def test_a_kick_lands_less_often_than_a_punch_but_hits_harder():
    s, _, _ = _world()
    res = {}
    for limb in (E.LIMB_HAND, E.LIMB_FOOT):
        st, _, _, info = _many(s, _how(E.PART_TORSO, limb, E.MODE_STRIKE))
        landed = np.asarray(info.blow_landed[:, 0])
        res[limb] = landed.mean(), float(np.asarray(st.part_damage[:, 1, E.PART_TORSO])[landed].mean())
    assert res[E.LIMB_FOOT][0] < res[E.LIMB_HAND][0] - 0.1
    assert res[E.LIMB_FOOT][1] > 1.3 * res[E.LIMB_HAND][1]
    # ... and much less often at the head
    _, _, _, info = _many(s, _how(E.PART_HEAD, E.LIMB_FOOT, E.MODE_STRIKE))
    assert abs(np.asarray(info.blow_landed[:, 0]).mean()
               - CFG.part_hit[E.PART_HEAD] * CFG.limb_accuracy[E.LIMB_FOOT] * CFG.kick_high_accuracy) < 0.07


def test_a_shove_moves_and_staggers_and_a_heavier_body_resists_it():
    s, west, mid = _world()
    st, _, _, info = _many(s, _how(E.PART_TORSO, E.LIMB_BODY, E.MODE_PUSH))
    moved = np.asarray(info.shoved[:, 1])
    assert 0.2 < moved.mean() < 0.95
    pos = np.asarray(st.pos[:, 1])
    assert (pos[moved] == np.array([mid[0], mid[1] + 1])).all() and (pos[~moved] == np.array(mid)).all()
    assert (np.asarray(st.off_balance[:, 1])[moved] >= 1).all()
    # staggered: the next tick it cannot step
    one = jax.tree_util.tree_map(lambda x: x[int(np.flatnonzero(moved)[0])], st)
    nxt, _, _, _ = E.step(one, jnp.array([E.REST, STEP_EAST]), CFG, _how(E.PART_TORSO, 0, 0))
    assert (np.asarray(nxt.pos[1]) == np.asarray(one.pos[1])).all()
    # a stronger, heavier victim is moved less often
    g = s.genotype._replace(strength=jnp.array([1.0, 1.6]), build=jnp.array([1.0, 1.4]))
    big = s._replace(genotype=g, weight=s.weight * jnp.array([1.0, 1.4]))
    _, _, _, info2 = _many(big, _how(E.PART_TORSO, E.LIMB_BODY, E.MODE_PUSH))
    assert np.asarray(info2.shoved[:, 1]).mean() < moved.mean() - 0.1


def test_a_hold_pins_and_opens_the_held_to_blows():
    s, _, mid = _world()
    st, _, _, info = _many(s, _how(E.PART_ARMS, E.LIMB_HAND, E.MODE_HOLD))
    got = np.asarray(info.grabbed[:, 1])
    assert 0.2 < got.mean() < 0.95 and (np.asarray(st.held_by[:, 1])[got] == 0).all()
    assert (np.asarray(st.held_by[:, 1])[~got] == -1).all()
    held = jax.tree_util.tree_map(lambda x: x[int(np.flatnonzero(got)[0])], st)
    grip = float(held.hold_grip[1])
    # the held one tries to step away: it fails about as often as the grip
    _, _, _, i2 = _many(held, _how(E.PART_TORSO, 0, 0), actions=(E.REST, STEP_EAST))
    assert abs(np.asarray(i2.pinned[:, 1]).mean() - grip) < 0.08
    # a blow at the head lands far more often on the held
    _, _, _, free = _many(s, _how(E.PART_HEAD, 0, 0))
    _, _, _, on_held = _many(held, _how(E.PART_HEAD, 0, 0))
    p = CFG.part_hit[E.PART_HEAD]
    assert abs(np.asarray(on_held.blow_landed[:, 0]).mean() - (p + (1 - p) * grip)) < 0.08
    assert np.asarray(on_held.blow_landed[:, 0]).mean() > np.asarray(free.blow_landed[:, 0]).mean() + 0.1
    # a head cannot grip
    _, _, _, i3 = _many(s, _how(E.PART_ARMS, E.LIMB_HEAD, E.MODE_HOLD))
    assert not np.asarray(i3.grabbed[:, 1]).any()


def test_hold_then_push_is_a_throw():
    s, _, _ = _world()
    held = s._replace(held_by=jnp.array([-1, 0]), hold_grip=jnp.array([0.0, 0.5]))
    st, _, _, info = _many(held, _how(E.PART_TORSO, E.LIMB_HAND, E.MODE_PUSH))
    thrown = np.asarray(info.thrown[:, 1])
    _, _, _, plain = _many(s, _how(E.PART_TORSO, E.LIMB_HAND, E.MODE_PUSH))
    assert thrown.mean() > np.asarray(plain.shoved[:, 1]).mean() + 0.1          # leverage: more often
    assert not np.asarray(plain.thrown[:, 1]).any()                              # no hold, no throw
    assert (np.asarray(st.off_balance[:, 1])[thrown] == CFG.throw_ticks).all()
    torso = np.asarray(st.part_damage[:, 1, E.PART_TORSO])
    assert torso[thrown].mean() > 2 * torso[~thrown].mean()


def test_a_headbutt_jars_ones_own_head():
    s, _, _ = _world()
    st, _, _, info = _many(s, _how(E.PART_HEAD, E.LIMB_HEAD, E.MODE_STRIKE))
    landed = np.asarray(info.blow_landed[:, 0])
    own = np.asarray(st.part_damage[:, 0, E.PART_HEAD])
    assert (own[landed] > 0).all() and (own[~landed] == 0).all()
    st2, _, _, _ = _many(s, _how(E.PART_HEAD, E.LIMB_HAND, E.MODE_STRIKE))
    assert (np.asarray(st2.part_damage[:, 0, E.PART_HEAD]) == 0).all()


def test_without_techniques_every_blow_is_a_hand_strike():
    s, _, _ = _world()
    off = dataclasses.replace(CFG, techniques=False)
    a, _, _, ia = _many(s, _how(E.PART_TORSO, E.LIMB_BODY, E.MODE_PUSH), cfg=off, k=50)
    b, _, _, ib = _many(s, _how(E.PART_TORSO, 0, 0), cfg=off, k=50)
    assert np.allclose(np.asarray(a.part_damage), np.asarray(b.part_damage))
    assert not np.asarray(ia.shoved).any()


def test_names_come_after_the_fact():
    assert technique_name("torso", E.LIMB_HAND, E.MODE_STRIKE) == "punch"
    assert technique_name("head", E.LIMB_FOOT, E.MODE_STRIKE) == "kick to the head"
    assert technique_name("torso", E.LIMB_FOOT, E.MODE_STRIKE, victim_down=True) == "stomp"
    assert technique_name("legs", E.LIMB_HAND, E.MODE_PUSH, thrown=True) == "trip throw"
    assert technique_name("neck", E.LIMB_HAND, E.MODE_HOLD, grabbed=True) == "chokehold"
