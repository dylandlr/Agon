"""Body parts and aimed blows (EnvConfig.body_parts), and the physiology of needs and force."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import fauna as fa
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=12, n_agents=2, max_steps=1000, climate_volatility=0.0, season_temp=(18.0,) * 4,
                day_temp_swing=0.0, lightning_prob=0.0, downed=True)
SP = E.species_params(CFG)


def _world(cfg=CFG):
    s = E.reset(jax.random.PRNGKey(5), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    g = s.genotype._replace(strength=jnp.ones(2), height=jnp.ones(2), build=jnp.ones(2), pain_aversion=jnp.ones(2))
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators), genotype=g,
                   conditions=jnp.zeros_like(s.conditions), pain_aversion=jnp.ones(2),
                   birth_age=jnp.full(2, int(25 * cfg.ticks_per_bio_year), jnp.int32))
    t = np.asarray(s.terrain)
    west, mid = next(((r, c - 1), (r, c)) for r in range(12) for c in range(1, 11)
                     if t[r, c] != E.WATER and t[r, c - 1] != E.WATER)
    return s._replace(pos=jnp.asarray([west, mid], jnp.int32)), west, mid


def _aim(part, n=2):
    o = np.zeros((n, E.OUT_SLOTS), np.int32)
    o[:, -1] = part
    return jnp.asarray(o)


def _swings(s, part, cfg=CFG, k=400, actions=(E.STRIKE_N + 2, E.REST)):
    keys = jax.random.split(jax.random.PRNGKey(7), k)
    return jax.vmap(lambda kk: E.step(s._replace(key=kk), jnp.array(actions), cfg, _aim(part, len(actions))))(keys)


def test_where_a_blow_is_aimed_sets_whether_it_lands_and_what_it_does():
    s, _, _ = _world()
    for part, p in zip(range(E.N_PARTS), CFG.part_hit):
        st, _, _, info = _swings(s, part)
        landed = np.asarray(info.blow_landed[:, 0])
        assert abs(landed.mean() - p) < 0.07, (E.PARTS[part], landed.mean())
        hurt = np.asarray(st.part_damage[:, 1, part])
        assert (hurt[landed] > 0).all() and (hurt[~landed] == 0).all()           # a miss does nothing
        assert (np.asarray(st.part_damage[:, 1])[:, [q for q in range(E.N_PARTS) if q != part]] == 0).all()
    # how much health a landed blow takes follows the part's lethality
    loss = {}
    for part in (E.PART_NECK, E.PART_ARMS):
        st, _, _, info = _swings(s, part)
        landed = np.asarray(info.blow_landed[:, 0])
        loss[part] = float((1 - np.asarray(st.health[:, 1]))[landed].mean())
    ratio = loss[E.PART_NECK] / loss[E.PART_ARMS]
    assert abs(ratio - CFG.part_lethality[E.PART_NECK] / CFG.part_lethality[E.PART_ARMS]) < 0.6


def test_the_downed_are_hit_wherever_one_aims_and_a_ruined_neck_kills():
    s, _, _ = _world()
    down = s._replace(pain=jnp.array([0.0, 1.5]))
    _, _, _, info = _swings(down, E.PART_NECK, k=100)
    assert np.asarray(info.blow_landed[:, 0]).all()
    near = down._replace(part_damage=down.part_damage.at[1, E.PART_NECK].set(0.95))
    st, _, _, info = E.step(near, jnp.array([E.STRIKE_N + 2, E.REST]), CFG, _aim(E.PART_NECK))
    assert not bool(st.alive[1]) and int(info.death_cause[1]) == E.CAUSE_VIOLENCE


def test_a_blow_to_the_head_can_knock_one_out():
    s, _, _ = _world()
    st = s._replace(part_damage=s.part_damage.at[1, E.PART_HEAD].set(CFG.concussion_at))
    assert bool(E.downed_of(st, CFG)[1]) and not bool(E.downed_of(st, CFG)[0])


def test_hurt_arms_weaken_and_ruined_legs_stop_walking():
    s, _, _ = _world()
    hurt = s._replace(part_damage=s.part_damage.at[0, E.PART_ARMS].set(0.5).at[1, E.PART_LEGS].set(0.95))
    b0, b1 = E.body(s, CFG), E.body(hurt, CFG)
    assert np.isclose(float(b1["strength"][0]), float(b0["strength"][0]) * (1 - CFG.arm_weakness * 0.5))
    mask = np.asarray(E.action_mask(hurt, CFG))
    assert not mask[1, E.MOVE_N:E.MOVE_N + 8].any() and mask[1, E.REST]
    assert mask[0, E.MOVE_N:E.MOVE_N + 8].any()


def test_a_blow_to_a_leg_lames_an_animal():
    from test_byproducts import CFG as ONE, _put, _world as one_world
    s, west, mid = one_world()
    s = s._replace(large=_put(s.large, "deer", mid, float(SP["hp"][fa.IDX["deer"]])))
    keys = jax.random.split(jax.random.PRNGKey(3), 200)
    lame = lambda part: float((jax.vmap(lambda kk: E.step(s._replace(key=kk), jnp.array([E.STRIKE_N + 2]), ONE,
                                                         _aim(part, 1))[0].large.lame[0])(keys) > 0).mean())
    assert lame(E.PART_LEGS) > 0.6 and lame(E.PART_TORSO) == 0.0


def test_ones_own_wounds_are_felt_part_by_part():
    s, _, _ = _world()
    s = s._replace(part_damage=s.part_damage.at[0, E.PART_LEGS].set(0.4))
    v, c = CFG.view_size, E.N_OBS_CHANNELS
    own = np.asarray(E.observe(s, CFG))[0, v * v * c + 25: v * v * c + 25 + E.N_PARTS]
    assert np.allclose(own, [0, 0, 0, 0, 0.4])


def test_bigger_frames_and_harder_work_need_more():
    s, _, _ = _world()
    big = s._replace(genotype=s.genotype._replace(height=jnp.array([1.1, 1.0]), build=jnp.array([1.2, 1.0])))
    st = big._replace(satiety=jnp.full(2, 0.9))
    loss = lambda a, cfg=CFG: [0.9 - float(x) for x in E.step(st, jnp.array(a), cfg)[0].satiety]
    rest = loss([E.REST, E.REST])
    assert rest[0] > rest[1]                                    # the bigger frame burns more
    walk = loss([E.REST, E.MOVE_N + 2])
    flat = loss([E.REST, E.MOVE_N + 2], dataclasses.replace(CFG, activity_needs=False))
    assert walk[1] > rest[1] and abs(flat[1] - flat[0] * (rest[1] / rest[0])) < 1e-9 or flat[1] < walk[1]


def test_heavier_practised_people_hit_harder_and_no_two_blows_are_alike():
    s, _, _ = _world()
    cfg = dataclasses.replace(CFG, downed=False, pain_shock=0.0)
    keys = jax.random.split(jax.random.PRNGKey(11), 300)

    def hurt(st):
        st2, _, _, info = jax.vmap(lambda kk: E.step(st._replace(key=kk), jnp.array([E.STRIKE_N + 2, E.REST]), cfg,
                                                     _aim(E.PART_TORSO)))(keys)
        lost = 1 - np.asarray(st2.health[:, 1])
        return lost[np.asarray(info.blow_landed[:, 0])]
    base = hurt(s)
    big = s._replace(genotype=s.genotype._replace(height=jnp.array([1.1, 1.0]), build=jnp.array([1.2, 1.0])))
    heavy = hurt(big._replace(weight=E.body(big, cfg)["ideal_kg"]))         # at a healthy weight for that frame
    practised = hurt(s._replace(life_strikes=jnp.array([10_000, 0], jnp.int32)))
    assert heavy.mean() > 1.1 * base.mean() and practised.mean() > 1.15 * base.mean()
    assert base.std() > 0.1 * base.mean()                       # luck: blows vary
    off = dataclasses.replace(cfg, blow_by_mass=False)
    st2, _, _, info = jax.vmap(lambda kk: E.step(s._replace(key=kk), jnp.array([E.STRIKE_N + 2, E.REST]), off,
                                                 _aim(E.PART_TORSO)))(keys[:50])
    lost = (1 - np.asarray(st2.health[:, 1]))[np.asarray(info.blow_landed[:, 0])]
    assert lost.std() < 1e-6
