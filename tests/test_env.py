import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import genotype as gt
from lifesim.config import EnvConfig
from lifesim.recorder import random_policy, record_episode, rewind

CFG = EnvConfig(grid_size=24, n_agents=4, max_steps=300)


def _rollout(seed, cfg=CFG, steps=200):
    key = jax.random.PRNGKey(seed)
    state = E.reset(key, cfg)
    step = jax.jit(lambda s, a: E.step(s, a, cfg))
    k = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        k, sk = jax.random.split(k)
        a, _ = random_policy(sk, None, E.action_mask(state, cfg), None)
        state, *_ = step(state, a)
    return state


def test_determinism_same_seed():
    s1, s2 = _rollout(3), _rollout(3)
    for a, b in zip(jax.tree_util.tree_leaves(s1), jax.tree_util.tree_leaves(s2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_different_seeds_differ():
    s1, s2 = _rollout(3), _rollout(4)
    assert not np.array_equal(np.asarray(s1.terrain), np.asarray(s2.terrain))


def test_terrain_has_all_types_and_spawns_on_land():
    for seed in range(5):
        s = E.reset(jax.random.PRNGKey(seed), EnvConfig(grid_size=32, n_agents=50))
        types = set(np.unique(np.asarray(s.terrain)).tolist())
        assert types == {E.GRASS, E.FOREST, E.ROCK, E.WATER}
        terr_at = np.asarray(s.terrain)[np.asarray(s.pos[:, 0]), np.asarray(s.pos[:, 1])]
        assert not np.any(terr_at == E.WATER)


def test_masked_moves_never_enter_water_or_leave_grid():
    s = _rollout(7, steps=300)
    p = np.asarray(s.pos)
    assert np.all((p >= 0) & (p < CFG.grid_size))
    assert not np.any(np.asarray(s.terrain)[p[:, 0], p[:, 1]] == E.WATER)


def test_invalid_action_becomes_rest():
    s = E.reset(jax.random.PRNGKey(0), CFG)
    s = s._replace(inv=jnp.zeros((CFG.n_agents, E.N_MAT)))
    s2, _, _, info = E.step(s, jnp.full((CFG.n_agents,), E.EAT), CFG)
    assert np.all(np.asarray(info.outcome) == E.OUT_REST)


def test_contested_gather_single_winner_and_conservation():
    cfg = EnvConfig(grid_size=12, n_agents=3, max_steps=10)
    s = E.reset(jax.random.PRNGKey(1), cfg)
    s = s._replace(meat=jnp.zeros_like(s.meat))
    land = np.argwhere(np.asarray(s.terrain) != E.WATER)[0]
    pos = jnp.tile(jnp.asarray(land, dtype=jnp.int32), (3, 1))
    plants = s.plants.at[land[0], land[1]].set(2.0)
    s = s._replace(pos=pos, plants=plants)
    s2, _, _, info = E.step(s, jnp.full((3,), E.GRASP), cfg)
    out = np.asarray(info.outcome)
    assert (out == E.OUT_GATHER).sum() == 1
    assert (out == E.OUT_GATHER_LOST).sum() == 2
    # Food is conserved between the cell and inventories (before regrowth)
    regrow = float(s2.plants[land[0], land[1]]) - (2.0 - float(info.gathered.sum()))
    assert 0.0 <= regrow <= cfg.plant_regrow_forest + 1e-6
    assert np.isclose(float(s2.inv[:, E.PLANT].sum()), float(info.gathered.sum()))


def test_starvation_kills_with_cause():
    cfg = EnvConfig(grid_size=12, n_agents=1, max_steps=5000, season_temp=(18.0,) * 4,
                    day_temp_swing=0.0, climate_volatility=0.0, predators_per_1000=0.0,
                    lightning_prob=0.0)
    s = E.reset(jax.random.PRNGKey(2), cfg)
    s = s._replace(predators=s.predators._replace(alive=jnp.zeros_like(s.predators.alive)))
    s = s._replace(satiety=jnp.zeros(1), hydration=jnp.ones(1), energy=jnp.ones(1))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    died, cause = False, None
    for _ in range(400):
        s = s._replace(hydration=jnp.ones(1), energy=jnp.ones(1))
        s, _, done, info = step(s, jnp.array([E.REST]))
        if bool(info.died[0]):
            died, cause = True, int(info.death_cause[0])
            break
    assert died and cause == E.CAUSE_STARVATION


def test_observation_shape_and_bounds():
    s = E.reset(jax.random.PRNGKey(0), CFG)
    o = E.observe(s, CFG)
    assert o.shape == (CFG.n_agents, E.obs_dim(CFG))
    assert np.all(np.isfinite(np.asarray(o)))


def test_rewind_reproduces_exact_state():
    cfg = EnvConfig(grid_size=16, n_agents=3, max_steps=250)
    rec = record_episode(random_policy, None, seed=11, cfg=cfg, snapshot_every=50)
    target = 137
    replayed = rewind(rec, target)
    # Independent straight-line replay from t=0
    straight = rewind(type(rec)(cfg=cfg, seed=11, snapshot_every=10**9,
                                snapshots={0: rec.snapshots[0]}, actions=rec.actions), target)
    for a, b in zip(jax.tree_util.tree_leaves(replayed), jax.tree_util.tree_leaves(straight)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert int(replayed.t) == target


def test_genotype_recombination_in_bounds():
    k = jax.random.PRNGKey(0)
    m, f = gt.sample(k, 100), gt.sample(jax.random.PRNGKey(1), 100)
    c = gt.recombine_and_mutate(jax.random.PRNGKey(2), m, f, mutation_scale=0.5)
    assert np.all((np.asarray(c.patience) >= 0.95) & (np.asarray(c.patience) <= 0.999))
    assert np.all((np.asarray(c.strength) >= 0.25) & (np.asarray(c.strength) <= 4.0))


# ------------------------------------------------------------ new mechanics
def _quiet_world(n=2, **kw):
    """Small deterministic world with animals removed, constant mild climate."""
    kw.setdefault("lightning_prob", 0.0)
    cfg = EnvConfig(grid_size=12, n_agents=n, max_steps=1000, climate_volatility=0.0,
                    season_temp=(18.0,) * 4, day_temp_swing=0.0, **kw)
    s = E.reset(jax.random.PRNGKey(5), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   conditions=jnp.zeros_like(s.conditions),          # no inborn conditions: exact numbers
                   birth_age=jnp.full(n, int(25 * cfg.ticks_per_bio_year), jnp.int32))  # prime age
    land = np.argwhere(np.asarray(s.terrain) != E.WATER)
    return cfg, s, land


def _place_animal(an, i, cell, hp):
    return an._replace(pos=an.pos.at[i].set(jnp.asarray(cell, jnp.int32)),
                       alive=an.alive.at[i].set(True), hp=an.hp.at[i].set(hp),
                       rest=an.rest.at[i].set(0))


def _neighbor_pair(terrain):
    """A land cell with a land neighbor to its east and one to its west."""
    t = np.asarray(terrain)
    for r in range(t.shape[0]):
        for c in range(1, t.shape[1] - 1):
            if t[r, c] != E.WATER and t[r, c - 1] != E.WATER and t[r, c + 1] != E.WATER:
                return (r, c - 1), (r, c), (r, c + 1)
    raise AssertionError("no land triple")


def test_large_prey_reward_teamwork_more_than_solo_effort():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, east = _neighbor_pair(s.terrain)
    s = s._replace(genotype=s.genotype._replace(strength=jnp.ones(2)),
                   pos=jnp.asarray([west, east], jnp.int32))
    s = s._replace(large=_place_animal(s.large, 0, mid, cfg.large_prey_hp))
    strike_e, strike_w = E.STRIKE_N + 2, E.STRIKE_N + 6
    s1, *_ = E.step(s, jnp.array([strike_e, E.REST]), cfg)
    s2, *_ = E.step(s, jnp.array([strike_e, strike_w]), cfg)
    solo = cfg.large_prey_hp - float(s1.large.hp[0])
    pair = cfg.large_prey_hp - float(s2.large.hp[0])
    assert 0.15 < solo < 0.35                    # a lone blow wounds, a little
    assert pair > 2.5 * solo                     # two together do far more than twice as much


def test_small_prey_kill_drops_meat_and_credits_hunter():
    cfg, s, _ = _quiet_world(n=1)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west], jnp.int32),
                   genotype=s.genotype._replace(strength=jnp.ones(1)))
    s = s._replace(small=_place_animal(s.small, 0, mid, cfg.small_prey_hp))
    s2, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2]), cfg)
    assert bool(info.kills[0, 0]) and not bool(s2.small.alive[0])
    assert float(s2.meat[mid]) > 0.9 * cfg.small_prey_meat


def test_predator_attacks_lone_agent_but_not_group():
    cfg, s, _ = _quiet_world(n=2, predator_attack_prob=1.0)
    west, mid, east = _neighbor_pair(s.terrain)
    pred = _place_animal(s.predators, 0, west, cfg.predator_hp)
    # Lone agent at mid; the other agent far away
    far = tuple(np.argwhere(np.asarray(s.terrain) != E.WATER)[-1])
    lone = s._replace(pos=jnp.asarray([mid, far], jnp.int32), predators=pred)
    _, _, _, info = E.step(lone, jnp.array([E.REST, E.REST]), cfg)
    assert bool(info.attacked_by_predator[0])
    # Two agents side by side: the predator is deterred
    grouped = s._replace(pos=jnp.asarray([mid, east], jnp.int32), predators=pred)
    _, _, _, info = E.step(grouped, jnp.array([E.REST, E.REST]), cfg)
    assert not bool(info.attacked_by_predator.any())


def test_strike_on_agent_is_violence():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    s2, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    assert bool(info.struck_by_agent[1]) and float(s2.health[1]) < 1.0
    assert int(info.outcome[0]) == E.OUT_STRIKE_AGENT


def test_disease_runs_its_course_then_immunity():
    cfg, s, _ = _quiet_world(n=1, disease_duration=20)
    s = s._replace(infection=jnp.array([20], jnp.int32))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    recovered = False
    for _ in range(25):
        s, _, _, info = step(s, jnp.array([E.REST]))
        recovered |= bool(info.recovered[0])
    assert recovered and float(s.immunity[0]) > 0.9 and int(s.infection[0]) == 0


def test_contact_transmission():
    cfg, s, _ = _quiet_world(n=2, contact_infection=1.0)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                   infection=jnp.array([100, 0], jnp.int32))
    _, _, _, info = E.step(s, jnp.array([E.REST, E.REST]), cfg)
    assert bool(info.newly_infected[1])


def test_winter_cold_kills_in_the_open():
    cfg = EnvConfig(grid_size=12, n_agents=1, max_steps=5000, climate_volatility=0.0,
                    season_temp=(-15.0,) * 4, forest_fraction=0.0, lightning_prob=0.0)
    s = E.reset(jax.random.PRNGKey(3), cfg)
    s = s._replace(predators=s.predators._replace(alive=jnp.zeros_like(s.predators.alive)))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    cause = None
    for _ in range(2000):
        s = s._replace(satiety=jnp.ones(1), hydration=jnp.ones(1), energy=jnp.ones(1))
        s, _, _, info = step(s, jnp.array([E.REST]))
        if bool(info.died[0]):
            cause = int(info.death_cause[0])
            break
    assert cause == E.CAUSE_COLD


def test_weather_spells_have_bounded_duration():
    cfg = EnvConfig(grid_size=12, n_agents=1, max_steps=3000, climate_volatility=0.5)
    s = E.reset(jax.random.PRNGKey(9), cfg)
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    changes, seen = 0, set()
    for _ in range(1500):
        s = s._replace(health=jnp.ones(1), alive=jnp.ones(1, bool))
        s, _, _, info = step(s, jnp.array([E.REST]))
        changes += int(bool(info.weather_changed))
        seen.add(int(info.weather))
        assert 0 <= int(s.weather_timer) <= cfg.spell_max
    assert changes >= 3 and len(seen) >= 3


def test_trauma_raises_pain_aversion_phenotype():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                   genotype=s.genotype._replace(temperament_drift=jnp.ones(2)))
    before = float(s.pain_aversion[1])
    s2, *_ = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    assert float(s2.pain_aversion[1]) > before
    assert float(s2.stress[1]) > 0.0


def test_huddling_keeps_agents_warmer():
    cfg = EnvConfig(grid_size=12, n_agents=2, max_steps=1000, climate_volatility=0.0,
                    season_temp=(0.0,) * 4, day_temp_swing=0.0, forest_fraction=0.0)
    s = E.reset(jax.random.PRNGKey(4), cfg)
    west, mid, _ = _neighbor_pair(s.terrain)
    far = tuple(np.argwhere(np.asarray(s.terrain) != E.WATER)[-1])
    g = s.genotype._replace(insulation=jnp.ones(2))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    temps = {}
    for name, pos in [("together", [west, mid]), ("apart", [west, far])]:
        st = s._replace(pos=jnp.asarray(pos, jnp.int32), genotype=g)
        for _ in range(100):
            st, *_ = step(st, jnp.array([E.REST, E.REST]))
        temps[name] = float(st.body_temp[0])
    assert temps["together"] > temps["apart"] + 0.03


# ------------------------------------------------------------ Phase 2 social
def test_actions_beyond_strikes_are_not_strikes():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    for a in [E.RELEASE, E.TOUCH_N + 2, E.SELECT_0, E.SELECT_0 + 4, E.PLACE_N, E.PLACE_N + 7]:
        s2, _, _, info = E.step(s, jnp.array([a, E.REST]), cfg)
        assert not bool(info.struck_by_agent.any()), E.ACTION_NAMES[a]


def test_a_gift_is_food_someone_else_takes():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                   inv=jnp.zeros((2, E.N_MAT)).at[0, E.RAW_MEAT].set(2.0).at[0, E.STONE].set(1.0),
                   hand=jnp.array([E.RAW_MEAT, E.PLANT], jnp.int32),
                   meat=jnp.zeros_like(s.meat), plants=jnp.zeros_like(s.plants),
                   wood=jnp.zeros_like(s.wood), stones=jnp.zeros_like(s.stones))
    s1, _, _, info = E.step(s, jnp.array([E.RELEASE, E.REST]), cfg)
    assert np.isclose(float(s1.inv[0, E.RAW_MEAT]), 1.0 * (1 - cfg.meat_spoil_inventory))
    assert float(s1.meat[west]) > 0.9
    assert not bool(info.gifts.any())                        # dropping is not yet giving
    taker = s1._replace(pos=jnp.asarray([west, west], jnp.int32))
    s2, _, _, info = E.step(taker, jnp.array([E.REST, E.GRASP]), cfg)
    assert bool(info.gifts[0, 1])                            # the other agent took it: a gift
    assert float(s2.memory[1, 0, E.MEM_GAVE_NEAR_ME]) > 0.9
    _, _, _, info = E.step(s1, jnp.array([E.GRASP, E.REST]), cfg)
    assert not bool(info.gifts.any())                        # taking back one's own drop is not
    stone = s._replace(hand=jnp.array([E.STONE, E.PLANT], jnp.int32))
    s3, *_ = E.step(stone, jnp.array([E.RELEASE, E.REST]), cfg)
    assert int(s3.drop_owner[west]) == -1                    # dropping a stone gives nothing


def test_touch_memory_and_social_reward():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    _, r_rest, _, _ = E.step(s, jnp.array([E.REST, E.REST]), cfg)
    s2, r, _, info = E.step(s, jnp.array([E.TOUCH_N + 2, E.REST]), cfg)
    assert bool(info.touches[0, 1])
    assert float(s2.memory[1, 0, E.MEM_TOUCHED_ME]) > 0.9
    assert float(r[0]) > float(r_rest[0]) and float(r[1]) > float(r_rest[1])


def test_violence_is_remembered_by_victim_and_witnesses():
    cfg, s, _ = _quiet_world(n=3)
    west, mid, east = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid, east], jnp.int32))
    s2, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST, E.REST]), cfg)
    assert bool(info.hits[0, 1]) and not bool(info.hits[0, 2])
    assert float(s2.memory[1, 0, E.MEM_STRUCK_ME]) > 0.9
    assert float(s2.memory[2, 0, E.MEM_SEEN_VIOLENT]) > 0.9   # third party saw it
    assert float(s2.memory[0, 0, E.MEM_SEEN_VIOLENT]) == 0.0  # striker does not witness itself
    assert float(s2.memory[2, 0, E.MEM_STRUCK_ME]) == 0.0


def test_messages_are_heard_by_nearby_listeners_next_tick():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    msgs = jnp.zeros((2, E.MSG_LEN), jnp.int32).at[0, 0].set(3).at[0, 2].set(7)
    s2, _, _, info = E.step(s, jnp.array([E.REST, E.REST]), cfg, msgs)
    assert bool(info.spoke[0]) and not bool(info.spoke[1])
    obs = np.asarray(E.observe(s2, cfg))
    v, r, c = cfg.view_size, cfg.view_radius, E.N_OBS_CHANNELS
    win = obs[1, :v * v * c].reshape(v, v, c)
    spoke_ch = c - E.N_SOCIAL_FEATURES + E.N_MEM + E.N_APPEARANCE   # memory, appearance, then the spoke flag
    assert win[r, r - 1, spoke_ch] == 1.0                    # the speaker's cell shows it spoke
    h0 = v * v * c + E.N_BODY_FEATURES - E.READ_FEATURES - E.HEARD_FEATURES   # heard, then what one reads
    heard = obs[1, h0:h0 + E.HEARD_FEATURES]
    slot = heard[: E.MSG_LEN * E.VOCAB].reshape(E.MSG_LEN, E.VOCAB)
    assert slot[0, 3] == 1 and slot[1, 0] == 1 and slot[2, 7] == 1 and slot.sum() == E.MSG_LEN
    np.testing.assert_allclose(heard[E.MSG_LEN * E.VOCAB: E.MSG_LEN * E.VOCAB + 2], [0.0, -1.0 / r])
    assert obs[0, h0:h0 + E.HEARD_FEATURES].sum() == 0  # silent partner
    # Speaking costs a little energy
    quiet, *_ = E.step(s, jnp.array([E.REST, E.REST]), cfg)
    assert float(s2.energy[0]) < float(quiet.energy[0])


def test_fixed_population_across_worlds():
    cfg = EnvConfig(grid_size=16, n_agents=3)
    g = gt.sample(jax.random.PRNGKey(1), 3)
    app = jnp.ones((3, E.N_APPEARANCE)) * 0.5
    s1 = E.reset(jax.random.PRNGKey(2), cfg, genotype=g, appearance=app)
    s2 = E.reset(jax.random.PRNGKey(3), cfg, genotype=g, appearance=app)
    np.testing.assert_array_equal(np.asarray(s1.genotype.strength), np.asarray(s2.genotype.strength))
    assert not np.array_equal(np.asarray(s1.terrain), np.asarray(s2.terrain))


def test_patch_and_native_conv_agree():
    from lifesim.policy import PatchConv
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 11, 11, 5))
    for stride in (1, 2):
        m_patch, m_nat = PatchConv(7, stride, "patch"), PatchConv(7, stride, "native")
        p = m_patch.init(jax.random.PRNGKey(1), x)
        np.testing.assert_allclose(np.asarray(m_patch.apply(p, x)), np.asarray(m_nat.apply(p, x)),
                                   rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------ materials and fire
def _fire_world(n=1, **kw):
    cfg, s, _ = _quiet_world(n=n, **kw)
    west, mid, east = _neighbor_pair(s.terrain)
    s = s._replace(wood=jnp.zeros_like(s.wood), plants=jnp.zeros_like(s.plants),
                   meat=jnp.zeros_like(s.meat), cooked=jnp.zeros_like(s.cooked),
                   fire=jnp.zeros_like(s.fire))
    return cfg, s, west, mid, east


def test_select_then_grasp_picks_the_selected_material():
    cfg, s, west, mid, east = _fire_world()
    s = s._replace(pos=jnp.asarray([mid], jnp.int32), wood=s.wood.at[mid].set(2.0),
                   plants=s.plants.at[mid].set(2.0))
    s1, *_ = E.step(s, jnp.array([E.GRASP]), cfg)                 # hand starts as plant
    assert float(s1.inv[0, E.PLANT]) > 0.9 and float(s1.inv[0, E.WOOD]) == 0.0
    s2, *_ = E.step(s1, jnp.array([E.SELECT_0 + E.WOOD]), cfg)
    s3, *_ = E.step(s2, jnp.array([E.GRASP]), cfg)
    assert int(s3.hand[0]) == E.WOOD and float(s3.inv[0, E.WOOD]) > 0.9


def test_hard_object_in_hand_strikes_harder():
    cfg, s, west, mid, east = _fire_world(n=2)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                   genotype=s.genotype._replace(strength=jnp.ones(2)))
    bare, *_ = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    armed = s._replace(inv=s.inv.at[0, E.STONE].set(1.0), hand=jnp.array([E.STONE, 0], jnp.int32))
    tool, *_ = E.step(armed, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    assert float(tool.health[1]) < float(bare.health[1])


def test_stone_struck_on_rock_with_wood_can_start_a_fire():
    cfg, s, west, mid, east = _fire_world(spark_prob=1.0, ignite_prob=1.0)
    s = s._replace(pos=jnp.asarray([mid], jnp.int32), terrain=s.terrain.at[east].set(E.ROCK),
                   wood=s.wood.at[east].set(1.0), inv=s.inv.at[0, E.STONE].set(1.0),
                   hand=jnp.array([E.STONE], jnp.int32))
    s2, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2]), cfg)
    assert float(s2.fire[east]) > 0 and int(info.ignited[0]) == 1
    # Same strike with an empty hand makes no spark
    bare = s._replace(inv=jnp.zeros_like(s.inv))
    s3, _, _, info = E.step(bare, jnp.array([E.STRIKE_N + 2]), cfg)
    assert float(s3.fire[east]) == 0 and int(info.ignited[0]) == 0


def test_heat_cooks_meat_and_cooked_meat_carries_no_infection():
    cfg, s, west, mid, east = _fire_world(raw_meat_infection=1.0)
    s = s._replace(pos=jnp.asarray([west], jnp.int32), fire=s.fire.at[east].set(500.0),
                   meat=s.meat.at[mid].set(1.0))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    for _ in range(20):
        s, *_ = step(s, jnp.array([E.REST]))
    assert float(s.cooked[mid]) > 0.5 and float(s.meat[mid]) < 0.5
    eat = lambda mat: s._replace(fire=jnp.zeros_like(s.fire), inv=jnp.zeros_like(s.inv).at[0, mat].set(1.0),
                                 hand=jnp.array([mat], jnp.int32), immunity=jnp.zeros(1),
                                 genotype=s.genotype._replace(disease_resistance=jnp.ones(1)))
    _, _, _, info_c = E.step(eat(E.COOKED_MEAT), jnp.array([E.EAT]), cfg)
    _, _, _, info_r = E.step(eat(E.RAW_MEAT), jnp.array([E.EAT]), cfg)
    assert bool(info_c.ate_cooked[0]) and not bool(info_c.newly_infected[0])
    assert bool(info_r.newly_infected[0])


def test_fire_warms_burns_and_keeps_predators_off():
    cfg, s, west, mid, east = _fire_world(predator_attack_prob=1.0)
    pred = _place_animal(s.predators, 0, west, cfg.predator_hp)
    base = s._replace(pos=jnp.asarray([mid], jnp.int32), predators=pred)
    _, _, _, cold = E.step(base, jnp.array([E.REST]), cfg)
    lit = base._replace(fire=base.fire.at[east].set(500.0))
    _, _, _, warm = E.step(lit, jnp.array([E.REST]), cfg)
    assert float(warm.ambient_temp[0]) > float(cold.ambient_temp[0]) + 10
    assert bool(cold.attacked_by_predator[0]) and not bool(warm.attacked_by_predator[0])
    inside = lit._replace(pos=jnp.asarray([east], jnp.int32), predators=s.predators)
    s2, _, _, info = E.step(inside, jnp.array([E.REST]), cfg)
    assert bool(info.burned[0]) and float(s2.health[0]) < 1.0


def test_burning_wood_carries_fire_to_a_new_place():
    cfg, s, west, mid, east = _fire_world()
    s = s._replace(pos=jnp.asarray([mid], jnp.int32), fire=s.fire.at[mid].set(500.0),
                   hand=jnp.array([E.WOOD], jnp.int32))
    s1, _, _, info = E.step(s, jnp.array([E.GRASP]), cfg)
    assert bool(info.torch_lit[0]) and int(s1.torch[0]) > 0 and float(s1.inv[0, E.WOOD]) > 0.9
    s2, _, _, info = E.step(s1, jnp.array([E.PLACE_N + 2]), cfg)
    assert float(s2.fire[east]) > 0 and int(info.ignited[0]) == 2


# ------------------------------------------------------------ clock, daylight, darkness
def test_clock_stamps_every_tick_in_30_minute_steps():
    cfg = EnvConfig()
    assert E.clock(0, cfg)["stamp"] == "Y1 Spring D1 00:00"
    assert E.clock(27, cfg)["stamp"] == "Y1 Spring D1 13:30"
    assert E.clock(48, cfg)["stamp"] == "Y1 Spring D2 00:00"
    assert E.clock(480, cfg)["stamp"] == "Y1 Summer D1 00:00"
    assert E.clock(1920 + 47, cfg)["stamp"] == "Y2 Spring D1 23:30"


def test_daylight_depends_on_season():
    cfg = EnvConfig()
    summer_5am = 480 + 4 * 48 + 10            # Summer day 5, 05:00
    winter_5am = 1440 + 4 * 48 + 10           # Winter day 5, 05:00
    assert not bool(E.is_night(jnp.int32(summer_5am), cfg))
    assert bool(E.is_night(jnp.int32(winter_5am), cfg))
    assert bool(E.is_night(jnp.int32(summer_5am - 2), cfg))      # 04:00 in summer is still dark
    assert not bool(E.is_night(jnp.int32(winter_5am + 14), cfg))  # noon in winter is light


def test_warmest_mid_afternoon_coldest_before_dawn():
    cfg = EnvConfig()
    temp = lambda t: float(E.ambient_temperature(jnp.int32(t), jnp.int32(E.CLEAR), jnp.int32(E.GRASS), cfg))
    assert temp(30) > temp(24) > temp(6)       # 15:00 > 12:00 > 03:00
    assert abs(temp(30) - temp(6) - 2 * cfg.day_temp_swing) < 1e-4


def test_darkness_hides_distant_things_unless_fire_lights_them():
    cfg, s, _ = _quiet_world(n=1)
    s = s._replace(pos=jnp.asarray([[6, 2]], jnp.int32),
                   small=_place_animal(s.small, 0, (6, 6), cfg.small_prey_hp))
    v, r, c = cfg.view_size, cfg.view_radius, E.N_OBS_CHANNELS
    prey_at = lambda st: np.asarray(E.observe(st, cfg))[0, :v * v * c].reshape(v, v, c)[r, r + 4, E.N_TERRAIN + 3]
    noon, midnight = s._replace(t=jnp.int32(24)), s._replace(t=jnp.int32(0))
    assert prey_at(noon) == 1.0 and prey_at(midnight) == 0.0
    lit = midnight._replace(fire=midnight.fire.at[6, 7].set(100.0))
    assert prey_at(lit) == 1.0
    terrain_mid = np.asarray(E.observe(midnight, cfg))[0, :v * v * c].reshape(v, v, c)[r, r + 4, :E.N_TERRAIN]
    assert terrain_mid.sum() == 1.0            # terrain shapes stay visible at night


def test_events_carry_timestamps_and_resolution_phase():
    from lifesim.recorder import PHASES
    cfg = EnvConfig(grid_size=16, n_agents=3, max_steps=200)
    rec = record_episode(random_policy, None, seed=5, cfg=cfg, snapshot_every=100)
    assert rec.events and all("time" in ev and "phase" in ev for ev in rec.events)
    keys = [(ev["t"], PHASES.index(ev["phase"])) for ev in rec.events]
    assert keys == sorted(keys)
    assert rec.events[0]["time"].startswith("Y1 Spring D1")


# ------------------------------------------------------------ moral sentiments
def _moral_world(n, **traits):
    """All moral traits zero except those given, so each test isolates one sentiment."""
    cfg, s, _ = _quiet_world(n=n)
    zero = {k: jnp.zeros(n) for k in ["empathy", "kin_care", "envy", "guilt", "reciprocity", "vengefulness"]}
    zero.update({k: jnp.asarray(v, jnp.float32) for k, v in traits.items()})
    g = s.genotype._replace(**zero)
    return cfg, s._replace(genotype=g)


def test_empathy_makes_witnessing_harm_costly():
    for emp, sign in [(0.0, 0), (1.0, -1)]:
        cfg, s = _moral_world(3, empathy=[0.0, 0.0, emp])
        west, mid, east = _neighbor_pair(s.terrain)
        s = s._replace(pos=jnp.asarray([west, mid, east], jnp.int32))
        _, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST, E.REST]), cfg)
        m = float(info.moral_reward[2])
        assert (m < -0.05) if sign else (m == 0.0)


def test_envy_and_guilt_respond_to_who_is_better_fed():
    cfg, s = _moral_world(2, envy=[1.0, 1.0], guilt=[1.0, 1.0])
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32), satiety=jnp.array([0.2, 0.9]))
    _, _, _, info = E.step(s, jnp.array([E.REST, E.REST]), cfg)
    assert float(info.moral_reward[0]) < 0 and float(info.moral_reward[1]) < 0
    cfg, s2 = _moral_world(2, envy=[1.0, 1.0], guilt=[0.0, 0.0])
    s2 = s2._replace(pos=jnp.asarray([west, mid], jnp.int32), satiety=jnp.array([0.2, 0.9]))
    _, _, _, info = E.step(s2, jnp.array([E.REST, E.REST]), cfg)
    assert float(info.moral_reward[0]) < 0 and float(info.moral_reward[1]) == 0.0


def test_revenge_and_reciprocity_reward_answering_past_acts():
    cfg, s = _moral_world(2, vengefulness=[1.0, 1.0], reciprocity=[1.0, 1.0])
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    struck_before = s._replace(memory=s.memory.at[0, 1, E.MEM_STRUCK_ME].set(2.0))
    _, _, _, info = E.step(struck_before, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    _, _, _, info_fresh = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    assert float(info.moral_reward[0]) > float(info_fresh.moral_reward[0])
    # Repaying: agent 1 once gave to agent 0; agent 0 drops food and agent 1 takes it
    gave_before = s._replace(memory=s.memory.at[0, 1, E.MEM_GAVE_NEAR_ME].set(2.0),
                             inv=s.inv.at[0, E.PLANT].set(2.0), hand=jnp.array([E.PLANT, E.PLANT], jnp.int32),
                             plants=jnp.zeros_like(s.plants), meat=jnp.zeros_like(s.meat))
    s1, _, _, info = E.step(gave_before, jnp.array([E.RELEASE, E.REST]), cfg)
    assert float(info.moral_reward[0]) == 0.0                # dropping alone pays nothing
    s1 = s1._replace(pos=jnp.asarray([west, west], jnp.int32))
    s2, _, _, info = E.step(s1, jnp.array([E.REST, E.GRASP]), cfg)
    assert np.isclose(float(info.moral_reward[0]), cfg.reciprocity_reward)
    # Repaying again right away still counts, but feels weaker (habituation)
    s3, *_ = E.step(s2._replace(inv=s2.inv.at[0, E.PLANT].set(1.0)), jnp.array([E.RELEASE, E.REST]), cfg)
    _, _, _, info = E.step(s3, jnp.array([E.REST, E.GRASP]), cfg)
    second = float(info.moral_reward[0])
    assert bool(info.gifts[0, 1]) and 0.0 < second < 0.5 * cfg.reciprocity_reward


def test_real_365_day_year_is_one_setting():
    cfg = EnvConfig(days_per_year=365)
    assert cfg.year_length == 17520 and cfg.season_starts == (92, 183, 274)
    assert E.clock(0, cfg)["stamp"] == "Y1 Spring D1 00:00"
    assert E.clock(91 * 48 + 47, cfg)["stamp"] == "Y1 Spring D92 23:30"
    assert E.clock(92 * 48, cfg)["stamp"] == "Y1 Summer D1 00:00"
    assert E.clock(274 * 48, cfg)["stamp"] == "Y1 Winter D1 00:00"
    assert E.clock(364 * 48 + 47, cfg)["stamp"] == "Y1 Winter D91 23:30"
    assert E.clock(365 * 48, cfg)["stamp"] == "Y2 Spring D1 00:00"
    for t in [0, 92 * 48 - 1, 92 * 48, 183 * 48, 274 * 48, 364 * 48 + 47]:   # clock and physics agree
        assert E.SEASON_NAMES[int(E.season_of(jnp.int32(t), cfg))] == E.clock(t, cfg)["season"]
    midsummer, midwinter = int(cfg.solstice_frac * 17520), int((cfg.solstice_frac + 0.5) * 17520)
    assert float(E.daylight_hours(jnp.int32(midsummer), cfg)) > 14.9
    assert float(E.daylight_hours(jnp.int32(midwinter), cfg)) < 9.1


def test_deterrence_is_counted_for_the_protected_agent():
    cfg, s, _ = _quiet_world(n=2, predator_attack_prob=1.0)
    west, mid, east = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([mid, east], jnp.int32),
                   predators=_place_animal(s.predators, 0, west, cfg.predator_hp))
    _, _, _, info = E.step(s, jnp.array([E.REST, E.REST]), cfg)
    assert bool(info.deterred[0]) and not bool(info.attacked_by_predator[0])
    assert bool(info.huddling[0]) and bool(info.huddling[1])



# ------------------------------------------------------------ fixes from the playback review
def test_founders_start_together_as_a_band():
    cfg = EnvConfig(grid_size=32, n_agents=8)
    for seed in range(5):
        s = E.reset(jax.random.PRNGKey(seed), cfg)
        p = np.asarray(s.pos)
        assert np.max(np.abs(p[:, None, :] - p[None, :, :]).max(-1)) <= 2 * cfg.founder_spread
        assert not np.any(np.asarray(s.terrain)[p[:, 0], p[:, 1]] == E.WATER)
    scattered = E.reset(jax.random.PRNGKey(0), EnvConfig(grid_size=32, n_agents=8, founder_spread=0))
    q = np.asarray(scattered.pos)
    assert np.max(np.abs(q[:, None, :] - q[None, :, :]).max(-1)) > 2 * cfg.founder_spread


def test_useless_actions_are_masked_but_affordances_stay():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, east = _neighbor_pair(s.terrain)
    far = tuple(np.argwhere(np.asarray(s.terrain) != E.WATER)[-1])
    s = s._replace(pos=jnp.asarray([mid, far], jnp.int32), stones=jnp.zeros_like(s.stones),
                   inv=jnp.zeros_like(s.inv), wood=jnp.zeros_like(s.wood), plants=jnp.zeros_like(s.plants),
                   meat=jnp.zeros_like(s.meat), cooked=jnp.zeros_like(s.cooked),
                   terrain=s.terrain.at[east].set(E.GRASS).at[mid].set(E.GRASS))
    m = np.asarray(E.action_mask(s, cfg))[0]
    assert not m[E.TOUCH_N:E.TOUCH_N + 8].any()                 # nobody to touch
    assert not m[E.STRIKE_N + 2]                                # empty grass to the east
    assert not m[E.SELECT_0 + E.COOKED_MEAT]                    # nothing cooked anywhere
    near = s._replace(pos=jnp.asarray([mid, east], jnp.int32))
    m2 = np.asarray(E.action_mask(near, cfg))[0]
    assert m2[E.TOUCH_N + 2] and m2[E.STRIKE_N + 2]             # someone is there now
    rock = s._replace(terrain=s.terrain.at[east].set(E.ROCK))
    assert np.asarray(E.action_mask(rock, cfg))[0][E.STRIKE_N + 2]   # hard surface: can still spark
    carried = s._replace(inv=s.inv.at[0, E.STONE].set(1.0))
    assert np.asarray(E.action_mask(carried, cfg))[0][E.SELECT_0 + E.STONE]


def test_company_is_logged():
    cfg, s, _ = _quiet_world(n=3)
    west, mid, _ = _neighbor_pair(s.terrain)
    far = tuple(np.argwhere(np.asarray(s.terrain) != E.WATER)[-1])
    s = s._replace(pos=jnp.asarray([west, mid, far], jnp.int32))
    _, _, _, info = E.step(s, jnp.array([E.REST, E.REST, E.REST]), cfg)
    c = np.asarray(info.company)
    assert c[0] and c[1] and (not c[2] or max(abs(np.array(far) - np.array(mid))) <= cfg.view_radius)



def test_a_bare_handed_blow_hurts_but_does_not_maim():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                   genotype=s.genotype._replace(strength=jnp.ones(2)))
    s2, _, _, info = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    loss = 1.0 - float(s2.health[1])
    assert 0.1 < loss < 0.2 and bool(info.struck_by_agent[1])



def test_grasp_takes_what_is_there_and_holds_it():
    cfg, s, _ = _quiet_world(n=1)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([mid], jnp.int32), inv=jnp.zeros_like(s.inv),
                   plants=jnp.zeros_like(s.plants), wood=jnp.zeros_like(s.wood),
                   meat=jnp.zeros_like(s.meat).at[mid].set(2.0), hand=jnp.array([E.PLANT], jnp.int32))
    assert np.asarray(E.action_mask(s, cfg))[0][E.GRASP]          # nothing selected is here, hand is empty
    s1, _, _, info = E.step(s, jnp.array([E.GRASP]), cfg)
    assert int(s1.hand[0]) == E.RAW_MEAT and float(s1.inv[0, E.RAW_MEAT]) > 0.9
    s2, _, _, info = E.step(s1, jnp.array([E.EAT]), cfg)          # two steps to eat meat, not three
    assert bool(info.meat_eaten[0])
    # Holding something else, grasp still takes what is here and switches the hand to it
    full = s._replace(inv=s.inv.at[0, E.WOOD].set(1.0), hand=jnp.array([E.WOOD], jnp.int32))
    s3, *_ = E.step(full, jnp.array([E.GRASP]), cfg)
    assert int(s3.hand[0]) == E.RAW_MEAT and float(s3.inv[0, E.WOOD]) > 0.9



def test_witnessing_violence_needs_light():
    for t, expect in [(24, True), (0, False)]:                           # noon, midnight
        cfg, s, _ = _quiet_world(n=3)
        west, mid, _ = _neighbor_pair(s.terrain)
        land = np.argwhere(np.asarray(s.terrain) != E.WATER)
        far = next(tuple(p) for p in land if 3 <= max(abs(p[0] - mid[0]), abs(p[1] - mid[1])) <= cfg.view_radius)
        s = s._replace(t=jnp.int32(t), pos=jnp.asarray([west, mid, far], jnp.int32))
        s2, *_ = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST, E.REST]), cfg)
        assert (float(s2.memory[2, 0, E.MEM_SEEN_VIOLENT]) > 0.5) == expect



def test_scarcity_scales_starting_food_and_regrowth():
    rich, poor = EnvConfig(grid_size=24), EnvConfig(grid_size=24, plant_abundance=0.4)
    a, b = E.reset(jax.random.PRNGKey(0), rich), E.reset(jax.random.PRNGKey(0), poor)
    assert abs(float(b.plants.sum()) / float(a.plants.sum()) - 0.4) < 1e-3     # starting stock
    a = a._replace(plants=jnp.zeros_like(a.plants))
    b = b._replace(plants=jnp.zeros_like(b.plants))
    n = a.pos.shape[0]
    a2, *_ = E.step(a, jnp.zeros(n, jnp.int32), rich)
    b2, *_ = E.step(b, jnp.zeros(n, jnp.int32), poor)
    assert abs(float(b2.plants.sum()) / float(a2.plants.sum()) - 0.4) < 1e-3   # regrowth



def test_camp_starts_near_water():
    cfg = EnvConfig(grid_size=32, n_agents=6)
    worst = 0
    for seed in range(20):
        s = E.reset(jax.random.PRNGKey(seed), cfg)
        water = np.argwhere(np.asarray(s.terrain) == E.WATER)
        p = np.asarray(s.pos)
        d = min(np.max(np.abs(water - q), axis=1).min() for q in p)   # closest founder to water
        worst = max(worst, d)
    assert worst <= cfg.camp_water_radius + 1 + cfg.founder_spread



def test_repeated_feelings_fade_and_recover_but_acts_are_never_blocked():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                   genotype=s.genotype._replace(social_reward=jnp.ones(2)))
    touch = jnp.array([E.TOUCH_N + 2, E.REST])
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    felt = []
    rest = jnp.array([E.REST, E.REST])
    for _ in range(48):                                  # touch every tick for a whole day
        s = s._replace(hydration=jnp.ones(2), energy=jnp.ones(2))
        _, r_rest, _, _ = step(s, rest)                  # same state, no touch: isolates the touch term
        s, r, _, info = step(s, touch)
        assert bool(info.touches[0, 1])                  # the act always happens
        felt.append(float(r[0] - r_rest[0]))
    assert np.isclose(felt[0], cfg.touch_reward, atol=1e-6)
    assert felt[1] < 0.5 * felt[0] and felt[10] < 0.01 * felt[0]       # it fades fast
    assert sum(felt) < 2.0 * cfg.touch_reward            # a day of nonstop touching ~ under 2 touches
    fed = lambda st: st._replace(satiety=jnp.ones(2), hydration=jnp.ones(2), energy=jnp.ones(2),
                                 health=jnp.ones(2))     # keep the test agents alive

    def touch_feeling(st):
        _, r_rest, _, _ = step(st, rest)
        st2, r, _, _ = step(st, touch)
        return st2, float(r[0] - r_rest[0])

    def apart(st, ticks):
        for _ in range(ticks):
            st, *_ = step(fed(st), rest)
        return fed(st)

    # A day of nonstop touching builds tolerance that lasts: after 2 days apart it is still faint
    assert touch_feeling(apart(s, 2 * cfg.day_length))[1] < 0.1 * cfg.touch_reward
    # An ordinary act fades and returns within a day
    _, cfg_s, _ = _quiet_world(n=2)
    t = cfg_s._replace(pos=jnp.asarray([west, mid], jnp.int32),
                       genotype=cfg_s.genotype._replace(social_reward=jnp.ones(2)))
    t, first = touch_feeling(fed(t))
    _, again = touch_feeling(fed(t))
    t = apart(t, cfg.day_length)
    _, later = touch_feeling(t)
    assert again < 0.5 * first and later > 0.5 * first


# ------------------------------------------------------------ survival reward mode
def _survival_world(n=2, **traits):
    cfg, s, _ = _quiet_world(n=n, reward_mode="survival")
    g = s.genotype._replace(**{k: jnp.asarray(v, jnp.float32) for k, v in traits.items()})
    return cfg, s._replace(genotype=g, pain_aversion=g.pain_aversion)


def test_survival_mode_pays_nothing_for_acts():
    cfg, s = _survival_world(social_reward=[1.0, 1.0], novelty=[1.0, 1.0], empathy=[0.0, 0.0],
                             kin_care=[0.0, 0.0], reciprocity=[0.0, 0.0], vengefulness=[0.0, 0.0])
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    # Compare touching with another non-resting act (selecting what one holds), so fatigue is equal
    s = s._replace(inv=s.inv.at[:, E.PLANT].set(1.0), hand=jnp.array([E.PLANT, E.PLANT], jnp.int32))
    _, r_other, _, _ = E.step(s, jnp.array([E.SELECT_0 + E.PLANT, E.REST]), cfg)
    _, r_touch, _, info = E.step(s, jnp.array([E.TOUCH_N + 2, E.REST]), cfg)
    assert bool(info.touches[0, 1])
    assert np.isclose(float(r_touch[0]), float(r_other[0]), atol=1e-6)      # toucher: nothing extra
    assert np.isclose(float(r_touch[1]), float(r_other[1]), atol=1e-6)      # touched: nothing extra


def test_survival_mode_cannot_be_farmed_by_juggling_food():
    cfg, s = _survival_world(empathy=[1.0, 1.0], reciprocity=[1.0, 1.0])
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, west], jnp.int32), plants=jnp.zeros_like(s.plants),
                   meat=jnp.zeros_like(s.meat), memory=s.memory.at[:, :, E.MEM_GAVE_NEAR_ME].set(2.0),
                   inv=jnp.zeros_like(s.inv).at[:, E.PLANT].set(1.0), hand=jnp.array([E.PLANT, E.PLANT], jnp.int32))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    g = cfg.shaping_gamma
    juggle, idle, t = 0.0, 0.0, s
    for k in range(40):                                  # pass one plant back and forth
        a = [E.RELEASE, E.REST] if k % 2 == 0 else [E.REST, E.GRASP]
        if k % 4 >= 2:
            a = a[::-1]
        t, r, _, _ = step(t, jnp.array(a))
        juggle += g ** k * float(r.sum())                # the discounted return agents optimize
    hold = E.SELECT_0 + E.PLANT                          # same rhythm of acts, but no food changes hands
    for k in range(40):
        a = [hold, E.REST] if k % 2 == 0 else [E.REST, hold]
        if k % 4 >= 2:
            a = a[::-1]
        s, r, _, _ = step(s, jnp.array(a))
        idle += g ** k * float(r.sum())
    # Felt changes telescope, so the return depends only on how well off agents end up. Bodies end
    # the same either way, so passing the same food back and forth earns nothing extra.
    assert abs(juggle - idle) < 1e-3


def test_survival_mode_feels_body_needs_and_others_fates():
    cfg, s = _survival_world(empathy=[1.0, 0.0], kin_care=[0.0, 0.0], vengefulness=[0.0, 1.0])
    west, mid, _ = _neighbor_pair(s.terrain)
    water_next = None
    t = np.asarray(s.terrain)
    for r in range(1, t.shape[0] - 1):
        for c in range(1, t.shape[1] - 1):
            if t[r, c] != E.WATER and (t[r - 1:r + 2, c - 1:c + 2] == E.WATER).any():
                water_next = (r, c)
                break
        if water_next:
            break
    thirsty = s._replace(pos=jnp.asarray([water_next, water_next], jnp.int32), hydration=jnp.full(2, 0.2))
    _, r_drink, _, _ = E.step(thirsty, jnp.array([E.DRINK, E.REST]), cfg)
    _, r_rest, _, _ = E.step(thirsty, jnp.array([E.REST, E.REST]), cfg)
    assert float(r_drink[0]) > float(r_rest[0]) + 0.01   # quenching thirst feels good: it raises survival
    near = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    _, r_hit, _, _ = E.step(near, jnp.array([E.REST, E.STRIKE_N + 6]), cfg)    # 1 strikes 0
    _, r_calm, _, _ = E.step(near, jnp.array([E.REST, E.REST]), cfg)
    assert float(r_hit[0]) < float(r_calm[0])            # being hurt is felt
    feud = near._replace(memory=near.memory.at[1, 0, E.MEM_STRUCK_ME].set(2.0))
    _, r_feud, _, _ = E.step(feud, jnp.array([E.REST, E.STRIKE_N + 6]), cfg)
    _, r_feud_calm, _, _ = E.step(feud, jnp.array([E.REST, E.REST]), cfg)
    assert float(r_feud[1]) > float(r_feud_calm[1])      # a vengeful agent is satisfied by an enemy's harm
    empath = near._replace(pos=jnp.asarray([west, mid], jnp.int32))
    _, r_seen, _, _ = E.step(empath, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)  # 0 strikes 1
    _, r_none, _, _ = E.step(empath, jnp.array([E.REST, E.REST]), cfg)
    assert float(r_seen[0]) < float(r_none[0])           # the empathic striker feels the victim's harm


# ------------------------------------------------------------ company: loneliness, buffering, healing
def _company_world(n_agents, together, sociability=1.0, ticks=400, health=0.5):
    cfg, s, _ = _quiet_world(n=n_agents)
    t = np.asarray(s.terrain)
    land = np.argwhere(t != E.WATER)
    anchor = tuple(land[len(land) // 2])
    if together:
        cells = [anchor] * n_agents
    else:   # spread out of each other's sight
        cells, i = [], 0
        for p in land:
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > cfg.view_radius for q in cells):
                cells.append(tuple(p))
            if len(cells) == n_agents:
                break
        assert len(cells) == n_agents
    s = s._replace(pos=jnp.asarray(cells, jnp.int32), health=jnp.full(n_agents, health),
                   genotype=s.genotype._replace(social_reward=jnp.full(n_agents, sociability)))
    step = jax.jit(lambda st: E.step(st, jnp.full(n_agents, E.REST), cfg))
    for _ in range(ticks):
        s = s._replace(satiety=jnp.ones(n_agents), hydration=jnp.ones(n_agents), energy=jnp.ones(n_agents))
        s, *_ = step(s)
    return cfg, s


def test_loneliness_stresses_company_calms_first_companion_matters_most():
    cfg, alone = _company_world(1, together=False)
    _, pair = _company_world(2, together=True)
    _, four = _company_world(4, together=True)
    a, p, f = float(alone.stress[0]), float(pair.stress[0]), float(four.stress[0])
    assert 0.25 < a < cfg.stress_heal_block              # an average agent alone: stressed, still heals
    assert p < 0.5 * a and f < p                          # company calms; more helps
    assert (a - p) > 2 * (p - f)                          # the first companion matters most


def test_lonely_sociable_agents_heal_slower_but_loneliness_alone_never_kills():
    _, alone = _company_world(1, together=False, sociability=1.5)
    _, pair = _company_world(2, together=True, sociability=1.5)
    assert float(pair.health[0]) > float(alone.health[0]) + 0.05     # company speeds healing
    _, very = _company_world(1, together=False, sociability=3.0, ticks=1500, health=1.0)
    assert bool(very.alive[0]) and float(very.health[0]) > 0.99      # isolation alone does no damage


# ------------------------------------------------------------ impressions: you feel what you perceive
def _impression_world():
    """Observer 0 and friend 1 at noon; empathy only, so feelings are easy to read."""
    cfg, s = _survival_world(empathy=[1.0, 1.0], kin_care=[0.0, 0.0], envy=[0.0, 0.0], guilt=[0.0, 0.0],
                             reciprocity=[0.0, 0.0], vengefulness=[0.0, 0.0])
    s = s._replace(t=jnp.int32(24), genotype=s.genotype._replace(social_reward=jnp.zeros(2)))
    land = np.argwhere(np.asarray(s.terrain) != E.WATER)
    friend = tuple(land[len(land) // 2])
    in_view = next(tuple(p) for p in land if 2 <= max(abs(p[0] - friend[0]), abs(p[1] - friend[1])) <= 4)
    away = next(tuple(p) for p in land[::-1] if max(abs(p[0] - friend[0]), abs(p[1] - friend[1])) > cfg.view_radius)
    fed = lambda st: st._replace(satiety=jnp.ones(2), hydration=jnp.ones(2), energy=jnp.ones(2))
    step = jax.jit(lambda st: E.step(fed(st), jnp.array([E.REST, E.REST]), cfg))
    place = lambda st, obs: st._replace(pos=jnp.asarray([obs, friend], jnp.int32))
    return cfg, s, step, place, in_view, away


def test_harm_out_of_sight_is_felt_on_the_next_sighting():
    cfg, s, step, place, in_view, away = _impression_world()
    s = place(s, away)._replace(health=jnp.array([1.0, 0.4]))         # hurt while the observer is away
    s, _, _, info = step(s)
    assert float(info.moral_reward[0]) == 0.0                           # unseen: nothing felt
    before = float(s.impression[0, 1])
    s, _, _, info = step(place(s, in_view))                             # comes back into view: wounds show
    felt = float(info.moral_reward[0])
    assert felt < -0.1
    assert np.isclose(felt, cfg.care_empathy * (float(s.impression[0, 1]) - before), atol=1e-5)


def test_harm_healed_unseen_is_never_felt():
    cfg, s, step, place, in_view, away = _impression_world()
    s = place(s, away)._replace(health=jnp.array([1.0, 0.4]))
    s, *_ = step(s)
    s = s._replace(health=jnp.ones(2))                                  # healed before the next sighting
    s, _, _, info = step(place(s, in_view))
    assert abs(float(info.moral_reward[0])) < 1e-3


def test_hunger_and_stress_cannot_be_seen():
    cfg, s, step, place, in_view, away = _impression_world()
    s = place(s, in_view)
    s, *_ = step(s)
    hungry = s._replace(stress=jnp.array([0.0, 0.9]))
    _, _, _, info = E.step(hungry._replace(satiety=jnp.array([1.0, 0.05]), hydration=jnp.ones(2), energy=jnp.ones(2)),
                           jnp.array([E.REST, E.REST]), cfg)
    assert abs(float(info.moral_reward[0])) < 1e-3                      # inner states don't show


def test_a_death_is_felt_on_seeing_the_body():
    cfg, s, step, place, in_view, away = _impression_world()
    s = place(s, away)._replace(alive=jnp.array([True, False]), health=jnp.array([1.0, 0.0]))
    s, _, _, info = step(s)
    assert float(info.moral_reward[0]) == 0.0                           # died out of sight: unknown
    obs_before = E.observe(place(s, in_view), cfg)
    s, _, _, info = step(place(s, in_view))
    assert float(info.moral_reward[0]) < -cfg.care_empathy * cfg.death_penalty   # grief on finding the body
    v, r, c = cfg.view_size, cfg.view_radius, E.N_OBS_CHANNELS
    win = np.asarray(obs_before[0, :v * v * c]).reshape(v, v, c)
    assert win[..., E.CH_BODIES].sum() == 1.0                      # and the body is visible
    s, _, _, info = step(s)
    assert abs(float(info.moral_reward[0])) < 1e-6                      # felt once, not every tick


def test_a_healthy_friend_in_view_earns_nothing_by_presence():
    cfg, s, step, place, in_view, away = _impression_world()
    s = place(s, in_view)
    felt = 0.0
    for _ in range(100):
        s, _, _, info = step(s)
        felt += float(info.moral_reward[0])
    assert abs(felt) < 1e-3


# ------------------------------------------------------------ scars and lasting injuries
def _healing_run(injury, ticks=1200):
    cfg, s, _ = _quiet_world(n=1)
    s = s._replace(injury=jnp.array([injury]))
    step = jax.jit(lambda st: E.step(st._replace(satiety=jnp.ones(1), hydration=jnp.ones(1), energy=jnp.ones(1),
                                                 stress=jnp.zeros(1)), jnp.array([E.REST]), cfg))
    for _ in range(ticks):
        s, *_ = step(s)
    return cfg, s


def test_minor_injuries_heal_severe_ones_leave_a_lasting_part():
    cfg, minor = _healing_run(0.3)
    assert float(minor.injury[0]) < 1e-3 and float(minor.lasting_injury[0]) == 0.0
    cfg, severe = _healing_run(0.9)
    lasting = cfg.lasting_share * (0.9 - cfg.severe_injury)
    assert np.isclose(float(severe.lasting_injury[0]), lasting, atol=1e-5)
    assert np.isclose(float(severe.injury[0]), lasting, atol=1e-5)        # heals down to the floor, no further
    assert bool(severe.alive[0]) and float(severe.health[0]) > 0.99       # a lasting injury never bleeds out
    cfg, worst = _healing_run(1.0, ticks=10)
    assert float(worst.lasting_injury[0]) <= cfg.max_lasting_injury < 0.5


def test_a_lasting_injury_weakens_strikes_for_life():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32), genotype=s.genotype._replace(strength=jnp.ones(2)))
    hit = jnp.array([E.STRIKE_N + 2, E.REST])
    s_ok, *_ = E.step(s, hit, cfg)
    lame = s._replace(injury=jnp.array([0.3, 0.0]), lasting_injury=jnp.array([0.3, 0.0]))
    s_lame, *_ = E.step(lame, hit, cfg)
    assert float(s_lame.health[1]) > float(s_ok.health[1])               # the blow lands softer


def test_wounds_leave_scars_hunger_does_not_and_scars_never_fade():
    cfg, s, _ = _quiet_world(n=2)
    west, mid, _ = _neighbor_pair(s.terrain)
    s = s._replace(pos=jnp.asarray([west, mid], jnp.int32))
    for _ in range(3):
        s, *_ = E.step(s, jnp.array([E.STRIKE_N + 2, E.REST]), cfg)
    scarred = float(s.scars[1])
    assert scarred > 0.1 and float(s.scars[0]) == 0.0                     # the struck one is marked
    starving = s._replace(satiety=jnp.zeros(2), hydration=jnp.ones(2))
    starving, *_ = E.step(starving, jnp.array([E.REST, E.REST]), cfg)
    assert float(starving.scars[0]) == 0.0                                # hunger hurts but leaves no scar
    t = s
    step = jax.jit(lambda st: E.step(st._replace(satiety=jnp.ones(2), hydration=jnp.ones(2), energy=jnp.ones(2)),
                                     jnp.array([E.REST, E.REST]), cfg))
    for _ in range(500):
        t, *_ = step(t)
    assert float(t.health[1]) > 0.99 and np.isclose(float(t.scars[1]), scarred)   # healed, still scarred


def test_scars_are_seen_and_felt_after_a_wound_healed_out_of_sight():
    cfg, s, step, place, in_view, away = _impression_world()
    s = place(s, away)._replace(scars=jnp.array([0.0, 0.4]))             # healed while away, but marked
    s, _, _, info = step(s)
    assert float(info.moral_reward[0]) == 0.0
    s, _, _, info = step(place(s, in_view))
    assert float(info.moral_reward[0]) < -0.01                            # the scars tell
    v, c = cfg.view_size, E.N_OBS_CHANNELS
    win = np.asarray(E.observe(s, cfg)[0, :v * v * c]).reshape(v, v, c)
    base = c - E.N_SOCIAL_FEATURES + E.N_MEM + E.N_APPEARANCE + 1
    assert np.isclose(win[..., base + 2].max(), 0.4)                      # visible in the observation


# ------------------------------------------------------------ cliffs and falls
def _cliff_world(**kw):
    cfg, s, _ = _quiet_world(n=1, **kw)
    t = np.asarray(s.terrain)
    spot = next((r, c) for r in range(1, t.shape[0] - 1) for c in range(1, t.shape[1] - 2)
                if t[r, c] not in (E.WATER, E.ROCK))
    dest = (spot[0], spot[1] + 1)
    s = s._replace(pos=jnp.asarray([spot], jnp.int32), terrain=s.terrain.at[dest].set(E.ROCK),
                   genotype=s.genotype._replace(speed=jnp.ones(1)), t=jnp.int32(24))
    return cfg, s, dest


def _fall_outcomes(cfg, s, dest, cliff, n=2000, night=False, injury=0.0):
    def one(key):
        st = s._replace(key=key, cliff=s.cliff.at[dest].set(cliff), injury=jnp.array([injury]),
                        t=jnp.int32(0 if night else 24))
        st2, _, _, info = E.step(st, jnp.array([E.MOVE_N + 2]), cfg)
        return st2.alive[0], st2.lasting_injury[0], info.death_cause[0], st2.health[0], st2.injury[0], st2.scars[0]
    return [np.asarray(x) for x in jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(3), n))]


def test_cliffs_come_from_elevation_and_are_visible_in_the_dark():
    cfg = EnvConfig(grid_size=32, n_agents=2)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    cl, t = np.asarray(s.cliff), np.asarray(s.terrain)
    assert np.isclose(cl.max(), 1.0) and cl.min() >= 0.0
    assert cl[t == E.ROCK].mean() > 1.5 * cl[t == E.GRASS].mean()        # rock edges are the steep ones
    assert cl[t == E.GRASS].mean() < 0.2                                 # ordinary ground is gentle
    rock = tuple(np.argwhere((t == E.ROCK) & (cl > 0.3))[0])
    dark = s._replace(t=jnp.int32(0), pos=s.pos.at[0].set(jnp.asarray(rock)))
    v, r, c = cfg.view_size, cfg.view_radius, E.N_OBS_CHANNELS
    win = np.asarray(E.observe(dark, cfg)[0, :v * v * c]).reshape(v, v, c)
    assert np.isclose(win[r, r, E.CH_CLIFF], cl[rock])                  # seen even at midnight


def test_falls_are_mostly_minor_sometimes_lasting_rarely_fatal_and_worse_from_high_cliffs():
    cfg, s, dest = _cliff_world(fall_prob=100.0)                         # force the fall
    flat = _fall_outcomes(cfg, s, dest, 0.0)
    high = _fall_outcomes(cfg, s, dest, 1.0)
    assert flat[0].all()                                                 # nobody dies falling on flat rock
    assert 0.03 < 1 - high[0].mean() < 0.15                              # a few die from high cliffs
    dead = ~high[0]
    assert (high[2][dead] == E.CAUSE_FALL).all()                         # and the cause is the fall
    survived = high[0]
    assert 0.25 < (high[1][survived] > 0).mean() < 0.6                   # many are hurt for life
    assert (flat[4] < 0.1).mean() > 0.5                                  # most flat-rock falls are minor


def test_a_fall_hurts_injures_and_scars_in_proportion():
    cfg, s, dest = _cliff_world(fall_prob=100.0)
    alive, lasting, cause, health, injury, scars = _fall_outcomes(cfg, s, dest, 0.5, n=200)
    ok = alive & (injury < 0.99) & (health < 0.99)                       # skip capped cases
    lost = 1.0 - health[ok]                                              # (the tick's healing shifts this ~0.002)
    assert (lost > 0).all()
    assert np.allclose(injury[ok], np.maximum(cfg.fall_injury_ratio * lost, lasting[ok]), atol=0.01)
    assert np.allclose(scars[ok], cfg.scar_per_wound * lost, atol=0.005)  # scars by health lost, like any wound


def test_falls_are_likelier_at_high_edges_in_the_dark_and_when_injured():
    cfg, s, dest = _cliff_world(injury_move_fail=0.0)                    # everyone reaches the rock
    rate = lambda **kw: float((_fall_outcomes(cfg, s, dest, n=4000, **kw)[4] > kw.get("injury", 0.0) + 1e-6).mean())
    low, high = rate(cliff=0.0), rate(cliff=1.0)
    assert high > 3 * low > 0
    assert rate(cliff=1.0, night=True) > 1.5 * high
    assert rate(cliff=1.0, injury=0.3) > 1.1 * high


# ------------------------------------------------------------ height (2.5D)
def test_height_orders_the_terrain_water_low_rock_high():
    cfg = EnvConfig(grid_size=32, n_agents=2)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    h, t = np.asarray(s.height), np.asarray(s.terrain)
    assert h.min() == 0.0 and np.isclose(h.max(), 1.0)
    assert h[t == E.WATER].max() <= h[t == E.GRASS].mean() < h[t == E.ROCK].min()


def test_climbing_costs_energy_and_high_ground_is_colder():
    cfg, s, _ = _quiet_world(n=1)
    t = np.asarray(s.terrain)
    spot = next((r, c) for r in range(1, t.shape[0] - 1) for c in range(1, t.shape[1] - 2)
                if t[r, c] == E.GRASS and t[r, c + 1] == E.GRASS)
    east = (spot[0], spot[1] + 1)
    s = s._replace(pos=jnp.asarray([spot], jnp.int32), genotype=s.genotype._replace(speed=jnp.ones(1)),
                   cliff=jnp.zeros_like(s.cliff))
    flat = s._replace(height=jnp.zeros_like(s.height))
    uphill = flat._replace(height=flat.height.at[east].set(0.5).at[0, 0].set(1.0))
    downhill = flat._replace(height=flat.height.at[spot].set(0.5).at[0, 0].set(1.0))
    move = jnp.array([E.MOVE_N + 2])
    e_flat = float(E.step(flat, move, cfg)[0].energy[0])
    e_up = float(E.step(uphill, move, cfg)[0].energy[0])
    e_down = float(E.step(downhill, move, cfg)[0].energy[0])
    assert e_up < e_flat and np.isclose(e_down, e_flat)                 # climbing costs, descending does not
    low = float(E.ambient_temperature(jnp.int32(24), jnp.int32(E.CLEAR), jnp.int32(E.GRASS), cfg, 0.0))
    high = float(E.ambient_temperature(jnp.int32(24), jnp.int32(E.CLEAR), jnp.int32(E.GRASS), cfg, 1.0))
    assert np.isclose(low - high, cfg.altitude_cold)


def test_agents_see_ground_height_relative_to_their_own_even_at_night():
    cfg = EnvConfig(grid_size=32, n_agents=2)
    s = E.reset(jax.random.PRNGKey(0), cfg)._replace(t=jnp.int32(0))
    v, r, c = cfg.view_size, cfg.view_radius, E.N_OBS_CHANNELS
    win = np.asarray(E.observe(s, cfg)[0, :v * v * c]).reshape(v, v, c)
    p = np.asarray(s.pos[0])
    h = np.asarray(s.height)
    assert win[r, r, E.CH_HEIGHT] == 0.0                                 # own cell is the reference
    q = (min(p[0] + 1, cfg.grid_size - 1), p[1])
    assert np.isclose(win[r + (q[0] - p[0]), r, E.CH_HEIGHT], h[q] - h[tuple(p)])


def test_steep_banks_can_trip_but_flat_ground_never_does():
    cfg, s, dest = _cliff_world(fall_prob=100.0)
    grass = s._replace(terrain=s.terrain.at[dest].set(E.GRASS))
    def falls(st, cliff):
        def one(key):
            st2, *_ = E.step(st._replace(key=key, cliff=st.cliff.at[dest].set(cliff)), jnp.array([E.MOVE_N + 2]), cfg)
            return st2.injury[0]
        return np.asarray(jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(5), 500)))
    assert (falls(grass, 0.0) == 0.0).all()                              # flat grass: no falls at all
    assert (falls(grass, 0.8) > 0.0).mean() > 0.9                        # a steep bank: falls happen off rock too
