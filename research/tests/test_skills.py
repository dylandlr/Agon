"""Given skills (caveman slate) and rare selflessness."""
import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import genotype as gt
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=32, n_agents=2, skills=True, climate_volatility=0.0, lightning_prob=0.0)


def _world():
    s = E.reset(jax.random.PRNGKey(5), CFG)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    return s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators))


def _water_dist(s, p):
    w = np.argwhere(np.asarray(s.terrain) == E.WATER)
    return int(np.abs(w - np.asarray(p)).max(1).min())


def test_blank_slate_has_no_skills_and_caveman_does():
    s = _world()
    assert np.asarray(E.action_mask(s, CFG))[0, E.SEEK_WATER:E.SEEK_SHELTER + 1].all()
    import dataclasses
    blank = dataclasses.replace(CFG, skills=False)
    assert not np.asarray(E.action_mask(s, blank))[0, E.SEEK_WATER:E.SEEK_SHELTER + 1].any()


def test_seek_water_uses_only_remembered_water_and_gets_there():
    s = _world()
    step = jax.jit(lambda st: E.step(st, jnp.array([E.SEEK_WATER, E.REST]), CFG))
    # With an empty map there is no known water: the skill explores instead of beelining
    blind = s._replace(mind_map=jnp.zeros_like(s.mind_map))
    land = np.argwhere(np.asarray(s.terrain) != E.WATER)
    far = next(p for p in land if _water_dist(s, p) >= 8)
    blind = blind._replace(pos=blind.pos.at[0].set(jnp.asarray(far)))
    out, _, _, info = step(blind)
    assert int(info.skill[0]) == 1
    # Knowing where water is, it walks there and drinks
    knows = blind._replace(mind_map=blind.mind_map.at[0, ..., 0].set((s.terrain == E.WATER).astype(jnp.float32))
                           .at[0, ..., 2].set(1.0))
    d0 = _water_dist(s, far)
    st = knows
    drank = False
    for _ in range(3 * d0):
        st = st._replace(hydration=jnp.full(2, 0.5))
        st, _, _, info = step(st)
        drank |= int(info.outcome[0]) == E.OUT_DRINK
    assert drank and _water_dist(st, st.pos[0]) == 1


def test_seek_food_gathers_then_eats():
    s = _world()
    r, c = map(int, np.asarray(s.pos[0]))
    s = s._replace(plants=s.plants.at[r, c].set(CFG.plant_max), satiety=jnp.full(2, 0.4),
                   hand=jnp.full(2, E.WOOD, jnp.int32), inv=jnp.zeros_like(s.inv))
    step = jax.jit(lambda st: E.step(st, jnp.array([E.SEEK_FOOD, E.REST]), CFG))
    outcomes = []
    for _ in range(4):
        s, _, _, info = step(s)
        outcomes.append(E.OUTCOME_NAMES[int(info.outcome[0])])
    assert outcomes[:3] == ["select", "gather", "select"] or "eat" in outcomes
    assert "eat" in outcomes


def test_flee_steps_away_from_a_visible_predator():
    s = E.reset(jax.random.PRNGKey(5), CFG)
    land = np.argwhere(np.asarray(s.terrain) != E.WATER)
    p = next(q for q in land if 4 <= q[0] <= 27 and 4 <= q[1] <= 27
             and all(s.terrain[q[0] + dr, q[1] + dc] != E.WATER for dr in (-1, 0, 1) for dc in (-1, 0, 1)))
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(p)),
                   predators=s.predators._replace(pos=s.predators.pos.at[0].set(jnp.asarray(p) + jnp.array([0, 2])),
                                                  alive=s.predators.alive.at[0].set(True)))
    mask = E.action_mask(s, CFG)
    a = E.resolve_skills(s, jnp.array([E.FLEE, E.REST]), mask, CFG)
    after = np.asarray(p) + np.asarray(E.MOVE_DELTAS[int(a[0])])
    assert int(a[0]) != E.REST
    assert np.abs(after - (np.asarray(p) + [0, 2])).max() > 2


def test_routes_go_around_water_and_avoid_steep_ground():
    g = 8
    cfg = EnvConfig(grid_size=g, n_agents=1, skills=True)
    terrain = jnp.full((g, g), E.GRASS, jnp.int32).at[0:7, 4].set(E.WATER)      # water, open only at the bottom
    targets = jnp.zeros((1, g, g), bool).at[0, 1, 7].set(True)
    flat = jnp.zeros((g, g))
    d = np.asarray(E.route_field(targets, terrain, flat, cfg))[0]
    assert d[1, 0] == 12                                                   # 7 straight, 12 around the water
    steep = flat.at[7, 3:6].set(1.0)                                        # the only way round is steep
    d2 = np.asarray(E.route_field(targets, terrain, steep, cfg))[0]
    assert d2[1, 0] > d[1, 0] + 10


def test_selflessness_is_rare_not_inherited_and_counts_strangers():
    import dataclasses
    many = dataclasses.replace(CFG, n_agents=64, saint_rate=1e-6)
    assert float(E.reset(jax.random.PRNGKey(0), many).genotype.selflessness.max()) == 0.0
    common = dataclasses.replace(CFG, n_agents=64, saint_rate=0.5)
    sf = np.asarray(E.reset(jax.random.PRNGKey(0), common).genotype.selflessness)
    assert 10 < (sf > 0).sum() < 54 and sf.max() <= 1.5 and sf[sf > 0].min() >= 0.5
    m = gt.sample(jax.random.PRNGKey(1), 4)._replace(selflessness=jnp.full(4, 1.0))
    child = gt.inherit(jax.random.PRNGKey(2), m, m, m)
    assert float(child.selflessness.max()) == 0.0                           # not passed on


def test_seek_shelter_walks_to_a_remembered_forest_and_rests_there():
    s = _world()
    t = np.asarray(s.terrain)
    forest = np.argwhere(t == E.FOREST)
    assert len(forest), "this world needs a forest"
    land = np.argwhere((t != E.WATER) & (t != E.FOREST) & (t != E.CAVE))
    dist = lambda p: int(np.abs(forest - p).max(1).min())
    start = next(p for p in land if 4 <= dist(p) <= 8)
    s = s._replace(pos=s.pos.at[0].set(jnp.asarray(start)),
                   mind_map=s.mind_map.at[0, ..., 2].set(1.0))                      # it has seen the whole map
    step = jax.jit(lambda st: E.step(st, jnp.array([E.SEEK_SHELTER, E.REST]), CFG))
    st = s
    for _ in range(4 * dist(start)):
        st = st._replace(hydration=jnp.ones(2), satiety=jnp.ones(2), energy=jnp.ones(2))
        st, _, _, info = step(st)
    r, c = map(int, np.asarray(st.pos[0]))
    assert t[r, c] == E.FOREST
    assert int(info.skill[0]) == 4 and int(info.outcome[0]) == E.OUT_REST          # arrived: rests in cover
    # an empty map: nothing known, so it explores rather than heading for an unseen forest
    blind = s._replace(mind_map=jnp.zeros_like(s.mind_map))
    _, _, _, info = step(blind)
    assert int(info.skill[0]) == 4


def test_seek_food_leaves_a_thinned_patch_for_a_better_known_one():
    """Charnov's marginal value theorem (cfg.patch_leave): a patch much poorer than the average one
    knows is left, not grazed to the bone; with patch_leave=0 the old skill stays and gathers."""
    import dataclasses
    s = _world()
    r, c = map(int, np.asarray(s.pos[0]))
    far = (r, c + 5) if c + 5 < CFG.grid_size else (r, c - 5)
    thin = 0.15 * CFG.plant_max                                     # above the 10% floor, far below average
    plants = s.plants.at[r, c].set(thin).at[far].set(0.9 * CFG.plant_max)
    mm = jnp.zeros_like(s.mind_map).at[0, r, c, 1].set(thin / CFG.plant_max).at[0, far[0], far[1], 1].set(0.9) \
        .at[0, r, c, 2].set(1.0).at[0, far[0], far[1], 2].set(1.0)
    s = s._replace(plants=plants, mind_map=mm, satiety=jnp.full(2, 0.4), hand=jnp.full(2, E.PLANT, jnp.int32),
                   inv=jnp.zeros_like(s.inv))
    _, _, _, info = E.step(s, jnp.array([E.SEEK_FOOD, E.REST]), CFG)
    assert E.OUTCOME_NAMES[int(info.outcome[0])] == "move"
    old = dataclasses.replace(CFG, patch_leave=0.0)
    _, _, _, info = E.step(s, jnp.array([E.SEEK_FOOD, E.REST]), old)
    assert E.OUTCOME_NAMES[int(info.outcome[0])] == "gather"
