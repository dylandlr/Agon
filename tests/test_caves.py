"""The underground layer (EnvConfig.underground): cave geography and formations, delving, layers
kept apart, darkness, animals that follow, and what lies on the cave floor."""
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import fauna as fa
from lifesim import minerals as mn
from lifesim.config import EnvConfig

CFG = EnvConfig(grid_size=32, n_agents=3, biomes=True, underground=True, climate_volatility=0.0,
                lightning_prob=0.0, reward_mode="survival")
STEP = jax.jit(lambda s, a: E.step(s, a, CFG))


def _world(seed=1):
    """A cave world with animals cleared away, everyone fed and rested, at noon."""
    s = E.reset(jax.random.PRNGKey(seed), CFG)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    n = CFG.n_agents
    return s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators), t=jnp.int32(24),
                      satiety=jnp.ones(n), hydration=jnp.ones(n), energy=jnp.ones(n), immunity=jnp.ones(n))


def _mouth_and_next(s):
    """A cave mouth and a passage cell beside it."""
    m, ok = E.cave_mouths(s, CFG)
    passage = np.asarray(s.under & ~s.under_water)
    for (r, c), o in zip(np.asarray(m), np.asarray(ok)):
        if not o:
            continue
        for k, (dr, dc) in enumerate(np.asarray(E.NEIGHBORS8)):
            if 0 <= r + dr < CFG.grid_size and 0 <= c + dc < CFG.grid_size and passage[r + dr, c + dc]:
                return (int(r), int(c)), (int(r + dr), int(c + dc)), E.MOVE_N + k
    raise AssertionError("no mouth with a passage beside it")


def _act(*a):
    return jnp.array(a, jnp.int32)


def test_off_by_default_nothing_changes():
    cfg = EnvConfig(grid_size=24, n_agents=3, biomes=True)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    assert not bool(s.under.any()) and not bool(s.below.any())
    assert not bool(E.action_mask(s, cfg)[:, E.DELVE].any())
    bat = fa.IDX["bat"]
    assert not any(bool((an.species == bat).any()) for an in (s.small, s.large, s.predators))


def test_every_passage_connects_to_a_mouth():
    for seed in range(3):
        s = E.reset(jax.random.PRNGKey(seed), CFG)
        u, w = np.asarray(s.under), np.asarray(s.under_water)
        walk = u & ~w
        m, ok = E.cave_mouths(s, CFG)
        seen = np.zeros_like(walk)
        q = deque(tuple(p) for p, o in zip(np.asarray(m), np.asarray(ok)) if o)
        for p in q:
            seen[p] = True
        while q:
            r, c = q.popleft()
            for dr, dc in np.asarray(E.NEIGHBORS8):
                rr, cc = r + dr, c + dc
                if 0 <= rr < CFG.grid_size and 0 <= cc < CFG.grid_size and walk[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        assert walk.sum() > 20 and (seen | ~walk).all(), seed
        assert not (u & (np.asarray(s.terrain) == E.WATER)).any()          # never under open water
        # every pool can be drunk from a passage beside it
        for r, c in zip(*np.nonzero(w)):
            assert walk[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any()


def test_formations_set_minerals_and_air():
    s = E.reset(jax.random.PRNGKey(2), CFG)
    kinds = np.asarray(s.under_kind)
    floor = np.asarray(s.cave_floor)[..., E.MIN0:]
    assert (kinds[np.asarray(s.under)] > 0).all() and (kinds[~np.asarray(s.under)] == 0).all()
    for k in np.unique(kinds[kinds > 0]):
        allowed = np.array(mn.CAVE_CHANCE[k]) > 0
        assert floor[kinds == k][:, ~allowed].sum() == 0, mn.KINDS[k]           # only that formation's minerals
    summer = jnp.int32(CFG.year_length // 4 + 12)
    ice, karst, lava = (float(E.cave_air(summer, CFG, 0.0, jnp.int32(k))) for k in (E.K_ICE, E.K_KARST, E.K_LAVA))
    assert ice < 5.0 < karst < lava


def test_delving_down_and_up_and_walking_only_along_passages():
    s = _world()
    mouth, nxt, mv = _mouth_and_next(s)
    s = s._replace(pos=s.pos.at[0].set(jnp.array(mouth)))
    assert bool(E.action_mask(s, CFG)[0, E.DELVE])
    s, _, _, info = STEP(s, _act(E.DELVE, E.REST, E.REST))
    assert bool(s.below[0]) and bool(info.delved[0])
    mask = np.asarray(E.action_mask(s, CFG))[0]
    passage = np.asarray(s.under & ~s.under_water)
    for k, (dr, dc) in enumerate(np.asarray(E.NEIGHBORS8)):
        r, c = mouth[0] + dr, mouth[1] + dc
        inside = 0 <= r < CFG.grid_size and 0 <= c < CFG.grid_size and passage[r, c]
        assert mask[E.MOVE_N + k] == inside
    s, _, _, _ = STEP(s, _act(E.DELVE, E.REST, E.REST))
    assert not bool(s.below[0])


def test_the_layers_do_not_meet():
    s = _world()
    mouth, nxt, mv = _mouth_and_next(s)
    k = mv - E.MOVE_N
    # 0 below in the passage beside the mouth; 1 above, on the mouth; 2 above on 0's cell
    s = s._replace(pos=jnp.array([nxt, mouth, nxt], jnp.int32), below=jnp.array([True, False, False]))
    mask = np.asarray(E.action_mask(s, CFG))
    assert mask[1, E.TOUCH_N + k]                                           # 2 is there, above
    obs = np.asarray(E.observe(s, CFG))
    v, C = CFG.view_size, E.N_OBS_CHANNELS
    win = obs[:, :v * v * C].reshape(-1, v, v, C)
    agents = E.N_TERRAIN + 2
    assert win[0, ..., agents].sum() == 0                                   # below, it sees no one
    assert win[1, ..., agents].sum() == 1 and win[2, ..., agents].sum() == 1   # above, only each other
    s2, _, _, info = STEP(s, _act(E.REST, E.STRIKE_N + k, E.REST))          # 1 strikes at that cell
    assert bool(info.struck_by_agent[2]) and not bool(info.struck_by_agent[0])
    heard = E.heard_messages(s._replace(last_msg=jnp.zeros((3, E.MSG_LEN), jnp.int32).at[1].set(3)), CFG)
    assert float(np.abs(np.asarray(heard[0])).sum()) == 0 and float(np.abs(np.asarray(heard[2])).sum()) > 0


def test_darkness_frightens_and_a_torch_lights_the_way():
    s = _world()
    mouth, nxt, mv = _mouth_and_next(s)
    s = s._replace(pos=jnp.array([nxt, nxt, (0, 0)], jnp.int32), below=jnp.array([True, False, False]),
                   alive=jnp.array([True, True, False]))
    s1, _, _, info = STEP(s, _act(E.REST, E.REST, E.REST))
    assert bool(info.dark[0]) and not bool(info.dark[1])
    assert float(s1.stress[0]) > float(s1.stress[1])
    lit = s._replace(torch=s.torch.at[0].set(30), inv=s.inv.at[0, E.WOOD].set(1.0), hand=s.hand.at[0].set(E.WOOD))
    _, _, _, info = STEP(lit, _act(E.REST, E.REST, E.REST))
    assert not bool(info.dark[0])
    obs_dark = np.asarray(E.observe(s, CFG))[0, :CFG.view_size ** 2 * E.N_OBS_CHANNELS]
    obs_lit = np.asarray(E.observe(lit, CFG))[0, :CFG.view_size ** 2 * E.N_OBS_CHANNELS]
    assert (np.abs(obs_lit) > 0).sum() > (np.abs(obs_dark) > 0).sum()      # the torch shows more


def test_mining_and_what_is_left_below_stays_below():
    s = _world()
    floor = np.asarray(s.cave_floor)[..., E.MIN0:].sum(-1)
    r, c = map(int, np.argwhere(floor > 0)[0])
    m = int(np.argmax(np.asarray(s.cave_floor)[r, c, E.MIN0:]))
    s = s._replace(pos=s.pos.at[0].set(jnp.array([r, c])), below=s.below.at[0].set(True),
                   hand=s.hand.at[0].set(E.MIN0 + m))
    before = float(s.cave_floor[r, c, E.MIN0 + m])
    s, _, _, _ = STEP(s, _act(E.GRASP, E.REST, E.REST))
    assert float(s.inv[0, E.MIN0 + m]) > 0 and float(s.cave_floor[r, c, E.MIN0 + m]) < before
    s = s._replace(inv=s.inv.at[0, E.RAW_MEAT].set(2.0), hand=s.hand.at[0].set(E.RAW_MEAT))
    meat_above = float(s.meat[r, c])
    s, _, _, _ = STEP(s, _act(E.RELEASE, E.REST, E.REST))
    assert float(s.cave_floor[r, c, E.RAW_MEAT]) > 0.5 and float(s.meat[r, c]) == meat_above


def test_a_predator_follows_its_quarry_into_the_cave():
    s = _world()
    mouth, nxt, mv = _mouth_and_next(s)
    wolf = fa.IDX["wolf"]
    beside = next((mouth[0] + dr, mouth[1] + dc) for dr, dc in np.asarray(E.NEIGHBORS8)
                  if 0 <= mouth[0] + dr < CFG.grid_size and 0 <= mouth[1] + dc < CFG.grid_size
                  and int(s.terrain[mouth[0] + dr, mouth[1] + dc]) != E.WATER)
    p = s.predators
    p = p._replace(alive=p.alive.at[0].set(True), species=p.species.at[0].set(wolf), rest=p.rest.at[0].set(0),
                   pos=p.pos.at[0].set(jnp.array(beside, jnp.int32)), below=p.below.at[0].set(False))
    s = s._replace(predators=p, pos=s.pos.at[0].set(jnp.array(mouth)), below=s.below.at[0].set(True),
                   t=jnp.int32(CFG.day_length - 2))                             # night: wolves hunt
    followed = False
    for _ in range(30):
        s, _, _, _ = STEP(s, _act(E.REST, E.REST, E.REST))
        followed |= bool(s.predators.below[0])
    assert followed


def test_bats_roost_below_and_leave_saltpetre():
    s = E.reset(jax.random.PRNGKey(3), CFG)
    bat = np.asarray(s.small.species) == fa.IDX["bat"]
    assert bat.any() and np.asarray(s.small.below)[bat].all()
    passage = np.asarray(s.under & ~s.under_water)
    assert all(passage[r, c] for r, c in np.asarray(s.small.pos)[bat])
    sp = E.MIN0 + mn.IDX["saltpetre"]
    before = float(s.cave_floor[..., sp].sum())
    for _ in range(5):
        s, _, _, _ = STEP(s, _act(E.REST, E.REST, E.REST))
    assert float(s.cave_floor[..., sp].sum()) > before
