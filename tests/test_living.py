"""The downed (EnvConfig.downed) and grazing animals with hungry predators (EnvConfig.grazing):
both off by default, on in the living world."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim import fauna as fa
from lifesim.config import EnvConfig, living_world


def _calm(cfg):
    return dataclasses.replace(cfg, climate_volatility=0.0, lightning_prob=0.0)


def test_off_by_default_on_in_the_living_world():
    base = EnvConfig()
    assert not base.downed and not base.grazing
    live = living_world(EnvConfig())
    assert live.downed and live.grazing
    assert not living_world(EnvConfig(), downed=False, grazing=False).downed


def test_the_downed_cannot_stand_but_can_eat_and_call_out():
    cfg = _calm(living_world(EnvConfig(grid_size=16, n_agents=3)))
    s = E.reset(jax.random.PRNGKey(0), cfg)
    s = s._replace(injury=s.injury.at[0].set(0.9), inv=s.inv.at[0, E.PLANT].set(1.0), hand=s.hand.at[0].set(E.PLANT))
    m = np.asarray(E.action_mask(s, cfg))
    allowed = {E.REST, E.GRASP, E.EAT, E.DRINK, E.RELEASE, E.SLEEP}
    assert all(not m[0, a] for a in range(m.shape[1]) if a not in allowed)       # no walking, striking, touching
    assert m[0, E.EAT] and m[0, E.REST]
    assert np.asarray(E.downed_of(s, cfg)).tolist() == [True, False, False]
    # others see the downed lying, as they see a sleeper
    off = dataclasses.replace(cfg, downed=False)
    assert not np.asarray(E.downed_of(s, off)).any() and np.asarray(E.action_mask(s, off))[0, E.MOVE_N:E.MOVE_N + 8].any()
    _, _, _, info = jax.jit(lambda st, a: E.step(st, a, cfg))(s, jnp.array([E.MOVE_N, E.REST, E.REST]))
    assert bool(info.downed[0])
    assert int(info.pos[0, 0]) == int(s.pos[0, 0]) and int(info.pos[0, 1]) == int(s.pos[0, 1])   # did not move


def test_grazers_eat_the_ground_and_starve_without_it():
    cfg = _calm(living_world(EnvConfig(grid_size=16, n_agents=1), downed=False))
    s = E.reset(jax.random.PRNGKey(1), cfg)
    deer = fa.IDX["deer"]
    L = s.large
    alive = jnp.zeros_like(L.alive).at[0].set(True)
    grass = np.argwhere(np.asarray(s.terrain) == E.GRASS)[0]
    L = L._replace(alive=alive, species=L.species.at[0].set(deer), pos=L.pos.at[0].set(jnp.array(grass, jnp.int32)),
                   below=jnp.zeros_like(L.below))
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    full = jnp.full_like(s.plants, cfg.plant_max)
    s1 = s._replace(large=L, small=dead(s.small), predators=dead(s.predators), plants=full)
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    s2, _, _, _ = step(s1, jnp.array([E.REST]))
    r, c = map(int, np.asarray(s2.large.pos[0]))
    assert float(s2.plants[r, c]) < cfg.plant_max                   # it ate where it stands
    # on bare ground (no plants anywhere) it weakens
    bare = s1._replace(plants=jnp.zeros_like(s1.plants), t=jnp.int32(cfg.year_length * 3 // 4 + 5))   # winter: no regrowth
    hp0 = float(bare.large.hp[0])
    for _ in range(20):
        bare, _, _, _ = step(bare, jnp.array([E.REST]))
    assert float(bare.large.hp[0]) < hp0


def test_a_kill_sates_a_predator_and_hunger_weakens_it():
    # what a kill does, not how often one lands: no dozing, no hunting penalty on the odds
    cfg = dataclasses.replace(_calm(living_world(EnvConfig(grid_size=16, n_agents=1), downed=False)),
                              animal_sleep=0.0, hunt_kill_scale=1.0)
    s = E.reset(jax.random.PRNGKey(2), cfg)
    wolf, rabbit = fa.IDX["wolf"], fa.IDX["rabbit"]
    cell = jnp.array(np.argwhere(np.asarray(s.terrain) == E.GRASS)[0], jnp.int32)
    P = s.predators._replace(alive=jnp.zeros_like(s.predators.alive).at[0].set(True), species=s.predators.species.at[0].set(wolf),
                             pos=s.predators.pos.at[0].set(cell), rest=jnp.zeros_like(s.predators.rest),
                             below=jnp.zeros_like(s.predators.below))
    far = jnp.array([15 - int(cell[0]), 15 - int(cell[1])], jnp.int32)
    s = s._replace(predators=P, large=s.large._replace(alive=jnp.zeros_like(s.large.alive)),
                   pos=s.pos.at[0].set(far))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    hungry = s._replace(small=s.small._replace(alive=jnp.zeros_like(s.small.alive)))
    hp0 = float(hungry.predators.hp[0])
    for _ in range(10):
        hungry, _, _, _ = step(hungry, jnp.array([E.REST]))
    assert float(hungry.predators.hp[0]) < hp0                       # no prey, no carcass: it weakens
    # a rabbit on its cell: sooner or later it kills and is sated (rests)
    sated = False
    for k in range(40):
        Sm = s.small._replace(alive=s.small.alive.at[0].set(True), species=s.small.species.at[0].set(rabbit),
                              pos=s.small.pos.at[0].set(s.predators.pos[0]), below=s.small.below.at[0].set(False))
        s, _, _, _ = step(s._replace(small=Sm), jnp.array([E.REST]))
        if int(s.predators.rest[0]) > 0:
            sated = True
            break
    assert sated


def test_clothing_differs_by_founder_babies_are_bare_and_it_keeps_heat_in():
    cfg = _calm(living_world(EnvConfig(grid_size=16, n_agents=6)))
    s = E.reset(jax.random.PRNGKey(4), cfg)
    c = np.asarray(s.clothing)
    assert (c >= cfg.clothing_min).all() and (c <= 1.0).all() and len(np.unique(np.round(c, 3))) == 6
    assert not np.asarray(E.reset(jax.random.PRNGKey(4), dataclasses.replace(cfg, clothing=False)).clothing).any()
    # cold: the clothed one's body cools more slowly than the bare one's
    cold = dataclasses.replace(cfg, season_temp=(-10.0,) * 4, n_agents=2)
    s = E.reset(jax.random.PRNGKey(5), cold)
    s = s._replace(clothing=jnp.array([1.0, 0.0]), body_temp=jnp.zeros(2), pos=jnp.stack([s.pos[0], s.pos[0]]),
                   genotype=s.genotype._replace(insulation=jnp.ones(2)))
    s2, _, _, _ = E.step(s, jnp.array([E.REST, E.REST]), cold)
    assert float(s2.body_temp[0]) > float(s2.body_temp[1])
