"""Trees: each forest cell's trees are of one kind, with an age. The kind is a real species' base
(its height, growth, lifespan, crop and dead wood); none of that is known to the agents, who see
only how tall a stand is and the tone of its leaves (needles dark, broad leaves light). Cheap: two
(G, G) arrays, aged once a tick.

Kinds are chosen by each cell's mean temperature and wetness, in stands (smooth noise per kind)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

# name, latin, max height m, years to ~63% of it, lifespan years, leaf tone (0 dark needles .. 1 light
# broad leaves), fruit and nut crop (x the cell's fruit and nut capacity when grown), dead wood
# (x the forest's dead wood), preferred mean temperature C and its spread, wet (+1 by water, -1 dry)
SPECIES = [
    dict(name="pine", latin="Pinus sylvestris", max_h=30, tau=40, life=300, leaf=0.15, fruit=0.0, nuts=0.3, deadfall=1.0, t_opt=6, t_wid=7, wet=-0.3),
    dict(name="spruce", latin="Picea abies", max_h=40, tau=50, life=300, leaf=0.08, fruit=0.0, nuts=0.0, deadfall=0.9, t_opt=3, t_wid=6, wet=0.2),
    dict(name="birch", latin="Betula pendula", max_h=25, tau=20, life=90, leaf=0.85, fruit=0.0, nuts=0.0, deadfall=1.2, t_opt=6, t_wid=8, wet=0.0),
    dict(name="oak", latin="Quercus robur", max_h=35, tau=60, life=600, leaf=0.55, fruit=0.0, nuts=1.0, deadfall=1.0, t_opt=11, t_wid=6, wet=0.0),
    dict(name="beech", latin="Fagus sylvatica", max_h=35, tau=50, life=300, leaf=0.6, fruit=0.0, nuts=0.6, deadfall=0.9, t_opt=10, t_wid=5, wet=0.3),
    dict(name="hazel", latin="Corylus avellana", max_h=8, tau=10, life=80, leaf=0.7, fruit=0.0, nuts=1.5, deadfall=1.3, t_opt=10, t_wid=7, wet=0.1),
    dict(name="wild apple", latin="Malus sylvestris", max_h=10, tau=15, life=100, leaf=0.7, fruit=1.5, nuts=0.0, deadfall=0.8, t_opt=11, t_wid=6, wet=0.0),
    dict(name="willow", latin="Salix alba", max_h=25, tau=20, life=100, leaf=0.9, fruit=0.0, nuts=0.0, deadfall=1.1, t_opt=10, t_wid=9, wet=1.0),
    dict(name="juniper", latin="Juniperus communis", max_h=6, tau=25, life=300, leaf=0.2, fruit=0.5, nuts=0.0, deadfall=0.6, t_opt=4, t_wid=9, wet=-1.0),
    dict(name="wild olive", latin="Olea europaea", max_h=10, tau=40, life=1000, leaf=0.45, fruit=1.0, nuts=0.0, deadfall=0.7, t_opt=17, t_wid=4, wet=-0.8),
]
NAMES = [s["name"] for s in SPECIES]
K = len(SPECIES)
col = lambda key: jnp.array([s[key] for s in SPECIES], jnp.float32)


def plant(key, forest, mean_temp, wet, stand_scale: int = 4):
    """(G, G) int32 kind (-1 where no forest) and (G, G) age in years. mean_temp (G, G) C; wet (G, G) 0..1."""
    g = forest.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    fit = -((mean_temp[None] - col("t_opt")[:, None, None]) / col("t_wid")[:, None, None]) ** 2 \
        + 1.5 * col("wet")[:, None, None] * (2.0 * wet[None] - 1.0)                     # (K, G, G)
    m = max(1, g // stand_scale)                                                         # stands: a coarse field per kind
    coarse = jax.random.normal(k1, (K, m, m))
    stands = jax.image.resize(coarse, (K, g, g), "linear")
    score = fit + 1.2 * stands + jax.random.gumbel(k2, (K, g, g)) * 0.5
    kind = jnp.where(forest, jnp.argmax(score, 0), -1).astype(jnp.int32)
    life = col("life")[jnp.clip(kind, 0)]
    age = jax.random.uniform(k3, (g, g)) * 0.8 * life                                    # of every age
    return kind, jnp.where(forest, age, 0.0)


def height(kind, age):
    """(G, G) metres: grows toward its kind's height, fast young and slower as it nears it."""
    k = jnp.clip(kind, 0)
    return jnp.where(kind >= 0, col("max_h")[k] * (1.0 - jnp.exp(-age / col("tau")[k])), 0.0)


def grown(kind, age):
    """(G, G) 0..1: share of its kind's full height (how much it bears and drops)."""
    return jnp.where(kind >= 0, height(kind, age) / col("max_h")[jnp.clip(kind, 0)], 0.0)


def age_step(kind, age, years, burning):
    """Age every stand by `years`; one past its lifespan falls (returns where) and a seedling of its kind
    takes its place; fire kills a stand back to seedlings."""
    age = jnp.where(kind >= 0, age + years, 0.0)
    fell = (kind >= 0) & (age >= col("life")[jnp.clip(kind, 0)])
    return jnp.where(fell | burning, 0.0, age), fell


def crop(kind, age):
    """(G, G, 4) x the cell's greens, fruit, nuts, roots capacity and growth: mature fruit and nut trees
    bear (none until half grown); greens and roots are the ground's, not the trees'."""
    k = jnp.clip(kind, 0)
    bearing = jnp.clip((grown(kind, age) - 0.5) / 0.5, 0.0, 1.0)
    fruit = jnp.where(kind >= 0, 0.5 + col("fruit")[k] * bearing, 1.0)
    nuts = jnp.where(kind >= 0, 0.5 + col("nuts")[k] * bearing, 1.0)
    one = jnp.ones_like(fruit)
    return jnp.stack([one, fruit, nuts, one], -1)


def looks(kind, age):
    """(G, G, 2) what anyone can see of a stand: its height (0..1 of 40 m) and leaf tone."""
    return jnp.stack([height(kind, age) / 40.0, jnp.where(kind >= 0, col("leaf")[jnp.clip(kind, 0)], 0.0)], -1)


def table():
    """For the playback: each kind's facts (hidden from the agents)."""
    return [{k: s[k] for k in ("name", "latin", "max_h", "life", "leaf", "fruit", "nuts")} for s in SPECIES]
