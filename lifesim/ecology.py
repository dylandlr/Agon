"""Plant food as living things (EnvConfig.ecology): what grows where, how it grows back, how fast
people can gather it, and what it is worth.

Four kinds of plant food, the ones foragers actually live on:

  greens  leaves, shoots and herbs: low in energy, slow to gather, grow back from what is left
  fruit   berries and fruit: ripen in summer and autumn, rot within weeks once the season ends
  nuts    nuts and seeds: fall in autumn, keep through winter, dense in energy
  roots   roots, tubers and rhizomes: there all year, dug up (faster with a digging stick), and slow
          to grow back once dug, because digging takes the plant

Every amount is in food units: one unit is the energy of one unit of plant food in hand
(cfg.plant_food_value of satiety, about 4,400 kcal at the calibration). Biome figures are per km^2
and become per-cell amounts through the cell's area, so the ecology keeps its density whatever the
tick length makes a cell's size (2 km across at 30-minute ticks, 67 m at 1-minute ticks).

How each grows:
  greens, roots: logistic regrowth from what is left (r S (1 - S / K)), plus a little from a seed bank
    and from seed blowing in from neighbouring cells. A patch grazed or dug bare comes back slowly; a
    patch left alone fills up. Greens die back in winter; roots lie dormant.
  fruit, nuts: the trees and shrubs that bear them are not harmed by picking. They make a seasonal
    crop at the biome's rate, up to what the patch can hold; fruit rots fast out of season, nuts slowly.
  All growth follows the season, rain (faster), drought (slower) and snow (none); fire burns greens,
  fruit and nuts on the ground, but roots survive it underground.

Gathering takes time: an act gathers what the kind's return rate (kcal per hour, typical of foraging
studies) yields in the tick, times how dense the food is here (a type II functional response: in a
thin patch most of the time goes to searching). A rich patch pays; a picked-over one does not, so a
camp that stays on one patch eats it out and has to move, as foragers do.

The figures are order-of-magnitude estimates in the range reported for foragers' environments (e.g.
Kelly, The Lifeways of Hunter-Gatherers, 2013, on return rates and population densities), chosen so
that a forest supports roughly 0.2-0.5 people per km^2 indefinitely and tundra or desert a tenth of
that. They are knobs, not measurements.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

KINDS = ("greens", "fruit", "nuts", "roots")
GREENS, FRUIT, NUTS, ROOTS = range(4)
N_KINDS = 4

# Per biome (terrain index: grass, forest, rock, water, desert, tundra, marsh, cave) and kind:
# K = standing crop a patch can hold at most (kcal / km^2), P = mean production (kcal / km^2 / day)
_K = {
    "grass":  (60e3, 30e3, 60e3, 250e3),
    "forest": (40e3, 150e3, 400e3, 300e3),
    "rock":   (4e3, 8e3, 0.0, 0.0),
    "water":  (0.0, 0.0, 0.0, 0.0),
    "desert": (5e3, 15e3, 20e3, 20e3),
    "tundra": (10e3, 40e3, 0.0, 15e3),
    "marsh":  (60e3, 20e3, 40e3, 600e3),
    "cave":   (0.0, 0.0, 0.0, 0.0),
}
_P = {
    "grass":  (400.0, 150.0, 300.0, 400.0),
    "forest": (150.0, 800.0, 1500.0, 500.0),
    "rock":   (20.0, 30.0, 0.0, 0.0),
    "water":  (0.0, 0.0, 0.0, 0.0),
    "desert": (30.0, 60.0, 80.0, 50.0),
    "tundra": (60.0, 150.0, 0.0, 40.0),
    "marsh":  (400.0, 100.0, 200.0, 1500.0),
    "cave":   (0.0, 0.0, 0.0, 0.0),
}
TERRAIN_ORDER = ("grass", "forest", "rock", "water", "desert", "tundra", "marsh", "cave")
K_TABLE = jnp.array([_K[t] for t in TERRAIN_ORDER])          # (N_TERRAIN, 4)
P_TABLE = jnp.array([_P[t] for t in TERRAIN_ORDER])

# Return rates while gathering (kcal per hour of work, processing included) and the density at
# which they halve (kcal / km^2): greens are slow; nuts need cracking; roots need digging
RATE_KCAL_H = jnp.array([500.0, 1500.0, 1000.0, 1200.0])
HALF_DENSITY = jnp.array([10e3, 20e3, 30e3, 40e3])
DIG_STICK_BONUS = 1.5                                         # roots, with wood in hand

# Seasonal pattern (spring, summer, autumn, winter): growth multiplier per kind. Fruit and nuts:
# share of the year's crop made in that season, as a rate multiplier (mean over the year = 1).
SEASON_GROWTH = jnp.array([
    [1.8, 1.3, 0.6, 0.0],     # greens: spring flush, winter dieback (below)
    [0.0, 2.2, 1.8, 0.0],     # fruit: summer and autumn
    [0.0, 0.0, 4.0, 0.0],     # nuts: the autumn mast
    [1.4, 1.4, 0.9, 0.3],     # roots: grow most of the year, near dormant in winter
])
# Loss per day (rotting, dieback, animals) by season and kind
SEASON_LOSS = jnp.array([
    [0.01, 0.01, 0.02, 0.05],   # greens: wither in autumn, die back in winter
    [0.10, 0.02, 0.03, 0.10],   # fruit: keeps on the bush in season, rots out of it
    [0.005, 0.01, 0.005, 0.005],  # nuts: keep (cached by trees, taken slowly by animals)
    [0.0, 0.0, 0.0, 0.0],       # roots: stay in the ground
])
SEED_BANK = jnp.array([0.02, 0.0, 0.0, 0.005])    # share of K that regrows from nothing, x r
SPREAD = jnp.array([0.3, 0.0, 0.0, 0.1])          # seed from neighbours, x their fullness x r x K
LOGISTIC = jnp.array([True, False, False, True])
BURNS = jnp.array([True, True, True, False])      # roots survive fire


def cell_km2(cfg) -> float:
    return float(cfg.cell_km) ** 2


def unit_kcal(cfg) -> float:
    """kcal in one unit of plant food: its satiety value times the kcal one satiety unit stands for
    (a day's need, cfg.kcal_per_day, is what hunger takes in a day)."""
    return float(cfg.plant_food_value) * float(cfg.kcal_per_day) / (float(cfg.hunger_rate) * float(cfg.day_length))


def capacity(terrain, patch, cfg):
    """(G, G, 4) most each cell can hold, in food units."""
    return K_TABLE[terrain] * patch * cfg.plant_abundance * cell_km2(cfg) / unit_kcal(cfg)


def production(terrain, patch, cfg):
    """(G, G, 4) mean production per day, in food units."""
    return P_TABLE[terrain] * patch * cfg.plant_abundance * cell_km2(cfg) / unit_kcal(cfg)


def init(key, terrain, cfg):
    """Patches (each kind's richness here, fixed for the world: plant communities are patchy) and the
    standing crop at the start (spring: greens coming up, little fruit, last autumn's nuts left, roots
    in the ground)."""
    k1, k2 = jax.random.split(key)
    g = terrain.shape[0]
    patch = jnp.exp(0.6 * jax.random.normal(k1, (g, g, N_KINDS)) - 0.18)   # lognormal, mean ~1
    start = jnp.array([0.4, 0.0, 0.25, 0.7])
    flora = capacity(terrain, patch, cfg) * start * jax.random.uniform(k2, (g, g, N_KINDS), minval=0.6, maxval=1.0)
    return flora, patch


def grow(flora, terrain, patch, season, growth_mult, burning, cfg, bear=None):
    """One tick of growth and loss. growth_mult (G, G): rain, drought and snow; burning (G, G) bool;
    bear: None, or (G, G, 4) x each kind's capacity and growth (what the trees there bear: trees.crop)."""
    dt = float(cfg.tick_minutes) / 1440.0                            # days per tick
    K = capacity(terrain, patch, cfg)
    P = production(terrain, patch, cfg)
    if bear is not None:
        K, P = K * bear, P * bear
    s = SEASON_GROWTH[:, season]                                     # (4,)
    loss = SEASON_LOSS[:, season]
    Ks = jnp.maximum(K, 1e-9)
    full = flora / Ks
    # logistic kinds: r from the production rate (maximum sustained yield r K / 4 = P)
    r = 4.0 * P / Ks
    pad = jnp.pad(full, ((1, 1), (1, 1), (0, 0)), mode="edge")
    g = flora.shape[0]
    nb = sum(pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr, dc) != (0, 0)) / 8.0
    logistic = r * flora * (1 - full) + r * K * (SEED_BANK + SPREAD * nb) * (1 - full)
    pulse = P * (1 - full)                                           # trees and shrubs bear up to what they hold
    gain = jnp.where(LOGISTIC, logistic, pulse) * s * growth_mult[..., None]
    flora = flora + dt * (gain - loss * flora)
    flora = jnp.where(burning[..., None] & BURNS, 0.0, flora)
    return jnp.clip(flora, 0.0, K)


def density(flora, cfg):
    """kcal per km^2 of each kind."""
    return flora * unit_kcal(cfg) / cell_km2(cfg)


def quality(flora, cfg):
    """(..., 4) in [0, 1): the share of the full return rate gathering each kind here would get."""
    d = density(flora, cfg)
    return d / (d + HALF_DENSITY)


def gather_yield(flora_here, digging, cfg):
    """(N, 4) food units one act of gathering would get of each kind here (not more than there is).
    flora_here (N, 4); digging (N,) bool: a stick in hand for roots."""
    hours = float(cfg.tick_minutes) / 60.0
    rate = RATE_KCAL_H * jnp.where(digging[:, None] & (jnp.arange(N_KINDS) == ROOTS), DIG_STICK_BONUS, 1.0)
    got = rate * hours * quality(flora_here, cfg) / unit_kcal(cfg)
    return jnp.minimum(got, flora_here)
