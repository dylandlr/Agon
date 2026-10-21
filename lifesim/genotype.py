"""Heritable traits. Genes encode drives, capacities and learning biases,
never concepts or behaviors. All traits are float arrays of shape (n_agents,)."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Genotype(NamedTuple):
    # Physical: multipliers around 1.0
    strength: jnp.ndarray
    speed: jnp.ndarray
    metabolism: jnp.ndarray       # higher = gets hungry faster
    lifespan: jnp.ndarray
    disease_resistance: jnp.ndarray  # scales infection chance and severity down
    insulation: jnp.ndarray       # slows body-temperature drift toward ambient
    height: jnp.ndarray           # adult height multiplier (x base height)
    build: jnp.ndarray            # frame: ideal weight multiplier for a given height
    # Temperament: weights on intrinsic reward terms
    pain_aversion: jnp.ndarray    # weight on health loss
    social_reward: jnp.ndarray    # weight on social contact (unused until Phase 2)
    novelty: jnp.ndarray          # weight on visiting new cells
    patience: jnp.ndarray         # discount factor gamma in [0.95, 0.999]
    # Plasticity
    lr_scale: jnp.ndarray         # multiplies optimizer learning rate
    critical_period: jnp.ndarray  # ticks over which learning rate decays
    social_learning: jnp.ndarray  # imitation loss weight (unused until Phase 2)
    temperament_drift: jnp.ndarray  # how far experience shifts temperament phenotype
    # Moral sentiments: feelings that add to or subtract from reward. They are
    # motives, not rules; what a group comes to punish or share must emerge.
    empathy: jnp.ndarray          # distress at seeing any agent hurt
    kin_care: jnp.ndarray         # extra distress when the hurt agent looks like oneself
    envy: jnp.ndarray             # displeasure at being worse fed than visible others
    guilt: jnp.ndarray            # displeasure at being better fed than visible others
    reciprocity: jnp.ndarray      # satisfaction in giving to someone who gave to oneself
    vengefulness: jnp.ndarray     # satisfaction in striking someone who struck oneself
    sadism: jnp.ndarray           # pleasure in the pain one inflicts, on anyone (near 0 for most)
    # Drive: how much stress intimacy relieves. Pleasure is felt through the body (calm),
    # so selection can tune the drive without it becoming a reward of its own.
    libido: jnp.ndarray
    # Attraction to one's own sex, 0..1 (attraction_of): below 0.25 straight, 0.25-0.75
    # bisexual, above 0.75 gay. Mating needs mutual attraction; between two of the same sex it is
    # intimacy (calm, attachment), never a conception.
    same_sex: jnp.ndarray
    # Rare: weighs others' visible fortunes like one's own, kin or not (0 for nearly everyone).
    # Not inherited: it arises anew with probability EnvConfig.saint_rate (set by the environment).
    selflessness: jnp.ndarray


# (log-mean, log-std) for multiplicative traits; (low, high) for bounded ones
_LOGNORMAL = {
    "strength": 0.15, "speed": 0.15, "metabolism": 0.15, "lifespan": 0.10,
    "disease_resistance": 0.20, "insulation": 0.15, "height": 0.05, "build": 0.10,
    "pain_aversion": 0.30, "social_reward": 0.30, "novelty": 0.30,
    "lr_scale": 0.30, "libido": 0.30,
}
_BOUNDS = {
    "patience": (0.95, 0.999),
    "critical_period": (2000.0, 20000.0),
    "social_learning": (0.0, 1.0),
    "temperament_drift": (0.0, 1.0),
    "empathy": (0.0, 1.0), "kin_care": (0.0, 1.0), "envy": (0.0, 1.0), "guilt": (0.0, 1.0),
    "reciprocity": (0.0, 1.0), "vengefulness": (0.0, 1.0), "sadism": (0.0, 1.0), "same_sex": (0.0, 1.0),
    "selflessness": (0.0, 2.0),
}
_FIELDS = Genotype._fields


def sample(key: jax.Array, n: int) -> Genotype:
    """Founder genotypes: independent draws per trait."""
    keys = jax.random.split(key, len(_FIELDS))
    vals = {}
    for k, name in zip(keys, _FIELDS):
        if name == "selflessness":
            vals[name] = jnp.zeros((n,))
        elif name == "sadism":                  # skewed: most feel almost none, a few a great deal
            vals[name] = jax.random.uniform(k, (n,)) ** 4
        elif name == "same_sex":                # U^12: about 89% straight, 8.5% bisexual, 2.5% gay
            vals[name] = jax.random.uniform(k, (n,)) ** 12
        elif name in _LOGNORMAL:
            vals[name] = jnp.exp(_LOGNORMAL[name] * jax.random.normal(k, (n,)))
        else:
            lo, hi = _BOUNDS[name]
            vals[name] = jax.random.uniform(k, (n,), minval=lo, maxval=hi)
    return Genotype(**vals)


def clip(g: Genotype) -> Genotype:
    """Keep traits in biologically sane ranges after mutation."""
    vals = {}
    for name in _FIELDS:
        v = getattr(g, name)
        if name in _LOGNORMAL:
            vals[name] = jnp.clip(v, 0.25, 4.0)
        else:
            lo, hi = _BOUNDS[name]
            vals[name] = jnp.clip(v, lo, hi)
    return Genotype(**vals)


# Share of a trait's variation passed from parents, as in human twin studies:
# body size is highly heritable, temperament and moral sentiments much less.
HERITABILITY = {
    "height": 0.8, "build": 0.6, "strength": 0.5, "speed": 0.5, "metabolism": 0.5,
    "lifespan": 0.25, "disease_resistance": 0.3, "insulation": 0.5,
    "pain_aversion": 0.4, "social_reward": 0.4, "novelty": 0.4, "patience": 0.4,
    "lr_scale": 0.4, "critical_period": 0.4, "social_learning": 0.4, "temperament_drift": 0.4,
    "empathy": 0.35, "kin_care": 0.35, "envy": 0.35, "guilt": 0.35,
    "reciprocity": 0.35, "vengefulness": 0.35, "sadism": 0.35, "libido": 0.4,
    "same_sex": 0.3,                    # twin studies: about 0.3
}


def attraction_of(g: Genotype):
    """(to the other sex, to one's own sex) as bool arrays: straight, bisexual (both) or gay."""
    return g.same_sex < 0.75, g.same_sex > 0.25


def inherit(key: jax.Array, mother: Genotype, father: Genotype, population: Genotype) -> Genotype:
    """A child's genotype: the parents' average pulled toward the population mean
    by (1 - heritability), plus random variation of the size real siblings show.

    Multiplicative traits are blended on the log scale; bounded ones directly.
    Shapes: mother/father (k,), population (N,) used only for its mean and spread."""
    keys = jax.random.split(key, len(_FIELDS))
    vals = {}
    for k, name in zip(keys, _FIELDS):
        h2 = HERITABILITY.get(name, 0.4)
        log = name in _LOGNORMAL
        tr = (lambda x: jnp.log(x)) if log else (lambda x: x)
        m, f, pop = tr(getattr(mother, name)), tr(getattr(father, name)), tr(getattr(population, name))
        mid = (m + f) / 2
        mean, sd = pop.mean(), pop.std() + 1e-6
        child = mean + h2 * (mid - mean) + jnp.sqrt(1 - h2 ** 2 / 2) * sd * jax.random.normal(k, m.shape)
        vals[name] = jnp.exp(child) if log else child
    vals["selflessness"] = jnp.zeros_like(vals["selflessness"])      # drawn anew by the environment
    return clip(Genotype(**vals))


def mutate(key: jax.Array, child: Genotype, population: Genotype, rate: float = 0.01, size: float = 2.5):
    """De novo mutations: each trait of each child mutates with probability `rate`, jumping by
    `size` population standard deviations up or down (on the log scale for multiplicative traits),
    then clipped to the trait's range. Far rarer and larger than the ordinary variation between
    siblings that inherit() already gives. Selflessness is left alone (it is drawn by the
    environment). Returns the genotype and the jumps in standard deviations, (k, n_traits)."""
    keys = jax.random.split(key, len(_FIELDS))
    vals, jumps = {}, []
    for k, name in zip(keys, _FIELDS):
        x = getattr(child, name)
        k1, k2 = jax.random.split(k)
        hit = (jax.random.uniform(k1, x.shape) < rate) & (name != "selflessness")
        jump = jnp.where(hit, jnp.where(jax.random.bernoulli(k2, 0.5, x.shape), size, -size), 0.0)
        log = name in _LOGNORMAL
        pop = jnp.log(getattr(population, name)) if log else getattr(population, name)
        sd = pop.std() + 1e-6
        y = (jnp.log(x) if log else x) + jump * sd
        vals[name] = jnp.exp(y) if log else y
        jumps.append(jump)
    return clip(Genotype(**vals)), jnp.stack(jumps, -1)


def recombine_and_mutate(key: jax.Array, mother: Genotype, father: Genotype,
                         mutation_scale: float = 0.05) -> Genotype:
    """Per-trait uniform crossover, then multiplicative (or additive, for
    bounded traits) Gaussian mutation. Shapes must match."""
    k_cross, k_mut = jax.random.split(key)
    cross_keys = jax.random.split(k_cross, len(_FIELDS))
    mut_keys = jax.random.split(k_mut, len(_FIELDS))
    vals = {}
    for ck, mk, name in zip(cross_keys, mut_keys, _FIELDS):
        m, f = getattr(mother, name), getattr(father, name)
        pick = jax.random.bernoulli(ck, 0.5, m.shape)
        child = jnp.where(pick, m, f)
        noise = jax.random.normal(mk, m.shape)
        if name in _LOGNORMAL:
            child = child * jnp.exp(mutation_scale * noise)
        else:
            lo, hi = _BOUNDS[name]
            child = child + mutation_scale * (hi - lo) * noise
        vals[name] = child
    return clip(Genotype(**vals))
