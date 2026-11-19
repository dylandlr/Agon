"""Conditions: discrete traits an agent has, good or bad, each with a rarity.

They come four ways, as in people:
- inborn: present from birth. Most are heritable: a child of a parent who has one gets it with
  `inherit` probability (else the base rate); the rest arise anew.
- age: may begin each year past an onset age.
- event: triggered by something that happened (a near-death, an illness survived, a long
  hardship, a crippling injury); some fade with time.
- use: built by doing something many times (walking, striking).

Each acts only on physical quantities the world already has (strength, speed, sight, healing,
disease, bleeding, warmth, fatigue, stress, lifespan, fertility, walking cost, blow force); none
prescribes behaviour. EFFECTS gives each one's multipliers; env.condition_effects combines them.
"""
from __future__ import annotations

# name, kind, good?, base chance, inherit chance (inborn), onset (age: years; event/use: see env),
# fade per day (event conditions that can heal), plain description
CONDITIONS = [
    ("keen eyes", "inborn", True, 0.03, 0.5, None, 0.0, "sees one cell further at night and resists sleepy blur"),
    ("poor eyesight", "inborn", False, 0.05, 0.5, None, 0.0, "sees one cell less, day and night"),
    ("robust", "inborn", True, 0.04, 0.4, None, 0.0, "20% stronger; heals 20% faster"),
    ("frail", "inborn", False, 0.04, 0.4, None, 0.0, "20% weaker; 30% more prone to infection"),
    ("iron stomach", "inborn", True, 0.03, 0.4, None, 0.0, "raw meat and bad water rarely make it ill"),
    ("long-lived", "inborn", True, 0.02, 0.5, None, 0.0, "natural lifespan 20% longer"),
    ("slow clotting", "inborn", False, 0.01, 0.5, None, 0.0, "bad wounds bleed twice as much"),
    ("cold-hardy", "inborn", True, 0.04, 0.4, None, 0.0, "keeps warm 40% better"),
    ("infertile", "inborn", False, 0.02, 0.0, None, 0.0, "cannot conceive or father a child"),
    ("stiff joints", "age", False, 0.03, 0.0, 45.0, 0.0, "walks 25% slower (each year past 45: 3%)"),
    ("failing sight", "age", False, 0.04, 0.0, 50.0, 0.0, "sees one cell less (each year past 50: 4%)"),
    ("lame", "event", False, 1.0, 0.0, None, 0.0, "a crippling injury that never healed: walks 30% slower, stumbles more"),
    ("chronic cough", "event", False, 0.1, 0.0, None, 0.02, "after an illness (10%): tires 30% faster; may clear"),
    ("haunted", "event", False, 0.3, 0.0, None, 0.01, "after nearly dying (30%): stress rises 50% faster; fades slowly"),
    ("hardened", "event", True, 0.2, 0.0, None, 0.0, "after a long hardship survived (20%): stress rises 30% slower"),
    ("enduring", "use", True, 1.0, 0.0, None, 0.0, "after walking a great deal: walking costs 15% less"),
    ("heavy-handed", "use", True, 1.0, 0.0, None, 0.0, "after many blows struck: blows 20% harder"),
]
NAMES = [c[0] for c in CONDITIONS]
K = len(CONDITIONS)
IDX = {n: i for i, n in enumerate(NAMES)}

# multiplier (or additive, for sight) effects, by condition
EFFECTS = {
    "keen eyes": {"night_sight": 1, "sleepy_sight": -1},
    "poor eyesight": {"sight": -1},
    "robust": {"strength": 1.2, "healing": 1.2},
    "frail": {"strength": 0.8, "infection": 1.3},
    "iron stomach": {"food_water_infection": 0.3},
    "long-lived": {"lifespan": 1.2},
    "slow clotting": {"bleeding": 2.0},
    "cold-hardy": {"insulation": 1.4},
    "infertile": {"fertile": 0.0},
    "stiff joints": {"speed": 0.75},
    "failing sight": {"sight": -1},
    "lame": {"speed": 0.7, "stumble": 1.5},
    "chronic cough": {"fatigue": 1.3},
    "haunted": {"stress_gain": 1.5},
    "hardened": {"stress_gain": 0.7},
    "enduring": {"walk_cost": 0.85},
    "heavy-handed": {"blow": 1.2},
}
INBORN = [i for i, c in enumerate(CONDITIONS) if c[1] == "inborn"]


def rarity(p: float) -> str:
    """Rarity label from a base chance."""
    return "common" if p >= 0.1 else "uncommon" if p >= 0.04 else "rare" if p >= 0.015 else "very rare"


def table():
    """For the playback: name, kind, good, rarity, description."""
    return [{"name": n, "kind": k, "good": g, "rarity": rarity(p) if k == "inborn" else
             ("with age" if k == "age" else "from what happens" if k == "event" else "earned by use"),
             "base": p, "inherit": h, "about": d} for n, k, g, p, h, _, _, d in CONDITIONS]
