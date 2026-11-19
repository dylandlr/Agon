"""Threat knobs and injected shocks: turn the world's pressures up or down, and inject crises.

Baseline knobs (a whole run; train_phase2 and every script that rebuilds a run's world):

    food       x plant abundance: starting stock and regrowth (the --scarcity dial; 0.4 = scarce)
    predators  x predator numbers and respawning; 0 = predators that never attack
    disease    x every chance of infection: bad water, raw meat, contact with the sick
    hazards    x environmental danger: how often weather turns extreme, how extreme it gets,
               lightning, falls

Shocks (the continuous world and viability.py), KIND:DAY:DAYS:STRENGTH, e.g. famine:40:10:0.2

    famine     food falls to STRENGTH x normal: standing plants are cut to that share at the start
               and regrowth runs at that rate for DAYS
    plague     infection chances x STRENGTH
    predators  predators grow bold: attack chance x STRENGTH (up to certain), chase harder, rest
               less between attacks
    cold       STRENGTH degrees colder everywhere
    heat       STRENGTH degrees hotter everywhere
    drought    thirst x STRENGTH and no rain to catch

Nothing here tells agents anything: a shock changes the world, and they meet it as they meet
any hard season. In the event log a shock is a world event, and the hardships it brings (hunger
in a famine, infection in a plague, attacks, freezing, overheating, thirst) list it as a cause.
"""
from __future__ import annotations

import dataclasses

KNOBS = ("food", "predators", "disease", "hazards")
SHOCKS = ("famine", "plague", "predators", "cold", "heat", "drought")
# the events a shock makes more likely, which name it as a cause while it lasts
SHOCK_EFFECTS = {"famine": ("starving",), "plague": ("infected",), "predators": ("predator_attack",),
                 "cold": ("freezing",), "heat": ("overheating",), "drought": ("parched",)}


def with_threats(cfg, food: float = 1.0, predators: float = 1.0, disease: float = 1.0, hazards: float = 1.0):
    """cfg with the baseline pressures scaled (1 = as designed)."""
    for name, v in (("food", food), ("predators", predators), ("disease", disease), ("hazards", hazards)):
        if v < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
    kw = dict(plant_abundance=cfg.plant_abundance * food,
              raw_meat_infection=min(1.0, cfg.raw_meat_infection * disease),
              water_infection=min(1.0, cfg.water_infection * disease),
              contact_infection=min(1.0, cfg.contact_infection * disease),
              climate_volatility=min(1.0, cfg.climate_volatility * hazards),
              extreme_temp_shift=cfg.extreme_temp_shift * hazards,
              lightning_prob=min(1.0, cfg.lightning_prob * hazards),
              fall_prob=min(1.0, cfg.fall_prob * hazards))
    if predators > 0:
        kw.update(predators_per_1000=cfg.predators_per_1000 * predators,
                  predator_respawn=min(1.0, cfg.predator_respawn * predators))
    else:                     # the world always holds at least one predator: make it harmless
        kw.update(predators_per_1000=0.0, predator_respawn=0.0, predator_attack_prob=0.0)
    return dataclasses.replace(cfg, **kw)


def parse_shock(text: str) -> dict:
    """'famine:40:10:0.2' -> {kind, day, days, strength}."""
    parts = text.split(":")
    if len(parts) != 4 or parts[0] not in SHOCKS:
        raise ValueError(f"shock {text!r}: use KIND:DAY:DAYS:STRENGTH with KIND one of {', '.join(SHOCKS)}")
    kind, day, days, strength = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
    if days <= 0 or strength < 0:
        raise ValueError(f"shock {text!r}: DAYS must be > 0 and STRENGTH >= 0")
    return {"kind": kind, "day": day, "days": days, "strength": strength}


def active(shocks, day: float) -> tuple:
    """Indices of the shocks in force on this (fractional) day."""
    return tuple(i for i, s in enumerate(shocks) if s["day"] <= day < s["day"] + s["days"])


def cfg_during(cfg, shocks, which) -> object:
    """cfg with the given shocks applied."""
    for i in which:
        s = shocks[i]
        k, x = s["kind"], s["strength"]
        if k == "famine":
            cfg = dataclasses.replace(cfg, plant_abundance=cfg.plant_abundance * x)
        elif k == "plague":
            cfg = with_threats(cfg, disease=x)
        elif k == "predators":
            cfg = dataclasses.replace(
                cfg, predator_attack_prob=min(1.0, cfg.predator_attack_prob * x),
                predator_chase_prob=1.0 - (1.0 - cfg.predator_chase_prob) / max(x, 1e-6) if x >= 1 else cfg.predator_chase_prob * x,
                predator_rest_ticks=max(1, int(cfg.predator_rest_ticks / max(x, 1e-6))))
        elif k in ("cold", "heat"):
            d = -x if k == "cold" else x
            cfg = dataclasses.replace(cfg, season_temp=tuple(v + d for v in cfg.season_temp))
        elif k == "drought":
            cfg = dataclasses.replace(cfg, thirst_rate=cfg.thirst_rate * x, rain_drink=0.0)
    return cfg


def describe(s) -> str:
    k, x = s["kind"], s["strength"]
    what = {"famine": f"food at {x:g}x normal", "plague": f"infection {x:g}x as likely",
            "predators": f"predators {x:g}x as aggressive", "cold": f"{x:g} C colder", "heat": f"{x:g} C hotter",
            "drought": f"thirst {x:g}x as fast, no rain to catch"}[k]
    return f"{k} for {s['days']:g} days ({what})"
