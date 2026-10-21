"""Minerals, ores and metals: physical materials with properties, never recipes.

Each is a material like wood or stone: it lies in deposits where geology puts it, can be picked
up, carried, held, dropped, placed on a neighbouring cell and struck with or against. What it does
comes only from its properties, through mechanics the world already has:

- hardness: a blow with a hard thing in hand lands harder; hard struck on hard can spark.
- sharpness: an edge in hand lands harder still and cuts more meat from a carcass and more wood.
- spark: struck against something hard it throws sparks (pyrite against flint is the classic
  fire kit; iron against flint too); sparks on dry fuel can start a fire.
- fuel: burns like wood; coal burns four times as long and hot enough to smelt ore.
- fired / smelted: heat changes some materials where they lie: clay in any fire becomes fired
  clay; ores in a coal-hot fire become metal (copper, tin; iron more slowly); copper and tin
  heated together become bronze.
- preserve: carrying salt slows the spoiling of carried meat.

Agents are told none of this, and do not see names: they see each cell's minerals as colour and
lustre (how they look), and feel what they carry. Ochre (a pigment) and fired clay (a vessel) have
no use yet: there is no way to make a mark or carry water.
"""
from __future__ import annotations

# columns: name, colour (r, g, b), lustre, hardness, sharpness, spark, fuel (ticks per unit),
#          becomes (in fire: product, needs coal-hot heat, rate per tick), preserve, where found
#          (terrain -> chance per cell), amount per deposit (low, high)
_T = ["grass", "forest", "rock", "water", "desert", "tundra", "marsh", "cave"]
MINERALS = [
    ("flint", (0.33, 0.32, 0.31), 0.4, 1.0, 0.8, 0.3, 0.0, None, 0.0, {"rock": 0.06, "grass": 0.01}, (2, 5)),
    ("obsidian", (0.07, 0.06, 0.09), 0.9, 0.7, 1.0, 0.1, 0.0, None, 0.0, {"rock": 0.015}, (1, 3)),
    ("pyrite", (0.76, 0.66, 0.26), 0.9, 0.9, 0.1, 1.0, 0.0, None, 0.0, {"rock": 0.03, "cave": 0.05}, (1, 3)),
    ("clay", (0.66, 0.43, 0.31), 0.1, 0.0, 0.0, 0.0, 0.0, ("fired clay", False, 0.05), 0.0,
     {"marsh": 0.15, "grass": 0.01}, (2, 6)),
    ("fired clay", (0.76, 0.41, 0.26), 0.2, 0.5, 0.3, 0.0, 0.0, None, 0.0, {}, (0, 0)),
    ("ochre", (0.70, 0.26, 0.12), 0.05, 0.1, 0.0, 0.0, 0.0, None, 0.0, {"desert": 0.04, "rock": 0.02}, (1, 4)),
    ("salt", (0.95, 0.95, 0.92), 0.5, 0.2, 0.0, 0.0, 0.0, None, 0.75, {"desert": 0.06, "marsh": 0.02}, (2, 5)),
    ("coal", (0.12, 0.12, 0.12), 0.3, 0.3, 0.0, 0.0, 160.0, None, 0.0, {"rock": 0.03, "forest": 0.005}, (2, 6)),
    ("copper ore", (0.20, 0.55, 0.45), 0.3, 0.4, 0.0, 0.0, 0.0, ("copper", True, 0.05), 0.0, {"rock": 0.02}, (1, 4)),
    ("tin ore", (0.30, 0.25, 0.22), 0.6, 0.8, 0.0, 0.0, 0.0, ("tin", True, 0.05), 0.0, {"rock": 0.01}, (1, 3)),
    ("iron ore", (0.56, 0.31, 0.20), 0.3, 0.6, 0.0, 0.0, 0.0, ("iron", True, 0.02), 0.0,
     {"rock": 0.02, "marsh": 0.03}, (1, 4)),
    ("copper", (0.80, 0.46, 0.26), 1.0, 0.5, 0.6, 0.0, 0.0, None, 0.0, {}, (0, 0)),
    ("tin", (0.76, 0.76, 0.76), 0.9, 0.2, 0.1, 0.0, 0.0, None, 0.0, {}, (0, 0)),
    ("bronze", (0.70, 0.51, 0.26), 1.0, 0.9, 0.9, 0.0, 0.0, None, 0.0, {}, (0, 0)),
    ("iron", (0.40, 0.40, 0.42), 0.7, 1.0, 0.9, 0.8, 0.0, None, 0.0, {}, (0, 0)),
    # only underground (cfg.underground), so the surface is as before: see CAVE_DEPOSITS for where each cave formation holds them
    ("limestone", (0.82, 0.80, 0.72), 0.1, 0.4, 0.0, 0.0, 0.0, ("quicklime", True, 0.03), 0.0, {}, (2, 5)),
    ("quicklime", (0.97, 0.96, 0.93), 0.05, 0.1, 0.0, 0.0, 0.0, None, 0.0, {}, (0, 0)),
    ("quartz", (0.92, 0.92, 0.96), 0.85, 0.9, 0.6, 0.4, 0.0, None, 0.0, {}, (1, 3)),
    ("sulfur", (0.93, 0.85, 0.20), 0.3, 0.1, 0.0, 0.0, 40.0, None, 0.0, {}, (1, 3)),
    ("gypsum", (0.95, 0.93, 0.88), 0.4, 0.1, 0.0, 0.0, 0.0, None, 0.0, {}, (2, 4)),
    ("saltpetre", (0.97, 0.97, 0.95), 0.2, 0.1, 0.0, 0.0, 0.0, None, 0.5, {}, (0, 0)),
    ("galena", (0.45, 0.47, 0.52), 0.95, 0.3, 0.0, 0.0, 0.0, ("lead", False, 0.03), 0.0, {}, (1, 3)),
    ("lead", (0.55, 0.56, 0.60), 0.6, 0.2, 0.1, 0.0, 0.0, None, 0.0, {}, (0, 0)),
    ("gold", (1.00, 0.82, 0.20), 1.0, 0.2, 0.1, 0.0, 0.0, None, 0.0, {}, (1, 2)),
    ("basalt", (0.20, 0.20, 0.23), 0.2, 0.9, 0.2, 0.1, 0.0, None, 0.0, {}, (2, 5)),
]
NAMES = [m[0] for m in MINERALS]
M = len(MINERALS)
IDX = {n: i for i, n in enumerate(NAMES)}
COLOR = [m[1] for m in MINERALS]
LUSTER = [m[2] for m in MINERALS]
HARDNESS = [m[3] for m in MINERALS]
SHARPNESS = [m[4] for m in MINERALS]
SPARK = [m[5] for m in MINERALS]
FUEL = [m[6] for m in MINERALS]
PRESERVE = [m[8] for m in MINERALS]
# heat transforms: (from, to, needs coal-hot fire, rate)
FIRING = [(i, IDX[m[7][0]], m[7][1], m[7][2]) for i, m in enumerate(MINERALS) if m[7]]
ALLOY = (IDX["copper"], IDX["tin"], IDX["bronze"], 3.0)   # 3 copper + 1 tin -> 4 bronze, in coal-hot fire
DEPOSIT_CHANCE = [[m[9].get(t, 0.0) for t in _T] for m in MINERALS]   # (M, N_TERRAIN)
DEPOSIT_AMOUNT = [m[10] for m in MINERALS]

# Cave formations (cfg.underground): each cave system takes its kind from the land at its mouth,
# and its walls hold minerals by kind (chance per passage cell; amounts as above, for minerals
# only made elsewhere, 1-3).
#   karst       limestone dissolved by water: wide chambers, many pools, limestone, some galena;
#               a quarter of its chambers are geodes lined with quartz (and now and then gold)
#   lava tube   high volcanic ground: long straight tubes, no pools, basalt, obsidian, sulfur,
#               warm air; hot springs bubble up beside it (warm, drinkable, sulfur at the edge)
#   ice cave    cold north: pools frozen but breakable, air near freezing all year
#   salt cave   desert: salt and gypsum, no water at all, dry air
KINDS = ["none", "karst", "lava tube", "ice cave", "salt cave", "geode"]
CAVE_DEPOSITS = {
    "karst": {"limestone": 0.25, "galena": 0.03, "flint": 0.04, "pyrite": 0.03, "clay": 0.04, "ochre": 0.03},
    "lava tube": {"basalt": 0.3, "obsidian": 0.08, "sulfur": 0.08, "pyrite": 0.03, "copper ore": 0.03},
    "ice cave": {"flint": 0.05, "iron ore": 0.04, "tin ore": 0.03, "coal": 0.04},
    "salt cave": {"salt": 0.3, "gypsum": 0.2, "ochre": 0.04},
    "geode": {"quartz": 0.5, "gold": 0.04, "pyrite": 0.05},
}
CAVE_CHANCE = [[CAVE_DEPOSITS.get(k, {}).get(n, 0.0) for n in [m[0] for m in MINERALS]] for k in KINDS]   # (K, M)


def table():
    """For the playback: what each mineral is and does."""
    uses = {"flint": "hard, takes a sharp edge; sparks a little against stone",
            "obsidian": "volcanic glass: the sharpest edge, brittle",
            "pyrite": "struck against something hard (flint) it throws strong sparks",
            "clay": "soft; in a fire it hardens into fired clay",
            "fired clay": "hard, brittle shards; could hold water, but nothing carries water yet",
            "ochre": "red earth pigment; no use yet (there is no way to make a mark)",
            "salt": "carried with meat, slows its spoiling",
            "coal": "burns four times as long as wood, hot enough to smelt ore",
            "copper ore": "in a coal-hot fire becomes copper",
            "tin ore": "in a coal-hot fire becomes tin",
            "iron ore": "in a coal-hot fire slowly becomes iron",
            "copper": "soft metal with an edge; with tin in a coal-hot fire becomes bronze",
            "tin": "soft metal; with copper in a coal-hot fire becomes bronze",
            "bronze": "hard metal that holds a sharp edge",
            "iron": "the hardest metal, sharp; strikes sparks from flint",
            "limestone": "soft pale stone of karst caves; in a coal-hot fire it burns to quicklime",
            "quicklime": "burnt lime; with water it would make plaster or mortar (no use yet)",
            "quartz": "clear crystal from geodes; hard, takes an edge, sparks against iron",
            "sulfur": "yellow, from lava tubes and hot springs; burns with a choking smoke",
            "gypsum": "soft white stone of salt caves; no use yet",
            "saltpetre": "left by roosting bats; carried with meat, cures it against spoiling",
            "galena": "heavy, shining lead ore; even a campfire smelts it to lead",
            "lead": "soft heavy metal; no use yet",
            "gold": "rare, soft, shining; no use but what people may come to give it",
            "basalt": "dark hard lava stone; heavy and tough"}
    return [{"name": n, "color": "#%02x%02x%02x" % tuple(int(255 * x) for x in COLOR[i]), "lustre": LUSTER[i],
             "hardness": HARDNESS[i], "sharpness": SHARPNESS[i], "about": uses[n],
             "found": ", ".join(f"{t} {100 * p:g}%" for t, p in MINERALS[i][9].items()) or "only made, never found"}
            for i, n in enumerate(NAMES)]
