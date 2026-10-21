"""Animals and bugs: species with the bodies and habits of their kind.

Three groups share the world's animal slots, as before: small prey, large prey and predators.
Each animal now belongs to a species (fixed for its slot; a slot's next animal is the same
kind), and the species sets its body and its habits. Nothing here tells agents anything: they
see an animal's size, its colour and whether it has fangs or tusks, and learn the rest.

Habits, all from mechanics the world already has or adds here:
- flee / chase: notice range and how often it acts on what it notices; speed 2 runs two cells.
- habitat: where it can be at all (fish and crocodiles stay in water, the rest on land).
- herd: when not fleeing it drifts toward its own kind.
- night: nocturnal hunters are far more active in the dark.
- evade: a share of blows simply miss (birds take off, fish dart); an edge in hand (a spear
  point) cuts the miss chance.
- fights back: struck, it may turn on the striker (aurochs and mammoth trample, boar gores,
  bear mauls).
- pack: each extra pack member next to the prey cancels one defender standing with it.
- ambush: barely notices from afar but strikes hard when adjacent (big cats from cover).
- provoked only: attacks only when struck first or startled at close range (bears).
- hibernates: inactive in winter (bears).
- venom: a bite adds poison damage and lasting pain (snakes).
- byproducts: a kill leaves hide, bone and feathers by species (BYPRODUCTS), the raw stuff of
  cloaks, beds and points; fowl also moult feathers where they live.
- caves (cfg.underground): any species whose habitat includes cave can go underground at a
  mouth. Predators follow a quarry that ducks into a cave close by (scent reaches 2 cells across
  the layers); prey that is fleeing may bolt down a mouth; denning species keep to the caves at
  their time (bats roost below by day and fly at night, bears den below in winter). Bats exist
  only with caves (cave_only: no share of the slots otherwise), and roosting bats leave guano.

Bugs are fields, not individuals: mosquitoes (marsh and water edges in the warm seasons, worse at
night, driven off by smoke) carry fever; flies gather on meat and bodies, speeding rot and
sickness; bees keep hives of honey in woods and sting whoever robs them unless smoke calms them;
termites build mounds of edible grubs in grass and desert, which a stick pushed in draws out
far better than fingers.
"""
from __future__ import annotations

SMALL, LARGE, PRED = 0, 1, 2
GROUPS = ["small_prey", "large_prey", "predator"]
_T = ["grass", "forest", "rock", "water", "desert", "tundra", "marsh", "cave"]
_LAND = ("grass", "forest", "rock", "desert", "tundra", "marsh", "cave")

# name, group, share of the group's slots, habitat (terrains it can be on), hp, armour, meat,
# notice, act (flee or chase chance), speed, herd, nocturnal, evade, counter (chance, damage),
# attack (chance, damage, injury, rest ticks), pack, ambush, provoked_only, hibernates, venom,
# look (size 0..1, fangs 0..1, colour 0..1)
SPECIES = [
    # small prey
    dict(name="rabbit", group=SMALL, share=0.45, habitat=("grass", "forest", "desert", "tundra"), hp=0.5, armor=0.0,
         meat=1.0, notice=2, act=0.7, speed=1, evade=0.0, look=(0.1, 0.0, 0.55)),
    dict(name="fowl", group=SMALL, share=0.25, habitat=("grass", "forest", "marsh"), hp=0.4, armor=0.0,
         meat=0.7, notice=3, act=0.8, speed=2, evade=0.5, look=(0.1, 0.0, 0.3)),
    dict(name="fish", group=SMALL, share=0.2, habitat=("water",), hp=0.3, armor=0.0,
         meat=0.8, notice=1, act=0.5, speed=1, evade=0.6, look=(0.1, 0.0, 0.75)),
    dict(name="tortoise", group=SMALL, share=0.1, habitat=("grass", "desert"), hp=0.8, armor=0.5,
         meat=1.2, notice=1, act=0.1, speed=1, evade=0.0, look=(0.15, 0.0, 0.4)),
    dict(name="bat", group=SMALL, share=0.15, habitat=("cave", "forest", "rock"), hp=0.2, armor=0.0,
         meat=0.2, notice=2, act=0.9, speed=2, evade=0.7, nocturnal=1.0, den=1, cave_only=1,
         look=(0.05, 0.0, 0.15)),
    # large prey
    dict(name="deer", group=LARGE, share=0.35, habitat=("grass", "forest", "tundra"), hp=3.0, armor=0.8,
         meat=6.0, notice=2, act=0.35, speed=2, herd=0.3, look=(0.5, 0.0, 0.6)),
    dict(name="aurochs", group=LARGE, share=0.2, habitat=("grass", "marsh", "forest"), hp=5.0, armor=1.2,
         meat=12.0, notice=2, act=0.2, speed=1, herd=0.6, counter=(0.4, 0.15), look=(0.8, 0.3, 0.2)),
    dict(name="boar", group=LARGE, share=0.2, habitat=("forest", "marsh", "grass"), hp=3.0, armor=0.8,
         meat=6.0, notice=2, act=0.3, speed=1, counter=(0.6, 0.2), look=(0.5, 0.5, 0.25)),
    dict(name="horse", group=LARGE, share=0.15, habitat=("grass", "desert", "tundra"), hp=3.0, armor=0.6,
         meat=8.0, notice=3, act=0.6, speed=2, herd=0.6, look=(0.65, 0.0, 0.45)),
    dict(name="mammoth", group=LARGE, share=0.1, habitat=("tundra", "grass"), hp=10.0, armor=2.0,
         meat=30.0, notice=2, act=0.15, speed=1, herd=0.7, counter=(0.3, 0.3), look=(1.0, 0.6, 0.35)),
    # predators
    dict(name="wolf", group=PRED, share=0.4, habitat=_LAND, hp=1.5, armor=0.3, meat=2.0, notice=6, act=0.85,
         speed=1, nocturnal=1.0, attack=(0.3, 0.12, 0.15, 150), pack=1, takes_large=1, look=(0.4, 1.0, 0.5)),
    dict(name="big cat", group=PRED, share=0.2, habitat=("forest", "grass", "rock"), hp=2.0, armor=0.4, meat=3.0,
         notice=3, act=0.7, speed=2, nocturnal=0.5, attack=(0.6, 0.18, 0.2, 250), ambush=1, takes_large=1,
         look=(0.5, 1.0, 0.7)),
    dict(name="bear", group=PRED, share=0.15, habitat=("forest", "tundra", "cave", "rock"), hp=4.0, armor=0.8,
         meat=10.0, notice=3, act=0.5, speed=1, attack=(0.5, 0.2, 0.25, 200), provoked=1, hibernates=1, den=1,
         counter=(0.8, 0.2), takes_large=1, look=(0.8, 1.0, 0.25)),
    dict(name="snake", group=PRED, share=0.15, habitat=("grass", "desert", "rock", "marsh"), hp=0.4, armor=0.0,
         meat=0.4, notice=1, act=0.05, speed=1, attack=(0.3, 0.06, 0.02, 100), venom=0.12, look=(0.15, 1.0, 0.45)),
    dict(name="crocodile", group=PRED, share=0.1, habitat=("water", "marsh"), hp=3.5, armor=1.0, meat=8.0,
         notice=2, act=0.6, speed=1, attack=(0.4, 0.25, 0.25, 300), ambush=1, takes_large=1, look=(0.7, 1.0, 0.3)),
]
# What a carcass leaves besides meat (units of lifesim/minerals.py organics): hide (skin and fur),
# bone, feathers. A tortoise's "bone" is its shell; a fish leaves bones and scales.
BYPRODUCTS = {'rabbit': (0.5, 0.3, 0), 'fowl': (0, 0.2, 3), 'fish': (0, 0.2, 0), 'tortoise': (0, 1.0, 0), 'bat': (0, 0, 0), 'deer': (2, 2, 0), 'aurochs': (3, 3, 0), 'boar': (2, 2, 0), 'horse': (2, 2, 0), 'mammoth': (4, 6, 0), 'wolf': (1, 1, 0), 'big cat': (1.5, 1, 0), 'bear': (3, 2, 0), 'snake': (0.3, 0.1, 0), 'crocodile': (2, 2, 0)}
for s in SPECIES:
    s["hide"], s["bone"], s["feather"] = BYPRODUCTS[s["name"]]
# How readily a species comes to trust people who feed it and do it no harm (x every bond gain;
# EnvConfig.taming): wolves (scavengers at camps, the first animal kept) fastest; boar, fowl, horses
# and aurochs (all once domesticated) less; deer seldom; big cats, bears and mammoths hardly;
# fish, bats, snakes and crocodiles never.
TAMEABLE = {'rabbit': 0.4, 'fowl': 0.7, 'fish': 0.0, 'tortoise': 0.2, 'bat': 0.0, 'deer': 0.25, 'aurochs': 0.5,
            'boar': 0.7, 'horse': 0.6, 'mammoth': 0.1, 'wolf': 1.0, 'big cat': 0.15, 'bear': 0.1, 'snake': 0.0,
            'crocodile': 0.0}
for s in SPECIES:
    s["tameable"] = TAMEABLE[s["name"]]
for s in SPECIES:                       # defaults
    for k, v in dict(herd=0.0, nocturnal=0.0, evade=0.0, counter=(0.0, 0.0), attack=(0.0, 0.0, 0.0, 0), pack=0,
                     ambush=0, provoked=0, hibernates=0, venom=0.0, takes_large=0, den=0, cave_only=0).items():
        s.setdefault(k, v)

NAMES = [s["name"] for s in SPECIES]
S = len(SPECIES)
IDX = {n: i for i, n in enumerate(NAMES)}
IN_GROUP = [[i for i, s in enumerate(SPECIES) if s["group"] == g] for g in range(3)]
HABITAT = [[t in s["habitat"] for t in _T] for s in SPECIES]                     # (S, N_TERRAIN)


def column(key, i=None):
    """A per-species parameter as a list (or element i of a tuple-valued one)."""
    return [s[key] if i is None else s[key][i] for s in SPECIES]


def table():
    """For the playback: what each species is and does."""
    habits = {"rabbit": "flees fast", "fowl": "flies off (blows often miss), runs two cells",
              "fish": "lives in water; struck from the bank, most blows miss without a point",
              "tortoise": "slow, shelled", "deer": "bounds away two cells, herds loosely",
              "aurochs": "herds; struck, may trample the striker", "boar": "struck, often gores the striker",
              "horse": "herds, runs two cells, wary", "mammoth": "huge, herds; struck, may trample; feeds a band",
              "wolf": "hunts at night in packs: each extra wolf cancels one defender",
              "big cat": "ambushes from cover: notices late, strikes hard",
              "bear": "attacks only when struck or startled close by; hibernates in winter; mauls back",
              "snake": "barely moves; a bite is venomous",
              "bat": "roosts in caves by day, flies at night; most blows miss; its guano is saltpetre", "crocodile": "lies in water: strikes whoever stands at the edge"}
    return [{"name": s["name"], "group": GROUPS[s["group"]], "size": s["look"][0], "fangs": s["look"][1],
             "colour": s["look"][2], "hp": s["hp"], "meat": s["meat"], "about": habits[s["name"]],
             "lives": ", ".join(s["habitat"])} for s in SPECIES]


BUGS = {"mosquitoes": "marsh and water edges in warm seasons, worse at night; carry fever; smoke drives them off",
        "flies": "gather on meat and bodies; meat rots faster and sickens more",
        "bees": "hives of honey in woods; sting whoever robs them, unless smoke calms them",
        "termites": "mounds of edible grubs in grass and desert; a stick pushed in draws out far more"}
