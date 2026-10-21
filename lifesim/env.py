"""Core world simulation. Pure, jittable, deterministic given the PRNG key.

Mechanics: terrain with rock falls, seasonal plants, Markov weather (rain,
drought, cold snaps, heat waves), ambient temperature and body temperature,
hunger/thirst/fatigue, injury, disease with immunity, stress and trauma-driven
temperament drift, aging, small prey, large prey that need group hunting,
predators deterred by groups, meat that spoils.

All agents act in the same tick; conflicts resolve with seeded randomness.
Nothing here scripts social behavior: strikes hit whatever is in the target
cell, including other agents.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import conditions as cond
from . import minerals as mn
from . import fauna as fa
from . import genotype as gt
from . import ecology as eco
from .config import EnvConfig

# ---------------------------------------------------------------- terrain
GRASS, FOREST, ROCK, WATER, DESERT, TUNDRA, MARSH, CAVE = range(8)
TERRAIN_NAMES = ["grass", "forest", "rock", "water", "desert", "tundra", "marsh", "cave"]
N_TERRAIN = 8
TERRAIN_MOVE_COST = jnp.array([1.0, 1.3, 2.0, 0.0, 1.1, 1.4, 1.8, 1.0])

# ---------------------------------------------------------------- weather
CLEAR, RAIN, DROUGHT, COLD_SNAP, HEAT_WAVE = range(5)
N_WEATHER = 5
WEATHER_NAMES = ["clear", "rain", "drought", "cold_snap", "heat_wave"]

# ---------------------------------------------------------------- actions
REST = 0
MOVE_N, MOVE_NE, MOVE_E, MOVE_SE, MOVE_S, MOVE_SW, MOVE_W, MOVE_NW = range(1, 9)
GRASP, EAT, DRINK = 9, 10, 11
STRIKE_N = 12                     # STRIKE_N .. STRIKE_NW = 12 .. 19
RELEASE = 20                      # drop one unit of carried food on the current cell
TOUCH_N = 21                      # TOUCH_N .. TOUCH_NW = 21 .. 28
SELECT_0 = 29                     # 29..60: put material k in hand (N_MAT = 32 materials)
PLACE_N = SELECT_0 + 7 + mn.M     # 61..68: place one unit of the hand material on a neighbor cell
MATE = PLACE_N + 8                # 69: mate with an adjacent partner who chooses the same (mutual)
SLEEP = MATE + 1                  # 70: fall or stay asleep (asleep, REST means waking up)
SEEK_WATER, SEEK_FOOD, FLEE = SLEEP + 1, SLEEP + 2, SLEEP + 3   # 71-73: given skills (cfg.skills)
SEEK_SHELTER = SLEEP + 4                                         # 74: to known forest or cave, then rest
N_SKILLS = 4
DELVE = SEEK_SHELTER + 1          # 75: on a cave mouth, go down into the caves, or back up (cfg.underground)
RELIEVE = DELVE + 1               # 76: relieve oneself where one stands (cfg.excretion)
WRITE = RELIEVE + 1               # 77: scratch one's message into the ground here (cfg.writing)
N_ACTIONS = WRITE + 1
LAYER_OFF = 4096                  # added to a below agent's column in layered_pos: no one above is ever near

# ---------------------------------------------------------------- materials
# Agents see materials as distinct channels; they are never told what any of
# them is for. Everything a material does comes from these physical properties.
PLANT, RAW_MEAT, COOKED_MEAT, WOOD, STONE, HONEY, GRUBS = range(7)
N_BASE_MAT = 7
MIN0 = N_BASE_MAT                 # minerals, ores and metals follow (lifesim/minerals.py)
N_MAT = MIN0 + mn.M
MAT_NAMES = ["plant", "raw_meat", "cooked_meat", "wood", "stone", "honey", "grubs"] \
    + [x.replace(" ", "_") for x in mn.NAMES]
EDIBLE = jnp.zeros(N_MAT, bool).at[jnp.array([PLANT, RAW_MEAT, COOKED_MEAT, HONEY, GRUBS])].set(True)
_BASE0 = [0.0] * N_BASE_MAT
HARDNESS = jnp.array([0.0, 0.0, 0.0, 0.3, 1.0, 0.0, 0.0] + mn.HARDNESS)   # hard on hard makes sparks; hard tools hit harder
SHARPNESS = jnp.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0] + mn.SHARPNESS)  # an edge hits harder and cuts more
# Writing needs a tool and something to write on. A mark is cut with an edge (sharp enough; on stone
# only with a point at least as hard as stone, flint, quartz, obsidian or metal) or drawn with a
# pigment (ochre, charcoal-like coal). What takes a mark: a piece of wood lying here (put down by
# someone, it stands as a sign), a loose stone, a rock face, or the bark of a tree. Bare earth, sand,
# snow and water take none. A mark goes with its surface: carry off or burn the wood, and it is gone.
PIGMENT = jnp.zeros(N_MAT).at[MIN0 + mn.IDX["ochre"]].set(1.0).at[MIN0 + mn.IDX["coal"]].set(1.0)
SURF_NONE, SURF_WOOD, SURF_STONE, SURF_ROCK, SURF_BARK = range(5)
SURFACE_NAMES = ["", "wood", "a stone", "rock", "bark"]
SURF_HARD = jnp.array([9.0, 0.0, 1.0, 1.0, 0.0])            # how hard the cutting point must be (scaled by cfg)
MARK_PAINTED = 8                                            # mark_kind = surface + 8 when drawn in pigment
SPARK = jnp.array(_BASE0 + mn.SPARK)                                       # throws sparks struck against something hard
MIN_FUEL = jnp.array(mn.FUEL)                                    # (M,) ticks of burning per unit (coal)
MIN_COLOR = jnp.array(mn.COLOR)                                  # (M, 3) how each looks
MIN_LUSTER = jnp.array(mn.LUSTER)
# Organic byproducts (lifesim/minerals.py ORGANIC), materials like any other
LEAVES, FIBER, FEATHER, HIDE, BONE, SPEAR = (MIN0 + mn.IDX[k] for k in ("leaves", "fiber", "feather", "hide", "bone", "spear"))
INSULATION = jnp.array(_BASE0 + mn.INSULATION)                   # (N_MAT,) warmth of one carried unit
MIN_BEDDING = jnp.array(mn.BEDDING)                              # (M,) of one unit lying where one sleeps
MIN_TINDER = jnp.array(mn.TINDER)                                # (M,) how readily a spark catches on it
MIN_ROT = jnp.array(mn.ROT)                                      # (M,) share lost per 30-minute tick, lying out
MIN_SHARP = jnp.array(mn.SHARPNESS)
N_MINERAL_CHANNELS = 5            # per cell: minerals present, and their colour (r, g, b) and lustre
N_FAUNA_CHANNELS = 7              # per cell: animals' size, fangs, colour; mosquitoes, flies, hive, mound
N_FLORA_CHANNELS = 6              # per cell: how well greens, fruit, nuts, roots would gather here; waste; an inscription
assert PLACE_N == SELECT_0 + N_MAT, "one select action per material"
_DIRS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
ACTION_NAMES = (["rest"] + [f"move_{d}" for d in _DIRS] + ["grasp", "eat", "drink"]
                + [f"strike_{d}" for d in _DIRS] + ["release"]
                + [f"touch_{d}" for d in _DIRS]
                + [f"select_{m}" for m in MAT_NAMES]
                + [f"place_{d}" for d in _DIRS] + ["mate", "sleep", "seek_water", "seek_food", "flee", "seek_shelter"]
                + ["delve", "relieve", "write"])

MOVE_DELTAS = jnp.array([[0, 0], [-1, 0], [-1, 1], [0, 1], [1, 1],
                         [1, 0], [1, -1], [0, -1], [-1, -1]], dtype=jnp.int32)
NEIGHBORS8 = MOVE_DELTAS[1:]

# ---------------------------------------------------------------- outcomes
(OUT_NONE, OUT_REST, OUT_MOVE, OUT_GATHER, OUT_GATHER_LOST, OUT_EAT, OUT_DRINK,
 OUT_STRIKE_MISS, OUT_STRIKE_ANIMAL, OUT_STRIKE_AGENT, OUT_KILL, OUT_MOVE_FAILED,
 OUT_RELEASE, OUT_TOUCH, OUT_TOKEN, OUT_SELECT, OUT_PLACE, OUT_MATE, OUT_SLEEP, OUT_RELIEVE, OUT_WRITE) = range(21)
OUTCOME_NAMES = ["none", "rest", "move", "gather", "gather_lost", "eat", "drink",
                 "strike_miss", "strike_animal", "strike_agent", "kill", "move_failed",
                 "release", "touch", "token", "select", "place", "mate", "sleep", "relieve", "write"]

# Death causes: index 1.. match the damage component order in step()
CAUSE_NAMES = ["none", "starvation", "thirst", "exhaustion", "cold", "heat",
               "disease", "predation", "violence", "injury", "burns", "fall", "childbirth", "age"]
(CAUSE_NONE, CAUSE_STARVATION, CAUSE_THIRST, CAUSE_EXHAUSTION, CAUSE_COLD, CAUSE_HEAT,
 CAUSE_DISEASE, CAUSE_PREDATION, CAUSE_VIOLENCE, CAUSE_INJURY, CAUSE_BURNS, CAUSE_FALL,
 CAUSE_CHILDBIRTH, CAUSE_AGE) = range(14)
FEMALE, MALE = 0, 1

ANIMAL_SMALL, ANIMAL_LARGE, ANIMAL_PREDATOR = 0, 1, 2

SEASON_REGROW = (1.0, 0.8, 0.5)

# Relationship memory: what observer i has seen agent j do (decayed counts)
MEM_STRUCK_ME, MEM_GAVE_NEAR_ME, MEM_TOUCHED_ME, MEM_SEEN_VIOLENT = range(4)
# Feelings that reward an act fade when the act is repeated with the same partner and
# recover with time apart (habituation). Acts are never limited, only how rewarding they feel.
HAB_REPAY, HAB_TOUCH, HAB_REVENGE, HAB_INTIMACY = range(4)
N_HABIT = 4
N_MEM = 4
MEM_NAMES = ["struck_me", "gave_near_me", "touched_me", "seen_violent"]
N_APPEARANCE = 3
# ---------------------------------------------------------------- language
# Every agent emits a message each tick on a channel separate from its action:
# MSG_LEN symbols from a vocabulary of VOCAB, symbol 0 = silence; a first symbol of 0
# means silence. Listeners hear the N_HEARD nearest speakers within view.
# Emergent mode (cfg.language_mode "emergent", the default): no symbol has a built-in
# meaning. Grounded mode ("grounded"): every symbol is a word of LEXICON, and a phrase
# naming water or food with a direction (and here / near / far) is understood: it puts
# that place, as heard, into each listener's remembered map ("no water east" clears it).
# What to say, when, to whom, and whom to believe is left to learning.
MSG_LEN = 4
VOCAB = 32
# What a brain puts out besides its action: MSG_LEN symbols said and MSG_LEN symbols written (the
# text WRITE cuts or draws when it fires; chosen apart from what is said).
OUT_TOKENS = 2 * MSG_LEN
OUT_LOGITS = OUT_TOKENS * VOCAB
# Body parts a blow can be aimed at (people and animals alike; for animals arms are forelegs).
PARTS = ("head", "neck", "torso", "arms", "legs")
N_PARTS = len(PARTS)
PART_HEAD, PART_NECK, PART_TORSO, PART_ARMS, PART_LEGS = range(N_PARTS)
# How a strike is delivered (EnvConfig.techniques): with what (the limb) and to what end (the mode).
# No named moves: a punch is hand+strike, a kick foot+strike, a shove hand+push, a grab hand+hold;
# what they do comes from the physics in step(), and sequences (grab then push: a throw; trip then
# strike) are found, not given. The recorder names them after the fact (technique_name).
LIMBS = ("hand", "foot", "head", "body")
N_LIMBS = len(LIMBS)
LIMB_HAND, LIMB_FOOT, LIMB_HEAD, LIMB_BODY = range(N_LIMBS)
MODES = ("strike", "push", "hold")
N_MODES = len(MODES)
MODE_STRIKE, MODE_PUSH, MODE_HOLD = range(N_MODES)
N_HOW = N_PARTS * N_LIMBS * N_MODES     # one joint choice: part + N_PARTS * (limb + N_LIMBS * mode)


def how_of(part, limb=0, mode=0):
    """The joint index of (part aimed at, limb, mode); a hand strike at part k is k."""
    return part + N_PARTS * (limb + N_LIMBS * mode)


def how_parts(how):
    """Joint index -> (part, limb, mode)."""
    return how % N_PARTS, (how // N_PARTS) % N_LIMBS, how // (N_PARTS * N_LIMBS)


# Per tick a brain gives: an action; how a strike is delivered (the part aimed at, limb and mode,
# one joint choice of N_HOW); MSG_LEN symbols said and MSG_LEN written. Packed as ints:
# out = [said..., written..., how] (OUT_SLOTS), logits as
# [action (N_ACTIONS) | how (N_HOW) | symbols (OUT_TOKENS x VOCAB)].
OUT_SLOTS = OUT_TOKENS + 1
EXTRA_LOGITS = N_HOW + OUT_LOGITS
LEXICON = ["", "water", "food", "meat", "fire", "wood", "stone", "shelter", "predator", "prey", "waste",
           "north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest",
           "here", "near", "far", "come", "go away", "give", "help", "follow", "yes", "no", "me", "you", "many"]
assert len(LEXICON) == VOCAB
W_WATER, W_FOOD = 1, 2
W_DIR0 = 11                       # 11..18: north .. northwest, in MOVE_DELTAS order
W_HERE, W_NEAR, W_FAR, W_NO = 19, 20, 21, 28
N_HEARD = 3
HEARD_FEATURES = N_HEARD * (MSG_LEN * VOCAB + 2 + N_MEM)
READ_FEATURES = MSG_LEN * VOCAB + 2   # an inscription on one's own cell or beside it, and where it is
# Per visible agent: memory about it, its appearance, whether it spoke last tick
N_SOCIAL_FEATURES = N_MEM + N_APPEARANCE + 1 + 8 + 1 + 1   # ... + height, weight, scars, lasting injury, sex, pregnant, asleep, courting, attachment, wincing

# Observation: terrain one-hot, plants, meat, agents, small prey, large prey,
# predators, out-of-bounds, then social features of agents in each cell
# Material and fire channels: cooked meat, wood, stone on the ground, burning, heat
N_OBS_CHANNELS = N_TERRAIN + 7 + 8 + 2 + N_MINERAL_CHANNELS + N_FAUNA_CHANNELS + N_FLORA_CHANNELS \
    + N_SOCIAL_FEATURES   # ... minerals, fauna, flora, social
CH_BODIES = N_TERRAIN + 12
CH_CLIFF = N_TERRAIN + 13
CH_HEIGHT = N_TERRAIN + 14
CH_RAIN = N_TERRAIN + 15
CH_SNOW = N_TERRAIN + 16
CH_MINERALS = N_TERRAIN + 17      # N_MINERAL_CHANNELS channels, then fauna, then the social features
CH_FAUNA = CH_MINERALS + N_MINERAL_CHANNELS
CH_FLORA = CH_FAUNA + N_FAUNA_CHANNELS
OWN_FEATURES = 25 + N_PARTS + 2 + cond.K   # own body (felt pain, heartache, damage to each part, held, off balance) and conditions
SHARED_START = OWN_FEATURES + N_MAT + N_MAT + 1    # after inventory, hand and torch: time and weather
N_BODY_FEATURES = OWN_FEATURES + N_MAT + N_MAT + 1 + 5 + N_WEATHER + HEARD_FEATURES + READ_FEATURES


# ---------------------------------------------------------------- state
class Animals(NamedTuple):
    pos: jnp.ndarray      # (M, 2) int32
    below: jnp.ndarray    # (M,) bool: underground (cfg.underground)
    alive: jnp.ndarray    # (M,) bool
    hp: jnp.ndarray       # (M,) float32
    rest: jnp.ndarray     # (M,) int32 ticks of not hunting (predators only)
    species: jnp.ndarray  # (M,) int32 index into lifesim/fauna.py SPECIES (fixed for the slot)
    anger: jnp.ndarray    # (M,) int32 ticks left of being provoked (struck)
    lame: jnp.ndarray     # (M,) int32 ticks left lame from a blow to a leg (EnvConfig.body_parts)


class State(NamedTuple):
    key: jax.Array
    t: jnp.ndarray
    terrain: jnp.ndarray        # (G, G) int32
    height: jnp.ndarray         # (G, G) in [0, 1]: ground height (2.5D)
    cliff: jnp.ndarray          # (G, G) in [0, 1]: each cell's drop to its lowest neighbor
    plants: jnp.ndarray         # (G, G) plant food here: loose (dropped) + standing crop (flora, with ecology)
    meat: jnp.ndarray           # (G, G) raw meat lying on the ground
    cooked: jnp.ndarray         # (G, G) cooked meat on the ground
    wood: jnp.ndarray           # (G, G) dead wood on the ground
    stones: jnp.ndarray         # (G, G) loose stones on the ground (rock cells hold unlimited)
    minerals: jnp.ndarray       # (G, G, M) minerals, ores and metals lying here (lifesim/minerals.py)
    kiln: jnp.ndarray           # (G, G) ticks of coal-hot heat left in a burning cell (smelting heat)
    mutations: jnp.ndarray      # (N, n_traits) de novo mutations at birth, in population SDs (0 = none)
    honey: jnp.ndarray          # (G, G) honey lying here (in a hive, or dropped)
    grubs: jnp.ndarray          # (G, G) grubs lying here (in a termite mound, or dropped)
    hive: jnp.ndarray           # (G, G) bool: a bees' hive is here
    mound: jnp.ndarray          # (G, G) bool: a termite mound is here
    fire: jnp.ndarray           # (G, G) remaining fuel of a burning cell, in ticks; 0 = not burning
    weather: jnp.ndarray        # scalar int32
    clouds: jnp.ndarray         # (G, G) cloud cover 0..1 (dynamic weather)
    snow: jnp.ndarray           # (G, G) snow depth 0..1
    wind: jnp.ndarray           # (2,) int32: drift direction (rows, columns)
    climate: jnp.ndarray        # scalar: the climate's random drift, C (the cycle is computed from t)
    weather_timer: jnp.ndarray  # scalar int32
    # agents
    pos: jnp.ndarray            # (N, 2) int32
    satiety: jnp.ndarray
    hydration: jnp.ndarray
    energy: jnp.ndarray
    health: jnp.ndarray
    inv: jnp.ndarray            # (N, N_MAT) carried amount of each material
    hand: jnp.ndarray           # (N,) int32 material currently in hand
    torch: jnp.ndarray          # (N,) int32 ticks left on a burning branch in hand, 0 = none
    age: jnp.ndarray            # (N,) int32
    alive: jnp.ndarray          # (N,) bool
    visited: jnp.ndarray        # (N, G, G) bool
    mind_map: jnp.ndarray       # (N, G, G, 3) what i remembers of each cell: water seen, plants seen, recency
    sleepiness: jnp.ndarray     # (N,) 0 rested .. 1 exhausted for want of sleep
    asleep: jnp.ndarray         # (N,) bool
    sleep_depth: jnp.ndarray    # (N,) int: ticks of unbroken sleep
    collapsed: jnp.ndarray      # (N,) bool: fell asleep from exhaustion; cannot wake by choice yet
    body_temp: jnp.ndarray      # (N,) thermal state, 0 = comfortable
    injury: jnp.ndarray         # (N,) in [0, 1]
    infection: jnp.ndarray      # (N,) int32 ticks of illness remaining
    immunity: jnp.ndarray       # (N,) in [0, 1]
    stress: jnp.ndarray         # (N,) in [0, 1]
    clothing: jnp.ndarray       # (N,) in [0, 1]: how much of the body hides cover (EnvConfig.clothing)
    pain_aversion: jnp.ndarray  # (N,) phenotype; drifts from genotype with trauma
    weight: jnp.ndarray         # (N,) kg; rises when well fed, burns when starving
    drop_owner: jnp.ndarray     # (G, G) int32: who last dropped food on this cell (-1: nobody)
    drop_time: jnp.ndarray      # (G, G) int32: when
    habituation: jnp.ndarray    # (N, N, 4) how used i is to each feeling toward j (repaying, touch, revenge, intimacy)
    attachment: jnp.ndarray     # (N, N) 0..1: how much j's survival is felt as part of i's
    conditions: jnp.ndarray     # (N, K) 0/1: conditions each has (lifesim/conditions.py)
    father_cond: jnp.ndarray    # (N, K) the father's conditions, kept from conception for the birth
    life_moves: jnp.ndarray     # (N,) int: steps walked in this life
    life_strikes: jnp.ndarray   # (N,) int: blows struck in this life
    hardship: jnp.ndarray       # (N,) int: ticks in a row with a critical need
    impression: jnp.ndarray     # (N, N) how j looked to i when last seen (visible condition; -1 seen dead)
    scars: jnp.ndarray          # (N,) in [0, 1]: visible marks of past wounds; never fade
    # Reproduction (all present even when off). A slot is "existed" once someone was born into it.
    sex: jnp.ndarray            # (N,) FEMALE or MALE, visible to others
    existed: jnp.ndarray        # (N,) bool: founder or born; unborn slots are empty and invisible
    gestation: jnp.ndarray      # (N,) int32 ticks pregnant (0: not pregnant)
    courting: jnp.ndarray       # (N,) int32 ticks a choice to mate stays open (0: not courting)
    father: jnp.ndarray         # (N,) int32 father of the current pregnancy (-1: none)
    mother: jnp.ndarray         # (N,) int32 this agent's mother (-1: founder)
    relatedness: jnp.ndarray    # (N, N) genetic relatedness: 1 self, 0.5 parent/child/sibling, ...
    fetus_r: jnp.ndarray        # (N, N) the unborn child's relatedness to everyone, fixed at conception
    father_geno: gt.Genotype    # (N,) father's genotype for each pregnancy
    father_look: jnp.ndarray    # (N, N_APPEARANCE) father's appearance for each pregnancy
    lasting_injury: jnp.ndarray # (N,) floor that injury can no longer heal below
    pain: jnp.ndarray           # (N,) fresh pain from recent wounds, fading (pain_of adds the ache of injury)
    heartache: jnp.ndarray      # (N,) emotional pain: others' hurt seen (x empathy) and grief; fades over days
    part_damage: jnp.ndarray    # (N, N_PARTS) damage to head, neck, torso, arms, legs, 0..1 (EnvConfig.body_parts)
    held_by: jnp.ndarray        # (N,) int32 who got hold of one last tick (-1: free; EnvConfig.techniques)
    hold_grip: jnp.ndarray      # (N,) that hold's strength: the chance it pins one this tick
    off_balance: jnp.ndarray    # (N,) int32 ticks left staggered or on the ground: cannot step or strike
    birth_age: jnp.ndarray      # (N,) int32 biological age in ticks when the run began
    genotype: gt.Genotype
    appearance: jnp.ndarray     # (N, 3) heritable look; lets kin resemble each other
    memory: jnp.ndarray         # (N, N, N_MEM) observer i's decayed record of agent j
    last_msg: jnp.ndarray       # (N, MSG_LEN) int32 message emitted last tick, all 0 = silent
    # underground (cfg.underground; empty otherwise)
    under: jnp.ndarray          # (G, G) bool: a cave passage (or pool) lies under this cell
    under_water: jnp.ndarray    # (G, G) bool: an underground pool (drinkable, not walkable)
    under_kind: jnp.ndarray     # (G, G) int: the cave formation here (minerals.KINDS; 0 none)
    spring: jnp.ndarray         # (G, G) bool: a hot spring on the surface (warm, drinkable)
    cave_floor: jnp.ndarray     # (G, G, N_MAT) what lies underground here: ore in the walls, and whatever
                                #   was dropped, placed or killed below (stone is unlimited, the walls)
    below: jnp.ndarray          # (N,) bool: the agent is underground
    # plant ecology and camp fouling (cfg.ecology, cfg.fouling; zeros otherwise)
    flora: jnp.ndarray          # (G, G, 4) standing crop of greens, fruit, nuts, roots, in food units
    flora_patch: jnp.ndarray    # (G, G, 4) how rich each patch is in each kind (fixed for the world)
    foul: jnp.ndarray           # (G, G) waste where people have stayed, person-days per km^2
    urge: jnp.ndarray           # (N,) 0..1 need to relieve oneself (cfg.excretion)
    marks: jnp.ndarray          # (G, G, MSG_LEN) int32: what is written on each cell (all 0: nothing)
    mark_time: jnp.ndarray      # (G, G) int32: when it was written (-1: never)
    mark_kind: jnp.ndarray      # (G, G) int32: what it is written on (SURF_*), + MARK_PAINTED if drawn in pigment
    # animals
    small: Animals
    large: Animals
    predators: Animals


class StepInfo(NamedTuple):
    outcome: jnp.ndarray        # (N,) outcome code
    died: jnp.ndarray
    death_cause: jnp.ndarray
    gathered: jnp.ndarray
    health_loss: jnp.ndarray
    pos: jnp.ndarray
    t: jnp.ndarray
    attacked_by_predator: jnp.ndarray  # (N,) bool
    struck_by_agent: jnp.ndarray       # (N,) bool
    newly_infected: jnp.ndarray        # (N,) bool
    recovered: jnp.ndarray             # (N,) bool
    fell: jnp.ndarray                  # (N,) bool, rock fall
    kills: jnp.ndarray                 # (N, 3) animals killed this tick by type
    weather: jnp.ndarray               # scalar
    weather_changed: jnp.ndarray       # scalar bool
    ambient_temp: jnp.ndarray          # (N,) degrees at each agent
    hits: jnp.ndarray                  # (N, N) bool: i struck j this tick
    gifts: jnp.ndarray                 # (N, N) bool: j took food that i had dropped (a completed gift)
    touches: jnp.ndarray               # (N, N) bool: i touched j
    message: jnp.ndarray               # (N, MSG_LEN) int32 message emitted this tick
    spoke: jnp.ndarray                 # (N,) bool: the message was not silent
    ignited: jnp.ndarray               # (N,) int32 fire started by agent: 0 none, 1 spark, 2 burning wood
    lightning: jnp.ndarray             # scalar bool: lightning started a fire this tick
    burning_cells: jnp.ndarray         # scalar int32
    near_fire: jnp.ndarray             # (N,) bool: agent feels a fire's heat
    fed_fire: jnp.ndarray              # (N,) bool: placed wood onto a burning cell
    meat_to_heat: jnp.ndarray          # (N,) bool: placed raw meat onto a heated cell
    ate_cooked: jnp.ndarray            # (N,) bool
    torch_lit: jnp.ndarray             # (N,) bool: picked up burning wood
    burned: jnp.ndarray                # (N,) bool: stood in fire
    moral_reward: jnp.ndarray          # (N,) the moral-sentiment part of this tick's reward
    deterred: jnp.ndarray              # (N,) bool: a predator backed off because a neighbor stood by
    huddling: jnp.ndarray              # (N,) bool: at least one living neighbor adjacent
    winter_night: jnp.ndarray          # scalar bool
    company: jnp.ndarray               # (N,) bool: another living agent within view
    meat_eaten: jnp.ndarray            # (N,) bool: ate raw or cooked meat this tick
    large_prey_hit: jnp.ndarray        # (N,) bool: a strike landed on a large prey animal
    mated: jnp.ndarray                 # (N,) bool: took part in a mutual mating
    slept: jnp.ndarray                 # (N,) bool: asleep this tick
    woken: jnp.ndarray                 # (N,) bool: a sleeper woken by being struck, attacked, burned or touched
    fell_asleep: jnp.ndarray           # (N,) bool: went to sleep this tick
    skill: jnp.ndarray                 # (N,) int: given skill chosen (0 none, 1 water, 2 food, 3 flee)
    season: jnp.ndarray                # () int: season this tick (0 spring .. 3 winter)
    new_conditions: jnp.ndarray        # (N, K) bool: conditions gained this tick
    made: jnp.ndarray                  # (G, G, M) minerals that heat made this tick (fired clay, metals)
    bound: jnp.ndarray                 # (N,) bool: bound a point to a stick with fiber (a spear) this tick
    bedded: jnp.ndarray                # (N,) bool: asleep on leaves or a hide (a bed)
    kill_species: jnp.ndarray          # (N, 3) species of the small prey, large prey, predator it killed (-1 none)
    attacker_species: jnp.ndarray      # (N,) species of the predator that attacked it (-1 none)
    gored: jnp.ndarray                 # (N,) damage from an animal it struck that fought back
    gored_by: jnp.ndarray              # (N,) that animal's species (-1 none)
    stung: jnp.ndarray                 # (N,) bool: stung by bees while taking honey
    venomed: jnp.ndarray               # (N,) bool: a venomous bite
    mosquitoes: jnp.ndarray            # (N,) mosquito density where it stands (0..1)
    small_preyed_by: jnp.ndarray       # (n_small,) species of the predator that took it this tick (-1 none)
    large_preyed_by: jnp.ndarray       # (n_large,) same for large prey (a carcass is left)
    small_species: jnp.ndarray         # (n_small,) species of each small-prey slot
    large_species: jnp.ndarray         # (n_large,)
    small_pos: jnp.ndarray             # (n_small, 2) where each was
    large_pos: jnp.ndarray             # (n_large, 2)
    collapsed_now: jnp.ndarray         # (N,) bool: fell asleep from exhaustion this tick
    conceived: jnp.ndarray             # (N,) bool: became pregnant
    no_room: jnp.ndarray               # (N,) bool: would have conceived, but every slot was taken
    gave_birth: jnp.ndarray            # (N,) bool: mother whose child was born this tick
    born: jnp.ndarray                  # (N,) bool: slot that received a newborn this tick
    nursed: jnp.ndarray                # (N,) bool: child nursed by its mother this tick
    genetic_reward: jnp.ndarray        # (N,) relatives' survival, weighted by relatedness
    below: jnp.ndarray                 # (N,) bool: underground after this tick
    delved: jnp.ndarray                # (N,) bool: went down or came up this tick
    dark: jnp.ndarray                  # (N,) bool: below with no light where it stands (afraid, groping)
    stress_from: jnp.ndarray           # (N, 8) stress gained this tick by source (STRESS_SOURCES)
    downed: jnp.ndarray                # (N,) bool: down after this tick (EnvConfig.downed)
    relieved: jnp.ndarray              # (N,) bool: relieved itself by choice (EnvConfig.excretion)
    gathered_kind: jnp.ndarray         # (N,) int: plant food gathered: ecology.KINDS index, 4 loose, -1 none
    aim: jnp.ndarray                   # (N,) int: the part its blow was aimed at (env.PARTS), -1 if none (or it missed an animal)
    blow_landed: jnp.ndarray           # (N,) bool: its blow at someone landed
    how: jnp.ndarray                   # (N,) int: its strike's joint choice (how_parts: part, limb, mode), -1 if none
    shoved: jnp.ndarray                # (N,) bool: pushed off its cell this tick (EnvConfig.techniques)
    thrown: jnp.ndarray                # (N,) bool: thrown to the ground by someone holding it
    grabbed: jnp.ndarray               # (N,) bool: someone got hold of it this tick
    pinned: jnp.ndarray                # (N,) bool: a hold from last tick kept it from stepping
    same_sex_mated: jnp.ndarray        # (N,) bool: intimacy with a partner of its own sex (no conception)
    soiled: jnp.ndarray                # (N,) bool: could not hold it any longer


GRAZERS = ("rabbit", "fowl", "tortoise", "deer", "aurochs", "boar", "horse", "mammoth")   # eat plants (cfg.grazing)
STRESS_SOURCES = ("hurt", "danger", "unmet needs", "loneliness", "missing someone", "sleep debt", "darkness",
                  "needing to relieve oneself", "heartache")


# ---------------------------------------------------------------- helpers
def season_of(t, cfg: EnvConfig):
    day = (t % cfg.year_length) // cfg.day_length
    b1, b2, b3 = cfg.season_starts
    return (day >= b1).astype(jnp.int32) + (day >= b2) + (day >= b3)


def hour_of_day(t, cfg: EnvConfig):
    """Clock time in hours, 0 <= h < 24."""
    return 24.0 * (t % cfg.day_length) / cfg.day_length


def daylight_hours(t, cfg: EnvConfig):
    """Hours of daylight: 12 in spring and autumn, 12 + swing at midsummer, 12 - swing midwinter."""
    year_frac = (t % cfg.year_length) / cfg.year_length            # 0 = first day of spring
    return 12.0 + cfg.daylight_swing * jnp.cos(2 * jnp.pi * (year_frac - cfg.solstice_frac))


def is_night(t, cfg: EnvConfig):
    """Dark between sunset and sunrise; solar noon is 12:00."""
    h, d = hour_of_day(t, cfg), daylight_hours(t, cfg)
    return (h < 12.0 - d / 2) | (h >= 12.0 + d / 2)


SEASON_NAMES = ["Spring", "Summer", "Autumn", "Winter"]


def clock(t: int, cfg: EnvConfig) -> dict:
    """Host-side calendar for a tick: year, season, day, hh:mm and a stamp."""
    t = int(t)
    day_index = t // cfg.day_length
    year = day_index // cfg.days_per_year + 1
    day_of_year = day_index % cfg.days_per_year
    starts = (0,) + cfg.season_starts
    season = sum(day_of_year >= b for b in cfg.season_starts)
    day_in_season = day_of_year - starts[season] + 1
    minutes = (t % cfg.day_length) * 24 * 60 // cfg.day_length
    hh, mm = divmod(minutes, 60)
    return {"year": year, "season": SEASON_NAMES[season], "day": day_in_season,
            "day_of_year": day_of_year + 1,
            "hour": hh, "minute": mm,
            "stamp": f"Y{year} {SEASON_NAMES[season]} D{day_in_season} {hh:02d}:{mm:02d}"}


def _smooth_noise(key, size, coarse):
    small = jax.random.normal(key, (coarse, coarse))
    return jax.image.resize(small, (size, size), method="linear")


def generate_terrain(key, cfg: EnvConfig):
    if cfg.biomes:
        return generate_biomes(key, cfg)
    k_elev, k_moist = jax.random.split(key)
    coarse = max(4, cfg.grid_size // 6)
    elev = _smooth_noise(k_elev, cfg.grid_size, coarse)
    moist = _smooth_noise(k_moist, cfg.grid_size, coarse)
    water_thr = jnp.quantile(elev, cfg.water_fraction)
    rock_thr = jnp.quantile(elev, 1.0 - cfg.rock_fraction)
    forest_thr = jnp.quantile(moist, 1.0 - cfg.forest_fraction)
    terrain = jnp.full(elev.shape, GRASS, dtype=jnp.int32)
    terrain = jnp.where(moist >= forest_thr, FOREST, terrain)
    terrain = jnp.where(elev >= rock_thr, ROCK, terrain)
    return jnp.where(elev <= water_thr, WATER, terrain)


def generate_biomes(key, cfg: EnvConfig):
    """Biomes from elevation, moisture and coldness (north and high ground are colder): water in
    the lowest ground, marsh just above it where wet, rock on the highest, tundra where coldest,
    desert where driest and warm, forest where wet, grass elsewhere; caves in rock outcrops."""
    k_elev, k_moist = jax.random.split(key)
    g = cfg.grid_size
    coarse = max(4, g // 6)
    elev = _smooth_noise(k_elev, g, coarse)
    moist = _smooth_noise(k_moist, g, coarse)
    rows = jnp.broadcast_to(jnp.arange(g)[:, None] / max(g - 1, 1), (g, g))
    e01 = (elev - elev.min()) / jnp.maximum(elev.max() - elev.min(), 1e-6)
    cold = 0.6 * (1 - rows) + 0.4 * e01
    water_thr = jnp.quantile(elev, cfg.water_fraction)
    marsh_thr = jnp.quantile(elev, cfg.water_fraction + cfg.marsh_fraction * 2)
    rock_thr = jnp.quantile(elev, 1.0 - cfg.rock_fraction)
    forest_thr = jnp.quantile(moist, 1.0 - cfg.forest_fraction)
    desert_thr = jnp.quantile(moist, cfg.desert_fraction * 1.6)
    tundra_thr = jnp.quantile(cold, 1.0 - cfg.tundra_fraction)
    t = jnp.full((g, g), GRASS, jnp.int32)
    t = jnp.where(moist >= forest_thr, FOREST, t)
    t = jnp.where((moist <= desert_thr) & (cold < jnp.median(cold)), DESERT, t)
    t = jnp.where(cold >= tundra_thr, TUNDRA, t)
    t = jnp.where((elev <= marsh_thr) & (moist >= jnp.median(moist)), MARSH, t)
    t = jnp.where(elev >= rock_thr, ROCK, t)
    t = jnp.where(elev <= water_thr, WATER, t)
    # Caves: rock cells on an outcrop's edge (beside walkable ground), a few per map
    pad = jnp.pad((t != ROCK) & (t != WATER), 1)
    edge = jnp.zeros((g, g), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            edge = edge | pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g]
    candidates = ((t == ROCK) & edge).reshape(-1).astype(jnp.float32)
    pick = jax.random.choice(jax.random.fold_in(key, 3), g * g, (cfg.caves,), replace=False,
                             p=(candidates + 1e-9) / (candidates + 1e-9).sum())
    has = candidates.sum() > 0
    t = t.reshape(-1).at[pick].set(jnp.where(has, CAVE, t.reshape(-1)[pick])).reshape(g, g)
    return t


def drinkable(terrain):
    """Cells one can drink from: open water and marsh."""
    return (terrain == WATER) | (terrain == MARSH)


def generate_height(key, cfg: EnvConfig):
    """Height of every cell in [0, 1]: the same elevation map that placed water and rock."""
    k_elev, _ = jax.random.split(key)
    coarse = max(4, cfg.grid_size // 6)
    elev = _smooth_noise(k_elev, cfg.grid_size, coarse)
    lo, hi = jnp.min(elev), jnp.max(elev)
    return (elev - lo) / jnp.maximum(hi - lo, 1e-6)


def _lowest_neighbor(height):
    g = height.shape[0]
    pad = jnp.pad(height, 1, mode="edge")
    return jnp.min(jnp.stack([pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g]
                              for dr in (-1, 0, 1) for dc in (-1, 0, 1)]), axis=0)


def generate_cliffs(height):
    """How dangerous each cell's edge is: its drop to its lowest neighbor relative to the steepest
    drop on the map, squared, since danger rises sharply with steepness. Gentle slopes are near 0,
    the steepest edge is 1. Rock edges are steepest, but banks and slopes anywhere can be steep."""
    drop = height - _lowest_neighbor(height)
    return jnp.clip(drop / jnp.maximum(jnp.max(drop), 1e-6), 0.0, 1.0) ** 2


def steepest_drop(height):
    """The map's steepest single-step drop: the unit that climbing cost and cliffs are scaled by."""
    return jnp.maximum(jnp.max(height - _lowest_neighbor(height)), 1e-6)


def _in_bounds(p, g):
    return jnp.all((p >= 0) & (p < g), axis=-1)


def _cheb(a, b):
    return jnp.max(jnp.abs(a - b), axis=-1)


def adjacent_to_water(terrain, pos, cfg: EnvConfig):
    g = cfg.grid_size
    nb = pos[:, None, :] + NEIGHBORS8[None]
    ok = _in_bounds(nb, g)
    nbc = jnp.clip(nb, 0, g - 1)
    on_marsh = terrain[jnp.clip(pos[:, 0], 0, g - 1), jnp.clip(pos[:, 1], 0, g - 1)] == MARSH
    return jnp.any(ok & drinkable(terrain[nbc[..., 0], nbc[..., 1]]), axis=1) | on_marsh


def open_water_beside(terrain, pos, cfg: EnvConfig):
    g = cfg.grid_size
    nb = pos[:, None, :] + NEIGHBORS8[None]
    nbc = jnp.clip(nb, 0, g - 1)
    return jnp.any(_in_bounds(nb, g) & (terrain[nbc[..., 0], nbc[..., 1]] == WATER), axis=1)


def _random_land_cells(key, terrain, n):
    g = terrain.shape[0]
    walkable = (terrain != WATER).reshape(-1).astype(jnp.float32)
    cells = jax.random.choice(key, g * g, (n,), replace=True, p=walkable / walkable.sum())
    return jnp.stack([cells // g, cells % g], axis=-1).astype(jnp.int32)


def _spawn_band(key, terrain, n, spread, water_radius=-1):
    """Founders start together: land cells within `spread` of one camp point
    (people are born into groups). The camp is within `water_radius` cells of
    drinkable water, as real camps are; -1 puts it anywhere on land.
    spread = 0 scatters founders over the whole map."""
    if spread <= 0:
        return _random_land_cells(key, terrain, n)
    g = terrain.shape[0]
    k_center, k_members = jax.random.split(key)
    rr, cc = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    land = terrain != WATER
    if water_radius >= 0:
        water = (terrain == WATER).astype(jnp.float32)
        size = 2 * (water_radius + 1) + 1                  # drinkable = next to water, so radius + 1
        near_water = jax.lax.reduce_window(water, 0.0, jax.lax.max, (size, size), (1, 1), "SAME") > 0
        ok = land & near_water
        ok = jnp.where(ok.any(), ok, land)                 # a world with no water: anywhere on land
    else:
        ok = land
    w0 = ok.reshape(-1).astype(jnp.float32)
    cidx = jax.random.choice(k_center, g * g, (), p=w0 / w0.sum())
    center = jnp.stack([cidx // g, cidx % g]).astype(jnp.int32)
    near = (jnp.maximum(jnp.abs(rr - center[0]), jnp.abs(cc - center[1])) <= spread) & (terrain != WATER)
    w = near.reshape(-1).astype(jnp.float32)
    cells = jax.random.choice(k_members, g * g, (n,), replace=True, p=w / w.sum())
    return jnp.stack([cells // g, cells % g], axis=-1).astype(jnp.int32)


def _k(cfg: EnvConfig) -> float:
    """How many 30-minute calibration ticks one tick is (1/30 at 1-minute ticks): rates written
    as numbers in this file are per calibration tick and are scaled by it (config.with_tick_minutes
    does the same for the config's own rates)."""
    return cfg.tick_minutes / 30.0


def _ticks(cfg: EnvConfig, n: float) -> int:
    """n calibration ticks of game time, in this world's ticks."""
    return max(1, int(round(n / _k(cfg))))


def _occupancy(pos, alive, g):
    return jnp.zeros((g, g)).at[pos[:, 0], pos[:, 1]].add(alive.astype(jnp.float32))


# ---------------------------------------------------------------- underground
def layered_pos(pos, below):
    """(N, 2) positions with the layer folded in: a below agent's column is moved LAYER_OFF away,
    so every Chebyshev distance between layers is huge and nothing crosses them (sight, hearing,
    touch, blows, mating, contagion, huddling, predators, imitation). Same layer: unchanged."""
    return pos + jnp.stack([jnp.zeros_like(below, jnp.int32), below.astype(jnp.int32) * LAYER_OFF], -1)


def cave_mouths(state, cfg: EnvConfig):
    """((K, 2) cells, (K,) real): the cave mouths, K = cfg.caves + cfg.sinkholes (fewer real ones
    on some maps)."""
    g = cfg.grid_size
    k = max(cfg.caves + cfg.sinkholes, 1)
    flat = ((state.terrain == CAVE) & state.under).reshape(-1).astype(jnp.float32)
    score, cell = jax.lax.top_k(flat, k)
    return jnp.stack([cell // g, cell % g], -1).astype(jnp.int32), score > 0.5


def under_terrain(state):
    """(G, G) how the cave layer looks in terrain terms: passage = CAVE, pool = WATER, wall = ROCK."""
    return jnp.where(state.under_water, WATER, jnp.where(state.under, CAVE, ROCK)).astype(jnp.int32)


def spring_warmth(state, cfg: EnvConfig):
    """(G, G) C added near hot springs: on the spring and the cells around it."""
    pad = jnp.pad(state.spring, 1)
    g = state.spring.shape[0]
    near = sum(pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) > 0
    return jnp.where(near, cfg.spring_heat, 0.0)


def cave_air(t, cfg: EnvConfig, climate=0.0, kind=None):
    """Degrees C underground: the season's mean (no day swing, no weather), pulled most of the way
    to the air the cave holds: comfort, warmer in lava tubes, near freezing in ice caves."""
    mean = jnp.array(cfg.season_temp)[season_of(t, cfg)] + climate
    held = cfg.comfort_temp + (0.0 if kind is None else jnp.array(_KIND_AIR)[kind])
    return mean + cfg.deep_cave_buffer * (held - mean)


# cave formation kinds (lifesim/minerals.py KINDS)
K_NONE, K_KARST, K_LAVA, K_ICE, K_SALT, K_GEODE = range(6)
_KIND_TURN = (0.3, 0.35, 0.1, 0.3, 0.3, 0.3)          # lava tubes run long and straight
_KIND_CHAMBER = (0.08, 0.12, 0.03, 0.08, 0.08, 0.08)
_KIND_POOL = (0.0, 0.5, 0.0, 0.4, 0.0, 0.0)           # karst is wet, ice caves have frozen pools, salt is dry
_KIND_AIR = (0.0, 0.0, 4.0, -19.0, 0.0, 1.0)          # the air the cave holds, as C from comfort


def cave_kind_of_mouths(terrain, height, mouth, cfg: EnvConfig):
    """(K,) formation of each mouth's cave system, from the land around it (radius 3): an ice cave
    in tundra or the far north, a salt cave in desert, a lava tube on the highest ground, karst
    otherwise."""
    g = cfg.grid_size
    rr, cc = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    near = (jnp.maximum(jnp.abs(rr[None] - mouth[:, 0, None, None]), jnp.abs(cc[None] - mouth[:, 1, None, None])) <= 3)
    has = lambda t: jnp.any(near & (terrain == t)[None], axis=(1, 2))
    high = height[mouth[:, 0], mouth[:, 1]] >= 0.75
    north = mouth[:, 0] < 0.15 * g
    return jnp.where(has(TUNDRA) | north, K_ICE, jnp.where(has(DESERT), K_SALT, jnp.where(high, K_LAVA, K_KARST)))


def generate_caves(key, terrain, cfg: EnvConfig, height=None):
    """The cave layer, as a dict: terrain (with sinkholes and hot springs added), under,
    under_water, under_kind, spring, cave_floor, and surface_minerals (sulfur at the springs).

    From each mouth (a CAVE cell), cave_tunnels random walks of cave_len steps carve passages,
    never passing under open water, now and then opening a 3x3 chamber, some with a pool in the
    middle (only where all eight cells around it are passage, so a pool never blocks a tunnel).
    How they wind, how often chambers open and pools form, the air and the minerals in the walls
    follow the system's formation (karst, lava tube, ice cave, salt cave; a quarter of karst
    chambers are quartz geodes). Karst opens sinkholes, new mouths onto grass or forest above
    it (cfg.sinkholes); lava tubes feed hot springs beside them (cfg.hot_springs: warm, drinkable,
    sulfur at the edge)."""
    g, m = cfg.grid_size, mn.M
    empty = dict(terrain=terrain, under=jnp.zeros((g, g), bool), under_water=jnp.zeros((g, g), bool),
                 under_kind=jnp.zeros((g, g), jnp.int32), spring=jnp.zeros((g, g), bool),
                 cave_floor=jnp.zeros((g, g, N_MAT)), surface_minerals=jnp.zeros((g, g, m)))
    if not cfg.underground:
        return empty
    height = jnp.zeros((g, g)) if height is None else height
    k_pick, k_dir, k_walk, k_ore, k_amt, k_geo, k_sink, k_spr = jax.random.split(key, 8)
    k = max(cfg.caves, 1)
    flat = (terrain == CAVE).reshape(-1).astype(jnp.float32)
    score, cell = jax.lax.top_k(flat + 1e-3 * jax.random.uniform(k_pick, (g * g,)), k)
    mouth = jnp.stack([cell // g, cell % g], -1).astype(jnp.int32)                 # (k, 2)
    real = score > 0.5                                                              # fewer mouths than k
    kinds = cave_kind_of_mouths(terrain, height, mouth, cfg)
    nt = cfg.cave_tunnels
    pos = jnp.repeat(mouth, nt, 0)
    on = jnp.repeat(real, nt, 0)
    wk = jnp.repeat(kinds, nt, 0)                                                   # each walker's formation
    turn_p, cham_p, pool_p = (jnp.array(x)[wk] for x in (_KIND_TURN, _KIND_CHAMBER, _KIND_POOL))
    d = NEIGHBORS8[jax.random.randint(k_dir, (k * nt,), 0, 8)]
    under = jnp.zeros((g, g), bool).at[mouth[:, 0], mouth[:, 1]].max(real)
    kind = jnp.zeros((g, g), jnp.int32).at[mouth[:, 0], mouth[:, 1]].max(jnp.where(real, kinds, 0))
    dry = terrain != WATER
    box = jnp.array([[dr, dc] for dr in range(-2, 3) for dc in range(-2, 3)], jnp.int32)

    def walk(t, carry):
        under, kind, pools, pos, d, key = carry
        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
        w = pos.shape[0]
        turn = jax.random.uniform(k1, (w,)) < turn_p
        d = jnp.where(turn[:, None], NEIGHBORS8[jax.random.randint(k2, (w,), 0, 8)], d)
        new = jnp.clip(pos + d, 1, g - 2)
        ok = dry[new[:, 0], new[:, 1]]
        # a diagonal step also opens a cell sharing an edge with both, so tunnels are not corner-to-corner
        side_a = jnp.stack([new[:, 0], pos[:, 1]], -1)
        side_b = jnp.stack([pos[:, 0], new[:, 1]], -1)
        side = jnp.where(dry[side_a[:, 0], side_a[:, 1]][:, None], side_a, side_b)
        diag = ok & (d != 0).all(-1) & dry[side[:, 0], side[:, 1]]
        under = under.at[side[:, 0], side[:, 1]].max(on & diag)
        kind = kind.at[side[:, 0], side[:, 1]].max(jnp.where(on & diag, wk, 0))
        pos = jnp.where(ok[:, None], new, pos)
        d = jnp.where(ok[:, None], d, -d)                                           # turn back at water
        under = under.at[pos[:, 0], pos[:, 1]].max(on)
        kind = kind.at[pos[:, 0], pos[:, 1]].max(jnp.where(on, wk, 0))
        chamber = on & (jax.random.uniform(k3, (w,)) < cham_p)
        geode = chamber & (wk == K_KARST) & (jax.random.uniform(k5, (w,)) < 0.25)
        # a chamber: the 3x3 around, and ragged edges out to two cells
        cells = jnp.clip(pos[:, None, :] + box[None], 0, g - 1)                    # (w, 25, 2)
        ragged = (jnp.abs(box).max(-1) <= 1)[None] | (jax.random.uniform(jax.random.fold_in(k3, 1), (w, box.shape[0])) < 0.35)
        open_ = chamber[:, None] & ragged & dry[cells[..., 0], cells[..., 1]]
        under = under.at[cells[..., 0], cells[..., 1]].max(open_)
        kind = kind.at[cells[..., 0], cells[..., 1]].max(jnp.where(open_, jnp.where(geode, K_GEODE, wk)[:, None], 0))
        pool = chamber & ~geode & (jax.random.uniform(k4, (w,)) < pool_p) & (t > 3)
        pools = pools.at[pos[:, 0], pos[:, 1]].max(pool)
        return under, kind, pools, pos, d, key

    under, kind, pools, _, _, _ = jax.lax.fori_loop(0, cfg.cave_len or g, walk,
                                                    (under, kind, jnp.zeros((g, g), bool), pos, d, k_walk))
    ring = sum(jnp.pad(under, 1)[1 + dr:1 + dr + g, 1 + dc:1 + dc + g].astype(jnp.int32)
               for dr, dc in NEIGHBORS8.tolist())
    under_water = pools & under & (ring == 8) & (terrain != CAVE)
    # Ore in the walls, by formation
    chance = jnp.minimum(jnp.array(mn.CAVE_CHANCE)[kind] * cfg.mineral_abundance * cfg.cave_ore_mult, 1.0)   # (G, G, M)
    lo, hi = (jnp.array([max(a[i], 1) for a in mn.DEPOSIT_AMOUNT], jnp.float32) for i in (0, 1))
    hi = jnp.maximum(hi, jnp.where(jnp.array([a[1] for a in mn.DEPOSIT_AMOUNT]) == 0, 3.0, hi))
    here = (jax.random.uniform(k_ore, (g, g, m)) < chance) & (under & ~under_water)[..., None]
    amount = jnp.round(lo + (hi - lo) * jax.random.uniform(k_amt, (g, g, m)))
    floor = jnp.concatenate([jnp.zeros((g, g, N_BASE_MAT)), jnp.where(here, amount, 0.0)], -1)

    def pick(key, ok, count):
        """Up to `count` random cells where ok, as (count,) flat indices and (count,) real."""
        w = ok.reshape(-1).astype(jnp.float32)
        sc, ix = jax.lax.top_k(w + 1e-3 * jax.random.uniform(key, (g * g,)), max(count, 1))
        return ix, (sc > 0.5) & (count > 0)
    # Sinkholes: karst roof fallen in, a new mouth onto grass or forest
    sink_ok = under & ~under_water & ((kind == K_KARST) | (kind == K_GEODE)) & ((terrain == GRASS) | (terrain == FOREST))
    six, sreal = pick(k_sink, sink_ok, cfg.sinkholes)
    t = terrain.reshape(-1)
    t = t.at[six].set(jnp.where(sreal, CAVE, t[six]))
    # Hot springs: beside a lava tube, open land becomes a warm, drinkable pool's edge (marsh)
    pad = jnp.pad(under & (kind == K_LAVA), 1)
    beside = sum(pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g] for dr, dc in NEIGHBORS8.tolist()) > 0
    land = (terrain != WATER) & (terrain != ROCK) & (terrain != CAVE)
    hix, hreal = pick(k_spr, beside & land & ~under, cfg.hot_springs)
    t = t.at[hix].set(jnp.where(hreal, MARSH, t[hix]))
    spring = jnp.zeros(g * g, bool).at[hix].max(hreal).reshape(g, g)
    surf = jnp.zeros((g, g, m)).at[..., mn.IDX["sulfur"]].set(jnp.where(spring, 2.0, 0.0))
    return dict(terrain=t.reshape(g, g), under=under, under_water=under_water, under_kind=jnp.where(under, kind, 0),
                spring=spring, cave_floor=floor, surface_minerals=surf)


def species_params(cfg: EnvConfig) -> dict:
    """(S,) arrays of every species' body and habits (lifesim/fauna.py). The first species of each
    group (rabbit, deer, wolf) takes the group's values from the config, so the group knobs and
    the classic world keep working; predator attack, chase and rest scale with the config's."""
    col = lambda k, i=None: jnp.array(fa.column(k, i), jnp.float32)
    p = {"hp": col("hp"), "armor": col("armor"), "meat": col("meat"), "notice": col("notice"),
         "act": col("act"), "speed": col("speed"), "herd": col("herd"), "nocturnal": col("nocturnal"),
         "evade": col("evade"), "counter_p": col("counter", 0), "counter_dmg": col("counter", 1),
         "atk_p": col("attack", 0), "atk_dmg": col("attack", 1), "atk_inj": col("attack", 2),
         "atk_rest": col("attack", 3), "pack": col("pack"), "ambush": col("ambush"), "provoked": col("provoked"),
         "hibernates": col("hibernates"), "venom": col("venom"), "takes_large": col("takes_large"),
         "den": col("den"), "hide": col("hide"), "bone": col("bone"), "feather": col("feather"),
         "size": col("look", 0), "fangs": col("look", 1), "colour": col("look", 2),
         "habitat": jnp.array(fa.HABITAT, bool),
         "grazer": jnp.array([s["name"] in GRAZERS for s in fa.SPECIES])}
    for g, (hp, armor, meat, notice, act) in enumerate([
            (cfg.small_prey_hp, 0.0, cfg.small_prey_meat, cfg.small_prey_notice, cfg.small_prey_flee_prob),
            (cfg.large_prey_hp, cfg.large_prey_armor, cfg.large_prey_meat, cfg.large_prey_notice, cfg.large_prey_flee_prob),
            (cfg.predator_hp, cfg.predator_armor, 2.0, cfg.predator_notice, cfg.predator_chase_prob)]):
        i = fa.IN_GROUP[g][0]
        for k, v in (("hp", hp), ("armor", armor), ("meat", meat), ("notice", notice), ("act", act)):
            p[k] = p[k].at[i].set(v)
    pred = jnp.array([s["group"] == fa.PRED for s in fa.SPECIES])
    p["atk_p"] = jnp.where(pred, jnp.minimum(p["atk_p"] * cfg.predator_attack_prob / 0.3, 1.0), 0.0)
    p["atk_dmg"] = p["atk_dmg"] * cfg.predator_attack_damage / 0.12
    p["atk_rest"] = jnp.maximum(1.0, p["atk_rest"] * cfg.predator_rest_ticks / 150)
    p["act"] = jnp.where(pred & (jnp.arange(fa.S) != fa.IN_GROUP[2][0]),
                         jnp.minimum(p["act"] * cfg.predator_chase_prob / 0.85, 1.0), p["act"])
    return p


def assign_species(group: int, n: int, cfg: EnvConfig):
    """(n,) species for a group's slots: slot 0 is the group's first species, the rest follow the
    species' shares, deterministically (a stable mix for every world)."""
    members = fa.IN_GROUP[group]
    if not cfg.fauna_diversity or n == 1:
        return jnp.full(n, members[0], jnp.int32)
    shares = jnp.array([0.0 if (fa.SPECIES[i]["cave_only"] and not cfg.underground) else fa.SPECIES[i]["share"]
                        for i in members])                         # cave-only species: only with caves
    cum = jnp.cumsum(shares / shares.sum())
    q = (jnp.arange(n) + 0.5) / n
    k = jnp.minimum(jnp.searchsorted(cum, q), len(members) - 1)
    return jnp.array(members, jnp.int32)[k].at[0].set(members[0])


def _habitat_cells(key, terrain, species, cfg: EnvConfig):
    """(n, 2) a random cell for each animal, within its species' habitat (any land if none there)."""
    hab = jnp.array(fa.HABITAT, bool)[species][:, terrain]                              # (n, G, G)
    land = (terrain != WATER)[None]
    ok = jnp.where(jnp.any(hab, axis=(1, 2))[:, None, None], hab, land)
    g = terrain.shape[0]
    w = ok.reshape(species.shape[0], -1).astype(jnp.float32)
    keys = jax.random.split(key, species.shape[0])
    idx = jax.vmap(lambda k, p: jax.random.choice(k, g * g, p=p / p.sum()))(keys, w)
    return jnp.stack([idx // g, idx % g], -1).astype(jnp.int32)


def _roost(key, species, passage, cfg: EnvConfig):
    """(cells (n, 2), roosting (n,) bool): cave-only species (bats) start and are born in the caves."""
    n = species.shape[0]
    if not cfg.underground:
        return jnp.zeros((n, 2), jnp.int32), jnp.zeros(n, bool)
    g = passage.shape[0]
    w = passage.reshape(-1).astype(jnp.float32)
    cell = jax.random.choice(key, g * g, (n,), p=(w + 1e-9) / (w + 1e-9).sum())
    only = jnp.array(fa.column("cave_only"), bool)[species] & passage.any()
    return jnp.stack([cell // g, cell % g], -1).astype(jnp.int32), only


def _new_animals(key, terrain, n, group, cfg: EnvConfig, passage=None):
    sp = assign_species(group, n, cfg)
    pos = _habitat_cells(key, terrain, sp, cfg)
    if passage is not None:
        cells, roost = _roost(jax.random.fold_in(key, 5), sp, passage, cfg)
        pos = jnp.where(roost[:, None], cells, pos)
    else:
        roost = jnp.zeros(n, bool)
    return Animals(pos=pos, below=roost, alive=jnp.ones(n, bool),
                   hp=species_params(cfg)["hp"][sp], rest=jnp.zeros(n, jnp.int32), species=sp,
                   anger=jnp.zeros(n, jnp.int32), lame=jnp.zeros(n, jnp.int32))


def fire_heat(fire, cfg: EnvConfig):
    """(G, G) extra degrees from fires: full heat within 1 cell, less within 2."""
    burning = (fire > 0).astype(jnp.float32)
    near1 = jax.lax.reduce_window(burning, 0.0, jax.lax.max, (3, 3), (1, 1), "SAME")
    near2 = jax.lax.reduce_window(burning, 0.0, jax.lax.max, (5, 5), (1, 1), "SAME")
    return cfg.fire_heat * near1 + cfg.fire_heat_far * (near2 - near1)


def _neighbor_count(mask_f):
    """(G, G) number of the 8 neighbors where mask_f is 1."""
    return jax.lax.reduce_window(mask_f, 0.0, jax.lax.add, (3, 3), (1, 1), "SAME") - mask_f


def body(state, cfg: EnvConfig) -> dict:
    """Physical characteristics as they manifest now, from genes, age and food.

    Height grows to adult size by maturity and shrinks a little after 60.
    Strength and speed rise through youth, peak in the prime years, then decline.
    Condition (weight against the ideal for one's height and frame) scales
    strength; extra weight slows movement."""
    g = state.genotype
    years = (state.age + state.birth_age) / cfg.ticks_per_bio_year
    maturity = jnp.clip(years / cfg.maturity_years, 0.3, 1.0)
    shrink = 1 - 0.03 * jnp.clip((years - 60.0) / 20.0, 0.0, 1.0)
    height = cfg.base_height_cm * g.height * (0.55 + 0.45 * maturity) * shrink
    ideal = cfg.ideal_bmi * (height / 100.0) ** 2 * g.build
    youth = 0.6 + 0.4 * jnp.clip((years - 10.0) / 10.0, 0.0, 1.0)
    decline = 1 - 0.5 * jnp.clip((years - cfg.prime_until_years) / (cfg.max_bio_years - cfg.prime_until_years), 0.0, 1.0)
    age_factor = youth * decline
    condition = jnp.clip(state.weight / ideal, 0.7, 1.05)
    fx = condition_effects(state.conditions)
    arms = 1 - cfg.arm_weakness * state.part_damage[:, PART_ARMS] if cfg.body_parts else 1.0
    legs = 1 - cfg.leg_lameness * state.part_damage[:, PART_LEGS] if cfg.body_parts else 1.0
    strength = g.strength * age_factor * condition * fx.get("strength", 1.0) * arms
    speed = g.speed * age_factor * jnp.clip(jnp.sqrt(ideal / state.weight), 0.8, 1.1) * fx.get("speed", 1.0) * legs
    weight_factor = jnp.clip(jnp.sqrt(ideal / state.weight), 0.8, 1.1)
    return {"years": years, "height_cm": height, "weight_kg": state.weight, "ideal_kg": ideal,
            "strength": strength, "speed": speed, "maturity": maturity,
            # the factors strength and speed are made of (for the inspector; their products are above)
            "age_factor": age_factor, "condition": condition, "fx_strength": fx.get("strength", 1.0) * jnp.ones_like(years),
            "fx_speed": fx.get("speed", 1.0) * jnp.ones_like(years), "arms": arms * jnp.ones_like(years),
            "legs": legs * jnp.ones_like(years), "weight_factor": weight_factor}


def body_breakdown(state, cfg: EnvConfig, actions=None) -> dict:
    """(N,) arrays: how strength, a blow, speed and food and water needs are made up, factor by
    factor, as step() computes them (for the inspector). actions: what each is doing this tick."""
    bod = body(state, cfg)
    g = state.genotype
    ref_kg = cfg.ideal_bmi * (cfg.base_height_cm / 100.0) ** 2
    mass_force = (bod["ideal_kg"] / ref_kg) ** (2.0 / 3.0) if cfg.blow_by_mass else jnp.ones_like(bod["years"])
    practice = 1.0 + cfg.practice_max * (1.0 - 0.5 ** (state.life_strikes / cfg.practice_half)) \
        if cfg.blow_by_mass else jnp.ones_like(bod["years"])
    size_need = (bod["ideal_kg"] / ref_kg) ** 0.75 if cfg.needs_by_size else jnp.ones_like(bod["years"])
    if actions is not None and cfg.activity_needs:
        a = actions
        doing = jnp.where(a == SLEEP, cfg.activity_sleep, jnp.where((a >= STRIKE_N) & (a < STRIKE_N + 8), cfg.activity_strike,
                jnp.where((a >= MOVE_N) & (a <= MOVE_NW), cfg.activity_walk,
                          jnp.where((a == GRASP) | ((a >= PLACE_N) & (a < PLACE_N + 8)) | (a == WRITE),
                                    cfg.activity_work, cfg.activity_rest)))) / cfg.activity_norm
    else:
        doing = jnp.ones_like(bod["years"])
    preg = 1 + (state.gestation > 0) * cfg.pregnancy_need
    sick = 1 + (state.infection > 0) * cfg.disease_hunger_mult
    per_hour = 60.0 / cfg.tick_minutes
    return {"strength_gene": g.strength, "age_factor": bod["age_factor"], "condition": bod["condition"],
            "fx_strength": bod["fx_strength"], "arms": bod["arms"], "strength": bod["strength"],
            "mass_force": mass_force, "practice": practice, "injury_factor": 1 - 0.5 * state.injury,
            "blow": cfg.strike_damage * bod["strength"] * mass_force * practice * (1 - 0.5 * state.injury),
            "speed_gene": g.speed, "weight_factor": bod["weight_factor"], "fx_speed": bod["fx_speed"],
            "legs": bod["legs"], "speed": bod["speed"],
            "metabolism": g.metabolism, "size_need": size_need, "activity": doing, "pregnancy": preg, "sickness": sick,
            "food_per_hour": cfg.hunger_rate * g.metabolism * size_need * doing * preg * sick * per_hour,
            "water_per_hour": cfg.thirst_rate * size_need * doing * per_hour}


def climate_offset(t, drift, cfg: EnvConfig):
    """C added to every temperature: a slow warm-cool cycle plus random drift (0 when off)."""
    if cfg.climate_cycle_years <= 0:
        return jnp.float32(0.0)
    cycle = cfg.climate_cycle_years * cfg.year_length
    return cfg.climate_amp * jnp.sin(2 * jnp.pi * t / cycle) + drift


def ambient_temperature(t, weather, terrain_at, cfg: EnvConfig, height_at=0.0, row_at=None, climate=0.0):
    """Degrees C at given cells. Seasons, day/night, weather, altitude, latitude, biome, climate,
    forest and cave buffering."""
    season = season_of(t, cfg)
    base = jnp.array(cfg.season_temp)[season]
    hour = hour_of_day(t, cfg)
    temp = base + cfg.day_temp_swing * jnp.cos(2 * jnp.pi * (hour - cfg.warmest_hour) / 24.0)
    temp = temp + jnp.where(weather == COLD_SNAP, -cfg.extreme_temp_shift, 0.0)
    temp = temp + jnp.where(weather == HEAT_WAVE, cfg.extreme_temp_shift, 0.0)
    temp = temp - cfg.altitude_cold * height_at                       # colder up high
    temp = temp + climate
    if cfg.biomes:
        if row_at is not None:
            temp = temp - cfg.latitude_cold * (0.5 - row_at / max(cfg.grid_size - 1, 1))   # north colder
        temp = temp + jnp.where(terrain_at == DESERT, cfg.desert_heat, 0.0) \
            - jnp.where(terrain_at == TUNDRA, cfg.tundra_cold, 0.0)
        temp = jnp.where(terrain_at == CAVE, temp + cfg.cave_buffer * (cfg.comfort_temp - temp), temp)
    buffered = temp + cfg.forest_buffer * (cfg.comfort_temp - temp)
    return jnp.where(terrain_at == FOREST, buffered, temp)


def update_weather(state, weather, season, key, cfg: EnvConfig):
    """Dynamic weather for one tick: wind, clouds (evaporation, drift, fronts, decay), where it
    rains or snows, snow lying and melting, storms. Returns clouds, snow, wind, raining, storms."""
    g = cfg.grid_size
    k_wind, k_dir, k_front, k_pos = jax.random.split(key, 4)
    shift = jax.random.bernoulli(k_wind, cfg.wind_change_prob)
    new_dir = NEIGHBORS8[jax.random.randint(k_dir, (), 0, 8)]
    wind = jnp.where(shift, new_dir, state.wind).astype(jnp.int32)
    evap = jnp.array([1.0, 1.3, 1.0, 0.6])[season] * cfg.evaporation
    source = drinkable(state.terrain).astype(jnp.float32) * evap
    clouds = state.clouds * cfg.cloud_decay * jnp.where(state.terrain == DESERT, 0.97 ** _k(cfg), 1.0) + source
    # drift one cell downwind every cloud_drift_every ticks (clouds leave the map at the edge)
    pad = jnp.pad(clouds, 1)
    moved = jax.lax.dynamic_slice(pad, (1 - wind[0], 1 - wind[1]), (g, g))
    clouds = jnp.where(state.t % cfg.cloud_drift_every == 0, moved, clouds)
    # a front: a broad band of cloud arriving from upwind
    rr, cc = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    centre = jax.random.randint(k_pos, (2,), 0, g)
    blob = jnp.exp(-((rr - centre[0]) ** 2 + (cc - centre[1]) ** 2) / (2 * (g / 6) ** 2))
    clouds = clouds + jnp.where(jax.random.bernoulli(k_front, cfg.front_prob * jnp.where(weather == DROUGHT, 0.2, 1.0)),
                                cfg.front_strength * blob, 0.0)
    # smooth a little so cover is patchy, not speckled
    pad = jnp.pad(clouds, 1, mode="edge")
    blur = sum(pad[1 + dr:1 + dr + g, 1 + dc:1 + dc + g] for dr in (-1, 0, 1) for dc in (-1, 0, 1)) / 9.0
    clouds = jnp.clip(0.8 * clouds + 0.2 * blur, 0.0, 1.0)
    raining = clouds > cfg.rain_threshold
    clouds = jnp.where(raining, clouds - cfg.rain_use, clouds)
    storms = clouds > cfg.storm_threshold
    rows = jnp.broadcast_to(jnp.arange(g)[:, None], (g, g)).astype(jnp.float32)
    air = ambient_temperature(state.t, weather, state.terrain, cfg, state.height, rows,
                              climate_offset(state.t, state.climate, cfg))
    snowing = raining & (air < 0.0) & (state.terrain != WATER) & (state.terrain != CAVE)
    snow = jnp.clip(state.snow + jnp.where(snowing, cfg.snow_rate, 0.0)
                    - cfg.snow_melt * jnp.maximum(air, 0.0), 0.0, 1.0)
    return clouds, snow, wind, raining & ~snowing, storms


# ---------------------------------------------------------------- reset
_COND_BASE = jnp.array([c[3] for c in cond.CONDITIONS])
_COND_INHERIT = jnp.array([c[4] for c in cond.CONDITIONS])
_COND_INBORN = jnp.array([c[1] == "inborn" for c in cond.CONDITIONS])


def draw_inborn(key, n, mother, father):
    """(n, K) inborn conditions: each at its base rate, or its inherit rate if a parent has it."""
    has_parent = jnp.zeros((n, cond.K), bool) if mother is None else (mother > 0.5) | (father > 0.5)
    p = jnp.where(has_parent & (_COND_INHERIT > 0), _COND_INHERIT, _COND_BASE)
    return (jax.random.uniform(key, (n, cond.K)) < p) & _COND_INBORN


def condition_effects(c):
    """Combined effects of each agent's conditions (c: (N, K) 0/1) as a dict of (N,) arrays:
    multipliers (1 = no effect) and, for sight, cells added."""
    out = {}
    for name, eff in cond.EFFECTS.items():
        on = c[..., cond.IDX[name]] > 0.5
        for k, v in eff.items():
            if k in ("sight", "night_sight", "sleepy_sight"):
                out[k] = out.get(k, 0) + jnp.where(on, v, 0)
            else:
                out[k] = out.get(k, 1.0) * jnp.where(on, v, 1.0)
    return out


def dark_radius(state, cfg: EnvConfig):
    """(N,) how far each agent sees in the dark (keen or poor eyes change it)."""
    fx = condition_effects(state.conditions)
    return jnp.maximum(cfg.night_view_radius + fx.get("night_sight", 0) + fx.get("sight", 0), 1)


def draw_selfless(key, n, cfg: EnvConfig):
    """(n,) selflessness: 0 for nearly everyone; with probability saint_rate, between 0.5 and 1.5
    (enough that a stranger's death can weigh about as much as one's own)."""
    k1, k2 = jax.random.split(key)
    return jnp.where(jax.random.uniform(k1, (n,)) < cfg.saint_rate, jax.random.uniform(k2, (n,), minval=0.5, maxval=1.5), 0.0)


def reset(key: jax.Array, cfg: EnvConfig, genotype: gt.Genotype | None = None,
          appearance: jnp.ndarray | None = None) -> State:
    """New world. Pass `genotype`/`appearance` to keep a fixed population
    (the same individuals) across many replica worlds."""
    ks = jax.random.split(key, 11)
    g, n = cfg.grid_size, cfg.n_agents
    terrain = generate_terrain(ks[0], cfg)
    height = generate_height(ks[0], cfg)
    cv = generate_caves(jax.random.fold_in(key, 55), terrain, cfg, height)
    terrain = cv["terrain"]                                   # with sinkholes and hot springs
    under, under_water, cave_floor = cv["under"], cv["under_water"], cv["cave_floor"]
    base = jnp.select([terrain == FOREST, terrain == GRASS, terrain == MARSH, terrain == TUNDRA, terrain == DESERT],
                      [0.7, 0.4, 0.5, 0.15, 0.1], 0.0)
    plants = base * cfg.plant_max * cfg.plant_abundance * jax.random.uniform(ks[1], (g, g))
    if cfg.ecology:
        flora, flora_patch = eco.init(jax.random.fold_in(key, 61), terrain, cfg)
        plants = flora.sum(-1)
    else:
        flora, flora_patch = jnp.zeros((g, g, eco.N_KINDS)), jnp.ones((g, g, eco.N_KINDS))
    pos = _spawn_band(ks[2], terrain, n, cfg.founder_spread, cfg.camp_water_radius)
    body = jax.random.uniform(ks[3], (3, n), minval=0.8, maxval=1.0)
    geno = gt.sample(ks[4], n) if genotype is None else genotype
    geno = geno._replace(selflessness=jnp.maximum(geno.selflessness, draw_selfless(jax.random.fold_in(ks[4], 9), n, cfg)))
    sex = (jnp.arange(n) % 2).astype(jnp.int32)
    if cfg.archetypes:
        from . import archetypes as arch
        geno, sex = arch.apply(geno, sex, cfg.archetypes, cfg.n_founders if cfg.n_founders > 0 else n)
    if appearance is None:
        appearance = jax.random.uniform(ks[9], (n, N_APPEARANCE))
    visited = jnp.zeros((n, g, g), bool).at[jnp.arange(n), pos[:, 0], pos[:, 1]].set(True)
    founders = jnp.arange(n) < (cfg.n_founders if cfg.n_founders > 0 else n)
    return State(
        key=ks[8], t=jnp.int32(0), terrain=terrain, height=height, cliff=generate_cliffs(height), plants=plants,
        flora=flora, flora_patch=flora_patch, foul=jnp.zeros((g, g)), urge=jnp.zeros(n),
        marks=jnp.zeros((g, g, MSG_LEN), jnp.int32), mark_time=jnp.full((g, g), -1, jnp.int32),
        mark_kind=jnp.zeros((g, g), jnp.int32),
        meat=jnp.zeros((g, g)), cooked=jnp.zeros((g, g)),
        wood=jnp.where(terrain == FOREST, cfg.wood_max * jax.random.uniform(ks[10], (g, g)), 0.0),
        stones=jnp.zeros((g, g)), minerals=generate_minerals(jax.random.fold_in(key, 99), terrain, cfg) + cv["surface_minerals"],
        kiln=jnp.zeros((g, g)), fire=jnp.zeros((g, g)), weather=jnp.int32(CLEAR),
        mutations=jnp.zeros((n, len(gt.Genotype._fields))),
        clouds=jnp.zeros((g, g)), snow=jnp.zeros((g, g)), wind=jnp.array([0, 1], jnp.int32),
        climate=jnp.float32(0.0),
        weather_timer=jnp.int32(cfg.spell_min),
        pos=pos, satiety=body[0], hydration=body[1], energy=body[2], health=jnp.ones(n),
        inv=jnp.zeros((n, N_MAT)), hand=jnp.zeros(n, jnp.int32), torch=jnp.zeros(n, jnp.int32),
        age=jnp.zeros(n, jnp.int32),
        alive=founders, visited=visited, mind_map=jnp.zeros((n, g, g, N_MAP_STORED)),
        sleepiness=jnp.zeros(n), asleep=jnp.zeros(n, bool), sleep_depth=jnp.zeros(n, jnp.int32),
        collapsed=jnp.zeros(n, bool), body_temp=jnp.zeros(n),
        injury=jnp.zeros(n), infection=jnp.zeros(n, jnp.int32), immunity=jnp.zeros(n),
        stress=jnp.zeros(n), pain_aversion=geno.pain_aversion, genotype=geno,
        clothing=jnp.where(founders, jax.random.uniform(jax.random.fold_in(key, 71), (n,), minval=cfg.clothing_min,
                                                         maxval=1.0), 0.0) if cfg.clothing else jnp.zeros(n),
        weight=cfg.ideal_bmi * (cfg.base_height_cm * geno.height / 100.0) ** 2 * geno.build,
        drop_owner=jnp.full((g, g), -1, jnp.int32), drop_time=jnp.zeros((g, g), jnp.int32),
        habituation=jnp.zeros((n, n, N_HABIT)), attachment=jnp.zeros((n, n)),
        conditions=draw_inborn(jax.random.fold_in(ks[4], 17), n, None, None).astype(jnp.float32), father_cond=jnp.zeros((n, cond.K)),
        life_moves=jnp.zeros(n, jnp.int32), life_strikes=jnp.zeros(n, jnp.int32), hardship=jnp.zeros(n, jnp.int32),
        impression=jnp.full((n, n), 2.5 * cfg.wellbeing_scale),   # founders know each other, all looking well
        scars=jnp.zeros(n), lasting_injury=jnp.zeros(n), pain=jnp.zeros(n), heartache=jnp.zeros(n), part_damage=jnp.zeros((n, N_PARTS)),
        held_by=jnp.full(n, -1, jnp.int32), hold_grip=jnp.zeros(n), off_balance=jnp.zeros(n, jnp.int32),
        sex=sex, existed=founders, gestation=jnp.zeros(n, jnp.int32), courting=jnp.zeros(n, jnp.int32),
        father=jnp.full(n, -1, jnp.int32), mother=jnp.full(n, -1, jnp.int32),
        relatedness=jnp.eye(n), fetus_r=jnp.zeros((n, n)), father_geno=geno, father_look=appearance,
        birth_age=(jax.random.uniform(ks[10], (n,), minval=cfg.founder_age_range[0],
                                      maxval=cfg.founder_age_range[1]) * cfg.ticks_per_bio_year).astype(jnp.int32),
        appearance=appearance, memory=jnp.zeros((n, n, N_MEM)),
        last_msg=jnp.zeros((n, MSG_LEN), jnp.int32),
        under=under, under_water=under_water, under_kind=cv["under_kind"], spring=cv["spring"], cave_floor=cave_floor, below=jnp.zeros(n, bool),
        small=_new_animals(ks[5], terrain, cfg.n_small_prey, fa.SMALL, cfg, under & ~under_water),
        large=_new_animals(ks[6], terrain, cfg.n_large_prey, fa.LARGE, cfg, under & ~under_water),
        predators=_new_animals(ks[7], terrain, cfg.n_predators, fa.PRED, cfg, under & ~under_water),
        **generate_bugs(jax.random.fold_in(key, 77), terrain, cfg),
    )


# ---------------------------------------------------------------- materials on cells
def generate_minerals(key, terrain, cfg: EnvConfig):
    """(G, G, M) deposits where geology puts them: each mineral has a chance per cell by terrain
    (clay mostly by water, salt in deserts, ores in rock, bog iron in marsh) and an amount."""
    g = cfg.grid_size
    chance = jnp.array(mn.DEPOSIT_CHANCE)[:, terrain] * cfg.mineral_abundance          # (M, G, G)
    lo, hi = (jnp.array([a[k] for a in mn.DEPOSIT_AMOUNT], jnp.float32)[:, None, None] for k in (0, 1))
    k1, k2 = jax.random.split(key)
    here = jax.random.uniform(k1, chance.shape) < chance
    amount = lo + (hi - lo) * jax.random.uniform(k2, chance.shape)
    return jnp.moveaxis(jnp.where(here, jnp.round(amount), 0.0), 0, -1)


def available_here(state: State, r, c, below=None):
    """(N_MAT, N) amount of each material an agent at (r, c) could pick up. below (N,) bool: those
    underground can take only stone (the walls, unlimited) and the ore in the walls there."""
    rock = state.terrain[r, c] == ROCK
    top = jnp.concatenate([jnp.stack([state.plants[r, c], state.meat[r, c], state.cooked[r, c], state.wood[r, c],
                                      jnp.where(rock, 1e6, state.stones[r, c]), state.honey[r, c], state.grubs[r, c]]),
                           state.minerals[r, c].T])
    if below is None:
        return top
    deep = state.cave_floor[r, c].T.at[STONE].set(1e6)
    return jnp.where(below[None, :], deep, top)


def surface_at(state: State, r, c, cfg: EnvConfig):
    """What would take a mark at (r, c) (SURF_*): wood lying there first (a board, a sign), then a
    loose stone, then the rock face, then a tree's bark; SURF_NONE on bare ground."""
    t = state.terrain[r, c]
    return jnp.where(state.wood[r, c] >= cfg.sign_min, SURF_WOOD,
           jnp.where(state.stones[r, c] >= cfg.sign_min, SURF_STONE,
           jnp.where(t == ROCK, SURF_ROCK, jnp.where(t == FOREST, SURF_BARK, SURF_NONE))))


def write_means(state: State, cfg: EnvConfig):
    """(N,) whether each agent can write where it stands, (N,) on what (SURF_*) and (N,) whether in
    pigment: something in hand (an edge that can cut that surface, or a pigment) and a surface here."""
    n = state.pos.shape[0]
    r, c = state.pos[:, 0], state.pos[:, 1]
    hand = state.hand
    holding = state.inv[jnp.arange(n), hand] > 0.05
    surf = surface_at(state, r, c, cfg)
    cut = holding & (SHARPNESS[hand] >= cfg.write_edge) & (HARDNESS[hand] >= SURF_HARD[surf] * cfg.write_hard)
    paint = holding & (PIGMENT[hand] > 0) & (surf != SURF_BARK)
    ok = (surf != SURF_NONE) & (cut | paint)
    return ok, surf, paint & ~cut


def loose_plants(state: State, cfg: EnvConfig):
    """(G, G) plant food lying loose on the ground (dropped or placed), apart from what grows."""
    if not cfg.ecology:
        return state.plants
    return jnp.maximum(state.plants - state.flora.sum(-1), 0.0)


def forage_grid(state: State, cfg: EnvConfig):
    """(G, G) how good each cell is for getting plant food, on the classic scale (0 .. plant_max).
    With ecology: the best share of the full return rate any kind would give here, or loose food."""
    if not cfg.ecology:
        return state.plants
    q = eco.quality(state.flora, cfg).max(-1)
    return jnp.maximum(q, 1.0 - jnp.exp(-loose_plants(state, cfg))) * cfg.plant_max


# ---------------------------------------------------------------- masks
def action_mask(state: State, cfg: EnvConfig) -> jnp.ndarray:
    g = cfg.grid_size
    deep = cfg.underground
    below = state.below
    tgt = state.pos[:, None, :] + NEIGHBORS8[None]
    ok = _in_bounds(tgt, g)
    tc = jnp.clip(tgt, 0, g - 1)
    can_move = ok & (state.terrain[tc[..., 0], tc[..., 1]] != WATER)
    if deep:                                                         # below: only along passages
        passage = state.under & ~state.under_water
        can_move = jnp.where(below[:, None], ok & passage[tc[..., 0], tc[..., 1]], can_move)
    r, c = state.pos[:, 0], state.pos[:, 1]
    n = state.pos.shape[0]
    carried = state.inv.sum(axis=1)
    held = state.inv[jnp.arange(n), state.hand]
    avail = available_here(state, r, c, below if deep else None)     # (N_MAT, N)
    here = avail[state.hand, jnp.arange(n)]
    torch_source = (state.hand == WOOD) & (state.fire[r, c] > 0) & ~below   # a branch can be pulled from a fire
    empty_handed = held <= 0.05
    anything_here = jnp.any(avail > 0.1, axis=0)
    can_grasp = ((here > 0.1) | torch_source | anything_here) \
        & (carried < cfg.inventory_cap - 1e-3)
    edible = EDIBLE[state.hand]
    can_eat = edible & (held > 0.05)
    can_drink = adjacent_to_water(state.terrain, state.pos, cfg)
    # Affordances: touch needs someone there; a strike needs something to hit
    # (an agent, an animal, or a hard surface that a hard object could spark on)
    occ_top = _occupancy(state.pos, state.alive & ~below, g)
    agent_there = occ_top[tc[..., 0], tc[..., 1]] > 0
    animals_on = lambda deep_: sum(_occupancy(an.pos, an.alive & (an.below == deep_), g)
                                   for an in (state.small, state.large, state.predators))
    animal_there = animals_on(False)[tc[..., 0], tc[..., 1]] > 0
    hard_there = (state.terrain[tc[..., 0], tc[..., 1]] == ROCK) | (state.stones[tc[..., 0], tc[..., 1]] > 0.1) \
        | jnp.any((state.minerals[tc[..., 0], tc[..., 1]] > 0.1) & (jnp.array(mn.HARDNESS) >= 0.5), axis=-1)
    if deep:                                                         # below: those below, and the walls
        b2 = below[:, None]
        can_drink = jnp.where(below, jnp.any(ok & state.under_water[tc[..., 0], tc[..., 1]], axis=1), can_drink)
        occ_deep = _occupancy(state.pos, state.alive & below, g)
        agent_there = jnp.where(b2, occ_deep[tc[..., 0], tc[..., 1]] > 0, agent_there)
        animal_there = jnp.where(b2, animals_on(True)[tc[..., 0], tc[..., 1]] > 0, animal_there)
        hard_deep = ~state.under[tc[..., 0], tc[..., 1]] \
            | jnp.any((state.cave_floor[tc[..., 0], tc[..., 1], MIN0:] > 0.1) & (jnp.array(mn.HARDNESS) >= 0.5), axis=-1)
        hard_there = jnp.where(b2, hard_deep, hard_there)
    can_strike = ok & (agent_there | animal_there | hard_there)
    can_touch = ok & agent_there
    can_release = held > 0.05
    # Select only what one carries or what lies here (wood also from a fire underfoot)
    can_select = (state.inv > 0.05) | (avail.T > 0.1) \
        | ((jnp.arange(N_MAT) == WOOD)[None, :] & ((state.fire[r, c] > 0) & ~below)[:, None])
    # Reproduction: infants cannot walk or forage; adults can offer to mate with an adjacent
    # adult of the other sex (it takes both choosing it); a pregnant woman cannot conceive.
    years = (state.age + state.birth_age) / cfg.ticks_per_bio_year
    if cfg.reproduction:
        infant = years < cfg.infant_years
        can_move = can_move & ~infant[:, None]
        can_grasp = can_grasp & ~infant
        can_strike = can_strike & ~infant[:, None]                   # a baby cannot land a blow
        adult = years >= cfg.adult_years
        lp = layered_pos(state.pos, below)
        near = (_cheb(lp[:, None, :], lp[None, :, :]) <= 1) & ~jnp.eye(state.pos.shape[0], dtype=bool)
        to_other, to_same = gt.attraction_of(state.genotype)
        other_sex = state.sex[None, :] != state.sex[:, None]
        drawn = jnp.where(other_sex, to_other[:, None], to_same[:, None])          # i drawn to j's sex
        partner_there = jnp.any(near & state.alive[None, :] & adult[None, :] & drawn, axis=1)
        can_mate = adult & partner_there & ~((state.sex == FEMALE) & (state.gestation > 0))
    else:
        can_mate = jnp.zeros_like(can_eat)
    can_place = can_move & (held > 0.05)[:, None]
    if cfg.body_parts:                                               # legs too badly hurt to walk on
        legs_ok = (state.part_damage[:, PART_LEGS] < cfg.legs_unusable)[:, None]
        can_move = can_move & legs_ok
    # Delve: on a cave mouth, down into the caves or back up (as walking: not an infant)
    if deep:
        can_delve = (state.terrain[r, c] == CAVE) & state.under[r, c]
        if cfg.reproduction:
            can_delve = can_delve & ~infant
    else:
        can_delve = jnp.zeros_like(can_eat)
    if cfg.body_parts:
        can_delve = can_delve & legs_ok[:, 0]
    skills_ok = jnp.broadcast_to(jnp.asarray(cfg.skills), (n, N_SKILLS)) & jnp.any(can_move, axis=1)[:, None] \
        & ~below[:, None]                                            # the given skills know only the surface
    can_relieve = (state.urge >= cfg.relieve_min) if cfg.excretion else jnp.zeros_like(can_eat)
    can_write = (~below & ~state.asleep & write_means(state, cfg)[0]) if cfg.writing else jnp.zeros_like(can_eat)
    # Sleep needs sleep pressure (the body clock lowers the bar at night); asleep, one may sleep on
    can_sleep = state.asleep | (state.sleepiness >= jnp.where(is_night(state.t, cfg), cfg.sleep_onset_night,
                                                              cfg.sleep_onset))
    mask = jnp.concatenate([jnp.ones_like(can_eat)[:, None], can_move, can_grasp[:, None],
                            can_eat[:, None], can_drink[:, None], can_strike,
                            can_release[:, None], can_touch, can_select, can_place, can_mate[:, None],
                            can_sleep[:, None], skills_ok, can_delve[:, None], can_relieve[:, None],
                            can_write[:, None]],
                           axis=1)
    dead = jnp.zeros_like(mask).at[:, REST].set(True)
    # Asleep, one can only sleep on or wake up (REST); after collapsing from exhaustion, only sleep on
    asleep_row = jnp.zeros_like(mask).at[:, SLEEP].set(True)
    asleep_row = asleep_row.at[:, REST].set(~(state.collapsed & (state.sleepiness >= cfg.sleep_wake_block)))
    mask = jnp.where(state.asleep[:, None], asleep_row, mask)
    if cfg.downed:                                                   # down: lie, eat, drink, sleep, call out
        keep = jnp.zeros(mask.shape[1], bool).at[jnp.array([REST, GRASP, EAT, DRINK, RELEASE, SLEEP, RELIEVE])].set(True)
        mask = jnp.where((downed_of(state, cfg) & ~state.asleep)[:, None], mask & keep[None, :], mask)
    return jnp.where(state.alive[:, None], mask, dead)


def pain_of(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N,) how much it hurts: the ache of standing injury plus the fresh pain of recent wounds."""
    return cfg.pain_per_injury * state.injury + state.pain


def shock_threshold(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N,) pain at which one goes into shock and drops: cfg.pain_shock, lower for those who feel
    pain more (the heritable pain_aversion trait: a wimp drops sooner, the stoic later)."""
    return jnp.clip(cfg.pain_shock / jnp.sqrt(state.pain_aversion), 0.2, 0.99)


def felt_pain(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N,) pain as one feels it: measured against one's own tolerance, so 1 is the point of shock.
    Felt by the one in pain (observe) and shown to those who can see them (a wince)."""
    tol = shock_threshold(state, cfg) if cfg.pain_shock > 0 else 0.8 / jnp.sqrt(state.pain_aversion)
    return jnp.clip(pain_of(state, cfg) / tol, 0.0, 1.5)


def downed_of(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N,) bool: down (EnvConfig.downed): too hurt to stand, from wounds (injury), failing health,
    or pain shock (RimWorld's: pain past one's threshold, until it fades).
    The downed lie where they are and cannot walk, strike, touch, place or mate; they can still pick
    up and eat what is on their own cell, drink if water is beside them, sleep, and call out."""
    if not cfg.downed:
        return jnp.zeros_like(state.alive)
    shock = (pain_of(state, cfg) >= shock_threshold(state, cfg)) if cfg.pain_shock > 0 else jnp.zeros_like(state.alive)
    knocked_out = state.part_damage[:, PART_HEAD] >= cfg.concussion_at if cfg.body_parts else jnp.zeros_like(state.alive)
    return state.alive & ((state.health < cfg.downed_health) | (state.injury >= cfg.downed_injury) | shock | knocked_out)


def _techniques(state, cfg, bod, strength, mass_force, limb, mode, sdir, landed, blow, alive0, below0, down0,
                is_strike, new_pos, part_by_blow, violence, fall_damage, injury, key):
    """What pushes and holds do (EnvConfig.techniques), given who reached whom this tick: landed
    (striker, victim), the blow each dealt, each striker's limb and mode and direction. Returns the
    updated part damage, violence, positions, fall damage and injury, and the next hold and balance."""
    n, g = cfg.n_agents, cfg.grid_size
    k = jax.random.split(key, 5)
    idx = jnp.arange(n)
    power = strength * mass_force * bod["legs"]                       # footing matters in a push
    resist = power * jnp.where(down0 | (state.off_balance > 0), cfg.unsteady_resist, 1.0)

    def contest(lev):                                                 # (striker,) -> (striker, victim)
        a_ = (lev * power)[:, None]
        return a_ / (a_ + resist[None, :] + 1e-8)
    held_last = state.held_by[None, :] == idx[:, None]               # striker held victim last tick
    # Pushes: a shove moves the victim a cell the way it was pushed; on someone held, a throw
    pushing = landed & (mode == MODE_PUSH)[:, None]
    p_push = jnp.minimum(contest(jnp.asarray(cfg.limb_push)[limb]) * jnp.where(held_last, cfg.throw_leverage, 1.0), 0.95)
    shove = pushing & (jax.random.uniform(k[0], (n, n)) < p_push)
    thrown_m = shove & held_last
    delta = jnp.clip(jnp.round(shove.astype(jnp.float32).T @ MOVE_DELTAS[sdir].astype(jnp.float32)), -1, 1).astype(jnp.int32)
    shoved = jnp.any(shove, axis=0) & alive0
    dest_raw = new_pos + delta
    dest = jnp.clip(dest_raw, 0, g - 1)
    room = _in_bounds(dest_raw, g) & jnp.where(below0, (state.under & ~state.under_water)[dest[:, 0], dest[:, 1]], True)
    displaced = shoved & room & jnp.any(delta != 0, axis=-1)
    new_pos = jnp.where(displaced[:, None], dest, new_pos)
    # ... onto rough or steep ground: a fall
    hz = jnp.where(below0, 0.0, jnp.where(state.terrain[dest[:, 0], dest[:, 1]] == ROCK, 0.2, 0.0)
                   + state.cliff[dest[:, 0], dest[:, 1]])
    falls = displaced & (jax.random.uniform(k[1], (n,)) < jnp.minimum(cfg.shove_fall_mult * hz, 1.0))
    extra_fall = jnp.where(falls, hz * cfg.fall_severity * jax.random.exponential(k[2], (n,)), 0.0)
    fall_damage = fall_damage + extra_fall
    injury = injury + cfg.fall_injury_ratio * extra_fall
    # A throw: to the ground, on one's back
    throw = cfg.throw_damage * (thrown_m.astype(jnp.float32) * (strength * mass_force)[:, None]).sum(0)
    thrown = jnp.any(thrown_m, axis=0) & alive0
    violence = violence + throw
    part_by_blow = part_by_blow.at[:, PART_TORSO].add(throw * cfg.part_fragility[PART_TORSO])
    # A headbutt jars one's own head too
    butt = jnp.where(limb == LIMB_HEAD, cfg.headbutt_recoil * blow.sum(1), 0.0)
    part_by_blow = part_by_blow.at[:, PART_HEAD].add(butt)
    # Holds: the strongest hold that took is the one that pins next tick
    holding = landed & (mode == MODE_HOLD)[:, None]
    p_hold = contest(jnp.asarray(cfg.limb_grip)[limb])
    got = holding & (jax.random.uniform(k[3], (n, n)) < p_hold)
    grip = jnp.where(got, p_hold, 0.0)
    grabbed = jnp.any(got, axis=0) & alive0
    held_by = jnp.where(grabbed, jnp.argmax(grip, axis=0), -1).astype(jnp.int32)
    hold_grip = jnp.where(grabbed, grip.max(0), 0.0)
    # Balance: a shove staggers, a throw floors, a kick may cost the kicker its footing
    kick_slip = is_strike & (limb == LIMB_FOOT) & (jax.random.uniform(k[4], (n,))
                                                   < cfg.kick_balance * (1 + state.part_damage[:, PART_LEGS]))
    knock = jnp.maximum(jnp.where(shoved, cfg.shove_ticks, 0), jnp.where(thrown, cfg.throw_ticks, 0))
    knock = jnp.maximum(knock, jnp.where(kick_slip, 1, 0))
    off_balance = jnp.maximum(state.off_balance - 1, knock).astype(jnp.int32)
    tech = dict(held_by=held_by, hold_grip=hold_grip, off_balance=off_balance,
                shoved=displaced, thrown=thrown, grabbed=grabbed)
    return part_by_blow, violence, new_pos, fall_damage, injury, tech


def caregiver_of(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N,) int: who nurses (and, for an infant, carries) each young child: its mother while she
    lives; an orphan, the most closely related woman within reach who is nursing a child of her own
    (cfg.wet_nursing; allomaternal nursing is common among foragers). -1: no one."""
    n = state.pos.shape[0]
    years = (state.age + state.birth_age) / cfg.ticks_per_bio_year
    young = state.alive & (state.mother >= 0) & (years < cfg.nurse_years)
    mom = jnp.clip(state.mother, 0)
    has_mom = young & state.alive[mom]
    if not cfg.wet_nursing:
        return jnp.where(has_mom, mom, -1).astype(jnp.int32)
    nursing_own = jnp.zeros(n, bool).at[mom].max(has_mom)
    lactating = state.alive & (state.sex == FEMALE) & nursing_own
    lp = layered_pos(state.pos, state.below)
    near = _cheb(lp[:, None, :], lp[None, :, :]) <= 1
    cand = near & lactating[None, :] & ~jnp.eye(n, dtype=bool)
    best = jnp.argmax(jnp.where(cand, state.relatedness + 0.01, -1.0), axis=1)
    return jnp.where(has_mom, mom, jnp.where(young & jnp.any(cand, axis=1), best, -1)).astype(jnp.int32)


# ---------------------------------------------------------------- given skills
def route_field(targets, terrain, cliff, cfg: EnvConfig):
    """(N, G, G) steps from each cell to the nearest target cell, per agent, walking around water
    and counting steep cells as longer (inf where no target can be reached)."""
    g = cfg.grid_size
    land = terrain != WATER
    cost = 1.0 + cfg.skill_cliff_avoid * cliff                               # (G, G)
    big = jnp.float32(1e9)
    d0 = jnp.where(targets & land[None], 0.0, big)

    def sweep(_, d):
        p = jnp.pad(d, ((0, 0), (1, 1), (1, 1)), constant_values=big)
        best = d
        for dr, dc in NEIGHBORS8.tolist():
            best = jnp.minimum(best, p[:, 1 + dr:1 + dr + g, 1 + dc:1 + dc + g] + cost[None])
        return jnp.where(land[None], best, big)

    return jax.lax.fori_loop(0, cfg.skill_search_steps, sweep, d0)


def _step_down(field, pos, mask_move, cfg: EnvConfig):
    """(N,) the move (MOVE_N..MOVE_NW) to the neighbour lowest in the field, and whether one exists."""
    g = cfg.grid_size
    nb = jnp.clip(pos[:, None, :] + NEIGHBORS8[None], 0, g - 1)              # (N, 8, 2)
    vals = field[jnp.arange(pos.shape[0])[:, None], nb[..., 0], nb[..., 1]]
    vals = jnp.where(mask_move, vals, 1e9)
    k = jnp.argmin(vals, axis=1)
    return MOVE_N + k, vals[jnp.arange(pos.shape[0]), k] < 1e8


def resolve_skills(state: State, a, mask, cfg: EnvConfig):
    """Turn a chosen skill into this tick's primitive action, using only what the agent knows."""
    g, n = cfg.grid_size, state.pos.shape[0]
    r, c = state.pos[:, 0], state.pos[:, 1]
    mm = state.mind_map
    known = mm[..., 2] > 0.05                                                # remembered at all
    pad = jnp.pad(mm[..., 0] * known, ((0, 0), (1, 1), (1, 1)))
    near_water = jnp.zeros((n, g, g))
    for dr, dc in NEIGHBORS8.tolist():
        near_water = jnp.maximum(near_water, pad[:, 1 + dr:1 + dr + g, 1 + dc:1 + dc + g])
    water_spots = near_water > 0.5
    food_spots = (mm[..., 1] > 0.2) & known
    # Patch choice (Charnov's marginal value theorem): leave, and skip, patches poorer than a share
    # (cfg.patch_leave) of the average one knows, so a thinned patch is left for a better one
    if cfg.patch_leave > 0:
        avg = (mm[..., 1] * food_spots).sum((1, 2)) / jnp.maximum(food_spots.sum((1, 2)), 1)
        bar = cfg.patch_leave * avg                                              # (N,) share of a full patch
        richer = food_spots & (mm[..., 1] > bar[:, None, None])
        food_spots = jnp.where(jnp.any(richer, axis=(1, 2))[:, None, None], richer, food_spots)
    else:
        bar = jnp.zeros(n)
    unseen = ~known
    seek_water, seek_food, flee, shelter = a == SEEK_WATER, a == SEEK_FOOD, a == FLEE, a == SEEK_SHELTER
    # shelter: forest or cave cells it has seen (terrain is seen, so a remembered cell's terrain is known)
    cover = (state.terrain == FOREST) | (state.terrain == CAVE)
    shelter_spots = cover[None] & known
    targets = jnp.where(seek_water[:, None, None], water_spots,
                        jnp.where(shelter[:, None, None], shelter_spots, food_spots))
    has = jnp.any(targets, axis=(1, 2))
    targets = jnp.where(has[:, None, None], targets, unseen)                   # nothing known: explore
    field = route_field(targets, state.terrain, state.cliff, cfg)
    can_move = mask[:, MOVE_N:MOVE_NW + 1]
    go, reachable = _step_down(field, state.pos, can_move, cfg)
    go = jnp.where(reachable, go, REST)
    # Arrived: drink; or select food, gather it, eat it (never raw meat: that stays a choice)
    held = state.inv[jnp.arange(n), state.hand]
    edible_in_hand = ((state.hand == PLANT) | (state.hand == COOKED_MEAT)) & (held > 0.05)
    carried = jnp.where(state.inv[:, COOKED_MEAT] > 0.05, COOKED_MEAT, PLANT)
    carrying_food = (state.inv[:, COOKED_MEAT] > 0.05) | (state.inv[:, PLANT] > 0.05)
    plants_here = forage_grid(state, cfg)[r, c] > jnp.maximum(0.1 * cfg.plant_max if cfg.ecology else 0.1,
                                                              bar * cfg.plant_max)
    food_act = jnp.where(edible_in_hand, EAT,
               jnp.where(carrying_food, SELECT_0 + carried,
               jnp.where(plants_here & (state.hand == PLANT) & mask[:, GRASP], GRASP,
               jnp.where(plants_here, SELECT_0 + PLANT, go))))
    water_act = jnp.where(mask[:, DRINK], DRINK, go)
    # Flee: the step that most increases distance from the nearest predator in sight
    sight = jnp.where(is_night(state.t, cfg), cfg.night_view_radius, cfg.view_radius)
    dp = _cheb(layered_pos(state.pos, state.below)[:, None, :],
               layered_pos(state.predators.pos, state.predators.below)[None])  # (N, P), own layer only
    dp = jnp.where(state.predators.alive[None], dp, 10 ** 6)
    threat = jnp.argmin(dp, axis=1)
    seen = jnp.min(dp, axis=1) <= sight
    nb = state.pos[:, None, :] + NEIGHBORS8[None]
    away = _cheb(nb, state.predators.pos[threat][:, None, :]).astype(jnp.float32) \
        - cfg.skill_cliff_avoid * state.cliff[jnp.clip(nb[..., 0], 0, g - 1), jnp.clip(nb[..., 1], 0, g - 1)]
    away = jnp.where(can_move, away, -1e9)
    k = jnp.argmax(away, axis=1)
    flee_act = jnp.where(seen & (away[jnp.arange(n), k] > -1e8), MOVE_N + k, REST)
    in_cover = cover[r, c]
    a = jnp.where(shelter, jnp.where(in_cover, REST, go), a)
    a = jnp.where(seek_water, water_act, a)
    a = jnp.where(seek_food, food_act, a)
    return jnp.where(flee, flee_act, a).astype(jnp.int32)


# ---------------------------------------------------------------- observation
def observe(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N, obs_dim): egocentric window (gathered, not padded) + body/time/weather."""
    g, r, n, v = cfg.grid_size, cfg.view_radius, cfg.n_agents, cfg.view_size
    offs = jnp.arange(-r, r + 1)
    rows = state.pos[:, 0, None, None] + offs[None, :, None]          # (N, V, 1)
    cols = state.pos[:, 1, None, None] + offs[None, None, :]          # (N, 1, V)
    rows, cols = jnp.broadcast_arrays(rows, cols)                     # (N, V, V)
    inb = (rows >= 0) & (rows < g) & (cols >= 0) & (cols < g)
    rc, cc = jnp.clip(rows, 0, g - 1), jnp.clip(cols, 0, g - 1)

    def take(grid):
        return jnp.where(inb, grid[rc, cc], 0.0)

    agents_grid = _occupancy(state.pos, state.alive & ~state.below, g)     # those above (below: own window)
    bod = body(state, cfg)
    showing = (state.gestation > 0.5 * cfg.gestation_years * cfg.ticks_per_bio_year).astype(jnp.float32)
    physique = jnp.stack([bod["height_cm"] / 200.0, state.weight / 100.0,
                          state.scars, state.lasting_injury, state.sex.astype(jnp.float32),
                          showing, (state.asleep | downed_of(state, cfg)).astype(jnp.float32),   # lying down
                          (state.courting > 0).astype(jnp.float32)], axis=-1)       # visible to others
    wince = jnp.minimum(felt_pain(state, cfg), 1.0)[:, None]                          # pain shows
    heat = fire_heat(state.fire, cfg)
    chans = [jnp.where(inb[..., None], jax.nn.one_hot(state.terrain[rc, cc], N_TERRAIN), 0.0),
             take(forage_grid(state, cfg) / cfg.plant_max)[..., None],
             take(jnp.minimum(state.meat / cfg.large_prey_meat, 1.0))[..., None],
             take(agents_grid)[..., None],
             take(_occupancy(state.small.pos, state.small.alive & ~state.small.below, g))[..., None],
             take(_occupancy(state.large.pos, state.large.alive & ~state.large.below, g))[..., None],
             take(_occupancy(state.predators.pos, state.predators.alive & ~state.predators.below, g))[..., None],
             (~inb).astype(jnp.float32)[..., None],
             take(jnp.minimum(state.cooked / cfg.large_prey_meat, 1.0))[..., None],
             take(state.wood / cfg.wood_max)[..., None],
             take(jnp.minimum(state.stones, 1.0))[..., None],
             take((state.fire > 0).astype(jnp.float32))[..., None],
             take(heat / cfg.fire_heat + (spring_warmth(state, cfg) / cfg.fire_heat if cfg.underground else 0.0))[..., None],
             take(_occupancy(state.pos, state.existed & ~state.alive & ~state.below, g))[..., None],   # bodies of the dead
             take(state.cliff)[..., None],                                    # cliff height (terrain shape)
             jnp.where(inb, state.height[rc, cc] - state.height[state.pos[:, 0], state.pos[:, 1]][:, None, None],
                       0.0)[..., None],                                       # ground height relative to own
             take(state.clouds * (state.clouds > cfg.rain_threshold))[..., None],   # rain (felt and heard)
             take(state.snow)[..., None]]                                     # snow lying
    # Minerals are seen as they look, not by name: how much lies here, its colour and lustre
    amt = state.minerals.sum(-1)                                                   # (G, G)
    w = state.minerals / jnp.maximum(amt, 1e-6)[..., None]
    looks = jnp.concatenate([jnp.minimum(amt, 3.0)[..., None] / 3.0, w @ MIN_COLOR, (w @ MIN_LUSTER)[..., None]], -1)
    chans.append(jnp.where(inb[..., None], looks[rc, cc], 0.0))
    # Animals as they look (the biggest here, whether it has fangs or tusks, its colour), and bugs
    SPc = species_params(cfg)
    look = jnp.zeros((g, g, 3))
    for an in (state.small, state.large, state.predators):
        f = jnp.stack([SPc["size"], SPc["fangs"], SPc["colour"]], -1)[an.species] * (an.alive & ~an.below)[:, None]
        look = look.at[an.pos[:, 0], an.pos[:, 1]].max(f)
    night_now = is_night(state.t, cfg)
    season_now = season_of(state.t, cfg)
    bugs = jnp.stack([mosquito_density(state, cfg, season_now, night_now, heat),
                      fly_density(state.meat, _occupancy(state.pos, state.existed & ~state.alive & ~state.below, g).astype(jnp.float32), heat),
                      jnp.minimum(state.honey, 1.0) * state.hive, jnp.minimum(state.grubs, 1.0) * state.mound], -1)
    chans.append(jnp.where(inb[..., None], jnp.concatenate([look, bugs], -1)[rc, cc], 0.0))
    # Plants by kind (how well each would gather here) and waste, seen and smelled
    flora_look = jnp.concatenate([eco.quality(state.flora, cfg),
                                  (1.0 - jnp.exp(-state.foul / cfg.foul_ref))[..., None],
                                  jnp.any(state.marks > 0, axis=-1).astype(jnp.float32)[..., None]], -1)
    chans.append(jnp.where(inb[..., None], flora_look[rc, cc], 0.0))
    # Social features: each observer sees, in the cell of every other visible agent,
    # its own memory of that agent, the agent's appearance and last token.
    spoke = (jnp.any(state.last_msg > 0, axis=1) & state.alive).astype(jnp.float32)[:, None]

    def social_window(i):
        rel = state.pos - state.pos[i] + r                             # (N, 2)
        vis = jnp.all((rel >= 0) & (rel < v), axis=-1) & state.alive & (jnp.arange(n) != i) \
            & (state.below == state.below[i])                          # only those on one's own layer
        feats = jnp.concatenate([jnp.log1p(state.memory[i]), state.appearance, spoke, physique,
                                 state.attachment[i][:, None], wince], axis=-1)
        feats = jnp.where(vis[:, None], feats, 0.0)
        relc = jnp.clip(rel, 0, v - 1)
        return jnp.zeros((v, v, N_SOCIAL_FEATURES)).at[relc[:, 0], relc[:, 1]].add(feats)

    social = jax.vmap(social_window)(jnp.arange(n))                    # (N, V, V, F)
    win = jnp.concatenate(chans + [social], axis=-1)                   # (N, V, V, C)
    # Darkness: at night only nearby cells and cells lit by fire are seen. Terrain
    # shapes and the map edge stay visible; food, animals and agents do not.
    cheb_off = jnp.maximum(jnp.abs(offs)[:, None], jnp.abs(offs)[None, :])
    near = cheb_off[None] <= dark_radius(state, cfg)[:, None, None]
    lit = take((heat > 0).astype(jnp.float32)) > 0
    seen = near | lit | ~is_night(state.t, cfg)
    always = jnp.zeros(N_OBS_CHANNELS, bool).at[:N_TERRAIN].set(True).at[N_TERRAIN + 6].set(True) \
        .at[CH_CLIFF].set(True).at[CH_HEIGHT].set(True).at[CH_RAIN].set(True).at[CH_SNOW].set(True)
    win = jnp.where(always[None, None, None, :] | seen[..., None], win, 0.0)
    if cfg.underground:
        # Below: the cave layer instead, in total darkness but for what is next to one (felt) or
        # within a burning branch's light. Plants, animals, weather and fire above do not show.
        zero = jnp.zeros((n, v, v, 1))
        occ_deep = _occupancy(state.pos, state.alive & state.below, g)
        dead_deep = _occupancy(state.pos, state.existed & ~state.alive & state.below, g)
        amt_d = state.cave_floor[..., MIN0:].sum(-1)
        w_d = state.cave_floor[..., MIN0:] / jnp.maximum(amt_d, 1e-6)[..., None]
        looks_d = jnp.concatenate([jnp.minimum(amt_d, 3.0)[..., None] / 3.0, w_d @ MIN_COLOR,
                                   (w_d @ MIN_LUSTER)[..., None]], -1)
        fl = state.cave_floor
        occ_below = lambda an: _occupancy(an.pos, an.alive & an.below, g)
        look_d = jnp.zeros((g, g, 3))
        for an in (state.small, state.large, state.predators):
            f = jnp.stack([SPc["size"], SPc["fangs"], SPc["colour"]], -1)[an.species] * (an.alive & an.below)[:, None]
            look_d = look_d.at[an.pos[:, 0], an.pos[:, 1]].max(f)
        deep = jnp.concatenate(
            [jnp.where(inb[..., None], jax.nn.one_hot(under_terrain(state)[rc, cc], N_TERRAIN), 0.0),
             zero, take(jnp.minimum(fl[..., RAW_MEAT] / cfg.large_prey_meat, 1.0))[..., None],
             take(occ_deep)[..., None], take(occ_below(state.small))[..., None], take(occ_below(state.large))[..., None],
             take(occ_below(state.predators))[..., None], (~inb).astype(jnp.float32)[..., None],
             take(jnp.minimum(fl[..., COOKED_MEAT] / cfg.large_prey_meat, 1.0))[..., None],
             take(fl[..., WOOD] / cfg.wood_max)[..., None], take(jnp.minimum(fl[..., STONE], 1.0))[..., None],
             zero, zero, take(dead_deep)[..., None], zero, zero, zero, zero,
             jnp.where(inb[..., None], looks_d[rc, cc], 0.0),
             jnp.where(inb[..., None], jnp.concatenate([look_d, jnp.zeros((g, g, N_FAUNA_CHANNELS - 3))], -1)[rc, cc], 0.0),
             jnp.zeros((n, v, v, N_FLORA_CHANNELS)), social], axis=-1)
        # What is seen below: one's own cell and those next to it (felt), and whatever any burning
        # branch below lights (one's own or another's)
        rr_, cc_ = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
        d_t = jnp.maximum(jnp.abs(rr_[None] - state.pos[:, 0, None, None]), jnp.abs(cc_[None] - state.pos[:, 1, None, None]))
        bearers = state.alive & state.below & (state.torch > 0)
        torchlit = jnp.any((d_t <= cfg.torch_light) & bearers[:, None, None], axis=0)
        felt = (cheb_off[None] <= 1) | (take(torchlit.astype(jnp.float32)) > 0)
        deep = jnp.where(felt[..., None], deep, 0.0)
        win = jnp.where(state.below[:, None, None, None], deep, win)
    # Asleep: eyes closed, only what is next to one (touch, sound); sleepy: sight narrows
    reach = sight_radius(state, cfg)
    within = jnp.maximum(jnp.abs(offs)[:, None], jnp.abs(offs)[None, :])[None] <= reach[:, None, None]
    win = jnp.where(within[..., None], win, 0.0)
    win = win.at[:, r, r, N_TERRAIN + 2].add(-state.alive.astype(jnp.float32))  # not self

    terr_here = state.terrain[state.pos[:, 0], state.pos[:, 1]]
    amb = ambient_temperature(state.t, state.weather, terr_here, cfg,
                              state.height[state.pos[:, 0], state.pos[:, 1]], state.pos[:, 0].astype(jnp.float32),
                              climate_offset(state.t, state.climate, cfg)) \
        + heat[state.pos[:, 0], state.pos[:, 1]]
    if cfg.underground:
        amb = amb + jnp.where(state.below, 0.0, spring_warmth(state, cfg)[state.pos[:, 0], state.pos[:, 1]])
        amb = jnp.where(state.below, cave_air(state.t, cfg, climate_offset(state.t, state.climate, cfg),
                                              state.under_kind[state.pos[:, 0], state.pos[:, 1]]), amb)
    day_ph = 2 * jnp.pi * (state.t % cfg.day_length) / cfg.day_length
    yr_ph = 2 * jnp.pi * (state.t % cfg.year_length) / cfg.year_length
    shared = jnp.concatenate([
        jnp.array([jnp.sin(day_ph), jnp.cos(day_ph), jnp.sin(yr_ph), jnp.cos(yr_ph),
                   is_night(state.t, cfg).astype(jnp.float32)]),
        jax.nn.one_hot(state.weather, N_WEATHER)])
    own = jnp.stack([
        state.satiety, state.hydration, state.energy, state.health,
        state.age / cfg.max_age, state.body_temp, (amb - cfg.comfort_temp) / cfg.temp_scale,
        state.injury, (state.infection > 0).astype(jnp.float32), state.stress,
        state.pain_aversion / 4.0, bod["height_cm"] / 200.0, state.weight / 100.0,
        bod["strength"], bod["years"] / cfg.max_bio_years,
        state.scars, state.lasting_injury, state.sex.astype(jnp.float32),
        state.gestation / (cfg.gestation_years * cfg.ticks_per_bio_year),
        state.sleepiness, state.asleep.astype(jnp.float32),
        state.below.astype(jnp.float32), state.urge, felt_pain(state, cfg),
        jnp.minimum(state.heartache, 1.0)], axis=-1)                                     # (N, 25)
    own = jnp.concatenate([own, state.part_damage], axis=-1)                           # + each part's damage
    own = jnp.concatenate([own, (state.held_by >= 0).astype(jnp.float32)[:, None],     # + held, off balance
                           (state.off_balance > 0).astype(jnp.float32)[:, None]], axis=-1)
    own = jnp.concatenate([own, state.conditions], axis=-1)                          # + its conditions
    feats = jnp.concatenate([own, state.inv / cfg.inventory_cap,
                             jax.nn.one_hot(state.hand, N_MAT),
                             (state.torch > 0).astype(jnp.float32)[:, None],
                             jnp.where(state.below[:, None] & (jnp.arange(shared.shape[0]) >= 5)[None, :],
                                       0.0, jnp.broadcast_to(shared, (n, shared.shape[0]))),   # below: no weather
                             heard_messages(state, cfg), read_marks(state, cfg)], axis=-1)
    return jnp.concatenate([win.reshape(n, -1), feats, read_mind_map(state.mind_map, state.pos, cfg)], axis=-1)


def _marks_with_surface(marks, mark_time, mark_kind, wood, stones, cfg: EnvConfig):
    """A mark goes with what it is on: wood carried off or burned, a stone taken, and it is gone."""
    on = mark_kind % MARK_PAINTED
    lost = (mark_time >= 0) & (((on == SURF_WOOD) & (wood < cfg.sign_min)) | ((on == SURF_STONE) & (stones < cfg.sign_min)))
    return dict(marks=jnp.where(lost[..., None], 0, marks), mark_time=jnp.where(lost, -1, mark_time),
                mark_kind=jnp.where(lost | (mark_time < 0), 0, mark_kind))


def mark_beside(state: State, cfg: EnvConfig):
    """(N,) whether each agent can read an inscription (on its own cell, else the first beside it,
    above ground and awake), and (N, 2) where."""
    g = cfg.grid_size
    cand = state.pos[:, None, :] + MOVE_DELTAS[None]                         # own cell first
    ok = _in_bounds(cand, g)
    cc = jnp.clip(cand, 0, g - 1)
    has = ok & jnp.any(state.marks[cc[..., 0], cc[..., 1]] > 0, axis=-1)
    first = jnp.argmax(has, axis=1)
    can = jnp.any(has, axis=1) & state.alive & ~state.below & ~state.asleep
    return can, cc[jnp.arange(state.pos.shape[0]), first]


def read_marks(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N, READ_FEATURES): the inscription one can read (one-hot symbols) and where it is."""
    can, at = mark_beside(state, cfg)
    words = state.marks[at[:, 0], at[:, 1]]
    feats = jnp.concatenate([jax.nn.one_hot(words, VOCAB).reshape(-1, MSG_LEN * VOCAB),
                             (at - state.pos).astype(jnp.float32)], axis=-1)
    return jnp.where(can[:, None], feats, 0.0)


def phrase_place(msgs, src, cfg: EnvConfig):
    """Read phrases as the grounded language means them. msgs (S, MSG_LEN) from sources at src (S, 2).
    Returns (S,) channel of the remembered map it speaks of (0 water, 1 food, -1 neither), (S,) the
    value it gives (1 there is, 0 there is none), and (S, 2) the place: the source + direction x
    distance (here 0, near half the view radius, far twice it; no distance said = near)."""
    g = cfg.grid_size
    is_thing = (msgs == W_WATER) | (msgs == W_FOOD)
    thing_at, has_thing = jnp.argmax(is_thing, axis=1), jnp.any(is_thing, axis=1)
    thing = msgs[jnp.arange(msgs.shape[0]), thing_at]
    is_dir = (msgs >= W_DIR0) & (msgs < W_DIR0 + 8)
    dir_at, has_dir = jnp.argmax(is_dir, axis=1), jnp.any(is_dir, axis=1)
    d = MOVE_DELTAS[1 + jnp.clip(msgs[jnp.arange(msgs.shape[0]), dir_at] - W_DIR0, 0, 7)]
    here = jnp.any(msgs == W_HERE, axis=1)
    near, far = max(2, cfg.view_radius // 2), 2 * cfg.view_radius
    dist = jnp.where(jnp.any(msgs == W_FAR, axis=1), far, near)
    place = jnp.clip(src + jnp.where((here & ~has_dir)[:, None], 0, d * dist[:, None]), 0, g - 1)
    channel = jnp.where(has_thing & (has_dir | here), jnp.where(thing == W_WATER, 0, 1), -1)
    value = jnp.where(jnp.any(msgs == W_NO, axis=1), 0.0, 1.0)
    return channel, value, place


def comprehend(mind_map, heard, channel, value, place, cfg: EnvConfig):
    """Each listener (row of heard (L, S)) takes in what each source it heard said: the place goes
    into its remembered map, as heard (recency cfg.heard_recency, not consolidated)."""
    L, S = heard.shape
    li = jnp.broadcast_to(jnp.arange(L)[:, None], (L, S))
    pr = jnp.broadcast_to(place[None, :, 0], (L, S))
    pc = jnp.broadcast_to(place[None, :, 1], (L, S))
    ch = jnp.broadcast_to(jnp.maximum(channel, 0)[None, :], (L, S))
    ok = heard & (channel >= 0)[None, :]
    old = mind_map[li, pr, pc]                                                 # (L, S, 4)
    new = old.at[..., 0].set(jnp.where(ch == 0, value[None, :], old[..., 0]))
    new = new.at[..., 1].set(jnp.where(ch == 1, value[None, :], old[..., 1]))
    new = new.at[..., 2].set(jnp.maximum(old[..., 2], cfg.heard_recency))
    return mind_map.at[li, pr, pc].set(jnp.where(ok[..., None], new, old))


def heard_messages(state: State, cfg: EnvConfig) -> jnp.ndarray:
    """(N, HEARD_FEATURES): for each listener, the nearest speakers within view
    (up to N_HEARD): their message (one-hot symbols), where they are, and the
    listener's memory of them. Empty slots are zeros."""
    n, r = state.pos.shape[0], cfg.view_radius
    spoke = jnp.any(state.last_msg > 0, axis=1) & state.alive
    lp = layered_pos(state.pos, state.below)                          # no voice carries between layers
    d = _cheb(lp[:, None, :], lp[None, :, :])
    hear = jnp.where(state.asleep, 1, r)                              # asleep: only the closest voices
    cand = spoke[None, :] & ~jnp.eye(n, dtype=bool) & (d <= hear[:, None])
    k = min(N_HEARD, n)
    score, who = jax.lax.top_k(jnp.where(cand, -d, -10 ** 6).astype(jnp.float32), k)  # (N, k)
    ok = (score > -10 ** 5)[..., None]
    msg = jax.nn.one_hot(state.last_msg[who], VOCAB).reshape(n, k, MSG_LEN * VOCAB)
    rel = (state.pos[who] - state.pos[:, None, :]) / r
    mem = jnp.log1p(state.memory[jnp.arange(n)[:, None], who])
    slots = jnp.where(ok, jnp.concatenate([msg, rel, mem], axis=-1), 0.0)
    slots = jnp.pad(slots, ((0, 0), (0, N_HEARD - k), (0, 0)))
    return slots.reshape(n, HEARD_FEATURES)


def sight_radius(state, cfg: EnvConfig):
    """(N,) how far each agent can see this tick, before darkness: 1 asleep, less when sleepy."""
    loss = jnp.round(cfg.sleepy_sight_loss * jnp.clip((state.sleepiness - 0.5) / 0.5, 0.0, 1.0)).astype(jnp.int32)
    fx = condition_effects(state.conditions)
    loss = jnp.maximum(loss + fx.get("sleepy_sight", 0), 0)
    return jnp.where(state.asleep, 1, jnp.clip(cfg.view_radius - loss + fx.get("sight", 0), 1, cfg.view_radius))


def obs_dim(cfg: EnvConfig) -> int:
    return cfg.view_size ** 2 * N_OBS_CHANNELS + N_BODY_FEATURES + map_features(cfg)


N_MAP_STORED = 4      # per cell: water seen, plants seen (share of max), recency (1 = just seen), consolidated
N_MAP_CHANNELS = 4    # read as: water x recency, plants x recency, recency, inside the world


def map_features(cfg: EnvConfig) -> int:
    return cfg.map_cells ** 2 * N_MAP_CHANNELS


def update_mind_map(mind_map, pos, alive, terrain, plants, sight, cfg: EnvConfig, consolidate=None):
    """Cells within sight are remembered as they are now; everything else fades.
    Only what an agent has seen itself is ever in its map. A new memory fades fast (half a day's
    worth in a day) until deep sleep consolidates it; consolidated memories last (10-day half-life).
    consolidate: (N,) bool, agents in deep sleep this tick."""
    g = cfg.grid_size
    rr, cc = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
    d = jnp.maximum(jnp.abs(rr[None] - pos[:, 0, None, None]), jnp.abs(cc[None] - pos[:, 1, None, None]))
    seen = (d <= sight[:, None, None]) & alive[:, None, None]                    # (N, G, G)
    if consolidate is None:
        consolidate = jnp.zeros(pos.shape[0], bool)
    kept = mind_map[..., 3] > 0.5
    rate = jnp.where(kept, 0.5 ** (1.0 / cfg.map_half_life), 0.5 ** (1.0 / cfg.map_fresh_half_life))
    faded = mind_map.at[..., 2].multiply(rate)
    faded = faded.at[..., 3].set(jnp.where(consolidate[:, None, None] & (faded[..., 2] > 0.05), 1.0, faded[..., 3]))
    now = jnp.stack([jnp.broadcast_to(drinkable(terrain).astype(jnp.float32), seen.shape),
                     jnp.broadcast_to(plants / cfg.plant_max, seen.shape),
                     jnp.ones(seen.shape), faded[..., 3]], axis=-1)              # (N, G, G, 4)
    return jnp.where(seen[..., None], now, faded)


def read_mind_map(mind_map, pos, cfg: EnvConfig):
    """(N, map_features): the map centred on the agent, covering twice the world's width so the
    whole world is in it from anywhere, pooled into map_cells x map_cells blocks."""
    g, p = cfg.grid_size, cfg.map_cells
    block = max(1, (2 * g) // p)
    span = p * block
    rec = mind_map[..., 2:3]
    cells = jnp.concatenate([mind_map[..., 0:1] * rec, mind_map[..., 1:2] * rec, rec,
                             jnp.ones_like(rec)], axis=-1)                         # (N, G, G, 4)
    half = span // 2
    padded = jnp.pad(cells, ((0, 0), (half, span - half), (half, span - half), (0, 0)))

    def crop(m, rc):
        return jax.lax.dynamic_slice(m, (rc[0], rc[1], 0), (span, span, N_MAP_CHANNELS))

    win = jax.vmap(crop)(padded, pos).reshape(-1, p, block, p, block, N_MAP_CHANNELS)
    pooled = jnp.concatenate([win[..., 0:1].max(axis=(2, 4)), win[..., 1:2].mean(axis=(2, 4)),
                              win[..., 2:3].max(axis=(2, 4)), win[..., 3:4].mean(axis=(2, 4))], axis=-1)
    return pooled.reshape(pos.shape[0], -1)


# ---------------------------------------------------------------- animals
def animal_asleep(key, an: Animals, sp: dict, night, cfg: EnvConfig):
    """(M,) bool: asleep this tick (cfg.animal_sleep). Each species rests in its own half of the day,
    day-active ones at night and nocturnal ones by day ('nocturnal' 0.5: either), and in that half
    is asleep a share cfg.animal_sleep of the time. Hibernation is separate (winter, in _move_animals)."""
    noct = sp["nocturnal"][an.species]
    resting = jnp.where(night, 1.0 - noct, noct)
    return an.alive & (jax.random.uniform(key, an.alive.shape) < cfg.animal_sleep * resting)


def _move_animals(key, an: Animals, agent_pos, agent_alive, terrain, approach, cfg: EnvConfig, night, season,
                  sp: dict, caves=None, plants=None, others=None, asleep=None):
    """Each animal by its species: flee from (approach=False) or chase (approach=True) the nearest
    agent it notices; otherwise wander, or drift toward its own kind if it herds. Fast species
    take two steps; nocturnal hunters are less active by day; hibernators sleep through winter.
    Each stays within its habitat (water for fish and crocodiles, land for the rest).

    caves: None, or dict(below (N,) agents' layer, passage (G, G), mouths (K, 2), mouth_ok (K,)).
    Then animals live on two layers: they notice agents on their own layer by sight, and on the
    other layer only by scent within cfg.scent_range (someone who just ducked into a cave). A
    chaser whose quarry is on the other layer heads for the nearest mouth and, on it, follows; a
    fleeing animal on a mouth may bolt down (cfg.prey_bolt_prob); a calm one on a mouth goes to
    the layer it keeps (denning species: bats below by day, bears below in winter; the rest
    above) with cfg.den_prob. Below, they walk only along passages, trying a random way when the
    way they want is rock.

    others: None, or dict(pos (P, 2), below (P,), ok (M, P) bool): animals that count as quarry
    (approach=True: prey a predator can take) or as threat (prey: predators that could take it),
    besides the agents (cfg.animal_hunting). Returns the nearest agent and its distance (for attacks).
    asleep: None, or (M,) bool (animal_asleep): a sleeper stays put and notices only what is beside it."""
    g = cfg.grid_size
    k_act, k_dir, k_wander, k_herd = jax.random.split(key, 4)
    m = an.pos.shape[0]
    spc = an.species
    notice, act, speed = sp["notice"][spc], sp["act"][spc], sp["speed"][spc]
    act = act * jnp.where(night, 1.0, 1.0 - 0.6 * sp["nocturnal"][spc])                # day: nocturnal ones idle
    awake = ~((sp["hibernates"][spc] > 0) & (season == 3))
    if asleep is not None:
        notice = jnp.where(asleep, jnp.minimum(notice, 1), notice)       # woken only by what is beside it
    habitat = sp["habitat"][spc]                                                       # (M, N_TERRAIN)
    d = _cheb(an.pos[:, None, :], agent_pos[None, :, :])
    if caves is not None:
        cross = an.below[:, None] != caves["below"][None, :]
        d = jnp.where(cross & (d > cfg.scent_range), 10 ** 6, d)      # the other layer: scent only
    d = jnp.where(agent_alive[None, :], d, 10 ** 6)
    nearest = jnp.argmin(d, axis=1)                      # the nearest agent (returned for attacks)
    dist = jnp.min(d, axis=1)
    # what it reacts to: agents, and other animals (quarry or threat) when given
    all_pos, d_all = agent_pos, d
    all_below = caves["below"] if caves is not None else None
    if others is not None:
        d_o = _cheb(an.pos[:, None, :], others["pos"][None, :, :])
        if caves is not None:
            cross_o = an.below[:, None] != others["below"][None, :]
            d_o = jnp.where(cross_o & (d_o > cfg.scent_range), 10 ** 6, d_o)
            all_below = jnp.concatenate([caves["below"], others["below"]])
        d_o = jnp.where(others["ok"], d_o, 10 ** 6)
        all_pos, d_all = jnp.concatenate([agent_pos, others["pos"]]), jnp.concatenate([d, d_o], axis=1)
    near_all = jnp.argmin(d_all, axis=1)
    if others is not None and approach:
        # A predator prefers its usual prey: a person counts as cfg.prey_preference cells farther
        # off when choosing whom to chase (attacks on people beside it are unchanged)
        n_ag = agent_pos.shape[0]
        score = d_all + jnp.where(jnp.arange(d_all.shape[1]) < n_ag, cfg.prey_preference + 0.01, 0.0)[None, :]
        near_all = jnp.argmin(score, axis=1)
    dist_all = jnp.take_along_axis(d_all, near_all[:, None], axis=1)[:, 0]
    hunting = (an.rest <= 0) & ((sp["provoked"][spc] == 0) | (an.anger > 0)) if approach else jnp.ones_like(an.alive)
    if approach and asleep is not None:
        hunting = hunting & ~asleep                                          # a sleeping predator does not hunt
    engaged = an.alive & awake & (dist_all <= notice) & hunting
    toward = jnp.sign(all_pos[near_all] - an.pos)
    if caves is not None:
        mouths, mouth_ok = caves["mouths"], caves["mouth_ok"]
        dm = jnp.where(mouth_ok[None, :], _cheb(an.pos[:, None, :], mouths[None]), 10 ** 6)   # (M, K)
        on_mouth = jnp.min(dm, axis=1) == 0
        other = all_below[near_all] != an.below
        toward = jnp.where((other & ~on_mouth)[:, None], jnp.sign(mouths[jnp.argmin(dm, axis=1)] - an.pos), toward)
    step_engaged = toward if approach else -toward
    # herding: when calm, drift toward the nearest animal of the same species (on its own layer)
    same = (spc[:, None] == spc[None, :]) & an.alive[None, :] & ~jnp.eye(spc.shape[0], dtype=bool) \
        & (an.below[:, None] == an.below[None, :])
    dh = jnp.where(same, _cheb(an.pos[:, None, :], an.pos[None, :, :]), 10 ** 6)
    mate = jnp.argmin(dh, axis=1)
    herd_step = jnp.where((jnp.min(dh, axis=1) > 1)[:, None] & (jnp.min(dh, axis=1) < 10 ** 6)[:, None],
                          jnp.sign(an.pos[mate] - an.pos), 0)
    rand_dir = MOVE_DELTAS[jax.random.randint(k_dir, (an.pos.shape[0],), 0, 9)]
    calm_dir = jnp.where((jax.random.uniform(k_herd, act.shape) < sp["herd"][spc])[:, None], herd_step, rand_dir)
    if plants is not None:                               # grazers: often toward the richest ground beside them
        cand = jnp.clip(an.pos[:, None, :] + MOVE_DELTAS[None], 0, g - 1)            # (M, 9, 2), staying first
        food = plants[cand[..., 0], cand[..., 1]] + 1e-3 * (jnp.arange(9) == 0)    # ties: stay
        food = jnp.where(habitat[jnp.arange(m)[:, None], terrain[cand[..., 0], cand[..., 1]]], food, -1.0)
        graze_step = MOVE_DELTAS[jnp.argmax(food, axis=1)]
        seek = sp["grazer"][spc] & ~an.below & (jax.random.uniform(jax.random.fold_in(k_herd, 9), act.shape) < cfg.graze_seek)
        calm_dir = jnp.where(seek[:, None], graze_step, calm_dir)
    acts = jax.random.uniform(k_act, (an.pos.shape[0],))
    move_engaged = engaged & (acts < act)
    move_wander = ~engaged & awake & (jax.random.uniform(k_wander, acts.shape) < cfg.wander_prob)
    if asleep is not None:
        move_wander = move_wander & ~asleep
    delta = jnp.where(move_engaged[:, None], step_engaged, jnp.where(move_wander[:, None], calm_dir, 0))
    pos, below = an.pos, an.below
    if caves is not None:
        # Changing layer at a mouth (instead of a step this tick)
        can_cave = habitat[:, CAVE] & an.alive & awake
        k_l = jax.random.split(jax.random.fold_in(key, 7), 2)
        follow = engaged & other & on_mouth & (acts < act) if approach else jnp.zeros_like(engaged)
        bolt = (engaged & ~other & on_mouth & (jax.random.uniform(k_l[0], (m,)) < cfg.prey_bolt_prob)) \
            if not approach else jnp.zeros_like(engaged)
        keeps_below = (sp["den"][spc] > 0) & jnp.where(sp["nocturnal"][spc] > 0, ~night, season == 3)
        settle = ~engaged & on_mouth & (keeps_below != below) & (jax.random.uniform(k_l[1], (m,)) < cfg.den_prob)
        switch = can_cave & (follow | bolt | settle)
        below = below ^ switch
        delta = jnp.where(switch[:, None], 0, delta)
        rand2 = NEIGHBORS8[jax.random.randint(jax.random.fold_in(k_dir, 3), (m,), 0, 8)]
    lame = an.lame > 0                                   # a lamed animal cannot run and manages half its steps
    hobble = jax.random.uniform(jax.random.fold_in(k_act, 13), (m,)) < 0.5
    for k in range(2):                                   # fast species run a second cell when engaged
        go = an.alive & ((k == 0) | (move_engaged & (speed >= 2))) & ~(lame & ((k == 1) | hobble))
        new = pos + delta
        ok = _in_bounds(new, g)
        newc = jnp.clip(new, 0, g - 1)
        ok_top = habitat[jnp.arange(pos.shape[0]), terrain[newc[:, 0], newc[:, 1]]]
        if caves is not None:
            ok = ok & jnp.where(below, caves["passage"][newc[:, 0], newc[:, 1]], ok_top) & go
            # below, rock in the way: try a random way along the passage instead
            alt = jnp.clip(pos + rand2, 0, g - 1)
            alt_ok = below & ~ok & go & (delta != 0).any(-1) & caves["passage"][alt[:, 0], alt[:, 1]]
            newc = jnp.where(alt_ok[:, None], alt, newc)
            ok = ok | alt_ok
        else:
            ok = ok & ok_top & go
        pos = jnp.where(ok[:, None], newc, pos)
    return an._replace(pos=pos, below=below, anger=jnp.maximum(an.anger - 1, 0),
                       lame=jnp.maximum(an.lame - 1, 0)), nearest, dist


def _respawn(key, an: Animals, terrain, prob, cfg: EnvConfig, hp_of, passage=None):
    """A dead animal's slot is filled again, now and then, by one of the same species in its habitat
    (bats in the caves)."""
    k_p, k_c = jax.random.split(key)
    born = ~an.alive & (jax.random.uniform(k_p, an.alive.shape) < prob)
    cells = _habitat_cells(k_c, terrain, an.species, cfg)
    roost = jnp.zeros_like(born)
    if passage is not None:
        deep, roost = _roost(jax.random.fold_in(k_c, 5), an.species, passage, cfg)
        cells = jnp.where(roost[:, None], deep, cells)
    return an._replace(pos=jnp.where(born[:, None], cells, an.pos), alive=an.alive | born,
                       below=jnp.where(born, roost, an.below),
                       hp=jnp.where(born, hp_of[an.species], an.hp), rest=jnp.where(born, 0, an.rest),
                       anger=jnp.where(born, 0, an.anger), lame=jnp.where(born, 0, an.lame))


def generate_bugs(key, terrain, cfg: EnvConfig) -> dict:
    """Bees' hives in woods (with honey) and termite mounds in grass and desert (with grubs)."""
    k1, k2 = jax.random.split(key)
    g = terrain.shape[0]
    hive = (terrain == FOREST) & (jax.random.uniform(k1, (g, g)) < cfg.hive_density)
    mound = ((terrain == GRASS) | (terrain == DESERT)) & (jax.random.uniform(k2, (g, g)) < cfg.mound_density)
    return {"hive": hive, "mound": mound, "honey": jnp.where(hive, cfg.honey_max, 0.0),
            "grubs": jnp.where(mound, cfg.grubs_max, 0.0)}


def mosquito_density(state, cfg: EnvConfig, season, night, heat):
    """(G, G) 0..1: marsh and water edges in the warm seasons, worse at night and after rain;
    smoke (any fire's heat) drives them off."""
    wet = (state.terrain == MARSH) | drinkable(state.terrain)
    warm = jnp.array([0.6, 1.0, 0.5, 0.0])[season]
    rain = 1.0 + 0.5 * (state.clouds > cfg.rain_threshold)
    return jnp.where(wet & (heat <= 0), jnp.minimum(warm * rain * jnp.where(night, 1.0, 0.5), 1.0), 0.0)


def fly_density(meat, bodies, heat):
    """(G, G) 0..1: flies on raw meat and the bodies of the dead; smoke drives them off."""
    return jnp.where(heat > 0, 0.0, jnp.minimum(meat / 2.0 + bodies, 1.0))


# ---------------------------------------------------------------- step
def visible_condition(health, injury, sick, weight, ideal_kg, scars, cfg: EnvConfig, wince=0.0):
    """How an agent looks to others: wounds, injuries (including lasting ones), sickness,
    thinness and scars. Hunger, thirst, fatigue and stress are internal and cannot be seen."""
    thin = jnp.clip((weight / ideal_kg - 0.7) / 0.3, 0.0, 1.0)      # 1 at a healthy weight or above
    return cfg.wellbeing_scale * (health + 0.5 * (1.0 - injury) + 0.5 * (1.0 - sick) + 0.5 * thin
                                  - 0.25 * scars - 0.5 * wince)            # pain shows: one winces, cries out


def wellbeing(health, satiety, hydration, energy, body_temp, stress, pain_weight, cfg: EnvConfig,
              sleepiness=0.0):
    """Felt well-being from the body alone: a fixed predictor of survival, not a reward for acts.
    Health is weighted by the pain-aversion trait; needs, warmth and calm add to it. Dead agents: 0."""
    warmth = jnp.clip(1.0 - jnp.maximum(-body_temp / -cfg.cold_threshold, body_temp / cfg.heat_threshold), 0.0, 1.0)
    felt = lambda x: cfg.need_weight * (1.0 - jnp.clip(1.0 - x, 0.0, 1.0) ** cfg.need_curve)
    needs = felt(satiety) + felt(hydration) + felt(energy) + felt(1.0 - sleepiness)
    return cfg.wellbeing_scale * (pain_weight * health + needs + 0.5 * warmth + 0.5 * (1.0 - stress))


def _survival_reward(state, cfg, geno, alive0, alive, dead_now, similarity,
                     health, satiety, hydration, energy, body_temp, stress, pain_aversion,
                     impression, new_impression, own_look, sleepiness):
    """Survival is the only reward; dispositions decide how it is felt.

    Each agent's survival signal: +alive_reward while alive, -death_penalty on dying, plus the
    change in felt well-being, gamma * W(after) - W(before). That change is potential-based
    (Ng, Harada & Russell, 1999): any loop of actions that ends where it started sums to about
    zero, so nothing can be farmed by repetition. Feelings about others come from changes in one's
    impression of them (updated only on sight), weighted by dispositions. Envy and guilt compare
    impressions with how one looks oneself and only ever cost reward."""
    g = cfg.shaping_gamma
    ones = jnp.ones_like(health)
    # Own experience: pain aversion weighs one's own health
    w0 = jnp.where(alive0, wellbeing(state.health, state.satiety, state.hydration, state.energy,
                                     state.body_temp, state.stress, state.pain_aversion, cfg,
                                     state.sleepiness), 0.0)
    w1 = jnp.where(alive, wellbeing(health, satiety, hydration, energy, body_temp, stress, pain_aversion, cfg,
                                    sleepiness), 0.0)
    own = jnp.where(alive0, jnp.where(alive, cfg.alive_reward, 0.0)
                    - jnp.where(dead_now, cfg.death_penalty, 0.0) + g * w1 - w0, 0.0)
    # Others are felt only through what one perceives of them: each agent keeps an impression of
    # how every other looked when last seen, and feels changes to that impression. Harm out of
    # sight is felt on the next sighting if it left visible marks; harm healed unseen is never
    # felt; a death is felt on seeing the body. Changes telescope, so no loop of acts pays.
    fed_me = jnp.clip(state.memory[:, :, MEM_GAVE_NEAR_ME], 0.0, 1.0)
    struck_me = jnp.clip(state.memory[:, :, MEM_STRUCK_ME], 0.0, 1.0)
    struck_by_me = struck_me.T                                          # [i, j]: j remembers i striking j
    care = (cfg.care_empathy * geno.empathy[:, None]
            + cfg.care_kin * geno.kin_care[:, None] * similarity
            + cfg.care_reciprocity * geno.reciprocity[:, None] * fed_me
            - cfg.care_revenge * geno.vengefulness[:, None] * struck_me
            - cfg.care_sadism * geno.sadism[:, None] * struck_by_me          # glad to see one's victims suffer
            + geno.selflessness[:, None]                                  # the rare selfless: anyone, as oneself
            + cfg.care_attachment * state.attachment)                      # those who keep one alive
    n = alive0.shape[0]
    others = ~jnp.eye(n, dtype=bool)
    caring = jnp.where(others, care * (new_impression - impression), 0.0).sum(1)
    # Envy and guilt: how others are believed to be doing against how one looks oneself
    known_alive = others & (new_impression > -0.5 * cfg.death_penalty)
    gap = new_impression - own_look[:, None]
    n_known = jnp.maximum(known_alive.sum(1), 1)
    inequity = (geno.envy * jnp.where(known_alive, jnp.maximum(gap, 0.0), 0.0).sum(1)
                + geno.guilt * jnp.where(known_alive, jnp.maximum(-gap, 0.0), 0.0).sum(1)) / n_known
    social = jnp.where(alive0, caring - cfg.care_inequity * inequity, 0.0)
    return own + social, social


def _give_birth(ns: State, birthing, key, cfg: EnvConfig):
    """Each mother giving birth places a newborn in a free slot (never used first, else a dead
    agent's slot) at her cell. The child's traits come from both parents (heritable with random
    variation), it looks like them, and its kinship to everyone was fixed at conception.
    Returns the new state and a (N,) mask of the slots that received a child."""
    n = ns.pos.shape[0]
    born = jnp.zeros(n, bool)

    def one(i, carry):
        s, born, key = carry
        key, k_sex, k_geno, k_look = jax.random.split(key, 4)
        score = jnp.where(~s.alive, jnp.where(~s.existed, 2, 1), 0)
        c = jnp.argmax(score)
        do = birthing[i] & (score[c] > 0)
        row = lambda x: jax.tree_util.tree_map(lambda v: v[i][None], x)
        child_geno = gt.inherit(k_geno, row(s.genotype), row(s.father_geno), s.genotype)
        child_geno, jumps = gt.mutate(jax.random.fold_in(k_geno, 23), child_geno, s.genotype,
                                      cfg.mutation_rate, cfg.mutation_size)
        child_geno = child_geno._replace(selflessness=draw_selfless(jax.random.fold_in(k_geno, 9), 1, cfg))
        put = lambda arr, val: arr.at[c].set(jnp.where(do, val, arr[c]))
        geno = jax.tree_util.tree_map(lambda arr, v: put(arr, v[0]), s.genotype, child_geno)
        height = cfg.base_height_cm * child_geno.height[0] * (0.55 + 0.45 * 0.3)    # newborn size
        ideal = cfg.ideal_bmi * (height / 100.0) ** 2 * child_geno.build[0]
        look = jnp.clip(0.5 * (s.appearance[i] + s.father_look[i])
                        + 0.05 * jax.random.normal(k_look, (N_APPEARANCE,)), 0.0, 1.0)
        rel = s.fetus_r[i].at[c].set(1.0)
        relatedness = jnp.where(do, s.relatedness.at[c, :].set(rel).at[:, c].set(rel), s.relatedness)
        # Other pregnancies: the coming child's kinship to this newborn, through both its parents
        dad = jnp.clip(s.father, 0)
        col = 0.5 * (relatedness[:, c] + relatedness[dad, c])
        fetus_r = jnp.where(do & (s.gestation > 0), s.fetus_r.at[:, c].set(col), s.fetus_r)
        fetus_r = put(fetus_r, jnp.zeros(n))
        clear = lambda m: jnp.where(do, m.at[c].set(0).at[:, c].set(0), m)    # forget the slot's last occupant
        new_look = 2.5 * cfg.wellbeing_scale                                  # a healthy-looking newborn
        s = s._replace(
            alive=put(s.alive, True), existed=put(s.existed, True),
            sex=put(s.sex, jax.random.bernoulli(k_sex).astype(jnp.int32)),
            pos=put(s.pos, s.pos[i]), below=put(s.below, s.below[i]), satiety=put(s.satiety, 0.8), hydration=put(s.hydration, 0.8),
            energy=put(s.energy, 1.0), health=put(s.health, 1.0), inv=put(s.inv, jnp.zeros(N_MAT)),
            hand=put(s.hand, PLANT), torch=put(s.torch, 0), age=put(s.age, 0), birth_age=put(s.birth_age, 0),
            body_temp=put(s.body_temp, 0.0), injury=put(s.injury, 0.0), infection=put(s.infection, 0),
            immunity=put(s.immunity, 0.0), stress=put(s.stress, 0.0), clothing=put(s.clothing, 0.0),   # born bare
            pain_aversion=put(s.pain_aversion, child_geno.pain_aversion[0]), genotype=geno,
            weight=put(s.weight, ideal), scars=put(s.scars, 0.0), lasting_injury=put(s.lasting_injury, 0.0), pain=put(s.pain, 0.0), heartache=put(s.heartache, 0.0),
            part_damage=put(s.part_damage, jnp.zeros(N_PARTS)),
            held_by=put(s.held_by, -1), hold_grip=put(s.hold_grip, 0.0), off_balance=put(s.off_balance, 0),
            appearance=put(s.appearance, look), last_msg=put(s.last_msg, jnp.zeros(MSG_LEN, jnp.int32)),
            mutations=put(s.mutations, jumps[0]),
            visited=put(s.visited, jnp.zeros(s.visited.shape[1:], bool).at[s.pos[i, 0], s.pos[i, 1]].set(True)),
            mind_map=put(s.mind_map, jnp.zeros(s.mind_map.shape[1:])),
            sleepiness=put(s.sleepiness, 0.0), asleep=put(s.asleep, False), sleep_depth=put(s.sleep_depth, 0),
            collapsed=put(s.collapsed, False),
            gestation=put(s.gestation, 0), courting=put(s.courting, 0), father=put(s.father, -1),
            # children of the slot's last occupant are orphans now: they must not take the newborn for their mother
            mother=put(jnp.where(do & (s.mother == c), -1, s.mother), i),
            relatedness=relatedness, fetus_r=fetus_r,
            memory=clear(s.memory), habituation=clear(s.habituation), attachment=clear(s.attachment),
            conditions=put(s.conditions, draw_inborn(jax.random.fold_in(k_geno, 17), 1, s.conditions[i][None],
                                                     s.father_cond[i][None])[0].astype(jnp.float32)),
            life_moves=put(s.life_moves, 0), life_strikes=put(s.life_strikes, 0), hardship=put(s.hardship, 0),
            impression=jnp.where(do, s.impression.at[:, c].set(new_look).at[c, :].set(s.impression[i]), s.impression),
            drop_owner=jnp.where(do & (s.drop_owner % n == c) & (s.drop_owner >= 0), -1, s.drop_owner),
        )
        return s, born | (do & (jnp.arange(n) == c)), key

    ns, born, _ = jax.lax.fori_loop(0, n, one, (ns, born, key))
    return ns, born



def senescence_hazard(years, lifespan_mult, cfg):
    """Deaths from senescence per biological year at this age: Gompertz, b exp(b (x - M)), with x
    the age measured against one's own lifespan (gene and conditions) and M = max_bio_years the most
    likely age of death. Never a fixed limit: anyone may die at any age, ever more likely with age."""
    b = cfg.senescence_rate
    x = years / lifespan_mult
    return b * jnp.exp(jnp.minimum(b * (x - cfg.max_bio_years), 20.0))

def step(state: State, actions: jnp.ndarray, cfg: EnvConfig, messages: jnp.ndarray | None = None):
    """Resolve one tick for all agents simultaneously.
    Returns (next_state, reward (N,), done (scalar), info: StepInfo)."""
    g, n = cfg.grid_size, cfg.n_agents
    keys = jax.random.split(state.key, 17)
    key = keys[0]
    alive0 = state.alive
    geno = state.genotype
    idx = jnp.arange(n)
    season = season_of(state.t, cfg)
    night = is_night(state.t, cfg)
    # How each strike is delivered (last of the packed outputs): the part aimed at, the limb, the
    # mode (a hand strike at the torso when none is given; hand strikes only without cfg.techniques)
    how = jnp.clip(messages[:, OUT_TOKENS].astype(jnp.int32), 0, N_HOW - 1) \
        if messages is not None and messages.shape[1] >= OUT_SLOTS else jnp.full(n, PART_TORSO, jnp.int32)
    aim, limb, mode = how_parts(how)
    if not cfg.techniques:
        how, limb, mode = aim, jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32)

    # ---------------------------------------------------------- weather
    timer = state.weather_timer - 1
    change = timer <= 0
    clim = climate_offset(state.t, state.climate, cfg)
    warm = jnp.maximum(clim, 0.0) / max(cfg.climate_amp, 1e-6) if cfg.climate_cycle_years > 0 else 0.0
    extreme = jax.random.uniform(keys[1]) < cfg.climate_volatility * (1 + 0.5 * warm)   # warm phases: more extremes
    # Which extremes are possible depends on the season
    extreme_w = jnp.array([[0.0, 0.0, 0.5, 0.5, 0.0],   # spring: drought, cold
                           [0.0, 0.0, 0.5, 0.0, 0.5],   # summer: drought, heat
                           [0.0, 0.0, 0.5, 0.5, 0.0],   # autumn
                           [0.0, 0.0, 0.0, 1.0, 0.0]])[season]
    mild_w = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]) if cfg.dynamic_weather else jnp.array([0.6, 0.4, 0.0, 0.0, 0.0])
    probs = jnp.where(extreme, extreme_w, mild_w)
    new_weather = jax.random.choice(keys[2], N_WEATHER, p=probs)
    weather = jnp.where(change, new_weather, state.weather).astype(jnp.int32)
    dur = jax.random.randint(keys[3], (), cfg.spell_min, cfg.spell_max)
    timer = jnp.where(change, dur, timer).astype(jnp.int32)
    weather_changed = change & (weather != state.weather)
    if cfg.dynamic_weather:
        clouds, snow, wind, raining, storms = update_weather(state, weather, season, jax.random.fold_in(keys[2], 51), cfg)
    else:
        clouds, snow, wind = state.clouds, state.snow, state.wind
        raining = jnp.broadcast_to(weather == RAIN, (g, g))
        storms = jnp.zeros((g, g), bool)
    climate_drift = state.climate
    if cfg.climate_cycle_years > 0:
        climate_drift = jnp.clip(state.climate + cfg.climate_drift / jnp.sqrt(cfg.day_length)
                                 * jax.random.normal(jax.random.fold_in(keys[2], 52)),
                                 -cfg.climate_amp, cfg.climate_amp)

    # ---------------------------------------------------------- decode actions
    mask = action_mask(state, cfg)
    valid = jnp.take_along_axis(mask, actions[:, None], axis=1)[:, 0]
    a = jnp.where(valid & alive0, actions, jnp.where(state.asleep, SLEEP, REST))   # asleep: sleep on
    chosen = a                                           # as chosen (a skill stays a skill in the record)
    if cfg.skills:
        a = resolve_skills(state, a, mask, cfg)
        a = jnp.where(jnp.take_along_axis(mask, a[:, None], axis=1)[:, 0], a, REST)
    # Exhaustion: past sleep_forced one falls asleep whatever one chose
    collapse = alive0 & (state.sleepiness >= cfg.sleep_forced)
    a = jnp.where(collapse, SLEEP, a)
    # Staggered or thrown to the ground (cfg.techniques): no step and no blow this tick
    staggered = alive0 & (state.off_balance > 0)
    a = jnp.where(staggered & (((a >= MOVE_N) & (a <= MOVE_NW)) | ((a >= STRIKE_N) & (a < STRIKE_N + 8))), REST, a)
    sleeping = a == SLEEP
    is_move = (a >= MOVE_N) & (a <= MOVE_NW)
    is_strike = (a >= STRIKE_N) & (a < STRIKE_N + 8)
    resting = (a == REST) | sleeping                     # sleep is also rest for the muscles and warmth
    is_select = (a >= SELECT_0) & (a < SELECT_0 + N_MAT)
    is_place = (a >= PLACE_N) & (a < PLACE_N + 8)
    is_mate = a == MATE
    # Layers: below0 during this tick's acts, below_after once delving is done (an infant on its
    # mother's cell goes with her)
    below0 = state.below
    delving = (a == DELVE) & alive0
    if cfg.reproduction:
        care0 = caregiver_of(state, cfg)                 # mother, or a wet nurse for an orphan
        mom0 = jnp.clip(care0, 0)
        infant0 = (state.age + state.birth_age) / cfg.ticks_per_bio_year < cfg.infant_years
        held_infant = alive0 & infant0 & (care0 >= 0) & (below0 == below0[mom0]) \
            & (jnp.max(jnp.abs(state.pos - state.pos[mom0]), axis=-1) <= (1 if cfg.carry_infants else 0))
        delving = delving | (held_infant & delving[mom0])
    else:
        held_infant, mom0 = jnp.zeros(n, bool), jnp.zeros(n, jnp.int32)
    below_after = below0 ^ delving
    lp0 = layered_pos(state.pos, below0)
    light0 = jnp.where(state.torch > 0, cfg.torch_light, 1)                  # how far one sees below
    hand0 = state.hand                                   # this tick acts with the current hand
    held0 = state.inv[idx, hand0]
    hand = jnp.where(is_select, a - SELECT_0, state.hand).astype(jnp.int32)
    k_fire = jax.random.split(keys[16], 6)               # fire randomness (own key)
    SP = species_params(cfg)                             # every species' body and habits
    heat0 = fire_heat(state.fire, cfg)

    # Capabilities from the body as it is now: age, genes, condition
    bod = body(state, cfg)
    strength_now, speed_now = bod["strength"], bod["speed"]

    # ---------------------------------------------------------- movement
    fx = condition_effects(state.conditions)                       # this tick's conditions
    move_fail = jax.random.uniform(keys[4], (n,)) < state.injury * cfg.injury_move_fail * fx.get("stumble", 1.0)
    step_dir = jnp.where(is_move, a, 0)
    # Held (cfg.techniques): a hold from last tick pins one with its grip, if the holder is still
    # beside one, on one's layer, up and on its feet
    holder = jnp.clip(state.held_by, 0)
    hold_on = (state.held_by >= 0) & alive0 & alive0[holder] & ~downed_of(state, cfg)[holder] \
        & (state.off_balance[holder] == 0) & (below0 == below0[holder]) \
        & (jnp.max(jnp.abs(state.pos - state.pos[holder]), axis=-1) <= 1)
    grip_now = jnp.where(hold_on, state.hold_grip, 0.0)
    pinned = hold_on & (jax.random.uniform(jax.random.fold_in(keys[4], 23), (n,)) < state.hold_grip)
    move_fail = move_fail | (is_move & pinned)
    # Light below: burning branches light the cells within torch_light of whoever carries one
    if cfg.underground:
        bearers = alive0 & below0 & (state.torch > 0)
        rr_, cc_ = jnp.meshgrid(jnp.arange(g), jnp.arange(g), indexing="ij")
        d_t = jnp.maximum(jnp.abs(rr_[None] - state.pos[:, 0, None, None]), jnp.abs(cc_[None] - state.pos[:, 1, None, None]))
        torchlit = jnp.any((d_t <= cfg.torch_light) & bearers[:, None, None], axis=0)          # (G, G)
        in_dark = alive0 & below0 & ~torchlit[state.pos[:, 0], state.pos[:, 1]]
        # In the dark one gropes: a step often goes a random way (into rock: no step) or fails
        k_dk = jax.random.split(jax.random.fold_in(keys[4], 9), 3)
        veer = is_move & in_dark & (jax.random.uniform(k_dk[0], (n,)) < cfg.dark_veer)
        step_dir = jnp.where(veer, jax.random.randint(k_dk[1], (n,), MOVE_N, MOVE_NW + 1), step_dir)
        groped = jnp.clip(state.pos + MOVE_DELTAS[step_dir], 0, g - 1)
        into_rock = veer & ~(state.under & ~state.under_water)[groped[:, 0], groped[:, 1]] \
            | veer & ~_in_bounds(state.pos + MOVE_DELTAS[step_dir], g)
        move_fail = move_fail | into_rock | (is_move & in_dark & (jax.random.uniform(k_dk[2], (n,)) < cfg.dark_stumble))
    else:
        torchlit = jnp.zeros((g, g), bool)
        in_dark = jnp.zeros(n, bool)
    moving = is_move & ~move_fail
    new_pos = jnp.where(moving[:, None], state.pos + MOVE_DELTAS[step_dir], state.pos)
    terr_new = jnp.where(below0, CAVE, state.terrain[new_pos[:, 0], new_pos[:, 1]])   # below: cave floor
    rise = jnp.maximum(state.height[new_pos[:, 0], new_pos[:, 1]] - state.height[state.pos[:, 0], state.pos[:, 1]], 0.0)
    rise = jnp.where(below0, 0.0, rise)
    climb = 1.0 + cfg.climb_cost * rise / steepest_drop(state.height)
    move_energy = jnp.where(moving, cfg.move_cost * TERRAIN_MOVE_COST[terr_new] * climb
                            * (1 + cfg.snow_move_mult * state.snow[new_pos[:, 0], new_pos[:, 1]] * ~below0)
                            * (1 + state.injury) * (state.weight / bod["ideal_kg"]) / speed_now, 0.0)
    if cfg.reproduction and cfg.carry_infants:
        # An infant within reach is carried (cfg.carry_infants): it goes where its carrier goes,
        # and the carrier's steps cost more by the infant's share of her weight
        new_pos = jnp.where(held_infant[:, None], new_pos[mom0], new_pos)
        load = jnp.zeros(n).at[mom0].add(jnp.where(held_infant, state.weight, 0.0)) / jnp.maximum(state.weight, 1.0)
        move_energy = move_energy * (1 + cfg.carry_cost * load)
    # Falls: rock is rough ground (0.2), and any steep cell is dangerous in proportion to its drop
    cliff_new = state.cliff[new_pos[:, 0], new_pos[:, 1]]
    hazard = jnp.where(below0, 0.0, jnp.where(terr_new == ROCK, 0.2, 0.0) + cliff_new)
    p_fall = cfg.fall_prob * hazard * (1 + state.injury) * jnp.where(is_night(state.t, cfg), 2.0, 1.0) \
        * (1 + cfg.sleepy_fall_mult * state.sleepiness)
    fell = moving & (jax.random.uniform(keys[5], (n,)) < p_fall)
    severity = hazard * cfg.fall_severity * jax.random.exponential(jax.random.fold_in(keys[5], 1), (n,))
    fall_damage = jnp.where(fell, severity, 0.0)                      # can exceed 1: instant death
    injury = state.injury + cfg.fall_injury_ratio * fall_damage

    # Animals asleep this tick (cfg.animal_sleep): fixed for the tick, so a blow, a move and an attack agree
    if cfg.animal_sleep > 0:
        asleep_s = animal_asleep(jax.random.fold_in(keys[6], 77), state.small, SP, night, cfg)
        asleep_l = animal_asleep(jax.random.fold_in(keys[7], 77), state.large, SP, night, cfg)
        asleep_p = animal_asleep(jax.random.fold_in(keys[8], 77), state.predators, SP, night, cfg)
    else:
        asleep_s = asleep_l = asleep_p = None

    # ---------------------------------------------------------- strikes
    # Grids here are two layers stacked, (2G, G): row r + G for a cell below, so a blow, a dodge,
    # a kill or a counterattack only ever meets what is on the striker's own layer.
    sdir = jnp.where(is_strike, a - STRIKE_N + 1, 0)
    tgt = state.pos + MOVE_DELTAS[sdir]
    tgt_ok = is_strike & _in_bounds(tgt, g)
    tgt = jnp.clip(tgt, 0, g - 1)
    lrow = lambda p, b: p[:, 0] + g * b.astype(jnp.int32)
    occ2 = lambda p, b, w: jnp.zeros((2 * g, g)).at[lrow(p, b), p[:, 1]].add(w.astype(jnp.float32))
    occ_pred = occ2(state.predators.pos, state.predators.below, state.predators.alive)
    occ_large = occ2(state.large.pos, state.large.below, state.large.alive)
    occ_small = occ2(state.small.pos, state.small.below, state.small.alive)
    occ_agent = occ2(state.pos, below0, alive0)
    tr, tc = tgt[:, 0], tgt[:, 1]
    trl = tr + g * below0.astype(jnp.int32)                              # the target cell, on one's layer
    hit_pred = tgt_ok & (occ_pred[trl, tc] > 0)
    hit_large = tgt_ok & ~hit_pred & (occ_large[trl, tc] > 0)
    hit_small = tgt_ok & ~hit_pred & ~hit_large & (occ_small[trl, tc] > 0)
    hit_agent = tgt_ok & ~hit_pred & ~hit_large & ~hit_small & (occ_agent[trl, tc] > 0)
    holding = held0 > 0.05
    tool = HARDNESS[hand0] * holding                     # a hard object in hand
    edge = SHARPNESS[hand0] * holding                    # ... with a sharp edge
    # Some animals dodge: birds take off, fish dart; a point (an edge in hand) cuts the misses
    def dodge_grid(an, asleep):
        ev = SP["evade"][an.species] * (1.0 if asleep is None else jnp.where(asleep, 0.2, 1.0))   # a sleeper barely dodges
        return jnp.zeros((2 * g, g)).at[lrow(an.pos, an.below), an.pos[:, 1]].max(jnp.where(an.alive, ev, 0.0))
    k_miss = jax.random.split(k_fire[5], 3)
    for grp, (an, kk, zz) in enumerate([(state.predators, k_miss[0], asleep_p), (state.large, k_miss[1], asleep_l),
                                        (state.small, k_miss[2], asleep_s)]):
        miss = jax.random.uniform(kk, (n,)) < dodge_grid(an, zz)[trl, tc] * (1.0 - 0.6 * edge)
        if grp == 0:
            hit_pred = hit_pred & ~miss
        elif grp == 1:
            hit_large = hit_large & ~miss
        else:
            hit_small = hit_small & ~miss
    dmg = cfg.strike_damage * strength_now * (1 - 0.5 * state.injury) * fx.get("blow", 1.0) \
        * (1 + cfg.tool_bonus * tool + cfg.edge_bonus * edge)
    ref_kg = cfg.ideal_bmi * (cfg.base_height_cm / 100.0) ** 2
    mass_force = (bod["ideal_kg"] / ref_kg) ** (2.0 / 3.0) if cfg.blow_by_mass else jnp.ones(n)
    if cfg.blow_by_mass:
        # Force follows muscle cross-section (lean mass^(2/3)), technique grows with practice, and no
        # two blows are alike: some glance off
        practice = 1.0 + cfg.practice_max * (1.0 - 0.5 ** (state.life_strikes / cfg.practice_half))
        k_luck = jax.random.split(jax.random.fold_in(k_fire[5], 17))
        luck = jnp.exp(cfg.blow_luck * jax.random.normal(k_luck[0], (n,)) - 0.5 * cfg.blow_luck ** 2) \
            * jnp.where(jax.random.uniform(k_luck[1], (n,)) < cfg.glancing_prob, 0.5, 1.0)
        dmg = dmg * mass_force * practice * luck
    if cfg.techniques:
        # The limb: a kick drives the legs' larger muscles (about 1.5x a punch's force; less accurate,
        # worse at a head or neck, and it can cost one's balance), a headbutt the neck and trunk
        # behind the skull (it jars one's own head too), a shoulder or hip the whole body's mass
        # (mass^1, not mass^(2/3); easy to land, weak per blow). Kicks need sound legs, not arms.
        # The mode: a strike deals the blow; a push or a hold spends the effort moving or holding.
        limb_f = jnp.asarray(cfg.limb_force)[limb]
        limb_f = limb_f * jnp.where(limb == LIMB_FOOT, bod["legs"] / jnp.maximum(bod["arms"], 1e-3), 1.0) \
            * jnp.where((limb == LIMB_BODY) & cfg.blow_by_mass, (bod["ideal_kg"] / ref_kg) ** (1.0 / 3.0), 1.0)
        dmg = dmg * limb_f * jnp.asarray(cfg.mode_damage)[mode]
    # Aim (cfg.body_parts): the part a blow is aimed at decides whether it lands and how much of it
    # tells. A head or neck is hard to hit and deadly; a torso easy; a limb easy and less harmful,
    # but on an animal a blow to a leg lames it. Someone downed is hit wherever one aims.
    if cfg.body_parts:
        p_land = jnp.asarray(cfg.part_hit)[aim]
        if cfg.techniques:
            high = (aim == PART_HEAD) | (aim == PART_NECK)
            p_land = jnp.minimum(p_land * jnp.asarray(cfg.limb_accuracy)[limb]
                                 * jnp.where((limb == LIMB_FOOT) & high, cfg.kick_high_accuracy, 1.0), 1.0)
        lethal = jnp.asarray(cfg.part_lethality)[aim]
        roll = jax.random.uniform(jax.random.fold_in(k_fire[5], 11), (n,))
        lands_animal = roll < p_land
        # The downed and the staggered cannot ward off; one held is open in proportion to the grip
        exposed = jnp.where((downed_of(state, cfg) | (state.off_balance > 0)) & alive0, 1.0, grip_now)
        exposure = jnp.zeros((2 * g, g)).at[lrow(state.pos, below0), state.pos[:, 1]].max(jnp.where(alive0, exposed, 0.0))
        lands_agent = roll < p_land + (1.0 - p_land) * exposure[trl, tc]
        hit_pred, hit_large, hit_small = hit_pred & lands_animal, hit_large & lands_animal, hit_small & lands_animal
        lame_blow = (aim == PART_ARMS) | (aim == PART_LEGS)
    else:
        lethal, lands_agent, lame_blow = jnp.ones(n), jnp.ones(n, bool), jnp.zeros(n, bool)
    # Sparks: something hard struck against something hard (rock ground, a loose stone, a mineral);
    # far likelier when one of the two throws sparks (pyrite, iron) and the other is hard (flint)
    on_target = state.minerals[tr, tc] > 0.1                                       # (N, M)
    surf_hard = jnp.maximum(jnp.maximum(jnp.where(state.terrain[tr, tc] == ROCK, 1.0, 0.0),
                                        jnp.where(state.stones[tr, tc] > 0.1, 1.0, 0.0)),
                            jnp.max(jnp.where(on_target, jnp.array(mn.HARDNESS), 0.0), axis=-1))
    surf_spark = jnp.max(jnp.where(on_target, jnp.array(mn.SPARK), 0.0), axis=-1)
    pair = tool * surf_hard + cfg.spark_kit_mult * jnp.maximum(SPARK[hand0] * holding * surf_hard, surf_spark * tool)
    spark = tgt_ok & ~below0 & (jax.random.uniform(k_fire[0], (n,)) < jnp.minimum(cfg.spark_prob * pair, 1.0))   # no fire below

    def dmg_grid(hit, scale=1.0):
        return jnp.zeros((2 * g, g)).at[trl, tc].add(jnp.where(hit, dmg * scale, 0.0))

    def lame_grid(hit):
        return jnp.zeros((2 * g, g), jnp.int32).at[trl, tc].max(
            jnp.where(hit & lame_blow, _ticks(cfg, cfg.animal_lame_ticks), 0))

    def wound(an: Animals, grid, lamed=None):
        # Damage reduction, not a hard threshold: armor (by species) blunts weak blows most, so
        # a lone hunter wounds slowly and several striking together wound much faster.
        total = grid[lrow(an.pos, an.below), an.pos[:, 1]]
        armor = SP["armor"][an.species]
        taken = total * total / (total + armor + 1e-8)
        hp = an.hp - jnp.where(an.alive, taken, 0.0)
        killed = an.alive & (hp <= 0)
        struck = an.alive & (total > 0)
        lame = an.lame if lamed is None else jnp.maximum(an.lame, jnp.where(an.alive, lamed[lrow(an.pos, an.below), an.pos[:, 1]], 0))
        return an._replace(hp=hp, alive=an.alive & ~killed, anger=jnp.where(struck, _ticks(cfg, 60), an.anger),
                           lame=lame), killed, taken

    predators, killed_pred, pred_taken = wound(state.predators, dmg_grid(hit_pred, lethal), lame_grid(hit_pred))
    large, killed_large, _ = wound(state.large, dmg_grid(hit_large, lethal), lame_grid(hit_large))
    small, killed_small, _ = wound(state.small, dmg_grid(hit_small, lethal), lame_grid(hit_small))
    meat, cave_floor = state.meat, state.cave_floor

    def carcass(meat, floor, an, amount):
        """A kill's meat on the ground where it fell: above, or on the cave floor."""
        meat = meat.at[an.pos[:, 0], an.pos[:, 1]].add(jnp.where(~an.below, amount, 0.0))
        floor = floor.at[an.pos[:, 0], an.pos[:, 1], RAW_MEAT].add(jnp.where(an.below, amount, 0.0))
        return meat, floor
    meat, cave_floor = carcass(meat, cave_floor, large, jnp.where(killed_large, SP["meat"][large.species], 0.0))
    meat, cave_floor = carcass(meat, cave_floor, small, jnp.where(killed_small, SP["meat"][small.species], 0.0))
    meat, cave_floor = carcass(meat, cave_floor, predators, jnp.where(killed_pred, SP["meat"][predators.species], 0.0))
    mins = state.minerals

    def leftovers(mins, floor, an, killed, kinds=("hide", "bone", "feather")):
        """What a carcass leaves besides meat: hide, bone, feathers by species (cfg.byproducts)."""
        if not cfg.byproducts:
            return mins, floor
        for kind, mat in (("hide", HIDE), ("bone", BONE), ("feather", FEATHER)):
            if kind not in kinds:
                continue
            amt = jnp.where(killed, SP[kind][an.species], 0.0)
            mins = mins.at[an.pos[:, 0], an.pos[:, 1], mat - MIN0].add(jnp.where(~an.below, amt, 0.0))
            floor = floor.at[an.pos[:, 0], an.pos[:, 1], mat].add(jnp.where(an.below, amt, 0.0))
        return mins, floor
    mins, cave_floor = leftovers(mins, cave_floor, large, killed_large)
    mins, cave_floor = leftovers(mins, cave_floor, small, killed_small)
    mins, cave_floor = leftovers(mins, cave_floor, predators, killed_pred)
    # Struck, some fight back: aurochs and mammoths trample, boars gore, bears maul (the striker)
    def counter(an, hit, kk):
        rows = lrow(an.pos, an.below)
        on = jnp.zeros((2 * g, g)).at[rows, an.pos[:, 1]].max(jnp.where(an.alive, SP["counter_p"][an.species], 0.0))
        hurt = jnp.zeros((2 * g, g)).at[rows, an.pos[:, 1]].max(jnp.where(an.alive, SP["counter_dmg"][an.species], 0.0))
        who = jnp.zeros((2 * g, g), jnp.int32).at[rows, an.pos[:, 1]].max(
            jnp.where(an.alive & (SP["counter_p"][an.species] > 0), an.species, -1))
        keep_off = 1.0 - cfg.spear_reach * ((hand0 == SPEAR) & holding)          # a spear's length keeps it off
        back = hit & (jax.random.uniform(kk, (n,)) < on[trl, tc] * keep_off)
        return jnp.where(back, hurt[trl, tc], 0.0), jnp.where(back, who[trl, tc], -1)
    k_ctr = jax.random.split(jax.random.fold_in(k_fire[5], 3), 2)
    gore_l, gore_l_sp = counter(large, hit_large, k_ctr[0])
    gore_p, gore_p_sp = counter(predators, hit_pred, k_ctr[1])
    gored_dmg = gore_l + gore_p
    gored_by = jnp.where(gore_l > 0, gore_l_sp, gore_p_sp)
    injury = injury + 0.5 * gored_dmg

    def kill_credit(killed, an, hit):
        """Every striker who hit a cell where an animal died gets credit."""
        kgrid = jnp.zeros((2 * g, g)).at[lrow(an.pos, an.below), an.pos[:, 1]].add(killed.astype(jnp.float32))
        return hit & (kgrid[trl, tc] > 0)

    kills = jnp.stack([kill_credit(killed_small, small, hit_small),
                       kill_credit(killed_large, large, hit_large),
                       kill_credit(killed_pred, predators, hit_pred)], axis=-1)

    def kill_species(killed, an):
        grid = jnp.full((2 * g, g), -1, jnp.int32).at[lrow(an.pos, an.below), an.pos[:, 1]].max(
            jnp.where(killed, an.species, -1))
        return grid[trl, tc]
    kill_sp = jnp.stack([kill_species(killed_small, small), kill_species(killed_large, large),
                         kill_species(killed_pred, predators)], axis=-1)                  # (N, 3) species or -1
    # The downed cannot ward off, turn or roll away: every blow lands square (cfg.downed_hit_mult)
    down0 = downed_of(state, cfg)
    if cfg.body_parts:
        # Blow by blow: who struck whom (on one's own layer), whether it landed where it was aimed,
        # how much of it told (the part's lethality), and the damage to that part (its fragility)
        on_cell = (trl[:, None] == lrow(state.pos, below0)[None, :]) & (tc[:, None] == state.pos[None, :, 1])
        attempt = hit_agent[:, None] & on_cell & alive0[None, :] & ~jnp.eye(n, dtype=bool)        # (striker, victim)
        landed = attempt & lands_agent[:, None]
        blow = jnp.where(landed, (dmg * cfg.agent_strike_scale)[:, None], 0.0) \
            * jnp.where(down0, cfg.downed_hit_mult, 1.0)[None, :]
        violence = (blow * lethal[:, None]).sum(0)
        part_by_blow = blow.T @ (jax.nn.one_hot(aim, N_PARTS) * jnp.asarray(cfg.part_fragility)[None, :])   # (victim, part)
        struck_by_agent = jnp.any(attempt, axis=0)           # struck at, landed or not: one knows one was attacked
        blow_landed = jnp.any(landed, axis=1)
        if cfg.techniques:
            part_by_blow, violence, new_pos, fall_damage, injury, tech = _techniques(
                state, cfg, bod, strength_now, mass_force, limb, mode, sdir, landed, blow, alive0, below0, down0,
                is_strike, new_pos, part_by_blow, violence, fall_damage, injury, jax.random.fold_in(k_fire[5], 29))
    else:
        agent_dmg_grid = dmg_grid(hit_agent) * cfg.agent_strike_scale        # a blow lands on one's own layer
        violence = jnp.where(alive0, agent_dmg_grid[lrow(state.pos, below0), state.pos[:, 1]], 0.0)
        struck_by_agent = violence > 0
        violence = violence * jnp.where(down0, cfg.downed_hit_mult, 1.0)
        # without parts, a blow on the downed may land on something vital and end it (cfg.coup_prob)
        coup = down0 & struck_by_agent & (jax.random.uniform(jax.random.fold_in(k_fire[5], 7), (n,)) < cfg.coup_prob)
        violence = violence + jnp.where(coup, state.health, 0.0)
        part_by_blow = jnp.zeros((n, N_PARTS))
        blow_landed = hit_agent
    if not (cfg.body_parts and cfg.techniques):
        tech = dict(held_by=jnp.full(n, -1, jnp.int32), hold_grip=jnp.zeros(n),
                    off_balance=jnp.maximum(state.off_balance - 1, 0).astype(jnp.int32),
                    shoved=jnp.zeros(n, bool), thrown=jnp.zeros(n, bool), grabbed=jnp.zeros(n, bool))
    injury = injury + 0.5 * violence

    # ---------------------------------------------------------- animals act
    pos_after = jnp.where(alive0[:, None], new_pos, state.pos)
    lpa = layered_pos(pos_after, below_after)
    if cfg.underground:
        mouths, mouth_ok = cave_mouths(state, cfg)
        caves = dict(below=below_after, passage=state.under & ~state.under_water, mouths=mouths, mouth_ok=mouth_ok)
    else:
        caves = None
    graze_on = state.plants if cfg.grazing else None
    if cfg.animal_hunting:
        # Animals among animals: prey flee predators that could take them (cfg.animal_hunting), and a
        # hunting predator chases the nearest quarry it notices, a person or small prey (large prey
        # too with cfg.stalk_large: a few large-prey slots cannot feed six stalking predators)
        p_sp = predators.species
        takes_big = SP["takes_large"][p_sp] > 0                                          # (P,)
        big_ok = SP["size"][large.species] < 0.95                                         # not a mammoth
        flee_small = dict(pos=predators.pos, below=predators.below,
                          ok=jnp.broadcast_to(predators.alive[None, :], (small.pos.shape[0], p_sp.shape[0])))
        flee_large = dict(pos=predators.pos, below=predators.below,
                          ok=big_ok[:, None] & (predators.alive & takes_big)[None, :])
        small, _, _ = _move_animals(keys[6], small, pos_after, alive0, state.terrain, False, cfg, night, season, SP,
                                    caves, graze_on, flee_small, asleep_s)
        large, _, _ = _move_animals(keys[7], large, pos_after, alive0, state.terrain, False, cfg, night, season, SP,
                                    caves, graze_on, flee_large, asleep_l)
        quarry = dict(pos=jnp.concatenate([small.pos, large.pos]), below=jnp.concatenate([small.below, large.below]),
                      ok=jnp.concatenate([jnp.broadcast_to(small.alive[None, :], (p_sp.shape[0], small.pos.shape[0])),
                                          takes_big[:, None] & (large.alive & big_ok)[None, :] & cfg.stalk_large], axis=1))
        predators, target, pdist = _move_animals(keys[8], predators, pos_after, alive0, state.terrain, True, cfg,
                                                 night, season, SP, caves, None, quarry, asleep_p)
    else:
        small, _, _ = _move_animals(keys[6], small, pos_after, alive0, state.terrain, False, cfg, night, season, SP, caves,
                                    graze_on, None, asleep_s)
        large, _, _ = _move_animals(keys[7], large, pos_after, alive0, state.terrain, False, cfg, night, season, SP, caves,
                                    graze_on, None, asleep_l)
        predators, target, pdist = _move_animals(keys[8], predators, pos_after, alive0, state.terrain, True, cfg,
                                                 night, season, SP, caves, None, None, asleep_p)
    lp_pred = layered_pos(predators.pos, predators.below)
    # Predator attacks, by species: an adjacent hunting predator attacks its target unless the target
    # stands with others; each extra pack member beside the target cancels one defender. Ambushers
    # (big cats, crocodiles) strike at once; bears only when provoked or startled close by; snakes
    # bite with venom; nocturnal hunters are less keen by day; nothing attacks into fire's heat.
    # A predator reaches only someone on its own layer.
    psp = predators.species
    pdist = _cheb(lp_pred, lpa[target])
    group = jnp.sum((_cheb(lpa[target][:, None, :], lpa[None, :, :]) <= 1)
                    & alive0[None, :], axis=1)
    pack_here = (predators.alive[None, :] & (psp[None, :] == psp[:, None]) & (SP["pack"][psp][:, None] > 0)
                 & (_cheb(lp_pred[None, :, :], lpa[target][:, None, :]) <= 1)).sum(axis=1)
    awake = ~((SP["hibernates"][psp] > 0) & (season == 3))
    if asleep_p is not None:
        awake = awake & ~asleep_p                                           # asleep: no attack, no hunt
    startled = (SP["provoked"][psp] > 0) & (pdist <= 1) & (jax.random.uniform(jax.random.fold_in(keys[10], 1), psp.shape) < 1 - 0.95 ** _k(cfg))
    willing = (SP["provoked"][psp] == 0) | (predators.anger > 0) | startled
    fire_guard = (heat0[pos_after[target][:, 0], pos_after[target][:, 1]] > 0) & ~below_after[target]
    can_attack = predators.alive & awake & willing & (predators.rest <= 0) & (pdist <= 1) & alive0[target] \
        & ~fire_guard
    defenders = group - jnp.maximum(pack_here - 1, 0)
    deterred = can_attack & (defenders >= 2) & (SP["ambush"][psp] == 0)
    keen = SP["atk_p"][psp] * jnp.where(night | predators.below, 1.0, 1.0 - 0.5 * SP["nocturnal"][psp])   # caves are night
    attack = can_attack & ~deterred & (jax.random.uniform(keys[9], target.shape) < keen)
    bite = jnp.where(attack, SP["atk_dmg"][psp] + SP["venom"][psp], 0.0) \
        * jnp.where(down0[target], cfg.downed_hit_mult, 1.0)                    # a downed one is easy meat
    n_attacks = jnp.zeros(n).at[target].add(attack.astype(jnp.float32))
    predation = jnp.zeros(n).at[target].add(bite) + gored_dmg              # bitten, mauled, gored, trampled
    if cfg.body_parts:                                    # each bite lands on a part (throat, legs, ...)
        bite_part = jax.random.categorical(jax.random.fold_in(keys[9], 21), jnp.log(jnp.asarray(cfg.bite_parts)),
                                           shape=psp.shape)
        part_by_bite = jnp.zeros((n, N_PARTS)).at[target, bite_part].add(bite * jnp.asarray(cfg.part_fragility)[bite_part])
    else:
        part_by_bite = jnp.zeros((n, N_PARTS))
    attacked = n_attacks > 0
    attacker_sp = jnp.full(n, -1, jnp.int32).at[target].max(jnp.where(attack, psp, -1))
    venomed = jnp.zeros(n).at[target].add(jnp.where(attack, SP["venom"][psp], 0.0)) > 0
    injury = injury + jnp.zeros(n).at[target].add(jnp.where(attack, SP["atk_inj"][psp], 0.0))
    rest = jnp.where(attack | deterred, SP["atk_rest"][psp].astype(jnp.int32), jnp.maximum(predators.rest - 1, 0))
    predators = predators._replace(rest=rest)
    # Predators also hunt animals: small prey on their cell; large prey too for those built for it
    # (a pack of wolves, a big cat, a bear, a crocodile), never a mammoth; a large kill leaves a
    # carcass with meat for scavengers
    hunter = predators.alive & awake & (predators.rest <= 0)
    # chasing (cfg.animal_hunting) brings predator and prey together far more often, so each
    # meeting kills less often: herds hold at about three quarters of their slots
    hunt_scale = cfg.hunt_kill_scale if cfg.animal_hunting else 1.0
    prow = lrow(predators.pos, predators.below)
    pred_sp_grid = jnp.full((2 * g, g), -1, jnp.int32).at[prow, predators.pos[:, 1]].max(jnp.where(hunter, psp, -1))
    big_grid = jnp.zeros((2 * g, g)).at[prow, predators.pos[:, 1]].add(jnp.where(hunter, SP["takes_large"][psp], 0.0))
    srow, lrow_ = lrow(small.pos, small.below), lrow(large.pos, large.below)
    eaten_by_pred = small.alive & (pred_sp_grid[srow, small.pos[:, 1]] >= 0) & \
        (jax.random.uniform(keys[10], small.alive.shape) < cfg.predator_prey_kill_prob * hunt_scale)
    small_preyed_by = jnp.where(eaten_by_pred, pred_sp_grid[srow, small.pos[:, 1]], -1)
    small = small._replace(alive=small.alive & ~eaten_by_pred)
    can_take = SP["size"][large.species] < 0.95                                   # not a mammoth
    large_taken = large.alive & can_take & (jax.random.uniform(jax.random.fold_in(keys[10], 2), large.alive.shape)
                                           < cfg.predator_large_kill_prob * hunt_scale * big_grid[lrow_, large.pos[:, 1]])
    large_preyed_by = jnp.where(large_taken, pred_sp_grid[lrow_, large.pos[:, 1]], -1)
    meat, cave_floor = carcass(meat, cave_floor, large, jnp.where(large_taken, 0.5 * SP["meat"][large.species], 0.0))
    mins, cave_floor = leftovers(mins, cave_floor, large, large_taken, ("hide", "bone"))
    mins, cave_floor = leftovers(mins, cave_floor, small, eaten_by_pred, ("feather",))
    large = large._replace(alive=large.alive & ~large_taken)
    if cfg.grazing:                                      # a kill sates the hunter on that cell for a while
        kill_grid = jnp.zeros((2 * g, g)).at[srow, small.pos[:, 1]].add(eaten_by_pred.astype(jnp.float32)) \
            .at[lrow_, large.pos[:, 1]].add(large_taken.astype(jnp.float32))
        sated = hunter & (kill_grid[prow, predators.pos[:, 1]] > 0)
        predators = predators._replace(rest=jnp.where(sated, SP["atk_rest"][psp].astype(jnp.int32), predators.rest))

    # ---------------------------------------------------------- gathering (contested)
    gathering = a == GRASP
    cid = state.pos[:, 0] * g + state.pos[:, 1] + below0.astype(jnp.int32) * g * g   # contested per layer
    score = jnp.where(gathering, strength_now * jax.random.uniform(keys[11], (n,)), -1.0)
    best = jax.ops.segment_max(score, cid, num_segments=2 * g * g)
    winner = gathering & (score >= best[cid]) & (score > 0)
    r0, c0 = state.pos[:, 0], state.pos[:, 1]
    room = cfg.inventory_cap - state.inv.sum(axis=1)
    # Plant ecology: what grows (flora) and what lies loose. A change made to plants from outside (a
    # test, a shock) is honoured: more than the flora holds is loose food; less scales the flora down.
    if cfg.ecology:
        fsum0 = state.flora.sum(-1)
        flora = state.flora * jnp.where(fsum0 > 1e-9, jnp.minimum(state.plants / jnp.maximum(fsum0, 1e-9), 1.0),
                                        1.0)[..., None]
        loose0 = jnp.maximum(state.plants - fsum0, 0.0)
    # Ground layers as one stack, so every material follows the same rules
    ground = jnp.concatenate([jnp.stack([state.plants, meat, state.cooked, state.wood, state.stones,
                                         state.honey, state.grubs]),
                              jnp.moveaxis(mins, -1, 0)])
    avail_here = available_here(state._replace(meat=meat, cave_floor=cave_floor, minerals=mins), r0, c0,
                                below0 if cfg.underground else None)   # (N_MAT, N)
    # Grasp takes the selected material if present, otherwise whatever is most
    # plentiful here (no preference is built in); the hand then holds what was taken.
    empty_hand = held0 <= 0.05
    plenty = avail_here
    if cfg.ecology:                                   # growing plants count by what a grasp would get, not the stock
        plenty = plenty.at[PLANT].set(jnp.where(below0 if cfg.underground else False, avail_here[PLANT],
                                                forage_grid(state, cfg)[r0, c0]))
    most = jnp.argmax(jnp.minimum(plenty, 3.0), axis=0)
    sel_present = avail_here[hand0, idx] > 0.1
    grab = jnp.where(sel_present, hand0, most).astype(jnp.int32)
    here_sel = avail_here[grab, idx]
    pull_branch = winner & (hand0 == WOOD) & (state.fire[r0, c0] > 0) & (avail_here[hand0, idx] <= 0.1) & ~below0
    grab = jnp.where(pull_branch, WOOD, grab)
    # an edge in hand cuts more meat from a carcass and more wood
    cut = jnp.where((grab == RAW_MEAT) | (grab == WOOD), 1.0 + cfg.cut_bonus * edge, 1.0)
    # Termite mounds give few grubs to fingers, far more to a stick pushed in
    probing = (hand0 == WOOD) & holding
    cut = jnp.where((grab == GRUBS) & state.mound[r0, c0],
                    cfg.grub_hand_yield * jnp.where(probing, 1.0 + cfg.probe_bonus, 1.0), cut)
    take = jnp.where(winner, jnp.minimum(jnp.minimum(here_sel, cfg.gather_amount * cut), room), 0.0)
    if cfg.ecology:
        # Gathering plants takes time: loose food is picked up whole; otherwise the kind that pays
        # best here, at its return rate x how dense it is (roots faster with a digging stick)
        yields = eco.gather_yield(flora[r0, c0], (hand0 == WOOD) & holding, cfg)        # (N, 4)
        kind = jnp.argmax(yields, axis=1)
        loose_here = loose0[r0, c0]
        from_loose = loose_here > 0.05
        # Loose food goes to one (the winner, as with anything lying there); growing plants are a
        # patch, not an object: everyone gathering it gets their share (scaled down if the patch runs out)
        foraging = gathering & (grab == PLANT) & ~below0 & ~from_loose & alive0
        want = jnp.where(foraging, yields[idx, kind], 0.0)
        slot = (r0 * g + c0) * eco.N_KINDS + kind
        demand = jax.ops.segment_sum(want, slot, num_segments=g * g * eco.N_KINDS)
        stock = flora.reshape(-1)[slot]
        share = jnp.where(demand[slot] > 0, jnp.minimum(1.0, stock / jnp.maximum(demand[slot], 1e-9)), 0.0)
        take = jnp.where(winner & (grab == PLANT) & ~below0 & from_loose,
                         jnp.minimum(jnp.minimum(loose_here, cfg.gather_amount), room), take)
        take = jnp.where(foraging, jnp.minimum(want * share, room), take)
        winner = winner | (foraging & (take > 0))                         # (for the record: gathered, not lost)
        took_kind = jnp.where(foraging & (take > 0), kind, jnp.where((grab == PLANT) & from_loose & (take > 0), 4, -1))
    take = jnp.where(pull_branch, jnp.minimum(1.0, room), take)
    # Stripping greens from grass and sedge leaves stems: fiber comes with them
    grassy = (state.terrain[r0, c0] == GRASS) | (state.terrain[r0, c0] == MARSH)
    if cfg.ecology:
        stripped = foraging & (kind == 0) & (take > 0)
    else:
        stripped = winner & (grab == PLANT) & (take > 0)
    fiber_got = jnp.where(stripped & grassy & ~below0 & cfg.byproducts,
                          jnp.minimum(cfg.fiber_per_greens * take, jnp.maximum(room - take, 0.0)), 0.0)
    # Robbing a hive: the bees sting, unless smoke (a fire's heat here) has calmed them
    stung = (take > 0) & (grab == HONEY) & state.hive[r0, c0] & (heat0[r0, c0] <= 0)
    from_ground = jnp.where(pull_branch | ((grab == STONE) & (state.terrain[r0, c0] == ROCK)) | below0, 0.0, take)
    ground = jnp.maximum(ground.at[grab, r0, c0].add(-from_ground), 0.0)
    if cfg.ecology:
        flora = jnp.maximum(flora.at[r0, c0, kind].add(-jnp.where((grab == PLANT) & ~from_loose, from_ground, 0.0)), 0.0)
    mined = jnp.where(below0 & (grab != STONE), take, 0.0)               # ore out of the wall, or what lay here
    cave_floor = jnp.maximum(cave_floor.at[r0, c0, grab].add(-mined), 0.0)
    inv = state.inv.at[idx, grab].add(take).at[idx, FIBER].add(fiber_got)
    hand = jnp.where(take > 0, grab, hand).astype(jnp.int32)       # what you pick up is in your hand
    # A gift is food that someone else dropped here recently and this agent takes.
    # Dropping alone, taking back one's own drop, or dropping non-food is not giving.
    is_food = (grab == PLANT) | (grab == RAW_MEAT) | (grab == COOKED_MEAT)
    owner_raw = state.drop_owner[r0, c0]                   # dropper + N if it was dropped below
    owner = jnp.where(owner_raw >= 0, owner_raw % n, -1)
    same_floor = (owner_raw >= n) == below0                # a drop above is not one below
    recent = (state.t - state.drop_time[r0, c0]) <= cfg.gift_window
    picked_up = (from_loose | (grab != PLANT)) if cfg.ecology else True    # not what grows: only what was dropped
    giver = jnp.where((take > 0) & is_food & recent & (owner >= 0) & (owner != idx) & same_floor & picked_up, owner, -1)
    gifts = (jnp.arange(n)[:, None] == giver[None, :]) & alive0[None, :]   # [i, j]: i gave food to j
    torch = jnp.where(pull_branch, cfg.torch_ticks, state.torch)
    fire = state.fire.at[r0, c0].add(-jnp.where(pull_branch, 0.5 * cfg.wood_fuel, 0.0))
    fire = jnp.maximum(fire, 0.0)

    # ---------------------------------------------------------- eat / drink
    eating = a == EAT
    food_value = jnp.zeros(N_MAT).at[jnp.array([PLANT, RAW_MEAT, COOKED_MEAT, HONEY, GRUBS])].set(
        jnp.array([cfg.plant_food_value, cfg.meat_food_value, cfg.cooked_food_value, cfg.honey_food_value,
                   cfg.grubs_food_value]))
    pathogen = jnp.zeros(N_MAT).at[RAW_MEAT].set(cfg.raw_meat_infection)
    eaten = jnp.where(eating, jnp.minimum(inv[idx, hand0], 1.0), 0.0)
    inv = inv.at[idx, hand0].add(-eaten)
    satiety = state.satiety + eaten * food_value[hand0]
    flies = fly_density(meat, _occupancy(state.pos, state.existed & ~state.alive & ~state.below, g).astype(jnp.float32), heat0)
    mosq = mosquito_density(state, cfg, season, night, heat0)
    eat_risk = jnp.where(eaten > 0, pathogen[hand0], 0.0) * fx.get("food_water_infection", 1.0) \
        * (1.0 + cfg.fly_infection_mult * flies[state.pos[:, 0], state.pos[:, 1]] * ~below0)   # flies on the meat
    ate_cooked = (eaten > 0) & (hand0 == COOKED_MEAT)
    drinking = a == DRINK
    hydration = state.hydration + jnp.where(drinking, cfg.drink_value, 0.0)

    # ---------------------------------------------------------- social primitives
    not_self = ~jnp.eye(n, dtype=bool)
    same_layer = below0[:, None] == below0[None, :]
    same_cell = jnp.all(tgt[:, None, :] == state.pos[None, :, :], axis=-1) & same_layer   # strike target == j
    hits = hit_agent[:, None] & same_cell & alive0[None, :] & not_self
    # Release: drop one unit (meat first) on the current cell; agents adjacent see a gift
    releasing = a == RELEASE
    dropped = jnp.where(releasing, jnp.minimum(inv[idx, hand0], 1.0), 0.0)
    inv = inv.at[idx, hand0].add(-dropped)
    ground = ground.at[hand0, r0, c0].add(jnp.where(below0, 0.0, dropped))
    cave_floor = cave_floor.at[r0, c0, hand0].add(jnp.where(below0, dropped, 0.0))     # below: on the cave floor
    adjacent = (_cheb(lp0[:, None, :], lp0[None, :, :]) <= 1) & not_self
    food_drop = releasing & (dropped > 0) & ((hand0 == PLANT) | (hand0 == RAW_MEAT) | (hand0 == COOKED_MEAT))
    drop_owner = state.drop_owner.at[r0, c0].set(jnp.where(food_drop, idx + n * below0, state.drop_owner[r0, c0]))
    drop_time = state.drop_time.at[r0, c0].set(jnp.where(food_drop, state.t, state.drop_time[r0, c0]))
    # Place: one unit of the hand material onto a neighboring cell
    pdir = jnp.where(is_place, a - PLACE_N + 1, 0)
    ptgt = jnp.clip(state.pos + MOVE_DELTAS[pdir], 0, g - 1)
    placed = jnp.where(is_place, jnp.minimum(inv[idx, hand0], 1.0), 0.0)
    inv = inv.at[idx, hand0].add(-placed)
    ground = ground.at[hand0, ptgt[:, 0], ptgt[:, 1]].add(jnp.where(below0, 0.0, placed))
    cave_floor = cave_floor.at[ptgt[:, 0], ptgt[:, 1], hand0].add(jnp.where(below0, placed, 0.0))
    # Binding: fiber placed where a stick and a point (anything that takes an edge) lie together ties
    # them into a spear. Nothing names it: it is what placing those three in one spot does.
    there = ground[:, ptgt[:, 0], ptgt[:, 1]]                                   # (N_MAT, N)
    point_ok = (there[MIN0:] >= 1.0) & (MIN_SHARP[:, None] >= cfg.haft_edge) \
        & (jnp.arange(mn.M)[:, None] != SPEAR - MIN0)                           # (M, N)
    point = jnp.argmax(jnp.where(point_ok, MIN_SHARP[:, None], -1.0), axis=0)   # the sharpest there
    bound = is_place & (placed > 0) & (hand0 == FIBER) & ~below0 & (there[WOOD] >= 1.0) \
        & (there[FIBER] >= 1.0) & jnp.any(point_ok, axis=0) & cfg.byproducts
    b1 = bound.astype(jnp.float32)
    ground = ground.at[WOOD, ptgt[:, 0], ptgt[:, 1]].add(-b1).at[FIBER, ptgt[:, 0], ptgt[:, 1]].add(-b1) \
        .at[MIN0 + point, ptgt[:, 0], ptgt[:, 1]].add(-b1).at[SPEAR, ptgt[:, 0], ptgt[:, 1]].add(b1)
    ground = jnp.maximum(ground, 0.0)
    # Burning wood leaves the hand with the unit it is attached to (below it goes out: no fire there)
    lit_drop = (torch > 0) & (hand0 == WOOD) & ((placed > 0) | (dropped > 0))
    snuffed = lit_drop & below0
    lit_drop = lit_drop & ~below0
    lit_cell = jnp.where((placed > 0)[:, None], ptgt, state.pos)
    torch = jnp.where(lit_drop | snuffed, 0, torch)
    is_fuel = (hand0 == WOOD) | ((hand0 >= MIN0) & (jnp.concatenate([jnp.zeros(MIN0), MIN_FUEL])[hand0] > 0))
    fed_fire = (placed > 0) & is_fuel & (state.fire[ptgt[:, 0], ptgt[:, 1]] > 0) & ~below0
    meat_to_heat = (placed > 0) & (hand0 == RAW_MEAT) & (heat0[ptgt[:, 0], ptgt[:, 1]] > 0) & ~below0
    # Touch
    is_touch = (a >= TOUCH_N) & (a < TOUCH_N + 8)
    tdir = jnp.where(is_touch, a - TOUCH_N + 1, 0)
    ttgt = state.pos + MOVE_DELTAS[tdir]
    touches = is_touch[:, None] & jnp.all(ttgt[:, None, :] == state.pos[None, :, :], axis=-1) \
        & alive0[None, :] & not_self & same_layer
    # Messages: a separate channel, emitted alongside the action
    msg = jnp.zeros((n, MSG_LEN), jnp.int32) if messages is None else messages.astype(jnp.int32)
    msg = jnp.clip(msg, 0, VOCAB - 1)
    written = msg[:, MSG_LEN:2 * MSG_LEN] if msg.shape[1] >= 2 * MSG_LEN else None   # the text to write, if given
    msg = msg[:, :MSG_LEN]
    if cfg.channel == "none":
        msg = jnp.zeros_like(msg)
    spoke_now = alive0 & (msg[:, 0] > 0)                   # first symbol 0 = staying silent
    msg = jnp.where(spoke_now[:, None], msg, 0)
    # What listeners hear from each speaker next tick: the message itself, or (channel "shuffled")
    # another agent's this tick, so content is cut from the speaker's state (a control, not a world)
    if cfg.channel == "shuffled":
        heard_msg = msg[jax.random.permutation(jax.random.fold_in(keys[0], 95), n)]
    else:
        heard_msg = msg
    # Writing: one's message this tick (else the last one), cut or drawn on what is here (write_means)
    if cfg.writing:
        w_ok, w_surf, w_paint = write_means(state, cfg)
        writing = (a == WRITE) & alive0 & ~below0 & ~state.asleep & w_ok
    else:
        writing, w_surf, w_paint = jnp.zeros(n, bool), jnp.zeros(n, jnp.int32), jnp.zeros(n, bool)
    # The text written: chosen for writing when the brain gives it (its written symbols), else
    # (older callers) one's message this tick or the last one
    text = written if written is not None else jnp.where(spoke_now[:, None], msg, state.last_msg)
    writing = writing & jnp.any(text > 0, axis=1)
    wcell = jnp.where(writing[:, None], state.pos, g)                     # out of range: dropped
    marks = state.marks.at[wcell[:, 0], wcell[:, 1]].set(text, mode="drop")
    mark_time = state.mark_time.at[wcell[:, 0], wcell[:, 1]].set(state.t, mode="drop")
    mark_kind = state.mark_kind.at[wcell[:, 0], wcell[:, 1]].set((w_surf + MARK_PAINTED * w_paint).astype(jnp.int32), mode="drop")
    age_days = (state.t - mark_time) * cfg.tick_minutes / 1440.0
    on = mark_kind % MARK_PAINTED
    lasts = jnp.where(mark_kind >= MARK_PAINTED, cfg.mark_days_paint,
                      jnp.where((on == SURF_ROCK) | (on == SURF_STONE), cfg.mark_days_rock, cfg.mark_days))
    gone = (mark_time >= 0) & (age_days > lasts)
    marks = jnp.where(gone[..., None], 0, marks)
    mark_time = jnp.where(gone, -1, mark_time)
    # Relationship memory: observer (row) about actor (column)
    sight_now = jnp.where(night, dark_radius(state, cfg), cfg.view_radius)  # darkness limits witnessing
    sight_now = jnp.where(below0, light0, sight_now)
    seeing = (_cheb(lp0[:, None, :], lp0[None, :, :]) <= sight_now[:, None]) \
        & not_self & alive0[:, None]                                   # k can see j
    witnessed = seeing.astype(jnp.float32) @ hits.T.astype(jnp.float32)  # [k, i]: k saw i strike someone
    witnessed = witnessed * not_self                                   # a striker does not witness itself
    mem_inc = jnp.stack([hits.T, gifts.T, touches.T], axis=-1).astype(jnp.float32)
    mem_inc = jnp.concatenate([mem_inc, witnessed[..., None]], axis=-1)
    memory = state.memory * cfg.memory_decay + mem_inc * alive0[:, None, None]

    # ---------------------------------------------------------- disease
    susceptibility = (1 - state.immunity) / geno.disease_resistance * fx.get("infection", 1.0)
    water_risk = cfg.water_infection * jnp.where(weather == DROUGHT, 2.0, 1.0)
    if cfg.biomes:
        water_risk = water_risk * jnp.where(open_water_beside(state.terrain, state.pos, cfg) | below0, 1.0,
                                            cfg.marsh_infection_mult)          # a cave pool is open water
    if cfg.fouling:                                       # waste beside the water one drinks from
        fp = jnp.pad(state.foul, 1)
        foul_near = jnp.max(jnp.stack([fp[1 + dr:1 + dr + g, 1 + dc:1 + dc + g] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]), 0)
        dirty = jnp.minimum(1.0 + foul_near[state.pos[:, 0], state.pos[:, 1]] / cfg.foul_ref, cfg.foul_max_mult)
        water_risk = water_risk * jnp.where(below0, 1.0, dirty)              # a cave pool is not fouled from above
    infected_now = state.infection > 0
    near = (_cheb(lpa[:, None, :], lpa[None, :, :]) <= 1) & ~jnp.eye(n, dtype=bool)
    sick_neighbors = jnp.sum(near & (infected_now & alive0)[None, :], axis=1)
    bitten = mosq[pos_after[:, 0], pos_after[:, 1]] * cfg.mosquito_infection * ~below_after   # fever from bites
    p_inf = 1 - (1 - eat_risk) \
        * (1 - jnp.where(drinking, water_risk, 0.0)) \
        * (1 - cfg.contact_infection) ** sick_neighbors * (1 - bitten)
    newly_infected = alive0 & ~infected_now & \
        (jax.random.uniform(keys[12], (n,)) < jnp.clip(p_inf * susceptibility, 0.0, 1.0))
    infection = jnp.where(newly_infected, cfg.disease_duration,
                          jnp.maximum(state.infection - 1, 0)).astype(jnp.int32)
    recovered = alive0 & infected_now & (infection == 0)
    immunity = jnp.where(recovered, 1.0, jnp.maximum(state.immunity - cfg.immunity_decay, 0.0))
    sick = infection > 0

    # ---------------------------------------------------------- temperature
    terr_here = jnp.where(below_after, CAVE, state.terrain[pos_after[:, 0], pos_after[:, 1]])
    amb = ambient_temperature(state.t, weather, terr_here, cfg, state.height[pos_after[:, 0], pos_after[:, 1]],
                              pos_after[:, 0].astype(jnp.float32), clim) \
        + heat0[pos_after[:, 0], pos_after[:, 1]]
    if cfg.underground:
        amb = amb + jnp.where(below_after, 0.0, spring_warmth(state, cfg)[pos_after[:, 0], pos_after[:, 1]])
        amb = jnp.where(below_after, cave_air(state.t, cfg, clim, state.under_kind[pos_after[:, 0], pos_after[:, 1]]),
                        amb)                                             # below: no sun, weather or fire above
    rained_on = raining[pos_after[:, 0], pos_after[:, 1]] & (terr_here != CAVE) & alive0
    snow_here = snow[pos_after[:, 0], pos_after[:, 1]] * ~below_after
    load = (amb - cfg.comfort_temp) / cfg.temp_scale \
        + jnp.where(resting & (amb < cfg.comfort_temp), cfg.rest_warmth, 0.0)   # curling up keeps heat in, only when cold
    load = load + jnp.where(moving, 0.05, 0.0)               # activity produces heat
    if cfg.dynamic_weather:
        load = load - jnp.where(rained_on, cfg.rain_chill, 0.0) - cfg.snow_chill * snow_here
    huddle = jnp.minimum(jnp.sum(near & alive0[None, :], axis=1), 3)
    load = load + cfg.huddle_warmth * huddle                   # body heat of neighbors
    # clothing keeps heat in when it is colder than comfort (and does little against heat)
    if cfg.byproducts:          # hides and feathers carried keep warm; leaves or a hide under a sleeper is a bed
        cover = jnp.clip(state.clothing + inv @ INSULATION, 0.0, 1.0)
        bed_mins = ground[MIN0:, pos_after[:, 0], pos_after[:, 1]]                 # (M, N)
        bed = jnp.minimum(MIN_BEDDING @ bed_mins, cfg.bedding_max)
        bedded = sleeping & alive0 & ~below_after & (bed >= 0.1)
        bed = jnp.where(bedded, bed, 0.0)
    else:
        cover, bed, bedded = state.clothing, jnp.zeros(n), jnp.zeros(n, bool)
    wrap = 1.0 + (cfg.clothing_insulation * cover + bed) * (load < 0)
    body_temp = state.body_temp + cfg.thermal_rate / (geno.insulation * fx.get("insulation", 1.0) * wrap) * (load - state.body_temp)
    body_temp = jnp.clip(body_temp, -1.5, 1.5)

    # ---------------------------------------------------------- metabolism
    carrying_child = (state.gestation > 0).astype(jnp.float32) * cfg.pregnancy_need   # 0 unless pregnant
    # Kleiber's law: needs ~ lean mass^0.75, from one's own frame and growth (body's ideal weight
    # against an average adult's); and what one is doing (physical activity level)
    ref_kg = cfg.ideal_bmi * (cfg.base_height_cm / 100.0) ** 2
    size_need = (bod["ideal_kg"] / ref_kg) ** 0.75 if cfg.needs_by_size else 1.0
    if cfg.activity_needs:
        doing = jnp.where(sleeping, cfg.activity_sleep, jnp.where(is_strike, cfg.activity_strike,
                jnp.where(moving, cfg.activity_walk, jnp.where(gathering | is_place | (a == WRITE), cfg.activity_work,
                                                               cfg.activity_rest))))
        size_need = size_need * doing / cfg.activity_norm
    hunger = cfg.hunger_rate * geno.metabolism * (1 + jnp.where(sick, cfg.disease_hunger_mult, 0.0)) \
        * (1 + carrying_child) * size_need
    thirst = size_need * cfg.thirst_rate * jnp.where(weather == DROUGHT, cfg.drought_thirst_mult, 1.0) \
        * jnp.where(terr_here == DESERT, cfg.desert_thirst_mult if cfg.biomes else 1.0, 1.0) \
        * (1 + cfg.sweat_thirst_mult * jnp.maximum(body_temp - 0.2, 0.0))
    rest_gain = jnp.where(resting, cfg.rest_recovery * jnp.where(night, cfg.night_rest_bonus, 1.0), 0.0)
    # Sleep: deepens over unbroken ticks; deep sleep at night restores most
    sleep_depth = jnp.where(sleeping, state.sleep_depth + 1, 0)
    depth = jnp.minimum(sleep_depth, cfg.sleep_deep_ticks) / cfg.sleep_deep_ticks
    rest_gain = rest_gain + jnp.where(sleeping, cfg.sleep_energy * depth, 0.0)
    sleep_eff = jnp.where(night, 1.0, cfg.sleep_day_efficiency)
    sleepiness = jnp.where(sleeping, state.sleepiness - cfg.sleep_recovery * (0.5 + 0.5 * depth) * sleep_eff,
                           state.sleepiness + cfg.sleep_pressure * jnp.where(night, cfg.sleep_night_mult, 1.0))
    sleepiness = jnp.where(alive0, jnp.clip(sleepiness, 0.0, 1.0), state.sleepiness)
    energy = state.energy + rest_gain - jnp.where(resting, 0.0, cfg.fatigue_rate * fx.get("fatigue", 1.0)) \
        - move_energy * fx.get("walk_cost", 1.0) \
        - jnp.where(sick, cfg.disease_fatigue, 0.0) - cfg.stress_energy_cost * state.stress \
        - jnp.where(spoke_now, cfg.speak_cost, 0.0)
    satiety = jnp.clip(satiety - hunger, 0.0, 1.0)
    # Body reserves: store fat when well fed, burn it to blunt hunger when starving
    gain = jnp.where((satiety > 0.7) & (state.weight < 1.15 * bod["ideal_kg"]), cfg.fat_gain_kg, 0.0)
    burn = jnp.where((satiety < 0.2) & (state.weight > 0.8 * bod["ideal_kg"]), cfg.fat_burn_kg, 0.0)
    # Growth: a child's lean mass follows its growing frame while it is fed at all (a starving
    # child is stunted). Fat (above) is separate; without this, children could only gain weight as
    # fat and every child born grew up emaciated, below the weight needed to conceive.
    grown = body(state._replace(age=state.age + 1), cfg)["ideal_kg"] - bod["ideal_kg"]
    growth = jnp.where(alive0 & (satiety > 0.3), jnp.maximum(grown, 0.0), 0.0)
    deficit = jnp.maximum(bod["ideal_kg"] - state.weight, 0.0)          # catch-up growth (EnvConfig.catch_up_years)
    growth = growth + jnp.where(alive0 & (satiety > 0.5) & (bod["years"] < cfg.catch_up_until_years),
                                deficit / (cfg.catch_up_years * cfg.ticks_per_bio_year), 0.0)
    weight = state.weight + gain - burn + growth
    satiety = jnp.clip(satiety + burn * cfg.satiety_per_kg, 0.0, 1.0)
    hydration = jnp.clip(hydration - thirst * (1 + carrying_child), 0.0, 1.0)
    if cfg.dynamic_weather:
        hydration = jnp.clip(hydration + jnp.where(rained_on, cfg.rain_drink, 0.0), 0.0, 1.0)   # catching rain
    energy = jnp.clip(energy, 0.0, 1.0)
    # Nursing: a young child within 1 cell of its mother takes milk; she pays most of it
    if cfg.reproduction:
        mom = mom0                                       # caregiver_of: mother, or a wet nurse
        nursing = alive0 & (care0 >= 0) & alive0[mom] & (bod["years"] < cfg.nurse_years) \
            & (_cheb(lpa, lpa[mom]) <= 1)
        milk_food = jnp.where(nursing, jnp.minimum(cfg.nurse_rate, 1.0 - satiety), 0.0)
        milk_water = jnp.where(nursing, jnp.minimum(cfg.nurse_rate, 1.0 - hydration), 0.0)
        satiety = jnp.clip(satiety + milk_food - jnp.zeros(n).at[mom].add(cfg.nurse_cost * milk_food), 0.0, 1.0)
        hydration = jnp.clip(hydration + milk_water - jnp.zeros(n).at[mom].add(cfg.nurse_cost * milk_water), 0.0, 1.0)
    else:
        nursing = jnp.zeros(n, bool)

    # ---------------------------------------------------------- damage and healing
    burned = alive0 & (state.fire[pos_after[:, 0], pos_after[:, 1]] > 0) & ~below_after
    injury = jnp.clip(injury + jnp.where(burned, cfg.burn_injury, 0.0), 0.0, 1.0)
    # Body parts: what this tick's blows, bites and falls did to each part; a destroyed head or neck
    # kills (counted to the blow, else the bite)
    if cfg.body_parts:
        part_new = state.part_damage + part_by_blow + part_by_bite \
            + jnp.zeros((n, N_PARTS)).at[:, PART_LEGS].set(fall_damage)
        vital_gone = alive0 & ((part_new[:, PART_HEAD] >= 1.0) | (part_new[:, PART_NECK] >= 1.0))
        by_blow = (part_by_blow[:, PART_HEAD] + part_by_blow[:, PART_NECK]) > 0
        violence = violence + jnp.where(vital_gone & by_blow, 1.0, 0.0)
        predation = predation + jnp.where(vital_gone & ~by_blow, 1.0, 0.0)
    components = jnp.stack([
        jnp.where(satiety <= 0, cfg.starve_damage, 0.0),
        jnp.where(hydration <= 0, cfg.thirst_damage, 0.0),
        jnp.where(energy <= 0, cfg.exhaust_damage, 0.0),
        jnp.where(body_temp < cfg.cold_threshold, cfg.cold_damage, 0.0),
        jnp.where(body_temp > cfg.heat_threshold, cfg.heat_damage, 0.0),
        jnp.where(sick, cfg.disease_damage / geno.disease_resistance, 0.0),
        predation + jnp.where(stung, cfg.sting_damage, 0.0),     # bitten, gored, stung
        violence,
        cfg.injury_bleed * jnp.maximum(injury - 0.5, 0.0) * 2 * fx.get("bleeding", 1.0),   # bleeding from bad injuries
        jnp.where(burned, cfg.burn_damage, 0.0),
        fall_damage,
    ], axis=-1)
    damage = components.sum(-1)
    thriving = (satiety > 0.3) & (hydration > 0.3) & (energy > 0.3) & ~sick \
        & (state.stress < cfg.stress_heal_block)
    calm = jnp.clip(1.0 - state.stress / cfg.stress_heal_block, 0.0, 1.0)   # healing slows with stress
    calm = calm * jnp.where(sleeping, cfg.sleep_heal_mult, 1.0)                # and quickens in sleep
    health = jnp.clip(state.health - damage + jnp.where(thriving, cfg.heal_rate * calm * fx.get("healing", 1.0), 0.0), 0.0, 1.0)
    health = jnp.where(alive0, health, state.health)
    health_loss = jnp.maximum(state.health - health, 0.0)
    # Lasting marks: severe injuries leave a part that never heals; wounds leave scars
    lasting_injury = jnp.where(alive0, jnp.clip(jnp.maximum(
        state.lasting_injury, cfg.lasting_share * (jnp.clip(injury, 0.0, 1.0) - cfg.severe_injury)),
        0.0, cfg.max_lasting_injury), state.lasting_injury)
    injury = jnp.clip(injury - jnp.where(thriving, cfg.injury_heal_rate * calm, 0.0), lasting_injury, 1.0)
    part_damage = jnp.clip(part_new - jnp.where(thriving, cfg.injury_heal_rate * calm, 0.0)[:, None], 0.0, 1.0) \
        if cfg.body_parts else state.part_damage                              # parts heal as injuries do
    # Every wound scars by the health it takes: blows, predator attacks, burns and falls
    wound = predation + violence + jnp.where(burned, cfg.burn_damage, 0.0) + fall_damage
    scars = jnp.where(alive0, jnp.clip(state.scars + cfg.scar_per_wound * wound, 0.0, 1.0), state.scars)
    # Fresh pain: each wound hurts by the health it takes, and the hurt fades over a few hours.
    # An old wound that never healed flares now and then with no new hurt (chronic and phantom pain).
    flare = jax.random.uniform(jax.random.fold_in(keys[0], 93), (n,)) \
        < (1.0 - (1.0 - cfg.phantom_prob) ** _k(cfg)) * state.lasting_injury / cfg.max_lasting_injury
    pain = jnp.clip(state.pain * (1.0 - cfg.pain_fade) ** _k(cfg) + cfg.pain_per_wound * wound
                    + jnp.where(flare & alive0, cfg.phantom_pain, 0.0), 0.0, 2.0)

    # ---------------------------------------------------------- psychology
    # Below without light where one stands (no burning branch near): fear of the dark
    dark_now = (alive0 & below_after & ~torchlit[pos_after[:, 0], pos_after[:, 1]] & ~(torch > 0)).astype(jnp.float32) \
        if cfg.underground else jnp.zeros(n)
    pred_near = jnp.any((_cheb(lpa[:, None, :], layered_pos(predators.pos, predators.below)[None]) <= 3)
                        & predators.alive[None, :], axis=1)
    critical = ((satiety < 0.15).astype(jnp.float32) + (hydration < 0.15) + (energy < 0.15)
                + (body_temp < cfg.cold_threshold) + (body_temp > cfg.heat_threshold))
    # Company in view: the first companion matters most (each one halves loneliness)
    in_sight = jnp.minimum(jnp.where(night, dark_radius(state, cfg), cfg.view_radius), sight_radius(state, cfg))
    in_sight = jnp.where(below_after, jnp.minimum(light0, sight_radius(state, cfg)), in_sight)
    companions = ((_cheb(lpa[:, None, :], lpa[None, :, :]) <= in_sight[:, None])
                  & ~jnp.eye(n, dtype=bool) & alive0[None, :]).sum(1)
    lonely = 0.5 ** companions.astype(jnp.float32)                   # 1 alone, 0.5, 0.25, ...
    in_view = (_cheb(lpa[:, None, :], lpa[None, :, :]) <= in_sight[:, None]) \
        & ~jnp.eye(n, dtype=bool) & alive0[None, :]
    attached_near = jnp.minimum((state.attachment * in_view).sum(1), 1.0)
    attached_away = jnp.minimum((state.attachment * (~in_view & alive0[None, :])).sum(1), 1.0)
    # Excretion: the urge builds with what the body processes; relieved by choice, or past full anyway
    if cfg.excretion:
        made = cfg.urge_rate * (0.5 + state.hydration) * jnp.where(state.asleep, cfg.urge_sleep_mult, 1.0)
        urge_now = jnp.where(alive0, state.urge + made, state.urge)
        relieving = (a == RELIEVE) & alive0
        if cfg.reproduction:
            no_control = (state.age + state.birth_age) / cfg.ticks_per_bio_year < cfg.infant_continence_years
        else:
            no_control = jnp.zeros(n, bool)
        soiled = alive0 & ~relieving & ((urge_now >= 1.0) | (no_control & (urge_now >= 0.5)))
        relieved = jnp.where(relieving | soiled, jnp.minimum(urge_now, 1.0), 0.0)
        urge = jnp.where(relieved > 0, 0.0, jnp.minimum(urge_now, 1.0))
        urge_stress = cfg.urge_stress * jnp.clip((urge_now - 0.7) / 0.3, 0.0, 1.0) \
            + jnp.where(soiled & ~no_control, cfg.soil_stress, 0.0)
    else:
        relieving = soiled = no_control = jnp.zeros(n, bool)
        relieved, urge, urge_stress = jnp.zeros(n), state.urge, jnp.zeros(n)
    gain = fx.get("stress_gain", 1.0)
    stress_src = jnp.stack([                       # what raised stress this tick, by source (STRESS_SOURCES)
        gain * cfg.stress_from_damage * health_loss, gain * jnp.where(pred_near, cfg.stress_from_threat, 0.0),
        gain * cfg.stress_from_need * critical, cfg.loneliness_stress * geno.social_reward * lonely,
        cfg.attach_absence * attached_away, cfg.sleep_debt_stress * jnp.clip((state.sleepiness - 0.7) / 0.3, 0.0, 1.0),
        gain * cfg.dark_stress * dark_now * ~sleeping, urge_stress,
        cfg.heartache_stress * state.heartache], axis=-1) * jnp.ones((n, 1))
    stress = state.stress * (cfg.stress_decay - cfg.social_buffering * (1.0 - lonely)
                             - cfg.attach_calm * attached_near) \
        + cfg.attach_absence * attached_away \
        + fx.get("stress_gain", 1.0) * (cfg.stress_from_damage * health_loss
                                        + jnp.where(pred_near, cfg.stress_from_threat, 0.0) + cfg.stress_from_need * critical) \
        + cfg.loneliness_stress * geno.social_reward * lonely \
        + cfg.sleep_debt_stress * jnp.clip((state.sleepiness - 0.7) / 0.3, 0.0, 1.0) \
        + fx.get("stress_gain", 1.0) * cfg.dark_stress * dark_now * ~sleeping + urge_stress \
        + cfg.heartache_stress * state.heartache \
        - jnp.where(sleeping, cfg.sleep_stress_relief * depth, 0.0)
    stress = jnp.clip(stress, 0.0, 1.0)
    pain_aversion = state.pain_aversion + geno.temperament_drift * cfg.trauma_rate * health_loss \
        - cfg.trauma_recovery * geno.temperament_drift * (state.pain_aversion - geno.pain_aversion)
    pain_aversion = jnp.clip(pain_aversion, 0.25, 4.0)

    # ---------------------------------------------------------- aging and death
    age = state.age + alive0.astype(jnp.int32)
    # No fixed death age: senescence is a Gompertz hazard, h(x) = b exp(b (x - M)) per biological
    # year, whose most likely age of death M is max_bio_years scaled by the lifespan gene and
    # conditions. Other deaths (hunger, cold, disease, predators...) are modelled as they happen,
    # so this is only the intrinsic part. It is drawn once per biological day (each agent on its own
    # phase) rather than every tick, so the chance stays above float32's resolution at fine ticks.
    tpy_ = cfg.ticks_per_bio_year
    every = max(1, int(round(tpy_ / 365.0)))
    total_age = age + state.birth_age
    hazard = senescence_hazard(total_age / tpy_, geno.lifespan * fx.get("lifespan", 1.0), cfg)
    p_old = 1.0 - jnp.exp(-hazard * every / tpy_)
    old = alive0 & (total_age % every == 0) \
        & (jax.random.uniform(jax.random.fold_in(keys[0], 91), (n,)) < p_old)
    dead_now = alive0 & ((health <= 0.0) | old)
    alive = alive0 & ~dead_now
    cause = jnp.where(old, CAUSE_AGE, 1 + jnp.argmax(components, axis=-1))
    cause = jnp.where(dead_now, cause, CAUSE_NONE)

    # ---------------------------------------------------------- reproduction
    if cfg.reproduction:
        tpy = cfg.ticks_per_bio_year
        years = (age + state.birth_age) / tpy
        female = state.sex == FEMALE
        adult = years >= cfg.adult_years
        adj = (_cheb(lpa[:, None, :], lpa[None, :, :]) <= 1) & ~jnp.eye(n, dtype=bool)
        # Consent is mutual but need not fall on the same half-hour tick: a choice to mate stays
        # open (visibly, as courting) for mate_window ticks, so a pair mates when one chooses it
        # while the other, awake and adjacent, has chosen it within the window (1: same tick only).
        mating = is_mate & alive
        stays_now = (a == REST) | ((a >= TOUCH_N) & (a < TOUCH_N + 8))
        willing = mating | ((state.courting > 0) & alive & stays_now)
        both = (mating[:, None] & willing[None, :]) | (willing[:, None] & mating[None, :])
        to_other, to_same = gt.attraction_of(geno)
        pair = both & adj & female[:, None] & ~female[None, :] \
            & adult[:, None] & adult[None, :] \
            & to_other[:, None] & to_other[None, :]                       # [woman, man], both chose it
        # Two of the same sex, both drawn to their own sex: intimacy, never a conception
        same_pair = both & adj & (female[:, None] == female[None, :]) & adult[:, None] & adult[None, :] \
            & to_same[:, None] & to_same[None, :]
        # Lactational amenorrhea: a mother with a living child young enough to nurse does not ovulate
        # (it returns if the child dies). With gestation and the wait to conceive this spaces births
        # about 3.4 biological years apart, as among foragers (3-4).
        infant = alive0 & (state.mother >= 0) & (years < cfg.lactation_infertile_years)
        has_infant = jnp.zeros(n, bool).at[jnp.clip(state.mother, 0)].max(infant)
        fertile_f = female & alive & adult & (years < cfg.menopause_years) & (state.gestation == 0) & ~has_infant \
            & (weight >= cfg.min_condition * bod["ideal_kg"]) & ~sick & (state.stress < cfg.stress_heal_block) \
            & (fx.get("fertile", 1.0) > 0)
        fertile_m = ~female & alive & adult & ~sick & (fx.get("fertile", 1.0) > 0)
        partner = jnp.argmax(pair, axis=1)
        k_c, k_b = jax.random.split(jax.random.fold_in(keys[0], 77))
        conceive = jnp.any(pair, axis=1) & fertile_f & fertile_m[partner] \
            & (jax.random.uniform(k_c, (n,)) < cfg.conception_prob)
        free = (~alive).sum() - (state.gestation > 0).sum()             # slots left for children
        could = conceive
        conceive = conceive & (jnp.cumsum(conceive) <= free)
        no_room = could & ~conceive           # the slot cap, not the world, stopped this child
        mated = jnp.any(pair, axis=1) | jnp.any(pair, axis=0) | jnp.any(same_pair, axis=1)
        same_sex_mated = jnp.any(same_pair, axis=1)
        # An open offer lasts only while the agent stays with it: resting beside or touching the
        # other keeps it; doing anything else (moving away, striking, eating, sleeping...) withdraws
        # it at once. No one can be mated without having chosen it themselves within the window.
        stays = mating | (a == REST) | ((a >= TOUCH_N) & (a < TOUCH_N + 8))
        courting = jnp.where(mated | ~alive | ~stays, 0,
                             jnp.where(mating, cfg.mate_window - 1, jnp.maximum(state.courting - 1, 0)))
        # Intimacy calms both partners, by their libido, less with each repeat with the same partner
        # (same habituation as other felt acts). Stress cannot go below zero, so it cannot be farmed.
        intimate = (pair | pair.T | same_pair).astype(jnp.float32)
        fresh_i = jnp.exp(-state.habituation[..., HAB_INTIMACY] * 0.5 ** (1.0 / cfg.habituation_half_life))
        relief = cfg.intimacy_relief * geno.libido * (intimate * fresh_i).sum(1)
        stress = jnp.clip(stress - relief, 0.0, 1.0)
        gestation = jnp.where(conceive, 1, jnp.where((state.gestation > 0) & alive, state.gestation + 1, 0))
        birthing = alive & (state.gestation > 0) & (gestation >= int(cfg.gestation_years * tpy))
        risk = cfg.maternal_death * jnp.where(health < 0.5, 2.0, 1.0)
        died_giving_birth = birthing & (jax.random.uniform(k_b, (n,)) < risk)
        dead_now = dead_now | died_giving_birth
        alive = alive & ~died_giving_birth
        health = jnp.where(died_giving_birth, 0.0, health)
        cause = jnp.where(died_giving_birth, CAUSE_CHILDBIRTH, cause)
        gestation = jnp.where(birthing, 0, gestation)
        father = jnp.where(conceive, partner, state.father)
        fetus_r = jnp.where(conceive[:, None], 0.5 * (state.relatedness + state.relatedness[partner]), state.fetus_r)
        father_cond = jnp.where(conceive[:, None], state.conditions[partner], state.father_cond)
        father_geno = jax.tree_util.tree_map(lambda f, g_: jnp.where(conceive, g_[partner], f),
                                             state.father_geno, geno)
        father_look = jnp.where(conceive[:, None], state.appearance[partner], state.father_look)
    else:
        mated = conceive = birthing = no_room = same_sex_mated = jnp.zeros(n, bool)
        courting = state.courting
        intimate = jnp.zeros((n, n))
        gestation, father, fetus_r = state.gestation, state.father, state.fetus_r
        father_cond = state.father_cond
        father_geno, father_look = state.father_geno, state.father_look

    # ---------------------------------------------------------- world dynamics
    regrow = jnp.array([*SEASON_REGROW, cfg.winter_regrow_factor])[season]
    regrow = regrow * jnp.where(weather == DROUGHT, cfg.drought_regrow_mult, 1.0)
    breed = regrow * jnp.where(weather == RAIN, cfg.rain_regrow_mult, 1.0)   # animals breed with plant growth
    regrow = regrow * jnp.where(raining, cfg.rain_regrow_mult, 1.0) * (1.0 - snow)   # (G, G): local rain, snow
    grass = cfg.plant_regrow_grass
    base_rate = cfg.plant_abundance * jnp.select(
        [state.terrain == FOREST, state.terrain == GRASS, state.terrain == MARSH,
         state.terrain == TUNDRA, state.terrain == DESERT],
        [cfg.plant_regrow_forest, grass, grass * cfg.marsh_regrow, grass * cfg.tundra_regrow,
         grass * cfg.desert_regrow], 0.0)
    plants, meat, cooked, wood, stones, honey, grubs = ground[:MIN0]
    minerals = jnp.moveaxis(ground[MIN0:], 0, -1)                               # (G, G, M)
    if cfg.ecology:
        loose = jnp.maximum(plants - flora.sum(-1), 0.0)                        # dropped or placed plant food
    # ---- fire: ignition (sparks, burning wood, lightning, spread), fuel, burning down
    dryness = jnp.array(cfg.fire_dryness)[weather]
    if cfg.dynamic_weather:
        dryness = dryness * jnp.where(raining | (snow > 0.2), 0.1, 1.0)   # wet ground barely burns
    coal_here = minerals @ (MIN_FUEL / _k(cfg))                                              # ticks of coal fuel lying here
    if cfg.ecology:                                   # what burns: the plants above ground (not roots) and loose food
        Kc = eco.capacity(state.terrain, state.flora_patch, cfg)[..., :3].sum(-1)
        above = jnp.where(Kc > 0, flora[..., :3].sum(-1) / jnp.maximum(Kc, 1e-9), 0.0)
        plant_fuel_amount = cfg.plant_max * jnp.where(state.terrain == FOREST, 0.7, 0.4) * above + jnp.minimum(loose, cfg.plant_max)
    else:
        plant_fuel_amount = plants
    fuel_here = plant_fuel_amount * cfg.plant_fuel + wood * cfg.wood_fuel + coal_here
    burning0 = (fire > 0).astype(jnp.float32)
    spark_grid = jnp.zeros((g, g)).at[tr, tc].max(spark.astype(jnp.float32))
    torch_grid = jnp.zeros((g, g)).at[lit_cell[:, 0], lit_cell[:, 1]].max(lit_drop.astype(jnp.float32))
    storm = (season == 1) | (weather == DROUGHT) | (weather == HEAT_WAVE)
    strike_cell = jax.random.randint(k_fire[1], (2,), 0, g)
    if cfg.dynamic_weather:                               # lightning comes from storm clouds
        w = storms.reshape(-1).astype(jnp.float32)
        idx_s = jax.random.choice(k_fire[1], g * g, p=(w + 1e-9) / (w + 1e-9).sum())
        strike_cell = jnp.stack([idx_s // g, idx_s % g])
        storm = jnp.any(storms)
    bolt = storm & (jax.random.uniform(k_fire[2]) < cfg.lightning_prob)
    bolt_grid = jnp.zeros((g, g)).at[strike_cell[0], strike_cell[1]].set(bolt.astype(jnp.float32))
    p_spread = 1 - (1 - cfg.spread_prob * dryness) ** _neighbor_count(burning0)
    spread = jax.random.uniform(k_fire[3], (g, g)) < p_spread
    tinder = jnp.clip(minerals @ MIN_TINDER, 0.0, 1.0) * cfg.byproducts           # dry leaves, fiber
    p_catch = jnp.minimum(cfg.ignite_prob * dryness * (1.0 + cfg.tinder_mult * tinder), 1.0)
    catch = jax.random.uniform(k_fire[4], (g, g)) < p_catch
    ignite = (fire <= 0) & (fuel_here > 0) & (
        ((spark_grid > 0) & catch) | (torch_grid > 0) | (bolt_grid > 0) | spread)
    on = (fire > 0) | ignite
    fire = jnp.where(on, fire + fuel_here, fire)                 # burning cells consume their fuel
    plants = jnp.where(on, 0.0, plants)
    if cfg.ecology:
        loose = jnp.where(on, 0.0, loose)
    wood = jnp.where(on, 0.0, wood)
    kiln = jnp.where(on, state.kiln + coal_here, state.kiln)     # coal burns hot: smelting heat while it lasts
    minerals = jnp.where((on[..., None]) & (MIN_FUEL > 0), 0.0, minerals)
    burn_rate = cfg.fire_burn_rate * jnp.where(raining, cfg.rain_burn_mult, 1.0)
    fire = jnp.maximum(fire - burn_rate, 0.0)
    kiln = jnp.where(fire > 0, jnp.maximum(kiln - burn_rate, 0.0), 0.0)
    # ---- heat changes what lies in it: clay fires in any fire; ores smelt in coal-hot fire;
    # copper and tin together become bronze
    burning, hot = fire > 0, (fire > 0) & (kiln > 0)
    made = jnp.zeros_like(minerals)                                              # what heat made, this tick
    for src, dst, needs_hot, rate in mn.FIRING:
        moved = jnp.where(hot if needs_hot else burning, jnp.minimum(minerals[..., src], rate * _k(cfg)), 0.0)
        minerals = minerals.at[..., src].add(-moved).at[..., dst].add(moved)
        made = made.at[..., dst].add(moved)
    cu, sn, bz, ratio = mn.ALLOY
    alloy = jnp.where(hot, jnp.minimum(jnp.minimum(minerals[..., cu] / ratio, minerals[..., sn]), 0.02 * _k(cfg)), 0.0)
    minerals = minerals.at[..., cu].add(-ratio * alloy).at[..., sn].add(-alloy).at[..., bz].add((ratio + 1) * alloy)
    made = made.at[..., bz].add((ratio + 1) * alloy)
    # ---- heat cooks raw meat; cooked meat left in the flames chars away
    cook = jnp.minimum(meat, cfg.cook_rate * heat0 / cfg.fire_heat)
    meat, cooked = meat - cook, cooked + cook
    cooked = cooked * (1 - jnp.where(burning0 > 0, cfg.char_rate, 0.0))
    # ---- animals caught in flames die; prey leave cooked carcasses
    def burn_animals(an, meat_amount):
        hit = an.alive & (burning0[an.pos[:, 0], an.pos[:, 1]] > 0) & ~an.below
        return an._replace(alive=an.alive & ~hit), hit, meat_amount
    small, small_burned, _ = burn_animals(small, 0.0)
    large, large_burned, _ = burn_animals(large, 0.0)
    predators, _, _ = burn_animals(predators, 0.0)
    cooked = cooked.at[small.pos[:, 0], small.pos[:, 1]].add(jnp.where(small_burned, SP["meat"][small.species], 0.0))
    cooked = cooked.at[large.pos[:, 0], large.pos[:, 1]].add(jnp.where(large_burned, SP["meat"][large.species], 0.0))
    # ---- grazing: plant-eaters eat where they stand (shared when a cell is crowded); hungry
    # predators eat from carcasses; condition (hp) follows how well each has eaten
    if cfg.grazing:
        def graze(an, rate):
            want = jnp.where(an.alive & ~an.below & SP["grazer"][an.species], rate, 0.0)
            cell = an.pos[:, 0] * g + an.pos[:, 1]
            demand = jax.ops.segment_sum(want, cell, num_segments=g * g).reshape(g, g)
            share = jnp.where(demand > 0, jnp.minimum(plants / jnp.maximum(demand, 1e-9), 1.0), 0.0)
            got = share[an.pos[:, 0], an.pos[:, 1]]
            return demand * share, got >= 0.5
        if cfg.ecology:                                       # grazers eat greens (and what lies loose)
            plants = loose + flora[..., eco.GREENS]
        eaten_s, fed_s = graze(small, cfg.graze_small)
        eaten_l, fed_l = graze(large, cfg.graze_large)
        plants = jnp.maximum(plants - eaten_s - eaten_l, 0.0)
        if cfg.ecology:
            ate = eaten_s + eaten_l
            from_loose_g = jnp.minimum(ate, loose)
            loose = loose - from_loose_g
            flora = flora.at[..., eco.GREENS].set(jnp.maximum(flora[..., eco.GREENS] - (ate - from_loose_g), 0.0))
        hungry = predators.alive & (predators.rest <= 0)
        on_carcass = hungry & ~predators.below & (meat[predators.pos[:, 0], predators.pos[:, 1]] >= cfg.predator_scavenge)
        meat = meat.at[predators.pos[:, 0], predators.pos[:, 1]].add(jnp.where(on_carcass, -cfg.predator_scavenge, 0.0))
        meat = jnp.maximum(meat, 0.0)
        predators = predators._replace(rest=jnp.where(on_carcass, SP["atk_rest"][predators.species].astype(jnp.int32),
                                                      predators.rest))
    # ---- regrowth and spoilage
    if cfg.ecology:
        weather_mult = jnp.where(weather == DROUGHT, cfg.drought_regrow_mult, 1.0) \
            * jnp.where(raining, cfg.rain_regrow_mult, 1.0) * (1.0 - snow)
        flora = eco.grow(flora, state.terrain, state.flora_patch, season, weather_mult, fire > 0, cfg)
        loose = loose * (1.0 - cfg.loose_food_spoil) ** (cfg.tick_minutes / 1440.0)
        plants = loose + flora.sum(-1)
    else:
        flora = state.flora
        plants = jnp.minimum(plants + base_rate * regrow * (fire <= 0), cfg.plant_max)
    wood = jnp.minimum(wood + jnp.where((state.terrain == FOREST) & (fire <= 0), cfg.wood_regrow, 0.0),
                       cfg.wood_max)
    if cfg.byproducts:          # leaves fall under trees (most in autumn) and rot; fowl moult feathers
        fall = cfg.leaf_fall * _k(cfg) * jnp.where(season == 2, cfg.autumn_leaf_mult, 1.0)
        minerals = minerals * (1.0 - MIN_ROT * _k(cfg))
        lv = LEAVES - MIN0
        minerals = minerals.at[..., lv].set(jnp.where((state.terrain == FOREST) & (fire <= 0),
                                                      jnp.minimum(minerals[..., lv] + fall, cfg.leaf_max), minerals[..., lv]))
        fowl = small.alive & ~small.below & (small.species == fa.IDX["fowl"]) \
            & (jax.random.uniform(jax.random.fold_in(keys[12], 41), small.alive.shape) < cfg.molt_prob * _k(cfg))
        minerals = minerals.at[small.pos[:, 0], small.pos[:, 1], FEATHER - MIN0].add(fowl.astype(jnp.float32))
    meat = meat * (1 - cfg.meat_spoil_ground * (1.0 + cfg.fly_spoil_mult * flies))     # flies speed the rot
    cooked = cooked * (1 - cfg.cooked_spoil_ground)
    if cfg.underground:                                   # the cave floor: cool, no flies
        rot = jnp.zeros(N_MAT).at[RAW_MEAT].set(cfg.meat_spoil_ground).at[COOKED_MEAT].set(cfg.cooked_spoil_ground) \
            .at[GRUBS].set(cfg.meat_spoil_ground)
        cave_floor = cave_floor * (1 - cfg.cave_spoil_mult * rot)
    spoil = jnp.zeros(N_MAT).at[RAW_MEAT].set(cfg.meat_spoil_inventory).at[COOKED_MEAT].set(cfg.cooked_spoil_inventory) \
        .at[GRUBS].set(cfg.meat_spoil_inventory)                                       # honey keeps
    # salt carried with meat slows its spoiling
    salted = 1.0 - jnp.max(jnp.where(inv[:, MIN0:] >= 0.2, jnp.array(mn.PRESERVE)[None, :], 0.0), axis=-1)   # (N,)
    inv = inv * (1 - spoil[None, :] * salted[:, None])
    # ---- a burning branch burns down and is used up
    torch_out = (torch == 1)
    inv = inv.at[idx, WOOD].add(-jnp.where(torch_out, jnp.minimum(inv[:, WOOD], 1.0), 0.0))
    torch = jnp.where(inv[:, WOOD] > 0.05, jnp.maximum(torch - 1, 0), 0)
    ignited_by = jnp.where(ignite[tr, tc] & spark, 1,
                           jnp.where(ignite[lit_cell[:, 0], lit_cell[:, 1]] & lit_drop, 2, 0))
    # animals breed with the season's growth (breed, set above)
    passage = (state.under & ~state.under_water) if cfg.underground else None
    if cfg.underground:                                   # roosting bats leave guano: saltpetre
        roosting = small.alive & small.below & (small.species == fa.IDX["bat"])
        cave_floor = cave_floor.at[small.pos[:, 0], small.pos[:, 1], MIN0 + mn.IDX["saltpetre"]].add(
            jnp.where(roosting, cfg.guano_rate, 0.0))
    if cfg.grazing:
        # condition: grazers that ate (and the rest: fish, bats) mend; grazers that found too little, and
        # hunters long unfed, weaken and may die of it. Births follow how well fed the living are.
        def condition(an, fed, starve):
            top = SP["hp"][an.species]
            hp = jnp.where(fed, jnp.minimum(an.hp + cfg.animal_hp_regen, top), an.hp - starve * top)
            return an._replace(hp=hp, alive=an.alive & (hp > 0))
        def thriving(an):                                  # share of the slots alive and in good condition
            return jnp.mean((an.alive & (an.hp >= 0.8 * SP["hp"][an.species])).astype(jnp.float32))
        small = condition(small, fed_s | ~SP["grazer"][small.species] | small.below, cfg.graze_starve)
        large = condition(large, fed_l | ~SP["grazer"][large.species] | large.below, cfg.graze_starve)
        sleeping_off = (SP["hibernates"][predators.species] > 0) & (season == 3)    # hibernating: lives on its fat
        predators = condition(predators, (predators.rest > 0) | sleeping_off, cfg.predator_starve)
        food_s = cfg.immigration + 2.0 * thriving(small)
        food_l = cfg.immigration + 2.0 * thriving(large)
        prey_about = jnp.mean(jnp.concatenate([small.alive, large.alive]).astype(jnp.float32))
        food_p = cfg.immigration + 2.0 * prey_about
    else:
        food_s = food_l = food_p = 1.0
    small = _respawn(keys[13], small, state.terrain, cfg.small_prey_respawn * breed * food_s, cfg, SP["hp"], passage)
    large = _respawn(keys[14], large, state.terrain, cfg.large_prey_respawn * breed * food_l, cfg, SP["hp"], passage)
    predators = _respawn(keys[15], predators, state.terrain, cfg.predator_respawn * food_p, cfg, SP["hp"], passage)
    if not cfg.grazing:
        small = small._replace(hp=jnp.minimum(small.hp + cfg.animal_hp_regen, SP["hp"][small.species]))
        large = large._replace(hp=jnp.minimum(large.hp + cfg.animal_hp_regen, SP["hp"][large.species]))
        predators = predators._replace(hp=jnp.minimum(predators.hp + cfg.animal_hp_regen, SP["hp"][predators.species]))
    # ---- bees refill their hives and termites their mounds in the growing seasons
    grow = jnp.array([1.0, 1.0, 0.5, 0.0])[season]
    honey = jnp.where(state.hive, jnp.minimum(honey + cfg.honey_regrow * grow, cfg.honey_max), honey)
    grubs = jnp.where(state.mound, jnp.minimum(grubs + cfg.grubs_regrow * grow, cfg.grubs_max), grubs)

    # ---------------------------------------------------------- cognitive map
    map_sight = jnp.where(sleeping, 0, jnp.minimum(jnp.where(night, dark_radius(state, cfg), cfg.view_radius),
                                                   sight_radius(state, cfg)))
    deep_sleep = sleeping & (sleep_depth >= cfg.sleep_deep_ticks)              # sleep consolidates the day
    # Camp fouling: waste builds where people are and fades with time, faster in rain
    if cfg.fouling:
        days = cfg.tick_minutes / 1440.0
        fade = 0.5 ** (days / cfg.foul_half_life_days * jnp.where(raining, cfg.foul_rain_mult, 1.0))
        if cfg.excretion:                             # waste where each relieved itself (not below ground)
            left = jnp.where(below_after, 0.0, relieved * cfg.waste_per_urge)
            here_now = jnp.zeros((g, g)).at[pos_after[:, 0], pos_after[:, 1]].add(left)
            foul = state.foul * fade + here_now / eco.cell_km2(cfg)
        else:                                         # people's refuse wherever they are
            here_now = _occupancy(pos_after, alive & ~below_after, g).astype(jnp.float32)
            foul = state.foul * fade + here_now * days / eco.cell_km2(cfg)
    else:
        foul = state.foul
    forage_now = forage_grid(state._replace(plants=plants, flora=flora), cfg)
    heard_map = state.mind_map
    if cfg.language_mode == "grounded":
        # what was said this tick is understood by those who heard it (as heard_messages: within
        # view, on one's layer, the closest only when asleep); what is written, by those beside it
        lp_s = layered_pos(state.pos, below0)
        d_s = _cheb(lp_s[:, None, :], lp_s[None, :, :])
        hear_r = jnp.where(state.asleep, 1, cfg.view_radius)
        heard_now = spoke_now[None, :] & ~jnp.eye(n, dtype=bool) & (d_s <= hear_r[:, None]) & alive0[:, None]
        ch, val, place = phrase_place(heard_msg, state.pos, cfg)
        heard_map = comprehend(heard_map, heard_now, ch, val, place, cfg)
        can_read, at = mark_beside(state, cfg)
        ch_r, val_r, place_r = phrase_place(state.marks[at[:, 0], at[:, 1]], at, cfg)
        heard_map = comprehend(heard_map, jnp.eye(n, dtype=bool) & can_read[:, None], ch_r, val_r, place_r, cfg)
    mind_map = update_mind_map(heard_map, pos_after, alive & ~below_after, state.terrain, forage_now, map_sight, cfg,
                               consolidate=deep_sleep)

    # ---------------------------------------------------------- novelty
    new_cell = alive & ~state.visited[idx, pos_after[:, 0], pos_after[:, 1]]
    visited = state.visited.at[idx, pos_after[:, 0], pos_after[:, 1]].set(
        state.visited[idx, pos_after[:, 0], pos_after[:, 1]] | alive)

    # ---------------------------------------------------------- moral sentiments
    # Witnessing uses sight: the full view by day, the short night radius in the dark.
    sight = jnp.minimum(jnp.where(night, dark_radius(state, cfg), cfg.view_radius), sight_radius(state, cfg))
    sight = jnp.where(below_after, jnp.minimum(light0, sight), sight)
    sight = jnp.where(sleeping, 1, sight)
    vis = (_cheb(lpa[:, None, :], lpa[None, :, :]) <= sight[:, None]) & not_self \
        & alive0[:, None] & alive0[None, :]
    visf = vis.astype(jnp.float32)
    n_vis = jnp.maximum(visf.sum(1), 1.0)
    # How hurt someone looks: the health they lose, as loudly as they feel it (a wimp cries out)
    shows = jnp.sqrt(state.pain_aversion)
    seen_hurt = health_loss * shows
    others_harm = visf @ seen_hurt
    similarity = jnp.exp(-3.0 * jnp.linalg.norm(state.appearance[:, None, :] - state.appearance[None, :, :], axis=-1))
    kin_harm = (visf * similarity) @ seen_hurt
    gap = satiety[None, :] - satiety[:, None]                           # [i, j] = j's minus i's
    worse_off = (visf * jnp.maximum(gap, 0.0)).sum(1) / n_vis
    better_off = (visf * jnp.maximum(-gap, 0.0)).sum(1) / n_vis
    # Habituation: each repetition with the same partner feels e^-h as good, where h rises by one
    # per felt act and halves every habituation_half_life ticks. A nonstop burst is worth at most
    # 1 / (1 - e^-1) ~ 1.6 single acts, so repeating an act cannot be farmed, yet nothing forbids it.
    habit = state.habituation * 0.5 ** (1.0 / cfg.habituation_half_life)
    fresh = jnp.exp(-habit)                                             # (N, N, 3)
    repay = (gifts & (state.memory[:, :, MEM_GAVE_NEAR_ME] > 0.5)).astype(jnp.float32)
    revenge = (hits & (state.memory[:, :, MEM_STRUCK_ME] > 0.5)).astype(jnp.float32)
    contact = (touches | touches.T).astype(jnp.float32)                 # touched or was touched by j
    # Cruelty: the hurt one's own blows cause, as it shows on the one struck (sadism); satisfaction
    # in striking habituates as one whole, revenge or cruelty
    inflicted = hits.astype(jnp.float32) * (violence * shows)[None, :]
    sadistic = (inflicted * fresh[:, :, HAB_REVENGE]).sum(1)
    felt = jnp.stack([repay, contact, revenge, intimate], axis=-1)      # (N, N, 4): acts this tick
    feeling = (felt * fresh).sum(axis=1)                                # (N, 3): how good they felt
    cruel = (geno.sadism[:, None] > 0.05) & hits                        # cruelty wears off like revenge does
    habituation = habit + felt.at[:, :, HAB_REVENGE].max(cruel.astype(jnp.float32))
    reciprocated = feeling[:, HAB_REPAY]
    retaliated = feeling[:, HAB_REVENGE]
    moral = (- geno.empathy * cfg.empathy_weight * others_harm
             - geno.kin_care * cfg.kin_weight * kin_harm
             - cfg.inequity_weight * (geno.envy * worse_off + geno.guilt * better_off)
             + cfg.reciprocity_reward * geno.reciprocity * reciprocated
             + cfg.revenge_reward * geno.vengefulness * retaliated
             + cfg.sadism_reward * geno.sadism * sadistic)
    moral = jnp.where(alive0, moral, 0.0)
    # Heartache: others' hurt imagined as one's own (x empathy), and grief for the attached who die
    grief = state.attachment @ dead_now.astype(jnp.float32)
    others_wounded = visf @ (wound * shows)                             # pain seen: wounds, not hunger or illness
    heartache = state.heartache * (1.0 - cfg.heartache_fade) ** _k(cfg) \
        + cfg.heartache_empathy * geno.empathy * others_wounded + cfg.heartache_grief * grief
    heartache = jnp.where(alive0, jnp.clip(heartache, 0.0, 2.0), 0.0)

    # ---------------------------------------------------------- conditions: onset, events, use, fading
    kc = jax.random.split(jax.random.fold_in(keys[0], 91), 4)
    roll = jax.random.uniform(kc[0], (n, cond.K))
    years_now = (age + state.birth_age) / cfg.ticks_per_bio_year
    gain_c = jnp.zeros((n, cond.K), bool)
    for name, kind, good, base, inherit, onset, fade, about in cond.CONDITIONS:
        i_c = cond.IDX[name]
        if kind == "age":
            per_tick = base / cfg.ticks_per_bio_year
            gain_c = gain_c.at[:, i_c].set((years_now >= onset) & (roll[:, i_c] < per_tick))
    life_moves = state.life_moves + moving.astype(jnp.int32)
    life_strikes = state.life_strikes + is_strike.astype(jnp.int32)
    hardship = jnp.where(critical > 0, state.hardship + 1, 0)
    near_death = alive & (health < 0.15) & (state.health >= 0.15)
    gain_c = gain_c.at[:, cond.IDX["lame"]].set(lasting_injury > 0.15)
    gain_c = gain_c.at[:, cond.IDX["chronic cough"]].set(recovered & (roll[:, cond.IDX["chronic cough"]] < 0.1))
    gain_c = gain_c.at[:, cond.IDX["haunted"]].set(near_death & (roll[:, cond.IDX["haunted"]] < 0.3))
    gain_c = gain_c.at[:, cond.IDX["hardened"]].set((state.hardship >= cfg.day_length) & (critical == 0) & alive
                                                     & (roll[:, cond.IDX["hardened"]] < 0.2))
    gain_c = gain_c.at[:, cond.IDX["enduring"]].set(life_moves >= _ticks(cfg, 1500))
    gain_c = gain_c.at[:, cond.IDX["heavy-handed"]].set(life_strikes >= 200)
    had = state.conditions > 0.5
    new_cond = gain_c & ~had & alive[:, None]
    fade_p = jnp.array([c[6] for c in cond.CONDITIONS]) / cfg.day_length
    healed_c = had & (jax.random.uniform(kc[1], (n, cond.K)) < fade_p[None, :])
    conditions = jnp.where(alive[:, None], ((had | new_cond) & ~healed_c).astype(jnp.float32), state.conditions)

    # ---------------------------------------------------------- attachment
    # Evidence that j keeps i alive, from this tick (all [i, j]):
    adj = (_cheb(lpa[:, None, :], lpa[None, :, :]) <= 1) & ~jnp.eye(n, dtype=bool) \
        & alive0[:, None] & alive0[None, :]
    held_off = jnp.zeros(n).at[target].add(deterred.astype(jnp.float32))           # attacks held off, per agent
    R = state.relatedness
    shares_child = (R >= 0.45) & alive0[None, :]
    coparent = ((shares_child.astype(jnp.float32) @ shares_child.T.astype(jnp.float32)) > 0) \
        & (R < 0.2) & ~jnp.eye(n, dtype=bool)
    mom = jnp.clip(state.mother, 0)
    nursed_by = jnp.zeros((n, n)).at[jnp.arange(n), mom].set(nursing.astype(jnp.float32))
    gain = (cfg.attach_food * gifts.T.astype(jnp.float32) * (state.satiety < cfg.attach_hungry)[:, None]   # fed when it mattered
            + cfg.attach_warmth * (adj & (body_temp < -0.2)[:, None])
            + cfg.attach_protect * held_off[:, None] * adj
            + cfg.attach_company * in_view
            + cfg.attach_intimacy * intimate
            + cfg.attach_coparent * coparent
            + cfg.attach_nurse * nursed_by)
    # Fit: temperament similarity and the other's fitness scale how fast the evidence counts
    temper = jnp.stack([geno.empathy, geno.vengefulness, geno.reciprocity, geno.kin_care,
                        jnp.log(jnp.maximum(geno.social_reward, 0.05)),       # floor: a zero drive must not give -inf
                        jnp.log(jnp.maximum(geno.novelty, 0.05))], axis=-1)
    similar = jnp.exp(-1.5 * jnp.linalg.norm(temper[:, None, :] - temper[None, :, :], axis=-1))   # 1 = alike
    fit = 0.5 * health * jnp.clip(weight / bod["ideal_kg"], 0.0, 1.0) + 0.5 * jnp.clip(bod["strength"] / 1.5, 0.0, 1.0)
    compat = (1 - cfg.attach_similarity + cfg.attach_similarity * similar) \
        * (1 - cfg.attach_fitness + cfg.attach_fitness * fit[None, :])
    # Danger faced together, and teamwork against the same animal
    both_threatened = pred_near[:, None] & pred_near[None, :] & in_view & in_view.T
    hunter = hit_pred | hit_large | hit_small
    teamed = hunter[:, None] & hunter[None, :] & jnp.all(tgt[:, None, :] == tgt[None, :, :], axis=-1) \
        & ~jnp.eye(n, dtype=bool)
    gain = gain * compat + cfg.attach_shared_threat * both_threatened + cfg.attach_synergy * teamed
    attachment = state.attachment + jnp.clip(gain, 0.0, 1.0) * (1.0 - state.attachment)
    attachment = jnp.where(hits.T, attachment * (1.0 - cfg.attach_harm), attachment)       # struck by them
    attachment = attachment * jnp.where(in_view, 1.0, 0.5 ** (1.0 / cfg.attach_half_life))
    attachment = jnp.where(alive0[:, None], attachment, state.attachment) * ~jnp.eye(n, dtype=bool)

    # ---------------------------------------------------------- impressions
    # i sees j (alive or a body) within the view square; at night only nearby or fire-lit cells
    lit = fire_heat(fire, cfg) > 0
    d_ij = _cheb(lpa[:, None, :], lpa[None, :, :])
    lighting = ~night | (d_ij <= dark_radius(state, cfg)[:, None]) | lit[pos_after[None, :, 0], pos_after[None, :, 1]]
    lighting = jnp.where(below_after[:, None], d_ij <= light0[:, None], lighting)   # below: only one's own light
    sees = alive0[:, None] & ~jnp.eye(n, dtype=bool) & (d_ij <= sight_radius(state, cfg)[:, None]) \
        & ~(sleeping[:, None] & (d_ij > 1)) & state.existed[None, :] & lighting
    tol = shock_threshold(state._replace(pain_aversion=pain_aversion), cfg) if cfg.pain_shock > 0 \
        else 0.8 / jnp.sqrt(pain_aversion)
    wince_now = jnp.clip((cfg.pain_per_injury * injury + pain) / tol, 0.0, 1.0)
    looks = visible_condition(health, injury, (infection > 0).astype(jnp.float32), weight, bod["ideal_kg"],
                              scars, cfg, wince_now)
    looks = jnp.where(alive, looks, -cfg.death_penalty)               # a body: seen dead
    new_impression = jnp.where(sees, looks[None, :], state.impression)

    # ---------------------------------------------------------- reward
    if cfg.reward_mode == "survival":
        reward, moral = _survival_reward(state, cfg, geno, alive0, alive, dead_now, similarity,
                                         health, satiety, hydration, energy, body_temp, stress, pain_aversion,
                                         state.impression, new_impression, looks, sleepiness)
    else:
        reward = (jnp.where(alive, cfg.alive_reward, 0.0)
                  - jnp.where(dead_now, cfg.death_penalty, 0.0)
                  - state.pain_aversion * health_loss
                  + jnp.where(new_cell, cfg.novelty_reward * geno.novelty, 0.0)
                  + cfg.touch_reward * geno.social_reward * feeling[:, HAB_TOUCH]
                  + moral)
    # Survival of one's genes: relatives' survival counts, weighted by relatedness (true kinship,
    # seen or not). Mating is never rewarded in itself; children count once they are alive.
    if cfg.reproduction and cfg.reward_mode == "survival":
        signal = jnp.where(alive0, jnp.where(alive, cfg.alive_reward, 0.0) - jnp.where(dead_now, cfg.death_penalty, 0.0), 0.0)
        genetic = jnp.where(alive0, (state.relatedness * ~jnp.eye(n, dtype=bool)) @ signal, 0.0)
        reward = reward + genetic
    else:
        genetic = jnp.zeros(n)

    # ---------------------------------------------------------- outcomes
    any_kill = jnp.any(kills, axis=-1)
    # Waking: a sleeper is woken by being struck, attacked, burned or touched, or wakes rested
    disturbed = struck_by_agent | attacked | burned | jnp.any(touches, axis=0) | fell | (soiled & ~no_control)
    asleep = alive & sleeping & ~disturbed & (sleepiness > 0.0)
    collapsed = asleep & (collapse | state.collapsed)
    sleep_depth = jnp.where(asleep, sleep_depth, 0)

    outcome = jnp.select(
        [~alive0, sleeping, relieving, writing, resting, is_move & move_fail, is_move, winner, gathering, eating, drinking,
         any_kill, hit_agent, hit_pred | hit_large | hit_small, is_strike,
         releasing, is_touch, is_select, is_place, is_mate],
        [OUT_NONE, OUT_SLEEP, OUT_RELIEVE, OUT_WRITE, OUT_REST, OUT_MOVE_FAILED, OUT_MOVE, OUT_GATHER, OUT_GATHER_LOST, OUT_EAT,
         OUT_DRINK, OUT_KILL, OUT_STRIKE_AGENT, OUT_STRIKE_ANIMAL, OUT_STRIKE_MISS,
         OUT_RELEASE, OUT_TOUCH, OUT_SELECT, OUT_PLACE, OUT_MATE], OUT_NONE)

    def keep(new, old):
        return jnp.where(alive0, new, old)

    t = state.t + 1
    next_state = State(
        key=key, t=t, terrain=state.terrain, height=state.height, cliff=state.cliff, plants=plants, meat=meat,
        cooked=cooked, wood=wood,
        stones=stones, minerals=minerals, kiln=kiln, fire=fire, mutations=state.mutations,
        honey=honey, grubs=grubs, hive=state.hive, mound=state.mound,
        weather=weather, weather_timer=timer, pos=pos_after,
        clouds=clouds, snow=snow, wind=wind, climate=climate_drift,
        satiety=keep(satiety, state.satiety), hydration=keep(hydration, state.hydration),
        energy=keep(energy, state.energy), health=health,
        inv=inv, hand=jnp.where(alive0, hand, state.hand), torch=torch,
        age=age, alive=alive, visited=visited, mind_map=mind_map,
        body_temp=keep(body_temp, state.body_temp), injury=keep(injury, state.injury),
        infection=jnp.where(alive0, infection, state.infection),
        immunity=keep(immunity, state.immunity), stress=keep(stress, state.stress), clothing=state.clothing,
        sleepiness=sleepiness, asleep=asleep, sleep_depth=sleep_depth.astype(jnp.int32), collapsed=collapsed,
        pain_aversion=keep(pain_aversion, state.pain_aversion), genotype=geno,
        weight=keep(weight, state.weight), birth_age=state.birth_age,
        drop_owner=drop_owner, drop_time=drop_time, habituation=habituation, impression=new_impression,
        attachment=attachment, conditions=conditions, father_cond=father_cond,
        life_moves=keep(life_moves, state.life_moves), life_strikes=keep(life_strikes, state.life_strikes),
        hardship=keep(hardship, state.hardship),
        scars=scars, lasting_injury=lasting_injury, pain=keep(pain, state.pain), heartache=keep(heartache, state.heartache),
        part_damage=jnp.where(alive0[:, None], part_damage, state.part_damage),
        held_by=jnp.where(alive0, tech["held_by"], -1), hold_grip=jnp.where(alive0, tech["hold_grip"], 0.0),
        off_balance=jnp.where(alive0, tech["off_balance"], 0),
        sex=state.sex, existed=state.existed, gestation=gestation, courting=courting, father=father, mother=state.mother,
        relatedness=state.relatedness, fetus_r=fetus_r, father_geno=father_geno, father_look=father_look,
        appearance=state.appearance, memory=memory, last_msg=heard_msg,
        under=state.under, under_water=state.under_water, under_kind=state.under_kind, spring=state.spring,
        cave_floor=cave_floor,
        below=jnp.where(alive0, below_after, state.below),
        small=small, large=large, predators=predators,
        flora=flora, flora_patch=state.flora_patch, foul=foul, urge=keep(urge, state.urge),
        **_marks_with_surface(marks, mark_time, mark_kind, wood, stones, cfg),
    )
    if cfg.reproduction:
        next_state, born = _give_birth(next_state, birthing, jax.random.fold_in(keys[0], 78), cfg)
        alive = next_state.alive
    else:
        born = jnp.zeros(n, bool)
    done = (~jnp.any(alive)) | (t >= cfg.max_steps)
    info = StepInfo(
        outcome=outcome, died=dead_now, death_cause=cause,
        gathered=take, health_loss=health_loss, pos=pos_after, t=state.t,
        attacked_by_predator=attacked, struck_by_agent=struck_by_agent,
        newly_infected=newly_infected, recovered=recovered, fell=fell, kills=kills,
        weather=weather, weather_changed=weather_changed, ambient_temp=amb,
        hits=hits, gifts=gifts, touches=touches, message=msg, spoke=spoke_now,
        ignited=ignited_by.astype(jnp.int32), lightning=bolt & ((bolt_grid * ignite).sum() > 0),
        burning_cells=(fire > 0).sum().astype(jnp.int32),
        near_fire=alive0 & (heat0[pos_after[:, 0], pos_after[:, 1]] > 0) & ~below_after,
        fed_fire=fed_fire, meat_to_heat=meat_to_heat, ate_cooked=ate_cooked,
        torch_lit=pull_branch, burned=burned, moral_reward=moral,
        deterred=jnp.zeros(n).at[target].add(deterred.astype(jnp.float32)) > 0,
        huddling=alive0 & (huddle > 0), winter_night=(season == 3) & night,
        company=alive0 & jnp.any((_cheb(lpa[:, None, :], lpa[None, :, :]) <= cfg.view_radius)
                                 & ~jnp.eye(n, dtype=bool) & alive0[None, :], axis=1),
        meat_eaten=(eaten > 0) & ((hand0 == RAW_MEAT) | (hand0 == COOKED_MEAT)),
        large_prey_hit=hit_large, slept=asleep & alive0, woken=sleeping & alive0 & disturbed,
        fell_asleep=sleeping & alive0 & ~state.asleep,
        season=season, new_conditions=new_cond, made=made, bound=bound, bedded=bedded, kill_species=kill_sp, attacker_species=attacker_sp,
        gored=gored_dmg, gored_by=gored_by, stung=stung & alive0, venomed=venomed,
        mosquitoes=mosq[pos_after[:, 0], pos_after[:, 1]] * ~below_after,
        small_preyed_by=small_preyed_by, large_preyed_by=large_preyed_by,
        small_species=state.small.species, large_species=state.large.species,
        small_pos=state.small.pos, large_pos=state.large.pos,
        skill=jnp.where(alive0 & (chosen >= SEEK_WATER), chosen - SEEK_WATER + 1, 0).astype(jnp.int32), collapsed_now=collapse & ~state.asleep, mated=mated, conceived=conceive, no_room=no_room, gave_birth=birthing, born=born,
        nursed=nursing, genetic_reward=genetic, below=next_state.below, delved=delving, dark=dark_now > 0,
        stress_from=stress_src, downed=downed_of(next_state, cfg), relieved=relieving & alive0, soiled=soiled,
        gathered_kind=took_kind if cfg.ecology else jnp.full(n, -1, jnp.int32),
        aim=jnp.where(hit_agent | hit_pred | hit_large | hit_small, aim, -1), blow_landed=blow_landed & hit_agent,
        how=jnp.where(is_strike, how, -1), shoved=tech["shoved"], thrown=tech["thrown"], grabbed=tech["grabbed"],
        pinned=pinned, same_sex_mated=same_sex_mated)
    return next_state, reward, done, info
