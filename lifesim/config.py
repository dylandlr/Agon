"""Static configuration for the world. All fields are hashable so the config
can be passed as a static argument to jitted functions. Every mechanic is a
dial here, so experiments change config, never code."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    # ------------------------------------------------------------ world
    grid_size: int = 64
    n_agents: int = 1
    max_steps: int = 1920          # episode time limit (ticks): one year
    # Calendar. One tick = 30 minutes. By default a year is compressed to 40 days
    # (seasons of 10 days) so winter arrives within a learnable horizon; set
    # days_per_year=365 for a real year (seasons of 91.25 days, 17,520 ticks).
    # Per-tick physiology, weather and growth rates do not change with it.
    day_length: int = 48           # ticks per day (24 h)
    # Minutes of game time per tick. Every rate below is calibrated at 30; with_tick_minutes()
    # rescales them all for another tick length (1 = RimWorld-like: a day of 1,440 decisions).
    tick_minutes: float = 30.0
    days_per_year: int = 40
    daylight_swing: float = 3.0    # hours: days are 12 +/- this (15 h summer, 9 h winter)
    solstice_frac: float = 0.307   # longest day, as a fraction of the year from spring's start (21 Jun from 1 Mar)
    founder_spread: int = 3        # founders start within this many cells of one point (a band); 0 = scattered
    camp_water_radius: int = 4     # the band's camp is within this many cells of drinkable water; -1 = anywhere
    warmest_hour: float = 15.0     # daily temperature peaks mid-afternoon, lowest 12 h earlier
    night_view_radius: int = 2     # at night agents see only this far, except where fire lights

    # Terrain generation (fractions are approximate)
    water_fraction: float = 0.08
    rock_fraction: float = 0.10
    forest_fraction: float = 0.30  # remainder is grassland
    # Falls. Rock cells have a cliff height: how far they drop to the lowest neighboring ground
    # (0 on a plateau, 1 on the steepest edges). Moving onto rock risks a fall, more so at a high
    # edge, when already injured, or in the dark. Severity = (0.2 + cliff) x fall_severity x an
    # exponential draw: most falls are minor, some injure for weeks or for life (injury above 0.5
    # leaves a lasting part), and a few from high cliffs kill outright (health lost >= 1).
    fall_prob: float = 0.02        # per move, x (0.2 on rock + cliff) x (1 + injury) x 2 at night
    fall_severity: float = 0.35    # health lost per unit of (0.2 + cliff) x exponential draw
    fall_injury_ratio: float = 1.5 # injury added per unit of health lost in a fall
    # Height (2.5D). Every cell has a height in [0, 1] from the elevation map that placed water
    # (lowest) and rock (highest). Climbing costs energy, high ground is colder, and cliff is each
    # cell's drop to its lowest neighbor anywhere (steep banks and slopes too, not only rock).
    climb_cost: float = 2.0        # move energy x (1 + climb_cost x rise / steepest drop on the map)
    altitude_cold: float = 6.0     # degrees C colder at the highest point than at the lowest

    # ------------------------------------------------------------ plants
    plant_max: float = 3.0
    plant_abundance: float = 1.0       # scales BOTH the starting stock and regrowth: the scarcity dial
    plant_regrow_forest: float = 0.020
    plant_regrow_grass: float = 0.008
    winter_regrow_factor: float = 0.1
    gather_amount: float = 1.0

    # ------------------------------------------------------------ weather
    # Markov weather: clear, rain, drought, cold snap, heat wave.
    climate_volatility: float = 0.3   # chance a new weather spell is an extreme event
    spell_min: int = 60
    spell_max: int = 300
    rain_regrow_mult: float = 1.5
    drought_regrow_mult: float = 0.3
    drought_thirst_mult: float = 1.5
    extreme_temp_shift: float = 10.0  # degrees for cold snaps and heat waves

    # ------------------------------------------------------------ temperature
    season_temp: tuple = (12.0, 25.0, 12.0, 2.0)   # spring, summer, autumn, winter (C)
    day_temp_swing: float = 6.0
    forest_buffer: float = 0.5      # forest pulls ambient 50% toward comfort
    comfort_temp: float = 18.0
    temp_scale: float = 20.0        # degrees away from comfort for full thermal load
    thermal_rate: float = 0.03      # body temp relaxation rate (divided by insulation)
    rest_warmth: float = 0.15       # resting curled up shifts thermal load warmer (only when the air is cold)
    huddle_warmth: float = 0.05     # per adjacent living agent, up to 3
    cold_threshold: float = -0.7    # body-temp state below this causes damage
    heat_threshold: float = 0.85    # damage from about 35 C held (was 0.5 = 28 C, which killed resting
                                    # people in shade in an ordinary summer; humans sweat)
    cold_damage: float = 0.006
    heat_damage: float = 0.004
    sweat_thirst_mult: float = 1.0  # extra thirst per unit body temp above 0.2

    # ------------------------------------------------------------ body
    hunger_rate: float = 0.004
    thirst_rate: float = 0.006
    fatigue_rate: float = 0.002
    move_cost: float = 0.002
    rest_recovery: float = 0.02
    night_rest_bonus: float = 1.5
    plant_food_value: float = 0.35
    meat_food_value: float = 0.8    # meat is denser food
    drink_value: float = 0.5
    starve_damage: float = 0.01
    thirst_damage: float = 0.015
    exhaust_damage: float = 0.005
    heal_rate: float = 0.002
    inventory_cap: float = 5.0
    max_age: int = 20000
    senescence_start: float = 0.6   # (unused since the body model; kept for old configs)

    # Body. Biological age runs on its own clock: max_age ticks = max_bio_years (a scale, not a
    # limit: see senescence_rate),
    # so one biological year is ~286 ticks (~6 simulated days) by default.
    max_bio_years: float = 70.0     # the most likely age of death from senescence (not a limit)
    # Senescence: a Gompertz hazard b exp(b (age - max_bio_years)) a biological year. b 0.085 doubles
    # the risk about every 8 years, the range fitted to hunter-gatherers (Gurven & Kaplan 2007, whose
    # modal age of adult death is about 72). From senescence alone about 1 in 800 die a year at 20,
    # 1 in 100 at 45, 1 in 12 at 70; about 84% of 15-year-olds reach 50, and half live past 66.
    senescence_rate: float = 0.085
    # Days per biological year with reproduction (with_reproduction): 1 compresses a life into
    # about 70 days so generations fit in a run; 365 is real time (9-month pregnancies, adults at 15
    # calendar years). With days_per_year this sets the whole time scale (see with_time_scale).
    days_per_bio_year: float = 1.0
    founder_age_range: tuple = (18.0, 35.0)   # biological years at the start of a run
    maturity_years: float = 18.0

    # Reproduction (off by default; see with_reproduction()). n_agents is the number of slots;
    # n_founders are alive at the start and births fill empty or dead slots.
    reproduction: bool = False
    n_founders: int = 0                # 0: every slot is a founder (no births possible)
    adult_years: float = 15.0          # fertile from this biological age
    menopause_years: float = 45.0
    conception_prob: float = 0.15      # per mutual mating between fertile adults (with_reproduction sets it from the two below)
    # Fecundity under compressed biological time. A fertile couple mating regularly conceives within
    # a biological year with probability conception_per_bio_year (about 0.2 per menstrual cycle gives
    # ~0.9 a year; 0.8 allows for cycles missed). An act of mating in game time stands for
    # 1 / (mating_acts_per_day x days_per_bio_year) of a biological year, so at 1 day per biological
    # year one act carries most of a year's chance (0.8), and in real time (365 days per year) about
    # 0.4% an act. Before this, 0.15 an act held at every compression: at 1 day per year a woman had
    # one or two chances per "year", and trained bands bore too few children to outlive their founders.
    conception_per_bio_year: float = 0.8
    mating_acts_per_day: float = 1.0
    mate_window: int = 4               # ticks a choice to mate stays open for a partner (1: same tick only)
    min_condition: float = 0.85        # a woman thinner than this share of ideal weight does not conceive
    gestation_years: float = 0.75
    pregnancy_need: float = 0.3        # extra food and water while pregnant (share of normal)
    maternal_death: float = 0.01       # risk of dying in childbirth, doubled if health < 0.5
    infant_years: float = 1.0          # cannot walk or forage
    mutation_rate: float = 0.01        # chance per trait per birth of a de novo mutation
    mutation_size: float = 2.5         # its size, in population standard deviations (up or down)
    nurse_years: float = 2.5           # mother nurses within 1 cell until this age (foragers wean at ~2.5-3)
    wet_nursing: bool = True           # an orphan is nursed by the closest kin among women nursing their own
    carry_infants: bool = True         # an infant within 1 cell of its carer is carried as she moves
    carry_cost: float = 1.0            # her steps cost x (1 + carry_cost x infant's share of her weight)
    # Children need less: food and water needs scale with body mass to the 3/4 power (Kleiber's law),
    # so a toddler needs a bit over half of what an adult does (on this body model's sizes) and an
    # adult is unchanged. Off, every age needs an adult's share.
    needs_by_size: bool = True
    # ... and with each person's own frame (height and build genes: lean mass^0.75 against an average
    # adult's), and with what they do this tick: physical activity levels, as multiples of resting
    # needs (sleep a little below, walking well above, fighting most). activity_norm divides them so
    # that a typical forager's day (about 40% asleep, 30% resting, 25% walking, 5% working or
    # fighting) costs what it did before. Hunger and thirst alike (exertion sweats).
    activity_needs: bool = True
    activity_sleep: float = 0.9
    activity_rest: float = 1.0
    activity_walk: float = 1.6
    activity_work: float = 1.3      # gathering, placing, writing
    activity_strike: float = 2.0
    activity_norm: float = 1.13
    # A blow's force: strength x (lean mass / an average adult's)^(2/3) (muscle force follows its
    # cross-section), x practice (up to +practice_max, half of it after practice_half strikes in a
    # life: technique), x luck (lognormal, sd blow_luck; a glancing_prob chance of half force).
    blow_by_mass: bool = True
    practice_max: float = 0.25
    practice_half: float = 200.0
    blow_luck: float = 0.25
    glancing_prob: float = 0.1
    # Techniques (env.LIMBS x env.MODES; needs body_parts for push and hold): no named moves, only
    # what each limb and mode does. Per limb (hand, foot, head, body):
    techniques: bool = True
    limb_force: tuple = (1.0, 1.5, 1.2, 0.7)      # a kick ~1.5x a punch's peak force; body also x mass^(1/3)
    limb_accuracy: tuple = (1.0, 0.8, 0.85, 1.15) # x the part's chance to land
    kick_high_accuracy: float = 0.7               # x again for a kick at a head or neck
    kick_balance: float = 0.1                     # a kick's chance to leave the kicker staggered (x (1 + leg damage))
    headbutt_recoil: float = 0.3                  # share of a landed headbutt's blow one's own head takes
    limb_push: tuple = (0.8, 1.0, 0.5, 1.4)       # leverage in a push (a shoulder or hip shoves best)
    limb_grip: tuple = (1.0, 0.25, 0.0, 0.6)      # leverage in a hold (hands grip; a head cannot)
    # Per mode (strike, push, hold): the share of the blow's damage dealt
    mode_damage: tuple = (1.0, 0.2, 0.1)
    # A push or hold is a contest: leverage x strength x lean mass^(2/3) x sound legs against the
    # other's; someone down or staggered resists at unsteady_resist. A shove moves one a cell (into
    # rough or steep ground: a fall, shove_fall_mult x that cell's hazard) and staggers one for
    # shove_ticks; a push on someone one held last tick is a throw (throw_leverage x the chance):
    # throw_damage x the thrower's force to the torso, and throw_ticks on the ground. A hold pins:
    # next tick one steps only past its grip (the contest's chance), and blows land as on the downed
    # in that proportion.
    unsteady_resist: float = 0.3
    shove_fall_mult: float = 3.0
    shove_ticks: int = 1
    throw_leverage: float = 1.5
    throw_damage: float = 0.15
    throw_ticks: int = 2
    lactation_infertile_years: float = 2.0   # a mother cannot conceive while her child is younger (0: off)
    nurse_rate: float = 0.03           # food and water a nursing child can take per tick
    nurse_cost: float = 0.6            # share of what the child takes that the mother pays
    prime_until_years: float = 35.0
    base_height_cm: float = 165.0
    ideal_bmi: float = 22.0
    fat_gain_kg: float = 0.01        # per tick when well fed, up to 115% of ideal weight
    # Catch-up growth: a child or youth below its frame's lean weight (stunted by hunger earlier)
    # regrows toward it while fed, closing the gap with this time constant in biological years
    # (catch-up growth after malnutrition is real, if often incomplete). Without it, a child hungry
    # for a while stayed thin for life and the next generation could not conceive.
    catch_up_years: float = 1.0
    catch_up_until_years: float = 20.0
    fat_burn_kg: float = 0.08        # per tick when starving, down to 80% of ideal weight
    satiety_per_kg: float = 0.05     # burning 1 kg of reserves restores this much satiety

    # Injury
    injury_heal_rate: float = 0.001
    injury_move_fail: float = 0.5   # move fails with prob injury * this
    injury_bleed: float = 0.003     # health loss per tick at full injury above 0.5
    # Lasting marks. Scars form from wounds (blows, predators, burns, falls) and never fade; they
    # do no harm but others can see them. Part of a severe injury never heals: injury heals down
    # to a lasting floor, so weaker strikes and costlier, riskier movement stay for life.
    scar_per_wound: float = 0.5     # per unit of health lost to a wound: a bare-handed blow ~0.07, ~13 to saturate
    severe_injury: float = 0.5      # injury above this leaves a lasting part
    lasting_share: float = 0.5      # share of the injury above that level which never heals
    max_lasting_injury: float = 0.35  # below the bleeding level: a lasting injury never kills
    # Down (RimWorld's "downed"): past these one cannot stand. One lies where one fell and can only
    # rest, sleep, call out, drink if water is beside one, and pick up and eat what is on one's own
    # cell: others decide whether to bring food, keep watch, or finish one off. Off by default.
    # Clothing (very basic): each founder starts with hides covering some share of the body, from
    # clothing_min to all of it, different for each; babies are born bare. It keeps heat in when it is
    # colder than comfort (body temperature drifts toward the cold more slowly: x 1 + insulation x
    # coverage). It does not wear out. Children born bare can keep warm with what they carry: hides
    # and feathers (byproducts below) count toward coverage while carried.
    clothing: bool = False
    # Byproducts of the living world (lifesim/minerals.py ORGANIC): leaf litter under trees, fiber
    # from foraging grasses, feathers, hides and bones from kills, a spear bound from a stick, a point
    # and fiber. Carried hides and feathers keep warm; leaves and hides under a sleeper are a bed;
    # leaves and fiber are tinder for sparks; a spear's length keeps a struck animal off.
    byproducts: bool = True
    leaf_fall: float = 0.004           # units per forest cell per 30-minute tick (x autumn_leaf_mult in autumn)
    autumn_leaf_mult: float = 5.0
    leaf_max: float = 3.0
    fiber_per_greens: float = 0.5      # fiber per unit of greens foraged on grass or marsh
    molt_prob: float = 0.004           # a feather per fowl per 30-minute tick
    tinder_mult: float = 3.0           # a spark catches this much more readily on dry leaves or fiber
    bedding_max: float = 0.6           # at most this much added insulation from what one lies on
    haft_edge: float = 0.5             # how sharp a point must be to bind into a spear
    spear_reach: float = 0.5           # a struck animal fights back this much less against a spear
    # Plant ecology (lifesim/ecology.py): greens, fruit, nuts and roots per cell, from biome densities
    # per km^2 times the cell's area; logistic regrowth and seed spread, seasonal crops, gathering
    # that takes time and pays less where food is thin. Off by default (the classic plants: one stock
    # per cell regrowing at a fixed rate); on in the living world.
    ecology: bool = False
    kcal_per_day: float = 2400.0       # an active forager's daily need: what hunger takes in a day
    loose_food_spoil: float = 0.05     # per day: plant food left lying on the ground
    # Camp fouling: where people stay, waste builds up (person-days per km^2), and drinking near it
    # risks infection: the infection chance per drink is water_infection x (1 + foul / foul_ref, at
    # most foul_max_mult). It fades with time, faster in rain. (Why foragers move camp, besides food.)
    fouling: bool = False
    # Excretion (on in the living world): an urge builds as the body processes food and water (more
    # when well watered, less asleep at night), relieved by RELIEVE wherever one stands, which leaves
    # waste there (the fouling above, which then comes only from this). Past full, it happens anyway:
    # soiling oneself is stressful and wakes a sleeper. Infants have no control. Choosing where to go
    # (away from the water one drinks) is left to learning.
    # Language (lifesim/env.py LEXICON): "emergent" (symbols mean nothing given) or "grounded" (a
    # defined lexicon; phrases about water and food with a direction are understood by listeners and
    # put into their remembered maps). writing: WRITE puts one's message on what is here, read by
    # anyone beside it, if one holds the means (lifesim/env.py write_means): an edge (sharpness >=
    # write_edge; on stone a point of hardness >= write_hard) or a pigment (ochre, coal), and a surface
    # (wood lying here, which makes a sign, a loose stone, a rock face, a tree's bark; not bare
    # ground). Cut in wood or bark it lasts mark_days, in stone mark_days_rock; drawn in pigment
    # mark_days_paint (weather wears it off).
    language_mode: str = "emergent"
    writing: bool = False
    # The message channel, for the communication controls (Lowe et al. 2019): "full"; "none" (no one
    # is ever heard: every message is silence); "shuffled" (each tick the messages heard are those of
    # other speakers, shuffled among them, so content is cut from its speaker's state).
    channel: str = "full"
    heard_recency: float = 0.6         # how fresh a heard place is in one's map (seen is 1)
    mark_days: float = 90.0
    mark_days_rock: float = 365.0
    mark_days_paint: float = 30.0
    write_edge: float = 0.5            # flint 0.8, obsidian 1.0, quartz 0.6, copper 0.6; a plain stone 0.2 is not enough
    write_hard: float = 0.7            # the point's hardness needed to cut stone (flint 1.0, obsidian 0.7, quartz 0.9)
    sign_min: float = 0.5              # how much wood (or stone) lying here makes a surface to write on
    excretion: bool = False
    urge_rate: float = 0.06            # per tick at 30 minutes, x (0.5 + hydration): ~4-6 reliefs a day
    urge_sleep_mult: float = 0.4       # made more slowly asleep (the body holds it at night)
    relieve_min: float = 0.15          # nothing to go with less than this
    urge_stress: float = 0.01          # stress per tick at full urge (from 0.7 up)
    soil_stress: float = 0.08          # stress from soiling oneself (per time)
    waste_per_urge: float = 0.3        # person-days of waste in a full urge
    infant_continence_years: float = 2.5
    foul_ref: float = 50.0             # person-days per km^2 that double the risk from water nearby
    foul_max_mult: float = 20.0
    foul_half_life_days: float = 20.0
    foul_rain_mult: float = 3.0        # fades this much faster while it rains
    clothing_min: float = 0.1
    clothing_insulation: float = 0.8
    downed: bool = False
    downed_health: float = 0.2      # health below this
    downed_injury: float = 0.8      # or injury at or above this (bleeding hard)
    # Pain shock (RimWorld's): pain = pain_per_injury x injury + fresh pain (pain_per_wound x health a
    # wound takes, fading by pain_fade per 30 minutes). At pain_shock / sqrt(pain_aversion) one drops
    # (only with downed on) and gets up again once the fresh pain fades. 0 turns shock off.
    pain_shock: float = 0.8
    pain_per_injury: float = 0.6
    pain_per_wound: float = 2.5
    pain_fade: float = 0.08        # half-life about 4 hours
    # Finishing off: blows and bites on the downed land square (x downed_hit_mult) and each blow from
    # a person has coup_prob of striking something vital and killing outright (RimWorld has no
    # multiplier: its downed simply cannot dodge or block, and die when a vital part is destroyed)
    downed_hit_mult: float = 2.0
    coup_prob: float = 0.15         # only without body_parts (with them, a blow to the neck finishes)
    # Body parts (env.PARTS: head, neck, torso, arms, legs). Every blow is aimed at one (the brain's
    # aim output). By part: the chance it lands on someone standing (the downed are always hit), how
    # much of its force reaches health, and how much damage the part itself takes per unit of blow.
    # A head at concussion_at drops one (knocked out); a head or neck at 1 kills; hurt arms weaken
    # (strength x (1 - arm_weakness x damage)), hurt legs slow (speed x (1 - leg_lameness x damage))
    # and at legs_unusable one cannot walk. Parts heal as injuries do. Predator bites land on a part
    # drawn from bite_parts; falls hurt the legs. For animals arms are forelegs, and a leg or foreleg
    # blow lames the animal for animal_lame_ticks (no running, half its steps).
    body_parts: bool = True
    part_hit: tuple = (0.6, 0.45, 0.9, 0.8, 0.8)
    part_lethality: tuple = (1.4, 2.0, 1.0, 0.4, 0.5)
    part_fragility: tuple = (1.5, 2.0, 1.0, 1.5, 1.5)
    concussion_at: float = 0.6
    arm_weakness: float = 0.6
    leg_lameness: float = 0.6
    legs_unusable: float = 0.9
    bite_parts: tuple = (0.1, 0.25, 0.35, 0.15, 0.15)
    animal_lame_ticks: int = 96     # calibration ticks (2 days at half-hour ticks)

    # Disease (SIR-style per agent)
    disease_duration: int = 96      # 2 days (was 200: with no healing while sick, a 0.8 health loss
    disease_damage: float = 0.004   # that killed anyone not at full health; now ~0.4 for a healthy adult)
    disease_hunger_mult: float = 0.5
    disease_fatigue: float = 0.002
    raw_meat_infection: float = 0.03
    water_infection: float = 0.002  # per drink, doubled in drought
    contact_infection: float = 0.02 # per tick per infected neighbor (multi-agent)
    immunity_decay: float = 0.0005

    # Stress (psychological state, felt by the agent)
    stress_from_damage: float = 3.0  # per unit health lost
    stress_from_threat: float = 0.03 # per tick with a predator within 3 cells
    stress_from_need: float = 0.005  # per critical need per tick
    stress_decay: float = 0.99
    stress_energy_cost: float = 0.002
    stress_heal_block: float = 0.7   # healing slows as stress rises and stops at this level
    # Company: loneliness raises stress (scaled by the sociability trait), each companion in view
    # halving it; company also speeds recovery from any stress (social buffering). Its worth is
    # whatever calmer, faster-healing bodies are worth: real, bounded, not a reward in itself.
    loneliness_stress: float = 0.004  # per tick alone, x sociability; ~0.4 settled stress for an average agent
    social_buffering: float = 0.01    # extra stress decay per tick with company
    intimacy_relief: float = 0.2      # stress removed by mating, x libido, fading with repeats per partner
    trauma_rate: float = 2.0         # pain-aversion phenotype gain per health lost
    trauma_recovery: float = 0.0005  # drift back toward genotype per tick

    # ------------------------------------------------------------ animals
    # Densities are per 1000 cells; counts are fixed array capacities.
    small_prey_per_1000: float = 12.0
    large_prey_per_1000: float = 8.0   # 8 on a 32 x 32 map: enough encounters to learn group hunting
    predators_per_1000: float = 6.0   # 6 on a 32 x 32 map: 2 wolves, 2 big cats, a bear, a crocodile (lifesim/fauna.py)
    small_prey_hp: float = 0.5
    large_prey_hp: float = 3.0
    large_prey_armor: float = 0.8    # damage taken = total^2 / (total + armor): one hunter ~0.26/tick, two ~0.72
    predator_hp: float = 1.5
    predator_armor: float = 0.3
    small_prey_meat: float = 1.0
    large_prey_meat: float = 6.0
    strike_damage: float = 0.6       # scaled by strength, age and injury
    agent_strike_scale: float = 0.25 # people are hurt less per blow than prey: a bare hand does ~0.15
    small_prey_notice: int = 2
    large_prey_notice: int = 2
    predator_notice: int = 5
    small_prey_flee_prob: float = 0.7
    large_prey_flee_prob: float = 0.35  # large, slow animals: flee less often than small prey
    predator_chase_prob: float = 0.85
    wander_prob: float = 0.3
    predator_attack_prob: float = 0.3
    predator_attack_damage: float = 0.12
    predator_attack_injury: float = 0.15
    predator_rest_ticks: int = 150   # after an attack the predator stops hunting
    predator_prey_kill_prob: float = 0.2
    predator_large_kill_prob: float = 0.03   # per tick per able predator on a large animal's cell
    animal_hp_regen: float = 0.02
    small_prey_respawn: float = 0.002
    large_prey_respawn: float = 0.0005
    predator_respawn: float = 0.0005
    # Grazing (RimWorld's herbivores, with food-limited numbers): plant-eaters eat the plants where they
    # stand and drift toward richer ground; one that finds too little loses condition (hp) and can
    # starve; births fill empty slots faster the better fed the rest of its kind are (a trickle still
    # wanders in). Predators get hungry: a kill or a carcass sates them for a while; one that goes long
    # unfed weakens and dies; predator births follow how much prey is about. Off by default.
    grazing: bool = False
    graze_small: float = 0.003          # plant food a small grazer eats per tick (a cell holds up to plant_max)
    graze_large: float = 0.02           # a large one
    graze_seek: float = 0.6             # chance a calm grazer steps toward the richest ground beside it
    graze_starve: float = 0.002         # share of its hp a grazer loses per tick with too little to eat
    predator_starve: float = 0.0015     # share of its hp a hungry predator loses per tick
    predator_scavenge: float = 0.5      # meat a hungry predator eats from a carcass it stands on
    immigration: float = 0.25           # births at this share of the base rate even when none are well fed
    meat_spoil_ground: float = 0.01
    meat_spoil_inventory: float = 0.002

    # ------------------------------------------------------------ materials and fire
    # Physics only. No rule mentions fire-making, cooking or torches: those are
    # sequences agents may or may not discover.
    cooked_food_value: float = 1.0   # heat makes meat more nourishing
    cooked_spoil_ground: float = 0.005
    cooked_spoil_inventory: float = 0.001
    wood_max: float = 2.0            # dead wood per forest cell
    wood_regrow: float = 0.0005
    wood_fuel: float = 40.0          # ticks of burning per unit of wood
    plant_fuel: float = 8.0          # ticks of burning per unit of dry plants
    tool_bonus: float = 0.5          # strike damage x (1 + bonus x hardness of hand item)
    edge_bonus: float = 0.5          # ... + this x sharpness of the hand item (flint, obsidian, metal)
    cut_bonus: float = 1.0           # an edge in hand gathers up to (1 + this x sharpness) meat or wood
    spark_kit_mult: float = 4.0      # sparks x this when a spark-thrower (pyrite, iron) meets something hard
    mineral_abundance: float = 1.0   # x every mineral deposit chance (0 = no minerals)

    # ------------------------------------------------------------ fauna and bugs (lifesim/fauna.py)
    fauna_diversity: bool = True     # species within each animal group (False: one per group)
    hive_density: float = 0.04       # chance a forest cell holds a bees' hive
    mound_density: float = 0.03      # chance a grass or desert cell holds a termite mound
    honey_max: float = 3.0
    grubs_max: float = 3.0
    honey_regrow: float = 0.002      # per tick in spring and summer
    grubs_regrow: float = 0.004
    honey_food_value: float = 1.2    # dense and sweet; never spoils
    grubs_food_value: float = 0.5
    grub_hand_yield: float = 0.25    # share of a normal grasp that fingers draw from a mound
    probe_bonus: float = 3.0         # a stick (wood in hand) draws (1 + this) x as many
    sting_damage: float = 0.06       # robbing a hive without smoke
    mosquito_infection: float = 0.0015   # infection chance per tick at full mosquito density
    fly_infection_mult: float = 2.0  # raw meat eaten amid flies: infection chance x (1 + this x flies)
    fly_spoil_mult: float = 2.0      # meat on the ground rots (1 + this x flies) x as fast
    spark_prob: float = 0.05         # per strike of hard on hard
    ignite_prob: float = 0.5         # a spark catches on fuel (x dryness)
    fire_dryness: tuple = (1.0, 0.1, 1.5, 0.8, 1.3)  # by weather: clear, rain, drought, cold, heat
    fire_burn_rate: float = 1.0      # fuel ticks consumed per tick
    rain_burn_mult: float = 3.0
    spread_prob: float = 0.003       # per burning neighbor per tick (x dryness)
    lightning_prob: float = 0.006    # per tick in summer, drought or heat waves (~2.7 fires/year)
    fire_heat: float = 15.0          # degrees within 1 cell of a fire
    fire_heat_far: float = 7.0       # degrees within 2 cells
    burn_damage: float = 0.05        # per tick standing in flames
    burn_injury: float = 0.05
    cook_rate: float = 0.1           # meat units cooked per tick at full heat
    char_rate: float = 0.05          # cooked meat destroyed per tick in the flames
    torch_ticks: int = 60            # a branch pulled from a fire burns this long

    # ------------------------------------------------------------ social
    memory_decay: float = 0.999      # relationship memory half-life ~700 ticks
    touch_reward: float = 0.005      # scaled by the social_reward gene, for both parties
    speak_cost: float = 0.0005       # energy per non-silent message, so noise is not free

    # ------------------------------------------------------------ moral sentiments
    # Weights on each agent's heritable moral traits (all in [0, 1]); see genotype.
    empathy_weight: float = 0.5      # per unit of health a visible other loses, x how much it shows (sqrt pain_aversion)
    kin_weight: float = 0.5          # same, scaled by appearance similarity
    inequity_weight: float = 0.01    # per tick, per unit of mean satiety gap to visible others
    reciprocity_reward: float = 0.05 # repaying someone who gave you food (fades with repetition)
    # ------------------------------------------------------------ biomes and climate
    # biomes: besides grass, forest, rock and water, the land holds desert (hot by day, little food,
    # faster thirst), tundra (cold, sparse, slow going), marsh (rich, drinkable but riskier water,
    # slow going) and caves in rock outcrops (shelter: temperature held near comfort, dry in rain).
    # Placement follows elevation, moisture and a north-south temperature gradient.
    biomes: bool = False
    desert_fraction: float = 0.12
    tundra_fraction: float = 0.12
    marsh_fraction: float = 0.06
    caves: int = 3
    latitude_cold: float = 8.0          # C colder at the north edge than the south edge
    desert_heat: float = 6.0
    tundra_cold: float = 6.0
    cave_buffer: float = 0.8            # caves pull the air 80% of the way to comfort
    # underground (needs biomes, whose cave cells are the mouths): a second layer under the map.
    # Geography: from each mouth, tunnels wind out under the land (never under open water),
    # opening ragged chambers, some with a pool. Each cave system's formation comes from the land
    # at its mouth (minerals.KINDS): karst (limestone, wet, many pools, stalagmites; a quarter
    # of its chambers quartz geodes, now and then gold; sinkholes open new mouths onto grass or
    # forest), lava tube on high ground (long straight tubes, basalt, obsidian, sulfur, warm air;
    # hot springs beside it: warm, drinkable, sulfur at the edge), ice cave in the cold north
    # (frozen pools, air near freezing all year), salt cave in desert (salt, gypsum, no water).
    # Movement: an agent on a mouth can DELVE to go down or come back up (an infant on its
    # mother's cell goes with her); below it walks only along passages.
    # Darkness: below it is always dark. One feels only the cells next to one, and sees what a
    # burning branch lights (torch_light, one's own or another's). Awake in the dark, fear builds
    # (dark_stress) and steps go astray (dark_veer: a random way, into rock is no step) or fail
    # (dark_stumble). A torch removes all of it.
    # Life below: the air holds near the season's mean, pulled toward the formation's own; no
    # rain, snow, plants or fire; only pools give water; the walls hold the formation's minerals
    # and unlimited stone; what is dropped, placed or killed below lies on the cave floor (meat
    # rots slower in the cool), and a burning branch set down goes out.
    # Layers: agents above and below do not see, hear, touch, strike, mate with, infect, imitate
    # or warm each other. Animals live on both: they notice the other layer only by scent close
    # by, chasers follow a quarry down, fleeing prey may bolt down a mouth, bats roost below by
    # day (leaving saltpetre), bears den below in winter.
    # Off (the default): the layer is empty, DELVE is never available, bats have no slots, and
    # the world is as before.
    underground: bool = False
    cave_tunnels: int = 3               # tunnels from each mouth
    cave_len: int = 0                   # steps per tunnel (0: the grid's width)
    # per formation, tunnels' turning, chambers' frequency and pools are set in env.py (_KIND_*)
    cave_ore_mult: float = 2.0
    deep_cave_buffer: float = 0.9       # below, the air is pulled 90% from the season's mean to comfort
    torch_light: int = 3                # how far a burning branch lights below
    cave_spoil_mult: float = 0.5        # meat lying in the cool of a cave rots at half the rate
    scent_range: int = 2                # animals notice someone on the other layer only this close
    prey_bolt_prob: float = 0.3         # a fleeing animal on a mouth bolts down (per tick)
    den_prob: float = 0.3               # a calm animal on a mouth goes to the layer it keeps (per tick)
    dark_stress: float = 0.02           # per awake tick below with no light where one stands: fear
    dark_veer: float = 0.35             # a step in the dark goes a random way along the passage
    dark_stumble: float = 0.25          # ... or fails
    guano_rate: float = 0.002           # saltpetre left per roosting bat per tick
    sinkholes: int = 2                  # karst roofs fallen in: extra mouths onto grass or forest
    hot_springs: int = 2                # warm drinkable pools beside lava tubes
    spring_heat: float = 8.0            # C of warmth at a hot spring and the cells around it
    desert_thirst_mult: float = 1.5
    marsh_infection_mult: float = 2.0   # drinking marsh water, with no open water beside one
    desert_regrow: float = 0.25         # x grass regrowth
    tundra_regrow: float = 0.35
    marsh_regrow: float = 1.3
    # dynamic_weather: clouds form over water and marsh (more in summer, less in winter), drift
    # with a wind that shifts direction, and rain where they are thick; fronts sweep in now and
    # then. Rain is local: it waters plants, damps fire, chills those caught in it (not in caves)
    # and can be drunk. Below freezing it falls as snow, which lies, slows walking, chills and
    # stops growth until it melts. Lightning strikes under storm clouds.
    dynamic_weather: bool = False
    evaporation: float = 0.04
    cloud_decay: float = 0.97
    front_prob: float = 0.04
    front_strength: float = 0.9
    rain_threshold: float = 0.5
    rain_use: float = 0.04              # cloud spent per tick of rain
    storm_threshold: float = 0.85
    wind_change_prob: float = 0.01
    cloud_drift_every: int = 2          # ticks per cell of drift
    rain_chill: float = 0.1             # thermal load while rained on
    rain_drink: float = 0.01            # water caught per tick in the rain
    snow_rate: float = 0.05
    snow_melt: float = 0.004            # per degree above freezing per tick
    snow_chill: float = 0.15
    snow_move_mult: float = 1.0         # walking cost x (1 + this x snow depth)
    # climate: a slow warm-cool cycle plus random drift, added to every temperature; warm phases
    # bring more droughts and heat waves. 0 = off.
    climate_cycle_years: float = 0.0
    climate_amp: float = 3.0
    climate_drift: float = 0.05         # C of random drift per day

    # Given skills (the "caveman" slate): whole survival behaviours the agent can choose like any
    # action. Each picks one step toward its goal from the agent's own cognitive map and sight
    # (never the true map): seek water (walk to remembered water, drink there), seek food (walk to
    # remembered plants, gather, eat), flee (step away from the nearest visible predator). Routes
    # avoid water and steep ground; with nothing known, the agent explores unseen ground. When and
    # whether to use them stays the agent's choice. Off = the "blank" slate (primitive actions only).
    skills: bool = False
    skill_search_steps: int = 64       # route-planning sweeps (enough for any path on a 32 x 32 world)
    skill_cliff_avoid: float = 20.0    # a fully steep cell counts as this many steps on a route
    animal_sleep: float = 0.6          # share of its resting half of the day an animal sleeps (0: never)
    animal_hunting: bool = True        # predators chase prey they can take; prey flee such predators
    prey_preference: float = 3.0       # a predator choosing whom to chase counts a person this many cells farther
    stalk_large: bool = False          # predators also stalk large prey (8 slots against 6 hunters: herds collapse)
    hunt_kill_scale: float = 0.3       # with animal_hunting: x predator kill chances per meeting (more meetings)
    # Taming, emergent: no act tames. Each animal is more or less at ease with each person (a bond,
    # 0..1), built when it eats food that person dropped and, slowly, from that person's harmless
    # company; a blow from them (and, less, from anyone) breaks it; it fades apart. At ease
    # (tame_calm), it neither flees nor hunts that person; a tame predator beside them stands off wild
    # ones as a defender; closer still (tame_follow) it follows them. Species differ (fauna TAMEABLE).
    taming: bool = True
    tame_feed: float = 0.2             # bond gained each tick it eats what that person dropped (x tameable)
    tame_company: float = 0.002        # gained each tick that person is beside it doing it no harm (x tameable)
    tame_harm: float = 0.8             # share of its bond with a striker lost to a blow (half that with anyone)
    tame_half_life_days: float = 20.0  # bonds fade by half in this many days
    tame_drop_window_days: float = 1.0  # food dropped within this long counts as fed by the dropper
    tame_calm: float = 0.5             # at or above: at ease with that person
    tame_follow: float = 0.7           # at or above: follows that person (cfg.tame_follow_prob of its steps)
    tame_follow_prob: float = 0.6
    scavenge_seek: float = 0.6         # a hungry predator not chasing anything steps toward meat beside it
    stalk_hungry_hp: float = 0.75     # a predator below this share of its hp stalks large prey too (0: never)
    animal_sleep_jitter: float = 0.075  # each slot's sleep bout starts up to this share of a day early or late
    kill_asleep_mult: float = 3.0      # x a predator's kill chance on prey caught asleep
    kill_lame_mult: float = 1.0        # + this x on lame prey (x2 in all)
    kill_weak_mult: float = 2.0        # + this x the share of its hp the prey has lost (x3 at none left)
    patch_leave: float = 0.5           # seek-food skips and leaves patches poorer than this x the known average (0: off)
    # A rare temperament: selflessness weighs others' visible fortunes like one's own, kin or not,
    # so such an agent may give up its last food or its life for a stranger. Most agents have none;
    # each founder or newborn has it with probability saint_rate (not inherited; arises anew).
    saint_rate: float = 1e-6
    # Archetypes placed among the founders, e.g. (("tyrant", 1), ("saint", 1)); see lifesim/archetypes.py
    archetypes: tuple = ()
    # Sleep: a built-in need, as in people. Sleepiness builds while awake (faster at night, a body
    # clock) and sleep removes it. Asleep, one sees and hears only what is next to one, and wakes
    # when struck, attacked, burned or touched. Sleep restores energy, calms and speeds healing;
    # going without it stresses, narrows sight and makes falls likelier. Past sleep_forced one falls
    # asleep whatever one chooses, and cannot wake by choice until sleepiness is below sleep_wake_block.
    sleep_pressure: float = 0.0125     # per tick awake (x sleep_night_mult at night): ~0.45 after 16 h, collapse ~30 h
    sleep_night_mult: float = 1.5
    sleep_recovery: float = 0.04       # per tick of deep sleep at night; light sleep half, by day x0.7 (~8 h a night)
    sleep_deep_ticks: int = 4          # ticks of unbroken sleep to reach deep sleep (2 h)
    sleep_day_efficiency: float = 0.7
    sleep_forced: float = 0.95
    # Sleep is honest: one can only fall asleep with real sleep pressure (lower at night, when the
    # body clock opens the gate). Lying down without it is rest, nothing more: no deep sleep, no sleep
    # healing or calm. (Before this, choosing sleep at no sleepiness gave sleep's benefits for free.)
    sleep_onset: float = 0.3           # sleepiness needed to fall asleep by day (~12 h awake)
    sleep_onset_night: float = 0.1     # ... at night
    sleep_wake_block: float = 0.6
    sleep_energy: float = 0.01         # extra energy per tick of deep sleep, beyond resting
    sleep_stress_relief: float = 0.01  # stress removed per tick of deep sleep
    sleep_heal_mult: float = 1.5       # healing while asleep
    sleep_debt_stress: float = 0.01    # stress per tick at full sleepiness (from 0.7 up)
    sleepy_sight_loss: int = 2         # view radius lost at full sleepiness (from 0.5 up)
    sleepy_fall_mult: float = 1.0      # falls x (1 + this x sleepiness)
    # Felt needs are convex (homeostatic): a small deficit is mild, a large one is felt strongly.
    # Each need (food, water, energy, rest from sleepiness) contributes
    # need_weight * (1 - deficit ** need_curve) to felt well-being.
    need_weight: float = 0.6
    need_curve: float = 2.0
    habituation_half_life: int = 48  # ticks for a repeated feeling to recover half its strength (1 day)
    gift_window: int = 20            # dropped food taken by someone else within this many ticks is a gift
    revenge_reward: float = 0.05     # per strike on someone who has struck oneself
    sadism_reward: float = 0.5       # per unit of health one's blows take from someone, x how much it shows;
                                     # x the sadism trait; habituates with revenge (satisfaction in striking)
    # Emotional pain (heartache): seeing others hurt (x empathy) and losing those one is attached to.
    # It fades over days, raises stress by heartache_stress per tick, and is felt (observed); it is not
    # physical pain and never sends one into shock.
    heartache_empathy: float = 1.0   # per unit of health visible others lose to wounds (x how it shows), x empathy
    heartache_grief: float = 0.8     # per death of someone one is attached to, x attachment
    heartache_fade: float = 0.03     # per 30 minutes (half-life about 11 hours)
    heartache_stress: float = 0.002  # stress per tick at full heartache: settles stress near 0.2, like
                                     # loneliness (0.4) and below the healing block (0.7); 0.01 blocked healing
    # Phantom and chronic pain: a lasting injury flares now and then, with no new wound
    phantom_prob: float = 0.02       # chance per 30 minutes at full lasting injury (x lasting / max)
    phantom_pain: float = 0.3        # fresh pain a flare adds

    # ------------------------------------------------------------ observation
    view_radius: int = 5
    # Cognitive map: each agent remembers what it has seen of the land (water, plants, when),
    # fading with time; read as a coarse map centred on itself (map_cells x map_cells blocks).
    map_cells: int = 8
    map_half_life: int = 480       # ticks for a consolidated memory of a place to fade by half (10 days)
    map_fresh_half_life: int = 48  # a memory not yet consolidated by deep sleep fades by half in a day

    # ------------------------------------------------------------ reward
    # "acts": the original design (touch, novelty and moral sentiments reward acts and harms).
    # "survival": survival is the only reward; dispositions decide how it is felt. An agent feels
    # its own survival (alive, death, and changes in felt well-being from its body) and, for agents
    # it can see, theirs, weighted by empathy, kin care, reciprocity (+) and vengefulness (-).
    reward_mode: str = "acts"
    wellbeing_scale: float = 0.5     # felt well-being = scale x (health x pain aversion + needs + warmth)
    shaping_gamma: float = 0.997     # must match the trainer's discount: keeps felt changes loop-proof
    care_empathy: float = 0.5        # weight on a visible agent's survival signal, x empathy
    care_kin: float = 0.5            # extra weight, x kin care x appearance similarity
    care_reciprocity: float = 0.5    # extra weight for those who have fed you, x reciprocity
    care_revenge: float = 0.5        # negative weight for those who have struck you, x vengefulness
    care_sadism: float = 0.5         # negative weight for those you have struck, x sadism: their pain pleases
    care_inequity: float = 0.01      # envy and guilt over gaps in felt well-being (cost only)
    # Attachment: survival felt toward particular others. i's attachment to j (0..1) grows with
    # evidence that j keeps i alive (food j left that i took when hungry, warmth huddled beside j when cold,
    # predators held off with j beside, calm time together in view, intimacy, a living child they
    # share, milk for an infant) and falls when j strikes i; it fades with time apart. It acts only
    # through survival: j's visible condition counts in i's reward (care += care_attachment x A),
    # an attached companion in view calms i more, and attached ones out of sight leave i less
    # secure. Nothing rewards attachment for its own sake.
    care_attachment: float = 1.0
    attach_food: float = 0.05           # per gift taken while hungry
    attach_hungry: float = 0.6          # a gift counts only if the taker's food was below this
    attach_warmth: float = 0.003        # per tick huddled beside them while cold
    attach_protect: float = 0.15        # per predator attack held off with them beside
    attach_company: float = 0.0003      # per tick in view
    attach_intimacy: float = 0.04       # per mating
    attach_coparent: float = 0.002      # per tick a child they share is alive
    attach_nurse: float = 0.01          # per tick an infant is nursed (toward its mother)
    attach_harm: float = 0.3            # share lost per blow from them
    attach_half_life: int = 960         # ticks apart for attachment to fade by half (20 days)
    attach_calm: float = 0.03           # extra stress decay with attached company in view
    # How fast it grows depends on fit, each a reason two survive better together: similar
    # temperament (each can predict the other), the other's visible fitness (health, condition,
    # strength: a better bet), danger faced together (both threatened, in view of each other) and
    # teamwork (blows on the same animal in the same tick).
    attach_similarity: float = 0.5      # weight of temperament similarity on growth
    attach_fitness: float = 0.5         # weight of the other's fitness on growth
    attach_shared_threat: float = 0.02  # per tick both are threatened in view of each other
    attach_synergy: float = 0.05        # per tick both strike the same animal
    attach_absence: float = 0.002       # stress per tick per unit of attachment out of sight
    alive_reward: float = 0.01
    death_penalty: float = 1.0
    novelty_reward: float = 0.002

    @property
    def view_size(self) -> int:
        return 2 * self.view_radius + 1

    @property
    def ticks_per_bio_year(self) -> float:
        return self.max_age / self.max_bio_years

    @property
    def year_length(self) -> int:
        return self.days_per_year * self.day_length

    @property
    def season_starts(self) -> tuple:
        """First day (0-based) of summer, autumn and winter: whole days, so a
        season never changes mid-day. 40-day year: 10, 20, 30; 365: 92, 183, 274."""
        return tuple(-(-k * self.days_per_year // 4) for k in (1, 2, 3))

    @property
    def season_length(self) -> float:
        """Average ticks per season."""
        return self.year_length / 4

    # Nominal physical scale, derived from rules the world already has. One move per tick at a
    # walking pace of 4 km/h makes a cell 2 km across at 30-minute ticks (67 m at 1 minute); the altitude cooling (6 C over the
    # map's full relief) at the usual 6.5 C per km makes the relief about 900 m. Close-range rules
    # (striking or huddling with someone in a neighbouring cell) do not fit a 2 km cell. The plant
    # ecology (ecology=True) and camp fouling use it: their amounts are per km^2 times a cell's area.
    walk_kmh: float = 4.0
    lapse_c_per_km: float = 6.5

    @property
    def cell_km(self) -> float:
        return self.walk_kmh * self.tick_minutes / 60.0

    @property
    def relief_m(self) -> float:
        return 1000.0 * self.altitude_cold / self.lapse_c_per_km

    def _count(self, per_1000: float) -> int:
        return max(1, int(round(per_1000 * self.grid_size ** 2 / 1000)))

    @property
    def n_small_prey(self) -> int:
        return self._count(self.small_prey_per_1000)

    @property
    def n_large_prey(self) -> int:
        return self._count(self.large_prey_per_1000)

    @property
    def n_predators(self) -> int:
        return self._count(self.predators_per_1000)


def with_reproduction(cfg: "EnvConfig", n_founders: int | None = None, **kw) -> "EnvConfig":
    """A config for worlds with births: by default 1 day per biological year (adult about day 15,
    deaths from age most likely about day 70; days_per_bio_year changes it), founders filling half the slots unless
    n_founders is given."""
    import dataclasses
    founders = n_founders if n_founders is not None else max(2, cfg.n_agents // 2)
    dpby = kw.pop("days_per_bio_year", cfg.days_per_bio_year)
    if "conception_prob" not in kw:            # per act, from the yearly chance and how much time an act stands for
        acts = max(dpby * kw.get("mating_acts_per_day", cfg.mating_acts_per_day), 1.0)
        kw["conception_prob"] = round(1.0 - (1.0 - kw.get("conception_per_bio_year", cfg.conception_per_bio_year)) ** (1.0 / acts), 4)
    return dataclasses.replace(cfg, reproduction=True, n_founders=founders, days_per_bio_year=dpby,
                               max_age=int(round(cfg.day_length * dpby * cfg.max_bio_years)), **kw)


def with_time_scale(cfg: "EnvConfig", days_per_year: int = 40, days_per_bio_year: float = 1.0) -> "EnvConfig":
    """Set the calendar (days per year: seasons, climate) and biological time (days per year of
    age: growth, fertility, pregnancy, lifespan). The defaults compress both so that winters and
    generations fit in a run; (365, 365) is real time. Apply before with_reproduction.

    What real time costs: a child is fertile at 15 x 365 = 5,475 days (262,800 ticks); a 100M
    agent-step run of 16 agents in 32 worlds lives about 195,000 ticks per world, so no one born
    in training would grow up, and a 400-day continuous world is 1.1 years."""
    import dataclasses
    return dataclasses.replace(cfg, days_per_year=int(days_per_year), days_per_bio_year=float(days_per_bio_year))


# ---------------------------------------------------------------- tick length
# How each field changes when a tick lasts k = minutes / 30 times as long as the calibration.
# Every rule is kept the same per unit of GAME TIME; per act (a blow, a bite, a drink, a step, a
# mating) nothing changes, so a finer tick means more, shorter acts, as in a colony sim.
_PER_TICK_AMOUNT = (       # added or lost per tick: x k
    "hunger_rate", "thirst_rate", "fatigue_rate", "rest_recovery", "starve_damage", "thirst_damage",
    "exhaust_damage", "heal_rate", "cold_damage", "heat_damage", "fat_gain_kg", "fat_burn_kg",
    "injury_heal_rate", "injury_bleed", "disease_damage", "disease_fatigue", "immunity_decay",
    "stress_from_threat", "stress_from_need", "stress_energy_cost", "loneliness_stress", "social_buffering",
    "trauma_recovery", "plant_regrow_forest", "plant_regrow_grass", "animal_hp_regen", "wood_regrow",
    "honey_regrow", "grubs_regrow", "cook_rate", "burn_damage", "burn_injury", "nurse_rate",
    "inequity_weight", "care_inequity", "speak_cost", "alive_reward", "novelty_reward",
    "sleep_pressure", "sleep_recovery", "sleep_energy", "sleep_stress_relief", "sleep_debt_stress",
    "attach_warmth", "attach_company", "attach_coparent", "attach_nurse", "attach_shared_threat",
    "attach_absence", "attach_calm", "dark_stress", "guano_rate", "evaporation", "rain_use", "rain_drink",
    "snow_rate", "snow_melt", "move_cost", "graze_small", "graze_large", "graze_starve", "predator_starve", "urge_rate", "urge_stress",
    "heartache_stress", "tame_feed", "tame_company")
_PER_TICK_CHANCE = (       # a chance each tick: 1 - (1 - p) ** k
    "contact_infection", "mosquito_infection", "spread_prob", "lightning_prob", "predator_prey_kill_prob",
    "predator_large_kill_prob", "small_prey_respawn", "large_prey_respawn", "predator_respawn",
    "predator_attack_prob", "prey_bolt_prob", "den_prob", "front_prob", "wind_change_prob", "conception_prob",
    "meat_spoil_ground", "meat_spoil_inventory", "cooked_spoil_ground", "cooked_spoil_inventory", "char_rate",
    "thermal_rate", "fall_prob")
_PER_TICK_KEEP = ("stress_decay", "memory_decay", "cloud_decay", "shaping_gamma")   # kept per tick: ** k
_DURATION_TICKS = (        # a stretch of game time counted in ticks: / k
    "max_steps", "max_age", "spell_min", "spell_max", "disease_duration", "predator_rest_ticks", "torch_ticks",
    "wood_fuel", "plant_fuel", "gift_window", "sleep_deep_ticks", "habituation_half_life", "map_half_life",
    "map_fresh_half_life", "attach_half_life")


def with_tick_minutes(cfg: "EnvConfig", minutes: float) -> "EnvConfig":
    """The same world with ticks of `minutes` game minutes (30 = the calibration). Needs, healing,
    sleep, illness, growth, fire, weather, spoilage, attachment and rewards keep their pace in game
    time; acts keep their size per act. Space follows from one step per tick at walking pace: a cell
    is walk_kmh x minutes / 60 across (67 m at 1 minute), so a 32 x 32 map is a ~2 km village area.
    What cannot stay the same: on a fixed grid of smaller cells everything is closer in game time
    (food, water, others, predators), so the ecology needs re-checking (scripts/viability.py).
    Apply to a config that is still at its own tick length (any order with the other helpers)."""
    import dataclasses
    k = float(minutes) / float(cfg.tick_minutes)
    if k == 1.0:
        return cfg
    day = 1440.0 / float(minutes)
    if abs(day - round(day)) > 1e-6:
        raise ValueError(f"tick_minutes must divide a day of 1440 minutes, got {minutes}")
    kw = {"tick_minutes": float(minutes), "day_length": int(round(day))}
    for f in _PER_TICK_AMOUNT:
        kw[f] = getattr(cfg, f) * k
    for f in _PER_TICK_CHANCE:
        kw[f] = 1.0 - (1.0 - getattr(cfg, f)) ** k
    for f in _PER_TICK_KEEP:
        kw[f] = getattr(cfg, f) ** k
    for f in _DURATION_TICKS:
        v = getattr(cfg, f) / k
        kw[f] = type(getattr(cfg, f))(max(1, round(v)) if isinstance(getattr(cfg, f), int) else v)
    return dataclasses.replace(cfg, **kw)


def living_world(cfg: "EnvConfig", **kw) -> "EnvConfig":
    """The 'living' world: biomes, dynamic weather and a 3-year climate cycle, with the downed (too
    hurt to stand) and grazing animals whose numbers follow their food."""
    import dataclasses
    # Excretion stays off unless asked for (an optional complexity: urge, relieving oneself, soiling).
    kw = {"downed": True, "grazing": True, "clothing": True, "ecology": True, "fouling": True, "excretion": False, **kw}
    return dataclasses.replace(cfg, biomes=True, dynamic_weather=True, climate_cycle_years=3.0, **kw)
