"""Can this world be lived in at all? A scripted, competent band, for an upper bound.

    python scripts/viability.py --world living --days 160 --seeds 3
    python scripts/viability.py --preset tribe --days 240

Training can only find behaviour the world allows. This runs the same world as the experiments
with hand-scripted agents who do what a competent forager band would do, with full knowledge of
the map (so it is an upper bound, not a baseline): flee predators when alone and stand together otherwise, drink and eat before needs get
low, sleep at night, go to forest or cave and huddle when cold, mothers stay with their nursing
children, and couples mate when fed. No fire, tools, hunting or sharing: those are left for
learning. If this band cannot keep its numbers up for years, no amount of training will, and the
world is the thing to fix.

Prints, per seed, the population over time, births, deaths by cause and season, and how long the
band lasted; then a verdict. Writes viability.json next to --out if given.
"""
from __future__ import annotations

import argparse
import dataclasses
import collections
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lifesim import env as E  # noqa: E402
from lifesim.config import EnvConfig, living_world, with_reproduction  # noqa: E402
from lifesim.presets import PRESETS  # noqa: E402
from lifesim import threats  # noqa: E402
from lifesim.threats import with_threats  # noqa: E402


def _toward(pos, target, can_move, g):
    """(N,) the move that most reduces Chebyshev distance to target (per agent), or REST."""
    nb = pos[:, None, :] + E.NEIGHBORS8[None]
    d = jnp.max(jnp.abs(nb - target[:, None, :]), axis=-1).astype(jnp.float32)
    d = jnp.where(can_move, d, 1e9)
    k = jnp.argmin(d, axis=1)
    here = jnp.max(jnp.abs(pos - target), axis=-1)
    return jnp.where((d[jnp.arange(pos.shape[0]), k] < here) & (d[jnp.arange(pos.shape[0]), k] < 1e8),
                     E.MOVE_N + k, E.REST)


def _down(field, pos, can_move, g):
    nb = jnp.clip(pos[:, None, :] + E.NEIGHBORS8[None], 0, g - 1)
    v = jnp.where(can_move, field[nb[..., 0], nb[..., 1]], 1e9)
    k = jnp.argmin(v, axis=1)
    here = field[pos[:, 0], pos[:, 1]]
    return jnp.where(v[jnp.arange(pos.shape[0]), k] < jnp.minimum(here, 1e8), E.MOVE_N + k, E.REST)


def scripted_policy(cfg: EnvConfig, knowledge: str = "full"):
    """knowledge="full": routes from the true map (an upper bound). "own": only the caveman slate's
    given skills, which route from what each agent has seen (what trained agents can do)."""
    g, n = cfg.grid_size, cfg.n_agents

    def field(targets, state):
        return E.route_field(targets[None], state.terrain, state.cliff, cfg)[0]

    @jax.jit
    def act(state):
        mask = E.action_mask(state, cfg)
        can_move = mask[:, E.MOVE_N:E.MOVE_NW + 1]
        idx = jnp.arange(n)
        pos, alive = state.pos, state.alive
        years = (state.age + state.birth_age) / cfg.ticks_per_bio_year
        night = E.is_night(state.t, cfg)
        r, c = pos[:, 0], pos[:, 1]

        # where things are (full knowledge: an upper bound)
        water_cells = E.drinkable(state.terrain)
        shelter = (state.terrain == E.FOREST) | (state.terrain == E.CAVE)
        forage = E.forage_grid(state, cfg)
        if cfg.ecology:                    # a competent forager heads for rich patches and leaves thin ones
            rich = forage > 0.5 * cfg.plant_max
            food_cells = jnp.where(jnp.any(rich), rich, forage > 0.15 * cfg.plant_max)
        else:
            food_cells = forage > 0.3
        f_water, f_food, f_shelter = field(water_cells, state), field(food_cells, state), field(shelter, state)
        go_water = jnp.where(mask[:, E.DRINK], E.DRINK, _down(f_water, pos, can_move, g))
        held = state.inv[idx, state.hand]
        edible = ((state.hand == E.PLANT) | (state.hand == E.COOKED_MEAT)) & (held > 0.05)
        plants_here = food_cells[r, c] if cfg.ecology else forage[r, c] > 0.1
        go_food = jnp.where(edible, E.EAT,
                  jnp.where(state.inv[:, E.PLANT] > 0.05, E.SELECT_0 + E.PLANT,
                  jnp.where(plants_here & (state.hand == E.PLANT) & mask[:, E.GRASP], E.GRASP,
                  jnp.where(plants_here, E.SELECT_0 + E.PLANT, _down(f_food, pos, can_move, g)))))
        in_shelter = shelter[r, c]
        go_shelter = _down(f_shelter, pos, can_move, g)
        if knowledge == "own":
            go_water, go_food, go_shelter = (jnp.full(n, E.SEEK_WATER), jnp.full(n, E.SEEK_FOOD),
                                             jnp.full(n, E.SEEK_SHELTER))

        # the youngest nursing child of each mother
        child_years = jnp.where(alive & (state.mother >= 0), years, 1e9)
        mine = (state.mother[None, :] == idx[:, None]) & (child_years[None, :] < cfg.nurse_years)
        has_child = jnp.any(mine, axis=1)
        child = jnp.argmin(jnp.where(mine, child_years[None, :], 1e9), axis=1)
        child_far = has_child & (jnp.max(jnp.abs(pos - pos[child]), axis=-1) > 1)
        to_child = _toward(pos, pos[child], can_move, g)

        # nearest living agent, for huddling
        d_ag = jnp.max(jnp.abs(pos[:, None] - pos[None]), axis=-1)
        d_ag = jnp.where(alive[None] & ~jnp.eye(n, dtype=bool), d_ag, 10 ** 6)
        mate_near = jnp.argmin(d_ag, axis=1)
        d_near = jnp.min(d_ag, axis=1)
        to_mate = _toward(pos, pos[mate_near], can_move, g)

        dp = jnp.max(jnp.abs(pos[:, None] - state.predators.pos[None]), axis=-1)
        danger = jnp.min(jnp.where(state.predators.alive[None], dp, 10 ** 6), axis=1) <= 3

        a = jnp.full(n, E.REST)
        fed = (state.satiety > 0.6) & (state.hydration > 0.6)
        # lowest priority first; later lines override
        a = jnp.where(state.satiety < 0.8, go_food, a)
        a = jnp.where(state.hydration < 0.8, go_water, a)
        a = jnp.where(mask[:, E.MATE] & fed & ~has_child, E.MATE, a)
        cold = state.body_temp < -0.35
        hot = state.body_temp > 0.35
        a = jnp.where(hot & ~in_shelter, go_shelter, a)
        a = jnp.where(hot & in_shelter, E.REST, a)
        a = jnp.where(cold & ~in_shelter, go_shelter, a)
        a = jnp.where(cold & in_shelter & (d_near > 1), to_mate, a)
        a = jnp.where(cold & in_shelter & (d_near <= 1), E.REST, a)
        # a mother forages only near her child, unless need is pressing
        pressing = (state.satiety < 0.35) | (state.hydration < 0.35)
        a = jnp.where(child_far & ~pressing, to_child, a)
        # Parents feed their hungry children (under 12): a fed parent gathers food, carries it to the
        # child and drops it where the child stands; the child picks it up with its own food skill.
        # (Trained agents must learn this; the script does it so the check measures what the world
        # allows, not what a script that never shares can do.)
        parent_of = (state.relatedness >= 0.45) & ((years[:, None] - years[None, :]) >= 14.0) \
            & alive[:, None] & alive[None, :] & ~jnp.eye(n, dtype=bool)                    # [i, j]: i parent of j
        hungry_kid = alive & (years < 12.0) & (years >= cfg.infant_years) & (state.satiety < 0.5)
        kid_d = jnp.where(parent_of & hungry_kid[None, :], jnp.max(jnp.abs(pos[:, None] - pos[None]), -1), 10 ** 6)
        kid = jnp.argmin(kid_d, axis=1)
        provider = (jnp.min(kid_d, axis=1) < 10 ** 6) & (state.satiety > 0.55) & ~pressing
        # grasp while the cell has plants, up to 0.5; then (or once the patch runs out) take it over.
        # Leaving with less matters: away from plants, the foraging branch eats what one carries.
        carried = state.inv[:, E.PLANT]
        carrying = (carried >= 0.5) | ((carried >= 0.05) & ~plants_here)
        with_kid = jnp.all(pos == pos[kid], axis=-1)
        give = jnp.where(state.hand != E.PLANT, E.SELECT_0 + E.PLANT,
               jnp.where(with_kid, E.RELEASE, _toward(pos, pos[kid], can_move, g)))
        fetch = jnp.where(plants_here & (state.hand == E.PLANT) & mask[:, E.GRASP], E.GRASP,
                jnp.where(plants_here, E.SELECT_0 + E.PLANT, go_food))
        a = jnp.where(provider, jnp.where(carrying, give, fetch), a)
        sleepy_time = (night & (state.sleepiness > 0.25)) | (state.sleepiness > 0.8)
        a = jnp.where(sleepy_time & ~pressing & ~((cold | hot) & ~in_shelter) & ~child_far, E.SLEEP, a)
        a = jnp.where(state.satiety < 0.25, go_food, a)
        a = jnp.where(state.hydration < 0.25, go_water, a)
        # alone, flee; with a companion beside, stand together (predators rarely attack a pair)
        a = jnp.where(danger & (d_near > 1), E.FLEE if cfg.skills else E.REST, a)
        # relieve oneself before it is pressing, away from the water (beside it only if it cannot wait)
        if cfg.excretion:
            beside_water = mask[:, E.DRINK]
            a = jnp.where(~danger & (((state.urge > 0.6) & ~beside_water) | (state.urge > 0.9)), E.RELIEVE, a)
        # asleep: sleep on until rested and day (or woken by danger or need)
        wake = (~night & (state.sleepiness < 0.2)) | (danger & (d_near > 1)) | pressing
        if cfg.excretion:
            wake = wake | (state.urge > 0.9)
        a = jnp.where(state.asleep, jnp.where(wake, E.REST, E.SLEEP), a)
        ok = mask[idx, a]
        return jnp.where(ok, a, E.REST).astype(jnp.int32)

    return act


def build_cfg(args):
    p = PRESETS[args.preset]
    cfg = EnvConfig(grid_size=p["grid"], n_agents=p["agents"], skills=p.get("slate") == "caveman",
                    plant_abundance=args.food, reward_mode="survival")
    if (args.world or p.get("world")) == "living":
        cfg = living_world(cfg, ecology=getattr(args, "ecology", True), fouling=getattr(args, "fouling", True))
    from lifesim.config import with_time_scale
    bio = getattr(args, "days_per_bio_year", None)
    cfg = with_time_scale(cfg, p.get("days_per_year", 40), bio if bio is not None else p.get("days_per_bio_year", 1.0))
    if p.get("reproduction"):
        cfg = with_reproduction(cfg, n_founders=p.get("founders"))
    cfg = with_threats(cfg, predators=getattr(args, "predators", 1.0), disease=getattr(args, "disease", 1.0),
                       hazards=getattr(args, "hazards", 1.0))
    from lifesim.config import with_tick_minutes
    minutes = getattr(args, "tick_minutes", None)
    cfg = with_tick_minutes(cfg, minutes if minutes is not None else p.get("tick_minutes", 30.0))
    if getattr(args, "patch_leave", None) is not None:
        cfg = dataclasses.replace(cfg, patch_leave=args.patch_leave)
    return cfg


def run(cfg, seed, days, knowledge="full", shocks=()):
    ticks = {}

    def tick_for(which):                              # the world under the shocks in force
        if which not in ticks:
            wc = threats.cfg_during(cfg, list(shocks), which)
            act = scripted_policy(wc, knowledge)

            @jax.jit
            def tick(s):
                a = act(s)
                s2, _, _, info = E.step(s, a, wc)
                return s2, info
            ticks[which] = tick
        return ticks[which]

    s = E.reset(jax.random.PRNGKey(seed), cfg)
    in_force = ()
    deaths, seasons, pop = collections.Counter(), collections.Counter(), []
    births, peak, last_alive_day, no_room = 0, int(s.alive.sum()), 0, 0
    born_here = np.zeros(cfg.n_agents, bool)          # slot's occupant was born in this world
    births_to_born_here = 0                           # children of mothers born here: a second generation
    for t in range(int(days * cfg.day_length)):
        now = threats.active(list(shocks), t / cfg.day_length)
        for i in set(now) - set(in_force):
            if shocks[i]["kind"] == "famine":
                s = s._replace(plants=s.plants * shocks[i]["strength"])
        in_force = now
        s, info = tick_for(in_force)(s)
        died = np.asarray(info.died)
        for i in np.flatnonzero(died):
            deaths[E.CAUSE_NAMES[int(np.asarray(info.death_cause)[i])]] += 1
            seasons[E.SEASON_NAMES[int(E.season_of(jnp.int32(t), cfg))]] += 1
        gave = np.asarray(info.gave_birth)
        births += int(gave.sum())
        no_room += int(np.asarray(info.no_room).sum())
        births_to_born_here += int((gave & born_here).sum())
        born_here |= np.asarray(info.born)
        alive = int(np.asarray(s.alive).sum())
        peak = max(peak, alive)
        if t % cfg.day_length == cfg.day_length - 1:
            pop.append(alive)
            if alive:
                last_alive_day = len(pop)
        if alive == 0:
            break
    return {"seed": seed, "days_lasted": last_alive_day if pop and pop[-1] == 0 else len(pop),
            "extinct": int(np.asarray(s.alive).sum()) == 0, "alive_end": int(np.asarray(s.alive).sum()),
            "peak": peak, "births": births, "births_to_mothers_born_here": births_to_born_here,
            "conceptions_blocked_by_slot_cap": no_room, "deaths": dict(deaths), "deaths_by_season": dict(seasons),
            "population_by_day": pop}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="society", choices=sorted(PRESETS))
    ap.add_argument("--world", default=None, choices=["classic", "living"])
    ap.add_argument("--ecology", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fouling", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--food", type=float, default=1.0, help="x plant abundance (scarcity)")
    ap.add_argument("--predators", type=float, default=1.0, help="x predator numbers (0 = harmless)")
    ap.add_argument("--disease", type=float, default=1.0, help="x infection chances")
    ap.add_argument("--hazards", type=float, default=1.0, help="x extreme weather, lightning, falls")
    ap.add_argument("--shock", action="append", default=[], metavar="KIND:DAY:DAYS:STRENGTH",
                    help="inject a crisis, e.g. famine:40:10:0.2 (lifesim/threats.py)")
    ap.add_argument("--days", type=float, default=1200,
                    help="1,200 = 30 years: at 40 days per year of age, long enough for a child to grow up "
                         "(600 days) and have children of its own")
    ap.add_argument("--tick-minutes", type=float, default=None, help="default: the preset's (1 for society)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--patch-leave", type=float, default=None,
                    help="seek-food leaves patches poorer than this x the known average (default 0.5; 0 = off)")
    ap.add_argument("--knowledge", default="full", choices=["full", "own"],
                    help="full: true map (upper bound); own: the given skills, from what each has seen")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = build_cfg(args)
    res = []
    for seed in range(args.seeds):
        r = run(cfg, seed, args.days, args.knowledge, [threats.parse_shock(x) for x in args.shock])
        res.append(r)
        pop = r["population_by_day"]
        print(f"seed {seed}: lasted {r['days_lasted']} days{' (extinct)' if r['extinct'] else ''}, "
              f"alive at end {r['alive_end']}, peak {r['peak']}, births {r['births']} "
              f"({r['births_to_mothers_born_here']} to mothers born here)", flush=True)
        print(f"  population every 10 days: {pop[9::10]}", flush=True)
        print(f"  deaths: {r['deaths']}; by season {r['deaths_by_season']}", flush=True)
    persisted = sum(not r["extinct"] for r in res)
    print(f"\nVERDICT: {persisted}/{len(res)} scripted bands persisted {args.days:g} days "
          f"({args.days / (cfg.year_length / cfg.day_length):.1f} years)")
    if args.out:
        Path(args.out).write_text(json.dumps({"config": vars(args), "runs": res}, indent=1))


if __name__ == "__main__":
    main()
