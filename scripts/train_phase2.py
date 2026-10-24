"""Phase 2: N individuals with their own networks share a world (in B replica
worlds). Logs social behavior rates per 1000 living agent-ticks.

    python scripts/train_phase2.py --agents 8 --worlds 16 --ticks 400000 --out runs/p2
    python scripts/train_phase2.py --scarcity 0.4 --out runs/p2_scarce   # E1-style dial
    python scripts/train_phase2.py --preset society --ticks 200000 --worlds 64 --out runs/s0

--preset fills in a named set of settings (lifesim/presets.py); flags given explicitly still win.
Every run records the code version it was trained with; --resume refuses to continue a run
with different code, so results from different versions of the world are never mixed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax
import numpy as np

from lifesim import checkpoint as ckpt
from lifesim import env as E
from lifesim.config import EnvConfig, living_world, with_reproduction, with_tick_minutes, with_time_scale
from lifesim.threats import with_threats  # noqa: E402
from lifesim.ppo_multi import MAConfig, make_train
from lifesim import archetypes
from lifesim.presets import PRESETS, code_version


def gini(x) -> float:
    """Inequality of hunting credit across agents: 0 = equal, near 1 = one agent does it all."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--scarcity", type=float, default=1.0, help="plant abundance: scales starting stock and regrowth (1 = normal)")
    ap.add_argument("--encoder", choices=["conv", "mlp"], default="conv")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--trait-gamma", action="store_true")
    ap.add_argument("--no-agent-lr", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--msg-ent-coef", type=float, default=0.0005,
                    help="entropy bonus on messages (actions use 0.005)")
    ap.add_argument("--speech-attention", type=float, default=0.0,
                    help="extra chance of watching agents who speak (off: it inflates speech)")
    ap.add_argument("--teach-bonus", type=float, default=0.0)
    ap.add_argument("--no-sleep-learning", action="store_true",
                    help="learn from all experience, not only what sleep has consolidated (ablation)")
    ap.add_argument("--reproduction", action="store_true",
                    help="births: --agents is the number of slots, --founders start alive")
    ap.add_argument("--founders", type=int, default=0, help="founders with --reproduction (default half the slots)")
    ap.add_argument("--reward-mode", choices=["acts", "survival"], default="acts",
                    help="acts: original rewards for acts; survival: survival is the only reward, felt by disposition")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="prosociality: 0 own reward, 1 group-mean reward")
    ap.add_argument("--imitation", type=float, default=0.5, help="social learning weight; 0 = off")
    ap.add_argument("--imitation-bias", choices=["uniform", "success", "kin"], default="success")
    ap.add_argument("--imitate-all", action="store_true",
                    help="copy every watched action, not only ones that paid off")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", type=str, default="runs/phase2")
    ap.add_argument("--ckpt-every", type=int, default=50, help="updates between checkpoints")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint.pkl")
    ap.add_argument("--slate", choices=["blank", "caveman"], default="blank",
                    help="what agents start with: blank = primitive actions only; caveman = plus given "
                         "survival skills (seek water, seek food, flee)")
    ap.add_argument("--saint-rate", type=float, default=1e-6,
                    help="chance each founder or newborn is selfless toward anyone (rare altruism)")
    ap.add_argument("--consolidation-ticks", type=float, default=8.0,
                    help="deep sleep (per 30 minutes of game time) that must follow a step before it is "
                         "learned in full; less sleep, a share of it")
    ap.add_argument("--retention-coef", type=float, default=0.1,
                    help="how strongly each sleep pulls the policy back toward old stored moments")
    ap.add_argument("--channel", choices=["full", "none", "shuffled"], default="full",
                    help="message channel control: none = no one is heard; shuffled = messages heard are "
                         "other speakers', shuffled each tick (Lowe et al. 2019 controls)")
    ap.add_argument("--msg-net", choices=["branch", "stopgrad", "separate"], default="branch",
                    help="where speech and writing branch off the brain (policy.ActorCritic.msg_net)")
    ap.add_argument("--msg-frozen", action="store_true",
                    help="speech and writing never learn (their initial weights): the floor for speaker-side measures")
    ap.add_argument("--memory-report", action="store_true",
                    help="compile one update, print XLA's memory estimate for it, and exit")
    ap.add_argument("--agent-chunk", type=int, default=0,
                    help="compute the loss this many agents at a time (0: all at once); lowers memory, "
                         "same gradients, ~30%% slower. Try 8 on a T4 or a CPU")
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="base learning rate (each agent's is this x its lr_scale gene unless --no-agent-lr)")
    ap.add_argument("--ent-coef", type=float, default=0.005,
                    help="exploration: how strongly learning keeps choices varied (entropy bonus)")
    ap.add_argument("--world", choices=["classic", "living"], default="classic",
                    help="classic: grass, forest, rock, water, global weather; living: plus desert, tundra, "
                         "marsh and caves, moving clouds, local rain and snow, wind, and a climate cycle")
    ap.add_argument("--underground", action="store_true",
                    help="with --world living: caves under the land, entered at cave mouths (delve), with their "
                         "own formations, minerals, animals and darkness (EnvConfig.underground)")
    ap.add_argument("--downed", action=argparse.BooleanOptionalAction, default=True,
                    help="with --world living: the badly hurt fall and cannot stand (EnvConfig.downed); "
                         "--no-downed for the ablation")
    ap.add_argument("--grazing", action=argparse.BooleanOptionalAction, default=True,
                    help="with --world living: animals eat plants and their numbers follow food; predators "
                         "get hungry (EnvConfig.grazing); --no-grazing for the ablation")
    ap.add_argument("--clothing", action=argparse.BooleanOptionalAction, default=True,
                    help="with --world living: founders start with hides covering some of the body (different "
                         "for each), babies are born bare; it keeps heat in (EnvConfig.clothing)")
    ap.add_argument("--ecology", action=argparse.BooleanOptionalAction, default=True,
                    help="with --world living: greens, fruit, nuts and roots that grow, spread and run out, and "
                         "gathering that takes time (EnvConfig.ecology)")
    ap.add_argument("--fouling", action=argparse.BooleanOptionalAction, default=True,
                    help="with --world living: waste builds where people stay; drinking beside it risks "
                         "infection (EnvConfig.fouling)")
    ap.add_argument("--excretion", action=argparse.BooleanOptionalAction, default=False,
                    help="with --world living: an urge to relieve oneself, RELIEVE, soiling (EnvConfig.excretion). "
                         "Off by default: an optional complexity. The network's shape is the same either way, so a "
                         "run trained without it can be continued with it")
    ap.add_argument("--language", choices=["emergent", "grounded"], default="emergent",
                    help="emergent: symbols mean nothing given; grounded: a defined lexicon (env.LEXICON) whose "
                         "phrases about water and food with a direction are understood by listeners")
    ap.add_argument("--writing", action="store_true",
                    help="WRITE scratches one's message into the ground, read by anyone beside it")
    ap.add_argument("--archetypes", default="",
                    help="temperament profiles for the first founders, e.g. tyrant:1,saint:1 (lifesim/archetypes.py)")
    ap.add_argument("--world-model", action=argparse.BooleanOptionalAction, default=False,
                    help="a learned model of its own world that imagines a few ticks ahead before choosing "
                         "(ideas); on in the presets, --no-world-model for the ablation")
    ap.add_argument("--share-trunk", action="store_true",
                    help="one trunk (encoder, memory, world model) for the population, tied across the agents; "
    ap.add_argument("--entropy-target", type=float, default=0.0,
                    help="an entropy floor in nats (0 = off): the entropy bonus grows under it, shrinks above "
                         "(MAConfig.entropy_target); the validate run collapsed to 0.07 of 4.4 and stopped learning")
                         "policy, value, messages and trust stay each agent's own (MAConfig.share_trunk)")
    ap.add_argument("--predators", type=float, default=1.0,
                    help="x predator numbers and respawning (0 = predators that never attack)")
    ap.add_argument("--disease", type=float, default=1.0, help="x every chance of infection")
    ap.add_argument("--hazards", type=float, default=1.0,
                    help="x environmental danger: extreme weather, lightning, falls")
    ap.add_argument("--tick-minutes", type=float, default=30.0,
                    help="game minutes per tick: 30 is the calibration; 1 is RimWorld-like (a day of 1,440 "
                         "decisions, 67 m cells); every rate keeps its pace in game time (with_tick_minutes)")
    ap.add_argument("--days-per-year", type=int, default=40,
                    help="calendar: 40 compresses a year so winter is learnable; 365 is a real year")
    ap.add_argument("--days-per-bio-year", type=float, default=1.0,
                    help="days per year of age: 1 compresses a life into ~70 days; 365 is real time")
    ap.add_argument("--episode-years", type=float, default=1.0,
                    help="training worlds restart after this many years (or when everyone has died); "
                         "a cut-off by the clock is bootstrapped, not treated as death")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume even though the simulation code changed since the run started")
    pre, _ = ap.parse_known_args()
    if pre.preset:
        ap.set_defaults(**PRESETS[pre.preset])
    args = ap.parse_args()
    args.code_version = code_version()

    base = EnvConfig()
    # Scarcity scales both the starting plant stock and regrowth. Scaling regrowth
    # alone barely mattered: the starting stock outlasted most agents' lives.
    env_cfg = EnvConfig(grid_size=args.grid, n_agents=args.agents, plant_abundance=args.scarcity,
                        reward_mode=args.reward_mode, shaping_gamma=args.gamma,
                        skills=args.slate == "caveman", saint_rate=args.saint_rate,
                        archetypes=archetypes.parse(args.archetypes))
    if args.world == "living":
        env_cfg = living_world(env_cfg, underground=args.underground, downed=args.downed, grazing=args.grazing,
                               clothing=args.clothing, ecology=args.ecology, fouling=args.fouling,
                               excretion=args.excretion)
    elif args.underground:
        raise SystemExit("--underground needs --world living (the cave mouths are a biome)")
    import dataclasses as _dc
    env_cfg = _dc.replace(env_cfg, language_mode=args.language, writing=args.writing, channel=args.channel)
    env_cfg = with_time_scale(env_cfg, args.days_per_year, args.days_per_bio_year)
    if args.reproduction:
        env_cfg = with_reproduction(env_cfg, n_founders=args.founders or None)
    env_cfg = with_threats(env_cfg, predators=args.predators, disease=args.disease, hazards=args.hazards)
    env_cfg = with_tick_minutes(env_cfg, args.tick_minutes)
    env_cfg = dataclasses.replace(env_cfg, max_steps=int(round(args.episode_years * env_cfg.year_length)))
    cfg = MAConfig(num_worlds=args.worlds, num_steps=args.steps, total_ticks=args.ticks, lr=args.lr, agent_chunk=args.agent_chunk,
                   msg_net=args.msg_net, msg_frozen=args.msg_frozen,
                   consolidation_ticks=args.consolidation_ticks, retention_coef=args.retention_coef,
                   encoder=args.encoder, gamma=args.gamma, trait_gamma=args.trait_gamma,
                   per_agent_lr=not args.no_agent_lr, seed=args.seed,
                   num_minibatches=min(4, args.worlds), imitation_coef=args.imitation,
                   imitation_bias=args.imitation_bias, imitate_positive_only=not args.imitate_all,
                   alpha=args.alpha, msg_ent_coef=args.msg_ent_coef, ent_coef=args.ent_coef,
                   speech_attention=args.speech_attention, teach_bonus=args.teach_bonus,
                   sleep_learning=not args.no_sleep_learning, world_model=args.world_model,
                   share_trunk=args.share_trunk, entropy_target=args.entropy_target,
                   day_ticks=env_cfg.day_length, tick_scale=env_cfg.tick_minutes / 30.0,
                   memory_span=env_cfg.day_length)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    net, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(cfg.seed))
    if args.memory_report:
        ma = update.lower(runner, pop).compile().memory_analysis()
        gb = lambda x: round(x / 2 ** 30, 2)
        print(json.dumps({"temp_GiB": gb(ma.temp_size_in_bytes), "arguments_GiB": gb(ma.argument_size_in_bytes),
                          "output_GiB": gb(ma.output_size_in_bytes), "agents": args.agents, "worlds": args.worlds,
                          "agent_chunk": args.agent_chunk}))
        return
    history, acc, start = [], None, 0
    ck_path = out / "checkpoint.pkl"
    if args.resume and ck_path.exists():
        old = json.load(open(out / "history.json"))["args"].get("code_version") \
            if (out / "history.json").exists() else None
        if old and old != args.code_version and not args.allow_code_change:
            raise SystemExit(f"{out} was trained with code {old}, but this code is {args.code_version}. "
                             "Start a new run (a new --out), or pass --allow-code-change to mix versions.")
        state = ckpt.load(ck_path)
        runner, pop, start = state["runner"], state["pop"], int(state["update"])
        history = json.load(open(out / "history.json"))["history"] if (out / "history.json").exists() else []
        print(f"resumed from update {start}", flush=True)
    print(f"backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    t0, steps0 = time.time(), start
    for u in range(start, cfg.num_updates):
        runner, m = update(runner, pop)
        m = jax.tree_util.tree_map(np.asarray, m)
        acc = m if acc is None else jax.tree_util.tree_map(lambda a, x: a + x, acc, m)
        if (u + 1) % args.log_every == 0 or u == cfg.num_updates - 1:
            alive = max(float(acc["alive"]), 1.0)
            deaths = float(acc["deaths"])
            per_k = lambda x: round(1000 * float(x) / alive, 3)
            row = {
                "update": u + 1, "ticks": (u + 1) * cfg.num_steps,
                "agent_steps": (u + 1) * cfg.num_steps * cfg.num_worlds * args.agents,
                "mean_age_at_death": round(float(acc["age_at_death"]) / deaths, 1) if deaths else None,
                "deaths": {E.CAUSE_NAMES[c]: int(k) for c, k in enumerate(acc["cause"]) if c and k},
                "deaths_by_season": dict(zip(["spring", "summer", "autumn", "winter"],
                                             (int(k) for k in acc["deaths_by_season"]))),
                # Global reward: all agents' base rewards (no alpha mixing, no moral sentiments) per world per day
                # Mode-independent survival score: 0.01 per agent-tick alive, -1 per death, per world-day.
                # Comparable across reward modes and alpha, unlike global reward, whose terms depend on the mode.
                "survival_score_per_world_day": round((env_cfg.alive_reward * float(acc["alive"])
                                                       - env_cfg.death_penalty * deaths)
                                                      / max(float(acc["world_ticks"]), 1.0) * env_cfg.day_length, 4),
                "global_reward_per_world_day": round(float(acc["own_reward"]) / max(float(acc["world_ticks"]), 1.0)
                                                     * env_cfg.day_length, 4),
                # Group survival at the end of each world's episode (the time limit, or everyone dead)
                "episodes_ended": int(acc["world_ends"]),
                "share_alive_at_episode_end": round(float(acc["survivors_at_end"]) / (int(acc["world_ends"]) * args.agents), 3)
                                              if int(acc["world_ends"]) else None,
                "share_of_worlds_reaching_year_end": round(float(acc["full_years"]) / int(acc["world_ends"]), 3)
                                                     if int(acc["world_ends"]) else None,
                "per_1000_agent_ticks": {"strikes_on_agents": per_k(acc["hits"]),
                                         "gifts": per_k(acc["gifts"]),
                                         "touches": per_k(acc["touches"]),
                                         "messages": per_k(acc["messages"])},
                "fire_ladder_per_1000": {k: per_k(acc[k]) for k in
                                         ["near_fire", "fed_fire", "meat_to_heat", "ate_cooked",
                                          "pulled_burning_wood", "fires_by_spark",
                                          "fires_by_burning_wood"]},
                "lightning_fires": int(acc["lightning_fires"]),
                "moral_reward_per_1000": per_k(acc["moral_reward"]),
                "deterred_per_1000": per_k(acc["deterred"]),
                "in_company_share": round(float(acc["company"]) / alive, 3),
                "births": int(acc["births"]), "conceptions": int(acc["conceptions"]),
                "conceptions_blocked_by_slot_cap": int(acc.get("no_room", 0)),
                "matings_per_1000": per_k(acc["matings"]),
                "asleep_share_night": round(float(acc["asleep_night"]) / max(float(acc["alive_night"]), 1.0), 3),
                "asleep_share_day": round(float(acc["asleep_day"]) / max(alive - float(acc["alive_night"]), 1.0), 3),
                "woken_per_1000": per_k(acc["woken"]),
                "collapses_per_1000": per_k(acc["collapses"]),
                "skills_per_1000": {"seek_water": per_k(acc["skill_water"]), "seek_food": per_k(acc["skill_food"]),
                                    "flee": per_k(acc["skill_flee"]), "seek_shelter": per_k(acc["skill_shelter"])},
                "scarred_share": round(float(acc["scarred"]) / alive, 3),
                "lasting_injury_share": round(float(acc["lasting_injured"]) / alive, 3),
                "meat_eaten_per_1000": per_k(acc["meat_eaten"]),
                "large_prey_hits_per_1000": per_k(acc["large_prey_hits"]),
                "winter_huddle_share": round(float(acc["winter_huddle_ticks"]) / max(float(acc["winter_night_ticks"]), 1.0), 3),
                "hunting_gini": round(gini(acc["large_kills_by_agent"]), 3),
                "kills": {"small_prey": int(acc["kills"][0]), "large_prey": int(acc["kills"][1]),
                          "predator": int(acc["kills"][2])},
                "entropy": round(float(m["entropy"]), 3),
                "imitation_loss": round(float(m["imitation_loss"]), 3),
                "watched_fraction": round(float(m["watched_fraction"]), 3),
                "learned_share": round(float(m["learned_share"]), 3),
                # Sleep: share of new steps still waiting for sleep, how far the policy has drifted
                # from what it did at stored old moments, and how full the long-term store is
                "sleep_memory": {"waiting_share": round(float(m["waiting_share"]), 3),
                                 "retention_kl": round(float(m["retention_kl"]), 5),
                                 "long_term_filled": round(float(m["long_term_filled"]), 3)},
                # Per output (action, aim, said, written): share of updates clipped, and the KL moved
                "per_output": {name: {"clip_fraction": round(float(m["clip_fraction"][j]), 4),
                                      "approx_kl": round(float(m["approx_kl"][j]), 5)}
                               for j, name in enumerate(["action", "how", "said", "written"])},
                **({"world_model": {"own_state_error": round(float(m["wm_own"]), 5),
                                    "belief_error": round(float(m["wm_latent"]), 5),
                                    "reward_error": round(float(m["wm_reward"]), 5),
                                    "trust_in_ideas": round(float(m["trust"]), 4)}} if cfg.world_model else {}),
                "agent_sps": round((u + 1 - steps0) * cfg.num_steps * cfg.num_worlds * args.agents
                                   / (time.time() - t0), 1),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            acc = None
        if (u + 1) % args.ckpt_every == 0 or u == cfg.num_updates - 1:
            ckpt.save(ck_path, {"runner": runner, "pop": pop, "update": u + 1})
            json.dump({"args": vars(args), "history": history}, open(out / "history.json", "w"), indent=1)
    from flax import serialization
    (out / "params.msgpack").write_bytes(serialization.to_bytes(runner[0]))
    with open(out / "population.pkl", "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, pop), f)
    json.dump({"args": vars(args), "history": history}, open(out / "history.json", "w"), indent=1)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
