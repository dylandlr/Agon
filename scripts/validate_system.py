"""Does the system support emergent life as intended?

Learning is not tested here. Each check uses hand-scripted agents to show that a
social or technological behavior (1) is possible through the generic actions
alone and (2) actually pays off when performed. If either fails, no amount of
training could make the behavior emerge. A final audit lists everything the
reward pays for, to show none of these behaviors is rewarded directly.

    python scripts/validate_system.py            # writes VALIDATION.md
"""
from __future__ import annotations

import inspect
import random

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import analysis, env as E, social
from lifesim.config import EnvConfig
from lifesim.policy import make_network

RESULTS = []


def check(name, passed, detail):
    RESULTS.append((name, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def world(n, **kw):
    """Small calm world: no weather extremes, lightning or roaming animals; prime-age agents."""
    base = dict(grid_size=16, n_agents=n, climate_volatility=0.0, lightning_prob=0.0,
                season_temp=(18.0,) * 4, day_temp_swing=0.0, founder_spread=0)
    base.update(kw)
    cfg = EnvConfig(**base)
    s = E.reset(jax.random.PRNGKey(3), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   birth_age=jnp.full(n, int(25 * cfg.ticks_per_bio_year), jnp.int32),
                   genotype=s.genotype._replace(strength=jnp.ones(n), disease_resistance=jnp.ones(n)),
                   immunity=jnp.zeros(n))
    t = np.asarray(s.terrain)
    for r in range(t.shape[0]):
        for c in range(1, t.shape[1] - 1):
            if (t[r, c - 1:c + 2] != E.WATER).all():
                return cfg, s, (r, c - 1), (r, c), (r, c + 1)
    raise RuntimeError("no land triple")


def place(an, i, cell, hp):
    return an._replace(pos=an.pos.at[i].set(jnp.asarray(cell, jnp.int32)), alive=an.alive.at[i].set(True),
                       hp=an.hp.at[i].set(hp), rest=an.rest.at[i].set(0))


def run(cfg, s, actions_fn, ticks):
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    infos = []
    for t in range(ticks):
        s, _, _, info = step(s, jnp.asarray(actions_fn(t, s), jnp.int32))
        infos.append(info)
    return s, infos


# ------------------------------------------------------------------ 1. group hunting
def group_hunting():
    out = {}
    for hunters in (1, 2):
        cfg, s, west, mid, east = world(2, large_prey_flee_prob=0.0, wander_prob=0.0)
        s = s._replace(pos=jnp.asarray([west, east], jnp.int32), large=place(s.large, 0, mid, cfg.large_prey_hp))
        acts = [E.STRIKE_N + 2, E.STRIKE_N + 6 if hunters == 2 else E.REST]
        ticks = None
        step = jax.jit(lambda st, a: E.step(st, a, cfg))
        for t in range(60):
            s, _, _, info = step(s, jnp.asarray(acts, jnp.int32))
            if not bool(s.large.alive[0]):
                ticks = t + 1
                break
        out[hunters] = (ticks, float(s.meat[mid]))
    solo, pair = out[1][0], out[2][0]
    check("Group hunting pays", pair is not None and (solo is None or pair < solo / 2),
          f"a large animal falls in {pair} ticks to two hunters vs {solo or 'more than 60'} to one; "
          f"it leaves {out[2][1]:.1f} units of meat")


# ------------------------------------------------------------------ 2. fire, cooking, safe food
def fire_and_cooking():
    cfg, s, west, mid, east = world(1, raw_meat_infection=1.0)
    s = s._replace(pos=jnp.asarray([mid], jnp.int32), terrain=s.terrain.at[east].set(E.ROCK),
                   wood=jnp.zeros_like(s.wood), plants=jnp.zeros_like(s.plants),
                   inv=jnp.zeros_like(s.inv).at[0, E.WOOD].set(2.0).at[0, E.STONE].set(1.0),
                   hand=jnp.array([E.WOOD], jnp.int32))
    step = jax.jit(lambda st, a: E.step(st, a, cfg))
    s, *_ = step(s, jnp.array([E.PLACE_N + 2]))                  # wood onto the rock to the east
    s, *_ = step(s, jnp.array([E.SELECT_0 + E.STONE]))
    strikes = None
    for t in range(400):                                        # strike stone on rock beside the wood
        s, _, _, info = step(s, jnp.array([E.STRIKE_N + 2]))
        if float(s.fire[east]) > 0:
            strikes = t + 1
            break
    lit = strikes is not None
    check("Fire can be made with generic actions", lit,
          f"place wood on rock, take stone in hand, strike: fire after {strikes} strikes"
          if lit else "no fire after 400 strikes")
    if not lit:
        return
    cold_cfg, cold_s, *_ = world(1)
    temp_without = float(E.step(cold_s._replace(pos=jnp.asarray([mid], jnp.int32)), jnp.array([E.REST]), cold_cfg)[3].ambient_temp[0])
    s = s._replace(inv=s.inv.at[0, E.RAW_MEAT].set(1.0), hand=jnp.array([E.RAW_MEAT], jnp.int32),
                   fire=s.fire.at[east].set(500.0))
    s, _, _, info = step(s, jnp.array([E.RELEASE]))             # drop raw meat beside the fire
    warm = float(info.ambient_temp[0])
    for _ in range(20):
        s, *_ = step(s, jnp.array([E.REST]))
    cooked = float(s.cooked[mid])
    s, *_ = step(s, jnp.array([E.GRASP]))                       # takes the cooked meat, holds it
    s, _, _, info = step(s, jnp.array([E.EAT]))
    check("Heat cooks food, and cooked food is safe", cooked > 0.5 and bool(info.ate_cooked[0]) and not bool(info.newly_infected[0]),
          f"meat dropped by the fire became {cooked:.2f} units cooked in 20 ticks; eaten with "
          f"{'no' if not bool(info.newly_infected[0]) else 'an'} infection, though raw meat here infects every time")
    check("Fire warms", warm > temp_without + 10, f"{warm:.1f} C beside the fire vs {temp_without:.1f} C without")


# ------------------------------------------------------------------ 3. huddling
def huddling():
    temps = {}
    for together in (False, True):
        cfg, s, west, mid, east = world(2, season_temp=(0.0,) * 4, forest_fraction=0.0)
        far = tuple(np.argwhere(np.asarray(s.terrain) != E.WATER)[-1])
        s = s._replace(pos=jnp.asarray([west, mid if together else far], jnp.int32),
                       genotype=s.genotype._replace(insulation=jnp.ones(2)),
                       height=jnp.zeros_like(s.height))                 # flat ground: air really is 0 C
        s, _ = run(cfg, s, lambda t, st: [E.REST, E.REST], 100)
        temps[together] = float(s.body_temp[0])
    thr = EnvConfig().cold_threshold
    if temps[True] >= thr > temps[False]:
        note = f"Small, but here it is the difference between staying above the cold-damage threshold ({thr}) and falling below it"
    else:
        note = f"Small: both are {'above' if temps[False] >= thr else 'below'} the cold-damage threshold ({thr})"
    check("Huddling keeps warm", temps[True] > temps[False] + 0.03,
          f"body temperature after 100 ticks at 0 C: {temps[True]:+.2f} together vs {temps[False]:+.2f} alone. {note}")


# ------------------------------------------------------------------ 4. predator deterrence
def deterrence():
    attacks = {}
    for together in (False, True):
        cfg, s, west, mid, east = world(2, predator_attack_prob=1.0, predator_rest_ticks=0)
        far = tuple(np.argwhere(np.asarray(s.terrain) != E.WATER)[-1])
        s = s._replace(pos=jnp.asarray([mid, east if together else far], jnp.int32),
                       predators=place(s.predators, 0, west, cfg.predator_hp))
        n_att = 0
        step = jax.jit(lambda st, a: E.step(st, a, cfg))
        for _ in range(20):
            s = s._replace(health=jnp.ones(2), predators=place(s.predators, 0, west, cfg.predator_hp))
            s, _, _, info = step(s, jnp.array([E.REST, E.REST]))
            n_att += int(info.attacked_by_predator[0])
        attacks[together] = n_att
    check("Standing together deters predators", attacks[True] == 0 and attacks[False] > 10,
          f"attacks on an agent beside a predator over 20 ticks: {attacks[False]} alone, {attacks[True]} with a companion")


# ------------------------------------------------------------------ 5. social learning transmits behavior
def transmission():
    import optax
    cfg = EnvConfig(grid_size=16, n_agents=2)
    net = make_network(cfg, hidden=64, encoder="mlp", messages=True, conv_impl="patch")
    A, M = E.N_ACTIONS, E.MSG_LEN * E.VOCAB
    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    states = [E.reset(k, cfg) for k in keys]
    obs = jnp.stack([E.observe(st, cfg)[1] for st in states])          # agent 1 is the demonstrator
    pos_d = jnp.stack([st.pos[1] for st in states])
    pos_o = jnp.stack([st.pos[0] for st in states])
    seen = social.restrict_to_observer_view(obs, pos_d, pos_o, jnp.zeros(64, bool), cfg)
    masks = jnp.stack([E.action_mask(st, cfg)[1] for st in states])
    target = jnp.where(masks[:, E.REST], E.REST, 0)                    # the demonstrator always rests
    full = jnp.concatenate([masks, jnp.ones((64, M), bool)], -1)
    params = net.init(jax.random.PRNGKey(1), jnp.zeros((64, 64)), (seen[None], jnp.ones((1, 64), bool), full[None]))
    def p_target(p):
        _, logits, _ = net.apply(p, jnp.zeros((64, 64)), (seen[None], jnp.ones((1, 64), bool), full[None]))
        return jnp.exp(jax.nn.log_softmax(logits[0, :, :A]))[jnp.arange(64), target].mean()
    def loss(p):
        _, logits, _ = net.apply(p, jnp.zeros((64, 64)), (seen[None], jnp.ones((1, 64), bool), full[None]))
        return -jax.nn.log_softmax(logits[0, :, :A])[jnp.arange(64), target].mean()
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    before = float(p_target(params))
    g = jax.jit(jax.grad(loss))
    for _ in range(150):
        upd, opt = tx.update(g(params), opt)
        params = optax.apply_updates(params, upd)
    after = float(p_target(params))
    check("Watching transmits behavior", after > 0.8 and after > before + 0.5,
          f"an observer that only watches (no reward) picks the demonstrated action with probability "
          f"{before:.2f} before and {after:.2f} after, seeing only what it could see")


# ------------------------------------------------------------------ 6. names and meaning are detectable
def language_detection():
    rng = random.Random(0)
    def frames(rule):
        out = []
        for t in range(300):
            spk, tgt = rng.choice([0, 1]), rng.choice([2, 3, 4])
            agents = [{"r": 30, "c": 30, "alive": True, "msg": [0, 0, 0]} for _ in range(5)]
            agents[spk] = {"r": 0, "c": 0, "alive": True, "msg": rule(tgt)}
            agents[tgt] = {"r": 0, "c": 1, "alive": True, "msg": [0, 0, 0]}
            out.append({"t": t, "agents": agents, "pred": [[0, 2]] if tgt == 2 else [], "large": [], "fire": []})
        return {"frames": out, "view_radius": 5}
    named = analysis.language_metrics(frames(lambda tgt: [tgt + 1, 1, 1]))
    noise = analysis.language_metrics(frames(lambda tgt: [rng.randint(1, 15), rng.randint(0, 15), rng.randint(0, 15)]))
    ok = len(named["names_given"]) == 3 and not noise["names_given"] and named["context_bits"]["predator"] > 0.3
    check("Invented names and meaning would be detected", ok,
          f"shared words for individuals: {len(named['names_given'])} names found "
          f"({', '.join(g['name'] for g in named['names_given'])}); random speech: {len(noise['names_given'])}; "
          f"messages tied to a predator carry {named['context_bits']['predator']:.2f} bits vs "
          f"{noise['context_bits']['predator']:.2f} for noise")


# ------------------------------------------------------------------ 7. audit: what does reward pay for?
def reward_audit():
    src = inspect.getsource(E.step)
    reward_block = src[src.index("    reward = ("):src.index("# ---------------------------------------------------------- outcomes")]
    terms = ["alive_reward", "death_penalty", "pain_aversion", "novelty_reward", "touch_reward", "moral"]
    found = [t for t in terms if t in reward_block]
    forbidden = ["kills", "gifts", "fire", "cooked", "spoke", "huddle", "deterred", "company", "large"]
    leaks = [w for w in forbidden if w in reward_block]
    check("Reward pays for no social or technological behavior", not leaks,
          f"reward terms: staying alive, death, pain, novelty, touch, moral sentiments ({', '.join(found)}). "
          f"Nothing rewards hunting, giving, fire, cooking, speech, huddling, deterrence or company in general; "
          f"two inborn sentiments do reward answering a past act in kind (reciprocity: giving to someone who "
          f"gave to you; vengefulness: striking someone who struck you)"
          + (f"; LEAK: {leaks}" if leaks else ""))


if __name__ == "__main__":
    for fn in (group_hunting, fire_and_cooking, huddling, deterrence, transmission, language_detection, reward_audit):
        try:
            fn()
        except Exception as exc:                                  # a crash is a failure, not a skip
            check(fn.__name__, False, f"crashed: {exc!r}")
    passed = sum(p for _, p, _ in RESULTS)
    lines = ["# System validation", "",
             "Each check shows a behavior is possible with generic actions and pays off; none is trained or rewarded directly.", "",
             f"**{passed} of {len(RESULTS)} checks pass.**", "",
             "| Check | Result | Evidence |", "| --- | --- | --- |"]
    lines += [f"| {n} | {'Pass' if p else 'FAIL'} | {d} |" for n, p, d in RESULTS]
    open("VALIDATION.md", "w").write("\n".join(lines) + "\n")
    print(f"\n{passed} of {len(RESULTS)} checks pass; wrote VALIDATION.md")
