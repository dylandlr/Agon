"""How much does the critic's TARGET know about how long an agent has left?

    python scripts/probe_target_info.py --seeds 8

The critic is trained on `targets = adv + value`. If those targets carry little information about
remaining lifetime, then no critic -- however capable, and the architecture fits remaining lifetime
to train R^2 0.995 -- learns to predict survival, because the thing it is fitted to does not
contain it. So the quantity to measure is R^2(target, remaining lifetime), per gae_lambda.

This probe was written to test a claim in docs/MEASUREMENT.md 5d -- that raising `gae_lambda` works
by putting lifetime information into the target, and that the usual variance objection is weak here
because the verified return identity is affine in lifetime. It measured the opposite, and 5d now
records that. The identity is proven for a WHOLE-LIFE return from reset, with Phi_0 the initial
potential; a windowed mid-life GAE target also carries Phi(s_t) and a gamma^k * V(cut) bootstrap,
so it is not affine in remaining lifetime and the argument does not transfer.

Method: roll a random policy until every agent dies, recording the real env reward and the real
network's value each tick. The network is FRESHLY INITIALISED on purpose: the measured trained
critic is near-constant (V moves 0.2% of the return range from comfortable to lethal cold), so a
near-constant V is the regime the run is stuck in. The probe prints the spread of its own V so that
premise is checked, not assumed. GAE is then computed over 176-tick windows (--steps 128 plus the
carried memory tail of 48), bootstrapping from V at each cut, which is what the learner does.

Reported per lambda: R^2 of the target on remaining lifetime; R^2 on the wellbeing potential Phi;
sd(target) and sd(adv); and R^2 restricted to windows that contain the agent's own death.

Limits, which matter: a random policy and an untrained V. The probe measures the signal the critic
is handed at INITIALISATION, not at the fixed point. As V becomes informative its bootstrap carries
real information, and that could change the ordering. This probe cannot see that, and no claim here
should be read as one about the end of training.
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lifesim.config as C                      # noqa: E402
import lifesim.env as E                         # noqa: E402
from lifesim.policy import make_network         # noqa: E402
from lifesim.ppo_multi import MAConfig, gae     # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--agents", type=int, default=8)
ap.add_argument("--ticks", type=int, default=900, help="cap; the roll stops when everyone is dead")
ap.add_argument("--steps", type=int, default=128, help="rollout length, as --steps in training")
ap.add_argument("--memory-span", type=int, default=48, help="carried tail; the window is steps + this")
ap.add_argument("--lambdas", type=float, nargs="+", default=[0.95, 0.99, 0.995, 1.0])
ap.add_argument("--gamma", type=float, default=MAConfig().gamma)
ap.add_argument("--hidden", type=int, default=128)
args = ap.parse_args()

N, HID = args.agents, args.hidden
WINDOW = args.steps + min(args.memory_span, args.steps)
cfg = C.EnvConfig(n_agents=N, grid_size=32, reward_mode="survival", max_steps=50_000)
A = E.N_ACTIONS + E.EXTRA_LOGITS
net = make_network(cfg, hidden=HID, encoder="conv", messages=True, world_model=True)
step = jax.jit(lambda s, a: E.step(s, a, cfg))
obs_f = jax.jit(lambda s: E.observe(s, cfg))
phi_f = jax.jit(lambda s: E.wellbeing(s.health, s.satiety, s.hydration, s.energy, s.body_temp,
                                      s.stress, s.genotype.pain_aversion, cfg, s.sleepiness))
mk = lambda o, reset: (o[None], jnp.full((1, N), reset), jnp.ones((1, N, A), bool))
params = net.init(jax.random.PRNGKey(1), jnp.zeros((N, HID)), mk(jnp.zeros((N, E.obs_dim(cfg))), True))
apply = jax.jit(lambda p, c, o, r: net.apply(p, c, mk(o, r)))


def roll(seed):
    key = jax.random.PRNGKey(seed)
    s = E.reset(key, cfg)
    carry = jnp.zeros((N, HID))
    R, V, D, AL, PHI = [], [], [], [], []
    death = np.full(N, -1)
    for t in range(args.ticks):
        PHI.append(np.asarray(phi_f(s)))
        o, alive = obs_f(s), np.asarray(s.alive)
        carry, _, v = apply(params, carry, o, t == 0)
        key, k = jax.random.split(key)
        s, r, _, info = step(s, jax.random.randint(k, (N,), 0, E.N_ACTIONS))
        d = np.asarray(info.died)
        R.append(np.asarray(r)); V.append(np.asarray(v)[0]); D.append(d); AL.append(alive)
        for i in range(N):
            if d[i] and death[i] < 0:
                death[i] = t
        if not np.asarray(s.alive).any():
            break
    return (np.asarray(R), np.asarray(V), np.asarray(D), np.asarray(AL), np.asarray(PHI),
            death, len(R))


def advantages(reward, value, died, lam):
    """GAE over 176-tick windows, bootstrapping from V after each cut -- the learner's own path.

    gae()'s scan is seeded with (0, last_val), so last_val IS the window bootstrap. `boot` is a
    different mechanism: the value to continue from when a WORLD is cut off by the clock.
    """
    T = reward.shape[0]
    g = jnp.full((N, 1), args.gamma)
    cuts = list(range(0, T, WINDOW)) + [T]
    adv = np.zeros((T, N), np.float64)
    for a, b in zip(cuts[:-1], cuts[1:]):
        last = jnp.asarray(value[b][:, None]) if b < T else jnp.zeros((N, 1))
        x = gae(jnp.asarray(reward[a:b])[:, :, None], jnp.asarray(value[a:b])[:, :, None],
                jnp.asarray(died[a:b].astype(np.float32))[:, :, None],
                jnp.zeros((b - a, N, 1)), last, g, lam)
        adv[a:b] = np.asarray(x)[:, :, 0]
    return adv


def r2(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(y) < 10 or y.std() < 1e-12 or x.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


keys = ("r2_life", "r2_phi", "sd_t", "sd_a", "r2_life_deathwin")
acc = {lam: {k: [] for k in keys} for lam in args.lambdas}
vlo, vhi, shares = [], [], []
print(f"window {WINDOW} ticks (steps {args.steps} + tail {min(args.memory_span, args.steps)}), "
      f"gamma {args.gamma}, {N} agents, {args.seeds} seeds\n", flush=True)
for seed in range(args.seeds):
    reward, value, died, alive, phi, death, T = roll(seed)
    left = np.full((T, N), -1.0)
    for i in range(N):
        if death[i] >= 0:
            left[:, i] = death[i] - np.arange(T)
    keep = (alive[:T] > 0) & (left >= 0)                    # alive, and lifetime not censored
    vlo.append(value[keep].min()); vhi.append(value[keep].max())

    cuts = list(range(0, T, WINDOW)) + [T]
    nwin = ndeath = 0
    deathwin = np.zeros((T, N), bool)
    for a, b in zip(cuts[:-1], cuts[1:]):
        for i in range(N):
            if keep[a:b, i].any():
                nwin += 1
                if died[a:b, i].any():
                    ndeath += 1
                    deathwin[a:b, i] = True
    shares.append(ndeath / max(nwin, 1))
    print(f"  seed {seed}: {T} ticks, {int((death >= 0).sum())}/{N} died, lifetimes "
          f"{death[death >= 0].min()}-{death[death >= 0].max()}, {int(keep.sum())} agent-ticks",
          flush=True)

    for lam in args.lambdas:
        adv = advantages(reward, value, died, lam)
        tgt = adv + value[:T]
        k2 = keep & deathwin
        acc[lam]["r2_life"].append(r2(left[keep], tgt[keep]))
        acc[lam]["r2_phi"].append(r2(phi[:T][keep], tgt[keep]))
        acc[lam]["sd_t"].append(tgt[keep].std())
        acc[lam]["sd_a"].append(adv[keep].std())
        acc[lam]["r2_life_deathwin"].append(r2(left[k2], tgt[k2]) if k2.sum() > 10 else np.nan)

print(f"\nV at fresh init: {min(vlo):.4f} to {max(vhi):.4f} (spread {max(vhi) - min(vlo):.4f})"
      "  -- near-constant, the regime the trained critic was measured in")
print(f"176-tick agent-windows containing the agent's death: {100 * np.mean(shares):.0f}%\n")
print(f"{'lambda':>7}{'R2 life':>10}{'R2 Phi':>9}{'sd(target)':>12}{'sd(adv)':>10}{'R2 life | death':>17}")
for lam in args.lambdas:
    a = acc[lam]
    print(f"{lam:>7}{np.nanmean(a['r2_life']):>10.4f}{np.nanmean(a['r2_phi']):>9.4f}"
          f"{np.mean(a['sd_t']):>12.4f}{np.mean(a['sd_a']):>10.4f}"
          f"{np.nanmean(a['r2_life_deathwin']):>17.4f}")

base, top = args.lambdas[0], args.lambdas[-1]
d = np.array(acc[base]["r2_life"]) - np.array(acc[top]["r2_life"])
if len(d) > 2 and np.isfinite(d).all():
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\npaired, R2 life at lambda {base} minus lambda {top}: mean {d.mean():+.4f} "
          f"t {d.mean() / se:.2f} df {len(d) - 1}, {int((d > 0).sum())}/{len(d)} seeds favour {base}")
print("\nR2 life = share of the target explained by remaining lifetime alone. Raising lambda does "
      "NOT raise it;\nsd(target) rises while R2 falls, so lambda buys variance, not signal. See "
      "docs/MEASUREMENT.md 0 and 5d.")
