"""Which level-free formulation of "progression toward death" should candidate C's head predict?

    python scripts/probe_label_form.py --seeds 6

docs/MEASUREMENT.md 5d settles that C's primary label is progression toward death, and that it must
be LEVEL-FREE: the trunk tracks an agent's progression but cannot place its absolute level on an
unseen lifetime (R² within-life +0.479 against R² overall −2.139), and an MSE head on an absolute
label chases a per-lifetime offset no observation supplies -- which is how the earlier probe got
train R² 0.995 and held out below 0.3.

"Level-free" admits several formulations, and they are not equally learnable or equally
implementable inside the learner:

  within-centred lifetime   remaining lifetime minus this life's mean. Not computable online, and
                            not computable inside a 176-tick window either -- the window does not
                            know the life's mean. Measured here only as the reference point.
  fraction elapsed          t / T. Bounded [0,1], level-free by construction, known in hindsight.
  log remaining             log(1 + remaining), centred. Compresses the long tail.
  hazard at k               will this agent die within the next k ticks. Binary, bounded, needs
                            only a reverse scan over `died` inside the window, and is exactly the
                            "nearness to death" the critic needs. Scored by AUC, so 0.5 is chance.

Each is ridge- or logistic-regressed from the carried GRU state at fresh init -- a reservoir, hence
a LOWER bound on a trained trunk -- with whole agent-lifetimes held out and the penalty tuned inside
train, averaged over 5 fixed splits shared by every label.

The deciding criteria are learnability AND computability inside a rollout window. A label that
scores well but cannot be built from `traj.died` without the whole life is not implementable.
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lifesim.config as C                  # noqa: E402
import lifesim.env as E                     # noqa: E402
from lifesim.policy import make_network     # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--agents", type=int, default=8)
ap.add_argument("--ticks", type=int, default=900)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--hazard-k", type=int, nargs="+", default=[12, 24, 48, 96])
ap.add_argument("--train-share", type=float, default=0.7)
ap.add_argument("--splits", type=int, default=5)
args = ap.parse_args()

N, HID = args.agents, args.hidden
cfg = C.EnvConfig(n_agents=N, grid_size=32, reward_mode="survival", max_steps=50_000)
A = E.N_ACTIONS + E.EXTRA_LOGITS
net = make_network(cfg, hidden=HID, encoder="conv", messages=True, world_model=True)
step = jax.jit(lambda s, a: E.step(s, a, cfg))
obs_f = jax.jit(lambda s: E.observe(s, cfg))
mk = lambda o, r: (o[None], jnp.full((1, N), r), jnp.ones((1, N, A), bool))
params = net.init(jax.random.PRNGKey(1), jnp.zeros((N, HID)), mk(jnp.zeros((N, E.obs_dim(cfg))), True))
apply = jax.jit(lambda p, c, o, r: net.apply(p, c, mk(o, r)))

HH, LEFT, TOT, GRP = [], [], [], []
for seed in range(args.seeds):
    key = jax.random.PRNGKey(seed)
    s = E.reset(key, cfg)
    carry = jnp.zeros((N, HID))
    Hs, AL = [], []
    death = np.full(N, -1)
    for t in range(args.ticks):
        o, alive = obs_f(s), np.asarray(s.alive)
        carry, _, _ = apply(params, carry, o, t == 0)
        Hs.append(np.asarray(carry)); AL.append(alive)
        key, k = jax.random.split(key)
        s, _, _, info = step(s, jax.random.randint(k, (N,), 0, E.N_ACTIONS))
        d = np.asarray(info.died)
        for i in range(N):
            if d[i] and death[i] < 0:
                death[i] = t
        if not np.asarray(s.alive).any():
            break
    T = len(Hs)
    Hs, AL = np.asarray(Hs), np.asarray(AL)
    for i in range(N):
        if death[i] < 0:
            continue
        last = min(death[i], T - 1)
        m = (AL[:last + 1, i] > 0)
        ts = np.flatnonzero(m)
        if len(ts) < 30:
            continue
        HH.append(Hs[ts, i])
        LEFT.append((last - ts).astype(np.float64))
        TOT.append(np.full(len(ts), float(last + 1)))
        GRP.append(np.full(len(ts), seed * N + i))
    print(f"  seed {seed}: {T} ticks, {int((death >= 0).sum())}/{N} died", flush=True)

H = np.concatenate(HH); left = np.concatenate(LEFT)
tot = np.concatenate(TOT); grp = np.concatenate(GRP)
print(f"\n{len(H)} agent-ticks, {len(np.unique(grp))} lifetimes, trunk {H.shape[1]}-dim")
print(f"lifetimes {int(tot.min())}-{int(tot.max())} ticks, remaining {int(left.min())}-{int(left.max())}\n")
assert np.isfinite(H).all() and (left >= 0).all() and (tot > 0).all()


def splits_of(g, s):
    gs = np.random.default_rng(s).permutation(np.unique(g))
    n_tr = max(1, int(args.train_share * len(gs)))
    return np.isin(g, gs[:n_tr]), np.isin(g, gs[n_tr:])


def _fit(Fs, y, m, a):
    return np.linalg.solve(Fs[m].T @ Fs[m] + a * np.eye(Fs.shape[1]), Fs[m].T @ y[m])


def auc(y, p):
    """Rank AUC. 0.5 is chance; needs both classes present."""
    if y.min() == y.max():
        return float("nan")
    r = np.empty(len(p), float)
    r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(X, y, g, binary):
    """Mean held-out score over fixed grouped splits. R^2 for continuous, AUC for binary."""
    out = []
    F = np.concatenate([X, np.ones((len(X), 1))], 1).astype(np.float64)
    for s in range(args.splits):
        tr, te = splits_of(g, s)
        if te.sum() < 20 or tr.sum() < 20:
            continue
        gs_in = np.unique(g[tr])
        cut = max(1, int(0.75 * len(gs_in)))
        itr, iva = np.isin(g, gs_in[:cut]), np.isin(g, gs_in[cut:])
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Fs = (F - mu) / sd
        best, best_a = -np.inf, 1.0
        if iva.sum() > 20:
            for a in (1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4):
                p = Fs[iva] @ _fit(Fs, y, itr, a)
                sc = auc(y[iva], p) if binary else 1 - ((y[iva] - p) ** 2).sum() / max(
                    ((y[iva] - y[iva].mean()) ** 2).sum(), 1e-12)
                if np.isfinite(sc) and sc > best:
                    best, best_a = sc, a
        p = Fs[te] @ _fit(Fs, y, tr, best_a)
        sc = auc(y[te], p) if binary else 1 - ((y[te] - p) ** 2).sum() / max(
            ((y[te] - y[te].mean()) ** 2).sum(), 1e-12)
        if np.isfinite(sc):
            out.append(sc)
    return (float(np.mean(out)), float(np.std(out))) if out else (float("nan"), float("nan"))


def centre(y, g):
    o = y.copy()
    for gg in np.unique(g):
        m = g == gg
        o[m] -= o[m].mean()
    return o


print(f"{'formulation':<34}{'metric':>8}{'held-out':>16}{'in-window?':>13}")
rows = [
    ("within-centred remaining life", "R2", centre(left, grp), False,
     "no: needs the life's mean"),
    ("fraction of life elapsed", "R2", 1.0 - left / tot, False,
     "no: needs total T"),
    ("log(1+remaining), centred", "R2", centre(np.log1p(left), grp), False,
     "no: needs the life's mean"),
]
for lab, metric, y, binary, note in rows:
    m, sd = score(H, y, grp, binary)
    print(f"{lab:<34}{metric:>8}{m:>10.3f}+-{sd:.2f}   {note}")

for k in args.hazard_k:
    y = (left < k).astype(np.float64)              # dies within the next k ticks
    m, sd = score(H, y, grp, True)
    print(f"{f'hazard: dies within {k} ticks':<34}{'AUC':>8}{m:>10.3f}+-{sd:.2f}   "
          f"yes: reverse scan over died   (base rate {y.mean():.3f})")

print("\nR2 0 = no better than the mean; AUC 0.5 = chance. Frozen random trunk, so these are LOWER")
print("bounds on a trained one. Grouped by whole lifetime; penalty tuned inside train.")
print("'in-window?' is whether the learner can build the label from a 176-tick rollout alone --")
print("a label needing the whole life is not implementable in ppo_multi's update, however it scores.")
