"""Which auxiliary labels would candidate C actually be able to learn?

    python scripts/probe_aux_tasks.py --seeds 6

docs/MEASUREMENT.md 5d now ranks C (auxiliary prediction heads) first. Its premise is that a
supervised head supplies state-dependent signal the critic's target lacks. But an earlier probe
fitted this architecture to remaining lifetime at train R^2 0.995 and held-out R^2 below 0.3 on a
grouped split -- memorization, not generalization. A head that memorizes teaches the trunk nothing,
so before writing C, measure which labels survive a held-out split.

The world model already has the head C would extend: `decode` predicts the CHANGE in the eight
own-state features at `DECODE_INDEX`, one step ahead. C's version is the same head at a longer
horizon (the proposal's "consequence models"). So the candidate labels are:

  remaining lifetime            the hard one, and the one V ultimately needs
  needs change at horizon h     what `decode` already does at h=1, extended

Instrument: roll a random policy, carry the GRU state properly, and take h_t -- a 128-dim random
nonlinear feature map of the agent's whole history, i.e. a reservoir. Ridge-regress each label from
h_t with a split GROUPED BY WHOLE AGENT-LIFETIME, and report held-out R^2. Alongside it, the same
regression from the eight current needs alone, as a triviality control: a label that the current
needs already predict teaches the trunk little, however well it scores.

Read the result asymmetrically. A frozen random trunk is a LOWER bound on what a trained trunk
could extract, so a high held-out R^2 supports C's premise for that label. A low one is
discouraging but not conclusive -- a trained trunk may find structure a random one misses.

Splitting: grouped by lifetime on purpose. A random split leaks, because consecutive agent-ticks
are near-identical observations with near-identical labels (it once inflated a reference R^2 from
-553 to +0.91). A temporal split instead measures the monotone drift of hydration. Both were
real bugs in earlier versions of this measurement.
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lifesim.config as C                          # noqa: E402
import lifesim.env as E                             # noqa: E402
from lifesim.policy import DECODE_INDEX, make_network   # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--agents", type=int, default=8)
ap.add_argument("--ticks", type=int, default=900)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--horizons", type=int, nargs="+", default=[1, 24, 48, 200])
ap.add_argument("--train-share", type=float, default=0.7)
args = ap.parse_args()

N, HID = args.agents, args.hidden
cfg = C.EnvConfig(n_agents=N, grid_size=32, reward_mode="survival", max_steps=50_000)
A = E.N_ACTIONS + E.EXTRA_LOGITS
NWIN = cfg.view_size ** 2 * E.N_OBS_CHANNELS
DEC = np.asarray(DECODE_INDEX) + NWIN
SCALARS = ("satiety", "hydration", "energy", "health", "body_temp", "sleepiness", "stress")

net = make_network(cfg, hidden=HID, encoder="conv", messages=True, world_model=True)
step = jax.jit(lambda s, a: E.step(s, a, cfg))
obs_f = jax.jit(lambda s: E.observe(s, cfg))
mk = lambda o, r: (o[None], jnp.full((1, N), r), jnp.ones((1, N, A), bool))
params = net.init(jax.random.PRNGKey(1), jnp.zeros((N, HID)), mk(jnp.zeros((N, E.obs_dim(cfg))), True))
apply = jax.jit(lambda p, c, o, r: net.apply(p, c, mk(o, r)))


def roll(seed):
    key = jax.random.PRNGKey(seed)
    s = E.reset(key, cfg)
    carry = jnp.zeros((N, HID))
    H, F, S, AL, ACT = [], [], [], [], []
    death = np.full(N, -1)
    for t in range(args.ticks):
        o, alive = obs_f(s), np.asarray(s.alive)
        S.append(np.stack([np.asarray(getattr(s, k)) for k in SCALARS], 1))
        F.append(np.asarray(o)[:, DEC])
        carry, _, _ = apply(params, carry, o, t == 0)
        H.append(np.asarray(carry))
        AL.append(alive)
        key, k = jax.random.split(key)
        act = jax.random.randint(k, (N,), 0, E.N_ACTIONS)
        ACT.append(np.asarray(act))
        s, _, _, info = step(s, act)
        d = np.asarray(info.died)
        for i in range(N):
            if d[i] and death[i] < 0:
                death[i] = t
        if not np.asarray(s.alive).any():
            break
    return (np.asarray(H), np.asarray(F), np.asarray(S), np.asarray(AL),
            np.asarray(ACT), death, len(H))


def standardize(X, tr):
    """Scale X by TRAIN statistics and append a WORKING intercept column.

    The order matters and getting it wrong is a silent, severe bug. This probe first did the
    opposite -- append a ones column, then divide the lot by `X[tr].std(0) + 1e-9` -- which turns
    the intercept into (1-1)/1e-9 = exactly 0. The ridge was therefore fitted with NO INTERCEPT
    against an UNCENTRED target, which costs roughly (mean(y))^2 / var(y) of R^2 and costs nothing
    at all when the target happens to be centred. That is why the within-centred labels here were
    unaffected while `fraction of life elapsed` was not. See docs/MEASUREMENT.md 0.

    Columns with no train variance are zeroed instead of divided by an epsilon, which would
    otherwise amplify their rounding noise by ~1e9 into the Gram matrix.
    """
    X = np.asarray(X, np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    live = sd > 1e-8
    Z = np.zeros_like(X)
    Z[:, live] = (X[:, live] - mu[live]) / sd[live]
    return np.concatenate([Z, np.ones((len(X), 1))], 1)


def _one_split(X, y, grp, split):
    """Held-out R^2 for ONE split, with whole groups held out and the penalty tuned inside train.

    The split is derived from `split` alone, never from a shared generator. An earlier version
    threaded one advancing rng through every call, so each label was scored on a DIFFERENT split
    and the comparisons between labels were unpaired -- which moved a figure by 0.03, the same size
    as the differences being read off it.
    """
    gs = np.random.default_rng(split).permutation(np.unique(grp))
    n_tr = max(1, int(args.train_share * len(gs)))
    tr, te = np.isin(grp, gs[:n_tr]), np.isin(grp, gs[n_tr:])
    if te.sum() < 20 or tr.sum() < 20 or y[tr].std() < 1e-9:
        return float("nan")
    inner = gs[:n_tr]
    cut = max(1, int(0.75 * len(inner)))
    itr, iva = np.isin(grp, inner[:cut]), np.isin(grp, inner[cut:])
    Fs = standardize(X, tr)

    def fit(m, a):
        return np.linalg.solve(Fs[m].T @ Fs[m] + a * np.eye(Fs.shape[1]), Fs[m].T @ y[m])

    best, best_a = -np.inf, 1.0
    if iva.sum() > 20:
        for a in (1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4):
            p = Fs[iva] @ fit(itr, a)
            sc = 1 - ((y[iva] - p) ** 2).sum() / max(((y[iva] - y[iva].mean()) ** 2).sum(), 1e-12)
            if sc > best:
                best, best_a = sc, a
    p = Fs[te] @ fit(tr, best_a)
    return float(1 - ((y[te] - p) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-12))


def ridge_grouped(X, y, grp, splits=5):
    """Mean held-out R^2 over `splits` fixed grouped splits. Every label sees the same splits, so
    labels are compared paired rather than each on its own luck of the draw."""
    v = [_one_split(X, y, grp, s) for s in range(splits)]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


HH, FF, SS, LEFT, GRP, AA = [], [], [], [], [], []
for seed in range(args.seeds):
    H, Fe, S, AL, ACT, death, T = roll(seed)
    left = np.full((T, N), -1.0)
    for i in range(N):
        if death[i] >= 0:
            left[:, i] = death[i] - np.arange(T)
    keep = (AL[:T] > 0) & (left >= 0)
    for i in range(N):
        m = keep[:, i]
        if m.sum() < 30:
            continue
        HH.append(H[:T][m, i]); FF.append(Fe[:T][m, i]); SS.append(S[:T][m, i])
        LEFT.append(left[m, i]); GRP.append(np.full(int(m.sum()), seed * N + i))
        AA.append(ACT[:T][m, i])
    print(f"  seed {seed}: {T} ticks, {int((death >= 0).sum())}/{N} died", flush=True)

H = np.concatenate(HH); Fe = np.concatenate(FF); S = np.concatenate(SS)
left = np.concatenate(LEFT); grp = np.concatenate(GRP)
act = np.concatenate(AA)
# one-hot the realized action, so a window's counts are a plain sum over it
A1 = np.zeros((len(act), E.N_ACTIONS), np.float64)
A1[np.arange(len(act)), act] = 1.0


def action_counts(src, h, g):
    """How many times each action was taken over [t, t+h), per source tick.

    The world model's head is ACTION-CONDITIONED -- decode(dyn(z, a)) -- so predicting a needs
    change without the actions asks a different and partly impossible question: hydration rises
    when the agent drinks, and under a random policy the future action is noise. An earlier version
    of this probe omitted them and recorded hydration as unlearnable.
    """
    out = np.zeros((len(src), E.N_ACTIONS), np.float64)
    pos = {gg: np.flatnonzero(grp == gg) for gg in np.unique(g)}
    cums = {gg: np.vstack([np.zeros(E.N_ACTIONS), np.cumsum(A1[idx], 0)]) for gg, idx in pos.items()}
    for r, (i, gg) in enumerate(zip(src, g)):
        idx = pos[gg]
        off = int(np.flatnonzero(idx == i)[0])
        out[r] = cums[gg][off + h] - cums[gg][off]
    assert np.allclose(out.sum(1), h), "action counts must sum to the horizon"
    return out
print(f"\n{len(H)} agent-ticks from {len(np.unique(grp))} lifetimes, trunk {H.shape[1]}-dim\n")

# label the eight decoded features empirically rather than guessing the observation layout
print("the eight features `decode` predicts, matched to named state scalars by correlation:")
for j in range(len(DECODE_INDEX)):
    c = [abs(np.corrcoef(Fe[:, j], S[:, k])[0, 1]) for k in range(len(SCALARS))]
    k = int(np.nanargmax(c))
    print(f"  DECODE_INDEX[{j}] = obs[{DECODE_INDEX[j]}]  ~ {SCALARS[k]:<11} |r| {c[k]:.3f}")

def between_share(y, grp):
    """Share of the label's variance that lies BETWEEN lifetimes rather than within them.

    This is the part no per-tick observation can supply when the lifetime is held out: under a
    random policy, how long an agent happens to live is mostly luck. A label whose variance is
    mostly between-lifetime makes a bad auxiliary target -- an MSE head chases an offset it cannot
    predict, which is gradient noise, and is exactly how a head ends up memorizing.
    """
    m = np.array([y[grp == g].mean() for g in np.unique(grp)])
    w = np.array([len(y[grp == g]) for g in np.unique(grp)], float)
    between = (w * (m - y.mean()) ** 2).sum() / w.sum()
    return float(between / max(y.var(), 1e-12))


def within(y, grp):
    """The label with each lifetime's own mean removed."""
    out = y.astype(np.float64).copy()
    for g in np.unique(grp):
        s = grp == g
        out[s] -= out[s].mean()
    return out


def fmt(mean_sd):
    m, sd = mean_sd
    return f"{m:>8.3f}+-{sd:.2f}" if np.isfinite(m) else f"{'n/a':>13}"


def report(label, X, y, g, ctrl=None, tail=""):
    ctl = ridge_grouped(ctrl, y, g) if ctrl is not None else (float("nan"), float("nan"))
    print(f"{label:<30}{between_share(y, g):>8.2f}  {fmt(ridge_grouped(X, y, g))}"
          f"  {fmt(ridge_grouped(X, within(y, g), g))}  {fmt(ctl)}{tail}")


print(f"\n{'label':<30}{'btwn':>8}  {'R2 all':>13}  {'R2 within-life':>13}  {'R2 ctrl':>13}"
      f"  {'+actions':>13}")
print("  (R2 within-life uses each held-out lifetime's own mean to define the within component, so")
print("   it measures within-life PREDICTABILITY, not a predictor you could deploy as-is.)")
report("remaining lifetime", H, left, grp, Fe)

for h in args.horizons:
    # change in each decoded feature h ticks ahead, within the same lifetime
    per_t = []
    for g in np.unique(grp):
        m = grp == g
        if m.sum() <= h:
            continue
        per_t.append((np.flatnonzero(m)[:-h], np.flatnonzero(m)[h:]))
    if not per_t:
        continue
    src = np.concatenate([a for a, _ in per_t]); dst = np.concatenate([b for _, b in per_t])
    g = grp[src]
    AC = action_counts(src, h, g)
    HA = np.concatenate([H[src], AC], 1)          # the trunk PLUS what the agent actually did
    bs, ra, rw, rc, rwa, sds = [], [], [], [], [], []
    for j in range(len(DECODE_INDEX)):
        y = (Fe[dst, j] - Fe[src, j]).astype(np.float64)
        bs.append(between_share(y, g))
        ra.append(ridge_grouped(H[src], y, g)[0])
        rw.append(ridge_grouped(H[src], within(y, g), g)[0])
        rc.append(ridge_grouped(Fe[src], y, g)[0])
        rwa.append(ridge_grouped(HA, within(y, g), g)[0])
        sds.append(float(within(y, g).std()))
    print(f"{f'needs change, h={h} (mean of 8)':<30}{np.nanmean(bs):>8.2f}  "
          f"{np.nanmean(ra):>13.3f}  {np.nanmean(rw):>13.3f}  {np.nanmean(rc):>13.3f}"
          f"  {np.nanmean(rwa):>13.3f}   n={len(src)}, {len(np.unique(g))} lifetimes")
    if len(args.horizons) > 1 and h == args.horizons[1]:        # per-scalar for one horizon
        for j, s in enumerate(("satiety", "hydration", "energy", "health", "temp", "stress",
                               "sleepy", "health2")[:len(DECODE_INDEX)]):
            print(f"    {s:<26}{bs[j]:>8.2f}  {ra[j]:>13.3f}  {rw[j]:>13.3f}  {rc[j]:>13.3f}"
                  f"  {rwa[j]:>13.3f}  sd {sds[j]:.5f}")

DRINK, SEEKW = E.ACTION_NAMES.index("drink"), E.ACTION_NAMES.index("seek_water")
print(f"\nvalidation: drink chosen on {100 * A1[:, DRINK].mean():.2f}% of ticks, seek_water on "
      f"{100 * A1[:, SEEKW].mean():.2f}% -- a random policy over {E.N_ACTIONS} actions. If a label's"
      "\n  within-life sd is ~0 the label barely varies, so its R2 is noise and the probe is"
      "\n  UNINFORMATIVE for it rather than evidence against it.")
print("\nGrouped split: whole agent-lifetimes held out; penalty tuned inside train only.")
print("between-life = share of the label's variance lying between lifetimes rather than within one.")
print("R2 within-life removes each lifetime's own mean, isolating structure a head could learn.")
print("R2 ctrl predicts the same label from the eight CURRENT needs: a triviality control.")
print("A frozen random trunk is a LOWER bound on a trained one, so a high score supports C's")
print("premise for that label; a low one is discouraging but not conclusive.")
