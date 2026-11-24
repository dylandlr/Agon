"""Candidate D's premise: do the 532 language dimensions crowd the 8 reward scalars out of
`feat_dense`'s 64-unit bottleneck?

    python scripts/probe_bottleneck.py --seeds 4

docs/MEASUREMENT.md 5d ranks D second and says plainly that its premise is unmeasured: "Interference
is unmeasured, so this candidate rests on a hypothesis nothing in 0 supports." A is the cautionary
example of what happens when a candidate is implemented on an argument instead -- the whole-life
return identity did not apply to a windowed mid-life target, and the experiment cost a cycle. So D
gets measured before it gets code.

`encode` splits the observation in two: an 11x11x64 window through three convolutions, and
everything else (924 dims) through ONE `nn.Dense(64)`. Inside those 924:

      8   the reward scalars, at DECODE_INDEX -- the only inputs survival depends on
    128   the rest of own body, inventory, hand, torch, time, weather
    532   heard (402) and read (130) symbols
    256   the mind map

D's claim is that the 532 swamp the 8. Four measurements, in order of how much they can settle:

M1  INPUT ENERGY SHARE. For a Dense layer dL/dW = x (x) delta, so the gradient reaching the row of
    W fed by input j has norm |x_j| * ||delta||, and ON ONE SAMPLE a block's share of ||dL/dW||_F^2
    is exactly its share of sum_j x_j^2 -- for any loss, at any point in training, because
    ||delta|| factors out. Over a DATASET that becomes an approximation: the reported share is
    sum_{j in B} E[x_j^2] / sum_j E[x_j^2], which equals the true average share only if ||delta||^2
    is uncorrelated with how x's energy is split between blocks. Treat it as the share for a
    typical tick, not a theorem about the run.

M2  LEAVE-ONE-BLOCK-OUT at the 64-d output. Zero a block, recompute relu(feat_dense(feats)), and
    report how much the activation moves. Non-analytic because of the relu, and it says whether the
    bottleneck's state is actually a function of the block or merely receives it.

M3  RECOVERABILITY -- the decisive test. Ridge-probe the 8 reward scalars and remaining lifetime
    from the 64-d output, against the 924-d input as a ceiling. If 64 dims preserve what the critic
    needs, the bottleneck destroys nothing and D's crowding premise is FALSE whatever M1 says: a
    gradient share is not a loss of information.

M4  THE ABLATION THAT WOULD CONFIRM IT. Re-run M3 with the language block zeroed. If recoverability
    rises once the 532 are gone, they were crowding. If it does not move, they were not.

Read it asymmetrically, as with the other probes. The weights here are at INITIALIZATION, so:
  - A high M3 kills D. Information preserved through a random 924->64 projection is preserved,
    and no amount of gradient share makes a well-conditioned projection lose it.
  - A low M3, or an M4 that moves, does NOT confirm D on its own -- a trained `feat_dense` can
    allocate its 64 units deliberately, which is the whole thing a random one cannot do. That
    needs a checkpoint, and the same asymmetry already decided hydration in probe_aux_tasks.
Note also that at init M2's and M1's shares coincide analytically (W is iid, so a block's
contribution variance is proportional to its input variance); M2 is reported as a check that the
decomposition is right, not as independent evidence.

Splitting is grouped by whole agent-lifetime, and the penalty is tuned inside train, for the
reasons probe_aux_tasks.py documents: a random split leaks badly between consecutive agent-ticks.
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
from lifesim.policy import DECODE_INDEX, DECODE_NAMES, make_network, sample_outputs   # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--agents", type=int, default=8)
ap.add_argument("--ticks", type=int, default=900)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--train-share", type=float, default=0.7)
ap.add_argument("--splits", type=int, default=5)
ap.add_argument("--silent", action="store_true",
                help="pass messages=None to E.step, as every earlier probe did: the 532 language "
                     "dimensions are then identically zero and D cannot be measured at all")
args = ap.parse_args()
SPEAK = not args.silent

N, HID = args.agents, args.hidden
cfg = C.EnvConfig(n_agents=N, grid_size=32, reward_mode="survival", max_steps=50_000)
A = E.N_ACTIONS + E.EXTRA_LOGITS
NWIN = cfg.view_size ** 2 * E.N_OBS_CHANNELS

# ---- the blocks of `feats`, which is obs[..., NWIN:] ----------------------------------------
LANG0 = E.OWN_FEATURES + 2 * E.N_MAT + 1 + 5 + E.N_WEATHER     # after inventory, hand, torch, time, weather
LANG1 = LANG0 + E.HEARD_FEATURES + E.READ_FEATURES             # == N_BODY_FEATURES
NFEAT = E.obs_dim(cfg) - NWIN
assert LANG1 == E.N_BODY_FEATURES, (LANG1, E.N_BODY_FEATURES)
SCAL = np.asarray(DECODE_INDEX)                                # the 8 reward scalars, inside [0, OWN_FEATURES)

BLOCKS = {
    "reward scalars (8)": SCAL,
    "other own/inv/time (128)": np.asarray([i for i in range(LANG0) if i not in set(SCAL.tolist())]),
    "language heard+read (532)": np.arange(LANG0, LANG1),
    "mind map (256)": np.arange(LANG1, NFEAT),
}
assert sum(len(v) for v in BLOCKS.values()) == NFEAT, "blocks must tile `feats` exactly"

net = make_network(cfg, hidden=HID, encoder="conv", messages=True, world_model=True)
step = jax.jit(lambda s, a, m: E.step(s, a, cfg, m))
obs_f = jax.jit(lambda s: E.observe(s, cfg))
mk = lambda o, r: (o[None], jnp.full((1, N), r), jnp.ones((1, N, A), bool))
params = net.init(jax.random.PRNGKey(1), jnp.zeros((N, HID)), mk(jnp.zeros((N, E.obs_dim(cfg))), True))
W = np.asarray(params["params"]["feat_dense"]["kernel"], np.float64)      # (924, 64)
B = np.asarray(params["params"]["feat_dense"]["bias"], np.float64)
assert W.shape == (NFEAT, 64), W.shape


def standardize(X, tr):
    """Scale X by TRAIN statistics and append a WORKING intercept column.

    The order matters and getting it wrong is a silent, severe bug. The pattern these probes used
    first -- append a ones column, then divide the lot by `X[tr].std(0) + 1e-9` -- turns the
    intercept into (1-1)/1e-9 = exactly 0. Every such ridge was therefore fitted with NO INTERCEPT
    against an UNCENTRED target, which costs about (mean(y))^2 / var(y) of R^2 and nothing at all
    when the target happens to be centred. It announced itself here as `food` reconstructed at
    R^2 -6.2 from an input set that contains the food column itself.

    Columns with no train variance are zeroed rather than divided by an epsilon: they carry no
    information, and `feats` has hundreds of them (unused inventory slots, unsaid symbols), each of
    which would otherwise have its rounding noise amplified ~1e9 into the Gram matrix.
    """
    X = np.asarray(X, np.float64)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    live = sd > 1e-8
    Z = np.zeros_like(X)
    Z[:, live] = (X[:, live] - mu[live]) / sd[live]
    return np.concatenate([Z, np.ones((len(X), 1))], 1)


def bottleneck(F, drop=None):
    """relu(feat_dense(feats)), optionally with one block of inputs zeroed."""
    X = F.astype(np.float64)
    if drop is not None:
        X = X.copy()
        X[:, drop] = 0.0
    return np.maximum(X @ W + B, 0.0)


def roll(seed, speak):
    """A uniform-random action policy, carrying the GRU state as training does.

    `speak` decides whether `messages` is passed to `E.step`. It matters more than it looks:
    `E.step(state, actions, cfg, messages=None)` defaults to silence, and EVERY probe written
    before this one left it at the default. So in all of them the 402 heard and 130 read dimensions
    were identically zero, and nothing could have been measured about them. With `speak` the
    symbols are sampled from the network's own message head at initialization -- which carries
    `silence_bias` 5.75, i.e. speech in roughly 9% of slots -- so the language channels are as
    sparse as a fresh run's, and the action distribution is unchanged. One variable moved.
    """
    key = jax.random.PRNGKey(seed)
    s = E.reset(key, cfg)
    carry = jnp.zeros((N, HID))
    F, AL = [], []
    said = 0.0
    death = np.full(N, -1)
    for t in range(args.ticks):
        o = obs_f(s)
        F.append(np.asarray(o)[:, NWIN:].astype(np.float32))
        AL.append(np.asarray(s.alive))
        carry, logits, _ = net.apply(params, carry, mk(o, t == 0))
        key, k, km = jax.random.split(key, 3)
        act = jax.random.randint(k, (N,), 0, E.N_ACTIONS)
        msg = None
        if speak:
            _, out = sample_outputs(km, logits[0])            # keep the uniform action, take the symbols
            msg = out
            # `E.step` gates a whole utterance on its FIRST symbol (`spoke_now = alive & msg[:,0]>0`),
            # so the speech rate is that slot alone. Counting non-silent slots across all 8 instead
            # reads ~0.86, because `silence_bias` biases only logit 0 and the other seven slots are
            # uniform -- but those slots are discarded for a silent speaker, so the number is
            # meaningless. This is the rate that reaches a listener.
            said += float((np.asarray(out)[:, 0] > 0)[np.asarray(s.alive)].mean())
        s, _, _, info = step(s, act, msg)
        d = np.asarray(info.died)
        for i in range(N):
            if d[i] and death[i] < 0:
                death[i] = t
        if not np.asarray(s.alive).any():
            break
    return np.asarray(F), np.asarray(AL), death, len(F), said / max(len(F), 1)


def _one_split(X, y, grp, split):
    """Held-out R^2 for ONE split, whole lifetimes held out, penalty tuned inside train.

    Derived from `split` alone so every (feature set, label) pair is scored on the SAME splits --
    the unpaired version of this moved a figure by 0.03 in probe_aux_tasks, the size of the
    differences being read off it.
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


def ridge(X, y, grp):
    v = [_one_split(X, y, grp, s) for s in range(args.splits)]
    v = [x for x in v if np.isfinite(x)]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


# ------------------------------------------------------------------ collect
FF, LEFT, GRP = [], [], []
print(f"rolling {args.seeds} seeds, messages {'sampled from the head' if SPEAK else 'OFF (silent)'}")
for seed in range(args.seeds):
    F, AL, death, T, said = roll(seed, SPEAK)
    # Remaining lifetime is NaN for an agent still alive at the end of the roll: that label's rows
    # are dropped from its regression only. M1, M2 and the eight scalar labels need no death at
    # all, so a roll where nobody dies still measures everything except the lifetime row.
    left = np.full((T, N), np.nan)
    for i in range(N):
        if death[i] >= 0:
            left[:death[i] + 1, i] = death[i] - np.arange(death[i] + 1)
    for i in range(N):
        m = AL[:T, i] > 0
        if m.sum() < 30:
            continue
        FF.append(F[:T][m, i]); LEFT.append(left[m, i])
        GRP.append(np.full(int(m.sum()), seed * N + i))
    print(f"  seed {seed}: {T} ticks, {int((death >= 0).sum())}/{N} died, "
          f"{said:.3f} of living agents spoke per tick", flush=True)

if not FF:
    sys.exit("no agent lived 30 ticks in any roll -- raise --ticks or --agents")
F = np.concatenate(FF); left = np.concatenate(LEFT); grp = np.concatenate(GRP)
print(f"\n{len(F)} alive agent-ticks from {len(np.unique(grp))} lifetimes "
      f"({int(np.isfinite(left).sum())} with a known remaining lifetime), "
      f"feats = {F.shape[1]} dims\n")

# ------------------------------------------------------------------ M1: input energy share
print("M1. share of feat_dense's weight gradient each block drives, for a typical tick")
print("    dL/dW = x (x) delta, so per sample a block's share of ||dL/dW||_F^2 is its share of")
print("    sum_j x_j^2 exactly; averaging over ticks assumes ||delta|| does not covary with the split")
e = (F.astype(np.float64) ** 2).mean(0)
tot = e.sum()
print(f"    {'block':<28}{'dims':>6}{'share':>9}{'per-dim':>12}")
for name, idx in BLOCKS.items():
    print(f"    {name:<28}{len(idx):>6}{e[idx].sum() / tot:>9.4f}{e[idx].mean():>12.3e}")

# ------------------------------------------------------------------ M2: leave-one-block-out
print("\nM2. leave-one-block-out at the 64-d output (relu, so not analytic)")
h_full = bottleneck(F)
nz = (h_full > 0).mean()
print(f"    full activation: ||h|| = {np.linalg.norm(h_full, axis=1).mean():.4f}, "
      f"{nz:.3f} of units active")
print(f"    {'block zeroed':<28}{'rel ||dh||':>12}{'corr(h_abl,h)':>16}{'units flipped':>15}")
for name, idx in BLOCKS.items():
    h = bottleneck(F, idx)
    rel = np.linalg.norm(h - h_full, axis=1).mean() / np.linalg.norm(h_full, axis=1).mean()
    cor = np.corrcoef(h.ravel(), h_full.ravel())[0, 1]
    flip = ((h > 0) != (h_full > 0)).mean()
    print(f"    {name:<28}{rel:>12.4f}{cor:>16.4f}{flip:>15.4f}")

# ------------------------------------------------------------------ M3 / M4: recoverability
h_nolang = bottleneck(F, BLOCKS["language heard+read (532)"])
sets = [("924-d in", F.astype(np.float64)),
        ("64-d out", h_full),
        ("64-d, lang=0", h_nolang)]

# --- M3: reconstruction. Is each reward scalar still THERE after the projection? ----------------
# Split on ROWS, not lifetimes, and deliberately so. This asks whether a FIXED linear map plus relu
# destroys a quantity -- the same W faces the same agents at training time, so there is no
# generalization question and a grouped split would answer a different one (it measures the
# per-lifetime level offset, which is what 0's "trunk learns the trend, not the level" is about
# and what sank `fraction of life elapsed`). Remaining lifetime below IS predictive, so it keeps
# the grouped split.
print("\nM3. reconstruction of each reward scalar after the bottleneck (row split, R^2)")
print(f"    {'scalar':<18}" + "".join(f"{n:>16}" for n, _ in sets) + f"{'in-var':>12}")
rows = np.random.default_rng(0).permutation(len(F))
n_tr = int(0.7 * len(F))
tr, va = rows[:int(0.8 * n_tr)], rows[int(0.8 * n_tr):n_tr]
te = rows[n_tr:]
for X in (s[1] for s in sets):
    if X.shape[1] + 1 > len(tr):
        print(f"    WARNING: {X.shape[1] + 1} features against {len(tr)} train rows -- "
              f"underdetermined, raise --ticks or --seeds")
        break


def recon(X, y):
    """R^2 of reconstructing y from X on held-out ROWS, penalty chosen on a validation slice.

    The penalty is tuned, not fixed: with 925 features the 924-d set is underdetermined at small
    --ticks, and a fixed tiny ridge then interpolates train and reads R^2 of -5000 on test. That
    is a property of the probe's scale, not of the bottleneck.
    """
    Z = standardize(X, tr)
    G = Z[tr].T @ Z[tr]
    rhs = Z[tr].T @ y[tr]
    best, bw = -np.inf, None
    for a in (1e-4, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4):
        w = np.linalg.solve(G + a * np.eye(Z.shape[1]), rhs)
        d = max(((y[va] - y[va].mean()) ** 2).sum(), 1e-12)
        sc = 1 - ((y[va] - Z[va] @ w) ** 2).sum() / d
        if sc > best:
            best, bw = sc, w
    d = max(((y[te] - y[te].mean()) ** 2).sum(), 1e-12)
    return float(1 - ((y[te] - Z[te] @ bw) ** 2).sum() / d)


for nm, j in zip(DECODE_NAMES, SCAL):
    y = F[:, j].astype(np.float64)
    print(f"    {nm:<18}" + "".join(f"{recon(X, y):>16.4f}" for _, X in sets)
          + f"{y.var():>12.2e}")

# --- M4: the predictive label, grouped by lifetime ----------------------------------------------
print(f"\nM4. remaining lifetime, grouped by lifetime, {args.splits} fixed splits "
      f"(train share {args.train_share})")
ok = np.isfinite(left)
if ok.sum() < 60 or len(np.unique(grp[ok])) < 3:
    print(f"    n/a: {int(ok.sum())} rows, {len(np.unique(grp[ok]))} completed lifetimes")
else:
    for nm, X in sets:
        m, sd = ridge(X[ok], left[ok], grp[ok])
        print(f"    {nm:<18}{m:>10.3f} ±{sd:.3f}")

print("""
How to read it. A high "64-d out" column kills D: a quantity that survives a random 924->64
projection is not being crowded out, whatever share of the gradient M1 gives the language block --
a gradient share is not a loss of information. "64-d, lang=0" above "64-d out" by more than the ±
is the only result that supports D, and even then only at initialization: a trained feat_dense
allocates its 64 units deliberately, which is the one thing a random one cannot do, so confirming
D would still need a checkpoint. Run with --silent to see what every earlier probe measured.""")
