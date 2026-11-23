"""Can a TRAINED encoder learn the hydration label that a random-init trunk cannot?

    python scripts/probe_trained_trunk.py --seeds 4 --epochs 8

docs/MEASUREMENT.md 0 records that `hydration` change at h=24 is essentially unpredictable from the
carried GRU state (within-life R^2 0.024, or 0.074 given the realized actions), while `body_temp`
change reaches 0.528 on a *smaller* label variance. That result survived three challenges, and the
surviving explanation is about the representation rather than the label: hydration change needs a
conjunction -- choosing `drink` while standing on water -- that a randomly initialised trunk does
not carry, where temperature is driven by ambient conditions the observation window supplies
directly.

That explanation is testable without a trained checkpoint: train the real network, supervised, on
the label itself, and see whether the encoder acquires it.

  recovers hydration  -> the random-trunk figure was a representation limit, and C's spec can
                         include a hydration head
  does not            -> hydration is not learnable at this horizon under this policy, and C
                         should exclude it

`body_temp` is the positive control: it must come out high, or the setup is broken rather than the
label being hard. The split holds out WHOLE agent-lifetimes, because a random split leaks across
near-identical consecutive agent-ticks and a temporal split measures hydration's own monotone drift
-- both were real bugs in earlier versions of this measurement.

Limits: still a random policy, so how often the agent drinks is exogenous. This asks whether the
architecture CAN represent the label given dense supervision, not whether RL would deliver it.
"""
import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lifesim.config as C                          # noqa: E402
import lifesim.env as E                             # noqa: E402
from lifesim.policy import DECODE_INDEX, make_network   # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--agents", type=int, default=8)
ap.add_argument("--ticks", type=int, default=900)
ap.add_argument("--horizon", type=int, default=24)
ap.add_argument("--stride", type=int, default=2, help="subsample ticks to keep the obs tensor in RAM")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--labels", nargs="+", default=["hydration", "temp"])
args = ap.parse_args()

N, HID, H = args.agents, args.hidden, args.horizon
cfg = C.EnvConfig(n_agents=N, grid_size=32, reward_mode="survival", max_steps=50_000)
A = E.N_ACTIONS + E.EXTRA_LOGITS
D = E.obs_dim(cfg)
NWIN = cfg.view_size ** 2 * E.N_OBS_CHANNELS
DEC = np.asarray(DECODE_INDEX) + NWIN
# DECODE_INDEX order, matched to named scalars by correlation in probe_aux_tasks.py
COL = {"satiety": 0, "hydration": 1, "energy": 2, "health": 3, "temp": 4, "stress": 5, "sleepy": 6}

net = make_network(cfg, hidden=HID, encoder="conv", messages=True, world_model=True)
step = jax.jit(lambda s, a: E.step(s, a, cfg))
obs_f = jax.jit(lambda s: E.observe(s, cfg))
mk = lambda x: (jnp.asarray(x)[None], jnp.ones((1, len(x)), bool), jnp.ones((1, len(x), A), bool))

print(f"collecting: {args.seeds} seeds x {N} agents, horizon {H}, stride {args.stride}", flush=True)
OBS, Y, GRP = [], [], []
for seed in range(args.seeds):
    key = jax.random.PRNGKey(seed)
    s = E.reset(key, cfg)
    ob, fe, al = [], [], []
    death = np.full(N, -1)
    for t in range(args.ticks):
        o = np.asarray(obs_f(s), np.float32)
        ob.append(o)
        fe.append(o[:, DEC])
        al.append(np.asarray(s.alive))
        key, k = jax.random.split(key)
        s, _, _, info = step(s, jax.random.randint(k, (N,), 0, E.N_ACTIONS))
        d = np.asarray(info.died)
        for i in range(N):
            if d[i] and death[i] < 0:
                death[i] = t
        if not np.asarray(s.alive).any():
            break
    T = len(ob)
    ob, fe, al = np.asarray(ob), np.asarray(fe), np.asarray(al)
    for i in range(N):
        if death[i] < 0:
            continue
        last = min(death[i], T - 1)
        ts = np.arange(0, last - H, args.stride)
        if len(ts) < 20:
            continue
        OBS.append(ob[ts, i])
        Y.append(fe[ts + H, i] - fe[ts, i])
        GRP.append(np.full(len(ts), seed * N + i))
    print(f"  seed {seed}: {T} ticks, {int((death >= 0).sum())}/{N} died", flush=True)

OBS = np.concatenate(OBS); Y = np.concatenate(Y); GRP = np.concatenate(GRP)
print(f"\n{len(OBS)} samples, {len(np.unique(GRP))} lifetimes, obs {OBS.shape[1]}, "
      f"{OBS.nbytes / 1e6:.0f} MB", flush=True)
assert np.isfinite(OBS).all() and np.isfinite(Y).all(), "non-finite inputs or labels"

# grouped split: whole lifetimes held out
gs = np.random.default_rng(0).permutation(np.unique(GRP))
n_tr = max(1, int(0.7 * len(gs)))
tr = np.isin(GRP, gs[:n_tr])
order = np.concatenate([np.flatnonzero(tr), np.flatnonzero(~tr)])
OBS, Y, GRP = OBS[order], Y[order], GRP[order]
cut = int(tr.sum())
print(f"grouped split: {n_tr}/{len(gs)} lifetimes train -> {cut} train / {len(Y) - cut} test samples")


def within(y, g):
    out = y.astype(np.float64).copy()
    for gg in np.unique(g):
        m = g == gg
        out[m] -= out[m].mean()
    return out


@jax.jit
def predict(p, xb):
    return net.apply(p, jnp.zeros((len(xb), HID)), mk(xb))[2][0, :]


def run(label):
    j = COL[label]
    y = within(Y[:, j], GRP)                    # level-free: the trunk learns trend, not level
    sd = y[:cut].std() + 1e-9
    yn = (y - y[:cut].mean()) / sd
    params = net.init(jax.random.PRNGKey(1), jnp.zeros((args.batch, HID)), mk(OBS[:args.batch]))
    opt = optax.adam(args.lr)
    st = opt.init(params)

    def loss_fn(p, xb, yb):
        return jnp.mean((net.apply(p, jnp.zeros((len(xb), HID)), mk(xb))[2][0, :] - yb) ** 2)

    @jax.jit
    def train_step(p, st, xb, yb):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(0)
    B = args.batch
    ntr = (cut // B) * B
    nte = ((len(yn) - cut) // B) * B
    best = -np.inf
    for ep in range(args.epochs):
        idx = rng.permutation(ntr)
        tot = 0.0
        for i in range(0, ntr, B):
            b = idx[i:i + B]
            params, st, l = train_step(params, st, OBS[b], yn[b])
            tot += float(l)
        pv = np.concatenate([np.asarray(predict(params, OBS[cut + i:cut + i + B]))
                             for i in range(0, nte, B)])
        yt = yn[cut:cut + len(pv)]
        r2 = 1 - ((yt - pv) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-12)
        best = max(best, r2)
        tr_pv = np.concatenate([np.asarray(predict(params, OBS[i:i + B])) for i in range(0, min(ntr, 2048), B)])
        ytr = yn[:len(tr_pv)]
        r2tr = 1 - ((ytr - tr_pv) ** 2).sum() / max(((ytr - ytr.mean()) ** 2).sum(), 1e-12)
        print(f"  {label:<10} epoch {ep:>2}  train mse {tot / (ntr // B):.4f}  "
              f"train R2 {r2tr:+.4f}  held-out R2 {r2:+.4f}", flush=True)
    return best


print()
res = {lab: run(lab) for lab in args.labels}
print(f"\n{'label':<12}{'best held-out R2, TRAINED encoder':>36}{'random trunk (probe_aux_tasks)':>34}")
REF = {"hydration": 0.024, "temp": 0.528, "energy": 0.374, "sleepy": 0.281}
for lab, v in res.items():
    print(f"{lab:<12}{v:>36.4f}{REF.get(lab, float('nan')):>34.3f}")
print("\ntemp is the positive control: if it is not high, the setup is broken, not the label hard.")
print("A trained encoder recovering hydration would mean the random-trunk figure was a")
print("representation limit and C may include a hydration head. Not recovering it means hydration")
print("is not learnable at this horizon under a random policy, and C should leave it out.")
