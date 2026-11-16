"""Recurrent actor-critic, optionally with a learned world model that imagines.

Memory lives in the GRU state; invalid actions are masked in the logits. Each agent (or lineage)
owns its own copy of these parameters, stacked and vmapped.

World model (world_model=True). The GRU state h is the agent's belief about its situation. A
small dynamics network predicts the next belief from the current one and an action,
h' ~ dyn(h, a); heads read a belief as the reward it expects and as its own needs (food, water,
energy, health, warmth, stress, sleepiness, pain). All of it is learned from the agent's own experience
(see ppo_multi: world-model loss), so it can only imagine what its own life has taught it.

Ideas. At each decision the agent takes its few most likely actions and, for each, imagines a
short future in its own model (following its own habits after the first step), scoring it by the
rewards it expects plus its value of where it ends up. The policy's choice is nudged toward the
action whose imagined future looks best, by a learned weight (`trust`, starting at 0): an agent
uses its ideas only as far as acting on them has paid off (trust can turn negative if they mislead
it). After the first step the imagined agent follows its habits among the actions possible now. An "idea" in the playback is the
best-looking imagined future, especially one the habits alone would rarely choose.
"""
from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp

MASK_VALUE = -1e9
# own-state features the world model decodes (indices into the body vector after the window)
DECODE_NAMES = ["food", "water", "energy", "health", "body temperature", "stress", "sleepiness", "pain"]
DECODE_INDEX = [0, 1, 2, 3, 5, 9, 19, 23]


class ScannedGRU(nn.Module):
    """GRU scanned over time. Hidden state is zeroed where `resets` is True
    (the start of a new life)."""

    @functools.partial(nn.scan, variable_broadcast="params", in_axes=0,
                       out_axes=0, split_rngs={"params": False})
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x
        carry = jnp.where(resets[:, None], jnp.zeros_like(carry), carry)
        new_carry, y = nn.GRUCell(features=carry.shape[-1])(carry, ins)
        return new_carry, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden: int):
        return jnp.zeros((batch_size, hidden))


class PatchConv(nn.Module):
    """3x3 'same' convolution with one parameter layout and two compute paths.
    impl="native" uses lax.conv (fast on GPU); impl="patch" uses patch
    extraction + matmul (XLA's conv gradients are pathologically slow on CPU).
    Both give the same result, so checkpoints move freely between CPU and GPU."""
    features: int
    stride: int = 1
    impl: str = "patch"

    @nn.compact
    def __call__(self, x):                              # x: (N, H, W, C)
        c = x.shape[-1]
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (3, 3, c, self.features))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        if self.impl == "native":
            y = jax.lax.conv_general_dilated(x, kernel, (self.stride, self.stride), "SAME",
                                             dimension_numbers=("NHWC", "HWIO", "NHWC"))
        else:
            h, w = x.shape[1], x.shape[2]
            xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
            patches = jnp.concatenate([xp[:, i:i + h, j:j + w, :]
                                       for i in range(3) for j in range(3)], axis=-1)
            if self.stride > 1:
                patches = patches[:, ::self.stride, ::self.stride, :]
            y = patches @ kernel.reshape(9 * c, self.features)
        return y + bias


class ActorCritic(nn.Module):
    """encoder="conv" reads the egocentric window as a (V, V, C) image and the
    body/time/weather features separately; encoder="mlp" flattens everything."""
    n_actions: int
    hidden: int = 128
    encoder: str = "conv"
    view_size: int = 11
    channels: int = 11
    conv_impl: str = "patch"
    msg_out: int = 0          # logits for the symbols said and written (OUT_TOKENS x VOCAB), 0 = none
    aim_out: int = 0          # logits for how a strike is delivered (env.N_HOW: part x limb x mode), 0 = none
    # Where speech and writing branch off (Lowe et al. 2019: sharing the layers that pick the action
    # makes messages look meaningful when they are not): "branch" their own hidden layer on the shared
    # trunk; "stopgrad" the same, but their learning does not shape the trunk; "separate" their own
    # small network from the raw observation (a control: no memory, no shared features).
    msg_net: str = "branch"
    world_model: bool = False
    imagine_k: int = 6        # actions considered per decision
    imagine_h: int = 6        # ticks imagined for each
    imagine_gamma: float = 0.97
    # The own-state head predicts the CHANGE in each need over the next tick, times this (a tick's
    # change is a few hundredths: scaled up so what an action does is the whole signal, not a sliver
    # of the level, which the belief already holds). Levels come back as now + the summed changes.
    own_scale: float = 10.0

    def setup(self):
        if self.encoder == "conv":
            self.conv1 = PatchConv(16, impl=self.conv_impl)
            self.conv2 = PatchConv(32, stride=2, impl=self.conv_impl)
            self.conv3 = PatchConv(32, stride=2, impl=self.conv_impl)
            self.win_dense = nn.Dense(self.hidden)
            self.feat_dense = nn.Dense(64)
        else:
            self.in_dense = nn.Dense(self.hidden)
        self.mid_dense = nn.Dense(self.hidden)
        self.gru = ScannedGRU()

        def silence_bias(key, shape, dtype=jnp.float32):
            # Start mostly silent: favour symbol 0 in the first slot said. With 31 others at 0,
            # a bias b gives speech 31 / (e^b + 31): 5.75 is ~9% (5.0 gave ~17%).
            return jnp.zeros(shape, dtype).at[0].set(5.75)
        self.pol_hidden = nn.Dense(self.hidden)
        self.pol_out = nn.Dense(self.n_actions + self.aim_out, kernel_init=nn.initializers.orthogonal(0.01))
        if self.msg_out:
            if self.msg_net == "separate":
                self.msg_in = nn.Dense(self.hidden)
            self.msg_hidden = nn.Dense(self.hidden)
            self.msg_head = nn.Dense(self.msg_out, kernel_init=nn.initializers.orthogonal(0.01),
                                     bias_init=silence_bias)
        self.val_hidden = nn.Dense(self.hidden)
        self.val_out = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))
        if self.world_model:
            self.dyn1 = nn.Dense(self.hidden)
            self.dyn2 = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(0.1))
            self.rew_hidden = nn.Dense(64)
            self.rew_out = nn.Dense(1)
            self.dec_hidden = nn.Dense(64)
            self.dec_out = nn.Dense(len(DECODE_INDEX))
            self.trust = self.param("trust", nn.initializers.zeros, ())

    # ------------------------------------------------------------ pieces
    def encode(self, obs):
        if self.encoder == "conv":
            t, b, _ = obs.shape
            n_win = self.view_size ** 2 * self.channels
            win = obs[..., :n_win].reshape(t * b, self.view_size, self.view_size, self.channels)
            feats = obs[..., n_win:]
            z = nn.relu(self.conv1(win))
            z = nn.relu(self.conv2(z))
            z = nn.relu(self.conv3(z))
            z = z.reshape(t, b, -1)
            h = jnp.concatenate([nn.relu(self.win_dense(z)), nn.relu(self.feat_dense(feats))], axis=-1)
        else:
            h = nn.relu(self.in_dense(obs))
        return nn.relu(self.mid_dense(h))

    def action_logits(self, h):
        """Logits of the action and of how a strike is delivered (part, limb, mode: env.how_parts)."""
        return self.pol_out(nn.relu(self.pol_hidden(h)))

    def msg_logits(self, h, obs):
        if self.msg_net == "separate":
            x = nn.relu(self.msg_in(obs))
        elif self.msg_net == "stopgrad":
            x = jax.lax.stop_gradient(h)
        else:
            x = h
        return self.msg_head(nn.relu(self.msg_hidden(x)))

    def policy_logits(self, h, obs=None):
        a = self.action_logits(h)
        return jnp.concatenate([a, self.msg_logits(h, obs)], axis=-1) if self.msg_out else a

    def value_of(self, h):
        return self.val_out(nn.relu(self.val_hidden(h))).squeeze(-1)

    def dyn(self, h, a):
        """Next belief from a belief and an action index (residual)."""
        x = jnp.concatenate([h, jax.nn.one_hot(a, self.n_actions)], axis=-1)
        return h + self.dyn2(nn.relu(self.dyn1(x)))

    def reward_of(self, h):
        return self.rew_out(nn.relu(self.rew_hidden(h))).squeeze(-1)

    def decode(self, h):
        """The change in each need (DECODE_NAMES) expected over the next tick, times own_scale."""
        return self.dec_out(nn.relu(self.dec_hidden(h)))

    def habit(self, z, mask_a):
        """The action its habits would take in an imagined situation, among actions possible now
        (the model does not predict what will be possible, so today's options stand in)."""
        return jnp.argmax(jnp.where(mask_a, self.action_logits(z)[..., :self.n_actions], MASK_VALUE), axis=-1)

    def imagine(self, h, logits_masked, mask):
        """Imagined return of each of the k most likely valid actions: (cand (..., k), q (..., k))."""
        A = self.n_actions
        _, cand = jax.lax.top_k(logits_masked[..., :A], self.imagine_k)
        z = jnp.broadcast_to(h[..., None, :], h.shape[:-1] + (self.imagine_k, h.shape[-1]))
        mask_a = mask[..., None, :A]
        a, ret, disc = cand, 0.0, 1.0
        for _ in range(self.imagine_h):
            z = self.dyn(z, a)
            ret = ret + disc * self.reward_of(z)
            disc = disc * self.imagine_gamma
            a = self.habit(z, mask_a)                                     # then its own habits
        ret = ret + disc * self.value_of(z)
        return cand, ret

    # ------------------------------------------------------------ forward
    def trace(self, carry, x):
        """Everything one forward pass computes: carry, h, prior logits, final logits, value,
        and (with a world model) candidates and their imagined returns."""
        obs, resets, mask = x
        carry, h = self.gru(carry, (self.encode(obs), resets))
        prior = jnp.where(mask, self.policy_logits(h, obs), MASK_VALUE)
        out = {"h": h, "prior": prior, "value": self.value_of(h)}
        logits = prior
        if self.world_model:
            cand, q = self.imagine(h, prior, mask)
            q = jax.lax.stop_gradient(q)                       # ideas are trained by the model loss only
            ok = jnp.take_along_axis(mask[..., :self.n_actions], cand, -1)   # fewer valid actions than k
            nok = jnp.maximum(ok.sum(-1, keepdims=True), 1)
            mean = (q * ok).sum(-1, keepdims=True) / nok
            # Standardised over the candidates: imagined returns differ by hundredths (a few ticks of
            # a slowly changing life), so raw differences would need an enormous trust to move a
            # logit, and trust's own gradient would be as small. Scaled, trust 1 means "one standard
            # idea's worth of logit".
            std = jnp.sqrt((((q - mean) ** 2) * ok).sum(-1, keepdims=True) / nok) + 1e-3
            bias = self.trust * (q - mean) / std * ok
            full = jnp.sum(jax.nn.one_hot(cand, self.n_actions + self.aim_out + self.msg_out) * bias[..., None], axis=-2)
            logits = jnp.where(mask, prior + full, MASK_VALUE)
            out.update(cand=cand, q=q)
            if self.is_initializing():
                self.decode(h)                                 # create the own-state head's parameters
        out["logits"] = logits
        return carry, out

    def __call__(self, carry, x):
        """x = (obs [T, B, D], resets [T, B], mask [T, B, A])."""
        carry, out = self.trace(carry, x)
        return carry, out["logits"], out["value"]

    # ------------------------------------------------------------ learning and inspection
    def learn_terms(self, carry, x, actions):
        """What the learner needs in one pass: logits and value, and with a world model the
        beliefs, the predicted next beliefs, expected rewards and expected own state."""
        carry, out = self.trace(carry, x)
        res = {"logits": out["logits"], "value": out["value"]}
        if self.world_model:
            z = self.dyn(out["h"], actions)
            res.update(h=out["h"], z=z, r=self.reward_of(z), own=self.decode(z))
        return res

    def model_terms(self, carry, x, actions):
        """Per-step world-model predictions for training: from h_t and the action taken, the
        predicted next belief, the reward expected and the own state expected next."""
        carry, out = self.trace(carry, x)
        z = self.dyn(out["h"], actions)
        return out["h"], z, self.reward_of(z), self.decode(z)

    def idea(self, carry, x, first_action):
        """Imagine a future starting with `first_action` (per batch item): the actions it would
        take after (its habits), the own state expected each step, the reward expected, and the
        return. For the playback's Ideas view."""
        carry, out = self.trace(carry, x)
        z, a = out["h"], first_action
        n_win = self.view_size ** 2 * self.channels
        now = x[0][..., jnp.asarray(DECODE_INDEX) + n_win]           # the needs as observed now
        level = now
        acts, states, rewards = [], [], []
        for _ in range(self.imagine_h):
            acts.append(a)
            z = self.dyn(z, a)
            level = level + self.decode(z) / self.own_scale          # the level expected after this step
            states.append(level)
            rewards.append(self.reward_of(z))
            a = self.habit(z, x[2][..., :self.n_actions])
        return jnp.stack(acts, -1), jnp.stack(states, -2), jnp.stack(rewards, -1), now


def make_network(env_cfg, hidden: int = 128, encoder: str = "conv",
                 conv_impl: str = "auto", messages: bool = False, world_model: bool = False,
                 msg_net: str = "branch") -> ActorCritic:
    """messages=True adds the aim and the symbols said and written after the action logits
    (env.EXTRA_LOGITS; split_logits, sample_outputs); the mask passed in must then cover them too
    (all True). conv_impl "auto" picks native convolutions
    on GPU/TPU and the patch path on CPU. world_model=True adds the dynamics, reward and
    own-state heads and imagination (see the module docstring)."""
    from . import env as E
    if conv_impl == "auto":
        conv_impl = "patch" if jax.default_backend() == "cpu" else "native"
    return ActorCritic(n_actions=E.N_ACTIONS, hidden=hidden, encoder=encoder,
                       view_size=env_cfg.view_size, channels=E.N_OBS_CHANNELS,
                       conv_impl=conv_impl,
                       msg_out=E.OUT_LOGITS if messages else 0, aim_out=E.N_HOW if messages else 0,
                       msg_net=msg_net, world_model=world_model)


def split_logits(logits):
    """(action (..., N_ACTIONS), how (..., N_HOW), symbols (..., OUT_TOKENS, VOCAB))."""
    from . import env as E
    A, K = E.N_ACTIONS, E.N_HOW
    return logits[..., :A], logits[..., A:A + K], logits[..., A + K:].reshape(logits.shape[:-1] + (E.OUT_TOKENS, E.VOCAB))


def sample_outputs(key, logits):
    """An action and the packed rest, [said..., written..., aim] (env.OUT_SLOTS ints)."""
    act, aim, sym = split_logits(logits)
    ka, kb, kc = jax.random.split(key, 3)
    out = jnp.concatenate([jax.random.categorical(kc, sym), jax.random.categorical(kb, aim)[..., None]], axis=-1)
    return jax.random.categorical(ka, act), out
