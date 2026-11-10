"""Care of infants (EnvConfig.carry_infants, wet_nursing): carried by whoever nurses them, and an
orphan nursed by a woman nursing her own child. Slots 3 and 4 are infants of women 0 and 2."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from test_reproduction import _world, fed

EAST = E.MOVE_N + 2


def _band():
    cfg, s = _world(n=5, founders=5)
    tpy = cfg.ticks_per_bio_year
    r, c = map(int, np.asarray(s.pos[0]))
    t = np.asarray(s.terrain)
    assert t[r, c + 1] != E.WATER and c + 2 < cfg.grid_size
    s = s._replace(sex=jnp.array([E.FEMALE, E.MALE, E.FEMALE, E.MALE, E.FEMALE], jnp.int32),
                   mother=jnp.array([-1, -1, -1, 0, 2], jnp.int32),
                   birth_age=jnp.array([25 * tpy, 25 * tpy, 25 * tpy, 0, 0], jnp.int32),
                   age=jnp.array([0, 0, 0, int(0.3 * tpy), int(0.3 * tpy)], jnp.int32),
                   weight=s.weight.at[3:].set(6.0),
                   pos=jnp.asarray([(r, c)] * 5, jnp.int32),
                   relatedness=jnp.eye(5).at[0, 3].set(0.5).at[3, 0].set(0.5).at[2, 4].set(0.5).at[4, 2].set(0.5))
    return cfg, fed(s), (r, c)


def test_an_infant_goes_where_its_mother_goes_and_she_pays_for_it():
    cfg, s, (r, c) = _band()
    acts = jnp.array([EAST, E.REST, E.REST, E.REST, E.REST])
    out, _, _, _ = E.step(s, acts, cfg)
    assert tuple(np.asarray(out.pos[0])) == (r, c + 1)
    assert tuple(np.asarray(out.pos[3])) == (r, c + 1)                     # carried along
    assert tuple(np.asarray(out.pos[4])) == (r, c)                         # the other infant stays with its mother
    # with no infant in arms the same step costs her less
    free = s._replace(mother=s.mother.at[3].set(-1))
    out2, _, _, _ = E.step(free, acts, cfg)
    assert float(out.energy[0]) < float(out2.energy[0])
    # off: infants stay put
    off = dataclasses.replace(cfg, carry_infants=False)
    out3, _, _, _ = E.step(s, acts, off)
    assert tuple(np.asarray(out3.pos[3])) == (r, c)


def test_an_orphan_is_nursed_and_carried_by_a_nursing_woman():
    cfg, s, (r, c) = _band()
    orphan = s._replace(alive=s.alive.at[0].set(False), satiety=s.satiety.at[3].set(0.3),
                        hydration=s.hydration.at[3].set(0.3))
    care = np.asarray(E.caregiver_of(orphan, cfg))
    assert care[3] == 2 and care[4] == 2
    out, _, _, info = E.step(orphan, jnp.array([E.REST, E.REST, EAST, E.REST, E.REST]), cfg)
    assert bool(info.nursed[3]) and float(out.hydration[3]) > 0.3
    assert tuple(np.asarray(out.pos[3])) == (r, c + 1) == tuple(np.asarray(out.pos[2]))
    # a woman not nursing a child of her own has no milk to give
    dry = orphan._replace(mother=orphan.mother.at[4].set(-1))
    assert int(E.caregiver_of(dry, cfg)[3]) == -1
    # and without wet nursing the orphan has no one
    off = dataclasses.replace(cfg, wet_nursing=False)
    assert int(E.caregiver_of(orphan, off)[3]) == -1
