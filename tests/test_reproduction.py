"""Natural reproduction: mutual mating, fertility, pregnancy, birth, nursing, kinship and genetic survival."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lifesim import env as E
from lifesim.config import EnvConfig, with_reproduction


def _world(n=4, founders=2, **kw):
    base = EnvConfig(grid_size=12, n_agents=n, climate_volatility=0.0, lightning_prob=0.0,
                     season_temp=(18.0,) * 4, day_temp_swing=0.0, founder_spread=0, reward_mode="survival")
    opts = dict(conception_prob=1.0, maternal_death=0.0)
    opts.update(kw)
    cfg = with_reproduction(base, n_founders=founders, **opts)
    s = E.reset(jax.random.PRNGKey(1), cfg)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))
    t = np.asarray(s.terrain)
    spot = next((r, c) for r in range(1, 11) for c in range(1, 10)
                if (t[r, c:c + 2] != E.WATER).all())
    pos = [spot, (spot[0], spot[1] + 1)] + [spot] * (n - 2)
    tpy = cfg.ticks_per_bio_year
    s = s._replace(small=dead(s.small), large=dead(s.large), predators=dead(s.predators),
                   pos=jnp.asarray(pos, jnp.int32), birth_age=jnp.full(n, int(25 * tpy), jnp.int32),
                   immunity=jnp.ones(n), t=jnp.int32(24))
    return cfg, s


def fed(s):
    n = s.pos.shape[0]
    return s._replace(satiety=jnp.ones(n), hydration=jnp.ones(n), energy=jnp.ones(n), stress=jnp.zeros(n))


def run(cfg, s, actions, ticks):
    step = jax.jit(lambda st, a: E.step(fed(st), a, cfg))
    infos = []
    for _ in range(ticks):
        s, r, _, info = step(s, jnp.asarray(actions, jnp.int32))
        infos.append((r, info))
    return s, infos


REST, MATE = E.REST, E.MATE


def test_unborn_slots_are_empty_invisible_and_feel_nothing():
    cfg, s = _world()
    assert list(np.asarray(s.alive)) == [True, True, False, False]
    assert list(np.asarray(s.existed)) == [True, True, False, False]
    v, c = cfg.view_size, E.N_OBS_CHANNELS
    win = np.asarray(E.observe(s, cfg)[0, :v * v * c]).reshape(v, v, c)
    assert win[..., E.CH_BODIES].sum() == 0                              # no "bodies" of the unborn
    mask = np.asarray(E.action_mask(s, cfg))
    assert mask[2].sum() == 1 and mask[2, REST]
    _, infos = run(cfg, s, [REST] * 4, 3)
    assert all(float(r[2]) == 0.0 and float(r[3]) == 0.0 for r, _ in infos)


def test_conception_takes_both_choosing_to_mate():
    cfg, s = _world()
    assert bool(E.action_mask(s, cfg)[0, MATE]) and bool(E.action_mask(s, cfg)[1, MATE])
    one, infos = run(cfg, s, [MATE, REST, REST, REST], 1)
    assert int(one.gestation[0]) == 0 and not bool(infos[0][1].mated.any())
    both, infos = run(cfg, s, [MATE, MATE, REST, REST], 1)
    assert int(both.gestation[0]) == 1 and int(both.father[0]) == 1 and bool(infos[0][1].conceived[0])
    assert int(both.gestation[1]) == 0                                   # men do not get pregnant
    same = s._replace(sex=jnp.array([E.FEMALE, E.FEMALE, 0, 1], jnp.int32))
    assert not bool(E.action_mask(same, cfg)[0, MATE])                   # needs a partner of the other sex


def test_no_conception_when_too_young_too_thin_sick_or_already_pregnant():
    cfg, s = _world()
    young = s._replace(birth_age=s.birth_age.at[0].set(int(10 * cfg.ticks_per_bio_year)))
    assert not bool(E.action_mask(young, cfg)[0, MATE])
    for unfit in (s._replace(weight=s.weight.at[0].multiply(0.7)), s._replace(infection=s.infection.at[0].set(50))):
        out, _ = run(cfg, unfit, [MATE, MATE, REST, REST], 1)
        assert int(out.gestation[0]) == 0
    pregnant, _ = run(cfg, s, [MATE, MATE, REST, REST], 1)
    assert not bool(E.action_mask(pregnant, cfg)[0, MATE])


def test_no_conception_without_a_free_slot():
    cfg, s = _world(n=2, founders=2)
    out, _ = run(cfg, s, [MATE, MATE], 1)
    assert int(out.gestation[0]) == 0


def _pregnant_until_birth(cfg, s):
    s, _ = run(cfg, s, [MATE, MATE] + [REST] * (s.pos.shape[0] - 2), 1)
    g = int(cfg.gestation_years * cfg.ticks_per_bio_year)
    s, infos = run(cfg, s, [REST] * s.pos.shape[0], g - 1)
    return s, infos


def test_pregnancy_ends_in_a_birth_with_inherited_traits_and_kinship():
    cfg, s = _world()
    s, infos = _pregnant_until_birth(cfg, s)
    last = infos[-1][1]
    assert bool(last.gave_birth[0]) and bool(last.born[2])               # never-used slot first
    assert bool(s.alive[2]) and bool(s.existed[2]) and int(s.mother[2]) == 0 and int(s.age[2]) == 0
    assert tuple(np.asarray(s.pos[2])) == tuple(np.asarray(s.pos[0]))    # born where the mother is
    R = np.asarray(s.relatedness)
    assert np.isclose(R[2, 0], 0.5) and np.isclose(R[2, 1], 0.5) and np.isclose(R[0, 1], 0.0)
    assert np.allclose(R, R.T) and np.isclose(R[2, 2], 1.0)
    assert int(s.gestation[0]) == 0
    look = np.asarray(s.appearance)
    assert np.abs(look[2] - 0.5 * (look[0] + look[1])).max() < 0.25      # looks like its parents
    for name in ("height", "strength", "empathy"):
        v = float(getattr(s.genotype, name)[2])
        assert np.isfinite(v) and v > 0


def test_infants_cannot_walk_or_forage_and_are_nursed_by_their_mother():
    cfg, s = _world()
    s, _ = _pregnant_until_birth(cfg, s)
    mask = np.asarray(E.action_mask(s, cfg))[2]
    assert not mask[E.MOVE_N:E.MOVE_NW + 1].any() and not mask[E.GRASP]
    hungry = s._replace(satiety=s.satiety.at[2].set(0.3), hydration=s.hydration.at[2].set(0.3))
    step = jax.jit(lambda st: E.step(st, jnp.full(4, REST, jnp.int32), cfg))
    with_mom, _, _, info = step(hungry)
    assert bool(info.nursed[2]) and float(with_mom.satiety[2]) > 0.3
    assert float(with_mom.satiety[0]) < float(hungry.satiety[0])         # the mother pays
    land = np.argwhere(np.asarray(s.terrain) != E.WATER)
    far = next(tuple(p) for p in land[::-1] if max(abs(p[0] - int(s.pos[2, 0])), abs(p[1] - int(s.pos[2, 1]))) > 2)
    apart = hungry._replace(pos=hungry.pos.at[0].set(jnp.asarray(far)))
    alone, _, _, info = step(apart)
    assert not bool(info.nursed[2]) and float(alone.satiety[2]) < 0.3


def test_relatives_survival_counts_in_the_reward():
    cfg, s = _world()
    s, _ = _pregnant_until_birth(cfg, s)
    step = jax.jit(lambda st: E.step(fed(st), jnp.full(4, REST, jnp.int32), cfg))
    _, r, _, info = step(s)
    assert np.isclose(float(info.genetic_reward[0]), 0.5 * cfg.alive_reward)   # a living child
    dying = s._replace(health=s.health.at[2].set(0.0001), injury=s.injury.at[2].set(1.0))
    _, _, _, info = step(dying)
    assert bool(info.died[2])
    assert np.isclose(float(info.genetic_reward[0]), -0.5 * cfg.death_penalty)  # losing a child
    assert float(info.genetic_reward[3]) == 0.0                                 # unborn: nothing


def test_childbirth_can_kill_the_mother_and_the_child_still_lives():
    cfg, s = _world(maternal_death=1.0)
    s, infos = _pregnant_until_birth(cfg, s)
    last = infos[-1][1]
    assert bool(last.died[0]) and int(last.death_cause[0]) == E.CAUSE_CHILDBIRTH
    assert bool(s.alive[2])


def test_a_reused_slot_forgets_its_previous_occupant():
    cfg, s = _world(n=3, founders=3)
    s = s._replace(memory=s.memory.at[0, 2, E.MEM_STRUCK_ME].set(3.0),
                   alive=s.alive.at[2].set(False), health=s.health.at[2].set(0.0))   # slot 2 died earlier
    s, infos = _pregnant_until_birth(cfg, s)
    assert bool(infos[-1][1].born[2]) and bool(s.alive[2])
    assert float(s.memory[0, 2].sum()) == 0.0                           # nobody remembers the newborn as an enemy
    assert np.isclose(float(s.relatedness[2, 0]), 0.5)


def _mate_once(cfg, s, actions):
    """One tick from given stress, without the fed() reset of stress."""
    full = lambda st: st._replace(satiety=jnp.ones(4), hydration=jnp.ones(4), energy=jnp.ones(4))
    out, _, _, info = jax.jit(lambda st: E.step(full(st), jnp.asarray(actions, jnp.int32), cfg))(s)
    return out, info


def test_intimacy_calms_both_partners_by_their_libido():
    cfg, s = _world(conception_prob=0.0)
    s = s._replace(stress=jnp.full(4, 0.5), genotype=s.genotype._replace(libido=jnp.array([1.0, 2.0, 1.0, 1.0])))
    alone, _ = _mate_once(cfg, s, [MATE, REST, REST, REST])
    both, info = _mate_once(cfg, s, [MATE, MATE, REST, REST])
    assert bool(info.mated[0]) and bool(info.mated[1])
    relief = np.asarray(alone.stress[:2]) - np.asarray(both.stress[:2])
    assert np.allclose(relief, [cfg.intimacy_relief, 2 * cfg.intimacy_relief], atol=1e-5)
    calm = s._replace(stress=jnp.zeros(4))
    out, _ = _mate_once(cfg, calm, [MATE, MATE, REST, REST])
    assert float(out.stress[0]) >= 0.0                                   # nothing to gain when already calm


def test_intimacy_fades_with_repetition_and_recovers_with_time_apart():
    cfg, s = _world(conception_prob=0.0)
    s = s._replace(stress=jnp.full(4, 0.8), genotype=s.genotype._replace(libido=jnp.ones(4)))
    base, _ = _mate_once(cfg, s, [REST] * 4)
    first, _ = _mate_once(cfg, s, [MATE, MATE, REST, REST])
    again, _ = _mate_once(cfg, first._replace(stress=s.stress), [MATE, MATE, REST, REST])
    r1 = float(base.stress[0] - first.stress[0])
    r2 = float(base.stress[0] - again.stress[0])
    assert np.isclose(r1, cfg.intimacy_relief, atol=1e-5) and r2 < 0.5 * r1
    rested = first._replace(stress=s.stress, habituation=first.habituation * 0.5 ** 6)
    later, _ = _mate_once(cfg, rested, [MATE, MATE, REST, REST])
    assert float(base.stress[0] - later.stress[0]) > 0.9 * r1


def test_a_fed_child_grows_lean_mass_and_a_starving_one_is_stunted():
    """Children grow into their frame; before this, weight could only rise as fat (0.01 kg a tick)
    and every child born grew up below the weight needed to conceive."""
    import jax
    import jax.numpy as jnp
    from lifesim import env as E
    from lifesim.config import EnvConfig, with_reproduction
    cfg = with_reproduction(EnvConfig(grid_size=12, n_agents=2, climate_volatility=0.0, lightning_prob=0.0,
                                      season_temp=(18.0,) * 4, day_temp_swing=0.0), n_founders=2)
    s = E.reset(jax.random.PRNGKey(2), cfg)
    child_age = jnp.full(2, int(5 * cfg.ticks_per_bio_year), jnp.int32)            # 5 years old
    s = s._replace(birth_age=child_age, age=jnp.zeros(2, jnp.int32), conditions=jnp.zeros_like(s.conditions))
    s = s._replace(weight=E.body(s, cfg)["ideal_kg"])
    start = np.asarray(E.body(s, cfg)["ideal_kg"])
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST]), cfg))
    for _ in range(int(5 * cfg.ticks_per_bio_year)):                               # five more years
        s = s._replace(satiety=jnp.array([1.0, 0.1]), hydration=jnp.ones(2), energy=jnp.ones(2),
                       health=jnp.ones(2), infection=jnp.zeros(2, jnp.int32))
        s, *_ = step(s)
    ideal = np.asarray(E.body(s, cfg)["ideal_kg"])
    ratio = np.asarray(s.weight) / ideal
    assert ideal[0] > 1.15 * start[0]                    # the frame grew
    assert ratio[0] > 0.97                               # fed: kept up with it
    assert ratio[1] < 0.9                                # starving: stunted


def test_a_stunted_child_catches_up_once_fed():
    """Catch-up growth (EnvConfig.catch_up_years): a 6-year-old at 70% of its frame's weight, then
    well fed, regains most of it within a couple of years; an adult past catch_up_until_years does not."""
    import jax
    import jax.numpy as jnp
    from lifesim import env as E
    from lifesim.config import EnvConfig, with_reproduction
    cfg = with_reproduction(EnvConfig(grid_size=12, n_agents=2, climate_volatility=0.0, lightning_prob=0.0,
                                      season_temp=(18.0,) * 4, day_temp_swing=0.0), n_founders=2)
    s = E.reset(jax.random.PRNGKey(3), cfg)
    ages = jnp.array([int(6 * cfg.ticks_per_bio_year), int(30 * cfg.ticks_per_bio_year)], jnp.int32)
    s = s._replace(birth_age=ages, age=jnp.zeros(2, jnp.int32), conditions=jnp.zeros_like(s.conditions))
    s = s._replace(weight=0.7 * E.body(s, cfg)["ideal_kg"])
    step = jax.jit(lambda st: E.step(st, jnp.array([E.REST, E.REST]), cfg))
    for _ in range(int(2 * cfg.ticks_per_bio_year)):
        s = s._replace(satiety=jnp.full(2, 0.6), hydration=jnp.ones(2), energy=jnp.ones(2),
                       health=jnp.ones(2), infection=jnp.zeros(2, jnp.int32))
        s, *_ = step(s)
    ratio = np.asarray(s.weight) / np.asarray(E.body(s, cfg)["ideal_kg"])
    assert ratio[0] > 0.85                               # the child caught up past the fertility line
    assert ratio[1] < 0.75                               # the adult did not (fat only above satiety 0.7)


def _seq(cfg, s, plan):
    """Run a list of per-tick action lists; return the final state and each tick's info."""
    step = jax.jit(lambda st, a: E.step(fed(st), a, cfg))
    infos = []
    for acts in plan:
        s, _, _, info = step(s, jnp.asarray(acts, jnp.int32))
        infos.append(info)
    return s, infos


def test_a_choice_to_mate_stays_open_for_a_few_ticks_and_is_seen():
    cfg, s = _world()                                                    # mate_window 4 by default
    she = [MATE, REST, REST, REST]
    he = [REST, MATE, REST, REST]
    later, infos = _seq(cfg, s, [she, he])
    assert not bool(infos[0].mated.any()) and bool(infos[1].conceived[0])
    # Courting shows: the man sees her as courting right after she chose it
    s1, _ = _seq(cfg, s, [she])
    v, c = cfg.view_size, E.N_OBS_CHANNELS
    win = np.asarray(E.observe(s1, cfg)[1, :v * v * c]).reshape(v, v, c)
    courting_ch = c - E.N_SOCIAL_FEATURES + E.N_MEM + E.N_APPEARANCE + 1 + 7
    r = cfg.view_radius
    assert win[r, r - 1, courting_ch] == 1.0 and win[r, r, courting_ch] == 0.0
    # The offer lapses after the window: three more ticks of waiting, then too late
    late, infos = _seq(cfg, s, [she, [REST] * 4, [REST] * 4, [REST] * 4, he])
    assert not any(bool(i.mated.any()) for i in infos)
    # Window 1: the same tick only, as before
    cfg1, s = _world(mate_window=1)
    _, infos = _seq(cfg1, s, [she, he])
    assert not any(bool(i.mated.any()) for i in infos)


def test_a_sleeping_partner_is_not_courting():
    cfg, s = _world()
    s = s._replace(sleepiness=s.sleepiness.at[0].set(0.5))         # tired enough to sleep (sleep is honest)
    _, infos = _seq(cfg, s, [[MATE, REST, REST, REST], [E.SLEEP, MATE, REST, REST]])
    assert not any(bool(i.mated.any()) for i in infos)


def test_doing_anything_else_withdraws_an_offer():
    """Consent is current: an offer stays open only while one stays with it (resting or touching);
    any other act withdraws it, so no one is mated without having chosen it themselves."""
    cfg, s = _world()
    she, he = [MATE, REST, REST, REST], [REST, MATE, REST, REST]
    _, infos = _seq(cfg, s, [she, [E.GRASP, MATE, REST, REST]])          # she turns to something else
    assert not any(bool(i.mated.any()) for i in infos)
    _, infos = _seq(cfg, s, [she, [E.TOUCH_N + 2, MATE, REST, REST]])    # touching him keeps it open
    assert bool(infos[1].mated[0])
    _, infos = _seq(cfg, s, [she, he])
    assert bool(infos[1].mated[0])


def test_real_time_makes_pregnancy_nine_months():
    from lifesim.config import with_time_scale
    base = with_time_scale(EnvConfig(grid_size=12, n_agents=4), 365, 365)
    cfg = with_reproduction(base, n_founders=2)
    assert cfg.year_length == 365 * 48 and cfg.ticks_per_bio_year == 365 * 48
    assert round(cfg.gestation_years * cfg.ticks_per_bio_year / cfg.day_length) == 274      # days
    assert cfg.max_age == 70 * 365 * 48
    compressed = with_reproduction(EnvConfig(grid_size=12, n_agents=4), n_founders=2)
    assert compressed.ticks_per_bio_year == 48


def test_conception_per_act_follows_the_time_scale():
    """A year of regular mating gives the same chance of a child at any compression."""
    from lifesim.config import EnvConfig, with_reproduction, with_time_scale
    for dpby in (1.0, 10.0, 365.0):
        cfg = with_reproduction(with_time_scale(EnvConfig(), 365, dpby))
        acts = dpby * cfg.mating_acts_per_day
        assert 1 - (1 - cfg.conception_prob) ** acts == pytest.approx(cfg.conception_per_bio_year, abs=2e-3)
    assert with_reproduction(EnvConfig(), conception_prob=0.3).conception_prob == 0.3


def test_there_is_no_fixed_death_age():
    """Senescence is a hazard rising with age (Gompertz), not a limit: deaths spread over decades,
    peaking near max_bio_years, and a longer-lived gene shifts the whole curve."""
    cfg = EnvConfig()
    h = lambda y, m=1.0: float(E.senescence_hazard(jnp.float32(y), m, cfg))
    assert h(20) < 0.002 and 0.005 < h(45) < 0.02 and h(70) == pytest.approx(cfg.senescence_rate, rel=1e-4)
    assert h(90) < 1.0 and h(84, 1.2) == pytest.approx(h(70), rel=1e-4)
    years = np.arange(15.0, 130.0, 1 / 52)                       # life table from age 15, weekly
    surv = np.exp(-np.cumsum(np.asarray(E.senescence_hazard(jnp.asarray(years), 1.0, cfg))) / 52)
    a = years[np.searchsorted(-surv, -np.random.default_rng(0).random(20000))]   # sampled ages at death
    assert 60 < np.median(a) < 72 and np.percentile(a, 10) < 50 and np.percentile(a, 99) > 85


def test_a_nursing_mother_does_not_conceive_until_her_child_is_weaned():
    cfg, s = _world()
    tpy = cfg.ticks_per_bio_year
    with_child = lambda years: s._replace(alive=s.alive.at[2].set(True), existed=s.existed.at[2].set(True),
                                          mother=s.mother.at[2].set(0), age=s.age.at[2].set(0),
                                          birth_age=s.birth_age.at[2].set(int(years * tpy)))
    both = [MATE, MATE, REST, REST]
    _, infos = run(cfg, with_child(1.0), both, 1)
    assert bool(infos[0][1].mated[0]) and not bool(infos[0][1].conceived[0])
    _, infos = run(cfg, with_child(3.0), both, 1)                  # weaned: fertile again
    assert bool(infos[0][1].conceived[0])


def test_a_conception_the_slot_cap_stopped_is_counted():
    cfg, s = _world(n=2, founders=2)
    _, infos = run(cfg, s, [MATE, MATE], 1)
    i = infos[0][1]
    assert bool(i.mated[0]) and not bool(i.conceived.any()) and bool(i.no_room[0])
