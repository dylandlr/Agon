import random

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import analysis, env as E
from lifesim.config import EnvConfig
from lifesim.ppo_multi import MAConfig, make_train
from lifesim.recorder import Recording, record_episode, rewind


def test_no_names_exist_until_agents_invent_them():
    from lifesim.names import names_over_time, spoken_form
    assert "name" not in " ".join(E.State._fields + E.StepInfo._fields)
    assert spoken_form([3, 1, 1]) == "Mikaka" and spoken_form([0, 0, 0]) == ""
    assert names_over_time([]) == []


def test_names_appear_only_once_naming_starts():
    from lifesim.names import names_over_time
    # Random speech for 200 ticks, then speakers start using one word per person
    data = _frames(lambda target, rng: [rng.randint(1, 15), rng.randint(0, 15), rng.randint(0, 15)], n_frames=200)
    later = _frames(lambda target, rng: [target + 1, 1, 1], n_frames=200, seed=1)
    for f in later["frames"]:
        f["t"] += 200
    data["frames"] += later["frames"]
    given = names_over_time(analysis.utterances(data))
    assert {g["agent"]: g["name"] for g in given} == {2: "Mikaka", 3: "Takaka", 4: "Rukaka"}
    assert all(g["t"] >= 200 for g in given)          # nobody is named during random speech


def test_mutual_information_basics():
    xs = [0, 1, 2, 3] * 25
    assert abs(analysis.mutual_information(xs, xs) - 2.0) < 1e-9
    assert abs(analysis.mutual_information(xs, [0] * 100)) < 1e-9


def _frames(rule, n_frames=300, seed=0):
    """Agents 0 and 1 speak; one of agents 2-4 stands next to the speaker each tick."""
    rng = random.Random(seed)
    frames = []
    for t in range(n_frames):
        speaker, target = rng.choice([0, 1]), rng.choice([2, 3, 4])
        agents = [{"r": 30, "c": 30, "alive": True, "msg": [0, 0, 0]} for _ in range(5)]
        agents[speaker] = {"r": 0, "c": 0, "alive": True, "msg": rule(target, rng)}
        agents[target] = {"r": 0, "c": 1, "alive": True, "msg": [0, 0, 0]}
        frames.append({"t": t, "agents": agents, "pred": [], "large": [], "fire": []})
    return {"frames": frames, "view_radius": 5}


def test_naming_index_finds_invented_names():
    data = _frames(lambda target, rng: [target + 1, 1, 1])     # same word for the same person
    m = analysis.language_metrics(data)
    found = {n["target"]: n["message"] for n in m["naming"]["names"]}
    assert m["naming"]["index_bits"] > 1.0
    assert found == {2: [3, 1, 1], 3: [4, 1, 1], 4: [5, 1, 1]}
    assert {g["agent"] for g in m["names_given"]} == {2, 3, 4}


def test_naming_index_finds_nothing_in_random_speech():
    data = _frames(lambda target, rng: [rng.randint(1, 15), rng.randint(0, 15), rng.randint(0, 15)])
    m = analysis.language_metrics(data)
    assert abs(m["naming"]["index_bits"]) < 0.3 and m["naming"]["names"] == [] and m["names_given"] == []


def test_replay_is_exact_with_speech():
    cfg = EnvConfig(grid_size=16, n_agents=3, max_steps=150)

    def speaking_policy(key, obs, mask, carry):
        ka, km = jax.random.split(key)
        a = jax.random.categorical(ka, jnp.where(mask, 0.0, -1e9))
        return (a, jax.random.randint(km, (3, E.MSG_LEN), 0, E.VOCAB)), carry

    rec = record_episode(speaking_policy, None, seed=3, cfg=cfg, snapshot_every=50, names=["Ama", "Bao", "Chen"])
    assert len(rec.messages) == len(rec.actions) and any(np.asarray(m)[:, 0].any() for m in rec.messages)
    straight = rewind(Recording(cfg=cfg, seed=3, snapshot_every=10 ** 9, snapshots={0: rec.snapshots[0]},
                                actions=rec.actions, messages=rec.messages), 120)
    for x, y in zip(jax.tree_util.tree_leaves(rewind(rec, 120)), jax.tree_util.tree_leaves(straight)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert all("name" in ev for ev in rec.events if ev["agent"] >= 0)


def test_alpha_group_reward_trains_and_counters_are_logged():
    env_cfg = EnvConfig(grid_size=12, n_agents=3)
    cfg = MAConfig(num_worlds=4, num_steps=16, total_ticks=32, num_minibatches=2,
                   encoder="mlp", hidden=32, alpha=1.0)
    _, init, update = make_train(env_cfg, cfg)
    runner, pop = init(jax.random.PRNGKey(0))
    runner, m = update(runner, pop)
    for k in ["deterred", "winter_night_ticks", "winter_huddle_ticks", "large_kills_by_agent",
              "own_reward", "world_ticks", "survivors_at_end", "full_years"]:
        assert k in m
    assert int(m["world_ticks"]) == 4 * 16 and np.isfinite(float(m["own_reward"]))
    assert m["large_kills_by_agent"].shape == (3,) and np.isfinite(float(m["entropy"]))


def test_gini_of_hunting_credit():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("tp2", pathlib.Path("scripts/train_phase2.py"))
    tp2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp2)
    assert tp2.gini([5, 5, 5, 5]) == 0.0
    assert abs(tp2.gini([0, 0, 0, 12]) - 0.75) < 1e-9


def test_context_information_is_corrected_for_chance():
    rng = random.Random(1)
    flags = [rng.random() < 0.5 for _ in range(300)]
    unique_msgs = [(i, 0, 0) for i in range(300)]                 # every message different: pure noise
    raw = analysis.mutual_information(unique_msgs, flags)
    assert raw > 0.9                                               # the plug-in estimate reads noise as ~1 bit
    assert abs(analysis.mi_above_chance(unique_msgs, flags)) < 0.05
    meaningful = [(1, 0, 0) if f else (2, 0, 0) for f in flags]    # message determined by the context
    assert analysis.mi_above_chance(meaningful, flags) > 0.9



# ------------------------------------------------------------ body, aging, inheritance
from lifesim import genotype as gt


def _body_at(years, **genes):
    cfg = EnvConfig(grid_size=12, n_agents=1, climate_volatility=0.0, season_temp=(18.0,) * 4,
                    day_temp_swing=0.0, lightning_prob=0.0)
    s = E.reset(jax.random.PRNGKey(0), cfg)
    g = s.genotype._replace(**{k: jnp.array([v], jnp.float32) for k, v in genes.items()})
    s = s._replace(genotype=g, birth_age=jnp.array([int(years * cfg.ticks_per_bio_year)], jnp.int32))
    ideal = cfg.ideal_bmi * (cfg.base_height_cm * g.height / 100) ** 2 * g.build
    return cfg, s._replace(weight=ideal)


def test_traits_manifest_with_age():
    out = {}
    for years in (8, 25, 65):
        cfg, s = _body_at(years, strength=1.0, speed=1.0, height=1.0, build=1.0)
        out[years] = {k: float(v[0]) for k, v in E.body(s, cfg).items()}
    assert out[8]["height_cm"] < out[25]["height_cm"] and out[65]["height_cm"] < out[25]["height_cm"]
    assert out[8]["strength"] < out[25]["strength"] and out[65]["strength"] < out[25]["strength"]
    assert abs(out[25]["strength"] - 1.0) < 1e-6 and abs(out[25]["height_cm"] - 165.0) < 1e-3


def test_fat_reserves_buffer_starvation():
    cfg, s = _body_at(25, metabolism=1.0)
    dead = lambda an: an._replace(alive=jnp.zeros_like(an.alive))       # no predators, no disease sources
    s = s._replace(satiety=jnp.zeros(1), hydration=jnp.ones(1), energy=jnp.ones(1),
                   small=dead(s.small), large=dead(s.large), predators=dead(s.predators))
    w0 = float(s.weight[0])
    for _ in range(50):
        s = s._replace(hydration=jnp.ones(1), energy=jnp.ones(1))
        s, _, _, info = E.step(s, jnp.array([E.REST]), cfg)
    assert float(s.weight[0]) < w0 - 3.0              # burned reserves
    assert float(s.health[0]) > 0.99                  # and was spared starvation damage meanwhile


def test_inheritance_is_heritable_but_random():
    pop = gt.sample(jax.random.PRNGKey(0), 200)
    tall = pop._replace(height=jnp.full(200, 1.10))
    short = pop._replace(height=jnp.full(200, 0.90))
    kids_tall = gt.inherit(jax.random.PRNGKey(1), tall, tall, pop)
    kids_short = gt.inherit(jax.random.PRNGKey(2), short, short, pop)
    assert float(kids_tall.height.mean()) > float(kids_short.height.mean()) + 0.1   # heritable
    assert float(kids_tall.height.std()) > 0.01                                     # but random
    # Low-heritability traits regress further toward the population mean than height does
    hi = pop._replace(empathy=jnp.full(200, 1.0), height=jnp.full(200, 1.10))
    kids = gt.inherit(jax.random.PRNGKey(3), hi, hi, pop)
    emp_pull = (1.0 - float(kids.empathy.mean())) / (1.0 - float(pop.empathy.mean()))
    hgt_pull = (jnp.log(1.10) - jnp.log(kids.height).mean()) / (jnp.log(1.10) - jnp.log(pop.height).mean())
    assert emp_pull > float(hgt_pull)



def test_baselines_run_in_the_multi_agent_setting():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("eb", pathlib.Path("scripts/eval_baselines.py"))
    eb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eb)
    cfg = EnvConfig(grid_size=16, n_agents=3, max_steps=60)
    for policy in ("random", "scripted"):
        r = eb.evaluate(policy, cfg, worlds=2, seed=0)
        assert 0.0 <= r["share_alive_at_episode_end"] <= 1.0
        assert 0.0 <= r["share_of_worlds_reaching_year_end"] <= 1.0
        assert np.isfinite(r["global_reward_per_world_day"])


def test_compare_runs_keeps_conditions_apart(tmp_path, capsys):
    import importlib.util, json, pathlib
    spec = importlib.util.spec_from_file_location("cr", pathlib.Path("scripts/compare_runs.py"))
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    row = {"mean_age_at_death": 300.0, "kills": {"large_prey": 0},
           "per_1000_agent_ticks": {"gifts": 1.0, "strikes_on_agents": 2.0}}
    def run(name, **args):
        d = tmp_path / name
        d.mkdir()
        json.dump({"args": {"alpha": 0.0, "scarcity": 1.0, "imitation": 0.5, "seed": 0, **args},
                   "history": [row]}, open(d / "history.json", "w"))
        return str(d)
    paths = [run("a", seed=0), run("b", seed=1), run("c", imitation=0.0)]
    cr.main(paths)
    lines = [l for l in capsys.readouterr().out.splitlines()[1:] if l.strip()]
    assert len(lines) == 2                                   # imitation on vs off stay separate
    seeds = sorted(int(l.split()[1]) for l in lines)
    assert seeds == [1, 2]                                   # the two seeds pool into one row


def test_presets_and_code_version():
    from lifesim.presets import PRESETS, code_version
    v = code_version()
    assert len(v) == 12 and v == code_version()                       # stable for the same code
    society = PRESETS["society"]
    assert society["reproduction"] and society["reward_mode"] == "survival" and society["agents"] > society["founders"]


def test_experiment_conditions():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("experiment", Path(__file__).parents[1] / "scripts" / "experiment.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    core = ex.items("core")
    assert len(core) == 18 and len({n for n, _ in core}) == 18
    assert [n for n, _ in ex.items("validate")] == ["caveman", "caveman_no_sleep_learning", "blank",
                                                    "caveman_no_world_model"]
    inh = ex.items("inheritance")
    assert len(inh) == 6 and {e[0] for _, e in inh} == {"parent", "innate"}
    from lifesim.presets import PRESETS
    assert PRESETS["society"]["world_model"] and PRESETS["tribe"]["agents"] == 64
