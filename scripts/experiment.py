"""The thesis experiments, identical on Colab and HPRC.

    python scripts/experiment.py validate --root runs/exp --worlds 64 --agent-steps 1e8
    python scripts/experiment.py core     --root runs/exp --worlds 64 --agent-steps 5e7
    python scripts/experiment.py inheritance --root runs/exp      # after validate
    python scripts/experiment.py stress --root runs/exp           # after validate
    python scripts/experiment.py report   --root runs/exp

validate  Four runs, each followed by a 10-year continuous world: the "society" preset
          (caveman slate: given survival skills, world model), the same without sleep-consolidated
          learning, the same without the world model, and the blank slate (primitive actions only).
inheritance  History with learning: the validated caveman run's agents live on in a continuous
          world where every individual keeps learning; newborns get a parent's learned brain
          (parent, Lamarckian) or the brain the parent was born with (innate, Darwinian), 3 seeds
          each. Does culture survive when learning cannot pass down through the brain?
stress    How the validated caveman agents meet a crisis: one continuous world per shock
          (famine, plague, bold predators, a cold spell, drought), 10 days long from day 40,
          against the run's own unshocked world. The report gives numbers before, at the worst,
          and 40 days after, and what killed them during the shock.
archetypes  Intervention: one tyrant or one saint among the founders (alpha 0, normal food,
          3 seeds each); the control is core a0.0_f1.0_s*.
placed    Intervention while living: the validated caveman agents' continuous world with learning,
          one founder given an archetype's temperament (--which, default all 17), 2 seeds, against the
          same world with none. Does a chief come to lead, a warrior to rank, an enforcer to make
          others gentler? Far cheaper than retraining per archetype.
culture   Replicate worlds: the same map and founders (map seed 21), different histories (seeds
          31..), the validated agents learning as they live with innate inheritance (learning does
          not pass down through the brain); `social` as trained, `solitary` with no learning by
          watching (--replicates each, default 4). scripts/culture.py compares them.
core      The proposal's core experiment: reward sharing alpha in {0, 0.5, 1} x food {1.0, 0.4}
          x seeds {0, 1, 2} = 18 runs of the same preset, each followed by a continuous world.
gate      Did the validated caveman agents persist in their continuous world (births, and people
          born there reaching fertile age)? core, archetypes,
          inheritance and stress refuse to start until it passes (--skip-gate to override), so
          hours are not spent on agents that cannot persist.
probe     The current world as it now stands, on the 250 x 250 "society" preset (1-minute ticks),
          excretion off, sleep-consolidated learning on: the full run, the same without sleep
          learning, with caves underground, and with a grounded language (each symbol a word), each
          followed by its continuous world (playback pages with the Decisions tab), plus a baseline
          playback of untrained brains deciding in the same world (export_playback --decisions).
          On an A100 (40 GB) use --worlds 24 --agent-chunk 4 (about 21 GB; 64 worlds do not fit).
report    Training results grouped by condition (scripts/compare_runs.py) and a table of how each
          continuous world went.

Everything resumes: finished runs are skipped, unfinished ones continue from their checkpoint.
A run trained with other code is never reused: the driver stops before any work and lists it
(keep one --root per code version; the Colab notebook does this by itself).
On HPRC each array task runs one item with --only K (see hprc/*.slurm); without --only, all
items run one after another (Colab).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lifesim.presets import PRESETS, code_version  # noqa: E402
sys.path.insert(0, str(HERE))
from society_data import day_length_of  # noqa: E402

PRESET = "society"
# Shock, start day, length (days), strength (lifesim/threats.py)
STRESS = [("famine", "famine:40:10:0.2"), ("plague", "plague:40:10:10"), ("predators", "predators:40:10:3"),
          ("cold", "cold:40:10:10"), ("drought", "drought:40:10:2")]
PLACED_SEEDS = (21, 22)


def which(args, default):
    from lifesim import archetypes as arch
    w = getattr(args, "which", None) or default
    names = sorted(arch.PROFILES) if w == "all" else [x.strip() for x in w.split(",") if x.strip()]
    bad = [x for x in names if x not in arch.PROFILES]
    if bad:
        raise SystemExit(f"unknown archetypes {bad}; known: {', '.join(sorted(arch.PROFILES))}")
    return names


def items(kind, args=None):
    if kind == "archetypes":
        # Intervention in training: one archetype among the founders; the control is core a0.0_f1.0_s*
        return [(f"{a}_s{seed}", ["--archetypes", f"{a}:1", "--seed", str(seed)])
                for a in which(args, "tyrant,saint") for seed in (0, 1, 2)]
    if kind == "placed":
        # Intervention while living: the validated caveman agents keep learning, one founder is
        # given an archetype's temperament; the control is the same world with none
        out = []
        for seed in PLACED_SEEDS:
            out.append((f"none_s{seed}", ["--seed", str(seed), "--map-seed", str(seed)]))
            out += [(f"{a}_s{seed}", ["--seed", str(seed), "--map-seed", str(seed), "--archetypes", f"{a}:1"])
                    for a in which(args, "all")]
        return out
    if kind == "culture":
        # Same map and founders, different histories; brains do not pass learning down (innate), so
        # what persists across generations passes by watching. solitary: no learning by watching.
        reps = getattr(args, "replicates", None) or 4
        return [(f"{cond}_r{r}", ["--seed", str(31 + r), "--map-seed", "21", "--inherit", "innate"]
                 + (["--imitation", "0"] if cond == "solitary" else []))
                for cond in ("social", "solitary") for r in range(reps)]
    if kind == "probe":
        base = ["--no-excretion"]
        return [("full", base), ("no_sleep_learning", base + ["--no-sleep-learning"]),
                ("caves", base + ["--underground"]), ("grounded_language", base + ["--language", "grounded"])]
    if kind == "validate":
        return [("caveman", []), ("caveman_no_sleep_learning", ["--no-sleep-learning"]),
                ("blank", ["--slate", "blank"]), ("caveman_no_world_model", ["--no-world-model"])]
    if kind == "stress":
        return [(k, [s]) for k, s in STRESS]
    if kind == "inheritance":
        return [(f"{mode}_s{seed}", [mode, str(seed)]) for mode in ("parent", "innate") for seed in (21, 22, 23)]
    out = []
    for alpha in (0.0, 0.5, 1.0):
        for food in (1.0, 0.4):
            for seed in (0, 1, 2):
                out.append((f"a{alpha}_f{food}_s{seed}",
                            ["--alpha", str(alpha), "--scarcity", str(food), "--seed", str(seed)]))
    return out


def run(cmd, log):
    print("$ " + " ".join(cmd), flush=True)
    with open(log, "a") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if line.startswith("{"):
                r = json.loads(line)
                if "update" in r:
                    d = r["deaths"]
                    print(f"  steps {r['agent_steps']:>12,}  lifespan {str(r['mean_age_at_death']):>7}  "
                          f"thirst {d.get('thirst', 0) / max(1, sum(d.values())):4.0%}  births {r.get('births')} (cap stopped {r.get('conceptions_blocked_by_slot_cap', 0)})  "
                          f"asleep n/d {r.get('asleep_share_night')}/{r.get('asleep_share_day')}  "
                          f"learned {r.get('learned_share')}  {r.get('agent_sps', 0):,.0f}/s", flush=True)
                elif "chunk" in r:
                    p = r["population"]
                    print(f"  chunk {r['chunk']:>3} {r['to']:>20}  alive {p['alive']:>2}  "
                          f"births so far {r['births_so_far']}  generation {p['max_generation']}  slots {r.get('slots')}", flush=True)
            elif line.strip() and not line.startswith(("backend=", "using checkpoint")):
                print("  " + line.rstrip(), flush=True)
        return proc.wait()


def version_of(run_dir: Path):
    f = run_dir / "history.json"
    return json.load(open(f))["args"].get("code_version") if f.exists() else None


def stale(args, kind, todo):
    """Runs this command would reuse or resume that were made by other code."""
    now, bad = code_version(), []
    dirs = [Path(args.root) / kind / name for name, _ in todo] if kind in ("validate", "core", "archetypes", "probe") \
        else [Path(args.root) / "validate" / "caveman"]
    for d in dirs:
        v = version_of(d)
        if (d / "params.msgpack").exists() or (d / "checkpoint.pkl").exists():
            if v != now:
                bad.append(f"{d} (code {v})")
    return bad


def gate(root, adult_days=15.0, day_length=None):
    """(passed, lines): did the caveman agents' continuous world outlast its founders? Passed when
    someone born in the world reached fertile age (15 biological years, i.e. 15 days): the next
    generation could carry on. Births alone are not enough (infants die), nor is a long-lived world
    of founders (most die of age within a few dozen days of the start)."""
    cont = Path(root) / "validate" / "caveman" / "continuous"
    f = cont / "summary.json"
    if not f.exists() or not json.load(open(f)):
        return False, [f"no continuous world for {cont.parent}: run `experiment.py validate` first"]
    S = json.load(open(f))
    last = S[-1]
    births = last["births_so_far"]
    matings = sum(c.get("reproduction", {}).get("matings", 0) for c in S)
    lives = json.load(open(cont / "lives.json")) if (cont / "lives.json").exists() else []
    h = cont.parent / "history.json"
    if h.exists():                  # adulthood in days follows the run's time scale
        adult_days = 15.0 * json.load(open(h))["args"].get("days_per_bio_year", 1.0)
    day_length = day_length or day_length_of(cont)
    grown = [l for l in lives if l.get("born_t", 0) > 0 and l["ticks"] >= adult_days * day_length]
    ok = len(grown) > 0
    capped = sum(c.get("reproduction", {}).get("blocked_by_slot_cap", 0) for c in S)
    lines = [f"caveman continuous world: {5 * len(S)} days, {matings} matings, {births} births, "
             f"{len(grown)} born there reached adulthood, alive at the end {last['population']['alive']}",
             "GATE PASSED: a generation born in the world grew up" if ok else
             "GATE FAILED: no one born in the trained agents' world grew up, so it cannot outlast its "
             "founders. Fix that before spending GPU hours on core, archetypes, inheritance or stress "
             "(--skip-gate runs them anyway)."]
    if capped:
        lines.append(f"  slot cap: {capped} conceptions found every slot taken, so the cap, not the world, "
                     "limited births (a world grows when nearly full; this many arrived faster than it could)")
    return ok, lines


def do_item(name, extra, args, kind):
    out = Path(args.root) / kind / name
    out.parent.mkdir(parents=True, exist_ok=True)
    a = PRESETS[PRESET]
    ticks = int(args.agent_steps / (args.worlds * a["agents"]))
    ticks = max(a["steps"], ticks // a["steps"] * a["steps"])
    if not (out / "params.msgpack").exists():
        cmd = [sys.executable, str(HERE / "train_phase2.py"), "--preset", PRESET, "--worlds", str(args.worlds),
               "--ticks", str(ticks), "--log-every", "10", "--ckpt-every", "10", "--out", str(out), "--resume",
               *(["--agent-chunk", str(args.agent_chunk)] if getattr(args, "agent_chunk", 0) else []), *extra]
        if run(cmd, f"{out}.log") != 0:
            raise SystemExit(f"training {name} failed; see {out}.log")
    else:
        print(f"{name}: trained", flush=True)
    cont = out / "continuous"
    if args.days > 0 and not (cont / "finished").exists():
        pb_days = args.playback_days if args.playback_days is not None else (0.25 if kind == "probe" else None)
        cmd = [sys.executable, str(HERE / "continuous_world.py"), "--run", str(out), "--days", str(args.days),
               "--chunk-days", "5", "--playback-every", str(args.playback_every), "--seed", "21",
               "--out-dir", str(cont), *(["--playback-days", str(pb_days)] if pb_days is not None else [])]
        if run(cmd, f"{out}.continuous.log") != 0:
            raise SystemExit(f"continuous world for {name} failed; see {out}.continuous.log")
        (cont / "finished").write_text("")
    print(f"{name}: done", flush=True)


def do_baseline_decisions(args):
    """Untrained brains deciding in the probe's world: the baseline the trained runs' playbacks are
    read against (what a fresh network does before any learning)."""
    out = Path(args.root) / "probe" / "baseline_decisions.html"
    if out.exists():
        print("baseline decisions: done", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    a = PRESETS[PRESET]
    cmd = [sys.executable, str(HERE / "export_playback.py"), "--decisions", "--world", a.get("world", "living"),
           "--slate", a.get("slate", "caveman"), "--grid", str(a["grid"]), "--agents", str(a["agents"]),
           "--tick-minutes", str(a.get("tick_minutes", 30.0)), "--encoder", a.get("encoder", "conv"),
           "--days", "0.25", "--seed", "21", "--out", str(out)]
    if run(cmd, f"{out}.log") != 0:
        raise SystemExit(f"baseline decisions playback failed; see {out}.log")


def do_inheritance(name, extra, args):
    mode, seed = extra
    run_dir = Path(args.root) / "validate" / "caveman"
    if not (run_dir / "params.msgpack").exists():
        raise SystemExit(f"{run_dir} is not trained yet: run `experiment.py validate` first")
    out = Path(args.root) / "inheritance" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "finished").exists():
        print(f"{name}: done", flush=True)
        return
    cmd = [sys.executable, str(HERE / "continuous_world.py"), "--run", str(run_dir), "--days", str(args.days),
           "--chunk-days", "5", "--playback-every", str(args.playback_every), "--seed", seed,
           "--learn", "--inherit", mode, "--out-dir", str(out)]
    if run(cmd, f"{out}.log") != 0:
        raise SystemExit(f"inheritance world {name} failed; see {out}.log")
    (out / "finished").write_text("")
    print(f"{name}: done", flush=True)


def do_living(kind, name, extra, args):
    """A continuous world of the validated caveman agents, learning as they live."""
    run_dir = Path(args.root) / "validate" / "caveman"
    if not (run_dir / "params.msgpack").exists():
        raise SystemExit(f"{run_dir} is not trained yet: run `experiment.py validate` first")
    out = Path(args.root) / kind / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "finished").exists():
        print(f"{name}: done", flush=True)
        return
    cmd = [sys.executable, str(HERE / "continuous_world.py"), "--run", str(run_dir), "--days", str(args.days),
           "--chunk-days", "5", "--playback-every", str(args.playback_every), "--learn", "--out-dir", str(out), *extra]
    if run(cmd, f"{out}.log") != 0:
        raise SystemExit(f"{kind} world {name} failed; see {out}.log")
    (out / "finished").write_text("")
    print(f"{name}: done", flush=True)


def do_stress(name, extra, args):
    run_dir = Path(args.root) / "validate" / "caveman"
    if not (run_dir / "params.msgpack").exists():
        raise SystemExit(f"{run_dir} is not trained yet: run `experiment.py validate` first")
    out = Path(args.root) / "stress" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "finished").exists():
        print(f"{name}: done", flush=True)
        return
    cmd = [sys.executable, str(HERE / "continuous_world.py"), "--run", str(run_dir), "--days", str(min(args.days, 120)),
           "--chunk-days", "5", "--playback-every", "2", "--seed", "21", "--shock", extra[0], "--out-dir", str(out)]
    if run(cmd, f"{out}.log") != 0:
        raise SystemExit(f"stress world {name} failed; see {out}.log")
    (out / "finished").write_text("")
    print(f"{name}: done", flush=True)


def stress_row(label, S, start=40, length=10):
    """Alive before the shock, at the worst point up to 20 days after it ends, and 40 days after;
    deaths during the shock by cause."""
    alive_at = lambda day: next((s["population"]["alive"] for s in S if 5 * (s["chunk"] + 1) >= day), S[-1]["population"]["alive"])
    window = [s for s in S if start < 5 * (s["chunk"] + 1) <= start + length + 20]
    worst = min((s["population"]["alive"] for s in window), default=None)
    during = {}
    for s in S:
        if start < 5 * (s["chunk"] + 1) <= start + length:
            for k, v in s["deaths"].items():
                during[k] = during.get(k, 0) + v
    return (f"{label:<14}{alive_at(start):>8}{str(worst):>8}{alive_at(start + 40):>9}  "
            + (", ".join(f"{k} {v}" for k, v in sorted(during.items(), key=lambda kv: -kv[1])) or "none"))


def world_row(label, S):
    deaths, seasons = {}, {}
    for s in S:
        for k, v in s["deaths"].items():
            deaths[k] = deaths.get(k, 0) + v
        for k, v in s.get("deaths_by_season", {}).items():
            seasons[k] = seasons.get(k, 0) + v
    winter = f"{seasons.get('Winter', 0) / max(1, sum(seasons.values())):.0%}" if seasons else "-"
    last = S[-1]
    sl = last.get("sleep", {})
    return (f"{label:<30}{5 * len(S):>6}{last['population']['alive']:>7}"
            f"{last['births_so_far']:>8}{str(last['population']['max_generation']):>5}"
            f"{str(sl.get('asleep_share_night')) + '/' + str(sl.get('asleep_share_day')):>12}{winter:>8}  "
            + ", ".join(f"{k} {v}" for k, v in sorted(deaths.items(), key=lambda kv: -kv[1])))


def society_section(root, runs):
    """Society in each run's continuous world (scripts/society.py), the last full year: groups,
    company sought beyond chance, bands, sharing, joint hunts, division of labour, punishment; the
    full per-year results go to <root>/society.json."""
    sys.path.insert(0, str(HERE))
    import society
    from society_data import World
    rows, full = [], {}
    for r in runs:
        cont = r / "continuous"
        if not (cont / "trace.npz").exists():
            continue
        try:
            res = society.analyse(World(cont, need_trace=True), window_days=40.0, nulls=39)
        except (SystemExit, Exception) as e:                     # one bad world must not stop the report
            print(f"  society: {r.parent.name}/{r.name}: {e}")
            continue
        full[f"{r.parent.name}/{r.name}"] = res
        w = next((x for x in reversed(res["windows"]) if x["bands"]), None)
        if w is None:
            continue
        b, s, h, lab, pu = w["bands"], w["sharing"], w["hunting"], w["labour"], w["punishment"]
        f = society._f
        rows.append(f"{r.parent.name + '/' + r.name:<30}{w['from_day']:>6.0f}{f(b['typical_group_size']):>7}{f(b['alone']):>7}"
                    f"{f(b.get('association_ratio')):>8} ({f(b.get('gregarious_p'), 3):>5}){len(b['bands']):>6}"
                    f"{f(s.get('gifts_per_person_year')):>9}{f(s.get('reciprocity_r')):>7}{f(h.get('joint_share')):>7}"
                    f"{f(lab.get('dol')):>7} ({f(lab.get('dol_p'), 3):>5}){f(pu.get('punishment_ratio')):>7} ({f(pu.get('p'), 3):>5})")
    if rows:
        print("\n=== society in the continuous worlds (last year; scripts/society.py for every year)")
        print(f"{'run':<30}{'day':>6}{'group':>7}{'alone':>7}{'assoc x':>8} {'(p)':>7}{'bands':>6}{'gifts/py':>9}"
              f"{'recip':>7}{'joint':>7}{'DOL':>7} {'(p)':>7}{'punish':>7} {'(p)':>7}")
        print("\n".join(rows))
        (Path(root) / "society.json").write_text(json.dumps(full, indent=1, default=str))


def report(root):
    root = Path(root)
    runs = sorted(p.parent for p in root.glob("*/*/history.json"))
    if not runs:
        raise SystemExit(f"no runs under {root}")
    versions = {json.load(open(r / "history.json"))["args"].get("code_version") for r in runs}
    if len(versions) > 1:
        print(f"WARNING: runs come from {len(versions)} code versions {sorted(map(str, versions))}; "
              "compare_runs keeps them apart", flush=True)
    for kind in ("validate", "probe", "core", "archetypes"):
        group = [str(r) for r in runs if r.parent.name == kind]
        if group:
            print(f"\n=== {kind}: training (last 20% of each run, mean and 95% interval over seeds)")
            subprocess.run([sys.executable, str(HERE / "compare_runs.py"), *group])
    print("\n=== continuous worlds")
    print(f"{'run':<30}{'days':>6}{'alive':>7}{'births':>8}{'gen':>5}{'asleep n/d':>12}{'winter':>8}  deaths")
    for r in runs:
        f = r / "continuous" / "summary.json"
        if f.exists() and json.load(open(f)):
            print(world_row(r.parent.name + "/" + r.name, json.load(open(f))))
    society_section(root, runs)
    st = sorted((root / "stress").glob("*/summary.json"))
    if st:
        print("\n=== stress: a 10-day shock from day 40 (control: the run's own unshocked world)")
        print(f"{'shock':<14}{'before':>8}{'worst':>8}{'+40 d':>9}  deaths during the shock")
        ctrl = root / "validate" / "caveman" / "continuous" / "summary.json"
        if ctrl.exists() and json.load(open(ctrl)):
            print(stress_row("none", json.load(open(ctrl))))
        for f in st:
            S = json.load(open(f))
            if S:
                print(stress_row(f.parent.name, S))
    inh = sorted((root / "inheritance").glob("*/summary.json"))
    if inh:
        print("\n=== inheritance: worlds where individuals keep learning (parent = Lamarckian, innate = Darwinian)")
        print(f"{'world':<30}{'days':>6}{'alive':>7}{'births':>8}{'gen':>5}{'asleep n/d':>12}{'winter':>8}  deaths")
        for f in inh:
            S = json.load(open(f))
            if S:
                print(world_row(f.parent.name, S))
    sys.path.insert(0, str(HERE))
    import hierarchy
    from society_data import World
    pl = sorted((root / "placed").glob("*/summary.json"))
    if pl:
        print("\n=== placed: one founder with an archetype's temperament, learning as they live")
        print(f"{'world':<22}{'days':>6}{'alive':>7}{'births':>8}{'gen':>5}{'asleep n/d':>12}{'winter':>8}  deaths")
        for f in pl:
            S = json.load(open(f))
            if S:
                print(world_row(f.parent.name, S))
        print("\nwhere the archetype stood (dominance score 0..N-1, net leadership -1..1, children):")
        for f in pl:
            if f.parent.name.startswith("none_"):
                continue
            try:
                w = World(f.parent)
                h = hierarchy.analyse(w)
                arch = [l for l in w.lives if l.get("archetype")]
                kids = w.children()
                whole = {}
                for win in h["windows"]:
                    whole.update(win.get("scores", {}))
                for l in arch:
                    print(f"  {f.parent.name:<20} {l['archetype']:<11} lived {l['ticks'] / day_length_of(f.parent):5.1f} d, "
                          f"dominance {whole.get(l['id'], 'n/a')}, leadership "
                          f"{h['leadership']['net_by_life'].get(l['id'], 'n/a')}, children {kids[l['id']]}")
            except SystemExit as e:
                print(f"  {f.parent.name}: {e}")
    soc = sorted(p.parent for p in (root / "culture").glob("social_*/lives.json"))
    sol = sorted(p.parent for p in (root / "culture").glob("solitary_*/lives.json"))
    if soc:
        import culture
        print("\n=== culture: replicate worlds, same map, different histories (scripts/culture.py)")
        print("\n".join(culture.render(culture.analyse(soc), "Learning by watching (social)")))
        if sol:
            print("\n".join(culture.render(culture.analyse(sol), "No learning by watching (solitary)")))
    cav = root / "validate" / "caveman" / "continuous"
    if (cav / "lives.json").exists():
        print("\n=== hierarchy in the caveman continuous world (scripts/hierarchy.py)")
        print("\n".join(hierarchy.render(hierarchy.analyse(World(cav)))))
    for kind in ("validate", "probe", "core", "archetypes"):
        if list((root / kind).glob("*/continuous/lives.json")):
            print(f"\n=== {kind}: roles agents fell into (scripts/archetypes.py)")
            subprocess.run([sys.executable, str(HERE / "archetypes.py"), str(root / kind)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["validate", "probe", "core", "archetypes", "placed", "culture", "inheritance", "stress",
                                     "gate", "report"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--agent-steps", type=float, default=1e8, help="training budget per run")
    ap.add_argument("--days", type=float, default=1200,
                    help="continuous world after training (0 = none); 1,200 days = 30 years at 40 days per "
                         "year of age, long enough for a child born early to grow up (600 days)")
    ap.add_argument("--playback-every", type=int, default=8)
    ap.add_argument("--only", type=int, default=None, help="run just item K (HPRC array task)")
    ap.add_argument("--playback-days", type=float, default=None,
                    help="days kept in each continuous-world playback page (probe default 0.25: at 1-minute "
                         "ticks a whole day on a 250 x 250 map is a very large page)")
    ap.add_argument("--agent-chunk", type=int, default=0,
                    help="passed to training: agents per chunk in the update (less memory; 4 on an A100 for society)")
    ap.add_argument("--skip-gate", action="store_true", help="run even if the persistence gate fails")
    ap.add_argument("--preset", default="society", choices=sorted(PRESETS),
                    help="run settings for trained runs (lifesim/presets.py); realtime = 365-day years and real "
                         "biological time")
    ap.add_argument("--which", default=None,
                    help="archetypes / placed: comma-separated archetypes or 'all' (lifesim/archetypes.py); "
                         "default tyrant,saint for archetypes, all for placed")
    ap.add_argument("--replicates", type=int, default=4, help="culture: worlds per condition")
    args = ap.parse_args()
    global PRESET
    PRESET = args.preset
    if args.kind == "report":
        return report(args.root)
    if args.kind == "gate":
        ok, lines = gate(args.root)
        print("\n".join(lines))
        raise SystemExit(0 if ok else 1)
    print(f"code version {code_version()}", flush=True)
    todo = items(args.kind, args)
    if args.only is not None:
        todo = [todo[args.only]]
    bad = stale(args, args.kind, todo)
    if bad:
        raise SystemExit("These runs were made by other code and would be reused or resumed:\n  "
                         + "\n  ".join(bad) + f"\nUse a fresh --root for code {code_version()} "
                         "(results from different code must not be mixed).")
    if args.kind not in ("validate", "probe") and not args.skip_gate:
        ok, lines = gate(args.root)
        if not ok:
            raise SystemExit("\n".join(lines))
    if args.kind == "probe" and args.only is None:
        do_baseline_decisions(args)              # first: it needs no training (untrained brains deciding)
    for name, extra in todo:
        if args.kind == "inheritance":
            do_inheritance(name, extra, args)
        elif args.kind in ("placed", "culture"):
            do_living(args.kind, name, extra, args)
        elif args.kind == "stress":
            do_stress(name, extra, args)
        else:
            do_item(name, extra, args, args.kind)


if __name__ == "__main__":
    main()
