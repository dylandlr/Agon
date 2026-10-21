"""Record the scripted band of scripts/viability.py as a replay and a causal log.

    python scripts/record_scripted.py --out-dir runs/scripted_s0 --days 1200 --seed 0
    python scripts/causal_story.py runs/scripted_s0 --out runs/scripted_s0/story.txt

The same world and the same hand-scripted competent band as viability.py (an upper bound on what
the world allows, not learned behaviour), with the playback recorder on: every event of every
tick goes to events.jsonl.gz (what causal_story.py reads), and every --record-every-th chunk of
--chunk-days becomes a playback page (chunk_NNN.html), linked from index.html with the population
over time. Pages are kept for a few chunks only because a page of 5 days is a few MB.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifesim import env as E  # noqa: E402
from lifesim import techtree  # noqa: E402
from lifesim.analysis import language_metrics  # noqa: E402
from lifesim.presets import PRESETS  # noqa: E402
from lifesim.recorder import _stamp_events, extract_events  # noqa: E402
from export_playback import build_frame, knowledge_snapshot, page_html, world_static  # noqa: E402
from viability import build_cfg, scripted_policy  # noqa: E402

MORALS = ["empathy", "kin_care", "envy", "guilt", "reciprocity", "vengefulness", "sadism"]


def agents_meta(state, events):
    geno = jax.tree_util.tree_map(np.asarray, state.genotype)
    n = state.pos.shape[0]
    skills = [dict() for _ in range(n)]
    for ev in events:
        if 0 <= ev.get("agent", -1) < n:
            for b in techtree.blocks_of(ev):
                skills[ev["agent"]][b] = skills[ev["agent"]].get(b, 0) + 1
    return [{"id": i, "appearance": [round(float(x), 3) for x in np.asarray(state.appearance)[i]],
             "morals": {m: round(float(getattr(geno, m)[i]), 2) for m in MORALS},
             "traits": {t: round(float(getattr(geno, t)[i]), 3) for t in geno._fields},
             "mutations": {t: round(float(v), 1) for t, v in zip(geno._fields, np.asarray(state.mutations)[i]) if v},
             "skills": skills[i]} for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preset", default="society30", choices=sorted(PRESETS))
    ap.add_argument("--world", default=None, choices=["classic", "living"])
    ap.add_argument("--food", type=float, default=1.0)
    ap.add_argument("--knowledge", default="own", choices=["full", "own"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=1200)
    ap.add_argument("--chunk-days", type=float, default=5)
    ap.add_argument("--record-every", type=int, default=20, help="record every Nth chunk as a page (1 = all)")
    ap.add_argument("--tick-minutes", type=float, default=None)
    ap.add_argument("--patch-leave", type=float, default=None,
                    help="seek-food leaves patches poorer than this x the known average (default 0.5; 0 = off)")
    args = ap.parse_args()
    cfg = build_cfg(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    act = scripted_policy(cfg, args.knowledge)

    @jax.jit
    def tick(s):
        a = act(s)
        s2, _, _, info = E.step(s, a, cfg)
        return s2, a, info

    state = E.reset(jax.random.PRNGKey(args.seed), cfg)
    n, day = cfg.n_agents, cfg.day_length
    chunk_ticks, total = int(args.chunk_days * day), int(args.days * day)
    last_event, pages, pop = {}, [], []
    log = gzip.open(out / "events.jsonl.gz", "wt")
    t, chunk = 0, 0
    while t < total:
        record = chunk % max(1, args.record_every) == 0
        frames, events, knowledge = [], [], []
        t0 = t
        for _ in range(min(chunk_ticks, total - t)):
            prev = state
            state, a, info = tick(state)
            evs = _stamp_events(extract_events(info, np.asarray(a), last_event, state, prev, cfg), cfg)
            for ev in evs:
                log.write(json.dumps(ev, separators=(",", ":")) + "\n")
            if record:
                events.extend(evs)
                frames.append(build_frame(t, prev, state, a, info, evs, cfg, n))
                if (t - t0) % 12 == 0:
                    knowledge.append(knowledge_snapshot(t, state))
            if t % day == 0:
                pop.append({"day": t // day, "alive": int(np.asarray(state.alive).sum())})
            t += 1
        if record:
            stamp = lambda tt: E.clock(tt, cfg)["stamp"]
            data = {"grid": cfg.grid_size, "terrain": "".join(str(int(v)) for v in np.asarray(state.terrain).reshape(-1)),
                    "source": f"scripted band (viability.py, knowledge {args.knowledge}), seed {args.seed}: "
                              f"{stamp(t0)} to {stamp(t - 1)}",
                    "view_radius": cfg.view_radius, "inventory_cap": cfg.inventory_cap,
                    "agents": agents_meta(state, events), "frames": frames, "events": events,
                    "knowledge": knowledge, **world_static(state, cfg)}
            data["language"] = language_metrics(data)
            name = f"chunk_{chunk:03d}.html"
            (out / name).write_text(page_html(data))
            pages.append({"page": name, "from": stamp(t0), "to": stamp(t - 1),
                          "alive": int(np.asarray(state.alive).sum())})
            print(f"wrote {name}: {stamp(t0)} to {stamp(t - 1)}, {pages[-1]['alive']} alive", flush=True)
        chunk += 1
    log.close()
    json.dump({"pages": pages, "population": pop, "args": vars(args)}, open(out / "summary.json", "w"), indent=1)
    peak = max(p["alive"] for p in pop)
    bars = "".join(f'<i title="day {p["day"]}: {p["alive"]}" style="height:{100 * p["alive"] / max(peak, 1):.0f}%"></i>'
                   for p in pop[::max(1, len(pop) // 300)])
    links = "".join(f'<li><a href="{p["page"]}">{p["from"]} to {p["to"]}</a>, {p["alive"]} alive at the end</li>'
                    for p in pages)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Scripted band replay</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:2em auto;padding:0 16px;background:#f7f5ef;color:#222}}
.pop{{display:flex;align-items:flex-end;height:120px;gap:1px;border-bottom:1px solid #999}}.pop i{{flex:1;background:#6b8f4e}}
@media (prefers-color-scheme:dark){{body{{background:#1b1c1a;color:#ddd}}a{{color:#9cc4ff}}}}</style>
<h1>Scripted band, seed {args.seed}</h1>
<p>{args.days:g} days of the hand-scripted band from <code>viability.py</code> ({args.preset}, knowledge {args.knowledge}).
Population each day (peak {peak}):</p><div class="pop">{bars}</div>
<h2>Recorded stretches</h2><ul>{links}</ul>
<p>Every event of every day is in <code>events.jsonl.gz</code>; <code>python scripts/causal_story.py {out}</code> tells it as chains of cause and effect.</p>""")
    print(f"done: {len(pages)} pages, events.jsonl.gz, index.html in {out}")


if __name__ == "__main__":
    main()
