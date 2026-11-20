# Agon noise-floor panel (r34+). Fixed, reusable set of worlds for every future A/B.
#
# Since r33 the sim is deterministic: repeat runs of one world are byte-identical, so
# run-to-run variance is zero and the ONLY noise is between-seed spread. That means an
# A/B must run this identical panel under both arms and be compared as PAIRED per-seed
# differences, never as two independent means. Do not change PANEL without saying so.
#
# Usage:  python3 -u tests/noise/panel.py [ticks] [out.json]
import asyncio, json, sys, time, statistics as st
from playwright.async_api import async_playwright

BIOMES = ['temperate', 'forest', 'grassland', 'desert', 'tundra', 'highlands',
          'volcanic', 'alpine', 'badlands', 'wetland', 'savanna', 'jungle']
SEEDS = [3, 21]                      # two worlds per biome -> 24 worlds
PANEL = [(b, s) for b in BIOMES for s in SEEDS]

RUN = r"""
(arg) => {
  const [ticks, seed, biome] = arg;
  __agon.setWorld(undefined, undefined, biome, seed);
  return new Promise((res) => setTimeout(() => {
    __agon.setState({ playing: false });
    // time-to-extinction is a far better endpoint than the surviving count: at 6000 ticks the
    // count is floored at 0 in over half the panel, which throws away most of the signal.
    let deathTick = -1, peak = 0;
    for (let i = 0; i < ticks; i++) {
      __agon.advance();
      const A0 = __agon.state.agents || [];
      let n = 0;
      for (let j = 0; j < A0.length; j++) if (A0[j].alive && !A0[j].fac) n++;
      if (n > peak) peak = n;
      if (n === 0 && deathTick < 0) deathTick = __agon.state.tick;
    }
    const s = __agon.state, store = s.store || {}, A = s.agents || [];
    const live = A.filter(a => a.alive && !a.fac);
    const h = (x) => { let n = 5381; for (let i = 0; i < x.length; i++) { n = ((n * 33) ^ x.charCodeAt(i)) >>> 0; } return n.toString(36); };
    // wall tiles still owed in enabled rooms: tells us whether a colony ever finishes a room,
    // which decides whether gating shelter on a finished roof would strand it
    let todo = -1;
    try { todo = __agon.E.campTodo(new Set((store.built) || [])).length; } catch (e) {}
    res({
      tick: s.tick,
      todo,
      deathTick,            // -1 if the colony was still alive at the end
      survived: deathTick < 0 ? ticks : deathTick,
      peak,
      alive: live.length,
      walls: __agon.E.WALLS ? __agon.E.WALLS.size : -1,
      built: (store.built || []).length,
      techs: (store.techs || []).length,
      food: Math.round((store.food || 0) * 100) / 100,
      hash: h(JSON.stringify(A) + '|' + JSON.stringify(store)),
      BS: window.__BS || null,
      biome: s.biome, wseed: s.wseed
    });
  }, 1800));
}
"""

async def run_one(ticks, seed, biome, flags=''):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width': 1000, 'height': 640})
        errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.add_init_script("window.AGON_P={paused:1%s}" % (',' + flags if flags else ''))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(400):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(500)
        r = await pg.evaluate(RUN, [ticks, seed, biome])
        await b.close()
        return r, errs

async def main():
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    out_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mar/panel.json'
    flags = sys.argv[3] if len(sys.argv) > 3 else ''
    rows = []
    t0 = time.time()
    for i, (biome, seed) in enumerate(PANEL):
        r, errs = await run_one(ticks, seed, biome, flags)
        r['seed'] = seed
        r['errs'] = errs[:2]
        rows.append(r)
        print(f"[{i+1:2d}/{len(PANEL)}] {biome:10s} seed {seed:3d}  tick={r['tick']} "
              f"alive={r['alive']:2d} walls={r['walls']:3d} built={r['built']:3d} "
              f"techs={r['techs']:2d} todo={r['todo']:3d} food={r['food']:7.2f} "
              f"{'ERR '+str(errs[:1]) if errs else ''} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(out_path, 'w'), indent=1)

    def stats(key):
        v = [r[key] for r in rows]
        return v, st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), min(v), max(v)
    print(f"\nPANEL: {len(rows)} worlds, {ticks} ticks, between-seed spread (NOT measurement error)")
    print(f"{'metric':8s} {'mean':>8s} {'sd':>8s} {'min':>5s} {'max':>5s}")
    for key in ('survived', 'peak', 'alive', 'walls', 'built', 'techs'):
        v, m, sd, lo, hi = stats(key)
        print(f"{key:8s} {m:8.2f} {sd:8.2f} {lo:5d} {hi:5d}")
    ext = sum(1 for r in rows if r['alive'] == 0)
    print(f"extinct: {ext}/{len(rows)} = {100*ext/len(rows):.0f}%")
    print(f"distinct world hashes: {len({r['hash'] for r in rows})}/{len(rows)}")
    print(f"page errors: {sum(1 for r in rows if r['errs'])}")
    print(f"wall-clock: {time.time()-t0:.0f}s")

if __name__ == '__main__':      # ab.py imports PANEL and RUN from here; don't run on import
    asyncio.run(main())
