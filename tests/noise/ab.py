# Paired A/B over the fixed seed panel.
#
# Since r33 repeat runs of one world are byte-identical, so run-to-run variance is zero and the
# only noise is between-seed spread. Both arms therefore run the SAME 24 worlds and are compared
# as paired per-seed differences. An unpaired comparison on this panel cannot resolve anything
# smaller than ~2.9 colonists, which is larger than the mean.
#
# Usage:  python3 -u tests/noise/ab.py [ticks] [out.json] [flagsB]
#   flagsB defaults to "noroof:1"; arm A is always the plain build with no flags.
#   (r43: roofs are default-on, so the plain arm has roofs and "noroof:1" is the old baseline.)
import asyncio, json, sys, time, math, statistics as st
from playwright.async_api import async_playwright

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from panel import PANEL, RUN            # one source of truth for the panel and the measurement


async def run_one(ticks, seed, biome, flags):
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


async def arm(ticks, flags, label):
    rows = []
    t0 = time.time()
    for i, (biome, seed) in enumerate(PANEL):
        r, errs = await run_one(ticks, seed, biome, flags)
        r['biome_'], r['seed'] = biome, seed
        r['errs'] = errs[:2]
        rows.append(r)
        print(f"  [{label}] {i+1:2d}/{len(PANEL)} {biome:10s} s{seed:<3d} "
              f"alive={r['alive']:2d} survived={r['survived']:4d} peak={r['peak']:2d} "
              f"built={r['built']:3d} roofTodo={r['todo']:3d}"
              f"{'  ERR ' + str(errs[:1]) if errs else ''} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def paired(A, B, key):
    """Paired difference B - A on one metric. Returns the numbers that decide significance."""
    a = [r[key] for r in A]
    b = [r[key] for r in B]
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    md, sd_d = st.mean(d), (st.stdev(d) if n > 1 else 0.0)
    se = sd_d / math.sqrt(n) if sd_d else 0.0
    t = md / se if se else float('nan')
    # correlation between arms: what pairing actually bought
    try:
        rho = st.correlation(a, b)
    except Exception:
        rho = float('nan')
    mde = 2.93 * se if se else float('nan')
    return dict(metric=key, mean_a=st.mean(a), mean_b=st.mean(b), mean_diff=md,
                sd_d=sd_d, se=se, t=t, rho=rho, mde=mde,
                better=sum(1 for x in d if x > 0), worse=sum(1 for x in d if x < 0),
                same=sum(1 for x in d if x == 0))


async def main():
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/mar/ab.json'
    flagsB = sys.argv[3] if len(sys.argv) > 3 else 'noroof:1'
    print(f"paired A/B, {len(PANEL)} worlds, {ticks} ticks.  A = baseline, B = {flagsB}\n")
    A = await arm(ticks, '', 'A')
    B = await arm(ticks, flagsB, 'B')
    json.dump({'ticks': ticks, 'flagsB': flagsB, 'A': A, 'B': B}, open(out, 'w'), indent=1)

    print(f"\n{'metric':9s} {'A':>7s} {'B':>7s} {'B-A':>7s} {'sd_d':>7s} {'SE':>6s} "
          f"{'t':>6s} {'rho':>6s} {'MDE':>6s}   B better/worse/same")
    for k in ('survived', 'alive', 'peak', 'built', 'techs', 'spoiled'):
        s = paired(A, B, k)
        print(f"{s['metric']:9s} {s['mean_a']:7.2f} {s['mean_b']:7.2f} {s['mean_diff']:7.2f} "
              f"{s['sd_d']:7.2f} {s['se']:6.2f} {s['t']:6.2f} {s['rho']:6.2f} {s['mde']:6.2f}   "
              f"{s['better']}/{s['worse']}/{s['same']}")
    # the r36 target: worlds that actually finish what they start
    # wallTodo is wall-only and therefore comparable across roof arms; todo is not (it counts
    # roof tiles when roofs are on), which made r40's "finished walls" line meaningless.
    fin = lambda r: r.get('wallTodo', r['todo']) == 0 and r['built'] > 0
    finA = sum(1 for r in A if fin(r))
    finB = sum(1 for r in B if fin(r))
    zeroA = sum(1 for r in A if r['built'] == 0)
    zeroB = sum(1 for r in B if r['built'] == 0)
    print(f"\nfinished walls: A {finA}/{len(A)}   B {finB}/{len(B)}"
          f"      never placed a tile: A {zeroA}/{len(A)}   B {zeroB}/{len(B)}")
    exA = sum(1 for r in A if r['alive'] == 0)
    exB = sum(1 for r in B if r['alive'] == 0)
    print(f"\nextinct: A {exA}/{len(A)}   B {exB}/{len(B)}")
    # McNemar on the discordant pairs
    b01 = sum(1 for x, y in zip(A, B) if x['alive'] == 0 and y['alive'] > 0)
    b10 = sum(1 for x, y in zip(A, B) if x['alive'] > 0 and y['alive'] == 0)
    print(f"discordant worlds: B saved {b01}, B killed {b10} (McNemar needs these, not the totals)")
    print(f"page errors: A {sum(1 for r in A if r['errs'])}, B {sum(1 for r in B if r['errs'])}")
    print("\n|t| >= 2.07 is significant at p<0.05 with 23 df. Anything under the MDE column is "
          "not resolvable by this panel.")

asyncio.run(main())
