# Sweep several genuinely different worlds and report survivors plus any page error.
#
# NOTE: window.AGON_P has no seed or biome key -- it is a bag of behaviour flags. Passing
# {seed:3,biome:'desert'} sets keys nothing reads, so every such run is the same default world
# (wseed 33403, temperate). World choice goes through setWorld, which is POSITIONAL:
#   setWorld(size, form, biome, seed, nRivals, gp, cell)
# An object argument hits `if (!SIZES[kk]) return;` and silently does nothing.
import asyncio
from playwright.async_api import async_playwright

WORLDS = [(3, 'temperate'), (5, 'desert'), (7, 'tundra'), (13, 'jungle'), (21, 'temperate')]

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for seed, biome in WORLDS:
            pg = await b.new_page(viewport={'width': 1400, 'height': 900})
            errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
            await pg.add_init_script("window.AGON_P={paused:1}")
            await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
            for _ in range(80):
                if await pg.evaluate("()=>window.__agon?1:null"): break
                await pg.wait_for_timeout(50)
            await pg.wait_for_timeout(400)
            await pg.evaluate("""([sd,bm])=>{__agon.setWorld(undefined,undefined,bm,sd);}""", [seed, biome])
            await pg.wait_for_timeout(1500)
            await pg.evaluate("()=>__agon.setState({playing:false})")
            r = await pg.evaluate("""(()=>{const A=__agon;for(let i=0;i<1500;i++)A.advance();
              const s=A.state;
              return JSON.stringify({world:s.wseed,biome:s.biome,tick:s.tick,
                alive:s.agents.filter(a=>a.alive).length,
                colony:s.agents.filter(a=>a.alive&&!a.fac).length})})()""")
            await pg.wait_for_timeout(400)
            print('seed %-3d %-10s %s  pageerrors: %s' % (seed, biome, r, errs[:3] if errs else 'none'))
            await pg.close()
        await b.close()

asyncio.run(main())
