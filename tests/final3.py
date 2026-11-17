import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for seed,params in [(3,"{seed:3}"),(5,"{seed:5,biome:'desert'}"),(7,"{seed:7,biome:'tundra'}"),(21,"{seed:21,sea:1}")]:
            pg = await b.new_page(viewport={'width':1400,'height':900})
            errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
            await pg.add_init_script("window.AGON_P=%s" % params)
            await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
            r = await pg.evaluate("""(()=>{const A=__agon;A.setState({playing:false});for(let i=0;i<1500;i++)A.advance();
              return JSON.stringify({tick:A.state.tick,alive:A.state.agents.filter(a=>a.alive).length})})()""")
            await pg.wait_for_timeout(600)
            print(seed, params, r, 'pageerrors:', errs[:3] if errs else 'none')
            await pg.close()
        await b.close()
asyncio.run(main())
