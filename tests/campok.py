import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1200,'height':800})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
        for bio,seed in [('temperate',3),('temperate',5),('taiga',9),('jungle',13),('tundra',7),('desert',11)]:
            r = await pg.evaluate("""(([bio,seed])=>{const A=__agon;A.setState({playing:false});
              A.setWorld('medium',A.state.form,bio,seed,1);
              const E=A.E,W=E.W,H=E.H;
              // invariant: nothing the colony builds on may be forest or blocked
              let badBuild=0;
              for(const k of E.WALLS) { const [x,y]=k.split(',').map(Number); if(E.TER[y*W+x]===2)badBuild++; }
              for(const k of E.FLOORS){ const [x,y]=k.split(',').map(Number); if(E.TER[y*W+x]===2)badBuild++; }
              let forestInCamp=0, campN=0;
              for(let y=0;y<H;y++)for(let x=0;x<W;x++) if(E.inCamp(x,y)){campN++; if(E.TER[y*W+x]===2)forestInCamp++;}
              for(let i=0;i<2500;i++)A.advance();
              const al=A.state.agents.filter(a=>a.alive).length;
              return JSON.stringify({biome:bio,seed,campTiles:campN,forestInCamp,
                buildTilesOnForest:badBuild, walls:E.WALLS.size, aliveAt2500:al});
            })""", [bio,seed])
            print(r)
        print('errors:', errs[:3] if errs else 'none')
        await b.close()
asyncio.run(main())
