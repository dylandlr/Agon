import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for seed in [3,5,21]:
            pg = await b.new_page(viewport={'width':1400,'height':900})
            errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:150]))
            await pg.add_init_script("window.AGON_P={seed:%d, disc:'all', buildT:1}" % seed)
            await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
            r = await pg.evaluate("""(async () => {
              const A = window.__agon; A.setState({playing:false});
              const counts=[];
              for (let t=0;t<5000;t++) { A.advance(); }
              const E=A.E, W=E.WALLS; let corners=0;
              const pts=[];
              for (const k of W) { const [x,y]=k.split(',').map(Number);
                if (W.has((x+1)+','+y) && W.has(x+','+(y+1)) && W.has((x+1)+','+(y+1))) { corners++; if(pts.length<6) pts.push(k); } }
              return JSON.stringify({tick:A.state.tick, walls:W.size, corners, pts});
            })()""")
            print('seed', seed, r, 'errors:', errs[:2])
            await pg.close()
        await b.close()
asyncio.run(main())
