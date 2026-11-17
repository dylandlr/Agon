import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for seed in [3,5,7,11,21]:
            pg = await b.new_page(viewport={'width':1400,'height':900})
            await pg.add_init_script("window.AGON_P={seed:%d, disc:'all', buildT:1}" % seed)
            await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3500)
            r = await pg.evaluate("""(() => {
              const E = window.__agon && window.__agon.E; if (!E || !E.WALLS) return 'no E.WALLS';
              const W = E.WALLS; let corners=0, walls=W.size;
              for (const k of W) { const [x,y]=k.split(',').map(Number);
                if (W.has((x+1)+','+y) && W.has(x+','+(y+1)) && W.has((x+1)+','+(y+1))) corners++; }
              return JSON.stringify({walls, corners});
            })()""")
            print('seed', seed, r)
            await pg.close()
        await b.close()
asyncio.run(main())
