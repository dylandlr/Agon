import asyncio, re
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1400,'height':900})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
        await pg.add_init_script("window.AGON_P={seed:3, disc:'all', buildT:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
        r = await pg.evaluate("""(()=>{const A=__agon;A.setState({playing:false});
          // run to night so the darkness mask is active
          while ((A.state.tick % 50) < 40 || A.state.tick < 60) A.advance();
          const out=[];
          const probe=(label)=>{ const rv=A.renderVals ? A.renderVals() : null;
            const ns = rv && rv.nightStyle ? rv.nightStyle : '';
            // count hearth light circles inside the mask uri
            const circles = (decodeURIComponent(ns).match(/<circle/g)||[]).length;
            out.push({label, fire: !!A.E.CAMPON.fire, maskCircles: circles,
                      nightVisible: ns.indexOf('display: none')===-1,
                      fireSprite: rv && rv.fire ? rv.fire.outer.indexOf('display: none')===-1 : null,
                      glow: rv && rv.fireGlow ? rv.fireGlow.indexOf('display: none')===-1 : null}); };
          probe('fire on');
          A.E.CAMPON.fire = false;   // hearth gone
          probe('fire gone');
          A.E.CAMPON.fire = true;
          return JSON.stringify(out,null,1);
        })()""")
        print(r); print('errors:', errs[:2] if errs else 'none')
        await b.close()
asyncio.run(main())
