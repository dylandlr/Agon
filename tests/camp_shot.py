import asyncio, json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1500,'height':950}, device_scale_factor=2)
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
        await pg.add_init_script("window.AGON_P={seed:3, disc:'all', buildT:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
        r = await pg.evaluate("""(()=>{const A=__agon;A.setState({playing:false});
          for(let i=0;i<212;i++)A.advance(); // late morning, shelter built
          const E=A.E; const d=E.DOORS[0]; const h=E.HEARTH;
          return JSON.stringify({door:d?{x:d.x,y:d.y}:null,hearth:h,tick:A.state.tick})})()""")
        print(r, 'errors:', errs[:2])
        await pg.wait_for_timeout(1200)
        d=json.loads(r)
        # scroll the map container so the camp is visible, then clip around it
        box = await pg.evaluate("""(()=>{const sc=document.querySelector('#agon-root');
          const el=sc&&(sc.scrollWidth>sc.clientWidth?sc:null)||document.scrollingElement; 
          return 'no-scroll' })()""")
        x,y = d['door']['x']*48, d['door']['y']*48
        await pg.evaluate(f"(()=>{{const c=[...document.querySelectorAll('*')].find(e=>e.scrollWidth>e.clientWidth+200&&e.clientWidth>800); if(c){{c.scrollLeft={x}-600;c.scrollTop={y}-400;return 'scrolled'}} return 'none'}})()")
        await pg.wait_for_timeout(800)
        await pg.screenshot(path='/tmp/mar/camp.png')
        await b.close()
asyncio.run(main())
