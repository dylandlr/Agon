import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1400,'height':900})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
        await pg.add_init_script("window.AGON_P={seed:3}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
        r = await pg.evaluate("""(()=>{const A=__agon,E=A.E;A.setState({playing:false});
          const W=E.W,H=E.H,TER=E.TER;
          const campCells=[]; for(let y=0;y<H;y++)for(let x=0;x<W;x++) if(E.inCamp(x,y)) campCells.push(y*W+x);
          const soil=()=>campCells.filter(i=>TER[i]===0).length;
          const grassAll=()=>{let n=0;for(let i=0;i<TER.length;i++)if(TER[i]===1)n++;return n;};
          const out=[{t:0, campCells:campCells.length, campSoil:soil(), grassTotal:grassAll()}];
          for (const yrs of [1,2,4,8]) {
            while (A.state.tick < yrs*2000) A.advance();
            out.push({t:A.state.tick, year:yrs, campSoil:soil(), grassTotal:grassAll()});
          }
          return JSON.stringify(out);
        })()""")
        print(r); print('errors:', errs[:2] if errs else 'none')
        await b.close()
asyncio.run(main())
