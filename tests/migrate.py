import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1300,'height':900})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:250]))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3200)
        r = await pg.evaluate("""(()=>{const A=__agon;A.setState({playing:false});
          const snap=(t)=>{const s=A.state;return {when:t, tick:s.tick, cell:s.cell?(s.cell.x+','+s.cell.y):null,
            alive:(s.agents||[]).filter(a=>a.alive).length,
            ids:(s.agents||[]).filter(a=>a.alive).slice(0,3).map(a=>a.id+':'+a.name),
            food:s.store?s.store.food:null, techs:s.store&&s.store.techs?s.store.techs.length:null,
            sat:(s.agents||[]).filter(a=>a.alive).slice(0,1).map(a=>+(a.sat||0).toFixed(2))[0]};};
          // run a while so the colony develops tech and age
          for(let i=0;i<1200;i++)A.advance();
          const before=snap('before');
          const c=A.state.cell||{x:0,y:0};
          A.migrate({x:c.x+1, y:c.y});
          const after=snap('after migrate');
          return JSON.stringify({before,after},null,1);
        })()""")
        print(r); print('errors:', errs[:3] if errs else 'none')
        await b.close()
asyncio.run(main())
