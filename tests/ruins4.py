import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1300,'height':900})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:250]))
        # slow build so the ruin head-start is observable before rebuild catches up
        await pg.add_init_script("window.AGON_P={seed:3, buildT:400, startTechs:['firecraft','shelter','bedding','carpentry','storage']}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3200)
        r = await pg.evaluate("""(()=>{const A=__agon;A.setState({playing:false});
          const snap=(t)=>{const s=A.state,E=A.E;
            return {when:t, tick:s.tick, cell:s.cell.x+','+s.cell.y, walls:E.WALLS.size,
              floors:E.FLOORS.size, doors:E.DOORS.length, beds:E.BEDS.length, fire:!!E.CAMPON.fire,
              ruins:Object.keys(s.ruins||{}).length};};
          const out=[];
          for(let i=0;i<2600;i++)A.advance();     // let the whole camp go up
          out.push(snap('camp fully built'));
          const h=A.state.cell, hx=h.x, hy=h.y;
          A.migrate({x:hx+1,y:hy}); A.advance();
          out.push(snap('fresh area, nothing built'));
          // age the ruin, then go home
          A.setState({tick:A.state.tick+2000});
          A.migrate({x:hx,y:hy}); A.advance();
          out.push(snap('home after 1 yr: ruin + rebuild starting'));
          for(let i=0;i<1200;i++)A.advance();
          out.push(snap('rebuilding on top of the ruin'));
          return JSON.stringify(out,null,1);
        })()""")
        print(r); print('errors:', errs[:3] if errs else 'none')
        await b.close()
asyncio.run(main())
