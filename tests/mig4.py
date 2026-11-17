import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1300,'height':900})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:250]))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3200)
        r = await pg.evaluate("""(()=>{const A=__agon;A.setState({playing:false});
          for(let i=0;i<300;i++)A.advance();
          A.state.agents.forEach(a=>{a.sat=1;a.hyd=1;a.ene=1;}); A.state.store.food=99;
          const snap=()=>{const s=A.state,al=s.agents.filter(a=>a.alive);
            return {tick:s.tick, pcell:s.pcell.x+','+s.pcell.y, alive:al.length,
              first:al[0]?al[0].name:null, techs:s.store.techs.length, food:s.store.food,
              sat:al[0]?+al[0].sat.toFixed(2):null};};
          const out={start:snap()};
          // find a land region nearby to trek to
          const p0=A.state.pcell; let target=null;
          for(let d=1;d<=3&&!target;d++)
            for(const [dx,dy] of [[d,0],[0,d],[-d,0],[0,-d],[d,d],[-d,-d]]){
              const before=A.state.pcell.x+','+A.state.pcell.y;
              A.migrate({px:p0.x+dx,py:p0.y+dy});
              if(A.state.pcell.x+','+A.state.pcell.y!==before){target=[p0.x+dx,p0.y+dy];break;}
            }
          out.target=target; out.afterTrek=snap();
          // a refused migration must not poison the next world change
          A.migrate({px:0,py:0});
          out.afterRefused=snap();
          return JSON.stringify(out,null,1);
        })()""")
        print(r); print('errors:', errs[:3] if errs else 'none')
        await b.close()
asyncio.run(main())
