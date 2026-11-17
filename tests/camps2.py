import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1200,'height':800})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:150]))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(2800)
        for bio,seed in [('temperate',3),('desert',5),('tundra',7),('savanna',11),('jungle',13),('taiga',17)]:
            r = await pg.evaluate("""(([bio,seed])=>{const A=__agon;A.setState({playing:false});
              A.setWorld('medium', A.state.form, bio, seed, 1);
              const E=A.E, W=E.W,H=E.H,TER=E.TER; const c={}; let n=0,bad=0;
              for(let y=0;y<H;y++)for(let x=0;x<W;x++) if(E.inCamp(x,y)){const t=TER[y*W+x];c[t]=(c[t]||0)+1;n++;
                if(t===2||t===4||t===5||t===12||t===13)bad++;}
              return JSON.stringify({biome:bio,camp:n,terr:c,unbuildable:bad});
            })""", [bio,seed])
            print(f"{bio:11s} {r}")
        print('errors:', errs[:2] if errs else 'none')
        await b.close()
asyncio.run(main())
