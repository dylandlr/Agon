import asyncio, json
from playwright.async_api import async_playwright

RUN = r"""
(k) => {
  __agon.setState({ playing: false });
  const q = (v) => (typeof v === 'number' ? (Number.isInteger(v) ? v : Math.round(v*1e12)/1e12) : v);
  const proj = () => {
    const s = __agon.state, A = s.agents || [], out = [s.tick];
    for (let i = 0; i < A.length; i++) {
      const a = A[i];
      out.push(a.id, a.x, a.y, q(a.sat), q(a.hyd), q(a.ene), q(a.temp), q(a.hap), q(a.ang),
               a.alive?1:0, a.g||'', a.act||'', a.esc||0, a.fx||0, q(a.stress), q(a.rw));
    }
    const st = s.store || {};
    out.push((st.built||[]).length, (st.techs||[]).length, q(st.water), (st.items||[]).length);
    for (const kk of Object.keys(st.mats||{}).sort()) out.push(kk, q(st.mats[kk]));
    out.push((s.animals||[]).length, (s.bushes||[]).length, (s.fires||[]).length, (s.burnt||[]).length);
    return out.join(',');
  };
  const h = (str)=>{let x=5381;for(let i=0;i<str.length;i++){x=((x*33)^str.charCodeAt(i))>>>0;}return x.toString(36);};
  const tr = [];
  for (let i = 0; i < k; i++) { __agon.advance(); tr.push(h(proj())); }
  return tr;
}
"""

async def run(ticks):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1000,'height':640})
        await pg.add_init_script("window.AGON_P={paused:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html')
        for _ in range(120):
            if await pg.evaluate("()=>window.__agon?1:null"): break
            await pg.wait_for_timeout(50)
        await pg.wait_for_timeout(500)
        r = await pg.evaluate(RUN, ticks)
        await b.close()
        return r

async def main():
    trs=[]
    for i in range(6):
        t = await run(1300); trs.append(t); print('page',i,'done')
    print()
    for i in range(6):
        for j in range(i+1,6):
            d = next((n+1 for n,(x,y) in enumerate(zip(trs[i],trs[j])) if x!=y), None)
            print(f'  page{i} vs page{j}: first differing tick = {d}')

asyncio.run(main())
