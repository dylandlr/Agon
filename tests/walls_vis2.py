import asyncio, json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':900,'height':700})
        await pg.add_init_script("window.AGON_P={seed:3, disc:'all', buildT:1}")
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
        css = await pg.evaluate("typeof __agon.E.SPRITE_CSS==='string' ? __agon.E.SPRITE_CSS : JSON.stringify(Object.keys(__agon.E.SPRITE_CSS||{}))")
        doors = await pg.evaluate("""(()=>{const E=__agon.E;const o={};for(const ax of ['h','v'])for(const op of [false,true])o[ax+(op?'o':'c')]=E.doorParts({x:0,y:0,axis:ax},op);return JSON.stringify(o)})()""")
        await b.close()
        d = json.loads(doors)
        def door(q):
            return ('<div style="position:relative;width:48px;height:48px;background:#8a6a4a">'
                    +'<div style="'+q['outer']+'"><div class="'+q['cls']+'" style="'+q['inner']+'"></div><div style="'+q['detail']+'"></div></div></div>')
        mk=lambda cls,x,y,w=48,h=48: f'<div class="{cls}" style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px"></div>'
        cells=''
        for mt in ['w','t','s']:
            h = mk(f'wl{mt}60',0,0)+mk(f'wl{mt}120',48,0)+mk(f'wl{mt}30',0,48)+mk(f'wl{mt}90',48,48)+mk(f'wcf{mt}',36,36,24,24)
            h += mk(f'wl{mt}60',0,150)+mk(f'wl{mt}120',48,150)+mk(f'wl{mt}30',0,198)+mk(f'wl{mt}90',48,198)
            cells += f'<div style="position:relative;width:96px;height:250px">{h}</div>'
        doorrow = door(d['hc'])+door(d['ho'])+door(d['vc'])+door(d['vo'])
        html = ('<!doctype html><meta charset="utf-8"><style>'+css+'</style>'
                '<body style="background:#6a6152;margin:0;padding:30px">'
                '<div style="display:flex;gap:50px">'+cells+'</div>'
                '<div style="display:flex;gap:30px;margin-top:30px">'+doorrow+'</div></body>')
        open('/tmp/mar/board.html','w').write(html)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':700,'height':480}, device_scale_factor=2)
        await pg.goto('file:///tmp/mar/board.html'); await pg.wait_for_timeout(400)
        await pg.screenshot(path='/tmp/mar/board.png')
        await b.close()
asyncio.run(main())
