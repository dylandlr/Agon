import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={'width':1500,'height':1000})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:200]))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(3000)
        # start the audio context the way a real first interaction does
        await pg.mouse.click(700, 400); await pg.wait_for_timeout(700)
        async def snap():
            return await pg.evaluate("""(()=>{const A=__agon;const o=JSON.parse(localStorage.getItem('agon.opts')||'{}');
              const btn=[...document.querySelectorAll('button')].find(b=>/^(Sound on|Muted)$/.test(b.textContent.trim()));
              return JSON.stringify({aOn:o.aOn===undefined?'(default true)':o.aOn,
                btnText:btn?btn.textContent.trim():null, pressed:btn?btn.getAttribute('aria-pressed'):null,
                title:btn?btn.getAttribute('title'):null,
                masterGain:A._au?+A._au.master.gain.value.toFixed(3):'no-ctx'})})()""")
        async def clickMute():
            await pg.evaluate("""(()=>{const b=[...document.querySelectorAll('button')].find(b=>/^(Sound on|Muted)$/.test(b.textContent.trim()));b.click();})()""")
            await pg.wait_for_timeout(800)
        print('initial      :', await snap())
        await clickMute(); print('after mute   :', await snap())
        await clickMute(); print('after unmute :', await snap())
        print('errors:', errs[:2] if errs else 'none')
        await b.close()
asyncio.run(main())
