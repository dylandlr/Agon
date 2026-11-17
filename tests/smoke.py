import asyncio, json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={'width':1400,'height':900})
        errs=[]; pg.on('pageerror', lambda e: errs.append(str(e)[:300]))
        await pg.goto('file:///tmp/ab/app_zoom/Agon.html'); await pg.wait_for_timeout(4000)
        ok = await pg.evaluate("!!window.__agon && !!window.__agon.state")
        info = await pg.evaluate("window.__agon ? JSON.stringify({tick:__agon.state.tick, agents:(__agon.state.agents||[]).filter(a=>a.alive).length}) : 'none'")
        print('mounted:', ok, info, 'errors:', errs[:3])
        await b.close()
asyncio.run(main())
