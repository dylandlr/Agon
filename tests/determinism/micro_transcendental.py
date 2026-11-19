import asyncio
from playwright.async_api import async_playwright

TEST = r"""
() => {
  const fns = {
    sin: x => Math.sin(x), cos: x => Math.cos(x), tan: x => Math.tan(x),
    exp: x => Math.exp(x), log: x => Math.log(1 + x),
    pow1: x => Math.pow(x, 1.37), pow2: x => Math.pow(1.0001, x),
    powop: x => x ** 1.37,
    atan: x => Math.atan(x), atan2: x => Math.atan2(x, 1.7),
    hypot: x => Math.hypot(x, 1.3), sqrt: x => Math.sqrt(x),
    cbrt: x => Math.cbrt(x)
  };
  const xs = []; for (let i = 1; i <= 400; i++) xs.push(i * 0.37 + i / 1000);
  const digest = (f) => { let s = ''; for (const x of xs) s += f(x).toExponential(17) + '|'; return s; };
  const out = {};
  for (const n of Object.keys(fns)) {
    const cold = digest(fns[n]);
    let sink = 0;
    for (let i = 0; i < 300000; i++) sink += fns[n](1 + (i % 97) * 0.013);
    const hot = digest(fns[n]);
    let firstDiff = null;
    if (cold !== hot) {
      const a = cold.split('|'), b = hot.split('|');
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { firstDiff = [xs[i], a[i], b[i]]; break; }
    }
    out[n] = firstDiff ? ('DIFFERS at x=' + firstDiff[0] + '  cold=' + firstDiff[1] + '  hot=' + firstDiff[2]) : 'stable';
    if (sink === 123456789) out[n] += '!';
  }
  return out;
}
"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto('about:blank')
        r = await pg.evaluate(TEST)
        for k,v in r.items(): print(f'  {k:7s}: {v}')
        await b.close()

asyncio.run(main())
