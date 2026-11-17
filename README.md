# Agon — colony sim (p14)

A RimWorld-like emergent colony simulation: one self-contained HTML app, no
dependencies at runtime. Colonists with needs, emergent technology (houses and
tech grow more complex over time), ecology-plausible flora/fauna with unique
per-species art, weather, biomes, rivals.

**This branch (`colony-sim`) is the home of the colony sim and has its own
history. It shares no files with `main` or the other branches (the Agon: Eos
RL/notebook line). Never merge it into `main` and never merge `main` into it.**

## Layout

```
src/     .dc.html components + art.js — the source. Main.dc.html is the game.
pack/    build.js (packs src into one html), dc-runtime.js, recover.js
bin/     built, runnable single-file builds (Agon_p14_rNN.html)
assets/  canvas.json and other non-code assets
docs/    TODO.md round notes, design/research docs
tests/   Playwright harness scripts (python)
```

## Build

```sh
MAINONLY=1 node pack/build.js src /tmp/ab/app_zoom   # writes Agon.html
# full build (all pages): node pack/build.js src /tmp/ab/app_zoom
```

## Test

Tests expect the build at `/tmp/ab/app_zoom/Agon.html` and use
Python + Playwright (chromium):

```sh
python3 tests/smoke.py     # mounts, no page errors
python3 tests/final3.py    # 4 biome configs, 1500 ticks each, error sweep
python3 tests/corner2.py   # 2x2 wall-block scan across seeds
python3 tests/walls_vis2.py # renders a synthetic wall/door board to /tmp/mar/board.png
```

Test contract: drive `window.__agon` (call `setState({playing:false})` before
measuring, then `advance()` in a loop), always capture `pageerror`. Runs are
non-deterministic; survival measurements use 5 runs per seed.

## Recovery

`pack/recover.js <built.html> <wrapperSrcDir> <outDir>` reconstructs the full
`.dc.html` sources from any packed build (the build embeds every component's
script and template verbatim). This is how r18 was recovered after the cloud
workspace was reclaimed; keep committing source + build here so it is never
needed again.

## Working rules

- Back up / commit before each round; end each round with verified / rough /
  not-done notes in docs/TODO.md.
- Never push to `main`.
- The Design-canvas artifact (claude.ai) holds an older copy; this branch is
  canonical for the colony sim.
