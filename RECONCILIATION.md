# Agon: reconciled workspace

This branch brings the confirmed `colony-sim` and `ecology` versions into one
checkout without replacing any file from either source. Start here; the original
root README remains the colony branch's historical documentation.

## Sources and layout

Exact source commits are pinned in `workspace/sources.json`.

| Version | Source | Location here |
| --- | --- | --- |
| Browser colony game, r25 HUD, Unity port | `colony-sim` at `9609399` | Original root paths: `src/`, `pack/`, `bin/`, `unity/`, `tests/` |
| Eos research simulation, PPO, experiments, playback | `ecology` at `d315c04` | Complete original tree under `research/` |

Both source histories are parents of the reconciliation commit. No original
branches are merged into or overwritten. The colony README's warning against
merging into `main` still applies: this is a separate integration branch. Its
statement that the trees share no files describes the original branch, not this
workspace. The Claude canvas may be older than the canonical colony branch;
the linked sessions could not be read, so this reconciliation uses the two
GitHub branches confirmed by the user.

## Run and verify

From the repository root, with Node installed (no npm install needed):

```sh
npm run verify
npm run build
npm test
```

The fresh browser game is `node_modules/.cache/agon/Agon.html`; open it in a
browser. Historical `bin/` builds are retained verbatim and may lag current
source. The wrapper creates the output directory and builds Main only because
the upstream full-page pack list references an absent `StyleGuide.dc.html`.

For the research engine, use Python 3.13 in its own environment. Always work
from `research/` so notebook, script, data and test-relative paths retain their
original meaning:

```sh
cd research
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements-ci.txt
python -m pip install -e . --no-deps
python -m pytest -q
python scripts/export_playback.py --random --agents 6 --grid 32 --days 2
```

See `research/README.md` for training and `research/hprc/README.md` for the
cluster workflow. For cluster jobs, upload/use `research/` as the project root;
do not submit them from this repository root. Existing notebooks that clone
the original ecology branch still target that branch, not this integration.
The original research CI is retained as reference inside `research/.github/`;
the active root workflow adapts its working directory and dependency paths.

Unity remains a separate port; see `unity/README.md`. Node reference tests do
not compile or execute C# and must not be treated as proof that Unity runs.

## What is reconciled and what remains distinct

| Concern | Colony game | Eos research | Integration decision |
| --- | --- | --- | --- |
| Agent control | Scripted survival and colony behavior | Learned PPO policies, local observations and action masks | Keep separate; scripted game behavior must not be reported as learned emergence |
| State and replay | Browser component state, migration and ruins | JAX arrays, recorder, deterministic replay | No direct save/checkpoint conversion |
| Calendar and balance | Game tick/year/lifespan constants | Configurable physical tick and seasonal calendar | No implicit unit conversion; preserve both balances |
| Technology | Game tech progression and staged camp structures | Generic physical actions and research mechanisms | Preserve provenance; do not substitute game unlocks for experimental evidence |
| Presentation | Current browser HUD and developing Unity port | Playback renderer and body model | Retain both renderers; research playback remains the supported way to inspect a trained run |
| Validation | Browser sweeps, independent JS checks, Unity edit-mode tests | Python/JAX suite and experiment validation | Run each suite in its own working directory |

This is a source/history and build-workflow reconciliation, **not yet one shared
simulation engine**. Neither version's policies can drive the other engine yet.
A follow-up runtime bridge should start with read-only research replay in the
game UI, not live state mutation: versioned snapshots, explicit tick duration,
stable agent IDs, coordinates, normalized needs with units, and event provenance.
Only after replay equivalence is checked should action/observation adapters be
considered. Policy weights require compatible observations and actions; they
cannot simply be loaded into the browser or Unity pawn AI.

Known upstream issues are retained, not silently fixed: see `docs/TODO.md` for
the game (age scale, larder/thirst behavior, unreachable Unity job targets) and
`research/VALIDATION.md` for experimental validation. Rebalancing them while
reconciling would prevent comparison to the original versions.

## Validation of this reconciliation

- All 208 original tracked files match their pinned Git blobs.
- Current browser source packs successfully to a fresh 1.6 MB HTML build.
- All 83 existing Node reference checks pass (links, A*, jobs, world generation).
- Research Python files byte-compile; playback and body-model JavaScript parse;
  all 10 body-model tests pass.
- Full JAX/pytest, browser Playwright and Unity tests were not run locally:
  this environment has Python 3.11 and lacks JAX, Flax, pytest and Playwright.
  The root workflow configures the original pinned research tests on Python 3.13;
  its existence is not a claim those tests have passed.
