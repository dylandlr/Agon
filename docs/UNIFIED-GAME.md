# Unified Agon game: acceptance ledger

Goal: one playable Unity game, with the browser colony experience and the Eos
ecology/agent systems reconciled explicitly. Original branches and reference
implementations remain available. This ledger is an implementation checklist,
not a claim of feature completeness.

## Product decisions

- Unity is the delivery runtime requested by the user.
- `unity-game/` is the developing game; `unity/` is the unchanged original port.
- `src/` remains the browser reference; `research/` remains the Python research
  reference. Do not execute multiple independent simulations and call them one.
- User confirmed real learned policies and research mechanics are essential.
  Eos must remain the authoritative simulation, initially through a local Python
  process and a versioned Unity client protocol. Unity's scripted job system is
  a temporary presentation scaffold, not the finished intelligence system.
  Scripted survival must never be labelled learned intelligence.
- Prefer a local desktop game; no cloud service, account or paid API required.

## Acceptance criteria

| Area | Required evidence | Current status |
| --- | --- | --- |
| Foundation | Unity compiles; edit/play tests execute, not just JS approximations | 63 EditMode tests pass; real Eos PlayMode integration test passes |
| Playable world | Terrain, resources, structures and moving/selectable pawns visible | Live Eos terrain and pawns; roster selection verified; art/resources/structures pending |
| Controls | Camera pan/zoom, pause/speed, selection, needs/jobs inspector | HUD layout inspected; Step and selection submit events verified; full mouse/keyboard coverage pending |
| Survival | Food/water/rest loop replenishes from finite ecological resources; no free stores | Original Eos simulation runs; long-duration survival acceptance pending |
| Ecology | Seeded biome/season/weather, regrowth, animal/resource interactions | Historical living-world engine runs; ecology presentation and long-run validation pending |
| Agents | Explicit policy source, action/observation contract, reproducible behavior | Recovered trained baseline runs in Unity; direct inference parity passes; online learning/inheritance pending |
| Progression | Resource use, camp construction and technology with visible consequences | Pending |
| Continuity | Save/load equivalence; migration preserves colonists, supplies and abandoned structures | Random full-state and trained snapshot continuation pass; migration pending |
| Presentation | Usable colony HUD, readable art, alerts and feedback | Pending |
| Delivery | Runnable desktop build; advancing gameplay verified; errors checked | Pending |

## Sequence

1. Compile the inherited code using the installed Unity Editor and run its real tests.
2. Make its world and inhabitants visible and interactive; create a reproducible
   startup scene and build entry point. Verify advancement, not a static screenshot.
3. Replace the exhaustible demonstration with a coherent ecology/survival loop.
4. Integrate local Eos stepping, masked policy inference, checkpoints and read-only
   snapshots into Unity. Validate snapshots against direct Eos runs for identical
   seeds/actions. Preserve policy memory and RNG across save/load; do not replace
   learned decisions with Unity jobs. Document every deliberate model difference.
5. Add progression, persistence, migration and remaining colony presentation.
6. Exercise acceptance scenarios and ship a verified build.

Keep test logs and build output ignored. Record concrete failures and next work
here as they are found, without declaring an untested milestone complete.

## Verified foundation — 2026-11-19

- Unity CLI updated to 1.0.1 with user approval.
- Isolated development project upgraded by Editor to 6000.6.5f1; package lock and
  generated asset metadata retained for reproducibility.
- First compile exposed an invalid NUnit AreNotEqual tolerance overload in
  WorldGenTests. Fixed using a constraint assertion in the development copy.
- Actual EditMode result: 63 passed, zero failed or skipped. Local report:
  `unity-game/Logs/editmode.xml` (ignored build evidence).
- Local isolated Python 3.13 environment has the original pinned dependencies;
  JAX 0.11.2, Flax 0.12.10, Optax 0.2.8 and lifesim import successfully on CPU.
- Eos currently has 84 actions. No trained weights are tracked in the repository;
  requested the location of the user's latest trained run.
- Next: implement and test `EOS-UNITY-CONTRACT.md`, then Unity visualization.

## Live bridge milestone — 2026-11-19

- `bridge/eos_server.py` provides local NDJSON reset/step/snapshot and explicit
  random/manual/trained provenance. Current trained path requires an exact
  configuration manifest and fingerprint match before unpickling any population.
- Four Python tests pass, including direct Eos full-state equality after stepping,
  malformed request handling, rejection of absent/mismatched trained runs, and
  rejection of invalid actions without changing state or PRNG.
- Real Unity PlayMode test passes in 20.75 seconds: launches the Python process,
  obtains a 24x24 world and four agents, creates terrain/agent renderers, verifies
  pause and exactly one tick advance. Headless evidence does not verify appearance.
- Editor generated `Assets/Eos.unity`, configured as the startup scene. It is a
  paused diagnostic scaffold; use `bridge/README.md`. No finished game UI yet.
- Drive run provenance was inspected read-only; matching historical source is
  still being recovered. Private audit stays in ignored local cache. No trained
  weights have been loaded, and trained-policy/game acceptance remains pending.
- Next: recover version-compatible runs, expose Unity controls/inspector, extend
  snapshots and implement full-state persistence with continuation equivalence.

## Persistence and HUD milestone — 2026-11-19

- Six Python bridge tests pass: full state/PRNG continuation equals an uninterrupted
  random-mode run; wrong-mode and corrupt loads preserve current state; array
  storage preserves recurrent carry and bfloat16 parameter arrays.
- Real Unity PlayMode test passes in 25.17 seconds: UI Toolkit document loads,
  client saves, advances and restores an identical snapshot. UI input routing and
  rendered appearance still need interactive verification.
- Startup scene contains run/pause, step, speed, save/load, pan/zoom and an agent
  inspector. Diagnostic random policy is clearly identified.
- `workspace/find-eos-versions.py` searches every locally fetched branch from raw
  Git blobs. 127 commits checked; two older reported fingerprints matched, the
  two newest did not. Historical run recovery remains open.
- Next: visually verify the scene, recover a compatible trained run, expand
  snapshots/rendering and produce a portable runtime/build. Save/load covers
  inference, not PPO optimizer state or live training.

## Historical trained-policy milestone — 2026-11-19

- Recovered source fingerprint `f4498b87343e` from commit
  `b406a249d9fab097503ee355e16ce85e98d1c106`. Original imported research stays
  unchanged; the historical checkout and private run remain in ignored cache.
- Baseline history reaches 390,528 requested ticks. Its exported weights and
  population load using that source and its exact 445-field environment config.
- Added explicit engine-root selection before Python imports, configuration export
  from the matching training code, and an opt-in trained parity verifier.
- Independent direct inference agrees with bridge fields/agent states for three
  ticks. Trained save/load continuation agrees after two further ticks. Weight
  SHA256: `ae6bd0b749d9f7312ef59563e2b41259dd20c590b76f2a4f33978027c0a0f038`.
- Live Unity loaded that learned policy and advanced from tick 0 to 1 through
  a HUD submit event; colonist selection showed needs; console errors were zero.
- Visual inspection found and fixed the wrong stylesheet resource and collapsed
  document height. Added layout assertions. The larger population also exposed
  scroll clipping and unused-slot display, corrected using Eos's existed flag.
- This remains fixed-policy inference. Online learning, newborn policy inheritance,
  rich ecology rendering, progression, migration and standalone delivery are open.
- Final regression: six Python tests pass (43.03 seconds); real Unity PlayMode
  integration passes (32.16 seconds), including a 64-entry scrolling roster and
  Save/Step/Load UI submit events. Current compilation and console error count are
  clear. All 208 original source files still match their pinned commits.
