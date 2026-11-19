# Verification gates and release evidence

Every milestone records the commit, source/policy/config hashes, platform, commands,
artifact paths, observed result and unresolved limitations. Keep private histories
and weights out of public CI logs/artifacts. A gate is not passed by writing its test.
An individual work item marked verified does not imply its entire gate is complete.

Planning-foundation checks on 2026-11-19: 30 dependency-linked work items validate;
46 Unity asset files have metadata with 63 unique file/directory GUIDs; 37 SVGs
regenerate identically and parse as standalone XML; 83 colony reference checks
pass; 208 original source blobs remain unchanged. The gallery is available locally,
but the connected browser could not reach the local preview server; browser-render
and Unity visual-fidelity checks remain open. New CI jobs are configured, not yet
reported as passing remote runs.

| Gate | Required evidence |
| --- | --- |
| G00 Repository | 208 original blobs unchanged; dependency graph acyclic; source paths resolve; Unity metadata present/unique; package pins valid; generated art reproduces |
| G01 Compatibility | Missing/mismatched engine/config/architecture/population rejected before loader; correct trusted run loads; no random fallback; historical/current source distinguished |
| G02 Art | Browser-versus-export-versus-Unity contact sheets; representative body/coat/equipment variants; wall connection/door test board; no lost strokes, alpha, pivots or missing atlas sprites |
| G03 Playable slice | Real pointer/keyboard inputs at 720p/900p/1080p; pause stable; speed advances real ticks; camera/minimap/selection agree; focus prevents unwanted shortcuts; mute persists |
| G04 Learned behavior | Direct runtime equivalence for masks, actions, messages, recurrent state and world state; births reset memory correctly; inherited policies/optimizer/sleep learning match reference; held-out evaluation reports seeds and uncertainty |
| G05 Modes | Capability checks server-side; unauthorized/stale/duplicate commands rejected safely; Director intervention fork recorded/replayed; returning to Overseer does not erase provenance |
| G06 Persistence | Same-platform exact continuation across save/load, sleep update and birth; interrupted writes/corrupt files preserve last good save; version migration retains originals; readable incompatibility errors |
| G07 Survival/building | Resource conservation; no free stores; material/work costs; navigation and shelter effects; tech derives from real events; new action schemas trigger compatibility/retraining gates |
| G08 Migration | Stable people/lineage/relationships/policies and inventory through journey; transactional origin/destination saves; return finds camp/ruins; failed migration cannot duplicate or lose colony |
| G09 Research | Event/decision/technology/body views match source data; predictions labelled; seeded replay agrees; intervention provenance exported; ablation protocols preserved |
| G10 Delivery | Clean Windows machine without Editor/Python/Git; offline start and policy load; advancing game; save/restart/load; crash recovery; no absolute developer paths; complete licenses/checksums |

## Test layers

1. Fast repository checks and original JS reference checks on each integration PR.
2. Python protocol, invalid-input, conservation and persistence tests on affected PRs.
3. Unity EditMode for pure models, content catalogs, imports, units and IDs.
4. Unity PlayMode against the real Python process for controls, actual advancement,
   rendering synchronization and state restoration. A mocked transport cannot replace it.
5. Private policy compatibility/direct-inference tests when engine/weights/schema changes.
6. Packaged Windows end-to-end tests before each release candidate.
7. Multi-seed survival/research evaluation and long-duration soak before acceptance.

For exact state equality, use the same source, dependency lock, hardware backend and
floating-point configuration. Cross-platform GPU/CPU comparisons need documented
tolerances and behavioral metrics; do not claim bitwise identity from approximate
floating-point agreement. Existing three-tick inference parity is a smoke test.

## Performance measurements

Record hardware and distinguish cold compile, warm inference, learning update,
serialization, rendering and save/load times. Proposed initial targets, to validate
or revise after M3/M4 profiling: 60 FPS presentation at 1080p, ordinary UI response
under 100 ms while Python is busy, and memory usage that stabilizes in a one-hour
run. Never hide cold-start/learning waits behind a frozen UI. No simulation tick-rate
promise is made until agent/grid-size benchmarks are recorded.

Stress profiles: 8 living founders in 64 slots at 32x32; crowded 64-agent state;
larger 64x64 world; multi-region migration; many events; prolonged learning.
Compare memory before/after region unloading, repeated save/load and policy switches.
Run a one-hour soak for each candidate and an overnight soak for a release candidate.
These durations are proposed release checks, not claims of completed runs.

## Build and release layout

```text
Agon/
  Agon.exe + Unity player data
  Runtime/<version>/            # Python, pinned dependencies, canonical/historical engine
  Content/<catalog-version>/    # local assets and content catalogs
  PolicyPacks/                  # explicitly installed compatible policies
  build-manifest.json           # source/package/content hashes and supported schemas
```

User saves, logs and settings belong in persistent user storage, not the install
directory. Include a bounded diagnostics log, explicit cancellation and child-process
cleanup. The build must not depend on the developer's `node_modules/.cache` layout.

CI currently lacks a configured licensed Unity runner. The repository adds automated
source/asset/bridge checks; it must not imply a Unity cloud build has run. Configure
a Windows runner and supported Unity licensing path before marking automated Unity
or packaged release gates complete. Keep credential setup separate from source.

Release checklist: gate evidence attached; no temporary test scene or private run
tracked; clean content build; reviewed asset/code provenance (including inherited
wall references); all required notices; artifacts hashed; original branches unchanged;
previous installer and saves retained for rollback.
