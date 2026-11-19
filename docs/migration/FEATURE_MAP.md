# Feature reconciliation matrix

Source paths are evidence of implementation intent, not claims that every upstream
behavior is correct. Each row must receive its named acceptance evidence before
being marked complete. C = colony reference; E = Eos reference. New authoritative
behavior belongs in the planned writable runtime, not preserved source directories.

| ID | Feature / source evidence | Reconciliation decision | Target owner / gate |
| --- | --- | --- | --- |
| F01 | Agent decisions: C `unity/Assets/Agon/Core/AI/`; E `lifesim/policy.py`, `ppo_multi.py` | Eos policies choose autonomous actions; retain scripted jobs only as reference/test baselines | Runtime; G04 |
| F02 | Calendar: C `src/Main.dc.html` calOf; E `lifesim/config.py` | Use EnvConfig tick_minutes/day/year/bio-year; explicit UI units; do not port colony lifespan constants | Contracts/UI; G03 |
| F03 | Worldgen: C `AgonMacro` in `src/art.js`; E `lifesim/env.py` reset | Eos owns playable terrain; adapt colony rendering to it. Macro regions require a documented map interface | Runtime/world views; G03/G08 |
| F04 | Flora, fauna, weather: C Main species art; E `ecology.py`, `fauna.py`, `trees.py`, env | Preserve Eos rates/state; map to semantic art IDs, including missing species placeholders | Runtime/content; G03/G07 |
| F05 | Food, water, rest: C needs/jobs and stores; E env physiology | Eos owns consumption and resource limits; UI does not refill colony stores | Runtime; G07 |
| F06 | Body, illness, injury: C Pawn/Xray/Character; E conditions/env/body exporter | Render Eos state with explicit unit mappings; unsupported visual traits are cosmetic, not invented biological genes | UI/content; G09 |
| F07 | Personality, relationships and kinship: C art.js/Thoughts; E genotype/social/analysis | Eos genotype and learned relationships authoritative; retain colony presentation vocabulary only with source definitions | Runtime/research UI; G04/G09 |
| F08 | Appearance: C art.js look/phenotype/Pawn/Portrait; E appearance vector | Deterministically derive a cosmetic profile from Eos appearance and life ID; persist mapping version | Content/contracts; G02/G04 |
| F09 | Sleep, memory and imitation: E ppo_multi/continuous_world | Extract ongoing-learning orchestration with identical buffers/scheduling; never substitute authored thought text | Runtime; G04 |
| F10 | World model / ideas: E policy and export_playback | Preserve learned predictions and action distributions; label confidence/interpretation | Runtime/research UI; G04/G09 |
| F11 | Reproduction/lineage: E env and continuous_world inherit_brain/newborn_memory | Stable life identities, explicit parent/innate inheritance modes, reset reused-slot carry, save complete lineage | Runtime; G04/G08 |
| F12 | Speech/writing: C glyphs/message UI; E policy outputs, names, language metrics | Display actual tokens/events; do not assign semantic meaning unless measured or using explicitly grounded language mode | Contracts/UI; G09 |
| F13 | Fire, tools, materials: C Gear/art.js material templates; E minerals/env | Reuse item visuals; retain Eos physical recipes, quantities and masking | Runtime/content; G02/G07 |
| F14 | Technology: C staged TECH/CAMPON; E techtree event matchers | Autonomous tech is discovered/established/lost from events; browser unlock timing is not transferred | Runtime/research UI; G07/G09 |
| F15 | Walls, rooms, roofs: C linked walls/staged camp; E shelter currently forest/caves | Implement actual construction and thermal/navigation semantics as new engine work; policies may need new observations/actions and retraining | Runtime/content; G02/G07 |
| F16 | Beds/storage/farming: C camp objects/plots; E physical resources | Add authoritative objects and conservation rules where absent; no duplicated scripted larder | Runtime; G07 |
| F17 | Overseer mode: user request | Inspect autonomous colony; define nonbinding planning only if policies can perceive it | Application/runtime; G05 |
| F18 | Director mode: user request | Explicit scenario/intervention powers; receipts, branch save, provenance and deterministic replay | Application/runtime; G05 |
| F19 | HUD: C docs/TODO round25 and Main | Restore chosen layout A: inspector bottom-left; minimap/clock/speed bottom-right; alerts right; roster top | UI; G03 |
| F20 | Camera, selection, input, audio: C Main/mute/migration tests | Port interaction behavior, map coordinate orientation explicitly, test real pointer/keyboard/focus, preserve mute settings | UI; G03 |
| F21 | Saving/rewind: C browser state; E recorder and current bridge persistence | Versioned unified saves; actions/events and complete learning state; no silent browser-save conversion | Runtime/storage; G06 |
| F22 | Migration: C migrate/applyCarry and tests/migrate.py | Transactional regional transfer of stable people, possessions, learned state and relationships | Runtime/world registry; G08 |
| F23 | Ruins: C round22 and tests/ruins4.py | Persist abandoned region state; deterministic decay; explicit dormant-region approximation if chosen | Runtime; G08 |
| F24 | Analysis: E scripts/society.py, hierarchy.py, culture.py | Preserve measurements and ablation controls; annotate intervention contamination instead of claiming spontaneous emergence | Research UI/export; G09 |
| F25 | Policy versions: current/historical incompatibility | Isolated matching source packages; schema/weights/config checks; new canonical engine retrained/evaluated separately | Runtime/packaging; G01/G04 |
| F26 | Delivery: original single HTML vs Unity/Python | Offline Windows package with runtime/dependencies, writable saves, policy pack installation and restart recovery | Build/release; G10 |

Remaining product decisions are explicit: exact mode powers, whether Overseer has
planning designations, dormant-region simulation policy, eventual distribution of
trained weights, and target hardware performance. They do not block art extraction
or baseline preservation. They do block affected semantics from being declared done.
