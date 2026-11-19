# Full migration and reconciliation plan

Status: implementation plan, 2026-11-19. This document records decisions, their
rationale, dependencies and evidence required to proceed. Milestones are gates,
not promises of elapsed time. Machine-readable work items are in
[`workspace/migration.json`](../../workspace/migration.json).

## Product contract

Deliver one local Unity colony game with Agon's existing visual identity,
autonomous learned inhabitants, living ecology, persistent lives and camps,
observable research mechanics, and Overseer/Director modes. Preserve the originals.
The player must be able to tell what an agent actually did, what its model predicts,
and what a player intervention caused. Do not present authored thoughts as neural
explanations or scripted colony jobs as learned behavior.

The user confirmed the two mode names. Their detailed powers below are proposed
working definitions, not a claim that the user specified every interaction:

| Mode | Proposed player experience | Boundary |
| --- | --- | --- |
| Overseer | Follow the colony, inspect needs, relationships, ideas, ecology and history; select, pause, change viewing speed, save, replay | Agents choose actions. Planning designations can be added only after the policy/environment supports perceiving them; no hidden forced-job fallback |
| Director | Create scenarios and intervene in weather, resources, threats, population, construction or movement through an explicit command palette | Every intervention is validated by the engine, logged and replayable; edited worlds are labelled as intervention runs |

Both modes share one world and one simulation. Entering Director alone does not
change state; the first accepted intervention creates a branch point and preserves
the prior save. Returning to Overseer never erases the intervention history.
Direct pawn orders, if included, are a Director capability, never silently enabled
for autonomous research. Exact powers and planning affordances are resolved in M1.

## What exists and what does not

- Source merge preserves 208 original files and both source histories.
- Unity 6000.6.5f1 compiles; 63 inherited EditMode tests previously passed.
- Six bridge tests and a real Unity PlayMode integration test pass.
- Historical baseline `f4498b87343e` loads with matching source and 445 configuration
  fields. Three direct inference ticks and trained save continuation were verified.
- Current UI is a diagnostic foundation; it is not the browser game's final HUD.
- First asset extraction yields 23 original glyphs and 14 equipment SVGs. Thirteen
  HTML components are inventoried. Pawns, animals, terrain and animations are not
  yet converted or imported into Unity.
- Online learning, newborn brain inheritance, player modes, construction,
  migration, rich rendering and standalone packaging are not complete.
- The newest reported runs still have unresolved source provenance. The recovered
  historical run does not verify the latest research source or fix its old bugs.

## M0 — Establish a repeatable baseline

Keep source commits pinned. Make `START_HERE.md` the entry point, keep generated
assets reproducible, verify Unity metadata and package pins, and run lightweight
repository checks on the integration branch. Preserve the baseline trained run
outside Git with hashes and an engine manifest. Keep measured evidence distinct
from proposed targets.

Exit: clean checkout passes repository/preservation checks; committed asset exports
regenerate byte-for-byte; baseline evidence and missing dependencies are visible.
Rollback: revert integration-only changes; the two original branches are untouched.

## M1 — Freeze the contracts and mode semantics

Write the complete feature mapping before porting individual systems. Confirm
Overseer/Director powers, pause behavior during commands, and whether players may
create nonbinding plans. Define protocol v2: capabilities, command IDs, immutable
world/run IDs, per-life IDs, tick units, field encodings, policy identity, errors,
events and interventions. Define rules for invalid, duplicate and stale commands.

Choose a canonical development engine in a new writable runtime area; retain
historical engines as compatibility packages. A source or action-space change
must receive a new fingerprint and compatibility report. Do not load old weights
merely because tensor dimensions happen to match. Separate architecture identity,
environment identity, weight identity and training history.

Exit: contract fixtures work in Python and C#; every matrix row has an owner,
decision, test and milestone; mode capabilities are explicit. No gameplay migration
may invent an implicit conversion between incompatible state models.

## M2 — Recover and curate Agon's artwork

Inventory HTML templates, imported components, palettes, functions, SVG paths,
canvas tiles, animation names, material variations, body layers and art dependencies.
Export static symbols first; capture deterministic representative pawn/animal
poses next; recover terrain/wall/door variants and seasonal states last.

Retain editable vector masters and generator recipes. Extract rendering only:
`art.js` also contains genetics, spoilage, technology and decision logic, which must
not become a second simulation in Unity. Build the visual catalog before replacing
placeholders. Compare original browser render and Unity output side by side at
the same scale, palette, pose, seed and animation phase.

Exit: all required visual categories have semantic IDs and source hashes; source
renders and Unity imports agree; no missing GUIDs or unsupported SVG features;
all 16 wall connections, corners, door orientations and open/closed states checked.
No external art pack replaces the selected Agon style by default.

## M3 — Build the playable presentation slice

Split process transport, immutable snapshot models, rendering, HUD, camera and
selection out of the monolithic client without changing Eos stepping. Import
approved-by-validation assets with stable metadata, build atlases and implement
layered pawns, animals, materials, fire, weather and map overlays.

Restore the chosen colony layout: resources top-left, colonist roster across the
top, inspector bottom-left, minimap/clock/speed bottom-right, alerts on the right,
tools/log panels sharing the inspector slot. Add readable typography, key rebinding,
focus-safe shortcuts, zoom/pan, selection highlights, audio/mute and load/error UI.
First make a one-map session usable; do not implement a second economy in the HUD.

Exit: a fresh start, inspect, run, pause, step, save, load and exit loop works using
real input at 1280x720, 1440x900 and 1920x1080. Terrain, agents, fauna and resources
reflect the same authoritative snapshots. One visible clock owns simulation time.

## M4 — Preserve learned and research behavior

Extract continuous-world learning orchestration into the writable runtime, retaining
original equations and tests. Bring across sleep consolidation, memory reservoir,
imitation, world-model updates, optimizer state, update scheduling and lineage brain
inheritance. Reset newborn recurrent state and assign a new life ID on slot reuse.
Keep inference, ongoing learning and deterministic replay as explicitly separate
execution modes. Surface ideas as predictions with supporting values, not facts.

Save the complete runner, pending learning buffers, innate/current policy trees,
optimizer, memory, inheritance RNG, world RNG and scheduler. Version every schema.
Budget learning work separately from rendering; measure pauses and throughput before
choosing batching. A packaged CPU baseline must remain usable if no GPU is present.

Exit: direct continuous-world parity for births, deaths, sleep and optimizer updates;
save/load across a learning update and birth matches uninterrupted execution on the
same pinned platform; controls and simulation speed do not change learned decisions.
New training is required where semantics changed; historical playback stays available.

## M5 — Implement Overseer and Director

Build a capability-based command router on the Python side. Read-only views work in
both modes. Director interventions execute at tick boundaries with before/after
events, explicit resource consequences and idempotent command IDs. Validate an
entire command before mutation; show a preview for large changes and retain a
branch save. Track mode and intervention history in saves and research exports.

Exit: disallowed actions are rejected by the engine even if a UI control is bypassed;
failed/duplicate commands preserve state; replay reproduces interventions; switching
modes preserves people, world, policy memory and prior research provenance.

## M6 — Reconcile survival, construction and progression

Expose existing Eos mechanisms first: flora, fauna, seasons, water, fire, tools,
disease, body changes, waste, communication and knowledge. Keep units from EnvConfig.
Replace browser staged construction with actual resource/work/state transitions
in the authoritative engine. Add foundations, walls, doors, roofs, beds and storage
only with explicit material costs, accessibility, thermal effects and observation
semantics. Extend masks/policies and retrain if new actions are needed.

Use Eos event-derived technology discovery/establishment/loss for autonomous worlds.
The colony glossary becomes a presentation of that evidence. Director may author
structures and unlock scenarios, but those events remain attributable to the player.
Do not advance houses by elapsed time or invent free stores to hide policy failure.

Exit: gather-to-store-to-consume conserves materials; construction spends real stock;
walls/doors affect navigation and shelter; technology evidence references real events;
baseline and new-policy evaluations separate survival improvement from intervention.

## M7 — Persistent regions, migration and ruins

Introduce a world registry before moving colonies. A life ID survives transfers;
slot numbers may change. Carry relationships, possessions, physiology, current and
innate policies, recurrent state, learned memories and clocks. Persist abandoned
structures, inventories and ecological state by region. Choose and document dormant
region advancement: exact replay/steps or a labelled approximation, never a hidden
world reset. Treat journey time and consumption as authoritative transitions.

Commit migration transactionally: departure inventory, journey cost, destination
allocation and both region saves succeed together, or the origin remains intact.
Save schema migration writes a new file and retains the old save.

Exit: leave/return preserves identities and camp history; no duplicate inventories;
invalid destinations and crashes cannot lose the colony; loading either side of a
journey resumes consistently. Ruin decay has a stated rule and deterministic evidence.

## M8 — Make the research legible

Add body/needs, relationships, lineage, messages, technology, ecology, event causality,
decision distributions and learned-model views using the existing research exporters.
Separate observed state, inferred interpretation and model prediction. Include run
manifest, interventions, seed, config, source and weights hashes in exports. Provide
baseline/ablation comparisons with sample sizes and uncertainty; no quality claim
from a three-tick smoke test or a visually convincing animation.

Exit: every inspector field has an authoritative source and units; timeline/rewind
and exported events agree; mode labels and provenance survive export and reload.

## M9 — Package and validate a desktop release

Build Unity with a local, versioned Python runtime and locked dependencies. Resolve
paths relative to the installation, use user-writable persistent save/log folders,
and run with no developer checkout, uv, Git, Colab or Editor installed. Define how
compatible policy packs are installed without committing private weights. Add
startup diagnostics, cancellation, shutdown, log bounds and process-crash recovery.

Exit: clean-machine offline install/start/learn-or-infer/save/restart/load works;
overnight soak and performance budgets pass; source/license manifests and build
checksums accompany the artifact. Review distribution provenance of inherited
wall-related code before shipping. No release is gated solely on a screenshot.

## Execution discipline

One reviewable feature slice per commit/PR; every slice names its matrix rows,
schema effects, acceptance evidence and rollback. Refactors and behavior changes
are separate commits. Move Unity assets through the Editor with their `.meta`
files. Never rearrange preserved originals for cosmetic repository tidiness.

Critical path: M0 → M1 → M3/M4 → M5/M6 → M7/M8 → M9. M2 artwork recovery can
progress alongside contract work. Independent workstreams here describe task
dependencies, not an instruction to spawn agents. Do not estimate completion
dates until M3/M4 throughput and training requirements have been measured.
