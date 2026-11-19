# Agon asset library and migration pipeline

Visual authority: the existing Agon colony game's RimWorld-style presentation,
as requested by the user. This means Agon's actual silhouettes, palettes, walls,
icons, portraits and HUD language; it is not a request to obtain RimWorld assets.

## Inventory and recovered world library

[`asset-library/catalog.json`](../../asset-library/catalog.json) is generated from
the preserved source. It records component props, SVG viewBoxes, source hashes,
art families and each exported asset's provenance. The
[local gallery](../../asset-library/index.html) contains 370 standalone SVGs,
with search and category filters. This is an extraction milestone, not complete
conversion of the original game art.

The supplied `Agon_p14_r18 (2) (1).html` contains serialized component definitions
and embedded original artwork. Its 330 generated world sprites match the current
colony source byte-for-byte after deterministic SVG extraction. The current source
adds three wall-corner fillers. The embedded `art.js` and nine non-Main component
scripts also match; Main differs. Template hashes are recorded separately, without
claiming all templates match. The original HTML was not modified.

| Exported category | SVG files |
| --- | ---: |
| Trees, including seasonal variants | 50 |
| Rocks | 15 |
| Plants, forage states and bushes | 94 |
| Aquatic life and plants | 50 |
| Small birds/butterflies | 3 |
| Structures: 96 walls, 16 shadows, 3 corners, 1 windbreak/tent | 116 |
| Fire bases and overlay layers | 5 |
| Original glyphs and representative equipment | 37 |
| **Total** | **370** |

Recovered r18 sprites share semantic IDs with identical current sprites; the
catalog records both origins instead of duplicating files. Changed future recoveries
receive version-qualified IDs. `source/recovered/r18/manifest.json` records the
original document SHA256, component script/template hashes, dependency closure and
each sprite hash. `world.js` preserves just the selected artwork declarations for
reproducible comparison without redistributing the entire packed application.

| Source | Content | Conversion strategy |
| --- | --- | --- |
| `src/art.js` GLYPH / glyphPaths | 23 status/message symbols | Exported standalone SVG; preserve 24x24 viewBox and exact path/color data |
| `src/Gear.dc.html`, art.js TPL/itemShapes | 14 equipment templates across weapon/armor/clothing | Exported 48x48 SVG with explicit representative material choices; additional material variants generated on demand |
| `src/Pawn.dc.html` | Layered 48x48 pawn, clothes, hair, limbs, gear, state animation | Evaluate component with fixed props; export per-layer masters and pose fixtures; preserve common ground pivot |
| `src/Portrait.dc.html` | 160x200 portrait with nested pawn/body rendering | Use the same appearance recipe as map pawns; capture portrait composition separately |
| `src/Animal.dc.html` | 64x64 animals, body plans and palette variations | Species-to-Eos mapping; export rig/layer families, state poses and coat palette metadata |
| `src/Main.dc.html` | Terrain, flora, walls, doors, seasons, water, UI/canvas layers | 333 generated world sprites extracted; canvas terrain, doors, weather composition and remaining dynamic layers still pending |
| `src/Xray.dc.html`, Character, Thoughts, Helix, Hexagraph, Encyclopedia, Workshop | Body/research/equipment interfaces | Preserve data concepts and useful glyphs; rebuild interaction in Unity rather than embedding HTML screens |

Inventory counts are categories, not counts of finished sprite files. In particular,
82 palette variants are not 82 independently converted species. Animation CSS is
not a Unity animation clip. Extracted SVGs are not yet Unity-imported assets.

## Pipeline stages

1. **Catalog.** Pin source commit; collect SHA256 with normalized LF; record symbol,
   props, template dependencies, viewBox and semantic ID. Preserve original files.
2. **Reference render.** Build the browser reference. Freeze random seed, simulation,
   animation time, pose, camera and props. Capture source renders and layered SVG.
3. **Extract.** Static paths use `tools/assets/export-agon.cjs`. Dynamic templates
   need a deterministic component harness using `pack/dc-runtime.js`, with no live
   simulation ticks. Export only visual data; do not execute colony AI in Unity.
   For world sprites, Acorn and eslint-scope resolve the exact lexical dependencies
   of `ENGINE.SPRITE_CSS`; only those declarations and `art.js` run in an isolated
   VM. The parser rejects ambiguous/missing bindings and unsupported parameters.
   This is a dependency selector for reviewed project source, not a security
   sandbox for arbitrary downloaded JavaScript. Packed JSON is parsed, not evaluated.
4. **Validate source-derived art.** Parse SVG, reject missing/external dependencies,
   check bounds, palettes, transparency, pivots and IDs. Make contact sheets for
   body types, directions, gear and material variants. Compare with source captures.
5. **Import in Unity.** Use an Editor importer with explicit settings and persistent
   `.meta` files. SVG features unsupported by the installed vector package get a
   deterministic raster bake; do not silently drop masks, filters or strokes.
6. **Assemble.** Prefabs/layer recipes reference semantic asset IDs. Use shared
   palettes and materials; separate cosmetic selection from simulation genotype.
7. **Pack.** Separate world, pawns, fauna and UI into atlas/content groups. Validate
   padding, edge bleed, compression, alpha and texture memory in an actual build.
8. **Integrate.** Replace placeholders category by category. Verify both visual
   fidelity and authoritative state mapping before deleting development placeholders.

## Layout and naming

```text
asset-library/
  catalog.json                 # generated provenance and inventory
  index.html                   # generated contact sheet
  generated/agon/
    glyphs/                    # first 23 SVG exports
    gear/                      # first 14 representative equipment exports
    world/                     # 333 SVGs grouped by category
      source.css               # original sprite rules, URLs rewritten locally
      presentation.css         # original Main style blocks, including 31 keyframes
    palette.json               # exact original palette data
  source/recovered/r18/         # artwork-only closure and document/hash manifest
  references/                  # future deterministic browser render fixtures
tools/assets/
  export-agon.cjs               # implemented offline extractor/checker
  source-art.cjs                # packed JSON parser and lexical dependency resolver
  recover-packed.cjs            # explicit recovery, refuses existing output labels
  render-review.cjs             # all-SVG raster validation and contact sheets
unity-game/Assets/Agon/Content/ # planned: imported runtime assets and catalog
```

IDs use `agon.<family>.<name>`; variants receive explicit material/pose identifiers.
Unity GUIDs identify imported files; semantic IDs identify game meaning. Neither
should depend on an enum's incidental position. Track generator version and source
hash with every batch. Never regenerate GUIDs when only image content changes.

Do not flatten all characters into a giant combination atlas: body/skin/hair/clothes/
equipment create a combinatorial explosion. Reuse layers, palette variants and
small animation sets; measure compositing cost. Idle/walk/work/sleep/hurt/dead poses,
front/back/side directions, child/adult proportions and body-state changes need
explicit fixtures. Some old aesthetic genetics may become cosmetic recipe inputs;
they must not fabricate biological traits absent from Eos.

World art coverage includes terrain transitions, cliffs, water edges, flora growth/
harvest states, animals alive/dead, resources, fire/fuel/snow, wall 16-way connections,
corner fillers, doors in both orientations, floors, roofs, windows, ruins and damage.
Match the existing art's outlined SVG appearance. Do not force nearest-neighbor
pixel filtering onto smooth vector-derived art without a side-by-side comparison.

## Reproduction and checks

```sh
npm ci
npm run assets:export
npm run assets:check
npm run assets:test
npm run assets:render
npm run check:repository
```

The exporter uses local project sources and pinned parser dependencies. It regenerates
owned output paths without editing originals. `--check` recomputes expected content
and fails on stale/missing files. The current gear batch selects the first allowed
material per part; that selection is recorded, not guessed at runtime.

Recover another packed build explicitly with
`node tools/assets/recover-packed.cjs "path/to/build.html" version-label`.
The command refuses to overwrite an existing recovery or substitute a different
embedded `art.js`. Review the closure before running it for an unfamiliar source.

`assets:render` uses pinned Sharp/librsvg to decode every SVG and generate ten PNG
contact sheets plus a report in ignored `node_modules/.cache/asset-review/`.
All 370 decode successfully; only `world.wsh0` intentionally has no visible pixels
(no connected wall, no cast shadow). Review PNGs are diagnostic and are not Unity
runtime textures. A first renderer, resvg-js 2.6.2, crashed on an original clipped
rock; it was removed, and the original SVG was kept intact. Browser/export/Unity
fidelity comparison, runtime pivots, scale and atlas packing remain required.

Fire overlays retain their CSS selectors and declarations. Main's original style
blocks retain 31 keyframe definitions. CSS is reference material, not converted
Unity animation. World metadata separates pseudo-element overlays from base sprites.

Source provenance and third-party licensing are different things. Preserve the
project origin for recovered art. Record creator, license, source URL, modification
history and redistribution terms if any external assets are introduced later.
No external asset purchase or download is needed for this initial library.

Unity stores import settings and GUIDs alongside assets in metadata; keep those
files with the imported assets ([Asset Database documentation](https://docs.unity.com/en-us/engine/6000.6/manual/assets-and-media/asset-database/contents)).
Use the current Sprite Atlas workflow when packing sprites
([Unity 6.6 atlas guide](https://docs.unity.com/en-us/engine/6000.6/manual/unity2d/sprite/atlas/create-sprite-atlas)).
The exact atlas sizes and memory budgets are chosen from measured target scenes,
not assumed from the number of source SVG files.
