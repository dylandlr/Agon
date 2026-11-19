# Agon asset library and migration pipeline

Visual authority: the existing Agon colony game's RimWorld-style presentation,
as requested by the user. This means Agon's actual silhouettes, palettes, walls,
icons, portraits and HUD language; it is not a request to obtain RimWorld assets.

## Inventory and first deliverable

[`asset-library/catalog.json`](../../asset-library/catalog.json) is generated from
the preserved source. It records component props, SVG viewBoxes, source hashes,
art families and each exported asset's provenance. The
[local gallery](../../asset-library/index.html) shows the first batch.

| Source | Content | Conversion strategy |
| --- | --- | --- |
| `src/art.js` GLYPH / glyphPaths | 23 status/message symbols | Exported standalone SVG; preserve 24x24 viewBox and exact path/color data |
| `src/Gear.dc.html`, art.js TPL/itemShapes | 14 equipment templates across weapon/armor/clothing | Exported 48x48 SVG with explicit representative material choices; additional material variants generated on demand |
| `src/Pawn.dc.html` | Layered 48x48 pawn, clothes, hair, limbs, gear, state animation | Evaluate component with fixed props; export per-layer masters and pose fixtures; preserve common ground pivot |
| `src/Portrait.dc.html` | 160x200 portrait with nested pawn/body rendering | Use the same appearance recipe as map pawns; capture portrait composition separately |
| `src/Animal.dc.html` | 64x64 animals, body plans and palette variations | Species-to-Eos mapping; export rig/layer families, state poses and coat palette metadata |
| `src/Main.dc.html` | Terrain, flora, walls, doors, seasons, water, UI/canvas layers | Isolate renderer functions and capture seeded fixtures; do not screenshot entire map as one game texture |
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
    palette.json               # exact original palette data
  source/                      # future editable masters, scoped LFS for large binaries
  references/                  # future deterministic browser render fixtures
tools/assets/
  export-agon.cjs               # implemented offline extractor/checker
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
npm run assets:export
npm run assets:check
npm run check:repository
```

The exporter uses only local project sources and Node built-ins. It regenerates
owned output paths without editing originals. `--check` recomputes expected content
and fails on stale/missing files. The current gear batch selects the first allowed
material per part; that selection is recorded, not guessed at runtime.

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
