# Agon source asset library

Open [index.html](index.html) to browse the first 37 recovered SVG assets. The
[catalog](catalog.json) records source hashes, original symbols, selected materials
and 13 inventoried HTML components. These are Agon's existing glyphs and equipment,
not replacement artwork. They have not yet been imported into Unity.

Run `npm run assets:export` to reproduce this library, or `npm run assets:check`
to verify it without changing files. See the
[asset pipeline](../docs/migration/ASSET_PIPELINE.md) for the remaining pawn,
portrait, animal, terrain, wall, animation and Unity import stages.

Generated content is owned by the exporter. Edit the extraction recipe or add a
new editable master with provenance; do not hand-edit generated output. Preserved
`src/` files stay unchanged. Large new binary masters belong under `source/` with
the scoped Git LFS rules; do not move or rewrite historical assets.
