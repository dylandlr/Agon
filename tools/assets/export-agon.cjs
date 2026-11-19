// Deterministic, offline extraction from the preserved Agon artwork.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const check = process.argv.includes('--check');
const sha = text => crypto.createHash('sha256').update(text).digest('hex');
const read = name => fs.readFileSync(path.join(root, name), 'utf8').replace(/\r\n/g, '\n');
const xml = value => String(value).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const source = read('src/art.js');
const sandbox = { window: {} };
vm.createContext(sandbox);
vm.runInContext(source, sandbox, { timeout: 5000, filename: 'src/art.js' });
const art = sandbox.window.AgonArt;
const outputs = new Map(), entries = [];
function add(id, group, viewBox, shapes, provenance) {
  if (!/^[a-z0-9.-]+$/.test(id) || !shapes.length) throw new Error(`Invalid asset ${id}`);
  const body = shapes.map(p => `<path d="${xml(p.d)}" fill="${xml(p.fill)}" stroke="${xml(p.stroke)}" stroke-width="${xml(p.sw)}" stroke-linecap="round" stroke-linejoin="round"/>`).join('\n');
  const text = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" role="img" aria-label="${xml(id)}">\n${body}\n</svg>\n`;
  const file = `asset-library/generated/agon/${group}/${id}.svg`;
  outputs.set(file, text);
  entries.push({ id: `agon.${id}`, group, file, viewBox, sha256: sha(text), provenance,
    status: 'extracted-svg', unityImported: false });
}
for (const key of Object.keys(art.GLYPH).sort()) {
  add(`glyph.${key}`, 'glyphs', '0 0 24 24', art.glyphPaths(key).map(p => ({ d: p.d, fill: p.f, stroke: p.s, sw: p.w })),
    { source: 'src/art.js', symbol: `AgonArt.GLYPH.${key}` });
}
for (const category of Object.keys(art.TPL).sort()) for (const key of Object.keys(art.TPL[category]).sort()) {
  const template = art.TPL[category][key];
  const materials = template.parts.map(part => part[2][0]);
  add(`gear.${category}.${key}`, 'gear', '0 0 48 48', art.itemShapes(category, key, materials),
    { source: 'src/art.js', component: 'src/Gear.dc.html', symbol: `AgonArt.TPL.${category}.${key}`, materials });
}
const components = fs.readdirSync(path.join(root, 'src')).filter(n => n.endsWith('.dc.html')).sort().map(name => {
  const file = `src/${name}`, text = read(file), props = /data-props='([^']*)'/.exec(text);
  return { file, sha256: sha(text), svgRoots: [...text.matchAll(/<svg\b/g)].length,
    viewBoxes: [...new Set([...text.matchAll(/viewBox="([^"]+)"/g)].map(m => m[1]))],
    properties: props ? JSON.parse(props[1]) : {},
    imports: [...text.matchAll(/<dc-import[^>]*\bname="([^"]+)"/g)].map(m => m[1]),
    status: 'inventoried-not-converted' };
});
const keys = name => Object.keys(art[name]).sort();
const catalog = {
  schemaVersion: 1,
  sourceCommit: JSON.parse(read('workspace/sources.json')).colony,
  source: { path: 'src/art.js', sha256: sha(source), hashNormalization: 'UTF-8, LF' },
  style: 'Existing Agon colony HTML/SVG art; retain its visual identity',
  license: 'Project source provenance retained; no new third-party license claimed',
  inventories: { expressions: keys('EXPR'), motions: keys('MOTION'), animalFamilies: keys('ANIMALS'),
    animalPaletteVariants: keys('ANIMAL_VARS'), bodyBuilds: keys('BUILDS'), hairStyles: [...art.HAIRNAMES],
    outfits: [...art.OUTFITNAMES], materials: keys('MATS') },
  components, assets: entries
};
outputs.set('asset-library/catalog.json', JSON.stringify(catalog, null, 2) + '\n');
outputs.set('asset-library/generated/agon/palette.json', JSON.stringify(Object.fromEntries(
  ['INK','SKIN','HAIR','CLOTH','SOIL','GRASS','FLOOR','WALL','FOREST','ROCK','WATER','WATER_EDGE','TRIM','NEEDCOL'].map(k => [k, art[k]])), null, 2) + '\n');
const cards = entries.map(e => `<figure><img src="${e.file.replace('asset-library/', '')}" alt="${xml(e.id)}"><figcaption>${xml(e.id)}</figcaption></figure>`).join('\n');
outputs.set('asset-library/index.html', `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Agon source asset library</title><style>body{background:#211b16;color:#eee4ce;font:16px system-ui;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}figure{margin:0;background:#403a2e;border:1px solid #655a45;border-radius:6px;padding:16px}img{width:96px;height:96px;display:block;margin:12px auto}figcaption{font-size:12px;overflow-wrap:anywhere}p{max-width:850px;line-height:1.6}</style><h1>Agon source asset library</h1><p>37 SVG exports from the original colony art: 23 glyphs and 14 equipment templates. Each equipment preview uses the first allowed material for each part. Pawns, portraits, animals and terrain are inventoried; their layered conversion is pending. These exports are not yet imported into Unity.</p><main>${cards}</main></html>\n`);
let failures = 0;
for (const [file, text] of outputs) {
  const target = path.join(root, file);
  if (check) {
    if (!fs.existsSync(target) || read(file) !== text) { console.error(`Stale/missing generated asset: ${file}`); failures++; }
  } else { fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, text); }
}
if (failures) process.exitCode = 1;
else console.log(`${check ? 'Verified' : 'Exported'} ${entries.length} SVG assets and inventoried ${components.length} components; original sources unchanged.`);
