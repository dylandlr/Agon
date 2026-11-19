// Deterministic, offline extraction from the preserved Agon artwork.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const { isolateDeclarations, renderWorld } = require('./source-art.cjs');
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
const mainScript = /<script[^>]*type="text\/x-dc"[^>]*>([\s\S]*?)<\/script>/.exec(read('src/Main.dc.html'))[1];
const presentationCss = [...read('src/Main.dc.html').matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)].map(m => m[1]).join('\n');
outputs.set('asset-library/generated/agon/world/presentation.css', presentationCss + '\n');
const worldClosure = isolateDeclarations(mainScript, ['SPRITE_CSS']);
const world = renderWorld(worldClosure.code, source);
const worldSources = [{ label: 'colony-source', source: 'src/Main.dc.html',
  scriptSha256: sha(mainScript), closureSha256: sha(worldClosure.code), symbols: worldClosure.symbols,
  spriteCount: world.assets.length, presentationCss: 'asset-library/generated/agon/world/presentation.css',
  keyframes: [...presentationCss.matchAll(/@keyframes\s+([^\s{]+)/g)].map(m => m[1]) }];
function worldGroup(key) {
  if (/^(q\d\.)?t-/.test(key)) return 'trees';
  if (/^rk/.test(key)) return 'rocks';
  if (/^(p-|bush|fg-)/.test(key)) return 'plants';
  if (/^(fs-|ms-|mp-|fish$)/.test(key)) return 'aquatic';
  if (/^(bird|bfly)/.test(key)) return 'wildlife';
  if (/^fire/.test(key)) return 'effects';
  if (/^(wl|wsh|wcf|tent)/.test(key)) return 'structures';
  throw new Error(`Unclassified world artwork: ${key}`);
}
function addWorld(asset, provenance, version = '') {
  const group = worldGroup(asset.key), id = `world.${version}${asset.key}`;
  const file = `asset-library/generated/agon/world/${group}/${id}.svg`;
  outputs.set(file, asset.svg);
  const entry = { id: `agon.${id}`, group, file, viewBox: asset.viewBox, sha256: sha(asset.svg),
    provenance, presentation: { selector: asset.selector, declarations: asset.css,
      composition: asset.selector.includes(':') ? 'CSS pseudo-element layer; assemble with its base sprite' : 'Source CSS sprite',
      animationConverted: false }, status: 'extracted-svg', unityImported: false };
  entries.push(entry);
  return entry;
}
const currentWorld = new Map(world.assets.map(a => [a.selector,
  addWorld(a, { source: 'src/Main.dc.html', symbol: 'ENGINE.SPRITE_CSS',
    closureSha256: sha(worldClosure.code), alsoPresentIn: [] })]));
// Retain source CSS layout and keyframes, but point image URLs to standalone files.
let cssIndex = 0;
const sourceCss = world.css.replace(/url\('data:image\/svg\+xml;utf8,[^']*'\)/g, () => {
  const entry = currentWorld.get(world.assets[cssIndex++].selector);
  return `url('${entry.file.replace('asset-library/generated/agon/world/', '')}')`;
});
if (cssIndex !== world.assets.length) throw new Error('World CSS/image correspondence changed');
outputs.set('asset-library/generated/agon/world/source.css', sourceCss + '\n');
const recoveredRoot = path.join(root, 'asset-library/source/recovered');
for (const label of fs.existsSync(recoveredRoot) ? fs.readdirSync(recoveredRoot).sort() : []) {
  const base = `asset-library/source/recovered/${label}`;
  const manifest = JSON.parse(read(`${base}/manifest.json`)), closure = read(`${base}/world.js`);
  if (sha(closure) !== manifest.closure.sha256 || sha(source) !== manifest.sharedArt.sha256)
    throw new Error(`Recovered source hash changed: ${label}`);
  const recovered = renderWorld(closure, source);
  const recorded = manifest.sprites;
  if (JSON.stringify(recovered.assets.map(a => ({ selector: a.selector, key: a.key, sha256: sha(a.svg) }))) !== JSON.stringify(recorded))
    throw new Error(`Recovered sprite output changed: ${label}`);
  let identical = 0, variants = 0;
  for (const asset of recovered.assets) {
    const current = currentWorld.get(asset.selector);
    const provenance = { source: `${base}/world.js`, manifest: `${base}/manifest.json`, artifactSha256: manifest.artifact.sha256 };
    if (current?.sha256 === sha(asset.svg)) { current.provenance.alsoPresentIn.push(provenance); identical++; }
    else { addWorld(asset, provenance, `${label}.`); variants++; }
  }
  worldSources.push({ label, manifest: `${base}/manifest.json`, spriteCount: recovered.assets.length,
    identicalToCurrent: identical, additionalVariants: variants });
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
  schemaVersion: 2,
  sourceCommit: JSON.parse(read('workspace/sources.json')).colony,
  source: { path: 'src/art.js', sha256: sha(source), hashNormalization: 'UTF-8, LF' },
  style: 'Existing Agon colony HTML/SVG art; retain its visual identity',
  license: 'Project source provenance retained; no new third-party license claimed',
  inventories: { expressions: keys('EXPR'), motions: keys('MOTION'), animalFamilies: keys('ANIMALS'),
    animalPaletteVariants: keys('ANIMAL_VARS'), bodyBuilds: keys('BUILDS'), hairStyles: [...art.HAIRNAMES],
    outfits: [...art.OUTFITNAMES], materials: keys('MATS') },
  worldSources, components, assets: entries
};
outputs.set('asset-library/catalog.json', JSON.stringify(catalog, null, 2) + '\n');
outputs.set('asset-library/generated/agon/palette.json', JSON.stringify(Object.fromEntries(
  ['INK','SKIN','HAIR','CLOTH','SOIL','GRASS','FLOOR','WALL','FOREST','ROCK','WATER','WATER_EDGE','TRIM','NEEDCOL'].map(k => [k, art[k]])), null, 2) + '\n');
const groups = [...new Set(entries.map(e => e.group))].sort();
const cards = entries.map(e => `<figure data-group="${e.group}"><img loading="lazy" src="${e.file.replace('asset-library/', '')}" alt="${xml(e.id)}"><figcaption>${xml(e.id)}</figcaption><small>${xml(e.presentation?.selector || e.provenance.symbol)}</small></figure>`).join('\n');
outputs.set('asset-library/index.html', `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Agon source asset library</title>
<style>body{background:#211b16;color:#eee4ce;font:16px system-ui;margin:32px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}figure{margin:0;background:#403a2e;border:1px solid #655a45;border-radius:6px;padding:16px}figure[hidden]{display:none}img{width:112px;height:112px;object-fit:contain;display:block;margin:12px auto}figcaption,small{font-size:12px;overflow-wrap:anywhere}small{color:#c2b79f}p{max-width:900px;line-height:1.6}input,select{font:inherit;padding:10px;margin:0 12px 12px 0;background:#403a2e;color:#eee4ce;border:1px solid #97876a}nav{position:sticky;top:0;background:#211b16;padding:12px 0;z-index:1}</style>
<h1>Agon source asset library</h1><p>${entries.length} source-derived SVG exports, including ${world.assets.length} world sprites. The recovered r18 HTML supplies independent provenance for matching sprites. Equipment uses the first allowed material per part. Effects are separate static layers; original CSS keyframes are retained for conversion. Layered pawns, portraits, larger animals and canvas terrain still need conversion. These files are not yet imported into Unity.</p>
<nav aria-label="Filter assets"><label>Search <input id="search" type="search" placeholder="oak, wall, fish, fire…"></label><label>Category <select id="group"><option value="">All categories</option>${groups.map(g => `<option value="${g}">${g} (${entries.filter(e => e.group === g).length})</option>`).join('')}</select></label><span id="count" role="status">${entries.length} assets</span></nav><main>${cards}</main>
<script>const search=document.querySelector('#search'),group=document.querySelector('#group'),cards=[...document.querySelectorAll('figure')];function filter(){let count=0;for(const card of cards){card.hidden=!!(group.value&&card.dataset.group!==group.value)||!card.textContent.toLowerCase().includes(search.value.toLowerCase());if(!card.hidden)count++;}document.querySelector('#count').textContent=count+' assets';}search.addEventListener('input',filter);group.addEventListener('change',filter);</script></html>\n`);
let failures = 0;
for (const [file, text] of outputs) {
  const target = path.join(root, file);
  if (check) {
    if (!fs.existsSync(target) || read(file) !== text) { console.error(`Stale/missing generated asset: ${file}`); failures++; }
  } else { fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, text); }
}
if (failures) process.exitCode = 1;
else console.log(`${check ? 'Verified' : 'Exported'} ${entries.length} SVG assets and inventoried ${components.length} components; original sources unchanged.`);
