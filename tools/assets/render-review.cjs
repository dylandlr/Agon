// Render every standalone SVG and make review sheets. Not a Unity fidelity test.
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const root = path.resolve(__dirname, '../..');
const catalog = JSON.parse(fs.readFileSync(path.join(root, 'asset-library/catalog.json'), 'utf8'));
const out = path.join(root, 'node_modules/.cache/asset-review');
fs.mkdirSync(out, { recursive: true });
const rows = [], pngs = new Map();
async function main() {
for (const asset of catalog.assets) {
  fs.writeFileSync(path.join(out, 'last-attempted.txt'), asset.id);
  const svg = fs.readFileSync(path.join(root, asset.file));
  if (/(?:href|src)=["'](?:https?:|\/\/)/.test(svg.toString('utf8'))) throw new Error(`External images in ${asset.id}`);
  const renderer = sharp(svg).resize({ width: 128 }).ensureAlpha();
  const { data: pixels, info } = await renderer.clone().raw().toBuffer({ resolveWithObject: true });
  let visible = 0;
  for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) visible++;
  // An unconnected wall has no cast shadow; its deliberately empty tile is valid.
  if (!visible && asset.id !== 'agon.world.wsh0') throw new Error(`Empty raster: ${asset.id}`);
  rows.push({ id: asset.id, width: info.width, height: info.height, visiblePixels: visible });
  pngs.set(asset.id, (await renderer.png().toBuffer()).toString('base64'));
}
async function sheet(name, assets) {
  const cols = 6, cellW = 184, cellH = 180, height = 60 + Math.ceil(assets.length / cols) * cellH;
  const tiles = assets.map((a, i) => {
    const x = (i % cols) * cellW + 12, y = Math.floor(i / cols) * cellH + 60;
    return `<g transform="translate(${x},${y})"><rect width="168" height="164" rx="5" fill="#403a2e"/>
      <image href="data:image/png;base64,${pngs.get(a.id)}" x="20" y="8" width="128" height="128" preserveAspectRatio="xMidYMid meet"/>
      <text x="8" y="153" font-size="11">${a.id.replace('agon.', '')}</text></g>`;
  }).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${cols * cellW}" height="${height}" viewBox="0 0 ${cols * cellW} ${height}">
    <rect width="100%" height="100%" fill="#211b16"/><g fill="#eee4ce" font-family="sans-serif"><text x="20" y="34" font-size="22">Agon source art · ${name}</text>${tiles}</g></svg>`;
  await sharp(Buffer.from(svg)).png().toFile(path.join(out, `${name}.png`));
}
const groups = [...new Set(catalog.assets.map(a => a.group))].sort();
for (const group of groups) await sheet(group, catalog.assets.filter(a => a.group === group));
const sampleKeys = ['t-oak0', 'q0.t-birch0', 'q2.t-oak0', 'q3.t-birch0', 't-pine0', 't-palm0',
  'p-fern0', 'p-shroom0', 'fg-grain2', 'bush2', 'rk20', 'p-log', 'ms-reefshark', 'ms-octopus',
  'fs-trout', 'mp-coral', 'bird', 'bfly0', 'wlw50', 'wlt101', 'wls150', 'tent0', 'fire1', 'fire1--before'];
await sheet('overview', sampleKeys.map(key => catalog.assets.find(a => a.id === `agon.world.${key}`)));
fs.writeFileSync(path.join(out, 'render-report.json'), JSON.stringify({ renderer: sharp.versions,
  scope: 'SVG decode and raster review; browser/Unity comparison pending', assets: rows }, null, 2) + '\n');
console.log(`Rendered ${rows.length} SVGs; ${groups.length + 1} contact sheets in ${out}`);
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
