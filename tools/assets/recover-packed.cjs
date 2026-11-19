// Explicit, local recovery step. The packed document is data, never an app to run.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { packedComponents, isolateDeclarations, renderWorld } = require('./source-art.cjs');
const root = path.resolve(__dirname, '../..');
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const normalize = value => value.replace(/\r\n/g, '\n');
const [file, label] = process.argv.slice(2);
if (!file || !/^[a-z0-9-]+$/.test(label || '')) throw new Error('Usage: node tools/assets/recover-packed.cjs <HTML file> <version-label>');
const output = path.join(root, 'asset-library/source/recovered', label);
if (fs.existsSync(output)) throw new Error(`Recovery already exists; refusing to overwrite ${label}`);
const bytes = fs.readFileSync(file), html = bytes.toString('utf8');
const definitions = packedComponents(html);
const main = definitions.find(d => d.n === 'Main');
if (!main) throw new Error('Packed Main component missing');
const embeddedArt = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)]
  .map(m => m[1]).find(s => s.includes('window.AgonArt'));
const art = normalize(fs.readFileSync(path.join(root, 'src/art.js'), 'utf8'));
if (!embeddedArt || normalize(embeddedArt).trim() !== art.trim())
  throw new Error('Embedded art differs from src/art.js; recover it separately before rendering');
const closure = isolateDeclarations(normalize(main.s), ['SPRITE_CSS']);
const rendered = renderWorld(closure.code, art);
const manifest = {
  schemaVersion: 1, label,
  artifact: { fileName: path.basename(file), bytes: bytes.length, sha256: sha(bytes) },
  method: 'Parse packed JSON; resolve SPRITE_CSS lexical dependencies; execute only recovered artwork declarations',
  sharedArt: { source: 'src/art.js', sha256: sha(art), embeddedContentMatches: true },
  closure: { file: 'world.js', sha256: sha(closure.code), symbols: closure.symbols },
  components: definitions.map(d => {
    const local = path.join(root, 'src', `${d.n}.dc.html`);
    const text = fs.existsSync(local) ? normalize(fs.readFileSync(local, 'utf8')) : '';
    const script = /<script[^>]*type="text\/x-dc"[^>]*>([\s\S]*?)<\/script>/.exec(text)?.[1];
    return { name: d.n, scriptSha256: sha(normalize(d.s)), templateSha256: sha(normalize(d.t)),
      scriptMatchesRepository: script?.trim() === normalize(d.s).trim() };
  }),
  sprites: rendered.assets.map(a => ({ selector: a.selector, key: a.key, sha256: sha(a.svg) }))
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, 'world.js'), closure.code);
fs.writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`Recovered ${rendered.assets.length} world sprites from ${label}; original HTML unchanged.`);
