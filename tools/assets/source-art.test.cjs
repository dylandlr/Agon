const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { packedComponents, isolateDeclarations, renderWorld } = require('./source-art.cjs');
const root = path.resolve(__dirname, '../..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8').replace(/\r\n/g, '\n');
const sha = text => crypto.createHash('sha256').update(text).digest('hex');

test('packed definitions decode escaped strings without executing document scripts', () => {
  const definitions = [{ n: 'Main', s: 'const quote = "\\n";', t: '<svg></svg>' }];
  const html = `<script>throw Error('must not run'); var D=${JSON.stringify(definitions)}; launch();</script>`;
  assert.deepEqual(packedComponents(html), definitions);
  assert.throws(() => packedComponents('<script>var D=launch();</script>'));
});

test('dependency closure follows lexical bindings and excludes simulation startup', () => {
  const input = `const ink='outer'; function ENGINE(){const ink='inner';
    function color(){return ink;} const SPRITE_CSS=color(); launchSimulation();}
    throw new Error('must not execute');`;
  const selected = isolateDeclarations(input, ['SPRITE_CSS']);
  assert.deepEqual(selected.symbols, ['ink', 'color', 'SPRITE_CSS']);
  assert.ok(selected.code.includes("ink='inner'"));
  assert.ok(!selected.code.includes('outer'));
  assert.ok(!selected.code.includes('launchSimulation'));
});

test('closure fails for missing, ambiguous or unsupported dependencies', () => {
  assert.throws(() => isolateDeclarations('const SPRITE_CSS=missing();', ['SPRITE_CSS']), /Unresolved/);
  assert.throws(() => isolateDeclarations('const SPRITE_CSS=1; function f(){const SPRITE_CSS=2;}', ['SPRITE_CSS']), /found 2/);
  assert.throws(() => isolateDeclarations('function f(value){const SPRITE_CSS=value;}', ['SPRITE_CSS']), /Unsupported/);
});

const script = /<script[^>]*type="text\/x-dc"[^>]*>([\s\S]*?)<\/script>/.exec(read('src/Main.dc.html'))[1];
const closure = isolateDeclarations(script, ['SPRITE_CSS']);
const rendered = renderWorld(closure.code, read('src/art.js'));

test('world recovery includes every wall mask/window/material and seasonal tree variant', () => {
  const keys = new Set(rendered.assets.map(a => a.key));
  assert.equal(keys.size, 333);
  for (const material of ['w', 't', 's']) for (let mask = 0; mask < 16; mask++)
    for (let window = 0; window < 2; window++) assert.ok(keys.has(`wl${material}${mask}${window}`));
  assert.equal([...keys].filter(k => /^(q\d\.)?t-/.test(k)).length, 50);
  assert.equal([...keys].filter(k => /^wcf/.test(k)).length, 3);
  assert.equal(rendered.assets.filter(a => a.selector.includes('::')).length, 3);
  assert.ok(rendered.assets.some(a => a.css.includes('animation:flameB')));
  assert.ok(read('asset-library/generated/agon/world/presentation.css').includes('@keyframes flameB'));
});

test('r18 independently reproduces all 330 matching sprites, without current corner fillers', () => {
  const recovered = read('asset-library/source/recovered/r18/world.js');
  const manifest = JSON.parse(read('asset-library/source/recovered/r18/manifest.json'));
  assert.equal(sha(recovered), manifest.closure.sha256);
  const old = renderWorld(recovered, read('src/art.js'));
  assert.equal(old.assets.length, 330);
  const current = new Map(rendered.assets.map(a => [a.selector, a.svg]));
  for (const asset of old.assets) assert.equal(asset.svg, current.get(asset.selector));
  assert.equal(manifest.components.filter(c => c.scriptMatchesRepository).length, 9);
});

test('standalone world SVGs carry dimensions and no unresolved or external image dependencies', () => {
  for (const asset of rendered.assets) {
    assert.match(asset.svg, /^<svg\b/);
    assert.match(asset.viewBox, /^0 0 \d+(?:\.\d+)? \d+(?:\.\d+)?$/);
    assert.doesNotMatch(asset.svg, /(?:undefined|NaN|<script\b|<foreignObject\b|(?:href|src)=["'](?:https?:|\/\/))/);
  }
});
