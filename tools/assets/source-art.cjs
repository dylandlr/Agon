const acorn = require('acorn');
const scope = require('eslint-scope');
const vm = require('node:vm');

const parse = text => acorn.parse(text, { ecmaVersion: 2022, sourceType: 'script', ranges: true });

function packedComponents(html) {
  const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)];
  for (const script of scripts) {
    const at = script[1].indexOf('var D=');
    if (at < 0) continue;
    const expression = acorn.parseExpressionAt(script[1], at + 6, { ecmaVersion: 2022 });
    // The packer serializes definitions as JSON. Never execute the packed app.
    const definitions = JSON.parse(script[1].slice(expression.start, expression.end));
    if (!Array.isArray(definitions) || !definitions.every(d => typeof d.n === 'string' && typeof d.s === 'string' && typeof d.t === 'string'))
      throw new Error('Unrecognized packed component definitions');
    return definitions;
  }
  throw new Error('No packed component definitions found');
}

function isolateDeclarations(source, targets) {
  const analyzed = scope.analyze(parse(source), { ecmaVersion: 2022, sourceType: 'script', optimistic: true, ignoreEval: true });
  const variables = analyzed.scopes.flatMap(s => s.variables);
  const references = analyzed.scopes.flatMap(s => s.references);
  const builtins = new Set(['window', 'Math', 'Object', 'Array', 'String', 'Number', 'Boolean', 'Set', 'Map',
    'parseInt', 'parseFloat', 'encodeURIComponent', 'decodeURIComponent', 'JSON', 'Infinity', 'NaN', 'undefined']);
  const selected = new Map();
  function collect(variable) {
    const name = variable.name;
    if (selected.get(name)?.variable === variable) return;
    if (selected.has(name)) throw new Error(`Artwork depends on conflicting lexical bindings: ${name}`);
    if (variable.defs.length !== 1) throw new Error(`Artwork binding has multiple definitions: ${name}`);
    const def = variable.defs[0], node = def.node;
    let code;
    if (def.type === 'FunctionName' && node.type === 'FunctionDeclaration') code = source.slice(node.start, node.end);
    else if (def.type === 'Variable' && node.id.type === 'Identifier') code = `${def.parent.kind} ${source.slice(node.start, node.end)};`;
    else throw new Error(`Unsupported artwork binding: ${name} (${def.type})`);
    selected.set(name, { variable, start: node.start, code });
    for (const ref of references) {
      if (ref.identifier.start < node.start || ref.identifier.end > node.end) continue;
      if (!ref.resolved) {
        if (!builtins.has(ref.identifier.name)) throw new Error(`Unresolved artwork dependency: ${ref.identifier.name}`);
      } else if (!ref.resolved.defs.some(d => d.node.start >= node.start && d.node.end <= node.end)) collect(ref.resolved);
    }
  }
  for (const name of targets) {
    const matches = variables.filter(v => v.name === name && v.defs.length);
    if (matches.length !== 1) throw new Error(`Expected one artwork binding for ${name}, found ${matches.length}`);
    collect(matches[0]);
  }
  const ordered = [...selected].sort((a, b) => a[1].start - b[1].start);
  return { symbols: ordered.map(([name]) => name), code: ordered.map(([, d]) => d.code).join('\n\n') + '\n' };
}

function renderWorld(declarations, artSource) {
  const sandbox = { window: {} };
  vm.createContext(sandbox, { codeGeneration: { strings: false, wasm: false } });
  vm.runInContext(artSource, sandbox, { timeout: 5000, filename: 'Agon art source' });
  vm.runInContext(declarations + '\nwindow.__assetCss = SPRITE_CSS;', sandbox, { timeout: 5000, filename: 'isolated Agon world artwork' });
  const assets = [];
  for (const match of sandbox.window.__assetCss.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    const image = /background-image:\s*url\('data:image\/svg\+xml;utf8,([^']*)'\)/.exec(match[2]);
    if (!image) continue;
    const selector = match[1].trim();
    const svg = decodeURIComponent(image[1]);
    const viewBox = /viewBox="([^"]+)"/.exec(svg)?.[1];
    if (!viewBox || /(?:undefined|NaN)/.test(svg)) throw new Error(`Invalid SVG for ${selector}`);
    assets.push({ key: selector.replace(/^\./, '').replace(/[^a-zA-Z0-9.-]/g, '-').toLowerCase(), selector,
      viewBox, svg: svg + '\n', css: match[2].replace(image[0], '').replace(/^;/, '') });
  }
  if (!assets.length || new Set(assets.map(a => a.key)).size !== assets.length) throw new Error('Missing or duplicate world artwork IDs');
  return { assets, css: sandbox.window.__assetCss };
}

module.exports = { packedComponents, isolateDeclarations, renderWorld };
