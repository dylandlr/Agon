// Reconstruct .dc.html sources from a packed Agon build (pack/build.js output).
// Usage: node pack/recover.js <built.html> <wrapperSrcDir> <outDir>
// The build embeds each component verbatim: art.js as the first <script>, and
// D=[{n,s,t,p}] (name, script, template, props) in the mount script. Script and
// template are exact; the .dc.html wrapper (helmet, data-props) is taken from
// wrapperSrcDir's copy of the same component, with its script and template swapped out.
const fs = require('fs'), path = require('path');
const [BUILT, WRAP, OUT] = process.argv.slice(2);
const html = fs.readFileSync(BUILT, 'utf8');
fs.mkdirSync(OUT, { recursive: true });

// art.js: first <script> tag body (build wrote art first, then runtime, then mount)
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (scripts.length < 3) throw new Error('expected 3 scripts, got ' + scripts.length);
const art = scripts[0].replace(/<\\\/script/gi, '</script');
fs.writeFileSync(path.join(OUT, 'art.js'), art);

// defs: var D=[...];try{
const mnt = scripts[scripts.length - 1];
const m = /var D=(\[[\s\S]*?\]);try\{/.exec(mnt);
if (!m) throw new Error('D array not found');
const D = JSON.parse(m[1]);
console.log('components in build:', D.map(d => d.n).join(', '));

for (const d of D) {
  const wp = path.join(WRAP, d.n + '.dc.html');
  if (!fs.existsSync(wp)) { console.log('NO WRAPPER for ' + d.n + ' — writing bare');
    fs.writeFileSync(path.join(OUT, d.n + '.dc.html'),
      '<script type="text/x-dc">' + d.s + '</script>\n<x-dc>' + d.t + '</x-dc>\n');
    continue; }
  let w = fs.readFileSync(wp, 'utf8');
  // swap script body (keep the opening tag with data-props etc.)
  const sm = /(<script type="text\/x-dc"[^>]*>)([\s\S]*?)(<\/script>)/.exec(w);
  if (!sm) throw new Error('no x-dc script in wrapper ' + d.n);
  w = w.slice(0, sm.index) + sm[1] + d.s + sm[3] + w.slice(sm.index + sm[0].length);
  // swap template body, preserving wrapper's <helmet> (build strips it from t)
  const tm = /(<x-dc>)([\s\S]*?)(<\/x-dc>)/.exec(w);
  if (!tm) throw new Error('no x-dc template in wrapper ' + d.n);
  const helm = (/<helmet>[\s\S]*?<\/helmet>/.exec(tm[2]) || [''])[0];
  // build rewrote href="X.dc.html" -> href="Agon.html"/"X.html" for packed pages; undo
  let t = d.t.replace(/href="Agon\.html"/g, 'href="Main.dc.html"')
             .replace(/href="(Encyclopedia|Character|StyleGuide|Workshop)\.html"/g, 'href="$1.dc.html"');
  w = w.slice(0, tm.index) + tm[1] + helm + t + tm[3] + w.slice(tm.index + tm[0].length);
  fs.writeFileSync(path.join(OUT, d.n + '.dc.html'), w);
  const orig = fs.readFileSync(wp, 'utf8');
  console.log(d.n + ': ' + (w === orig ? 'identical to wrapper source' : 'DIFFERS from wrapper source') + ' (' + w.length + ' bytes)');
}
