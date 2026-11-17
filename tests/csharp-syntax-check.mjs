import Parser from '/tmp/node_modules/tree-sitter/index.js';
import CSharp from '/tmp/node_modules/tree-sitter-c-sharp/bindings/node/index.js';
import fs from 'fs'; import path from 'path';
const p = new Parser(); p.setLanguage(CSharp);
function walk(dir, out=[]) { for (const e of fs.readdirSync(dir,{withFileTypes:true})) {
  const f = path.join(dir, e.name);
  if (e.isDirectory()) walk(f, out); else if (e.name.endsWith('.cs')) out.push(f); } return out; }
let bad = 0, n = 0;
for (const f of walk(process.argv[2])) {
  const src = fs.readFileSync(f, 'utf8'); const tree = p.parse(src); n++;
  const errs = [], lines = src.split('\n');
  (function visit(node){ if (node.type==='ERROR'||node.isMissing) errs.push({r:node.startPosition.row+1,x:(lines[node.startPosition.row]||'').trim().slice(0,90)});
    for (let i=0;i<node.childCount;i++) visit(node.child(i)); })(tree.rootNode);
  if (errs.length) { bad++; console.log('SYNTAX ' + f); errs.slice(0,4).forEach(e=>console.log('   line '+e.r+': '+e.x)); }
}
console.log(bad ? `\n${bad}/${n} files have syntax errors` : `all ${n} C# files parse clean`);
process.exit(bad?1:0);
