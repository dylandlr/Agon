// Verify all original tracked files against their source Git blobs, including
// binary assets. Git hash-object applies the repository's normal line endings.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const refs = require('./sources.json');
const root = path.resolve(__dirname, '..');
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' });
let count = 0;
for (const [ref, prefix] of [[refs.colony, ''], [refs.ecology, refs.ecologyPrefix]]) {
  for (const entry of git('ls-tree', '-rz', ref).split('\0').filter(Boolean)) {
    const [metadata, name] = entry.split('\t');
    const [mode, type, expected] = metadata.split(' ');
    if (type !== 'blob') throw new Error(`Unsupported source type: ${type} ${name}`);
    const target = prefix + name;
    if (!fs.existsSync(path.join(root, target))) throw new Error(`Missing: ${target}`);
    const actual = git('hash-object', '--', target).trim();
    if (actual !== expected) throw new Error(`Original changed: ${target} (${mode})`);
    count++;
  }
}
console.log(`Verified ${count} original files against both pinned source commits.`);
