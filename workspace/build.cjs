// Build the current colony source without replacing a historical bin/ artifact.
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'node_modules', '.cache', 'agon');
fs.mkdirSync(out, { recursive: true });
const result = spawnSync(process.execPath,
  [path.join(root, 'pack/build.js'), path.join(root, 'src'), out],
  { cwd: root, env: { ...process.env, MAINONLY: '1' }, stdio: 'inherit' });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
console.log(`Open ${path.join(out, 'Agon.html')}`);
