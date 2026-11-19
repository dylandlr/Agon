const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const read = p => fs.readFileSync(path.join(root, p), 'utf8');
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const plan = JSON.parse(read('workspace/migration.json'));
const ids = new Set(plan.tasks.map(t => t.id));
assert(ids.size === plan.tasks.length, 'Duplicate migration task IDs');
const byId = new Map(plan.tasks.map(t => [t.id, t]));
const visited = new Set(), active = new Set();
function visit(id) {
  assert(ids.has(id), `Missing dependency ${id}`);
  assert(!active.has(id), `Dependency cycle at ${id}`);
  if (visited.has(id)) return;
  active.add(id);
  const task = byId.get(id);
  assert(['planned', 'in_progress', 'verified', 'blocked'].includes(task.status), `Invalid status ${id}`);
  assert(/^G\d{2}$/.test(task.gate), `Missing gate ${id}`);
  assert(task.features.length && task.features.every(f => /^F\d{2}$/.test(f)), `Missing features ${id}`);
  if (task.status === 'verified') assert(task.evidence.length, `Verified task lacks evidence: ${id}`);
  for (const file of task.evidence) assert(fs.existsSync(path.join(root, file)), `Missing evidence ${file}`);
  task.dependsOn.forEach(visit);
  active.delete(id); visited.add(id);
}
plan.tasks.forEach(t => visit(t.id));
const assets = path.join(root, 'unity-game/Assets'), guids = new Map();
let assetFiles = 0;
function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const file = path.join(dir, entry.name);
    if (entry.name.endsWith('.meta')) {
      assert(fs.existsSync(file.slice(0, -5)), `Orphan metadata: ${file}`);
      const match = /^guid: ([a-f0-9]{32})$/m.exec(fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n'));
      assert(match, `Invalid GUID: ${file}`);
      assert(!guids.has(match[1]), `Duplicate GUID: ${file} and ${guids.get(match[1])}`);
      guids.set(match[1], file);
    } else {
      assert(fs.existsSync(file + '.meta'), `Missing metadata: ${file}`);
      if (entry.isDirectory()) walk(file); else assetFiles++;
    }
  }
}
walk(assets);
assert(read('unity-game/ProjectSettings/EditorSettings.asset').includes('m_SerializationMode: 2'), 'Use Force Text serialization');
assert(read('unity-game/ProjectSettings/VersionControlSettings.asset').includes('m_Mode: Visible Meta Files'), 'Keep visible metadata');
assert(read('unity-game/ProjectSettings/ProjectVersion.txt').includes('6000.6.5f1'), 'Editor pin changed: update migration evidence');
const packages = JSON.parse(read('unity-game/Packages/manifest.json')).dependencies;
const locked = JSON.parse(read('unity-game/Packages/packages-lock.json')).dependencies;
for (const [name, version] of Object.entries(packages)) {
  assert(/^\d+\.\d+\.\d+(?:[-+].+)?$/.test(version), `Package not version-pinned: ${name}`);
  assert(locked[name]?.version === version, `Package lock mismatch: ${name}`);
}
const catalog = JSON.parse(read('asset-library/catalog.json'));
assert(new Set(catalog.assets.map(a => a.id)).size === catalog.assets.length, 'Duplicate semantic asset IDs');
for (const asset of catalog.assets) {
  assert(asset.file.startsWith('asset-library/generated/agon/') && !asset.file.includes('..'), `Invalid asset path: ${asset.id}`);
  assert(fs.existsSync(path.join(root, asset.file)), `Missing asset: ${asset.id}`);
  assert(fs.existsSync(path.join(root, asset.provenance.source)), `Missing asset provenance: ${asset.id}`);
}
console.log(`Repository checks passed: ${plan.tasks.length} acyclic work items, ${assetFiles} Unity assets, ${guids.size} unique GUIDs, ${catalog.assets.length} catalog assets.`);
