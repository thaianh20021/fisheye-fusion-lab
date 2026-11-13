// Copies the backend Python scripts from ../tools into python/ so `npm pack` ships them.
// The repository keeps tools/ as the single source; python/ is generated and git-ignored.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tools = path.resolve(root, '..', 'tools');
const out = path.join(root, 'python');
const scripts = new Set();
for (const file of fs.readdirSync(path.join(root, 'backends')).filter((f) => f.endsWith('.json'))) {
  const b = JSON.parse(fs.readFileSync(path.join(root, 'backends', file), 'utf8'));
  for (const c of Object.values(b.commands)) scripts.add(c.script);
}
fs.mkdirSync(out, { recursive: true });
for (const s of scripts) {
  const src = path.join(tools, s);
  if (!fs.existsSync(src)) throw new Error(`missing ${src}`);
  fs.copyFileSync(src, path.join(out, s));
  console.log(`python/${s}`);
}
