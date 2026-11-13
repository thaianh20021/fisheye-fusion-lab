// Backends and accelerators are JSON manifests, so a new method or speed-up is a new file,
// not a code change. FFLAB_PLUGINS_DIR adds a directory with its own backends/ and accelerators/.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const PKG_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const STATUSES = new Set(['stable', 'experimental', 'builtin', 'planned']);
const TYPES = new Set(['path', 'string', 'int', 'float', 'bool']);

function roots() {
  const extra = process.env.FFLAB_PLUGINS_DIR;
  return extra ? [PKG_ROOT, path.resolve(extra)] : [PKG_ROOT];
}

function loadDir(kind) {
  const items = new Map();
  for (const root of roots()) {
    const dir = path.join(root, kind);
    if (!fs.existsSync(dir)) continue;
    for (const file of fs.readdirSync(dir).filter((f) => f.endsWith('.json')).sort()) {
      const full = path.join(dir, file);
      let data;
      try {
        data = JSON.parse(fs.readFileSync(full, 'utf8'));
      } catch (err) {
        throw new Error(`${full}: invalid JSON (${err.message})`);
      }
      data.__file = full;
      data.__root = root;
      items.set(data.id, data);
    }
  }
  return items;
}

export function validateBackend(b) {
  const where = b.__file || b.id;
  const problems = [];
  if (!b.id || !/^[a-z0-9][a-z0-9-]*$/.test(b.id)) problems.push('id must be lower-case kebab');
  if (!STATUSES.has(b.status)) problems.push(`status must be one of ${[...STATUSES].join(', ')}`);
  if (!b.commands || typeof b.commands !== 'object') problems.push('commands missing');
  for (const [name, cmd] of Object.entries(b.commands || {})) {
    if (!cmd.script) problems.push(`commands.${name}.script missing`);
    const flags = new Set();
    for (const o of cmd.options || []) {
      if (!o.flag || !o.env || !TYPES.has(o.type)) problems.push(`commands.${name}: option needs flag, env and a known type (${JSON.stringify(o)})`);
      if (flags.has(o.flag)) problems.push(`commands.${name}: duplicate flag --${o.flag}`);
      flags.add(o.flag);
    }
    if (cmd.outputs && !flags.has(cmd.outputs)) problems.push(`commands.${name}.outputs names unknown flag ${cmd.outputs}`);
  }
  if (problems.length) throw new Error(`${where}: ${problems.join('; ')}`);
  return b;
}

export function validateAccelerator(a) {
  const problems = [];
  if (!a.id) problems.push('id missing');
  if (!STATUSES.has(a.status)) problems.push('bad status');
  if (!Array.isArray(a.backends) || !a.backends.length) problems.push('backends must list at least one backend id');
  if (problems.length) throw new Error(`${a.__file || a.id}: ${problems.join('; ')}`);
  return a;
}

export function loadRegistry() {
  const backends = loadDir('backends');
  const accelerators = loadDir('accelerators');
  backends.forEach(validateBackend);
  accelerators.forEach(validateAccelerator);
  return { backends, accelerators };
}

export function getBackend(reg, id) {
  const b = reg.backends.get(id);
  if (!b) throw new Error(`unknown backend "${id}". Available: ${[...reg.backends.keys()].join(', ')}`);
  if (b.status === 'planned') throw new Error(`backend "${id}" is planned and not runnable yet`);
  return b;
}

/** Accelerators the user asked for, checked against the backend. Builtins are reported, not toggled. */
export function resolveAccelerators(reg, backend, ids) {
  const chosen = [];
  const notes = [];
  for (const id of ids) {
    const a = reg.accelerators.get(id);
    if (!a) throw new Error(`unknown accelerator "${id}". Available: ${[...reg.accelerators.keys()].join(', ')}`);
    if (!a.backends.includes(backend.id)) throw new Error(`accelerator "${id}" does not support backend "${backend.id}"`);
    if (a.status === 'planned') throw new Error(`accelerator "${id}" is planned and not usable yet`);
    if (a.status === 'builtin') {
      notes.push(`${id} is built into ${backend.id} and always on`);
      continue;
    }
    chosen.push(a);
  }
  return { chosen, notes };
}
