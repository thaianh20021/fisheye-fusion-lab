// Turns "fflab <command> --flags" into the exact script + environment a backend runs with.
// Every option of the command is set explicitly (defaults included): a variable left over in a
// shell or a Colab kernel can then never change a run silently.
import path from 'node:path';
import { parseBool } from './args.js';

export const GLOBAL_FLAGS = new Set([
  'backend', 'accel', 'config', 'python', 'venv', 'dry-run', 'help',
  'session', 'gpu', 'drive', 'remote-root', 'download', 'download-dir', 'poll', 'skip-setup', 'with-ckpt', 'run-name', 'torch-index-url',
]);

export function newRunName(date = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  return `run_${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}_${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
}

export function booleanFlags(backend) {
  const set = new Set(['dry-run', 'help', 'drive', 'download', 'skip-setup', 'with-ckpt']);
  for (const cmd of Object.values(backend.commands)) for (const o of cmd.options || []) if (o.type === 'bool') set.add(o.flag);
  return set;
}

function coerce(option, value, mode, remoteRoot) {
  const { flag, type } = option;
  if (type === 'bool') return (typeof value === 'boolean' ? value : parseBool(value, flag)) ? '1' : '0';
  if (type === 'int') {
    const n = Number(value);
    if (!Number.isInteger(n)) throw new Error(`--${flag} expects an integer, got "${value}"`);
    return String(n);
  }
  if (type === 'float') {
    const n = Number(value);
    if (!Number.isFinite(n)) throw new Error(`--${flag} expects a number, got "${value}"`);
    return String(n);
  }
  if (type === 'path') {
    const s = String(value);
    if (!s) return '';
    if (mode === 'colab') {
      if (s.startsWith('/')) return s;
      return path.posix.join(remoteRoot, s.replace(/^\.\//, ''));
    }
    return path.resolve(s);
  }
  return String(value);
}

/**
 * @param backend   manifest
 * @param command   'train' | 'render' | 'check'
 * @param opts      parsed CLI options (+ config-file values already merged)
 * @param accels    accelerator manifests to apply
 * @param mode      'local' | 'colab'
 */
export function buildPlan({ backend, command, opts, accels = [], mode = 'local', remoteRoot = '/content/fflab', run }) {
  const cmd = backend.commands[command];
  if (!cmd) throw new Error(`backend "${backend.id}" has no "${command}" command`);
  const known = new Set((cmd.options || []).map((o) => o.flag));
  const unknown = Object.keys(opts).filter((k) => !known.has(k) && !GLOBAL_FLAGS.has(k));
  if (unknown.length) {
    throw new Error(`unknown option(s) for ${backend.id} ${command}: ${unknown.map((k) => `--${k}`).join(', ')}\n`
      + `valid: ${[...known].map((k) => `--${k}`).join(' ')}`);
  }
  const runName = run || (opts['run-name'] ? String(opts['run-name']) : newRunName());
  const env = { ...(cmd.fixedEnv || {}) };
  const values = {};
  for (const o of cmd.options || []) {
    let v = opts[o.flag] !== undefined ? opts[o.flag] : o.default;
    if (typeof v === 'string') v = v.replaceAll('{run}', runName);
    if (v === null || v === undefined) continue;
    env[o.env] = coerce(o, v, mode, remoteRoot);
    values[o.flag] = env[o.env];
  }
  for (const a of accels) Object.assign(env, a.env || {});
  const outputs = cmd.outputs ? values[cmd.outputs] : undefined;
  let logFile;
  if (backend.logEnv && outputs) {
    const join = mode === 'colab' ? path.posix : path;
    const dir = join.extname(outputs) ? join.dirname(outputs) : outputs;
    logFile = join.join(dir, `fflab-${command}.log`);
    env[backend.logEnv] = logFile;
  }
  return {
    backend: backend.id,
    command,
    script: cmd.script,
    run: runName,
    env,
    outputs,
    logFile,
    accelerators: accels.map((a) => a.id),
    requiresCuda: cmd.requiresCuda ?? backend.requiresCuda ?? false,
  };
}
