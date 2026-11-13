// Finding Python, managing the venv and running backend scripts on Linux, macOS and Windows.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { PKG_ROOT } from './registry.js';

export const IS_WINDOWS = process.platform === 'win32';
export const DEFAULT_VENV = path.join(os.homedir(), '.fflab', 'venv');
const stateFile = () => path.join(os.homedir(), '.fflab', 'venv-path');

/** `fflab setup --venv X` remembers X, so later commands find it without repeating --venv. */
export function rememberVenv(venv) {
  fs.mkdirSync(path.dirname(stateFile()), { recursive: true });
  fs.writeFileSync(stateFile(), path.resolve(venv));
}

export function defaultVenv() {
  if (process.env.FFLAB_VENV) return path.resolve(process.env.FFLAB_VENV);
  try {
    const remembered = fs.readFileSync(stateFile(), 'utf8').trim();
    if (remembered) return remembered;
  } catch { /* no venv remembered yet */ }
  return path.join(os.homedir(), '.fflab', 'venv');
}

/** Python scripts: ../tools in a git checkout, python/ inside an installed package. */
export function scriptsDir() {
  if (process.env.FFLAB_SCRIPTS_DIR) return path.resolve(process.env.FFLAB_SCRIPTS_DIR);
  // A git checkout uses tools/ directly, so a stale python/ copy left by `npm pack` is never run.
  const repoTools = path.resolve(PKG_ROOT, '..', 'tools');
  if (fs.existsSync(path.join(repoTools, 'colab_cli_kitti360.py'))) return repoTools;
  return path.join(PKG_ROOT, 'python');
}

export function scriptPath(name) {
  const p = path.join(scriptsDir(), name);
  if (!fs.existsSync(p)) throw new Error(`backend script not found: ${p}`);
  return p;
}

export function venvPython(venv) {
  return IS_WINDOWS ? path.join(venv, 'Scripts', 'python.exe') : path.join(venv, 'bin', 'python');
}

function tryPython(cmd, args) {
  const r = spawnSync(cmd, [...args, '-c', 'import sys; print(sys.version.split()[0]); print(sys.executable)'],
    { encoding: 'utf8', timeout: 20000, windowsHide: true });
  if (r.status !== 0 || !r.stdout) return null;
  const [version, executable] = r.stdout.trim().split(/\r?\n/);
  const [maj, min] = version.split('.').map(Number);
  if (maj !== 3 || min < 9) return null;
  return { cmd, args, version, executable };
}

export function findSystemPython() {
  const candidates = IS_WINDOWS
    ? [['py', ['-3']], ['python', []], ['python3', []]]
    : [['python3', []], ['python', []]];
  for (const [cmd, args] of candidates) {
    const found = tryPython(cmd, args);
    if (found) return found;
  }
  return null;
}

/** --python wins, then an existing venv (--venv or ~/.fflab/venv), then the system Python. */
export function resolvePython({ python, venv } = {}) {
  if (python) {
    const found = tryPython(python, []);
    if (!found) throw new Error(`--python ${python} is not a working Python 3.9+`);
    return { ...found, source: '--python' };
  }
  const venvDir = venv ? path.resolve(venv) : defaultVenv();
  if (fs.existsSync(venvPython(venvDir))) {
    const found = tryPython(venvPython(venvDir), []);
    if (found) return { ...found, source: `venv ${venvDir}` };
  }
  const sys = findSystemPython();
  if (!sys) throw new Error('no Python 3.9+ found. Install Python or pass --python <path>.');
  return { ...sys, source: 'system' };
}

/** Run Python code and parse its last stdout line as JSON. */
export function pyJson(py, code, timeout = 120000) {
  const r = spawnSync(py.cmd, [...py.args, '-c', code], { encoding: 'utf8', timeout, windowsHide: true });
  const line = (r.stdout || '').trim().split(/\r?\n/).pop();
  try {
    return JSON.parse(line);
  } catch {
    return { error: (r.stderr || r.stdout || 'no output').trim().split(/\r?\n/).slice(-3).join(' | ') };
  }
}

/** Run a command, stream its output to the terminal and to logFile, resolve with its exit code. */
export function runStreaming(cmd, args, { env = {}, logFile, cwd } = {}) {
  return new Promise((resolve, reject) => {
    let log;
    if (logFile) {
      fs.mkdirSync(path.dirname(logFile), { recursive: true });
      log = fs.createWriteStream(logFile, { flags: 'w' });
    }
    const child = spawn(cmd, args, {
      env: { ...process.env, PYTHONUNBUFFERED: '1', ...env },
      cwd,
      stdio: ['inherit', 'pipe', 'pipe'],
      windowsHide: true,
    });
    const pipe = (stream, sink) => stream.on('data', (chunk) => {
      sink.write(chunk);
      if (log) log.write(chunk);
    });
    pipe(child.stdout, process.stdout);
    pipe(child.stderr, process.stderr);
    const forward = (sig) => child.kill(sig);
    process.on('SIGINT', forward);
    child.on('error', (err) => {
      process.off('SIGINT', forward);
      reject(err);
    });
    child.on('close', (code) => {
      process.off('SIGINT', forward);
      if (log) log.end();
      resolve(code ?? 1);
    });
  });
}

export function pipInstallArgs(packages, { noBuildIsolation = false, indexUrl } = {}) {
  const args = ['-m', 'pip', 'install'];
  if (noBuildIsolation) args.push('--no-build-isolation');
  if (indexUrl) args.push('--index-url', indexUrl);
  return [...args, ...packages];
}
