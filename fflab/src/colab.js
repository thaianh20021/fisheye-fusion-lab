// Drives a Google Colab VM through the official `colab` CLI (google-colab-cli).
// Long jobs run detached on the VM with a log and an exit file; we poll the log in chunks, because a
// `colab exec` that stays open for minutes tends to lose its connection. The environment of every job
// is written into its launcher, never passed with `colab exec --env`, which persists in the kernel.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

export const colabBin = () => process.env.FFLAB_COLAB_BIN || 'colab';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function colab(args, { timeout = 600000, inherit = false } = {}) {
  const r = spawnSync(colabBin(), args, {
    encoding: 'utf8',
    timeout,
    stdio: inherit ? 'inherit' : 'pipe',
    env: process.env,
  });
  if (r.error && r.error.code === 'ENOENT') {
    throw new Error('the `colab` CLI is not installed. Install it with: uv tool install google-colab-cli --with keyrings.alt');
  }
  return { code: r.status ?? 1, out: r.stdout || '', err: r.stderr || '' };
}

export function checkColabAvailable() {
  if (process.platform === 'win32') {
    throw new Error('google-colab-cli supports Linux and macOS only. On Windows run fflab inside WSL, or use the notebook.');
  }
  colab(['--help'], { timeout: 30000 });
}

export function listSessions() {
  const r = colab(['sessions'], { timeout: 60000 });
  return [...r.out.matchAll(/^\[([^\]]+)\]/gm)].map((m) => m[1]);
}

export function ensureSession(name, gpu) {
  if (listSessions().includes(name)) return 'existing';
  const args = ['new', '-s', name];
  if (gpu && gpu.toLowerCase() !== 'none') args.push('--gpu', gpu);
  const r = colab(args, { timeout: 600000 });
  if (r.code !== 0 || !/READY/i.test(r.out + r.err)) {
    throw new Error(`could not create Colab session "${name}" (${gpu || 'CPU'}): ${(r.err || r.out).trim().split('\n').slice(-2).join(' ')}`);
  }
  return 'created';
}

/** Run a Python snippet on the VM and return its stdout; retried because the tunnel drops now and then. */
export async function execPython(session, code, { timeoutSec = 60, attempts = 3 } = {}) {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'fflab-')), 'snippet.py');
  fs.writeFileSync(file, code);
  let last = '';
  for (let i = 0; i < attempts; i++) {
    const r = colab(['exec', '-s', session, '--timeout', String(timeoutSec), '-f', file], { timeout: (timeoutSec + 30) * 1000 });
    if (r.code === 0 && !/Connection was lost/.test(r.out + r.err)) {
      fs.rmSync(path.dirname(file), { recursive: true, force: true });
      return r.out;
    }
    last = (r.err || r.out).trim().split('\n').slice(-2).join(' ');
    await sleep(3000 * (i + 1));
  }
  fs.rmSync(path.dirname(file), { recursive: true, force: true });
  throw new Error(`colab exec failed after ${attempts} attempts: ${last}`);
}

/** The upload API refuses (HTTP 500) a path whose parent directory does not exist yet. */
export async function mkdirs(session, dirs) {
  const out = await execPython(session, `import os
for d in ${JSON.stringify(dirs)}:
    os.makedirs(d, exist_ok=True)
print('@@MKDIR')
`);
  if (!out.includes('@@MKDIR')) throw new Error(`could not create ${dirs.join(', ')} on the VM: ${out.trim()}`);
}

export function upload(session, local, remote) {
  const r = colab(['upload', '-s', session, local, remote], { timeout: 600000 });
  if (r.code !== 0) throw new Error(`upload ${local} failed: ${(r.err || r.out).trim()}`);
}

export function download(session, remote, local) {
  fs.mkdirSync(path.dirname(local), { recursive: true });
  const r = colab(['download', '-s', session, remote, local], { timeout: 1800000 });
  if (r.code !== 0 || /failed/i.test(r.out)) throw new Error(`download ${remote} failed: ${(r.err || r.out).trim()}`);
}

// A Python expression for json.loads(...): JSON text inside a (JSON-escaped) string literal.
const pyLiteral = (obj) => JSON.stringify(JSON.stringify(obj));

/** Start `argv` detached on the VM with exactly `env` on top of the kernel's. */
export async function launchJob(session, { name, root, argv, env }) {
  const job = { log: `${root}/jobs/${name}.log`, exit: `${root}/jobs/${name}.exit` };
  const code = `import json, os, shlex, subprocess, sys, pathlib
cfg = json.loads(${pyLiteral({ ...job, argv, env })})
pathlib.Path(cfg['log']).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(cfg['exit']).unlink(missing_ok=True)
argv = [sys.executable if a == '{python}' else a for a in cfg['argv']]
env = dict(os.environ); env.update(cfg['env']); env['PYTHONUNBUFFERED'] = '1'
cmd = f"{shlex.join(argv)} > {shlex.quote(cfg['log'])} 2>&1; echo $? > {shlex.quote(cfg['exit'])}"
subprocess.Popen(['bash', '-c', cmd], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True)
print('@@LAUNCHED')
`;
  const out = await execPython(session, code);
  if (!out.includes('@@LAUNCHED')) throw new Error(`job ${name} did not start: ${out.trim()}`);
  return job;
}

/** Read the job log from `offset`, only whole lines; report the exit code once the log is drained. */
export function pollCode(job, offset, chunk = 65536) {
  return `import os
log, exitf, off = ${JSON.stringify(job.log)}, ${JSON.stringify(job.exit)}, ${Number(offset)}
data = b''
if os.path.exists(log):
    with open(log, 'rb') as f:
        f.seek(off)
        data = f.read(${chunk})
done = os.path.exists(exitf)
more = len(data) == ${chunk}
if not (done and not more):
    data = data[:data.rfind(b'\\n') + 1]
print('@@OFF', off + len(data))
for line in data.decode('utf-8', 'replace').splitlines():
    print('@@L ' + line)
if done and not more:
    print('@@EXIT', (open(exitf).read().strip() or '1'))
`;
}

export function parsePoll(out) {
  let offset = null;
  let exit = null;
  const lines = [];
  for (const raw of out.split(/\r?\n/)) {
    if (raw.startsWith('@@OFF ')) offset = Number(raw.slice(6));
    else if (raw.startsWith('@@L ')) lines.push(raw.slice(4));
    else if (raw === '@@L') lines.push('');
    else if (raw.startsWith('@@EXIT ')) exit = Number(raw.slice(7));
  }
  return { offset, exit, lines };
}

export async function followJob(session, job, { intervalMs = 15000, onLine = (l) => console.log(l), maxFailures = 20 } = {}) {
  let offset = 0;
  let failures = 0;
  for (;;) {
    let res;
    try {
      res = parsePoll(await execPython(session, pollCode(job, offset), { attempts: 1 }));
      failures = 0;
    } catch (err) {
      failures++;
      if (failures >= maxFailures) throw new Error(`lost contact with the job after ${failures} polls (${err.message}); it keeps running on the VM: re-attach with fflab colab logs`);
      await sleep(intervalMs);
      continue;
    }
    res.lines.forEach(onLine);
    if (res.offset !== null) offset = res.offset;
    if (res.exit !== null) return res.exit;
    if (!res.lines.length) await sleep(intervalMs);
  }
}

export async function listRemote(session, target) {
  const out = await execPython(session, `import json, os
t = ${JSON.stringify(target)}
files = []
if os.path.isfile(t):
    files.append([t, os.path.getsize(t)])
elif os.path.isdir(t):
    for d, _, names in os.walk(t):
        for n in sorted(names):
            p = os.path.join(d, n)
            files.append([p, os.path.getsize(p)])
print('@@FILES ' + json.dumps(files))
`);
  const line = out.split(/\r?\n/).find((l) => l.startsWith('@@FILES '));
  return line ? JSON.parse(line.slice(8)).map(([p, size]) => ({ path: p, size })) : [];
}

/** Remote setup: install only what the import check says is missing (a command may narrow both lists). */
export function setupArgv(backend, accels, command) {
  const cmd = (command && backend.commands[command]) || {};
  const imports = [...(cmd.importCheck || backend.importCheck || []), ...accels.flatMap((a) => a.importCheck || [])];
  const groups = [
    { pkgs: cmd.requirements || backend.requirements || [], iso: true },
    { pkgs: cmd.buildRequirements || backend.buildRequirements || [], iso: false },
    ...accels.map((a) => ({ pkgs: [...(a.requirements || []), ...(a.buildRequirements || [])], iso: !(a.buildRequirements || []).length })),
  ].filter((g) => g.pkgs.length);
  const code = `import importlib.util, json, subprocess, sys
imports, groups = json.loads(${pyLiteral({ imports, groups })}).values()
missing = [m for m in imports if importlib.util.find_spec(m) is None]
print('missing modules:', missing or 'none', flush=True)
if missing:
    for g in groups:
        cmd = [sys.executable, '-m', 'pip', 'install', '-q'] + ([] if g['iso'] else ['--no-build-isolation']) + g['pkgs']
        print('+', ' '.join(cmd[1:]), flush=True)
        subprocess.check_call(cmd)
if importlib.util.find_spec('torch'):
    import torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
`;
  return ['{python}', '-c', code];
}
