import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { parseArgs } from './args.js';
import { loadRegistry, getBackend, resolveAccelerators, PKG_ROOT } from './registry.js';
import { buildPlan, booleanFlags, GLOBAL_FLAGS } from './plan.js';
import {
  IS_WINDOWS, DEFAULT_VENV, findSystemPython, pipInstallArgs, pyJson, resolvePython, runStreaming,
  scriptPath, scriptsDir, venvPython,
} from './python.js';
import * as remote from './colab.js';

export const DEFAULT_BACKEND = '3dgut-gsplat';
const RUN_COMMANDS = ['train', 'render', 'check'];
const STATE_FILE = path.join(os.homedir(), '.fflab', 'colab-jobs.json');
const VERSION = JSON.parse(fs.readFileSync(path.join(PKG_ROOT, 'package.json'), 'utf8')).version;
const say = (...a) => console.log(...a);

const USAGE = `fflab ${VERSION} - Fisheye Fusion Lab CLI

Usage
  fflab list                         backends and accelerators
  fflab doctor                       check Python, PyTorch, CUDA, GPU and the colab CLI
  fflab setup [--venv DIR] [--accel ID] [--torch-index-url URL]
                                     create a venv and install the backend (+ accelerators)
  fflab train  [options]             train on this machine (needs a CUDA GPU)
  fflab render [options]             fly-through video from a trained PLY
  fflab check  [options]             LiDAR-to-camera projection check (CPU is enough)
  fflab colab train|render|check [options] [--session S] [--gpu T4] [--drive]
              [--download-dir DIR] [--no-download] [--with-ckpt] [--skip-setup]
                                     run the same command on a Google Colab VM
  fflab colab logs [--run NAME]      re-attach to a Colab job
  fflab colab stop [--session S]     release the Colab VM
  fflab help <train|render|check>    options of a command

Common options
  --backend ID       backend (default ${DEFAULT_BACKEND})
  --accel ID[,ID]    accelerators to enable (see fflab list)
  --config FILE      JSON with defaults: {"backend": "...", "accel": [...], "train": {...}, "colab": {...}}
  --python PATH      Python to use (default: venv, then system)
  --venv DIR         venv (default ${DEFAULT_VENV})
  --dry-run          print the plan, run nothing
`;

function scanValue(argv, key) {
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === `--${key}`) return argv[i + 1];
    if (argv[i].startsWith(`--${key}=`)) return argv[i].slice(key.length + 3);
  }
  return undefined;
}

function loadConfig(file) {
  if (!file) return {};
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`--config ${file}: ${err.message}`);
  }
}

/** Options for one command: config file values first, CLI flags on top. */
function parseFor(reg, argv, command) {
  const config = loadConfig(scanValue(argv, 'config'));
  const backendId = scanValue(argv, 'backend') || config.backend || DEFAULT_BACKEND;
  const backend = getBackend(reg, backendId);
  const parsed = parseArgs(argv, { booleans: booleanFlags(backend), lists: new Set(['accel']) });
  const opts = { ...(config.colab || {}), ...(config[command] || {}), ...parsed.opts };
  if (!opts.accel && config.accel) opts.accel = config.accel;
  opts.backend = backend.id;
  return { backend, opts, positionals: parsed._ };
}

function printPlan(plan, py) {
  const kv = (k, v) => say(`${k.padEnd(13)}${v}`);
  kv('backend', `${plan.backend} (${plan.command})`);
  kv('run', plan.run);
  if (py) kv('python', `${py.executable} (${py.version}, ${py.source})`);
  kv('script', plan.script);
  if (plan.accelerators.length) kv('accelerators', plan.accelerators.join(', '));
  if (plan.outputs) kv('outputs', plan.outputs);
}

function printEnv(env) {
  for (const [k, v] of Object.entries(env)) say(`  ${k}=${v}`);
}

// ------------------------------------------------------------------------------------------- list
function cmdList(reg) {
  say('Backends');
  for (const b of reg.backends.values()) {
    say(`  ${b.id.padEnd(18)} ${b.status.padEnd(12)} ${b.name}`);
    say(`  ${''.padEnd(18)} commands: ${Object.keys(b.commands).join(', ')}; accelerators: ${(b.accelerators || []).join(', ') || 'none'}`);
  }
  say('\nAccelerators');
  for (const a of reg.accelerators.values()) {
    say(`  ${a.id.padEnd(18)} ${a.status.padEnd(12)} ${a.name}`);
    say(`  ${''.padEnd(18)} ${a.description}`);
  }
  say(`\nPlugins: drop more manifests in $FFLAB_PLUGINS_DIR/{backends,accelerators}/*.json`);
  return 0;
}

function cmdHelp(reg, topic) {
  if (!RUN_COMMANDS.includes(topic)) {
    say(USAGE);
    return 0;
  }
  for (const b of reg.backends.values()) {
    const cmd = b.commands[topic];
    if (!cmd) continue;
    say(`fflab ${topic} --backend ${b.id}`);
    for (const o of cmd.options) {
      const def = o.type === 'bool' ? (o.default ? 'on' : 'off') : JSON.stringify(o.default);
      say(`  --${(o.type === 'bool' ? `[no-]${o.flag}` : `${o.flag} ${o.type.toUpperCase()}`).padEnd(26)} ${o.help} (default ${def})`);
    }
    say('');
  }
  return 0;
}

// ----------------------------------------------------------------------------------------- doctor
const PROBE = `import importlib.util, json, platform
r = {'python': platform.python_version()}
for m in ['torch', 'gsplat', 'cv2', 'scipy', 'fused_ssim']:
    r[m] = importlib.util.find_spec(m) is not None
try:
    import torch
    r['torch_version'] = torch.__version__
    r['torch_cuda'] = torch.version.cuda
    r['cuda'] = torch.cuda.is_available()
    r['gpu'] = torch.cuda.get_device_name(0) if r['cuda'] else None
except Exception as e:
    r['torch_error'] = str(e)[:200]
print(json.dumps(r))
`;

function cmdDoctor(reg, opts) {
  const row = (k, v) => say(`  ${k.padEnd(14)} ${v}`);
  say('System');
  row('platform', `${process.platform} ${os.arch()} (${os.release()})`);
  row('node', process.version);
  const smi = spawnSync('nvidia-smi', ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], { encoding: 'utf8', windowsHide: true });
  row('nvidia-smi', smi.status === 0 ? smi.stdout.trim().replace(/\n/g, '; ') : 'not found (no NVIDIA GPU or driver)');
  let py;
  try {
    py = resolvePython(opts);
    row('python', `${py.executable} (${py.version}, ${py.source})`);
  } catch (err) {
    row('python', err.message);
  }
  if (py) {
    const p = pyJson(py, PROBE);
    say('Python packages');
    if (p.error) row('probe', p.error);
    else {
      row('torch', p.torch ? `${p.torch_version} (CUDA build ${p.torch_cuda || 'none'})` : 'missing');
      row('cuda', p.cuda ? `yes - ${p.gpu}` : 'no');
      for (const m of ['gsplat', 'cv2', 'scipy', 'fused_ssim']) row(m, p[m] ? 'installed' : 'missing');
    }
  }
  say('Colab');
  if (IS_WINDOWS) row('colab CLI', 'not supported on Windows (use WSL or the notebook)');
  else {
    const c = spawnSync(remote.colabBin(), ['--help'], { encoding: 'utf8' });
    row('colab CLI', c.error ? 'not installed (uv tool install google-colab-cli --with keyrings.alt)' : 'installed');
  }
  row('scripts', scriptsDir());
  say(`\nBackends: ${[...reg.backends.keys()].join(', ')}. Next: fflab setup, then fflab train (GPU) or fflab colab train.`);
  return 0;
}

// ------------------------------------------------------------------------------------------ setup
async function cmdSetup(reg, argv) {
  const { backend, opts } = parseFor(reg, argv, 'setup');
  const { chosen, notes } = resolveAccelerators(reg, backend, opts.accel || []);
  notes.forEach((n) => say(`note: ${n}`));
  const venv = path.resolve(opts.venv || DEFAULT_VENV);
  const base = opts.python ? resolvePython({ python: opts.python }) : findSystemPython();
  if (!base) throw new Error('no Python 3.9+ found; install Python first');
  const vpy = venvPython(venv);
  const steps = [];
  if (!fs.existsSync(vpy)) steps.push([base.cmd, [...base.args, '-m', 'venv', venv], `create venv ${venv}`]);
  steps.push([vpy, ['-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'], 'upgrade pip']);
  const hasTorch = fs.existsSync(vpy) && pyJson({ cmd: vpy, args: [] }, 'import importlib.util,json;print(json.dumps(importlib.util.find_spec("torch") is not None))') === true;
  if (!hasTorch) {
    const url = opts['torch-index-url'];
    if (!url && IS_WINDOWS) say('warning: the PyPI torch wheel for Windows is CPU-only; pass --torch-index-url https://download.pytorch.org/whl/cu124 (match your CUDA driver)');
    steps.push([vpy, pipInstallArgs(['torch'], { indexUrl: url }), `install PyTorch${url ? ` from ${url}` : ''}`]);
  }
  if (backend.requirements?.length) steps.push([vpy, pipInstallArgs(backend.requirements), 'backend requirements']);
  if (backend.buildRequirements?.length) {
    steps.push([vpy, pipInstallArgs(backend.buildRequirements, { noBuildIsolation: true }), 'backend CUDA packages (compiled)']);
  }
  for (const a of chosen) {
    if (a.requirements?.length) steps.push([vpy, pipInstallArgs(a.requirements), `accelerator ${a.id}`]);
    if (a.buildRequirements?.length) steps.push([vpy, pipInstallArgs(a.buildRequirements, { noBuildIsolation: true }), `accelerator ${a.id} (compiled)`]);
  }
  if (IS_WINDOWS && backend.requiresCuda) say('note: compiling CUDA packages on Windows needs the CUDA toolkit and Visual Studio Build Tools (C++).');
  for (const [cmd, args, desc] of steps) {
    say(`\n> ${desc}\n  ${cmd} ${args.join(' ')}`);
    if (opts['dry-run']) continue;
    const code = await runStreaming(cmd, args);
    if (code !== 0) throw new Error(`setup step failed (${desc}), exit ${code}`);
  }
  say(opts['dry-run'] ? '\n(dry run: nothing installed)' : `\nDone. fflab uses ${venv} automatically. Check with: fflab doctor`);
  return 0;
}

// ---------------------------------------------------------------------------- train/render/check
async function cmdRunLocal(reg, command, argv) {
  const { backend, opts } = parseFor(reg, argv, command);
  const { chosen, notes } = resolveAccelerators(reg, backend, opts.accel || []);
  notes.forEach((n) => say(`note: ${n}`));
  const plan = buildPlan({ backend, command, opts, accels: chosen });
  const script = scriptPath(plan.script);
  const py = resolvePython(opts);
  printPlan(plan, py);
  if (opts['dry-run']) {
    say('env');
    printEnv(plan.env);
    say(`command\n  ${py.cmd} ${[...py.args, script].join(' ')}`);
    return 0;
  }
  if (plan.requiresCuda) {
    const p = pyJson(py, PROBE);
    if (!p.cuda) {
      throw new Error(`${backend.id} ${command} needs a CUDA GPU and none is usable from ${py.executable}`
        + `${p.error ? ` (${p.error})` : ''}. Run "fflab doctor", or run it on Colab: fflab colab ${command} ...`);
    }
  }
  say('');
  const code = await runStreaming(py.cmd, [...py.args, script], { env: plan.env, logFile: plan.logFile });
  say(code === 0 ? `\nfinished. outputs: ${plan.outputs}` : `\nfailed with exit code ${code}. log: ${plan.logFile || 'terminal'}`);
  return code;
}

// ------------------------------------------------------------------------------------------ colab
function saveJob(entry) {
  fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true });
  const all = fs.existsSync(STATE_FILE) ? JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')) : [];
  all.push({ ...entry, time: new Date().toISOString() });
  fs.writeFileSync(STATE_FILE, JSON.stringify(all.slice(-50), null, 2));
}

async function downloadOutputs(session, plan, localDir, withCkpt) {
  const files = (await remote.listRemote(session, plan.outputs))
    .filter((f) => withCkpt || !/(ckpt\.pt|_partial\.ply)(\.tmp)?$/.test(f.path));
  const isFile = files.length === 1 && files[0].path === plan.outputs;
  for (const f of files) {
    const rel = isFile ? path.posix.basename(f.path) : path.posix.relative(plan.outputs, f.path);
    const dst = path.join(localDir, ...rel.split('/'));
    say(`  download ${rel} (${(f.size / 1e6).toFixed(1)} MB)`);
    remote.download(session, f.path, dst);
  }
  return files.length;
}

async function cmdColab(reg, sub, argv) {
  if (sub === 'stop') {
    const { opts } = parseFor(reg, argv, 'colab');
    remote.checkColabAvailable();
    const r = remote.colab(['stop', '-s', opts.session || 'fflab']);
    say((r.out || r.err).trim());
    return r.code;
  }
  if (sub === 'logs') {
    const opts = parseArgs(argv).opts;
    const all = fs.existsSync(STATE_FILE) ? JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')) : [];
    const entry = opts.run ? all.reverse().find((j) => j.run === opts.run) : all.at(-1);
    if (!entry) throw new Error('no Colab job recorded on this machine');
    say(`following ${entry.run} (${entry.command}) on session ${entry.session}`);
    remote.checkColabAvailable();
    const code = await remote.followJob(entry.session, entry.job, { intervalMs: Number(opts.poll || 15) * 1000 });
    say(`exit code ${code}; outputs on the VM: ${entry.outputs}`);
    return code;
  }
  if (!RUN_COMMANDS.includes(sub)) throw new Error('usage: fflab colab train|render|check|logs|stop [options]');

  const { backend, opts } = parseFor(reg, argv, sub);
  const { chosen, notes } = resolveAccelerators(reg, backend, opts.accel || []);
  notes.forEach((n) => say(`note: ${n}`));
  const root = opts['remote-root'] || '/content/fflab';
  const plan = buildPlan({ backend, command: sub, opts, accels: chosen, mode: 'colab', remoteRoot: root });
  const session = opts.session || 'fflab';
  const gpu = opts.gpu || (plan.requiresCuda ? 'T4' : 'none');
  const scripts = [...new Set(Object.values(backend.commands).map((c) => c.script))];
  printPlan(plan);
  say(`${'colab'.padEnd(13)}session ${session}, ${gpu === 'none' ? 'CPU' : gpu}, root ${root}`);
  if (opts['dry-run']) {
    say('env');
    printEnv(plan.env);
    say(`would upload ${scripts.join(', ')} to ${root}/scripts and run ${plan.script} detached`);
    return 0;
  }
  remote.checkColabAvailable();
  say(`\n> session: ${remote.ensureSession(session, gpu)}`);
  if (opts.drive) {
    say('> mounting Google Drive: open the link it prints, approve, then press Enter');
    remote.colab(['drivemount', '-s', session], { inherit: true, timeout: 1800000 });
  }
  say('> uploading scripts');
  await remote.mkdirs(session, [`${root}/scripts`, `${root}/jobs`]);
  for (const s of scripts) remote.upload(session, scriptPath(s), `${root}/scripts/${s}`);
  const poll = Number(opts.poll || 15) * 1000;
  if (!opts['skip-setup']) {
    say('> checking packages on the VM (first GPU run installs gsplat; its CUDA kernels then compile once, ~15 min)');
    const setup = await remote.launchJob(session, { name: `${plan.run}-setup`, root, argv: remote.setupArgv(backend, chosen, sub), env: {} });
    const code = await remote.followJob(session, setup, { intervalMs: Math.min(poll, 5000), onLine: (l) => say(`  ${l}`) });
    if (code !== 0) throw new Error(`setup on the VM failed (exit ${code})`);
  }
  say(`> starting ${sub} (detached; Ctrl+C stops watching, not the job)`);
  const job = await remote.launchJob(session, {
    name: `${plan.run}-${sub}`, root, argv: ['{python}', `${root}/scripts/${plan.script}`], env: plan.env, log: plan.logFile,
  });
  saveJob({ run: plan.run, command: sub, session, job, outputs: plan.outputs });
  const code = await remote.followJob(session, job, { intervalMs: poll });
  if (code !== 0) {
    say(`\n${sub} failed on the VM (exit ${code}). Full log: ${job.log}`);
    return code;
  }
  if (opts.download !== false && plan.outputs) {
    const localDir = path.resolve(opts['download-dir'] || path.join('fflab-runs', plan.run));
    say(`\n> downloading outputs to ${localDir}${opts['with-ckpt'] ? '' : ' (checkpoints and partial PLYs skipped; --with-ckpt to include)'}`);
    const n = await downloadOutputs(session, plan, localDir, opts['with-ckpt']);
    say(`  ${n} file(s)`);
  }
  say(`\ndone. The VM keeps running: fflab colab stop --session ${session}`);
  return 0;
}

// ------------------------------------------------------------------------------------------- main
export async function main(argv) {
  const [command, ...rest] = argv;
  if (!command || ['-h', '--help', 'help'].includes(command)) {
    const reg = loadRegistry();
    return cmdHelp(reg, rest[0]);
  }
  if (['-v', '--version', 'version'].includes(command)) {
    say(VERSION);
    return 0;
  }
  const reg = loadRegistry();
  switch (command) {
    case 'list': return cmdList(reg);
    case 'doctor': return cmdDoctor(reg, parseArgs(rest).opts);
    case 'setup': return cmdSetup(reg, rest);
    case 'train':
    case 'render':
    case 'check': return cmdRunLocal(reg, command, rest);
    case 'colab': return cmdColab(reg, rest[0], rest.slice(1));
    default: throw new Error(`unknown command "${command}". Run: fflab help`);
  }
}

export { GLOBAL_FLAGS };
