// fflab convert / nurec / eval: KITTI-360 -> NCore -> Instant NuRec -> PSNR, on this machine.
// Same tools and layout as the Web UI (fisheye_fusion/pipeline.py): clips in outputs/ncore/<name>,
// reconstructions in outputs/reconstruction/<run>, evaluations in outputs/eval/<run>.
import fs from 'node:fs';
import path from 'node:path';
import { parseArgs } from './args.js';
import { PKG_ROOT } from './registry.js';
import { IS_WINDOWS, resolvePython, runStreaming } from './python.js';

const say = (...a) => console.log(...a);

export const KITTI_CAMERAS = {
  image_00: 'camera_front_left_00',
  image_01: 'camera_front_right_01',
  image_02: 'camera_fisheye_left_02',
  image_03: 'camera_fisheye_right_03',
};
export const FISHEYE_MODELS = ['rectify120', 'ftheta', 'opencv'];
// Context cameras in Instant NuRec's canonical order (front, cross-left, cross-right) for a KITTI-360 clip.
export const MODEL_CAMERAS = {
  'pa-multiview': ['camera_front_left_00', 'camera_fisheye_left_02', 'camera_fisheye_right_03'],
  'pa-front': ['camera_front_left_00'],
  'pq-front': [],
};
const CHECKPOINT_STEM = { 'pa-front': 'pa_front', 'pa-multiview': 'pa_multiview', 'pq-front': 'pq_road' };

export const NUREC_USAGE = `KITTI-360 -> NCore -> Instant NuRec (needs backends/instant-nurec with its .venv)

  fflab convert [--data ./kitti360] [--sequence 2013_05_28_drive_0003_sync] [--start 0] [--frames 40]
                [--cameras image_00,image_01,image_02,image_03] [--fisheye-model rectify120|ftheta|opencv]
                [--no-lidar] [--name NAME]
        writes outputs/ncore/NAME/NAME.json and a LiDAR projection check in outputs/ncore/NAME/check
  fflab nurec   --clip outputs/ncore/NAME/NAME.json [--model pa-multiview|pa-front|pq-front]
                [--cameras ID,ID,ID] [--gpu N] [--n-gaussians 2000000] [--max-chunks 8] [--name RUN]
        writes outputs/reconstruction/RUN/<...>/ply/<clip>/<clip>.ply (+ sky cubemap)
  fflab eval    --ply FILE.ply --clip CLIP.json [--every 4] [--gpu N] [--name RUN]
        renders the PLY with gsplat (fflab setup venv) at the clip's cameras: outputs/eval/RUN/metrics.json

  fflab carla   --ply FILE.ply --clip CLIP.json [--no-carla] [--npc 3] [--start 0] [--duration 10]
                [--lateral 0] [--lift 0] [--fov 90] [--gpu N] [--port 2000] [--name RUN]
        camera along the clip's recorded path through the reconstruction (+ NuRec sky); with a CARLA
        server on localhost, CARLA vehicles are composited in by depth: outputs/carla/RUN/composite.mp4
  fflab carla-server [--adapter 0] [--port 2000]
        run CARLA 0.9.16 offscreen in the foreground (CARLA_ROOT, default ~/carla); --adapter is the
        Vulkan adapter index, which may differ from nvidia-smi's GPU number (the Web UI finds it)

  --workspace DIR   repo root holding tools/, backends/, outputs/ (default: the repo fflab lives in)
  --dry-run         print the commands, run nothing`;

export function workspaceOf(opts) {
  return path.resolve(opts.workspace || process.env.FFLAB_WORKSPACE || path.join(PKG_ROOT, '..'));
}

// A relative path is taken from the current directory if it exists there, else from the workspace.
function userPath(ws, p) {
  const here = path.resolve(p);
  return fs.existsSync(here) ? here : path.resolve(ws, p);
}

function nurecRepo(ws) {
  return path.join(ws, 'backends', 'instant-nurec');
}

export function nurecPython(ws) {
  return path.join(nurecRepo(ws), '.venv', IS_WINDOWS ? 'Scripts' : 'bin', IS_WINDOWS ? 'python.exe' : 'python');
}

function int(opts, key, def, lo, hi) {
  const raw = opts[key] ?? def;
  const v = Number(raw);
  if (!Number.isInteger(v) || v < lo || v > hi) throw new Error(`--${key} must be an integer in ${lo}..${hi}, got "${raw}"`);
  return v;
}

function gpuEnv(opts) {
  if (opts.gpu === undefined) return {};
  if (!/^\d+(,\d+)*$/.test(String(opts.gpu))) throw new Error(`--gpu expects a device index, got "${opts.gpu}"`);
  return { CUDA_VISIBLE_DEVICES: String(opts.gpu) };
}

function stamp() {
  return new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
}

function checkName(name) {
  if (!/^[A-Za-z0-9_.-]{1,80}$/.test(name)) throw new Error(`name "${name}": letters, digits, _ . - only`);
  return name;
}

function newOutput(ws, kind, name, dryRun) {
  const out = path.join(ws, 'outputs', kind, checkName(name));
  if (fs.existsSync(out)) throw new Error(`output already exists: ${path.relative(ws, out)} (pick another --name)`);
  if (!dryRun) fs.mkdirSync(out, { recursive: true });
  return out;
}

export function findCheckpoint(ws, model) {
  const dir = path.join(nurecRepo(ws), 'checkpoints', 'pth');
  const stem = CHECKPOINT_STEM[model];
  const hits = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.startsWith(`instant_nurec_${stem}_`) && f.endsWith('.pth')).sort() : [];
  if (!hits.length) throw new Error(`no ${model} checkpoint in ${path.relative(ws, dir) || dir} (instant_nurec_${stem}_*.pth)`);
  return path.join(dir, hits.at(-1));
}

// ------------------------------------------------------------------ plans (pure: no side effects but mkdir)
export function convertPlan(opts, ws, { dryRun = false } = {}) {
  const sequence = opts.sequence || '2013_05_28_drive_0003_sync';
  if (!/^2013_05_28_drive_\d{4}_sync$/.test(sequence)) throw new Error(`--sequence must look like 2013_05_28_drive_0003_sync, got "${sequence}"`);
  const start = int(opts, 'start', 0, 0, 1e9);
  const frames = int(opts, 'frames', 40, 2, 2000);
  const cameras = opts.cameras || Object.keys(KITTI_CAMERAS);
  const bad = cameras.filter((c) => !KITTI_CAMERAS[c]);
  if (bad.length || !cameras.length) throw new Error(`--cameras: choose from ${Object.keys(KITTI_CAMERAS).join(', ')}`);
  const fisheye = opts['fisheye-model'] || 'rectify120';
  if (!FISHEYE_MODELS.includes(fisheye)) throw new Error(`--fisheye-model: one of ${FISHEYE_MODELS.join(', ')}`);
  const lidar = opts.lidar !== false;
  const name = opts.name || `kitti360_${sequence.split('_drive_')[1].slice(0, 4)}_${String(start).padStart(6, '0')}_${frames}f`;
  const out = newOutput(ws, 'ncore', name, dryRun);
  const py = nurecPython(ws);
  const tools = path.join(ws, 'tools');
  const clip = path.join(out, `${name}.json`);
  const steps = [{
    label: 'convert', cmd: py,
    args: [path.join(tools, 'kitti360_to_ncore.py'), '--data', userPath(ws, opts.data || 'kitti360'), '--out', out,
      '--sequence', sequence, '--start', String(start), '--frames', String(frames), '--cameras', cameras.join(','),
      '--sequence-id', name, '--fisheye-model', fisheye, ...(lidar ? [] : ['--no-lidar'])],
  }];
  if (lidar) {
    steps.push({ label: 'LiDAR projection check', cmd: py, args: [path.join(tools, 'ncore_projection_check.py'), clip, '--out', path.join(out, 'check'), '--frame', '0'] });
  }
  return { steps, outputs: out, clip, cwd: ws, env: {} };
}

export function nurecPlan(opts, ws, { dryRun = false } = {}) {
  const model = opts.model || 'pa-multiview';
  if (!MODEL_CAMERAS[model]) throw new Error(`--model: one of ${Object.keys(MODEL_CAMERAS).join(', ')}`);
  if (!opts.clip) throw new Error('--clip CLIP.json is required (make one with: fflab convert)');
  const clip = userPath(ws, opts.clip);
  if (!dryRun && !fs.existsSync(clip)) throw new Error(`clip not found: ${clip}`);
  const cameras = opts.cameras || MODEL_CAMERAS[model];
  const allowed = { 'pa-front': [1], 'pa-multiview': [1, 3, 5], 'pq-front': [0] }[model];
  if (!allowed.includes(cameras.length)) throw new Error(`${model} takes ${allowed.join(' or ')} context camera(s), got ${cameras.length}`);
  const checkpoint = findCheckpoint(ws, model);
  const out = newOutput(ws, 'reconstruction', opts.name || `cli-${stamp()}`, dryRun);
  const args = [path.join(nurecRepo(ws), 'run_inference.py'), '--model', model, '--ncore-path', clip, '--output-dir', out, '--merge',
    '--n-gaussians', String(int(opts, 'n-gaussians', 2000000, 100000, 2000000)),
    '--max-chunks', String(int(opts, 'max-chunks', 8, 1, 8))];
  for (const c of cameras) args.push('--camera-id', c);
  return {
    steps: [{ label: `Instant NuRec ${model}`, cmd: nurecPython(ws), args }],
    outputs: out, cwd: nurecRepo(ws),
    env: { INSTANT_NUREC_FULL_PT: checkpoint, HF_HUB_OFFLINE: '1', ...gpuEnv(opts) },
  };
}

export function evalPlan(opts, ws, gsplatPython, { dryRun = false } = {}) {
  if (!opts.ply || !opts.clip) throw new Error('--ply FILE.ply and --clip CLIP.json are required');
  const ply = userPath(ws, opts.ply);
  const clip = userPath(ws, opts.clip);
  for (const f of dryRun ? [] : [ply, clip]) if (!fs.existsSync(f)) throw new Error(`not found: ${f}`);
  const every = int(opts, 'every', 4, 1, 100);
  const out = newOutput(ws, 'eval', opts.name || `cli-${stamp()}`, dryRun);
  if (!dryRun) fs.writeFileSync(path.join(out, 'job.json'), JSON.stringify({ ply: path.relative(ws, ply), clip: path.relative(ws, clip) }));
  const tools = path.join(ws, 'tools');
  const views = path.join(out, 'views.npz');
  return {
    steps: [
      { label: 'export views', cmd: nurecPython(ws), args: [path.join(tools, 'ncore_eval_export.py'), clip, '--out', views, '--every', String(every)] },
      { label: 'render + PSNR', cmd: gsplatPython, args: [path.join(tools, 'ncore_eval_render.py'), ply, views, '--out', out] },
    ],
    outputs: out, cwd: ws,
    // gsplat JIT-compiles with ninja from the venv; keep that venv's bin on PATH
    env: { ...gpuEnv(opts), PATH: `${path.dirname(gsplatPython)}${path.delimiter}${process.env.PATH || ''}` },
  };
}

export function carlaPlan(opts, ws, gsplatPython, { dryRun = false } = {}) {
  if (!opts.ply || !opts.clip) throw new Error('--ply FILE.ply and --clip CLIP.json are required');
  const ply = userPath(ws, opts.ply);
  const clip = userPath(ws, opts.clip);
  for (const f of dryRun ? [] : [ply, clip]) if (!fs.existsSync(f)) throw new Error(`not found: ${f}`);
  const num = (key, def, lo, hi) => {
    const v = Number(opts[key] ?? def);
    if (!Number.isFinite(v) || v < lo || v > hi) throw new Error(`--${key} must be a number in ${lo}..${hi}, got "${opts[key]}"`);
    return String(v);
  };
  const useCarla = opts.carla !== false;
  const out = newOutput(ws, 'carla', opts.name || `cli-${stamp()}`, dryRun);
  const traj = path.join(out, 'traj.npz');
  const info = { ply: path.relative(ws, ply), clip: path.relative(ws, clip), use_carla: useCarla };
  if (!dryRun) fs.writeFileSync(path.join(out, 'job.json'), JSON.stringify(info));
  const bridge = [path.join(ws, 'tools', 'carla_nurec_bridge.py'), ply, traj, '--out', out,
    '--start', num('start', 0, 0, 3600), '--duration', num('duration', 10, 0, 3600),
    '--lateral', num('lateral', 0, -10, 10), '--lift', num('lift', 0, -1, 20), '--fov', num('fov', 90, 30, 150)];
  if (useCarla) bridge.push('--carla', `127.0.0.1:${int(opts, 'port', 2000, 1, 65535)}`, '--npc', String(int(opts, 'npc', 3, 0, 8)));
  // a 3DGUT run (fflab train) keeps config.json with its frames next to the PLY: align its KITTI-360 world to the clip
  const cfg = path.join(path.dirname(ply), 'config.json');
  if (fs.existsSync(cfg) && JSON.parse(fs.readFileSync(cfg, 'utf8')).frame_ids) {
    bridge.push('--trainer-config', cfg, '--kitti-data', userPath(ws, opts.data || 'kitti360'));
  }
  return {
    steps: [
      { label: 'export trajectory', cmd: nurecPython(ws), args: [path.join(ws, 'tools', 'ncore_trajectory_export.py'), clip, '--out', traj] },
      { label: useCarla ? 'render with CARLA' : 'render', cmd: gsplatPython, args: bridge },
    ],
    outputs: out, cwd: ws,
    env: { ...gpuEnv(opts), PATH: `${path.dirname(gsplatPython)}${path.delimiter}${process.env.PATH || ''}` },
  };
}

// ------------------------------------------------------------------ runner
async function runPlan(plan, dryRun) {
  for (const s of plan.steps) {
    say(`\n> ${s.label}\n  ${s.cmd} ${s.args.join(' ')}`);
    if (dryRun) continue;
    const code = await runStreaming(s.cmd, s.args, { env: plan.env, cwd: plan.cwd });
    if (code !== 0) {
      say(`\n${s.label} failed (exit ${code}). Output folder: ${plan.outputs}`);
      return code;
    }
  }
  if (dryRun) {
    say(`\nenv ${JSON.stringify(plan.env)}\ncwd ${plan.cwd}\n(dry run: nothing executed)`);
    return 0;
  }
  say(`\ndone: ${plan.outputs}`);
  return 0;
}

function requireNurec(ws) {
  if (!fs.existsSync(nurecPython(ws))) {
    throw new Error(`Instant NuRec venv not found at ${nurecPython(ws)}. See docs/reconstruction.md (clone NVIDIA/instant-nurec into backends/instant-nurec and install it).`);
  }
}

export async function cmdPipeline(command, argv) {
  const { opts } = parseArgs(argv, { booleans: new Set(['lidar', 'dry-run', 'carla']), lists: new Set(['cameras']) });
  const ws = workspaceOf(opts);
  const dryRun = Boolean(opts['dry-run']);
  if (command === 'carla-server') {
    const script = path.join(ws, 'tools', 'carla_server.sh');
    const args = [script, String(int(opts, 'adapter', 0, 0, 63)), String(int(opts, 'port', 2000, 1, 65535))];
    say(`> bash ${args.join(' ')}   (CARLA_ROOT=${process.env.CARLA_ROOT || '~/carla'}; Ctrl+C stops it)`);
    return dryRun ? 0 : runStreaming('bash', args, {});
  }
  if (!dryRun) requireNurec(ws);
  if (command === 'convert') {
    const plan = convertPlan(opts, ws, { dryRun });
    const code = await runPlan(plan, dryRun);
    if (code === 0 && !dryRun) say(`clip: ${path.relative(ws, plan.clip)}\nnext: fflab nurec --clip ${path.relative(ws, plan.clip)} --gpu N`);
    return code;
  }
  if (command === 'nurec') {
    const plan = nurecPlan(opts, ws, { dryRun });
    const code = await runPlan(plan, dryRun);
    if (code === 0 && !dryRun) {
      const ply = findFiles(plan.outputs, '.ply');
      ply.forEach((p) => say(`PLY: ${path.relative(ws, p)}`));
      if (ply.length && opts.clip) say(`next: fflab eval --ply ${path.relative(ws, ply[0])} --clip ${opts.clip} --gpu N`);
    }
    return code;
  }
  if (command === 'eval') {
    const py = resolvePython(opts);
    const plan = evalPlan(opts, ws, py.executable, { dryRun });
    const code = await runPlan(plan, dryRun);
    const metrics = path.join(plan.outputs, 'metrics.json');
    if (code === 0 && fs.existsSync(metrics)) {
      const m = JSON.parse(fs.readFileSync(metrics, 'utf8'));
      say('\ncamera                      PSNR (dB)  views');
      for (const [c, v] of Object.entries(m)) say(`${c.padEnd(28)}${v.psnr_mean.toFixed(2).padStart(9)}  ${v.views}`);
    }
    return code;
  }
  if (command === 'carla') {
    const py = resolvePython(opts);
    const plan = carlaPlan(opts, ws, py.executable, { dryRun });
    const code = await runPlan(plan, dryRun);
    if (code === 0 && !dryRun) findFiles(plan.outputs, '.mp4').forEach((v) => say(`video: ${path.relative(ws, v)}`));
    return code;
  }
  throw new Error(`unknown pipeline command ${command}`);
}

function findFiles(dir, ext) {
  const out = [];
  const walk = (d) => {
    for (const e of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.name.endsWith(ext)) out.push(p);
    }
  };
  if (fs.existsSync(dir)) walk(dir);
  return out;
}
