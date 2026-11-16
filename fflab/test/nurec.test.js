import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { carlaPlan, convertPlan, evalPlan, nurecPlan, findCheckpoint, MODEL_CAMERAS } from '../src/nurec.js';

function workspace() {
  const ws = fs.mkdtempSync(path.join(os.tmpdir(), 'fflab-nurec-'));
  const pth = path.join(ws, 'backends', 'instant-nurec', 'checkpoints', 'pth');
  fs.mkdirSync(pth, { recursive: true });
  fs.writeFileSync(path.join(pth, 'instant_nurec_pa_multiview_1.0.0.pth'), '');
  fs.writeFileSync(path.join(pth, 'instant_nurec_pa_multiview_1.1.0.pth'), '');
  fs.mkdirSync(path.join(ws, 'outputs', 'ncore', 'c'), { recursive: true });
  fs.writeFileSync(path.join(ws, 'outputs', 'ncore', 'c', 'c.json'), '{}');
  return ws;
}
const flag = (args, name) => args[args.indexOf(name) + 1];

test('convert: defaults, rectify120, projection check, output folder created', () => {
  const ws = workspace();
  const plan = convertPlan({}, ws);
  const [conv, check] = plan.steps;
  assert.equal(path.basename(conv.args[0]), 'kitti360_to_ncore.py');
  assert.equal(flag(conv.args, '--fisheye-model'), 'rectify120');
  assert.equal(flag(conv.args, '--sequence-id'), 'kitti360_0003_000000_40f');
  assert.equal(flag(conv.args, '--cameras'), 'image_00,image_01,image_02,image_03');
  assert.equal(path.basename(check.args[0]), 'ncore_projection_check.py');
  assert.ok(fs.existsSync(plan.outputs));
  assert.throws(() => convertPlan({}, ws), /already exists/);
});

test('convert: options and validation', () => {
  const ws = workspace();
  const plan = convertPlan({ frames: '150', start: '300', cameras: ['image_00'], 'fisheye-model': 'opencv', lidar: false, name: 'x' }, ws, { dryRun: true });
  assert.equal(plan.steps.length, 1);
  assert.ok(plan.steps[0].args.includes('--no-lidar'));
  assert.equal(flag(plan.steps[0].args, '--frames'), '150');
  assert.ok(!fs.existsSync(plan.outputs), 'dry run creates nothing');
  for (const bad of [{ frames: '1' }, { frames: 'x' }, { cameras: ['image_09'] }, { 'fisheye-model': 'pinhole' },
    { sequence: 'drive3' }, { name: '../x' }]) {
    assert.throws(() => convertPlan(bad, ws, { dryRun: true }), undefined, JSON.stringify(bad));
  }
});

test('nurec: multiview cameras in canonical order, newest checkpoint, GPU env', () => {
  const ws = workspace();
  const plan = nurecPlan({ clip: 'outputs/ncore/c/c.json', gpu: '1', name: 'r' }, ws);
  const args = plan.steps[0].args;
  assert.equal(path.basename(args[0]), 'run_inference.py');
  assert.equal(flag(args, '--model'), 'pa-multiview');
  assert.deepEqual(args.filter((_, i) => args[i - 1] === '--camera-id'), MODEL_CAMERAS['pa-multiview']);
  assert.ok(args.includes('--merge'));
  assert.equal(path.basename(plan.env.INSTANT_NUREC_FULL_PT), 'instant_nurec_pa_multiview_1.1.0.pth');
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, '1');
  assert.equal(plan.env.HF_HUB_OFFLINE, '1');
  assert.equal(plan.cwd, path.join(ws, 'backends', 'instant-nurec'));
});

test('nurec: validation', () => {
  const ws = workspace();
  assert.throws(() => nurecPlan({}, ws, { dryRun: true }), /--clip/);
  assert.throws(() => nurecPlan({ clip: 'missing.json' }, ws), /not found/);
  assert.throws(() => nurecPlan({ clip: 'outputs/ncore/c/c.json', cameras: ['a', 'b'] }, ws, { dryRun: true }), /1 or 3 or 5/);
  assert.throws(() => nurecPlan({ clip: 'outputs/ncore/c/c.json', model: 'pa-front' }, ws, { dryRun: true }), /no pa-front checkpoint/);
  assert.throws(() => nurecPlan({ clip: 'outputs/ncore/c/c.json', gpu: 'all' }, ws, { dryRun: true }), /--gpu/);
  assert.throws(() => nurecPlan({ clip: 'outputs/ncore/c/c.json', 'max-chunks': '9' }, ws, { dryRun: true }), /max-chunks/);
  assert.throws(() => findCheckpoint(ws, 'pq-front'), /pq_road/);
});

test('eval: export with the NuRec venv, render with the gsplat venv, job.json recorded', () => {
  const ws = workspace();
  fs.writeFileSync(path.join(ws, 'scene.ply'), '');
  const plan = evalPlan({ ply: 'scene.ply', clip: 'outputs/ncore/c/c.json', every: '2', gpu: '0', name: 'e' }, ws, '/venv/bin/python');
  const [exp, render] = plan.steps;
  assert.match(exp.cmd, /instant-nurec/);
  assert.equal(flag(exp.args, '--every'), '2');
  assert.equal(render.cmd, '/venv/bin/python');
  assert.ok(plan.env.PATH.startsWith('/venv/bin'));
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(plan.outputs, 'job.json'), 'utf8')), { ply: 'scene.ply', clip: 'outputs/ncore/c/c.json' });
});

test('fflab help nurec and a dry run through the real CLI', () => {
  const bin = path.join(import.meta.dirname, '..', 'bin', 'fflab.js');
  const help = spawnSync(process.execPath, [bin, 'help', 'nurec'], { encoding: 'utf8' });
  assert.match(help.stdout, /fflab nurec\s+--clip/);
  const ws = workspace();
  const dry = spawnSync(process.execPath, [bin, 'nurec', '--clip', 'outputs/ncore/c/c.json', '--workspace', ws, '--dry-run'], { encoding: 'utf8' });
  assert.equal(dry.status, 0, dry.stderr);
  assert.match(dry.stdout, /--camera-id camera_fisheye_right_03/);
  assert.match(dry.stdout, /dry run/);
});

test('carla: trajectory export then bridge; CARLA by default, --no-carla renders the scene only', () => {
  const ws = workspace();
  fs.writeFileSync(path.join(ws, 'scene.ply'), '');
  const plan = carlaPlan({ ply: 'scene.ply', clip: 'outputs/ncore/c/c.json', npc: '4', lateral: '-1.5', gpu: '1', name: 'v' }, ws, '/venv/bin/python');
  const [traj, bridge] = plan.steps;
  assert.equal(path.basename(traj.args[0]), 'ncore_trajectory_export.py');
  assert.match(traj.cmd, /instant-nurec/);
  assert.equal(bridge.cmd, '/venv/bin/python');
  assert.equal(path.basename(bridge.args[0]), 'carla_nurec_bridge.py');
  assert.equal(flag(bridge.args, '--carla'), '127.0.0.1:2000');
  assert.equal(flag(bridge.args, '--npc'), '4');
  assert.equal(flag(bridge.args, '--lateral'), '-1.5');
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, '1');
  assert.ok(!bridge.args.includes('--trainer-config'));
  const solo = carlaPlan({ ply: 'scene.ply', clip: 'outputs/ncore/c/c.json', carla: false, name: 'v2' }, ws, '/venv/bin/python');
  assert.ok(!solo.steps[1].args.includes('--carla'));
  assert.throws(() => carlaPlan({ ply: 'scene.ply', clip: 'outputs/ncore/c/c.json', fov: '400', name: 'v3' }, ws, '/p', { dryRun: true }), /--fov/);
});

test('carla: a 3DGUT run (config.json with frame_ids next to the PLY) gets aligned to the clip', () => {
  const ws = workspace();
  fs.mkdirSync(path.join(ws, 'fflab-runs', 'r'), { recursive: true });
  fs.writeFileSync(path.join(ws, 'fflab-runs', 'r', 'gaussians.ply'), '');
  fs.writeFileSync(path.join(ws, 'fflab-runs', 'r', 'config.json'), JSON.stringify({ frame_ids: [167, 172] }));
  const plan = carlaPlan({ ply: 'fflab-runs/r/gaussians.ply', clip: 'outputs/ncore/c/c.json', data: path.join(ws, 'kitti360'), name: 'g' }, ws, '/p');
  assert.equal(flag(plan.steps[1].args, '--trainer-config'), path.join(ws, 'fflab-runs', 'r', 'config.json'));
  assert.equal(flag(plan.steps[1].args, '--kitti-data'), path.join(ws, 'kitti360'));
});
