import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseArgs } from '../src/args.js';
import { buildPlan, newRunName } from '../src/plan.js';
import { loadRegistry, getBackend, resolveAccelerators, validateBackend } from '../src/registry.js';
import { parsePoll, pollCode } from '../src/colab.js';

const reg = loadRegistry();
const backend = getBackend(reg, '3dgut-gsplat');

test('args: booleans never swallow the next token, lists split on commas', () => {
  const r = parseArgs(['--sky', 'pos', '--no-ssim', '--frames', '30', '--accel', 'a,b', '--accel=c', '--x=1'],
    { booleans: new Set(['sky', 'ssim']), lists: new Set(['accel']) });
  assert.deepEqual(r._, ['pos']);
  assert.equal(r.opts.sky, true);
  assert.equal(r.opts.ssim, false);
  assert.equal(r.opts.frames, '30');
  assert.deepEqual(r.opts.accel, ['a', 'b', 'c']);
  assert.equal(r.opts.x, '1');
  assert.throws(() => parseArgs(['--frames']), /needs a value/);
});

test('plan: every option is set explicitly, defaults included', () => {
  const plan = buildPlan({ backend, command: 'train', opts: { frames: '30', ssim: false }, run: 'r1' });
  assert.equal(plan.env.KITTI360_N_FRAMES, '30');
  assert.equal(plan.env.KITTI360_SSIM, '0');
  assert.equal(plan.env.KITTI360_SKY, '1');
  assert.equal(plan.env.KITTI360_MASK_ONLY, '0', 'fixed env guards against leaked kernel variables');
  assert.equal(plan.env.KITTI360_RESUME, '0');
  assert.equal(plan.env.KITTI360_RUN_NAME, 'r1');
  assert.match(plan.env.KITTI360_OUT, /fflab-runs[\\/]r1$/);
});

test('plan: colab mode maps relative paths under the remote root', () => {
  const plan = buildPlan({ backend, command: 'render', opts: {}, mode: 'colab', remoteRoot: '/content/x', run: 'r2' });
  assert.equal(plan.env.FLY_PLY, '/content/x/fflab-runs/r2/gaussians.ply');
  assert.equal(plan.outputs, '/content/x/fflab-runs/r2/fly.mp4');
  assert.equal(plan.env.KITTI360_LOG, '/content/x/fflab-runs/r2/fflab-render.log');
});

test('plan: accelerators override fixed env; builtins are notes; bad input is refused', () => {
  const { chosen, notes } = resolveAccelerators(reg, backend, ['fused-ssim', 'selective-adam']);
  assert.deepEqual(chosen.map((a) => a.id), ['fused-ssim']);
  assert.equal(notes.length, 1);
  const plan = buildPlan({ backend, command: 'train', opts: {}, accels: chosen, run: 'r3' });
  assert.equal(plan.env.KITTI360_SSIM_IMPL, 'fused');
  assert.throws(() => buildPlan({ backend, command: 'train', opts: { framez: '1' } }), /unknown option/);
  assert.throws(() => buildPlan({ backend, command: 'train', opts: { steps: 'ten' } }), /integer/);
  assert.throws(() => resolveAccelerators(reg, backend, ['nope']), /unknown accelerator/);
});

test('registry: manifests are validated', () => {
  assert.throws(() => validateBackend({ id: 'Bad Id', status: 'x', commands: {} }), /kebab/);
  assert.match(newRunName(new Date(2026, 10, 13, 4, 5, 6)), /^run_20261113_040506$/);
});

test('colab poll protocol: offsets, whole lines, exit only when drained', () => {
  const r = parsePoll('noise\n@@OFF 42\n@@L step 1\n@@L\n@@EXIT 0\n');
  assert.deepEqual(r, { offset: 42, exit: 0, lines: ['step 1', ''] });
  assert.match(pollCode({ log: '/a b/log', exit: '/a/exit' }, 10), /f\.seek\(off\)/);
  assert.ok(pollCode({ log: '/a b/log', exit: '/a/exit' }, 10).includes('log, exitf, off = "/a b/log", "/a/exit", 10'),
    'paths reach Python as plain string literals');
});
