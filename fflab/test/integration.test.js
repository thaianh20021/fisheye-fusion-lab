// End-to-end runs of the CLI binary. The Colab flow uses test/fixtures/fake-colab, which treats
// "remote" paths as local ones, so launch, polling, exit codes and downloads are all exercised.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const bin = path.join(here, '..', 'bin', 'fflab.js');
const repo = path.resolve(here, '..', '..');
const fixtures = path.join(here, 'fixtures');

function fflab(args, env = {}, cwd) {
  return spawnSync(process.execPath, [bin, ...args], { encoding: 'utf8', env: { ...process.env, ...env }, cwd, timeout: 180000 });
}

const hasPy = (mods) => spawnSync('python3', ['-c', mods.map((m) => `import ${m}`).join(';')]).status === 0;

test('fflab colab train: setup, detached job, polling, downloads without checkpoints', { skip: process.platform === 'win32' }, () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fflab-it-'));
  const env = {
    FFLAB_COLAB_BIN: path.join(fixtures, 'fake-colab'),
    FAKE_COLAB_STATE: tmp,
    FFLAB_PLUGINS_DIR: path.join(fixtures, 'plugins'),
    FFLAB_SCRIPTS_DIR: path.join(fixtures, 'scripts'),
    HOME: tmp,
    USERPROFILE: tmp,
  };
  const root = path.join(tmp, 'content');
  const ok = fflab(['colab', 'train', '--backend', 'dummy', '--steps', '4', '--run-name', 'r1', '--remote-root', root,
    '--poll', '0.2', '--download-dir', path.join(tmp, 'local')], env, tmp);
  assert.equal(ok.status, 0, ok.stdout + ok.stderr);
  assert.match(ok.stdout, /session: created/);
  assert.match(ok.stdout, /missing modules: none/);
  assert.match(ok.stdout, /step 4\/4 fixed=1/);
  assert.ok(fs.existsSync(path.join(tmp, 'local', 'result.json')));
  assert.ok(fs.existsSync(path.join(tmp, 'local', 'sub', 'fly.mp4')));
  assert.ok(!fs.existsSync(path.join(tmp, 'local', 'ckpt.pt')), 'checkpoints are skipped by default');
  assert.ok(fs.existsSync(path.join(tmp, 'local', 'fflab-train.log')), 'the job log lands where the backend expects it');

  const bad = fflab(['colab', 'train', '--backend', 'dummy', '--fail', '--run-name', 'r2', '--remote-root', root,
    '--poll', '0.2', '--skip-setup'], env, tmp);
  assert.equal(bad.status, 3, bad.stdout + bad.stderr);
  assert.match(bad.stdout, /session: existing/);
  assert.match(bad.stdout, /simulated failure/);

  const logs = fflab(['colab', 'logs', '--run', 'r1', '--poll', '0.2'], env, tmp);
  assert.equal(logs.status, 0, logs.stdout + logs.stderr);
  assert.match(logs.stdout, /step 1\/4/);
});

test('fflab check runs the real projection check on a synthetic KITTI-360 archive (CPU)', { skip: !hasPy(['numpy', 'cv2']) }, () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fflab-check-'));
  const archive = path.join(tmp, 'archive');
  const build = spawnSync('python3', ['-c',
    `import sys; from pathlib import Path; sys.path.insert(0, ${JSON.stringify(path.join(repo, 'tests'))});`
    + `from test_projection_check import build_archive; build_archive(Path(${JSON.stringify(archive)}), [0, 1, 2, 5, 6])`],
  { encoding: 'utf8' });
  assert.equal(build.status, 0, build.stderr);
  const r = fflab(['check', '--data', archive, '--out', path.join(tmp, 'out'), '--run-name', 'c1'], {}, tmp);
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const report = JSON.parse(fs.readFileSync(path.join(tmp, 'out', 'projection_report.json'), 'utf8'));
  assert.equal(report.frame, 0);
  assert.ok(report.cameras.image_00.A_official.inside > 50);
});

test('fflab train refuses to start without a usable CUDA GPU and points to Colab', { skip: hasPy(['torch']) }, () => {
  const r = fflab(['train', '--frames', '2']);
  assert.equal(r.status, 1);
  assert.match(r.stderr, /needs a CUDA GPU.*fflab colab train/s);
});

test('fflab train puts the Python directory first on PATH (so a venv\'s ninja is found)', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fflab-path-'));
  const r = fflab(['train', '--backend', 'dummy', '--steps', '1', '--run-name', 'p1', '--out', path.join(tmp, 'out')],
    { FFLAB_PLUGINS_DIR: path.join(fixtures, 'plugins'), FFLAB_SCRIPTS_DIR: path.join(fixtures, 'scripts'), HOME: tmp, USERPROFILE: tmp }, tmp);
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const m = r.stdout.match(/PATHHEAD (\S+) PYDIR (\S+)/);
  assert.ok(m, r.stdout);
  assert.equal(fs.realpathSync(m[1]), fs.realpathSync(path.dirname(spawnSync('python3', ['-c', 'import sys;print(sys.executable)'], { encoding: 'utf8' }).stdout.trim())));
});
