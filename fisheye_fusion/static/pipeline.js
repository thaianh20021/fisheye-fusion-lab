// Fisheye Fusion Lab: KITTI-360 -> NCore -> Instant NuRec -> evaluation.
// Talks to /api/setup, /api/pipeline/*, /api/reconstruction/* and /api/outputs/* (fisheye_fusion/server.py).

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const url = (path) => path.split('/').map(encodeURIComponent).join('/');

const NUREC_REPO = 'backends/instant-nurec';
const NUREC_PYTHON = `${NUREC_REPO}/.venv/bin/python`;
// Context-camera slots per model, in Instant NuRec's canonical order, and the KITTI-360 camera that fits each.
const MODEL_SLOTS = {
  'pa-multiview': [['Trước', 'camera_front_left_00'], ['Chéo trái', 'camera_fisheye_left_02'], ['Chéo phải', 'camera_fisheye_right_03']],
  'pa-front': [['Trước', 'camera_front_left_00']],
  'pq-front': [],
};
const STATE_TEXT = { running: 'Đang chạy', cancelling: 'Đang huỷ', succeeded: 'Xong', failed: 'Lỗi', cancelled: 'Đã huỷ' };

let setup = null;

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(Array.isArray(d) ? d.map((e) => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('\n') : d || res.statusText);
  }
  return data;
}

let toastTimer;
function toast(msg, error = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `toast${error ? ' error' : ''}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, error ? 9000 : 4000);
}

function fillSelect(select, options, keep = true) {
  const old = select.value;
  select.innerHTML = options.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('');
  if (keep && options.some(([v]) => v === old)) select.value = old;
  select.disabled = !options.length;
}

function gpuOptions() {
  const gpus = setup?.gpus || [];
  return gpus.map((g) => [String(g.index), `GPU ${g.index}: ${g.name}, đang dùng ${(g.memory_used_mib / 1024).toFixed(1)} / ${(g.memory_total_mib / 1024).toFixed(0)} GB, ${g.utilization}%`]);
}

function freestGpu() {
  const gpus = setup?.gpus || [];
  return gpus.length ? String(gpus.reduce((a, b) => (b.memory_used_mib < a.memory_used_mib ? b : a)).index) : '';
}

function gallery(files, src, wide = false) {
  if (!files.length) return '';
  return `<div class="gallery${wide ? ' wide' : ''}">${files.map((f) => `
    <figure><img loading="lazy" src="${esc(src(f))}" alt="${esc(f)}"><figcaption>${esc(f)}</figcaption></figure>`).join('')}</div>`;
}

document.addEventListener('click', (e) => {
  if (e.target.matches('.gallery img')) {
    const box = document.createElement('div');
    box.className = 'lightbox';
    box.innerHTML = `<img src="${esc(e.target.src)}" alt="">`;
    box.onclick = () => box.remove();
    document.body.append(box);
  }
});

// ------------------------------------------------------------------ jobs
// Polls a job and renders it into `box`; `render(status)` returns extra HTML (images, metrics).
function watchJob(box, statusUrl, cancelUrl, render, onDone) {
  box.hidden = false;
  let stopped = false;
  const tick = async () => {
    let s;
    try {
      s = await api(statusUrl);
    } catch (err) {
      box.innerHTML = `<p class="state failed">${esc(err.message)}</p>`;
      return;
    }
    const active = s.state === 'running' || s.state === 'cancelling';
    box.innerHTML = `
      <div class="head">
        ${active ? '<span class="spinner"></span>' : ''}
        <span class="state ${esc(s.state)}">${esc(STATE_TEXT[s.state] || s.state)}${s.step ? ` · ${esc(s.step)}` : ''}</span>
        <code>${esc(s.output || '')}</code>
        ${active ? '<button class="small" data-cancel>Huỷ</button>' : ''}
      </div>
      ${s.error ? `<p class="state failed">${esc(s.error)}</p>` : ''}
      ${render ? render(s) : ''}
      <details ${active || s.state === 'failed' ? 'open' : ''}><summary>Log</summary><pre>${esc((s.log || '').slice(-12000))}</pre></details>`;
    const pre = $('pre', box);
    if (pre) pre.scrollTop = pre.scrollHeight;
    const cancel = $('[data-cancel]', box);
    if (cancel) cancel.onclick = async () => { cancel.disabled = true; await api(cancelUrl, {}).catch((e) => toast(e.message, true)); };
    if (active && !stopped) setTimeout(tick, 2000);
    else if (!active && onDone) onDone(s);
  };
  tick();
  return () => { stopped = true; };
}

// ------------------------------------------------------------------ setup / step 1
async function loadSetup() {
  setup = await api('/api/setup');
  const n = setup.nurec;
  const ckpts = Object.keys(n.checkpoints || {});
  $('#status').innerHTML = [
    n.ok ? `<span class="good">NuRec sẵn sàng</span> (${ckpts.length} model)` : '<span class="bad">Chưa có NuRec</span>',
    setup.gsplat.ok ? '<span class="good">gsplat</span>' : '<span class="bad">thiếu gsplat</span>',
    `${setup.gpus.length} GPU`,
  ].join(' · ');

  $('#env').innerHTML = `
    <dt>Workspace</dt><dd><code>${esc(setup.workspace)}</code></dd>
    <dt>Instant NuRec</dt><dd>${n.ok ? `<code>${esc(n.repo)}</code>` : '<span class="tag bad">không tìm thấy</span> cần <code>backends/instant-nurec</code> có <code>.venv</code>'}</dd>
    <dt>Checkpoint</dt><dd>${ckpts.map((m) => `<span class="tag ok">${esc(m)}</span>`).join('') || '<span class="tag bad">chưa có</span>'}</dd>
    <dt>gsplat (đánh giá)</dt><dd>${setup.gsplat.ok ? '<span class="tag ok">có</span>' : '<span class="tag bad">chưa cài: fflab setup</span>'}</dd>`;

  $('#gpus').innerHTML = setup.gpus.map((g) => {
    const pct = Math.round((100 * g.memory_used_mib) / g.memory_total_mib);
    return `<div class="item"><div class="row"><span class="name">GPU ${g.index} · ${esc(g.name)}</span>
      <span class="muted">${(g.memory_used_mib / 1024).toFixed(1)} / ${(g.memory_total_mib / 1024).toFixed(0)} GB · ${g.utilization}%</span></div>
      <div class="bar"><i style="width:${pct}%"></i></div></div>`;
  }).join('') || '<p class="muted">Không thấy GPU (nvidia-smi).</p>';

  $('#datasets').innerHTML = setup.datasets.map((d) => `
    <div class="item"><div class="row"><span class="name">${esc(d.path)}</span>
      ${d.calibration_pickles ? '<span class="tag ok">có calib</span>' : '<span class="tag bad">thiếu ImageSets-rosa-6class</span>'}</div>
      ${Object.entries(d.sequences).map(([seq, info]) => `<div><code>${esc(seq)}</code>
        ${info.cameras.map((c) => `<span class="tag">${esc(c)}</span>`).join('')}
        ${info.lidar ? '<span class="tag">LiDAR</span>' : ''}</div>`).join('') || '<p class="muted">Chưa có sequence nào đủ file.</p>'}
    </div>`).join('') || `<p class="muted">Chưa thấy bộ dữ liệu. Đặt thư mục KITTI-360 (có <code>data_poses.zip</code>) trong workspace.</p>`;

  const done = { data: setup.datasets.length > 0, convert: setup.clips.length > 0, nurec: setup.reconstructions.length > 0, results: setup.evaluations.length > 0, carla: setup.carla_runs.length > 0 };
  document.querySelectorAll('#steps a').forEach((a) => a.classList.toggle('done', done[a.dataset.step]));

  // step 2 form
  fillSelect($('#convertForm [name=ds]'), setup.datasets.map((d) => [d.path, d.path]));
  updateSequences();
  // step 3 + 4 forms
  const clips = setup.clips.map((c) => [c.path, `${c.sequence_id} · ${c.frames} frame · ${c.cameras.length} camera`]);
  fillSelect($('#nurecForm [name=clip]'), clips);
  fillSelect($('#evalForm [name=clip]'), clips);
  for (const sel of [$('#nurecForm [name=gpu]'), $('#evalForm [name=gpu]'), $('#carlaForm [name=gpu]'), $('#carlaServerForm [name=gpu]')]) {
    const had = sel.value;
    fillSelect(sel, gpuOptions());
    if (!had) sel.value = freestGpu();
  }
  updateSlots();
  renderResults();
  renderCarla();
}

// ------------------------------------------------------------------ step 2: convert
function updateSequences() {
  const form = $('#convertForm');
  const ds = setup.datasets.find((d) => d.path === form.ds.value);
  const seqs = ds ? Object.keys(ds.sequences) : [];
  fillSelect(form.sequence, seqs.map((s) => [s, s]));
  const cams = ds?.sequences[form.sequence.value]?.cameras || [];
  const names = setup.cameras || {};
  $('#convertCams').innerHTML = Object.keys(names).map((c) => `
    <label><input type="checkbox" name="cam" value="${esc(c)}" ${cams.includes(c) ? 'checked' : 'disabled'}>
      ${esc(c)} <span class="muted">${esc(names[c])}</span></label>`).join('');
}

$('#convertForm').addEventListener('change', (e) => {
  if (e.target.name === 'ds' || e.target.name === 'sequence') updateSequences();
});

$('#convertForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    dataset: f.ds.value, sequence: f.sequence.value, start: Number(f.start.value), frames: Number(f.frames.value),
    cameras: [...f.querySelectorAll('[name=cam]:checked')].map((x) => x.value),
    lidar: f.lidar.checked, name: f.clipname.value.trim(), fisheye_model: f.fisheye_model.value,
  };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/pipeline/convert', body);
    watchJob($('#convertJob'), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`,
      (s) => gallery(s.artifacts.filter((a) => a.endsWith('.png')), (a) => `/api/pipeline/${s.job_id}/artifacts/${url(a)}`),
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await loadSetup();
          $('#nurecForm [name=clip]').value = s.info.clip;
          $('#evalForm [name=clip]').value = s.info.clip;
          updateSlots();
          toast('Đã tạo clip. Kiểm tra ảnh chiếu LiDAR rồi sang bước 3.');
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

// ------------------------------------------------------------------ step 3: NuRec
function updateSlots() {
  const form = $('#nurecForm');
  const clip = setup?.clips.find((c) => c.path === form.clip.value);
  const cams = clip?.cameras || [];
  const slots = MODEL_SLOTS[form.model.value];
  $('#camHint').textContent = slots.length ? `${slots.length} camera, theo thứ tự` : 'model này dùng camera cố định';
  $('#nurecCams').innerHTML = slots.map(([label, preferred], i) => `
    <label>${esc(label)}<select name="slot${i}">${cams.map((c) => `<option ${c === preferred ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></label>`).join('')
    || '<p class="muted">pq-front chỉ chạy với camera tên <code>camera_front_wide_120fov</code>, nên không dùng được với clip KITTI-360.</p>';
}

$('#nurecForm').addEventListener('change', (e) => {
  if (e.target.name === 'clip' || e.target.name === 'model') updateSlots();
});

$('#nurecForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const model = f.model.value;
  const checkpoint = setup.nurec.checkpoints[model];
  if (!checkpoint) return toast(`Chưa có checkpoint cho ${model}`, true);
  const camera_ids = MODEL_SLOTS[model].map((_, i) => f[`slot${i}`].value);
  if (new Set(camera_ids).size !== camera_ids.length) return toast('Mỗi vị trí phải là một camera khác nhau', true);
  const body = {
    repo: NUREC_REPO, python: NUREC_PYTHON, checkpoint, ncore_path: f.clip.value, model, camera_ids,
    gpu: f.gpu.value === '' ? null : Number(f.gpu.value),
    n_gaussians: Number(f.n_gaussians.value), max_chunks: Number(f.max_chunks.value),
  };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/reconstruction', body);
    const clip = f.clip.value;
    watchJob($('#nurecJob'), `/api/reconstruction/${job.job_id}`, `/api/reconstruction/${job.job_id}/cancel`,
      (s) => {
        const ply = s.artifacts.filter((a) => a.endsWith('.ply'));
        return ply.map((p) => `<p>PLY: <a href="/api/reconstruction/${s.job_id}/artifacts/${url(p)}" download>${esc(p)}</a></p>`).join('')
          + gallery(s.artifacts.filter((a) => a.endsWith('.png')), (a) => `/api/reconstruction/${s.job_id}/artifacts/${url(a)}`);
      },
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await loadSetup();
          const ply = setup.reconstructions.find((r) => r.job_id === s.job_id);
          if (ply) $('#evalForm [name=ply]').value = ply.path;
          $('#evalForm [name=clip]').value = clip;
          toast('NuRec xong. Sang bước 4 để đánh giá.');
          location.hash = '#step-results';
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

// ------------------------------------------------------------------ step 4: results
function metricsTable(m) {
  const rows = Object.entries(m).sort(([a], [b]) => a.localeCompare(b));
  const mean = rows.reduce((s, [, v]) => s + v.psnr_mean, 0) / (rows.length || 1);
  return `<table class="metrics"><tr><th>Camera</th><th>PSNR (dB)</th><th>Số view</th></tr>
    ${rows.map(([c, v]) => `<tr><td>${esc(c)}</td><td class="num">${v.psnr_mean.toFixed(2)}</td><td class="num">${v.views}</td></tr>`).join('')}
    <tr><th>Trung bình</th><th class="num">${mean.toFixed(2)}</th><th></th></tr></table>`;
}

function renderResults() {
  const recons = setup.reconstructions;
  // the PSNR tools render in the clip's NCore world: only NuRec PLYs are in it (3DGUT has its own metrics.json)
  fillSelect($('#evalForm [name=ply]'), recons.filter((r) => r.kind !== '3dgut').map((r) => [r.path, `${r.path.split('/').at(-1)} · ${r.size_mib} MB · job ${r.job_id.slice(0, 8)}`]));
  $('#recons').innerHTML = recons.map((r) => `
    <div class="item"><div class="row"><span class="name">${esc(r.path.split('/').at(-1))}</span><span class="muted">${r.size_mib} MB</span></div>
      <code>${esc(r.path)}</code><br>
      <a href="/api/outputs/${url(r.path.replace(/^outputs\//, ''))}" download>Tải PLY</a>
      <span class="muted">· xem bằng cách kéo vào superspl.at/editor</span></div>`).join('')
    || '<p class="muted">Chưa có. Chạy bước 3.</p>';

  $('#evals').innerHTML = setup.evaluations.map((ev, i) => {
    const base = ev.path.replace(/^outputs\//, '');
    return `<details class="eval" ${i === 0 ? 'open' : ''}>
      <summary>${esc(ev.ply ? ev.ply.split('/').at(-1) : ev.path)} <span class="muted">so với ${esc(ev.clip ? ev.clip.split('/').at(-1) : '?')}</span></summary>
      ${metricsTable(ev.metrics)}
      <p class="hint">Mỗi ảnh: trái là ảnh thật, phải là ảnh render.</p>
      ${gallery(ev.images, (f) => `/api/outputs/${url(`${base}/${f}`)}`, true)}
    </details>`;
  }).join('');
}

$('#evalForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = { ply: f.ply.value, clip: f.clip.value, every: Number(f.every.value), gpu: f.gpu.value === '' ? null : Number(f.gpu.value) };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/pipeline/evaluate', body);
    watchJob($('#evalJob'), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`,
      (s) => (s.metrics ? metricsTable(s.metrics) : ''),
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await loadSetup();
          $('#evalJob').hidden = true;
          toast('Đánh giá xong.');
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

loadSetup().catch((err) => {
  $('#status').innerHTML = `<span class="bad">${esc(err.message)}</span>`;
});

// ------------------------------------------------------------------ step 5: CARLA
const SERVER_TEXT = { stopped: 'Đang tắt', starting: 'Đang khởi động…', running: 'Đang chạy', stopping: 'Đang tắt…', failed: 'Lỗi' };
let serverTimer;

function renderServer(c) {
  const state = c.port_open && c.state !== 'running' ? 'running' : c.state;
  const external = c.port_open && !c.managed;
  $('#carlaServer').innerHTML = !c.installed
    ? `<p><span class="tag bad">Chưa có CARLA</span> Giải nén CARLA 0.9.16 vào <code>${esc(c.root)}</code> (hoặc đặt biến <code>CARLA_ROOT</code>).</p>`
    : `<div class="row"><span class="name"><span class="state-dot ${esc(state)}"></span>${esc(SERVER_TEXT[state] || state)}</span>
        <span class="muted">localhost:${c.port}${c.gpu !== null && c.gpu !== undefined ? ` · GPU ${c.gpu}` : ''}</span></div>
      ${external ? '<p class="muted">Server này được bật từ ngoài giao diện; dùng được, nhưng tắt ở nơi đã bật nó.</p>' : ''}
      ${c.message ? `<p class="muted">${esc(c.message)}</p>` : ''}
      ${c.client ? '' : '<p><span class="tag bad">venv gsplat thiếu gói carla</span> <code>pip install carla==0.9.16</code></p>'}
      ${c.log ? `<details><summary>Log server</summary><pre>${esc(c.log)}</pre></details>` : ''}`;
  $('#carlaStart').disabled = !c.installed || c.port_open || ['starting', 'stopping'].includes(c.state);
  $('#carlaStop').disabled = !c.managed;
  $('#carlaForm [name=use_carla]').disabled = !c.port_open;
  if (!c.port_open) $('#carlaForm [name=use_carla]').checked = false;
  clearTimeout(serverTimer);
  if (['starting', 'stopping'].includes(c.state)) {
    serverTimer = setTimeout(async () => renderServer(await api('/api/carla/server').catch(() => c)), 2000);
  }
}

function matchClip() {
  const ply = $('#carlaForm [name=ply]').value;
  const stem = ply.split('/').at(-1).replace(/\.ply$/, '');
  // NuRec PLYs are named after their clip; a 3DGUT run matches the clip with the same frame count
  const rec = setup.reconstructions.find((r) => r.path === ply);
  const clip = setup.clips.find((c) => c.path.endsWith(`/${stem}.json`))
    || (rec?.kind === '3dgut' && setup.clips.find((c) => c.frames === rec.frames));
  if (clip) $('#carlaForm [name=clip]').value = clip.path;
}

function renderCarla() {
  renderServer(setup.carla);
  fillSelect($('#carlaForm [name=ply]'), setup.reconstructions.map((r) => [r.path, `${r.kind === '3dgut' ? '3DGUT' : 'NuRec'} · ${r.kind === '3dgut' ? r.job_id : r.path.split('/').at(-1)} · ${r.size_mib} MB`]));
  fillSelect($('#carlaForm [name=clip]'), setup.clips.map((c) => [c.path, `${c.sequence_id} · ${c.frames} frame`]));
  matchClip();
  $('#carlaRuns').innerHTML = setup.carla_runs.map((r, i) => {
    const base = r.path.replace(/^outputs\//, '');
    const what = `${r.kind === '3dgut' ? '3DGUT' : 'NuRec'} · ${r.use_carla ? `${r.npc} xe CARLA` : 'chỉ cảnh dựng lại'}`;
    const view = [r.lateral ? `lệch ${r.lateral} m` : '', r.lift ? `nâng ${r.lift} m` : '', r.fov ? `FOV ${r.fov}°` : ''].filter(Boolean).join(', ');
    return `<div class="run"><div class="row"><span class="name">${esc(r.ply ? r.ply.split('/').at(-1) : r.path)}</span>
      <span class="muted">${esc(what)}${view ? ` · ${esc(view)}` : ''} · <a href="/api/outputs/${url(`${base}/${r.video}`)}" download>tải video</a></span></div>
      ${i < 3 ? `<video controls preload="metadata" src="/api/outputs/${url(`${base}/${r.video}`)}"></video>` : ''}</div>`;
  }).join('') || '<p class="muted">Chưa có video nào.</p>';
}

$('#carlaForm').addEventListener('change', (e) => {
  if (e.target.name === 'ply') matchClip();
  if (e.target.name === 'use_carla') $('#carlaForm [name=npc]').disabled = !e.target.checked;
});

$('#carlaServerForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const gpu = e.target.gpu.value === '' ? null : Number(e.target.gpu.value);
  try {
    renderServer(await api('/api/carla/server/start', { gpu }));
  } catch (err) {
    toast(err.message, true);
  }
});

$('#carlaStop').addEventListener('click', async () => {
  $('#carlaStop').disabled = true;
  try {
    renderServer(await api('/api/carla/server/stop', {}));
  } catch (err) {
    toast(err.message, true);
  }
});

$('#carlaForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    ply: f.ply.value, clip: f.clip.value, use_carla: f.use_carla.checked, npc: Number(f.npc.value),
    start: Number(f.start.value), duration: Number(f.duration.value), lateral: Number(f.lateral.value),
    lift: Number(f.lift.value), fov: Number(f.fov.value), gpu: f.gpu.value === '' ? null : Number(f.gpu.value),
  };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/pipeline/carla', body);
    watchJob($('#carlaJob'), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`,
      (s) => {
        const video = s.state === 'succeeded' && s.artifacts.find((a) => a.endsWith('.mp4'));
        return video ? `<video controls autoplay muted src="/api/pipeline/${s.job_id}/artifacts/${url(video)}"></video>` : '';
      },
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await loadSetup();
          toast('Đã render xong video.');
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});
