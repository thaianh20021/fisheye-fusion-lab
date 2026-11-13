(() => {
  "use strict";
  const el = name => document.getElementById(`training${name}`);
  if (!el("Form")) return;
  const storageKey = "fusion.localTraining.activeJob";
  const terminal = state => ["succeeded", "failed", "cancelled", "interrupted"].includes(state);
  const states = {queued: "Đang chờ", running: "Đang huấn luyện",
    cancelling: "Đang dừng", succeeded: "Hoàn tất", failed: "Thất bại", cancelled: "Đã dừng"};
  let jobId = null, inspected = false, busy = false, polling = false;
  let cancelling = false, cancelRequested = false, timer = null;
  let cameras = [];
  const selected = () => cameras.filter(camera => camera.input.checked).map(camera => camera.id);
  const validSelection = () => !cameras.length || selected().length === Number(el("CameraCount").value);
  const text = value => typeof value === "string" ? value : JSON.stringify(value);

  function storage(value) {
    try {
      if (value === undefined) return localStorage.getItem(storageKey);
      if (value === null) localStorage.removeItem(storageKey);
      else localStorage.setItem(storageKey, value);
    } catch {
      el("Storage").textContent = "Không truy cập được bộ nhớ trình duyệt; không thể khôi phục job khi tải lại.";
    }
    return null;
  }

  function controls() {
    el("Settings").disabled = busy || Boolean(jobId);
    el("Start").disabled = busy || Boolean(jobId) || !inspected || !validSelection();
    el("Cancel").disabled = !jobId || busy || polling || cancelling || cancelRequested;
    el("Selection").textContent = cameras.length
      ? `Đã chọn ${selected().length} / ${el("CameraCount").value} camera.` : "";
  }

  async function request(url, body) {
    const response = await fetch(url, {
      method: body === undefined ? "GET" : "POST",
      cache: "no-store",
      headers: body === undefined ? {} : {"Content-Type": "application/json"},
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch { data = null; }
    if (!response.ok) {
      const error = new Error(text(data?.detail ?? data?.error ?? raw) || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      throw new Error("Phản hồi máy chủ không hợp lệ.");
    }
    return data;
  }

  function artifacts(data, id) {
    el("Artifacts").replaceChildren();
    el("Previews").replaceChildren();
    for (const name of Array.isArray(data.artifacts) ? data.artifacts : []) {
      if (typeof name !== "string" || !name) continue;
      const url = `/api/training/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(name)}`;
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.textContent = name;
      const item = document.createElement("li");
      item.append(link);
      el("Artifacts").append(item);
      if (!["after.png", "reference.png"].includes(name)) continue;
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      caption.textContent = name === "after.png" ? "Sau huấn luyện" : "Ảnh tham chiếu";
      const image = document.createElement("img");
      image.alt = caption.textContent;
      image.addEventListener("error", () => {
        image.hidden = true;
        caption.textContent += ": không tải được ảnh.";
      }, {once: true});
      image.src = url;
      figure.append(image, caption);
      el("Previews").append(figure);
    }
  }

  function display(data) {
    if (typeof data.state !== "string") throw new Error("Thiếu trạng thái job.");
    const steps = Number(data.steps);
    const step = Number(data.step);
    el("Progress").max = Number.isFinite(steps) && steps > 0 ? steps : 1;
    if (Number.isFinite(step)) el("Progress").value = Math.max(0, Math.min(step, el("Progress").max));
    else el("Progress").removeAttribute("value");
    el("Status").textContent = `${states[data.state] || data.state} · ${jobId}`
      + (Number.isFinite(step) && Number.isFinite(steps) ? ` · ${step} / ${steps} bước` : "");
    el("Log").textContent = Array.isArray(data.log) ? data.log.map(text).join("\n") : text(data.log ?? "");
    el("Error").textContent = data.error ? text(data.error) : "";
    el("Output").textContent = data.output ? `Thư mục kết quả: ${text(data.output)}` : "";
    if (terminal(data.state)) {
      artifacts(data, jobId);
      jobId = null;
      storage(null);
      clearTimeout(timer);
    }
    controls();
  }

  function schedule() {
    clearTimeout(timer);
    if (jobId) timer = setTimeout(poll, 1000);
  }

  async function poll() {
    if (!jobId || polling || cancelling) return;
    polling = true;
    controls();
    try {
      display(await request(`/api/training/${encodeURIComponent(jobId)}`));
    } catch (error) {
      el("Error").textContent = error.message;
      if (error.status === 400 || error.status === 404 || error.status === 410) {
        jobId = null;
        storage(null);
        el("Status").textContent = "Job đã lưu không còn tồn tại. Có thể kiểm tra dataset và chạy lại.";
      } else {
        el("Status").textContent = "Chưa lấy được trạng thái; đang thử lại.";
      }
    } finally {
      polling = false;
      controls();
      schedule();
    }
  }

  el("Dataset").addEventListener("input", () => {
    inspected = false;
    cameras = [];
    el("Cameras").replaceChildren();
    el("Rig").hidden = true;
    el("Inspection").textContent = "Đường dẫn đã thay đổi; cần kiểm tra lại.";
    controls();
  });
  el("CameraCount").addEventListener("change", controls);
  el("Inspect").addEventListener("click", async () => {
    if (busy || jobId) return;
    if (!el("Dataset").value.trim() || !el("Dataset").reportValidity()) return;
    busy = true;
    inspected = false;
    cameras = [];
    el("Rig").hidden = true;
    el("Cameras").replaceChildren();
    el("Error").textContent = "";
    el("Inspection").textContent = "Đang kiểm tra dataset...";
    controls();
    try {
      const data = await request("/api/training/inspect", {
        dataset: el("Dataset").value.trim(), camera_ids: null,
      });
      if (!Array.isArray(data.cameras) || !Number.isFinite(data.observations)
          || !Number.isFinite(data.train_count) || !Number.isFinite(data.test_count)) {
        throw new Error("Kết quả kiểm tra dataset không hợp lệ.");
      }
      const ids = new Set();
      for (const camera of data.cameras) {
        if (!camera || !["string", "number"].includes(typeof camera.id) || ids.has(camera.id)) {
          throw new Error("Danh sách camera không hợp lệ.");
        }
        ids.add(camera.id);
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = cameras.length === 0;
        input.addEventListener("change", controls);
        label.append(input, document.createTextNode(`${camera.id} · ${camera.model ?? "Không rõ model"}`));
        cameras.push({id: camera.id, input});
        el("Cameras").append(label);
      }
      el("CameraCount").value = "1";
      for (const option of el("CameraCount").options) option.disabled = Number(option.value) > cameras.length;
      el("Rig").hidden = !cameras.length;
      el("Inspection").textContent = `${data.observations} quan sát · ${data.train_count} train · ${data.test_count} test`
        + (!cameras.length ? " · Dataset không cung cấp camera để chọn; dùng cấu hình có sẵn." : "");
      inspected = true;
    } catch (error) {
      el("Error").textContent = error.message;
      el("Inspection").textContent = "Kiểm tra thất bại.";
    } finally {
      busy = false;
      controls();
    }
  });

  el("Form").addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || jobId || !inspected || !validSelection() || !el("Form").reportValidity()) return;
    const steps = Number(el("Steps").value);
    if (!Number.isInteger(steps) || steps < 1 || steps > 10000) return;
    busy = true;
    cancelRequested = false;
    el("Error").textContent = "";
    el("Log").textContent = "";
    el("Output").textContent = "";
    el("Artifacts").replaceChildren();
    el("Previews").replaceChildren();
    el("Progress").removeAttribute("value");
    el("Status").textContent = "Đang gửi yêu cầu huấn luyện...";
    controls();
    try {
      const data = await request("/api/training", {
        dataset: el("Dataset").value.trim(), camera_ids: cameras.length ? selected() : null, steps,
      });
      if (typeof data.job_id !== "string" || !data.job_id) throw new Error("Máy chủ không trả về job_id hợp lệ.");
      jobId = data.job_id;
      storage(jobId);
      display(data);
    } catch (error) {
      el("Error").textContent = error.message;
      el("Status").textContent = "Không xác nhận được trạng thái khởi chạy.";
      el("Progress").value = 0;
    } finally {
      busy = false;
      controls();
      schedule();
    }
  });

  el("Cancel").addEventListener("click", async () => {
    if (!jobId || busy || polling || cancelling || cancelRequested) return;
    cancelling = true;
    clearTimeout(timer);
    controls();
    try {
      const data = await request(`/api/training/${encodeURIComponent(jobId)}/cancel`, {});
      cancelRequested = true;
      if (typeof data.state === "string") display(data);
      else el("Status").textContent = "Đã gửi yêu cầu dừng.";
    } catch (error) {
      el("Error").textContent = error.message;
    } finally {
      cancelling = false;
      controls();
      schedule();
    }
  });

  jobId = storage();
  controls();
  if (jobId) {
    el("Status").textContent = "Đang khôi phục job đã lưu...";
    poll();
  }
})();
