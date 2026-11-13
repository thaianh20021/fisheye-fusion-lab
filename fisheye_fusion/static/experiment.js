(() => {
  function init() {
    const section = document.getElementById("realExperiment");
    const button = document.getElementById("loadExperiment");
    const content = document.getElementById("experimentContent");
    if (!section || !button || !content) return;
    const artifact = (name, run) => `/api/experiment/artifacts/${encodeURIComponent(name)}?run=${encodeURIComponent(run)}`;
    const number = new Intl.NumberFormat("vi-VN", {maximumFractionDigits: 3});
    const value = item => item == null ? "Chưa có dữ liệu"
      : typeof item === "number" ? number.format(item)
      : typeof item === "object" ? JSON.stringify(item) : String(item);
    function node(tag, text, className) {
      const el = document.createElement(tag);
      if (text !== undefined) el.textContent = text;
      if (className) el.className = className;
      return el;
    }
    const runs = {baseline: "real-gs-run", geer: "geer-run"};
    const controls = node("div", undefined, "experiment-controls");
    const runLabel = node("label", "Thí nghiệm");
    const select = node("select");
    select.id = "experimentRun";
    runLabel.htmlFor = select.id;
    [["baseline", "Baseline"], ["geer", "GEER"]].forEach(([id, label]) => {
      const option = node("option", label);
      option.value = id;
      select.append(option);
    });
    controls.append(runLabel, select);
    section.insertBefore(controls, content);
    const status = node("p", "", "experiment-status");
    status.setAttribute("role", "status");
    section.insertBefore(status, content);

    function image(name, caption, run) {
      const figure = node("figure");
      const img = node("img");
      img.alt = caption;
      const label = node("figcaption", caption);
      img.addEventListener("error", () => {
        img.hidden = true;
        label.textContent = `${caption}: ảnh chưa có hoặc không tải được.`;
      });
      img.src = artifact(name, run);
      figure.append(img, label);
      return figure;
    }

    function table(headings, rows) {
      const wrapper = node("div", undefined, "experiment-table");
      const result = node("table");
      const head = node("thead");
      const header = node("tr");
      headings.forEach(text => {
        const cell = node("th", text);
        cell.scope = "col";
        header.append(cell);
      });
      head.append(header);
      const body = node("tbody");
      rows.forEach(row => {
        const tr = node("tr");
        row.forEach(item => tr.append(node("td", value(item))));
        body.append(tr);
      });
      result.append(head, body);
      wrapper.append(result);
      return wrapper;
    }

    function display(metrics, run) {
      const summary = node("dl", undefined, "experiment-metrics");
      [
        ["Backend", metrics.backend], ["Bộ dữ liệu", metrics.dataset],
        ["GPU", metrics.gpu], ["Số bước", metrics.steps],
        ["Số Gaussian", metrics.gaussians],
        ["Độ phân giải", Array.isArray(metrics.resolution) ? metrics.resolution.join(" × ") : metrics.resolution],
        ["Thời gian (giây)", metrics.seconds], ["VRAM đỉnh (MiB)", metrics.peak_vram_mib],
      ].forEach(([label, item]) => summary.append(node("dt", label), node("dd", value(item))));
      content.append(summary, node("h3", "PSNR trên ảnh kiểm định (dB)"));
      const before = metrics.heldout_psnr_before || {};
      const after = metrics.heldout_psnr_after || {};
      const views = [...new Set([...Object.keys(before), ...Object.keys(after)])];
      content.append(views.length
        ? table(["Ảnh", "Trước huấn luyện", "Sau huấn luyện"],
          views.map(view => [view, before[view], after[view]]))
        : node("p", "Chưa có số liệu PSNR."));
      const comparison = node("div", undefined, "experiment-comparison");
      [["reference.png", "Ảnh tham chiếu"], ["before.png", "Trước huấn luyện"],
        ["after.png", "Sau huấn luyện"]].forEach(([name, label]) => comparison.append(image(name, label, run)));
      content.append(comparison);

      const links = node("div", undefined, "experiment-links");
      [["scene.ply", "Tải mô hình scene.ply"], ["metrics.json", "Tải số liệu metrics.json"]]
        .forEach(([name, label]) => {
          const link = node("a", label);
          link.href = artifact(name, run);
          link.download = name;
          links.append(link);
        });
      content.append(links, node("h3", "Góc nhìn mới"));
      const slider = node("input");
      slider.type = "range";
      slider.id = "experimentNovelView";
      slider.min = "0";
      slider.max = "8";
      slider.step = "1";
      slider.value = "0";
      const label = node("label", "Góc nhìn 1 / 9");
      label.htmlFor = slider.id;
      const preview = node("div", undefined, "experiment-novel");
      function updateView() {
        const index = Number(slider.value);
        label.textContent = `Góc nhìn ${index + 1} / 9`;
        slider.setAttribute("aria-valuetext", label.textContent);
        preview.replaceChildren(image(`novel-${String(index).padStart(2, "0")}.png`, label.textContent, run));
      }
      slider.addEventListener("input", updateView);
      updateView();
      content.append(label, slider, preview);

      if (Array.isArray(metrics.trace) && metrics.trace.length) {
        const details = node("details");
        details.append(node("summary", "Diễn tiến huấn luyện"),
          table(["Bước", "Sai số L1", "Thời gian (giây)"],
            metrics.trace.map(item => [item.step, item.l1, item.seconds])));
        content.append(details);
      }
      content.append(node("h3", "Giới hạn"),
        node("p", value(metrics.limitations), "experiment-limitations"));
    }

    async function load() {
      if (button.disabled) return;
      const run = select.value;
      button.disabled = true;
      select.disabled = true;
      content.setAttribute("aria-busy", "true");
      content.replaceChildren();
      status.textContent = "Đang tải kết quả huấn luyện...";
      try {
        const response = await fetch(`/api/experiment?run=${encodeURIComponent(run)}`, {cache: "no-store"});
        if (response.status === 404 || response.status === 400) {
          status.textContent = `Chưa có kết quả hợp lệ tại outputs/${runs[run]}.`;
          return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const metrics = data.metrics ?? data;
        if (!metrics || typeof metrics !== "object" || Array.isArray(metrics) || !metrics.backend) {
          throw new Error("Dữ liệu kết quả không hợp lệ.");
        }
        display(metrics, run);
        status.textContent = "Đã tải kết quả thí nghiệm 3DGS tĩnh.";
      } catch (error) {
        content.replaceChildren();
        status.textContent = `Không tải được kết quả: ${error.message}`;
      } finally {
        button.disabled = false;
        select.disabled = false;
        content.setAttribute("aria-busy", "false");
      }
    }
    button.addEventListener("click", load);
    select.addEventListener("change", load);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})();
