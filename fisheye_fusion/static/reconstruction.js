(() => {
  const element = id => document.getElementById(id);
  let jobId = null;
  let timer = null;
  let busy = false;

  function config() {
    return {
      repo: element("nurecRepo").value,
      python: element("nurecPython").value,
      checkpoint: element("nurecCheckpoint").value,
      ncore_path: element("nurecInput").value,
      model: element("nurecModel").value,
      max_chunks: Number(element("nurecChunks").value),
      n_gaussians: Number(element("nurecGaussians").value),
      render_preview: element("nurecPreview").checked,
      include_dynamic: element("nurecDynamic").checked,
    };
  }

  async function request(url, payload) {
    const response = await fetch(url, payload === undefined ? {} : {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = data.detail;
      throw Error(Array.isArray(detail) ? detail.map(item => item.msg).join("\n") : detail || response.statusText);
    }
    return data;
  }

  function display(data) {
    element("reconstructionState").textContent = data.state || (data.ok ? "Sẵn sàng" : "Thiếu môi trường");
    element("nurecLog").textContent = data.log
      ? [data.error, data.log].filter(Boolean).join("\n")
      : JSON.stringify(data, null, 2);
    element("nurecCancel").disabled = !["running", "cancelling"].includes(data.state);
    element("nurecRefresh").disabled = !jobId;
    const links = element("nurecArtifacts");
    links.replaceChildren();
    for (const artifact of data.artifacts || []) {
      const name = typeof artifact === "string" ? artifact : artifact.path;
      const url = `/api/reconstruction/${encodeURIComponent(jobId)}/artifacts/${name.split(/[\\/]/).map(encodeURIComponent).join("/")}`;
      const link = document.createElement("a");
      link.href = url;
      link.textContent = name;
      link.download = name.split(/[\\/]/).pop();
      links.append(link);
      if (name.endsWith(".render.png")) {
        element("nurecRender").src = url;
        element("nurecRender").hidden = false;
      }
    }
  }

  async function refresh() {
    clearTimeout(timer);
    if (!jobId) return;
    try {
      const data = await request(`/api/reconstruction/${encodeURIComponent(jobId)}`);
      display(data);
      if (["running", "cancelling"].includes(data.state)) timer = setTimeout(refresh, 2000);
    } catch (error) {
      element("nurecLog").textContent = error.message;
    }
  }

  async function action(task) {
    if (busy) return;
    busy = true;
    element("nurecCheck").disabled = element("nurecStart").disabled = true;
    try { await task(); }
    catch (error) {
      element("reconstructionState").textContent = "Chưa chạy / có lỗi";
      element("nurecLog").textContent = error.message;
    } finally {
      busy = false;
      element("nurecCheck").disabled = element("nurecStart").disabled = false;
    }
  }

  element("nurecCheck").onclick = () => action(async () => display(await request("/api/reconstruction/preflight", config())));
  element("nurecStart").onclick = () => action(async () => {
    const data = await request("/api/reconstruction", config());
    jobId = data.job_id;
    element("nurecRender").hidden = true;
    display(data);
    await refresh();
  });
  element("nurecCancel").onclick = () => action(async () => {
    display(await request(`/api/reconstruction/${encodeURIComponent(jobId)}/cancel`, {}));
    await refresh();
  });
  element("nurecRefresh").onclick = refresh;
})();
