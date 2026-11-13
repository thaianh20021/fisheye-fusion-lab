# Fisheye Fusion Tool Implementation Plan

> **For agentic workers:** Implement task-by-task with isolated write scopes. Run the listed checks after each task.

**Goal:** Build a native-friendly local Web UI and CLI for validated pinhole/fisheye manifests, 1/2/4-camera preview fusion, LiDAR overlay, and remote job bundle export.

**Architecture:** FastAPI serves a small browser UI and JSON API. A dependency-light Python core handles manifest validation, camera projection, transforms, and CPU image fusion. Backend adapters remain isolated and initially expose status/preflight rather than importing all upstream repositories.

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, NumPy, OpenCV, Pydantic, vanilla HTML/CSS/JS, pytest.

---

### Task 1: Core package and manifest validation

**Files:**
- Create: `fisheye_fusion/manifest.py`
- Create: `fisheye_fusion/models.py`
- Create: `tests/test_manifest.py`

- [ ] Define `CameraConfig`, `FrameConfig`, `FusionConfig`, and `ProjectManifest` with Pydantic models.
- [ ] Validate unique camera IDs, supported models, positive intrinsics, 4x4 transforms, referenced paths, and fusion camera membership.
- [ ] Add `load_manifest(path)` and `validate_manifest(manifest, root)`.
- [ ] Test valid pinhole, valid fisheye, 2-camera fusion, 4-camera fusion, duplicate IDs, invalid focal lengths, and missing image references.
- [ ] Run `python -m pytest tests/test_manifest.py -q`.

### Task 2: Camera geometry

**Files:**
- Create: `fisheye_fusion/geometry.py`
- Create: `tests/test_geometry.py`

- [ ] Implement homogeneous transform helpers and rigid-transform checks.
- [ ] Implement pinhole `pixel_to_ray` and OpenCV fisheye `pixel_to_ray`.
- [ ] Implement `project_points` for both models and positive-depth filtering.
- [ ] Test identity projection, principal-point ray, 90-degree rig rotations, and round-trip tolerance.
- [ ] Run `python -m pytest tests/test_geometry.py -q`.

### Task 3: CPU fusion and LiDAR overlay

**Files:**
- Create: `fisheye_fusion/fusion.py`
- Create: `fisheye_fusion/io.py`
- Create: `tests/test_fusion.py`

- [ ] Implement nearest timestamp selection and image loading.
- [ ] Implement single-camera pass-through, pair fusion, and four-camera equirectangular output using cached remap maps.
- [ ] Implement feather blending and valid-coverage mask.
- [ ] Implement LiDAR projection overlay with depth and color filters.
- [ ] Add synthetic checkerboard fixtures generated in the test.
- [ ] Run `python -m pytest tests/test_fusion.py -q`.

### Task 4: CLI

**Files:**
- Create: `fisheye_fusion/cli.py`
- Create: `tests/test_cli.py`

- [ ] Implement `validate`, `inspect`, `render`, and `build-map` commands.
- [ ] Emit machine-readable JSON errors with nonzero exit status.
- [ ] Make all output paths relative to the project root and reject traversal.
- [ ] Test manifest validation and synthetic render through the CLI.
- [ ] Run `python -m pytest tests/test_cli.py -q`.

### Task 5: Local Web UI

**Files:**
- Create: `fisheye_fusion/server.py`
- Create: `fisheye_fusion/static/index.html`
- Create: `fisheye_fusion/static/app.js`
- Create: `fisheye_fusion/static/styles.css`
- Create: `tests/test_server.py`

- [ ] Add endpoints for health, manifest inspection, preview render, and artifact listing.
- [ ] Build views for project/rig, fusion mode, preview, runtime/backend status, and outputs.
- [ ] Show explicit unsupported states for CUDA/backend adapters.
- [ ] Add accessible controls, clear error messages, and responsive layout.
- [ ] Test health and manifest endpoints with FastAPI TestClient.
- [ ] Run `python -m pytest tests/test_server.py -q`.

### Task 6: Runtime preflight and remote bundles

**Files:**
- Create: `fisheye_fusion/runtimes.py`
- Create: `fisheye_fusion/jobs.py`
- Create: `templates/run_colab.ipynb`
- Create: `templates/run_kaggle.ipynb`
- Create: `tests/test_jobs.py`

- [ ] Detect Python, CUDA visibility, GPU name, Ninja, disk, RAM, and VRAM without requiring CUDA.
- [ ] Generate a bounded job bundle with manifest, selected data, calibration, logs directory, and notebook template.
- [ ] Include only adapters explicitly marked runnable.
- [ ] Test bundle path safety, manifest inclusion, and notebook presence.
- [ ] Run `python -m pytest tests/test_jobs.py -q`.

### Task 7: Backend registry and research sources

**Files:**
- Create: `fisheye_fusion/backends.py`
- Create: `docs/backend_matrix.md`
- Create: `tests/test_backends.py`

- [ ] Register the 12 scoped methods with URL, task, camera model, LiDAR capability, runtime status, and license note.
- [ ] Mark `3DGUT`, `SplatAD`, and `Instant NuRec` as research adapters with explicit preflight requirements.
- [ ] Keep upstream checkout and environment commands as metadata, not automatic installs.
- [ ] Test registry completeness and unsupported-backend behavior.
- [ ] Run `python -m pytest tests/test_backends.py -q`.

### Task 8: Verification

**Files:**
- Modify: `README.md`
- Create: `examples/synthetic_rig/manifest.json`

- [ ] Run the full test suite.
- [ ] Run a local server and render the synthetic 4-camera fixture.
- [ ] Verify Windows path handling and browser loading.
- [ ] Document Colab/Kaggle export as bundle generation, not live orchestration.
- [ ] Record known gaps: no real dataset adapter, no neural training integration, no depth-aware seam correction.

