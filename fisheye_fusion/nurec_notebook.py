"""Opt-in, isolated InstantNuRec inference notebook export."""
import json
from textwrap import dedent


COMMIT = "f2ecafcdf28f2dc8f2cc17447a772571cde558be"
REPOSITORY = "https://github.com/NVIDIA/instant-nurec"


def make_nurec_notebook(platform: str) -> bytes:
    if platform not in ("colab", "kaggle"):
        raise ValueError("platform must be colab or kaggle")
    base = "/content" if platform == "colab" else "/kaggle/working"
    cells = []

    def cell(kind, source):
        item = {
            "cell_type": kind, "id": f"nurec-{len(cells)}",
            "metadata": {}, "source": dedent(source).strip() + "\n",
        }
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """
        # InstantNuRec: local-assets inference, not training
        Execute cells in order. Supply an existing NCore JSON clip and its local
        assets, plus the trusted pa-front 1.1.0 checkpoint. No dataset/checkpoint
        download is provided. No KITTI or canonical-manifest conversion is provided.

        Review NVIDIA's repository LICENSE and checkpoint/dataset terms before
        enabling setup. Code, dependencies, and managed Python downloads require
        ALLOW_SETUP=True. CUDA render extensions require a second explicit opt-in.
        A checkpoint may execute code when loaded: use a trusted official file.
        Filename checking cannot establish checkpoint authenticity or compatibility.

        T4 compatibility and memory sufficiency are not guaranteed. One chunk and
        200,000 merged Gaussians limit output, not peak GPU/RAM usage. Setup requires
        at least 8 GiB free; inference requires 3 GiB. These are preflight minimums,
        not hard storage quotas. No automatic giant ZIP is created.
    """)
    cell("code", f"""
        from pathlib import Path
        import importlib.util
        import os
        import shutil
        import subprocess
        import sys

        REPO_DIR = Path({base!r}) / "instant-nurec"
        NCORE_PATH = ""  # Absolute path to an existing local NCore JSON.
        CHECKPOINT = ""  # Existing instant_nurec_pa_front_1.1.0.pth.
        ALLOW_SETUP = False
        ALLOW_RENDER_SETUP = False
        RENDER_PREVIEW = False
        COMMIT = {COMMIT!r}
        REPOSITORY = {REPOSITORY!r}
        OUTPUT_DIR = REPO_DIR.parent / "instant-nurec-output"
    """)
    cell("code", """
        def run(args, **kwargs):
            return subprocess.run(
                [str(arg) for arg in args], check=True, **kwargs
            )

        def require_disk(path, gib):
            path = Path(path)
            while not path.exists():
                path = path.parent
            free = shutil.disk_usage(path).free
            if free < gib * 1024**3:
                raise RuntimeError(f"Need {gib} GiB free at {path}; have {free / 1024**3:.2f}")

        def local_file(value, suffix):
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError("Set NCORE_PATH and CHECKPOINT to existing absolute paths")
            path = Path(value).expanduser()
            if not path.is_absolute() or not path.is_file() or path.suffix.lower() != suffix:
                raise ValueError(f"Missing/invalid local {suffix} file: {value}")
            if path.stat().st_size == 0:
                raise ValueError(f"Empty asset: {path}")
            return path.resolve()

        def assets():
            ncore = local_file(NCORE_PATH, ".json")
            checkpoint = local_file(CHECKPOINT, ".pth")
            if checkpoint.name != "instant_nurec_pa_front_1.1.0.pth":
                raise ValueError("Expected the matching official pa-front 1.1.0 checkpoint")
            return ncore, checkpoint

        def verify_repo():
            if not (REPO_DIR / ".git").exists():
                raise RuntimeError("Repository missing; explicitly enable setup first")
            head = run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                       capture_output=True, text=True).stdout.strip()
            dirty = run(["git", "-C", REPO_DIR, "status", "--porcelain",
                         "--untracked-files=normal"], capture_output=True, text=True).stdout.strip()
            if head != COMMIT or dirty:
                raise RuntimeError("Existing repo must be clean at the pinned commit; no reset performed")
            for name in ("uv.lock", "pyproject.toml", "run_inference.py"):
                if not (REPO_DIR / name).is_file():
                    raise RuntimeError(f"Required upstream file missing: {name}")

        def setup_env():
            env = os.environ.copy()
            env["UV_PROJECT_ENVIRONMENT"] = str(REPO_DIR / ".venv")
            env["UV_CACHE_DIR"] = str(REPO_DIR.parent / ".nurec-uv-cache")
            env["UV_PYTHON_INSTALL_DIR"] = str(REPO_DIR.parent / ".nurec-python")
            env["MAX_JOBS"] = "2"
            return env
    """)
    cell("code", """
        if ALLOW_SETUP is not True:
            print("Setup skipped. Set ALLOW_SETUP=True only after reviewing licenses and disk usage.")
        else:
            assets()
            require_disk(REPO_DIR.parent, 8)
            if REPO_DIR.exists():
                verify_repo()
            else:
                run(["git", "init", REPO_DIR])
                run(["git", "-C", REPO_DIR, "remote", "add", "origin", REPOSITORY])
                run(["git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", COMMIT])
                run(["git", "-C", REPO_DIR, "checkout", "--detach", COMMIT])
                verify_repo()
            if importlib.util.find_spec("uv") is None:
                run([sys.executable, "-m", "pip", "install", "uv"])
            env = setup_env()
            if not (REPO_DIR / ".venv").exists():
                run([sys.executable, "-m", "uv", "venv", "--seed", "--python", "3.11",
                     REPO_DIR / ".venv"], env=env)
            run([sys.executable, "-m", "uv", "sync", "--frozen", "--python", "3.11",
                 "--no-dev", "--directory", REPO_DIR], env=env)
    """)
    cell("code", """
        if ALLOW_RENDER_SETUP is not True:
            print("Render setup skipped.")
        else:
            if ALLOW_SETUP is not True:
                raise RuntimeError("Render dependency downloads also require ALLOW_SETUP=True")
            verify_repo()
            require_disk(REPO_DIR.parent, 8)
            run([sys.executable, "-m", "uv", "sync", "--frozen", "--python", "3.11",
                 "--no-dev", "--directory", REPO_DIR, "--extra", "render"], env=setup_env())
    """)
    cell("code", """
        ncore, checkpoint = assets()
        verify_repo()
        require_disk(OUTPUT_DIR, 3)
        python = REPO_DIR / ".venv" / "bin" / "python"
        if not python.is_file():
            raise RuntimeError("Missing isolated Python; enable setup and execute preceding cells")
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   INSTANT_NUREC_FULL_PT=str(checkpoint), MAX_JOBS="2")
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        check = (
            "import sys, torch; "
            "assert sys.version_info[:2] == (3, 11), 'Python 3.11 required'; "
            "assert torch.cuda.is_available(), 'CUDA unavailable; enable GPU runtime'; "
            "print(torch.__version__, torch.cuda.get_device_name(0))"
        )
        run([python, "-c", check], cwd=REPO_DIR, env=env)
        if RENDER_PREVIEW:
            run([python, "-c", "import gsplat, nvdiffrast"],
                cwd=REPO_DIR, env=env)
        command = [
            python, "-u", "run_inference.py", "--model", "pa-front",
            "--ncore-path", ncore, "--output-dir", OUTPUT_DIR,
            "--merge", "--max-chunks", "1", "--n-gaussians", "200000",
        ]
        if RENDER_PREVIEW:
            command.append("--render-preview")
        run(command, cwd=REPO_DIR, env=env)
    """)
    cell("code", """
        from IPython.display import Image, display

        for preview in sorted(OUTPUT_DIR.rglob("*.render.png")):
            print(preview)
            display(Image(filename=str(preview)))
        for ply in sorted(OUTPUT_DIR.rglob("*.ply")):
            print(f"PLY: {ply} ({ply.stat().st_size / 1024**2:.1f} MiB)")
        print("Artifacts remain on disk; no ZIP or automatic upload.")
    """)
    return json.dumps({
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }, indent=2).encode("utf-8")
