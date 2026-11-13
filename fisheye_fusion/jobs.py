"""Portable CPU preview jobs; neural launchers are not implemented."""
import io
import json
import zipfile
from pathlib import Path
from .manifest import validate_manifest


def make_bundle(manifest, root, platform):
    if platform not in ("colab", "kaggle"):
        raise ValueError("unknown platform")
    root = Path(root).resolve()
    validate_manifest(manifest, root)
    paths = {p for frame in manifest.frames for p in frame.images.values()}
    paths.update(frame.lidar for frame in manifest.frames if frame.lidar)
    if sum((root / p).stat().st_size for p in paths) > 200 * 1024**2:
        raise ValueError("MVP bundle limit: 200 MiB. Select a smaller sequence.")
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# CPU fusion preview\nNot a neural training job. CUDA/T4 is not used.\n",
            "Extract the bundle, then set the notebook working directory to its folder.\n",
        ]},
        {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": [
            "%pip install numpy==2.2.6 opencv-python-headless==4.13.0.92 pydantic==2.12.3"
        ]},
        {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": [
            "from pathlib import Path\n",
            "assert Path('manifest.json').is_file(), 'Extract bundle and set working directory first'\n",
            "!python -m fisheye_fusion.cli render manifest.json --output preview.png\n",
        ]},
        {"cell_type": "code", "execution_count": None, "outputs": [], "metadata": {}, "source": [
            "from IPython.display import Image, display\n", "display(Image(filename='preview.png'))"
        ]},
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"fusion-{index}"
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, cells=cells)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("manifest.json", manifest.model_dump_json(indent=2))
        for p in sorted(paths):
            out.write(root / p, Path(p).as_posix())
        package = Path(__file__).parent
        for name in ("__init__.py", "geometry.py", "models.py", "manifest.py", "fusion.py", "cli.py"):
            out.write(package / name, f"fisheye_fusion/{name}")
        out.writestr(f"run_{platform}.ipynb", json.dumps(notebook, indent=2))
        out.writestr("job.json", json.dumps({"runtime": platform, "operation": "cpu-preview", "neural_training": False}))
    return data.getvalue()
