import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fisheye_fusion.nurec_notebook import COMMIT, make_nurec_notebook


def codes(platform="colab"):
    notebook = json.loads(make_nurec_notebook(platform))
    return [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize("platform,base", [("colab", "/content"), ("kaggle", "/kaggle/working")])
def test_notebook_is_executable_python(platform, base):
    data = make_nurec_notebook(platform)
    assert isinstance(data, bytes)
    notebook = json.loads(data)
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    for source in codes(platform):
        ast.parse(source)
        compile(source, "<notebook>", "exec")
    namespace = {}
    exec(codes(platform)[0], namespace)
    assert namespace["REPO_DIR"] == Path(base) / "instant-nurec"
    assert namespace["ALLOW_SETUP"] is False
    assert namespace["ALLOW_RENDER_SETUP"] is False
    assert namespace["RENDER_PREVIEW"] is False


def test_default_cells_never_download_or_launch():
    namespace = {}
    with patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")):
        for source in codes()[:4]:
            exec(source, namespace)
        with pytest.raises(ValueError, match="existing absolute paths"):
            exec(codes()[4], namespace)


def test_pinned_offline_commands_and_limits():
    source = "\n".join(codes())
    assert COMMIT == "f2ecafcdf28f2dc8f2cc17447a772571cde558be"
    for expected in [
        COMMIT, "--frozen", "--seed", '"3.11"', '"--extra", "render"',
        'HF_HUB_OFFLINE="1"', 'TRANSFORMERS_OFFLINE="1"',
        "INSTANT_NUREC_FULL_PT=str(checkpoint)", 'MAX_JOBS="2"',
        'require_disk(REPO_DIR.parent, 8)', 'require_disk(OUTPUT_DIR, 3)',
        '"--max-chunks", "1"', '"--n-gaussians", "200000"',
        '"--model", "pa-front"', "torch.cuda.is_available()",
    ]:
        assert expected in source
    assert "shell=True" not in source
    assert "reset" not in source.replace("no reset performed", "")
    assert "hf_hub_download" not in source
    assert "ZipFile" not in source


def test_assets_require_existing_matching_files(tmp_path):
    namespace = {}
    for source in codes()[:2]:
        exec(source, namespace)
    ncore = tmp_path / "clip.json"
    checkpoint = tmp_path / "instant_nurec_pa_front_1.1.0.pth"
    ncore.write_text("{}")
    checkpoint.write_bytes(b"fixture")
    namespace.update(NCORE_PATH=str(ncore), CHECKPOINT=str(checkpoint))
    assert namespace["assets"]() == (ncore.resolve(), checkpoint.resolve())
    namespace["CHECKPOINT"] = str(tmp_path / "missing.pth")
    with pytest.raises(ValueError, match="Missing/invalid"):
        namespace["assets"]()
    wrong = tmp_path / "wrong.pth"
    wrong.write_bytes(b"fixture")
    namespace["CHECKPOINT"] = str(wrong)
    with pytest.raises(ValueError, match="matching"):
        namespace["assets"]()


def test_disk_preflight_rejects_insufficient_space():
    namespace = {}
    for source in codes()[:2]:
        exec(source, namespace)
    with patch("shutil.disk_usage") as usage:
        usage.return_value.free = 1024**3
        with pytest.raises(RuntimeError, match="Need 8 GiB"):
            namespace["require_disk"](Path.cwd(), 8)


def test_unknown_platform_rejected():
    with pytest.raises(ValueError, match="platform"):
        make_nurec_notebook("windows")
