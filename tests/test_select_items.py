"""select_items from the trainer, extracted with ast (the trainer needs torch and gsplat to import)."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "tools" / "colab_cli_kitti360.py"


def load(seq="s1", start=0, n=4, gap=5):
    tree = ast.parse(SRC.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "select_items")
    ns = {"SEQ": seq, "START": start, "N_FRAMES": n, "MAX_GAP": gap, "print": lambda *a, **k: None}
    exec(compile(ast.Module([fn], []), str(SRC), "exec"), ns)
    return ns["select_items"]


def items(seq, ids):
    return [{"frame_id": i, "lidar_path": f"{seq}/velodyne/{i:010d}.bin"} for i in ids]


def test_first_long_enough_run_and_gaps_break_runs():
    pool = items("s1", [2, 7, 12] + [100, 105, 110, 115, 120, 125] + [300]) + items("s2", [1, 6, 11, 16])
    official = {i for i in range(0, 400)}
    got = [x["frame_id"] for x in load("s1", 0, 4)(pool, official)]
    assert got == [100, 105, 110, 115], "first run with 4 frames, not the short one at the start"


def test_start_and_poses_and_short_sequences():
    pool = items("s1", [2, 7, 12, 17, 22, 27])
    assert [x["frame_id"] for x in load("s1", 12, 3)(pool, set(range(40)))] == [12, 17, 22]
    assert [x["frame_id"] for x in load("s1", 0, 3)(pool, {12, 17, 22, 27})] == [12, 17, 22], "frames without poses are skipped"
    short = load("s1", 0, 10)(pool, set(range(40)))
    assert len(short) == 6, "fewer frames than asked: the longest run is returned"


def test_huge_max_gap_reproduces_taking_the_first_frames_with_poses():
    pool = items("s1", [12, 17, 22, 157, 167, 172, 177, 182, 187, 192])
    official = set(range(300))
    assert [x["frame_id"] for x in load("s1", 0, 5)(pool, official)] == [167, 172, 177, 182, 187]
    assert [x["frame_id"] for x in load("s1", 0, 5, gap=10**6)(pool, official)] == [12, 17, 22, 157, 167]
