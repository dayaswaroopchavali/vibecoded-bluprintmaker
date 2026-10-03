"""Leakage and freeze-ordering tests for the simulated-LiDAR opening batch."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import run_opening_lidar_batch as batch

HERE = Path(__file__).resolve().parent
INFERENCE_MODULES = ["detect_openings_depth.py", "paired_wall_measurement.py", "motion_plane_axis.py",
                     "estimate_height.py", "extract_planes.py"]
FORBIDDEN_IMPORTS = {"simulator", "capture_lidar_sim", "evaluate_floorplan", "evaluation", "house_layouts",
                     "opening_capture_protocol", "score_opening_location_diagnostic"}
FORBIDDEN_TEXT = ("private_ground_truth", "scene.json", "ground_truth_scene")
PILOT = (HERE.parent.parent / "outputs" / "sim-benchmark" / "opening-capture-lidar-pilot"
         / "house_104" / "capture_00" / "prediction.npz")


def _imports(tree: ast.AST) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse((HERE / "run_opening_lidar_batch.py").read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


class StaticLeakageChecks(unittest.TestCase):
    def test_inference_modules_do_not_import_private_or_scoring_code(self):
        for module in INFERENCE_MODULES:
            source = (HERE / module).read_text(encoding="utf-8")
            self.assertFalse(_imports(ast.parse(source)) & FORBIDDEN_IMPORTS, module)
            for text in FORBIDDEN_TEXT:
                self.assertNotIn(text, source, (module, text))

    def test_freeze_functions_only_touch_public_run_fields(self):
        for name in ("_freeze_one", "_freeze_truth_free"):
            fn = _function(name)
            self.assertFalse(_imports(fn) & FORBIDDEN_IMPORTS)
            used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
            self.assertFalse(used & {"scene", "scene_file", "path_file", "_evaluate", "_opening_metrics",
                                    "_geometric_truth_wall", "evaluate_floorplan"})
            strings = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            self.assertFalse(strings & {"scene_file", "path_file", "width", "length", "private"})
            keys = {n.slice.value for n in ast.walk(fn)
                    if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                    and isinstance(n.slice.value, str) and isinstance(n.value, ast.Name) and n.value.id == "run"}
            self.assertLessEqual(keys, {"label", "prediction", "frames", "seed", "path_index"})


class FreezeOrdering(unittest.TestCase):
    def test_evaluate_refuses_private_truth_if_freeze_hash_mismatches(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            (tmp / "prediction_freeze.json").write_text("{}", encoding="utf-8")
            (tmp / "prediction_freeze.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
            opened = []
            with mock.patch.object(batch, "_read", lambda p: opened.append(p) or {}):
                with self.assertRaisesRegex(RuntimeError, "changed after freezing"):
                    batch._evaluate({"runs": [{"label": "a"}]},
                                    [{"label": "a", "scene_file": tmp / "scene.json"}], tmp)
            self.assertEqual(opened, [])

    def test_run_orders_freeze_before_evaluation(self):
        events = []
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            scene_dir = tmp / "private" / "scene_7"
            scene_dir.mkdir(parents=True)
            (scene_dir / "scene.json").write_text("{}")
            (scene_dir / "path_00.json").write_text("{}")

            def fake_capture(run, public_dir, args):
                public_dir.mkdir(parents=True)
                np.savez(public_dir / "prediction.npz", depth=np.zeros((2, 2, 2), np.float32))
                events.append("capture")
                return public_dir / "prediction.npz"

            def fake_freeze(records, output):
                events.append("freeze")
                return {"runs": [{"label": r["label"]} for r in records]}

            def fake_evaluate(freeze, records, output):
                events.append("evaluate")
                return {}

            args = mock.Mock(private_root=tmp / "private", output_dir=tmp / "out", max_paths=None, seeds=None,
                             reuse_public_root=None, resume=False)
            with mock.patch.object(batch, "_capture", fake_capture), \
                    mock.patch.object(batch, "_freeze_truth_free", fake_freeze), \
                    mock.patch.object(batch, "_evaluate", fake_evaluate):
                batch.run(args)
        self.assertEqual(events, ["capture", "freeze", "evaluate"])


class OracleWallAssociation(unittest.TestCase):
    def test_geometric_wall_association_uses_scene_frame(self):
        scene = {"width": 6.0, "length": 8.0}
        boundary = {"vertices_xy": [[-3, -4], [3, -4], [3, 4], [-3, 4]], "coordinate_origin_world": [0, 0, 0],
                    "axes_world": {"e1": [1, 0, 0], "e2": [0, 1, 0], "up": [0, 0, 1]}}
        self.assertEqual([batch._geometric_truth_wall(boundary, i, scene) for i in range(4)], [2, 1, 0, 3])
        boundary["vertices_xy"] = [[10, 10], [20, 10], [20, 20], [10, 20]]
        self.assertIsNone(batch._geometric_truth_wall(boundary, 0, scene))


@unittest.skipUnless(PILOT.is_file(), "pilot public capture not present")
class RealCaptureDataFlow(unittest.TestCase):
    def test_inference_never_opens_private_files(self):
        opened = []

        def hook(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes, Path)):
                text = str(args[0]).replace("\\", "/")
                if "private_ground_truth" in text or text.endswith("scene.json"):
                    opened.append(text)

        sys.addaudithook(hook)
        with np.load(PILOT) as data:
            frames = int(len(data["depth"]))
        run = {"label": "audit", "seed": 104, "path_index": 0, "prediction": PILOT, "frames": frames,
               "scene_file": Path("private_ground_truth/scene_104/scene.json")}
        with tempfile.TemporaryDirectory() as d:
            row = batch._freeze_one(run, Path(d))
        self.assertEqual(opened, [])
        self.assertEqual(row["candidate_status"], "ok")
        self.assertTrue(row["openings"]["truth_free"])


if __name__ == "__main__":
    unittest.main()
