"""Fast CPU tests for colmap_poses_da3_depth: depth-scale alignment, refusal, npz schema, determinism, parsing, truth-free AST."""
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

import colmap_poses_da3_depth as cp

HERE = Path(__file__).resolve().parent
W, H = 40, 30


def _synthetic(n_frames=3, true_scale=7.0, n_points=60, few_in_frame=None, registered=(0, 1, 2), seed=0):
    """A fronto-parallel world: DA3 depth = z_colmap / true_scale at the observed pixels (+ smooth elsewhere)."""
    rng = np.random.default_rng(seed)
    depth = np.full((n_frames, H, W), 2.0, np.float32)
    images, points = {}, {}
    pid = 0
    for k in range(n_frames):
        n = few_in_frame if (few_in_frame is not None and k == 1) else n_points
        cols = rng.integers(0, W, n)
        rows = rng.integers(0, H, n)
        z = rng.uniform(2.0, 6.0, n)
        pids = []
        for c, r, zz in zip(cols, rows, z):
            depth[k, r, c] = zz / true_scale
        # identity pose: world == camera frame, so a point's depth is its z; one 3D point per observation
        for zz in z:
            points[pid] = np.array([0.0, 0.0, zz])
            pids.append(pid)
            pid += 1
        if k in registered:
            images[k] = {"name": f"{k:05d}.png", "camera_id": 1, "w2c": np.concatenate([np.eye(3), [[0.1 * k], [0.0], [0.0]]], axis=1),
                         "xy": np.stack([cols + 0.5, rows + 0.5], axis=1).astype(float), "pid": np.asarray(pids, np.int64)}
            # the identity-rotation pose has translation t, so camera depth = z; keep z unchanged (t has no z component)
    cams = {1: {"model": "SIMPLE_PINHOLE", "width": W, "height": H, "params": [30.0, W / 2, H / 2]}}
    model = {"cameras": cams, "images": images, "points": points}
    da3 = {"depth": depth, "confidence": np.ones_like(depth), "images": np.zeros((n_frames, H, W, 3), np.uint8),
           "source_indices": np.arange(n_frames, dtype=np.int64)}
    return model, da3


class AlignDepthScaleTests(unittest.TestCase):
    def test_recovers_scale_on_synthetic_points(self):
        depth = np.full((H, W), 2.0)
        cols, rows = np.arange(40) % W, np.arange(40) % H
        z = depth[rows, cols] * 5.5
        r = cp.align_depth_scale(depth, cols, rows, z)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["scale"], 5.5, places=9)
        self.assertEqual(r["n_valid"], 40)

    def test_median_is_robust_to_outliers(self):
        depth = np.full((H, W), 2.0)
        cols, rows = np.arange(50) % W, np.arange(50) % H
        z = np.full(50, 11.0)
        z[:12] = 100.0                       # 24 % gross outliers
        r = cp.align_depth_scale(depth, cols, rows, z)
        self.assertAlmostEqual(r["scale"], 5.5, places=9)

    def test_refuses_with_too_few_points(self):
        depth = np.full((H, W), 2.0)
        r = cp.align_depth_scale(depth, np.arange(5), np.arange(5), np.full(5, 4.0))
        self.assertEqual(r["status"], "refused_too_few_points")
        self.assertIsNone(r["scale"])
        r0 = cp.align_depth_scale(depth, [], [], [])
        self.assertEqual(r0["status"], "refused_too_few_points")

    def test_invalid_depths_and_points_behind_camera_do_not_count(self):
        depth = np.full((H, W), 2.0)
        depth[0, :] = 0.0                    # no DA3 depth on row 0
        depth[1, :] = np.nan
        cols = np.arange(40) % W
        rows = np.array([0] * 10 + [1] * 10 + [2] * 20)
        z = np.full(40, 4.0)
        z[30:] = -1.0                        # behind the camera
        r = cp.align_depth_scale(depth, cols, rows, z, min_points=5)
        self.assertEqual(r["n_valid"], 10)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["scale"], 2.0)

    def test_depth_pixels_maps_between_resolutions_and_clips(self):
        cols, rows = cp.depth_pixels([[0.5, 0.5], [479.9, 359.9], [480.0, 360.0]], (480, 360), (476, 364))
        self.assertEqual(cols.tolist(), [0, 475, 475])
        self.assertEqual(rows.tolist(), [0, 363, 363])


class BuildPredictionTests(unittest.TestCase):
    def test_schema_scale_and_dropped_frames(self):
        model, da3 = _synthetic(n_frames=4, true_scale=7.0, registered=(0, 1, 3), few_in_frame=5)
        arrays, report = cp.build_prediction(model, da3, min_points=30)
        # frame 1 registered but only 5 points -> refused; frame 2 unregistered -> dropped
        self.assertEqual(arrays["source_indices"].tolist(), [0, 3])
        self.assertEqual({f["source_index"]: f["status"] for f in report["frames"]},
                         {0: "kept", 1: "dropped_refused_too_few_points", 2: "dropped_unregistered", 3: "kept"})
        self.assertEqual((report["frames_kept"], report["frames_dropped_unregistered"], report["frames_dropped_too_few_points"]), (2, 1, 1))
        self.assertEqual(sorted(arrays), ["confidence", "depth", "extrinsics", "images", "intrinsics", "source_indices"])
        self.assertEqual(arrays["depth"].shape, (2, H, W))
        self.assertEqual(arrays["extrinsics"].shape, (2, 3, 4))
        self.assertEqual(arrays["intrinsics"].shape, (2, 3, 3))
        self.assertEqual(arrays["confidence"].shape, (2, H, W))
        self.assertEqual(arrays["images"].shape, (2, H, W, 3))
        self.assertEqual(arrays["source_indices"].dtype, np.int64)
        for name in ("depth", "confidence", "intrinsics", "extrinsics"):
            self.assertEqual(arrays[name].dtype, np.float32)
        # DA3 depth at track pixels was z / 7, so the scaled depth there must be z, i.e. scale ~ 7
        scale = [f["scale"] for f in report["frames"] if f["status"] == "kept"]
        np.testing.assert_allclose(scale, 7.0, rtol=1e-6)
        # poses are COLMAP's, untouched
        np.testing.assert_allclose(arrays["extrinsics"][1], model["images"][3]["w2c"], atol=1e-6)
        # intrinsics: SIMPLE_PINHOLE rescaled (here to the same size)
        np.testing.assert_allclose(arrays["intrinsics"][0], [[30, 0, W / 2], [0, 30, H / 2], [0, 0, 1]], atol=1e-5)

    def test_nothing_registered_gives_empty_but_valid_arrays(self):
        model, da3 = _synthetic(n_frames=2, registered=())
        arrays, report = cp.build_prediction(model, da3)
        self.assertEqual(report["frames_kept"], 0)
        self.assertEqual(arrays["depth"].shape, (0, H, W))
        self.assertEqual(arrays["source_indices"].shape, (0,))

    def test_deterministic_bytes(self):
        model, da3 = _synthetic()
        with tempfile.TemporaryDirectory() as t:
            a = cp.write_prediction(Path(t) / "a" / "p.npz", cp.build_prediction(model, da3)[0])
            b = cp.write_prediction(Path(t) / "b" / "p.npz", cp.build_prediction(model, da3)[0])
            self.assertEqual(a, b)
            with np.load(Path(t) / "a" / "p.npz", allow_pickle=False) as d:
                self.assertEqual(sorted(d.files), ["confidence", "depth", "extrinsics", "images", "intrinsics", "source_indices"])

    def test_intrinsics_rescaled_to_depth_map(self):
        cam = {"model": "SIMPLE_PINHOLE", "width": 480, "height": 360, "params": [300.0, 240.0, 180.0]}
        k = cp.camera_intrinsics(cam, (476, 364))
        np.testing.assert_allclose(k, [[300 * 476 / 480, 0, 240 * 476 / 480], [0, 300 * 364 / 360, 180 * 364 / 360], [0, 0, 1]])
        with self.assertRaises(ValueError):
            cp.camera_intrinsics({"model": "OPENCV", "width": 1, "height": 1, "params": []}, (1, 1))


class ParseSparseModelTests(unittest.TestCase):
    def test_parses_poses_observations_and_points(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "cameras.txt").write_text("# c\n1 SIMPLE_PINHOLE 480 360 300 240 180\n", encoding="utf-8")
            (d / "images.txt").write_text("# i\n7 1 0 0 0 0.5 0 2 1 00012.png\n10.5 20.5 3 30.0 40.0 -1\n"
                                          "8 1 0 0 0 0 0 0 1 00013.png\n\n", encoding="utf-8")
            (d / "points3D.txt").write_text("# p\n3 1.0 2.0 3.0 9 9 9 0.5 7 10 8 11\n", encoding="utf-8")
            m = cp.parse_sparse_model(d)
        self.assertEqual(sorted(m["images"]), [12, 13])
        np.testing.assert_allclose(m["images"][12]["w2c"], np.concatenate([np.eye(3), [[0.5], [0.0], [2.0]]], axis=1))
        self.assertEqual(m["images"][12]["pid"].tolist(), [3, -1])
        self.assertEqual(m["images"][13]["xy"].shape, (0, 2))
        xy, z = cp.sparse_points_in_frame(m["images"][12], m["points"])
        np.testing.assert_allclose(xy, [[10.5, 20.5]])
        np.testing.assert_allclose(z, [5.0])            # z of the point in camera frame = 3 + 2
        self.assertEqual(m["stats"]["num_points3D"], 1)


class TruthFreeTests(unittest.TestCase):
    def _fn(self, name):
        tree = ast.parse((HERE / "colmap_poses_da3_depth.py").read_text(encoding="utf-8"))
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def _tokens(self, fn):
        body = list(fn.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        out = []
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name):
                    out.append(node.id)
                elif isinstance(node, ast.Attribute):
                    out.append(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    out.append(node.value)
                elif isinstance(node, ast.arg):
                    out.append(node.arg)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    out += [a.name for a in node.names] + ([node.module] if getattr(node, "module", None) else [])
        return out

    def test_truth_free_functions_name_no_truth_simulator_or_evaluator(self):
        for name in cp.TRUTH_FREE_FUNCTIONS:
            toks = " ".join(self._tokens(self._fn(name))).lower()
            for bad in cp.FORBIDDEN_TOKENS:
                self.assertNotIn(bad, toks, f"{name} mentions {bad!r}")

    def test_all_truth_free_functions_exist_and_no_module_level_truth_imports(self):
        tree = ast.parse((HERE / "colmap_poses_da3_depth.py").read_text(encoding="utf-8"))
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertTrue(set(cp.TRUTH_FREE_FUNCTIONS) <= defined)
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name for n in top for a in n.names} | {n.module for n in top if isinstance(n, ast.ImportFrom) and n.module}
        for bad in ("simulator", "evaluate_floorplan", "evaluation", "audit_camera_trajectory", "compare_backends", "capture_parallax",
                    "run_axis_candidates", "run_rgb_batch"):
            self.assertNotIn(bad, names)

    def test_no_random_sources_in_module(self):
        src = (HERE / "colmap_poses_da3_depth.py").read_text(encoding="utf-8")
        self.assertNotIn("random", src.replace("np.random.default_rng", ""))


class ResumeTests(unittest.TestCase):
    def test_finished_stages_are_skipped_and_failure_stops_the_run(self):
        calls = []
        saved = {k: getattr(cp, k) for k in ("stage_colmap", "stage_build", "stage_chain", "stage_freeze")}

        def fake(stage, status="ok"):
            def f(run_dir, *a, **k):
                calls.append(stage)
                return cp.write_receipt(run_dir, stage, status, 0.0, {}, {})
            return f
        try:
            with tempfile.TemporaryDirectory() as t:
                cp.stage_colmap, cp.stage_build, cp.stage_chain, cp.stage_freeze = fake("colmap"), fake("build"), fake("chain"), fake("freeze")
                s = cp.run_truth_free_stages(Path(t), Path(t), "x", Path("c"), {})
                self.assertEqual(calls, ["colmap", "build", "chain", "freeze"])
                self.assertEqual(set(s.values()), {"ok"})
                calls.clear()
                s = cp.run_truth_free_stages(Path(t), Path(t), "x", Path("c"), {})
                self.assertEqual(calls, [])
                self.assertEqual(set(s.values()), {"skipped"})
            with tempfile.TemporaryDirectory() as t:
                cp.stage_colmap, cp.stage_build = fake("colmap", "failed"), fake("build")
                calls.clear()
                s = cp.run_truth_free_stages(Path(t), Path(t), "x", Path("c"), {})
                self.assertEqual(calls, ["colmap"])
                self.assertEqual(s, {"colmap": "failed"})
        finally:
            for k, v in saved.items():
                setattr(cp, k, v)


if __name__ == "__main__":
    unittest.main()
