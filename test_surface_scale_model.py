"""Tests for surface_scale_model.py.

Synthetic rooms are rendered HERE from the simulator (truth exists only in this test file) with injected, known inconsistencies:
a ceiling depth offset, per-frame depth scale errors and a focal error. The module must recover them from the prediction alone,
return the identity on unbiased input, refuse parameters it cannot identify, never alter poses and be deterministic.
"""
import ast
import inspect
import unittest
from pathlib import Path

import numpy as np

import simulator
import surface_scale_model as ssm

HERE = Path(__file__).resolve().parent
CEILING_GAIN = 0.95
SCALE_SD = 0.02
FX_ERR, FY_ERR = 1.02, 0.985
AXIS = [0.0, 0.0, 1.0]            # the simulator's vertical, passed in as a stand-in for a geometry_orientation hypothesis
_CACHE = {}


def render_room(seed=17, frames=24, width=200, height=150, fov=78.0, mode="bounded", capture_seed=5):
    scene = simulator.make_scene(seed)
    poses = simulator.make_paths(scene, 120, capture_seed, mode)[0]
    poses = [poses[i] for i in np.linspace(0, 119, frames, dtype=int)]
    depth, kinds, K, E, images = [], [], [], [], []
    yy, xx = np.mgrid[0:height, 0:width]
    for pose in poses:
        image, ray, intr, kind, _ = simulator.render(scene, pose, width, height, fov, True, True)
        depth.append(ray / np.sqrt(1 + ((xx - intr["cx"]) / intr["fx"]) ** 2 + ((yy - intr["cy"]) / intr["fy"]) ** 2))
        kinds.append(kind); images.append(image)
        right, up, forward = simulator.camera_basis(pose)
        R = np.stack([right, -up, forward])
        E.append(np.c_[R, -R @ np.asarray(pose["position"])])
        K.append(np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]))
    depth = np.array(depth)
    return {"depth": np.where(np.isfinite(depth), depth, 0.0), "kind": np.array(kinds), "K": np.array(K), "E": np.array(E),
            "images": np.array(images)}


def room():
    if "room" not in _CACHE:
        _CACHE["room"] = render_room()
    return _CACHE["room"]


def prediction(depth, K, r, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    d = depth * np.exp(rng.normal(0, noise, depth.shape)) if noise else depth
    return {"depth": d.astype(np.float32), "confidence": (r["depth"] > 0).astype(np.float32), "intrinsics": K, "extrinsics": r["E"],
            "images": r["images"]}


def biased():
    if "biased" not in _CACHE:
        r = room()
        rng = np.random.default_rng(1)
        scale = np.exp(rng.normal(0, SCALE_SD, len(r["depth"])))
        d = r["depth"] * scale[:, None, None]
        d = np.where(r["kind"] == 1, d * CEILING_GAIN, d)
        K = r["K"].copy(); K[:, 0, 0] *= FX_ERR; K[:, 1, 1] *= FY_ERR
        _CACHE["biased"] = (prediction(d, K, r, seed=2), scale)
    return _CACHE["biased"]


def corrected_biased():
    if "cor" not in _CACHE:
        pred, _ = biased()
        _CACHE["cor"] = ssm.correct_prediction(pred, AXIS, sample_stride=5)
    return _CACHE["cor"]


def ceiling_over_wall(depth_pred, r):
    ratio = depth_pred / np.where(r["depth"] > 0, r["depth"], np.nan)
    ceil = np.nanmedian(np.where(r["kind"] == 1, ratio, np.nan))
    wall = np.nanmedian(np.where(r["kind"] >= 2, ratio, np.nan))
    return float(ceil / wall)


class RecoveryTests(unittest.TestCase):
    def test_frame_scales_recovered_gauge_fixed_and_with_uncertainty(self):
        r = room(); pred, scale = biased()
        out = ssm.estimate_frame_scales(pred, K=r["K"], sample_stride=5)
        self.assertEqual(out["status"], "ok")
        truth = -np.log(scale); truth -= truth.mean()
        est = np.array(out["log_scale_gauged"])
        self.assertAlmostEqual(float(est.mean()), 0.0, places=9)                           # gauge fixed
        self.assertLess(float(np.abs(est - truth).max()), 0.006)                           # injected sd is 2 %
        sd = np.array(out["log_scale_sd_leave_one_view_out"])
        self.assertTrue(np.isfinite(sd).all()); self.assertEqual(len(sd), len(est))
        applied = np.array(out["applied"]); to_apply = np.array(out["log_scale_to_apply"])
        np.testing.assert_array_equal(to_apply[~applied], 0.0)                              # unidentified frames get the identity
        self.assertTrue(np.all(sd[applied] <= np.abs(est[applied]) + 1e-12))
        self.assertGreater(applied.sum(), len(applied) // 2)

    def test_full_correction_recovers_all_three_injected_inconsistencies(self):
        r = room(); pred, scale = biased()
        cor, rep = corrected_biased()
        # poses and every non-depth array are returned untouched
        np.testing.assert_array_equal(cor["extrinsics"], pred["extrinsics"])
        np.testing.assert_array_equal(cor["confidence"], pred["confidence"])
        self.assertFalse(rep["poses_modified"])
        self.assertEqual(set(cor), set(pred))
        # (c) focal: the injected factors are 1.02 / 0.985 on the true focal length
        ratio = cor["intrinsics"][:, 0, 0] / r["K"][:, 0, 0]
        self.assertLess(abs(float(np.median(ratio)) - 1.0), 0.008)
        self.assertLess(abs(float(np.median(cor["intrinsics"][:, 1, 1] / r["K"][:, 1, 1])) - 1.0), 0.008)
        foc = rep["intrinsics"]
        self.assertEqual(len(foc["sd_leave_one_view_out"]), 2); self.assertTrue(all(np.isfinite(foc["sd_leave_one_view_out"])))
        # (a) per-frame scales: residual per-frame error is far below the injected 2 %
        sc = rep["frame_scales"]
        truth = -np.log(scale); truth -= truth.mean()
        self.assertLess(float(np.abs(np.array(sc["log_scale_to_apply"]) - truth).max()), 0.006)
        # (b) ceiling offset relative to the walls: 5 % injected, at least half removed
        before = ceiling_over_wall(pred["depth"].astype(float), r)
        after = ceiling_over_wall(cor["depth"].astype(float), r)
        self.assertAlmostEqual(before, CEILING_GAIN, places=2)
        self.assertLess(abs(after - 1.0), 0.5 * abs(before - 1.0))
        gains = rep["surface_gains"]["iterations"][0]["gains"]
        self.assertTrue(any(g["applied"] for g in gains))
        for g in gains:
            if g["applied"]:
                self.assertLessEqual(g["sd_relative"], abs(g["eps_relative_to_walls"]))      # applied only when sd <= |value|
        self.assertEqual(rep["uses_ground_truth"], False)

    def test_unbiased_input_gives_near_identity(self):
        r = room()
        pred = prediction(r["depth"], r["K"], r, seed=3)
        cor, rep = ssm.correct_prediction(pred, AXIS, sample_stride=5)
        np.testing.assert_array_equal(cor["extrinsics"], pred["extrinsics"])
        self.assertLess(rep["summary"]["median_abs_log_depth_change"], 0.004)
        self.assertLess(float(np.abs(cor["intrinsics"][:, 0, 0] / r["K"][:, 0, 0] - 1).max()), 0.004)
        self.assertLess(float(np.abs(cor["intrinsics"][:, 1, 1] / r["K"][:, 1, 1] - 1).max()), 0.004)
        self.assertLess(float(np.abs(np.log(cor["depth"][r["depth"] > 0] / pred["depth"][r["depth"] > 0])).max()), 0.02)
        self.assertLess(abs(ceiling_over_wall(cor["depth"].astype(float), r) - 1.0), 0.01)


class RefusalAndDeterminismTests(unittest.TestCase):
    def test_refuse_factor_zero_applies_nothing(self):
        r = room(); pred, _ = biased()
        sc = ssm.estimate_frame_scales(pred, K=r["K"], sample_stride=5, refuse_factor=0.0)
        self.assertFalse(any(sc["applied"])); np.testing.assert_array_equal(sc["log_scale_to_apply"], 0.0)
        self.assertEqual(len(sc["not_identifiable_frames"]), len(sc["applied"]))
        foc = ssm.estimate_intrinsics(pred, sample_stride=9, refuse_factor=0.0, loo=True)
        self.assertEqual(foc["to_apply"], [0.0, 0.0]); self.assertEqual(sorted(foc["not_identifiable"]), ["log_fx_factor", "log_fy_factor"])

    def test_two_views_or_no_overlap_are_not_identifiable(self):
        r = room(); pred, _ = biased()
        two = {k: (v[:2] if isinstance(v, np.ndarray) else v) for k, v in pred.items()}
        self.assertEqual(ssm.estimate_frame_scales(two, K=r["K"][:2], sample_stride=5)["status"], "not_identifiable")
        self.assertEqual(ssm.estimate_intrinsics(two, sample_stride=9)["status"], "not_identifiable")
        cor, rep = ssm.correct_prediction(two, AXIS, sample_stride=5)
        np.testing.assert_array_equal(cor["depth"], two["depth"]); np.testing.assert_array_equal(cor["intrinsics"], two["intrinsics"])
        self.assertIn("frame_scales", rep["not_identifiable"])

    def test_surfaces_are_not_identifiable_without_images_or_texture(self):
        r = room(); pred, _ = biased()
        no_img = {k: v for k, v in pred.items() if k != "images"}
        out = ssm.estimate_surface_gains(no_img, AXIS)
        self.assertEqual(out["gains"], []); self.assertEqual(out["reason"], "no_images_for_photometric_plane_sweep")
        flat = dict(pred); flat["images"] = np.full(pred["images"].shape, 0.5, np.float32)       # no texture: no photometric evidence
        out = ssm.estimate_surface_gains(flat, AXIS)
        self.assertTrue(out["gains"])
        self.assertFalse(any(g["applied"] for g in out["gains"]))
        cor, rep = ssm.correct_prediction(flat, AXIS, estimate_scales=False, estimate_focal=False, sample_stride=5)
        np.testing.assert_array_equal(cor["depth"], flat["depth"])

    def test_determinism(self):
        r = room(); pred, _ = biased()
        a = ssm.estimate_frame_scales(pred, K=r["K"], sample_stride=6, loo=False)
        b = ssm.estimate_frame_scales(pred, K=r["K"], sample_stride=6, loo=False)
        self.assertEqual(a["log_scale_gauged"], b["log_scale_gauged"])
        c = ssm.estimate_surface_gains(pred, AXIS); d = ssm.estimate_surface_gains(pred, AXIS)
        self.assertEqual([g.get("eps_relative_to_walls") for g in c["gains"]], [g.get("eps_relative_to_walls") for g in d["gains"]])
        f1 = ssm.estimate_intrinsics(pred, sample_stride=9, loo=False); f2 = ssm.estimate_intrinsics(pred, sample_stride=9, loo=False)
        self.assertEqual(f1["estimate"], f2["estimate"])

    def test_apply_touches_only_gained_cap_pixels_and_never_poses(self):
        n, h, w = 2, 30, 30
        depth = np.full((n, h, w), 2.0)
        labels = np.full((n, 10, 10), -1, np.int32)
        labels[:, :4, :] = 0            # a gained cap plane
        labels[:, 6:, :] = 1            # a wall plane, never touched
        E = np.stack([np.c_[np.eye(3), [0, 0, 0.0]]] * 2); E[1, :, 3] = [0.1, 0, 0]
        K = np.stack([np.array([[30., 0, 15], [0, 30., 15], [0, 0, 1]])] * 2)
        rows = [{"normal": [0, 0, 1.0], "offset": -3.0}, {"normal": [1.0, 0, 0], "offset": -3.0}]
        gains = [{"plane_index": 0, "applied": True, "shift_fraction_to_apply": 0.1}]
        out = ssm.apply_surface_gains(depth, labels, gains, rows, None, K, E)
        self.assertTrue(np.allclose(out[:, 15:, :], 2.0))                                   # unlabelled-beyond-one-cell and wall pixels untouched
        # plane z = 3 seen from z = 0 moved away by 10 % of the 3 m distance: ratio (3 + 0.3) / 3 on cap pixels and one grown cell
        self.assertTrue(np.allclose(np.unique(np.round(out[:, :15, :] / 2.0, 6)), [1.1], atol=1e-5))


class CliTests(unittest.TestCase):
    def test_estimate_writes_new_folder_npz_and_reports_and_never_overwrites(self):
        import subprocess
        import sys
        import tempfile
        pred, _ = biased()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pred.npz"
            np.savez_compressed(src, **pred, source_indices=np.arange(len(pred["depth"])))
            out = Path(tmp) / "out"
            cmd = [sys.executable, str(HERE / "run_surface_scale_model.py"), "estimate", "--prediction", str(src), "--output-dir", str(out), "--axis", "0,0,1"]
            done = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertTrue((out / "pred" / "corrected_hyp0.npz").exists() and (out / "pred" / "report_hyp0.json").exists())
            self.assertTrue((out / "report.md").exists() and (out / "manifest.json").exists())
            with np.load(out / "pred" / "corrected_hyp0.npz") as c, np.load(src) as s:
                np.testing.assert_array_equal(c["extrinsics"], s["extrinsics"]); np.testing.assert_array_equal(c["source_indices"], s["source_indices"])
            self.assertNotEqual(subprocess.run(cmd, capture_output=True, text=True).returncode, 0)        # existing folder is refused


class StaticTests(unittest.TestCase):
    FORBIDDEN_IMPORTS = {"simulator", "evaluation", "evaluate_floorplan", "depth_bias_analysis", "compare_wall_boundaries", "blender_capture",
                         "house_layouts", "audit_camera_trajectory", "estimate_height", "analyze_floorplan_error", "orient_room",
                         "gyro_stabilize", "virtual_gyro", "hybrid_pose_depth", "capture_parallax", "evaluate_windowed"}
    ALLOWED_LOCAL = {"extract_planes"}
    FORBIDDEN_WORDS = ("ground_truth", "groundtruth", "gravity", "camera_up", "up_vector", "scene_json", "private_")

    def _tree(self):
        return ast.parse((HERE / "surface_scale_model.py").read_text(encoding="utf-8"))

    def test_no_forbidden_imports(self):
        mods = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertFalse(mods & self.FORBIDDEN_IMPORTS, mods & self.FORBIDDEN_IMPORTS)
        local = {p.stem for p in HERE.glob("*.py")}
        self.assertTrue((mods & local) <= self.ALLOWED_LOCAL, mods & local)

    def test_no_truth_camera_up_or_gravity_names_in_code(self):
        names = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
        bad = [n for n in names if any(word in n.lower() for word in self.FORBIDDEN_WORDS)]
        self.assertEqual(bad, [])

    def test_api_has_no_truth_parameters_and_reads_no_files_but_npz(self):
        for fn in (ssm.correct_prediction, ssm.estimate_frame_scales, ssm.estimate_intrinsics, ssm.estimate_surface_gains):
            for p in inspect.signature(fn).parameters:
                self.assertFalse(any(word in p for word in ("truth", "scene", "gt", "simulator", "up")), (fn.__name__, p))
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(self._tree()) if isinstance(n, ast.Call)}
        self.assertFalse(calls & {"open", "read_text", "read_bytes", "loads", "read_csv"})


if __name__ == "__main__":
    unittest.main()
