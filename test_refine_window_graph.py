"""refine_window_graph: synthetic windows with known transforms (truth only inside these tests, never in the module)."""
import ast
import contextlib
import copy
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import refine_window_graph as R
import windowed_reconstruction as W
from test_windowed_reconstruction import frames, make_windows, umeyama_align

P = dict(W.PARAMS)


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def perturb(links, seed, rot=0.012, scale=0.01, trans=0.01, only=None, extra_rot_deg=0.0):
    """Noisy copy of the accepted links (the link stage itself is exact on synthetic data)."""
    rng = np.random.default_rng(seed); out = []
    for l in links:
        l = copy.deepcopy(l)
        if l["ok"]:
            big = only is not None and (l["windows"][0] in only or l["windows"][1] in only)
            w = rng.normal(size=3) * rot
            if big:
                a = rng.normal(size=3); w = a / np.linalg.norm(a) * np.radians(extra_rot_deg)
            l["Q"] = (np.asarray(l["Q"]) @ exp_so3(w)).tolist()
            l["s"] = float(l["s"] * np.exp(rng.normal() * scale))
            l["b"] = (np.asarray(l["b"]) + rng.normal(size=3) * trans * l["extent"]).tolist()
        out.append(l)
    return out


def centre_error(pred, truth_frames):
    order = [list(truth_frames["source_indices"]).index(i) for i in pred["source_indices"]]
    truth = W.camera_center(truth_frames["extrinsics"])[order]; got = W.camera_center(pred["extrinsics"])
    s, Q, b = umeyama_align(got, truth)
    return float(np.sqrt(((s * got @ Q.T + b - truth) ** 2).sum(1).mean()) / np.linalg.norm(np.ptp(truth, axis=0)))


_CACHE = {}


def fixture():
    if "x" not in _CACHE:
        f = frames(); wins = make_windows(f, 6, 1, seed=21)
        base_pred, rep = W.merge_window_graph(wins, max_neighbors=3, prune_cycles=True)
        _CACHE["x"] = (f, wins, base_pred, rep)
    return _CACHE["x"]


class RefinementTests(unittest.TestCase):
    def test_refined_has_lower_truth_free_residuals_and_is_closer_to_truth_than_the_tree(self):
        f, wins, base_pred, rep = fixture()
        links = perturb(rep["links"], 1)
        out = R.refine_links(wins, links, P); r = out["report"]
        self.assertGreater(r["non_tree_links"], 10)
        self.assertLess(r["residuals_after_refined"]["all_links"]["weighted_mean_squared_residual"],
                        r["residuals_before_tree"]["all_links"]["weighted_mean_squared_residual"])
        self.assertLess(r["cycle_consistency_after"]["max_rotation_deg"], r["cycle_consistency_before"]["max_rotation_deg"])
        tree_pred = W._assemble(wins, out["component"], out["tree_transforms"])
        e_tree, e_ref = centre_error(tree_pred, f), centre_error(out["prediction"], f)
        self.assertLess(e_ref, e_tree)
        self.assertLess(r["objective_after"], r["objective_before"])
        # truth-free multi-window disagreement also improves
        s = r["per_frame_summary"]
        self.assertLess(s["multiwindow_rotation_deg_refined"]["mean"], s["multiwindow_rotation_deg_tree"]["mean"])

    def test_no_reflections_identity_components_and_source_indices_preserved(self):
        f, wins, base_pred, rep = fixture()
        out = R.refine_links(wins, perturb(rep["links"], 2), P); pred = out["prediction"]
        np.testing.assert_array_equal(pred["source_indices"], base_pred["source_indices"])
        np.testing.assert_array_equal(pred["component_ids"], base_pred["component_ids"])
        np.testing.assert_array_equal(pred["source_window"], base_pred["source_window"])        # same most-central rule
        for T in out["refined_transforms"]:
            self.assertAlmostEqual(np.linalg.det(T[1]), 1.0, places=9); self.assertGreater(T[0], 0)
        det = np.linalg.det(pred["extrinsics"][:, :, :3]); src = np.linalg.det(base_pred["extrinsics"][:, :, :3])
        np.testing.assert_allclose(det, src, atol=1e-8)                                         # inversion/roll (det sign) untouched
        for k in ("images", "intrinsics", "confidence"):
            np.testing.assert_array_equal(pred[k], base_pred[k])

    def test_depth_and_cameras_move_consistently(self):
        f, wins, base_pred, rep = fixture()
        out = R.refine_links(wins, perturb(rep["links"], 3), P); pred = out["prediction"]
        for row in (0, 7, 20):
            w = int(pred["source_window"][row]); w = [i for i in range(len(wins)) if out["component"][i] >= 0][w]; k =list(wins[w]["source_indices"]).index(int(pred["source_indices"][row]))
            s, Q, b = out["refined_transforms"][w]
            p0, _ = W.frame_points(wins[w], k, P)
            one = {n: pred[n][row:row + 1] for n in ("depth", "confidence", "intrinsics", "extrinsics")}
            p1, _ = W.frame_points(one, 0, P)
            np.testing.assert_allclose(p1, s * p0 @ Q.T + b, atol=1e-7 * max(1.0, np.abs(p1).max()))

    def test_corrupted_links_of_one_window_are_downweighted(self):
        f, wins, base_pred, rep = fixture()
        links = perturb(rep["links"], 4, rot=0.003, scale=0.003, trans=0.003, only={9}, extra_rot_deg=8.0)
        r = R.refine_links(wins, links, P)["report"]
        bad = [d["robust_weight_final"] for d in r["link_details"] if 9 in d["windows"]]
        good = [d["robust_weight_final"] for d in r["link_details"] if 9 not in d["windows"]]
        self.assertLess(np.median(bad), np.median(good))
        self.assertLess(np.mean(bad), 0.7 * np.mean(good))

    def test_tree_only_graph_has_nothing_to_refine_and_equals_tree(self):
        f = frames(); wins = make_windows(f, 6, 3, seed=22)
        pred, rep = W.merge_window_graph(wins, max_neighbors=1)
        out = R.refine_links(wins, rep["links"], P)
        self.assertEqual(out["report"]["non_tree_links"], 0)
        for a, b in zip(out["tree_transforms"], out["refined_transforms"]):
            np.testing.assert_allclose(a[0], b[0], rtol=1e-6); np.testing.assert_allclose(a[1], b[1], atol=1e-6); np.testing.assert_allclose(a[2], b[2], atol=1e-6)
        np.testing.assert_array_equal(out["prediction"]["source_indices"], pred["source_indices"])

    def test_disconnected_windows_stay_separate_components(self):
        f = frames(); a = make_windows(f, 6, 1, seed=23)[:6]; b = make_windows(f, 6, 1, seed=24)[12:18]
        wins = a + b; pred, rep = W.merge_window_graph(wins, max_neighbors=3)
        out = R.refine_links(wins, perturb(rep["links"], 5), P); p = out["prediction"]
        self.assertEqual(sorted(set(p["component_ids"].tolist())), [0, 1]); self.assertEqual(out["report"]["components"], 2)
        np.testing.assert_array_equal(p["source_indices"], pred["source_indices"]); np.testing.assert_array_equal(p["component_ids"], pred["component_ids"])
        self.assertLess(out["report"]["retained_capture_fraction"], 1.0)
        self.assertFalse(any(set(d["windows"]) & set(range(6)) and set(d["windows"]) & set(range(6, 12)) for d in out["report"]["link_details"]))

    def test_isolated_corrupt_window_is_excluded_not_bridged(self):
        f = frames(); wins = make_windows(f, 6, 1, seed=25)
        wins[9] = dict(wins[9], depth=np.random.default_rng(1).uniform(.5, 5, wins[9]["depth"].shape))
        pred, rep = W.merge_window_graph(wins, max_neighbors=3)
        out = R.refine_links(wins, rep["links"], P)
        self.assertIn(9, out["report"]["isolated_windows"]); self.assertEqual(out["report"]["windows_used"], len(wins) - 1)
        np.testing.assert_array_equal(out["prediction"]["source_indices"], pred["source_indices"])
        np.testing.assert_array_equal(out["prediction"]["source_window"], pred["source_window"])

    def test_deterministic(self):
        f, wins, base_pred, rep = fixture(); links = perturb(rep["links"], 6)
        a = R.refine_links(wins, links, P); b = R.refine_links(wins, links, P)
        for k in a["prediction"]:
            np.testing.assert_array_equal(a["prediction"][k], b["prediction"][k])

    def test_exact_links_are_a_fixed_point(self):
        f, wins, base_pred, rep = fixture()
        out = R.refine_links(wins, rep["links"], P)
        self.assertLess(out["report"]["per_window_correction_summary"]["rotation_deg"]["max"], 1e-3)


class WeightTests(unittest.TestCase):
    def test_weight_is_scale_free_monotone_and_uses_only_link_diagnostics(self):
        good = {"rms_fraction_of_extent": 0.002, "max_rotation_disagreement_deg": 0.5, "inlier_fraction": 0.95}
        worse = dict(good, rms_fraction_of_extent=0.02); rot = dict(good, max_rotation_disagreement_deg=8.0); inl = dict(good, inlier_fraction=0.55)
        wg = R.link_weight(good, P)[0]
        for l in (worse, rot, inl):
            self.assertLess(R.link_weight(l, P)[0], wg)
        self.assertLessEqual(wg, 1 / R.WEIGHT_FLOOR)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name)
        f, wins, base_pred, rep = fixture()
        pool = W.WindowPool(root / "pool", 64); groups = [[int(i) for i in w["source_indices"]] for w in wins]
        for g, w in zip(groups, wins):
            pool.put(g, w)
        (root / "base").mkdir(); np.savez_compressed(root / "base" / "prediction.npz", **base_pred)
        merge = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.items()}, "shared_intrinsics": None,
                 "cycle_pruning": {"enabled": True, "gates": dict(W.PRUNE)}, "links": rep["links"], "cycle_consistency": rep["cycle_consistency"]}
        (root / "base" / "manifest.json").write_text(json.dumps({"resolution": 64, "window_groups": groups, "merge": merge}, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_cli_runs_reproduces_baseline_writes_manifest_and_refuses_overwrite(self):
        out = self.root / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(R.main(["--window-pool", str(self.root / "pool"), "--baseline", str(self.root / "base"), "--output", str(out)]), 0)
        m = json.loads((out / "manifest.json").read_text())
        for key in ("inputs", "parameters", "links_accepted", "links_rejected", "baseline_verification", "refinement", "runtime_seconds", "tree_transforms", "refined_transforms"):
            self.assertIn(key, m)
        self.assertTrue(m["baseline_verification"]["matches"]); self.assertTrue(m["truth_free"]); self.assertFalse(m["ground_truth_read"])
        r = m["refinement"]
        for key in ("residuals_before_tree", "residuals_after_refined", "objective_before", "objective_after", "cycle_consistency_before",
                    "cycle_consistency_after", "outlier_links", "per_window_correction_vs_tree", "per_frame", "component_frame_counts",
                    "retained_capture_fraction", "link_details", "weights"):
            self.assertIn(key, r)
        self.assertEqual(len(m["inputs"]["pool_files"]), len(m["window_groups"]))
        self.assertIn("rotation_deg", r["link_details"][0]["before"]); self.assertIn("translation_fraction_of_extent", r["link_details"][0]["after"])
        self.assertTrue((out / "prediction.npz").exists() and (out / "largest_component.npz").exists())
        with np.load(out / "prediction.npz") as d, np.load(self.root / "base" / "prediction.npz") as b:
            np.testing.assert_array_equal(d["source_indices"], b["source_indices"]); self.assertEqual(set(d.files), set(b.files))
        before = (out / "prediction.npz").read_bytes()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            R.main(["--window-pool", str(self.root / "pool"), "--baseline", str(self.root / "base"), "--output", str(out)])
        with self.assertRaises(FileExistsError):
            R.run(self.root / "pool", self.root / "base", out)
        self.assertEqual((out / "prediction.npz").read_bytes(), before)

    def test_a_baseline_that_does_not_match_is_refused_before_anything_is_written(self):
        f, wins, base_pred, rep = fixture()
        bad = self.root / "badbase"; bad.mkdir()
        tweaked = dict(base_pred); tweaked["extrinsics"] = base_pred["extrinsics"] + 1e-3
        np.savez_compressed(bad / "prediction.npz", **tweaked)
        (bad / "manifest.json").write_text((self.root / "base" / "manifest.json").read_text())
        with self.assertRaises(RuntimeError):
            R.run(self.root / "pool", bad, self.root / "never")
        self.assertFalse((self.root / "never").exists())

    def test_missing_pool_window_is_an_error_not_an_inference(self):
        with self.assertRaises(FileNotFoundError):
            R.load_pool_windows(self.root / "pool", [[900, 901, 902]], 64)


class TruthFreeTests(unittest.TestCase):
    def test_module_has_no_truth_simulator_audit_or_evaluation_import(self):
        src = inspect.getsource(R); tree = ast.parse(src)
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertTrue(imported.isdisjoint({"simulator", "audit_camera_trajectory", "evaluation", "house_layouts", "evaluate_windowed",
                                             "evaluate_floorplan", "analyze_floorplan_error", "blender_capture"}))
        self.assertNotIn("private_ground_truth", src)


if __name__ == "__main__":
    unittest.main()
