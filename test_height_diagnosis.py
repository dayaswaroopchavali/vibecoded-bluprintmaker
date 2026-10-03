"""Synthetic checks for height_diagnosis (truth-free second height estimator and the diagnosis of estimate_height)."""
import ast
import unittest
from pathlib import Path

import numpy as np

import height_diagnosis as hd


def _rotation(theta=.63, phi=-.28):
    rz = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]])
    return rz @ rx


def _room(height=3.0, scale=1.0, seed=0, tilt_deg=0., furniture=True, ceiling=True, floor=True, views=12,
          origin=(3., -2., 5.)):
    """Cloud of a 5 x 4 room. Local z is vertical; points are rotated into an arbitrary world frame."""
    rng = np.random.default_rng(seed)
    w, d = 5., 4.
    parts = []

    def cap(z, n):
        xy = np.column_stack([rng.uniform(-w / 2, w / 2, n), rng.uniform(-d / 2, d / 2, n)])
        zz = z + rng.normal(0, .004, n) + np.tan(np.radians(tilt_deg)) * xy[:, 0]
        return np.column_stack([xy, zz])

    if floor:
        parts.append(cap(0., 6000))
    if ceiling:
        parts.append(cap(height, 5000))
    for _ in range(3000):  # walls
        pass
    n = 3000
    t = rng.uniform(0, 1, n)
    zz = rng.uniform(0, height, n)
    parts.append(np.column_stack([np.full(n, -w / 2), (t - .5) * d, zz]))
    parts.append(np.column_stack([np.full(n, w / 2), (t - .5) * d, zz]))
    parts.append(np.column_stack([(t - .5) * w, np.full(n, -d / 2), zz]))
    parts.append(np.column_stack([(t - .5) * w, np.full(n, d / 2), zz]))
    if furniture:  # a table top and a cabinet top: small, horizontal, mid-height
        m = 900
        parts.append(np.column_stack([rng.uniform(-.6, .6, m), rng.uniform(-.4, .4, m) + 1., np.full(m, .75)]))
        parts.append(np.column_stack([rng.uniform(-.3, .3, m), rng.uniform(-.3, .3, m) - 1., np.full(m, 1.5)]))
    local = np.concatenate(parts)
    view_ids = rng.integers(0, views, len(local))
    rot = _rotation()
    world = (np.asarray(origin) + local @ rot.T) * scale
    return world, rot[:, 2], view_ids, rot


class EstimatorTests(unittest.TestCase):
    def test_complete_with_furniture_arbitrary_orientation(self):
        pts, up, ids, _ = _room()
        r = hd.estimate_room_height(pts, up, ids)
        self.assertEqual(r["status"], "complete", r["reasons"])
        self.assertAlmostEqual(r["separation_unscaled"], 3.0, delta=.03)
        self.assertFalse(r["axis_sign_resolved"])
        self.assertTrue(r["leave_one_view_out"]["available"])
        self.assertLess(r["leave_one_view_out"]["relative_range"], .02)

    def test_inverted_axis_sign_gives_same_height(self):
        pts, up, ids, _ = _room(seed=3)
        a = hd.estimate_room_height(pts, up, ids)
        b = hd.estimate_room_height(pts, -up, ids)
        self.assertEqual(a["status"], "complete")
        self.assertEqual(b["status"], "complete")
        self.assertAlmostEqual(a["separation_unscaled"], b["separation_unscaled"], delta=1e-2)
        self.assertLess(a["sign_disagreement_relative"], .01)

    def test_tilted_axis_estimate_still_recovers_height(self):
        pts, up, ids, rot = _room(seed=5)
        tilted = up + 0.05 * rot[:, 0] - 0.04 * rot[:, 1]   # ~3.6 degrees off the true structural axis
        r = hd.estimate_room_height(pts, tilted / np.linalg.norm(tilted), ids)
        self.assertEqual(r["status"], "complete", r["reasons"])
        self.assertAlmostEqual(r["separation_unscaled"], 3.0, delta=.05)
        self.assertGreater(r["axis_refinement_deg"], 1.0)

    def test_tilted_floor_and_ceiling_are_measured_perpendicular(self):
        pts, up, ids, _ = _room(seed=6, tilt_deg=2.)
        r = hd.estimate_room_height(pts, up, ids)
        self.assertEqual(r["status"], "complete", r["reasons"])
        self.assertAlmostEqual(r["separation_unscaled"], 3.0 * np.cos(np.radians(2.)), delta=.03)
        self.assertLess(r["parallel_angle_degrees"], .5)

    def test_scale_equivariance_and_translation(self):
        pts, up, ids, _ = _room(seed=7)
        base = hd.estimate_room_height(pts, up, ids)
        scaled = hd.estimate_room_height(pts * 7.3 + 11., up, ids)
        self.assertEqual(scaled["status"], base["status"])
        self.assertAlmostEqual(scaled["separation_unscaled"] / base["separation_unscaled"], 7.3, delta=.03)

    def test_missing_ceiling_is_not_complete(self):
        pts, up, ids, _ = _room(seed=8, ceiling=False, furniture=False)
        r = hd.estimate_room_height(pts, up, ids)
        self.assertNotEqual(r["status"], "complete")

    def test_without_view_ids_never_complete(self):
        pts, up, _, _ = _room(seed=9)
        r = hd.estimate_room_height(pts, up)
        self.assertEqual(r["status"], "ambiguous")
        self.assertIn("view ids", r["reasons"][0])

    def test_competing_broad_level_is_ambiguous(self):
        pts, up, ids, rot = _room(seed=10, furniture=False)
        rng = np.random.default_rng(11)
        n = 5000  # a second broad full-footprint level (e.g. mezzanine) at one third of the height
        local = np.column_stack([rng.uniform(-2.5, 2.5, n), rng.uniform(-2, 2, n), np.full(n, 1.0) + rng.normal(0, .004, n)])
        extra = np.asarray((3., -2., 5.)) + local @ rot.T
        pts2 = np.concatenate([pts, extra])
        ids2 = np.concatenate([ids, rng.integers(0, 12, n)])
        r = hd.estimate_room_height(pts2, up, ids2)
        self.assertEqual(r["status"], "ambiguous")
        self.assertTrue(r["intermediate_broad_peaks"])
        self.assertGreaterEqual(len(r["alternatives"]), 3)

    def test_disagreement_with_reference_is_reported(self):
        pts, up, ids, _ = _room(seed=12)
        good = hd.estimate_room_height(pts, up, ids, reference={"status": "estimated_unscaled_separation",
                                                                  "separation_unscaled": 3.01})
        bad = hd.estimate_room_height(pts, up, ids, reference={"status": "estimated_unscaled_separation",
                                                                 "separation_unscaled": 2.0})
        none = hd.estimate_room_height(pts, up, ids, reference={"status": "incomplete"})
        self.assertEqual(good["agreement_with_existing"]["status"], "agrees")
        self.assertEqual(bad["agreement_with_existing"]["status"], "disagrees")
        self.assertEqual(none["agreement_with_existing"]["status"], "reference_not_estimated")

    def test_degenerate_inputs_are_incomplete(self):
        self.assertEqual(hd.estimate_room_height(np.zeros((10, 3)), [0, 0, 1])["status"], "incomplete")
        pts, up, ids, _ = _room(seed=13)
        self.assertEqual(hd.estimate_room_height(pts, [0, 0, 0], ids)["status"], "incomplete")
        pts[5] = np.nan
        self.assertEqual(hd.estimate_room_height(pts, up, ids)["status"], "complete")

    def test_evaluator_adapter_only_maps_complete(self):
        ok = hd.evaluator_compatible({"status": "complete", "separation_unscaled": 2.1, "reasons": []})
        self.assertEqual(ok["status"], "estimated_unscaled_separation")
        self.assertEqual(ok["separation_unscaled"], 2.1)
        amb = hd.evaluator_compatible({"status": "ambiguous", "separation_unscaled": 2.1, "reasons": ["x"]})
        self.assertEqual(amb["status"], "ambiguous")
        self.assertIsNone(amb["separation_unscaled"])


class DiagnosisTests(unittest.TestCase):
    def test_support_share_failure_is_named_with_numbers(self):
        pts, up, ids, rot = _room(seed=14, furniture=False)

        def plane(z, support):
            center = np.asarray((3., -2., 5.)) + rot[:, 2] * z
            return {"normal": up.tolist(), "offset": float(-up @ center), "center": center.tolist(),
                    "support_count": support,
                    "surface_patches": {"axes_world": [rot[:, 0].tolist(), rot[:, 1].tolist()], "patches": [
                        {"bounds_uv": [[-2.5, -2.], [2.5, 2.]], "support_count": support}]}}

        walls = [{"normal": rot[:, 0].tolist(), "offset": 0., "support_count": 9000}]
        data = {"planes": walls + [plane(0., 300), plane(3., 300)]}
        centers = np.asarray((3., -2., 5.)) + np.array([[0., 0., 1.4], [.5, .2, 1.6]]) @ rot.T
        d = hd.diagnose_existing(data, pts, up, centers)
        self.assertEqual(d["existing_status"], "incomplete")
        rows = {r["plane_id"]: r for r in d["candidate_horizontal_planes"]}
        self.assertEqual(set(rows), {1, 2})
        for row in rows.values():
            self.assertEqual(row["failed_checks"], ["candidate_support_share"])
            self.assertAlmostEqual(row["support_share_of_all_candidates"], 300 / 9600, places=6)
        self.assertEqual({rows[1]["side"], rows[2]["side"]}, {"below_all_cameras", "above_all_cameras"})
        cf = d["counterfactual_relaxed_support_share"]
        self.assertEqual(cf["status"], "estimated_unscaled_separation")
        self.assertAlmostEqual(cf["separation_unscaled"], 3.0, delta=.01)


class PurityTests(unittest.TestCase):
    def test_no_truth_camera_up_or_simulator_imports(self):
        path = Path(hd.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        allowed = {"argparse", "json", "numpy", "pathlib", "estimate_height", "extract_planes", "__future__"}
        self.assertTrue(set(names) <= allowed, set(names) - allowed)
        for name in names:
            for banned in ("simulator", "house_layouts", "evaluation", "evaluate", "layout_mesh", "blender",
                           "ground_truth", "geometry_orientation", "gyro", "virtual"):
                self.assertNotIn(banned, name)
        # the estimator itself must not import the existing estimator (independence): only the diagnosis does
        estimator_funcs = {"estimate_room_height", "_analyse_one_sign", "_find_peaks", "_search_axis", "_fit_plane",
                           "_leave_one_view_out", "_separation"}
        for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in estimator_funcs):
            for node in ast.walk(fn):
                self.assertNotIsInstance(node, (ast.Import, ast.ImportFrom), fn.name)
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, {"eh", "estimate_height", "extract_planes", "camera_centers"}, fn.name)
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("scene.json", "private_ground", "wall_height_m", "ceiling_height_m", "3.0 m", "3 m "):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
