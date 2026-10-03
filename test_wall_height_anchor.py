"""Synthetic checks for wall_height_anchor: the wall-based height must not move when the ceiling plane is mis-read."""
import ast
import unittest
from pathlib import Path

import numpy as np

import wall_height_anchor as wha


def _rotation(theta=.63, phi=-.28):
    rz = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi), np.cos(phi)]])
    return rz @ rx


def _room(height=3.0, ceiling_height=None, scale=1.0, seed=0, furniture=True, ceiling=True, views=12, wall_top=None,
          origin=(3., -2., 5.)):
    """5 x 4 room, local z vertical, arbitrary world rotation. `ceiling_height` places the ceiling SHEET (default: at `height`);
    walls always run from 0 to `height` (or `wall_top`). Furniture includes a cabinet against a wall."""
    rng = np.random.default_rng(seed)
    w, d = 5., 4.
    parts = []

    def cap(z, n):
        xy = np.column_stack([rng.uniform(-w / 2, w / 2, n), rng.uniform(-d / 2, d / 2, n)])
        return np.column_stack([xy, z + rng.normal(0, .004, n)])

    parts.append(cap(0., 6000))
    if ceiling:
        parts.append(cap(height if ceiling_height is None else ceiling_height, 5000))
    top = height if wall_top is None else wall_top
    n = 5000
    t = rng.uniform(0, 1, n)
    zz = rng.uniform(0, top, n)
    noise = lambda: rng.normal(0, .004, n)  # noqa: E731
    parts.append(np.column_stack([np.full(n, -w / 2) + noise(), (t - .5) * d, zz]))
    parts.append(np.column_stack([np.full(n, w / 2) + noise(), (t - .5) * d, zz]))
    parts.append(np.column_stack([(t - .5) * w, np.full(n, -d / 2) + noise(), zz]))
    parts.append(np.column_stack([(t - .5) * w, np.full(n, d / 2) + noise(), zz]))
    if furniture:   # table top and a cabinet standing against the +x wall
        m = 900
        parts.append(np.column_stack([rng.uniform(-.6, .6, m), rng.uniform(-.4, .4, m) + 1., np.full(m, .75)]))
        cab = rng.uniform(0, 1, (2500, 3)) * np.array([.5, 1.2, 1.4]) + np.array([w / 2 - .5, -.6, 0.])
        parts.append(cab)
    local = np.concatenate(parts)
    view_ids = rng.integers(0, views, len(local))
    rot = _rotation()
    world = (np.asarray(origin) + local @ rot.T) * scale
    return world, rot[:, 2], view_ids, rot


class WallHeightTests(unittest.TestCase):
    def test_complete_with_furniture_and_arbitrary_orientation(self):
        pts, up, ids, _ = _room()
        r = wha.estimate_wall_height(pts, up, ids)
        self.assertEqual(r["status"], "complete", r["reasons"])
        self.assertAlmostEqual(r["separation_unscaled"], 3.0, delta=.06)
        self.assertEqual(r["cap_agreement"]["status"], "agrees")
        self.assertTrue(r["leave_one_view_out"]["available"])
        self.assertFalse(r["axis_sign_resolved"])
        self.assertEqual(wha.evaluator_compatible(r)["status"], "estimated_unscaled_separation")

    def test_ceiling_plane_5_percent_nearer_does_not_move_wall_height_and_is_flagged(self):
        pts0, up, ids0, _ = _room(seed=1, furniture=False)
        pts, up, ids, _ = _room(seed=1, furniture=False, ceiling_height=3.0 * .95)
        r0 = wha.estimate_wall_height(pts0, up, ids0)
        r = wha.estimate_wall_height(pts, up, ids)
        self.assertAlmostEqual(r["separation_unscaled"], r0["separation_unscaled"], delta=.05)
        self.assertAlmostEqual(r["separation_unscaled"], 3.0, delta=.06)
        cap = r["cap_agreement"]
        self.assertEqual(cap["status"], "disagrees")
        self.assertAlmostEqual(cap["cap_plane_separation"], 2.85, delta=.05)
        self.assertGreater(cap["relative_difference_wall_minus_cap"], wha.CAP_AGREEMENT_TOLERANCE)
        self.assertEqual(r["status"], "ambiguous")
        self.assertTrue(r["ambiguity_flags"])
        # the evaluator adapter refuses an ambiguous height; the number is only exposed by the explicit diagnostic adapter
        self.assertIsNone(wha.evaluator_compatible(r)["separation_unscaled"])
        self.assertNotEqual(wha.evaluator_compatible(r)["status"], "estimated_unscaled_separation")
        self.assertEqual(wha.diagnostic_unscaled_separation(r)["diagnostic_status"], "ambiguous")
        names = [a["name"] for a in r["alternatives"]]
        self.assertIn("cap_plane_estimate (height_diagnosis)", names)

    def test_supplied_reference_is_only_used_for_agreement(self):
        pts, up, ids, _ = _room(seed=2, furniture=False)
        ref = {"status": "estimated_unscaled_separation", "separation_unscaled": 2.5}
        r = wha.estimate_wall_height(pts, up, ids, reference=ref)
        r2 = wha.estimate_wall_height(pts, up, ids, reference={"status": "estimated_unscaled_separation", "separation_unscaled": 3.0})
        self.assertAlmostEqual(r["separation_unscaled"], r2["separation_unscaled"], places=9)
        self.assertEqual(r["cap_agreement"]["status"], "disagrees")
        self.assertEqual(r2["cap_agreement"]["status"], "agrees")

    def test_inverted_axis_gives_same_height(self):
        pts, up, ids, _ = _room(seed=3)
        a = wha.estimate_wall_height(pts, up, ids, compute_reference=False)
        b = wha.estimate_wall_height(pts, -up, ids, compute_reference=False)
        self.assertEqual(a["status"], "complete", a["reasons"])
        self.assertEqual(b["status"], "complete", b["reasons"])
        self.assertAlmostEqual(a["separation_unscaled"], b["separation_unscaled"], delta=.02)
        self.assertLess(a["sign_disagreement_relative"], .01)

    def test_tilted_axis_is_refined_from_the_walls(self):
        pts, up, ids, rot = _room(seed=5)
        tilted = up + 0.05 * rot[:, 0] - 0.04 * rot[:, 1]     # about 3.6 degrees off
        r = wha.estimate_wall_height(pts, tilted / np.linalg.norm(tilted), ids, compute_reference=False)
        self.assertEqual(r["status"], "complete", r["reasons"])
        self.assertAlmostEqual(r["separation_unscaled"], 3.0, delta=.08)
        self.assertGreater(r["axis_refinement_deg"], 1.0)
        self.assertTrue(r["axis_refined_from_walls"])

    def test_scale_equivariance_and_translation(self):
        pts, up, ids, _ = _room(seed=7)
        base = wha.estimate_wall_height(pts, up, ids, compute_reference=False)
        scaled = wha.estimate_wall_height(pts * 7.3 + 11., up, ids, compute_reference=False)
        self.assertEqual(scaled["status"], base["status"])
        self.assertAlmostEqual(scaled["separation_unscaled"] / base["separation_unscaled"], 7.3, delta=.05)

    def test_missing_ceiling_does_not_use_a_cap_and_is_never_called_agreeing(self):
        pts, up, ids, _ = _room(seed=8, ceiling=False, furniture=False)
        r = wha.estimate_wall_height(pts, up, ids)
        # walls still reach the (unobserved) ceiling, so a wall-based number exists; the cap estimate cannot confirm it
        self.assertAlmostEqual(r["separation_unscaled"], 3.0, delta=.08)
        self.assertIn(r["cap_agreement"]["status"], ("reference_not_estimated", "disagrees"))
        self.assertNotEqual(r["cap_agreement"]["status"], "agrees")

    def test_walls_that_stop_short_are_flagged_not_trusted(self):
        # the upper third of every wall was never observed: extent is a lower bound; the cap planes still bracket 3 m
        pts, up, ids, _ = _room(seed=9, furniture=False, wall_top=2.0)
        r = wha.estimate_wall_height(pts, up, ids)
        self.assertLess(r["separation_unscaled"], 2.3)
        self.assertEqual(r["cap_agreement"]["status"], "disagrees")
        self.assertEqual(r["status"], "ambiguous")

    def test_without_view_ids_never_complete(self):
        pts, up, _, _ = _room(seed=9)
        r = wha.estimate_wall_height(pts, up, compute_reference=False)
        self.assertEqual(r["status"], "ambiguous")
        self.assertIn("view ids", r["reasons"][0])

    def test_no_walls_is_incomplete(self):
        rng = np.random.default_rng(0)
        flat = np.column_stack([rng.uniform(-2, 2, 4000), rng.uniform(-2, 2, 4000), rng.normal(0, .004, 4000)])
        r = wha.estimate_wall_height(flat, np.array([0., 0., 1.]), np.zeros(4000, int), compute_reference=False)
        self.assertEqual(r["status"], "incomplete")
        self.assertIsNone(r["separation_unscaled"])
        self.assertIsNone(wha.evaluator_compatible(r)["separation_unscaled"])

    def test_degenerate_inputs(self):
        self.assertEqual(wha.estimate_wall_height(np.zeros((10, 3)), [0, 0, 1])["status"], "incomplete")
        self.assertEqual(wha.estimate_wall_height(np.random.rand(500, 3), [0, 0, 0])["status"], "incomplete")


class PurityTests(unittest.TestCase):
    def test_no_truth_camera_up_or_simulator_imports(self):
        path = Path(wha.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        allowed = {"numpy", "height_diagnosis", "__future__"}
        self.assertTrue(set(names) <= allowed, set(names) - allowed)
        # height_diagnosis (cap estimate, agreement flag only) may be imported only inside cap_reference
        for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
            has_import = any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(fn))
            self.assertEqual(has_import, fn.name == "cap_reference", fn.name)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        self.assertFalse(used & {"scene", "truth", "ground_truth", "simulator", "camera_centers", "camera_up", "gravity"})
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("scene.json", "private_ground", "wall_height_m", "ceiling_height_m", "3.0 m", "3 m ", "camera_up",
                       "estimate_height", "extract_planes", "house_layouts", "evaluate_floorplan"):
            self.assertNotIn(banned, text)

    def test_cap_estimate_never_feeds_the_number(self):
        tree = ast.parse(Path(wha.__file__).read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_analyse_one_sign")
        used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        self.assertFalse(used & {"cap_reference", "reference", "height_diagnosis", "estimate_room_height"})


if __name__ == "__main__":
    unittest.main()
