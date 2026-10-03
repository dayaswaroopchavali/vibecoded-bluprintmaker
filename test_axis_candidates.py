"""Truth-free synthetic tests for axis_candidates: every true vertical of synthetic rooms must be among the candidates."""
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

import axis_candidates as ac
import extract_planes as ep

HERE = Path(__file__).resolve().parent


def _inside(poly, p):
    x, y = p[:, 0], p[:, 1]
    res = np.zeros(len(p), bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        cond = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (x1 - x0) * (y - y0) / (y1 - y0) + x0
        res ^= cond & (x < xi)
    return res


def prism(poly, height=3.0, seed=0, floor=True, ceiling=True, n_wall=2500, n_cap=5000, noise=.004):
    """Points of a vertical prism room; local z is vertical."""
    rng = np.random.default_rng(seed)
    poly = np.asarray(poly, float)
    parts = []
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
        t = rng.uniform(0, 1, (n_wall, 1))
        xy = p0 + t * (p1 - p0)
        parts.append(np.column_stack([xy, rng.uniform(0, height, n_wall)]))
    lo, hi = poly.min(0), poly.max(0)
    cap = rng.uniform(lo, hi, (n_cap * 4, 2))
    cap = cap[_inside(poly, cap)][:n_cap]
    for z, on in ((0., floor), (height, ceiling)):
        if on:
            parts.append(np.column_stack([cap, np.full(len(cap), z) + rng.normal(0, noise, len(cap))]))
    pts = np.concatenate(parts)
    return pts + rng.normal(0, noise, pts.shape)


def rotation(seed, flip=False):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    if flip:                                    # inverted: turn the room upside down
        q = q @ np.diag([1., -1., -1.])
    return q


def candidate_angle(result, vertical):
    return min(ac._angle_deg(np.asarray(c["axis_world"]), vertical) for c in result["candidates"])


def run(pts, use_normals=True):
    planes = ep.extract(pts)
    normals = ac.normals_from_points(pts) if use_normals else None
    return ac.enumerate_axis_candidates(planes, pts, normals=normals)


BOX = [(-2.5, -2), (2.5, -2), (2.5, 2), (-2.5, 2)]
L_SHAPE = [(0, 0), (6, 0), (6, 2.5), (2.5, 2.5), (2.5, 5), (0, 5)]
SKEW = [(0, 0), (5, 0), (6.8, 3.5), (1.8, 3.5)]                     # parallelogram: walls at ~63 deg, not Manhattan
PENTAGON = [(0, 0), (5, 0), (6, 2.5), (2.8, 4.5), (-.5, 2.5)]


class TrueVerticalIsACandidate(unittest.TestCase):
    def check(self, poly, seed, flip=False, **kw):
        pts = prism(poly, **kw)
        rot = rotation(seed, flip)
        world = pts @ rot.T + np.array([3., -2., 5.])
        vertical = rot @ np.array([0., 0., 1.])
        for use_normals in (True, False):
            res = run(world, use_normals)
            self.assertGreater(len(res["candidates"]), 0)
            self.assertLess(candidate_angle(res, vertical), 3.0, f"normals={use_normals}")
        return res

    def test_box_arbitrary_orientation(self):
        for seed in (1, 2, 3):
            self.check(BOX, seed)

    def test_box_inverted_and_rolled(self):
        for seed in (4, 5):
            self.check(BOX, seed, flip=True)

    def test_l_shape(self):
        for seed in (6, 7):
            self.check(L_SHAPE, seed)

    def test_non_manhattan_skew_and_pentagon(self):
        for poly in (SKEW, PENTAGON):
            res = self.check(poly, 8)
            top = [c for c in res["candidates"] if c["viable"]]
            self.assertTrue(top)

    def test_missing_ceiling_still_offers_vertical_via_wall_zone_axis(self):
        self.check(L_SHAPE, 9, ceiling=False)

    def test_missing_both_caps_uses_zone_axis_of_walls(self):
        self.check(PENTAGON, 10, floor=False, ceiling=False)

    def test_tilted_room_every_axis_distinct_and_unit(self):
        res = self.check(BOX, 11)
        axes = np.array([c["axis_world"] for c in res["candidates"]])
        np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-9)
        for i in range(len(axes)):
            for j in range(i):
                self.assertGreater(ac._angle_deg(axes[i], axes[j]), ac.CANDIDATE_MERGE_DEG - 1e-6)


class Evidence(unittest.TestCase):
    def test_box_is_ambiguous_and_sign_is_never_resolved(self):
        pts = prism(BOX)
        res = run(pts @ rotation(1).T)
        self.assertIn(res["status"], ("ambiguous_axes", "single_axis_sign_ambiguous"))
        self.assertFalse(res["sign_resolved"])
        self.assertFalse(res["gravity_known"])
        self.assertTrue(all(not c["sign_resolved"] for c in res["candidates"]))
        self.assertGreaterEqual(res["co_supported_count"], 2)                 # box symmetry: wall pairs look like caps
        self.assertTrue(any("unsigned" in s or "sign" in s for s in res["unresolved_ambiguity"]))
        for c in res["candidates"]:
            np.testing.assert_allclose(c["alternative_sign_axis_world"], -np.asarray(c["axis_world"]))

    def test_true_vertical_has_cap_pair_with_both_sign_views(self):
        pts = prism(BOX, height=2.0)
        rot = rotation(2)
        res = run(pts @ rot.T)
        vertical = rot @ np.array([0., 0., 1.])
        best = min(res["candidates"], key=lambda c: ac._angle_deg(np.asarray(c["axis_world"]), vertical))
        cap = best["evidence"]["cap_pair"]
        self.assertTrue(cap["pair_found"])
        self.assertAlmostEqual(cap["separation_unscaled"], 2.0, delta=0.4)
        signs = best["evidence"]["axis_sign_views"]
        self.assertAlmostEqual(signs["plus"]["floor_like_height"], -signs["minus"]["ceiling_like_height"], places=9)

    def test_scale_invariance_of_candidate_count(self):
        pts = prism(L_SHAPE, seed=3)
        a = run(pts)
        b = run(pts * 37.0)
        self.assertEqual(len(a["candidates"]), len(b["candidates"]))

    def test_gauss_map_clusters_find_three_orthogonal_directions(self):
        rng = np.random.default_rng(0)
        base = np.eye(3)
        n = np.concatenate([base[k] + rng.normal(0, .02, (3000, 3)) for k in range(3)])
        n = n * rng.choice([-1, 1], (len(n), 1))
        cl = ac.cluster_directions(n)
        self.assertEqual(len(cl), 3)
        for k in range(3):
            self.assertLess(min(ac._angle_deg(c["axis"], base[k]) for c in cl), 2.0)

    def test_polygon_runner_feeds_ranking_and_errors_are_recorded(self):
        pts = prism(BOX)
        planes = ep.extract(pts)
        seen = []

        def runner(axis):
            seen.append(np.asarray(axis).copy())
            if len(seen) == 1:
                raise RuntimeError("boom")
            return {"boundary_status": "complete_polygon_hypothesis"}
        res = ac.enumerate_axis_candidates(planes, pts, normals=ac.normals_from_points(pts), polygon_runner=runner)
        self.assertEqual(len(seen), len(res["candidates"]))                  # EVERY candidate is run
        statuses = [c["polygon"]["boundary_status"] for c in res["candidates"]]
        self.assertIn("runner_failed", statuses)
        self.assertIn("complete_polygon_hypothesis", statuses)

    def test_degenerate_inputs_report_status(self):
        self.assertEqual(ac.enumerate_axis_candidates({"planes": []}, np.zeros((10, 3)))["status"], "insufficient_points")
        pts = prism(BOX)
        self.assertEqual(ac.enumerate_axis_candidates({"planes": []}, pts)["status"], "insufficient_directions")


class Schema(unittest.TestCase):
    def test_same_schema_as_geometry_orientation(self):
        import geometry_orientation as go
        pts = prism(BOX)
        planes = ep.extract(pts)
        ref = go.axis_hypotheses(planes, pts)
        mine = ac.to_axis_hypotheses(ac.enumerate_axis_candidates(planes, pts, normals=ac.normals_from_points(pts)))
        self.assertTrue(set(ref) - {"hypotheses"} <= set(mine), set(ref) - set(mine))
        self.assertTrue(ref["hypotheses"])
        self.assertTrue(set(ref["hypotheses"][0]) <= set(mine["hypotheses"][0]), set(ref["hypotheses"][0]) - set(mine["hypotheses"][0]))
        for h in mine["hypotheses"]:
            self.assertEqual(len(h["axis_world"]), 3)
            self.assertIn("co_supported_with_best", h)
            self.assertIn("rank", h)

    def test_json_serialisable(self):
        import json
        pts = prism(BOX)
        res = ac.enumerate_axis_candidates(ep.extract(pts), pts, normals=ac.normals_from_points(pts))
        json.dumps(ac.to_axis_hypotheses(res), default=float)
        json.dumps(res, default=float)


class TruthFree(unittest.TestCase):
    FORBIDDEN = {"simulator", "evaluate_floorplan", "evaluation", "blender_capture", "render_appearance", "house_layouts",
                 "virtual_gyro", "gyro_orientation", "gyro_stabilize", "make_gt_viewer", "make_box_eval_bundle",
                 "analyze_floorplan_error", "evaluate_windowed", "evaluate_rgb_boundary"}

    def test_axis_candidates_imports_nothing_truthy(self):
        tree = ast.parse((HERE / "axis_candidates.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertFalse(names & self.FORBIDDEN, names & self.FORBIDDEN)
        self.assertTrue(names <= {"__future__", "itertools", "numpy", "scipy"}, names)

    def test_no_truth_or_gravity_identifiers(self):
        text = (HERE / "axis_candidates.py").read_text(encoding="utf-8")
        for token in ("private_ground_truth", "scene.json", "camera_up", "ground_truth", "image_up"):
            self.assertNotIn(token, text)
        tree = ast.parse(text)
        ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
        for bad in ("scene", "truth", "gt", "up_vector", "camera_up"):
            self.assertNotIn(bad, ids)

    def test_extraction_stage_cannot_see_truth(self):
        import inspect

        import run_axis_candidates as rac
        sig = inspect.signature(rac.run_extraction)
        self.assertFalse({"scene", "scenes", "truth"} & set(sig.parameters))
        src = inspect.getsource(rac.run_extraction)
        for token in ("scene", "evaluate_floorplan", "load_scene", "private_ground_truth"):
            self.assertNotIn(token, src)

    def test_evaluation_refuses_before_freezing(self):
        import run_axis_candidates as rac
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                rac.run_evaluation([], Path(d), {"budgets": []})


class RunnerEndToEnd(unittest.TestCase):
    """run_extraction + run_evaluation with stubbed heavy stages: every candidate is run and scored; truth never selects."""

    def test_every_candidate_run_and_scored(self):
        import run_axis_candidates as rac
        pts = prism(BOX, seed=5)
        ids = np.zeros(len(pts), int)
        events, scored = [], []
        hooks = {
            "load_prediction": lambda p: {"depth": np.zeros((2, 2, 2)), "extrinsics": np.zeros((2, 3, 4)), "source_indices": np.arange(2)},
            "per_view_points": lambda pred: (pts, ids),
            "normals_from_prediction": lambda pred: ac.normals_from_points(pts),
            "camera_centers": lambda e: np.zeros((2, 3)),
            "estimate_room_height": lambda p, axis, i: {"status": "complete", "separation_unscaled": 3.0, "reasons": []},
            "extract_wall_lines": lambda pred, axis: {"status": "ok", "lines": [], "observations": {"x": 1}},
            "assemble_boundary_ray": lambda pred, walls, k: {"status": "complete_polygon_hypothesis", "vertices_xy": [[0, 0], [1, 0], [1, 1]], "edges": []},
            "evaluate_floorplan": lambda b, s, h, m: scored.append(h) or {"status": "scored", "metrics": {"iou": .5, "max_abs_wall_error_cm": 9.0, "max_rel_wall_error_percent": 1.0},
                                                                             "custom_1_5cm_horizontal_diagnostic": {"pass": False}, "wall_length_3_percent_threshold_oracle_normalised": {"pass": True}},
            "load_scene": lambda p: {"width": 5, "length": 4, "height": 3},
        }
        budgets = [{"label": "x", "prediction": "p", "scene": "s"}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            out.mkdir()
            index = rac.run_extraction(budgets, out, hooks=hooks, events=events)
            rec = index["budgets"][0]
            self.assertEqual(rec["status"], "ok")
            n = len(rec["candidates"])
            self.assertGreaterEqual(n, 3)
            self.assertEqual(sum(1 for e in events if e[0] == "candidate_downstream"), n)
            for c in rec["candidates"]:
                self.assertIn("boundary", c["files"])
                self.assertIn("height", c["files"])
            self.assertNotIn("observations", (out / rec["candidates"][0]["files"]["walls_summary"]["path"]).read_text())
            rac.write_reports(index, out)
            self.assertTrue((out / "reports" / "x.md").exists())
            summary = rac.run_evaluation(budgets, out, index, hooks=hooks, events=events)
            self.assertEqual(len(summary["rows"]), n)
            self.assertEqual(len(scored), n)
            self.assertTrue(all(r["status"] == "scored" for r in summary["rows"]))
            self.assertFalse(summary["selection_by_truth"])
            self.assertEqual(scored[0]["status"], "estimated_unscaled_separation")
            self.assertIn("evaluate", [e[0] for e in events])
            self.assertLess(max(i for i, e in enumerate(events) if e[0] == "candidate_downstream"),
                            min(i for i, e in enumerate(events) if e[0] == "evaluate"))
            self.assertIn("1.5", rac.results_markdown(summary))


if __name__ == "__main__":
    unittest.main()
