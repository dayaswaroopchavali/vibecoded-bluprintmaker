"""Tests for compare_wall_boundaries on schema-conforming synthetic fixtures (no real A/B/DA3 needed)."""
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compare_wall_boundaries as C


def rect(w, h, angle=0.0, scale=1.0, shift=(0.0, 0.0), mirror=False, start=0):
    v = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) - [w / 2, h / 2]
    if mirror:
        v = (v * [-1, 1])[::-1]
    v = np.roll(v, -start, axis=0)
    c, s = math.cos(angle), math.sin(angle)
    return scale * v @ np.array([[c, -s], [s, c]]).T + np.asarray(shift)


def walls_for(poly, sd=0.01):
    lines = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        t0, t1 = float(d @ a), float(d @ b)
        L = t1 - t0
        lines.append({"id": k, "normal_xy": n.tolist(), "offset": float(n @ a), "direction_xy": d.tolist(),
                      "observed_spans": [[t0 + 0.1 * L, t0 + 0.7 * L]],
                      "support": {"distinct_frames": 12, "frame_ids": list(range(12)), "view_spread_deg": 80.0,
                                  "along_coverage_fraction": 0.6, "weight": 5.0},
                      "uncertainty": {"offset_sd": sd, "angle_sd_deg": 0.5, "method": "leave_view_out"}, "height_range": [0.0, 1.0]})
    return {"model": "observed_wall_lines", "status": "ok", "reasons": [], "axis_world": [0, 0, 1],
            "frame": {"origin_world": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0], "extent": 3.0}, "noise_scale": 0.01,
            "lines": lines, "observations": {"frame_ids": [], "camera_xy": [], "hits_xy": [], "hit_heights": []}}


def boundary_for(poly, status="complete_polygon_hypothesis", b_style=False):
    """b_style: edge k ENDS at vertex k (as boundary_assembly emits), else edge k starts at vertex k."""
    edges = []
    for k in range(len(poly)):
        s = (k - 1) % len(poly) if b_style else k
        a, b = poly[s], poly[(s + 1) % len(poly)]
        d = (b - a) / np.linalg.norm(b - a)
        obs = k == 0
        t0 = float(d @ a); L = float(np.linalg.norm(b - a))
        edges.append({"index": k, "line_id": k, "start_xy": a.tolist(), "end_xy": b.tolist(), "length": L,
                      "label": "observed" if obs else "extrapolated", "observed_fraction": 1.0 if obs else 0.6,
                      "observed_spans_xy": [[(a + 0.1 * (b - a)).tolist(), (a + 0.7 * (b - a)).tolist()]],
                      "uncertainty": {"offset_sd": 0.01}, "supporting_frames": 12, "free_space": {}})
    return {"status": status, "vertices_xy": poly.tolist(), "edges": edges, "alternatives": [], "incomplete_boundaries": [],
            "frame": {}, "axes_world": [], "coordinate_origin_world": [0, 0, 0], "units": "arbitrary"}


class IndexConventionTests(unittest.TestCase):
    def test_assembly_style_edge_indices_are_remapped_for_evaluator(self):
        poly = rect(2, 1.3333)
        b = boundary_for(poly, b_style=True)              # edge 0 runs vertex 3 -> vertex 0
        c = C.eval_copy(b)
        edge0 = next(e for e in c["edges"] if e["assembly_index"] == 0)
        self.assertEqual(edge0["index"], 3)
        self.assertEqual(edge0["label"], "observed")
        self.assertEqual(sorted(e["index"] for e in c["edges"]), [0, 1, 2, 3])
        self.assertEqual(sum(e["label"] == "inferred" for e in c["edges"]), 3)


class StabilityTests(unittest.TestCase):
    def test_identical_is_stable(self):
        b = boundary_for(rect(2, 1.3333))
        r = C.stability_pair(b, b)
        self.assertEqual(r["verdict"], "stable")
        self.assertLess(r["edge_length_change_max_percent"], 1e-6)

    def test_edge_count_change_is_unstable(self):
        a = boundary_for(rect(2, 1.3333))
        l_shape = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
        r = C.stability_pair(a, boundary_for(l_shape))
        self.assertEqual(r["verdict"], "unstable_edge_count")
        self.assertEqual((r["edge_count_a"], r["edge_count_b"]), (4, 6))

    def test_rigid_scale_copy_is_stable_even_mirrored_and_reindexed(self):
        base = rect(2, 1.3333)
        a = boundary_for(base)
        for kwargs in ({"angle": 0.7, "scale": 2.3, "shift": (5, -3)},
                       {"angle": -2.0, "scale": 0.4, "shift": (1, 1), "mirror": True, "start": 2}):
            r = C.stability_pair(a, boundary_for(rect(2, 1.3333, **kwargs)))
            self.assertEqual(r["verdict"], "stable", r)
            self.assertLess(r["corner_residual_max_fraction_of_diameter"], 1e-6)
            self.assertLess(r["edge_length_change_max_percent"], 1e-5)
        l_shape = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 2], [0, 2]], float)
        mirrored = (l_shape * [-1, 1])[::-1] * 1.7 + [4, 2]
        r = C.stability_pair(boundary_for(l_shape), boundary_for(np.roll(mirrored, 2, axis=0)))
        self.assertEqual(r["verdict"], "stable", r)
        self.assertTrue(r["fit"]["reflection_used"])

    def test_geometry_change_is_reported_unstable(self):
        r = C.stability_pair(boundary_for(rect(2, 1.3333)), boundary_for(rect(2, 1.9)))
        self.assertEqual(r["verdict"], "unstable_geometry")
        self.assertGreater(r["edge_length_change_max_percent"], 5)

    def test_missing_polygon_not_comparable_and_listed(self):
        a = boundary_for(rect(2, 1))
        inc = {"status": "incomplete", "vertices_xy": None, "edges": [], "incomplete_boundaries": [{"reason": "open"}]}
        m = C.stability_matrix([{"budget": "x", "hypothesis_index": 0, "boundary": a},
                                {"budget": "y", "hypothesis_index": 0, "boundary": inc},
                                {"budget": "y", "hypothesis_index": 1, "boundary": a}])
        verdicts = sorted(p["verdict"] for p in m["pairs"])
        self.assertEqual(verdicts, ["not_comparable", "stable"])        # nothing filtered
        self.assertEqual(m["per_result"]["y/hyp0"]["not_comparable"], 1)


def fake_prediction(n):
    return {"depth": np.zeros((n, 2, 2)), "extrinsics": np.tile(np.eye(4)[:3], (n, 1, 1)), "source_indices": np.arange(n) * 3}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.events = []
        self.scene_loaded_after_freeze = []
        self.frozen_present_at_eval = []
        self.eval_labels = []
        self.poly = rect(2.0, 4 / 3)
        out = self

        def axis_hyps(planes, points):
            hy = [{"axis_world": [0, 0, 1], "rank": 0, "score": 2.0, "co_supported_with_best": True},
                  {"axis_world": [1, 0, 0], "rank": 1, "score": 1.95, "co_supported_with_best": True},
                  {"axis_world": [0, 1, 0], "rank": 2, "score": 1.9, "co_supported_with_best": True},
                  {"axis_world": [1, 1, 0], "rank": 3, "score": 0.5, "co_supported_with_best": False}]
            return {"status": "ambiguous_axes", "hypotheses": hy}

        def extract_walls(pred, axis):
            if abs(axis[0] - 1) < 1e-9:
                raise RuntimeError("synthetic A failure")
            if abs(axis[1] - 1) < 1e-9:
                return {**walls_for(out.poly), "status": "insufficient", "reasons": ["too few wall lines"]}
            return walls_for(out.poly)

        def evaluate(prediction, scene, height, manifest):
            out.events.append(("hook_evaluate",))
            out.frozen_present_at_eval.append((self.root / "out" / "frozen_index.json").exists())
            out.eval_labels.append(sorted({e["label"] for e in prediction["edges"]}))
            import evaluate_floorplan
            return evaluate_floorplan.evaluate_floorplan(prediction, scene, height, manifest=manifest, axis_sign_resolved=False)

        def load_scene(path):
            out.scene_loaded_after_freeze.append((self.root / "out" / "frozen_index.json").exists())
            return {"width": 6.0, "length": 4.0, "height": 3.0}

        self.hooks = {"load_prediction": lambda p: fake_prediction(36), "points_from_prediction": lambda pred: np.zeros((10, 3)),
                      "extract_planes": lambda pts: {"planes": []}, "axis_hypotheses": axis_hyps,
                      "extract_wall_lines": extract_walls,
                      "assemble_boundary": lambda walls: boundary_for(out.poly),
                      "camera_centers": lambda e: np.zeros((3, 3)),
                      "estimate_height": lambda planes, pts, up, centers: {"status": "estimated_unscaled_separation", "separation_unscaled": 1.0},
                      "evaluate_floorplan": evaluate, "load_scene": load_scene}
        self.budgets = [{"label": "b1", "prediction": "p1", "scene": "s1", "frames_available": 120},
                        {"label": "b2", "prediction": "p2", "scene": "s2", "frames_available": 120}]

    def tearDown(self):
        self.tmp.cleanup()

    def run_it(self, **kw):
        return C.run_comparison(self.budgets, self.root / "out", hooks=self.hooks, events=self.events, **kw)

    def test_extraction_path_has_no_scene_and_never_loads_truth(self):
        self.assertNotIn("scene", " ".join(inspect.signature(C.run_extraction).parameters))
        self.hooks["load_scene"] = lambda p: self.fail("truth loaded in extraction")
        (self.root / "x").mkdir()
        index = C.run_extraction(self.budgets, self.root / "x", self.hooks)
        self.assertEqual(len(index["budgets"]), 2)

    def test_freeze_before_evaluation_order(self):
        self.run_it()
        names = [e[0] for e in self.events]
        first_eval = min(i for i, n in enumerate(names) if n in ("evaluate_budget", "hook_evaluate", "evaluate"))
        last_extract = max(i for i, n in enumerate(names) if n in ("extract_hypothesis", "assemble_hypothesis", "extract_budget"))
        self.assertLess(last_extract, names.index("frozen"))
        self.assertLess(names.index("frozen"), first_eval)
        self.assertTrue(all(self.frozen_present_at_eval) and self.frozen_present_at_eval)
        self.assertTrue(all(self.scene_loaded_after_freeze) and self.scene_loaded_after_freeze)

    def test_evaluation_reads_frozen_files_and_detects_tampering(self):
        out = self.root / "out"
        out.mkdir()
        index = C.run_extraction(self.budgets, out, self.hooks)
        h = index["budgets"][0]["hypotheses"][0]
        (out / h["files"]["boundary"]["path"]).write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            C.run_evaluation(self.budgets, out, index, self.hooks)

    def test_evaluation_refused_before_freeze(self):
        (self.root / "e").mkdir()
        with self.assertRaises(RuntimeError):
            C.run_evaluation(self.budgets, self.root / "e", {"budgets": []}, self.hooks)

    def test_failed_unavailable_and_skipped_hypotheses_are_reported_not_dropped(self):
        res = self.run_it()
        by = {(r["budget"], r["hypothesis"]): r for r in res["table"]}
        self.assertEqual(len(res["table"]), 8)                           # 4 hypotheses x 2 budgets, nothing dropped
        self.assertEqual(by[("b1", 0)]["run_status"], "ran")
        self.assertEqual(by[("b1", 1)]["run_status"], "failed")
        self.assertIn("synthetic A failure", by[("b1", 1)]["failure"])
        self.assertEqual(by[("b1", 2)]["run_status"], "unavailable")
        self.assertIn("too few wall lines", by[("b1", 2)]["failure"])
        self.assertEqual(by[("b1", 3)]["run_status"], "skipped_not_co_supported")
        self.assertTrue((self.root / "out" / "comparison.md").exists())

    def test_budget_load_failure_is_reported(self):
        def bad(p):
            raise OSError("cannot read npz")
        self.hooks["load_prediction"] = bad
        res = self.run_it(evaluate=False)
        self.assertEqual(res["table"][0]["status"], "failed_before_hypotheses")
        self.assertIn("cannot read npz", res["table"][0]["failure"])

    def test_provenance_blockers_and_label_mapping(self):
        res = self.run_it()
        row = next(r for r in res["table"] if r["budget"] == "b1" and r["hypothesis"] == 0)
        wrc = row["evaluation"]["whole_room_claim"]
        self.assertFalse(wrc["allowed"])
        self.assertIn("retained_capture_below_full", wrc["blockers"])
        self.assertAlmostEqual(row["retained_capture_fraction"], 36 / 120)
        self.assertEqual(self.eval_labels[0], ["inferred", "observed"])        # extrapolated -> inferred for the evaluator
        self.assertEqual(row["evaluation"]["status"], "scored")
        self.assertIsNotNone(row["evaluation"]["max_abs_wall_error_cm"])
        self.assertTrue(row["evaluation"]["custom_1_5cm_pass"])             # exact synthetic rectangle at the true scale
        self.assertEqual(row["evaluation"]["unmatched_true"], [])
        self.assertEqual(row["stability"]["stable"], 1)                     # only hyp0 ran in each budget
        self.assertTrue(res["oracle_choices"] and all("EVALUATOR-ORACLE" in c for c in res["oracle_choices"]))

    def test_residual_sources_use_numbers_and_flag_unmatched(self):
        res = self.run_it()
        rs = res["residual_sources"]["b1/hyp0"]
        self.assertTrue(rs["available"])
        self.assertEqual(len(rs["rows"]), 4)
        self.assertIsNotNone(rs["rows"][0]["propagated_endpoint_length_sd_cm"])
        self.assertAlmostEqual(rs["common_length_ratio_median"], 1.0, places=2)
        self.assertIn("statement", rs)

    def test_refuses_to_overwrite(self):
        (self.root / "out").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_it()
        with self.assertRaises(FileExistsError):
            self.root.joinpath("f.json").write_text("{}")
            C.write_new_json(self.root / "f.json", {"a": 1})

    def test_nonfinite_values_become_null_in_frozen_json(self):
        p = self.root / "n.json"
        C.write_new_json(p, {"a": float("nan"), "b": np.array([1.0, np.inf])})
        self.assertEqual(json.loads(p.read_text()), {"a": None, "b": [1.0, None]})


class RayComparisonTests(unittest.TestCase):
    setUp = PipelineTests.setUp
    tearDown = PipelineTests.tearDown

    def with_ray_hook(self, reject_k=None):
        def ray(pred, walls, k):
            b = boundary_for(self.poly)
            span = {"start_xy": [0, 0], "end_xy": [1, 0], "free_space_contradiction": False, "fraction_domain": "wall_patch_area_within_supplied_height_band"}
            calls = [{**span, "conflict_treated_as_diagnostic": True, "rejected_subspans": [{"column": 1}], "interior_gap_of_line": 0,
                      "columns": [{"state": "full_height_conflict"}], "rejected_column_fraction": 0.25}]
            b["free_space_scorer"] = {"k": k, "tolerance": 0.01 * k, "unique_extrapolated_spans_scored": 3, "spans_accepted": 3 - (k == reject_k),
                                      "spans_rejected": int(k == reject_k), "spans_conflict_diagnostic_only": 1, "calls": calls}
            if k == reject_k:
                b = {**boundary_for(self.poly, status="incomplete"), "vertices_xy": None, "edges": [], "free_space_scorer": b["free_space_scorer"]}
            return b
        self.hooks["assemble_boundary_ray"] = ray

    def test_variants_frozen_before_evaluation_and_without_truth(self):
        self.assertNotIn("scene", " ".join(inspect.signature(C.run_ray_variants).parameters))
        self.with_ray_hook()
        res = C.run_ray_comparison(self.budgets, self.root / "out", hooks=self.hooks, events=self.events)
        names = [e[0] for e in self.events]
        self.assertLess(names.index("ray_variants_frozen"), names.index("evaluate_budget"))
        self.assertEqual(res["k_grid"], list(C.RAY_K_GRID))
        schemes = {r["scheme"] for r in res["table"]}
        self.assertEqual(schemes, {"default"} | {C.variant_name(k) for k in C.RAY_K_GRID})
        for r in res["table"]:
            if r["scheme"] != "default":
                self.assertEqual(r["vs_default"]["change"], "identical_polygon")
                self.assertEqual(r["candidate_spans_scored"], 3)
        self.assertTrue(any(v for v in res["localized_examples"].values()))
        self.assertTrue((self.root / "out" / "ray_variants_index.json").exists())

    def test_polygon_change_is_reported_per_k_and_deterministic(self):
        self.with_ray_hook(reject_k=4.0)
        a = C.run_ray_comparison(self.budgets, self.root / "out1", hooks=self.hooks, evaluate=False)
        self.with_ray_hook(reject_k=4.0)
        b = C.run_ray_comparison(self.budgets, self.root / "out2", hooks=self.hooks, evaluate=False)
        row = next(r for r in a["table"] if r["budget"] == "b1" and r["scheme"] == C.variant_name(4.0))
        self.assertEqual(row["vs_default"]["change"], "polygon_lost")
        self.assertEqual(row["candidate_spans_rejected"], 1)
        self.assertEqual(a["table"], b["table"])

    def test_scorer_controls_wall_exists_free_space_and_behind(self):
        import test_boundary_assembly as tba
        self.with_ray_hook()
        self.hooks["load_prediction"] = lambda p: tba._pred(4)
        self.hooks["extract_wall_lines"] = lambda pred, axis: tba._wall_dict()
        self.budgets = self.budgets[:1]
        C.run_ray_comparison(self.budgets, self.root / "out", ks=(2.0,), hooks=self.hooks, evaluate=False)
        res = C.scorer_controls(self.root / "out", ks=(2.0,), hooks=self.hooks, stride=1)
        self.assertTrue((self.root / "out" / "scorer_controls" / "controls.json").exists())
        s = res["summary"]
        self.assertEqual(s["k2/observed_span"]["rejected"], 0)
        self.assertEqual(s["k2/observed_span"]["raw_scorer_boolean_true"], 0)
        self.assertEqual(s["k2/displaced_negative_normal"]["rejected"], s["k2/displaced_negative_normal"]["n"])   # z=4: in front of the wall
        self.assertEqual(s["k2/displaced_positive_normal"]["mostly_unobserved"], s["k2/displaced_positive_normal"]["n"])  # z=6: behind
        with self.assertRaises(FileExistsError):
            C.scorer_controls(self.root / "out", ks=(2.0,), hooks=self.hooks)

    def test_polygon_delta_cases(self):
        p = boundary_for(rect(2, 1.3333))
        self.assertEqual(C.polygon_delta(p, p)["change"], "identical_polygon")
        self.assertEqual(C.polygon_delta(p, boundary_for(rect(2, 1.9)))["change"], "geometry_changed")
        self.assertEqual(C.polygon_delta(p, boundary_for(np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)))["change"], "topology_changed")


if __name__ == "__main__":
    unittest.main()


