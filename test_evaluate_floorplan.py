"""Tests for the floor-plan evaluator (synthetic predictions against private scenes)."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import evaluate_floorplan as ev

SCENE = {"width": 5.2, "length": 4.0, "height": 2.6}
RECT = np.array([[-2.6, -2.0], [2.6, -2.0], [2.6, 2.0], [-2.6, 2.0]])
L_OUTLINE = np.array([[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]], float)
L_SCENE = {"polygon_xy_m": L_OUTLINE.tolist(), "ceiling_height_m": 2.6}


def _rot(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s], [s, c]])


def _prediction(outline, k=3.7, deg=33., shift=(11., -4.), start=0, mirror=False, labels=None, status="complete_polygon_hypothesis"):
    """A predicted plan in an arbitrary frame: unit scale k, rotated, translated, vertex order started elsewhere."""
    v = np.asarray(outline, float)
    if mirror:
        v = (v * np.array([-1., 1.]))[::-1]
    v = np.roll(v, -start, axis=0)
    v = v @ _rot(deg).T * k + np.asarray(shift)
    n = len(v)
    edges = [{"index": i, "start_xy": v[i].tolist(), "end_xy": v[(i + 1) % n].tolist(),
              "length": float(np.linalg.norm(v[(i + 1) % n] - v[i])), "label": (labels or ["observed"] * n)[i]} for i in range(n)]
    return {"status": status, "vertices_xy": v.tolist(), "edges": edges, "incomplete_boundaries": []}


def _height(k=3.7, height=2.6, status="estimated_unscaled_separation"):
    return {"status": status, "separation_unscaled": height * k, "reasons": []}


class ScoringTests(unittest.TestCase):
    def test_exact_prediction_passes_both_gates_separately(self):
        out = ev.evaluate_floorplan(_prediction(RECT, start=2), SCENE, _height())
        self.assertEqual(out["status"], "scored", out["failure_reasons"])
        m = out["metrics"]
        self.assertGreater(m["iou"], .995); self.assertAlmostEqual(m["area_ratio"], 1.0, places=3)
        self.assertEqual((m["predicted_edge_count"], m["true_edge_count"]), (4, 4))
        self.assertLess(m["max_abs_wall_error_cm"], .5); self.assertLess(m["max_rel_wall_error_percent"], .5)
        self.assertTrue(out["custom_1_5cm_horizontal_diagnostic"]["pass"])
        self.assertTrue(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertFalse(out["custom_1_5cm_horizontal_diagnostic"]["assessment_eligible"])
        self.assertFalse(out["wall_length_3_percent_threshold_oracle_normalised"]["assessment_eligible"])
        self.assertTrue(out["gates_never_merged"]); self.assertFalse(out["benchmark_stage_complete"])
        self.assertIn("EVALUATOR-ORACLE", out["alignment"]["correspondence"])
        self.assertAlmostEqual(out["scale"]["prediction_scale"], 3.0 / (2.6 * 3.7))
        self.assertAlmostEqual(out["scale"]["truth_scale"], 3.0 / 2.6)

    def test_small_stretch_fails_custom_but_passes_official_and_large_fails_both(self):
        small = _prediction(RECT * np.array([1.01, 1.0]))          # 1% wider: ~6 cm at 3 m normalisation
        out = ev.evaluate_floorplan(small, SCENE, _height())
        self.assertFalse(out["custom_1_5cm_horizontal_diagnostic"]["pass"])
        self.assertTrue(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertGreater(out["metrics"]["max_abs_wall_error_cm"], 1.5)
        large = ev.evaluate_floorplan(_prediction(RECT * np.array([1.06, 1.0])), SCENE, _height())
        self.assertFalse(large["custom_1_5cm_horizontal_diagnostic"]["pass"])
        self.assertFalse(large["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertEqual(large["status"], "scored")

    def test_noisy_vertices_degrade_iou_and_error_but_stay_scored(self):
        rng = np.random.default_rng(4)
        pred = _prediction(L_OUTLINE + rng.normal(size=L_OUTLINE.shape) * .03)
        out = ev.evaluate_floorplan(pred, L_SCENE, _height())
        self.assertEqual(out["status"], "scored", out["failure_reasons"])
        self.assertGreater(out["metrics"]["iou"], .9); self.assertLess(out["metrics"]["iou"], .9999)
        self.assertEqual(out["metrics"]["predicted_edge_count"], 6)

    def test_mirrored_prediction_only_scored_when_sign_unresolved(self):
        pred = _prediction(L_OUTLINE, mirror=True, start=3)
        free = ev.evaluate_floorplan(pred, L_SCENE, _height(), axis_sign_resolved=False)
        self.assertEqual(free["status"], "scored")
        self.assertTrue(free["alignment"]["mirrored"]); self.assertTrue(free["alignment"]["mirror_tried"])
        self.assertGreater(free["metrics"]["iou"], .995)
        self.assertTrue(free["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertIn("EVALUATOR-ORACLE", free["alignment"]["correspondence"])
        fixed = ev.evaluate_floorplan(pred, L_SCENE, _height(), axis_sign_resolved=True)
        self.assertFalse(fixed["alignment"]["mirror_tried"]); self.assertFalse(fixed["alignment"]["mirrored"])
        self.assertLess(fixed["metrics"]["iou"], .9)
        self.assertFalse(fixed["custom_1_5cm_horizontal_diagnostic"]["pass"])

    def test_wrong_height_scale_is_not_corrected(self):
        out = ev.evaluate_floorplan(_prediction(RECT), SCENE, _height(height=2.6 * 1.1))   # separation 10% too large
        self.assertEqual(out["status"], "scored")
        self.assertLess(out["metrics"]["area_ratio"], .85)                                  # prediction shrunk by 1/1.1
        self.assertFalse(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertGreater(out["metrics"]["max_rel_wall_error_percent"], 8)

    def test_prediction_scale_never_comes_from_truth(self):
        base = ev.evaluate_floorplan(_prediction(RECT), SCENE, _height())
        corrupted = ev.evaluate_floorplan(_prediction(RECT), {**SCENE, "height": 3.9}, _height())
        self.assertEqual(base["scale"]["prediction_scale"], corrupted["scale"]["prediction_scale"])
        self.assertNotEqual(base["scale"]["truth_scale"], corrupted["scale"]["truth_scale"])
        self.assertAlmostEqual(corrupted["scale"]["truth_scale"], 3.0 / 3.9)
        # the corrupted truth is normalised on its own, so a prediction that matched before no longer does
        self.assertTrue(base["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertFalse(corrupted["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        # and the truth cannot rescue a missing prediction scale
        missing = ev.evaluate_floorplan(_prediction(RECT), SCENE, None)
        self.assertEqual(missing["status"], "failed")
        self.assertIn("height_estimate_missing", missing["failure_reasons"])
        self.assertIsNone(missing["scale"]["prediction_scale"])
        self.assertIsNotNone(missing["scale"]["truth_scale"])

    def test_height_estimate_problems_fail_with_named_reasons(self):
        cases = ((_height(status="ambiguous"), "height_estimate_ambiguous"),
                 (_height(status="incomplete"), "height_estimate_not_estimated:incomplete"),
                 ({"status": "estimated_unscaled_separation", "separation_unscaled": None}, "height_estimate_separation_invalid"),
                 ({"status": "estimated_unscaled_separation", "separation_unscaled": -1.0}, "height_estimate_separation_invalid"))
        for estimate, reason in cases:
            with self.subTest(reason):
                out = ev.evaluate_floorplan(_prediction(RECT), SCENE, estimate)
                self.assertEqual(out["status"], "failed"); self.assertIn(reason, out["failure_reasons"])
                self.assertFalse(out["custom_1_5cm_horizontal_diagnostic"]["pass"])
                self.assertFalse(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        no_truth_height = ev.evaluate_floorplan(_prediction(RECT), {"width": 5.2, "length": 4.0}, _height())
        self.assertIn("truth_ceiling_height_unavailable", no_truth_height["failure_reasons"])

    def test_incomplete_or_missing_prediction_fails(self):
        incomplete = {"status": "incomplete", "vertices_xy": None, "edges": [],
                      "incomplete_boundaries": [{"reason": "fragmented_footprint"}]}
        out = ev.evaluate_floorplan(incomplete, SCENE, _height())
        self.assertEqual(out["status"], "failed")
        self.assertIn("prediction_incomplete", out["failure_reasons"])
        self.assertIn("incomplete_boundary:fragmented_footprint", out["failure_reasons"])
        self.assertIsNone(out["metrics"])
        self.assertFalse(out["custom_1_5cm_horizontal_diagnostic"]["pass"])
        self.assertFalse(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertIn("prediction_missing", ev.evaluate_floorplan(None, SCENE, _height())["failure_reasons"])
        self.assertEqual(ev.evaluate_floorplan(
            {**_prediction(RECT), "vertices_xy": [[0, 0], [1, 1]]}, SCENE, _height())["status"], "failed")

    def test_wrong_edge_count_fails_both_gates(self):
        pentagon = np.array([[-2.6, -2.0], [2.6, -2.0], [2.6, 2.0], [0.0, 2.8], [-2.6, 2.0]])
        out = ev.evaluate_floorplan(_prediction(pentagon), SCENE, _height())
        self.assertEqual(out["status"], "scored")
        self.assertEqual((out["metrics"]["predicted_edge_count"], out["metrics"]["true_edge_count"]), (5, 4))
        self.assertIn("edge_count_mismatch", out["custom_1_5cm_horizontal_diagnostic"]["failure_reasons"])
        self.assertFalse(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])

    def test_edge_labels_follow_the_original_edges_through_mirroring_and_reordering(self):
        labels = ["observed", "inferred", "observed", "inferred", "inferred", "observed"]
        pred = _prediction(L_OUTLINE, mirror=False, start=0, labels=labels)
        out = ev.evaluate_floorplan(pred, L_SCENE, _height(), axis_sign_resolved=True)
        self.assertEqual(out["metrics"]["edge_labels"], {"observed": 3, "inferred": 3})
        self.assertEqual(out["predicted_edge_labels"], labels)
        mirrored = ev.evaluate_floorplan(_prediction(L_OUTLINE, mirror=True, labels=labels), L_SCENE, _height())
        self.assertTrue(mirrored["alignment"]["mirrored"])
        # The mirrored polygon's edge j is original edge (n-2-j); labels belong to the predicted polygon's own edges.
        # Edges of unique length (6, 2 and 5 m: original edges 0, 3, 5) must keep the label of the predicted edge they match.
        by_len = {round(r["true_length_m"] / (3.0 / 2.6), 3): r["label"] for r in mirrored["metrics"]["edge_errors"]}
        self.assertEqual((by_len[6.0], by_len[2.0], by_len[5.0]), (labels[4], labels[1], labels[5]))
        self.assertEqual(sorted(r["label"] for r in mirrored["metrics"]["edge_errors"]), sorted(labels))
    def test_missing_provenance_and_partial_retention_block_any_whole_room_claim(self):
        import evaluate_floorplan as EF
        none = EF.whole_room_claim(EF.retained_capture_fraction(None), "scored")
        self.assertFalse(none["allowed"]); self.assertIn("capture_provenance_missing", none["blockers"])
        part = EF.whole_room_claim(EF.retained_capture_fraction({"retained_capture_fraction": 0.35}), "scored")
        self.assertIn("retained_capture_below_full", part["blockers"]); self.assertNotIn("capture_provenance_missing", part["blockers"])
        full = EF.whole_room_claim(EF.retained_capture_fraction({"retained_capture_fraction": 1.0}), "scored")
        self.assertFalse(full["allowed"])                                                # still an oracle-normalised diagnostic
        self.assertIn("oracle_normalised_alignment_diagnostic_is_not_an_assessment_metric", full["blockers"])
        self.assertIn("prediction_not_scored", EF.whole_room_claim(EF.retained_capture_fraction({"retained_capture_fraction": 1.0}), "failed")["blockers"])

    def test_retained_capture_fraction_is_read_from_the_manifest(self):

        manifest = {"stages": [{"name": "merge", "report": {"merge": {"retained_capture_fraction": 0.8125}}}]}
        out = ev.evaluate_floorplan(_prediction(RECT), SCENE, _height(), manifest=manifest)
        self.assertEqual(out["retained_capture"]["value"], .8125)
        self.assertTrue(out["retained_capture"]["supplied"])
        none = ev.evaluate_floorplan(_prediction(RECT), SCENE, _height())
        self.assertIsNone(none["retained_capture"]["value"]); self.assertFalse(none["retained_capture"]["supplied"])

    def test_house_layout_scene_and_multiroom_requires_room_id(self):
        layout = {"rooms": [{"room_id": "a", "polygon_xy_m": L_OUTLINE.tolist(), "ceiling_height_m": 2.6},
                            {"room_id": "b", "polygon_xy_m": RECT.tolist(), "ceiling_height_m": 2.6}]}
        out = ev.evaluate_floorplan(_prediction(L_OUTLINE), layout, _height(), room_id="a")
        self.assertEqual(out["status"], "scored"); self.assertGreater(out["metrics"]["iou"], .995)
        with self.assertRaises(ValueError):
            ev.evaluate_floorplan(_prediction(L_OUTLINE), layout, _height())

    def test_end_to_end_with_footprint_estimate(self):
        from footprint import estimate_footprint
        from test_footprint import _room, RECT as FIXTURE_RECT
        data, pts, axis, _truth, ids = _room(FIXTURE_RECT, 3)
        footprint = estimate_footprint(data, pts, axis, ids)
        self.assertEqual(footprint["status"], "complete_polygon_hypothesis")
        scene = {"width": 6.0, "length": 5.0, "height": 2.6}
        out = ev.evaluate_floorplan(footprint, scene, {"status": "estimated_unscaled_separation",
                                                      "separation_unscaled": footprint["floor_ceiling_separation_unscaled"]})
        self.assertEqual(out["status"], "scored", out["failure_reasons"])
        self.assertGreater(out["metrics"]["iou"], .97)
        self.assertTrue(out["wall_length_3_percent_threshold_oracle_normalised"]["pass"])
        self.assertTrue(out["alignment"]["mirror_tried"])          # footprint axis sign is unresolved


class OverlayTests(unittest.TestCase):
    def test_overlay_files_and_no_overwrite(self):
        labels = ["observed", "inferred", "observed", "inferred", "observed", "observed"]
        out = ev.evaluate_floorplan(_prediction(L_OUTLINE, labels=labels), L_SCENE, _height(),
                                    manifest={"retained_capture_fraction": 0.9})
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new_eval"
            paths = ev.write_overlay(out, target)
            svg = Path(paths["svg"]).read_text(encoding="utf-8")
            self.assertIn("stroke-dasharray", svg)                  # inferred edges are dashed
            self.assertIn("custom 1.5 cm diagnostic", svg); self.assertIn("3% threshold (oracle-normalised", svg)
            self.assertIn("EVALUATOR-ORACLE", svg)
            png = cv2.imread(paths["png"])
            self.assertIsNotNone(png); self.assertGreater(png.shape[0], 200)
            self.assertLess(int(png.mean()), 254)                    # something was drawn
            saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "scored")
            with self.assertRaises(FileExistsError):
                ev.write_overlay(out, target)

    def test_overlay_for_a_failed_prediction_shows_the_reason(self):
        out = ev.evaluate_floorplan({"status": "incomplete", "vertices_xy": None,
                                     "incomplete_boundaries": [{"reason": "no_bracketing_pair"}]}, SCENE, _height())
        with tempfile.TemporaryDirectory() as tmp:
            paths = ev.write_overlay(out, Path(tmp) / "failed")
            svg = Path(paths["svg"]).read_text(encoding="utf-8")
            self.assertIn("no_bracketing_pair", svg); self.assertIn("FAIL", svg)
            self.assertIsNotNone(cv2.imread(paths["png"]))

    def test_cli_reads_the_private_scene_from_the_capture_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "capture" / "private_ground_truth").mkdir(parents=True)
            (tmp / "capture" / "private_ground_truth" / "scene.json").write_text(json.dumps(SCENE), encoding="utf-8")
            (tmp / "pred.json").write_text(json.dumps(_prediction(RECT)), encoding="utf-8")
            (tmp / "height.json").write_text(json.dumps(_height()), encoding="utf-8")
            argv = ["evaluate_floorplan.py", "--prediction", str(tmp / "pred.json"), "--capture", str(tmp / "capture"),
                    "--height", str(tmp / "height.json"), "--output-dir", str(tmp / "out")]
            with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(ev.main(), 0)
            summary = json.loads(buf.getvalue())
            self.assertEqual(summary["status"], "scored"); self.assertTrue(summary["official_3pct_pass"])
            self.assertTrue((tmp / "out" / "overlay.svg").exists())
            with mock.patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ev.main()                                           # output folder exists: refuse


if __name__ == "__main__":
    unittest.main()


