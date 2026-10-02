import copy
import unittest

from evaluation import evaluate


def make_docs():
    pred_runs, gt_runs = [], []
    for stage in ("single_room", "multiroom", "multifloor"):
        for house in range(10):
            for seed in (1, 2, 3):
                common = {
                    "stage": stage, "house_id": f"h{house}", "capture_seed": seed,
                    "rooms": [{"room_id": "r1", "walls": [{"wall_id": "w1", "video_length_m": 4.0}],
                               "wall_height_m": 2.7, "openings": [{"opening_id": "d1", "width_m": .9, "height_m": 2.0}],
                               "shape_dimensions": [4.0, 3.0, 2.7]}],
                }
                gt_runs.append(copy.deepcopy(common))
                prediction = copy.deepcopy(common)
                prediction["scale_source"] = "documented_external_calibration"
                prediction["rooms"][0]["complete_surface_pairs"] = 3
                pred_runs.append(prediction)
    return {"schema_version": 1, "runs": pred_runs}, {"schema_version": 1, "runs": gt_runs}


class EvaluationTests(unittest.TestCase):
    def test_missing_run_fails_coverage_and_run_match(self):
        pred, gt = make_docs()
        pred["runs"].pop()
        result = evaluate(pred, gt)
        self.assertFalse(result["completion"])
        self.assertFalse(result["gates"]["matched_run_coverage"])
        self.assertFalse(result["coverage"]["multifloor"]["pass"])

    def test_erroneous_wall_height_and_phantoms_fail_gates(self):
        pred, gt = make_docs()
        target = next(r for r in pred["runs"] if r["stage"] == "single_room" and r["house_id"] == "h0" and r["capture_seed"] == 1)
        target["rooms"][0]["walls"][0]["video_length_m"] = 4.2
        target["rooms"][0]["wall_height_m"] = 2.72
        target["rooms"][0]["openings"].append({"opening_id": "phantom", "width_m": .8, "height_m": 2.0})
        result = evaluate(pred, gt)
        self.assertFalse(result["gates"]["wall_length_error_le_3_percent"])
        self.assertFalse(result["gates"]["wall_height_error_le_0_015m"])
        self.assertFalse(result["gates"]["height_spread_le_0_01m"])
        self.assertEqual(result["metric"]["phantom_openings"], 1)
        self.assertLess(result["metric"]["opening_penalized_score"], 1)

    def test_metric_opening_gate_is_per_capture_and_pooled_score_remains_diagnostic(self):
        pred, gt = make_docs()
        # 29 captures are exact; one capture misses its only opening.
        pred["runs"][0]["rooms"][0]["openings"] = []
        result = evaluate(pred, gt)
        self.assertGreater(result["metric"]["opening_penalized_score"], .95)
        row = result["metric"]["opening_score_by_capture"]["single_room|h0|1"]
        self.assertEqual(row["status"], "measured")
        self.assertEqual(row["penalized_score"], 0)
        self.assertFalse(result["gates"]["opening_score_ge_85_percent"])

    def test_metric_scene_with_no_openings_is_not_measured_and_does_not_pass(self):
        pred, gt = make_docs()
        for run in pred["runs"]:
            run["rooms"][0]["openings"] = []
        for run in gt["runs"]:
            run["rooms"][0]["openings"] = []
        result = evaluate(pred, gt)
        self.assertIsNone(result["metric"]["opening_penalized_score"])
        self.assertTrue(all(row["status"] == "not_measured" and row["penalized_score"] is None
                            for row in result["metric"]["opening_score_by_capture"].values()))
        self.assertFalse(result["gates"]["opening_score_ge_85_percent"])

    def test_missing_scale_provenance_is_rejected(self):
        pred, gt = make_docs()
        del pred["runs"][0]["scale_source"]
        with self.assertRaisesRegex(ValueError, "scale_source"):
            evaluate(pred, gt)

    def test_multiroom_and_multifloor_never_claim_supported_completion(self):
        pred, gt = make_docs()
        result = evaluate(pred, gt)
        self.assertFalse(result["gates"]["supported_stage_checks"])
        self.assertFalse(result["completion"])

    def test_shape_only_accepts_arbitrary_units_and_never_claims_metric_completion(self):
        pred, gt = make_docs()
        pred["evaluation_mode"] = "shape_only"
        for run in pred["runs"]:
            run.pop("scale_source")
            room = run["rooms"][0]
            room["walls"][0]["video_length_m"] *= 7.0
            room["wall_height_m"] *= 7.0
            room["shape_dimensions"] = [d * 7.0 for d in room["shape_dimensions"]]
        result = evaluate(pred, gt)
        self.assertFalse(result["completion"])
        self.assertEqual(result["metric_gates"], "not_evaluated")
        self.assertAlmostEqual(result["shape_only"]["max_cuboid_extent_relative_error"], 0)
        self.assertTrue(result["shape_only"]["provisional_cuboid_shape_gate_le_3_percent"])
        self.assertFalse(result["shape_only"]["height_recovered"])

    def test_shape_gate_fails_when_an_opposing_surface_pair_is_missing(self):
        pred, gt = make_docs()
        pred["evaluation_mode"] = "shape_only"
        for run in pred["runs"]:
            run.pop("scale_source")
        pred["runs"][0]["rooms"][0]["complete_surface_pairs"] = 2
        result = evaluate(pred, gt)
        self.assertFalse(result["shape_only"]["data_complete_for_provisional_gate"])
        self.assertEqual(result["shape_only"]["incomplete_surface_pair_room_count"], 1)

    def test_null_height_is_incomplete_measurement_not_invalid_schema(self):
        pred, gt = make_docs()
        pred["runs"][0]["rooms"][0]["wall_height_m"] = None
        result = evaluate(pred, gt)
        self.assertFalse(result["gates"]["wall_height_error_le_0_015m"])
        self.assertEqual(result["metric"]["missing_height_measurement_count"], 1)

    def test_height_normalized_3m_protocol_forces_height_and_gates_horizontal_extent(self):
        pred, gt = make_docs()
        pred["evaluation_mode"] = "height_normalized_3m"
        for run in pred["runs"]:
            run.pop("scale_source")
            room = run["rooms"][0]
            room["height_axis_index"] = 2
            room["shape_dimensions"] = [4.02, 3.0, 2.7]
        result = evaluate(pred, gt)
        first = result["height_normalized_3m"]["per_run"][0]
        self.assertEqual(first["axis_matching"], "independently_predicted_height_axis")
        self.assertAlmostEqual(first["horizontal_absolute_errors_cm"][0], 2.2222222222, places=6)
        self.assertFalse(first["custom_1_5cm_pass"])
        self.assertFalse(result["completion"])

    def test_shape_only_partial_extent_counts_as_incomplete(self):
        pred, gt = make_docs()
        pred["evaluation_mode"] = "shape_only"
        for run in pred["runs"]:
            run.pop("scale_source")
        pred["runs"][0]["rooms"][0]["shape_dimensions"] = [4.0, None, 2.7]
        result = evaluate(pred, gt)
        self.assertFalse(result["shape_only"]["data_complete_for_provisional_gate"])
        self.assertGreater(result["shape_only"]["missing_shape_dimension_count"], 0)


def anchor_docs(houses=10, seeds=(1, 2, 3), height=3.0):
    """Raw (arbitrary-unit) predictions with a different unknown scale per capture, plus ground truth."""
    pred_runs, gt_runs = [], []
    for house in range(houses):
        for seed in seeds:
            k = 0.37 + 1.9 * ((house * 7 + seed * 3) % 11) / 11  # unknown per-capture scale
            gt_room = {"room_id": "r1", "layout": "cuboid", "wall_height_m": height, "shape_dimensions": [4.0, 3.5, height],
                       "walls": [{"wall_id": w, "video_length_m": v} for w, v in (("w1", 4.0), ("w2", 3.5), ("w3", 4.0), ("w4", 3.5))],
                       "openings": [{"opening_id": "d1", "width_m": .9, "height_m": 2.0}, {"opening_id": "win1", "width_m": 1.2, "height_m": 1.0}]}
            pred_room = {"room_id": "r1", "layout_status": "complete_box_hypothesis", "up_source": "camera_motion_prior",
                         "height_axis_index": 2, "complete_surface_pairs": 3, "wall_height_m": height * k,
                         "shape_dimensions": [4.0 * k, 3.5 * k, height * k],
                         "walls": [{"wall_id": w, "video_length_m": v * k} for w, v in (("w1", 4.0), ("w2", 3.5), ("w3", 4.0), ("w4", 3.5))],
                         "openings": [{"opening_id": "d1", "width_m": .9 * k, "height_m": 2.0 * k},
                                      {"opening_id": "win1", "width_m": 1.2 * k, "height_m": 1.0 * k}]}
            anchors = {"wall": {"wall_id": "w1", "length_m": 4.0, "source": "declared_tape"},
                       "door": {"opening_id": "d1", "width_m": .9, "source": "declared_door_spec"}}
            gt_runs.append({"stage": "single_room", "house_id": f"h{house}", "capture_seed": seed, "rooms": [gt_room]})
            pred_runs.append({"stage": "single_room", "house_id": f"h{house}", "capture_seed": seed, "anchors": anchors, "rooms": [pred_room]})
    return ({"schema_version": 1, "evaluation_mode": "anchor_protocols", "prediction_units": "arbitrary_unscaled", "runs": pred_runs},
            {"schema_version": 1, "runs": gt_runs})


def passes(result, name):
    return result["protocols"][name]["official_gates_pass"]


class AnchorProtocolTests(unittest.TestCase):
    def test_exact_prediction_passes_each_protocol_separately_and_never_claims_completion(self):
        pred, gt = anchor_docs()
        result = evaluate(pred, gt)
        self.assertEqual(set(result["protocols"]), {"ceiling_height_3m", "known_horizontal_wall", "known_door_width"})
        for name in result["protocols"]:
            self.assertTrue(passes(result, name), (name, result["protocols"][name]["official_gates"]))
            self.assertTrue(result["protocols"][name]["custom_1_5cm_diagnostic"]["pass_all_scored_walls"])
            self.assertFalse(result["protocols"][name]["custom_1_5cm_diagnostic"]["official_gate"])
            self.assertTrue(result["protocols"][name]["custom_1_5cm_overall_dimension_diagnostic"]["pass_all_measured_heldout_dimensions"])
        self.assertFalse(result["completion"])
        self.assertIn("none", result["benchmark_claim"])
        self.assertFalse(result["orientation"]["oracle_axis_used_in_pass_results"])
        self.assertFalse(result["oracle_diagnostics"]["used_for_pass_results"])

    def test_ceiling_protocol_does_not_score_height_and_wall_protocol_does(self):
        pred, gt = anchor_docs()
        result = evaluate(pred, gt)
        a, b = result["protocols"]["ceiling_height_3m"], result["protocols"]["known_horizontal_wall"]
        self.assertIsNone(a["official_gates"]["wall_height_error_le_0_015m"])
        self.assertIsNone(a["metrics"]["max_wall_height_error_m"])
        self.assertIsNotNone(b["metrics"]["max_wall_height_error_m"])
        self.assertTrue(all("w1" in r["scored_wall_ids"] for r in a["runs"]))

    def test_anchor_is_excluded_from_scored_dimensions(self):
        pred, gt = anchor_docs()
        result = evaluate(pred, gt)
        for run in result["protocols"]["known_horizontal_wall"]["runs"]:
            self.assertNotIn("w1", run["scored_wall_ids"])
            self.assertEqual(sorted(run["scored_wall_ids"]), ["w2", "w3", "w4"])
        self.assertEqual(result["protocols"]["known_door_width"]["metrics"]["opening_total"], 30)  # win1 only
        self.assertEqual(result["protocols"]["known_horizontal_wall"]["metrics"]["opening_total"], 60)

    def test_one_failing_capture_cannot_hide_in_pooled_opening_score(self):
        pred, gt = anchor_docs(houses=10, seeds=(1, 2, 3))
        # Keep only the held-out window for the wall protocol; make one capture fail it.
        pred["runs"][0]["rooms"][0]["openings"][0]["width_m"] = .6
        result = evaluate(pred, gt)["protocols"]["known_door_width"]
        self.assertGreater(result["metrics"]["opening_penalized_score"], .95)
        self.assertLess(result["runs"][0]["opening_score"], .85)
        self.assertFalse(result["official_gates"]["opening_score_ge_85_percent"])

    def test_phantom_opening_is_penalized_per_capture(self):
        pred, gt = anchor_docs()
        pred["runs"][0]["rooms"][0]["openings"].append({"opening_id": "phantom", "width_m": .7, "height_m": .8})
        result = evaluate(pred, gt)["protocols"]["known_door_width"]
        self.assertFalse(result["official_gates"]["opening_score_ge_85_percent"])
        self.assertLess(result["runs"][0]["opening_score"], 1)

    def test_opening_score_with_no_heldout_openings_is_unmeasured_and_fails(self):
        pred, gt = anchor_docs()
        for run in pred["runs"]:
            run["rooms"][0]["openings"] = [o for o in run["rooms"][0]["openings"] if o["opening_id"] == "win1"]
        result = evaluate(pred, gt)["protocols"]["known_door_width"]
        self.assertIsNone(result["metrics"]["opening_penalized_score"])
        self.assertFalse(result["official_gates"]["opening_score_ge_85_percent"])

    def test_overall_dimension_diagnostic_catches_height_error_with_perfect_walls(self):
        pred, gt = anchor_docs()
        for run in pred["runs"]:
            room = run["rooms"][0]
            room["wall_height_m"] *= 1.01
            room["shape_dimensions"][2] *= 1.01
        result = evaluate(pred, gt)["protocols"]["known_horizontal_wall"]
        self.assertTrue(result["custom_1_5cm_diagnostic"]["pass_all_scored_walls"])
        self.assertFalse(result["custom_1_5cm_overall_dimension_diagnostic"]["pass_all_measured_heldout_dimensions"])

    def test_scale_comes_from_the_declared_anchor_not_ground_truth(self):
        pred, gt = anchor_docs()
        base = evaluate(pred, gt)
        # Corrupt every non-anchor ground-truth wall: scale must not move, the error must show.
        for run in gt["runs"]:
            for wall in run["rooms"][0]["walls"]:
                if wall["wall_id"] != "w1":
                    wall["video_length_m"] *= 1.2
        shifted = evaluate(pred, gt)
        first, again = (r["protocols"]["known_horizontal_wall"]["runs"][0] for r in (base, shifted))
        self.assertEqual(first["scale"], again["scale"])
        k = pred["runs"][0]["rooms"][0]["walls"][0]["video_length_m"]
        self.assertAlmostEqual(first["scale"], 4.0 / k)
        self.assertFalse(first["scale_provenance"]["groundtruth_used_for_scale"])
        self.assertFalse(passes(shifted, "known_horizontal_wall"))
        self.assertGreater(shifted["protocols"]["known_horizontal_wall"]["metrics"]["max_relative_wall_length_error"], .15)

    def test_missing_wall_anchor_fails_only_that_protocol(self):
        pred, gt = anchor_docs()
        for run in pred["runs"]:
            del run["anchors"]["wall"]
        result = evaluate(pred, gt)
        wall = result["protocols"]["known_horizontal_wall"]
        self.assertFalse(wall["official_gates_pass"])
        self.assertFalse(wall["official_gates"]["anchors_valid_all_runs"])
        self.assertEqual(wall["failure_reason_counts"], {"anchor_missing": 30})
        self.assertTrue(passes(result, "ceiling_height_3m")); self.assertTrue(passes(result, "known_door_width"))

    def test_ambiguous_or_contradicting_anchor_fails(self):
        pred, gt = anchor_docs()
        pred["runs"][0]["anchors"]["wall"] = [{"wall_id": "w1", "length_m": 4.0}, {"wall_id": "w2", "length_m": 3.0}]
        pred["runs"][1]["anchors"]["wall"]["length_m"] = 4.2  # declared value contradicts the scene
        result = evaluate(pred, gt)
        counts = result["protocols"]["known_horizontal_wall"]["failure_reason_counts"]
        self.assertEqual(counts, {"ambiguous_anchor": 1, "anchor_value_conflicts_with_groundtruth": 1})
        self.assertFalse(passes(result, "known_horizontal_wall"))

    def test_door_anchor_failure_leaves_other_protocols_alone(self):
        for mutate, reason in ((lambda r: r["rooms"][0]["openings"].pop(0), "anchor_not_in_prediction"),
                               (lambda r: r["rooms"][0]["openings"][0].update(width_m=None), "anchor_measurement_unavailable"),
                               (lambda r: r["anchors"]["door"].update(width_m=.95), "anchor_value_conflicts_with_groundtruth")):
            with self.subTest(reason):
                pred, gt = anchor_docs()
                mutate(pred["runs"][4])
                result = evaluate(pred, gt)
                door = result["protocols"]["known_door_width"]
                self.assertFalse(door["official_gates_pass"])
                self.assertEqual(door["failure_reason_counts"], {reason: 1})
                if reason == "anchor_value_conflicts_with_groundtruth":
                    self.assertTrue(passes(result, "known_horizontal_wall"))
                else:
                    self.assertFalse(passes(result, "known_horizontal_wall"))  # missing held-out opening is not a measured pass

    def test_wrong_height_axis_fails_and_oracle_axis_is_diagnostic_only(self):
        pred, gt = anchor_docs()
        for run in pred["runs"]:
            room = run["rooms"][0]
            room["height_axis_index"] = 0                       # declared prior picked the longest axis
            room["wall_height_m"] = room["shape_dimensions"][0]
        result = evaluate(pred, gt)
        for name in result["protocols"]:
            self.assertFalse(passes(result, name), name)
        entry = result["oracle_diagnostics"]["per_run"][0]
        self.assertEqual(entry["oracle_height_axis_index"], 2)
        self.assertEqual(entry["predicted_height_axis_index"], 0)
        self.assertFalse(entry["predicted_axis_matches_oracle"])
        self.assertFalse(entry["used_for_pass_results"]); self.assertFalse(entry["independently_recovered"])
        self.assertFalse(result["orientation"]["oracle_axis_used_in_pass_results"])

    def test_missing_axis_declaration_is_not_rescued_by_the_oracle(self):
        pred, gt = anchor_docs()
        for run in pred["runs"]:
            run["rooms"][0].pop("up_source"); run["rooms"][0].pop("height_axis_index")
        result = evaluate(pred, gt)
        for name in result["protocols"]:
            self.assertFalse(passes(result, name))
            self.assertFalse(result["protocols"][name]["official_gates"]["declared_up_prior_all_runs"])
        self.assertEqual(result["orientation"]["runs_declaring_camera_motion_prior"], 0)

    def test_incomplete_polygon_predictions_fail_every_protocol(self):
        pred, gt = anchor_docs()
        room = pred["runs"][2]["rooms"][0]
        room["layout_status"] = "incomplete"
        for wall in room["walls"]:
            wall["video_length_m"] = None
        result = evaluate(pred, gt)
        for name in result["protocols"]:
            self.assertFalse(passes(result, name))
            self.assertFalse(result["protocols"][name]["official_gates"]["complete_layout_all_runs"])

    def test_non_cuboid_room_is_scored_by_individual_walls(self):
        pred, gt = anchor_docs(houses=10)
        lengths = (("w1", 6.0), ("w2", 3.0), ("w3", 3.0), ("w4", 2.0), ("w5", 3.0), ("w6", 5.0))
        for gr, pr in zip(gt["runs"], pred["runs"]):
            k = pr["rooms"][0]["wall_height_m"] / 3.0
            gr["rooms"][0].update(layout="non_cuboid_polygon", shape_dimensions=None,
                                  walls=[{"wall_id": w, "video_length_m": v} for w, v in lengths])
            pr["rooms"][0].update(layout_status="complete_polygon_hypothesis", shape_dimensions=None, height_axis_index=None,
                                  walls=[{"wall_id": w, "video_length_m": v * k} for w, v in lengths])
            pr["anchors"]["wall"]["length_m"] = 6.0
        result = evaluate(pred, gt)
        wall = result["protocols"]["known_horizontal_wall"]
        self.assertTrue(wall["official_gates_pass"], wall["official_gates"])
        self.assertEqual(sorted(wall["runs"][0]["scored_wall_ids"]), ["w2", "w3", "w4", "w5", "w6"])
        self.assertEqual(result["oracle_diagnostics"]["per_run"], [])  # no cuboid axes to match
        pred["runs"][0]["rooms"][0]["walls"][3]["video_length_m"] *= 1.1  # one wrong wall cannot hide
        self.assertFalse(passes(evaluate(pred, gt), "known_horizontal_wall"))

    def test_ceiling_protocol_normalizes_prediction_and_ground_truth_to_3m_and_shows_drift(self):
        pred, gt = anchor_docs(height=2.7)                      # true height need not be 3 m
        result = evaluate(pred, gt)
        self.assertTrue(passes(result, "ceiling_height_3m"))   # exact geometry: no drift after normalization
        run = result["protocols"]["ceiling_height_3m"]["runs"][0]
        self.assertAlmostEqual(run["anchor"]["groundtruth_normalization_scale"], 3.0 / 2.7)
        self.assertEqual(result["protocols"]["ceiling_height_3m"]["failure_reason_counts"], {})
        k = pred["runs"][0]["rooms"][0]["wall_height_m"] / 2.7
        self.assertAlmostEqual(run["scale"], 3.0 / (2.7 * k))
        pred["runs"][0]["rooms"][0]["walls"][1]["video_length_m"] *= 1.05   # a drifting wall is exposed, not absorbed
        drift = evaluate(pred, gt)["protocols"]["ceiling_height_3m"]
        self.assertFalse(drift["official_gates_pass"])
        self.assertGreater(drift["metrics"]["max_relative_wall_length_error"], .04)
    def test_predictions_must_be_raw_and_unscaled(self):
        pred, gt = anchor_docs()
        del pred["prediction_units"]
        with self.assertRaisesRegex(ValueError, "unscaled"):
            evaluate(pred, gt)
        pred, gt = anchor_docs()
        pred["runs"][0]["scale_source"] = "already_scaled"
        result = evaluate(pred, gt)
        self.assertIn("prediction_already_carries_scale_source",
                      result["protocols"]["known_door_width"]["failure_reason_counts"])
        self.assertFalse(passes(result, "known_door_width"))

    def test_insufficient_house_coverage_fails_protocol_gates(self):
        pred, gt = anchor_docs(houses=4)
        result = evaluate(pred, gt)
        self.assertFalse(result["protocols"]["known_horizontal_wall"]["official_gates"]["coverage_single_room"])
        self.assertFalse(passes(result, "known_horizontal_wall"))

    def test_small_capture_noise_breaks_repeatability_not_the_anchor_or_accuracy(self):
        pred, gt = anchor_docs()
        pred["runs"][0]["rooms"][0]["walls"][1]["video_length_m"] *= 1.004  # +0.4% on one wall
        wall = evaluate(pred, gt)["protocols"]["known_horizontal_wall"]
        self.assertTrue(wall["official_gates"]["wall_length_error_le_3_percent"])
        self.assertFalse(wall["official_gates"]["repeatability_le_0_01m"])


if __name__ == "__main__":
    unittest.main()
