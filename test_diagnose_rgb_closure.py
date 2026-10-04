import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import diagnose_rgb_closure as dg

SOURCE = Path(dg.__file__)


def wall_line(i, normal, offset, spans, frames=20, sd=0.004):
    m = np.asarray(normal, float)
    m = m / np.linalg.norm(m)
    return {"id": i, "normal_xy": m.tolist(), "offset": offset, "direction_xy": [-m[1], m[0]], "observed_spans": spans,
            "support": {"distinct_frames": frames, "frame_ids": list(range(frames)), "view_spread_deg": 80.0, "along_coverage_fraction": 1.0, "weight": 1.0},
            "uncertainty": {"offset_sd": sd, "angle_sd_deg": 0.3, "method": "leave_view_out"}, "height_range": [-0.4, 0.5], "weak": False}


def box_walls(extra=(), drop=None, noise=0.005):
    lines = [wall_line(0, [1, 0], 1.0, [[-0.9, 0.9]]), wall_line(1, [0, 1], 0.8, [[-1.0, 1.0]]),
             wall_line(2, [-1, 0], 1.0, [[-0.9, 0.9]]), wall_line(3, [0, -1], 0.8, [[-1.0, 1.0]])]
    if drop is not None:
        lines = [l for l in lines if l["id"] != drop]
    lines += list(extra)
    cams = [[0.0, 0.0], [0.2, 0.1], [-0.2, -0.1]]
    obs = {"frame_ids": list(range(len(cams))), "camera_xy": cams, "hits_xy": [[] for _ in cams], "hit_heights": [[] for _ in cams]}
    return {"model": "observed_wall_lines", "status": "ok", "reasons": [], "axis_world": [0, 0, 1],
            "frame": {"origin_world": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0], "extent": 3.0}, "noise_scale": noise, "lines": lines,
            "observations": obs}


class ReasonCountingTests(unittest.TestCase):
    def test_labels_and_layout_reasons(self):
        closed = dg.classify_failure({"status": "complete_polygon_hypothesis", "boundary_search": {"rejected_by_reason": {"a": 1}}})
        self.assertEqual(closed["label"], "closed")
        amb = dg.classify_failure({"status": "ambiguous", "incomplete_boundaries": [{"reason": "equally_supported_layouts"}]})
        self.assertEqual(amb["label"], "ambiguous")
        inc = dg.classify_failure({"status": "incomplete", "incomplete_boundaries": [
            {"reason": "no_closed_boundary_survives_evidence_checks",
             "layouts_rejected_by_reason": {"unused_one_sided_stretch_is_a_boundary_face": 41, "unused_supported_stretch_outside_boundary": 7}}],
            "ignored_lines": [{"reason": "too_few_distinct_frames"}, {"reason": "too_few_distinct_frames"}], "boundary_search": {"closed_cycles": 8, "nodes": 58}})
        self.assertEqual(inc["label"], "no_closed_boundary_survives_evidence_checks")
        self.assertEqual(inc["layout_reasons"]["unused_one_sided_stretch_is_a_boundary_face"], 41)
        self.assertEqual(inc["ignored_lines"], {"too_few_distinct_frames": 2})
        self.assertEqual(inc["closed_cycles"], 8)
        none = dg.classify_failure(None, "insufficient", ["no line passed the multi-view support test"])
        self.assertEqual(none["label"], "walls_unavailable")
        many = dg.classify_failure({"status": "incomplete", "incomplete_boundaries": [{"reason": "too_many_wall_candidates_for_bounded_search", "nodes": 99}]})
        self.assertEqual(many["label"], "too_many_wall_candidates_for_bounded_search")

    def test_real_assembler_reasons_on_fixtures(self):
        import boundary_assembly as ba
        full = ba.assemble_boundary(box_walls())
        self.assertEqual(dg.classify_failure(full)["label"], "closed")
        three = ba.assemble_boundary(box_walls(drop=3))
        lab = dg.classify_failure(three)
        self.assertNotEqual(lab["label"], "closed")
        self.assertTrue(lab["top_reason"])
        two = ba.assemble_boundary({**box_walls(), "lines": box_walls()["lines"][:2]})
        self.assertEqual(dg.classify_failure(two)["label"], "fewer_than_three_supported_wall_lines")

    def test_top_and_vertical_flag(self):
        cands = [{"id": 0, "rank": 0, "score": 1.5, "viable": True, "cap_pair_found": False},
                 {"id": 1, "rank": 1, "score": 1.5, "viable": True, "cap_pair_found": True, "cap_support": 0.4},
                 {"id": 2, "rank": 2, "score": 1.0, "viable": True, "cap_pair_found": True, "cap_support": 0.9},
                 {"id": 3, "rank": 3, "score": 2.0, "viable": False, "cap_pair_found": True, "cap_support": 1.0}]
        self.assertEqual(dg.top_ranked_ids(cands), [0, 1])
        self.assertEqual(dg.vertical_flag_id(cands), 2)
        self.assertEqual(dg.top_ranked_ids([{"id": 0, "score": 1, "viable": False}]), [])
        self.assertIsNone(dg.vertical_flag_id([{"id": 0, "rank": 0, "score": 1, "viable": True, "cap_pair_found": False}]))


class LayeredPairTests(unittest.TestCase):
    def test_same_side_copy_is_layered_and_opposite_wall_is_not(self):
        walls = box_walls(extra=[wall_line(4, [1, 0], 1.12, [[-0.8, 0.8]])])      # copy of wall 0, 0.12 behind it, 4 % of extent
        res = dg.layered_pairs(walls, walls["observations"]["camera_xy"])
        layered = [p for p in res["pairs"] if p["layered"]]
        self.assertEqual({(p["line_a"], p["line_b"]) for p in layered}, {(0, 4)})
        p = layered[0]
        self.assertAlmostEqual(p["separation"], 0.12, places=6)
        self.assertAlmostEqual(p["separation_over_extent"], 0.04, places=6)
        self.assertTrue(p["drift_scale"] and p["same_side"])
        # walls 0 and 2 are opposite faces of the room: parallel, overlapping, 2.0 apart, but the cameras lie between them
        opp = [q for q in res["pairs"] if {q["line_a"], q["line_b"]} == {0, 2}]
        self.assertTrue(opp and opp[0]["same_side"] is False and not opp[0]["layered"])
        summ = dg.layered_summary(res)
        self.assertEqual(summ["n_layered"], 1)
        self.assertEqual(summ["n_layered_drift_scale"], 1)
        self.assertAlmostEqual(summ["max_drift_separation_over_extent"], 0.04, places=6)

    def test_within_noise_and_object_scale(self):
        close = box_walls(extra=[wall_line(4, [1, 0], 1.004, [[-0.8, 0.8]])])      # 0.004 < 3 * 0.005: coincident within noise
        self.assertEqual(dg.layered_summary(dg.layered_pairs(close, close["observations"]["camera_xy"]))["n_layered"], 0)
        far = box_walls(extra=[wall_line(4, [1, 0], 0.4, [[-0.5, 0.5]])])           # 0.6 = 20 % of extent: an object, not a drift
        s = dg.layered_summary(dg.layered_pairs(far, far["observations"]["camera_xy"]))
        self.assertEqual((s["n_layered"], s["n_layered_drift_scale"], s["n_layered_object_scale"]), (1, 0, 1))

    def test_unrelated_directions_and_no_overlap_ignored(self):
        w = box_walls(extra=[wall_line(4, [1, 0], 1.1, [[5.0, 6.0]])])               # parallel but far along the wall: no lateral overlap
        self.assertEqual(dg.layered_summary(dg.layered_pairs(w, w["observations"]["camera_xy"]))["n_layered"], 0)
        w2 = box_walls(extra=[wall_line(4, [1, 1], 1.5, [[-0.5, 0.5]])])             # 45 degrees: not near parallel
        self.assertEqual(dg.layered_summary(dg.layered_pairs(w2, w2["observations"]["camera_xy"]))["n_layered"], 0)

    def test_line_stats_numbers(self):
        s = dg.line_stats(box_walls())
        self.assertEqual(s["n_lines"], 4)
        self.assertAlmostEqual(s["noise_over_extent"], 0.005 / 3.0)
        self.assertAlmostEqual(s["lines"][0]["span_total_over_extent"], 1.8 / 3.0)
        self.assertAlmostEqual(s["median_offset_sd_over_noise"], 0.004 / 0.005)
        self.assertEqual(dg.line_stats({"status": "insufficient", "lines": []})["n_lines"], 0)


class LayoutClassificationTests(unittest.TestCase):
    def test_make_scene_is_box_with_furniture_and_openings(self):
        import simulator
        info = dg.classify_layout(simulator.make_scene(17))
        self.assertEqual(info["class"], "box_with_furniture_and_openings")
        self.assertEqual((info["furniture_count"], info["opening_count"], info["doorway_count"], info["window_count"]), (3, 2, 1, 1))
        self.assertGreater(info["furniture_footprint_fraction"], 0.0)
        self.assertGreaterEqual(info["aspect"], 1.0)
        bare = dict(simulator.make_scene(17), furniture=[], openings=[])
        self.assertEqual(dg.classify_layout(bare)["class"], "plain_box")

    def test_house_layouts_rectangle_and_l_shape(self):
        import house_layouts
        classes = {dg.classify_layout(house_layouts.generate_house("single_room", i, 20261003))["class"] for i in range(10)}
        self.assertEqual(classes, {"l_shape"})              # every generated single room of that suite is an L
        lshape = {"rooms": [{"polygon_xy_m": [[0, 0], [4, 0], [4, 2], [2, 2], [2, 5], [0, 5]]}]}
        self.assertEqual(dg.classify_layout(lshape)["class"], "l_shape")
        self.assertEqual(dg.classify_layout({"rooms": [{"polygon_xy_m": [[0, 0], [3, 0], [3, 2], [0, 2]]}]})["class"], "rectangle")
        self.assertEqual(dg.classify_layout({"foo": 1})["class"], "unknown")


class OracleHelperTests(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(dg.spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])[0], 1.0)
        self.assertAlmostEqual(dg.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])[0], -1.0)
        self.assertIsNone(dg.spearman([1, 1, 1, 1], [1, 2, 3, 4])[0])
        self.assertIsNone(dg.spearman([1, 2], [1, 2])[0])

    def test_similarity_recovers_scale_and_residual(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(30, 3))
        th = 0.7
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
        sim = dg.similarity_from_centres(src, dst)
        self.assertAlmostEqual(sim["s"], 2.5, places=6)
        self.assertLess(sim["rms"], 1e-9)

    def test_true_alignment_and_offset_error_with_known_walls(self):
        import simulator
        poses = [{"position": [0.5 * np.cos(a), 0.4 * np.sin(a), 1.4], "yaw": a, "pitch": 0.1 * np.sin(2 * a), "roll": 0.0}
                 for a in np.linspace(0, 6.0, 30)]
        # a perfect "DA3" prediction in the true frame: world->camera from the true pose, scale 1
        ext = []
        for p in poses:
            right, up, forward = simulator.camera_basis(p)
            R = np.column_stack((right, -up, forward)).T
            ext.append(np.column_stack((R, -R @ np.asarray(p["position"]))))
        ext = np.asarray(ext)
        al = dg.true_alignment(ext, poses)
        self.assertAlmostEqual(al["s"], 1.0, places=6)
        self.assertLess(al["centre_resid_rms_m"], 1e-6)
        self.assertLess(al["rot_scatter_deg_median"], 1e-3)
        self.assertLess(dg.axis_angle_to_true_up([0, 0, 1], al), 1e-4)
        self.assertGreater(dg.axis_angle_to_true_up([1, 0, 0], al), 89.0)
        scene = {"width": 6.0, "length": 5.0}
        # lines (identity frame): east wall at x=3.0 (error 0), a drifted copy at x=2.9 (error -10 cm), north wall y=2.55 (+5 cm)
        compact = {"frame": {"origin_world": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0], "extent": 8.0}, "lines": [
            {"id": 0, "normal_xy": [1, 0], "offset": 3.0, "direction_xy": [0, 1], "observed_spans": [[-2.0, 2.0]], "distinct_frames": 10},
            {"id": 1, "normal_xy": [1, 0], "offset": 2.9, "direction_xy": [0, 1], "observed_spans": [[-1.0, 1.0]], "distinct_frames": 10},
            {"id": 2, "normal_xy": [0, 1], "offset": 2.55, "direction_xy": [-1, 0], "observed_spans": [[-2.5, 2.5]], "distinct_frames": 10},
            {"id": 3, "normal_xy": [1, 0], "offset": 0.0, "direction_xy": [0, 1], "observed_spans": [[-1.0, 1.0]], "distinct_frames": 10}]}
        res = dg.line_offset_errors(compact, al, scene)
        err = {r["id"]: r for r in res["lines"]}
        self.assertEqual(err[0]["wall"], 1)
        self.assertAlmostEqual(err[0]["offset_error_cm"], 0.0, places=3)
        self.assertAlmostEqual(err[1]["offset_error_cm"], -10.0, places=3)
        self.assertEqual(err[2]["wall"], 0)
        self.assertAlmostEqual(err[2]["offset_error_cm"], 5.0, places=3)
        self.assertIsNone(err[3]["wall"])                       # an interior line is not matched to a wall
        self.assertEqual(res["walls_with_multiple_lines"], 1)
        self.assertEqual(res["n_unmatched"], 1)


class OverwriteAndFreezeTests(unittest.TestCase):
    def test_write_new_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "x.json"
            sha = dg.write_new_json(p, {"a": np.float32(1.5), "b": np.array([1, 2])})
            self.assertEqual(sha, dg.sha256_file(p))
            self.assertEqual(json.loads(p.read_text())["b"], [1, 2])
            with self.assertRaises(FileExistsError):
                dg.write_new_json(p, {"a": 2})
            self.assertEqual(json.loads(p.read_text())["a"], 1.5)

    def test_stage_diagnose_refuses_existing_folder(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                dg.stage_diagnose(d)

    def test_oracle_requires_freeze_and_detects_change(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            with self.assertRaises(RuntimeError):
                dg.stage_oracle(out)
            dg.write_new_json(out / "truth_free" / "b" / "r.json", {"run_id": "r"})
            idx = dg.stage_freeze(out)
            self.assertIn("truth_free/b/r.json", idx["files"])
            with self.assertRaises(FileExistsError):
                dg.stage_freeze(out)
            (out / "truth_free" / "b" / "r.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                dg.stage_oracle(out)


class NoTruthInInferenceTests(unittest.TestCase):
    def test_truth_free_functions_import_and_name_no_truth(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        # module-level imports
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for n in names:
                    self.assertNotIn(n.split(".")[0], dg.TRUTH_MODULES, f"module-level import of truth module {n}")
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in dg.TRUTH_FREE_FUNCTIONS:
            self.assertIn(name, funcs, name)
            node = funcs[name]
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    mods = [a.name for a in sub.names] + ([sub.module] if isinstance(sub, ast.ImportFrom) and sub.module else [])
                    for m in mods:
                        self.assertNotIn(m.split(".")[0], dg.TRUTH_MODULES, f"{name} imports truth module {m}")
                if isinstance(sub, ast.Name):
                    self.assertNotIn(sub.id.lower(), ("scene", "poses", "simulator", "oracle"), f"{name} names {sub.id}")
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    for bad in ("private_ground_truth", "scene.json", "generator_private", "path_00.json", "poses.json", "capture_qa"):
                        self.assertNotIn(bad, sub.value, f"{name} references {bad}")
        # the evaluator part exists and is NOT in the truth-free list
        for name in ("oracle_run", "oracle_control", "true_alignment", "line_offset_errors", "classify_layout", "stage_oracle"):
            self.assertIn(name, funcs)
            self.assertNotIn(name, dg.TRUTH_FREE_FUNCTIONS)

    def test_probe_uses_only_the_documented_confidence_value(self):
        self.assertEqual(dg.PROBE_CONF_PERCENTILE, 50.0)


if __name__ == "__main__":
    unittest.main()
