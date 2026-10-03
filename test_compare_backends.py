"""Unit tests for compare_backends scoring/provenance logic on synthetic data.

DA3 and COLMAP runs are CLI-only (`python compare_backends.py run ...`) and are deliberately NOT exercised here.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_camera_trajectory as A
import compare_backends as C


def make_gt_poses(n=40, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for k in range(n):
        a = 2 * np.pi * k / n
        poses.append({"position": [1.5 * np.cos(a), 1.2 * np.sin(a), 1.5 + 0.3 * np.sin(3 * a) + 0.01 * rng.normal()],
                      "yaw": float(a + 1.0), "pitch": float(0.2 * np.sin(2 * a)), "roll": float(0.3 * np.cos(a))})
    return poses


def w2c_from_gt(poses, scale=2.0, world_rot=None, shift=(0.3, -0.2, 0.5)):
    """Perfect prediction in an arbitrary similarity frame: X' = s R0 X + t0."""
    R0 = np.eye(3) if world_rot is None else world_rot
    out = []
    for p in poses:
        c2w = A.gt_opencv_c2w(p)
        c = scale * R0 @ np.asarray(p["position"]) + np.asarray(shift)
        c2w_new = R0 @ c2w
        R = c2w_new.T
        out.append(np.concatenate([R, (-R @ c)[:, None]], axis=1))
    return np.stack(out)


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


class FrameSelection(unittest.TestCase):
    def test_default_budget_is_every_third_and_recorded(self):
        b = C.select_frames(120)
        self.assertEqual(b["selected_indices"], list(range(0, 120, 3)))
        self.assertEqual(b["frames_selected"], 40)
        self.assertEqual(b["frame_step"], 3)
        self.assertTrue(b["is_default_budget"])
        self.assertAlmostEqual(b["selected_fraction_of_capture"], 1 / 3)
        self.assertEqual(b, C.select_frames(120))  # deterministic

    def test_dense_budget_labelled_different(self):
        b = C.select_frames(120, 1)
        self.assertEqual(b["frames_selected"], 120)
        self.assertFalse(b["is_default_budget"])
        self.assertIn("NON_DEFAULT_BUDGET", b["budget_label"])

    def test_invalid(self):
        for args in ((0, 3), (10, 0), (10, 3, 3)):
            with self.assertRaises(ValueError):
                C.select_frames(*args)

    def test_scratch_frames_keep_original_names_and_refuse_private_dir(self):
        with tempfile.TemporaryDirectory() as t:
            rgb = Path(t) / "rgb"
            rgb.mkdir()
            for i in range(9):
                (rgb / f"{i:06d}.png").write_bytes(bytes([i]) * 10)
            rows = C.write_scratch_frames(rgb, [0, 3, 6], Path(t) / "scratch")
            self.assertEqual([r["file"] for r in rows], ["000000.png", "000003.png", "000006.png"])
            self.assertEqual(sorted(p.name for p in (Path(t) / "scratch").iterdir()), [r["file"] for r in rows])
            with self.assertRaises(ValueError):
                C.write_scratch_frames(rgb, [0], Path(t) / "private_ground_truth" / "x")
            with self.assertRaises(FileNotFoundError):
                C.write_scratch_frames(rgb, [99], Path(t) / "s2")

    def test_remap_source_indices(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "p.npz"
            np.savez(raw, extrinsics=np.zeros((3, 3, 4)), source_indices=np.arange(3))
            ordered = C.remap_source_indices(raw, [6, 0, 3], Path(t) / "q.npz")
            self.assertEqual(ordered, [0, 3, 6])
            with np.load(Path(t) / "q.npz") as d:
                self.assertEqual(d["source_indices"].tolist(), [0, 3, 6])
            with np.load(raw) as d:
                self.assertEqual(d["source_indices"].tolist(), [0, 1, 2])  # raw untouched
            np.savez(raw.with_name("bad.npz"), extrinsics=np.zeros((3, 3, 4)), source_indices=np.array([0, 5, 9]))
            with self.assertRaises(ValueError):
                C.remap_source_indices(raw.with_name("bad.npz"), [0, 3, 6], Path(t) / "r.npz")


class PoseScoring(unittest.TestCase):
    def setUp(self):
        self.gt = make_gt_poses(40)
        self.sel = list(range(0, 40, 1))

    def test_perfect_prediction_in_arbitrary_frame_scores_near_zero(self):
        w2c = w2c_from_gt(self.gt, 2.0, rot_z(37))
        s = C.score_pose_set(w2c, self.sel, self.gt)
        self.assertEqual(s["status"], "scored")
        self.assertLess(s["centre_aligned_umeyama"]["rotation_error_after_centre_alignment"]["geodesic_max_deg"], 0.05)
        self.assertLess(s["rotation_aligned_kabsch"]["geodesic_max_deg"], 0.05)
        self.assertLess(s["centre_aligned_umeyama"]["centre_rmse_fraction_of_full_gt_extent"], 1e-6)
        self.assertAlmostEqual(s["centre_aligned_umeyama"]["uniform_scale"], 0.5, places=4)

    def test_wrong_rotations_are_scored_not_hidden(self):
        w2c = w2c_from_gt(self.gt)
        rng = np.random.default_rng(1)
        for k in range(len(w2c)):                      # right centres, random camera rotations
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            q *= np.sign(np.linalg.det(q))
            c = -w2c[k, :, :3].T @ w2c[k, :, 3]
            w2c[k, :, :3] = q
            w2c[k, :, 3] = -q @ c
        s = C.score_pose_set(w2c, self.sel, self.gt)
        self.assertEqual(s["status"], "scored")
        self.assertGreater(s["centre_aligned_umeyama"]["rotation_error_after_centre_alignment"]["geodesic_median_deg"], 20)
        self.assertLess(s["centre_aligned_umeyama"]["centre_rmse_fraction_of_full_gt_extent"], 1e-6)

    def test_too_few_frames_not_comparable(self):
        w2c = w2c_from_gt(self.gt)[:5]
        s = C.score_pose_set(w2c, list(range(5)), self.gt)
        self.assertEqual(s["status"], "not_comparable")
        self.assertIn("only 5", s["reason"])

    def test_collinear_centres_not_comparable(self):
        poses = [{"position": [0.1 * k, 0.0, 1.5], "yaw": 0.0, "pitch": 0.0} for k in range(12)]
        s = C.score_pose_set(w2c_from_gt(poses), list(range(12)), poses)
        self.assertEqual(s["status"], "not_comparable")
        self.assertIn("degenerate", s["reason"])

    def test_duplicate_indices_not_comparable(self):
        s = C.score_pose_set(w2c_from_gt(self.gt)[:10], [0] * 10, self.gt)
        self.assertEqual(s["status"], "not_comparable")

    def test_backend_fractions_known_registered_subset(self):
        w2c = w2c_from_gt(self.gt)
        selected = list(range(0, 40, 2))                       # 20 selected out of 40 captured
        registered = [i for i in selected if i % 4 == 0]         # 10 registered
        m = {"model_id": "0", "indices": registered, "w2c": w2c[registered]}
        out = C.score_backend([m], self.gt, selected, 40)
        self.assertEqual(out["frames_selected"], 20)
        self.assertEqual(out["frames_registered_all_models"], 10)
        self.assertAlmostEqual(out["registered_fraction_of_selected"], 0.5)
        self.assertAlmostEqual(out["retained_capture_fraction_of_full_capture"], 0.25)
        self.assertEqual(out["headline"]["status"], "scored")
        self.assertEqual(out["headline"]["frames_scored"], 10)

    def test_multiple_models_all_reported_headline_is_largest(self):
        w2c = w2c_from_gt(self.gt)
        a, b = list(range(0, 24)), list(range(24, 32))
        out = C.score_backend([{"model_id": "0", "indices": b, "w2c": w2c[b]}, {"model_id": "1", "indices": a, "w2c": w2c[a]}],
                              self.gt, list(range(40)), 40)
        self.assertEqual(out["model_count"], 2)
        self.assertEqual(out["headline_model_id"], "1")
        self.assertEqual(out["frames_registered_all_models"], 32)       # fractions count every model
        self.assertEqual({m["model_id"] for m in out["models"]}, {"0", "1"})
        self.assertAlmostEqual(out["headline_frames_fraction_of_selected"], 24 / 40)

    def test_no_models(self):
        out = C.score_backend([], self.gt, list(range(40)), 40)
        self.assertEqual(out["registered_fraction_of_selected"], 0.0)
        self.assertEqual(out["headline"]["status"], "not_comparable")

    def test_head_to_head_common_only(self):
        w2c = w2c_from_gt(self.gt)
        ia, ib = list(range(0, 30)), list(range(15, 40))
        r = C.head_to_head({"indices": ia, "w2c": w2c[ia]}, {"indices": ib, "w2c": w2c[ib]}, self.gt)
        self.assertEqual(r["status"], "scored")
        self.assertEqual(r["common_source_indices"], list(range(15, 30)))
        self.assertEqual(r["a"]["frames_scored"], 15)

    def test_head_to_head_too_few_common_is_not_comparable(self):
        w2c = w2c_from_gt(self.gt)
        ia, ib = list(range(0, 20)), list(range(15, 40))        # 5 in common
        r = C.head_to_head({"indices": ia, "w2c": w2c[ia]}, {"indices": ib, "w2c": w2c[ib]}, self.gt)
        self.assertEqual(r["status"], "not_comparable")
        self.assertEqual(r["common_count"], 5)
        self.assertNotIn("a", r)


FAKE_ANALYZER = """I20261003 14:00:00.000000 100 model_analyzer.cc:90] Rigs: 1
I20261003 14:00:00.000000 100 model_analyzer.cc:91] Cameras: 1
I20261003 14:00:00.000000 100 model_analyzer.cc:92] Registered images: 12
I20261003 14:00:00.000000 100 model_analyzer.cc:93] Points: 345
I20261003 14:00:00.000000 100 model_analyzer.cc:94] Observations: 1500
I20261003 14:00:00.000000 100 model_analyzer.cc:95] Mean track length: 4.347826
I20261003 14:00:00.000000 100 model_analyzer.cc:96] Mean observations per image: 125.000000
I20261003 14:00:00.000000 100 model_analyzer.cc:97] Mean reprojection error: 0.612345px
"""


class ColmapParsing(unittest.TestCase):
    def test_model_analyzer_text(self):
        d = C.parse_model_analyzer(FAKE_ANALYZER)
        self.assertEqual(d["registered_images"], 12)
        self.assertEqual(d["points"], 345)
        self.assertAlmostEqual(d["mean_reprojection_error"], 0.612345)
        self.assertAlmostEqual(d["mean_track_length"], 4.347826)

    def test_text_model(self):
        with tempfile.TemporaryDirectory() as t:
            m = Path(t)
            (m / "cameras.txt").write_text("# c\n1 SIMPLE_PINHOLE 480 360 576 240 180\n")
            (m / "images.txt").write_text(
                "# images\n1 1 0 0 0 0.5 0 2 1 000003.png\n10.0 20.0 1 30.0 40.0 -1\n"
                "2 1 0 0 0 0 0 0 1 000006.png\n\n")                 # second image has an EMPTY points line
            (m / "points3D.txt").write_text("# p\n1 0 0 1 10 10 10 0.5 1 0 2 5\n2 0 0 2 10 10 10 1.5 1 1\n3 0 0 3 1 1 1 1.0 1 1 2 2 3 3\n")
            pm = C.parse_text_model(m)
        self.assertEqual(pm["registered_images"], 2)
        self.assertEqual([i["source_index"] for i in pm["images"]], [3, 6])
        self.assertEqual(pm["images"][0]["num_points2D"], 2)
        self.assertEqual(pm["images"][1]["num_points2D"], 0)
        np.testing.assert_allclose(pm["images"][0]["w2c"][:, 3], [0.5, 0, 2])
        self.assertEqual(pm["num_points3D"], 3)
        self.assertEqual(pm["track_length"]["max"], 3)
        self.assertEqual(pm["track_length"]["min"], 1)
        # observation-weighted: (0.5*2 + 1.5*1 + 1.0*3) / 6
        self.assertAlmostEqual(pm["mean_reprojection_error_px_observation_weighted"], (1 + 1.5 + 3) / 6)

    def test_failure_lines_find_initial_pair(self):
        text = "I.. incremental_pipeline.cc:472] => No good initial image pair found.\nE.. sfm.cc:310] Failed to create any sparse model\n"
        lines = C.colmap_failure_lines(text)
        self.assertEqual(len(lines), 2)


class SummaryAndProvenance(unittest.TestCase):
    def test_provenance_uses_full_capture_denominator(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "prov.json"
            prov = C.write_provenance(p, selected=40, registered=40, available=120, note="x")
            self.assertAlmostEqual(prov["retained_capture_fraction"], 40 / 120)
            from evaluate_floorplan import retained_capture_fraction, whole_room_claim
            r = retained_capture_fraction(json.loads(p.read_text()))
            self.assertAlmostEqual(r["value"], 1 / 3)
            self.assertIn("retained_capture_below_full", whole_room_claim(r, "scored")["blockers"])

    def test_height_ratio(self):
        self.assertAlmostEqual(C.height_ratio_oracle(2.0, 1.5, 3.0), 1.0)
        self.assertIsNone(C.height_ratio_oracle(None, 1.5, 3.0))

    def test_summary_rows_and_markdown(self):
        gt = make_gt_poses(40)
        w2c = w2c_from_gt(gt)
        da3 = C.score_backend([{"model_id": "whole_video", "indices": list(range(40)), "w2c": w2c}], gt, list(range(40)), 120)
        col = C.score_backend([], gt, list(range(40)), 120)
        doc = {"variant": "v", "frame_budget": C.select_frames(120), "da3": {"pose_score": da3, "inference": {"inference_manifest_elapsed_seconds": 5.0, "gpu_peak_memory_gib": 3.0},
                                                                              "structural": {"structural_axis_hypotheses_status": "ambiguous_axes", "hypotheses": []}},
               "colmap": {"pose_score": col, "models": [], "no_model_reason": "COLMAP could not start", "floor_plan": {"reason": "n/a"}, "runtime_seconds": 1.0},
               "head_to_head": {"status": "not_comparable", "common_count": 0, "reason": "x"}}
        rows = C.summary_rows([doc])
        self.assertEqual([r["backend"] for r in rows], ["da3", "colmap", "head_to_head_common_frames"])
        self.assertEqual(rows[1]["status"], "not_comparable")
        self.assertEqual(rows[1]["registered_fraction"], 0.0)
        md = C.render_markdown(rows)
        self.assertIn("not_comparable", md)
        self.assertIn("no stage is claimed complete", md.lower())
        with tempfile.TemporaryDirectory() as t:
            src = Path(t) / "c.json"
            src.write_text(json.dumps(C._clean(doc)))
            out = Path(t) / "out"
            C.summarize([src], out)
            self.assertTrue((out / "summary.md").exists())
            with self.assertRaises(FileExistsError):
                C.summarize([src], out)

    def test_compare_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileExistsError):
                C.compare_variant(Path(t), Path(t))


if __name__ == "__main__":
    unittest.main()
