"""Focused deterministic checks for the procedural room prototype."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import simulator


class SimulatorTests(unittest.TestCase):
    def test_scene_and_paths_repeat_for_seed(self):
        a, b = simulator.make_scene(43), simulator.make_scene(43)
        self.assertEqual(a, b)
        self.assertEqual(simulator.make_paths(a, 12), simulator.make_paths(b, 12))

    def test_seed_changes_room_and_dimensions_change_path(self):
        a, b = simulator.make_scene(43), simulator.make_scene(44)
        self.assertNotEqual((a["width"], a["length"], a["height"]), (b["width"], b["length"], b["height"]))
        pa = simulator.make_paths(a, 12)
        pb = simulator.make_paths(b, 12)
        self.assertNotEqual(pa, pb)

    def test_three_distinct_paths_stay_inside_room(self):
        scene = simulator.make_scene(7)
        paths = simulator.make_paths(scene, 12, 102)
        self.assertNotEqual(paths, simulator.make_paths(scene, 12, 103))
        self.assertEqual(len(paths), 3)
        self.assertNotEqual(paths[0], paths[1])
        for path in paths:
            for pose in path:
                x, y, z = pose["position"]
                self.assertLess(abs(x), scene["width"] / 2)
                self.assertLess(abs(y), scene["length"] / 2)
                self.assertGreater(z, 0)
                self.assertLess(z, scene["height"])
                for f in scene["furniture"]:
                    fx, fy = f["center"]
                    fw, fd = f["size"][:2]
                    self.assertFalse(abs(x-fx) < fw/2+.42 and abs(y-fy) < fd/2+.42)

    def test_renderer_produces_rgb_frame_and_calibration(self):
        image, depth, intrinsics = simulator.render(simulator.make_scene(8),
            simulator.make_paths(simulator.make_scene(8), 4)[0][0], 80, 60, 78, True)
        self.assertEqual(image.shape, (60, 80, 3))
        self.assertEqual(depth.shape, (60, 80))
        self.assertTrue(np.isfinite(image).all())
        self.assertGreater(intrinsics["fx"], 0)


FRAMES = 120


def _diag(scene, path):
    return simulator.coverage_diagnostics(scene, path)


class CaptureCoverageTests(unittest.TestCase):
    def test_three_independent_seeds_cover_every_surface_without_collisions(self):
        for seed in (1, 7, 17, 23):
            scene = simulator.make_scene(seed)
            seeds = simulator.path_seeds(scene, 2026 + seed)
            self.assertEqual(len(set(seeds)), 3)
            paths = simulator.make_paths(scene, FRAMES, 2026 + seed)
            self.assertEqual([len(p) for p in paths], [FRAMES] * 3)
            self.assertNotEqual(paths[0], paths[1]); self.assertNotEqual(paths[1], paths[2])
            diags = [_diag(scene, p) for p in paths]
            for d in diags:
                self.assertTrue(d["complete"], (seed, d["observed"]))
                self.assertEqual(d["furniture_collisions"], 0)
                self.assertGreaterEqual(d["min_wall_clearance_m"], simulator.CAMERA_WALL_MARGIN - 1e-6)
            union = simulator.union_coverage(diags)
            self.assertTrue(union["complete"])
            for key in ("wall_north", "wall_east", "wall_south", "wall_west", "floor", "ceiling", "doorway", "window"):
                self.assertTrue(union["observed"][key], key)

    def test_orientation_sweeps_are_continuous_and_stay_off_the_pole_singularity(self):
        scene = simulator.make_scene(5)
        for path in simulator.make_paths(scene, FRAMES, 11):
            d = _diag(scene, path)
            self.assertGreaterEqual(d["yaw_range_deg"], 340 * 3)          # several yaw revolutions
            self.assertLessEqual(d["max_yaw_step_deg"], 16)
            self.assertLessEqual(d["pitch_max_deg"], 65); self.assertGreaterEqual(d["pitch_min_deg"], -65)
            self.assertGreater(d["pitch_max_deg"] - d["pitch_min_deg"], 110)
            self.assertGreater(d["frame_overlap_min"], .3)  # neighbouring frames share content

    def test_wall_heavy_paths_fail_coverage(self):
        for seed in (1, 7, 17):
            scene = simulator.make_scene(seed)
            base = simulator.make_paths(scene, FRAMES, 5)[0]; y0 = base[0]["yaw"]
            # Camera loops back and forth facing roughly one wall at fixed pitch.
            wall_heavy = [dict(p, yaw=y0 + .4 * math.sin(4 * math.pi * i / FRAMES), pitch=0.) for i, p in enumerate(base)]
            d = _diag(scene, wall_heavy)
            self.assertFalse(d["complete"])
            self.assertLess(sum(d["observed"][f"wall_{n}"] for n in simulator.WALL_NAMES), 4)
            self.assertFalse(simulator.union_coverage([d])["complete"])

    def test_non_default_room_dimensions_still_covered(self):
        scene = dict(simulator.make_scene(3), width=3.4, length=11.0, height=2.6)
        diags = []
        for path in simulator.make_paths(scene, FRAMES, 9):
            d = _diag(scene, path); diags.append(d)
            self.assertEqual(d["furniture_collisions"], 0)
        union = simulator.union_coverage(diags)
        self.assertTrue(union["complete"], union["observed"])  # the capture set, not each path, must see every surface

    def test_diagnostics_only_generate_keeps_rgb_manifest_free_of_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            simulator.generate(out, seed=4, frames=FRAMES, capture_seed=77, diagnostics_only=True)
            private = out / "private_ground_truth"
            report = json.loads((private / "coverage_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["paths"]), 3); self.assertEqual(len(set(report["path_seeds"])), 3)
            self.assertTrue(report["union"]["complete"])
            public = (out / "capture_manifest.json").read_text(encoding="utf-8")
            self.assertTrue(json.loads(public)["rgb_only"])
            for leaked in ("seed", "scene", "pose", "furniture"):
                self.assertNotIn(leaked, public)
            self.assertTrue((private / "seed_manifest.json").exists())
            self.assertFalse((out / "rgb").exists())


def _sphere_row(scene_seed, mode, capture_seed=3101):
    scene = simulator.make_scene(scene_seed)
    return scene, simulator.make_paths(scene, FRAMES, capture_seed, mode)


class AngularCoverageTests(unittest.TestCase):
    def test_sphere_directions_are_equal_area_and_deterministic(self):
        d = simulator.sphere_directions()
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1, atol=1e-12)
        np.testing.assert_array_equal(d, simulator.sphere_directions())
        self.assertAlmostEqual((d[:, 2] >= .5).mean(), .25, places=3)   # cap above 30 degrees is 25% of the sphere
        self.assertAlmostEqual((d[:, 2] >= 0).mean(), .5, places=3)
        self.assertAlmostEqual(d.mean(axis=0).max(), 0, places=3)

    def test_independent_audit_numbers_are_reproduced_for_the_legacy_loops(self):
        scene, paths = _sphere_row(17, "bounded")
        fractions = [simulator.angular_coverage(p, 78., 480, 360)["full_video"]["fraction"] for p in paths]
        selected = [simulator.angular_coverage(p, 78., 480, 360)["selected_inference_frames"]["fraction"] for p in paths]
        np.testing.assert_allclose(fractions, [.8623, .8378, .8614], atol=.0015)   # audit: 86.23, 83.78, 86.14 %
        np.testing.assert_allclose(selected, [.8584, .8334, .8574], atol=.0015)    # audit: 85.84, 83.34, 85.74 % (36 frames)

    def test_bounded_pitch_loops_are_rejected_as_full_sphere_captures(self):
        for scene_seed in range(1, 11):
            _scene, paths = _sphere_row(scene_seed, "bounded")
            for path in paths:
                report = simulator.angular_coverage(path)
                self.assertFalse(report["passes_full_sphere"])
                self.assertLess(report["full_video"]["fraction"], .9)
                self.assertLess(min(report["full_video"]["polar_up_fraction"], report["full_video"]["polar_down_fraction"]), .5)

    def test_spherical_sweeps_pass_on_ten_scenes_and_three_paths_each_including_the_poles(self):
        for scene_seed in range(1, 11):
            scene, paths = _sphere_row(scene_seed, "spherical")
            self.assertEqual(len(paths), 3)
            for path in paths:
                report = simulator.angular_coverage(path)
                full, picked = report["full_video"], report["selected_inference_frames"]
                self.assertTrue(report["passes_full_sphere"], (scene_seed, full, picked))
                self.assertGreaterEqual(full["fraction"], simulator.SPHERE_FULL_VIDEO_MIN)
                self.assertGreaterEqual(picked["fraction"], simulator.SPHERE_SELECTED_MIN)
                for polar in ("polar_up_fraction", "polar_down_fraction"):
                    self.assertGreaterEqual(full[polar], simulator.SPHERE_POLAR_MIN); self.assertGreaterEqual(picked[polar], simulator.SPHERE_POLAR_MIN)
                self.assertEqual(picked["frame_count"], simulator.INFERENCE_FRAME_COUNT)
                self.assertLessEqual(picked["fraction"], full["fraction"] + 1e-12)   # a subset of the frames can only see less

    def test_full_video_and_selected_frames_are_reported_independently(self):
        _scene, paths = _sphere_row(3, "spherical")
        path = paths[0]
        sparse = simulator.angular_coverage(path, selected=[0, 40, 80])
        self.assertEqual(sparse["selected_inference_frames"]["indices"], [0, 40, 80])
        self.assertLess(sparse["selected_inference_frames"]["fraction"], .5)
        self.assertGreater(sparse["full_video"]["fraction"], .98)
        self.assertFalse(sparse["passes_full_sphere"])                         # sparse selection is not rescued by the full video
        self.assertEqual(simulator.selected_frame_indices(120, 36), np.linspace(0, 119, 36, dtype=int).tolist())

    def test_pole_only_and_horizon_only_rings_fail_the_other_region(self):
        base = simulator.make_paths(simulator.make_scene(3), FRAMES, 5)[0]
        horizon = [dict(p, pitch=0., roll=0.) for p in base]
        up_only = [dict(p, pitch=-math.radians(64), roll=0.) for p in base]
        h, u = simulator.angular_coverage(horizon), simulator.angular_coverage(up_only)
        self.assertFalse(h["passes_full_sphere"]); self.assertLess(h["full_video"]["polar_up_fraction"], .3)
        self.assertFalse(u["passes_full_sphere"]); self.assertLess(u["full_video"]["polar_down_fraction"], .01)

    def test_real_vertical_fov_is_used(self):
        base = simulator.make_paths(simulator.make_scene(3), FRAMES, 5)[0]
        wide = simulator.angular_coverage(base, 78., 480, 360)["full_video"]["fraction"]
        square = simulator.angular_coverage(base, 78., 480, 480)["full_video"]["fraction"]
        narrow = simulator.angular_coverage(base, 60., 480, 360)["full_video"]["fraction"]
        self.assertLess(wide, square); self.assertLess(narrow, wide)
        self.assertAlmostEqual(simulator.angular_coverage(base, 78., 480, 360)["fov_vertical_deg"], 2 * math.degrees(math.atan(.75 * math.tan(math.radians(39)))), places=6)

    def test_orientation_coverage_is_not_visibility(self):
        scene, paths = _sphere_row(3, "spherical")
        other = simulator.make_scene(11)
        report = simulator.angular_coverage(paths[0])
        self.assertTrue(report["orientation_only"] and report["ignores_occlusion"] and report["ignores_translation"])
        self.assertIn("not visibility", report["note"])
        self.assertNotIn("scene", simulator.angular_coverage.__code__.co_varnames[:simulator.angular_coverage.__code__.co_argcount])
        # The same poses see different surfaces in different rooms, but their orientation coverage cannot change.
        hits_a = simulator.coverage_diagnostics(scene, paths[0])["pixel_fraction"]
        hits_b = simulator.coverage_diagnostics(other, paths[0])["pixel_fraction"]
        self.assertNotEqual(hits_a, hits_b)
        self.assertEqual(simulator.angular_coverage(paths[0]), report)
    def test_capture_qa_reports_ten_scenes_three_paths_with_no_collisions_and_no_reconstruction_claim(self):
        qa = simulator.capture_qa(range(1, 11), 3101, FRAMES)
        self.assertEqual(len(qa["rows"]), 10 * 3 * 2)
        self.assertEqual(qa["summary"]["spherical"]["paths"], 30)
        self.assertEqual(qa["summary"]["spherical"]["collision_or_wall_violations"], 0)
        self.assertEqual(qa["summary"]["bounded"]["passes_full_sphere"], 0)
        self.assertIn("none", qa["claims"]); self.assertIn("nothing here says any room was reconstructed", qa["claims"])
        self.assertEqual(set(qa["summary"]["spherical"]["frame_budget_tradeoff"]), {"24", "36", "48", "72", "120"})


class OrientationTests(unittest.TestCase):
    """Roll, tilt and inversion are first-class; smoothness is judged on SO(3), never on raw Euler angles."""

    def test_basis_is_right_handed_and_roll_rotates_about_the_optical_axis(self):
        for pose in ({"yaw": 0., "pitch": 0., "roll": 0.}, {"yaw": 2.1, "pitch": -.7, "roll": 1.3}, {"yaw": -3., "pitch": 2.0, "roll": 4.0}):
            right, up, forward = simulator.camera_basis(pose)
            np.testing.assert_allclose(np.cross(right, forward), up, atol=1e-12)        # (right, forward, up) is right-handed
            R = simulator.rotation_matrix(pose)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)
        np.testing.assert_allclose(simulator.camera_basis({"yaw": 0., "pitch": 0., "roll": 0.}), [[1, 0, 0], [0, 0, 1], [0, 1, 0]], atol=1e-12)
        base = simulator.camera_basis({"yaw": .4, "pitch": .2, "roll": 0.})
        rolled = simulator.camera_basis({"yaw": .4, "pitch": .2, "roll": .5})
        np.testing.assert_allclose(rolled[2], base[2], atol=1e-12)                       # forward unchanged
        np.testing.assert_allclose(rolled[0], math.cos(.5) * base[0] - math.sin(.5) * base[1], atol=1e-12)
        self.assertAlmostEqual(float(simulator.camera_basis({"yaw": 0., "pitch": 0., "roll": math.pi})[1][2]), -1.0, places=12)  # upside-down

    def test_renderer_and_blender_use_the_same_basis_including_roll(self):
        import blender_capture
        scene = simulator.make_scene(8)
        pose = dict(simulator.make_paths(scene, 24, 5)[0][3], roll=2.4)
        upright = simulator.render(scene, dict(pose, roll=0.0), 40, 30, 78.)
        rolled = simulator.render(scene, pose, 40, 30, 78.)
        self.assertFalse(np.array_equal(upright, rolled))
        # 180 degree roll is the same view rotated by half a turn.
        flipped = simulator.render(scene, dict(pose, roll=math.pi), 40, 30, 78.)
        np.testing.assert_array_equal(flipped, simulator.render(scene, dict(pose, roll=0.0), 40, 30, 78.)[::-1, ::-1])
        self.assertIs(blender_capture.camera_poses.__module__, "blender_capture")
        src = Path(blender_capture.__file__).read_text(encoding="utf-8")
        self.assertIn("simulator.camera_basis(pose)", src)                              # Blender does not re-derive the basis

    def test_paths_roll_fully_and_spend_a_large_share_of_time_upside_down(self):
        for scene_seed in (1, 9, 17):
            scene = simulator.make_scene(scene_seed)
            for path in simulator.make_paths(scene, FRAMES, 3101):
                motion = simulator.camera_motion_summary(scene, path)
                self.assertGreaterEqual(motion["roll_range_deg"], 330)
                self.assertGreater(motion["inverted_fraction"], .25); self.assertLess(motion["inverted_fraction"], .75)

    def test_relative_so3_steps_are_smooth_on_the_loop_and_at_inference_spacing(self):
        for scene_seed in range(1, 11):
            scene = simulator.make_scene(scene_seed)
            for path in simulator.make_paths(scene, FRAMES, 3101):
                step = simulator.relative_rotation_angles_deg(path)          # closed loop: includes the last->first step
                self.assertLessEqual(step.max(), simulator.MAX_RELATIVE_ROTATION_DEG)
                chosen = simulator.relative_rotation_angles_deg([path[i] for i in simulator.selected_frame_indices(FRAMES)])
                self.assertLessEqual(chosen.max(), 90.0)                       # reported tradeoff: coarse sampling turns more per frame

    def test_wrapped_euler_angles_do_not_look_like_jumps_to_the_so3_metric(self):
        scene = simulator.make_scene(2)
        path = simulator.make_paths(scene, FRAMES, 7)[1]
        wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
        wrapped = [dict(p, yaw=wrap(p["yaw"]), roll=wrap(p["roll"])) for p in path]
        raw_jump = max(abs(wrapped[i + 1]["yaw"] - wrapped[i]["yaw"]) for i in range(len(wrapped) - 1))
        self.assertGreater(math.degrees(raw_jump), 300)                         # raw differences jump across +-180
        np.testing.assert_allclose(simulator.relative_rotation_angles_deg(wrapped), simulator.relative_rotation_angles_deg(path), atol=1e-6)
        self.assertLessEqual(simulator.relative_rotation_angles_deg(wrapped).max(), simulator.MAX_RELATIVE_ROTATION_DEG)

    def test_an_abrupt_orientation_change_is_caught_by_the_smoothness_metric(self):
        scene = simulator.make_scene(2)
        path = simulator.make_paths(scene, FRAMES, 7)[0]
        broken = path[:60] + [dict(p, yaw=p["yaw"] + 2.5, roll=p["roll"] + 1.0) for p in path[60:]]
        self.assertGreater(simulator.relative_rotation_angles_deg(broken).max(), simulator.MAX_RELATIVE_ROTATION_DEG)

    def test_angular_coverage_uses_the_full_basis_including_roll(self):
        base = simulator.make_paths(simulator.make_scene(3), FRAMES, 5)[0]
        rolled = simulator.angular_coverage(base)["full_video"]["fraction"]
        unrolled = simulator.angular_coverage([dict(p, roll=0.) for p in base])["full_video"]["fraction"]
        self.assertNotEqual(rolled, unrolled)                                    # rectangle orientation matters for coverage
        # Pure 90-degree roll swaps the 4:3 frame to 3:4: coverage of the same forward directions changes.
        a = simulator.angular_coverage([dict(p, roll=0.) for p in base[:6]])["full_video"]["fraction"]
        b = simulator.angular_coverage([dict(p, roll=math.pi / 2) for p in base[:6]])["full_video"]["fraction"]
        self.assertNotAlmostEqual(a, b, places=3)

    def test_collisions_and_overlap_hold_for_every_frame_count(self):
        for frames in (12, 36, 120, 240):
            scene = simulator.make_scene(4)
            for path in simulator.make_paths(scene, frames, 3101):
                self.assertEqual(simulator.camera_motion_summary(scene, path)["furniture_or_wall_violations"], 0)

if __name__ == "__main__":
    unittest.main()
