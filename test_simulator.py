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

    def test_yaw_sweeps_full_circle_pitch_bounded_and_motion_continuous(self):
        scene = simulator.make_scene(5)
        for path in simulator.make_paths(scene, FRAMES, 11):
            d = _diag(scene, path)
            self.assertGreaterEqual(d["yaw_range_deg"], 340)
            self.assertLessEqual(d["max_yaw_step_deg"], 8)
            self.assertLessEqual(d["pitch_max_deg"], 37); self.assertGreaterEqual(d["pitch_min_deg"], -37)
            self.assertGreater(d["pitch_max_deg"] - d["pitch_min_deg"], 30)
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
        for path in simulator.make_paths(scene, FRAMES, 9):
            d = _diag(scene, path)
            self.assertTrue(d["complete"], d["observed"]); self.assertEqual(d["furniture_collisions"], 0)

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


if __name__ == "__main__":
    unittest.main()
