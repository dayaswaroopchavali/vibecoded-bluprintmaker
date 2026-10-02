"""Non-rendering checks for independently seeded Blender capture paths."""
import unittest

import blender_capture
import simulator


class BlenderCaptureTests(unittest.TestCase):
    def test_path_seeds_and_camera_loops_are_repeatable_and_independent(self):
        scene = simulator.make_scene(17)
        a = blender_capture.camera_poses(scene, 16, 2026)
        b = blender_capture.camera_poses(scene, 16, 2026)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(set(blender_capture.path_seeds(scene, 2026))), 3)
        self.assertNotEqual(a[0], a[1])
        self.assertNotEqual(a[0], blender_capture.camera_poses(scene, 16, 2027)[0])

    def test_camera_positions_remain_clear_of_room_and_furniture(self):
        scene = simulator.make_scene(17)
        for path in blender_capture.camera_poses(scene, 16, 2026):
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

    def test_blender_poses_match_simulator_and_cover_room(self):
        scene = simulator.make_scene(17)
        poses = blender_capture.camera_poses(scene, 120, 2026)
        self.assertEqual(poses, simulator.make_paths(scene, 120, 2026))
        self.assertEqual(blender_capture.path_seeds(scene, 2026), simulator.path_seeds(scene, 2026))
        self.assertTrue(simulator.union_coverage([simulator.coverage_diagnostics(scene, p) for p in poses])["complete"])

    def test_write_coverage_goes_to_private_truth_only(self):
        import json, tempfile
        from pathlib import Path
        scene = simulator.make_scene(17)
        with tempfile.TemporaryDirectory() as tmp:
            report = blender_capture.write_coverage(scene, Path(tmp), 120, 78., 2026)
            self.assertTrue(report["union"]["complete"])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["private_ground_truth"])
            data = json.loads((Path(tmp) / "private_ground_truth" / "coverage_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["paths"]), 3)


if __name__ == "__main__":
    unittest.main()
