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

    def test_texture_params_depend_only_on_texture_seed(self):
        import json
        scene = simulator.make_scene(17)
        a = blender_capture.texture_params(5, scene)
        self.assertEqual(a, blender_capture.texture_params(5, scene))
        self.assertNotEqual(a, blender_capture.texture_params(6, scene))
        moved = json.loads(json.dumps(scene))
        moved["seed"] = 999; moved["width"] *= 1.7; moved["length"] *= .6; moved["height"] += 1
        for f in moved["furniture"]: f["center"] = [0., 0.]; f["size"] = [1., 1., 1.]
        for o in moved["openings"]: o["center"] += .3
        self.assertEqual(a, blender_capture.texture_params(5, moved))   # no scene dimension/position/seed enters the draw
        blender_capture.camera_poses(scene, 8, 1); blender_capture.camera_poses(scene, 8, 2)  # capture seed never reaches it
        self.assertEqual(a, blender_capture.texture_params(5, scene))
        self.assertEqual(a["coverage"]["wall_area_with_structured_pattern_fraction"], 0.0)

    def test_poses_json_loading_hash_and_validation(self):
        import json, tempfile
        from pathlib import Path
        scene = simulator.make_scene(17); poses = blender_capture.camera_poses(scene, 6, 2026)[0]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "path_00.json"; p.write_text(json.dumps({"path_seed": 1, "poses": poses}), encoding="utf-8")
            self.assertEqual(blender_capture.load_poses_json(p)["poses"], poses)
            self.assertEqual(blender_capture.poses_sha256(poses), blender_capture.poses_sha256(json.loads(json.dumps(poses))))
            p.write_text(json.dumps({"poses": [{"position": [0, 0, 1]}]}), encoding="utf-8")
            with self.assertRaises(ValueError): blender_capture.load_poses_json(p)
            p.write_text(json.dumps({"poses": []}), encoding="utf-8")
            with self.assertRaises(ValueError): blender_capture.load_poses_json(p)


if __name__ == "__main__":
    unittest.main()
