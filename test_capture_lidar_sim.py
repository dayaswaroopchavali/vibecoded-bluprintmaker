import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import simulator
from capture_lidar_sim import capture


class SimulatedLidarCaptureTests(unittest.TestCase):
    def test_public_depth_is_camera_z_and_has_no_surface_labels(self):
        scene = simulator.make_scene(17)
        pose = {"position": [0.0, 0.0, 1.4], "yaw": 0.3, "pitch": -0.2, "roll": 1.1}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_file, pose_file = root / "scene.json", root / "poses.json"
            scene_file.write_text(json.dumps(scene))
            pose_file.write_text(json.dumps({"poses": [pose]}))
            capture(scene_file, pose_file, root / "capture", width=64, height=48,
                    depth_noise_m=0, position_noise_m=0, rotation_noise_deg=0)
            manifest = json.loads((root / "capture" / "manifest.json").read_text())
            self.assertNotIn("private_scene_sha256", manifest)
            self.assertNotIn("private_pose_sha256", manifest)
            with np.load(root / "capture" / "prediction.npz") as data:
                self.assertEqual(set(data.files), {"images", "depth", "confidence", "extrinsics",
                                                   "intrinsics", "source_indices"})
                with np.errstate(all="ignore"):
                    _, _, _, _, hitpos = simulator.render(scene, pose, 64, 48, 78,
                                                          depth_out=True, kind_out=True)
                _, _, forward = simulator.camera_basis(pose)
                expected = np.einsum("hwc,c->hw", hitpos - np.asarray(pose["position"]), forward)
                mask = data["confidence"][0] > 0
                np.testing.assert_allclose(data["depth"][0][mask], expected[mask], atol=1e-5)
                self.assertTrue(np.all(data["depth"][0][~mask] == 0))
                self.assertEqual(data["extrinsics"].shape, (1, 3, 4))
                self.assertEqual(data["intrinsics"].shape, (1, 3, 3))


if __name__ == "__main__":
    unittest.main()
