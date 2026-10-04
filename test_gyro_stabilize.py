import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from gyro_stabilize import stabilize, transport_roll_angles


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


class GyroStabilizeTests(unittest.TestCase):
    def test_pure_roll_is_unwrapped(self):
        angles = transport_roll_angles(np.repeat(rz(np.deg2rad(3))[None], 119, axis=0))
        self.assertAlmostEqual(float(np.rad2deg(angles[-1])), 357.0, places=6)

    def test_minimal_forward_transport_has_zero_axial_twist(self):
        a = np.deg2rad(8.0)
        c, s = np.cos(a), np.sin(a)
        ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        angles = transport_roll_angles(np.repeat(ry[None], 12, axis=0))
        np.testing.assert_allclose(angles, 0, atol=1e-9)

    def test_crop_contains_no_padding_even_at_45_degrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            for i in range(2):
                cv2.imwrite(str(images / f"{i:06d}.png"), np.full((360, 480, 3), 255, np.uint8))
            gyro = root / "gyro.json"
            gyro.write_text(json.dumps({"schema": "virtual_gyro_v1", "frame_count": 2,
                                        "delta_camera_to_previous": [rz(np.deg2rad(45)).tolist()]}))
            result = stabilize(images, gyro, root / "output", crop_no_padding=True)
            self.assertTrue(result["truth_free"])
            corrected = cv2.imread(str(root / "output" / "000001.png"))
            self.assertGreater(int(corrected.min()), 200)


if __name__ == "__main__":
    unittest.main()
