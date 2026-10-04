"""Known-coordinate sensor conversion checks; no reconstruction truth involved."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import import_lidar


class SensorConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'capture'
        (self.source / 'depth').mkdir(parents=True)
        (self.source / 'confidence').mkdir()
        self.config = dict(depth_scale_to_meters=.001, depth_kind='axial',
                           pose_translation_scale_to_meters=1.,
                           pose_direction='camera_to_world', camera_axes='opencv',
                           intrinsics_width=4, intrinsics_height=4, min_confidence=2,
                           depth_intrinsics_aligned=True)
        self.row = dict(timestamp=0, frame=0, x=1., y=2., z=3.,
                        qx=0., qy=0., qz=0., qw=1., fx=4., fy=4., cx=1.5, cy=1.5)

    def fixture(self, confidence=None):
        cv2.imwrite(str(self.source / 'depth/000000.png'), np.full((2, 2), 2000, np.uint16))
        cv2.imwrite(str(self.source / 'confidence/000000.png'),
                    np.full((2, 2), 2, np.uint8) if confidence is None else confidence)
        with (self.source / 'odometry.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerow(self.row)

    def convert(self):
        import_lidar.convert_capture(self.source, self.root / 'converted', self.config)
        with np.load(self.root / 'converted/prediction.npz', allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    @staticmethod
    def world_point(data):
        camera = np.linalg.solve(data['intrinsics'][0], [0., 0., 1.]) * data['depth'][0, 0, 0]
        pose = data['extrinsics'][0]
        return (camera - pose[:3, 3]) @ pose[:3, :3]

    def test_rotated_translated_camera(self):
        self.row.update(qz=2**-.5, qw=2**-.5)
        self.fixture()
        data = self.convert()
        np.testing.assert_allclose(data['intrinsics'][0], [[2,0,.5],[0,2,.5],[0,0,1]])
        np.testing.assert_allclose(self.world_point(data), [1.5, 1.5, 5.], atol=1e-5)

    def test_opengl_camera_basis(self):
        self.config['camera_axes'] = 'opengl'
        self.fixture()
        np.testing.assert_allclose(self.world_point(self.convert()), [.5, 2.5, 1.], atol=1e-5)

    def test_world_to_camera_translation_is_not_camera_center(self):
        self.config['pose_direction'] = 'world_to_camera'
        self.fixture()
        np.testing.assert_allclose(self.world_point(self.convert()), [-1.5, -2.5, -1.], atol=1e-5)

    def test_ray_range_converts_to_axial(self):
        self.config['depth_kind'] = 'ray'
        self.fixture()
        expected = 2 / np.sqrt(1 + .25**2 + .25**2)
        np.testing.assert_allclose(self.convert()['depth'], expected, atol=1e-6)

    def test_confidence_masks_geometry(self):
        self.fixture(np.array([[0, 1], [2, 2]], np.uint8))
        data = self.convert()
        np.testing.assert_allclose(data['depth'][0], [[0, 0], [2, 2]])
        self.assertTrue(np.all(data['confidence'][0, 0] == 0))

    def test_missing_unit_is_rejected(self):
        self.fixture()
        del self.config['depth_scale_to_meters']
        with self.assertRaises((ValueError, KeyError)):
            self.convert()

    def test_invalid_quaternion_is_rejected(self):
        self.row['qw'] = 0.
        self.fixture()
        with self.assertRaises(ValueError):
            self.convert()

    def test_unaligned_depth_is_rejected(self):
        self.config['depth_intrinsics_aligned'] = False
        self.fixture()
        with self.assertRaises(ValueError):
            self.convert()

    def test_sampling_preserves_whole_capture_span(self):
        self.fixture()
        with (self.source / 'odometry.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.row))
            writer.writeheader()
            for frame in range(6):
                writer.writerow({**self.row, 'frame': frame, 'timestamp': frame})
                for subdir in ['depth', 'confidence']:
                    (self.source / subdir / f'{frame:06d}.png').write_bytes(
                        (self.source / subdir / '000000.png').read_bytes())
        target = self.root / 'converted'
        import_lidar.convert_capture(self.source, target, self.config, max_frames=3)
        with np.load(target / 'prediction.npz') as data:
            np.testing.assert_array_equal(data['source_indices'], [0, 2, 5])

    def test_missing_frame_is_rejected(self):
        self.fixture()
        (self.source / 'confidence/000000.png').unlink()
        with self.assertRaises(ValueError):
            self.convert()

    def test_refuses_existing_output(self):
        self.fixture()
        (self.root / 'converted').mkdir()
        with self.assertRaises(FileExistsError):
            self.convert()


if __name__ == '__main__':
    unittest.main()
