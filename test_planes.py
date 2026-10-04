import unittest
import numpy as np
from extract_planes import extract, points_from_prediction


class PlaneTests(unittest.TestCase):
    def test_rotated_camera_matches_homogeneous_inverse(self):
        # Independent reference: DA3 exporter uses inv(world_to_camera) on
        # homogeneous back-projected pixels. Exercise rotation and translation.
        angle = .73
        c, s = np.cos(angle), np.sin(angle)
        pose = np.eye(4)
        pose[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        pose[:3, 3] = [1.2, -3.4, 2.1]
        K = np.array([[5., 0, 1.3], [0, 6., .7], [0, 0, 1]])
        depth = np.arange(1., 13.).reshape(3, 4)
        data = dict(depth=depth[None], confidence=np.ones((1, 3, 4)),
                    intrinsics=K[None], extrinsics=pose[None])
        y, x = np.mgrid[:3, :4]
        pixels = np.stack([x.ravel(), y.ravel(), np.ones(12)])
        camera = np.linalg.inv(K) @ pixels * depth.ravel()
        reference = (np.linalg.inv(pose) @ np.vstack([camera, np.ones(12)]))[:3].T
        np.testing.assert_allclose(points_from_prediction(data, stride=1), reference,
                                   atol=1e-12)

    def test_inverse_camera_transform(self):
        pose = np.eye(4, dtype=float)[:3]
        pose[:, 3] = [1, 2, 3]
        data = dict(depth=np.ones((1, 2, 2))*2,
                    confidence=np.ones((1, 2, 2)),
                    intrinsics=np.eye(3)[None], extrinsics=pose[None])
        points = points_from_prediction(data, stride=1)
        np.testing.assert_allclose(points[0], [-1, -2, -1])

    def test_scale_invariance(self):
        rng = np.random.default_rng(42)
        points = np.c_[rng.uniform(-2,2,(300,2)), np.ones(300)*3]
        first = extract(points, trials=30)
        second = extract(points*7, trials=30)
        self.assertEqual(len(first['planes']), 1)
        self.assertAlmostEqual(abs(second['planes'][0]['offset']),
                               abs(first['planes'][0]['offset'])*7)
        self.assertEqual(first['units'], 'arbitrary')


    def test_disjoint_coplanar_surfaces_stay_separate_patches(self):
        rng = np.random.default_rng(5)
        left = np.c_[rng.uniform(-4, -1.5, (400, 1)), rng.uniform(0, 2.5, (400, 1)), np.zeros((400, 1))]
        right = np.c_[rng.uniform(1.5, 4, (400, 1)), rng.uniform(0, 2.5, (400, 1)), np.zeros((400, 1))]
        result = extract(np.r_[left, right] + rng.normal(size=(800, 3)) * .004, trials=60)
        self.assertEqual(len(result['planes']), 1)
        info = result['planes'][0]['surface_patches']
        self.assertEqual(len(info['patches']), 2)
        u, v = np.asarray(info['axes_world'])
        for patch in info['patches']:
            lo, hi = np.asarray(patch['bounds_uv'])
            extents = np.sort(hi - lo)
            self.assertAlmostEqual(extents[0], 2.5, delta=.15)
            self.assertAlmostEqual(extents[1], 2.5, delta=.15)  # the 1.5 m gap is not bridged

    def test_rotated_translated_noisy_plane_keeps_patch_extents(self):
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        flat = np.c_[rng.uniform(0, 4, (900, 1)), rng.uniform(0, 2.5, (900, 1)), rng.normal(size=(900, 1)) * .004]
        moved = flat @ q.T + np.array([50., -20., 7.])
        first = extract(flat, trials=60)['planes'][0]['surface_patches']['patches'][0]
        second = extract(moved, trials=60)['planes'][0]['surface_patches']['patches'][0]
        sizes = lambda patch: np.sort(np.diff(np.asarray(patch['bounds_uv']), axis=0)[0])
        np.testing.assert_allclose(sizes(first), sizes(second), atol=.15)
        self.assertEqual(second['support_count'] > 800, True)

    def test_candidates_are_not_labelled_as_walls(self):
        rng = np.random.default_rng(1)
        points = np.c_[rng.uniform(0, 1, (300, 2)), np.zeros(300)]
        plane = extract(points, trials=30)['planes'][0]
        self.assertEqual(plane['semantic_class'], 'unknown')
        self.assertEqual(plane['measurement_interval'], None)

if __name__ == '__main__':
    unittest.main()
