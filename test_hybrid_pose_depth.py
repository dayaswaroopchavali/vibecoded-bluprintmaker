import unittest

import numpy as np

from hybrid_pose_depth import estimate_depth_scale, fit_window_to_direct, reproject_depth


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class HybridPoseDepthTests(unittest.TestCase):
    def test_pose_fit_recovers_similarity_and_camera_centres(self):
        Q = _rotz(0.37)
        scale = 2.4
        shift = np.array([1.2, -0.7, 0.35])
        centres = np.array([[0, 0, 0], [0.3, 0.1, 0.0], [0.6, -0.1, 0.1],
                            [0.7, 0.2, -0.2], [1.0, 0.1, 0.3]], dtype=float)
        local, direct = [], []
        for i, c in enumerate(centres):
            R = _rotz(0.08 * i)
            t = -R @ c
            R_dst = R @ Q.T
            t_dst = scale * t - R_dst @ shift
            local.append(np.column_stack((R, t)))
            direct.append(np.column_stack((R_dst, t_dst)))
        fit = fit_window_to_direct(np.asarray(local), np.asarray(direct))
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["s"], scale, places=10)
        np.testing.assert_allclose(fit["Q"], Q, atol=1e-10)
        np.testing.assert_allclose(fit["b"], shift, atol=1e-10)
        self.assertLess(fit["center_rms_residual"], 1e-10)
        self.assertLess(fit["orientation_max_deg"], 1e-5)

    def test_depth_uses_source_k_then_direct_k_and_pose_scale(self):
        h = w = 9
        ys, xs = np.indices((h, w))
        # Principal point is at image centre. Target focal is half the source focal,
        # so the target image sees a smaller central patch after reprojection.
        K_src = np.array([[8.0, 0, 4], [0, 8.0, 4], [0, 0, 1]])
        K_dst = np.array([[4.0, 0, 4], [0, 4.0, 4], [0, 0, 1]])
        E = np.column_stack((np.eye(3), np.zeros(3)))
        depth = np.full((h, w), 3.0, np.float32)
        conf = np.ones((h, w), np.float32)
        moved, moved_conf, stats = reproject_depth(depth, conf, K_src, E, K_dst, E,
                                                    2.0, np.eye(3), np.zeros(3))
        valid = moved > 0
        self.assertGreater(valid.sum(), 0)
        np.testing.assert_allclose(moved[valid], 6.0, atol=1e-6)
        self.assertTrue(np.all(moved_conf[valid] > 0))
        self.assertEqual(stats["valid_target_fraction"], float(valid.mean()))
        # The direct focal length controls pixel placement: valid splats stay in
        # the central area rather than inheriting source-K pixel coordinates.
        self.assertLess(np.where(valid)[1].max(), w - 1)

        same_k, _, same_stats = reproject_depth(depth, conf, K_src, E, K_src, E,
                                                 2.0, np.eye(3), np.zeros(3))
        self.assertGreater(same_stats["valid_target_fraction"], 0.95)
        np.testing.assert_allclose(same_k[1:-1, 1:-1], 6.0, atol=1e-6)

    def test_frame_scale_is_robust_to_depth_outliers(self):
        rng = np.random.default_rng(4)
        window = rng.uniform(1.0, 5.0, (64, 64)).astype(np.float32)
        direct = window * 1.23
        direct.ravel()[::37] *= 2.5
        wc = np.ones_like(window)
        dc = np.ones_like(window)
        fit = estimate_depth_scale(window, wc, direct, dc, min_pixels=100)
        self.assertTrue(fit["ok"])
        self.assertAlmostEqual(fit["scale"], 1.23, places=3)
        self.assertLess(fit["relative_mad"], 1e-5)
        inconsistent = direct.copy()
        inconsistent[:, 32:] *= 1.7
        poor = estimate_depth_scale(window, wc, inconsistent, dc, min_pixels=100)
        self.assertGreater(poor["relative_mad"], 0.10)


if __name__ == "__main__":
    unittest.main()
