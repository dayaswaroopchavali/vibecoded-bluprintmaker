import unittest

import numpy as np

from pathlib import Path

from audit_camera_trajectory import (audit_path, camera_centers, fit_rotation_alignment, fit_similarity,
                                     gt_opencv_c2w, orientation_audit, rotation_angles_deg)
from simulator import camera_basis, rotation_matrix


def rotation_xyz():
    ax, ay, az = .31, -.27, .53
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


class CameraTrajectoryAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[0, 0, 0], [1, 0, .2], [0, 2, .4], [.3, .5, 1.5], [1.2, 1.1, 2.]])

    def test_known_proper_similarity_aligns_to_zero(self):
        rotation = rotation_xyz()
        target = 2.7 * (self.source @ rotation.T) + np.array([4., -2., .8])
        result = fit_similarity(self.source, target)
        self.assertLess(result["rmse"], 1e-12)
        self.assertAlmostEqual(result["scale"], 2.7)
        self.assertAlmostEqual(np.linalg.det(result["rotation"]), 1.)

    def test_warped_path_has_nonzero_residual(self):
        target = 1.4 * (self.source @ rotation_xyz().T) + np.array([2., 3., -1.])
        target[2, 2] += .8
        self.assertGreater(fit_similarity(self.source, target)["rmse"], .01)

    def test_reflection_is_not_allowed_to_fit_exactly(self):
        reflected = self.source.copy()
        reflected[:, 0] *= -1
        result = fit_similarity(self.source, reflected)
        self.assertGreater(result["rmse"], .01)
        self.assertGreater(np.linalg.det(result["rotation"]), .999)

    def test_degenerate_trajectories_are_rejected(self):
        line = np.array([[i, 0., 0.] for i in range(5)])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            fit_similarity(line, line)

    def test_camera_centers_follow_world_to_camera_convention(self):
        rot = rotation_xyz()
        center = np.array([1.2, -2., .7])
        extrinsic = np.c_[rot, -rot @ center][None, ...]
        np.testing.assert_allclose(camera_centers(extrinsic), center[None, :], atol=1e-12)

    def test_orientation_error_is_invariant_to_one_global_world_rotation(self):
        indices = list(range(5))
        poses = [{"yaw": .12 * i, "pitch": -.07 + .025 * i} for i in indices]
        q = rotation_xyz()
        pred_w2c = []
        for pose in poses:
            right = np.array([np.cos(pose["yaw"]), -np.sin(pose["yaw"]), 0.])
            forward = np.array([np.sin(pose["yaw"]) * np.cos(pose["pitch"]),
                                np.cos(pose["yaw"]) * np.cos(pose["pitch"]), -np.sin(pose["pitch"])])
            up = np.cross(right, forward)
            gt_c2w = np.column_stack((right, -up, forward))
            pred_w2c.append(gt_c2w.T @ q)
        errors = rotation_angles_deg(np.asarray(pred_w2c), poses, indices, q)
        self.assertLess(errors["geodesic_max_deg"], 1e-4)  # arccos near 1 limits float precision to ~1e-6 deg
        self.assertLess(errors["forward_max_deg"], 1e-6)
        self.assertLess(errors["up_max_deg"], 1e-6)

    def test_per_frame_orientation_error_is_detected(self):
        indices = list(range(5))
        poses = [{"yaw": .12 * i, "pitch": -.07 + .025 * i} for i in indices]
        gt_c2w = []
        for pose in poses:
            right = np.array([np.cos(pose["yaw"]), -np.sin(pose["yaw"]), 0.])
            forward = np.array([np.sin(pose["yaw"]) * np.cos(pose["pitch"]),
                                np.cos(pose["yaw"]) * np.cos(pose["pitch"]), -np.sin(pose["pitch"])])
            up = np.cross(right, forward)
            gt_c2w.append(np.column_stack((right, -up, forward)))
        pred_w2c = np.asarray([r.T for r in gt_c2w])
        roll = np.array([[np.cos(.18), -np.sin(.18), 0], [np.sin(.18), np.cos(.18), 0], [0, 0, 1]])
        pred_w2c[2] = (gt_c2w[2] @ roll).T
        errors = rotation_angles_deg(pred_w2c, poses, indices, np.eye(3))
        self.assertGreater(errors["geodesic_max_deg"], 5.)

    def test_reflected_camera_rotation_is_rejected(self):
        ext = np.tile(np.c_[np.eye(3), np.zeros(3)], (3, 1, 1))
        ext[0, :, 0] *= -1
        with self.assertRaisesRegex(ValueError, "proper rotations"):
            camera_centers(ext)

def random_rotation(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else q @ np.diag([1., 1., -1.])


ROLLED_INVERTED = [{"yaw": .3 * i, "pitch": p, "roll": r} for i, (p, r) in enumerate(
    [(0., 0.), (.2, .7), (-.4, -1.1), (2.4, 0.), (1.9, 2.5), (.1, 3.0), (-1.2, -2.0), (3.0, .4)])]
EXPECTED_AUDIT_V1_FRACTIONS = (.0092, .0124, .0232)
FULL = Path(__file__).resolve().parents[2] / "outputs" / "sim-benchmark" / "blender-full"


class OrientationExtensionTests(unittest.TestCase):
    def pred_from_gt(self, poses, q):
        # prediction lives in a rotated world: R_w2c_pred = R_gt_w2c @ q  (so alignment q recovers GT)
        return np.asarray([gt_opencv_c2w(p).T @ q for p in poses])

    def test_gt_frame_matches_simulator_basis_including_roll_and_inversion(self):
        for pose in ROLLED_INVERTED:
            right, up, forward = camera_basis(pose)
            m = gt_opencv_c2w(pose)
            np.testing.assert_allclose(m[:, 0], right); np.testing.assert_allclose(m[:, 1], -up)
            np.testing.assert_allclose(m[:, 2], forward)
            self.assertAlmostEqual(np.linalg.det(m), 1.0)
            # simulator.rotation_matrix columns are (right, forward, up): same camera, different axis labelling
            sim = rotation_matrix(pose)
            np.testing.assert_allclose(m[:, 0], sim[:, 0]); np.testing.assert_allclose(m[:, 2], sim[:, 1])
            np.testing.assert_allclose(-m[:, 1], sim[:, 2])

    def test_roll_sign_matches_simulator(self):
        # positive roll: right -> right*cos r - up*sin r, so camera right acquires a -up component
        base = gt_opencv_c2w({"yaw": 0., "pitch": 0., "roll": 0.})
        rolled = gt_opencv_c2w({"yaw": 0., "pitch": 0., "roll": .3})
        self.assertAlmostEqual(rolled[2, 0], -np.sin(.3))  # right gains a -up component (world z)
        np.testing.assert_allclose(rolled[:, 0], camera_basis({"yaw": 0., "pitch": 0., "roll": .3})[0])

    def test_invariant_to_global_world_rotation_rolled_and_inverted(self):
        for seed in range(4):
            q = random_rotation(seed)
            pred = self.pred_from_gt(ROLLED_INVERTED, q)
            idx = list(range(len(ROLLED_INVERTED)))
            e = rotation_angles_deg(pred, ROLLED_INVERTED, idx, q)
            self.assertLess(e["geodesic_max_deg"], 1e-6)
            audit = orientation_audit(pred, ROLLED_INVERTED, idx, q)
            self.assertLess(audit["rotation_kabsch_aligned"]["geodesic_max_deg"], 1e-6)
            self.assertLess(audit["center_umeyama_aligned"]["geodesic_max_deg"], 1e-6)

    def test_camera_to_world_input_equivalent(self):
        q = random_rotation(9); idx = list(range(len(ROLLED_INVERTED)))
        w2c = self.pred_from_gt(ROLLED_INVERTED, q)
        c2w = np.transpose(w2c, (0, 2, 1))
        a = orientation_audit(w2c, ROLLED_INVERTED, idx, q)["rotation_kabsch_aligned"]
        b = orientation_audit(c2w, ROLLED_INVERTED, idx, q, pose_kind="camera_to_world")["rotation_kabsch_aligned"]
        self.assertAlmostEqual(a["geodesic_max_deg"], b["geodesic_max_deg"], places=6)
        self.assertLess(b["geodesic_max_deg"], 1e-6)

    def test_pure_roll_error_is_reported_exactly_and_per_frame(self):
        idx = list(range(len(ROLLED_INVERTED)))
        pred = self.pred_from_gt(ROLLED_INVERTED, np.eye(3))
        c, s = np.cos(np.radians(12)), np.sin(np.radians(12))
        roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])  # about camera forward (z)
        pred = pred.copy(); pred[5] = (gt_opencv_c2w(ROLLED_INVERTED[5]) @ roll).T
        e = rotation_angles_deg(pred, ROLLED_INVERTED, idx, np.eye(3))
        self.assertAlmostEqual(e["per_frame_geodesic_deg"][5], 12., places=6)
        self.assertAlmostEqual(e["forward_max_deg"], 0., places=6)  # forward axis unchanged: roll only
        self.assertAlmostEqual(e["up_max_deg"], 12., places=6)
        self.assertEqual(sum(v > 1e-6 for v in e["per_frame_geodesic_deg"]), 1)
        self.assertAlmostEqual(e["geodesic_median_deg"], 0., places=6)
        self.assertAlmostEqual(e["geodesic_mean_deg"], 12. / len(idx), places=6)

    def test_wrong_roll_sign_convention_would_be_caught(self):
        # a predictor with opposite roll sign must show error ~2*|roll|, proving roll is audited
        poses = [{"yaw": .2, "pitch": .1, "roll": .5}]
        flipped = [{"yaw": .2, "pitch": .1, "roll": -.5}]
        pred = np.asarray([gt_opencv_c2w(flipped[0]).T])
        e = rotation_angles_deg(pred, poses, [0], np.eye(3))
        self.assertAlmostEqual(e["geodesic_max_deg"], 57.29578, places=3)

    def test_kabsch_recovers_global_rotation_and_inverted_camera_error_detected(self):
        q = random_rotation(3); idx = list(range(len(ROLLED_INVERTED)))
        pred = self.pred_from_gt(ROLLED_INVERTED, q)
        gt = np.asarray([gt_opencv_c2w(p) for p in ROLLED_INVERTED])
        np.testing.assert_allclose(fit_rotation_alignment(pred, gt), q, atol=1e-10)
        pred = pred.copy()
        flip = np.diag([-1., -1., 1.])  # camera upside down (180 deg about forward)
        pred[1] = (gt_opencv_c2w(ROLLED_INVERTED[1]) @ flip).T @ q
        out = orientation_audit(pred, ROLLED_INVERTED, idx)["rotation_kabsch_aligned"]
        self.assertGreater(out["geodesic_max_deg"], 100.)  # a single 180 degree flip stays visible

    @unittest.skipUnless((FULL / "da3" / "path_00" / "prediction.npz").is_file(), "frozen blender-full artifacts absent")
    def test_camera_audit_v1_center_rmse_fractions_reproduce(self):
        for name, expected in zip(("path_00", "path_01", "path_02"), EXPECTED_AUDIT_V1_FRACTIONS):
            r = audit_path(FULL / "da3" / name / "prediction.npz", FULL / "private_ground_truth" / f"{name}.json", name)
            self.assertAlmostEqual(r["metrics"]["position_rmse_fraction_of_gt_extent"], expected, places=4)
            self.assertIn("orientation_extended", r["metrics"])

if __name__ == "__main__":
    unittest.main()




