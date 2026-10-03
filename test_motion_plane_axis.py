import unittest

import numpy as np

from motion_plane_axis import select_motion_axis


def poses(centers, rolls):
    out = []
    for c, angle in zip(centers, rolls):
        co, si = np.cos(angle), np.sin(angle)
        C = np.array([[co, -si, 0], [si, co, 0], [0, 0, 1.]])
        R = C.T
        out.append(np.column_stack((R, -R @ c)))
    return np.asarray(out)


AXES = [{"axis_world": [1, 0, 0]}, {"axis_world": [0, 1, 0]},
        {"axis_world": [0, 0, 1]}]


class MotionPlaneAxisTests(unittest.TestCase):
    def test_arbitrary_roll_does_not_change_planar_motion_axis(self):
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        centers = np.column_stack((2 * np.cos(t), 1.5 * np.sin(t), .05 * np.sin(2 * t)))
        rolls = np.linspace(0, 4 * np.pi, len(t))
        result = select_motion_axis(poses(centers, rolls), AXES)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["hypothesis_index"], 2)
        self.assertFalse(result["axis_sign_resolved"])

    def test_near_duplicate_axis_is_not_a_competing_axis(self):
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        centers = np.column_stack((2 * np.cos(t), 1.5 * np.sin(t), .02 * np.sin(2 * t)))
        poses_ = poses(centers, np.zeros(len(t)))
        dup = [{"axis_world": [0, 0, 1]}, {"axis_world": [0, -.127, -.992]}, {"axis_world": [1, 0, 0]}]
        result = select_motion_axis(poses_, dup)
        self.assertEqual((result["status"], result["hypothesis_index"]), ("selected", 0))
        # Two genuinely distinct, equally aligned directions are still ambiguous.
        tilted = [{"axis_world": [0, 0, 1]}, {"axis_world": [0, .6, .8]}]
        self.assertEqual(select_motion_axis(poses_, tilted)["reason"], "axis_alignment_ambiguous")

    def test_three_dimensional_motion_is_refused(self):
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        centers = np.column_stack((np.cos(t), np.sin(t), np.sin(2 * t)))
        result = select_motion_axis(poses(centers, np.zeros(len(t))), AXES)
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["reason"], "trajectory_not_planar")

    def test_nearly_linear_motion_is_refused(self):
        t = np.linspace(-1, 1, 120)
        centers = np.column_stack((t, .01 * np.sin(t), .01 * np.cos(t)))
        result = select_motion_axis(poses(centers, np.zeros(len(t))), AXES)
        self.assertEqual(result["status"], "refused")
        self.assertEqual(result["reason"], "trajectory_nearly_linear")


if __name__ == "__main__":
    unittest.main()
