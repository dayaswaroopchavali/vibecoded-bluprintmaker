"""Camera-motion up prior and polygon/box orientation: synthetic poses only, no scene truth."""
import unittest

import numpy as np

from orient_room import camera_up_prior, orient, orient_polygon
from room_hypothesis import estimate_polygon
from test_room_hypothesis import L_SHAPE, _polygon_room, _rotation


def _unit(v):
    return v / np.linalg.norm(v)


def extrinsics(up, centers, rng, roll_deg=0.0, noise_deg=2.0):
    """World-to-camera 3x4 poses, OpenCV convention: x right, y down, z forward, camera-up = -y."""
    up = _unit(np.asarray(up, float)); seed = np.array([1., 0, 0]) if abs(up[0]) < .9 else np.array([0, 1., 0])
    e1 = _unit(np.cross(up, seed)); e2 = np.cross(up, e1)
    poses = []
    for i, c in enumerate(centers):
        yaw = 2 * np.pi * i / len(centers) + rng.normal() * .05
        forward = _unit(np.cos(yaw) * e1 + np.sin(yaw) * e2 - .1 * up)
        down = -up - forward * (-up @ forward); down = _unit(down)
        right = np.cross(down, forward)
        roll = np.radians(roll_deg + rng.normal() * noise_deg)
        right, down = np.cos(roll) * right + np.sin(roll) * down, -np.sin(roll) * right + np.cos(roll) * down
        R = np.stack([right, down, forward])
        poses.append(np.c_[R, -R @ np.asarray(c, float)])
    return np.asarray(poses)


def loop_centers(up, count=24, radius=1.0, origin=(0., 0., 0.)):
    up = _unit(np.asarray(up, float)); seed = np.array([1., 0, 0]) if abs(up[0]) < .9 else np.array([0, 1., 0])
    e1 = _unit(np.cross(up, seed)); e2 = np.cross(up, e1)
    a = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.asarray(origin) + radius * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2) + .12 * np.sin(2 * a)[:, None] * up


class CameraUpPriorTests(unittest.TestCase):
    def test_recovers_up_from_camera_motion_and_declares_it_a_prior(self):
        for seed in range(6):
            rng = np.random.default_rng(seed); up = _unit(rng.normal(size=3))
            result = camera_up_prior(extrinsics(up, loop_centers(up, origin=rng.normal(size=3) * 5), rng))
            self.assertEqual(result["status"], "declared_prior")
            angle = np.degrees(np.arccos(np.clip(np.dot(result["up_world"], up), -1, 1)))
            self.assertLess(angle, 4.0)
            self.assertTrue(result["declared_prior"]); self.assertFalse(result["oracle_axis_matching_used"])
            self.assertTrue(result["independent_of_scene_geometry"])
            self.assertEqual(result["trajectory_check"]["status"], "checked")

    def test_translation_of_the_whole_capture_does_not_change_up(self):
        rng = np.random.default_rng(3); up = _unit(rng.normal(size=3)); centers = loop_centers(up)
        a = camera_up_prior(extrinsics(up, centers, np.random.default_rng(9)))["up_world"]
        b = camera_up_prior(extrinsics(up, centers + np.array([250., -90., 31.]), np.random.default_rng(9)))["up_world"]
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_landscape_capture_has_no_camera_up_consensus(self):
        rng = np.random.default_rng(4); up = _unit(rng.normal(size=3))
        result = camera_up_prior(extrinsics(up, loop_centers(up), rng, roll_deg=90.0))
        self.assertEqual(result["status"], "refused")
        self.assertIsNone(result["up_world"])

    def test_consistent_camera_up_that_contradicts_the_motion_plane_is_refused(self):
        # Phone held on its side the whole time: a strong, consistent, wrong camera-up.
        side_up = np.array([1., 0., 0.]); floor_normal = np.array([0., 0., 1.])
        result = camera_up_prior(extrinsics(side_up, loop_centers(floor_normal), np.random.default_rng(6)))
        self.assertEqual(result["status"], "refused")
        self.assertIn("disagrees", result["reason"])
        self.assertGreater(result["trajectory_check"]["angle_to_camera_up_degrees"], 60)

    def test_mixed_orientations_have_no_consensus(self):
        rng = np.random.default_rng(5); up = _unit(rng.normal(size=3))
        upright = extrinsics(up, loop_centers(up, 12), rng)
        rolled = extrinsics(up, loop_centers(up, 12), rng, roll_deg=90.0)
        result = camera_up_prior(np.concatenate([upright[::2], rolled[::2], upright[1::2][:2], rolled[1::2][:2]]))
        self.assertEqual(result["status"], "refused")

    def test_too_few_or_invalid_poses_are_refused(self):
        up = np.array([0., 0., 1.]); rng = np.random.default_rng(1)
        self.assertEqual(camera_up_prior(extrinsics(up, loop_centers(up, 2), rng))["status"], "refused")
        self.assertEqual(camera_up_prior(np.zeros((5, 2)))["status"], "refused")


class PolygonOrientationTests(unittest.TestCase):
    def setUp(self):
        self.data, self.points, self.up, self.truth = _polygon_room(L_SHAPE, seed=3)
        self.R = _rotation(8)  # the fixture's rotation for seed=3
        self.polygon = estimate_polygon(self.data, self.points, up=self.up)
        self.assertEqual(self.polygon["status"], "complete_polygon_hypothesis")
        local = [(1.2, 1.2), (4.5, 1.0), (5.0, 2.2), (2.0, 2.4), (1.0, 3.8), (2.2, 4.2), (1.6, 2.0), (3.5, 1.6)]
        self.centers = np.asarray([self.R @ np.array([x, y, 1.4]) for x, y in local * 3])

    def _poses(self, up, centers=None):
        return extrinsics(up, self.centers if centers is None else centers, np.random.default_rng(2))

    def test_complete_polygon_is_oriented_with_declared_up_and_unscaled_walls(self):
        result = orient_polygon(self.polygon, self._poses(self.up))
        self.assertEqual(result["status"], "oriented_unscaled_polygon_floorplan")
        self.assertEqual(result["up_source"], "camera_motion_prior")
        self.assertFalse(result["scale_applied"]); self.assertIsNone(result["metric_scale_source"])
        self.assertEqual(len(result["walls"]), 6)  # an L is not collapsed to a box
        self.assertIsNone(result["height_unscaled"])
        self.assertEqual(result["height_status"], "unavailable_not_estimated")
        np.testing.assert_allclose(sorted(result["wall_lengths_unscaled"]),
                                   sorted(np.linalg.norm(np.roll(self.truth, -1, axis=0) - self.truth, axis=1)), rtol=.03)

    def test_incomplete_polygon_is_refused(self):
        data, points, up, _ = _polygon_room(L_SHAPE, seed=3, missing=(4,))
        incomplete = estimate_polygon(data, points, up=up)
        result = orient_polygon(incomplete, self._poses(up))
        self.assertEqual(result["status"], "orientation_refused")
        self.assertEqual(result["walls"], [])
        self.assertIsNone(result["floorplan"])

    def test_polygon_declaring_a_different_up_is_refused(self):
        tilted = _unit(self.up + .4 * self.R[:, 0])           # about 22 degrees away from the prior
        polygon = {**self.polygon, "axes_world": {**self.polygon["axes_world"], "up": tilted.tolist()}}
        result = orient_polygon(polygon, self._poses(self.up))
        self.assertEqual(result["status"], "orientation_refused")
        self.assertIn("different up", result["reason"])
        self.assertGreater(result["up_disagreement_degrees"], 10)
        slight = _unit(self.up + .05 * self.R[:, 0])          # ~3 degrees: still the same declared prior
        polygon = {**self.polygon, "axes_world": {**self.polygon["axes_world"], "up": slight.tolist()}}
        self.assertEqual(orient_polygon(polygon, self._poses(self.up))["status"], "oriented_unscaled_polygon_floorplan")

    def test_flipped_up_is_refused_not_silently_accepted(self):
        polygon = estimate_polygon(self.data, self.points, up=-self.up)
        result = orient_polygon(polygon, self._poses(self.up))
        self.assertEqual(result["status"], "orientation_refused")

    def test_cameras_outside_the_polygon_are_refused(self):
        far = self.centers + self.R @ np.array([40., 40., 0.])
        result = orient_polygon(self.polygon, self._poses(self.up, far))
        self.assertEqual(result["status"], "orientation_refused")
        self.assertLess(result["camera_inside_fraction"], .7)

    def test_unusable_camera_prior_refuses_polygon(self):
        result = orient_polygon(self.polygon, extrinsics(self.up, self.centers, np.random.default_rng(2), roll_deg=90.0))
        self.assertEqual(result["status"], "orientation_refused")
        self.assertIn("prior unavailable", result["reason"])


def box_hypothesis(rotation, dims, scale=1.0):
    pairs = [{"complete": True, "low": -d * scale / 2, "high": d * scale / 2} for d in dims]
    return {"status": "complete_box_hypothesis", "coordinate_origin_world": [0., 0., 0.],
            "axes": [{"direction_world": rotation[:, i].tolist(), "surface_pair": pairs[i]} for i in range(3)],
            "dimensions": [d * scale for d in dims]}


class BoxOrientationTests(unittest.TestCase):
    def centers_in_box(self, rotation, dims, height_axis):
        pts = []
        for a in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            local = np.zeros(3); others = [i for i in range(3) if i != height_axis]
            local[others[0]] = .25 * dims[others[0]] * np.cos(a); local[others[1]] = .25 * dims[others[1]] * np.sin(a)
            local[height_axis] = -dims[height_axis] / 2 + .45 * dims[height_axis]
            pts.append(rotation @ local)
        return np.asarray(pts)

    def test_height_axis_follows_the_camera_prior_even_when_it_is_the_longest_extent(self):
        rotation = _rotation(21); dims = (7.0, 3.0, 4.0)
        up = rotation[:, 0]
        result = orient(box_hypothesis(rotation, dims), extrinsics(up, self.centers_in_box(rotation, dims, 0), np.random.default_rng(1)))
        self.assertEqual(result["status"], "oriented_unscaled_box_floorplan", result.get("reason"))
        self.assertEqual(result["raw_height_axis_index"], 0)
        self.assertAlmostEqual(result["height_unscaled"], 7.0)
        self.assertEqual(sorted(result["wall_lengths_unscaled"]), [3.0, 3.0, 4.0, 4.0])
        self.assertEqual(result["up_source"], "camera_motion_prior")
        self.assertFalse(result["scale_applied"])

    def test_up_between_two_axes_is_refused_rather_than_guessed(self):
        rotation = _rotation(22); dims = (3.0, 3.1, 4.0)
        up = _unit(rotation[:, 0] + rotation[:, 1])
        result = orient(box_hypothesis(rotation, dims), extrinsics(up, loop_centers(up), np.random.default_rng(1)))
        self.assertEqual(result["status"], "orientation_refused")
        self.assertIsNone(result["floorplan"])

    def test_incomplete_box_is_refused(self):
        rotation = _rotation(23); dims = (3.0, 5.0, 4.0)
        hypothesis = box_hypothesis(rotation, dims); hypothesis["status"] = "incomplete"
        result = orient(hypothesis, extrinsics(rotation[:, 2], self.centers_in_box(rotation, dims, 2), np.random.default_rng(1)))
        self.assertEqual(result["status"], "orientation_refused")


if __name__ == "__main__":
    unittest.main()
