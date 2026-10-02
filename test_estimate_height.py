"""Synthetic geometry checks for the RGB-derived height pairing heuristic."""
import unittest

import numpy as np

from estimate_height import estimate_height


def _scene(scale=1.0, origin=None):
    origin = np.asarray(origin if origin is not None else [0., 0., 0.], dtype=float)
    # A rigid rotation makes the declared up prior non-axis-aligned in world coordinates.
    theta, phi = .63, -.28
    rz = np.array([[np.cos(theta), -np.sin(theta), 0.],
                   [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., np.cos(phi), -np.sin(phi)],
                   [0., np.sin(phi), np.cos(phi)]])
    rotation = rz @ rx
    up, ex, ey = rotation[:, 2], rotation[:, 0], rotation[:, 1]
    rng = np.random.default_rng(4)
    local = np.column_stack([rng.uniform(-4, 4, 1000), rng.uniform(-3, 3, 1000),
                             rng.uniform(0, 3, 1000)]) * scale
    points = origin + local @ rotation.T
    centers_local = np.array([[-1., 0., 1.4], [0., 1., 1.6], [1., -1., 1.5]]) * scale
    centers = origin + centers_local @ rotation.T
    axes = [ex.tolist(), ey.tolist()]

    def plane(height, support, bounds=(-4, -3, 4, 3)):
        center = origin + up * (height * scale)
        lo_x, lo_y, hi_x, hi_y = np.asarray(bounds) * scale
        return {"normal": up.tolist(), "offset": float(-up @ center), "center": center.tolist(),
                "support_count": support,
                "surface_patches": {"axes_world": axes, "patches": [
                    {"bounds_uv": [[lo_x, lo_y], [hi_x, hi_y]], "support_count": support}]}}

    planes = [plane(0., 600), plane(3., 500)]
    return points, centers, up, {"planes": planes}, plane


class EstimateHeightTests(unittest.TestCase):
    def test_rotated_translated_room_pairs_supported_bracketing_planes(self):
        points, centers, up, data, _ = _scene(origin=[17., -8., 2.])
        result = estimate_height(data, points, up, centers)
        self.assertEqual(result["status"], "estimated_unscaled_separation")
        self.assertEqual((result["floor_plane_id"], result["ceiling_plane_id"]), (0, 1))
        self.assertAlmostEqual(result["separation_unscaled"], 3.)


    def test_estimate_is_scale_equivariant(self):
        p1, c1, u1, d1, _ = _scene(scale=1.)
        p2, c2, u2, d2, _ = _scene(scale=7.5)
        a = estimate_height(d1, p1, u1, c1)
        b = estimate_height(d2, p2, u2, c2)
        self.assertEqual(a["status"], "estimated_unscaled_separation")
        self.assertEqual(b["status"], "estimated_unscaled_separation")
        self.assertAlmostEqual(b["separation_unscaled"] / a["separation_unscaled"], 7.5)


    def test_low_table_is_not_accepted_as_room_floor_or_ceiling(self):
        points, centers, up, data, plane = _scene()
        data["planes"].append(plane(.8, 120, bounds=(-.7, -.45, .7, .45)))
        result = estimate_height(data, points, up, centers)
        self.assertEqual(result["status"], "estimated_unscaled_separation")
        self.assertEqual(result["floor_plane_id"], 0)
        self.assertEqual(result["ceiling_plane_id"], 1)
        self.assertFalse(result["support_evidence"]["candidate_planes"][2]["support_accepted"])


    def test_missing_ceiling_is_reported_incomplete(self):
        points, centers, up, data, _ = _scene()
        data["planes"] = data["planes"][:1]
        result = estimate_height(data, points, up, centers)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["separation_unscaled"])
        self.assertTrue(any("ceiling" in reason for reason in result["reasons"]))


    def test_competing_equally_supported_floors_are_ambiguous(self):
        points, centers, up, data, plane = _scene()
        data["planes"].append(plane(-.1, 600))
        result = estimate_height(data, points, up, centers)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["separation_unscaled"])
        self.assertGreaterEqual(len(result["competing_pairs"]), 2)

    def test_tilted_pair_is_translation_equivariant_and_reports_local_range(self):
        points, centers, up, data, _ = _scene()
        ex = np.asarray(data["planes"][0]["surface_patches"]["axes_world"][0])
        ceiling = data["planes"][1]
        tilted = up + .05 * ex
        tilted /= np.linalg.norm(tilted)
        center = np.asarray(ceiling["center"])
        ceiling["normal"] = tilted.tolist()
        ceiling["offset"] = float(-tilted @ center)
        first = estimate_height(data, points, up, centers)
        self.assertEqual(first["status"], "ambiguous")
        self.assertIsNotNone(first["local_separation_median_unscaled"])

        translation = np.array([100., -50., 20.])
        moved = {"planes": []}
        for original in data["planes"]:
            item = {**original}
            item["center"] = (np.asarray(item["center"]) + translation).tolist()
            item["offset"] = float(original["offset"] - np.asarray(original["normal"]) @ translation)
            moved["planes"].append(item)
        second = estimate_height(moved, points + translation, up, centers + translation)
        self.assertEqual(second["status"], first["status"])
        self.assertAlmostEqual(second["local_separation_median_unscaled"],
                               first["local_separation_median_unscaled"])
        np.testing.assert_allclose(second["local_separation_range_unscaled"],
                                   first["local_separation_range_unscaled"])

    def test_opposed_candidate_tilts_are_not_reported_as_one_height(self):
        points, centers, up, data, _ = _scene()
        ex = np.asarray(data["planes"][0]["surface_patches"]["axes_world"][0])
        for sign, plane in ((1., data["planes"][0]), (-1., data["planes"][1])):
            normal = up + sign * .09 * ex
            normal /= np.linalg.norm(normal)
            center = np.asarray(plane["center"])
            plane["normal"] = normal.tolist()
            plane["offset"] = float(-normal @ center)
        result = estimate_height(data, points, up, centers)
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["separation_unscaled"])
        self.assertGreater(result["pair_normal_angle_degrees"], 2.)


if __name__ == "__main__":
    unittest.main()
