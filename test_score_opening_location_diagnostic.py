import unittest

from score_opening_location_diagnostic import _predicted_world_interval, location_matches


class OpeningLocationDiagnosticTests(unittest.TestCase):
    def test_similarly_sized_opening_elsewhere_on_wall_is_not_a_match(self):
        self.assertFalse(location_matches((0.0, 0.9), (1.5, 2.4)))
        self.assertTrue(location_matches((0.0, 0.9), (0.05, 0.95)))

    def test_reversed_wall_edge_preserves_world_interval(self):
        boundary = {
            "vertices_xy": [[2, 1], [-2, 1], [-2, -1], [2, -1]],
            "coordinate_origin_world": [0, 0, 0],
            "axes_world": {"e1": [1, 0, 0], "e2": [0, 1, 0]},
        }
        interval = _predicted_world_interval(
            boundary, {"wall_index": 0, "u_interval_m": [0.5, 1.5]}, 0)
        self.assertAlmostEqual(interval[0], 0.5)
        self.assertAlmostEqual(interval[1], 1.5)


if __name__ == "__main__":
    unittest.main()
