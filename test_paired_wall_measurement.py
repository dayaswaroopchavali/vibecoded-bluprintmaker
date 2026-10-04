import unittest

from paired_wall_measurement import estimate_paired_wall_measurements


def rectangle_walls():
    values = [([0.0, 1.0], 0.0), ([1.0, 0.0], 10.02),
              ([0.0, -1.0], -6.10), ([-1.0, 0.0], 0.0)]
    return {"lines": [{"id": i, "normal_xy": normal, "offset": offset,
                       "uncertainty": {"offset_sd": 0.003, "angle_sd_deg": 0.1}}
                      for i, (normal, offset) in enumerate(values)]}


def rectangle_boundary():
    return {"status": "complete_polygon_hypothesis",
            "edges": [{"line_id": i} for i in range(4)]}


class PairedWallMeasurementTests(unittest.TestCase):
    def test_projects_opposite_plane_offsets_and_propagates_uncertainty(self):
        result = estimate_paired_wall_measurements(rectangle_walls(), rectangle_boundary())
        self.assertEqual(result["status"], "paired_plane_measurement")
        self.assertEqual(result["opposite_pairing"], [[0, 2], [1, 3]])
        self.assertEqual(result["edge_lengths_m_sorted"], [6.1, 6.1, 10.02, 10.02])
        self.assertAlmostEqual(result["pair_measurements"][0]["separation_sd_m"], 0.0042426407)

    def test_refuses_a_non_parallel_opposite_pair(self):
        walls = rectangle_walls()
        walls["lines"][2]["normal_xy"] = [0.2, -0.98]
        result = estimate_paired_wall_measurements(walls, rectangle_boundary())
        self.assertEqual(result["status"], "unsupported_shape")
        self.assertEqual(result["reason"], "opposite_walls_not_parallel_enough")

    def test_refuses_non_complete_or_non_four_edge_boundaries(self):
        self.assertEqual(estimate_paired_wall_measurements(
            rectangle_walls(), {"status": "incomplete", "edges": []})["status"],
            "unsupported_boundary")
        boundary = {"status": "complete_polygon_hypothesis", "edges": [{"line_id": i} for i in range(3)]}
        self.assertEqual(estimate_paired_wall_measurements(rectangle_walls(), boundary)["status"],
                         "unsupported_shape")


if __name__ == "__main__":
    unittest.main()
