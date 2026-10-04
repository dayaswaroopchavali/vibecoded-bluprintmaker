import unittest

import numpy as np

from rectify_rectangle import rectify


def boundary(vertices):
    return {"status": "complete_polygon_hypothesis", "shape_gate_eligible": True,
            "vertices_xy": vertices,
            "edges": [{"line_id": i, "start_xy": vertices[i],
                       "end_xy": vertices[(i + 1) % 4]} for i in range(4)]}


PAIRS = {"status": "paired_plane_measurement", "opposite_pair_alignments": [1, 1],
         "pair_axis_alignment_abs_dot": 0,
         "pair_measurements": [{"line_ids": [0, 2], "separation_m": 6.0},
                               {"line_ids": [1, 3], "separation_m": 4.0}]}


class RectangleRectificationTests(unittest.TestCase):
    def test_measured_dimensions_replace_noisy_corners(self):
        vertices = [[0.003, 0.0], [4.005, -.002], [4.001, 6.003], [-.002, 5.998]]
        openings = {"openings": [{"wall_index": 0, "u_interval_m": [1.0, 2.0], "width_m": 1.0}]}
        result = rectify(boundary(vertices), PAIRS, openings)
        self.assertEqual(result["status"], "rectified")
        actual = np.asarray(result["boundary"]["vertices_xy"])
        lengths = np.linalg.norm(np.roll(actual, -1, axis=0) - actual, axis=1)
        np.testing.assert_allclose(lengths, [4, 6, 4, 6], atol=1e-9)
        self.assertLess(result["max_corner_shift_m"], .01)
        self.assertLess(abs(result["openings"][0]["width_m"] - 1.0), .001)
        self.assertTrue(result["truth_free"])

    def test_trapezoid_refused_instead_of_forced_box(self):
        result = rectify(boundary([[0, 0], [4, 0], [3.5, 6], [0, 6]]), PAIRS)
        self.assertEqual(result["status"], "refused")

    def test_weak_plane_support_refused(self):
        paired = {**PAIRS, "opposite_pair_alignments": [.98, 1]}
        result = rectify(boundary([[0, 0], [4, 0], [4, 6], [0, 6]]), paired)
        self.assertEqual(result["status"], "refused")


if __name__ == "__main__":
    unittest.main()
