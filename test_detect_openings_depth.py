import unittest

import numpy as np

from detect_openings_depth import _supported_bounds, matching_axis_hypothesis


class MatchingAxisHypothesisTests(unittest.TestCase):
    def test_near_duplicate_tilted_candidate_does_not_make_pair_ambiguous(self):
        cands = [{"axis_world": [0, 0, 1], "bracketing_pair": {"plane_ids": [0, 1]}},
                 {"axis_world": [0, -.127, -.992], "bracketing_pair": {"plane_ids": [2, 3]}},
                 {"axis_world": [1, 0, 0], "bracketing_pair": {"plane_ids": [4, 5]}}]
        self.assertIs(matching_axis_hypothesis(cands, np.array([0, 0, 1.])), cands[0])

    def test_no_close_candidate_or_exact_tie_is_ambiguous(self):
        cands = [{"axis_world": [1, 0, 0]}]
        self.assertIsNone(matching_axis_hypothesis(cands, np.array([0, 0, 1.])))
        tie = [{"axis_world": [0, 0, 1], "bracketing_pair": {"plane_ids": [0, 1]}},
               {"axis_world": [0, 0, -1], "bracketing_pair": {"plane_ids": [2, 3]}}]
        self.assertIsNone(matching_axis_hypothesis(tie, np.array([0, 0, 1.])))


def test_supported_bounds_refine_coarse_component_edges_from_dropout_rays():
    # Three 1 cm cells form the opening in the coarse evidence map. The
    # projected dropout samples show that the actual visible span is 2.1 cm,
    # rather than the 3 cm cell-aligned bounding box.
    component = np.array([[True, True, True]])
    u_min = np.array([[.101, .108, .119]])
    u_max = np.array([[.104, .116, .122]])
    v_min = np.full((1, 3), .50)
    v_max = np.full((1, 3), .51)

    bounds = _supported_bounds(component, u_min, u_max, v_min, v_max,
                               (.10, .13, .50, .51))

    assert bounds == (.101, .122, .50, .51)
    assert bounds[1] - bounds[0] < .03


def test_supported_bounds_ignore_morphology_filled_cells_without_dropout_samples():
    # Morphological closing can fill a cell, but it must not pull the measured
    # interval out to the edge of an unsupported cell.
    component = np.array([[True, True, True]])
    u_min = np.array([[.201, np.inf, .219]])
    u_max = np.array([[.204, -np.inf, .222]])
    v_min = np.array([[.70, np.inf, .70]])
    v_max = np.array([[.71, -np.inf, .71]])

    bounds = _supported_bounds(component, u_min, u_max, v_min, v_max,
                               (.20, .23, .70, .71))

    assert bounds == (.201, .222, .70, .71)


def test_supported_bounds_fall_back_when_component_has_no_raw_dropout_support():
    component = np.ones((1, 2), dtype=bool)
    inf = np.full((1, 2), np.inf)
    neg_inf = np.full((1, 2), -np.inf)
    fallback = (.30, .32, .8, .82)

    assert _supported_bounds(component, inf, neg_inf, inf, neg_inf,
                             fallback) == fallback


