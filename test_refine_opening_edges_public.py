"""Focused tests for truth-free opening-edge consensus."""
import unittest

import refine_opening_edges_public as refine


class ConsensusIntervalTests(unittest.TestCase):
    def test_trimmed_frame_consensus_rejects_one_extreme_frame(self):
        observations = [(1.00, 2.00)] * 19 + [(0.90, 2.10)]
        interval = refine.consensus_interval(observations, trim_fraction=.05)
        self.assertIsNotNone(interval)
        self.assertGreaterEqual(interval[0], .995)
        self.assertLessEqual(interval[1], 2.005)

    def test_insufficient_frames_preserve_original_candidate(self):
        self.assertIsNone(refine.consensus_interval([(1.0, 2.0)] * 2))

    def test_bad_quantile_is_rejected(self):
        with self.assertRaises(ValueError):
            refine.consensus_interval([(1.0, 2.0)] * 5, trim_fraction=.25)

    def test_negative_quantile_is_rejected(self):
        with self.assertRaises(ValueError):
            refine.consensus_interval([(1.0, 2.0)] * 5, trim_fraction=-.01)


if __name__ == "__main__":
    unittest.main()
