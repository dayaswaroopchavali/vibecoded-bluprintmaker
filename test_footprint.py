"""Tests for the floor/ceiling footprint polygon (synthetic clouds in arbitrary frames)."""

import unittest

import numpy as np

from footprint import estimate_footprint, estimate_footprint_robust
from geometry_orientation import axis_hypotheses
from test_room_hypothesis import CONCAVE, L_SHAPE, PENTAGON, _polygon_room, _rigid_cyclic_error, _rotation

RECT = [(0, 0), (6, 0), (6, 5), (0, 5)]
HEIGHT = 2.6


def _area(poly):
    p = np.asarray(poly, float)
    return 0.5 * abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))


def _room(layout, seed=3, noise=.008, sign=1, shift=(0., 0., 0.), planes_fc=True, densify=4, **kw):
    """Cloud + planes (walls and floor/ceiling) + axis (sign * true up) + truth outline in the e1/e2 frame handedness.

    The fixture's own floor/ceiling is only ~33 points/m2, far sparser than a real DA3 cloud, so `densify` re-samples the
    same room with other seeds (same rotation) and concatenates the clouds; densify=1 is the fixture's native density.
    """
    data, pts, up, truth = _polygon_room(layout, seed=seed, noise=noise, shift=shift, **kw)
    extra = [_polygon_room(layout, seed=seed + 1000 * k, noise=noise, shift=shift, rotation=_rotation(seed + 5), **kw)[1]
             for k in range(1, densify)]
    pts = np.concatenate([pts] + extra)
    planes = list(data["planes"]); ids = None
    if planes_fc:
        ids = (len(planes), len(planes) + 1)
        planes.append({"normal": up.tolist(), "offset": float(-(up @ np.asarray(shift))), "support_count": 900})
        planes.append({"normal": (-up).tolist(), "offset": float(up @ np.asarray(shift) + HEIGHT), "support_count": 900})
    truth = np.asarray(truth, float)
    if sign < 0:                         # flipping the axis mirrors the plan; keep the vertex order counter-clockwise
        truth = (truth * np.array([-1., 1.]))[::-1]
    return {"planes": planes}, pts, sign * up, truth, ids


class FootprintShapeTests(unittest.TestCase):
    def _check(self, name, layout, seed, noise=.008, sign=1, shift=(0., 0., 0.), area_tol=.05, vertex_tol=.35, **kw):
        data, pts, axis, truth, ids = _room(layout, seed, noise, sign, shift, **kw)
        out = estimate_footprint(data, pts, axis, ids)
        self.assertEqual(out["status"], "complete_polygon_hypothesis", (name, out["incomplete_boundaries"]))
        self.assertEqual(out["model"], "floor_ceiling_footprint_polygon_hypothesis")
        self.assertEqual(len(out["vertices_xy"]), len(layout), (name, out["vertices_xy"]))
        self.assertLess(abs(out["area"] / _area(truth) - 1), area_tol, (name, out["area"], _area(truth)))
        self.assertLess(_rigid_cyclic_error(out["vertices_xy"], truth), vertex_tol, name)
        self.assertGreater(out["area"], 0)                                   # counter-clockwise
        self.assertAlmostEqual(out["perimeter"], sum(e["length"] for e in out["edges"]))
        for e in out["edges"]:
            nxt = out["edges"][(e["index"] + 1) % len(out["edges"])]
            np.testing.assert_allclose(e["end_xy"], nxt["start_xy"])
            self.assertIn(e["label"], ("observed", "inferred"))
        self.assertEqual(out["units"], "arbitrary")
        return out

    def test_shapes_two_noise_levels_and_axis_signs(self):
        shapes = (("rect", RECT), ("L", L_SHAPE), ("U", CONCAVE["U"]), ("pentagon", PENTAGON))
        for name, layout in shapes:
            for noise in (.008, .03):
                for sign in (1, -1):
                    with self.subTest(shape=name, noise=noise, sign=sign):
                        self._check(name, layout, seed=11, noise=noise, sign=sign)

    def test_native_sparse_fixture_density_rectangle_and_l(self):
        for name, layout in (("rect", RECT), ("L", L_SHAPE)):
            with self.subTest(name):
                data, pts, axis, truth, ids = _room(layout, 5, noise=.01, densify=1)
                out = estimate_footprint(data, pts, axis, ids)
                self.assertEqual(out["status"], "complete_polygon_hypothesis", out["incomplete_boundaries"])
                self.assertLess(abs(out["area"] / _area(truth) - 1), .05)
                self.assertLess(_rigid_cyclic_error(out["vertices_xy"], truth), .4)

    def test_translated_far_from_origin_and_other_frames(self):
        for seed in (2, 7, 19):
            with self.subTest(seed=seed):
                self._check("L", L_SHAPE, seed, shift=(1000., -450., 77.), sign=-1 if seed % 2 else 1)

    def test_scale_free(self):
        data, pts, axis, truth, ids = _room(PENTAGON, 5)
        a = estimate_footprint(data, pts, axis, ids)
        k = 37.5
        planes = [{**p, "offset": p["offset"] * k} for p in data["planes"]]
        b = estimate_footprint({"planes": planes}, pts * k, axis, ids)
        self.assertEqual(b["status"], "complete_polygon_hypothesis")
        self.assertEqual(len(a["vertices_xy"]), len(b["vertices_xy"]))
        self.assertLess(abs(b["area"] / (k * k * a["area"]) - 1), .02)

    def test_furniture_holes_and_stray_outliers(self):
        data, pts, axis, truth, ids = _room(L_SHAPE, 4, noise=.01)
        # carve furniture-sized holes (about 1 m wide) out of the floor and ceiling interiors in the room frame
        local = (pts - np.array([0., 0., 0.])) @ _frame_of(L_SHAPE, 4)
        floor_like = (np.abs(local[:, 2]) < .05) | (np.abs(local[:, 2] - HEIGHT) < .05)
        hole = floor_like & (np.abs(local[:, 0] - 1.5) < .6) & (np.abs(local[:, 1] - 1.5) < .6)
        hole |= floor_like & (np.abs(local[:, 0] - 4.5) < .5) & (np.abs(local[:, 1] - 1.0) < .4)
        self.assertGreater(hole.sum(), 50)
        out = estimate_footprint(data, pts[~hole], axis, ids)
        self.assertEqual(out["status"], "complete_polygon_hypothesis", out["incomplete_boundaries"])
        self.assertEqual(len(out["vertices_xy"]), len(L_SHAPE))
        self.assertLess(abs(out["area"] / _area(truth) - 1), .06)
        self.assertLess(_rigid_cyclic_error(out["vertices_xy"], truth), .4)

    def test_deterministic(self):
        data, pts, axis, truth, ids = _room(CONCAVE["U"], 8)
        a = estimate_footprint(data, pts, axis, ids); b = estimate_footprint(data, pts.copy(), axis, ids)
        self.assertEqual(a, b)

    def test_integrates_with_axis_hypotheses_and_auto_bracketing(self):
        data, pts, axis, truth, ids = _room(L_SHAPE, 3)
        hyp = axis_hypotheses(data, pts)
        match = [h for h in hyp["hypotheses"] if abs(float(np.dot(h["axis_world"], axis))) > .99 and h["bracketing_pair"]]
        self.assertTrue(match, "the structural axis should appear among the hypotheses")
        by_pair = estimate_footprint(data, pts, match[0]["axis_world"], match[0]["bracketing_pair"]["plane_ids"])
        auto = estimate_footprint(data, pts, match[0]["axis_world"])
        self.assertEqual(by_pair["status"], "complete_polygon_hypothesis", by_pair["incomplete_boundaries"])
        self.assertEqual(auto["status"], "complete_polygon_hypothesis", auto["incomplete_boundaries"])
        self.assertLess(abs(auto["area"] / _area(truth) - 1), .05)


def _frame_of(layout, seed):
    return _rotation(seed + 5)


class FootprintRobustTests(unittest.TestCase):
    def test_consensus_over_scales_gives_the_true_corner_count_on_noisy_clouds(self):
        for name, layout in (("rect", RECT), ("L", L_SHAPE), ("U", CONCAVE["U"])):
            for noise in (.01, .03):
                with self.subTest(shape=name, noise=noise):
                    data, pts, axis, truth, ids = _room(layout, 7, noise)
                    out = estimate_footprint_robust(data, pts, axis, ids)
                    self.assertEqual(out["status"], "complete_polygon_hypothesis", out.get("incomplete_boundaries"))
                    self.assertEqual(len(out["edges"]), len(layout))
                    self.assertLess(abs(out["area"] / _area(truth) - 1), .06)
                    self.assertGreaterEqual(out["consensus"]["scales_complete"] * 2, out["consensus"]["scales_tried"])

    def test_unstable_footprints_are_reported_incomplete_never_averaged_into_a_polygon(self):
        data, pts, axis, truth, ids = _room(RECT, 7, .01)
        keep = pts[(pts @ axis) < np.median(pts @ axis)]                     # only one of the two surfaces (and no walls) remain
        out = estimate_footprint_robust(data, keep, axis, ids)
        import json
        self.assertTrue(out["status"] != "complete_polygon_hypothesis" or "single_surface_observed" in json.dumps(out))   # flagged or refused
        none = estimate_footprint_robust({"planes": []}, pts, axis, None)
        self.assertEqual(none["status"], "incomplete")

class FootprintEvidenceTests(unittest.TestCase):
    def test_edges_without_wall_evidence_are_inferred_not_observed(self):
        full = _room(L_SHAPE, 3)
        data, pts, axis, truth, ids = full
        with_walls = estimate_footprint(data, pts, axis, ids)
        self.assertEqual(with_walls["observed_fraction"], 1.0, [e["wall_evidence_fraction"] for e in with_walls["edges"]])
        self.assertTrue(all(e["label"] == "observed" for e in with_walls["edges"]))
        # remove the wall points entirely: floor/ceiling still define the outline, but nothing is observed
        local = pts @ _frame_of(L_SHAPE, 3)
        keep = (local[:, 2] < .1) | (local[:, 2] > HEIGHT - .1) | (np.abs(local[:, 2]) > 20)
        floors_only = estimate_footprint(data, pts[keep], axis, ids)
        self.assertEqual(floors_only["status"], "complete_polygon_hypothesis", floors_only["incomplete_boundaries"])
        self.assertEqual(len(floors_only["vertices_xy"]), len(L_SHAPE))
        self.assertTrue(all(e["label"] == "inferred" for e in floors_only["edges"]))
        self.assertEqual(floors_only["observed_fraction"], 0.0)

    def test_one_wall_with_an_occluded_gap_is_partly_observed(self):
        data, pts, axis, truth, ids = _room(RECT, 6, gaps=((0, .05, .95),))
        out = estimate_footprint(data, pts, axis, ids)
        self.assertEqual(out["status"], "complete_polygon_hypothesis", out["incomplete_boundaries"])
        labels = sorted(e["label"] for e in out["edges"])
        self.assertEqual(labels, ["inferred", "observed", "observed", "observed"])


class FootprintIncompleteTests(unittest.TestCase):
    def test_missing_quarter_of_floor_and_ceiling_with_walls_beyond_is_incomplete(self):
        data, pts, axis, truth, ids = _room(RECT, 3)
        local = pts @ _frame_of(RECT, 3)
        on_fc = (np.abs(local[:, 2]) < .1) | (np.abs(local[:, 2] - HEIGHT) < .1)
        drop = on_fc & (local[:, 0] < 3.0) & (local[:, 1] < 2.5)
        self.assertGreater(drop.sum(), 100)
        out = estimate_footprint(data, pts[~drop], axis, ids)
        self.assertEqual(out["status"], "incomplete")
        self.assertIsNone(out["vertices_xy"]); self.assertEqual(out["edges"], [])
        self.assertEqual(out["incomplete_boundaries"][0]["reason"],
                         "wall_points_outside_floor_ceiling_footprint_unobserved_room")

    def test_fragmented_footprint_is_incomplete(self):
        data, pts, axis, truth, ids = _room(RECT, 3)
        local = pts @ _frame_of(RECT, 3)
        on_fc = (np.abs(local[:, 2]) < .1) | (np.abs(local[:, 2] - HEIGHT) < .1)
        drop = on_fc & (np.abs(local[:, 0] - 3.0) < 1.0)
        walls = (local[:, 2] > .1) & (local[:, 2] < HEIGHT - .1)        # keep floor/ceiling only, minus a central strip
        out = estimate_footprint(data, pts[~(drop | walls)], axis, ids)
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["incomplete_boundaries"][0]["reason"], "fragmented_footprint")

    def test_only_one_of_floor_and_ceiling_observed_is_allowed_but_flagged(self):
        data, pts, axis, truth, ids = _room(L_SHAPE, 3)
        local = pts @ _frame_of(L_SHAPE, 3)
        keep = ~(np.abs(local[:, 2] - HEIGHT) < .1)           # no ceiling points
        out = estimate_footprint(data, pts[keep], axis, ids)
        self.assertEqual(out["status"], "complete_polygon_hypothesis", out["incomplete_boundaries"])
        self.assertEqual(out["flags"][0]["flag"], "single_surface_observed")
        self.assertLess(abs(out["area"] / _area(truth) - 1), .06)
        both = estimate_footprint(data, pts, axis, ids)
        self.assertEqual(both["flags"], [])

    def test_neither_surface_observed_is_incomplete(self):
        data, pts, axis, truth, ids = _room(L_SHAPE, 3)
        local = pts @ _frame_of(L_SHAPE, 3)
        keep = (local[:, 2] > .3) & (local[:, 2] < HEIGHT - .3)
        out = estimate_footprint(data, pts[keep], axis, ids)
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["incomplete_boundaries"][0]["reason"], "insufficient_floor_ceiling_support")

    def test_nothing_bracketed_is_incomplete(self):
        data, pts, axis, truth, ids = _room(L_SHAPE, 3, planes_fc=False)
        out = estimate_footprint(data, pts, axis)             # only wall planes: no parallel pair to the axis
        self.assertEqual(out["status"], "incomplete")
        self.assertEqual(out["incomplete_boundaries"][0]["reason"], "no_bracketing_pair")
        self.assertIsNone(out["vertices_xy"])
        self.assertEqual(estimate_footprint(data, pts, axis, [0, 1])["incomplete_boundaries"][0]["reason"],
                         "bracketing_planes_unusable")           # two wall planes are not perpendicular to the axis

    def test_bad_inputs(self):
        data, pts, axis, truth, ids = _room(L_SHAPE, 3)
        self.assertEqual(estimate_footprint(data, pts[:20], axis, ids)["status"], "insufficient_geometry")
        self.assertEqual(estimate_footprint(data, pts, None, ids)["incomplete_boundaries"][0]["reason"], "axis_unavailable")
        self.assertIsNone(estimate_footprint(data, pts, np.zeros(3), ids)["vertices_xy"])


if __name__ == "__main__":
    unittest.main()


