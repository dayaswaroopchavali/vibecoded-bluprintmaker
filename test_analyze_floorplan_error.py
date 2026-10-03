"""Synthetic tests for analyze_floorplan_error (evaluation-only). Real DA3 runs are CLI-only and not exercised here."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_floorplan_error as F
import audit_camera_trajectory as A
from test_compare_backends import make_gt_poses, w2c_from_gt, rot_z

ROOM = {"width": 6.0, "length": 5.0, "height": 3.0}


def box_points(room=ROOM, ceiling_offset=0.0, north_shift=0.0, spacing=0.1, wall_top=None):
    w, l, h = room["width"], room["length"], room["height"]
    hc = h + ceiling_offset
    top = hc if wall_top is None else wall_top
    xs, ys = np.arange(-w / 2, w / 2 + 1e-9, spacing), np.arange(-l / 2, l / 2 + 1e-9, spacing)
    zs = np.arange(0, top + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    ceil = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, hc)], 1)
    ax, az = np.meshgrid(xs, zs)
    ay, bz = np.meshgrid(ys, zs)
    north = np.stack([ax.ravel(), np.full(ax.size, l / 2 + north_shift), az.ravel()], 1)
    south = np.stack([ax.ravel(), np.full(ax.size, -l / 2), az.ravel()], 1)
    east = np.stack([np.full(ay.size, w / 2), ay.ravel(), bz.ravel()], 1)
    west = np.stack([np.full(ay.size, -w / 2), ay.ravel(), bz.ravel()], 1)
    return np.vstack([floor, ceil, north, south, east, west])


class Residuals(unittest.TestCase):
    def test_perfect_box_has_zero_residuals_and_full_coverage(self):
        r = F.residual_analysis(box_points(), ROOM)
        for name, s in r["surfaces"].items():
            self.assertAlmostEqual(s["signed_distance_m"]["median"], 0.0, places=6, msg=name)
        self.assertAlmostEqual(sum(s["point_share"] for s in r["surfaces"].values()) + r["unassigned_share"], 1.0, places=9)
        self.assertGreater(r["surfaces"]["north"]["coverage_of_true_length_by_bins"], 0.95)
        self.assertAlmostEqual(r["vertical_extent"]["vertical_error_fraction"], 0.0, places=6)
        self.assertGreater(r["surfaces"]["east"]["height_range_fraction_of_true_height"], 0.9)

    def test_known_ceiling_offset_is_recovered_with_sign(self):
        r = F.residual_analysis(box_points(ceiling_offset=0.2), ROOM)
        # ceiling 0.2 m too HIGH => 0.2 m on the outer side => negative (interior positive)
        self.assertAlmostEqual(r["surfaces"]["ceiling"]["signed_distance_m"]["median"], -0.2, places=6)
        self.assertAlmostEqual(r["surfaces"]["floor"]["signed_distance_m"]["median"], 0.0, places=6)
        self.assertAlmostEqual(r["vertical_extent"]["vertical_error_fraction"], 0.2 / 3.0, places=6)
        self.assertAlmostEqual(r["vertical_extent"]["ceiling_minus_floor_median_m"], 3.2, places=6)

    def test_known_wall_offset_only_moves_that_wall(self):
        r = F.residual_analysis(box_points(north_shift=0.15), ROOM)
        self.assertAlmostEqual(r["surfaces"]["north"]["signed_distance_m"]["median"], -0.15, places=6)
        self.assertAlmostEqual(r["surfaces"]["south"]["signed_distance_m"]["median"], 0.0, places=6)

    def test_outliers_are_unassigned_and_saturation_flags_big_offsets(self):
        pts = np.vstack([box_points(), np.array([[20, 20, 20.0]] * 50)])
        r = F.residual_analysis(pts, ROOM)
        self.assertGreater(r["unassigned_share"], 0.0)
        r2 = F.residual_analysis(box_points(ceiling_offset=0.33), ROOM, window=0.35)
        self.assertGreater(r2["surfaces"]["ceiling"]["window_saturated_share"], 0.9)

    def test_partial_wall_coverage_reported(self):
        pts = box_points()
        keep = ~((np.abs(pts[:, 1] - 2.5) < 1e-6) & (pts[:, 0] > 0.0))     # remove half of the north wall
        r = F.residual_analysis(pts[keep], ROOM)
        self.assertLess(r["surfaces"]["north"]["coverage_of_true_length_by_bins"], 0.6)
        self.assertLess(r["surfaces"]["north"]["along_extent_fraction_of_true_length"], 0.6)


class Anchors(unittest.TestCase):
    TRUE = {"north": 6.0, "south": 6.0, "east": 5.0, "west": 5.0}

    def test_pure_vertical_scale_error(self):
        u = 0.37                                   # arbitrary prediction units
        pred = {n: v * u for n, v in self.TRUE.items()}
        a = F.anchor_diagnostics(pred, 3.0 * 1.07 * u, self.TRUE, 3.0)
        for n, err in a["protocol_1_ceiling_3m"]["wall_relative_error"].items():
            self.assertAlmostEqual(err, 1 / 1.07 - 1, places=9, msg=n)       # uniform -6.5 % under the 3 m protocol
        for an, d in a["protocol_2_single_wall_anchor_oracle"].items():
            for n, err in d["other_wall_relative_error"].items():
                self.assertAlmostEqual(err, 0.0, places=9, msg=f"{an}->{n}")  # one true wall removes the error
            self.assertAlmostEqual(d["implied_height_error_fraction"], 0.07, places=9)

    def test_horizontal_deformation_is_not_removed_by_one_anchor(self):
        pred = {"north": 6.0 * 1.07, "south": 6.0 * 1.07, "east": 5.0, "west": 5.0}   # x stretched, height exact
        a = F.anchor_diagnostics(pred, 3.0, self.TRUE, 3.0)
        p1 = a["protocol_1_ceiling_3m"]["wall_relative_error"]
        self.assertAlmostEqual(p1["north"], 0.07, places=9)
        self.assertAlmostEqual(p1["east"], 0.0, places=9)
        d = a["protocol_2_single_wall_anchor_oracle"]["east"]
        self.assertAlmostEqual(d["other_wall_relative_error"]["north"], 0.07, places=9)   # stays wrong under the anchor
        self.assertAlmostEqual(d["implied_height_error_fraction"], 0.0, places=9)

    def test_verdicts_separate_the_two_cases(self):
        u = 0.37
        vert = F.anchor_diagnostics({n: v * u for n, v in self.TRUE.items()}, 3.0 * 1.07 * u, self.TRUE, 3.0)
        v1 = F.verdict(vert, 0.07, [0.0, 0.001])
        self.assertEqual(v1["status"], "supported")
        horiz = F.anchor_diagnostics({"north": 6.4, "south": 6.4, "east": 5.0, "west": 5.0}, 3.0, self.TRUE, 3.0)
        v2 = F.verdict(horiz, 0.0, [0.0])
        self.assertEqual(v2["status"], "not_supported")
        v3 = F.verdict(horiz, 0.07, [0.0])
        self.assertEqual(v3["status"], "mixed_or_inconclusive")                  # over-estimate present but horizontal deformation too
        v4 = F.verdict(vert, 0.07, [0.2])                                       # polygon shortened a lot
        self.assertNotEqual(v4["status"], "supported")
        v5 = F.verdict(vert, 0.07, [], coverage_limited=True)
        self.assertEqual(v5["status"], "mixed_or_inconclusive")
        ok = F.anchor_diagnostics({n: v for n, v in self.TRUE.items()}, 3.0, self.TRUE, 3.0)
        self.assertEqual(F.verdict(ok, 0.0, [0.0])["status"], "no_material_wall_error")

    def test_unassessed_raster_is_stated(self):
        u = 0.5
        vert = F.anchor_diagnostics({n: v * u for n, v in self.TRUE.items()}, 3.0 * 1.07 * u, self.TRUE, 3.0)
        v = F.verdict(vert, 0.07, [])
        self.assertEqual(v["status"], "supported")
        self.assertIn("NOT assessed", v["statement"])

    def test_underestimated_height_does_not_support_the_bias_hypothesis(self):
        u = 0.5
        a = F.anchor_diagnostics({n: v * u for n, v in self.TRUE.items()}, 3.0 * 0.95 * u, self.TRUE, 3.0)
        self.assertEqual(F.verdict(a, -0.05, [0.0])["status"], "not_supported")

    def test_points_only_anchor_on_synthetic_clouds(self):
        # pure vertical overestimate with full coverage
        r = F.residual_analysis(box_points(ceiling_offset=0.21), ROOM)
        a = F.points_only_anchor(r, ROOM)
        self.assertEqual(a["status"], "ok")
        self.assertFalse(a["coverage_limited"])
        for d in a["protocol_2_single_wall_anchor_oracle"].values():
            for e in d["other_wall_relative_error"].values():
                self.assertLess(abs(e), 0.01)
        self.assertEqual(F.verdict(a, r["vertical_extent"]["vertical_error_fraction"], [])["status"], "supported")
        # same vertical error but only 70 % of the floor/ceiling x range seen => coverage-limited and flagged
        pts = box_points(ceiling_offset=0.21)
        pts = pts[~(((pts[:, 2] < 0.01) | (pts[:, 2] > 3.2)) & (pts[:, 0] > 1.0))]
        r2 = F.residual_analysis(pts, ROOM)
        a2 = F.points_only_anchor(r2, ROOM)
        self.assertTrue(a2["coverage_limited"])
        self.assertEqual(F.verdict(a2, r2["vertical_extent"]["vertical_error_fraction"], [], coverage_limited=True)["status"], "mixed_or_inconclusive")


class FootprintVsSource(unittest.TestCase):
    def test_polygon_shorter_than_raw_support_is_reported(self):
        gx, gy = np.meshgrid(np.arange(0, 10.0001, 0.05), np.arange(0, 6.0001, 0.05))
        S = np.stack([gx.ravel(), gy.ravel()], 1)
        poly = np.array([[0.3, 0], [9.7, 0], [9.7, 6], [0.3, 6]], float)   # CCW, left/right walls pulled in by 0.3 m
        rows = F.edge_source_comparison(poly, S, tol=0.12, cell=0.1, closing_cells=1)
        bottom = rows[0]
        self.assertAlmostEqual(bottom["polygon_edge_length"], 9.4)
        self.assertAlmostEqual(bottom["raw_support_extent"]["min_max"], 10.0, places=1)
        self.assertAlmostEqual(bottom["polygon_shorter_than_raw_support_by"], 0.6, delta=0.1)
        self.assertGreater(bottom["polygon_shorter_than_raw_support_fraction_of_edge"], 0.05)
        # right edge (x=9.7): contour sits 0.3 m INSIDE the outermost points
        right = rows[1]
        self.assertAlmostEqual(right["contour_vs_point_extreme"]["contour_inside_extreme_by"], 0.3, delta=0.06)
        self.assertGreater(bottom["raster_closed_filled_largest_extent"], 9.8)   # the raster itself did not shorten

    def test_exact_polygon_has_no_shortening(self):
        gx, gy = np.meshgrid(np.arange(0, 10.0001, 0.05), np.arange(0, 6.0001, 0.05))
        S = np.stack([gx.ravel(), gy.ravel()], 1)
        poly = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], float)
        rows = F.edge_source_comparison(poly, S, tol=0.12, cell=0.1, closing_cells=1)
        for r in rows:
            self.assertLess(abs(r["polygon_shorter_than_raw_support_fraction_of_edge"]), 0.03)


class AlignmentAndMatching(unittest.TestCase):
    def test_fixed_alignment_maps_prediction_points_back_to_truth(self):
        gt = make_gt_poses(30)
        R0 = rot_z(25)
        w2c = w2c_from_gt(gt, scale=2.5, world_rot=R0, shift=(1.0, 2.0, -0.5))
        al = F.fixed_alignment(w2c, list(range(30)), gt)
        pred_pts = 2.5 * (box_points(spacing=0.5) @ R0.T) + np.array([1.0, 2.0, -0.5])   # truth points as DA3 would see them
        back = F.to_truth_frame(pred_pts, al)
        np.testing.assert_allclose(back, box_points(spacing=0.5), atol=1e-6)
        self.assertLess(al["centre_rmse_fraction_of_full_gt_extent"], 1e-6)

    def test_match_walls_oracle_correspondence(self):
        poly = np.array([[-3.1, -2.4], [3.0, -2.5], [3.1, 2.6], [-3.0, 2.5]], float)   # CCW, slightly off
        m = F.match_walls(poly, ROOM)
        self.assertEqual(m, {"south": 0, "east": 1, "north": 2, "west": 3})
        shifted = np.roll(poly, -1, axis=0)
        m2 = F.match_walls(shifted, ROOM)
        self.assertEqual(m2, {"south": 3, "east": 0, "north": 1, "west": 2})

    def test_match_walls_missing_wall(self):
        tri = np.array([[-3, -2.5], [3, -2.5], [3, 2.5]], float)
        m = F.match_walls(tri, ROOM)
        self.assertIn("south", m)
        self.assertNotIn("west", m)

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileExistsError):
                F.analyze(Path(t), Path(t) / "x.npz", Path(t), Path(t))

    def test_room_from_scene(self):
        self.assertEqual(F.room_from_scene({"width": 6, "length": 5, "height": 3}), ROOM)
        with self.assertRaises(ValueError):
            F.room_from_scene({"width": 6, "length": -5, "height": 3})


class EndToEndSynthetic(unittest.TestCase):
    """Real extract_planes + footprint stages on a noisy synthetic box, identity alignment (prediction frame == truth frame)."""

    def run_case(self, pts):
        import footprint
        import extract_planes
        planes = extract_planes.extract(pts)
        fp = footprint.estimate_footprint_robust(planes, pts, [0, 0, 1.0])
        self.assertEqual(fp["status"], "complete_polygon_hypothesis", fp.get("incomplete_boundaries"))
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "footprint_axis_hypothesis_0.json").write_text(json.dumps(fp))
            (t / "height_axis_hypothesis_0.json").write_text(json.dumps({"status": "estimated_unscaled_separation",
                                                                         "separation_unscaled": fp["floor_ceiling_separation_unscaled"]}))
            al = {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)}
            res = F.residual_analysis(pts, ROOM)
            return F.analyze_hypothesis(0, t, planes, pts, al, ROOM, res)

    def test_vertical_overestimate_is_supported(self):
        rng = np.random.default_rng(0)
        pts = box_points(ceiling_offset=0.21, spacing=0.06)
        pts = pts + rng.normal(0, 0.01, pts.shape)
        h = self.run_case(pts)
        self.assertEqual(h["status"], "analysed")
        self.assertEqual(set(h["wall_to_edge_oracle_correspondence"]), {"north", "east", "south", "west"})
        for v in h["fixed_alignment_wall_error"].values():
            self.assertLess(abs(v["relative_error"]), 0.01)
        self.assertEqual(h["verdict"]["status"], "supported")
        self.assertAlmostEqual(h["height_estimate_vs_truth_under_fixed_similarity"]["estimate_height_ratio"], 3.21 / 3.0, delta=0.02)

    def test_horizontal_stretch_is_not_blamed_on_height(self):
        rng = np.random.default_rng(1)
        pts = box_points(spacing=0.06)
        pts[:, 0] *= 1.07                                  # x stretched; height exact
        pts = pts + rng.normal(0, 0.01, pts.shape)
        h = self.run_case(pts)
        self.assertEqual(h["status"], "analysed")
        self.assertAlmostEqual(h["fixed_alignment_wall_error"]["north"]["relative_error"], 0.07, delta=0.015)
        self.assertNotEqual(h["verdict"]["status"], "supported")

    def test_axis_not_vertical_is_reported_not_analysed(self):
        import footprint
        import extract_planes
        pts = box_points(spacing=0.06)
        fp = footprint.estimate_footprint_robust(extract_planes.extract(pts), pts, [0, 0, 1.0])
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "footprint_axis_hypothesis_0.json").write_text(json.dumps(fp))
            rot90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])          # fixed alignment tilts the axis by 90 deg
            al = {"scale": 1.0, "rotation": rot90, "translation": np.zeros(3)}
            h = F.analyze_hypothesis(0, t, {"planes": []}, pts, al, ROOM, {})
        self.assertEqual(h["status"], "axis_not_true_vertical")


if __name__ == "__main__":
    unittest.main()
