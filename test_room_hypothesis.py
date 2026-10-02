"""Synthetic tests for unscaled rectilinear hypotheses with distractor planes."""

import unittest

import numpy as np

from room_hypothesis import estimate, estimate_cuboid_baseline, estimate_polygon


def _fixture(seed, dimensions, rotation=None, missing=None, furniture=True, outliers=True):
    rng = np.random.default_rng(seed)
    dimensions = np.asarray(dimensions, dtype=float)
    if rotation is None:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rotation = q
    center = rng.normal(size=3) * 2
    candidates = []
    surface_points = []
    n_surface = 320
    for axis_index in range(3):
        axis = rotation[:, axis_index]
        others = [i for i in range(3) if i != axis_index]
        for side, coordinate in enumerate((-dimensions[axis_index] / 2, dimensions[axis_index] / 2)):
            if missing == (axis_index, side):
                continue
            local = np.zeros((n_surface, 3))
            local[:, axis_index] = coordinate
            for other in others:
                local[:, other] = rng.uniform(-dimensions[other] / 2, dimensions[other] / 2, n_surface)
            points = local @ rotation.T + center
            surface_points.append(points)
            plane_center = center + axis * coordinate
            candidates.append({
                "normal": axis.tolist(), "offset": float(-axis @ plane_center),
                "center": plane_center.tolist(), "support_count": n_surface,
                "observed_bounds_world": np.percentile(points, [2, 98], axis=0).tolist(),
                "semantic_class": "unknown",
            })
    cloud = np.concatenate(surface_points)
    if furniture:
        # Supported interior parallel surfaces should lose to the enclosing pair.
        axis = rotation[:, 0]
        for coordinate in (-dimensions[0] * .12, dimensions[0] * .12):
            point = center + axis * coordinate
            candidates.append({"normal": axis.tolist(), "offset": float(-axis @ point),
                              "center": point.tolist(), "support_count": 55,
                              "observed_bounds_world": [point.tolist(), point.tolist()]})
    if outliers:
        # A small cloud of finite outliers should not set the box dimensions.
        cloud = np.concatenate([cloud, center + rng.normal(size=(35, 3)) * dimensions.max() * 3])
    return {"status": "plane_candidates_only", "units": "arbitrary", "planes": candidates}, cloud, dimensions


class RoomHypothesisTests(unittest.TestCase):
    def test_rotated_boxes_recover_ratios_with_furniture_and_outliers(self):
        for seed, dims in ((13, (3.2, 5.7, 2.6)), (91, (8.0, 3.4, 4.5)), (207, (2.1, 2.8, 6.3))):
            with self.subTest(seed=seed):
                data, points, expected = _fixture(seed, dims)
                result = estimate(data, points)
                self.assertEqual(result["status"], "complete_box_hypothesis")
                self.assertIsNotNone(result["corners"])
                actual = np.sort(np.asarray(result["dimensions"]) / max(result["dimensions"]))
                target = np.sort(expected / expected.max())
                np.testing.assert_allclose(actual, target, atol=.015, rtol=.02)
                self.assertEqual(result["units"], "arbitrary")
                self.assertIsNone(result["metric_scale_source"])
                self.assertTrue(all(len(axis["surface_pair"]["planes"]) == 2 for axis in result["axes"]))

    def test_missing_opposing_surface_stays_incomplete(self):
        data, points, _ = _fixture(42, (3.5, 6.0, 2.8), missing=(1, 1), furniture=False)
        result = estimate(data, points)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["corners"])
        self.assertIsNone(result["dimension_ratios_to_longest"])
        self.assertEqual(sum(v is None for v in result["dimensions"]), 1)
        self.assertTrue(result["missing_surfaces"])

    def test_insufficient_cloud_is_refused(self):
        data, _, _ = _fixture(8, (4.0, 5.0, 3.0))
        result = estimate(data, np.zeros((120, 3)))
        self.assertEqual(result["status"], "insufficient_geometry")
        self.assertIsNone(result["corners"])
        self.assertEqual(result["missing_surfaces"], ["all"])

    def test_translation_equivariance_with_noisy_plane_normals(self):
        data, points, _ = _fixture(771, (3.1, 5.2, 2.7), furniture=False, outliers=True)
        rng = np.random.default_rng(991)
        noisy = {**data, "planes": []}
        for source in data["planes"]:
            plane = dict(source)
            normal = np.asarray(plane["normal"], dtype=float)
            normal = normal + rng.normal(size=3) * .018
            normal /= np.linalg.norm(normal)
            plane_center = np.asarray(plane["center"], dtype=float)
            plane["normal"] = normal.tolist()
            plane["offset"] = float(-normal @ plane_center)
            noisy["planes"].append(plane)
        before = estimate(noisy, points)
        shift = np.array([137.25, -82.5, 41.75])
        translated = {**noisy, "planes": []}
        for source in noisy["planes"]:
            plane = dict(source)
            normal = np.asarray(plane["normal"])
            plane["center"] = (np.asarray(plane["center"]) + shift).tolist()
            plane["offset"] = float(plane["offset"] - normal @ shift)
            plane["observed_bounds_world"] = (np.asarray(plane["observed_bounds_world"]) + shift).tolist()
            translated["planes"].append(plane)
        after = estimate(translated, points + shift)
        self.assertEqual(before["status"], "complete_box_hypothesis")
        self.assertEqual(after["status"], "complete_box_hypothesis")
        np.testing.assert_allclose(before["dimensions"], after["dimensions"], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(before["corners"]) + shift,
                                   np.asarray(after["corners"]), atol=1e-7, rtol=1e-8)


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


L_SHAPE = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
PENTAGON = [(0, 0), (5, 0), (6.5, 3), (2.5, 5.5), (-1, 3)]


def _polygon_room(vertices, height=2.6, seed=0, noise=.008, missing=(), gaps=(), furniture=(), rotation=None,
                  shift=(0., 0., 0.), density=130):
    """Synthetic RGB-derived-like cloud and plane candidates for an arbitrary CCW polygon.

    gaps: (edge, start_fraction, end_fraction) of a wall with no support (occlusion).
    furniture: dict(a=(x, y), b=(x, y), height=h) vertical faces seen in front of the walls.
    """
    rng = np.random.default_rng(seed)
    R = _rotation(seed + 5) if rotation is None else rotation
    verts = np.asarray(vertices, dtype=float); n = len(verts)
    local, planes, planes_at = [], [], []

    def add_plane(a, b, pts, kind):
        d = (b - a) / np.linalg.norm(b - a); m = np.array([d[1], -d[0]])
        normal = R @ np.array([m[0], m[1], 0.]); anchor = R @ np.array([a[0], a[1], 0.]) + np.asarray(shift)
        planes.append({"normal": normal.tolist(), "offset": float(-normal @ anchor), "support_count": len(pts),
                       "kind": kind})

    for k in range(n):
        a, b = verts[k], verts[(k + 1) % n]
        length = np.linalg.norm(b - a)
        if k in missing:
            continue
        count = int(length * density); s = rng.uniform(0, 1, count); z = rng.uniform(0, height, count)
        for e, lo, hi in gaps:
            if e == k:
                keep = (s < lo) | (s > hi); s, z = s[keep], z[keep]
        pts = np.c_[a + s[:, None] * (b - a), z]
        planes_at.append((len(local), len(planes))); local.append(pts); add_plane(a, b, pts, f"wall-{k}")
    for z0 in (0., height):  # floor and ceiling fill the whole footprint, including any missing wall's side
        lo, hi = verts.min(axis=0), verts.max(axis=0); cloud = []
        while len(cloud) < 900:
            q = rng.uniform(lo, hi, 2)
            if _point_in(verts, q):
                cloud.append(q)
        local.append(np.c_[np.array(cloud), np.full(len(cloud), z0)])
    for f, item in enumerate(furniture):
        a, b = np.asarray(item["a"], float), np.asarray(item["b"], float)
        count = int(np.linalg.norm(b - a) * density * 6); s = rng.uniform(0, 1, count); z = rng.uniform(0, item["height"], count)
        pts = np.c_[a + s[:, None] * (b - a), z]
        planes_at.append((len(local), len(planes))); local.append(pts); add_plane(a, b, pts, f"furniture-{f}")
    pts = np.concatenate(local); pts[:, :3] += rng.normal(size=pts.shape) * noise
    world = pts @ R.T + np.asarray(shift)
    world = np.concatenate([world, np.median(world, axis=0) + rng.normal(size=(25, 3)) * 30])  # stray outliers
    sizes = np.cumsum([0] + [len(c) for c in local])
    for block, plane_index in planes_at:  # noisy normals tilted about the support centroid, like fitted candidates
        plane = planes[plane_index]; centroid = world[sizes[block]:sizes[block + 1]].mean(axis=0)
        normal = np.asarray(plane["normal"]) + rng.normal(size=3) * .01; normal /= np.linalg.norm(normal)
        plane["normal"] = normal.tolist(); plane["offset"] = float(-normal @ centroid)
    return {"planes": planes}, world, R[:, 2], verts


def _point_in(poly, q):
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > q[1]) != (y2 > q[1]) and q[0] < (x2 - x1) * (q[1] - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _edge_lengths(vertices):
    v = np.asarray(vertices, dtype=float)
    return np.sort(np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1))


class PolygonHypothesisTests(unittest.TestCase):
    def test_non_rectangular_layouts_are_not_forced_into_a_box(self):
        for name, layout, seed in (("L", L_SHAPE, 3), ("pentagon", PENTAGON, 9)):
            with self.subTest(name):
                data, points, up, truth = _polygon_room(layout, seed=seed)
                result = estimate_polygon(data, points, up=up)
                self.assertEqual(result["status"], "complete_polygon_hypothesis", result["incomplete_boundaries"])
                self.assertEqual(len(result["vertices_xy"]), len(layout))
                self.assertEqual(len(result["edges"]), len(layout))
                np.testing.assert_allclose(_edge_lengths(result["vertices_xy"]), _edge_lengths(truth), rtol=.03)
                area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(truth, np.roll(truth, -1, axis=0))))
                self.assertAlmostEqual(result["area"] / area, 1.0, delta=.03)
                self.assertTrue(all(e["label"] in ("observed", "inferred") for e in result["edges"]))
                self.assertTrue(result["shape_gate_eligible"])
                self.assertIsNone(result["metric_scale_source"])

    def test_rigid_translation_and_rotation_do_not_change_lengths(self):
        data, points, up, _ = _polygon_room(L_SHAPE, seed=3)
        base = estimate_polygon(data, points, up=up)
        data2, points2, up2, _ = _polygon_room(L_SHAPE, seed=3, shift=(310.5, -87.25, 40.0), rotation=_rotation(77))
        moved = estimate_polygon(data2, points2, up=up2)
        self.assertEqual(base["status"], "complete_polygon_hypothesis")
        self.assertEqual(moved["status"], "complete_polygon_hypothesis")
        np.testing.assert_allclose(_edge_lengths(base["vertices_xy"]), _edge_lengths(moved["vertices_xy"]), rtol=.02)
        # Pure translation of the very same cloud: identical result up to numerical noise.
        shift = np.array([137.25, -82.5, 41.75])
        shifted_data = {"planes": [dict(p, offset=float(p["offset"] - np.asarray(p["normal"]) @ shift)) for p in data["planes"]]}
        shifted = estimate_polygon(shifted_data, points + shift, up=up)
        np.testing.assert_allclose(base["vertices_xy"], shifted["vertices_xy"], atol=1e-6)

    def test_missing_wall_is_incomplete_not_fabricated(self):
        for name, layout, missing in (("rectangle", [(0, 0), (6, 0), (6, 4), (0, 4)], (3,)),
                                      ("L", L_SHAPE, (4,)), ("pentagon", PENTAGON, (1,))):
            with self.subTest(name):
                data, points, up, _ = _polygon_room(layout, seed=21, missing=missing)
                result = estimate_polygon(data, points, up=up)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["vertices_xy"]); self.assertEqual(result["edges"], [])
                self.assertFalse(result["shape_gate_eligible"])
                self.assertTrue(result["incomplete_boundaries"])
                self.assertGreaterEqual(len(result["observed_wall_segments"]), 3)  # observed walls are still reported

    def test_large_unobserved_stretch_at_a_corner_is_incomplete(self):
        data, points, up, _ = _polygon_room([(0, 0), (6, 0), (6, 4), (0, 4)], seed=5, gaps=((0, .5, 1.0),))
        result = estimate_polygon(data, points, up=up)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["vertices_xy"])

    def test_small_occlusions_yield_labelled_inferred_edges(self):
        data, points, up, _ = _polygon_room([(0, 0), (6, 0), (6, 4), (0, 4)], seed=5, gaps=((0, .25, .85),))
        result = estimate_polygon(data, points, up=up)
        self.assertEqual(result["status"], "complete_polygon_hypothesis", result["incomplete_boundaries"])
        labels = {e["plane_index"]: e["label"] for e in result["edges"]}
        self.assertEqual(sorted(labels.values()), ["inferred", "observed", "observed", "observed"])
        inferred = [e for e in result["edges"] if e["label"] == "inferred"][0]
        self.assertLess(inferred["observed_fraction"], .6)
        self.assertEqual(result["inferred_edge_count"], 1)

    def test_furniture_never_becomes_a_wall_from_support_alone(self):
        rect = [(0, 0), (6, 0), (6, 4), (0, 4)]
        furniture = [dict(a=(1.5, 1.2), b=(3.0, 1.2), height=.9),     # low, free-standing, dense
                     dict(a=(4.0, 2.8), b=(5.5, 2.8), height=1.8),    # wardrobe, free-standing
                     dict(a=(.5, 3.65), b=(2.5, 3.65), height=.9)]    # against the north wall
        data, points, up, _ = _polygon_room(rect, seed=14, furniture=furniture)
        result = estimate_polygon(data, points, up=up)
        walls = {p["kind"]: i for i, p in enumerate(data["planes"])}
        self.assertEqual(result["status"], "complete_polygon_hypothesis", result["incomplete_boundaries"])
        used = {e["plane_index"] for e in result["edges"]}
        self.assertTrue(all(data["planes"][i]["kind"].startswith("wall") for i in used))
        self.assertEqual(len(used), 4)
        rejected = {r["plane_index"]: r["reason"] for r in result["rejected_candidates"]}
        for kind, index in walls.items():
            if kind.startswith("furniture"):
                self.assertIn(index, rejected, kind)
        # Dense furniture outweighs the walls yet the room with only two walls stays incomplete.
        data, points, up, _ = _polygon_room(rect, seed=14, missing=(1, 3), furniture=furniture)
        result = estimate_polygon(data, points, up=up)
        self.assertEqual(result["status"], "incomplete")
        self.assertGreater(max(p["support_count"] for p in data["planes"] if p["kind"].startswith("furniture")),
                           max(p["support_count"] for p in data["planes"] if p["kind"].startswith("wall")))

    def test_requires_declared_up_and_enough_geometry(self):
        data, points, up, _ = _polygon_room(L_SHAPE, seed=3)
        result = estimate_polygon(data, points)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["incomplete_boundaries"], [{"reason": "up_axis_unavailable"}])
        self.assertEqual(estimate_polygon(data, points[:100], up=up)["status"], "insufficient_geometry")

    def test_cuboid_baseline_is_a_labelled_diagnostic(self):
        data, points, _ = _fixture(13, (3.2, 5.7, 2.6))
        baseline = estimate_cuboid_baseline(data, points)
        self.assertEqual(baseline["status"], "complete_box_hypothesis")
        self.assertEqual(baseline["role"], "diagnostic_baseline_only")
        self.assertIs(estimate_cuboid_baseline, estimate)

CONCAVE = {
    "U": [(0, 0), (8, 0), (8, 5), (5.5, 5), (5.5, 2), (2.5, 2), (2.5, 5), (0, 5)],
    "C": [(0, 0), (8, 0), (8, 2), (2, 2), (2, 4), (8, 4), (8, 6), (0, 6)],           # collinear walls either side of a notch
    "S": [(0, 0), (4, 0), (4, 2), (6, 2), (6, 4), (2, 4), (2, 2), (0, 2)],           # collinear walls facing opposite ways
    "T": [(0, 0), (9, 0), (9, 2), (6, 2), (6, 6), (3, 6), (3, 2), (0, 2)],
}


def _rigid_cyclic_error(estimated, truth):
    """Smallest vertex error after one rigid 2D fit, over every cyclic start: wrong adjacency or order cannot pass."""
    est, ref = np.asarray(estimated, float), np.asarray(truth, float)
    best = np.inf
    for shift in range(len(ref)):
        r = np.roll(ref, -shift, axis=0)
        ce, cr = est.mean(0), r.mean(0)
        u, _s, vt = np.linalg.svd((est - ce).T @ (r - cr))
        rot = vt.T @ np.diag([1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        best = min(best, float(np.abs((est - ce) @ rot.T + cr - r).max()))
    return best


class BoundaryOrderTests(unittest.TestCase):
    def _complete(self, outline, seed, **kw):
        data, points, up, truth = _polygon_room(outline, seed=seed, **kw)
        result = estimate_polygon(data, points, up=up)
        self.assertEqual(result["status"], "complete_polygon_hypothesis", (outline, seed, result["incomplete_boundaries"]))
        self.assertEqual(len(result["vertices_xy"]), len(outline))
        self.assertLess(_rigid_cyclic_error(result["vertices_xy"], truth), .15, "vertex order / adjacency differs from the layout")
        for edge in result["edges"]:   # consecutive edges share exactly one corner
            nxt = result["edges"][(edge["index"] + 1) % len(result["edges"])]
            np.testing.assert_allclose(edge["end_xy"], nxt["start_xy"], atol=1e-9)
        self.assertIn("boundary_search", result)
        return result

    def test_task_reproduction_l_prism_that_angle_sorting_got_wrong(self):
        rng = np.random.default_rng(0); w, l = rng.uniform(4, 8, 2); a, b = w * rng.uniform(.2, .75), l * rng.uniform(.2, .75)
        self._complete([(0, 0), (w, 0), (w, b), (a, b), (a, l), (0, l)], seed=30)

    def test_randomized_l_proportions_complete_with_correct_adjacency(self):
        for seed in range(25):
            rng = np.random.default_rng(seed); w, l = rng.uniform(4, 8, 2); a, b = w * rng.uniform(.2, .75), l * rng.uniform(.2, .75)
            self._complete([(0, 0), (w, 0), (w, b), (a, b), (a, l), (0, l)], seed=30 + seed)

    def test_deeper_non_star_layouts_with_collinear_walls(self):
        for name, outline in CONCAVE.items():
            for seed in range(4):
                with self.subTest(name=name, seed=seed):
                    self._complete(outline, seed, noise=.008 if seed % 2 else .02)

    def test_collinear_walls_are_separate_edges_not_one_merged_wall(self):
        result = self._complete(CONCAVE["C"], 1)
        self.assertEqual(len(result["edges"]), 8)
        lengths = sorted(e["length"] for e in result["edges"])
        np.testing.assert_allclose(lengths, sorted(np.linalg.norm(np.roll(np.asarray(CONCAVE["C"], float), -1, 0) - np.asarray(CONCAVE["C"], float), axis=1)), rtol=.03)

    def test_missing_walls_in_concave_layouts_stay_incomplete_without_fabricated_corners(self):
        for name, outline in CONCAVE.items():
            for miss in range(len(outline)):
                data, points, up, _ = _polygon_room(outline, seed=5, missing=(miss,))
                result = estimate_polygon(data, points, up=up)
                with self.subTest(name=name, missing=miss):
                    self.assertEqual(result["status"], "incomplete")
                    self.assertIsNone(result["vertices_xy"]); self.assertEqual(result["edges"], [])
                    self.assertTrue(result["incomplete_boundaries"])
                    self.assertGreaterEqual(len(result["observed_wall_segments"]), 3)

    def test_alcove_wall_is_not_mistaken_for_furniture_but_furniture_still_is(self):
        # Alcove walls have outside space (no floor/ceiling) behind them; a table-height box against a wall does not.
        data, points, up, _ = _polygon_room(CONCAVE["C"], seed=2, furniture=[dict(a=(1.0, .5), b=(3.5, .5), height=1.0)])
        result = estimate_polygon(data, points, up=up)
        self.assertEqual(result["status"], "complete_polygon_hypothesis", result["incomplete_boundaries"])
        used = {e["plane_index"] for e in result["edges"]}
        self.assertFalse(any(data["planes"][i]["kind"].startswith("furniture") for i in used))
        self.assertTrue(any(data["planes"][r["plane_index"]]["kind"].startswith("furniture") for r in result["rejected_candidates"]))

    def test_rigid_motion_does_not_change_the_boundary_order(self):
        for name in ("L", "S"):
            outline = L_SHAPE if name == "L" else CONCAVE["S"]
            a = estimate_polygon(*_polygon_room(outline, seed=9)[:2], up=_polygon_room(outline, seed=9)[2])
            moved = _polygon_room(outline, seed=9, shift=(300., -90., 20.), rotation=_rotation(41))
            b = estimate_polygon(moved[0], moved[1], up=moved[2])
            self.assertEqual(a["status"], "complete_polygon_hypothesis"); self.assertEqual(b["status"], "complete_polygon_hypothesis")
            np.testing.assert_allclose(sorted(e["length"] for e in a["edges"]), sorted(e["length"] for e in b["edges"]), rtol=.02)
            self.assertLess(_rigid_cyclic_error(a["vertices_xy"], b["vertices_xy"]), .15)


if __name__ == "__main__":
    unittest.main()
