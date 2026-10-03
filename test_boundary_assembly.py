import ast
import json
import unittest
from pathlib import Path

import numpy as np

import boundary_assembly as ba
from boundary_assembly import FreeSpaceScorer, adapt_ray_scorer, assemble_boundary
from room_hypothesis import _inside
from test_room_hypothesis import _rigid_cyclic_error

RECT = [(0, 0), (10, 0), (10, 6), (0, 6)]
SHAPES = {
    "rect": RECT,
    "L": [(0, 0), (8, 0), (8, 3), (4, 3), (4, 7), (0, 7)],
    "U": [(0, 0), (8, 0), (8, 5), (5.5, 5), (5.5, 2), (2.5, 2), (2.5, 5), (0, 5)],
    "C": [(0, 0), (8, 0), (8, 2), (2, 2), (2, 4), (8, 4), (8, 6), (0, 6)],
    "S": [(0, 0), (4, 0), (4, 2), (6, 2), (6, 4), (2, 4), (2, 2), (0, 2)],
    "T": [(0, 0), (9, 0), (9, 2), (6, 2), (6, 6), (3, 6), (3, 2), (0, 2)],
    "pentagon": [(0, 0), (8, 0), (10, 4), (5, 7), (-1, 5)],
}


def _raycast(cam, direction, segments):
    best = np.inf
    for a, b in segments:
        e = b - a
        det = direction[0] * (-e[1]) - direction[1] * (-e[0])
        if abs(det) < 1e-12:
            continue
        rhs = a - cam
        t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / det
        u = (direction[0] * rhs[1] - direction[1] * rhs[0]) / det
        if t > 1e-9 and -1e-9 <= u <= 1 + 1e-9:
            best = min(best, t)
    return best


def make_walls(outline, spans=None, drop=(), occluders=(), cameras=None, extra_lines=(), rotation=0.0,
               shift=(0.0, 0.0), scale=1.0, seed=0, n_rays=360, scene_extra=(), offset_sd=0.02, angle_sd=1.0):
    """Synthetic observed_wall_lines dict. `spans`: {edge_index: [(f0, f1), ...]} fractions of the edge.
    `scene_extra`/`occluders`: extra 2D segments that rays hit (not lines). Collinear edges share one line."""
    c, s = np.cos(rotation), np.sin(rotation)
    rot = np.array([[c, -s], [s, c]])

    def xf(p):
        return scale * (rot @ np.asarray(p, float)) + np.asarray(shift, float)
    poly = [xf(p) for p in outline]
    n = len(poly)
    rng = np.random.default_rng(seed)
    if cameras is None:
        cameras = []
        arr = np.array(poly)
        while len(cameras) < 6:
            q = rng.uniform(arr.min(0), arr.max(0))
            if _inside(arr, q[None])[0] and min(np.linalg.norm(q - a) for a in poly) > 0.8 * scale:
                cameras.append(q)
    else:
        cameras = [xf(q) for q in cameras]
    segments = [(poly[k], poly[(k + 1) % n]) for k in range(n)]
    scene = segments + [(xf(a), xf(b)) for a, b in list(occluders) + list(scene_extra)]
    lines = []
    for k in range(n):
        if k in drop:
            continue
        a, b = segments[k]
        d = (b - a) / np.linalg.norm(b - a)
        m = np.array([d[1], -d[0]])
        off = float(m @ a)
        fr = (spans or {}).get(k, [(0.05, 0.95)])
        target = None
        for line in lines:
            lm = np.array(line["normal_xy"])
            if abs(abs(lm @ m) - 1) < 1e-9 and abs(line["offset"] - np.sign(lm @ m) * off) < 1e-9 * max(1, scale):
                target = line
        if target is None:
            target = {"id": len(lines), "normal_xy": m.tolist(), "offset": off, "direction_xy": d.tolist(), "observed_spans": [],
                      "support": {"distinct_frames": 5, "frame_ids": list(range(5)), "view_spread_deg": 60.0,
                                  "along_coverage_fraction": .8, "weight": 1.0},
                      "uncertainty": {"offset_sd": offset_sd * scale, "angle_sd_deg": angle_sd, "method": "bootstrap"},
                      "height_range": [0.0, 2.6]}
            lines.append(target)
        td = np.array(target["direction_xy"])
        for f0, f1 in fr:
            ts = sorted([float(td @ (a + f0 * (b - a))), float(td @ (a + f1 * (b - a)))])
            target["observed_spans"].append(ts)
    for extra in extra_lines:       # (p0, p1) extra supported stretches
        a, b = xf(extra[0]), xf(extra[1])
        d = (b - a) / np.linalg.norm(b - a)
        m = np.array([d[1], -d[0]])
        lines.append({"id": len(lines), "normal_xy": m.tolist(), "offset": float(m @ a), "direction_xy": d.tolist(),
                      "observed_spans": sorted([[float(d @ a), float(d @ b)]]),
                      "support": {"distinct_frames": 4, "frame_ids": [0, 1, 2, 3], "view_spread_deg": 40.0,
                                  "along_coverage_fraction": .9, "weight": 1.0},
                      "uncertainty": {"offset_sd": offset_sd * scale, "angle_sd_deg": angle_sd, "method": "bootstrap"},
                      "height_range": [0.0, 2.6]})
    hits = []
    for cam in cameras:
        pts = []
        for ang in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
            direction = np.array([np.cos(ang), np.sin(ang)])
            t = _raycast(cam, direction, scene)
            if np.isfinite(t):
                pts.append((cam + t * direction).tolist())
        hits.append(pts)
    ext = float(np.linalg.norm(np.array(poly).max(0) - np.array(poly).min(0)))
    return {"model": "observed_wall_lines", "status": "ok", "reasons": [], "axis_world": [0, 0, 1],
            "frame": {"origin_world": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0], "extent": ext},
            "noise_scale": 0.005 * scale, "lines": lines,
            "observations": {"frame_ids": list(range(len(cameras))), "camera_xy": [list(map(float, q)) for q in cameras],
                             "hits_xy": hits, "hit_heights": [[1.0] * len(h) for h in hits]}}, [p.tolist() for p in poly]


def _walls(*a, **k):
    return make_walls(*a, **k)[0]


class AssemblyTests(unittest.TestCase):
    def check_complete(self, name, outline, spans=None, tol=.2, **kw):
        walls, truth = make_walls(outline, spans=spans, **kw)
        result = assemble_boundary(walls)
        self.assertEqual(result["status"], "complete_polygon_hypothesis", (name, result["incomplete_boundaries"], result.get("boundary_search")))
        self.assertEqual(len(result["vertices_xy"]), len(outline))
        self.assertLess(_rigid_cyclic_error(result["vertices_xy"], truth), tol * kw.get("scale", 1.0), name)
        for e in result["edges"]:
            np.testing.assert_allclose(e["end_xy"], result["edges"][(e["index"] + 1) % len(result["edges"])]["start_xy"], atol=1e-9)
        self.assertGreater(result["area"], 0)
        return result

    def test_shapes_close_with_correct_adjacency(self):
        for name, outline in SHAPES.items():
            self.check_complete(name, outline, spans={k: [(0.1, 0.9)] for k in range(len(outline))})

    def test_partial_spans_close_with_extrapolated_corners_and_larger_uncertainty(self):
        full = self.check_complete("full", RECT, spans={k: [(0.0, 1.0)] for k in range(4)})
        part = self.check_complete("partial", RECT, spans={k: [(0.15, 0.85)] for k in range(4)})
        for e in part["edges"]:
            self.assertTrue(e["has_extrapolated_span"])
            self.assertAlmostEqual(e["observed_fraction"], 0.7, places=2)
            self.assertEqual(len(e["extrapolated_spans_xy"]), 2)
            labels = [s["label"] for s in e["spans"]]
            self.assertEqual(labels.count("extrapolated"), 2)
            self.assertEqual(labels.count("observed"), 1)
            self.assertEqual(e["label"], "observed")
            ext = sum(np.linalg.norm(np.subtract(b, a)) for a, b in e["extrapolated_spans_xy"])
            self.assertAlmostEqual(ext, 0.3 * e["length"], delta=0.05 * e["length"])
            self.assertEqual(e["free_space"]["checked_spans"], 2)
        for ef, ep in zip(full["edges"], part["edges"]):
            self.assertFalse(ef["has_extrapolated_span"])
            self.assertTrue(all(s["label"] == "observed" for s in ef["spans"]))
            self.assertGreater(ep["uncertainty"]["total_sd"], 1.2 * ef["uncertainty"]["total_sd"])

    def test_vertical_extent_and_furniture_height_are_irrelevant(self):
        walls = _walls(RECT, scene_extra=[((3, 2), (4, 2)), ((4, 2), (4, 3)), ((4, 3), (3, 3)), ((3, 3), (3, 2))])
        a = assemble_boundary(walls)
        for l in walls["lines"]:
            l["height_range"] = [0.9, 1.1]
        walls["observations"]["hit_heights"] = [[0.1] * len(h) for h in walls["observations"]["hits_xy"]]
        b = assemble_boundary(walls)
        self.assertEqual(a["status"], "complete_polygon_hypothesis")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_interior_partition_stretch_does_not_become_boundary(self):
        walls = _walls(RECT, extra_lines=[((3, 2), (6, 2))], scene_extra=[((3, 2), (6, 2))], cameras=[(1, 1), (2, 4), (4, 1), (5, 4), (8, 1), (8, 5), (4, 5)])
        result = assemble_boundary(walls)
        self.assertEqual(result["status"], "complete_polygon_hypothesis")
        self.assertEqual(len(result["vertices_xy"]), 4)

    RECESS = [(0, 0), (4, 0), (4, -1), (6, -1), (6, 0), (10, 0), (10, 6), (0, 6)]

    def test_supported_recess_is_followed(self):
        r = self.check_complete("recess", self.RECESS, spans={k: [(0.1, 0.9)] for k in range(8)})
        self.assertEqual(len(r["vertices_xy"]), 8)

    def test_occluded_stretch_without_contradiction_stays_extrapolated(self):
        # bottom wall observed only at its two ends; a screen hides the middle from every camera
        cams = [(1, 3), (2, 5), (3, 2), (7, 3), (8, 5), (9, 2), (5, 4)]
        walls = _walls(RECT, spans={0: [(0.05, 0.35), (0.65, 0.95)]}, cameras=cams, occluders=[((3.5, 0.6), (6.5, 0.6))])
        result = assemble_boundary(walls)
        self.assertEqual(result["status"], "complete_polygon_hypothesis", result["incomplete_boundaries"])
        bottom = [e for e in result["edges"] if abs(e["start_xy"][1]) < 1e-6 and abs(e["end_xy"][1]) < 1e-6][0]
        self.assertTrue(bottom["has_extrapolated_span"])
        mid = [s for s in bottom["spans"] if s["label"] == "extrapolated" and 3.4 < s["start_xy"][0] < 3.6]
        self.assertEqual(len(mid), 1)
        self.assertFalse(bottom["free_space"]["contradiction"])

    def test_gap_contradicted_by_free_space_is_rejected(self):
        # same observed spans, but the gap is a real recess that the cameras see into: rays cross y=0 before their hits
        walls = _walls(self.RECESS, spans={0: [(0.1, 0.9)], 1: [], 2: [], 3: [], 4: [(0.1, 0.9)], 5: [(0.1, 0.9)],
                                           6: [(0.1, 0.9)], 7: [(0.1, 0.9)]},
                       cameras=[(1, 3), (2, 5), (3, 2), (7, 3), (8, 5), (9, 2), (5, 4), (5, 1)])
        result = assemble_boundary(walls)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["vertices_xy"])
        self.assertGreater(result["boundary_search"]["rejected_by_reason"].get("free_space_contradiction", 0), 0)

    def test_missing_whole_wall_is_incomplete(self):
        for name, drop in (("rect", (2,)), ("pentagon", (3,)), ("L", (4,))):
            walls = _walls(SHAPES[name], drop=drop, spans={k: [(0.1, 0.9)] for k in range(len(SHAPES[name]))})
            result = assemble_boundary(walls)
            self.assertIn(result["status"], ("incomplete", "ambiguous"), name)
            self.assertIsNone(result["vertices_xy"], name)
            self.assertTrue(result["incomplete_boundaries"], name)

    def test_rigid_and_scale_invariance(self):
        base = assemble_boundary(_walls(SHAPES["L"], seed=3))
        for rot, shift, scale in ((0.7, (100.0, -40.0), 1.0), (2.3, (0.0, 0.0), 37.0), (-1.1, (5.0, 9.0), 0.03)):
            walls, truth = make_walls(SHAPES["L"], rotation=rot, shift=shift, scale=scale, seed=3)
            result = assemble_boundary(walls)
            self.assertEqual(result["status"], "complete_polygon_hypothesis", (rot, scale))
            self.assertLess(_rigid_cyclic_error(result["vertices_xy"], truth), .2 * scale)
            self.assertAlmostEqual(result["area"] / scale ** 2, base["area"], delta=0.05 * base["area"])

    def test_near_tie_is_ambiguous_with_alternatives(self):
        # a short slanted stretch across a corner: a chamfer wall, or a free-standing partition inside a rectangle.
        # With no camera evidence about which side is room, both layouts are supported.
        sq = [(0, 0), (8, 0), (8, 8), (0, 8)]
        walls = _walls(sq, spans={1: [(0.05, 0.625)], 2: [(0.375, 0.95)]}, extra_lines=[((7.2, 6.8), (6.8, 7.2))])
        walls["observations"] = {"frame_ids": [], "camera_xy": [], "hits_xy": [], "hit_heights": []}
        clear = assemble_boundary(walls, reach_fraction=0.4)       # chamfer needs clearly less extrapolation
        self.assertEqual(clear["status"], "complete_polygon_hypothesis")
        self.assertEqual(len(clear["vertices_xy"]), 5)
        result = assemble_boundary(walls, reach_fraction=0.4, tie_margin=0.5)
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(sorted(len(a["vertices_xy"]) for a in result["alternatives"]), [4, 5])
        self.assertIsNone(result["vertices_xy"])
    def test_deterministic(self):
        walls = _walls(SHAPES["U"], seed=5)
        self.assertEqual(json.dumps(assemble_boundary(walls), sort_keys=True), json.dumps(assemble_boundary(walls), sort_keys=True))

    def test_injected_scorer_is_honoured(self):
        full = _walls(RECT, spans={k: [(0.0, 1.0)] for k in range(4)})
        part = _walls(RECT, spans={k: [(0.15, 0.85)] for k in range(4)})
        veto = lambda a, b: {"free_space_contradiction": True, "free_space_fraction": 1.0, "occupied_fraction": 0.0,
                             "unobserved_fraction": 0.0, "supporting_frames": 0, "contradicting_frames": 5}
        self.assertEqual(assemble_boundary(full, scorer=veto)["status"], "complete_polygon_hypothesis")
        self.assertEqual(assemble_boundary(part, scorer=veto)["status"], "incomplete")

    def test_untrusted_producer_output_is_refused(self):
        walls = _walls(RECT)
        walls["status"] = "insufficient"
        self.assertEqual(assemble_boundary(walls)["status"], "incomplete")


def _ring_obs(n_frames, radius=5.0, offsets=None):
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1).tolist()
    offsets = offsets or [(0, 0), (.1, 0), (0, .1), (-.1, 0)][:n_frames]
    return {"camera_xy": [list(o) for o in offsets], "hits_xy": [ring] * len(offsets)}


class ScorerTests(unittest.TestCase):
    def test_free_occupied_and_unobserved_fractions(self):
        scorer = FreeSpaceScorer(_ring_obs(3), min_frames=3, tolerance=0.1)
        free = scorer((-2, 0), (2, 0.5))
        self.assertAlmostEqual(free["free_space_fraction"], 1.0, places=2)
        self.assertTrue(free["free_space_contradiction"])
        self.assertEqual(free["contradicting_frames"], 3)
        occupied = scorer((5, -1), (5, 1))
        self.assertGreater(occupied["occupied_fraction"], 0.95)
        self.assertFalse(occupied["free_space_contradiction"])
        outside = scorer((7, 0), (9, 0))
        self.assertAlmostEqual(outside["unobserved_fraction"], 1.0, places=2)
        half = scorer((2, 0), (8, 0))     # inside up to the ring, then beyond
        self.assertAlmostEqual(half["free_space_fraction"] + half["occupied_fraction"] + half["unobserved_fraction"], 1.0, places=6)
        self.assertAlmostEqual(half["free_space_fraction"], 0.5, delta=0.1)
        self.assertAlmostEqual(half["unobserved_fraction"], 0.5, delta=0.1)

    def test_same_frame_rays_are_not_independent_frames(self):
        one = FreeSpaceScorer(_ring_obs(1), min_frames=3, tolerance=0.05)((-2, 0), (2, 0))
        self.assertEqual(one["free_space_fraction"], 0.0)
        self.assertFalse(one["free_space_contradiction"])
        self.assertEqual(one["unobserved_fraction"], 1.0)
        two = FreeSpaceScorer(_ring_obs(2), min_frames=3, tolerance=0.05)((-2, 0), (2, 0))
        self.assertFalse(two["free_space_contradiction"])
        self.assertTrue(FreeSpaceScorer(_ring_obs(2), min_frames=2, tolerance=0.05)((-2, 0), (2, 0))["free_space_contradiction"])

    def test_empty_observations_are_unobserved(self):
        r = FreeSpaceScorer({}, min_frames=3)((0, 0), (1, 1))
        self.assertEqual((r["unobserved_fraction"], r["free_space_contradiction"]), (1.0, False))


class AdapterTests(unittest.TestCase):
    def test_adapt_fake_module_scorer(self):
        calls = []

        def fake(p0, p1):
            calls.append((p0, p1))
            return {"free_space_fraction": 0.4, "occupied_fraction": 0.1}
        out = adapt_ray_scorer(fake)((0, 0), (1, 0))
        self.assertTrue(out["free_space_contradiction"])
        self.assertAlmostEqual(out["unobserved_fraction"], 0.5)
        self.assertEqual(out["supporting_frames"], 0)
        self.assertEqual(len(calls), 1)
        bad = adapt_ray_scorer(lambda a, b: 1 / 0)((0, 0), (1, 0))
        self.assertFalse(bad["free_space_contradiction"])
        self.assertIn("scorer_error", bad)
        walls = _walls(RECT, spans={k: [(0.15, 0.85)] for k in range(4)})
        none = adapt_ray_scorer(lambda a, b: {"free_space_contradiction": False, "free_space_fraction": 0.0})
        self.assertEqual(assemble_boundary(walls, scorer=none)["status"], "complete_polygon_hypothesis")
        yes = adapt_ray_scorer(lambda a, b: {"free_space_contradiction": True})
        self.assertEqual(assemble_boundary(walls, scorer=yes)["status"], "incomplete")

    def test_ray_evidence_scorer_with_real_module(self):
        frame = {"origin_world": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0]}
        xs, zs = np.meshgrid(np.linspace(-3, 3, 13), np.linspace(-.8, .8, 5))
        ends = np.stack([xs.ravel(), np.full(xs.size, 5.0), zs.ravel()], axis=1)
        o, e, v = [], [], []
        for view, origin in enumerate(([0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0])):
            o.append(np.tile(origin, (len(ends), 1))); e.append(ends); v.append(np.full(len(ends), view))
        rays = (np.concatenate(o), np.concatenate(e), np.concatenate(v))
        scorer = ba.ray_evidence_scorer(None, frame, [0, 0, 1], [-1.0, 1.0], 0.1, sub_patches=4, min_frames=2, rays=rays)
        on_wall = scorer((-2, 5), (2, 5))
        self.assertFalse(on_wall["free_space_contradiction"])
        self.assertGreater(on_wall["occupied_fraction"], 0.9)
        crossing = scorer((-2, 3), (2, 3))
        self.assertTrue(crossing["free_space_contradiction"])
        self.assertGreater(crossing["free_space_fraction"], 0.9)
        behind = scorer((-2, 7), (2, 7))
        self.assertFalse(behind["free_space_contradiction"])
        self.assertGreater(behind["unobserved_fraction"], 0.9)
        self.assertEqual(crossing["contradicting_frames"], 3)


class PurityTests(unittest.TestCase):
    def test_no_truth_camera_up_or_simulator_imports(self):
        tree = ast.parse(Path(ba.__file__).read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        for name in names:
            for banned in ("simulator", "house_layouts", "evaluation", "evaluate", "layout_mesh", "blender", "ground_truth"):
                self.assertNotIn(banned, name)
        text = Path(ba.__file__).read_text(encoding="utf-8").lower()
        for banned in ("camera_up", "camera up", "ground_truth", "image_up"):
            self.assertNotIn(banned, text.replace("ground truth", "ground_truth").replace("camera-up", "camera_up").replace(
                "never reads ground_truth", "").replace("reads ground_truth, simulator data, camera_up", ""))


if __name__ == "__main__":
    unittest.main()


