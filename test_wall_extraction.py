"""Tests for wall_extraction: synthetic DA3-like predictions rendered per frame (simulator allowed in TESTS only)."""
from __future__ import annotations

import ast
import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

import simulator
from wall_extraction import (check_walls_schema, common_frame, extract_wall_lines, validate_walls_schema)

HERE = Path(__file__).resolve().parent
Z = np.array([0., 0., 1.])
FOV = 80.
W_PX, H_PX = 112, 84


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _pack(frames, noise=0.004, bias=0.003, seed=0):
    """frames: list of dict(depth z-depth, K, R, pos). Adds multiplicative noise and a per-frame scale error."""
    rng = np.random.default_rng(seed)
    depth, K, E = [], [], []
    for f in frames:
        z = f["depth"] * (1 + rng.normal(size=f["depth"].shape) * noise) * (1 + rng.normal() * bias)
        depth.append(z.astype(np.float32)); K.append(f["K"]); E.append(np.c_[f["R"], -f["R"] @ f["pos"]])
    n = len(depth)
    return {"depth": np.array(depth), "confidence": np.isfinite(np.array(depth)).astype(np.float32),
            "intrinsics": np.array(K, dtype=np.float32), "extrinsics": np.array(E, dtype=np.float32),
            "source_indices": np.arange(n) * 3}


def transform_prediction(pred, R_w, t_w, s):
    """world' = s R_w x + t_w; depth scales with s."""
    out = dict(pred)
    E = []
    for e in pred["extrinsics"].astype(float):
        R, t = e[:3, :3], e[:3, 3]
        pos = -R.T @ t
        Rn = R @ R_w.T
        E.append(np.c_[Rn, -Rn @ (s * R_w @ pos + t_w)])
    out["extrinsics"] = np.array(E, dtype=np.float32)
    out["depth"] = (pred["depth"].astype(float) * s).astype(np.float32)
    return out


# ---- rectangular simulator scene (openings + furniture) ----------------------------------------------------------
_SIM = {}


def sim_frames(scene_seed=17, count=36):
    key = (scene_seed, count)
    if key not in _SIM:
        scene = simulator.make_scene(scene_seed)
        poses = simulator.make_paths(scene, 120, 5, "spherical")[0]
        poses = [poses[i] for i in np.linspace(0, 119, count, dtype=int)]
        frames = []
        for pose in poses:
            _img, best, intr, kind, _pos = simulator.render(scene, pose, W_PX, H_PX, FOV, True, True)
            yy, xx = np.mgrid[0:H_PX, 0:W_PX]
            z = best / np.sqrt(1 + ((xx - intr["cx"]) / intr["fx"]) ** 2 + ((yy - intr["cy"]) / intr["fy"]) ** 2)
            right, up, fwd = simulator.camera_basis(pose)
            frames.append({"depth": z, "kind": kind, "K": np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.]]),
                           "R": np.stack([right, -up, fwd]), "pos": np.asarray(pose["position"], float)})
        _SIM[key] = (scene, frames)
    return _SIM[key]


def sim_truth(scene):
    w, l = scene["width"], scene["length"]
    return [((1., 0.), w / 2), ((1., 0.), -w / 2), ((0., 1.), l / 2), ((0., 1.), -l / 2)]


# ---- arbitrary polygon room renderer (z-depth), furniture and openings -------------------------------------------
def _in_poly(poly, x, y):
    inside = np.zeros(x.shape, bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            cond = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        inside ^= cond
    return inside


def render_polygon(verts, height, pose, furniture=(), openings=()):
    fx = W_PX / (2 * math.tan(math.radians(FOV) / 2)); cx, cy = (W_PX - 1) / 2, (H_PX - 1) / 2
    yy, xx = np.mgrid[0:H_PX, 0:W_PX]
    cam = np.stack(((xx - cx) / fx, -(yy - cy) / fx, np.ones_like(xx, dtype=float)), axis=-1)
    right, up, fwd = simulator.camera_basis(pose)
    dirs = cam[..., 0, None] * right + cam[..., 1, None] * up + cam[..., 2, None] * fwd     # z-depth == t
    o = np.asarray(pose["position"], float)
    best = np.full((H_PX, W_PX), np.inf); nan_hit = np.zeros((H_PX, W_PX), bool)

    def segment(a, b, top, opening_list):
        nonlocal best, nan_hit
        e = b - a; dx, dy = dirs[..., 0], dirs[..., 1]
        denom = dx * e[1] - dy * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((a[0] - o[0]) * e[1] - (a[1] - o[1]) * e[0]) / denom
            s = ((a[0] - o[0]) * dy - (a[1] - o[1]) * dx) / denom
        zhit = o[2] + t * dirs[..., 2]
        ok = (np.abs(denom) > 1e-12) & (t > .02) & (s >= 0) & (s <= 1) & (zhit >= 0) & (zhit <= top) & (t < best)
        hole = np.zeros_like(ok)
        for lo, hi, zlo, zhi in opening_list:
            hole |= (s >= lo) & (s <= hi) & (zhit >= zlo) & (zhit <= zhi)
        best = np.where(ok, t, best); nan_hit = np.where(ok, hole, nan_hit)

    for k, (a, b) in enumerate(zip(verts, np.roll(verts, -1, axis=0))):
        segment(a, b, height, [(lo, hi, zl, zh) for (e, lo, hi, zl, zh) in openings if e == k])
    for z0 in (0., height):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z0 - o[2]) / dirs[..., 2]
        px, py = o[0] + t * dirs[..., 0], o[1] + t * dirs[..., 1]
        ok = (t > .02) & (t < best) & _in_poly(verts, px, py)
        best = np.where(ok, t, best); nan_hit = np.where(ok, False, nan_hit)
    for lo, hi, h in furniture:                    # axis-aligned boxes: (xmin,ymin), (xmax,ymax), height
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        corners = [lo, (hi[0], lo[1]), hi, (lo[0], hi[1])]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            segment(np.asarray(a, float), np.asarray(b, float), h, [])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (h - o[2]) / dirs[..., 2]
        px, py = o[0] + t * dirs[..., 0], o[1] + t * dirs[..., 1]
        ok = (t > .02) & (t < best) & (px >= lo[0]) & (px <= hi[0]) & (py >= lo[1]) & (py <= hi[1])
        best = np.where(ok, t, best); nan_hit = np.where(ok, False, nan_hit)
    best = np.where(nan_hit | ~np.isfinite(best), np.nan, best)
    return best, right, up, fwd


def _seg_dist(p, a, b):
    e = b - a; s = np.clip((p - a) @ e / (e @ e), 0, 1)
    return float(np.linalg.norm(p - (a + s * e)))


def polygon_frames(verts, count=30, seed=1, height=2.6, furniture=(), openings=()):
    rng = np.random.default_rng(seed)
    verts = np.asarray(verts, float); frames = []
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    tries = 0
    while len(frames) < count:
        tries += 1
        p = rng.uniform(lo, hi)
        if not _in_poly(verts, np.array([p[0]]), np.array([p[1]]))[0]:
            continue
        if min(_seg_dist(p, a, b) for a, b in zip(verts, np.roll(verts, -1, axis=0))) < .45:
            continue
        if any(lo_[0] - .4 < p[0] < hi_[0] + .4 and lo_[1] - .4 < p[1] < hi_[1] + .4 for lo_, hi_, _h in furniture):
            continue
        yaw = 2 * math.pi * len(frames) / count + rng.normal() * .3
        pose = {"position": [p[0], p[1], rng.uniform(1.1, 1.7)], "yaw": yaw, "pitch": rng.uniform(-.45, .45), "roll": rng.uniform(-.1, .1)}
        z, right, up, fwd = render_polygon(verts, height, pose, furniture, openings)
        fx = W_PX / (2 * math.tan(math.radians(FOV) / 2))
        frames.append({"depth": z, "K": np.array([[fx, 0, (W_PX - 1) / 2], [0, fx, (H_PX - 1) / 2], [0, 0, 1.]]),
                       "R": np.stack([right, -up, fwd]), "pos": np.asarray(pose["position"], float)})
    return frames


# ---- helpers -----------------------------------------------------------------------------------------------------
def world_planes(result):
    """[(n3 unit world normal, c with n3.p = c, direction 3D)] for each line."""
    fr = result["frame"]; o, e1, e2 = (np.asarray(fr[k], float) for k in ("origin_world", "e1", "e2"))
    out = []
    for ln in result["lines"]:
        m3 = e1 * ln["normal_xy"][0] + e2 * ln["normal_xy"][1]
        d3 = e1 * ln["direction_xy"][0] + e2 * ln["direction_xy"][1]
        out.append((m3, ln["offset"] + m3 @ o, d3))
    return out


def match(result, n_t, c_t, tol, cos=math.cos(math.radians(3.))):
    best = None
    n_t = np.asarray(n_t, float)
    for ln, (m3, c, _d3) in zip(result["lines"], world_planes(result)):
        sign = float(m3[:2] @ n_t)
        if abs(sign) < cos:
            continue
        err = abs(math.copysign(1, sign) * c - c_t)
        if err < tol and (best is None or err < best[0]):
            best = (err, ln)
    return best


def run(pred, axis=Z, **kw):
    result = extract_wall_lines(pred, axis, **kw)
    return result


def span_cover(result, ln, a_xy, b_xy):
    """Fraction of the truth segment (world xy endpoints, axis z) covered by observed spans, and total span overshoot."""
    fr = result["frame"]; o = np.asarray(fr["origin_world"], float)[:2]
    e1, e2 = np.asarray(fr["e1"], float)[:2], np.asarray(fr["e2"], float)[:2]
    d3 = np.asarray(fr["e1"], float) * ln["direction_xy"][0] + np.asarray(fr["e2"], float) * ln["direction_xy"][1]
    # t = d.x_local with x_local the frame coordinates of world point p: p = o + e1 x + e2 y  (e1,e2 horizontal when axis=z)
    def t_of(p):
        q = np.asarray(p, float) - o
        return ln["direction_xy"][0] * (q @ e1) + ln["direction_xy"][1] * (q @ e2)
    t0, t1 = sorted((t_of(a_xy), t_of(b_xy)))
    covered = sum(max(0.0, min(t1, s1) - max(t0, s0)) for s0, s1 in ln["observed_spans"])
    over = sum(max(0.0, s1 - s0 - max(0.0, min(t1, s1) - max(t0, s0))) for s0, s1 in ln["observed_spans"])
    return covered / (t1 - t0), over


def assert_no_duplicates(tc, result):
    pl = world_planes(result)
    tol = result["params"]["tolerance"]
    for i in range(len(pl)):
        for j in range(i + 1, len(pl)):
            ci = math.copysign(1, float(pl[i][0] @ pl[j][0]))
            if abs(float(pl[i][0] @ pl[j][0])) > math.cos(math.radians(3.)):
                tc.assertGreater(abs(pl[i][1] - ci * pl[j][1]), tol, f"duplicate lines {i},{j}")


L_SHAPE = [(0, 0), (6, 0), (6, 3), (3, 3), (3, 5), (0, 5)]
PENTAGON = [(0, 0), (5, 0), (6.5, 3), (2.5, 5.5), (-1, 3)]
U_SHAPE = [(0, 0), (6, 0), (6, 4), (4, 4), (4, 2), (2, 2), (2, 4), (0, 4)]


def polygon_truth(verts):
    verts = np.asarray(verts, float); out = []
    for a, b in zip(verts, np.roll(verts, -1, axis=0)):
        d = (b - a) / np.linalg.norm(b - a); n = np.array([d[1], -d[0]])
        out.append((n, float(n @ a), a, b))
    return out


class RectangleWithOpeningsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene, frames = sim_frames(17, 36)
        cls.pred = _pack(frames, seed=3)
        cls.R_w = _rotation(11); cls.t_w = np.array([13., -7., 4.]); cls.s = 3.7
        cls.result = run(cls.pred)

    def test_all_four_walls_found_with_spans_multiview_support_and_valid_schema(self):
        r = self.result
        self.assertEqual(r["status"], "ok")
        check_walls_schema(r)
        extent = r["frame"]["extent"]
        for n, c in sim_truth(self.scene):
            m = match(r, n, c, 0.03 * extent)
            self.assertIsNotNone(m, (n, c))
            ln = m[1]
            self.assertGreaterEqual(ln["support"]["distinct_frames"], 4)
            self.assertFalse(ln["weak"])
            self.assertGreater(ln["support"]["view_spread_deg"], 10)
            self.assertGreater(sum(b - a for a, b in ln["observed_spans"]), .4 * extent / 1.7)
        self.assertEqual(len(r["observations"]["frame_ids"]), 36)
        self.assertEqual(len(r["observations"]["hits_xy"][0]), len(r["observations"]["hit_heights"][0]))
        assert_no_duplicates(self, r)

    def test_openings_are_not_covered_by_spans(self):
        r = self.result
        for op in self.scene["openings"]:
            n, c = sim_truth(self.scene)[{1: 0, 3: 1, 0: 2, 2: 3}[op["wall"]]]      # sim wall index -> (normal, offset)
            m = match(r, n, c, 0.03 * r["frame"]["extent"])
            self.assertIsNotNone(m)
            centre = np.array([c, op["center"]]) if n == (1., 0.) else np.array([op["center"], c])
            fr = r["frame"]; q = centre - np.asarray(fr["origin_world"])[:2]
            ln = m[1]
            t = ln["direction_xy"][0] * (q @ np.asarray(fr["e1"])[:2]) + ln["direction_xy"][1] * (q @ np.asarray(fr["e2"])[:2])
            self.assertFalse(any(a <= t <= b for a, b in ln["observed_spans"]), "span bridges an opening")

    def test_rigid_transform_and_uniform_scale_invariance(self):
        pred2 = transform_prediction(self.pred, self.R_w, self.t_w, self.s)
        r2 = run(pred2, self.R_w @ Z)
        check_walls_schema(r2)
        self.assertAlmostEqual(r2["frame"]["extent"] / self.result["frame"]["extent"], self.s, delta=0.02 * self.s)
        planes2 = world_planes(r2)
        for ln, (m3, c, _d) in zip(self.result["lines"], world_planes(self.result)):
            if ln["weak"]:
                continue
            m_new = self.R_w @ m3; c_new = self.s * c + m_new @ self.t_w
            errs = [(abs(c_new - math.copysign(1, float(m_new @ m2)) * c2) / r2["frame"]["extent"], m2)
                    for m2, c2, _ in planes2 if abs(float(m_new @ m2)) > math.cos(math.radians(2.))]
            self.assertTrue(errs and min(e for e, _ in errs) < 0.01, "line lost under rigid motion + scale")
        weak_relative = [ln["uncertainty"]["offset_sd"] / r2["frame"]["extent"] for ln in r2["lines"] if not ln["weak"]]
        orig_relative = [ln["uncertainty"]["offset_sd"] / self.result["frame"]["extent"] for ln in self.result["lines"] if not ln["weak"]]
        self.assertAlmostEqual(np.median(weak_relative), np.median(orig_relative), delta=0.5 * np.median(orig_relative) + 1e-3)

    def test_deterministic(self):
        a = json.dumps(run(self.pred), sort_keys=True); b = json.dumps(run(self.pred), sort_keys=True)
        self.assertEqual(a, b)

    def test_duplicated_frames_do_not_create_duplicate_lines(self):
        doubled = {k: np.concatenate([v, v]) for k, v in self.pred.items() if k != "source_indices"}
        doubled["source_indices"] = np.arange(72)
        r = run(doubled)
        assert_no_duplicates(self, r)
        strong = [ln for ln in r["lines"] if not ln["weak"]]
        self.assertLessEqual(len(strong), len([ln for ln in self.result["lines"] if not ln["weak"]]) + 1)

    def test_scale_error_between_frame_groups_is_merged_into_one_wall(self):
        biased = dict(self.pred)
        depth = self.pred["depth"].copy(); depth[::2] *= 1.012            # two parallel "sheets" per wall
        biased["depth"] = depth
        r = run(biased)
        assert_no_duplicates(self, r)
        for n, c in sim_truth(self.scene):
            self.assertIsNotNone(match(r, n, c, 0.04 * r["frame"]["extent"]))

    def test_uncertainty_shrinks_with_more_distinct_views(self):
        _scene, frames = sim_frames(17, 36)
        few = run(_pack(frames[::6], seed=3)); many = run(_pack(frames, seed=3))
        sd_few, sd_many, fr_few, fr_many = [], [], [], []
        for n, c in sim_truth(self.scene):
            a, b = match(few, n, c, 0.05 * few["frame"]["extent"]), match(many, n, c, 0.05 * many["frame"]["extent"])
            if a is None or b is None:
                continue
            sd_few.append(a[1]["uncertainty"]["offset_sd"] / few["frame"]["extent"]); sd_many.append(b[1]["uncertainty"]["offset_sd"] / many["frame"]["extent"])
            fr_few.append(a[1]["support"]["distinct_frames"]); fr_many.append(b[1]["support"]["distinct_frames"])
        self.assertGreaterEqual(len(sd_few), 3)
        self.assertLess(np.mean(fr_few), np.mean(fr_many))
        self.assertGreater(np.mean(sd_few), 1.3 * np.mean(sd_many))

    def test_wall_seen_from_one_frame_gets_flag_and_large_uncertainty(self):
        scene, frames = sim_frames(17, 36)
        used = {o["wall"] for o in scene["openings"]}
        wall = next(w for w in range(4) if w not in used)
        counts = [int((f["kind"] == 2 + wall).sum()) for f in frames]
        keep = int(np.argmax(counts))
        masked = []
        for i, f in enumerate(frames):
            z = f["depth"].copy()
            if i != keep:
                z[f["kind"] == 2 + wall] = np.nan
            masked.append({**f, "depth": z})
        r = run(_pack(masked, seed=3))
        n, c = sim_truth(scene)[{1: 0, 3: 1, 0: 2, 2: 3}[wall]]
        m = match(r, n, c, 0.04 * r["frame"]["extent"])
        self.assertIsNotNone(m, "single-frame wall should still be reported (flagged)")
        ln = m[1]
        self.assertEqual(ln["support"]["distinct_frames"], 1)
        self.assertTrue(ln["weak"])
        others = [l["uncertainty"]["offset_sd"] for l in r["lines"] if not l["weak"]]
        self.assertGreater(ln["uncertainty"]["offset_sd"], 3 * float(np.median(others)))
        self.assertGreater(ln["uncertainty"]["angle_sd_deg"], 1.0)


class PolygonRoomTests(unittest.TestCase):
    def _check(self, verts, seed, furniture=(), openings=(), count=34, expect_all=True):
        local = _pack(polygon_frames(verts, count, seed, furniture=furniture, openings=openings), seed=seed)
        R_w, t_w, s = _rotation(seed + 40), np.array([5., -9., 2.]), 1.7
        pred = transform_prediction(local, R_w, t_w, s)
        r = run(pred, R_w @ Z)
        check_walls_schema(r)
        self.assertEqual(r["status"], "ok")
        ext = r["frame"]["extent"]
        found = []
        for k, (n, c, a, b) in enumerate(polygon_truth(verts)):
            n3 = R_w @ np.r_[n, 0.]; p3 = R_w @ np.r_[a, 0.] * s + t_w; q3 = R_w @ np.r_[b, 0.] * s + t_w
            c3 = float(n3 @ p3)
            best = None
            for ln, (m3, cc, d3) in zip(r["lines"], world_planes(r)):
                if abs(float(m3 @ n3)) < math.cos(math.radians(3.)):
                    continue
                err = abs(math.copysign(1, float(m3 @ n3)) * cc - c3)
                if err < 0.03 * ext and (best is None or err < best[0]):
                    best = (err, ln, d3)
            found.append(best)
            if expect_all:
                self.assertIsNotNone(best, f"edge {k} not found")
                ln = best[1]
                t_a = float(best[2] @ (p3 - np.asarray(r["frame"]["origin_world"])))
                t_b = float(best[2] @ (q3 - np.asarray(r["frame"]["origin_world"])))
                t0, t1 = sorted((t_a, t_b))
                cov = sum(max(0.0, min(t1, s1) - max(t0, s0)) for s0, s1 in ln["observed_spans"]) / (t1 - t0)
                self.assertGreater(cov, 0.45, f"edge {k} span coverage")
                self.assertGreaterEqual(ln["support"]["distinct_frames"], 3)
        assert_no_duplicates(self, r)
        return r, found

    def test_l_shape_and_pentagon_all_walls_with_openings_and_furniture(self):
        self._check(L_SHAPE, 4, furniture=[((2.0, 1.0), (3.0, 1.8), 1.0)], openings=[(0, .3, .45, 0., 2.0)])
        self._check(PENTAGON, 9, furniture=[((1.5, 1.5), (2.5, 2.3), 0.9)], openings=[(2, .4, .6, 0.9, 2.0)])

    def test_low_furniture_hiding_wall_bottom_still_yields_the_wall(self):
        # a 1.0 m cabinet 25 cm in front of the lower part of edge 0 (y=0): the wall above it stays visible
        r, found = self._check(L_SHAPE, 6, furniture=[((1.0, .25), (4.5, .85), 1.0)], count=40)
        self.assertGreater(found[0][1]["height_range"][1], found[0][1]["height_range"][0])

    def test_u_shape_keeps_collinear_walls_separated_by_a_real_gap(self):
        r, found = self._check(U_SHAPE, 12, count=44, expect_all=False)
        # y=4 edges (x 4..6 and x 0..2) are collinear with a 2 unit notch gap between them
        ln = None
        for k in (2, 6):
            if found[k] is not None:
                ln = found[k][1]
        self.assertIsNotNone(ln)
        self.assertIs(found[2][1] is found[6][1], True, "collinear arms should share one line")
        self.assertGreaterEqual(len(ln["observed_spans"]), 2)
        gaps = [ln["observed_spans"][i + 1][0] - ln["observed_spans"][i][1] for i in range(len(ln["observed_spans"]) - 1)]
        self.assertGreater(max(gaps), 1.0 * 1.7 * 0.7)      # ~2 units (scaled by 1.7), allow margin


class SchemaAndHygieneTests(unittest.TestCase):
    def test_schema_validator_accepts_good_and_rejects_bad(self):
        scene, frames = sim_frames(17, 36)
        r = run(_pack(frames[::3], seed=1))
        self.assertEqual(validate_walls_schema(r), [])
        bad = copy.deepcopy(r); bad["lines"][0]["normal_xy"] = [2.0, 0.0]
        self.assertTrue(validate_walls_schema(bad))
        bad = copy.deepcopy(r); bad["lines"][0]["observed_spans"] = [[1., 2.], [1.5, 3.]]
        self.assertTrue(validate_walls_schema(bad))
        bad = copy.deepcopy(r); del bad["lines"][0]["uncertainty"]
        self.assertTrue(validate_walls_schema(bad))
        bad = copy.deepcopy(r); bad["observations"]["hit_heights"].pop()
        self.assertTrue(validate_walls_schema(bad))
        with self.assertRaises(ValueError):
            check_walls_schema({"model": "x"})

    def test_insufficient_input_is_refused(self):
        r = extract_wall_lines({"depth": np.full((2, 20, 20), np.nan, np.float32), "confidence": np.ones((2, 20, 20), np.float32),
                                "intrinsics": np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)),
                                "extrinsics": np.tile(np.eye(3, 4, dtype=np.float32), (2, 1, 1))}, Z)
        self.assertEqual(r["status"], "insufficient")
        self.assertEqual(r["lines"], [])
        self.assertEqual(validate_walls_schema(r), [])
        self.assertEqual(extract_wall_lines({}, [0, 0, 0])["status"], "insufficient")

    def test_frame_matches_interface_convention(self):
        a, e1, e2 = common_frame([0, 0, 2.], np.zeros(3))
        np.testing.assert_allclose(e1, [0, 1, 0] if False else np.cross(a, [1, 0, 0]) / 1.0)
        np.testing.assert_allclose(np.cross(a, e1), e2)

    def test_module_has_no_truth_simulator_or_camera_up_dependency(self):
        tree = ast.parse((HERE / "wall_extraction.py").read_text(encoding="utf-8-sig"))
        imported, names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Name):
                names.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                names.add(node.attr.lower())
            elif isinstance(node, ast.arg):
                names.add(node.arg.lower())
        self.assertLessEqual(imported, {"__future__", "argparse", "json", "pathlib", "numpy", "extract_planes", "geometry_orientation"})
        self.assertFalse([n for n in names if any(w in n for w in ("truth", "gravity", "camera_up", "simulator", "annotation"))])


if __name__ == "__main__":
    unittest.main()
