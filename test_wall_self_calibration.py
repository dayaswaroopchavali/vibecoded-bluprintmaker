"""Tests for wall_self_calibration. Truth appears ONLY in this file (the synthetic room that verifies the result); the module under
test is checked by an AST test to import nothing from the simulator / evaluators."""
from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

import numpy as np

import wall_extraction as we
import wall_self_calibration as wsc

HERE = Path(__file__).resolve().parent
FORBIDDEN_IMPORTS = {"simulator", "evaluate_floorplan", "evaluation", "house_layouts", "capture_lidar_sim", "blender_capture",
                     "capture_parallax", "make_box_eval_bundle", "run_rgb_batch", "diagnose_rgb_closure", "compare_wall_boundaries",
                     "run_axis_candidates", "analyze_floorplan_error"}
FORBIDDEN_NAMES = {"private_ground_truth", "ground_truth", "load_scene", "oracle", "true_alignment", "scene_input"}


# ---------------------------------------------------------------------------------------------------------- synthetic room
def _cast(origin, dirs, polygon, height, boxes):
    """First hit parameter t (z-depth, directions have unit camera z) of rays against a vertical-wall polygon room, floor,
    ceiling and axis-aligned boxes (xmin, ymin, zmin, xmax, ymax, zmax). Returns t with inf where nothing is hit."""
    n = len(dirs)
    best = np.full(n, np.inf)
    # floor / ceiling
    for z in (0.0, height):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - origin[2]) / dirs[:, 2]
        ok = (t > 1e-6) & (t < best)
        pt = origin[None, :2] + t[:, None] * dirs[:, :2]
        ok &= _inside_polygon(pt, polygon)
        best = np.where(ok, t, best)
    k = len(polygon)
    for i in range(k):
        a, b = np.asarray(polygon[i], float), np.asarray(polygon[(i + 1) % k], float)
        e = b - a
        nrm = np.array([e[1], -e[0]])
        denom = dirs[:, :2] @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((a - origin[:2]) @ nrm) / denom
        pt = origin[None, :2] + t[:, None] * dirs[:, :2]
        s = ((pt - a) @ e) / float(e @ e)
        z = origin[2] + t * dirs[:, 2]
        ok = (t > 1e-6) & (t < best) & (s >= 0) & (s <= 1) & (z >= 0) & (z <= height)
        best = np.where(ok, t, best)
    for bx in boxes:
        lo, hi = np.asarray(bx[:3], float), np.asarray(bx[3:], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo[None, :] - origin[None, :]) / dirs
            t2 = (hi[None, :] - origin[None, :]) / dirs
        tn = np.minimum(t1, t2).max(axis=1)
        tf = np.maximum(t1, t2).min(axis=1)
        ok = (tn <= tf) & (tn > 1e-6) & (tn < best)
        best = np.where(ok, tn, best)
    return best


def _inside_polygon(pt, polygon):
    poly = np.asarray(polygon, float)
    x, y = pt[:, 0], pt[:, 1]
    inside = np.zeros(len(pt), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cond
        j = i
    return inside


def _camera_basis(yaw, pitch):
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.column_stack([right, down, f])            # camera -> world (OpenCV: x right, y down, z forward)


def make_room_prediction(polygon, height=2.8, boxes=(), n_frames=60, size=(96, 72), fov=78.0, seed=0, noise=0.002,
                         scale_sd=0.0, yaw_walk_deg=0.0, shift_walk=0.0, sim=(0.4, 0.3, (1.0, 2.0, 0.5)), path=None):
    """Synthetic DA3-format prediction with INJECTED drift (truth lives only here). Returns (prediction, truth) with truth =
    {sim (lam, Rg, tg), polygon, true_positions, drift (scale, yaw, shift)}. A global similarity maps the true world into the
    prediction world, per-frame depth scales / yaw / shift errors are injected on top."""
    rng = np.random.default_rng(seed)
    polygon = np.asarray(polygon, float)
    cx, cy = polygon.mean(axis=0)
    W, H = size
    f = 0.5 * W / math.tan(math.radians(fov) / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    yy, xx = np.mgrid[0:H, 0:W]
    rays_cam = np.stack([xx.ravel(), yy.ravel(), np.ones(W * H)], axis=1) @ np.linalg.inv(K).T
    lam, ang, tg = sim
    cg, sg = math.cos(ang), math.sin(ang)
    Rg = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1.0]])
    tg = np.asarray(tg, float)
    extent_true = float(np.ptp(polygon, axis=0).max())
    depth = np.zeros((n_frames, H, W), np.float32)
    conf = np.ones((n_frames, H, W), np.float32)
    extr = np.zeros((n_frames, 3, 4), np.float32)
    positions = []
    scales, yaws, shifts = np.zeros(n_frames), np.zeros(n_frames), np.zeros((n_frames, 2))
    ls = 0.0
    yw = 0.0
    sh = np.zeros(2)
    for k in range(n_frames):
        a = 2 * math.pi * k / n_frames
        rx, ry = 0.3 * np.ptp(polygon, axis=0)
        pos = np.array([cx + rx * math.cos(a), cy + ry * math.sin(a), 1.2])
        if path is not None:
            pos = np.asarray(path(k, n_frames), float)
        yaw = 3.0 * a + 0.5 * math.sin(2 * a)
        c2w = _camera_basis(yaw, 0.0)
        d_cam = rays_cam
        dirs = d_cam @ c2w.T
        t = _cast(pos, dirs, polygon, height, boxes)
        t = np.where(np.isfinite(t), t, 0.0)
        t = t * (1.0 + noise * rng.standard_normal(t.shape))
        # injected errors: random walks (scale: smooth around zero)
        ls = 0.8 * ls + scale_sd * 0.6 * rng.standard_normal()
        yw += math.radians(yaw_walk_deg) * rng.standard_normal()
        sh += shift_walk * extent_true * rng.standard_normal(2)
        scales[k], yaws[k], shifts[k] = ls, yw, sh
        s = math.exp(ls)
        # prediction frame: global similarity of (drifted) true camera
        pos_d = pos + np.array([sh[0], sh[1], 0.0])
        Q = np.array([[math.cos(yw), -math.sin(yw), 0], [math.sin(yw), math.cos(yw), 0], [0, 0, 1.0]])
        c2w_d = Q @ c2w
        centre = lam * Rg @ pos_d + tg
        c2w_p = Rg @ c2w_d
        R = c2w_p.T
        extr[k, :, :3] = R
        extr[k, :, 3] = -R @ centre
        depth[k] = (lam * s * t).reshape(H, W).astype(np.float32)
        positions.append(pos)
    pred = {"depth": depth, "confidence": conf, "extrinsics": extr, "intrinsics": np.repeat(K[None], n_frames, axis=0).astype(np.float32),
            "source_indices": np.arange(n_frames)}
    truth = {"lam": lam, "Rg": Rg, "tg": tg, "polygon": polygon, "positions": np.asarray(positions), "scale_log": scales, "yaw": yaws,
             "shift": shifts, "axis_world": Rg @ np.array([0, 0, 1.0]), "extent_true": extent_true, "height": height}
    return pred, truth


def _umeyama_2d(src, dst):
    sm, dm = src.mean(0), dst.mean(0)
    A, B = src - sm, dst - dm
    U, S, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (A ** 2).sum())
    return s, R, dm - s * R @ sm


def line_errors_to_truth(walls, pred, truth):
    """Distance of each extracted line (midpoint) to the nearest true wall, after the similarity that best maps the prediction's
    camera centres onto the true camera positions (the evaluation-side alignment). Units: metres."""
    cen = wsc.camera_centres(pred["extrinsics"])
    fr = walls["frame"]
    o, e1, e2 = np.asarray(fr["origin_world"]), np.asarray(fr["e1"]), np.asarray(fr["e2"])
    # prediction world -> true world: invert the global similarity, then refine with the centres (2D horizontal similarity)
    lam, Rg, tg = truth["lam"], truth["Rg"], truth["tg"]
    to_true = lambda X: ((np.asarray(X) - tg) @ Rg) / lam
    ct = to_true(cen)[:, :2]
    s, R, t = _umeyama_2d(ct, truth["positions"][:, :2])
    poly = truth["polygon"]
    out = []
    for ln in walls["lines"]:
        m = np.asarray(ln["normal_xy"]); d = np.asarray(ln["direction_xy"])
        sp = ln["observed_spans"]
        tm = 0.5 * (sp[0][0] + sp[-1][1])
        base = ln["offset"] * m
        pm = base + (tm - float(d @ base)) * d
        X = o + pm[0] * e1 + pm[1] * e2
        q = s * (R @ to_true(X)[:2]) + t
        dirw = np.array([d @ [1, 0], d @ [0, 1]])
        dists = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            e = b - a
            L = np.linalg.norm(e)
            nrm = np.array([e[1], -e[0]]) / L
            # direction of the line in true coordinates
            p2 = pm + 0.1 * d
            q2 = s * (R @ to_true(o + p2[0] * e1 + p2[1] * e2)[:2]) + t
            ud = (q2 - q) / np.linalg.norm(q2 - q)
            if abs(float(ud @ e) / L) < math.cos(math.radians(12)):
                continue
            dists.append(abs(float((q - a) @ nrm)))
        out.append(min(dists) if dists else None)
    return out, s


# ---------------------------------------------------------------------------------------------------------------- tests
class TestAstFree(unittest.TestCase):
    def test_module_imports_no_truth(self):
        tree = ast.parse((HERE / "wall_self_calibration.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
            elif isinstance(node, ast.Name):
                self.assertNotIn(node.id, FORBIDDEN_NAMES)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, FORBIDDEN_NAMES)
        self.assertFalse(names & FORBIDDEN_IMPORTS, names & FORBIDDEN_IMPORTS)
        text = (HERE / "wall_self_calibration.py").read_text(encoding="utf-8").lower()
        for token in ("private_ground_truth", "ground_truth", "simulator.", "evaluate_floorplan", "true_alignment", "load_scene"):
            self.assertNotIn(token, text)


RECT = [(0, 0), (6, 0), (6, 5), (0, 5)]
BOXES = [(0.2, 0.2, 0, 1.2, 1.4, 0.9), (4.4, 3.6, 0, 5.7, 4.8, 1.6), (2.4, 0.1, 0, 3.4, 0.7, 2.0)]
L_SHAPE = [(0, 0), (6, 0), (6, 2.6), (3.4, 2.6), (3.4, 5), (0, 5)]
_CACHE = {}


def _scene(tag, polygon, seed, **inj):
    key = (tag, seed, tuple(sorted(inj.items())))
    if key not in _CACHE:
        pred, truth = make_room_prediction(polygon, boxes=BOXES, n_frames=90, size=(160, 120), seed=seed, **inj)
        _CACHE[key] = (pred, truth, we.extract_wall_lines(pred, truth["axis_world"]))
    return _CACHE[key]


MILD = {"scale_sd": 0.025, "yaw_walk_deg": 0.4, "shift_walk": 0.003}      # ~3 % depth scale, ~4 deg yaw and ~5 % extent shift random walks


def _wall_errors(walls, pred, truth, gate=0.5):
    e, _ = line_errors_to_truth(walls, pred, truth)
    return [x for x in e if x is not None and x < gate]


def _same_prediction(a, b, truth):
    return (np.array_equal(a["depth"], b["depth"]) and np.array_equal(a["extrinsics"], b["extrinsics"]))


class TestNoDrift(unittest.TestCase):
    def test_clean_room_is_identity(self):
        pred, truth, w0 = _scene("rect", RECT, 1)
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=False)
        self.assertFalse(rep["applied"])
        self.assertTrue(_same_prediction(pred, out, truth))
        self.assertFalse(rep["truth_used"])
        self.assertEqual(len(wsc.layered_pairs(w0)), 0)

    def test_nonconvex_room_with_true_layers_is_not_collapsed(self):
        """L-shaped room, furniture 0.2 m from a wall (true same-side layered fragments), no drift: nothing may move."""
        pred, truth, w0 = _scene("L", L_SHAPE, 4)
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=False)
        centres_in, centres_out = wsc.camera_centres(pred["extrinsics"]), wsc.camera_centres(out["extrinsics"])
        extent = w0["frame"]["extent"]
        self.assertLess(np.abs(centres_in - centres_out).max(), 0.005 * extent, rep["status"])
        ratio = out["depth"].astype(float) / np.maximum(pred["depth"].astype(float), 1e-9)
        self.assertLess(np.abs(ratio[pred["depth"] > 0] - 1.0).max(), 0.005)
        w1 = we.extract_wall_lines(out, truth["axis_world"])
        self.assertEqual(len(w1["lines"]), len(w0["lines"]))
        offs0 = sorted(round(l["offset"], 3) for l in w0["lines"])
        offs1 = sorted(round(l["offset"], 3) for l in w1["lines"])
        self.assertLess(max(abs(a - b) for a, b in zip(offs0, offs1)), 0.01 * extent)


class TestDrift(unittest.TestCase):
    def test_layers_vanish_and_offsets_shrink(self):
        pred, truth, w0 = _scene("rect", RECT, 4, **MILD)
        lay0 = len(wsc.layered_pairs(w0))
        self.assertGreaterEqual(lay0, 3)
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=True)
        self.assertEqual(rep["status"], "calibrated", rep.get("reasons"))
        w1 = we.extract_wall_lines(out, truth["axis_world"])
        self.assertEqual(len(wsc.layered_pairs(w1)), 0)
        e0, e1 = _wall_errors(w0, pred, truth), _wall_errors(w1, out, truth)
        self.assertLess(np.median(e1), 0.5 * np.median(e0))
        self.assertLess(max(e1), max(e0))
        self.assertLess(len(w1["lines"]), len(w0["lines"]))
        self.assertGreater(rep["parameters"]["refused_count"], 0)           # the weakly determined knots stay at DA3

    def test_polygon_closes_after_calibration(self):
        import boundary_assembly as ba
        pred, truth, w0 = _scene("rect", RECT, 1, **MILD)
        before = ba.assemble_boundary_ray(pred, w0, 2.0)["status"]
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=True)
        w1 = we.extract_wall_lines(out, truth["axis_world"])
        after = ba.assemble_boundary_ray(out, w1, 2.0)["status"]
        self.assertNotEqual(before, "complete_polygon_hypothesis")
        self.assertEqual(rep["status"], "calibrated")
        self.assertEqual(after, "complete_polygon_hypothesis")
        e0, e1 = _wall_errors(w0, pred, truth), _wall_errors(w1, out, truth)
        self.assertLess(np.median(e1), np.median(e0))

    def test_no_collapse_of_true_layers_under_drift(self):
        """Drifting L-shaped room with furniture close to a wall: the true structure survives (same count of true wall lines,
        and the box-face line stays separated from its wall by more than the noise)."""
        pred, truth, w0 = _scene("L", L_SHAPE, 4, **MILD)
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=True)
        w1 = we.extract_wall_lines(out, truth["axis_world"])
        n0, n1 = len(_wall_errors(w0, pred, truth, 0.3)), len(_wall_errors(w1, out, truth, 0.3))
        self.assertGreaterEqual(n1, 4)
        self.assertGreaterEqual(n1, n0 - 2)
        # all calibrated wall lines stay close to a true wall (a collapse onto furniture would move them away)
        self.assertLess(np.median(_wall_errors(w1, out, truth, 0.3)), 0.15)


class TestInterface(unittest.TestCase):
    def test_schema_dtypes_determinism_and_gauge(self):
        pred, truth, w0 = _scene("rect", RECT, 4, **MILD)
        keep = {k: np.array(v, copy=True) for k, v in pred.items()}
        out1, rep1 = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=False)
        out2, rep2 = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=False)
        for k in pred:                                                     # input never mutated
            self.assertTrue(np.array_equal(pred[k], keep[k]), k)
        self.assertEqual(set(out1), set(pred))                            # same npz schema
        for k in pred:
            self.assertEqual(out1[k].shape, pred[k].shape, k)
            self.assertEqual(out1[k].dtype, pred[k].dtype, k)
        self.assertTrue(np.array_equal(out1["depth"], out2["depth"]))     # deterministic
        self.assertTrue(np.array_equal(out1["extrinsics"], out2["extrinsics"]))
        self.assertEqual(rep1["status"], "calibrated")
        R = out1["extrinsics"][:, :3, :3].astype(float)
        self.assertTrue((np.linalg.det(R) > 0).all())                     # proper rotations only
        self.assertLess(np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max(), 1e-4)
        self.assertTrue(np.array_equal(out1["intrinsics"], pred["intrinsics"]))
        changed = np.flatnonzero(~np.all(out1["extrinsics"] == pred["extrinsics"], axis=(1, 2)))
        self.assertGreater(len(changed), 0)
        logs = [math.log(f["depth_scale"]) for f in rep1["frames"]]
        self.assertLess(abs(float(np.mean(logs))), 2e-3)                  # gauge: mean log depth scale = 0
        self.assertEqual(rep1["parameters"]["kept_at_da3"][:3], ["vertical translation", "pitch", "roll"])
        # the horizontal-only model: the vertical coordinate of every camera centre is unchanged
        a = np.asarray(truth["axis_world"])
        dz = (wsc.camera_centres(out1["extrinsics"]) - wsc.camera_centres(pred["extrinsics"])) @ a
        self.assertLess(np.abs(dz).max(), 1e-4)

    def test_reflection_is_refused(self):
        pred, truth, w0 = _scene("rect", RECT, 4, **MILD)
        bad = dict(pred)
        extr = pred["extrinsics"].copy()
        extr[5, :3, :3] = extr[5, :3, :3] @ np.diag([1.0, 1.0, -1.0])
        bad["extrinsics"] = extr
        out, rep = wsc.calibrate(bad, truth["axis_world"], walls=w0, verify=False)
        self.assertEqual(rep["status"], "refused_reflection")
        self.assertTrue(np.array_equal(out["extrinsics"], extr))

    def test_unidentifiable_parameters_are_refused(self):
        """A vanishing refusal factor makes every uncertainty exceed the magnitude: everything stays at DA3."""
        pred, truth, w0 = _scene("rect", RECT, 4, **MILD)
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=w0, verify=False, refuse_factor=1e-9)
        self.assertEqual(rep["status"], "unidentifiable")
        self.assertTrue(_same_prediction(pred, out, truth))

    def test_invalid_axis_and_no_lines(self):
        pred, truth, w0 = _scene("rect", RECT, 1)
        out, rep = wsc.calibrate(pred, [0.0, 0.0, 0.0], walls=w0)
        self.assertEqual(rep["status"], "invalid_axis")
        empty = {**w0, "status": "insufficient", "lines": [], "reasons": ["x"]}
        out, rep = wsc.calibrate(pred, truth["axis_world"], walls=empty)
        self.assertEqual(rep["status"], "no_lines")
        self.assertTrue(_same_prediction(pred, out, truth))


class TestRunnerTruthFree(unittest.TestCase):
    def test_truth_free_stages_of_the_runner(self):
        import inspect

        import run_wall_self_calibration as runner
        for name in runner.TRUTH_FREE_FUNCTIONS:
            src = inspect.getsource(getattr(runner, name))
            for token in runner.PRIVATE_NAMES:
                self.assertNotIn(token, src, f"{name} mentions {token}")


if __name__ == "__main__":
    unittest.main()
