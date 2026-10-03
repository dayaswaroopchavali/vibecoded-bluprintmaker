import unittest
"""Synthetic tests for the truth-free gyro orientation prior (no simulator, no ground truth)."""
import ast
import json
from pathlib import Path

import numpy as np

import gyro_orientation as go

HERE = Path(__file__).resolve().parent


def rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float)
    return go.rotation_exp(axis / np.linalg.norm(axis) * a)


def rolling_inverting_chain(n=40, pitch=200.0, roll=270.0):
    """Camera-to-world rotations: pitch sweeps through 200 deg (inversion) while roll sweeps 0..270 deg."""
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(rot([0, 0, 1], 25 * np.sin(3 * t)) @ rot([1, 0, 0], pitch * t) @ rot([0, 0, 1], roll * t))
    return np.asarray(out)


def deltas_of(chain):
    return np.asarray([chain[i - 1].T @ chain[i] for i in range(1, len(chain))])


def box_depth(centre, r_c2w, k, h, w, room=(2.0, 1.5, 1.2)):
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    ray_c = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], -1)
    ray = ray_c @ r_c2w.T
    t = np.full((h, w), np.inf)
    for axis in range(3):
        for sign in (-1, 1):
            with np.errstate(divide="ignore", invalid="ignore"):
                cand = (sign * room[axis] - centre[axis]) / ray[..., axis]
            t = np.where((cand > 0) & (cand < t), cand, t)
    # interior boxes break the planar-wall translation degeneracy (in-plane sliding is unobservable from planes alone)
    for lo, hi in (((0.6, -0.4, -1.2), (1.1, 0.1, 0.3)), ((-1.3, 0.3, -0.5), (-0.8, 0.9, 0.8))):
        lo, hi = np.asarray(lo), np.asarray(hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - centre) / ray[..., :], (hi - centre) / ray[..., :]
        tmin = np.minimum(t1, t2).max(-1)
        tmax = np.maximum(t1, t2).min(-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < t)
        t = np.where(hit, tmin, t)
    return t.astype(np.float32)          # z-depth: ray has unit z in the camera frame


class GyroOrientationTests(unittest.TestCase):
    def test_integrate_recovers_relative_orientation(self):
        chain = rolling_inverting_chain()
        rel = go.integrate_gyro(deltas_of(chain))
        expected = np.einsum("ji,njk->nik", chain[0], chain)
        assert np.allclose(rel, expected, atol=1e-9)
        assert np.allclose(rel[0], np.eye(3))
        assert np.allclose(np.linalg.det(rel), 1.0)


    def test_smooth_roll_and_inversion_are_preserved_not_removed(self):
        chain = go.integrate_gyro(deltas_of(rolling_inverting_chain()))
        down = chain[:, :, 1]
        assert down[:, 1].min() < -0.5 < 0.5 < down[:, 1].max() or down[:, 2].min() < -0.5   # sweeps through inversion
        # the camera "down" axis turns over by more than 90 deg relative to frame 0 and the chain keeps it
        assert go.geodesic_deg(np.einsum("ji,njk->nik", chain[0], chain)).max() > 150
        # successive steps stay small (smooth): nothing was wrapped or reflected away
        assert go.geodesic_deg(deltas_of(chain)).max() < 15


    def test_reflection_rejected_and_proper_rotation_enforced(self):
        d = deltas_of(rolling_inverting_chain(5))
        bad = d.copy()
        bad[1] = bad[1] @ np.diag([1.0, 1.0, -1.0])
        with self.assertRaises(ValueError):
            go.integrate_gyro(bad)
        assert np.all(np.linalg.det(go.integrate_gyro(d)) > 0.999)
        assert np.linalg.det(go.project_to_so3(np.diag([1.0, 1.0, -1.0]))) > 0


    def test_noise_free_injection_is_identity_and_drift_scales_with_noise(self):
        chain = rolling_inverting_chain(60)
        d = deltas_of(chain)
        t = np.arange(len(chain)) / 6.0
        assert np.allclose(go.perturb_deltas(d, t, 0.0, (0, 0, 0)), d)
        rows = go.drift_vs_noise(d, t, (0.0, 0.01, 0.04), seeds=range(6))
        assert rows[0]["final_deviation_deg"] < 1e-9
        assert rows[1]["mean_deviation_deg"] < rows[2]["mean_deviation_deg"]
        # white angular-rate noise integrates as a random walk: sigma*sqrt(T*dt) rad about each axis
        expected_deg = np.degrees(0.04 * np.sqrt(t[-1] * (t[1] - t[0])) * np.sqrt(3) * 0.8862)  # E|N(0,3)| approx
        assert 0.3 * expected_deg < rows[2]["final_deviation_deg"] < 3.0 * expected_deg


    def test_constant_bias_drifts_linearly(self):
        chain = np.asarray([np.eye(3)] * 31)
        d = deltas_of(chain)
        t = np.arange(31) / 6.0
        rows = go.drift_vs_noise(d, t, (0.0,), bias_rad_s=(0.0, 0.0, 0.01))
        assert rows[0]["final_deviation_deg"] == 0 or abs(rows[0]["final_deviation_deg"] - np.degrees(0.01 * t[-1])) < 1e-6


    def test_gauge_recovery_is_truth_free_and_robust_to_outliers(self):
        rng = np.random.default_rng(3)
        gyro = go.integrate_gyro(deltas_of(rolling_inverting_chain(50)))
        q_true = go.rotation_exp(rng.normal(size=3))
        da3 = np.einsum("ij,njk->nik", q_true, gyro)
        bad = rng.choice(50, 15, replace=False)
        for k in bad:
            da3[k] = go.rotation_exp(rng.normal(size=3) * 2) @ da3[k]
        out = go.robust_gauge(gyro, da3)
        assert np.allclose(out["Q"], q_true, atol=1e-6)
        assert abs(out["inlier_fraction_below_10deg"] - (1 - 15 / 50)) < 0.1
        assert abs(np.linalg.det(out["Q"]) - 1.0) < 1e-9


    def test_gauge_reports_disagreement_when_da3_is_unreliable(self):
        rng = np.random.default_rng(4)
        gyro = go.integrate_gyro(deltas_of(rolling_inverting_chain(50)))
        da3 = np.asarray([go.rotation_exp(rng.normal(size=3) * 2) for _ in range(50)])
        out = go.robust_gauge(gyro, da3)
        assert out["inlier_fraction_below_10deg"] < 0.3


    def test_black_border_mask_removes_padding_but_keeps_dark_scene_interior(self):
        img = np.full((100, 100, 3), 120, np.uint8)
        img[:, :15] = 0          # side padding
        img[:12, :] = 0          # top padding
        img[50:60, 50:60] = 0    # dark interior object, not touching the border
        m = go.valid_pixel_mask(img, margin=1)
        assert not m[50, 5] and not m[5, 50]
        assert m[55, 55] and m[80, 80]
        assert m.mean() > 0.6


    def test_solve_centres_refines_warm_start_with_known_rolling_rotations(self):
        # Depth consistency alone is only weakly conditioned for planar rooms (in-plane sliding), so the
        # synthetic check is a warm start: noisy centres must be pulled to the truth, never away from it.
        n, h, w, f = 10, 96, 128, 50.0
        k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        chain = np.einsum("ij,njk->nik", rot([0, 1, 0], 40) @ rot([1, 0, 0], -30), rolling_inverting_chain(n, pitch=60.0, roll=100.0))
        truth = np.stack([[0.3 * np.sin(2 * i / n * np.pi), 0.2 * np.cos(i / n * 3), 0.05 * i] for i in range(n)])
        truth = truth - truth[0]
        depth = np.stack([box_depth(truth[i], chain[i], k, h, w) for i in range(n)])
        start = truth + np.random.default_rng(2).normal(size=truth.shape) * 0.03
        sol = go.solve_centres(depth, np.repeat(k[None], n, 0), chain, stride=6, windows=(1000,), gates=(0.15,),
                               iterations=8, smooth_weights=None, initial_centres=start)
        before = np.linalg.norm((start - start[0]) - truth, axis=1).mean()
        after = np.linalg.norm(sol["centres"] - truth, axis=1).mean()
        assert after < before
        assert after < 0.04
        assert set(sol["observability"]) >= {"min_eig_over_max_eig", "weak_modes_below_1e-3_of_max"}
        assert np.allclose(sol["centres"][0], 0.0)
    def test_extrinsics_round_trip_and_proper(self):
        chain = rolling_inverting_chain(6)
        centres = np.random.default_rng(1).normal(size=(6, 3))
        ext = go.extrinsics_from(chain, centres)
        c, r = go.da3_centres_and_c2w(ext)
        assert np.allclose(c, centres, atol=1e-5) and np.allclose(r, chain, atol=1e-5)
        assert np.all(np.linalg.det(ext[:, :3, :3]) > 0.999)


    def test_module_has_no_simulator_truth_or_audit_imports(self):
        tree = ast.parse((HERE / "gyro_orientation.py").read_text(encoding="utf-8"))
        forbidden = ("simulator", "virtual_gyro", "audit", "evaluat", "ground_truth", "private", "blender", "scene")
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        assert names, "expected imports"
        assert not [n for n in names if any(f in n.lower() for f in forbidden)]
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
        assert not [s for s in strings if "ground_truth" in s or "scene.json" in s or "pose_audit" in s]


if __name__ == "__main__":
    unittest.main()
