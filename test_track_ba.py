"""Tests for track_ba: tracker, BA drift recovery, ceiling-vs-wall scale, determinism, and truth-free source."""
import ast
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import track_ba as tb

HERE = Path(__file__).resolve().parent
W_, L_, H_ = 6.0, 5.0, 3.0  # synthetic room: x, y, z (z up)


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def _room_points(n, seed):
    rng = np.random.default_rng(seed)
    pts = []
    per = n // 6
    for ax, val in ((0, 0.0), (0, W_), (1, 0.0), (1, L_), (2, 0.0), (2, H_)):
        p = np.column_stack([rng.uniform(0, W_, per), rng.uniform(0, L_, per), rng.uniform(0, H_, per)])
        p[:, ax] = val + rng.normal(0, 0.004, per)
        pts.append(p)
    return np.concatenate(pts)


def _look_at_w2c(centre, target, up=np.array([0., 0., 1.])):
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ centre


def _scene(n_cam=30, n_pts=6000, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    K = np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    pts = _room_points(n_pts, seed + 1)
    R, t = [], []
    for i in range(n_cam):
        a = 2 * np.pi * i / n_cam
        c = np.array([W_ / 2 + 1.2 * np.cos(a), L_ / 2 + 1.0 * np.sin(a), 1.4 + 0.2 * np.sin(3 * a)])
        look = c + np.array([np.cos(a * 2.3 + 0.7), np.sin(a * 2.3 + 0.7), 0.7 * np.sin(a * 4)])
        Ri, ti = _look_at_w2c(c, look)
        R.append(Ri)
        t.append(ti)
    R, t = np.array(R), np.array(t)
    ob_t, ob_f, ob_xy = [], [], []
    for f in range(n_cam):
        pc = pts @ R[f].T + t[f]
        uv = pc @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        vis = (pc[:, 2] > 0.3) & (uv[:, 0] > 5) & (uv[:, 0] < 635) & (uv[:, 1] > 5) & (uv[:, 1] < 475)
        idx = np.flatnonzero(vis)
        ob_t.append(idx)
        ob_f.append(np.full(len(idx), f))
        ob_xy.append(uv[idx] + rng.normal(0, noise, (len(idx), 2)))
    obs = {"track_id": np.concatenate(ob_t), "frame": np.concatenate(ob_f), "xy": np.concatenate(ob_xy)}
    order = np.lexsort((obs["frame"], obs["track_id"]))
    obs = {k: v[order] for k, v in obs.items()}
    return K, pts, R, t, tb.compact_tracks(obs, 3)


def _drift(R, t, seed=5, rot_deg=0.25, step=0.02):
    rng = np.random.default_rng(seed)
    C = tb.camera_centres(R, t)
    dr = np.cumsum(rng.normal(0, np.radians(rot_deg), (len(R), 3)), axis=0)
    dc = np.cumsum(rng.normal(0, step, (len(R), 3)), axis=0)
    dr[0] = 0
    dc[0] = 0
    R2 = Rotation.from_rotvec(dr).as_matrix() @ R
    C2 = C + dc
    return R2, -np.einsum("nij,nj->ni", R2, C2)


def _similarity_rmse(a, b):
    xc, yc = a.mean(0), b.mean(0)
    u, s, vt = np.linalg.svd((b - yc).T @ (a - xc))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = np.sum(s * np.diag(d)) / np.sum((a - xc) ** 2)
    al = scale * (a - xc) @ rot.T + yc
    return float(np.sqrt(np.mean(np.sum((al - b) ** 2, axis=1))))


class TrackerTests(unittest.TestCase):
    def _frames(self, n=8, shift=(3.0, 1.5)):
        base = _texture(300, 400, 3)
        frames = []
        for i in range(n):
            M = np.float32([[1, 0, shift[0] * i], [0, 1, shift[1] * i]])
            frames.append(cv2.warpAffine(base, M, (400, 300), borderMode=cv2.BORDER_REFLECT))
        return frames

    def test_translation_tracks_are_chained_and_accurate(self):
        frames = self._frames()
        cfg = tb.TrackConfig(max_corners_per_frame=200)
        obs = tb.track_features(frames, cfg)
        stats = tb.track_statistics(obs, len(frames))
        self.assertGreater(stats["tracks_len_ge5"], 30)
        self.assertGreaterEqual(stats["max_length"], 7)
        # displacement between consecutive observations of any track must equal the known shift (3, 1.5)
        tid, fr, xy = obs["track_id"], obs["frame"], obs["xy"]
        same = (tid[1:] == tid[:-1]) & (fr[1:] == fr[:-1] + 1)
        d = (xy[1:] - xy[:-1])[same]
        self.assertGreater(len(d), 200)
        self.assertLess(np.median(np.abs(d - np.array([3.0, 1.5]))), 0.15)
        self.assertLess(np.percentile(np.abs(d - np.array([3.0, 1.5])), 99), 1.5)

    def test_new_features_are_seeded_on_later_frames(self):
        obs = tb.track_features(self._frames(), tb.TrackConfig(max_corners_per_frame=200))
        starts = {}
        for t_, f_ in zip(obs["track_id"], obs["frame"]):
            starts.setdefault(t_, f_)
        self.assertGreater(len({s for s in starts.values()}), 3)

    def test_textureless_frames_yield_no_tracks(self):
        frames = [np.full((240, 320), 128, np.uint8)] * 4
        obs = tb.track_features(frames)
        self.assertEqual(tb.track_statistics(obs, 4)["tracks_len_ge2"], 0)

    def test_restrict_to_frames_renumbers_to_prediction_indices(self):
        obs = {"track_id": np.array([0, 0, 0, 1, 1]), "frame": np.array([0, 3, 6, 3, 4]),
               "xy": np.arange(10, dtype=float).reshape(5, 2)}
        out = tb.restrict_to_frames(obs, np.array([0, 3, 6]))
        np.testing.assert_array_equal(out["frame"], [0, 1, 2, 1])
        np.testing.assert_array_equal(out["track_id"], [0, 0, 0, 1])

    def test_deterministic(self):
        a = tb.track_features(self._frames(), tb.TrackConfig(max_corners_per_frame=150))
        b = tb.track_features(self._frames(), tb.TrackConfig(max_corners_per_frame=150))
        for k in ("track_id", "frame", "xy"):
            np.testing.assert_array_equal(a[k], b[k])


class BundleAdjustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.K, cls.pts, cls.R, cls.t, cls.obs = _scene()
        cls.R2, cls.t2 = _drift(cls.R, cls.t)
        cls.pred = {"extrinsics": np.concatenate([cls.R2, cls.t2[:, :, None]], 2),
                    "intrinsics": np.repeat(cls.K[None], len(cls.R), 0)}
        cls.res = tb.refine_prediction(cls.pred, cls.obs, tb.BAConfig(max_points=3000, max_nfev=30))

    def test_recovers_injected_drift(self):
        res = self.res
        self.assertTrue(res["ok"])
        Ct = tb.camera_centres(self.R, self.t)
        before = _similarity_rmse(tb.camera_centres(self.R2, self.t2), Ct)
        after = _similarity_rmse(res["C"], Ct)
        self.assertGreater(before, 0.05)
        self.assertLess(after, 0.35 * before)
        ang = lambda A, B: np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", A, B) - 1) / 2, -1, 1)))
        # rotation relative to truth, after removing the (shared) world rotation of camera 0
        self.assertLess(ang(res["R"], self.R).mean(), 0.5 * ang(self.R2, self.R).mean())

    def test_reprojection_residual_reduced_and_small(self):
        res = self.res
        self.assertLess(res["residual_stats_final_px"]["rms_px"], 0.8)
        self.assertGreater(res["residual_stats_before_first_round_px"]["rms_px"], 1.5 * res["residual_stats_final_px"]["rms_px"])

    def test_ceiling_vs_wall_scale_recovered_from_sparse_points(self):
        res = self.res
        # world frame of the BA solution is cam-0 anchored (cam 0 is undrifted), so z is up
        up = np.array([0., 0., 1.])
        pts, ids = res["points"], res["point_view_ids"]
        m = tb.cloud_room_metrics(pts, ids, up, [np.array([1., 0, 0]), np.array([0., 1, 0])])
        self.assertEqual(m["status"], "complete", m.get("reasons"))
        self.assertAlmostEqual(m["separation"] / (res["points"][:, 0].ptp() * 0 + H_), 1.0, delta=0.05)
        r = [w["ratio_extent_p2_p98"] for w in m["walls"]]
        self.assertAlmostEqual(r[0], W_ / H_, delta=0.06 * W_ / H_)
        self.assertAlmostEqual(r[1], L_ / H_, delta=0.06 * L_ / H_)

    def test_metric_detects_injected_ceiling_bias(self):
        up = np.array([0., 0., 1.])
        pts = _room_points(9000, 11)
        ids = np.random.default_rng(0).integers(0, 8, len(pts))
        dirs = [np.array([1., 0, 0]), np.array([0., 1, 0])]
        base = tb.cloud_room_metrics(pts, ids, up, dirs)
        biased = pts.copy()
        biased[:, 2] *= 0.95  # ceiling read 5% too near
        bm = tb.cloud_room_metrics(biased, ids, up, dirs)
        self.assertEqual(base["status"], "complete", base.get("reasons"))
        self.assertEqual(bm["status"], "complete", bm.get("reasons"))
        ratio = bm["walls"][0]["ratio_extent_p2_p98"] / base["walls"][0]["ratio_extent_p2_p98"]
        self.assertAlmostEqual(ratio, 1 / 0.95, delta=0.01)

    def test_deterministic(self):
        again = tb.refine_prediction(self.pred, self.obs, tb.BAConfig(max_points=3000, max_nfev=30))
        np.testing.assert_allclose(again["C"], self.res["C"], atol=1e-9)
        np.testing.assert_allclose(again["points"], self.res["points"], atol=1e-9)

    def test_outlier_tracks_are_rejected(self):
        obs = {k: v.copy() for k, v in self.obs.items()}
        rng = np.random.default_rng(1)
        bad = rng.choice(len(obs["xy"]), len(obs["xy"]) // 20, replace=False)
        obs["xy"][bad] += rng.uniform(-60, 60, (len(bad), 2))
        res = tb.refine_prediction(self.pred, obs, tb.BAConfig(max_points=3000, max_nfev=30, loss="cauchy"))
        self.assertTrue(res["ok"])
        Ct = tb.camera_centres(self.R, self.t)
        self.assertLess(_similarity_rmse(res["C"], Ct), 0.6 * _similarity_rmse(tb.camera_centres(self.R2, self.t2), Ct))


class SourceTruthFreeTests(unittest.TestCase):
    def test_track_ba_has_no_truth_or_simulator_imports(self):
        src = (HERE / "track_ba.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = {"simulator", "evaluation", "evaluate_floorplan", "audit_camera_trajectory", "blender_capture",
                  "house_layouts", "evaluate_rgb_boundary", "analyze_floorplan_error", "run_pipeline"}
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertFalse(names & banned, names & banned)
        for needle in ("private_ground_truth", "ground_truth", "scene.json", "generator_private"):
            self.assertNotIn(needle, src.replace("Ground truth, the simulator", ""))


if __name__ == "__main__":
    unittest.main()
