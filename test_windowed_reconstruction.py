"""Window linking and Sim(3) merging: synthetic windows with known transforms (perfect depth rendered from the simulator)."""
import unittest

import numpy as np

import simulator
import windowed_reconstruction as W

P = dict(W.PARAMS)


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def world_frames(count=24, start=40, scene_seed=17, width=64, height=48, fov=78.):
    """Consecutive frames of a rolling/inverted capture with true depth, poses and intrinsics in one world frame."""
    scene = simulator.make_scene(scene_seed)
    path = simulator.make_paths(scene, 120, 3101)[0][start:start + count]
    depth, conf, K, E, imgs, idx = [], [], [], [], [], []
    for k, pose in enumerate(path):
        with np.errstate(all="ignore"):
            image, best, intr = simulator.render(scene, pose, width, height, fov, True)
        yy, xx = np.mgrid[0:height, 0:width]
        z = best / np.sqrt(1 + ((xx - intr["cx"]) / intr["fx"]) ** 2 + ((yy - intr["cy"]) / intr["fy"]) ** 2)
        right, up, forward = simulator.camera_basis(pose)
        R = np.stack([right, -up, forward])
        E.append(np.c_[R, -R @ np.asarray(pose["position"])])
        K.append(np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.]]))
        depth.append(z.astype(np.float64)); conf.append(np.isfinite(z).astype(np.float64)); imgs.append(image); idx.append(start + k)
    return {"depth": np.array(depth), "confidence": np.array(conf), "intrinsics": np.array(K), "extrinsics": np.array(E),
            "images": np.array(imgs), "source_indices": np.array(idx)}


def to_local(frames, ids, s, Q, b, mirror=False):
    """The same frames expressed in a window-local frame with Xworld = s Q Xlocal + b (optionally mirrored)."""
    pick = [list(frames["source_indices"]).index(i) for i in ids]
    E = frames["extrinsics"][pick]
    R, t = E[:, :, :3], E[:, :, 3]
    Rl = R @ Q; tl = (np.einsum("nij,j->ni", R, b) + t) / s
    if mirror:
        Rl = Rl @ np.diag([-1., 1., 1.])
    return {"depth": frames["depth"][pick] / s, "confidence": frames["confidence"][pick], "intrinsics": frames["intrinsics"][pick],
            "extrinsics": np.concatenate([Rl, tl[:, :, None]], axis=2), "images": frames["images"][pick],
            "source_indices": np.array(ids)}


def make_windows(frames, size=8, stride=4, seed=0):
    rng = np.random.default_rng(seed); wins = []
    for g in W.make_windows(len(frames["source_indices"]), size, stride):
        ids = [int(frames["source_indices"][i]) for i in g]
        wins.append(to_local(frames, ids, rng.uniform(.4, 3.0), _rotation(rng.integers(1000)), rng.normal(size=3) * 5))
    return wins


def umeyama_align(src, dst):
    s, Q, b, _ = W.umeyama(src, dst)
    return s, Q, b


FRAMES = None


def frames():
    global FRAMES
    if FRAMES is None:
        FRAMES = world_frames()
    return FRAMES


class TransformConventionTests(unittest.TestCase):
    def test_transformed_window_unprojects_to_the_transformed_points_and_keeps_pixels(self):
        f = frames(); s, Q, b = 1.7, _rotation(3), np.array([2., -1., .5])
        local = to_local(f, [int(i) for i in f["source_indices"][:3]], 1.0, np.eye(3), np.zeros(3))
        moved = W.transform_window(local, s, Q, b)
        for k in range(3):
            p0, pix0 = W.frame_points(local, k, P); p1, pix1 = W.frame_points(moved, k, P)
            np.testing.assert_array_equal(pix0, pix1)
            np.testing.assert_allclose(p1, s * p0 @ Q.T + b, atol=1e-9)           # Xg = s Q Xl + b
            # pixel reprojection invariance: project the transformed points with the transformed pose, same intrinsics
            E = moved["extrinsics"][k]; cam = p1 @ E[:3, :3].T + E[:3, 3]
            uv = cam @ moved["intrinsics"][k].T; uv = uv[:, :2] / uv[:, 2:3]
            np.testing.assert_allclose(uv, pix0[:, ::-1], atol=1e-6)
        np.testing.assert_allclose(W.camera_center(moved["extrinsics"]), s * W.camera_center(local["extrinsics"]) @ Q.T + b, atol=1e-9)
        np.testing.assert_array_equal(moved["intrinsics"], local["intrinsics"])        # pixel intrinsics unchanged
        np.testing.assert_allclose(moved["depth"], local["depth"] * s)                 # axial depth multiplies by s

    def test_scaling_rotations_or_poses_without_depth_would_be_wrong(self):
        f = frames(); local = to_local(f, [int(f["source_indices"][0])], 1.0, np.eye(3), np.zeros(3))
        s, Q, b = 2.0, _rotation(5), np.array([1., 2., 3.])
        right = W.transform_window(local, s, Q, b)
        wrong = dict(right, depth=local["depth"])                                      # pose transformed, depth not scaled
        a, _ = W.frame_points(right, 0, P); c, _ = W.frame_points(wrong, 0, P)
        self.assertGreater(float(np.abs(a - c).max()), .1)


class LinkAndMergeTests(unittest.TestCase):
    def test_known_transforms_are_recovered_and_every_link_is_accepted(self):
        f = frames(); wins = make_windows(f)
        pred, rep = W.merge_windows(wins)
        self.assertTrue(rep["connected"]); self.assertEqual(rep["links_rejected"], 0); self.assertEqual(rep["components"], 1)
        self.assertEqual(rep["retained_capture_fraction"], 1.0)
        truth_centers = W.camera_center(f["extrinsics"])
        got = W.camera_center(pred["extrinsics"]); order = [list(f["source_indices"]).index(i) for i in pred["source_indices"]]
        s, Q, b = umeyama_align(got, truth_centers[order])
        extent = np.linalg.norm(np.ptp(truth_centers, axis=0))
        np.testing.assert_allclose(s * got @ Q.T + b, truth_centers[order], atol=1e-6 * extent)
        # rotations agree after the same global alignment
        for k, j in enumerate(order):
            Rg = pred["extrinsics"][k][:3, :3] @ Q.T                                    # global camera rotation in the world frame
            self.assertLess(W.rotation_angle_deg(Rg, f["extrinsics"][j][:3, :3]), 1e-4)
        np.testing.assert_array_equal(pred["source_indices"], np.sort(pred["source_indices"]))   # source indices preserved, ordered
        self.assertEqual(set(pred["component_ids"]), {0})

    def test_point_geometry_stays_consistent_with_the_cameras_after_merging(self):
        f = frames(); pred, _ = W.merge_windows(make_windows(f, seed=4))
        a = {k: pred[k] for k in ("depth", "confidence", "intrinsics", "extrinsics")}
        pts = np.concatenate([W.unproject(a["depth"][k], a["confidence"][k], a["intrinsics"][k], a["extrinsics"][k], 4)[0] for k in range(len(a["depth"]))])
        # all merged points lie on the (single, scaled) room: no frame sits in a different scale or frame
        near = pts[np.isfinite(pts).all(axis=1)]
        extent = np.linalg.norm(np.percentile(near, 95, axis=0) - np.percentile(near, 5, axis=0))
        centers = W.camera_center(pred["extrinsics"])
        self.assertLess(np.linalg.norm(np.ptp(centers, axis=0)), extent)

    def test_mirrored_window_is_rejected_not_fitted_with_a_reflection(self):
        f = frames(); ids = [int(i) for i in f["source_indices"][:8]]
        a = to_local(f, ids, 1.0, np.eye(3), np.zeros(3)); b = to_local(f, ids[4:] + [int(f["source_indices"][8 + j]) for j in range(4)],
                                                                         1.3, _rotation(2), np.array([1., 0, 2.]), mirror=True)
        link = W.link_windows(a, b)
        self.assertFalse(link["ok"]); self.assertTrue({"high_residual", "low_inlier_fraction", "shared_frame_rotation_disagreement", "shared_frame_center_disagreement"} & set(link["reasons"]), link["reasons"])
        _pred, rep = W.merge_windows([a, b])
        self.assertEqual(rep["components"], 2)

    def test_collinear_and_degenerate_overlaps_are_rejected(self):
        t = np.linspace(0, 5, 400)
        line = np.stack([t, 2 * t, -t], axis=1)
        out = W.robust_sim3(line, line * 1.5 + 1.0)
        self.assertFalse(out["ok"]); self.assertIn("rank_deficient_or_collinear_overlap", out["reasons"])
        self.assertIn("too_few_correspondences", W.robust_sim3(line[:10], line[:10])["reasons"])
        rng = np.random.default_rng(0); plane = np.c_[rng.uniform(-3, 3, (500, 2)), np.zeros(500)]
        s, Q, b = 1.8, _rotation(8), np.array([1., 2., 3.])
        fit = W.robust_sim3(plane, s * plane @ Q.T + b)
        self.assertTrue(fit["ok"]); self.assertAlmostEqual(fit["s"], s, places=6)           # a plane still fixes a proper Sim(3)

    def test_inconsistent_correspondences(self):
        rng = np.random.default_rng(1); pts = rng.uniform(-3, 3, (1500, 3)); s, Q, b = .7, _rotation(9), np.array([4., -2., 1.])
        dst = s * pts @ Q.T + b
        some = dst.copy(); bad = rng.choice(1500, 300, replace=False); some[bad] += rng.normal(size=(300, 3)) * 3
        fit = W.robust_sim3(pts, some)
        self.assertTrue(fit["ok"], fit["reasons"]); self.assertAlmostEqual(fit["s"], s, places=2)      # 20% outliers: robust
        most = dst.copy(); worse = rng.choice(1500, 1100, replace=False); most[worse] += rng.normal(size=(1100, 3)) * 3
        self.assertFalse(W.robust_sim3(pts, most)["ok"])                                                # 73% outliers: refused

    def test_shared_frame_pose_disagreement_is_reported_and_blocks_the_link(self):
        f = frames(); ids = [int(i) for i in f["source_indices"][:8]]; ids2 = ids[4:] + [int(i) for i in f["source_indices"][8:12]]
        a = to_local(f, ids, 1.0, np.eye(3), np.zeros(3)); b = to_local(f, ids2, 1.0, np.eye(3), np.zeros(3))
        good = W.link_windows(a, b); self.assertTrue(good["ok"])
        self.assertLess(good["max_rotation_disagreement_deg"], 1e-4)
        tilt = np.radians(12); Rz = np.array([[np.cos(tilt), -np.sin(tilt), 0], [np.sin(tilt), np.cos(tilt), 0], [0, 0, 1]])
        E = b["extrinsics"].copy(); E[0, :, :3] = Rz @ E[0, :, :3]; E[0, :, 3] = Rz @ E[0, :, 3]            # one shared frame's pose disagrees
        bad = W.link_windows(a, dict(b, extrinsics=E))
        self.assertIn("shared_frame_rotation_disagreement", bad["reasons"])
        self.assertGreater(bad["max_rotation_disagreement_deg"], 5)
        self.assertEqual(len(bad["shared_frame_pose_disagreement"]), 4)

    def test_windows_without_shared_frames_become_separate_components_never_bridged(self):
        f = frames(); a = to_local(f, [int(i) for i in f["source_indices"][:6]], 1.0, np.eye(3), np.zeros(3))
        b = to_local(f, [int(i) for i in f["source_indices"][12:18]], 2.0, _rotation(6), np.array([3., 1, 0]))
        pred, rep = W.merge_windows([a, b])
        self.assertEqual(rep["components"], 2); self.assertFalse(rep["connected"])
        self.assertEqual(rep["links"][0]["reasons"], ["insufficient_shared_frames"])
        self.assertLess(rep["retained_capture_fraction"], 1.0)
        self.assertEqual(sorted(set(pred["component_ids"])), [0, 1])
        self.assertEqual(rep["frames_in_largest_component"], 6)
        # each component keeps its own local frame: identity is not a bridge and nothing is fabricated
        np.testing.assert_allclose(pred["extrinsics"][pred["component_ids"] == 1], b["extrinsics"])
        self.assertEqual(len(pred["source_indices"]), 12)

    def test_a_failed_middle_link_splits_the_chain_and_reports_membership(self):
        f = frames(); wins = make_windows(f, 8, 4)
        wins[2] = dict(wins[2], depth=np.random.default_rng(0).uniform(.5, 5, wins[2]["depth"].shape))      # garbage depth in one window
        pred, rep = W.merge_windows(wins)
        self.assertGreaterEqual(rep["components"], 2)
        self.assertEqual(sum(rep["component_frame_counts"].values()), len(pred["source_indices"]))
        self.assertLess(rep["retained_capture_fraction"], 1.0)

    def test_window_planning_keeps_half_window_overlap_and_covers_every_frame(self):
        for count, size, stride in ((120, 10, 5), (37, 8, 4), (24, 12, 6)):
            groups = W.make_windows(count, size, stride)
            self.assertEqual(sorted({i for g in groups for i in g}), list(range(count)))
            for a, b in zip(groups, groups[1:]):
                self.assertGreaterEqual(len(set(a) & set(b)), size // 2)
        self.assertEqual(P["window"], 6); self.assertEqual(P["stride"], 1)       # frozen defaults


class GraphMergeTests(unittest.TestCase):
    def test_dense_windows_recover_exact_geometry_and_cycle_residuals_vanish(self):
        f = frames(); wins = make_windows(f, 6, 1, seed=11)
        pred, rep = W.merge_window_graph(wins, max_neighbors=3)
        self.assertEqual(rep["mode"], "graph"); self.assertTrue(rep["connected"]); self.assertEqual(rep["retained_capture_fraction"], 1.0)
        self.assertGreater(rep["cycle_consistency"]["count"], 10)
        self.assertLess(rep["cycle_consistency"]["max_rotation_deg"], 1e-4)
        self.assertLess(rep["cycle_consistency"]["max_translation_fraction_of_extent"], 1e-6)
        self.assertLess(rep["cycle_consistency"]["max_scale_ratio_deviation"], 1e-6)
        truth = W.camera_center(f["extrinsics"]); order = [list(f["source_indices"]).index(i) for i in pred["source_indices"]]
        got = W.camera_center(pred["extrinsics"]); s, Q, b = umeyama_align(got, truth[order])
        np.testing.assert_allclose(s * got @ Q.T + b, truth[order], atol=1e-6 * np.linalg.norm(np.ptp(truth, axis=0)))

    def test_redundant_overlaps_bypass_a_corrupt_window_that_would_break_a_chain(self):
        f = frames(); wins = make_windows(f, 6, 1, seed=12)
        wins[9] = dict(wins[9], depth=np.random.default_rng(1).uniform(.5, 5, wins[9]["depth"].shape))
        chain, chain_rep = W.merge_windows(wins)
        graph, graph_rep = W.merge_window_graph(wins, max_neighbors=3)
        self.assertGreaterEqual(chain_rep["components"], 2)                    # a chain splits at the bad window
        self.assertIn(9, graph_rep["isolated_windows"])                        # the bad window agrees with nothing: reported, not bridged
        self.assertTrue(graph_rep["connected"]); self.assertEqual(graph_rep["frames_not_in_any_component"], [])   # neighbours still cover every frame
        self.assertGreaterEqual(graph_rep["links_rejected"], 1)

    def test_a_gap_no_window_overlaps_still_leaves_separate_components(self):
        f = frames(); a = make_windows(f, 6, 1, seed=13)[:5]; b = make_windows(f, 6, 1, seed=14)[12:17]
        pred, rep = W.merge_window_graph(a + b, max_neighbors=3)
        self.assertEqual(rep["components"], 2); self.assertFalse(rep["connected"]); self.assertLess(rep["retained_capture_fraction"], 1.0)
        self.assertEqual(sorted(set(pred["component_ids"].tolist())), [0, 1])

    def test_consensus_intrinsics_is_the_median_of_the_predicted_ones(self):
        f = frames(); wins = make_windows(f, 8, 4, seed=15)
        for k, w in enumerate(wins):
            K = w["intrinsics"].copy(); K[:, 0, 0] *= 1 + .1 * (k - len(wins) // 2); K[:, 1, 1] = K[:, 0, 0]; wins[k] = dict(w, intrinsics=K)
        med, rep = W.consensus_intrinsics(wins)
        self.assertAlmostEqual(med[0, 0], np.median(np.concatenate([w["intrinsics"][:, 0, 0] for w in wins])))
        self.assertGreater(rep["fx_max"], rep["fx_min"])
        pred, merge_rep = W.merge_windows(wins, shared_intrinsics=True)
        self.assertTrue((pred["intrinsics"] == med).all()); self.assertIsNotNone(merge_rep["shared_intrinsics"])


class AdaptivePlanningTests(unittest.TestCase):
    def test_in_plane_angle_is_recovered_from_rgb_and_translation_is_not_mistaken_for_roll(self):
        import cv2
        rng = np.random.default_rng(0); base = (rng.uniform(0, 255, (240, 320)) * 0 + cv2.GaussianBlur(rng.uniform(0, 255, (240, 320)).astype(np.uint8), (0, 0), 2)).astype(np.uint8)
        base = cv2.cvtColor(cv2.equalizeHist(base), cv2.COLOR_GRAY2BGR)
        rot = cv2.warpAffine(base, cv2.getRotationMatrix2D((160, 120), 12, 1.0), (320, 240))
        shifted = np.roll(base, 9, axis=1)
        turned = W.consecutive_inplane_angles([base, rot])[0]; moved = W.consecutive_inplane_angles([base, shifted])[0]
        self.assertAlmostEqual(abs(turned[0]), 12, delta=1.5); self.assertGreater(turned[1], 40)
        self.assertLess(abs(moved[0]), 2.0)                                    # a shift is not a rotation

    def test_planner_limits_cumulative_roll_requires_matches_and_keeps_half_overlap(self):
        n = 60; angles = [4.0] * (n - 1); matches = [200] * (n - 1)
        groups = W.plan_windows_adaptive(angles, matches, n, dict(W.ADAPTIVE))
        for g in groups:
            self.assertLessEqual(len(g) - 1, 7)                               # 7 steps x 4 deg stays under 30 deg
        for a, b in zip(groups, groups[1:]):
            self.assertGreaterEqual(len(set(a) & set(b)), len(a) // 2)
        self.assertEqual(sorted({i for g in groups for i in g}), list(range(n)))
        poor = W.plan_windows_adaptive([float("nan")] * (n - 1), [0] * (n - 1), n, dict(W.ADAPTIVE))
        self.assertTrue(all(len(g) == W.ADAPTIVE["min_frames"] for g in poor[:-1]))   # unmatchable motion: shortest window, no guessing
        still = W.plan_windows_adaptive([0.0] * (n - 1), [200] * (n - 1), n, dict(W.ADAPTIVE))
        self.assertTrue(all(len(g) <= W.ADAPTIVE["max_frames"] for g in still))


class GapFillingTests(unittest.TestCase):
    def test_uncovered_runs_and_bridging_windows_cover_the_gaps_only(self):
        ids = np.array([0] * 10 + [1] * 5); src = np.array(list(range(0, 5)) + list(range(8, 13)) + list(range(20, 25)))
        pred = {"component_ids": ids, "source_indices": src}
        runs = W.uncovered_runs(pred, 30)
        self.assertEqual(runs[0], (5, 7)); self.assertEqual(runs[-1][1], 29)
        self.assertIn((5, 7), runs)
        new = W.plan_gap_windows([(5, 7)], 30, [list(range(2, 8))])
        self.assertTrue(all(len(g) in W.GAP_SIZES for g in new))
        self.assertTrue(all(min(g) >= 1 and max(g) <= 11 for g in new))      # gap plus a 4-frame margin, nothing elsewhere
        self.assertNotIn(list(range(2, 8)), new)
        self.assertEqual(W.plan_gap_windows([(5, 7)], 30, new), [])           # already-run windows are not repeated

class NoGroundTruthTests(unittest.TestCase):
    def test_module_has_no_ground_truth_or_simulator_dependency(self):
        import ast, inspect
        tree = ast.parse(inspect.getsource(W))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertTrue(imported.isdisjoint({"simulator", "audit_camera_trajectory", "evaluation", "house_layouts"}))
        self.assertNotIn("private_ground_truth", inspect.getsource(W).replace("ground_truth_read", ""))


if __name__ == "__main__":
    unittest.main()

