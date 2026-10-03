import ast
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import depth_bias_analysis as dba
import simulator


TRUE_EDGES = np.array([6.5, 5.6, 6.5, 5.6])
HEIGHT = 3.1


class ScaleProtocolTests(unittest.TestCase):
    def test_pure_scale_error_is_removed_by_one_uniform_scale(self):
        k = 0.4                                             # prediction units
        pred = TRUE_EDGES * k / 1.03                        # every wall 3% long after the (a) scale below
        height_unscaled = HEIGHT * k                        # (a) scale = HEIGHT / h = 1 / k  -> lengths = true / 1.03
        out = dba.scale_protocols(pred, TRUE_EDGES, height_unscaled, HEIGHT, trajectory_scale=1 / k)
        self.assertAlmostEqual(out["a_ceiling_normalisation"]["scale"], 1 / k)
        self.assertLess(out["d_oracle_best_uniform_scale"]["max_abs_error_cm"], 1e-6)
        self.assertTrue(out["classification"]["reading"].startswith("scale_transfer"))
        self.assertAlmostEqual(out["scale_ratios"]["a_over_d"], 1 / 1.03, places=9)
        self.assertLess(out["c_oracle_single_wall_anchor"]["max_abs_error_cm_other_edges_worst"], 1e-6)
        self.assertAlmostEqual(out["scale_ratios"]["b_over_d"], 1 / 1.03, places=9)
        for rel in out["a_ceiling_normalisation"]["rel_error_percent"]:
            self.assertAlmostEqual(rel, 100 * (1 / 1.03 - 1), places=6)

    def test_horizontal_deformation_survives_the_best_uniform_scale(self):
        pred = TRUE_EDGES.copy()
        pred[[1, 3]] *= 1.05                                # short walls stretched by 5%, long walls exact
        out = dba.scale_protocols(pred, TRUE_EDGES, HEIGHT, HEIGHT, trajectory_scale=1.0)
        self.assertGreater(out["d_oracle_best_uniform_scale"]["max_abs_error_cm"], 100 * dba.UNIFORM_SCALE_TOL_M)
        self.assertTrue(out["classification"]["reading"].startswith("horizontal_deformation_remains"))
        an = out["horizontal_anisotropy"]["scale_ratio_longer_over_shorter_walls"]
        self.assertAlmostEqual(an, 1.05, places=6)           # long-wall scale 1.0 vs short-wall scale 1/1.05 -> 1.05
        self.assertLess(out["horizontal_anisotropy"]["group_fits"][0]["max_abs_error_cm_with_group_scale"], 1e-6)

    def test_single_anchor_scales_every_other_edge(self):
        pred = TRUE_EDGES * np.array([1.0, 1.02, 0.99, 1.01])
        out = dba.scale_protocols(pred, TRUE_EDGES, HEIGHT, HEIGHT, None)
        self.assertNotIn("b_oracle_trajectory_similarity", out)
        anchors = out["c_oracle_single_wall_anchor"]["anchors"]
        self.assertEqual(len(anchors), 4)
        for a in anchors:                                    # anchor edge is exact by construction
            self.assertAlmostEqual(a["error_m"][a["anchor_edge"]], 0.0, places=9)
        self.assertGreaterEqual(out["c_oracle_single_wall_anchor"]["max_abs_error_cm_other_edges_worst"],
                                out["c_oracle_single_wall_anchor"]["max_abs_error_cm_other_edges_best"])
        # the best uniform scale is never worse than the worst anchor on the max error
        self.assertLessEqual(out["d_oracle_best_uniform_scale"]["max_abs_error_cm"],
                             out["c_oracle_single_wall_anchor"]["max_abs_error_cm_other_edges_worst"] + 1e-9)

    def test_height_scale_error_is_pure_scale_transfer(self):
        pred = TRUE_EDGES / 2.0
        out = dba.scale_protocols(pred, TRUE_EDGES, HEIGHT / 2.0 * 1.04, HEIGHT, 2.0)    # DA3 height 4% too large -> walls 3.85% short
        self.assertLess(out["b_oracle_trajectory_similarity"]["max_abs_error_cm"], 1e-6)
        self.assertLess(out["d_oracle_best_uniform_scale"]["max_abs_error_cm"], 1e-6)
        self.assertGreater(out["a_ceiling_normalisation"]["max_abs_error_cm"], 10)
        self.assertAlmostEqual(out["scale_ratios"]["a_over_b"], 1 / 1.04, places=9)

    def test_group_by_true_length(self):
        self.assertEqual(sorted(map(sorted, dba.group_by_true_length(TRUE_EDGES))), [[0, 2], [1, 3]])

    def test_trajectory_similarity_recovers_scale(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(12, 3))
        ang = .7
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        true = 2.5 * pts @ R.T + np.array([1, 2, 3])
        sim = dba.similarity_from_centres(pts, true)
        self.assertAlmostEqual(sim["scale"], 2.5, places=9)
        self.assertLess(sim["rmse_m"], 1e-9)
        np.testing.assert_allclose(dba.apply_similarity(sim, pts), true, atol=1e-9)


# ------------------------------------------------------------------------------------------------- synthetic depth bias
def _w2c(pose, scale):
    right, up, forward = simulator.camera_basis(pose)
    R = np.stack([right, -up, forward])
    c = np.asarray(pose["position"], float) * scale
    return np.c_[R, -R @ c]


class Fixture:
    """True z-depth + labels from the real renderer on a tiny grid, 'DA3 frame' = true frame scaled by 1/2."""
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            warnings.simplefilter("ignore")
            scene = simulator.make_scene(17)
            poses = simulator.make_paths(scene, 120, 5, "spherical")[0]
            poses = [poses[i] for i in np.linspace(0, 119, 16, dtype=int)]
            width, height, fov = 128, 96, 78.0
            depth, labels, centres = [], [], []
            for p in poses:
                _, ray, intr, kind, _ = simulator.render(scene, p, width, height, fov, True, True)
                depth.append(dba.ray_to_zdepth(ray, intr, width, height)); labels.append(kind)
            true_z, labels = np.array(depth, np.float64), np.array(labels)
            scale = 0.5
            E = np.array([_w2c(p, scale) for p in poses])
            Ktrue = np.repeat(np.array([[[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]]), len(poses), axis=0)
            centres = np.array([p["position"] for p in poses])
            cls._cache = dict(scene=scene, poses=poses, true_z=true_z, labels=labels, E=E, Ktrue=Ktrue, intr=intr, scale=scale,
                              centres=centres, pred_centres=centres * scale,
                              dims=(scene["width"], scene["length"], scene["height"]))
        return cls._cache


def stats_for(pred_z, K=None):
    f = Fixture.get()
    return dba.depth_bias_stats(f["true_z"], pred_z, f["labels"], f["Ktrue"] if K is None else K, f["intr"]["fx"], f["intr"]["fy"], stride=1)


class DepthBiasAttributionTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture.get()
        self.base = self.f["true_z"] * self.f["scale"]          # perfect DA3 depth in its own units (global scale 2)
        rng = np.random.default_rng(1)
        self.rng = rng

    def flags(self, stats):
        return dba.attribute_depth_bias(stats)["flags_at_threshold"]

    def test_clean_depth_flags_nothing_and_recovers_global_scale(self):
        s = stats_for(self.base)
        self.assertAlmostEqual(s["global_scale_s_g_median"], 1 / self.f["scale"], places=6)
        self.assertFalse(any(self.flags(s).values()), dba.attribute_depth_bias(s))
        self.assertAlmostEqual(s["pooled_power_law_gamma_pred_vs_true"], 1.0, places=3)

    def test_range_dependent_shrink_is_attributed_to_range_only(self):
        pred = np.exp(0.85 * np.log(self.f["true_z"])) * self.f["scale"]       # far depth shrinks relative to near
        s = stats_for(pred)
        fl = self.flags(s)
        self.assertTrue(fl["range_dependent_depth_shrink"])
        self.assertFalse(fl["per_frame_scale_inconsistency"])
        self.assertFalse(fl["focal_length_error_fx"])
        self.assertLess(s["pooled_power_law_gamma_pred_vs_true"], 0.9)
        self.assertLess(s["effects_percent"]["range_dependent_percent_far_p90_vs_near_p10"], -3.0)    # far reads shallower

    def test_per_frame_scale_errors_are_attributed_to_frames_only(self):
        gains = np.exp(self.rng.normal(0, 0.06, size=len(self.base)))
        s = stats_for(self.base * gains[:, None, None])
        fl = self.flags(s)
        self.assertTrue(fl["per_frame_scale_inconsistency"])
        self.assertFalse(fl["range_dependent_depth_shrink"])
        self.assertFalse(fl["radial_depth_bias"])
        self.assertFalse(fl["focal_length_error_fx"])
        self.assertAlmostEqual(s["per_frame_scale"]["adjusted_sd_percent"], 100 * np.std(np.log(gains)), delta=1.0)

    def test_lateral_focal_error_is_attributed_to_the_intrinsics_only(self):
        K = self.f["Ktrue"].copy(); K[:, 0, 0] *= 1.04
        s = stats_for(self.base, K)
        fl = self.flags(s)
        self.assertTrue(fl["focal_length_error_fx"])
        self.assertAlmostEqual(s["intrinsics"]["fx_ratio_pred_over_true_mean_percent"], 4.0, places=6)
        self.assertFalse(fl["range_dependent_depth_shrink"]); self.assertFalse(fl["per_frame_scale_inconsistency"])
        self.assertFalse(fl["radial_depth_bias"])

    def test_radial_depth_bias_is_attributed_to_image_radius(self):
        f = self.f
        h, w = f["true_z"].shape[1:]
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - f["intr"]["cx"]) / f["intr"]["fx"]) ** 2 + ((yy - f["intr"]["cy"]) / f["intr"]["fy"]) ** 2
        s = stats_for(self.base * (1 + 0.15 * r2))
        fl = self.flags(s)
        self.assertTrue(fl["radial_depth_bias"])
        self.assertFalse(fl["per_frame_scale_inconsistency"]); self.assertFalse(fl["focal_length_error_fx"])
        self.assertGreater(s["effects_percent"]["radial_percent_edge_p90_vs_centre_p10"], 2.0)

    def test_ceiling_only_offset_is_a_surface_effect_not_range(self):
        pred = self.base.copy()
        pred[self.f["labels"] == 1] *= 0.95
        s = stats_for(pred)
        fl = self.flags(s)
        self.assertTrue(fl["surface_type_offset_max_abs"])
        self.assertAlmostEqual(s["effects_percent"]["surface_offset_percent_vs_wall"]["ceiling"], -5.0, delta=0.6)
        self.assertFalse(fl["range_dependent_depth_shrink"]); self.assertFalse(fl["per_frame_scale_inconsistency"])
        self.assertFalse(fl["focal_length_error_fx"])


class CloudExtentsAttributionTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture.get()
        self.base = self.f["true_z"] * self.f["scale"]

    def run_variants(self, pred_z, pred_K=None):
        f = self.f
        K = f["Ktrue"] if pred_K is None else pred_K
        st = dba.depth_bias_stats(f["true_z"], pred_z, f["labels"], K, f["intr"]["fx"], f["intr"]["fy"], stride=1)
        sim = dba.similarity_from_centres(f["pred_centres"], f["centres"])
        a, g = dba.fit_range_law(f["true_z"], pred_z, f["labels"])
        pf = np.array(st["per_frame_scale"]["adjusted_frame_log_gain"])
        var = dba.variant_extents(f["true_z"], pred_z, f["labels"], K, f["Ktrue"], f["E"], sim, f["dims"], st["global_scale_s_g_median"], pf, (a, g), stride=2)
        pts, labs = dba.unproject_world(pred_z, K, f["E"], f["labels"], 2, 0.0, ~dba.edge_mask(f["labels"]))
        base = dba.cloud_extents(dba.apply_similarity(sim, pts), labs, f["dims"])
        return base, var, dba.ratio_deviation_shares(base, var)

    def test_perfect_depth_gives_unit_ratios_and_a_pure_scale_change_does_not_matter(self):
        base, _, _ = self.run_variants(self.base)
        for key in ("height_ratio_pred_over_true", "width_ratio_pred_over_true", "length_ratio_pred_over_true"):
            self.assertAlmostEqual(base[key], 1.0, delta=0.01, msg=key)
        self.assertAlmostEqual(dba.mean_wall_ratio(base), 1.0, delta=0.01)

    def test_focal_error_is_removed_by_the_true_K_variant(self):
        K = self.f["Ktrue"].copy(); K[:, 0, 0] *= 1.08          # fx only: DA3 reported fx/fy != 1
        base, var, shares = self.run_variants(self.base, K)
        self.assertGreater(abs(dba.mean_wall_ratio(base) - 1), 0.004)
        self.assertGreater(shares["variants"]["V1_da3_depth_true_K"]["share_of_deviation_removed"], 0.8)
        self.assertGreater(shares["variants"]["V4_true_depth_true_K_da3_poses"]["share_of_deviation_removed"], 0.8)

    def test_ceiling_depth_offset_is_removed_by_the_surface_variant_not_by_K(self):
        pred = self.base.copy()
        pred[self.f["labels"] == 1] *= 0.93
        base, var, shares = self.run_variants(pred)
        self.assertGreater(abs(dba.mean_wall_ratio(base) - 1), 0.01)
        self.assertGreater(shares["variants"]["V6_surface_offsets_removed"]["share_of_deviation_removed"], 0.7)
        self.assertLess(abs(shares["variants"]["V1_da3_depth_true_K"]["share_of_deviation_removed"]), 0.3)


# ------------------------------------------------------------------------------------------------- helpers and audit
class HelperTests(unittest.TestCase):
    def test_ray_to_zdepth_matches_axis_and_corner_geometry(self):
        intr = {"fx": 100.0, "fy": 100.0, "cx": 49.5, "cy": 39.5}
        z = dba.ray_to_zdepth(np.full((80, 100), 2.0), intr, 100, 80)
        self.assertAlmostEqual(z[39, 49], 2.0 / np.sqrt(1 + (0.5 / 100) ** 2 + (0.5 / 100) ** 2), places=6)
        self.assertLess(z[0, 0], z[39, 49])

    def test_edge_mask_flags_label_boundaries_and_border(self):
        lab = np.zeros((1, 20, 20), int); lab[0, :, 10:] = 3
        m = dba.edge_mask(lab, 2)[0]
        self.assertTrue(m[10, 9] and m[10, 10] and m[10, 11])
        self.assertFalse(m[10, 5]); self.assertTrue(m[0, 5])

    def test_surface_class(self):
        cls = dba.surface_class(np.array([0, 1, 2, 5, 7, 8, 9]))
        self.assertEqual(cls.tolist(), [1, 2, 0, 0, -1, 3, 3])

    def test_spread_reports_sign_consistency(self):
        self.assertTrue(dba._spread([0.5, 1.5])["same_sign"])
        self.assertFalse(dba._spread([-0.5, 1.5])["same_sign"])
        self.assertIsNone(dba._spread([None]))

    def test_run_analysis_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                dba.run_analysis(Path(tmp))


class TruthIsolationTests(unittest.TestCase):
    def test_only_evaluation_functions_touch_truth(self):
        touching = dba.evaluation_only_call_audit()
        offenders = [name for name, hit in touching.items() if hit and name not in dba.EVALUATION_ONLY_FUNCTIONS]
        self.assertEqual(offenders, [])
        for pure in ("scale_protocols", "depth_bias_stats", "attribute_depth_bias", "cloud_extents", "variant_extents",
                     "transfer_summary", "ratio_deviation_shares", "fit_range_law"):
            self.assertFalse(touching[pure], pure)

    def test_simulator_is_imported_only_inside_the_evaluation_renderer(self):
        tree = ast.parse(Path(dba.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                self.assertNotIn("simulator", names)
        render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render_true_depth")
        self.assertIn("simulator", [a.name for n in ast.walk(render) if isinstance(n, ast.Import) for a in n.names])

    def test_audit_detects_an_injected_truth_read(self):
        bad = "def helper(path):\n    return read_text(path)\n"
        self.assertTrue(dba.evaluation_only_call_audit(bad)["helper"])

    def test_output_declares_oracle_use(self):
        self.assertTrue(any("ORACLE (b)" in c for c in dba.ORACLE_CHOICES))
        self.assertTrue(any("ORACLE (c)" in c for c in dba.ORACLE_CHOICES))
        self.assertTrue(any("ORACLE (d)" in c for c in dba.ORACLE_CHOICES))


if __name__ == "__main__":
    unittest.main()



