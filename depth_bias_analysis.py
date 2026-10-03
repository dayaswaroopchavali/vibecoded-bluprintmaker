"""EVALUATION-ONLY diagnosis of wall-length error: scale transfer vs horizontal deformation vs DA3 depth bias.

Reads FROZEN predictions, frozen wall boundaries and the evaluator outputs, then (and only then) reads the private scene and
private camera poses to (1) compare four scale protocols on the frozen vertical-like polygon edges, (2) compare DA3 depth with
the TRUE per-pixel z-depth rendered at the same poses, and (3) judge whether the scale and depth bias transfer across the
predictions. It never changes a reconstruction, never selects an estimator, and never marks a benchmark stage complete.

Every use of private truth is confined to the functions named in EVALUATION_ONLY_FUNCTIONS (a unit test enforces this); the pure
helpers (scale_protocols, depth_bias_stats, cloud_extents, ...) take plain arrays and cannot read files or the simulator.

ORACLE choices (all labelled in the output):
  * protocol (b): similarity fitted from the prediction's camera centres to the TRUE camera centres (trajectory only);
  * protocol (c): one wall's TRUE length used as the anchor (diagnostic);
  * protocol (d): best-fit uniform scale from ALL true wall lengths;
  * which scored hypothesis is "vertical-like": angle between its axis and the TRUE up direction seen through (b);
  * the true z-depth, true surface labels and true intrinsics used in the depth comparison.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np

import audit_camera_trajectory as audit
import estimate_height

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
UNIFORM_SCALE_TOL_M = 0.01          # "one uniform oracle scale makes all edges <= ~1 cm"
VERTICAL_LIKE_MAX_DEG = 15.0
SIGNIFICANT_PERCENT = 1.0           # effect sizes below this are reported as not significant
KIND_NAMES = {0: "floor", 1: "ceiling", 2: "wall_north", 3: "wall_east", 4: "wall_south", 5: "wall_west"}
EDGE_MASK_RADIUS = 2
STATS_STRIDE = 2
CLOUD_STRIDE = 4
EVALUATION_ONLY_FUNCTIONS = ("render_true_depth", "load_truth", "polygon_records", "analyze_prediction", "run_analysis", "main")
ORACLE_CHOICES = [
    "ORACLE (b): similarity fitted from the prediction's camera centres to the TRUE camera centres (trajectory only)",
    "ORACLE (c): a single wall's TRUE length is the anchor (labelled diagnostic, tried for every wall)",
    "ORACLE (d): one uniform scale best-fitted to all TRUE wall lengths",
    "ORACLE: 'vertical-like' hypothesis = scored hypothesis whose axis is within %.0f deg of the TRUE up seen through (b)" % VERTICAL_LIKE_MAX_DEG,
    "ORACLE: TRUE per-pixel z-depth, TRUE surface labels (floor/ceiling/walls/furniture) and TRUE intrinsics from the private scene and poses",
    "ORACLE: the counterfactual cloud variants replace parts of the DA3 output by truth-fitted quantities; they diagnose, they are not estimators",
]

CONFIGS = [
    {"label": "a_linspace36", "kind": "control_budget", "root": "wall-boundaries-v1",
     "scene": "blender-coverage-v3/capture/private_ground_truth/scene.json",
     "poses": "blender-coverage-v3/capture/private_ground_truth/path_00.json"},
    {"label": "b_stride3_40", "kind": "control_budget", "root": "wall-boundaries-v1",
     "scene": "texture-ablation-v1/control_original/private_ground_truth/scene.json",
     "poses": "texture-ablation-v1/control_original/private_ground_truth/path_00.json"},
    {"label": "c_stride2_60_DIFFERENT_BUDGET", "kind": "control_budget", "root": "wall-boundaries-v1",
     "scene": "texture-ablation-v1/control_original/private_ground_truth/scene.json",
     "poses": "texture-ablation-v1/control_original/private_ground_truth/path_00.json"},
    {"label": "d_textured40", "kind": "control_budget", "root": "wall-boundaries-v1",
     "scene": "texture-ablation-v1/control_textured/private_ground_truth/scene.json",
     "poses": "texture-ablation-v1/control_textured/private_ground_truth/path_00.json"},
    {"label": "path_01_36", "kind": "independent_path", "root": "wall-boundary-paths-01-02-v1",
     "scene": "blender-coverage-v3/capture/private_ground_truth/scene.json",
     "poses": "blender-coverage-v3/capture/private_ground_truth/path_01.json"},
    {"label": "path_02_36", "kind": "independent_path", "root": "wall-boundary-paths-01-02-v1",
     "scene": "blender-coverage-v3/capture/private_ground_truth/scene.json",
     "poses": "blender-coverage-v3/capture/private_ground_truth/path_02.json"},
]


# ------------------------------------------------------------------------------------------------- generic helpers
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _pct(x):
    return None if x is None else float(100.0 * x)


# ------------------------------------------------------------------------------------------------- (1) scale protocols
def group_by_true_length(true_m, rel_tol: float = 0.02) -> list[list[int]]:
    """Group edge indices whose true lengths agree within rel_tol (a rectangle gives its two wall directions)."""
    true = np.asarray(true_m, float)
    groups: list[list[int]] = []
    for i in np.argsort(true):
        if groups and abs(true[i] - np.mean(true[groups[-1]])) <= rel_tol * np.mean(true[groups[-1]]):
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def _errors(scale, pred, true):
    err = scale * pred - true
    return {"scale": float(scale), "error_m": err.tolist(), "rel_error_percent": (100 * err / true).tolist(),
            "max_abs_error_cm": float(100 * np.max(np.abs(err))),
            "mean_signed_rel_error_percent": float(np.mean(100 * err / true))}


def scale_protocols(pred_raw, true_m, pred_height_unscaled: float, true_height_m: float, trajectory_scale: float | None) -> dict:
    """Edge-length error under four scale protocols. pred_raw: edge lengths in prediction units, true_m: true metres.

    (a) 3.0 m ceiling normalisation == true_height / pred_height metres per unit (the truth ceiling only fixes the unit of the
        comparison; the prediction's scale comes from its own floor-ceiling separation);
    (b) ORACLE trajectory similarity scale; (c) ORACLE single known wall anchor (every wall tried); (d) ORACLE best uniform
    scale from all true wall lengths."""
    pred, true = np.asarray(pred_raw, float), np.asarray(true_m, float)
    out = {"a_ceiling_normalisation": _errors(true_height_m / pred_height_unscaled, pred, true)}
    if trajectory_scale is not None:
        out["b_oracle_trajectory_similarity"] = _errors(trajectory_scale, pred, true)
    anchors = []
    for k in range(len(pred)):
        row = _errors(true[k] / pred[k], pred, true)
        row["anchor_edge"] = k
        row["max_abs_error_cm_other_edges"] = float(100 * max((abs(e) for j, e in enumerate(row["error_m"]) if j != k), default=0.0))
        anchors.append(row)
    best = min(anchors, key=lambda r: r["max_abs_error_cm_other_edges"])
    worst = max(anchors, key=lambda r: r["max_abs_error_cm_other_edges"])
    out["c_oracle_single_wall_anchor"] = {
        "anchors": anchors, "best_anchor_edge": best["anchor_edge"], "worst_anchor_edge": worst["anchor_edge"],
        "scale_mean_over_anchors": float(np.mean([a["scale"] for a in anchors])),
        "max_abs_error_cm_other_edges_best": best["max_abs_error_cm_other_edges"],
        "max_abs_error_cm_other_edges_worst": worst["max_abs_error_cm_other_edges"],
        "note": "diagnostic: the anchor wall's own error is zero by construction, so only the other edges are scored"}
    s_d = float(np.sum(pred * true) / np.sum(pred * pred))
    d = _errors(s_d, pred, true)
    out["d_oracle_best_uniform_scale"] = d
    groups = group_by_true_length(true)
    group_fits = []
    for g in groups:
        sg = float(np.sum(pred[g] * true[g]) / np.sum(pred[g] ** 2))
        group_fits.append({"edges": g, "true_length_m": float(np.mean(true[g])), "scale": sg,
                           "max_abs_error_cm_with_group_scale": float(100 * np.max(np.abs(sg * pred[g] - true[g])))})
    aniso = None
    if len(group_fits) == 2:
        aniso = float(group_fits[1]["scale"] / group_fits[0]["scale"])        # longer-wall scale / shorter-wall scale
    out["horizontal_anisotropy"] = {"group_fits": group_fits, "scale_ratio_longer_over_shorter_walls": aniso,
                                    "note": "one scale per wall direction; a ratio far from 1 is horizontal deformation, not scale transfer"}
    resid_cm = d["max_abs_error_cm"]
    out["classification"] = {
        "uniform_scale_residual_cm": resid_cm, "tolerance_cm": 100 * UNIFORM_SCALE_TOL_M,
        "reading": ("scale_transfer: one uniform oracle scale brings every edge within tolerance" if resid_cm <= 100 * UNIFORM_SCALE_TOL_M
                    else "horizontal_deformation_remains: a uniform oracle scale leaves edges beyond tolerance")}
    s_a = out["a_ceiling_normalisation"]["scale"]
    implied = {"a_ceiling_normalisation": s_a, "d_oracle_best_uniform_scale": s_d,
               "c_mean_over_anchors": out["c_oracle_single_wall_anchor"]["scale_mean_over_anchors"]}
    if trajectory_scale is not None:
        implied["b_oracle_trajectory_similarity"] = float(trajectory_scale)
    out["implied_scales_metres_per_prediction_unit"] = implied
    out["scale_ratios"] = {"a_over_d": s_a / s_d,
                           "a_over_b": (s_a / trajectory_scale) if trajectory_scale else None,
                           "b_over_d": (trajectory_scale / s_d) if trajectory_scale else None}
    return out


# ------------------------------------------------------------------------------------------------- geometry helpers
def similarity_from_centres(pred_centres, true_centres) -> dict:
    """ORACLE (b): proper similarity pred -> true (true = scale * R x + t), trajectory only."""
    fit = audit.fit_similarity(pred_centres, true_centres)
    return {"scale": fit["scale"], "rotation": fit["rotation"], "translation": fit["translation"], "rmse_m": fit["rmse"],
            "max_error_m": fit["max_error"], "gt_extent_m": fit["gt_extent"],
            "rotation_well_conditioned": fit["alignment_rotation_well_conditioned"]}


def apply_similarity(sim, points):
    return sim["scale"] * (np.asarray(points, float) @ sim["rotation"].T) + sim["translation"]


def ray_to_zdepth(ray_t, intr, width, height):
    """Ray distance -> z-depth (as test_integration.synthetic_prediction does)."""
    yy, xx = np.mgrid[0:height, 0:width]
    return ray_t / np.sqrt(1 + ((xx - intr["cx"]) / intr["fx"]) ** 2 + ((yy - intr["cy"]) / intr["fy"]) ** 2)


def edge_mask(labels, radius: int = EDGE_MASK_RADIUS):
    """True where the surface label changes within `radius` pixels (silhouettes / corners / depth discontinuities)."""
    lab = np.asarray(labels)
    bad = np.zeros(lab.shape, bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(lab, dy, axis=-2), dx, axis=-1)
            bad |= shifted != lab
    bad[..., :radius, :] = True; bad[..., -radius:, :] = True; bad[..., :, :radius] = True; bad[..., :, -radius:] = True
    return bad


def surface_class(labels):
    """0 wall, 1 floor, 2 ceiling, 3 furniture, -1 other (labels: simulator hit kinds)."""
    lab = np.asarray(labels)
    cls = np.full(lab.shape, -1, np.int8)
    cls[(lab >= 2) & (lab <= 5)] = 0
    cls[lab == 0] = 1
    cls[lab == 1] = 2
    cls[lab >= 8] = 3
    return cls


# ------------------------------------------------------------------------------------------------- (2) depth bias
def _binned_slope(x, y, bins: int = 20):
    """Slope of the bin-median of y against the bin-median of x (quantile bins of x)."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            xs.append(np.median(x[m])); ys.append(np.median(y[m]))
    if len(xs) < 3 or np.ptp(xs) <= 1e-9:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def _group_table(values, group, names):
    rows = {}
    for gid, name in names.items():
        m = group == gid
        if m.sum() >= 50:
            rows[name] = {"n_pixels": int(m.sum()), "median_bias_percent": float(np.median(values[m]))}
    return rows


def depth_bias_stats(true_z, pred_z, labels, pred_intrinsics, true_fx: float, true_fy: float | None = None,
                     stride: int = STATS_STRIDE, seed: int = 0) -> dict:
    """DA3 z-depth vs true z-depth under ONE global scale s_g = median(true / pred) over all valid pixels.

    bias% = 100 * (s_g * pred / true - 1): positive = DA3 reads deeper than truth after the global scale.
    Everything else is reported relative to that single scale so per-frame and per-range effects stay visible.
    pred_intrinsics: N x 3 x 3 (DA3 predicted K at the prediction's pixel grid); true_fx at that same grid."""
    true_z, pred_z, labels = (np.asarray(a)[:, ::stride, ::stride] for a in (true_z, pred_z, labels))
    n, h, w = true_z.shape
    K = np.asarray(pred_intrinsics, float)
    mask = ~edge_mask(np.asarray(labels))
    valid = np.isfinite(true_z) & np.isfinite(pred_z) & (true_z > 0) & (pred_z > 0) & mask & (np.asarray(labels) >= 0)
    ratio = np.where(valid, true_z / np.where(pred_z > 0, pred_z, 1), np.nan)
    s_g = float(np.nanmedian(ratio[valid]))
    s_g_logmean = float(np.exp(np.nanmean(np.log(ratio[valid]))))
    frame_ids, per_frame_s, per_frame_true_med = [], [], []
    for i in range(n):
        m = valid[i]
        if m.sum() >= 50:
            frame_ids.append(i); per_frame_s.append(float(np.median(ratio[i][m]))); per_frame_true_med.append(float(np.median(true_z[i][m])))
    pf = np.asarray(per_frame_s)
    log_pf = np.log(pf / s_g)
    corr = float(np.corrcoef(np.log(per_frame_true_med), log_pf)[0, 1]) if len(pf) >= 3 and np.std(log_pf) > 1e-12 else None
    # bias per pixel (percent) relative to the ONE global scale
    bias = np.where(valid, 100.0 * (s_g / ratio - 1.0), np.nan)
    tv, bv = true_z[valid], bias[valid]
    yy, xx = np.mgrid[0:h, 0:w]
    full_h, full_w = h * stride, w * stride
    fx_t = float(true_fx); fy_t = float(true_fy if true_fy is not None else true_fx)
    cx, cy = (full_w - 1) / 2.0, (full_h - 1) / 2.0
    tx = (xx * stride - cx) / fx_t
    ty = (yy * stride - cy) / fy_t
    rad = np.broadcast_to(np.sqrt(tx ** 2 + ty ** 2), true_z.shape)[valid]
    absx = np.broadcast_to(np.abs(tx), true_z.shape)[valid]
    absy = np.broadcast_to(np.abs(ty), true_z.shape)[valid]
    cls = surface_class(labels)[valid]
    lab = np.asarray(labels)[valid]
    frame_of = np.broadcast_to(np.arange(n)[:, None, None], true_z.shape)[valid]
    # ---- range bins: terciles of the true z-depth
    q = np.quantile(tv, [0.0, 1 / 3, 2 / 3, 1.0])
    range_rows = {}
    for name, (lo, hi) in zip(("near", "mid", "far"), zip(q[:-1], q[1:])):
        m = (tv >= lo) & (tv <= hi)
        range_rows[name] = {"true_zdepth_m": [float(lo), float(hi)], "n_pixels": int(m.sum()),
                            "median_bias_percent": float(np.median(bv[m]))}
    # ---- surface type
    surf_rows = _group_table(bv, cls, {0: "wall", 1: "floor", 2: "ceiling", 3: "furniture"})
    wall_rows = _group_table(bv, lab, {2: "wall_north", 3: "wall_east", 4: "wall_south", 5: "wall_west"})
    # ---- image radius (tangent units; image corner ~ 1.0)
    rad_edges = [0.0, 0.3, 0.6, 0.8, 10.0]
    radial_rows = {}
    for lo, hi in zip(rad_edges[:-1], rad_edges[1:]):
        m = (rad >= lo) & (rad < hi)
        if m.sum() >= 50:
            radial_rows[f"{lo:.1f}-{min(hi, 1.1):.1f}"] = {"n_pixels": int(m.sum()), "median_bias_percent": float(np.median(bv[m]))}
    lateral_rows = {}
    for name, arr, edges in (("abs_x_tan", absx, [0, .25, .5, .75, 10]), ("abs_y_tan", absy, [0, .15, .3, .45, 10])):
        sub = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (arr >= lo) & (arr < hi)
            if m.sum() >= 50:
                sub[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = {"n_pixels": int(m.sum()), "median_bias_percent": float(np.median(bv[m]))}
        lateral_rows[name] = sub
    # ---- joint regression with per-frame fixed effects:  log(true/pred) = a_frame + k*log(true) + rho*r^2 + surface dummies
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tv), size=min(len(tv), 400_000), replace=False)
    y = np.log(ratio[valid])[pick]
    cols = {"log_true_depth": np.log(tv[pick]), "radius_sq": rad[pick] ** 2, "floor": (cls[pick] == 1).astype(float),
            "ceiling": (cls[pick] == 2).astype(float), "furniture": (cls[pick] == 3).astype(float)}
    X = np.column_stack(list(cols.values()))
    fid = frame_of[pick]
    fmean_y, fmean_X = {}, {}
    for i in np.unique(fid):                                    # frame fixed effects by demeaning
        m = fid == i
        fmean_y[int(i)] = float(y[m].mean()); fmean_X[int(i)] = X[m].mean(axis=0)
        X[m] -= X[m].mean(axis=0); y[m] -= y[m].mean()
    keep = X.std(axis=0) > 1e-12
    coef = np.zeros(X.shape[1])
    if keep.any():
        coef[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
    coef_d = dict(zip(cols, coef.tolist()))
    # frame intercepts a_f of log(true/pred) after range / radius / surface effects: the content-independent per-frame scale
    a_f = {i: fmean_y[i] - float(fmean_X[i] @ coef) for i in fmean_y}
    a_med = float(np.median(list(a_f.values())))
    adj_log_gain = np.array([a_f.get(i, a_med) - a_med for i in range(n)])           # 0 for frames without samples
    adj_sd = float(np.std([a_f[i] for i in a_f]))
    # effect sizes in percent of DA3 depth (pred = true/ratio, so a +coef on log true means DA3 shrinks with range)
    lt = np.log(tv)
    p10, p90 = np.quantile(lt, [0.1, 0.9])
    r10, r90 = np.quantile(rad ** 2, [0.1, 0.9])
    # depth bias of DA3 (after s_g) = pred*s/true - 1 = exp(-coef*dx) - 1 for a contrast dx in the regressor
    present = {"floor": 1, "ceiling": 2, "furniture": 3}
    effects = {
        "range_dependent_percent_far_p90_vs_near_p10": _pct(math.exp(-coef_d["log_true_depth"] * (p90 - p10)) - 1),
        "radial_percent_edge_p90_vs_centre_p10": _pct(math.exp(-coef_d["radius_sq"] * (r90 - r10)) - 1),
        "per_frame_scale_sd_percent": _pct(adj_sd),                         # frame intercepts after range/radius/surface effects
        "per_frame_scale_sd_percent_raw_median_based": _pct(float(np.std(log_pf))),    # content-dependent, for reference
        "per_frame_scale_range_percent": [_pct(float(np.exp(log_pf.min()) - 1)), _pct(float(np.exp(log_pf.max()) - 1))],
        "surface_offset_percent_vs_wall": {k: _pct(math.exp(-coef_d[k]) - 1) for k, v in present.items() if (cls == v).any()},
    }
    # pooled and within-frame power-law exponent  pred ~ true^gamma  (gamma = 1 - slope of log(true/pred) on log true)
    lr = np.log(ratio[valid])
    gamma_pooled = None
    slope_pooled = _binned_slope(np.log(tv), lr)
    if slope_pooled is not None:
        gamma_pooled = 1.0 - slope_pooled
    # intrinsics
    fx = K[:, 0, 0]; fy = K[:, 1, 1]
    intr = {"fx_pred_mean": float(fx.mean()), "fx_pred_sd": float(fx.std()), "fx_pred_min": float(fx.min()), "fx_pred_max": float(fx.max()),
            "fy_pred_mean": float(fy.mean()), "fy_pred_sd": float(fy.std()),
            "fx_true": fx_t, "fy_true": fy_t,
            "fx_ratio_pred_over_true_mean_percent": _pct(float(fx.mean()) / fx_t - 1),
            "fy_ratio_pred_over_true_mean_percent": _pct(float(fy.mean()) / fy_t - 1),
            "fx_over_fy_pred_mean": float((fx / fy).mean()),
            "cx_pred_mean": float(K[:, 0, 2].mean()), "cy_pred_mean": float(K[:, 1, 2].mean()),
            "cx_true_integer_pixel_convention": cx, "cy_true_integer_pixel_convention": cy}
    return {
        "global_scale_s_g_median": s_g, "global_scale_s_g_log_mean": s_g_logmean,
        "n_valid_pixels": int(valid.sum()), "n_frames": int(n), "stride": stride,
        "per_frame_scale": {"median_ratio_true_over_pred_by_frame": pf.tolist(), "sd_percent_of_s_g": _pct(float(np.std(log_pf))),
                            "adjusted_frame_log_gain": adj_log_gain.tolist(), "adjusted_sd_percent": _pct(adj_sd),
                            "min_percent_vs_s_g": _pct(float(np.exp(log_pf.min()) - 1)),
                            "max_percent_vs_s_g": _pct(float(np.exp(log_pf.max()) - 1)),
                            "corr_log_frame_scale_vs_frame_median_true_depth": corr},
        "bias_percent_by_true_depth_tercile": range_rows,
        "bias_percent_by_surface": surf_rows, "bias_percent_by_wall": wall_rows,
        "bias_percent_by_image_radius_tan": radial_rows, "bias_percent_by_lateral_offset": lateral_rows,
        "pooled_power_law_gamma_pred_vs_true": gamma_pooled,
        "frame_fixed_effects_regression": {
            "model": "log(true/pred) = a_frame + k*log(true_z) + rho*r^2 + surface dummies (walls baseline); positive k = DA3 shrinks with range",
            "coefficients": coef_d, "n_samples": int(len(pick))},
        "effects_percent": effects,
        "intrinsics": intr,
        "bias_percent_overall": {"median": float(np.median(bv)), "p05": float(np.percentile(bv, 5)), "p95": float(np.percentile(bv, 95))},
    }


def attribute_depth_bias(stats: dict, threshold_percent: float = SIGNIFICANT_PERCENT) -> dict:
    """Which depth-bias mechanisms exceed the threshold (percent of depth)? Pure function of depth_bias_stats."""
    e = stats["effects_percent"]
    intr = stats["intrinsics"]
    comp = {
        "range_dependent_depth_shrink": e["range_dependent_percent_far_p90_vs_near_p10"],
        "radial_depth_bias": e["radial_percent_edge_p90_vs_centre_p10"],
        "per_frame_scale_inconsistency": e["per_frame_scale_sd_percent"],
        "focal_length_error_fx": intr["fx_ratio_pred_over_true_mean_percent"],
    }
    surf = e["surface_offset_percent_vs_wall"]
    # structural surfaces only (floor/ceiling vs wall); furniture is reported but excluded (small, mask-sensitive, off the room shell)
    comp["surface_type_offset_max_abs"] = max((abs(v) for k, v in surf.items() if k in ("floor", "ceiling")), default=0.0)
    flags = {k: bool(abs(v) >= threshold_percent) for k, v in comp.items()}
    return {"effect_percent": comp, "flags_at_threshold": flags, "threshold_percent": threshold_percent}


# ------------------------------------------------------------------------------------------------- cloud extents (variants)
def unproject_world(depth, K, E, labels, stride: int = CLOUD_STRIDE, pixel_offset: float = 0.0, mask=None):
    """World points (prediction frame) of every `stride`-th valid pixel. K: N x 3 x 3, E: N x 3 x 4 world-to-camera."""
    depth = np.asarray(depth)[:, ::stride, ::stride]
    lab = np.asarray(labels)[:, ::stride, ::stride]
    n, h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx * stride + pixel_offset).astype(float); v = (yy * stride + pixel_offset).astype(float)
    pts, labs = [], []
    ok_all = np.isfinite(depth) & (depth > 0) & (lab >= 0)
    if mask is not None:
        ok_all &= np.asarray(mask)[:, ::stride, ::stride]
    for i in range(n):
        ok = ok_all[i]
        if not ok.any():
            continue
        z = depth[i][ok].astype(float)
        x = (u[ok] - K[i][0, 2]) / K[i][0, 0] * z
        y = (v[ok] - K[i][1, 2]) / K[i][1, 1] * z
        cam = np.stack([x, y, z], axis=1)
        R, t = np.asarray(E[i], float)[:, :3], np.asarray(E[i], float)[:, 3]
        pts.append((cam - t) @ R)                       # R^T (p - t) for row vectors
        labs.append(lab[i][ok])
    return np.concatenate(pts), np.concatenate(labs)


def cloud_extents(points_true_frame, labels, scene_dims) -> dict:
    """Median plane coordinates of the labelled surfaces in the TRUE frame -> extents and their ratios to the truth.

    scene_dims = (W along x, L along y, H along z). Walls: 2 north (+y), 3 east (+x), 4 south (-y), 5 west (-x)."""
    p = np.asarray(points_true_frame, float)
    lab = np.asarray(labels)
    W, L, H = scene_dims

    def med(mask, axis):
        return float(np.median(p[mask, axis])) if mask.sum() >= 50 else None
    c = {"floor_z": med(lab == 0, 2), "ceiling_z": med(lab == 1, 2), "north_y": med(lab == 2, 1), "east_x": med(lab == 3, 0),
         "south_y": med(lab == 4, 1), "west_x": med(lab == 5, 0)}
    out = {"plane_coordinates_m": c}
    if None in (c["floor_z"], c["ceiling_z"]):
        return out
    Hh = c["ceiling_z"] - c["floor_z"]
    out["height_extent_m"] = Hh
    out["height_ratio_pred_over_true"] = Hh / H
    if c["east_x"] is not None and c["west_x"] is not None:
        Wh = c["east_x"] - c["west_x"]
        out["width_x_extent_m"] = Wh; out["width_ratio_pred_over_true"] = Wh / W
        out["length_ratio_x_walls_under_ceiling_norm"] = (Wh / W) / (Hh / H)
    if c["north_y"] is not None and c["south_y"] is not None:
        Lh = c["north_y"] - c["south_y"]
        out["length_y_extent_m"] = Lh; out["length_ratio_pred_over_true"] = Lh / L
        out["length_ratio_y_walls_under_ceiling_norm"] = (Lh / L) / (Hh / H)
    return out


def variant_extents(true_z, pred_z, labels, pred_K, true_K, pred_E, sim, scene_dims, s_g, frame_log_gain, range_fit,
                    stride: int = CLOUD_STRIDE) -> dict:
    """ORACLE counterfactual clouds, all with DA3 poses and the trajectory similarity (b). The wall-length ratio under
    the ceiling normalisation (W_hat/W)/(H_hat/H) is what the polygon common length-ratio measures, so a variant that removes
    its deviation from 1 isolates the responsible ingredient. These are diagnostics, not estimators."""
    pred_z = np.asarray(pred_z, float); true_z = np.asarray(true_z, float)
    n = len(pred_z)
    mask = ~edge_mask(labels)
    frame_gain = np.exp(np.asarray(frame_log_gain, float))[:, None, None]         # equalise the per-frame scale (frame intercepts)
    # per-surface-class gain (ORACLE labels): makes each class's median true/pred equal to s_g (walls, floor, ceiling, furniture)
    cls = surface_class(labels)
    surface_gain = np.ones(pred_z.shape)
    ok = np.isfinite(true_z) & np.isfinite(pred_z) & (pred_z > 0) & mask
    for c in (0, 1, 2, 3):
        m = ok & (cls == c)
        if m.sum() >= 200:
            surface_gain[cls == c] = float(np.median(true_z[m] / pred_z[m])) / s_g
    a, gamma = range_fit                                                        # log pred = a + gamma*log true (pred units)
    with np.errstate(invalid="ignore", divide="ignore"):
        # invert the fitted law -> "true-depth-like" metres, then back to prediction units with s_g
        range_fixed_pred = (np.exp((np.log(pred_z) - a) / gamma) / s_g) if gamma else pred_z
    variants = {
        "V0_da3_depth_da3_K": (pred_z, pred_K),
        "V1_da3_depth_true_K": (pred_z, true_K),
        "V2_per_frame_scale_equalised": (pred_z * frame_gain, pred_K),
        "V3_range_dependence_removed": (range_fixed_pred, pred_K),
        "V4_true_depth_true_K_da3_poses": (true_z / s_g, true_K),
        "V5_V1_V2_V3_combined": ((np.exp((np.log(pred_z * frame_gain) - a) / gamma) / s_g) if gamma else pred_z * frame_gain, true_K),
        "V6_surface_offsets_removed": (pred_z * surface_gain, pred_K),
        "V7_V1_true_K_plus_V6_surface": (pred_z * surface_gain, true_K),
    }
    out = {}
    for name, (depth, K) in variants.items():
        with np.errstate(invalid="ignore"):
            pts, labs = unproject_world(depth, K, pred_E, labels, stride, 0.0, mask)
        out[name] = cloud_extents(apply_similarity(sim, pts), labs, scene_dims)
    return out


def fit_range_law(true_z, pred_z, labels):
    """ORACLE: log(pred) = a + gamma*log(true) over valid pixels (bin-median slope); returns (a, gamma)."""
    true_z, pred_z = np.asarray(true_z), np.asarray(pred_z)
    valid = np.isfinite(true_z) & np.isfinite(pred_z) & (pred_z > 0) & (true_z > 0) & (np.asarray(labels) >= 0) & ~edge_mask(labels)
    ltrue, lpred = np.log(true_z[valid][::7].astype(float)), np.log(pred_z[valid][::7].astype(float))
    slope = _binned_slope(ltrue, lpred) or 1.0
    return float(np.median(lpred - slope * ltrue)), float(slope)


def mean_wall_ratio(ext: dict):
    vals = [ext.get("length_ratio_x_walls_under_ceiling_norm"), ext.get("length_ratio_y_walls_under_ceiling_norm")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def ratio_deviation_shares(as_is: dict, variants: dict) -> dict:
    """Share of the cloud's common wall-length-ratio deviation from 1 removed by each counterfactual variant."""
    base = mean_wall_ratio(as_is)
    out = {"as_is_mean_ratio": base, "variants": {}}
    for name, ext in variants.items():
        r = mean_wall_ratio(ext)
        out["variants"][name] = {"mean_ratio": r, "share_of_deviation_removed": None if (base is None or r is None or abs(base - 1) < 1e-9) else (base - r) / (base - 1)}
    return out


# ------------------------------------------------------------------------------------------------- evaluation-only truth access
def load_truth(scene_path: Path, poses_path: Path) -> dict:
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    pose_doc = json.loads(Path(poses_path).read_text(encoding="utf-8"))
    return {"scene": scene, "poses": pose_doc["poses"], "intrinsics": pose_doc.get("intrinsics", {})}


def render_true_depth(truth: dict, indices, width: int, height: int, fov: float, cache: dict | None = None):
    """EVALUATION ONLY. TRUE z-depth + surface labels at the capture poses `indices`, rendered directly on the prediction's
    pixel grid (width x height, same horizontal fov), so no resampling of depth is needed. Ray distance -> z-depth."""
    import simulator
    depth, labels, centres = [], [], []
    intr = None
    for i in indices:
        key = (id(truth), int(i), width, height, round(fov, 9))
        if cache is not None and key in cache:
            z, kind, intr, pos = cache[key]
        else:
            pose = truth["poses"][int(i)]
            _, ray, intr, kind, _ = simulator.render(truth["scene"], pose, width, height, fov, True, True)
            z = ray_to_zdepth(ray, intr, width, height).astype(np.float32)
            pos = np.asarray(pose["position"], float)
            if cache is not None:
                cache[key] = (z, kind, intr, pos)
        depth.append(z); labels.append(kind); centres.append(pos)
    return np.array(depth), np.array(labels), intr, np.array(centres)


# ------------------------------------------------------------------------------------------------- per-prediction analysis
def load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def _fov_from(truth: dict, capture_width: int = 480) -> float:
    fx = truth["intrinsics"].get("fx")
    if fx:
        return 2 * math.degrees(math.atan(capture_width / (2 * fx)))
    return 78.0


def polygon_records(cfg: dict, sim: dict, truth: dict, scale_traj: float) -> list[dict]:
    root = OUTPUTS / cfg["root"]
    index = json.loads((root / "frozen_index.json").read_text(encoding="utf-8"))
    rec = next(b for b in index["budgets"] if b["label"] == cfg["label"])
    H = float(truth["scene"]["height"])
    up_pred = sim["rotation"].T @ np.array([0.0, 0.0, 1.0])          # ORACLE: true up seen in the prediction frame
    records = []
    for h in rec["hypotheses"]:
        if h.get("status") != "ran":
            continue
        ev_path = root / "evaluation" / f"{cfg['label']}_hyp{h['hypothesis_index']}.json"
        if not ev_path.exists():
            continue
        doc = json.loads(ev_path.read_text(encoding="utf-8"))
        ev = doc.get("evaluation") or {}
        if ev.get("status") != "scored":
            continue
        boundary = json.loads((root / h["files"]["boundary"]["path"]).read_text(encoding="utf-8"))
        verts = np.asarray(boundary["vertices_xy"], float)
        axis = np.asarray(h["axis_world"], float)
        angle = float(np.degrees(np.arccos(min(1.0, abs(float(axis @ up_pred)) / np.linalg.norm(axis) / np.linalg.norm(up_pred)))))
        edges = ev["metrics"]["edge_errors"]
        if not edges:                                   # evaluator matched no edges (e.g. a non-vertical hypothesis)
            records.append({"hypothesis": int(h["hypothesis_index"]), "vertical_like_oracle": False, "skipped": "no_matched_edges",
                            "iou_from_evaluator": ev["metrics"]["iou"], "oracle_angle_to_true_up_deg": angle})
            continue
        pscale = float(ev["scale"]["prediction_scale"]); tscale = float(ev["scale"]["truth_scale"])
        sep = float(ev["scale"]["prediction_height"]["separation_unscaled"])
        raw, true_m, mismatch = [], [], 0.0
        for e in edges:
            j = int(e["pred_edge"])
            length = float(np.linalg.norm(verts[(j + 1) % len(verts)] - verts[j]))
            mismatch = max(mismatch, abs(length * pscale - e["pred_length_m"]))
            raw.append(length)
            true_m.append(e["true_length_m"] / tscale)
        protocols = scale_protocols(raw, true_m, sep, H, scale_traj)
        records.append({
            "hypothesis": int(h["hypothesis_index"]), "axis_world": axis.tolist(),
            "oracle_angle_to_true_up_deg": angle, "vertical_like_oracle": bool(angle <= VERTICAL_LIKE_MAX_DEG),
            "iou_from_evaluator": ev["metrics"]["iou"], "edges": [{"pred_edge": e["pred_edge"], "true_edge": e["true_edge"]} for e in edges],
            "pred_edge_length_unscaled": raw, "true_edge_length_m": true_m, "pred_height_unscaled": sep, "true_height_m": H,
            "evaluator_prediction_scale": pscale, "evaluator_truth_scale": tscale,
            "consistency_with_evaluator_max_abs_normalised_m": mismatch,
            "protocols": protocols})
    return records


def analyze_prediction(cfg: dict, cache: dict | None = None, fx_resolution_note: bool = True) -> dict:
    """EVALUATION ONLY: all analyses for one frozen prediction."""
    root = OUTPUTS / cfg["root"]
    index = json.loads((root / "frozen_index.json").read_text(encoding="utf-8"))
    rec = next(b for b in index["budgets"] if b["label"] == cfg["label"])
    pred_path = Path(rec["prediction"])
    pred = load_prediction(pred_path)
    src = [int(i) for i in pred.get("source_indices", rec["source_indices"])]
    truth = load_truth(OUTPUTS / cfg["scene"], OUTPUTS / cfg["poses"])
    n, h, w = pred["depth"].shape
    fov = _fov_from(truth)
    true_z, labels, intr, true_centres = render_true_depth(truth, src, w, h, fov, cache)
    pred_centres = estimate_height.camera_centers(pred["extrinsics"])
    sim = similarity_from_centres(pred_centres, true_centres)
    stats = depth_bias_stats(true_z, pred["depth"], labels, pred["intrinsics"], intr["fx"], intr["fy"])
    s_g = stats["global_scale_s_g_median"]
    attribution = attribute_depth_bias(stats)
    true_K = np.repeat(np.array([[[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.0]]]), n, axis=0)
    scene = truth["scene"]
    dims = (float(scene["width"]), float(scene["length"]), float(scene["height"]))
    # truth-fitted range law for the counterfactual only: log pred = a + gamma log true  (pred units: depth/s)
    a, slope = fit_range_law(true_z, pred["depth"], labels)
    pf = np.array(stats["per_frame_scale"]["adjusted_frame_log_gain"])
    variants = variant_extents(true_z, pred["depth"], labels, pred["intrinsics"], true_K, pred["extrinsics"], sim, dims, s_g, pf, (a, slope))
    base_pts, base_labs = unproject_world(pred["depth"], pred["intrinsics"], pred["extrinsics"], labels, CLOUD_STRIDE, 0.0, ~edge_mask(labels))
    base = cloud_extents(apply_similarity(sim, base_pts), base_labs, dims)
    polygons = polygon_records(cfg, sim, truth, sim["scale"])
    return {
        "label": cfg["label"], "kind": cfg["kind"], "prediction": str(pred_path), "frames": int(n), "pred_resolution": [int(w), int(h)],
        "true_render_fov_deg": fov, "source_indices": src,
        "trajectory_similarity": {"scale_metres_per_prediction_unit": sim["scale"], "rmse_m": sim["rmse_m"], "max_error_m": sim["max_error_m"],
                                  "gt_extent_m": sim["gt_extent_m"], "rotation_well_conditioned": sim["rotation_well_conditioned"]},
        "depth_scale_vs_trajectory_scale": {"s_g_depth_median": s_g, "s_b_trajectory": sim["scale"], "s_g_over_s_b": s_g / sim["scale"],
                                            "note": "!=1 means DA3 depth units and DA3 camera-translation units are inconsistent"},
        "depth_bias": stats, "depth_bias_attribution": attribution,
        "range_law_used_for_counterfactual": {"a": a, "gamma": slope},
        "cloud_extents_as_is": base, "cloud_extents_counterfactuals": variants,
        "common_ratio_deviation_shares": ratio_deviation_shares(base, variants),
        "scene_dims_xyz_m": list(dims),
        "polygons": polygons,
    }


# ------------------------------------------------------------------------------------------------- (3) transfer + reporting
def _spread(values):
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return None
    return {"values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "same_sign": bool(np.all(v > 0) or np.all(v < 0)),
            "relative_range_percent_of_mean": float(100 * np.ptp(v) / abs(v.mean())) if v.mean() else None}


def transfer_summary(results: list[dict]) -> dict:
    """Is the scale / depth bias consistent across the predictions? Dimensionless quantities only (DA3's own unit is arbitrary)."""
    rows = {"ceiling_vs_trajectory_scale_a_over_b": [], "ceiling_vs_best_uniform_scale_a_over_d": [],
            "trajectory_vs_best_uniform_scale_b_over_d": [], "depth_scale_over_trajectory_scale": [],
            "uniform_scale_residual_cm_after_d": [], "common_length_ratio_a": [], "anisotropy_longer_over_shorter_wall_scale": []}
    for r in results:
        rows["depth_scale_over_trajectory_scale"].append(r["depth_scale_vs_trajectory_scale"]["s_g_over_s_b"])
        for p in r["polygons"]:
            if not p["vertical_like_oracle"]:
                continue
            pr = p["protocols"]
            rows["ceiling_vs_trajectory_scale_a_over_b"].append(pr["scale_ratios"]["a_over_b"])
            rows["ceiling_vs_best_uniform_scale_a_over_d"].append(pr["scale_ratios"]["a_over_d"])
            rows["trajectory_vs_best_uniform_scale_b_over_d"].append(pr["scale_ratios"]["b_over_d"])
            rows["uniform_scale_residual_cm_after_d"].append(pr["d_oracle_best_uniform_scale"]["max_abs_error_cm"])
            rows["common_length_ratio_a"].append(1 + pr["a_ceiling_normalisation"]["mean_signed_rel_error_percent"] / 100)
            rows["anisotropy_longer_over_shorter_wall_scale"].append(pr["horizontal_anisotropy"]["scale_ratio_longer_over_shorter_walls"])
    dep = {
        "global_scale_s_g_over_trajectory_s_b": rows["depth_scale_over_trajectory_scale"],
        "fx_ratio_pred_over_true_percent": [r["depth_bias"]["intrinsics"]["fx_ratio_pred_over_true_mean_percent"] for r in results],
        "per_frame_scale_sd_percent_adjusted": [r["depth_bias"]["effects_percent"]["per_frame_scale_sd_percent"] for r in results],
        "per_frame_scale_sd_percent_raw": [r["depth_bias"]["per_frame_scale"]["sd_percent_of_s_g"] for r in results],
        "range_dependent_percent_far_vs_near": [r["depth_bias"]["effects_percent"]["range_dependent_percent_far_p90_vs_near_p10"] for r in results],
        "radial_percent_edge_vs_centre": [r["depth_bias"]["effects_percent"]["radial_percent_edge_p90_vs_centre_p10"] for r in results],
        "floor_offset_percent_vs_wall": [r["depth_bias"]["effects_percent"]["surface_offset_percent_vs_wall"].get("floor") for r in results],
        "ceiling_offset_percent_vs_wall": [r["depth_bias"]["effects_percent"]["surface_offset_percent_vs_wall"].get("ceiling") for r in results],
        "overall_median_bias_percent_after_global_scale": [r["depth_bias"]["bias_percent_overall"]["median"] for r in results],
        "cloud_height_ratio_pred_over_true": [r["cloud_extents_as_is"].get("height_ratio_pred_over_true") for r in results],
        "cloud_x_wall_length_ratio_under_ceiling_norm": [r["cloud_extents_as_is"].get("length_ratio_x_walls_under_ceiling_norm") for r in results],
        "cloud_y_wall_length_ratio_under_ceiling_norm": [r["cloud_extents_as_is"].get("length_ratio_y_walls_under_ceiling_norm") for r in results],
    }
    names = list(results[0]["common_ratio_deviation_shares"]["variants"]) if results else []
    shares = {n: _spread([r["common_ratio_deviation_shares"]["variants"][n]["share_of_deviation_removed"] for r in results]) for n in names}
    return {"labels": [r["label"] for r in results], "scale_protocol_ratios_vertical_like_polygons": {k: _spread(v) for k, v in rows.items()},
            "variant_share_of_common_ratio_removed": shares,
            "depth_bias_quantities": {k: _spread(v) for k, v in dep.items()},
            "scope": "descriptive consistency check over 6 predictions of ONE room (house 17); no estimator is selected and nothing here "
                     "validates any scale for other houses. Predictions share a scene, so they are not independent houses."}


def markdown(result: dict) -> str:
    L = ["# Depth-bias / scale-transfer diagnosis (house 17, frozen predictions)", "",
         "EVALUATION-ONLY. Private truth used only after the predictions were frozen. No estimator is proposed; no stage is complete.", "",
         "## Oracle choices", ""] + [f"- {c}" for c in result["oracle_choices"]]
    L += ["", "## 1. Scale protocols on the vertical-like polygon edges", "",
          "Errors in true metres. (a) ceiling normalisation, (b) trajectory similarity, (c) single wall anchor (worst/best other-edge max error), (d) best uniform scale.", "",
          "| prediction | hyp | angle to true up (deg) | edge | true m | (a) err cm | (a) rel % | (b) err cm | (d) err cm | (c) best/worst anchor max cm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["predictions"]:
        for p in r["polygons"]:
            if not p["vertical_like_oracle"]:
                continue
            pr = p["protocols"]
            c = pr["c_oracle_single_wall_anchor"]
            for k, t in enumerate(p["true_edge_length_m"]):
                L.append(f"| {r['label']} | {p['hypothesis']} | {p['oracle_angle_to_true_up_deg']:.1f} | {p['edges'][k]['true_edge']} | {t:.3f} | "
                         f"{100*pr['a_ceiling_normalisation']['error_m'][k]:+.1f} | {pr['a_ceiling_normalisation']['rel_error_percent'][k]:+.2f} | "
                         f"{100*pr['b_oracle_trajectory_similarity']['error_m'][k]:+.1f} | {100*pr['d_oracle_best_uniform_scale']['error_m'][k]:+.1f} | "
                         f"{c['max_abs_error_cm_other_edges_best']:.1f} / {c['max_abs_error_cm_other_edges_worst']:.1f} |")
    L += ["", "### Implied scales (metres per prediction unit) and ratios", "",
          "| prediction | hyp | s_a ceiling | s_b trajectory | s_c mean anchors | s_d uniform | a/b | a/d | b/d | max err after (d) cm | aniso (long/short scale) | reading |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["predictions"]:
        for p in r["polygons"]:
            if not p["vertical_like_oracle"]:
                continue
            pr = p["protocols"]; im = pr["implied_scales_metres_per_prediction_unit"]; sr = pr["scale_ratios"]
            an = pr["horizontal_anisotropy"]["scale_ratio_longer_over_shorter_walls"]
            L.append(f"| {r['label']} | {p['hypothesis']} | {im['a_ceiling_normalisation']:.4f} | {im['b_oracle_trajectory_similarity']:.4f} | {im['c_mean_over_anchors']:.4f} | "
                     f"{im['d_oracle_best_uniform_scale']:.4f} | {sr['a_over_b']:.4f} | {sr['a_over_d']:.4f} | {sr['b_over_d']:.4f} | "
                     f"{pr['d_oracle_best_uniform_scale']['max_abs_error_cm']:.1f} | {an if an is None else round(an, 4)} | {pr['classification']['reading'].split(':')[0]} |")
    L += ["", "## 2. DA3 depth vs true z-depth (one global scale per capture)", "",
          "bias% = 100*(s_g*pred/true - 1) after ONE global scale s_g = median(true/pred) over all valid pixels. Frame fixed-effects regression "
          "separates range, radius and surface effects from the per-frame scale.", "",
          "| prediction | s_g | s_b (trajectory) | s_g/s_b | per-frame scale sd % (adjusted / raw) | range far-vs-near % | radial edge-vs-centre % | floor vs wall % | ceiling vs wall % | fx pred/true % | pooled gamma |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["predictions"]:
        d = r["depth_bias"]; e = d["effects_percent"]; s = e["surface_offset_percent_vs_wall"]
        L.append(f"| {r['label']} | {d['global_scale_s_g_median']:.4f} | {r['trajectory_similarity']['scale_metres_per_prediction_unit']:.4f} | "
                 f"{r['depth_scale_vs_trajectory_scale']['s_g_over_s_b']:.4f} | {e['per_frame_scale_sd_percent']:.2f} / {e['per_frame_scale_sd_percent_raw_median_based']:.2f} | "
                 f"{e['range_dependent_percent_far_p90_vs_near_p10']:+.2f} | {e['radial_percent_edge_p90_vs_centre_p10']:+.2f} | "
                 f"{s.get('floor', float('nan')):+.2f} | {s.get('ceiling', float('nan')):+.2f} | {d['intrinsics']['fx_ratio_pred_over_true_mean_percent']:+.2f} | "
                 f"{d['pooled_power_law_gamma_pred_vs_true']:.3f} |")
    L += ["", "### Median bias % by true depth tercile / surface / image radius", "",
          "| prediction | near | mid | far | wall | floor | ceiling | furniture | r 0-0.3 | 0.3-0.6 | 0.6-0.8 | >0.8 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def g(tab, key):
        v = tab.get(key)
        return "-" if v is None else f"{v['median_bias_percent']:+.2f}"
    for r in result["predictions"]:
        d = r["depth_bias"]
        rt, su, ra = d["bias_percent_by_true_depth_tercile"], d["bias_percent_by_surface"], d["bias_percent_by_image_radius_tan"]
        rk = list(ra)
        L.append(f"| {r['label']} | {g(rt, 'near')} | {g(rt, 'mid')} | {g(rt, 'far')} | {g(su, 'wall')} | {g(su, 'floor')} | {g(su, 'ceiling')} | {g(su, 'furniture')} | "
                 + " | ".join(g(ra, k) for k in rk[:4]) + " |")
    L += ["", "### Cloud extents under the trajectory similarity (ceiling-normalised wall-length ratio = (Wx/W)/(H/Hh); 1.000 = no horizontal-vs-vertical anisotropy)", "",
          "| prediction | variant | height ratio | x-wall ratio | y-wall ratio |", "|---|---|---|---|---|"]
    for r in result["predictions"]:
        allv = {"as_is": r["cloud_extents_as_is"], **r["cloud_extents_counterfactuals"]}
        for name, ex in allv.items():
            L.append(f"| {r['label']} | {name} | {ex.get('height_ratio_pred_over_true', float('nan')):.4f} | "
                     f"{ex.get('length_ratio_x_walls_under_ceiling_norm', float('nan')):.4f} | {ex.get('length_ratio_y_walls_under_ceiling_norm', float('nan')):.4f} |")
    L += ["", "### Share of the cloud's common wall-length-ratio deviation removed by each counterfactual", "",
          "| prediction | as-is mean ratio | " + " | ".join(n.split("_")[0] for n in result["predictions"][0]["common_ratio_deviation_shares"]["variants"]) + " |",
          "|---|---|" + "---|" * len(result["predictions"][0]["common_ratio_deviation_shares"]["variants"])]
    for r in result["predictions"]:
        sh = r["common_ratio_deviation_shares"]
        L.append(f"| {r['label']} | {sh['as_is_mean_ratio']:.4f} | " + " | ".join(
            "-" if v["share_of_deviation_removed"] is None else f"{100*v['share_of_deviation_removed']:.0f}%" for v in sh["variants"].values()) + " |")
    L += ["", "V0 as-is; V1 true K; V2 per-frame scale equalised; V3 range law removed; V4 TRUE depth+TRUE K (DA3 poses only); V5 V1+V2+V3; "
          "V6 per-surface depth offsets removed (oracle labels); V7 V1+V6.", "",
          "### Trajectory similarity (b) quality", "", "| prediction | rmse m | trajectory extent m | rmse / extent % |", "|---|---|---|---|"]
    for r in result["predictions"]:
        ts = r["trajectory_similarity"]
        L.append(f"| {r['label']} | {ts['rmse_m']:.3f} | {ts['gt_extent_m']:.2f} | {100*ts['rmse_m']/ts['gt_extent_m']:.1f} |")
    t = result["transfer"]
    L += ["", "## 3. Transfer across the 6 predictions (descriptive; no estimator selected)", "",
          "| quantity | mean | sd | min | max | same sign | values |", "|---|---|---|---|---|---|---|"]
    for sect in ("scale_protocol_ratios_vertical_like_polygons", "depth_bias_quantities"):
        for k, v in t[sect].items():
            if v is None:
                continue
            L.append(f"| {k} | {v['mean']:.4f} | {v['sd']:.4f} | {v['min']:.4f} | {v['max']:.4f} | {v['same_sign']} | {', '.join(f'{x:.3f}' for x in v['values'])} |")
    L += ["", "## Reading (generated from the numbers above)", ""] + [f"- {s}" for s in result["reading"]]
    L += ["", "## Limitations", ""] + [f"- {s}" for s in result["limitations"]]
    return "\n".join(L) + "\n"


def reading(results: list[dict], transfer: dict) -> list[str]:
    out = []
    pol = [(r["label"], p) for r in results for p in r["polygons"] if p["vertical_like_oracle"]]
    if pol:
        resid = [p["protocols"]["d_oracle_best_uniform_scale"]["max_abs_error_cm"] for _, p in pol]
        a_err = [p["protocols"]["a_ceiling_normalisation"]["max_abs_error_cm"] for _, p in pol]
        b_err = [p["protocols"]["b_oracle_trajectory_similarity"]["max_abs_error_cm"] for _, p in pol]
        out.append(f"Scale transfer: ceiling-normalised max edge error {min(a_err):.1f}-{max(a_err):.1f} cm; trajectory-similarity (b) {min(b_err):.1f}-{max(b_err):.1f} cm; "
                   f"best uniform oracle scale (d) leaves {min(resid):.1f}-{max(resid):.1f} cm (tolerance {100*UNIFORM_SCALE_TOL_M:.0f} cm). "
                   + ("One uniform scale removes the error: scale transfer." if max(resid) <= 100 * UNIFORM_SCALE_TOL_M else
                      "A uniform scale does not remove the error beyond tolerance: part of it is horizontal deformation, the rest is scale transfer."))
        aob = transfer["scale_protocol_ratios_vertical_like_polygons"]["ceiling_vs_trajectory_scale_a_over_b"]
        aod = transfer["scale_protocol_ratios_vertical_like_polygons"]["ceiling_vs_best_uniform_scale_a_over_d"]
        if aob:
            out.append(f"Ceiling-normalisation scale relative to the trajectory scale (a/b): mean {aob['mean']:.4f}, range {aob['min']:.4f}-{aob['max']:.4f}; "
                       f"relative to the best uniform wall scale (a/d): mean {aod['mean']:.4f}, range {aod['min']:.4f}-{aod['max']:.4f} "
                       f"({'same sign on every prediction' if aod['same_sign'] and (aod['min']-1)*(aod['max']-1) > 0 else 'sign differs across predictions'}).")
    dq = transfer["depth_bias_quantities"]
    sb = dq["global_scale_s_g_over_trajectory_s_b"]
    out.append(f"Depth scale vs pose scale: s_g/s_b = {sb['mean']:.4f} (range {sb['min']:.4f}-{sb['max']:.4f}); a value away from 1 means DA3 depth and DA3 camera translations carry different units.")
    ce, fl_ = dq["ceiling_offset_percent_vs_wall"], dq["floor_offset_percent_vs_wall"]
    if ce and fl_:
        out.append(f"Surface-type depth offset after the one global scale (frame fixed effects, walls = baseline): ceiling {ce['mean']:+.2f}% (range {ce['min']:+.2f} to {ce['max']:+.2f}, "
                   f"{'same sign on all predictions' if ce['same_sign'] else 'sign differs'}), floor {fl_['mean']:+.2f}% (range {fl_['min']:+.2f} to {fl_['max']:+.2f}, "
                   f"{'same sign' if fl_['same_sign'] else 'sign differs'}). Range-dependent shrink, radius and per-bin effects are below 1% except where flagged below.")
    v4 = [r["cloud_extents_counterfactuals"].get("V4_true_depth_true_K_da3_poses", {}) for r in results]
    dev = [abs(x - 1) for v in v4 for x in (v.get("length_ratio_x_walls_under_ceiling_norm"), v.get("length_ratio_y_walls_under_ceiling_norm")) if x is not None]
    if dev and pol:
        out.append(f"Remainder after the best uniform scale: {min(resid):.1f}-{max(resid):.1f} cm. With TRUE depth and TRUE K but DA3 poses (V4) the per-direction wall-length ratio still deviates "
                   f"from 1 by up to {100*max(dev):.2f}% (~{6.0*max(dev)*100:.1f} cm on a 6 m wall): the part that survives perfect depth is DA3 pose / trajectory deformation.")
    for r in results:
        att = r["depth_bias_attribution"]
        fl = [k for k, v in att["flags_at_threshold"].items() if v]
        out.append(f"{r['label']}: mechanisms >= {att['threshold_percent']:.1f}% of depth: {', '.join(fl) if fl else 'none'} "
                   f"({', '.join(f'{k}={v:+.2f}' for k, v in att['effect_percent'].items())}).")
    for r in results:
        sh = r["common_ratio_deviation_shares"]
        out.append(f"{r['label']}: depth-cloud wall-length ratio under ceiling normalisation as-is {sh['as_is_mean_ratio']:.4f}; share of the deviation removed by "
                   + "; ".join(f"{k.split('_')[0]} {100*v['share_of_deviation_removed']:.0f}%" for k, v in sh["variants"].items()
                               if v["share_of_deviation_removed"] is not None) + ".")
    shares = transfer.get("variant_share_of_common_ratio_removed", {})
    single = {k: v for k, v in shares.items() if v and k.split("_")[0] in ("V1", "V2", "V3", "V6")}
    if single:
        top = max(single, key=lambda k: single[k]["mean"])
        out.append(f"Largest single removable ingredient of the common ratio (V1 K, V2 per-frame, V3 range, V6 surface offsets): {top.split('_')[0]} "
                   f"({'_'.join(top.split('_')[1:])}) at {100*single[top]['mean']:.0f}% mean share; it is the depth-bias mechanism that explains most of the ratio.")
    out.append("Renderer check: the captures are Blender renders while the TRUE depth here is simulator.render on the same scene and poses; blender_capture.py places the floor slab top at z=0 "
               "and the ceiling slab underside at z=H (cube('floor',(0,0,-.07),h=.14), cube('ceiling',(0,0,H+.045),h=.09)) so the floor and ceiling planes coincide with the simulator's. "
               "A ceiling offset of several percent is therefore read as DA3 behaviour, with the residual caveat of unverified Blender lens/furniture equality.")
    if shares:
        out.append("Across predictions, mean share of the common ratio removed: " + "; ".join(
            f"{k.split('_')[0]} {100*v['mean']:.0f}% (range {100*v['min']:.0f} to {100*v['max']:.0f})" for k, v in shares.items() if v) +
            ". V4 (TRUE depth + TRUE K, DA3 poses) isolates what remains after depth and intrinsics are perfect: DA3 poses / trajectory scale.")
    return out


def run_analysis(output_dir: Path, configs=None) -> dict:
    configs = configs or CONFIGS
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    cache: dict = {}
    results = [analyze_prediction(cfg, cache) for cfg in configs]
    transfer = transfer_summary(results)
    result = {
        "model": "depth_bias_analysis", "evaluation_only": True, "stage_complete": False, "oracle_choices": ORACLE_CHOICES,
        "predictions": results, "transfer": transfer, "reading": reading(results, transfer),
        "limitations": [
            "ONE room (house 17) with 6 predictions that share one scene: consistency across them is not generalisation to other houses.",
            "Scale protocols (b)-(d) use truth by construction; only (a) is a truth-free scale. Nothing here selects or validates an estimator.",
            "True depth is rendered by simulator.render on the private scene at the capture poses; all six captures are Blender renders, and the Blender renderer's "
            "lens and furniture are assumed to equal the simulator's (floor/ceiling planes checked by reading blender_capture.py only, not by rendering Blender depth).",
            "DA3 predicted K is used with integer pixel indices; DA3's own half-pixel convention is not known (<= 0.2% lateral).",
            "Pixels near label boundaries (+-%d px), window apertures (no true depth) and invalid DA3 depth are excluded; statistics use every %d-th pixel, clouds every %d-th." % (EDGE_MASK_RADIUS, STATS_STRIDE, CLOUD_STRIDE),
            "Counterfactual variants replace DA3 quantities by truth-fitted ones; each reports what would change, not what is estimable without truth.",
            "Polygon edge matching and alignment come from the frozen evaluator (itself oracle-aligned); lengths here are re-read from the frozen boundary vertices."]}
    output_dir.mkdir(parents=True)
    with (output_dir / "results.json").open("x", encoding="utf-8") as f:
        json.dump(_clean(result), f, indent=1, allow_nan=False)
    with (output_dir / "reading.md").open("x", encoding="utf-8") as f:
        f.write(markdown(result))
    return result


def evaluation_only_call_audit(source: str | None = None) -> dict:
    """Static check: which top-level functions call simulator.* / read scene or pose files. Used by a unit test."""
    tree = ast.parse(source if source is not None else Path(__file__).read_text(encoding="utf-8"))
    sensitive_names = {"simulator", "load_truth", "render_true_depth", "read_text", "open", "polygon_records", "load_prediction"}
    touching = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != "evaluation_only_call_audit":
            hit = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id in sensitive_names:
                    hit = True
                elif isinstance(sub, ast.Attribute) and sub.attr in sensitive_names:
                    hit = True
                elif isinstance(sub, ast.Import) and any(a.name == "simulator" for a in sub.names):
                    hit = True
            touching[node.name] = hit
    return touching


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW folder; existing folders are not overwritten")
    args = parser.parse_args()
    run_analysis(args.output_dir)
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
