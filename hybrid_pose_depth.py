"""Truth-free hybrid prediction: direct-video poses with calibrated window depths.

For every shared RGB frame, a robust scalar is estimated from high-confidence
same-pixel direct/window depth ratios. That scalar puts the window depth map in
the direct model's frame-local scale. Rays are unprojected with the window's own
intrinsics, attached to the direct predicted camera pose, and reprojected with
the direct intrinsics using a nearest-depth splat. No GT, gravity, camera-up, or
scene-shape prior is used. Frames with weak ratio support remain present with
zero depth and confidence, marked unresolved in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def camera_centres(extrinsics: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsics, dtype=np.float64)
    return -np.einsum("nji,nj->ni", e[:, :3, :3], e[:, :3, 3])


def _project_rotation_to_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    d = np.eye(3)
    d[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    return u @ d @ vt


def fit_window_to_direct(window_extrinsics, direct_extrinsics, min_shared=3):
    """Fit X_direct = s Q X_window + b from same-frame predicted camera poses."""
    ew = np.asarray(window_extrinsics, dtype=np.float64)
    ed = np.asarray(direct_extrinsics, dtype=np.float64)
    if ew.shape != ed.shape or ew.ndim != 3 or ew.shape[1:] != (3, 4):
        raise ValueError("matching (N,3,4) extrinsics are required")
    n = len(ew)
    if n < min_shared:
        raise ValueError(f"only {n} shared poses; need at least {min_shared}")

    # R_direct = R_window Q.T, so each same-frame pair votes Q=R_direct.T R_window.
    q_votes = np.einsum("nji,njk->nik", ed[:, :3, :3], ew[:, :3, :3])
    Q = _project_rotation_to_so3(q_votes.mean(axis=0))
    cw, cd = camera_centres(ew), camera_centres(ed)
    x = (Q @ cw.T).T
    xc, yc = x.mean(axis=0), cd.mean(axis=0)
    dx, dy = x - xc, cd - yc
    denom = float(np.sum(dx * dx))
    span_w = float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))
    span_d = float(np.sqrt(np.mean(np.sum(dy * dy, axis=1))))
    s = float(np.sum(dx * dy) / denom) if denom > 1e-12 else float("nan")
    b = yc - s * xc if np.isfinite(s) else np.full(3, np.nan)

    ctr_res = np.linalg.norm((s * x + b) - cd, axis=1) if np.isfinite(s) else np.full(n, np.inf)
    rot_res = []
    for rw, rd in zip(ew[:, :3, :3], ed[:, :3, :3]):
        predicted = rw @ Q.T
        c = (np.trace(predicted.T @ rd) - 1.0) / 2.0
        rot_res.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
    center_rms = float(np.sqrt(np.mean(ctr_res ** 2)))
    # Scale is poorly observed when matched centres barely move. Report a warning,
    # rather than substituting an arbitrary scale or claiming a confident fit.
    warnings = []
    if not np.isfinite(s) or s <= 0:
        warnings.append("non_positive_or_undefined_scale")
    if span_w < 1e-4 or span_d < 1e-4:
        warnings.append("camera_center_baseline_too_small_for_scale")
    if np.isfinite(s) and span_w > 0 and center_rms / max(span_d, 1e-12) > 0.25:
        warnings.append("large_shared_center_fit_residual")
    if float(np.median(rot_res)) > 5.0:
        warnings.append("orientation_disagreement_over_5deg")
    return {
        "ok": bool(np.isfinite(s) and s > 0), "s": s, "Q": Q, "b": b,
        "shared_pose_count": n, "window_center_rms_span": span_w,
        "direct_center_rms_span": span_d, "center_rms_residual": center_rms,
        "center_rms_residual_fraction_of_direct_span": center_rms / max(span_d, 1e-12),
        "orientation_median_deg": float(np.median(rot_res)),
        "orientation_max_deg": float(np.max(rot_res)), "warnings": warnings,
    }


def estimate_depth_scale(window_depth, window_confidence, direct_depth, direct_confidence,
                         confidence_percentile=50.0, min_pixels=500):
    """Robust positive scalar mapping window depth values to direct depth scale.

    Same-pixel matching is valid because both predictions are for the identical
    RGB frame and raster. The fit uses only pixels passing each prediction's own
    confidence percentile, then measures dispersion in log-ratio space.
    """
    wd, wc = np.asarray(window_depth, float), np.asarray(window_confidence, float)
    dd, dc = np.asarray(direct_depth, float), np.asarray(direct_confidence, float)
    if wd.shape != dd.shape or wc.shape != wd.shape or dc.shape != dd.shape:
        raise ValueError("direct and window depth/confidence maps must have matching shapes")
    wv = np.isfinite(wd) & (wd > 0) & np.isfinite(wc)
    dv = np.isfinite(dd) & (dd > 0) & np.isfinite(dc)
    if wv.any():
        wv &= wc >= np.percentile(wc[wv], confidence_percentile)
    if dv.any():
        dv &= dc >= np.percentile(dc[dv], confidence_percentile)
    valid = wv & dv
    ratios = dd[valid] / wd[valid]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    out = {"ok": False, "scale": None, "shared_high_confidence_pixels": int(len(ratios)),
           "relative_mad": None, "ratio_median": None, "warnings": []}
    if len(ratios) < min_pixels:
        out["warnings"].append("too_few_shared_high_confidence_depth_pixels")
        return out
    logs = np.log(ratios)
    center = float(np.median(logs))
    mad_log = float(np.median(np.abs(logs - center))) * 1.4826
    # 1.4826*MAD in log space is a stable relative-dispersion approximation.
    rel_mad = float(np.expm1(mad_log))
    scale = float(np.exp(center))
    out.update({"ok": bool(np.isfinite(scale) and scale > 0), "scale": scale,
                "relative_mad": rel_mad, "ratio_median": scale,
                "ratio_p10": float(np.percentile(ratios, 10)),
                "ratio_p90": float(np.percentile(ratios, 90)),
                "log_mad_sigma": mad_log})
    return out


def reproject_depth(depth, confidence, source_K, source_E, direct_K, direct_E, s, Q, b):
    """Unproject with source K/E; Sim(3)-move; splat into target K/E, preserving nearest z."""
    d = np.asarray(depth, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    h, w = d.shape
    y, x = np.indices((h, w), dtype=np.float64)
    valid = np.isfinite(d) & (d > 0) & np.isfinite(conf)
    if not valid.any():
        return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), {"valid_source_fraction": 0.0, "valid_target_fraction": 0.0}

    pix = np.stack((x[valid], y[valid], np.ones(int(valid.sum()))), axis=1)
    cam_s = (pix @ np.linalg.inv(np.asarray(source_K, float)).T) * d[valid, None]
    same_pose_identity = (float(s) == 1.0 and np.allclose(Q, np.eye(3), atol=1e-12)
                          and np.allclose(b, 0.0, atol=1e-12)
                          and np.allclose(source_E, direct_E, atol=1e-12))
    if same_pose_identity:
        cam_d = cam_s
    else:
        Rs, ts = np.asarray(source_E, float)[:, :3], np.asarray(source_E, float)[:, 3]
        world_s = (cam_s - ts) @ Rs
        world_d = (float(s) * (world_s @ np.asarray(Q, float).T)) + np.asarray(b, float)
        Rd, td = np.asarray(direct_E, float)[:, :3], np.asarray(direct_E, float)[:, 3]
        cam_d = world_d @ Rd.T + td
    z = cam_d[:, 2]
    keep = np.isfinite(z) & (z > 1e-6)
    uvh = cam_d[keep] @ np.asarray(direct_K, float).T
    uv = uvh[:, :2] / uvh[:, 2:3]
    u, v = uv[:, 0], uv[:, 1]
    cc = conf[valid][keep]

    # Bilinear forward splat; every candidate competes by target-camera depth.
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    flat_depth = np.full(h * w, np.inf, np.float64)
    flat_conf = np.zeros(h * w, np.float64)
    z = z[keep]
    weights = ((1 - (u-u0)) * (1 - (v-v0)), (u-u0) * (1 - (v-v0)),
               (1 - (u-u0)) * (v-v0), (u-u0) * (v-v0))
    offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    for wt, (ox, oy) in zip(weights, offsets):
        xi, yi = u0 + ox, v0 + oy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (wt > 1e-6)
        idx = yi[ok] * w + xi[ok]
        np.minimum.at(flat_depth, idx, z[ok])
    # Confidence belongs to the nearest visible splat (ties allow multiple samples).
    for wt, (ox, oy) in zip(weights, offsets):
        xi, yi = u0 + ox, v0 + oy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (wt > 1e-6)
        idx = yi[ok] * w + xi[ok]
        nearest = z[ok] <= flat_depth[idx] * (1.0 + 1e-6) + 1e-7
        np.maximum.at(flat_conf, idx[nearest], cc[ok][nearest])
    out = flat_depth.reshape(h, w)
    out_conf = flat_conf.reshape(h, w)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32), out_conf.astype(np.float32), {
        "valid_source_fraction": float(valid.mean()),
        "valid_target_fraction": float(np.count_nonzero(out) / (h * w)),
    }


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_hybrid(direct_path, window_prediction_path, pool_dir, output_dir, max_scale_relative_mad=0.10):
    direct_path, window_prediction_path, pool_dir, output_dir = map(Path, (direct_path, window_prediction_path, pool_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    direct = np.load(direct_path, allow_pickle=False)
    selected = np.load(window_prediction_path, allow_pickle=False)
    # NPZ members are compressed. Cache arrays before frame iteration so each
    # 120-frame key is decompressed only once, rather than once per frame lookup.
    direct_depth = np.asarray(direct["depth"])
    direct_confidence = np.asarray(direct["confidence"])
    direct_intrinsics = np.asarray(direct["intrinsics"])
    direct_extrinsics = np.asarray(direct["extrinsics"])
    direct_images = np.asarray(direct["images"])
    direct_source_indices = np.asarray(direct["source_indices"])
    selected_depth = np.asarray(selected["depth"])
    selected_confidence = np.asarray(selected["confidence"])
    selected_intrinsics = np.asarray(selected["intrinsics"])
    selected_extrinsics = np.asarray(selected["extrinsics"])
    selected_source_indices = np.asarray(selected["source_indices"])
    selected_components = np.asarray(selected["component_ids"], dtype=np.int64) if "component_ids" in selected.files else np.zeros(len(selected_source_indices), np.int64)
    selected_source_window = np.asarray(selected["source_window"], dtype=np.int64) if "source_window" in selected.files else np.full(len(selected_source_indices), -1, np.int64)
    pool_files = sorted(pool_dir.glob("*.npz"))
    if not pool_files:
        raise FileNotFoundError(f"no window files under {pool_dir}")
    direct_by_id = {int(f): i for i, f in enumerate(direct_source_indices)}
    sel_by_id = {int(f): i for i, f in enumerate(selected_source_indices)}
    if len(direct_by_id) != len(direct_source_indices):
        raise ValueError("direct prediction contains duplicate source indices")
    if len(sel_by_id) != len(selected_source_indices):
        raise ValueError("window prediction contains duplicate source indices; refusing to fuse components implicitly")
    if set(direct_by_id) != set(sel_by_id):
        raise ValueError("direct and selected window predictions must contain the same source indices")

    ids = np.asarray(direct_source_indices, dtype=np.int64)
    out_depth, out_conf, used_window, used_component, frame_stats = [], [], [], [], []
    for f in ids:
        di, si = direct_by_id[int(f)], sel_by_id[int(f)]
        comp = int(selected_components[si])
        scale_fit = estimate_depth_scale(selected_depth[si], selected_confidence[si],
                                         direct_depth[di], direct_confidence[di])
        resolved = bool(scale_fit["ok"] and scale_fit["relative_mad"] <= max_scale_relative_mad)
        if scale_fit["ok"] and not resolved:
            scale_fit["warnings"].append("depth_scale_ratio_dispersion_over_limit")
        if resolved:
            scaled_source_depth = np.asarray(selected_depth[si], dtype=np.float32) * scale_fit["scale"]
            # The scalar is anchored directly to the direct model's per-frame
            # depth scale. Source K supplies the window's RGB rays; direct E and
            # direct K define this hybrid frame's camera coordinate and raster.
            depth, conf, stats = reproject_depth(
                scaled_source_depth, selected_confidence[si], selected_intrinsics[si], direct_extrinsics[di],
                direct_intrinsics[di], direct_extrinsics[di], 1.0, np.eye(3), np.zeros(3))
        else:
            # Preserve the frame and direct pose, but never guess its scale.
            h, w = selected_depth.shape[-2:]
            depth, conf = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
            stats = {"valid_source_fraction": 0.0, "valid_target_fraction": 0.0, "unresolved_scale": True}
        out_depth.append(depth); out_conf.append(conf)
        used_window.append(int(selected_source_window[si]))
        used_component.append(comp)
        frame_stats.append({"source_index": int(f), "window": used_window[-1], "component": comp,
                            "scale_resolved": resolved, "depth_scale_fit": scale_fit, **stats})

    # Pool focal statistics are read one compressed file at a time; depth/image
    # arrays from all overlapping windows are never retained in memory.
    fx_per_window = []
    pool_focal_report = []
    for p in pool_files:
        with np.load(p, allow_pickle=False) as z:
            fx = np.asarray(z["intrinsics"][:, 0, 0], dtype=np.float64)
            fx_per_window.extend(fx.tolist())
            pool_focal_report.append({"file": p.name, "fx_min": float(fx.min()), "fx_median": float(np.median(fx)), "fx_max": float(fx.max())})

    output_dir.mkdir(parents=True)
    pred_path = output_dir / "prediction.npz"
    np.savez_compressed(pred_path,
        depth=np.stack(out_depth), confidence=np.stack(out_conf),
        intrinsics=direct_intrinsics, extrinsics=direct_extrinsics,
        images=direct_images, source_indices=ids,
        source_window=np.asarray(used_window, dtype=np.int64), component_ids=np.asarray(used_component, dtype=np.int64))
    focal_window = np.asarray(fx_per_window, dtype=np.float64)
    focal_direct = direct_intrinsics[:, 0, 0]
    manifest = {
        "kind": "truth_free_hybrid_direct_poses_window_depths",
        "truth_free": True, "ground_truth_read": False, "camera_up_or_shape_prior_used": False,
        "method": "per-frame robust median ratio from shared high-confidence direct/window depth pixels; scalar applied to the window depth map; source-intrinsic ray unprojection; direct-pose attachment; direct-intrinsics bilinear z-buffer reprojection",
        "inputs": {"direct_prediction": str(direct_path.resolve()), "direct_prediction_sha256": _sha256(direct_path),
                   "window_prediction": str(window_prediction_path.resolve()), "window_prediction_sha256": _sha256(window_prediction_path),
                   "window_pool": str(pool_dir.resolve()), "window_pool_file_count": len(pool_files),
                   "window_pool_filenames": [p.name for p in pool_files]},
        "output_prediction_sha256": _sha256(pred_path),
        "diagnostics": {"window_count": len(pool_files), "frames": len(ids), "component_count": len(set(used_component)),
                        "window_focal_fx_min": float(focal_window.min()),
                        "window_focal_fx_median": float(np.median(focal_window)), "window_focal_fx_max": float(focal_window.max()),
                        "direct_focal_fx_min": float(focal_direct.min()), "direct_focal_fx_median": float(np.median(focal_direct)),
                        "direct_focal_fx_max": float(focal_direct.max()), "per_pool_window_focal_fx": pool_focal_report,
                        "scale_calibration": {"method": "median of direct_depth/window_depth ratios in log space",
                                              "confidence_percentile_per_prediction": 50.0,
                                              "minimum_shared_high_confidence_pixels": 500,
                                              "max_relative_mad": float(max_scale_relative_mad)},
                        "max_scale_relative_mad": float(max_scale_relative_mad),
                        "resolved_frame_count": int(sum(bool(x["scale_resolved"]) for x in frame_stats)),
                        "unresolved_frame_count": int(sum(not bool(x["scale_resolved"]) for x in frame_stats)),
                        "valid_target_fraction_median": float(np.median([x["valid_target_fraction"] for x in frame_stats])),
                        "valid_target_fraction_min": float(np.min([x["valid_target_fraction"] for x in frame_stats])),
                        "per_frame_reprojection": frame_stats},
        "warnings": sorted({w for frame in frame_stats for w in frame["depth_scale_fit"]["warnings"]}),
        "unresolved_scale_frames": [int(x["source_index"]) for x in frame_stats if not x["scale_resolved"]],
        "reprojection_approximation": "Window focal variation is handled by unprojecting each selected frame with its own window K and reprojecting through direct-frame K. Forward bilinear splat leaves unsupported target pixels at depth/confidence zero. The direct prediction supplies only a robust per-frame scalar calibration and direct poses/intrinsics; no direct depth pixels are copied into output.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pred_path, manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direct", required=True, type=Path, help="direct full-video prediction.npz (poses and target intrinsics)")
    ap.add_argument("--window-prediction", required=True, type=Path, help="refined windowed prediction.npz (central-window frame selection)")
    ap.add_argument("--window-pool", required=True, type=Path, help="directory of raw window prediction NPZs (used for focal diagnostics)")
    ap.add_argument("--output", required=True, type=Path, help="new output directory; existing directories are refused")
    ap.add_argument("--max-scale-relative-mad", type=float, default=0.10,
                    help="flag/zero frames whose robust direct/window scale ratio has larger relative MAD (default: 0.10)")
    args = ap.parse_args(argv)
    pred, report = build_hybrid(args.direct, args.window_prediction, args.window_pool, args.output,
                                max_scale_relative_mad=args.max_scale_relative_mad)
    d = report["diagnostics"]
    print(json.dumps({"prediction": str(pred), "frames": d["frames"], "windows": d["window_count"],
                      "median_target_valid_fraction": d["valid_target_fraction_median"],
                      "unresolved_frames": d["unresolved_frame_count"],
                      "warnings": report["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
