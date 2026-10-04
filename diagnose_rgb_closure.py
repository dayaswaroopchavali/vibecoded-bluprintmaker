"""Diagnosis of WHY the frozen DA3 (RGB) wall boundaries stay open, on the 60 frozen RGB runs.

Read-only analysis (no fix, no existing module edited, no GPU, no training, no download). Inputs are the frozen chain outputs of
`rgb-batch-upright-v2` and `rgb-batch-realistic-v1` (houses/<h>/<path>/axis/frozen/...) and the frozen DA3 predictions next to them.

Stages (each refuses to overwrite; the output folder must be new):
  diagnose  TRUTH-FREE. Per run: every candidate's frozen wall lines / boundary failure reason; for the truth-free top-ranked
            candidate(s) and the truth-free vertical-flag candidate (cap-pair separation band): line stats, layered (drifted
            copy) pairs, fragment_association report, re-extraction check against the frozen walls, probe (g) with only the
            highest-confidence 50 % pixels per frame (one probe, not tuned), pairwise depth reprojection consistency, camera
            path extent, view coverage and DA3 confidence statistics.
  control   TRUTH-FREE. Per house (path 0): the DA3 prediction vs the simulated-LiDAR capture of the same path, through the SAME
            axis-candidate + wall_extraction + assemble_boundary_ray chain, plus the two hybrid cells (DA3 depth with sensor poses,
            LiDAR depth with DA3 poses; scale from the camera-centre similarity of the two public captures): a 2 x 2 factorial.
  freeze    sha256 of every truth-free output -> freeze_index.json.
  oracle    EVALUATOR-ONLY, after freeze (hashes verified first): layout class, true up axis in the prediction frame, observed-line
            offset error (cm) after an oracle similarity, per-wall coverage, capture QA labels.
  summarize correlations (Spearman) and tables.

    python diagnose_rgb_closure.py diagnose --output-dir <NEW dir under outputs/sim-benchmark>
    python diagnose_rgb_closure.py control  --output-dir <same dir>
    python diagnose_rgb_closure.py freeze   --output-dir <same dir>
    python diagnose_rgb_closure.py oracle   --output-dir <same dir>
    python diagnose_rgb_closure.py summarize --output-dir <same dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
BATCHES = ("rgb-batch-upright-v2", "rgb-batch-realistic-v1")
LIDAR_ROOT = "lidar-batch-10houses-widefix-v2"
RAY_K = 2.0
PROBE_CONF_PERCENTILE = 50.0          # keep the highest-confidence 50 % pixels per frame (default of the chain keeps 60 %)
LAYER_TOL_MULTIPLE = 3.0              # same convention as fragment_association tol_noise_multiple
DRIFT_FRACTION = 0.12                 # fragment_association drift allowance (separation / extent)
TRUTH_MODULES = ("simulator", "evaluate_floorplan", "evaluation", "house_layouts", "capture_lidar_sim", "blender_capture",
                 "compare_wall_boundaries", "run_axis_candidates", "run_rgb_batch")

# Functions that make up the truth-free inference part (AST test: no truth import / name / path in any of them).
TRUTH_FREE_FUNCTIONS = (
    "sha256_file", "write_new_json", "read_json", "jsonable", "list_runs", "top_ranked_ids", "vertical_flag_id", "classify_failure",
    "line_stats", "layered_pairs", "layered_summary", "camera_centres", "cams_in_frame", "confidence_stats", "path_extent_stats",
    "view_coverage", "depth_consistency", "fragment_summary", "probe_confidence", "analyze_candidate", "candidate_meta",
    "reproduction_check", "diagnose_run", "run_chain", "similarity_from_centres", "hybrid_predictions", "control_house",
    "lines_compact", "stage_diagnose", "stage_control", "stage_freeze",
)


# --------------------------------------------------------------------------------------------------------------- utilities
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_new_json(path, obj) -> str:
    """Write JSON; refuses to overwrite. Returns the sha256 of the file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8")
    return sha256_file(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _median(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


# ------------------------------------------------------------------------------------------------- run discovery / ranking
def list_runs(outputs_root=OUTPUTS, batches=BATCHES):
    runs = []
    for batch in batches:
        for run_dir in sorted((Path(outputs_root) / batch / "houses").glob("house_*/path_*")):
            index_path = run_dir / "axis" / "frozen_index.json"
            if not index_path.exists():
                continue
            house = int(run_dir.parent.name.split("_")[1])
            path_i = int(run_dir.name.split("_")[1])
            runs.append({"batch": batch, "run_dir": str(run_dir), "house": house, "path": path_i,
                         "run_id": f"{batch}:house{house}_path{path_i}"})
    return runs


def top_ranked_ids(cands):
    """Truth-free top rank: viable candidates whose score equals the best viable score (ties kept, as run_rgb_batch does)."""
    viable = [c for c in cands if c.get("viable")]
    if not viable:
        return []
    best = max(c["score"] for c in viable)
    return [c["id"] for c in viable if abs(c["score"] - best) < 1e-9]


def vertical_flag_id(cands):
    """Truth-free vertical-height candidate flag: among viable candidates with a found cap pair (floor/ceiling-like pair of
    height peaks separated by the cap separation band), the one with the largest cap support; ties -> lower rank."""
    pool = [c for c in cands if c.get("viable") and c.get("cap_pair_found")]
    if not pool:
        return None
    return max(pool, key=lambda c: (c.get("cap_support") or 0.0, -c.get("rank", 0)))["id"]


def classify_failure(boundary, walls_status=None, walls_reasons=None):
    """Failure label and reason counts for one assemble_boundary_ray output (or its absence)."""
    if boundary is None:
        return {"label": "walls_unavailable", "status": None, "top_reason": f"walls:{walls_status}",
                "walls_reasons": list(walls_reasons or []), "layout_reasons": {}, "ignored_lines": {}, "closed_cycles": None}
    status = boundary.get("status")
    inc = boundary.get("incomplete_boundaries") or []
    top = inc[0].get("reason") if inc else None
    layout = dict(inc[0].get("layouts_rejected_by_reason") or {}) if inc else {}
    search = boundary.get("boundary_search") or {}
    if not layout and search.get("rejected_by_reason"):
        layout = dict(search["rejected_by_reason"])
    ignored = Counter(i.get("reason") for i in boundary.get("ignored_lines") or [])
    if status == "complete_polygon_hypothesis":
        label = "closed"
    elif status == "ambiguous":
        label = "ambiguous"
    else:
        label = top or "incomplete_unknown"
    return {"label": label, "status": status, "top_reason": top, "layout_reasons": layout, "ignored_lines": dict(ignored),
            "closed_cycles": search.get("closed_cycles"), "nodes": search.get("nodes"), "corner_links": search.get("corner_links"),
            "supported_lines": (inc[0].get("supported_lines") if inc else None)}


# ----------------------------------------------------------------------------------------------------- wall line statistics
def line_stats(walls):
    """(a): lines extracted, observed span coverage per line, noise_scale against offsets and offset_sd."""
    lines_in = (walls or {}).get("lines") or []
    frame = (walls or {}).get("frame") or {}
    extent = float(frame.get("extent") or 0.0)
    noise = (walls or {}).get("noise_scale")
    lines = []
    for ln in lines_in:
        spans = ln.get("observed_spans") or []
        total = float(sum(b - a for a, b in spans))
        hull = float(spans[-1][1] - spans[0][0]) if spans else 0.0
        sd = (ln.get("uncertainty") or {}).get("offset_sd")
        sup = ln.get("support") or {}
        lines.append({"id": ln.get("id"), "offset": ln.get("offset"), "span_total": total,
                      "span_total_over_extent": total / extent if extent else None,
                      "hull_over_extent": hull / extent if extent else None, "span_fill_fraction": total / hull if hull else None,
                      "n_spans": len(spans), "along_coverage_fraction": sup.get("along_coverage_fraction"),
                      "distinct_frames": sup.get("distinct_frames"), "view_spread_deg": sup.get("view_spread_deg"),
                      "weak": bool(ln.get("weak")), "offset_sd": sd,
                      "offset_sd_over_noise": (sd / noise) if (sd is not None and noise) else None})
    offs = [abs(l["offset"]) for l in lines if l["offset"] is not None]
    sds = [l["offset_sd"] for l in lines if l["offset_sd"] is not None]
    return {"n_lines": len(lines), "n_weak": sum(1 for l in lines if l["weak"]), "extent": extent or None, "noise_scale": noise,
            "noise_over_extent": (noise / extent) if (noise is not None and extent) else None,
            "median_offset_sd": _median(sds), "median_offset_sd_over_noise": (_median(sds) / noise) if (sds and noise) else None,
            "noise_over_median_abs_offset": (noise / _median(offs)) if (noise is not None and offs and _median(offs)) else None,
            "median_span_over_extent": _median([l["span_total_over_extent"] for l in lines]),
            "total_span_over_extent": float(sum(l["span_total_over_extent"] or 0 for l in lines)) if lines else 0.0,
            "median_along_coverage": _median([l["along_coverage_fraction"] for l in lines]),
            "median_frames": _median([l["distinct_frames"] for l in lines]), "lines": lines}


def _hull_along(line_m, line_c, line_d, spans, d_ref):
    ends = []
    for a, b in spans:
        for t in (a, b):
            ends.append(float(d_ref @ (line_c * line_m + t * line_d)))
    return (min(ends), max(ends)) if ends else (0.0, 0.0)


def layered_pairs(walls, cams_xy=None, *, max_angle_deg=10.0, min_overlap=0.3, noise_multiple=LAYER_TOL_MULTIPLE,
                  drift_fraction=DRIFT_FRACTION):
    """(d): pairs of near-parallel, laterally overlapping lines seen from the same side whose offset separation exceeds the noise.

    `same_side`: the camera centroid lies on the same side of both lines (a wall pair across the room is not layered). A pair
    is `layered` when same-side and separation > noise_multiple * max(noise_scale, median offset_sd) (fragment_association's tol);
    `drift_scale` when also separation <= drift_fraction * extent (the allowance used by fragment_association), else it is an
    object-scale offset (furniture / partition), not a drift."""
    lines_in = (walls or {}).get("lines") or []
    extent = float(((walls or {}).get("frame") or {}).get("extent") or 0.0)
    noise = float((walls or {}).get("noise_scale") or 0.0)
    sds = [float((l.get("uncertainty") or {}).get("offset_sd") or 0.0) for l in lines_in]
    tol = noise_multiple * max(noise, float(np.median(sds)) if sds else 0.0)
    cen = None
    if cams_xy is not None and len(cams_xy):
        cen = np.asarray(cams_xy, float).reshape(-1, 2).mean(axis=0)
    recs = []
    for ln in lines_in:
        m = np.asarray(ln["normal_xy"], float)
        d = np.asarray(ln["direction_xy"], float)
        recs.append((ln.get("id"), m / np.linalg.norm(m), float(ln["offset"]) / np.linalg.norm(m), d, ln.get("observed_spans") or []))
    out = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            ida, ma, ca, da, sa = recs[i]
            idb, mb, cb, db, sb = recs[j]
            if not sa or not sb:
                continue
            dot = float(ma @ mb)
            ang = math.degrees(math.acos(min(1.0, abs(dot))))
            if ang > max_angle_deg:
                continue
            if dot < 0:
                mb, cb = -mb, -cb
            ha = _hull_along(ma, ca, da, sa, da)
            hb = _hull_along(mb, cb, db, sb, da)
            ov = max(0.0, min(ha[1], hb[1]) - max(ha[0], hb[0]))
            shorter = min(ha[1] - ha[0], hb[1] - hb[0])
            if shorter <= 0 or ov < min_overlap * shorter:
                continue
            tm = 0.5 * (max(ha[0], hb[0]) + min(ha[1], hb[1]))
            # point of line a at along-coordinate tm; its signed distance to line b
            base = ca * ma
            denom = float(da @ da)
            p = base + (tm - float(da @ base)) / denom * da
            sep_signed = float(mb @ p) - cb
            sep = abs(sep_signed)
            same_side = None
            if cen is not None:
                same_side = bool((float(ma @ cen) - ca) * (float(mb @ cen) - cb) > 0)
            layered = bool(same_side and sep > tol) if same_side is not None else bool(sep > tol)
            out.append({"line_a": ida, "line_b": idb, "angle_deg": ang, "overlap_over_shorter": ov / shorter, "separation": sep,
                        "separation_over_extent": sep / extent if extent else None,
                        "separation_over_noise": sep / noise if noise else None, "same_side": same_side, "layered": layered,
                        "drift_scale": bool(layered and extent and sep <= drift_fraction * extent)})
    return {"tol": tol, "pairs": out}


def layered_summary(result):
    pairs = result["pairs"]
    lay = [p for p in pairs if p["layered"]]
    drift = [p for p in lay if p["drift_scale"]]
    return {"tol": result["tol"], "n_parallel_overlapping_pairs": len(pairs), "n_layered": len(lay), "n_layered_drift_scale": len(drift),
            "n_layered_object_scale": len(lay) - len(drift),
            "max_drift_separation_over_extent": max((p["separation_over_extent"] for p in drift), default=None),
            "median_drift_separation_over_extent": _median([p["separation_over_extent"] for p in drift]),
            "max_drift_separation_over_noise": max((p["separation_over_noise"] for p in drift if p["separation_over_noise"] is not None),
                                                    default=None),
            "pairs": lay}


# ------------------------------------------------------------------------------------------------ prediction-only statistics
def camera_centres(extrinsics):
    E = np.asarray(extrinsics, float)
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)       # -R^T t


def cams_in_frame(frame, centres):
    o = np.asarray(frame["origin_world"], float)
    e1, e2 = np.asarray(frame["e1"], float), np.asarray(frame["e2"], float)
    rel = np.asarray(centres, float) - o
    return np.stack([rel @ e1, rel @ e2], axis=1)


def confidence_stats(pred):
    conf = np.asarray(pred.get("confidence"), float) if pred.get("confidence") is not None else None
    if conf is None:
        return None
    depth = np.asarray(pred["depth"], float)
    ok = np.isfinite(conf) & np.isfinite(depth) & (depth > 0)
    c = conf[ok][::7]
    if not len(c):
        return None
    q = np.percentile(c, [10, 25, 50, 75, 90])
    frame_med = np.array([np.median(conf[k][ok[k]]) for k in range(len(conf)) if ok[k].any()])
    return {"p10": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p90": q[4], "mean": float(c.mean()),
            "p90_over_p10": float(q[4] / q[0]) if q[0] > 0 else None, "per_frame_median_min": float(frame_med.min()),
            "per_frame_median_max": float(frame_med.max()), "per_frame_median_std": float(frame_med.std()),
            "valid_fraction": float(ok.mean())}


def path_extent_stats(pred, cloud_extent=None):
    c = camera_centres(pred["extrinsics"])
    lo, hi = np.percentile(c, [5, 95], axis=0)
    ext = float(np.linalg.norm(hi - lo))
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return {"n_frames": int(len(c)), "retained_fraction_of_120": float(len(c) / 120.0), "path_extent": ext,
            "path_extent_over_cloud_extent": (ext / cloud_extent) if cloud_extent else None,
            "path_length": float(step.sum()), "median_step": float(np.median(step)), "max_step_over_median": float(step.max() / max(np.median(step), 1e-12)),
            "path_length_over_cloud_extent": (float(step.sum()) / cloud_extent) if cloud_extent else None}


def _sphere_dirs(n=3000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def view_coverage(pred, axis=None, ring_deg=20.0, n_dirs=3000):
    """Predicted-view coverage of the direction sphere (and of the horizontal ring around `axis`): fraction of directions inside
    the frustum of >= 1 (and >= 3) predicted cameras. Uses only the prediction's own extrinsics / intrinsics."""
    E = np.asarray(pred["extrinsics"], float)
    K = np.asarray(pred["intrinsics"], float)
    if K.ndim == 2:
        K = np.repeat(K[None], len(E), axis=0)
    h, w = np.asarray(pred["depth"]).shape[1:3]
    dirs = _sphere_dirs(n_dirs)
    count = np.zeros(len(dirs), int)
    for k in range(len(E)):
        y = dirs @ E[k, :3, :3].T
        z = y[:, 2]
        with np.errstate(all="ignore"):
            u = K[k, 0, 0] * y[:, 0] / z + K[k, 0, 2]
            v = K[k, 1, 1] * y[:, 1] / z + K[k, 1, 2]
        count += ((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)).astype(int)
    out = {"sphere_ge1": float((count >= 1).mean()), "sphere_ge3": float((count >= 3).mean())}
    if axis is not None:
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        ring = np.abs(dirs @ a) <= math.sin(math.radians(ring_deg))
        out.update(ring_ge1=float((count[ring] >= 1).mean()), ring_ge3=float((count[ring] >= 3).mean()))
        az = np.arctan2(dirs[ring] @ np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < .9 else [0.0, 1.0, 0.0]),
                        dirs[ring] @ (np.array([1.0, 0.0, 0.0]) if abs(a[0]) < .9 else np.array([0.0, 1.0, 0.0])))
        bins = np.floor((az + np.pi) / (2 * np.pi) * 36).astype(int) % 36
        out["ring_azimuth_bins_seen_of_36"] = int(len({int(b) for b, c in zip(bins, count[ring]) if c >= 1}))
    return out


def depth_consistency(pred, *, every=4, stride=6, conf_percentile=40.0, min_samples=300, near_gap=8, far_gap=40, trim=0.25):
    """(e): pairwise reprojection depth disagreement, from the prediction only. Frame i's confident pixels are back-projected with
    its own depth and pose, projected into frame j, and the projected camera-z is compared with frame j's predicted depth at that
    pixel: rel = depth_j / z_proj - 1. Medians are robust to occlusion; `trim` drops |rel| > trim before the trimmed statistics.
    The median log(depth_j / z_proj) per pair measures a depth SCALE inconsistency between the two frames."""
    depth = np.asarray(pred["depth"], float)
    conf = pred.get("confidence")
    E = np.asarray(pred["extrinsics"], float)
    K = np.asarray(pred["intrinsics"], float)
    if K.ndim == 2:
        K = np.repeat(K[None], len(depth), axis=0)
    n, h, w = depth.shape
    ids = list(range(0, n, every))
    y, x = np.mgrid[0:h:stride, 0:w:stride]
    cache = {}
    for k in ids:
        z = depth[k][y, x]
        ok = np.isfinite(z) & (z > 0)
        if conf is not None:
            c = np.asarray(conf[k], float)[y, x]
            ok &= np.isfinite(c)
            if ok.any():
                ok &= c >= np.percentile(c[ok], conf_percentile)
        pix = np.stack([x[ok], y[ok], np.ones(int(ok.sum()))], axis=1)
        cam = (pix @ np.linalg.inv(K[k]).T) * z[ok, None]
        R, t = E[k, :3, :3], E[k, :3, 3]
        cache[k] = (cam - t) @ R
    pairs = []
    for ia, a in enumerate(ids):
        for b in ids[ia + 1:]:
            Xw = cache[a]
            Yc = Xw @ E[b, :3, :3].T + E[b, :3, 3]
            z = Yc[:, 2]
            with np.errstate(all="ignore"):
                u = np.rint(K[b, 0, 0] * Yc[:, 0] / z + K[b, 0, 2]).astype(int)
                v = np.rint(K[b, 1, 1] * Yc[:, 1] / z + K[b, 1, 2]).astype(int)
            ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if ok.sum() < min_samples:
                continue
            d = depth[b][v[ok], u[ok]]
            good = np.isfinite(d) & (d > 0)
            if good.sum() < min_samples:
                continue
            rel = d[good] / z[ok][good] - 1.0
            tr = np.abs(rel) < trim
            if tr.sum() < min_samples // 2:
                continue
            pairs.append({"gap": b - a, "n": int(good.sum()), "median_abs_rel": float(np.median(np.abs(rel))),
                          "median_abs_rel_trim": float(np.median(np.abs(rel[tr]))), "within2pct": float((np.abs(rel) < 0.02).mean()),
                          "log_ratio_trim": float(np.median(np.log1p(rel[tr])))})
    if not pairs:
        return {"n_pairs": 0}
    near = [p for p in pairs if p["gap"] <= near_gap]
    far = [p for p in pairs if p["gap"] >= far_gap]

    def agg(ps, key):
        return _median([p[key] for p in ps])
    logs = np.array([p["log_ratio_trim"] for p in pairs])
    return {"n_pairs": len(pairs), "median_abs_rel_all": agg(pairs, "median_abs_rel"), "median_abs_rel_trim_all": agg(pairs, "median_abs_rel_trim"),
            "within2pct_all": float(np.mean([p["within2pct"] for p in pairs])),
            "near_pairs": len(near), "near_median_abs_rel_trim": agg(near, "median_abs_rel_trim"),
            "far_pairs": len(far), "far_median_abs_rel_trim": agg(far, "median_abs_rel_trim"),
            "scale_logratio_abs_median": float(np.median(np.abs(logs))), "scale_logratio_std": float(logs.std()),
            "scale_logratio_p90_abs": float(np.percentile(np.abs(logs), 90))}


# ------------------------------------------------------------------------------------------------ candidate-level analysis
def lines_compact(walls):
    return {"frame": (walls or {}).get("frame"), "axis_world": (walls or {}).get("axis_world"),
            "noise_scale": (walls or {}).get("noise_scale"),
            "lines": [{"id": l.get("id"), "normal_xy": l.get("normal_xy"), "offset": l.get("offset"), "direction_xy": l.get("direction_xy"),
                       "observed_spans": l.get("observed_spans"), "distinct_frames": (l.get("support") or {}).get("distinct_frames")}
                      for l in (walls or {}).get("lines") or []]}


def analyze_candidate(walls, boundary, centres):
    """Everything truth-free about one candidate from its walls dict (summary form is enough) and boundary dict."""
    status = (walls or {}).get("status")
    fail = classify_failure(boundary, status, (walls or {}).get("reasons"))
    out = {"walls_status": status, "failure": fail, "closed": fail["label"] == "closed"}
    if status == "ok":
        stats = line_stats(walls)
        cams = cams_in_frame(walls["frame"], centres) if walls.get("frame") else None
        lay = layered_summary(layered_pairs(walls, cams))
        out.update(line_stats=stats, layered=lay)
        if cams is not None:
            out["cameras_xy_extent"] = float(np.linalg.norm(cams.max(axis=0) - cams.min(axis=0)))
            out["cameras_xy_extent_over_cloud_extent"] = out["cameras_xy_extent"] / walls["frame"]["extent"]
    return out


def candidate_meta(c, evidence):
    cp = (evidence or {}).get("cap_pair") or {}
    poly = c.get("polygon") or {}
    return {"id": c["candidate_id"] if "candidate_id" in c else c["id"], "rank": c["rank"], "score": c["score"], "viable": c["viable"],
            "co_supported_with_best": c.get("co_supported_with_best"), "kinds": c.get("kinds"),
            "axis_world": c.get("axis_world", c.get("axis")), "cap_pair_found": bool(cp.get("pair_found")),
            "cap_support": cp.get("cap_support"), "cap_separation": cp.get("separation_unscaled"),
            "camera_share_between_caps": cp.get("camera_share_between_caps"),
            "height_status": poly.get("height_status"), "height_separation_unscaled": poly.get("height_separation_unscaled")}


def fragment_summary(walls):
    """fragment_association API on a walls dict that carries observations."""
    import fragment_association as fa
    res = fa.associate_fragments(walls)
    rep = res.get("report") or {}
    extent = float(((walls or {}).get("frame") or {}).get("extent") or 0.0)
    pairs = rep.get("pairs") or []
    return {"status": res.get("status"), "counts": rep.get("counts"), "n_pairs": len(pairs),
            "pairs": [{"line": p.get("line"), "target": p.get("target"), "score": p.get("score"), "separation_max": p.get("separation_max"),
                       "separation_over_extent": (p.get("separation_max") / extent) if (extent and p.get("separation_max") is not None) else None,
                       "exclusivity": p.get("exclusivity"), "co_visible_frames": p.get("co_visible_frames"), "reason": p.get("reason")}
                      for p in pairs],
            "n_layered_groups": len(rep.get("layered_groups") or []), "layered_groups": rep.get("layered_groups")}


def probe_confidence(pred, axis, percentile=PROBE_CONF_PERCENTILE):
    """(g): wall extraction + assembly on the highest-confidence pixels only (one fixed value, not tuned)."""
    import boundary_assembly as ba
    import wall_extraction as we
    walls = we.extract_wall_lines(pred, axis, conf_percentile=percentile)
    boundary = ba.assemble_boundary_ray(pred, walls, RAY_K) if walls.get("status") == "ok" else None
    fail = classify_failure(boundary, walls.get("status"), walls.get("reasons"))
    stats = line_stats(walls) if walls.get("status") == "ok" else None
    return {"conf_percentile": percentile, "walls_status": walls.get("status"), "failure": fail, "closed": fail["label"] == "closed",
            "n_lines": stats["n_lines"] if stats else 0, "noise_over_extent": stats["noise_over_extent"] if stats else None,
            "line_stats": stats}


def reproduction_check(frozen_walls, frozen_boundary, walls, boundary):
    """Re-extraction under the current modules vs the frozen outputs (determinism / code drift)."""
    fl = [(l["id"], round(float(l["offset"]), 9)) for l in (frozen_walls or {}).get("lines") or []]
    nl = [(l["id"], round(float(l["offset"]), 9)) for l in (walls or {}).get("lines") or []]
    return {"walls_identical": fl == nl, "frozen_lines": len(fl), "rerun_lines": len(nl),
            "frozen_boundary_status": (frozen_boundary or {}).get("status"), "rerun_boundary_status": (boundary or {}).get("status")}


# ----------------------------------------------------------------------------------------------------------- run diagnosis
def diagnose_run(run, *, probe=True, fragments=True, events=None):
    """TRUTH-FREE diagnosis of one frozen RGB run. `run` is an entry of list_runs."""
    import boundary_assembly as ba
    import wall_extraction as we
    t0 = time.time()
    run_dir = Path(run["run_dir"])
    index = read_json(run_dir / "axis" / "frozen_index.json")["budgets"][0]
    label = index["label"]
    cands_file = read_json(run_dir / "axis" / index["files"]["candidates"]["path"])
    evid = {c["id"]: c.get("evidence") for c in cands_file["candidates"]}
    pred = _load_npz(run_dir / "da3" / "prediction.npz")
    centres = camera_centres(pred["extrinsics"])
    metas, sha_problems = [], []
    for c in index["candidates"]:
        meta = candidate_meta(c, evid.get(c["candidate_id"]))
        walls = boundary = None
        files = c.get("files") or {}
        if "walls_summary" in files:
            p = run_dir / "axis" / files["walls_summary"]["path"]
            if sha256_file(p) != files["walls_summary"]["sha256"]:
                sha_problems.append(str(p))
            walls = read_json(p)
        if "boundary" in files:
            p = run_dir / "axis" / files["boundary"]["path"]
            if sha256_file(p) != files["boundary"]["sha256"]:
                sha_problems.append(str(p))
            boundary = read_json(p)
        meta.update(analyze_candidate(walls, boundary, centres))
        metas.append(meta)
    top = top_ranked_ids(metas)
    vflag = vertical_flag_id(metas)
    focus_ids = sorted(set(top) | ({vflag} if vflag is not None else set()))
    meta_by_id = {m["id"]: m for m in metas}
    focus = {}
    extent_cloud = next((m["line_stats"]["extent"] for m in metas if m.get("line_stats") and m["line_stats"].get("extent")), None)
    for cid in focus_ids:
        m = meta_by_id[cid]
        roles = (["top_ranked"] if cid in top else []) + (["vertical_flag"] if cid == vflag else [])
        rec = {"roles": roles}
        axis = m["axis_world"]
        try:
            walls = we.extract_wall_lines(pred, axis)
            boundary = ba.assemble_boundary_ray(pred, walls, RAY_K) if walls.get("status") == "ok" else None
            fp = next(c for c in index["candidates"] if c["candidate_id"] == cid)["files"]
            fw = read_json(run_dir / "axis" / fp["walls_summary"]["path"]) if "walls_summary" in fp else None
            fb = read_json(run_dir / "axis" / fp["boundary"]["path"]) if "boundary" in fp else None
            rec["reproduction"] = reproduction_check(fw, fb, walls, boundary)
            if walls.get("frame"):
                extent_cloud = walls["frame"]["extent"]
                rec["view_coverage"] = view_coverage(pred, axis)
            if fragments and walls.get("status") == "ok":
                rec["fragment_association"] = fragment_summary(walls)
        except Exception as exc:                                  # a failure is data, never hidden
            rec["rerun_error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback_tail"] = traceback.format_exc()[-800:]
        if probe:
            try:
                rec["probe_top50_confidence"] = probe_confidence(pred, axis)
            except Exception as exc:
                rec["probe_error"] = f"{type(exc).__name__}: {exc}"
        focus[str(cid)] = rec
    out = {"schema": "rgb_closure_diagnosis_truth_free_v1", "truth_used": False, **{k: run[k] for k in ("batch", "run_id", "house", "path")},
           "run_dir": run["run_dir"], "label": label, "candidate_count": len(metas), "top_ranked_ids": top, "vertical_flag_id": vflag,
           "frozen_sha_problems": sha_problems, "candidates": metas, "focus": focus,
           "run_features": {"confidence": confidence_stats(pred), "path": path_extent_stats(pred, extent_cloud),
                            "view_coverage_sphere": view_coverage(pred), "depth_consistency": depth_consistency(pred)},
           "seconds": time.time() - t0}
    if events is not None:
        events.append(("diagnosed", run["run_id"]))
    return out


# ------------------------------------------------------------------------------------------------------- control (2 x 2)
def run_chain(pred, *, events=None):
    """The same truth-free chain as run_axis_candidates.run_extraction for one in-memory prediction: planes -> axis candidates
    (polygon runner = wall_extraction + assemble_boundary_ray) for every candidate. Returns per-candidate records."""
    import axis_candidates as ac
    import boundary_assembly as ba
    import extract_planes
    import height_diagnosis
    import wall_extraction as we
    points, ids = height_diagnosis._per_view_points(pred)
    planes = extract_planes.extract(points)
    normals = ac.normals_from_prediction(pred)
    centres = camera_centres(pred["extrinsics"])
    store = {}

    def runner(axis):
        axis = np.asarray(axis, float)
        item = {}
        try:
            walls = we.extract_wall_lines(pred, axis)
            item["walls"] = walls
            summary = {"walls_status": walls.get("status"), "boundary_status": "walls_unavailable"}
            if walls.get("status") == "ok":
                boundary = ba.assemble_boundary_ray(pred, walls, RAY_K)
                item["boundary"] = boundary
                summary["boundary_status"] = boundary.get("status")
        except Exception as exc:
            summary = {"boundary_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        try:
            h = height_diagnosis.estimate_room_height(points, axis, ids)
            summary["height_status"] = h.get("status")
            summary["height_separation_unscaled"] = h.get("separation_unscaled")
        except Exception as exc:
            summary["height_error"] = f"{type(exc).__name__}: {exc}"
        store[tuple(round(float(x), 10) for x in axis)] = item
        return summary
    result = ac.enumerate_axis_candidates(planes, points, normals=normals, camera_centers=centres, polygon_runner=runner)
    cands = []
    for c in result.get("candidates", []):
        item = store.get(tuple(round(float(x), 10) for x in c["axis_world"]), {})
        meta = candidate_meta({**c, "candidate_id": c["id"]}, c.get("evidence"))
        meta.update(analyze_candidate(item.get("walls"), item.get("boundary"), centres))
        meta["lines_compact"] = lines_compact(item.get("walls")) if (item.get("walls") or {}).get("status") == "ok" else None
        cands.append(meta)
    top = top_ranked_ids(cands)
    vflag = vertical_flag_id(cands)
    return {"candidate_count": len(cands), "top_ranked_ids": top, "vertical_flag_id": vflag,
            "closed_candidates": sum(1 for c in cands if c["closed"]), "candidates": cands,
            "view_coverage_sphere": view_coverage(pred),
            "vertical_flag_view_coverage": view_coverage(pred, next((c["axis_world"] for c in cands if c["id"] == vflag), None)) if vflag is not None else None,
            "path": path_extent_stats(pred), "confidence": confidence_stats(pred), "depth_consistency": depth_consistency(pred)}


def similarity_from_centres(src, dst):
    """Umeyama similarity dst ~ s R src + t (no reflection). Returns s, R, t and the rms residual in dst units."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (A ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var)
    t = md - s * R @ ms
    resid = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
    return {"s": s, "R": R, "t": t, "rms": float(np.sqrt((resid ** 2).mean())), "max": float(resid.max())}


def hybrid_predictions(da3, lidar):
    """Control cells from two PUBLIC captures of the same camera positions. NOTE: the LiDAR capture rolls its camera about the
    optical axis while the upright RGB capture does not (same centres, yaw, pitch), so ROTATIONS cannot be exchanged between the
    captures; only camera CENTRES can. The similarity (scale s DA3 units -> metres, R_s, t_s) is fitted on the camera centres only; no
    scene data is used. Cells (all in the sensor world, metres):
      A  DA3 depth, DA3 rotations, DA3 centres            (the RGB baseline, mapped by the similarity)
      B  LiDAR depth, sensor rotations, sensor centres     (the control)
      C  DA3 depth, DA3 rotations, SENSOR centres          (does the DA3 centre/translation drift matter?)
      D  LiDAR depth, sensor rotations, DA3 centres        (does DA3 centre drift break clean depth + rotations?)"""
    n = min(len(da3["depth"]), len(lidar["depth"]))
    sim = similarity_from_centres(camera_centres(da3["extrinsics"][:n]), camera_centres(lidar["extrinsics"][:n]))
    s, Rs, ts = sim["s"], sim["R"], sim["t"]

    def sub(p, **over):
        per_frame = ("depth", "confidence", "extrinsics", "images")
        out = {k: (v[:n] if isinstance(v, np.ndarray) and (k in per_frame or (k == "intrinsics" and v.ndim == 3)) else v) for k, v in p.items()}
        out["source_indices"] = np.arange(n)
        out.update(over)
        return out
    Ed = np.asarray(da3["extrinsics"][:n], float)
    Rd, td = Ed[:, :3, :3], Ed[:, :3, 3]
    Rm = Rd @ Rs.T                                              # DA3 rotation expressed in the sensor world
    c_da3_m = s * (camera_centres(Ed) @ Rs.T) + ts              # DA3 centres in the sensor world (metres)
    El = np.asarray(lidar["extrinsics"][:n], float)
    c_sen = camera_centres(El)

    def with_centres(R, centres):
        out = np.zeros((len(R), 3, 4))
        out[:, :3, :3] = R
        out[:, :3, 3] = -np.einsum("nij,nj->ni", R, centres)
        return out.astype(np.float32)
    depth_m = np.asarray(da3["depth"][:n], np.float32) * np.float32(s)
    cell_a = sub(da3, depth=depth_m, extrinsics=with_centres(Rm, c_da3_m))
    cell_c = sub(da3, depth=depth_m, extrinsics=with_centres(Rm, c_sen))
    cell_d = sub(lidar, extrinsics=with_centres(El[:, :3, :3], c_da3_m))
    return {"A_da3_depth_rot_centres": cell_a, "B_lidar_depth_rot_centres": sub(lidar), "C_da3_depth_rot__sensor_centres": cell_c,
            "D_lidar_depth_rot__da3_centres": cell_d}, {"scale_da3_to_m": s, "centre_alignment_rms_m": sim["rms"], "centre_alignment_max_m": sim["max"], "frames": n}


def control_house(outputs_root, house, path_index=0, batch="rgb-batch-upright-v2"):
    """(f) for one house: the four cells through the same chain."""
    root = Path(outputs_root)
    da3 = _load_npz(root / batch / "houses" / f"house_{house}" / f"path_{path_index:02d}" / "da3" / "prediction.npz")
    lidar = _load_npz(root / LIDAR_ROOT / f"house_{house}" / f"capture_{path_index:02d}" / "prediction.npz")
    cells, info = hybrid_predictions(da3, lidar)
    out = {"schema": "rgb_closure_control_truth_free_v1", "truth_used": False, "house": house, "path": path_index, "batch": batch,
           "lidar_capture": f"{LIDAR_ROOT}/house_{house}/capture_{path_index:02d}/prediction.npz", "pairing": info, "cells": {}}
    for name, pred in cells.items():
        t0 = time.time()
        try:
            out["cells"][name] = {**run_chain(pred), "seconds": time.time() - t0}
        except Exception as exc:
            out["cells"][name] = {"error": f"{type(exc).__name__}: {exc}", "traceback_tail": traceback.format_exc()[-800:]}
    return out


# ---------------------------------------------------------------------------------------------------- truth-free stage CLIs
def stage_diagnose(out_dir, outputs_root=OUTPUTS, batches=BATCHES, limit=None, probe=True):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output folder already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    runs = list_runs(outputs_root, batches)[:limit]
    index = []
    for run in runs:
        res = diagnose_run(run, probe=probe)
        rel = f"truth_free/{run['batch']}/house{run['house']}_path{run['path']}.json"
        index.append({"run_id": run["run_id"], "path": rel, "sha256": write_new_json(out_dir / rel, res)})
        print(f"{run['run_id']} {res['seconds']:.1f}s", flush=True)
    return index


def stage_control(out_dir, outputs_root=OUTPUTS, houses=(17, 101, 102, 103, 104, 105, 106, 107, 108, 109)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for h in houses:
        res = control_house(outputs_root, h)
        rel = f"control/house{h}_path0.json"
        index.append({"house": h, "path": rel, "sha256": write_new_json(out_dir / rel, res)})
        print(f"control house {h}", flush=True)
    return index


def stage_freeze(out_dir):
    out_dir = Path(out_dir)
    files = sorted(list((out_dir / "truth_free").rglob("*.json")) + list((out_dir / "control").rglob("*.json")))
    index = {"schema": "rgb_closure_freeze_v1", "note": "sha256 of every truth-free output, written BEFORE any truth is read",
             "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files}}
    write_new_json(out_dir / "freeze_index.json", index)
    return index


# ==================================================================================================================
# EVALUATOR-ONLY PART: everything below may read the private scene / poses. It runs only after freeze_index.json exists.
# ==================================================================================================================
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return None, int(ok.sum())

    def rank(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            sel = v == u
            r[sel] = r[sel].mean()
        return r
    rx, ry = rank(x[ok]), rank(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return None, int(ok.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def classify_layout(layout):
    """Oracle layout class from a simulator.make_scene dict (box room with furniture/openings) or a house_layouts room."""
    poly = None
    if isinstance(layout, dict) and layout.get("rooms"):
        poly = layout["rooms"][0].get("polygon_xy_m")
    elif isinstance(layout, dict) and layout.get("polygon_xy_m"):
        poly = layout["polygon_xy_m"]
    furniture = layout.get("furniture") or [] if isinstance(layout, dict) else []
    openings = layout.get("openings") or [] if isinstance(layout, dict) else []
    out = {"furniture_count": len(furniture), "opening_count": len(openings),
           "doorway_count": sum(1 for o in openings if o.get("type") == "doorway"),
           "window_count": sum(1 for o in openings if o.get("type") == "window")}
    if poly is not None:
        pts = np.asarray(poly, float)
        n = len(pts)
        cross = []
        for i in range(n):
            a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        convex = all(v >= -1e-9 for v in cross) or all(v <= 1e-9 for v in cross)
        out.update(vertex_count=n, convex=bool(convex))
        out["class"] = "rectangle" if (n == 4 and convex) else ("l_shape" if (n == 6 and not convex) else ("convex_polygon" if convex else "nonconvex_polygon"))
        return out
    if isinstance(layout, dict) and "width" in layout and "length" in layout:
        w, l, h = float(layout["width"]), float(layout["length"]), float(layout["height"])
        foot = sum(float(f["size"][0]) * float(f["size"][1]) for f in furniture)
        out.update(class_="box", width=w, length=l, height=h, floor_area=w * l, aspect=max(w, l) / min(w, l),
                   furniture_footprint_fraction=foot / (w * l), max_furniture_height_over_room=max((float(f["size"][2]) for f in furniture), default=0.0) / h,
                   opening_width_total=sum(float(o["width"]) for o in openings))
        out["class"] = "plain_box" if not furniture and not openings else ("box_with_openings" if not furniture else
                                                                         ("box_with_furniture" if not openings else "box_with_furniture_and_openings"))
        out.pop("class_")
        return out
    out["class"] = "unknown"
    return out


def true_alignment(extrinsics, poses, source_indices=None):
    """Oracle similarity prediction world -> true world from per-frame rotations and the camera centres.
    Also returns the per-frame rotation scatter (deg) and the centre residual (metres): drift / inconsistency of the DA3 trajectory."""
    import simulator
    E = np.asarray(extrinsics, float)
    ids = np.arange(len(E)) if source_indices is None else np.asarray(source_indices).astype(int)
    Rs = []
    pos = []
    for k, i in enumerate(ids):
        pose = poses[int(i)]
        right, up, forward = simulator.camera_basis(pose)
        c2w = np.column_stack((right, -up, forward))
        Rs.append(c2w @ E[k, :3, :3])
        pos.append(np.asarray(pose["position"], float))
    Rs = np.asarray(Rs)
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R0 = U @ D @ Vt                              # chordal mean of the per-frame rotations: fixes the tilt (robust for planar paths)
    cen = camera_centres(E)
    a = cen @ R0.T
    p = np.asarray(pos)
    am, pm = a.mean(axis=0), p.mean(axis=0)
    # refine the rotation about the vertical (and the scale) with the camera centres: 2-D similarity in the horizontal plane
    A, B = a[:, :2] - am[:2], p[:, :2] - pm[:2]
    theta = float(np.arctan2((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum(), (A * B).sum()))
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    Ra = Rz @ R0
    a = cen @ Ra.T
    am = a.mean(axis=0)
    s = float(((a - am) * (p - pm)).sum() / ((a - am) ** 2).sum())
    t = pm - s * Ra @ cen.mean(axis=0)
    resid = np.linalg.norm(s * a + t - p, axis=1)
    ang = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", Rs, Ra) - 1) / 2, -1, 1)))
    return {"R": Ra, "s": s, "t": t, "rot_scatter_deg_median": float(np.median(ang)), "rot_scatter_deg_p90": float(np.percentile(ang, 90)),
            "centre_resid_rms_m": float(np.sqrt((resid ** 2).mean())), "centre_resid_max_m": float(resid.max()),
            "true_up_in_pred_world": (Ra.T @ np.array([0.0, 0.0, 1.0])).tolist()}


def axis_angle_to_true_up(axis, align):
    up = np.asarray(align["true_up_in_pred_world"], float)
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return float(math.degrees(math.acos(min(1.0, abs(float(a @ up))))))


def line_offset_errors(compact, align, scene, *, angle_tol_deg=15.0, max_wall_distance_m=0.5):
    """Observed-line offset error (cm) after the oracle similarity. EVALUATOR-ONLY. Only meaningful for a near-vertical axis.
    Walls: 0 north y=+L/2, 1 east x=+W/2, 2 south y=-L/2, 3 west x=-W/2 (simulator convention)."""
    W, L = float(scene["width"]), float(scene["length"])
    frame = compact["frame"]
    o = np.asarray(frame["origin_world"], float)
    e1, e2 = np.asarray(frame["e1"], float), np.asarray(frame["e2"], float)
    Ra, s, t = align["R"], align["s"], np.asarray(align["t"], float)

    def to_true(xy):
        X = o + xy[0] * e1 + xy[1] * e2
        return (s * Ra @ X + t)[:2]
    walls = [("north", 1, L / 2), ("east", 0, W / 2), ("south", 1, -L / 2), ("west", 0, -W / 2)]
    wall_len = [W, L, W, L]
    per_line, per_wall = [], {i: [] for i in range(4)}
    for ln in compact.get("lines") or []:
        m = np.asarray(ln["normal_xy"], float)
        d = np.asarray(ln["direction_xy"], float)
        spans = ln["observed_spans"]
        base = ln["offset"] * m
        t0, t1 = spans[0][0], spans[-1][1]
        tm = 0.5 * (t0 + t1)
        # direction-coordinate of the base point is d.base; positions on the line: base + (t - d.base) d
        def at(tt):
            return base + (tt - float(d @ base)) * d
        p0, p1, pm = to_true(at(t0)), to_true(at(t1)), to_true(at(tm))
        u = p1 - p0
        nrm = np.linalg.norm(u)
        if nrm < 1e-9:
            continue
        u = u / nrm
        horizontal = math.degrees(math.acos(min(1.0, abs(u[0])))) <= angle_tol_deg      # runs along x -> north/south wall
        vertical = math.degrees(math.acos(min(1.0, abs(u[1])))) <= angle_tol_deg
        best = None
        for wi, (name, ax, pos) in enumerate(walls):
            if (ax == 1 and not horizontal) or (ax == 0 and not vertical):
                continue
            err = float(pm[ax] - pos)
            if best is None or abs(err) < abs(best[1]):
                best = (wi, err)
        rec = {"id": ln["id"], "wall": None, "offset_error_cm": None, "span_length_m": float(nrm), "frames": ln.get("distinct_frames")}
        if best is not None and abs(best[1]) <= max_wall_distance_m:
            wi, err = best
            rec.update(wall=wi, offset_error_cm=100.0 * err)
            ax = walls[wi][1]
            along = sorted((float(p0[1 - ax]), float(p1[1 - ax])))
            per_wall[wi].append(along)
        per_line.append(rec)
    coverage = {}
    for wi in range(4):
        segs = sorted(per_wall[wi])
        merged = []
        for a, b in segs:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        half = wall_len[wi] / 2
        cov = sum(max(0.0, min(b, half) - max(a, -half)) for a, b in merged)
        coverage[walls[wi][0]] = {"n_lines": sum(1 for r in per_line if r["wall"] == wi), "covered_fraction": cov / wall_len[wi]}
    errs = [abs(r["offset_error_cm"]) for r in per_line if r["offset_error_cm"] is not None]
    return {"lines": per_line, "per_wall": coverage, "n_matched": len(errs), "n_unmatched": len(per_line) - len(errs),
            "walls_with_a_line": sum(1 for v in coverage.values() if v["n_lines"] > 0),
            "walls_with_multiple_lines": sum(1 for v in coverage.values() if v["n_lines"] > 1),
            "median_abs_offset_error_cm": _median(errs), "max_abs_offset_error_cm": max(errs) if errs else None,
            "mean_wall_covered_fraction": float(np.mean([v["covered_fraction"] for v in coverage.values()]))}


def _frozen_compact(run_dir, candidate_id):
    run_dir = Path(run_dir)
    index = read_json(run_dir / "axis" / "frozen_index.json")["budgets"][0]
    c = next(c for c in index["candidates"] if c["candidate_id"] == candidate_id)
    if "walls_summary" not in (c.get("files") or {}):
        return None
    walls = read_json(run_dir / "axis" / c["files"]["walls_summary"]["path"])
    return lines_compact(walls) if walls.get("status") == "ok" else None


def oracle_run(tf, outputs_root=OUTPUTS):
    """Labels for one truth-free run record `tf`: layout class, truth alignment, per-candidate angle to the true up, offset errors
    for the focus candidates and the oracle vertical candidate."""
    run_dir = Path(tf["run_dir"])
    scene = read_json(run_dir / "capture" / "private_ground_truth" / "scene.json")
    pose_file = next((run_dir / "capture" / "private_ground_truth").glob("path_*.json"))
    poses = read_json(pose_file)["poses"]
    qa = read_json(run_dir / "generator_private" / "capture_qa.json") if (run_dir / "generator_private" / "capture_qa.json").exists() else {}
    pred = _load_npz(run_dir / "da3" / "prediction.npz")
    align = true_alignment(pred["extrinsics"], poses, pred.get("source_indices"))
    angles = {c["id"]: axis_angle_to_true_up(c["axis_world"], align) for c in tf["candidates"]}
    oracle_vertical = min(angles, key=angles.get) if angles else None
    vertical_ids = [i for i, a in angles.items() if a <= 15.0]
    res = {"schema": "rgb_closure_oracle_labels_v1", "EVALUATOR_ONLY": True, "run_id": tf["run_id"],
           "layout": classify_layout(scene), "alignment": {k: v for k, v in align.items() if k not in ("R", "t")},
           "angle_to_true_up_deg": angles, "oracle_vertical_id": oracle_vertical,
           "oracle_vertical_angle_deg": angles.get(oracle_vertical), "oracle_vertical_ids_within_15deg": vertical_ids,
           "truth_free_vertical_flag_is_oracle_vertical": (tf["vertical_flag_id"] in vertical_ids) if tf["vertical_flag_id"] is not None else None,
           "top_ranked_includes_oracle_vertical": bool(set(tf["top_ranked_ids"]) & set(vertical_ids)),
           "capture_qa": {"angular_full_fraction": (qa.get("angular") or {}).get("full_fraction"), "max_hole_deg": (qa.get("angular") or {}).get("max_hole_deg"),
                          "horizontal_baseline_m": (qa.get("motion") or {}).get("horizontal_baseline_m"),
                          "min_wall_clearance_m": (qa.get("motion") or {}).get("min_wall_clearance_m"), "qa_pass": qa.get("qa_pass"),
                          "frame_overlap_min": qa.get("frame_overlap_min")},
           "offset_errors": {}}
    path_extent_true = float(np.linalg.norm(np.percentile([p["position"] for p in poses], 95, axis=0) - np.percentile([p["position"] for p in poses], 5, axis=0)))
    res["true_path_extent_over_room_diagonal"] = path_extent_true / math.hypot(scene["width"], scene["length"])
    res["scale_da3_unit_to_m"] = align["s"]
    focus_ids = {int(k) for k in tf["focus"]} | ({oracle_vertical} if oracle_vertical is not None else set())
    for cid in sorted(focus_ids):
        if angles[cid] > 15.0:
            res["offset_errors"][str(cid)] = {"skipped": f"axis {angles[cid]:.1f} deg from the true up"}
            continue
        compact = _frozen_compact(run_dir, cid)
        res["offset_errors"][str(cid)] = line_offset_errors(compact, align, scene) if compact else {"skipped": "no ok walls"}
    return res


def oracle_control(ctl, outputs_root=OUTPUTS):
    """Offset errors of the four control cells (EVALUATOR-ONLY): sensor-frame cells are aligned with the same oracle routine."""
    root = Path(outputs_root)
    run_dir = root / ctl["batch"] / "houses" / f"house_{ctl['house']}" / f"path_{ctl['path']:02d}"
    scene = read_json(run_dir / "capture" / "private_ground_truth" / "scene.json")
    poses_rgb = read_json(next((run_dir / "capture" / "private_ground_truth").glob("path_*.json")))["poses"]
    poses_lidar = read_json(root / LIDAR_ROOT / f"house_{ctl['house']}" / "private_ground_truth" / f"path_{ctl['path']:02d}.json")["poses"]
    da3 = _load_npz(run_dir / "da3" / "prediction.npz")
    lidar = _load_npz(root / LIDAR_ROOT / f"house_{ctl['house']}" / f"capture_{ctl['path']:02d}" / "prediction.npz")
    cells, info = hybrid_predictions(da3, lidar)
    out = {"schema": "rgb_closure_control_oracle_v1", "EVALUATOR_ONLY": True, "house": ctl["house"], "cells": {}}
    for name, pred in cells.items():
        cell = ctl["cells"].get(name) or {}
        if "candidates" not in cell:
            out["cells"][name] = {"error": cell.get("error")}
            continue
        # rotations of B and D are the sensor's (the LiDAR path rolls its camera); A and C carry the upright RGB rotations
        poses = poses_lidar if name[0] in "BD" else poses_rgb
        align = true_alignment(pred["extrinsics"], poses, np.arange(len(pred["extrinsics"])))
        angles = {c["id"]: axis_angle_to_true_up(c["axis_world"], align) for c in cell["candidates"]}
        vert = [i for i, a in angles.items() if a <= 15.0]
        rec = {"alignment": {k: v for k, v in align.items() if k not in ("R", "t")}, "oracle_vertical_ids_within_15deg": vert,
               "vertical_flag_is_oracle_vertical": (cell["vertical_flag_id"] in vert) if cell["vertical_flag_id"] is not None else None,
               "closed_oracle_vertical": [i for i in vert if next(c for c in cell["candidates"] if c["id"] == i)["closed"]], "offset_errors": {}}
        ids = set([cell["vertical_flag_id"]] if cell["vertical_flag_id"] is not None else []) | set(cell["top_ranked_ids"]) | set(vert[:1])
        for cid in sorted(ids):
            c = next(c for c in cell["candidates"] if c["id"] == cid)
            if angles[cid] > 15.0 or not c.get("lines_compact"):
                continue
            rec["offset_errors"][str(cid)] = line_offset_errors(c["lines_compact"], align, scene)
        out["cells"][name] = rec
    return out


def stage_oracle(out_dir, outputs_root=OUTPUTS):
    out_dir = Path(out_dir)
    freeze = out_dir / "freeze_index.json"
    if not freeze.exists():
        raise RuntimeError("refusing to read truth before freeze_index.json exists")
    for rel, sha in read_json(freeze)["files"].items():
        if sha256_file(out_dir / rel) != sha:
            raise RuntimeError(f"frozen file changed: {rel}")
    index = []
    for p in sorted((out_dir / "truth_free").rglob("*.json")):
        tf = read_json(p)
        res = oracle_run(tf, outputs_root)
        rel = "oracle/" + str(p.relative_to(out_dir / "truth_free")).replace("\\", "/")
        index.append({"run_id": tf["run_id"], "path": rel, "sha256": write_new_json(out_dir / rel, res)})
        print("oracle", tf["run_id"], flush=True)
    for p in sorted((out_dir / "control").glob("*.json")):
        res = oracle_control(read_json(p), outputs_root)
        write_new_json(out_dir / "oracle" / "control" / p.name, res)
        print("oracle control", p.name, flush=True)
    return index


# ------------------------------------------------------------------------------------------------------------- summarize
def _first_focus(tf, role):
    for cid, rec in tf["focus"].items():
        if role in rec["roles"]:
            return int(cid), rec
    return None, None


def run_row(tf, orc):
    cands = tf["candidates"]
    by_id = {c["id"]: c for c in cands}
    top = [by_id[i] for i in tf["top_ranked_ids"]]
    vflag = by_id.get(tf["vertical_flag_id"]) if tf["vertical_flag_id"] is not None else None
    vert = [by_id[i] for i in orc["oracle_vertical_ids_within_15deg"] if i in by_id]
    best_vert = by_id.get(orc["oracle_vertical_id"])
    ls = (best_vert or {}).get("line_stats") or {}
    lay = (best_vert or {}).get("layered") or {}
    rf = tf["run_features"]
    dc = rf["depth_consistency"]
    off = orc["offset_errors"].get(str(orc["oracle_vertical_id"])) or {}
    layout = orc["layout"]
    probe_changes = []
    for cid, rec in tf["focus"].items():
        pr = rec.get("probe_top50_confidence")
        if pr:
            base = by_id[int(cid)]
            probe_changes.append({"id": int(cid), "baseline_closed": base["closed"], "probe_closed": pr["closed"], "baseline_label": base["failure"]["label"],
                                  "probe_label": pr["failure"]["label"], "baseline_lines": base.get("line_stats", {}).get("n_lines"), "probe_lines": pr["n_lines"]})
    return {
        "run_id": tf["run_id"], "batch": tf["batch"], "house": tf["house"], "path": tf["path"], "n_candidates": len(cands),
        "n_closed": sum(c["closed"] for c in cands), "any_closed": any(c["closed"] for c in cands),
        "top_closed": any(c["closed"] for c in top), "top_n": len(top), "vflag_closed": bool(vflag and vflag["closed"]),
        "oracle_vertical_in_set": bool(vert), "oracle_vertical_closed": any(c["closed"] for c in vert),
        "oracle_vertical_id": orc["oracle_vertical_id"], "oracle_vertical_angle_deg": orc["oracle_vertical_angle_deg"],
        "vflag_is_oracle_vertical": orc["truth_free_vertical_flag_is_oracle_vertical"], "top_includes_oracle_vertical": orc["top_ranked_includes_oracle_vertical"],
        "ov_label": (best_vert or {}).get("failure", {}).get("label"), "ov_layout_reasons": (best_vert or {}).get("failure", {}).get("layout_reasons"),
        "ov_n_lines": ls.get("n_lines"), "ov_noise_over_extent": ls.get("noise_over_extent"), "ov_median_span_over_extent": ls.get("median_span_over_extent"),
        "ov_median_along_cov": ls.get("median_along_coverage"), "ov_n_weak": ls.get("n_weak"), "ov_n_layered": lay.get("n_layered"),
        "ov_n_layered_drift": lay.get("n_layered_drift_scale"), "ov_max_drift_sep_over_extent": lay.get("max_drift_separation_over_extent"),
        "ov_walls_with_a_line": off.get("walls_with_a_line"), "ov_walls_with_multiple_lines": off.get("walls_with_multiple_lines"),
        "ov_median_off_cm": off.get("median_abs_offset_error_cm"), "ov_max_off_cm": off.get("max_abs_offset_error_cm"),
        "ov_mean_wall_cov": off.get("mean_wall_covered_fraction"), "ov_n_unmatched": off.get("n_unmatched"),
        "layout_class": layout.get("class"), "furniture_count": layout.get("furniture_count"), "opening_count": layout.get("opening_count"),
        "furniture_footprint_fraction": layout.get("furniture_footprint_fraction"), "aspect": layout.get("aspect"), "floor_area": layout.get("floor_area"),
        "max_furniture_height_over_room": layout.get("max_furniture_height_over_room"), "opening_width_total": layout.get("opening_width_total"),
        "room_height": layout.get("height"),
        "qa_full_fraction": orc["capture_qa"]["angular_full_fraction"], "qa_max_hole_deg": orc["capture_qa"]["max_hole_deg"],
        "qa_horizontal_baseline_m": orc["capture_qa"]["horizontal_baseline_m"], "true_path_extent_over_room_diag": orc["true_path_extent_over_room_diagonal"],
        "retained_frames": rf["path"]["n_frames"], "pred_path_extent_over_cloud": rf["path"]["path_extent_over_cloud_extent"],
        "pred_sphere_cov_ge1": rf["view_coverage_sphere"]["sphere_ge1"], "pred_sphere_cov_ge3": rf["view_coverage_sphere"]["sphere_ge3"],
        "conf_p50": (rf["confidence"] or {}).get("p50"), "conf_p10": (rf["confidence"] or {}).get("p10"), "conf_p90_over_p10": (rf["confidence"] or {}).get("p90_over_p10"),
        "dc_near": dc.get("near_median_abs_rel_trim"), "dc_far": dc.get("far_median_abs_rel_trim"), "dc_all": dc.get("median_abs_rel_trim_all"),
        "dc_within2": dc.get("within2pct_all"), "dc_scale_abs": dc.get("scale_logratio_abs_median"), "dc_scale_std": dc.get("scale_logratio_std"),
        "rot_scatter_deg": orc["alignment"]["rot_scatter_deg_median"], "centre_resid_rms_m": orc["alignment"]["centre_resid_rms_m"],
        "centre_resid_over_room_diag": orc["alignment"]["centre_resid_rms_m"] / math.hypot(layout.get("width", 1), layout.get("length", 1)) if layout.get("width") else None,
        "probe": probe_changes}


CORRELATION_FEATURES = ("furniture_footprint_fraction", "aspect", "floor_area", "max_furniture_height_over_room", "opening_width_total", "room_height",
                        "qa_full_fraction", "qa_max_hole_deg", "qa_horizontal_baseline_m", "true_path_extent_over_room_diag", "pred_path_extent_over_cloud",
                        "pred_sphere_cov_ge1", "pred_sphere_cov_ge3", "conf_p50", "conf_p10", "conf_p90_over_p10", "dc_near", "dc_far", "dc_all", "dc_within2",
                        "dc_scale_abs", "dc_scale_std", "rot_scatter_deg", "centre_resid_rms_m", "centre_resid_over_room_diag", "ov_noise_over_extent",
                        "ov_median_span_over_extent", "ov_median_along_cov", "ov_n_lines", "ov_n_layered", "ov_n_layered_drift", "ov_max_drift_sep_over_extent",
                        "ov_walls_with_a_line", "ov_walls_with_multiple_lines", "ov_median_off_cm", "ov_max_off_cm", "ov_mean_wall_cov")


def correlations(rows, outcome):
    y = [1.0 if r[outcome] else 0.0 for r in rows]
    out = {}
    for f in CORRELATION_FEATURES:
        x = [np.nan if r.get(f) is None else float(r[f]) for r in rows]
        rho, n = spearman(x, y)
        out[f] = {"rho": rho, "n": n}
    return out


def control_rows(out_dir):
    """2 x 2 factorial rows (one per house and cell) from the truth-free control files and their oracle labels."""
    rows = []
    for p in sorted((Path(out_dir) / "control").glob("*.json")):
        ctl = read_json(p)
        orc = read_json(Path(out_dir) / "oracle" / "control" / p.name)
        for name, cell in ctl["cells"].items():
            if "candidates" not in cell:
                rows.append({"house": ctl["house"], "cell": name, "error": cell.get("error")})
                continue
            o = orc["cells"][name]
            vert = o["oracle_vertical_ids_within_15deg"]
            flag = cell["vertical_flag_id"]
            cid = flag if flag in vert else (vert[0] if vert else None)
            cand = next((c for c in cell["candidates"] if c["id"] == cid), None) if cid is not None else None
            ls = (cand or {}).get("line_stats") or {}
            lay = (cand or {}).get("layered") or {}
            off = o["offset_errors"].get(str(cid)) or {}
            by_id = {c["id"]: c for c in cell["candidates"]}
            dc = cell["depth_consistency"]
            rows.append({"house": ctl["house"], "cell": name, "n_candidates": cell["candidate_count"], "n_closed": cell["closed_candidates"],
                         "top_closed": any(by_id[i]["closed"] for i in cell["top_ranked_ids"]),
                         "top_includes_oracle_vertical": bool(set(cell["top_ranked_ids"]) & set(vert)),
                         "vflag_is_oracle_vertical": o["vertical_flag_is_oracle_vertical"], "oracle_vertical_closed": bool(o["closed_oracle_vertical"]),
                         "v_label": (cand or {}).get("failure", {}).get("label"), "v_n_lines": ls.get("n_lines"),
                         "v_noise_over_extent": ls.get("noise_over_extent"), "v_median_span_over_extent": ls.get("median_span_over_extent"),
                         "v_n_layered_drift": lay.get("n_layered_drift_scale"), "v_max_drift_sep_over_extent": lay.get("max_drift_separation_over_extent"),
                         "v_walls_with_a_line": off.get("walls_with_a_line"), "v_median_off_cm": off.get("median_abs_offset_error_cm"),
                         "v_max_off_cm": off.get("max_abs_offset_error_cm"), "v_mean_wall_cov": off.get("mean_wall_covered_fraction"),
                         "dc_near": dc.get("near_median_abs_rel_trim"), "dc_scale_abs": dc.get("scale_logratio_abs_median"),
                         "rot_scatter_deg": o["alignment"]["rot_scatter_deg_median"], "centre_resid_rms_m": o["alignment"]["centre_resid_rms_m"],
                         "pairing_rms_m": ctl["pairing"]["centre_alignment_rms_m"], "pairing_scale": ctl["pairing"]["scale_da3_to_m"]})
    return rows


def _table(rows, cols):
    def fmt(v):
        return "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def stage_summarize(out_dir):
    out_dir = Path(out_dir)
    rows = []
    for p in sorted((out_dir / "truth_free").rglob("*.json")):
        tf = read_json(p)
        orc = read_json(out_dir / "oracle" / p.relative_to(out_dir / "truth_free"))
        rows.append(run_row(tf, orc))
    reasons_all, reasons_top, reasons_vflag, reasons_ov = Counter(), Counter(), Counter(), Counter()
    layouts_all, layouts_ov = Counter(), Counter()
    present_all, present_top, present_ov = Counter(), Counter(), Counter()    # candidates in which a reason occurs at least once
    for p in sorted((out_dir / "truth_free").rglob("*.json")):
        tf = read_json(p)
        orc = read_json(out_dir / "oracle" / p.relative_to(out_dir / "truth_free"))
        for c in tf["candidates"]:
            reasons_all[c["failure"]["label"]] += 1
            for k, v in c["failure"]["layout_reasons"].items():
                layouts_all[k] += v
                present_all[k] += 1
                if c["id"] in tf["top_ranked_ids"]:
                    present_top[k] += 1
                if c["id"] in orc["oracle_vertical_ids_within_15deg"]:
                    present_ov[k] += 1
            if c["id"] in tf["top_ranked_ids"]:
                reasons_top[c["failure"]["label"]] += 1
            if c["id"] == tf["vertical_flag_id"]:
                reasons_vflag[c["failure"]["label"]] += 1
            if c["id"] in orc["oracle_vertical_ids_within_15deg"]:
                reasons_ov[c["failure"]["label"]] += 1
                for k, v in c["failure"]["layout_reasons"].items():
                    layouts_ov[k] += v
    summary = {"schema": "rgb_closure_summary_v1", "n_runs": len(rows), "rows": rows,
               "reason_counts": {"all_candidates": dict(reasons_all), "top_ranked": dict(reasons_top), "vertical_flag": dict(reasons_vflag),
                                 "oracle_vertical": dict(reasons_ov)},
               "rejected_layout_reason_counts": {"all_candidates": dict(layouts_all), "oracle_vertical": dict(layouts_ov)},
               "candidates_with_layout_reason": {"all_candidates": dict(present_all), "top_ranked": dict(present_top), "oracle_vertical": dict(present_ov)},
               "correlations": {o: correlations(rows, o) for o in ("any_closed", "top_closed", "oracle_vertical_closed")},
               }
    write_new_json(out_dir / "summary.json", summary)
    tables = ["# Tables (machine generated; evaluator-only columns come from the oracle stage)", "", "## Per run", "",
              _table(rows, ["run_id", "n_candidates", "n_closed", "top_closed", "top_n", "oracle_vertical_in_set", "oracle_vertical_closed", "vflag_is_oracle_vertical",
                            "top_includes_oracle_vertical", "ov_label", "ov_n_lines", "ov_n_layered_drift", "ov_walls_with_a_line", "ov_median_off_cm", "ov_max_off_cm",
                            "dc_near", "dc_scale_abs", "centre_resid_over_room_diag", "rot_scatter_deg", "conf_p50"])]
    (out_dir / "tables.md").write_text("\n".join(tables), encoding="utf-8")
    return summary


def stage_summarize_control(out_dir):
    out_dir = Path(out_dir)
    rows = control_rows(out_dir)
    write_new_json(out_dir / "control_summary.json", {"schema": "rgb_closure_control_summary_v1", "rows": rows})
    cols = ["house", "cell", "n_candidates", "n_closed", "top_closed", "top_includes_oracle_vertical", "oracle_vertical_closed", "v_label", "v_n_lines",
            "v_noise_over_extent", "v_n_layered_drift", "v_walls_with_a_line", "v_median_off_cm", "v_max_off_cm", "dc_near", "dc_scale_abs",
            "centre_resid_rms_m", "rot_scatter_deg", "pairing_rms_m"]
    (out_dir / "control_tables.md").write_text("# Control cells (house, path 0)\n\n" + _table(rows, cols), encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["diagnose", "control", "freeze", "oracle", "summarize", "summarize-control"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--houses", type=int, nargs="*", default=None, help="control stage: house seeds (default: all ten)")
    args = parser.parse_args()
    if args.stage == "diagnose":
        stage_diagnose(args.output_dir, args.outputs_root, limit=args.limit)
    elif args.stage == "control":
        stage_control(args.output_dir, args.outputs_root, **({"houses": tuple(args.houses)} if args.houses else {}))
    elif args.stage == "freeze":
        stage_freeze(args.output_dir)
    elif args.stage == "oracle":
        stage_oracle(args.output_dir, args.outputs_root)
    elif args.stage == "summarize-control":
        stage_summarize_control(args.output_dir)
    else:
        stage_summarize(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
