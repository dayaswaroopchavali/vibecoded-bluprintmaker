"""EVALUATION ONLY: score a predicted floor-plan polygon against a private scene (custom normalised diagnostic).

This is the one place private ground truth may be read for floor plans. Nothing here may be imported by a
reconstruction stage. Protocol (the user's custom diagnostic; official gates are reported separately, never merged):

 1. Prediction and ground truth are each normalised to a 3.0 m ceiling height with ONE uniform scale each. The prediction
    scale is 3.0 / (floor-ceiling separation from an estimate_height.py output JSON); when that is missing, ambiguous or not
    estimated the protocol fails with a named reason. The prediction scale is never fitted to the truth, and the truth's
    height only scales the truth.
 2. The two 2D polygons are aligned with the best rigid motion (rotation + translation, no scale) by maximising IoU. The
    mirrored prediction is also tried ONLY when the structural-axis sign is unresolved (it always is for
    geometry_orientation hypotheses). Any mirroring, cyclic vertex start and edge correspondence is an
    EVALUATOR-ORACLE correspondence chosen with the truth in hand; it is not something the reconstruction recovered.
 3. Reported: IoU, area ratio, per-edge length errors (matched edges), max absolute wall error (cm) and relative (%), predicted
    versus true edge counts, observed/inferred labels and the retained capture fraction (from a manifest, if supplied).
    Pass/fail against (a) the custom 1.5 cm horizontal diagnostic and (b) the official 3% wall-length gate, separately.
    Incomplete or missing predictions are failures with reasons.

An SVG and PNG overlay (true outline vs predicted polygon, observed edges solid, inferred dashed) is written to a NEW
folder; existing output is never overwritten. This module never claims that a benchmark stage is complete.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np

CEILING_NORMALISED_M = 3.0
CUSTOM_TOLERANCE_M = 0.015
OFFICIAL_RELATIVE_TOLERANCE = 0.03
COMPLETE_STATUSES = ("complete_polygon_hypothesis", "complete_box_hypothesis")
MATCH_ANGLE_DEG = 25.0
MATCH_DISTANCE_M = 0.6
LIMITATIONS = [
    "Diagnostic only: the 3.0 m normalisation, the rigid alignment and the edge matching use the ground truth to choose the best alignment (EVALUATOR-ORACLE).",
    "The prediction scale comes only from the supplied estimate_height output; if that is wrong, every horizontal error scales with it.",
    "Gates are per room and per prediction; this result does not complete any benchmark stage or coverage requirement.",
    "Edges are matched one-to-one by direction and midpoint; unequal edge counts (including collinear splits) fail both gates.",
]


# ---------------------------------------------------------------------------------------------------- geometry
def _signed_area(poly) -> float:
    p = np.asarray(poly, float)
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _centroid(poly) -> np.ndarray:
    p = np.asarray(poly, float); q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = cross.sum() / 2
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    return np.array([((p[:, 0] + q[:, 0]) * cross).sum(), ((p[:, 1] + q[:, 1]) * cross).sum()]) / (6 * area)


def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _mask(poly, lo, shape, ppm) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    pts = np.round((np.asarray(poly, float) - lo) * ppm * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return img


def polygon_iou(a, b, ppm: float = 100.0) -> float:
    """IoU of two simple polygons by sub-pixel rasterisation (default 1 cm cells)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.vstack([a, b]); lo = both.min(axis=0) - 0.1; hi = both.max(axis=0) + 0.1
    shape = (int(math.ceil((hi[1] - lo[1]) * ppm)) + 1, int(math.ceil((hi[0] - lo[0]) * ppm)) + 1)
    if max(shape) > 6000:
        raise ValueError("polygons too far apart or too large to rasterise")
    ma, mb = _mask(a, lo, shape, ppm), _mask(b, lo, shape, ppm)
    union = int(np.count_nonzero(ma | mb))
    return float(np.count_nonzero(ma & mb) / union) if union else 0.0


def _mirror(poly: np.ndarray) -> np.ndarray:
    """x -> -x, vertex order reversed so the polygon stays counter-clockwise."""
    return (poly * np.array([-1.0, 1.0]))[::-1].copy()


def _kabsch(est: np.ndarray, ref: np.ndarray):
    ce, cr = est.mean(axis=0), ref.mean(axis=0)
    u, _s, vt = np.linalg.svd((est - ce).T @ (ref - cr))
    rot = vt.T @ np.diag([1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return rot, cr - rot @ ce


def align_rigid(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Rotation + translation (no scale) of `pred` onto `truth` maximising IoU (EVALUATOR-ORACLE)."""
    cp, ct = _centroid(pred), _centroid(truth)
    seeds = []                                                     # (theta, translation, origin)
    coarse = []
    for deg in np.arange(0.0, 360.0, 3.0):
        th = math.radians(deg); t = ct - _rotation(th) @ cp
        coarse.append((polygon_iou(pred @ _rotation(th).T + t, truth, 25.0), th, t))
    coarse.sort(key=lambda item: -item[0])
    seeds += [(th, t, "iou_grid", None) for _v, th, t in coarse[:3]]
    if len(pred) == len(truth):
        for shift in range(len(truth)):
            rot, t = _kabsch(pred, np.roll(truth, -shift, axis=0))
            seeds.append((math.atan2(rot[1, 0], rot[0, 0]), t, "vertex_fit", shift))
    best = None
    for th, t, kind, shift in seeds:
        cur = polygon_iou(pred @ _rotation(th).T + t, truth, 50.0)
        step_a, step_t = math.radians(2.0), 0.08
        while step_a > math.radians(0.02) or step_t > 0.0005:
            improved = False
            for da, dx, dy in ((step_a, 0, 0), (-step_a, 0, 0), (0, step_t, 0), (0, -step_t, 0), (0, 0, step_t), (0, 0, -step_t)):
                th2, t2 = th + da, t + np.array([dx, dy])
                val = polygon_iou(pred @ _rotation(th2).T + t2, truth, 50.0)
                if val > cur + 1e-9:
                    th, t, cur, improved = th2, t2, val, True
            if not improved:
                step_a, step_t = step_a / 2, step_t / 2
        final = polygon_iou(pred @ _rotation(th).T + t, truth, 100.0)
        if best is None or final > best["iou"]:
            best = {"iou": final, "theta": th, "translation": t, "seed": kind, "cyclic_start": shift}
    return best


def match_edges(pred: np.ndarray, truth: np.ndarray) -> list[tuple[int, int]]:
    """Greedy one-to-one (true edge, predicted edge) pairs by direction and midpoint, after alignment."""
    def edges(poly):
        a, b = poly, np.roll(poly, -1, axis=0)
        return (a + b) / 2, b - a
    mp, dp = edges(pred); mt, dt = edges(truth)
    cand = []
    for i in range(len(truth)):
        for j in range(len(pred)):
            cos = abs(float(dt[i] @ dp[j])) / max(float(np.linalg.norm(dt[i]) * np.linalg.norm(dp[j])), 1e-12)
            angle = math.degrees(math.acos(min(1.0, cos)))
            dist = float(np.linalg.norm(mt[i] - mp[j]))
            if angle <= MATCH_ANGLE_DEG and dist <= MATCH_DISTANCE_M:
                cand.append((dist + 0.01 * angle, i, j))
    cand.sort()
    used_t, used_p, pairs = set(), set(), []
    for _c, i, j in cand:
        if i not in used_t and j not in used_p:
            used_t.add(i); used_p.add(j); pairs.append((i, j))
    return sorted(pairs)


# ---------------------------------------------------------------------------------------------------- inputs
def load_truth(scene: dict, room_id: str | None = None) -> dict:
    """Ground-truth outline and ceiling height from a private scene (rectangle or house_layouts polygon)."""
    if not isinstance(scene, dict):
        raise ValueError("scene must be a JSON object")
    room = scene
    if isinstance(scene.get("rooms"), list):
        rooms = scene["rooms"]
        chosen = [r for r in rooms if room_id is None or r.get("room_id") == room_id]
        if len(chosen) != 1:
            raise ValueError("ambiguous_or_unknown_room: pass room_id for a multi-room layout")
        room = chosen[0]
    if "polygon_xy_m" in room:
        poly = np.asarray(room["polygon_xy_m"], float)
        height = room.get("ceiling_height_m", room.get("height"))
        kind = "polygon_layout"
    else:
        w, l, height = (room.get(k) for k in ("width", "length", "height"))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (w, l)):
            raise ValueError("scene lacks width/length or polygon_xy_m")
        poly = np.array([[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2], [-w / 2, l / 2]], float)
        kind = "rectangle_centred_at_origin"
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError("malformed truth polygon")
    if _signed_area(poly) < 0:
        poly = poly[::-1].copy()
    ok = isinstance(height, (int, float)) and not isinstance(height, bool) and math.isfinite(height) and height > 0
    return {"vertices": poly, "height": float(height) if ok else None, "kind": kind}


def prediction_scale(height_estimate) -> tuple[float | None, list[str], dict]:
    """3.0 m / predicted floor-ceiling separation, from an estimate_height output only (never from the truth)."""
    if not isinstance(height_estimate, dict):
        return None, ["height_estimate_missing"], {}
    status = height_estimate.get("status")
    sep = height_estimate.get("separation_unscaled")
    info = {"source": "estimate_height.separation_unscaled", "status": status, "separation_unscaled": sep}
    if status == "ambiguous":
        return None, ["height_estimate_ambiguous"], info
    if status != "estimated_unscaled_separation":
        return None, [f"height_estimate_not_estimated:{status}"], info
    if not isinstance(sep, (int, float)) or isinstance(sep, bool) or not math.isfinite(sep) or sep <= 0:
        return None, ["height_estimate_separation_invalid"], info
    return CEILING_NORMALISED_M / float(sep), [], info


def _find_key(obj, key, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v), f"{path}/{k}"
            hit = _find_key(v, key, f"{path}/{k}")
            if hit:
                return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hit = _find_key(v, key, f"{path}/{i}")
            if hit:
                return hit
    return None


def whole_room_claim(retained: dict, status: str) -> dict:
    """Whether this result could support a whole-room statement. It never can for this oracle-normalised diagnostic;
    the blockers say why, so missing capture provenance or partial retention cannot be promoted into success."""
    blockers = ["oracle_normalised_alignment_diagnostic_is_not_an_assessment_metric"]
    if not retained.get("supplied") or retained.get("value") is None:
        blockers.append("capture_provenance_missing")
    elif retained["value"] < 1.0:
        blockers.append("retained_capture_below_full")
    if status != "scored":
        blockers.append("prediction_not_scored")
    return {"allowed": False, "blockers": blockers}

def retained_capture_fraction(manifest) -> dict:
    hit = _find_key(manifest, "retained_capture_fraction") if manifest is not None else None
    return {"value": hit[0] if hit else None, "found_at": hit[1] if hit else None,
            "supplied": manifest is not None}


# ---------------------------------------------------------------------------------------------------- evaluation
def _failed(reasons, scale, base):
    gates = {"custom_1_5cm_horizontal_diagnostic": {"assessment_eligible": False, "tolerance_cm": 100 * CUSTOM_TOLERANCE_M,
                                                    "pass": False, "failure_reasons": list(reasons)},
             "wall_length_3_percent_threshold_oracle_normalised": {"assessment_eligible": False, "note": "3% threshold applied under 3 m normalisation and evaluator-oracle alignment; NOT an official metric assessment", "tolerance_percent": 100 * OFFICIAL_RELATIVE_TOLERANCE,
                                                     "pass": False, "failure_reasons": list(reasons)}}
    base = {**base, "whole_room_claim": whole_room_claim(base["retained_capture"], "failed")}
    return {**base, "status": "failed", "failure_reasons": list(reasons), "scale": scale, "alignment": None, "metrics": None,
            **gates, "gates_never_merged": True, "limitations": LIMITATIONS}


def evaluate_floorplan(prediction: dict, scene: dict, height_estimate, *, manifest=None, axis_sign_resolved: bool = False,
                       room_id: str | None = None) -> dict:
    """Score one predicted plan. `scene` is private truth: this function is for evaluation only."""
    truth = load_truth(scene, room_id)
    base = {"model": "custom_normalised_floorplan_diagnostic", "evaluation_only": True,
            "normalised_ceiling_height_m": CEILING_NORMALISED_M, "truth_kind": truth["kind"],
            "retained_capture": retained_capture_fraction(manifest),
            "benchmark_stage_complete": False}
    base["whole_room_claim"] = whole_room_claim(base["retained_capture"], "scored")
    scale_info = {"ceiling_height_m": CEILING_NORMALISED_M, "prediction_scale": None, "truth_scale": None,
                  "prediction_scale_source": "estimate_height output only; never fitted to truth", "prediction_height": {},
                  "truth_scale_source": "truth ceiling height (scales the truth only)"}
    reasons = []
    pred_scale, why, info = prediction_scale(height_estimate)
    scale_info["prediction_height"] = info
    reasons += why
    scale_info["prediction_scale"] = pred_scale
    if truth["height"] is None:
        reasons.append("truth_ceiling_height_unavailable")
    else:
        scale_info["truth_scale"] = CEILING_NORMALISED_M / truth["height"]
    pred = prediction if isinstance(prediction, dict) else {}
    status = pred.get("status")
    verts = pred.get("vertices_xy")
    if status not in COMPLETE_STATUSES or verts is None:
        reasons.append("prediction_incomplete" if status is not None else "prediction_missing")
        reasons += [f"incomplete_boundary:{b.get('reason')}" for b in pred.get("incomplete_boundaries", []) if isinstance(b, dict)]
    if reasons:
        return _failed(reasons, scale_info, {**base, "prediction_status": status, "true_outline_normalised_m": None,
                                             "true_outline_scaled_m": (truth["vertices"] * scale_info["truth_scale"]).tolist()
                                             if scale_info["truth_scale"] else truth["vertices"].tolist()})
    try:
        pv = np.asarray(verts, float)
        if pv.ndim != 2 or pv.shape[1] != 2 or len(pv) < 3 or not np.isfinite(pv).all():
            raise ValueError
    except (TypeError, ValueError):
        return _failed(["prediction_vertices_malformed"], scale_info, {**base, "prediction_status": status})
    if _signed_area(pv) < 0:
        pv = pv[::-1].copy()
        flip_labels = True
    else:
        flip_labels = False
    n = len(pv)
    labels = ["unlabelled"] * n
    for e in pred.get("edges", []) or []:
        if isinstance(e, dict) and isinstance(e.get("index"), int) and 0 <= e["index"] < n and e.get("label") in ("observed", "inferred"):
            labels[e["index"]] = e["label"]
    if flip_labels:                                                # reversing vertices maps old edge k to n-2-k
        labels = [labels[(n - 2 - j) % n] for j in range(n)]
    pn = pv * scale_info["prediction_scale"]
    tn = truth["vertices"] * scale_info["truth_scale"]
    candidates = [("as_predicted", pn, labels)]
    mirror_allowed = not axis_sign_resolved
    if mirror_allowed:
        candidates.append(("mirrored_by_evaluator_oracle", _mirror(pn), [labels[(n - 2 - j) % n] for j in range(n)]))
    scored = []
    for name, poly, lab in candidates:
        scored.append((align_rigid(poly, tn), name, poly, lab))
    fit, variant, poly, lab = max(scored, key=lambda item: item[0]["iou"])
    aligned = poly @ _rotation(fit["theta"]).T + fit["translation"]
    pairs = match_edges(aligned, tn)
    lengths_p = np.linalg.norm(np.roll(aligned, -1, axis=0) - aligned, axis=1)
    lengths_t = np.linalg.norm(np.roll(tn, -1, axis=0) - tn, axis=1)
    edge_rows = [{"true_edge": i, "pred_edge": j, "true_length_m": float(lengths_t[i]), "pred_length_m": float(lengths_p[j]),
                  "abs_error_cm": float(100 * abs(lengths_p[j] - lengths_t[i])),
                  "rel_error_percent": float(100 * abs(lengths_p[j] - lengths_t[i]) / lengths_t[i]), "label": lab[j]}
                 for i, j in pairs]
    unmatched_true = sorted(set(range(len(tn))) - {i for i, _ in pairs})
    unmatched_pred = sorted(set(range(len(aligned))) - {j for _, j in pairs})
    max_abs = max((r["abs_error_cm"] for r in edge_rows), default=None)
    max_rel = max((r["rel_error_percent"] for r in edge_rows), default=None)
    label_counts = {k: lab.count(k) for k in ("observed", "inferred", "unlabelled") if lab.count(k)}
    problems = []
    if len(aligned) != len(tn):
        problems.append("edge_count_mismatch")
    if unmatched_true or unmatched_pred:
        problems.append("edges_unmatched")
    metrics = {"iou": fit["iou"], "area_ratio": float(abs(_signed_area(aligned)) / abs(_signed_area(tn))),
               "predicted_edge_count": int(len(aligned)), "true_edge_count": int(len(tn)),
               "edge_labels": label_counts, "edge_errors": edge_rows, "unmatched_true_edges": unmatched_true,
               "unmatched_predicted_edges": unmatched_pred, "max_abs_wall_error_cm": max_abs, "max_rel_wall_error_percent": max_rel}
    custom_ok = not problems and max_abs is not None and max_abs <= 100 * CUSTOM_TOLERANCE_M
    official_ok = not problems and max_rel is not None and max_rel <= 100 * OFFICIAL_RELATIVE_TOLERANCE
    return {**base, "status": "scored", "failure_reasons": [], "prediction_status": status, "scale": scale_info,
            "alignment": {"variant": variant, "mirror_tried": mirror_allowed, "mirrored": variant != "as_predicted",
                          "axis_sign_resolved": bool(axis_sign_resolved),
                          "rotation_deg": float(math.degrees(fit["theta"])), "translation_m": fit["translation"].tolist(),
                          "seed": fit["seed"], "cyclic_start": fit["cyclic_start"],
                          "correspondence": "EVALUATOR-ORACLE: mirroring, cyclic start and edge matching were chosen with the truth in hand, not recovered by the reconstruction"},
            "metrics": metrics,
            "custom_1_5cm_horizontal_diagnostic": {"assessment_eligible": False, "tolerance_cm": 100 * CUSTOM_TOLERANCE_M,
                                                    "pass": bool(custom_ok), "failure_reasons": problems if not custom_ok else []},
            "wall_length_3_percent_threshold_oracle_normalised": {"assessment_eligible": False, "note": "3% threshold applied under 3 m normalisation and evaluator-oracle alignment; NOT an official metric assessment", "tolerance_percent": 100 * OFFICIAL_RELATIVE_TOLERANCE,
                                                    "pass": bool(official_ok), "failure_reasons": problems if not official_ok else []},
            "gates_never_merged": True,
            "true_outline_normalised_m": tn.tolist(), "predicted_outline_aligned_m": aligned.tolist(),
            "predicted_edge_labels": lab, "limitations": LIMITATIONS}


# ---------------------------------------------------------------------------------------------------- overlay
def _summary_lines(result: dict) -> list[str]:
    s = result["scale"]
    lines = [f"status: {result['status']}"]
    if result["status"] != "scored":
        lines += [f"FAIL: {r}" for r in result["failure_reasons"]]
    else:
        m, a = result["metrics"], result["alignment"]
        lines += [f"IoU {m['iou']:.3f}   area ratio {m['area_ratio']:.3f}   edges pred/true {m['predicted_edge_count']}/{m['true_edge_count']}",
                  f"max wall error {m['max_abs_wall_error_cm']:.2f} cm / {m['max_rel_wall_error_percent']:.2f} %" if m["max_abs_wall_error_cm"] is not None else "max wall error n/a",
                  f"labels {m['edge_labels']}   mirrored(oracle) {a['mirrored']}"]
    c, o = result["custom_1_5cm_horizontal_diagnostic"], result["wall_length_3_percent_threshold_oracle_normalised"]
    lines.append(f"custom 1.5 cm diagnostic: {'PASS' if c['pass'] else 'FAIL'}   3% threshold (oracle-normalised, not an assessment gate): {'PASS' if o['pass'] else 'FAIL'}   (separate)")
    pscale = s["prediction_scale"]
    lines.append(f"normalised to 3.0 m ceiling; prediction scale {pscale:.4g}" if pscale else "prediction scale unavailable")
    r = result["retained_capture"]
    if r["value"] is not None:
        lines.append(f"retained capture fraction {r['value']:.3f}")
    lines.append("alignment, mirroring and edge matching are EVALUATOR-ORACLE")
    return lines


def write_overlay(result: dict, out_dir: Path) -> dict:
    """SVG + PNG overlay into a NEW directory (refuses to overwrite)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output folder already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    truth = np.asarray(result.get("true_outline_normalised_m") or result.get("true_outline_scaled_m"), float)
    pred = np.asarray(result["predicted_outline_aligned_m"], float) if result.get("predicted_outline_aligned_m") else None
    labels = result.get("predicted_edge_labels", [])
    both = truth if pred is None else np.vstack([truth, pred])
    lo, hi = both.min(axis=0) - 0.5, both.max(axis=0) + 0.5
    ppm = min(900 / max(hi[0] - lo[0], 1e-6), 700 / max(hi[1] - lo[1], 1e-6))
    width, plot_h = int((hi[0] - lo[0]) * ppm), int((hi[1] - lo[1]) * ppm)
    lines = _summary_lines(result)
    margin = 18 * (len(lines) + 1) + 10
    height = plot_h + margin

    def px(p):
        return (float((p[0] - lo[0]) * ppm), float(margin + (hi[1] - p[1]) * ppm))     # y up in the plan, down on screen
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="white"/>']
    for i, text in enumerate(lines):
        svg.append(f'<text x="8" y="{18 * (i + 1)}" font-family="monospace" font-size="13" fill="#222">{html.escape(text)}</text>')
    svg.append('<polygon points="' + " ".join("%.1f,%.1f" % px(p) for p in truth) + '" fill="none" stroke="black" stroke-width="3"/>')
    png = np.full((height, width, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(png, text, (8, 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.polylines(png, [np.round([px(p) for p in truth]).astype(np.int32)], True, (0, 0, 0), 3, cv2.LINE_AA)
    if pred is not None:
        for j in range(len(pred)):
            a, b = pred[j], pred[(j + 1) % len(pred)]
            inferred = j < len(labels) and labels[j] == "inferred"
            color = (0, 128, 255) if inferred else (200, 80, 0)                       # BGR: orange dashed / blue solid
            (x1, y1), (x2, y2) = px(a), px(b)
            stroke = "#ff8000" if inferred else "#0050c8"
            dash = ' stroke-dasharray="9 6"' if inferred else ""
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="2.5"{dash}/>')
            length = math.hypot(x2 - x1, y2 - y1)
            if inferred:
                for s0 in np.arange(0, length, 15):
                    s1 = min(s0 + 9, length)
                    p0 = (round(x1 + (x2 - x1) * s0 / length), round(y1 + (y2 - y1) * s0 / length))
                    p1 = (round(x1 + (x2 - x1) * s1 / length), round(y1 + (y2 - y1) * s1 / length))
                    cv2.line(png, p0, p1, color, 2, cv2.LINE_AA)
            else:
                cv2.line(png, (round(x1), round(y1)), (round(x2), round(y2)), color, 2, cv2.LINE_AA)
        for p in pred:
            x, y = px(p); svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#0050c8"/>')
            cv2.circle(png, (round(x), round(y)), 3, (200, 80, 0), -1, cv2.LINE_AA)
    legend_y = height - 8
    svg.append(f'<text x="8" y="{legend_y}" font-family="monospace" font-size="12" fill="#222">black: true outline   blue solid: observed edge   orange dashed: inferred edge</text>')
    cv2.putText(png, "black: true outline   blue solid: observed edge   orange dashed: inferred edge", (8, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    svg.append("</svg>")
    (out_dir / "overlay.svg").write_text("\n".join(svg), encoding="utf-8")
    cv2.imwrite(str(out_dir / "overlay.png"), png)
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return {"svg": str(out_dir / "overlay.svg"), "png": str(out_dir / "overlay.png"), "json": str(out_dir / "evaluation.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prediction", type=Path, required=True, help="footprint.py or room_hypothesis.estimate_polygon JSON")
    parser.add_argument("--capture", type=Path, help="capture folder; reads <capture>/private_ground_truth/scene.json")
    parser.add_argument("--scene", type=Path, help="explicit private scene.json (overrides --capture)")
    parser.add_argument("--room-id", help="room id for a multi-room house layout scene")
    parser.add_argument("--height", type=Path, required=True, help="estimate_height.py output JSON (prediction scale source)")
    parser.add_argument("--manifest", type=Path, help="pipeline manifest / merge report holding retained_capture_fraction")
    parser.add_argument("--axis-sign-resolved", action="store_true", help="disable the mirrored-prediction attempt")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW folder for evaluation.json, overlay.svg, overlay.png")
    args = parser.parse_args()
    scene_path = args.scene or (args.capture / "private_ground_truth" / "scene.json" if args.capture else None)
    if scene_path is None:
        parser.error("give --scene or --capture")
    if args.output_dir.exists():
        parser.error(f"output folder already exists: {args.output_dir}")
    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8")) if p and Path(p).exists() else None
    result = evaluate_floorplan(read(args.prediction), read(scene_path), read(args.height), manifest=read(args.manifest),
                                axis_sign_resolved=args.axis_sign_resolved, room_id=args.room_id)
    paths = write_overlay(result, args.output_dir)
    print(json.dumps({"status": result["status"], "failure_reasons": result["failure_reasons"],
                      "custom_1_5cm_pass": result["custom_1_5cm_horizontal_diagnostic"]["pass"],
                      "official_3pct_pass": result["wall_length_3_percent_threshold_oracle_normalised"]["pass"],
                      "iou": result["metrics"]["iou"] if result["metrics"] else None, **paths}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
