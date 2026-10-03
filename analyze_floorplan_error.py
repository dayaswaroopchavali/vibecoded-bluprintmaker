"""EVALUATION ONLY (reads private truth): is the wall-length error of a DA3 floor plan caused by a vertical (ceiling-height)
bias, or by footprint rasterisation / horizontal deformation?

Nothing here feeds back to inference. One fixed alignment is used for every surface: a single similarity fitted from the
backend's camera centres to the ground-truth camera centres (never refitted per surface, no anisotropic scale, no forced
rectangle, no simulator dimensions in the prediction). Reported:

 1. residuals of the DA3 point cloud (in the truth frame) to the true floor, ceiling and each wall plane: signed distance
    (positive = toward the room interior), point share, coverage of the true wall length, height range per wall;
 2. for each structural axis hypothesis whose axis is the true vertical under that fixed alignment: the footprint polygon
    versus its SOURCE floor/ceiling slab points (offset of each edge from the point extreme, raw support extent versus
    raster-occupied versus closed versus polygon edge length: does rasterisation/closing/simplification shorten walls?);
 3. an anchor diagnostic: normalise the prediction by (1) the ceiling = 3.0 m protocol and (2) ONE known true wall (oracle
    diagnostic), then show the wall errors under both so height bias can be told apart from horizontal deformation.

Edge-to-wall correspondence is an evaluator oracle (direction + midpoint after the fixed alignment). No benchmark stage is
claimed complete.

  python analyze_floorplan_error.py --variant <variant folder> --prediction <prediction.npz with true source_indices>
         --pipeline-run <run_pipeline from-prediction dir> --output <NEW dir>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import audit_camera_trajectory as A

PROTOCOL_CEILING_M = 3.0
WINDOW_M = 0.35                 # point-to-surface assignment window (a point farther than this from every true surface is unassigned)
COVERAGE_BIN_M = 0.10
COVERAGE_MIN_POINTS = 3
MATCH_ANGLE_DEG = 25.0
MATCH_DISTANCE_M = 0.8
AXIS_ALIGNED_DEG = 15.0
# verdict thresholds (fractions); documented, not tuned against any one result
ERROR_MATERIAL = 0.03           # a wall error of at least this needs explaining
VERTICAL_MATERIAL = 0.03        # vertical overestimate of at least this counts as a bias
HORIZONTAL_SMALL = 0.02         # horizontal deformation below this is "small"
COVERAGE_LIMITED_BELOW = 0.90   # point support spanning less than this fraction of a true wall length is coverage-limited
RASTER_MATERIAL = 0.03         # polygon edge shorter than the raw support extent by this fraction of the edge


# ------------------------------------------------------------------------------------------------ truth geometry
def room_from_scene(scene: dict) -> dict:
    w, l, h = (float(scene[k]) for k in ("width", "length", "height"))
    if not all(math.isfinite(v) and v > 0 for v in (w, l, h)):
        raise ValueError("scene lacks positive width/length/height")
    return {"width": w, "length": l, "height": h}


def true_surfaces(room: dict) -> list[dict]:
    """Floor, ceiling and four walls (simulator convention: x east, y north, z up; room centred on the origin)."""
    w, l, h = room["width"], room["length"], room["height"]
    return [
        {"name": "floor", "kind": "horizontal", "n_in": (0, 0, 1), "p": (0, 0, 0), "along": None, "length": None},
        {"name": "ceiling", "kind": "horizontal", "n_in": (0, 0, -1), "p": (0, 0, h), "along": None, "length": None},
        {"name": "north", "kind": "wall", "n_in": (0, -1, 0), "p": (0, l / 2, 0), "along": 0, "length": w},
        {"name": "east", "kind": "wall", "n_in": (-1, 0, 0), "p": (w / 2, 0, 0), "along": 1, "length": l},
        {"name": "south", "kind": "wall", "n_in": (0, 1, 0), "p": (0, -l / 2, 0), "along": 0, "length": w},
        {"name": "west", "kind": "wall", "n_in": (1, 0, 0), "p": (-w / 2, 0, 0), "along": 1, "length": l},
    ]


def true_wall_segments(room: dict) -> dict:
    w, l = room["width"], room["length"]
    return {"north": np.array([[-w / 2, l / 2], [w / 2, l / 2]]), "east": np.array([[w / 2, -l / 2], [w / 2, l / 2]]),
            "south": np.array([[-w / 2, -l / 2], [w / 2, -l / 2]]), "west": np.array([[-w / 2, -l / 2], [-w / 2, l / 2]])}


# ------------------------------------------------------------------------------------------------ fixed alignment
def fixed_alignment(extrinsics, source_indices, gt_poses) -> dict:
    """ONE similarity from the backend's camera centres to the true centres (oracle diagnostic, never refitted)."""
    centres = A.camera_centers(np.asarray(extrinsics, float))
    gt = np.asarray([gt_poses[int(i)]["position"] for i in source_indices], float)
    fit = A.fit_similarity(centres, gt)
    full_extent = float(np.linalg.norm(np.ptp(np.asarray([p["position"] for p in gt_poses], float), axis=0)))
    return {"scale": fit["scale"], "rotation": np.asarray(fit["rotation"]), "translation": np.asarray(fit["translation"]),
            "centre_rmse_m": fit["rmse"], "centre_rmse_fraction_of_full_gt_extent": fit["rmse"] / full_extent,
            "frames_fitted": int(len(centres)), "rotation_well_conditioned": fit["alignment_rotation_well_conditioned"]}


def to_truth_frame(points, align) -> np.ndarray:
    return align["scale"] * (np.asarray(points, float) @ align["rotation"].T) + align["translation"]


# ------------------------------------------------------------------------------------------------ 1. residuals
def _stats(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "abs_p90": float(np.percentile(np.abs(v), 90)), "mean": float(v.mean())}


def assign_points(points, room, window: float = WINDOW_M):
    """Assign each point to its nearest true surface within `window` (and inside that face's lateral extent)."""
    X = np.asarray(points, float)
    w, l, h = room["width"], room["length"], room["height"]
    surfaces = true_surfaces(room)
    d = np.full((len(X), len(surfaces)), np.inf)
    signed = np.zeros((len(X), len(surfaces)))
    for k, s in enumerate(surfaces):
        sd = (X - np.asarray(s["p"])) @ np.asarray(s["n_in"], float)
        signed[:, k] = sd
        if s["kind"] == "horizontal":
            lateral = (np.abs(X[:, 0]) <= w / 2 + window) & (np.abs(X[:, 1]) <= l / 2 + window)
        else:
            along = X[:, s["along"]]
            lateral = (np.abs(along) <= s["length"] / 2 + window) & (X[:, 2] >= -window) & (X[:, 2] <= h + window)
        ok = lateral & (np.abs(sd) <= window)
        d[ok, k] = np.abs(sd[ok])
    best = d.argmin(axis=1)
    assigned = np.isfinite(d[np.arange(len(X)), best])
    return surfaces, np.where(assigned, best, -1), signed


def residual_analysis(points_truth, room, window: float = WINDOW_M) -> dict:
    """Signed residuals (positive toward the room interior) of a truth-frame cloud to each true plane."""
    X = np.asarray(points_truth, float)
    X = X[np.isfinite(X).all(axis=1)]
    surfaces, owner, signed = assign_points(X, room, window)
    h = room["height"]
    out = {"point_count": int(len(X)), "assignment_window_m": window, "unassigned_share": float((owner < 0).mean()),
           "sign_convention": "signed distance to the TRUE plane, positive toward the room interior (a wall too far OUT is negative; a ceiling too HIGH is negative)",
           "window_note": "residuals are truncated at the window: window_saturated_share near 1 means the true offset may exceed the window",
           "surfaces": {}}
    for k, s in enumerate(surfaces):
        m = owner == k
        sd = signed[m, k]
        row = {"points": int(m.sum()), "point_share": float(m.mean()), "signed_distance_m": _stats(sd),
               "window_saturated_share": float((np.abs(sd) > 0.9 * window).mean()) if m.any() else None}
        if s["kind"] == "wall" and m.any():
            along = X[m, s["along"]]
            z = X[m, 2]
            L = s["length"]
            edges = np.arange(-L / 2, L / 2 + COVERAGE_BIN_M, COVERAGE_BIN_M)
            counts, _ = np.histogram(np.clip(along, -L / 2, L / 2), edges)
            row.update(true_length_m=L, coverage_of_true_length_by_bins=float((counts >= COVERAGE_MIN_POINTS).mean()),
                       along_extent_p1_p99_m=[float(np.percentile(along, 1)), float(np.percentile(along, 99))],
                       along_extent_fraction_of_true_length=float((np.percentile(along, 99) - np.percentile(along, 1)) / L),
                       height_range_p2_p98_m=[float(np.percentile(z, 2)), float(np.percentile(z, 98))],
                       height_range_fraction_of_true_height=float((np.percentile(z, 98) - np.percentile(z, 2)) / h),
                       coverage_note="gaps include the true door/window openings; this is not a defect measure by itself")
        if s["kind"] == "horizontal" and m.any():
            row["median_z_m"] = float(np.median(X[m, 2]))
        out["surfaces"][s["name"]] = row
    support = {}
    for k, s in enumerate(surfaces):
        m = owner == k
        if s["kind"] == "horizontal" and m.sum() >= 50:
            support[s["name"]] = {"x_extent_p1_p99_m": float(np.percentile(X[m, 0], 99) - np.percentile(X[m, 0], 1)),
                                  "y_extent_p1_p99_m": float(np.percentile(X[m, 1], 99) - np.percentile(X[m, 1], 1))}
    out["floor_ceiling_support_extent"] = {"surfaces": support, "true_width_x_m": room["width"], "true_length_y_m": room["length"],
                                           "note": "raw point support of the floor/ceiling along the true x/y axes (no raster, no polygon)"}
    fl, ce = out["surfaces"]["floor"], out["surfaces"]["ceiling"]
    if fl["points"] and ce["points"]:
        sep = ce["median_z_m"] - fl["median_z_m"]
        out["vertical_extent"] = {"ceiling_minus_floor_median_m": float(sep), "true_height_m": h, "ratio_to_true": float(sep / h),
                                  "vertical_error_fraction": float(sep / h - 1),
                                  "note": "uses the single fixed camera-centre similarity; a scale error in that fit would move horizontal and vertical alike"}
    else:
        out["vertical_extent"] = None
    return out


# ------------------------------------------------------------------------------------------------ 2. footprint vs source points
def _outward_unit(a, b):
    d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-300)
    return d, np.array([d[1], -d[0]])         # CCW polygon: outward normal = (dy, -dx)


def edge_source_comparison(poly_xy, slab_xy, tol: float, cell: float | None = None, closing_cells: int | None = None) -> list[dict]:
    """Per polygon edge: how the contour sits against the SOURCE slab points and how long the raw support is along it.

    Prediction units. `poly_xy` is CCW. Points within `2*tol` of the edge line (any position along the edge, extended
    by 25% of its length beyond each end so the support may exceed the polygon edge) define that wall's raw support. The raster stages are recomputed with the footprint's
    own cell size / closing radius (sampling `slab_xy`) so each stage's extent can be compared with the polygon edge.
    """
    P = np.asarray(poly_xy, float)
    S = np.asarray(slab_xy, float)
    n = len(P)
    region = None
    if cell and S.shape[0]:
        pad = (closing_cells or 1) + 4
        lo = S.min(axis=0) - pad * cell
        shape = (np.ceil((S.max(axis=0) - lo) / cell).astype(int) + pad + 1)[::-1]
        if shape.max() <= 6000:
            occ = np.zeros(tuple(shape), np.uint8)
            cols = np.floor((S[:, 0] - lo[0]) / cell).astype(int)
            rows = np.floor((S[:, 1] - lo[1]) / cell).astype(int)
            occ[rows, cols] = 1
            k = int(closing_cells or 1)
            closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            filled = np.zeros_like(closed)
            if contours:
                big = max(contours, key=cv2.contourArea)
                cv2.drawContours(filled, [big], -1, 1, thickness=-1)
            region = {"lo": lo, "occupied": occ, "closed_filled_largest": filled}
    rows_out = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        d, nrm = _outward_unit(a, b)
        length = float(np.linalg.norm(b - a))
        s = (S - a) @ d
        out_dist = (S - a) @ nrm                      # positive = beyond the contour (outside)
        band = (np.abs(out_dist) <= 2 * tol) & (s >= -0.25 * length) & (s <= 1.25 * length)
        row = {"edge": i, "polygon_edge_length": length, "tolerance": float(tol), "band_points": int(band.sum())}
        if band.sum() >= 5:
            sb = s[band]
            lo_s, hi_s = float(np.percentile(sb, 0.5)), float(np.percentile(sb, 99.5))
            row["raw_support_extent"] = {"p0.5_p99.5": hi_s - lo_s, "min_max": float(sb.max() - sb.min())}
            inner = (s >= 0.1 * length) & (s <= 0.9 * length) & (np.abs(out_dist) <= 3 * tol)
            if inner.sum() >= 5:
                ext = float(np.percentile(out_dist[inner], 99.5))
                row["contour_vs_point_extreme"] = {"extreme_outward_p99.5": ext, "extreme_outward_max": float(out_dist[inner].max()),
                                                   "contour_inside_extreme_by": ext,
                                                   "note": "positive = the contour sits that far INSIDE the outermost source points (within 3 tol)"}
            row["polygon_shorter_than_raw_support_by"] = row["raw_support_extent"]["p0.5_p99.5"] - length
            row["polygon_shorter_than_raw_support_fraction_of_edge"] = (row["raw_support_extent"]["p0.5_p99.5"] - length) / length
        else:
            row["raw_support_extent"] = None
        if region is not None:
            for name in ("occupied", "closed_filled_largest"):
                ys, xs = np.nonzero(region[name])
                if len(xs) == 0:
                    row[f"raster_{name}_extent"] = None
                    continue
                c = region["lo"] + (np.stack([xs, ys], axis=1) + 0.5) * cell
                sc, oc = (c - a) @ d, (c - a) @ nrm
                bm = (np.abs(oc) <= 2 * tol) & (sc >= -0.25 * length) & (sc <= 1.25 * length)
                row[f"raster_{name}_extent"] = float(sc[bm].max() - sc[bm].min() + cell) if bm.any() else None
        rows_out.append(row)
    return rows_out


# ------------------------------------------------------------------------------------------------ 3. anchors
def match_walls(poly_truth_xy, room) -> dict:
    """Oracle correspondence true wall -> polygon edge (direction within 25 deg, nearest midpoint within 0.8 m), one-to-one."""
    P = np.asarray(poly_truth_xy, float)
    segs = true_wall_segments(room)
    cands = []
    for name, seg in segs.items():
        t = seg[1] - seg[0]
        tm = seg.mean(axis=0)
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            e = b - a
            cos = abs(float(t @ e)) / max(float(np.linalg.norm(t) * np.linalg.norm(e)), 1e-12)
            ang = math.degrees(math.acos(min(1.0, cos)))
            dist = float(np.linalg.norm((a + b) / 2 - tm))
            if ang <= MATCH_ANGLE_DEG and dist <= MATCH_DISTANCE_M:
                cands.append((dist + 0.01 * ang, name, i))
    cands.sort()
    used_w, used_e, out = set(), set(), {}
    for _c, name, i in cands:
        if name not in used_w and i not in used_e:
            used_w.add(name); used_e.add(i); out[name] = i
    return out


def anchor_diagnostics(pred_lengths: dict, pred_height: float, true_lengths: dict, true_height: float) -> dict:
    """Wall errors under (1) ceiling = 3.0 m normalisation and (2) a single true-wall anchor, for every choice of anchor.

    pred_lengths / true_lengths: wall name -> length (same names; prediction in arbitrary units, truth in metres).
    Under (2) the anchor wall is made exact; its implied height is `pred_height * scale` versus the true height. A pure
    vertical scale error gives zero error on the other walls under (2) while (1) shows a uniform wall error; a horizontal
    deformation leaves other walls wrong under (2).
    """
    names = sorted(set(pred_lengths) & set(true_lengths))
    out = {"walls_compared": names, "protocol_1_ceiling_3m": {}, "protocol_2_single_wall_anchor_oracle": {}}
    if pred_height <= 0 or true_height <= 0 or not names:
        out["status"] = "insufficient_input"
        return out
    s1 = PROTOCOL_CEILING_M / pred_height
    t1 = PROTOCOL_CEILING_M / true_height
    out["protocol_1_ceiling_3m"] = {"prediction_scale": s1, "truth_scale": t1,
                                    "wall_relative_error": {n: pred_lengths[n] * s1 / (true_lengths[n] * t1) - 1 for n in names},
                                    "wall_abs_error_cm": {n: 100 * (pred_lengths[n] * s1 - true_lengths[n] * t1) for n in names}}
    for a in names:
        s2 = true_lengths[a] / pred_lengths[a]
        out["protocol_2_single_wall_anchor_oracle"][a] = {
            "anchor_wall_true_length_m": true_lengths[a], "scale": s2, "label": "EVALUATION-ONLY oracle: the anchor's true length scales the prediction",
            "other_wall_relative_error": {n: pred_lengths[n] * s2 / true_lengths[n] - 1 for n in names if n != a},
            "implied_height_m": pred_height * s2, "implied_height_error_fraction": pred_height * s2 / true_height - 1}
    out["status"] = "ok"
    return out


def points_only_anchor(residuals: dict, room: dict) -> dict:
    """Anchor diagnostic from raw floor/ceiling point extents and the floor-ceiling separation (no footprint, truth-frame metres).

    Wall lengths: north/south = mean floor/ceiling extent along x, east/west = along y. Predicted height = ceiling - floor.
    """
    sup = residuals.get("floor_ceiling_support_extent", {}).get("surfaces", {})
    ve = residuals.get("vertical_extent")
    if not sup or not ve:
        return {"status": "insufficient_input"}
    ex = float(np.mean([s["x_extent_p1_p99_m"] for s in sup.values()]))
    ey = float(np.mean([s["y_extent_p1_p99_m"] for s in sup.values()]))
    pred = {"north": ex, "south": ex, "east": ey, "west": ey}
    true = {s["name"]: s["length"] for s in true_surfaces(room) if s["kind"] == "wall"}
    out = anchor_diagnostics(pred, ve["ceiling_minus_floor_median_m"], true, room["height"])
    out["wall_lengths_from"] = "mean floor/ceiling raw point support extent (p1-p99) along x (north/south) and y (east/west)"
    out["horizontal_extent_error_fraction_fixed_similarity"] = {"x_width": ex / room["width"] - 1, "y_length": ey / room["length"] - 1}
    out["coverage_limited"] = bool(min(ex / room["width"], ey / room["length"]) < COVERAGE_LIMITED_BELOW)
    out["coverage_note"] = ("floor/ceiling support covers less than %.0f%% of a true wall length: point-support lengths measure how much of the room was SEEN, "
                            "not the scale, so horizontal 'deformation' is confounded with partial coverage" % (100 * COVERAGE_LIMITED_BELOW)) if out["coverage_limited"] else None
    return out


def verdict(anchor: dict, vertical_error_fraction, raster_shortening_fractions, error_material=ERROR_MATERIAL, coverage_limited: bool = False) -> dict:
    """Which hypothesis do the numbers support? Explicit thresholds; reports the evidence it used."""
    p1 = anchor.get("protocol_1_ceiling_3m", {}).get("wall_relative_error", {})
    if not p1:
        return {"status": "not_assessable", "reason": "no matched walls"}
    max_err1 = max(abs(v) for v in p1.values())
    p2 = anchor.get("protocol_2_single_wall_anchor_oracle", {})
    max_err2 = max((abs(v) for a in p2.values() for v in a["other_wall_relative_error"].values()), default=None)
    implied_h = [abs(a["implied_height_error_fraction"]) for a in p2.values()]
    ev = vertical_error_fraction
    raster = max(raster_shortening_fractions) if raster_shortening_fractions else None
    ev_ok = ev is not None
    material_error = max_err1 >= error_material
    vertical_bias = bool(ev_ok and ev >= VERTICAL_MATERIAL)
    horiz_small = bool(max_err2 is not None and max_err2 <= HORIZONTAL_SMALL)
    raster_material = bool(raster is not None and raster >= RASTER_MATERIAL)
    reasons = {"max_abs_wall_error_protocol1": max_err1, "vertical_error_fraction_points": ev, "max_other_wall_error_under_single_wall_anchor": max_err2,
               "max_polygon_shortening_vs_raw_support_fraction_of_edge": raster,
               "thresholds": {"error_material": error_material, "vertical_material": VERTICAL_MATERIAL,
                              "horizontal_small": HORIZONTAL_SMALL, "raster_material": RASTER_MATERIAL}}
    if not material_error:
        return {"status": "no_material_wall_error", "supported": None, **reasons,
                "statement": "wall errors under the 3.0 m protocol are below the materiality threshold: nothing to attribute"}
    reasons["raster_assessed"] = raster is not None
    reasons["coverage_limited"] = bool(coverage_limited)
    if vertical_bias and coverage_limited:
        return {"status": "mixed_or_inconclusive", **reasons,
                "statement": "INCONCLUSIVE: the vertical extent is over-estimated, but the point support covers too little of each wall for lengths to measure scale "
                             "(partial coverage is confounded with horizontal deformation)."}
    if vertical_bias and horiz_small and not raster_material:
        tail = ("the footprint polygon is not materially shorter than its raw support." if raster is not None else
                "polygon rasterisation was NOT assessed (no complete footprint): supported at the point-support level only.")
        return {"status": "supported", "hypothesis": "vertical (ceiling-height) overestimate explains the wall error", **reasons,
                "statement": "SUPPORTED: walls are mutually consistent in length once ONE true wall fixes the horizontal scale (other-wall error small), "
                             "the vertical extent is over-estimated, and " + tail}
    if raster_material and not (vertical_bias and horiz_small):
        return {"status": "not_supported", "hypothesis": "rasterisation/closing/simplification shortens walls", **reasons,
                "statement": "NOT SUPPORTED as a vertical-bias effect: polygon edges are materially shorter than their raw point support and/or the vertical bias does not account for the error."}
    if not vertical_bias:
        return {"status": "not_supported", **reasons,
                "statement": "NOT SUPPORTED: the point cloud's vertical extent is not materially over-estimated, so a height bias cannot explain the wall error."}
    return {"status": "mixed_or_inconclusive", **reasons,
            "statement": "INCONCLUSIVE: a vertical over-estimate exists but horizontal deformation (other walls wrong under a single-wall anchor) and/or polygon shortening also contribute."}


# ------------------------------------------------------------------------------------------------ driver
def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_hypothesis(k, pipe: Path, planes_doc, pred_points, align, room, residuals):
    fp_path, hh_path = pipe / f"footprint_axis_hypothesis_{k}.json", pipe / f"height_axis_hypothesis_{k}.json"
    out = {"hypothesis": k}
    fp = _load(fp_path) if fp_path.exists() else None
    hh = _load(hh_path) if hh_path.exists() else {}
    if not fp or fp.get("status") != "complete_polygon_hypothesis":
        out["status"] = "footprint_incomplete"
        out["reasons"] = [b.get("reason") for b in (fp or {}).get("incomplete_boundaries", [])] or ["footprint missing"]
        return out
    up = np.asarray(fp["axes_world"]["up"], float)
    up_truth = align["rotation"] @ up
    ang = float(np.degrees(np.arccos(np.clip(abs(up_truth[2]) / np.linalg.norm(up_truth), 0, 1))))
    out["axis_angle_to_true_vertical_deg_under_fixed_alignment"] = ang
    if ang > AXIS_ALIGNED_DEG:
        out["status"] = "axis_not_true_vertical"
        out["note"] = "this hypothesis is a different structural axis: its 'footprint' is not a horizontal plan, so wall/height comparison does not apply (not discarded, reported)"
        return out
    scale = align["scale"]
    P = np.asarray(fp["vertices_xy"], float)
    n = len(P)
    # footprint in the truth frame via the single fixed similarity
    V = to_truth_frame(np.asarray(fp["vertices_world"], float), align)
    lengths_pred_units = [float(np.linalg.norm(P[(i + 1) % n] - P[i])) for i in range(n)]
    lengths_truth_m = [float(np.linalg.norm(V[(i + 1) % n, :2] - V[i, :2])) for i in range(n)]
    out["polygon"] = {"edge_count": n, "edge_lengths_prediction_units": lengths_pred_units, "edge_lengths_in_truth_frame_m_fixed_similarity": lengths_truth_m}
    match = match_walls(V[:, :2], room)
    out["wall_to_edge_oracle_correspondence"] = match
    true_len = {s["name"]: s["length"] for s in true_surfaces(room) if s["kind"] == "wall"}
    out["fixed_alignment_wall_error"] = {name: {"edge": i, "pred_length_m": lengths_truth_m[i], "true_length_m": true_len[name],
                                                 "relative_error": lengths_truth_m[i] / true_len[name] - 1,
                                                 "abs_error_cm": 100 * (lengths_truth_m[i] - true_len[name])} for name, i in match.items()}
    # source slab points in the footprint frame
    sup = fp["support"]
    planes = planes_doc["planes"]
    center = np.asarray(fp["coordinate_origin_world"], float)
    e1, e2 = np.asarray(fp["axes_world"]["e1"], float), np.asarray(fp["axes_world"]["e2"], float)
    rel = pred_points - center
    xy_all = np.stack([rel @ e1, rel @ e2], axis=1)
    extent = float(fp["extent"])
    near = np.abs(xy_all).max(axis=1) <= 1.2 * extent
    mask = np.zeros(len(pred_points), bool)
    used = []
    for pid, cnt in zip(sup["plane_ids"], sup["points_near_plane"]):
        nn = np.asarray(planes[pid]["normal"], float)
        norm = np.linalg.norm(nn)
        nn, dd = nn / norm, float(planes[pid]["offset"]) / norm
        m = (np.abs(pred_points @ nn + dd) < sup["slab_half_thickness"]) & near
        mask |= m
        used.append({"plane_id": pid, "points": int(m.sum()), "recorded_points_near_plane": cnt})
    out["slab"] = {"planes": used, "slab_half_thickness_units": sup["slab_half_thickness"], "points_used": int(mask.sum())}
    tol = float(fp["wall_evidence"]["tolerance"])
    raster = fp.get("raster", {})
    rows = edge_source_comparison(P, xy_all[mask], tol, raster.get("cell"), raster.get("closing_radius_cells"))
    for r in rows:
        r["to_truth_metres_factor"] = scale
        for key in ("polygon_edge_length",):
            r["polygon_edge_length_m"] = r[key] * scale
        if r.get("polygon_shorter_than_raw_support_by") is not None:
            r["polygon_shorter_than_raw_support_by_m"] = r["polygon_shorter_than_raw_support_by"] * scale
    out["edges_vs_source_points"] = rows
    out["edges_vs_source_points_note"] = ("edges from the footprint's own consensus scale; tolerance and raster cell/closing as recorded in the footprint JSON. "
                                          "'polygon_shorter_than_raw_support' > 0 means the polygon edge is shorter than the raw slab-point support along that wall")
    matched_edges = set(match.values())
    shortening = [r["polygon_shorter_than_raw_support_fraction_of_edge"] for r in rows if r["edge"] in matched_edges and r.get("polygon_shorter_than_raw_support_fraction_of_edge") is not None]
    out["raster_shortening_matched_edges"] = shortening
    # anchors
    sep_h = hh.get("separation_unscaled") if hh.get("status") == "estimated_unscaled_separation" else None
    sep_fp = fp.get("floor_ceiling_separation_unscaled")
    out["height_units"] = {"estimate_height_separation_unscaled": sep_h, "footprint_slab_median_separation_unscaled": sep_fp}
    pred_len = {name: lengths_pred_units[i] for name, i in match.items()}
    anchors = anchor_diagnostics(pred_len, sep_h, true_len, room["height"]) if sep_h else {"status": "no_height_estimate"}
    out["anchor_diagnostics"] = anchors
    out["anchor_diagnostics_using_footprint_slab_separation"] = anchor_diagnostics(pred_len, sep_fp, true_len, room["height"]) if sep_fp else None
    ve = residuals.get("vertical_extent")
    out["height_estimate_vs_truth_under_fixed_similarity"] = {
        "estimate_height_ratio": (sep_h * scale / room["height"]) if sep_h else None,
        "slab_median_ratio": (sep_fp * scale / room["height"]) if sep_fp else None,
        "point_plane_ratio": ve["ratio_to_true"] if ve else None}
    out["verdict"] = verdict(anchors, (sep_h * scale / room["height"] - 1) if sep_h else None, shortening) if anchors.get("status") == "ok" else {"status": "not_assessable", "reason": anchors.get("status")}
    out["status"] = "analysed"
    return out


def analyze(variant: Path, prediction: Path, pipe: Path, output: Path, log=print) -> dict:
    from extract_planes import points_from_prediction
    variant, output = Path(variant), Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists (never overwrite): {output}")
    scene = _load(variant / "private_ground_truth" / "scene.json")
    gt_poses = _load(variant / "private_ground_truth" / "path_00.json")["poses"]
    room = room_from_scene(scene)
    with np.load(prediction, allow_pickle=False) as d:
        ex, si = np.asarray(d["extrinsics"], float), np.asarray(d["source_indices"], int)
        pts = points_from_prediction(d)
    align = fixed_alignment(ex, si, gt_poses)
    truth_pts = to_truth_frame(pts, align)
    residuals = residual_analysis(truth_pts, room)
    planes_doc = _load(pipe / "planes.json")
    hyps = []
    k = 0
    while (pipe / f"footprint_axis_hypothesis_{k}.json").exists():
        hyps.append(analyze_hypothesis(k, Path(pipe), planes_doc, pts, align, room, residuals))
        k += 1
    doc = {"schema_version": 1, "evaluation_only": True, "benchmark_stage_complete": False, "variant": variant.name,
           "inputs": {"prediction": str(prediction), "prediction_sha256_prefix": A.sha256(Path(prediction))[:16], "pipeline_run": str(pipe),
                      "scene_sha256_prefix_evaluation_only": A.sha256(variant / "private_ground_truth" / "scene.json")[:16]},
           "true_room_m": room,
           "fixed_alignment": {"kind": "single similarity from camera centres to true camera centres (never refitted per surface)",
                               **{k2: (v.tolist() if isinstance(v, np.ndarray) else v) for k2, v in align.items()}},
           "residuals_to_true_planes": residuals,
           "points_only_diagnostic": _points_only_block(residuals, room),
           "hypotheses": hyps,
           "limitations": ["Evaluation only: truth is used for the alignment, plane residuals and the oracle edge/wall correspondence.",
                           "One fixed similarity: any error in the camera-centre scale shifts all surfaces together.",
                           "Point residuals are windowed (see window_saturated_share).",
                           "No claim that any benchmark stage is complete."]}
    output.mkdir(parents=True)
    (output / "floorplan_error_analysis.json").write_text(json.dumps(_clean(doc), indent=2, allow_nan=False), encoding="utf-8")
    (output / "reading.md").write_text(reading(doc), encoding="utf-8")
    return doc


def _points_only_block(residuals, room):
    anchor = points_only_anchor(residuals, room)
    ve = residuals.get("vertical_extent")
    v = (verdict(anchor, ve["vertical_error_fraction"] if ve else None, [], coverage_limited=anchor.get("coverage_limited", False))
         if anchor.get("status") == "ok" else {"status": "not_assessable"})
    return {"anchor_diagnostics": anchor, "verdict": v,
            "note": "independent of the footprint stage: uses raw floor/ceiling point extents and the floor-ceiling separation under the fixed similarity"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _f(v, nd=3, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:+.1f}%" if pct else f"{v:.{nd}f}"


def reading(doc: dict) -> str:
    r = doc["residuals_to_true_planes"]
    L = [f"# Floor-plan error reading: {doc['variant']}", "",
         "Evaluation only (reads private truth). One fixed camera-centre similarity; positive residual = toward the room interior. No stage is claimed complete.", "",
         f"Fixed alignment: {doc['fixed_alignment']['frames_fitted']} frames, centre RMSE {_f(doc['fixed_alignment']['centre_rmse_fraction_of_full_gt_extent'], 3)} of the full GT extent.", "",
         "## Point cloud versus the true planes", "", "| surface | point share | median m | p10 m | p90 m | window-saturated | wall coverage | along-extent / true length | height range / true |", "|---|---|---|---|---|---|---|---|---|"]
    for name, s in r["surfaces"].items():
        sd = s["signed_distance_m"] or {}
        L.append(f"| {name} | {_f(s['point_share'], 3)} | {_f(sd.get('median'))} | {_f(sd.get('p10'))} | {_f(sd.get('p90'))} | {_f(s.get('window_saturated_share'), 2)} | "
                 f"{_f(s.get('coverage_of_true_length_by_bins'), 2)} | {_f(s.get('along_extent_fraction_of_true_length'), 2)} | {_f(s.get('height_range_fraction_of_true_height'), 2)} |")
    ve = r.get("vertical_extent")
    if ve:
        L += ["", f"Vertical extent (ceiling-floor medians) / true height = {_f(ve['ratio_to_true'], 3)} ({_f(ve['vertical_error_fraction'], pct=True)}); unassigned point share {_f(r['unassigned_share'], 3)}."]
    po = doc["points_only_diagnostic"]
    pa = po["anchor_diagnostics"]
    L += ["", "## Point-support diagnostic (no footprint needed)"]
    sup = r["floor_ceiling_support_extent"]
    L.append("Raw floor/ceiling support extent (p1-p99) vs true: " + "; ".join(
        f"{n}: x {_f(s['x_extent_p1_p99_m'])} m / y {_f(s['y_extent_p1_p99_m'])} m" for n, s in sup["surfaces"].items()) +
        f" (true {_f(sup['true_width_x_m'])} x {_f(sup['true_length_y_m'])} m)")
    if pa.get("status") == "ok":
        L.append("(1) ceiling = 3.0 m protocol wall errors: " + "; ".join(f"{n} {_f(v, pct=True)}" for n, v in pa["protocol_1_ceiling_3m"]["wall_relative_error"].items()))
        for an, d in pa["protocol_2_single_wall_anchor_oracle"].items():
            L.append(f"- (2) anchor {an} true wall (oracle): other walls " + "; ".join(f"{n} {_f(v, pct=True)}" for n, v in d["other_wall_relative_error"].items()) +
                     f"; implied height error {_f(d['implied_height_error_fraction'], pct=True)}")
    if pa.get("coverage_note"):
        L.append("Coverage caveat: " + pa["coverage_note"])
    L.append(f"**Verdict (point level):** {po['verdict'].get('statement') or po['verdict']}")
    for h in doc["hypotheses"]:
        L += ["", f"## Axis hypothesis {h['hypothesis']}: {h['status']}"]
        if h["status"] != "analysed":
            L.append(h.get("note") or ", ".join(map(str, h.get("reasons", []))))
            continue
        fa = h["fixed_alignment_wall_error"]
        L.append(f"Footprint {h['polygon']['edge_count']} edges; wall errors under the fixed similarity (oracle correspondence): " +
                 "; ".join(f"{n} {_f(v['relative_error'], pct=True)} ({v['abs_error_cm']:+.1f} cm)" for n, v in fa.items()))
        L.append("")
        L.append("| edge | polygon length | raw support extent | raster occupied | raster closed | polygon shorter than raw (frac of edge) | contour inside outermost points |")
        L.append("|---|---|---|---|---|---|---|")
        for e in h["edges_vs_source_points"]:
            raw = (e.get("raw_support_extent") or {}).get("p0.5_p99.5")
            ins = (e.get("contour_vs_point_extreme") or {}).get("contour_inside_extreme_by")
            L.append(f"| {e['edge']} | {_f(e['polygon_edge_length'], 3)} | {_f(raw, 3)} | {_f(e.get('raster_occupied_extent'), 3)} | {_f(e.get('raster_closed_filled_largest_extent'), 3)} | "
                     f"{_f(e.get('polygon_shorter_than_raw_support_fraction_of_edge'), pct=True)} | {_f(ins, 3)} |")
        a = h["anchor_diagnostics"]
        if a.get("status") == "ok":
            p1 = a["protocol_1_ceiling_3m"]["wall_relative_error"]
            L += ["", "Anchor diagnostic: (1) ceiling = 3.0 m protocol wall errors " + "; ".join(f"{n} {_f(v, pct=True)}" for n, v in p1.items())]
            for an, d in a["protocol_2_single_wall_anchor_oracle"].items():
                L.append(f"- (2) anchor {an} true wall (oracle): other walls " + "; ".join(f"{n} {_f(v, pct=True)}" for n, v in d["other_wall_relative_error"].items()) +
                         f"; implied height error {_f(d['implied_height_error_fraction'], pct=True)}")
        hv = h["height_estimate_vs_truth_under_fixed_similarity"]
        L.append(f"Height / true height under the fixed similarity: estimate_height {_f(hv['estimate_height_ratio'], 3)}, slab medians {_f(hv['slab_median_ratio'], 3)}, point planes {_f(hv['point_plane_ratio'], 3)}.")
        L += ["", f"**Verdict (hypothesis: DA3 vertical overestimate, not rasterisation, causes the wall-length error):** {h['verdict'].get('statement') or h['verdict']}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", type=Path, required=True)
    p.add_argument("--prediction", type=Path, required=True, help="prediction.npz whose source_indices are the ORIGINAL frame indices")
    p.add_argument("--pipeline-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    doc = analyze(a.variant, a.prediction, a.pipeline_run, a.output)
    print(json.dumps({"verdicts": [h.get("verdict", {}).get("status") for h in doc["hypotheses"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
