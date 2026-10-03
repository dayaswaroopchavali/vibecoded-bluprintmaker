"""Truth-free room-height anchor measured on the WALL planes, not on the ceiling plane.

Why: after one global scale the ceiling PLANE can be read systematically nearer than the walls (depth-bias-v2), so a floor-to-ceiling
separation taken from the cap planes is a little too small and normalising it to the assessment height inflates every wall. This module never uses the
position of a cap plane. From the cloud, a structural axis (sign irrelevant, arbitrary orientation) and optional view ids only:

 1. wall detection: cells whose points fill the middle of the height range (walls) as opposed to the two ends only (caps); sequential
    Hough/total-least-squares lines of those points in the plane perpendicular to the axis; the axis is refined so every wall plane
    contains it (needs two wall azimuths; otherwise it is kept and the fact is reported);
 2. wall-plane inlier points: all cloud points within a narrow strip of each accepted wall line. Bins whose count towers over the wall's
    own plateau (floor/ceiling sheets that cross the strip) are removed so a cap cannot define the wall extent;
 3. estimator A (`wall_extent_robust_percentile`): per wall, the robust lowest and highest wall point (0.5 / 99.5 percentile) of the
    inlier points that belong to the wall's contiguous height support (sheets detached from the wall are ignored). The half-plateau
    extent (density falls to half the wall's own plateau) is reported as an alternative;
 4. estimator B (`junction_line_fit`): per wall, the low and high robust percentile of every column along the wall, a robust line is
    fitted to each (floor-wall and wall-top junction lines, so a tilted junction is handled), height = distance between the two lines
    at the wall's mid position;
 5. room level: the best observed wall (maximum over walls; unobserved junction regions can only SHORTEN a wall, never lengthen it) for
    each estimator, the two are combined when they agree; the spread over walls, A-vs-B, +/- axis runs, leave-one-view-out and the
    ramp width of the wall ends are truth-free ambiguity checks; the cap-plane estimate (height_diagnosis.estimate_room_height, or a
    supplied copy) is used ONLY to report agreement: disagreement downgrades the status to ambiguous and is never resolved by choosing.

Every threshold is a fraction of the cloud's own extent / counts; a uniform scale of the input scales the answer and leaves the status
unchanged. No simulator, truth, camera orientation, camera path, plane-extraction json or instance constant is read. The output is an
arbitrary-unit separation in the schema of height_diagnosis.evaluator_compatible; it is not a metric height.
"""
from __future__ import annotations

import numpy as np

# scale-free settings
LINE_TOL_FRACTION = 0.01        # wall line half-width / Hough bin, fraction of horizontal extent
CELL_FRACTION = 0.02            # wall-cell raster, fraction of horizontal extent
MID_BAND = (0.25, 0.75)         # central share of the 1-99 percentile height range
MIN_MID_POINTS = 3
MIN_MID_FRACTION = 0.2          # a wall cell has at least this share of its points in the mid band
MAX_HOUGH_POINTS = 20000
MAX_POINTS = 150000
MAX_WALLS = 8
MIN_WALL_LENGTH_FRACTION = 0.25  # wall length (2-98 pct along the line) / horizontal extent
MIN_WALL_SUPPORT_FRACTION = 0.04
HOUGH_ANGLES = 180
MIN_AZIMUTH_DEG = 20.0          # two wall azimuths needed to refine the axis
AXIS_ITERATIONS = 3
MAX_AXIS_REFINEMENT_DEG = 10.0
BIN_FRACTION = 0.005            # height bin, fraction of cloud extent
SPIKE_FACTOR = 2.5              # bin count above this many plateaus is a cap sheet crossing the strip
EDGE_FRACTION = 0.5             # half-plateau crossing
MIN_WALL_POINTS = 300
COLUMNS = 6
COLUMN_QUANTILE = 1.0           # percent
WALL_QUANTILE = 0.5             # percent: robust lowest / highest wall point
MIN_COLUMN_POINTS = 150
GAP_FRACTION = 0.04             # a dip shorter than this share of the wall's height span is still the same wall
SUPPORT_FRACTION = 0.1          # density share that bounds the wall's contiguous support interval
MIN_POINTS = 200
AB_TOLERANCE = 0.03             # relative A-vs-B agreement
WALL_SPREAD_TOLERANCE = 0.05    # relative spread of per-wall extents
MAX_LOVO_RELATIVE_RANGE = 0.05
MAX_SIGN_DISAGREEMENT = 0.02
CAP_AGREEMENT_TOLERANCE = 0.03  # relative disagreement with the cap-plane estimate that is flagged
MAX_RAMP_FRACTION = 0.10        # 25%-75% density ramp at a wall end, fraction of the wall extent
MAX_LOVO_VIEWS = 60
MIN_SEPARATION_FRACTION = 0.10


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


def _basis(axis):
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    a = _unit(np.cross(axis, seed))
    return a, np.cross(axis, a)


def _extent(points):
    return float(np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))


def _horizontal_extent(points, axis):
    horiz = points - np.outer(points @ axis, axis)
    return float(np.linalg.norm(np.percentile(horiz, 95, axis=0) - np.percentile(horiz, 5, axis=0)))


# ------------------------------------------------------------------------------------------------ wall detection
def _hough_line(xy, tol):
    th = np.radians(np.arange(HOUGH_ANGLES) * 180.0 / HOUGH_ANGLES)
    nrm = np.column_stack([np.cos(th), np.sin(th)])
    rho = xy @ nrm.T
    lo = float(rho.min())
    bins = np.floor((rho - lo) / tol).astype(np.int64)
    nb = int(bins.max()) + 2
    acc = np.zeros((HOUGH_ANGLES, nb))
    for j in range(HOUGH_ANGLES):
        c = np.bincount(bins[:, j], minlength=nb).astype(float)
        acc[j] = c + np.r_[c[1:], 0.]
    j, k = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return nrm[j], lo + (k + 1) * tol


def _tls_line(xy, n, r, tol):
    inl = np.abs(xy @ n - r) <= 1.5 * tol
    for _ in range(3):
        if inl.sum() < 5:
            break
        p = xy[inl]
        m = p.mean(axis=0)
        _vals, vecs = np.linalg.eigh(np.cov((p - m).T))
        n = vecs[:, 0]
        r = float(m @ n)
        inl = np.abs(xy @ n - r) <= 1.5 * tol
    return n, r, inl


def _detect_walls(points, axis, extent):
    """Wall lines in the plane perpendicular to `axis`. Returns (lines, info) with 3D mid-band support for axis refinement."""
    a, b = _basis(axis)
    h = points @ axis
    uv = np.column_stack([points @ a, points @ b])
    horiz = float(np.linalg.norm(np.percentile(uv, 95, axis=0) - np.percentile(uv, 5, axis=0)))
    hl, hh = np.percentile(h, [1, 99])
    rng_h = float(hh - hl)
    info = {"horizontal_extent": horiz, "height_range_1_99": rng_h}
    if horiz <= 1e-12 or rng_h <= 1e-12:
        return [], info
    cell = CELL_FRACTION * horiz
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(np.int64)
    key = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    _u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mid = (h > hl + MID_BAND[0] * rng_h) & (h < hl + MID_BAND[1] * rng_h)
    mid_cnt = np.bincount(inv, weights=mid.astype(float), minlength=len(cnt))
    wall_cell = (mid_cnt >= MIN_MID_POINTS) & (mid_cnt / np.maximum(cnt, 1) >= MIN_MID_FRACTION)
    sel = np.flatnonzero(wall_cell[inv] & mid)
    info["wall_cells"] = int(wall_cell.sum())
    info["mid_points"] = int(len(sel))
    if len(sel) < 50:
        return [], info
    if len(sel) > MAX_HOUGH_POINTS:
        sel = np.sort(np.random.default_rng(0).choice(sel, MAX_HOUGH_POINTS, replace=False))
    xy = uv[sel]
    total = len(xy)
    tol = LINE_TOL_FRACTION * horiz
    work = np.arange(total)
    lines = []
    for _ in range(MAX_WALLS):
        if len(work) < MIN_WALL_SUPPORT_FRACTION * total:
            break
        n, r = _hough_line(xy[work], tol)
        n, r, inl_local = _tls_line(xy[work], n, r, tol)
        inl = work[inl_local]
        if len(inl) < MIN_WALL_SUPPORT_FRACTION * total:
            break
        tang = np.array([-n[1], n[0]])
        s = xy[inl] @ tang
        s0, s1 = np.percentile(s, [2, 98])
        work = np.setdiff1d(work, inl, assume_unique=True)
        if s1 - s0 < MIN_WALL_LENGTH_FRACTION * horiz:
            continue
        lines.append({"n": n, "r": float(r), "s0": float(s0), "s1": float(s1), "length": float(s1 - s0),
                      "support": int(len(inl)), "mid_points_3d": points[sel[inl]]})
    info["tolerance"] = tol
    return lines, info


def _plane_normal(pts3):
    c = pts3 - pts3.mean(axis=0)
    _vals, vecs = np.linalg.eigh(c.T @ c)
    return vecs[:, 0]


def _refine_axis(points, axis0, extent):
    """Iterate wall detection and axis refinement; returns (axis, lines, info)."""
    axis = _unit(axis0)
    refined = False
    info = {}
    lines = []
    for _ in range(AXIS_ITERATIONS):
        lines, info = _detect_walls(points, axis, extent)
        if len(lines) < 2:
            break
        normals = [_plane_normal(l["mid_points_3d"]) for l in lines]
        weights = [l["support"] for l in lines]
        far = any(np.degrees(np.arccos(np.clip(abs(float(normals[i] @ normals[j])), 0., 1.))) >= MIN_AZIMUTH_DEG
                  for i in range(len(normals)) for j in range(i + 1, len(normals)))
        if not far:
            break
        scatter = sum(w * np.outer(n, n) for w, n in zip(weights, normals))
        _vals, vecs = np.linalg.eigh(scatter)
        new = vecs[:, 0]
        new = new if float(new @ axis) > 0 else -new
        step = float(np.degrees(np.arccos(np.clip(float(new @ axis), -1., 1.))))
        axis = _unit(new)
        refined = True
        if step < 0.1:
            break
    lines, info = _detect_walls(points, axis, extent)
    info["axis_refined"] = refined
    info["azimuths_available"] = len(lines) >= 2
    return axis, lines, info


# ------------------------------------------------------------------------------------------------ profiles
def _despike(h, bw):
    """Drop points of bins whose count towers over the wall's own plateau (cap sheets that cross the strip)."""
    lo = float(h.min())
    idx = np.floor((h - lo) / bw).astype(np.int64)
    cnt = np.bincount(idx)
    q = np.percentile(h, [25, 75])
    mid_bins = np.unique(np.floor((h[(h >= q[0]) & (h <= q[1])] - lo) / bw).astype(np.int64))
    plateau = float(np.median(cnt[mid_bins])) if len(mid_bins) else 0.
    if plateau <= 0:
        return np.ones(len(h), bool), 0
    bad = cnt > SPIKE_FACTOR * plateau
    return ~bad[idx], int(bad.sum())


def _walk(sm, thr, start, direction, gap_bins):
    """Outermost bin at/above `thr` reachable from `start` without crossing a run of more than `gap_bins` bins below it, so a sheet
    that is detached from the wall profile (a cap crossing the strip) cannot extend the wall."""
    last, run, i = start, 0, start
    while 0 <= i + direction < len(sm):
        i += direction
        if sm[i] >= thr:
            last, run = i, 0
        else:
            run += 1
            if run > gap_bins:
                break
    return last


def _cross(sm, centers, thr, last, direction):
    j = last + direction
    if j < 0 or j >= len(sm):
        return float(centers[last])
    y0, y1 = sm[j], sm[last]
    t = (thr - y0) / (y1 - y0) if y1 != y0 else 0.5
    return float(centers[j] + t * (centers[last] - centers[j]))


def _edge_profile(h, bw, min_points=MIN_WALL_POINTS):
    """Robust extent, half-plateau edges, 25%-75% ramp widths and contiguous support interval of a wall's height profile.
    None when too few points."""
    if len(h) < min_points:
        return None
    keep, spikes = _despike(h, bw)
    h = h[keep]
    if len(h) < min_points:
        return None
    edges = np.arange(h.min() - bw, h.max() + 2 * bw, bw)
    cnt, e = np.histogram(h, bins=edges)
    sm = np.convolve(cnt.astype(float), [.25, .5, .25], mode="same")
    c = 0.5 * (e[1:] + e[:-1])
    q = np.percentile(h, [25, 75])
    core = np.flatnonzero((c >= q[0]) & (c <= q[1]))
    if len(core) == 0:
        return None
    plateau = float(np.median(sm[core]))
    if plateau <= 0:
        return None
    start = int(core[np.argmax(sm[core])])
    gap = max(2, int(round(GAP_FRACTION * (h.max() - h.min()) / bw)))

    def edge(frac, side):
        d = -1 if side == "lo" else 1
        last = _walk(sm, frac * plateau, start, d, gap)
        return _cross(sm, c, frac * plateau, last, d)
    lo, hi = edge(EDGE_FRACTION, "lo"), edge(EDGE_FRACTION, "hi")
    if hi <= lo:
        return None
    ramp = [abs(edge(.75, side) - edge(.25, side)) for side in ("lo", "hi")]
    s_lo, s_hi = edge(SUPPORT_FRACTION, "lo"), edge(SUPPORT_FRACTION, "hi")
    inside = h[(h >= s_lo) & (h <= s_hi)]
    p_lo, p_hi = np.percentile(inside, [WALL_QUANTILE, 100 - WALL_QUANTILE])
    return {"half_plateau_low": lo, "half_plateau_high": hi, "half_plateau_extent": hi - lo,
            "low": float(p_lo), "high": float(p_hi), "extent": float(p_hi - p_lo), "plateau": plateau,
            "spike_bins_removed": spikes, "ramp_width_low": float(ramp[0]), "ramp_width_high": float(ramp[1]),
            "support_low": s_lo, "support_high": s_hi}


def _robust_line(x, y, rounds=4):
    A = np.column_stack([np.ones(len(x)), x])
    keep = np.ones(len(x), bool)
    coef = np.array([float(np.median(y)), 0.])
    for _ in range(rounds):
        if keep.sum() < 3:
            break
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
        res = y - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
        keep = np.abs(res) <= max(2.5 * mad, 1e-12)
    return coef, keep


def _junction_profile(h, s, s0, s1, bw, support):
    """Floor-wall and wall-top junction lines: robust percentile edges of each column along the wall (points detached from the wall's
    own support interval are ignored), a robust line through each; separation at the wall's mid position."""
    edges = np.linspace(s0, s1, COLUMNS + 1)
    col = np.clip(np.digitize(s, edges) - 1, 0, COLUMNS - 1)
    inside = (h >= support[0]) & (h <= support[1])
    xs, lows, highs = [], [], []
    for k in range(COLUMNS):
        hk = h[(col == k) & inside]
        if len(hk) < MIN_COLUMN_POINTS:
            continue
        xs.append(0.5 * (edges[k] + edges[k + 1]))
        lows.append(float(np.percentile(hk, COLUMN_QUANTILE)))
        highs.append(float(np.percentile(hk, 100 - COLUMN_QUANTILE)))
    if len(xs) < 3:
        return None
    xs, lows, highs = np.asarray(xs), np.asarray(lows), np.asarray(highs)
    x0 = float(np.median(xs))
    cl, kl = _robust_line(xs - x0, lows)
    ch, kh = _robust_line(xs - x0, highs)
    return {"extent": float(ch[0] - cl[0]), "low_at_mid": float(cl[0]), "high_at_mid": float(ch[0]),
            "low_slope": float(cl[1]), "high_slope": float(ch[1]), "columns_used": int(len(xs)),
            "low_inlier_columns": int(kl.sum()), "high_inlier_columns": int(kh.sum())}


def _wall_strips(points, axis, lines, tol):
    a, b = _basis(axis)
    uv = np.column_stack([points @ a, points @ b])
    h = points @ axis
    out = []
    for w in lines:
        n = w["n"]
        tang = np.array([-n[1], n[0]])
        d = uv @ n - w["r"]
        s = uv @ tang
        m = (np.abs(d) <= 1.5 * tol) & (s >= w["s0"]) & (s <= w["s1"])
        out.append({"index": np.flatnonzero(m), "h": h[m], "s": s[m]})
    return out


def _room_numbers(strips, lines, bw, mask_fn=None):
    """Per-wall A and B, then the room level (best observed wall) for each. `mask_fn(index) -> bool mask` removes points (LOVO)."""
    per = []
    for w, st in zip(lines, strips):
        keep = np.ones(len(st["h"]), bool) if mask_fn is None else mask_fn(st["index"])
        h, s = st["h"][keep], st["s"][keep]
        ea = _edge_profile(h, bw)
        jb = None if ea is None else _junction_profile(h, s, w["s0"], w["s1"], bw, (ea["support_low"], ea["support_high"]))
        per.append({"A": ea, "B": jb, "points": int(len(h))})
    return per


def _aggregate(per):
    a_vals = [p["A"]["extent"] for p in per if p["A"] is not None]
    hp_vals = [p["A"]["half_plateau_extent"] for p in per if p["A"] is not None]
    b_vals = [p["B"]["extent"] for p in per if p["B"] is not None]
    out = {"A_walls": a_vals, "B_walls": b_vals, "half_plateau_walls": hp_vals,
           "A_room": max(a_vals) if a_vals else None, "B_room": max(b_vals) if b_vals else None}
    if out["A_room"] is not None and out["B_room"] is not None:
        out["combined"] = 0.5 * (out["A_room"] + out["B_room"])
    else:
        out["combined"] = None
    return out


def _analyse_one_sign(pts, ids, axis0, extent):
    rec = {"status": "incomplete", "reasons": [], "separation_unscaled": None}
    axis, lines, info = _refine_axis(pts, axis0, extent)
    rec["axis_world"] = axis.tolist()
    rec["axis_refinement_deg"] = float(np.degrees(np.arccos(np.clip(float(axis @ axis0), -1., 1.))))
    rec["wall_detection"] = {k: v for k, v in info.items()}
    if not lines:
        rec["reasons"].append("no wall-like line support (mid-height cells forming long straight lines) found")
        return rec, None
    tol = info["tolerance"]
    bw = BIN_FRACTION * extent
    strips = _wall_strips(pts, axis, lines, tol)
    per = _room_numbers(strips, lines, bw)
    agg = _aggregate(per)
    rec["walls"] = [{"length": l["length"], "support": l["support"], "normal_2d": l["n"].tolist(), "offset": l["r"],
                     "strip_points": p["points"],
                     "A_extent": None if p["A"] is None else p["A"]["extent"],
                     "half_plateau_extent": None if p["A"] is None else p["A"]["half_plateau_extent"],
                     "A_low": None if p["A"] is None else p["A"]["low"], "A_high": None if p["A"] is None else p["A"]["high"],
                     "A_ramp_width_low": None if p["A"] is None else p["A"]["ramp_width_low"],
                     "A_ramp_width_high": None if p["A"] is None else p["A"]["ramp_width_high"],
                     "A_spike_bins_removed": None if p["A"] is None else p["A"]["spike_bins_removed"],
                     "B_extent": None if p["B"] is None else p["B"]["extent"],
                     "B_detail": p["B"]} for l, p in zip(lines, per)]
    rec["aggregate"] = agg
    if agg["combined"] is None:
        rec["reasons"].append("no wall with enough points for both the percentile and the junction-line estimators")
        return rec, None
    rec["separation_unscaled"] = float(agg["combined"])
    state = {"axis": axis, "lines": lines, "strips": strips, "bw": bw, "per": per, "agg": agg}
    return rec, state


def _leave_one_view_out(ids, state):
    if ids is None:
        return {"available": False}
    uniq = np.unique(ids)
    if len(uniq) > MAX_LOVO_VIEWS:
        uniq = uniq[np.linspace(0, len(uniq) - 1, MAX_LOVO_VIEWS).astype(int)]
    seps = []
    for v in uniq:
        per = _room_numbers(state["strips"], state["lines"], state["bw"], mask_fn=lambda idx, v=v: ids[idx] != v)
        c = _aggregate(per)["combined"]
        if c is not None:
            seps.append(c)
    if len(seps) < 3:
        return {"available": False, "reason": "fewer than three usable leave-one-view-out refits"}
    arr = np.asarray(seps)
    full = state["agg"]["combined"]
    return {"available": True, "views": int(len(arr)), "min": float(arr.min()), "max": float(arr.max()),
            "relative_range": float((arr.max() - arr.min()) / abs(full)),
            "max_relative_deviation_from_full": float(np.max(np.abs(arr - full)) / abs(full))}


def _cap_agreement(sep, reference):
    if not isinstance(reference, dict):
        return {"status": "no_reference"}
    ok = reference.get("status") in ("complete", "estimated_unscaled_separation")
    ref = reference.get("separation_unscaled")
    if not ok or not ref:
        return {"status": "reference_not_estimated", "reference_status": reference.get("status")}
    rel = (sep - float(ref)) / abs(float(ref))
    return {"status": "agrees" if abs(rel) <= CAP_AGREEMENT_TOLERANCE else "disagrees", "cap_plane_separation": float(ref),
            "wall_based_separation": float(sep), "relative_difference_wall_minus_cap": float(rel),
            "tolerance": CAP_AGREEMENT_TOLERANCE,
            "note": "wall-based above cap-based = the cap planes read nearer than the walls; below = wall ends under-observed"}


def cap_reference(points, axis, view_ids=None):
    """The cap-plane estimate (height_diagnosis.estimate_room_height) used ONLY to report agreement."""
    import height_diagnosis
    return height_diagnosis.estimate_room_height(points, axis, view_ids)


def _prepare(points, view_ids):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None, None
    ok = np.isfinite(pts).all(axis=1)
    ids = None
    if view_ids is not None:
        ids = np.asarray(view_ids)
        if ids.shape != (len(pts),):
            raise ValueError("view_ids must have one entry per point")
        ids = ids[ok]
    pts = pts[ok]
    if len(pts) > MAX_POINTS:
        sel = np.sort(np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False))
        pts = pts[sel]
        ids = None if ids is None else ids[sel]
    return pts, ids


def estimate_wall_height(points, axis, view_ids=None, *, reference=None, compute_reference=True):
    """Wall-based room-height estimate along the unsigned structural `axis` (sign irrelevant, both signs are run).

    `reference` may be a height_diagnosis.estimate_room_height result (or its evaluator_compatible form); when absent and
    `compute_reference` it is computed. It only produces the agreement flag; it never changes the number."""
    base = {"model": "wall_plane_vertical_extent", "units": "arbitrary", "metric_scale_source": None, "scale_applied": False,
            "status": "incomplete", "separation_unscaled": None, "reasons": [], "axis_sign_resolved": False, "alternatives": [],
            "ambiguity_flags": [],
            "inputs_used": ["point cloud", "structural axis (sign irrelevant)", "view ids (optional)",
                            "cap-plane estimate (agreement flag only)"],
            "inputs_not_used": ["camera up", "camera path", "camera centres", "plane-extraction json", "ground truth",
                                "simulator", "instance constants", "cap plane position (never enters the number)"],
            "limitations": [
                "Not a metric height; arbitrary units, no scale calibration.",
                "A wall can only be observed over the heights the views and the confidence filter kept: the extent of every wall is a "
                "LOWER bound on the room height; the maximum over walls is used and the spread over walls and the ramp width of the "
                "ends are reported as truth-free evidence of under-observation.",
                "Whether the cap planes or the wall extents are the better anchor is not decided here; disagreement is reported as "
                "ambiguity and the evaluator refuses to score an ambiguous result.",
                "Thresholds are heuristics scaled by the cloud extent; they are not validated across rooms."]}
    axis = _unit(axis)
    if np.linalg.norm(axis) < .9:
        return {**base, "reasons": ["structural axis missing or degenerate"]}
    pts, ids = _prepare(points, view_ids)
    if pts is None or len(pts) < MIN_POINTS:
        return {**base, "reasons": ["insufficient finite points"]}
    extent = _extent(pts)
    if extent <= 1e-12:
        return {**base, "reasons": ["degenerate point cloud extent"]}
    base["cloud_extent"] = extent
    runs, states = {}, {}
    for name, sgn in (("axis_plus", 1.), ("axis_minus", -1.)):
        runs[name], states[name] = _analyse_one_sign(pts, ids, sgn * axis, extent)
    base["sign_runs"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("walls",)} for k, v in runs.items()}
    base["refined_axis_world"] = runs["axis_plus"].get("axis_world")
    main, state = runs["axis_plus"], states["axis_plus"]
    if state is None or states["axis_minus"] is None:
        reasons = [f"{k}: {r}" for k, v in runs.items() for r in v["reasons"]]
        return {**base, "reasons": reasons or ["no wall extent could be measured"]}
    sp, sm = main["separation_unscaled"], runs["axis_minus"]["separation_unscaled"]
    sign_gap = abs(sp - sm) / max(abs(sp), abs(sm), 1e-12)
    sep = float(0.5 * (sp + sm))
    agg = state["agg"]
    result = {**base, "separation_unscaled": sep, "sign_disagreement_relative": float(sign_gap),
              "axis_refinement_deg": main["axis_refinement_deg"], "axis_refined_from_walls": main["wall_detection"].get("axis_refined"),
              "walls": main["walls"], "estimators": {
                  "wall_extent_robust_percentile": {"room": agg["A_room"], "per_wall": agg["A_walls"]},
                  "wall_extent_half_plateau": {"room": max(agg["half_plateau_walls"]), "per_wall": agg["half_plateau_walls"]},
                  "junction_line_fit": {"room": agg["B_room"], "per_wall": agg["B_walls"]}}}
    problems = []
    if main["axis_refinement_deg"] > MAX_AXIS_REFINEMENT_DEG:
        problems.append(f"axis refinement of {main['axis_refinement_deg']:.1f} deg exceeds {MAX_AXIS_REFINEMENT_DEG} deg")
    if sep < MIN_SEPARATION_FRACTION * extent:
        problems.append(f"separation {sep:.4g} below {MIN_SEPARATION_FRACTION:.0%} of cloud extent {extent:.4g}")
    ab = abs(agg["A_room"] - agg["B_room"]) / max(abs(agg["combined"]), 1e-12)
    result["A_vs_B_relative_difference"] = float(ab)
    if ab > AB_TOLERANCE:
        problems.append(f"percentile and junction-line estimators differ by {ab:.2%} (> {AB_TOLERANCE:.0%})")
    walls_a = agg["A_walls"]
    if len(walls_a) >= 2:
        spread = (max(walls_a) - min(walls_a)) / max(max(walls_a), 1e-12)
        result["wall_extent_relative_spread"] = float(spread)
        if spread > WALL_SPREAD_TOLERANCE:
            problems.append(f"wall extents differ by {spread:.1%} between walls (> {WALL_SPREAD_TOLERANCE:.0%}): "
                            "some walls are under-observed at one end")
    elif len(walls_a) == 1:
        problems.append("only one wall measured: no check of one wall against another")
    ramps = [max(w["A_ramp_width_low"] or 0., w["A_ramp_width_high"] or 0.) / w["A_extent"]
             for w in main["walls"] if w["A_extent"]]
    if ramps:
        result["wall_end_ramp_fraction_max"] = float(max(ramps))
        result["wall_end_ramp_fraction_best_wall"] = float(min(ramps))
        if min(ramps) > MAX_RAMP_FRACTION:
            problems.append(f"every wall end ramps over >{MAX_RAMP_FRACTION:.0%} of the wall extent: junction regions look "
                            "under-observed, so the extent is a lower bound")
    if sign_gap > MAX_SIGN_DISAGREEMENT:
        problems.append(f"axis +/- runs disagree by {sign_gap:.3%}")
    lovo = _leave_one_view_out(ids, state)
    result["leave_one_view_out"] = lovo
    if lovo.get("available") and lovo["relative_range"] > MAX_LOVO_RELATIVE_RANGE:
        problems.append(f"leave-one-view-out range {lovo['relative_range']:.2%} exceeds {MAX_LOVO_RELATIVE_RANGE:.0%}")
    if reference is None and compute_reference:
        try:
            reference = cap_reference(pts, axis, ids)
        except Exception as exc:                                    # recorded, never silently dropped
            reference = {"status": f"cap_estimate_failed: {type(exc).__name__}: {exc}"}
    cap = _cap_agreement(sep, reference)
    result["cap_agreement"] = cap
    if cap["status"] == "disagrees":
        flag = (f"wall-based separation {sep:.4g} vs cap-plane {cap['cap_plane_separation']:.4g}: "
                f"{cap['relative_difference_wall_minus_cap']:+.2%} (tolerance {CAP_AGREEMENT_TOLERANCE:.0%}); not resolved here")
        result["ambiguity_flags"].append(flag)
        problems.append(flag)
    alt = [{"name": "wall_extent_robust_percentile_max_over_walls", "separation_unscaled": agg["A_room"]},
           {"name": "wall_extent_half_plateau_max_over_walls", "separation_unscaled": max(agg["half_plateau_walls"])},
           {"name": "junction_line_fit_max_over_walls", "separation_unscaled": agg["B_room"]},
           {"name": "wall_extent_robust_percentile_median_over_walls", "separation_unscaled": float(np.median(agg["A_walls"]))}]
    if cap.get("cap_plane_separation") is not None:
        alt.append({"name": "cap_plane_estimate (height_diagnosis)", "separation_unscaled": cap["cap_plane_separation"]})
    result["alternatives"] = alt
    result["structural_problems"] = problems
    if problems:
        result["status"] = "ambiguous"
        result["reasons"] = problems
    elif not lovo.get("available"):
        result["status"] = "ambiguous"
        result["reasons"] = ["no view ids: uncertainty by leaving views out could not be computed"]
    else:
        result["status"] = "complete"
        result["reasons"] = ["wall extents from two estimators agree, stable under view removal, consistent with the cap planes or "
                             "no cap estimate available"]
    return result


def evaluator_compatible(result):
    """Same schema as height_diagnosis.evaluator_compatible: only a complete result maps to 'estimated_unscaled_separation'."""
    ok = result.get("status") == "complete"
    return {"status": "estimated_unscaled_separation" if ok else result.get("status"),
            "separation_unscaled": result.get("separation_unscaled") if ok else None,
            "source": "wall_height_anchor.estimate_wall_height", "units": "arbitrary", "scale_applied": False,
            "reasons": result.get("reasons", [])}


def diagnostic_unscaled_separation(result):
    """EVALUATION-ONLY adapter: the separation is passed on even when the status is ambiguous, with the status carried in
    `diagnostic_status`. Used only to show what the evaluator would do; the strict adapter above is the pipeline interface."""
    sep = result.get("separation_unscaled")
    return {"status": "estimated_unscaled_separation" if sep else result.get("status"), "separation_unscaled": sep,
            "source": "wall_height_anchor.estimate_wall_height (diagnostic override of status)", "units": "arbitrary",
            "scale_applied": False, "diagnostic_status": result.get("status"), "reasons": result.get("reasons", [])}
