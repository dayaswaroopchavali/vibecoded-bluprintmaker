"""Truth-free diagnosis of estimate_height.py plus a second, independent room-height estimator.

Part 1 (`diagnose_existing`): explains, with numbers, which floor/ceiling evidence estimate_height.py
rejected for a declared structural axis (plane alignment, patch area, camera coverage, candidate-support share,
bracketing, parallelism). It re-runs estimate_height.py unchanged and re-implements none of its thresholds.

Part 2 (`estimate_room_height`): a separate estimator that never reads the existing planes json, camera
centres, camera up or camera path. From the cloud alone it
  * searches a small cone around the supplied structural axis for the direction in which two narrow height peaks
    are sharpest (tilt tolerance), over the footprint interior only,
  * keeps height peaks that are prominent against their own neighbourhood AND cover a broad share of the
    footprint cells (furniture tops are small, floor and ceiling are broad),
  * takes the lowest and the highest such peak along the axis as floor-like and ceiling-like (the axis sign is
    never resolved; both signs are run and must agree), fits a robust plane to each, and reports the distance
    between the two (near-)parallel planes at the footprint centre,
  * quantifies uncertainty by leaving each view out, and optionally agreement with an existing estimate.
Every threshold is a fraction of the cloud's own extent or of its own counts, so a uniform scale of the input
scales the answer and leaves the status unchanged. No simulator, truth, camera orientation or instance constant
is used. The output is an arbitrary-unit separation; it is not a metric height.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Scale-free settings (fractions of the cloud's own extent / counts / areas).
BIN_FRACTION = 0.02            # height bin, fraction of cloud extent
MIN_PEAK_SEPARATION = 0.08     # two peaks closer than this fraction of extent are one structure
TILT_SEARCH_DEG = 8.0          # cone searched around the supplied axis
TILT_STEP_DEG = 2.0
INTERIOR_QUANTILES = (10., 90.)  # central share of each horizontal coordinate that counts as footprint interior
PROMINENCE = 3.0               # peak density vs the median density of its neighbourhood
NEIGHBOURHOOD = 0.20           # neighbourhood half-width, fraction of extent
MIN_COVERAGE = 0.15            # cap covers at least this share of footprint cells
COVERAGE_RATIO = 0.5           # ... and at least this share of the best-covering peak
SLAB_BINS = 1.5                # slab half-thickness in bins for the plane fit
PLANAR_MAD_FRACTION = 0.3      # a peak is planar when its robust residual spread is below this share of the slab
TRIM_FLOOR_FRACTION = 0.1      # trimming never tighter than this share of a bin
MAX_PARALLEL_DEG = 5.0         # floor-like / ceiling-like normals must agree
MIN_SEPARATION_FRACTION = 0.10  # room height at least this fraction of extent
MAX_LOVO_RELATIVE_RANGE = 0.05  # leave-one-view-out spread (relative)
MAX_SIGN_DISAGREEMENT = 0.01   # axis +/- runs must agree (relative)
AGREEMENT_TOLERANCE = 0.05     # relative agreement with an existing estimate
MULTI_LEVEL_RATIO = 0.8        # intermediate broad peak this close in coverage to the weaker cap => ambiguous
MIN_POINTS = 200
CELL_FRACTION = 0.03           # footprint raster cell, fraction of horizontal extent
MIN_CELL_POINTS = 2


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(3)


def _basis(axis):
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    a = _unit(np.cross(axis, seed))
    return a, np.cross(axis, a)


def _tilted(axis, a, b, ta, tb):
    return _unit(axis + np.tan(ta) * a + np.tan(tb) * b)


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
    return pts[ok], ids


def _interior_mask(points, axis):
    """Footprint interior in the plane perpendicular to `axis`: central share of the principal horizontal axes."""
    horiz = points - np.outer(points @ axis, axis)
    centred = horiz - np.median(horiz, axis=0)
    a, b = _basis(axis)
    uv = np.column_stack([centred @ a, centred @ b])
    if len(uv) > 2:
        _, vecs = np.linalg.eigh(np.cov(uv.T))
        uv = uv @ vecs
    lo, hi = np.percentile(uv, INTERIOR_QUANTILES, axis=0)
    return np.all((uv >= lo) & (uv <= hi), axis=1), uv


def _extent(points):
    return float(np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))


def _horizontal_extent(points, axis):
    horiz = points - np.outer(points @ axis, axis)
    return float(np.linalg.norm(np.percentile(horiz, 95, axis=0) - np.percentile(horiz, 5, axis=0)))


def _histogram(h, bin_width):
    lo, hi = np.percentile(h, [0.01, 99.99])
    lo -= 2 * bin_width
    n = int(min(6000, max(8, np.ceil((hi - lo) / bin_width) + 3)))
    counts, edges = np.histogram(h, bins=n, range=(lo, lo + n * bin_width))
    return counts.astype(float), edges


def _sharpness(h, bin_width, extent):
    """Sum of the two tallest bins that are far enough apart; larger when height peaks are sharper."""
    counts, edges = _histogram(h, bin_width)
    gap = max(1, int(np.ceil(0.5 * MIN_PEAK_SEPARATION * extent / bin_width)))
    first = int(np.argmax(counts))
    masked = counts.copy()
    masked[max(0, first - gap):first + gap + 1] = 0
    return float(counts[first] + masked.max())


def _search_axis(points, axis, extent):
    """Direction within a small cone of the supplied axis that makes the two tallest height peaks sharpest."""
    a, b = _basis(axis)
    mask, _ = _interior_mask(points, axis)
    sample = points[mask]
    if len(sample) > 60000:
        sample = sample[np.random.default_rng(0).choice(len(sample), 60000, replace=False)]
    bin_width = BIN_FRACTION * extent
    grid = np.radians(np.arange(-TILT_SEARCH_DEG, TILT_SEARCH_DEG + 1e-9, TILT_STEP_DEG))
    best = (-1., axis)
    for ta in grid:
        for tb in grid:
            cand = _tilted(axis, a, b, ta, tb)
            s = _sharpness(sample @ cand, bin_width, extent)
            # prefer the supplied axis on ties
            if s > best[0] * (1 + 1e-9):
                best = (s, cand)
    return best[1]


def _find_peaks(points, axis, extent):
    """Prominent narrow height peaks over the footprint interior. Returns list of dicts sorted by height."""
    mask, _ = _interior_mask(points, axis)
    h = points[mask] @ axis
    if len(h) < MIN_POINTS:
        return [], mask
    peaks = []
    for mult in (1, 2):
        bw = BIN_FRACTION * extent * mult
        counts, edges = _histogram(h, bw)
        half = max(2, int(round(NEIGHBOURHOOD * extent / bw)))
        for i in range(len(counts)):
            if counts[i] <= 0 or counts[i] < counts[max(0, i - 1):i + 2].max():
                continue
            lo, hi = max(0, i - half), min(len(counts), i + half + 1)
            neigh = np.concatenate([counts[lo:i], counts[i + 1:hi]])
            background = float(np.median(neigh)) if len(neigh) else 0.
            if counts[i] >= PROMINENCE * max(background, 1.0) and counts[i] >= 0.01 * len(h):
                peaks.append({"height": float(0.5 * (edges[i] + edges[i + 1])), "count": float(counts[i]),
                              "bin_width": float(bw), "slab": float(SLAB_BINS * BIN_FRACTION * extent),
                              "prominence": float(counts[i] / max(background, 1.0))})
    # greedy non-maximum suppression by count
    kept = []
    for p in sorted(peaks, key=lambda q: -q["count"]):
        if all(abs(p["height"] - q["height"]) >= MIN_PEAK_SEPARATION * extent for q in kept):
            kept.append(p)
    return sorted(kept, key=lambda q: q["height"]), mask


def _fit_plane(points, axis, a, b, center_h, slab, trim_floor=0., rounds=4):
    """Robust plane h = alpha + beta*u + gamma*v along `axis`, starting from the flat plane h = center_h.

    The slab follows the current plane, so a tilted or mis-centred cap is still captured. Returns None if degenerate."""
    u, v, h = points @ a, points @ b, points @ axis
    coef = np.array([center_h, 0., 0.])
    A_all = np.column_stack([np.ones(len(h)), u, v])
    for _ in range(rounds):
        sel = np.abs(h - A_all @ coef) <= slab
        if sel.sum() < 30:
            return None
        A, y = A_all[sel], h[sel]
        keep = np.ones(len(y), bool)
        for _ in range(4):
            if keep.sum() < 30:
                return None
            coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            res = y - A @ coef
            mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
            keep = np.abs(res) <= max(2.5 * mad, trim_floor)
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "gamma": float(coef[2]),
            "inliers": int(keep.sum()), "slab_points": int(sel.sum()), "mad": float(mad), "slab": float(slab),
            "selected": sel, "inlier_mask": keep}


def _plane_normal(fit, axis, a, b):
    return _unit(axis - fit["beta"] * a - fit["gamma"] * b)


def _separation(fit_lo, fit_hi, axis, a, b, uv0):
    n_lo, n_hi = _plane_normal(fit_lo, axis, a, b), _plane_normal(fit_hi, axis, a, b)
    mean_n = _unit(n_lo + n_hi)
    p_lo = uv0[0] * a + uv0[1] * b + (fit_lo["alpha"] + fit_lo["beta"] * uv0[0] + fit_lo["gamma"] * uv0[1]) * axis
    p_hi = uv0[0] * a + uv0[1] * b + (fit_hi["alpha"] + fit_hi["beta"] * uv0[0] + fit_hi["gamma"] * uv0[1]) * axis
    angle = float(np.degrees(np.arccos(np.clip(n_lo @ n_hi, -1., 1.))))
    return float(mean_n @ (p_hi - p_lo)), angle, mean_n


def _analyse_one_sign(points, view_ids, axis0, extent):
    """Full estimate along ONE signed axis. Returns a record with status and every intermediate number."""
    rec = {"status": "incomplete", "reasons": [], "separation_unscaled": None}
    axis = _search_axis(points, axis0, extent)
    rec["axis_refinement_deg"] = float(np.degrees(np.arccos(np.clip(axis @ axis0, -1., 1.))))
    peaks, mask = _find_peaks(points, axis, extent)
    rec["axis_world"] = axis.tolist()
    if len(peaks) < 2:
        rec["reasons"].append(f"fewer than two prominent height peaks over the footprint interior ({len(peaks)})")
        rec["peaks"] = peaks
        return rec, None
    a, b = _basis(axis)
    horiz_extent = _horizontal_extent(points, axis)
    cell = max(CELL_FRACTION * horiz_extent, 1e-12)
    uv_all = np.column_stack([points @ a, points @ b])
    uv_origin = np.median(uv_all[mask], axis=0)
    uv_c = uv_all - uv_origin
    cells = np.floor(uv_c[mask] / cell).astype(np.int64)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    footprint = {tuple(c) for c, n in zip(uniq, counts) if n >= MIN_CELL_POINTS}
    shifted = points[mask] - uv_origin[0] * a - uv_origin[1] * b
    trim_floor = TRIM_FLOOR_FRACTION * BIN_FRACTION * extent
    fits, planar, nonplanar = [], [], []
    for p in peaks:
        f = _fit_plane(shifted, axis, a, b, p["height"], p["slab"], trim_floor)
        if f is None or f["mad"] > PLANAR_MAD_FRACTION * p["slab"]:
            p["cap_coverage"] = 0.0
            nonplanar.append(p)
            continue
        covered_cells = np.floor(uv_c[mask][f["selected"]] / cell).astype(np.int64)
        uniq2, counts2 = np.unique(covered_cells, axis=0, return_counts=True)
        covered = {tuple(c) for c, n in zip(uniq2, counts2) if n >= MIN_CELL_POINTS}
        p["cap_coverage"] = float(len(covered & footprint) / max(1, len(footprint)))
        p["fit_alpha"] = f["alpha"]
        planar.append(p)
        fits.append(f)
    rec["peaks"] = peaks
    rec["footprint_cells"] = len(footprint)
    rec["nonplanar_peaks_rejected"] = [_public_peak(p) for p in nonplanar]
    if len(planar) < 2:
        rec["reasons"].append(f"fewer than two planar height peaks ({len(planar)} of {len(peaks)})")
        return rec, None
    best_cov = max(p["cap_coverage"] for p in planar)
    broad = [p for p in planar if p["cap_coverage"] >= max(MIN_COVERAGE, COVERAGE_RATIO * best_cov)]
    if len(broad) < 2:
        rec["reasons"].append("fewer than two broad (cap-like) height peaks: "
                              f"coverages {[round(p['cap_coverage'], 3) for p in planar]}")
        return rec, None
    floor_pk, ceil_pk = broad[0], broad[-1]
    fit_f, fit_c = fits[planar.index(floor_pk)], fits[planar.index(ceil_pk)]
    uv0 = np.zeros(2)
    sep, angle, mean_n = _separation(fit_f, fit_c, axis, a, b, uv0)
    rec.update(floor_like_height=float(floor_pk["height"]), ceiling_like_height=float(ceil_pk["height"]),
               floor_like_peak=_public_peak(floor_pk), ceiling_like_peak=_public_peak(ceil_pk),
               separation_unscaled=float(sep), parallel_angle_degrees=angle,
               floor_fit={k: v for k, v in fit_f.items() if k not in ("selected", "inlier_mask")},
               ceiling_fit={k: v for k, v in fit_c.items() if k not in ("selected", "inlier_mask")},
               mean_plane_normal_world=mean_n.tolist(), basis_origin_uv=uv_origin.tolist())
    rec["intermediate_broad_peaks"] = [_public_peak(p) for p in broad[1:-1]]
    rec["narrow_or_small_peaks_rejected"] = [_public_peak(p) for p in planar if p not in broad]
    state = {"axis": axis, "a": a, "b": b, "uv_origin": uv_origin, "fit_f": fit_f, "fit_c": fit_c,
             "mask": mask, "trim_floor": trim_floor, "floor_pk": floor_pk, "ceil_pk": ceil_pk, "broad": broad}
    bad = []
    if sep < MIN_SEPARATION_FRACTION * extent:
        bad.append(f"separation {sep:.4g} below {MIN_SEPARATION_FRACTION:.0%} of cloud extent {extent:.4g}")
    if angle > MAX_PARALLEL_DEG:
        bad.append(f"floor-like and ceiling-like planes are {angle:.2f} deg apart (> {MAX_PARALLEL_DEG} deg)")
    weaker = min(floor_pk["cap_coverage"], ceil_pk["cap_coverage"])
    multi = [p for p in broad[1:-1] if p["cap_coverage"] >= MULTI_LEVEL_RATIO * weaker]
    if multi:
        bad.append("an intermediate broad horizontal level competes with the caps: heights "
                   f"{[round(p['height'], 4) for p in multi]}")
    rec["structural_problems"] = bad
    return rec, state


def _public_peak(p):
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()}


def _leave_one_view_out(points, view_ids, state):
    """Refit both planes without each view (same peak structure). Returns relative separation spread."""
    axis, a, b, uv_origin = state["axis"], state["a"], state["b"], state["uv_origin"]
    mask = state["mask"]
    ids_i = view_ids[mask]
    pts_i = points[mask] - uv_origin[0] * a - uv_origin[1] * b
    seps = {}
    for vid in np.unique(ids_i):
        keep = ids_i != vid
        ff = _fit_plane(pts_i[keep], axis, a, b, state["floor_pk"]["height"], state["floor_pk"]["slab"], state["trim_floor"])
        fc = _fit_plane(pts_i[keep], axis, a, b, state["ceil_pk"]["height"], state["ceil_pk"]["slab"], state["trim_floor"])
        if ff is None or fc is None:
            continue
        seps[vid] = _separation(ff, fc, axis, a, b, np.zeros(2))[0]
    return seps


def estimate_room_height(points, axis, view_ids=None, *, reference=None):
    """Second, independent truth-free room-height estimate along the unsigned structural `axis`.

    `view_ids` (optional, one id per point) enables leave-one-view-out uncertainty; without it the status can
    never be 'complete'. `reference` is an optional estimate_height.py result used only to report agreement.
    """
    base = {"model": "height_histogram_parallel_plane_pair", "units": "arbitrary", "metric_scale_source": None,
            "scale_applied": False, "status": "incomplete", "separation_unscaled": None, "reasons": [],
            "axis_sign_resolved": False, "alternatives": [],
            "inputs_used": ["point cloud", "structural axis (sign irrelevant)", "view ids (optional)"],
            "inputs_not_used": ["camera up", "camera path", "camera centres", "plane-extraction json",
                                "ground truth", "simulator", "instance constants"],
            "limitations": ["Not a metric height; arbitrary units, no scale calibration.",
                            "Leave-one-view-out refits planes for the selected peaks; it does not re-run peak selection.",
                            "A broad furniture-free horizontal level other than floor/ceiling would be reported as "
                            "ambiguous, but a ceiling that was never observed cannot be detected as missing.",
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
    base["sign_runs"] = {k: {kk: vv for kk, vv in v.items() if kk != "axis_world"} for k, v in runs.items()}
    base["refined_axis_world"] = runs["axis_plus"].get("axis_world")
    main, state = runs["axis_plus"], states["axis_plus"]
    if state is None or states["axis_minus"] is None:
        reasons = []
        for k, v in runs.items():
            reasons += [f"{k}: {r}" for r in v["reasons"]]
        return {**base, "reasons": reasons or ["no floor-like / ceiling-like pair found"],
                "peaks": main.get("peaks", [])}
    sp, sm = runs["axis_plus"]["separation_unscaled"], runs["axis_minus"]["separation_unscaled"]
    sign_gap = abs(sp - sm) / max(abs(sp), abs(sm), 1e-12)
    result = {**base, "separation_unscaled": float(0.5 * (sp + sm)), "sign_disagreement_relative": float(sign_gap),
              "floor_like_height_axis_plus": main["floor_like_height"],
              "ceiling_like_height_axis_plus": main["ceiling_like_height"],
              "parallel_angle_degrees": main["parallel_angle_degrees"],
              "axis_refinement_deg": main["axis_refinement_deg"], "peaks": main["peaks"],
              "intermediate_broad_peaks": main["intermediate_broad_peaks"],
              "floor_like_peak": main["floor_like_peak"], "ceiling_like_peak": main["ceiling_like_peak"],
              "floor_fit": main["floor_fit"], "ceiling_fit": main["ceiling_fit"]}
    problems = list(main["structural_problems"]) + list(runs["axis_minus"]["structural_problems"])
    if sign_gap > MAX_SIGN_DISAGREEMENT:
        problems.append(f"axis +/- runs disagree by {sign_gap:.3%}")
    lovo = {"available": False}
    if ids is not None:
        seps = _leave_one_view_out(pts, ids, state)
        if len(seps) >= 3:
            arr = np.array(list(seps.values()))
            full = main["separation_unscaled"]
            lovo = {"available": True, "views": len(arr), "min": float(arr.min()), "max": float(arr.max()),
                    "relative_range": float((arr.max() - arr.min()) / abs(full)),
                    "max_relative_deviation_from_full": float(np.max(np.abs(arr - full)) / abs(full))}
            if lovo["relative_range"] > MAX_LOVO_RELATIVE_RANGE:
                problems.append(f"leave-one-view-out range {lovo['relative_range']:.2%} exceeds {MAX_LOVO_RELATIVE_RANGE:.0%}")
    result["leave_one_view_out"] = lovo
    # alternatives: every pair of broad peaks, lowest/highest first
    broad = state["broad"]
    result["alternatives"] = [{"floor_like_height": float(lo["height"]), "ceiling_like_height": float(hi["height"]),
                               "separation_along_axis_unscaled": float(hi["height"] - lo["height"]),
                               "cap_coverages": [lo["cap_coverage"], hi["cap_coverage"]]}
                              for i, lo in enumerate(broad) for hi in broad[i + 1:]]
    result["agreement_with_existing"] = _agreement(result["separation_unscaled"], reference)
    result["structural_problems"] = problems
    if problems:
        result["status"] = "ambiguous"
        result["reasons"] = problems
    elif not lovo["available"]:
        result["status"] = "ambiguous"
        result["reasons"] = ["no view ids: uncertainty by leaving views out could not be computed"]
    else:
        result["status"] = "complete"
        result["reasons"] = ["lowest and highest broad prominent height peaks, near-parallel, stable under view removal"]
    return result


def _agreement(sep, reference):
    if not isinstance(reference, dict):
        return {"status": "no_reference"}
    if reference.get("status") != "estimated_unscaled_separation" or not reference.get("separation_unscaled"):
        return {"status": "reference_not_estimated", "reference_status": reference.get("status")}
    ref = float(reference["separation_unscaled"])
    rel = abs(sep - ref) / max(abs(ref), 1e-12)
    return {"status": "agrees" if rel <= AGREEMENT_TOLERANCE else "disagrees", "reference_separation": ref,
            "relative_difference": float(rel), "tolerance": AGREEMENT_TOLERANCE}


def evaluator_compatible(result):
    """Adapter for evaluate_floorplan.py --height: it accepts only status 'estimated_unscaled_separation'.

    Complete results map to that status; anything else maps to its own status so the evaluator refuses to score."""
    ok = result.get("status") == "complete"
    return {"status": "estimated_unscaled_separation" if ok else result.get("status"),
            "separation_unscaled": result.get("separation_unscaled") if ok else None,
            "source": "height_diagnosis.estimate_room_height", "units": "arbitrary", "scale_applied": False,
            "reasons": result.get("reasons", [])}


# ---------------------------------------------------------------------------------------- diagnosis of existing
def _per_view_points(prediction):
    """Points of extract_planes.points_from_prediction plus the originating view index."""
    from extract_planes import points_from_prediction
    chunks, ids = [], []
    for i in range(len(prediction["depth"])):
        part = {k: prediction[k][i:i + 1] for k in ("depth", "confidence", "intrinsics", "extrinsics")}
        try:
            pts = points_from_prediction(part)
        except ValueError:
            continue
        chunks.append(pts)
        ids.append(np.full(len(pts), i))
    return np.concatenate(chunks), np.concatenate(ids)


def diagnose_existing(plane_data, points, axis, centers):
    """Run estimate_height.py unchanged and explain its refusal with numbers. Truth-free."""
    import estimate_height as eh
    existing = eh.estimate_height(plane_data, points, axis, centers)
    ev = existing.get("support_evidence", {})
    thr = ev.get("support_thresholds", {})
    rows, blockers = [], []
    for p in ev.get("candidate_planes", []):
        inter = np.asarray(p["camera_vertical_intersections_unscaled"], float)
        side = "below_all_cameras" if inter.max() < 0 else ("above_all_cameras" if inter.min() > 0 else "crosses_camera_heights")
        fails = []
        if p["patch_area_fraction_of_cloud_footprint"] < thr.get("minimum_patch_area_fraction", 0):
            fails.append("patch_area")
        if p["camera_centers_in_patch_fraction"] < thr.get("minimum_camera_coverage", 0):
            fails.append("camera_coverage")
        if p["support_fraction_of_candidates"] < thr.get("minimum_candidate_support_fraction", 0):
            fails.append("candidate_support_share")
        rows.append({"plane_id": p["plane_id"], "support_count": p["support_count"],
                     "normal_up_alignment": p["normal_up_alignment"],
                     "patch_area_fraction": p["patch_area_fraction_of_cloud_footprint"],
                     "camera_coverage": p["camera_centers_in_patch_fraction"],
                     "support_share_of_all_candidates": p["support_fraction_of_candidates"],
                     "camera_intersection_range": [float(inter.min()), float(inter.max())],
                     "side": side, "failed_checks": fails, "accepted": p["support_accepted"]})
    planes = plane_data.get("planes", []) if isinstance(plane_data, dict) else []
    total = sum(max(0, int(p.get("support_count", 0) or 0)) for p in planes)
    axis = _unit(axis)
    horizontal_ids = []
    for i, p in enumerate(planes):
        n = _unit(p["normal"])
        if abs(n @ axis) >= np.cos(np.radians(12)):
            horizontal_ids.append(i)
    pair_angle = None
    if len(rows) >= 2:
        ns = [_unit(planes[r["plane_id"]]["normal"]) for r in rows]
        pair_angle = float(np.degrees(np.arccos(np.clip(abs(ns[0] @ ns[1]), -1., 1.))))
    pts = np.asarray(points, float)
    h = pts @ axis
    # Counterfactual (diagnostic only, never used downstream): which next check would fail if the
    # candidate-support-share check alone were relaxed to just below the weakest cap-like plane.
    counterfactual = None
    cap_like = [r for r in rows if not (set(r["failed_checks"]) - {"candidate_support_share"})]
    if len(cap_like) >= 2 and existing["status"] == "incomplete":
        relaxed = min(r["support_share_of_all_candidates"] for r in cap_like) * 0.99
        alt = eh.estimate_height(plane_data, points, axis, centers, min_support_fraction=relaxed)
        counterfactual = {"min_support_fraction_used": relaxed, "status": alt["status"], "reasons": alt["reasons"],
                          "separation_unscaled": alt.get("separation_unscaled"),
                          "pair_normal_angle_degrees": alt.get("pair_normal_angle_degrees"),
                          "local_separation_range_unscaled": alt.get("local_separation_range_unscaled"),
                          "local_separation_median_unscaled": alt.get("local_separation_median_unscaled")}
    return {"counterfactual_relaxed_support_share": counterfactual, "existing_status": existing["status"], "existing_reasons": existing["reasons"],
            "declared_axis": axis.tolist(), "total_candidate_support_points": total,
            "axis_parallel_plane_ids_within_12deg": horizontal_ids,
            "thresholds_of_existing": thr,
            "candidate_horizontal_planes": rows,
            "angle_between_first_two_candidates_degrees": pair_angle,
            "camera_height_interval_unscaled": ev.get("camera_height_interval_unscaled"),
            "cloud_height_percentiles_0_1_5_50_95_99_100": np.percentile(h, [0, 1, 5, 50, 95, 99, 100]).tolist(),
            "existing_result": existing}


def run(prediction_path, planes_path, axes_path, hypothesis_indices=None):
    """Both parts for one frozen prediction; returns one record per co-supported hypothesis."""
    import estimate_height as eh
    with np.load(prediction_path) as pred:
        data = {k: pred[k] for k in ("depth", "confidence", "intrinsics", "extrinsics")}
    points, ids = _per_view_points(data)
    centers = eh.camera_centers(data["extrinsics"])
    planes = json.loads(Path(planes_path).read_text(encoding="utf-8"))
    axes = json.loads(Path(axes_path).read_text(encoding="utf-8"))
    out = []
    for k, hyp in enumerate(axes["hypotheses"]):
        if hypothesis_indices is not None and k not in hypothesis_indices:
            continue
        if not hyp.get("co_supported_with_best"):
            continue
        axis = np.asarray(hyp["axis_world"], float)
        diag = diagnose_existing(planes, points, axis, centers)
        est = estimate_room_height(points, axis, ids, reference=diag["existing_result"])
        out.append({"hypothesis_index": k, "axis_world": axis.tolist(), "diagnosis": diag, "estimate": est,
                    "evaluator_compatible_height": evaluator_compatible(est)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--axes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.prediction, args.planes, args.axes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, default=float), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
