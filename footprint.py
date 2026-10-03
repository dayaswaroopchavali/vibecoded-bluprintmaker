"""Floor/ceiling footprint polygon from RGB-derived point clouds (geometry only, arbitrary units, no gravity).

Wall planes from DA3 are sparse and oblique, but the floor and ceiling cover the whole room footprint densely. Given
a structural-axis hypothesis (geometry_orientation.axis_hypotheses; sign unresolved) and the two parallel planes that
bracket the cloud (floor/ceiling candidates), this module projects the points near those two planes onto the plane
perpendicular to the axis, rasterises them, closes small holes, keeps the largest connected region, extracts its outer
contour and simplifies it into a polygon. The result is an *observed-support estimate*: the outline of where floor or
ceiling was seen. Edges are labelled `observed` only where a wall-height band of points lies along them; every other
boundary stays in the polygon as `inferred`. No convex hull of the cloud is ever used, no right angles are imposed, and
no simulator, camera, annotation or ground-truth data are read.

Same frame convention as room_hypothesis.estimate_polygon: e1 = unit(cross(axis, seed)), e2 = cross(axis, e1), origin =
median of the finite cloud. Flipping the axis sign mirrors the plan (the handedness of e1, e2 flips); the evaluator
therefore tries the mirrored plan only when the sign is unresolved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from room_hypothesis import _inside, _is_simple, _segment_distance, _unit

PARALLEL_COS = float(np.cos(np.radians(12)))
CELL_FRACTION = 0.008            # raster cell, fraction of the cloud extent
SLAB_FRACTION = 0.02             # half-thickness of the floor/ceiling slab, fraction of extent
TOL_FRACTION = 0.02              # wall-evidence tolerance, fraction of extent
SIMPLIFY_FRACTION = 0.025        # Douglas-Peucker tolerance, fraction of extent
RESTORE_CORNER_FRACTION = 0.15   # edges shorter than this (and bulging < the simplify tolerance) between clearly non-parallel longer edges collapse to a corner
COLLINEAR_DEG = 8.0              # consecutive edges closer than this in direction are merged
BAND_MARGIN = 0.15               # wall band excludes this fraction of the separation next to floor/ceiling
OBSERVED_EDGE_FRACTION = 0.60    # wall-band coverage along an edge needed for the label "observed"
MAX_OUTSIDE_FRACTION = 0.04      # wall-band points beyond 2 tol outside the polygon => unobserved room
FRAGMENT_FRACTION = 0.10         # a second component this large (vs the largest) means a fragmented footprint
MIN_SURFACE_POINTS = 50
MIN_INTERIOR_CELLS = 25          # coarse (3% of extent) cells with occupied 4-neighbourhoods needed for a supported surface
CELL_OFFSET = 0.0                # outward offset of fitted edges, in cells (a pixel is occupied when any point falls in it)
EDGE_BIAS_SPACING = 0.0          # outward offset of fitted edges for the sampling bias, in mean point spacings


def _planes(plane_data):
    out = {}
    for i, p in enumerate(plane_data.get("planes", []) if isinstance(plane_data, dict) else []):
        try:
            n = np.asarray(p["normal"], dtype=float); d = float(p["offset"]); w = max(0, int(p.get("support_count", 0) or 0))
            norm = float(np.linalg.norm(n))
            if n.shape != (3,) or not np.isfinite(n).all() or not np.isfinite(d) or norm < 1e-10:
                continue
            out[i] = (n / norm, d / norm, w)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return out


def _base(extra):
    return {"model": "floor_ceiling_footprint_polygon_hypothesis", "units": "arbitrary", "metric_scale_source": None,
            "estimate_kind": "observed_floor_ceiling_support_outline",
            "vertices_xy": None, "vertices_world": None, "edges": [], "area": None, "perimeter": None,
            "observed_fraction": None, "incomplete_boundaries": [], "shape_gate_eligible": False,
            "limitations": [
                "Outline of observed floor/ceiling support only: not a wall survey. Edges without a wall-height band of points along them stay 'inferred'.",
                "Needs an explicit structural-axis hypothesis and its bracketing plane pair; the axis sign is unresolved, so the plan may be mirrored.",
                "Corners are rounded by sampling density and the raster cell; no right angles or axis alignment are imposed.",
                "Furniture occluding the boundary can pull an edge inward; interior furniture holes are filled.",
                "Incomplete results carry no polygon; a hull of all points is never substituted.",
                "No simulator data, camera poses, annotations or ground truth are read."], **extra}


def _interior_cells(xy, cell):
    """Occupied coarse cells whose four neighbours are occupied too (0 for points that only trace an outline)."""
    if len(xy) == 0:
        return 0
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(int)
    grid = np.zeros((ij[:, 1].max() + 3, ij[:, 0].max() + 3), dtype=np.uint8)
    grid[ij[:, 1] + 1, ij[:, 0] + 1] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))   # tolerate single empty cells in sparse data
    core = cv2.erode(grid, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return int(core.sum())


def _fill_largest(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 1, thickness=-1)
    return out


def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _fit_line(pts):
    c = pts.mean(axis=0)
    d = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    return c, d


def _intersect(c1, d1, c2, d2):
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-6:
        return None
    t = ((c2[0] - c1[0]) * (-d2[1]) - (c2[1] - c1[1]) * (-d2[0])) / det
    return c1 + t * d1


def _simplify(contour_xy, eps):
    """Douglas-Peucker, then merge near-collinear edges and drop needle-thin spikes. Returns vertex contour indices."""
    cont = contour_xy
    approx = cv2.approxPolyDP(cont.astype(np.float32).reshape(-1, 1, 2), float(eps), True).reshape(-1, 2)
    idx, last = [], -1
    for v in approx:
        hit = np.where(np.all(np.isclose(cont, v, atol=1e-4), axis=1))[0]
        hit = hit[hit > last] if (hit > last).any() else hit
        idx.append(int(hit[0])); last = idx[-1]
    idx = sorted(set(idx))
    verts = lambda ii: cont[ii]
    changed = True
    while changed and len(idx) > 3:
        changed = False
        v = verts(idx); n = len(idx)
        prev, nxt = np.roll(v, 1, axis=0), np.roll(v, -1, axis=0)
        a, b = v - prev, nxt - v
        ang = np.degrees(np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))))
        order = np.argsort(ang)
        lens = np.linalg.norm(b, axis=1); prev_lens = np.roll(lens, 1)
        spikes = np.where((ang > 110.0) & (np.minimum(lens, prev_lens) < 4 * eps))[0]   # needle-thin slit/spike: not a wall
        if len(spikes):
            del idx[int(spikes[0])]; changed = True; continue
        if ang[order[0]] < COLLINEAR_DEG:
            del idx[int(order[0])]; changed = True; continue
    return idx


def _unit2(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-300)


def _restore_corners(poly, lines, max_len, max_deviation):
    """Collapse short edges (corner chamfers left by simplification/closing) back to the intersection of their neighbours.

    Only when the neighbouring edges are clearly non-parallel and much longer; a genuine step between parallel walls is kept.
    """
    P = [np.asarray(p, float) for p in poly]; L = list(lines)
    while len(P) > 3:
        n = len(P)
        lens = [float(np.linalg.norm(P[(i + 1) % n] - P[i])) for i in range(n)]
        done = False
        for i in sorted(range(n), key=lambda k: lens[k]):
            if lens[i] >= max_len:
                break
            (c1, d1), (c2, d2) = L[i - 1], L[(i + 1) % n]
            if abs(float(_unit2(d1) @ _unit2(d2))) > np.cos(np.radians(25)):
                continue
            if min(lens[i - 1], lens[(i + 1) % n]) < 1.5 * lens[i]:
                continue
            p = _intersect(c1, d1, c2, d2)
            if p is None or float(_segment_distance(p[None, :], P[i], P[(i + 1) % n])[0]) > max_deviation:
                continue
            P = P[i:] + P[:i]; L = L[i:] + L[:i]
            P[0] = p; P.pop(1); L.pop(0)
            done = True
            break
        if not done:
            break
    return np.array(P), L


def _wall_evidence(edge_a, edge_b, wall_xy, tol, gap):
    ab = edge_b - edge_a
    length = float(np.linalg.norm(ab))
    if len(wall_xy) == 0 or length < 1e-12:
        return 0.0
    s = ((wall_xy - edge_a) @ ab) / (length * length)
    dist = _segment_distance(wall_xy, edge_a, edge_b)
    near = (dist < tol) & (s >= -0.0) & (s <= 1.0)
    bins = max(1, int(np.ceil(length / gap)))
    counts = np.bincount(np.minimum((s[near] * bins).astype(int), bins - 1), minlength=bins)
    return float((counts >= 3).sum() / bins)


def estimate_footprint(plane_data: dict, points: np.ndarray, axis, bracketing_plane_ids=None, *,
                       cell_fraction: float = CELL_FRACTION, slab_fraction: float = SLAB_FRACTION,
                       tol_fraction: float = TOL_FRACTION, simplify_fraction: float = SIMPLIFY_FRACTION,
                       snap_to_wall_lines: bool = True, min_surface_points: int | None = None) -> dict:
    """Footprint polygon of the observed floor/ceiling support; `incomplete` (no polygon) when it cannot be trusted."""
    points = np.asarray(points, dtype=float)
    finite = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 and points.shape[1:] == (3,) else np.empty((0, 3))
    axis = np.asarray(axis, dtype=float).ravel() if axis is not None else np.zeros(3)
    if len(finite) < 100:
        return _base({"status": "insufficient_geometry", "point_count": int(len(finite))})
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        return _base({"status": "incomplete", "point_count": int(len(finite)),
                      "incomplete_boundaries": [{"reason": "axis_unavailable"}]})
    axis = _unit(axis)
    seed = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
    e1 = _unit(np.cross(axis, seed)); e2 = np.cross(axis, e1)
    center = np.median(finite, axis=0)
    rel = finite - center
    height = rel @ axis
    xy_all = np.stack([rel @ e1, rel @ e2], axis=1)
    extent = float(np.linalg.norm(np.percentile(finite, 95, axis=0) - np.percentile(finite, 5, axis=0)))
    frame = {"point_count": int(len(finite)), "coordinate_origin_world": center.tolist(),
             "axes_world": {"e1": e1.tolist(), "e2": e2.tolist(), "up": axis.tolist()}, "extent": extent}
    if extent <= 0:
        return _base({**frame, "status": "incomplete", "incomplete_boundaries": [{"reason": "degenerate_point_cloud_extent"}]})
    planes = _planes(plane_data)
    min_pts = int(min_surface_points) if min_surface_points is not None else max(MIN_SURFACE_POINTS, int(0.015 * len(finite)))

    def position(pid):   # plane height along the axis relative to the cloud median
        n, d, _w = planes[pid]
        return float(-(d + n @ center) / (n @ axis))

    def parallel(pid):
        return abs(float(planes[pid][0] @ axis)) >= PARALLEL_COS

    # 1. Bracketing pair: supplied ids, or the largest parallel planes on opposite sides of the median.
    if bracketing_plane_ids is not None:
        ids = list(bracketing_plane_ids)
        if len(ids) != 2 or ids[0] == ids[1] or any(i not in planes for i in ids) or not all(parallel(i) for i in ids):
            return _base({**frame, "status": "incomplete", "bracketing_plane_ids": ids,
                          "incomplete_boundaries": [{"reason": "bracketing_planes_unusable"}]})
    else:
        par = [i for i in planes if parallel(i) and planes[i][2] > 0]
        below = [i for i in par if position(i) < 0]; above = [i for i in par if position(i) > 0]
        if not below or not above:
            return _base({**frame, "status": "incomplete", "incomplete_boundaries": [{"reason": "no_bracketing_pair"}]})
        ids = [max(below, key=lambda i: (planes[i][2], -i)), max(above, key=lambda i: (planes[i][2], -i))]
    pos = [position(i) for i in ids]
    ids = [ids[0], ids[1]] if pos[0] <= pos[1] else [ids[1], ids[0]]
    pos = sorted(pos)
    separation = pos[1] - pos[0]
    if separation < 0.05 * extent:
        return _base({**frame, "status": "incomplete", "bracketing_plane_ids": ids,
                      "incomplete_boundaries": [{"reason": "bracketing_planes_too_close", "separation": separation}]})
    slab = min(slab_fraction * extent, 0.2 * separation)
    tol = tol_fraction * extent
    near = np.abs(xy_all).max(axis=1) <= 1.2 * extent              # horizontal gate against stray outliers
    surfaces = []
    for pid in ids:
        n, d, _w = planes[pid]
        mask = (np.abs(finite @ n + d) < slab) & near
        surfaces.append(mask)
    counts = [int(m.sum()) for m in surfaces]
    interior = [_interior_cells(xy_all[m], 0.03 * extent) for m in surfaces]
    # Wall points just above/below a plane lie on the outline only; real floor/ceiling support has interior coverage.
    supported = [c >= min_pts and i >= MIN_INTERIOR_CELLS for c, i in zip(counts, interior)]
    support = {"plane_ids": ids, "slab_half_thickness": float(slab), "points_near_plane": counts, "interior_cells": interior,
               "minimum_points_per_surface": min_pts, "plane_positions_along_axis": pos}
    if not any(supported):
        return _base({**frame, "status": "incomplete", "support": support,
                      "incomplete_boundaries": [{"reason": "insufficient_floor_ceiling_support", "points": counts}]})
    use = np.zeros(len(finite), dtype=bool)
    for m, ok in zip(surfaces, supported):
        if ok:
            use |= m
    flags = [] if all(supported) else [{"flag": "single_surface_observed",
                                        "observed_plane_id": ids[supported.index(True)],
                                        "note": "only one of floor/ceiling has support; the outline rests on a single surface"}]
    h_lo = float(np.median(height[surfaces[0]])) if supported[0] else pos[0]
    h_hi = float(np.median(height[surfaces[1]])) if supported[1] else pos[1]
    band = near & (height > h_lo + BAND_MARGIN * (h_hi - h_lo)) & (height < h_hi - BAND_MARGIN * (h_hi - h_lo))
    wall_xy = xy_all[band]
    xy = xy_all[use]

    # 2. Raster, close, largest component, fill.
    cell = cell_fraction * extent
    kmax = max(1, int(round(0.04 * extent / cell)))
    pad = kmax + 4
    lo = xy.min(axis=0) - pad * cell
    shape = (np.ceil((xy.max(axis=0) - lo) / cell).astype(int) + pad + 1)[::-1]   # (rows=y, cols=x)
    if shape.max() > 4000:
        return _base({**frame, "status": "incomplete", "support": support,
                      "incomplete_boundaries": [{"reason": "footprint_raster_too_large"}]})
    cols = np.floor((xy[:, 0] - lo[0]) / cell).astype(int); rows = np.floor((xy[:, 1] - lo[1]) / cell).astype(int)
    occ = np.zeros(tuple(shape), dtype=np.uint8); occ[rows, cols] = 1
    k, spacing = 2, cell
    for _ in range(3):
        closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        filled = _fill_largest(closed)
        spacing = float(np.sqrt(filled.sum() * cell * cell / len(xy)))
        k = int(np.clip(round(2.0 * spacing / cell), 1, kmax))
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    raster = {"cell": float(cell), "closing_radius_cells": int(k), "mean_point_spacing": spacing,
              "components": int(len(areas)), "component_areas_cells": [int(a) for a in areas[order][:5]]}
    if len(areas) > 1 and areas[order[1]] >= FRAGMENT_FRACTION * areas[order[0]]:
        return _base({**frame, "status": "incomplete", "support": support, "raster": raster,
                      "incomplete_boundaries": [{"reason": "fragmented_footprint", "large_components": int((areas >= FRAGMENT_FRACTION * areas[order[0]]).sum())}]})
    region = _fill_largest((labels == 1 + int(order[0])).astype(np.uint8))
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    if len(contour) < 8:
        return _base({**frame, "status": "incomplete", "support": support, "raster": raster,
                      "incomplete_boundaries": [{"reason": "degenerate_footprint_region"}]})
    cont_xy = lo + (contour + 0.5) * cell          # pixel centres in the e1/e2 frame (x -> column, y -> row)
    if _signed_area(cont_xy) < 0:
        cont_xy = cont_xy[::-1].copy()

    # 3. Simplify (DP + collinear merge), then refit each edge from its own contour run and intersect neighbours.
    eps = simplify_fraction * extent
    idx = _simplify(cont_xy, eps)
    if len(idx) < 3:
        return _base({**frame, "status": "incomplete", "support": support, "raster": raster,
                      "incomplete_boundaries": [{"reason": "fewer_than_three_vertices"}]})
    v0 = cont_xy[idx]
    n = len(idx)
    lines = []
    for i in range(n):
        a, b = idx[i], idx[(i + 1) % n]
        run = cont_xy[a:b + 1] if b > a else np.vstack([cont_xy[a:], cont_xy[:b + 1]])
        trim = int(0.15 * len(run))
        core = run[trim:len(run) - trim] if len(run) - 2 * trim >= 4 else run
        lines.append(_fit_line(core) if len(core) >= 2 else (v0[i], _unit(v0[(i + 1) % n] - v0[i])))
    bias = EDGE_BIAS_SPACING * spacing
    shifted = []
    for c, d in lines:
        normal = np.array([d[1], -d[0]])                   # outward for a CCW polygon (direction sign fixed below)
        shifted.append((c, d, normal))
    verts = []
    for i in range(n):
        va, vb = v0[i], v0[(i + 1) % n]
        direction = vb - va
        c, d, nrm = shifted[i]
        if d @ direction < 0:
            d = -d; nrm = -nrm
        off = (CELL_OFFSET * cell * (abs(nrm[0]) + abs(nrm[1])) + bias) * nrm
        shifted[i] = (c + off, d)
    for i in range(n):
        (c1, d1), (c2, d2) = shifted[i - 1], shifted[i]
        p = _intersect(c1, d1, c2, d2)
        verts.append(p if p is not None and np.linalg.norm(p - v0[i]) < 3 * eps else v0[i])
    poly = np.array(verts)
    if _signed_area(poly) <= 0 or not _is_simple(poly):
        poly = v0                                             # refit failed: keep the raw simplified contour
    else:
        restored, _lines = _restore_corners(poly, shifted, RESTORE_CORNER_FRACTION * extent, eps)
        if _signed_area(restored) > 0 and _is_simple(restored):
            poly = restored
    if _signed_area(poly) <= 0 or not _is_simple(poly) or len(poly) < 3:
        return _base({**frame, "status": "incomplete", "support": support, "raster": raster,
                      "incomplete_boundaries": [{"reason": "polygon_not_simple_counter_clockwise"}]})

    # 4. Unobserved-room test: supported wall-band points lying outside the polygon.
    gap = 0.04 * extent
    outside_fraction, outside_points = 0.0, 0
    if len(wall_xy):
        out = wall_xy[~_inside(poly, wall_xy)]
        if len(out):
            dist = np.min([_segment_distance(out, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))], axis=0)
            out = out[dist > 2 * tol]
            if len(out):
                cells = np.floor(out / (0.03 * extent)).astype(np.int64)
                _u, inv, cnt = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
                outside_points = int((cnt[inv.ravel()] >= 5).sum())    # isolated stray points are not "a region"
        outside_fraction = outside_points / len(wall_xy)
    evidence_summary = {"wall_band_point_count": int(len(wall_xy)), "outside_wall_band_fraction": float(outside_fraction),
                        "outside_fraction_limit": MAX_OUTSIDE_FRACTION, "tolerance": float(tol)}
    if outside_fraction > MAX_OUTSIDE_FRACTION:
        return _base({**frame, "status": "incomplete", "support": support, "raster": raster, "wall_evidence": evidence_summary,
                      "rejected_candidate_vertices_xy": poly.tolist(),
                      "incomplete_boundaries": [{"reason": "wall_points_outside_floor_ceiling_footprint_unobserved_room",
                                                 "outside_fraction": float(outside_fraction)}]})

    # 5. Edge labels from wall evidence; optionally snap evidenced edges onto the wall points' own line.
    m = len(poly)
    fractions = [_wall_evidence(poly[i], poly[(i + 1) % m], wall_xy, tol, gap) for i in range(m)]
    snapped = [False] * m
    if snap_to_wall_lines and len(wall_xy):
        lines2 = []
        for i in range(m):
            a, b = poly[i], poly[(i + 1) % m]
            lines2.append((a, b - a))
            if fractions[i] < OBSERVED_EDGE_FRACTION:
                continue
            ab = b - a; length = float(np.linalg.norm(ab))
            s = ((wall_xy - a) @ ab) / (length * length)
            sel = (_segment_distance(wall_xy, a, b) < tol) & (s >= 0.1) & (s <= 0.9)
            if sel.sum() < 20:
                continue
            c, d = _fit_line(wall_xy[sel])
            if d @ ab < 0:
                d = -d
            angle = np.degrees(np.arccos(np.clip(abs(float(_unit(ab) @ d)), -1, 1)))
            nrm = np.array([ab[1], -ab[0]]) / length
            if angle <= 6.0 and abs(float((c - a) @ nrm)) <= tol:
                lines2[-1] = (c, d); snapped[i] = True
        if any(snapped):
            cand = []
            for i in range(m):
                p = _intersect(*lines2[i - 1], *lines2[i]) if (snapped[i - 1] or snapped[i]) else None
                cand.append(p if p is not None and np.linalg.norm(p - poly[i]) < 3 * tol else poly[i])
            cand = np.array(cand)
            if _signed_area(cand) > 0 and _is_simple(cand):
                poly = cand
                fractions = [_wall_evidence(poly[i], poly[(i + 1) % m], wall_xy, tol, gap) for i in range(m)]
    edges = []
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        edges.append({"index": i, "start_xy": a.tolist(), "end_xy": b.tolist(), "length": float(np.linalg.norm(b - a)),
                      "label": "observed" if fractions[i] >= OBSERVED_EDGE_FRACTION else "inferred",
                      "wall_evidence_fraction": float(fractions[i]), "snapped_to_wall_points": bool(snapped[i])})
    perimeter = float(sum(e["length"] for e in edges))
    observed_len = sum(e["length"] for e in edges if e["label"] == "observed")
    floor_world = center + h_lo * axis
    return _base({**frame, "status": "complete_polygon_hypothesis", "vertices_xy": poly.tolist(),
                  "vertices_world": [(floor_world + v[0] * e1 + v[1] * e2).tolist() for v in poly],
                  "edges": edges, "area": _signed_area(poly), "perimeter": perimeter,
                  "observed_fraction": float(observed_len / perimeter) if perimeter > 0 else 0.0,
                  "inferred_edge_count": sum(e["label"] == "inferred" for e in edges),
                  "support": support, "raster": raster, "wall_evidence": evidence_summary, "flags": flags,
                  "bracketing_plane_ids": ids, "floor_ceiling_separation_unscaled": float(h_hi - h_lo),
                  "outside_point_fraction": float(outside_fraction), "shape_gate_eligible": False,
                  "incomplete_boundaries": []})


ROBUST_GRID = [(c, s) for c in (0.008, 0.012, 0.015, 0.02) for s in (0.025, 0.05, 0.08, 0.12)]


def estimate_footprint_robust(plane_data, points, axis, bracketing_plane_ids=None, grid=ROBUST_GRID, area_tolerance=0.05):
    """Consensus over raster/simplification scales: the single-scale estimate flips between complete and incomplete and
    between 21 and 5 edges on noisy clouds. Keep the scales that produce a complete polygon, take the median area, drop
    results whose area is more than `area_tolerance` away, and return the one with the fewest edges (then most observed
    edges). If fewer than half of the scales complete, the result is incomplete (reasons reported). No truth is used."""
    runs = [estimate_footprint(plane_data, points, axis, bracketing_plane_ids, cell_fraction=c, simplify_fraction=s) for c, s in grid]
    done = [r for r in runs if r.get("status") == "complete_polygon_hypothesis"]
    summary = {"scales_tried": len(runs), "scales_complete": len(done),
               "edge_counts_complete": sorted(len(r["edges"]) for r in done)}
    if len(done) * 2 < len(runs):
        base = next((r for r in runs if r.get("status") != "complete_polygon_hypothesis"), runs[0])
        reasons = sorted({b.get("reason", "?") for r in runs for b in r.get("incomplete_boundaries", [])})
        return {**base, "status": "incomplete", "consensus": {**summary, "incomplete_reasons": reasons},
                "incomplete_boundaries": [{"reason": "footprint_not_stable_across_scales", **summary}] + base.get("incomplete_boundaries", [])}
    median_area = float(np.median([r["area"] for r in done]))
    stable = [r for r in done if abs(r["area"] - median_area) <= area_tolerance * median_area]
    best = sorted(stable, key=lambda r: (len(r["edges"]), -sum(e["label"] == "observed" for e in r["edges"])))[0]
    return {**best, "consensus": {**summary, "stable_scales": len(stable), "median_area": median_area, "chosen_edge_count": len(best["edges"])}}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planes", type=Path, required=True); parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--orientation", type=Path, required=True, help="geometry_orientation.py output JSON")
    parser.add_argument("--hypothesis", type=int, default=0, help="rank of the axis hypothesis to use")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    from extract_planes import points_from_prediction
    with np.load(args.prediction) as data:
        points = points_from_prediction(data)
    hyp = json.loads(args.orientation.read_text(encoding="utf-8"))["hypotheses"][args.hypothesis]
    pair = hyp.get("bracketing_pair")
    result = estimate_footprint_robust(json.loads(args.planes.read_text(encoding="utf-8")), points, hyp["axis_world"],
                                       pair["plane_ids"] if pair else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output, result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





