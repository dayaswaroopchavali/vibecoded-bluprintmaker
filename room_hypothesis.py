"""Estimate an unscaled floor-plan polygon hypothesis (and a cuboid diagnostic baseline).

estimate_polygon is the room hypothesis: arbitrary polygons, edges labelled observed or
inferred, missing boundaries reported as incomplete. estimate (the single rectilinear
box fit) is kept as estimate_cuboid_baseline for diagnostics only.

Uses only generic RGB-derived DA3 predictions and their plane-candidate JSON.
The result is a geometry hypothesis, not a house plan or verified boundary.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def _unit(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    return value / length if length > 1e-10 else np.zeros(3, dtype=float)


def _cluster_directions(planes: list[dict], cosine: float = 0.94) -> list[dict]:
    usable = []
    for index, plane in enumerate(planes):
        try:
            normal = _unit(np.asarray(plane["normal"], dtype=float))
            support = max(0, int(plane.get("support_count", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(normal).all() and np.linalg.norm(normal) > 0.9 and support > 0:
            usable.append((index, normal, support))
    usable.sort(key=lambda item: item[2], reverse=True)
    clusters: list[dict] = []
    for index, normal, support in usable:
        best, best_dot = None, cosine
        for cluster in clusters:
            dot = abs(float(normal @ cluster["direction"]))
            if dot > best_dot:
                best, best_dot = cluster, dot
        if best is None:
            clusters.append({"members": [], "direction": normal.copy(), "weight": 0.0})
            best = clusters[-1]
        aligned = normal if normal @ best["direction"] >= 0 else -normal
        weight = float(np.sqrt(support))
        best["members"].append((index, normal, support))
        best["direction"] = _unit(best["direction"] * best["weight"] + aligned * weight)
        best["weight"] += weight
    return clusters


def _positions(cluster: dict, planes: list[dict], axis: np.ndarray,
               cloud_center: np.ndarray) -> list[tuple[int, float, int]]:
    result = []
    for index, normal, support in cluster["members"]:
        plane = planes[index]
        try:
            offset = float(plane["offset"])
        except (KeyError, TypeError, ValueError):
            continue
        denom = float(normal @ axis)
        if abs(denom) > 0.75 and np.isfinite(offset):
            # Intersect the plane with the line through a shared robust cloud
            # center. This remains translation-equivariant for noisy normals.
            result.append((index, -(offset + float(normal @ cloud_center)) / denom, support))
    return result


def _pair_for_axis(cluster: dict, planes: list[dict], axis: np.ndarray,
                   points: np.ndarray, cloud_center: np.ndarray) -> dict:
    coordinates = (points - cloud_center) @ axis
    low_ref, high_ref = np.percentile(coordinates, [5, 95])
    span = float(high_ref - low_ref)
    candidates = _positions(cluster, planes, axis, cloud_center)
    best = None
    if span <= 1e-8:
        return {"complete": False, "reason": "degenerate point-cloud extent", "candidate_count": len(candidates)}
    for first, second in itertools.combinations(candidates, 2):
        low, high = sorted((first[1], second[1]))
        separation = high - low
        ratio = separation / span
        if ratio < 0.48:
            continue
        # Prefer a pair bracketing the bulk of the cloud with support on both sides.
        endpoint_error = (abs(low - low_ref) + abs(high - high_ref)) / span
        support_balance = min(first[2], second[2]) / max(first[2], second[2])
        support_score = np.log1p(first[2]) + np.log1p(second[2])
        score = support_score + 3.0 * support_balance - 5.0 * endpoint_error
        pair = {"complete": True, "low": float(low), "high": float(high),
                "length": float(separation), "support_score": float(score),
                "planes": [{"index": int(first[0]), "support_count": int(first[2])},
                           {"index": int(second[0]), "support_count": int(second[2])}],
                "point_extent_5_95": [float(low_ref), float(high_ref)],
                "bracket_ratio": float(ratio), "endpoint_error": float(endpoint_error)}
        if best is None or score > best["support_score"]:
            best = pair
    if best is not None:
        return best
    return {"complete": False, "reason": "no sufficiently separated supported opposing plane pair",
            "candidate_count": len(candidates), "point_extent_5_95": [float(low_ref), float(high_ref)]}


def estimate(plane_data: dict, points: np.ndarray, min_points: int = 500) -> dict:
    """Return an arbitrary-unit box hypothesis, or an explicit incomplete result."""
    points = np.asarray(points, dtype=float)
    finite = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 and points.shape[1:] == (3,) else np.empty((0, 3))
    base = {"model": "single_rectilinear_room_box_hypothesis", "role": "diagnostic_baseline_only", "units": "arbitrary",
            "metric_scale_source": None, "limitations": [
                "This is a single rectilinear box hypothesis, not a general house model.",
                "Planes and observed bounds may include furniture or incomplete surfaces.",
                "Openings are not detected or validated.",
                "No metric scale, simulator data, or annotations are used."]}
    if len(finite) < min_points:
        return {**base, "status": "insufficient_geometry", "point_count": int(len(finite)),
                "required_point_count": int(min_points), "axes": [], "corners": None,
                "missing_surfaces": ["all"]}
    cloud_center = np.median(finite, axis=0)
    planes = plane_data.get("planes", []) if isinstance(plane_data, dict) else []
    clusters = _cluster_directions(planes)
    if len(clusters) < 3:
        return {**base, "status": "incomplete_orientation", "point_count": int(len(finite)),
                "direction_family_count": len(clusters), "axes": [], "corners": None,
                "missing_surfaces": ["third orthogonal orientation family"]}

    triples = []
    for ids in itertools.combinations(range(len(clusters)), 3):
        dirs = [clusters[i]["direction"] for i in ids]
        dots = [abs(float(dirs[0] @ dirs[1])), abs(float(dirs[0] @ dirs[2])), abs(float(dirs[1] @ dirs[2]))]
        if max(dots) > 0.42:
            continue
        axes = [dirs[0], _unit(dirs[1] - dirs[0] * float(dirs[0] @ dirs[1]))]
        axes.append(_unit(np.cross(axes[0], axes[1])))
        if abs(float(axes[2] @ dirs[2])) < 0.78:
            continue
        pair_results = [_pair_for_axis(clusters[i], planes, axes[j], finite, cloud_center)
                        for j, i in enumerate(ids)]
        support = sum(np.log1p(sum(member[2] for member in clusters[i]["members"])) for i in ids)
        pair_score = sum(p.get("support_score", 0.0) if p["complete"] else 0.0 for p in pair_results)
        missing_penalty = sum(not p["complete"] for p in pair_results) * 2.5
        score = support + pair_score - 8.0 * sum(dots) - missing_penalty
        triples.append({"ids": ids, "axes": axes, "pairs": pair_results,
                        "score": float(score), "orthogonality_error": float(max(dots))})
    if not triples:
        return {**base, "status": "incomplete_orientation", "point_count": int(len(finite)),
                "direction_family_count": len(clusters), "axes": [], "corners": None,
                "missing_surfaces": ["three near-orthogonal orientation families"]}
    triples.sort(key=lambda t: t["score"], reverse=True)
    chosen = triples[0]
    complete = all(pair["complete"] for pair in chosen["pairs"])
    ambiguous = len(triples) > 1 and (chosen["score"] - triples[1]["score"]) < max(1.0, abs(chosen["score"]) * 0.025)

    # Re-orthogonalized axes form a right-handed coordinate frame. Their signs
    # only affect corner ordering; lengths remain arbitrary and unscaled.
    axes = chosen["axes"]
    axis_rows = []
    lengths = []
    missing = []
    for j, (axis, pair, cluster_id) in enumerate(zip(axes, chosen["pairs"], chosen["ids"])):
        if pair["complete"]:
            lengths.append(pair["length"])
        else:
            lengths.append(None)
            missing.append(f"axis_{j}: {pair['reason']}")
        axis_rows.append({"name": f"axis_{j}", "direction_world": axis.tolist(),
                          "surface_pair": pair, "candidate_plane_indices": [m[0] for m in clusters[cluster_id]["members"]]})

    corners = None
    ratios = None
    if complete:
        corners = []
        for bits in itertools.product((0, 1), repeat=3):
            position = cloud_center + sum((chosen["pairs"][j]["high"] if bits[j] else chosen["pairs"][j]["low"]) * axes[j]
                                          for j in range(3))
            corners.append(position.tolist())
        normalized = np.asarray(lengths, dtype=float) / max(lengths)
        ratios = normalized.tolist()

    result = {**base, "status": "ambiguous" if ambiguous else ("complete_box_hypothesis" if complete else "incomplete"),
              "point_count": int(len(finite)), "plane_candidate_count": len(planes),
              "coordinate_origin_world": cloud_center.tolist(),
              "orientation_score": chosen["score"], "orthogonality_error_max_abs_dot": chosen["orthogonality_error"],
              "competing_orientation_score": triples[1]["score"] if ambiguous else None,
              "axes": axis_rows, "dimensions": lengths if complete else [v if v is not None else None for v in lengths],
              "dimension_ratios_to_longest": ratios, "corners": corners,
              "missing_surfaces": missing,
              "support_diagnostics": {"selected_plane_indices": [p["index"] for axis in chosen["pairs"] if axis["complete"] for p in axis["planes"]],
                                      "orientation_families_considered": len(clusters),
                                      "alternative_orthogonal_triplets": max(0, len(triples)-1)}}
    return result


estimate_cuboid_baseline = estimate  # diagnostic baseline only; estimate_polygon is the room hypothesis


# ---------------------------------------------------------------------------
# Floor-plan polygon hypothesis
# ---------------------------------------------------------------------------
# Thresholds are fractions of the cloud extent so they are scale-free. They were
# chosen from geometry (a wall must reach the ceiling, have an empty far side,
# etc.), not tuned to any scene.
TOL_FRACTION = 0.01          # plane/line inlier distance
GAP_FRACTION = 0.05          # break in a wall's support along its length
VERTICAL_TILT = 0.15         # |n . up| below this => vertical plane
BAND_FRACTION = 0.15         # wall support is read between 15% and 85% of the vertical extent
MIN_WALL_HEIGHT_FRACTION = 0.6
CEILING_REACH_FRACTION = 0.2
MAX_BEHIND_FRACTION = 0.05   # support on the far side of a wall patch
BEHIND_DEPTH_FRACTION = 0.12  # depth of the slab behind a patch inspected for a wall it might be hiding
BEHIND_ROOM_FRACTION = 0.1   # ...counts as "still room" only if this share of it is floor/ceiling
CORNER_REACH_FRACTION = 0.15  # largest tolerated unobserved stretch at a corner
OVERSHOOT_FRACTION = 0.04
OBSERVED_EDGE_FRACTION = 0.6
MIN_EDGE_FRACTION = 0.15
MAX_OUTSIDE_FRACTION = 0.04  # points beyond the polygon
PARALLEL_SIN = np.sin(np.radians(8))
MAX_SEARCH_WALLS = 10        # bounded local search for the closed boundary through all supported walls
SEARCH_STEP_BUDGET = 200000
MAX_GROUPING_VARIANTS = 64


def _fit_line(xy: np.ndarray) -> tuple[np.ndarray, float]:
    center = xy.mean(axis=0)
    direction = np.linalg.svd(xy - center, full_matrices=False)[2][0]
    normal = np.array([direction[1], -direction[0]])
    return normal, float(normal @ center)


def _intervals(t: np.ndarray, gap: float, min_len: float, min_count: int = 5) -> list[tuple[float, float, int]]:
    t = np.sort(t)
    groups = np.split(t, np.where(np.diff(t) > gap)[0] + 1)
    return [(float(g[0]), float(g[-1]), int(len(g))) for g in groups if len(g) >= min_count and g[-1] - g[0] >= min_len]


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    s = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-300), 0, 1)
    return np.linalg.norm(points - (a + s[:, None] * ab), axis=1)


def _proper_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


def _is_simple(poly: np.ndarray) -> bool:
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _proper_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return False
    return True


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= crosses & (x < x_at)
    return inside


def _polygon_base(extra: dict) -> dict:
    return {"model": "floor_plan_polygon_hypothesis", "units": "arbitrary", "metric_scale_source": None,
            "vertices_xy": None, "vertices_world": None, "edges": [], "area": None, "perimeter": None,
            "observed_wall_segments": [], "rejected_candidates": [], "incomplete_boundaries": [],
            "shape_gate_eligible": False,
            "limitations": [
                "Floor-plan polygon only; no ceiling height, openings, or metric scale.",
                "Requires an explicit structural-axis (up) hypothesis supplied from geometry; camera image-up and camera path are never valid sources, and none is inferred here.",
                "Boundary order comes from a bounded search over evidence-supported corners (up to 10 wall nodes); more walls, or equally supported layouts, stay incomplete.",
                "Inferred edges are continuations of observed planes and are labelled; incomplete results carry no polygon.",
                "Full-height furniture flush with the ceiling and hollow behind could still pass the wall tests.",
                "No simulator data, annotations, or ground truth are read."], **extra}


def estimate_polygon(plane_data: dict, points: np.ndarray, up: np.ndarray | None = None,
                     min_points: int = 500) -> dict:
    """Arbitrary floor-plan polygon from vertical plane candidates; `incomplete` when support is missing."""
    points = np.asarray(points, dtype=float)
    finite = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 and points.shape[1:] == (3,) else np.empty((0, 3))
    if len(finite) < min_points:
        return _polygon_base({"status": "insufficient_geometry", "point_count": int(len(finite)),
                              "required_point_count": int(min_points)})
    if up is None or not np.isfinite(np.asarray(up, dtype=float)).all() or np.linalg.norm(up) < 1e-9:
        return _polygon_base({"status": "incomplete", "point_count": int(len(finite)),
                              "incomplete_boundaries": [{"reason": "up_axis_unavailable"}]})
    up = _unit(np.asarray(up, dtype=float))
    seed_axis = np.array([1., 0., 0.]) if abs(up[0]) < .9 else np.array([0., 1., 0.])
    e1 = _unit(np.cross(up, seed_axis)); e2 = np.cross(up, e1)
    center = np.median(finite, axis=0)
    rel = finite - center
    xy = np.stack([rel @ e1, rel @ e2], axis=1); height = rel @ up
    extent = float(np.linalg.norm(np.percentile(finite, 95, axis=0) - np.percentile(finite, 5, axis=0)))
    tol, gap = TOL_FRACTION * extent, GAP_FRACTION * extent
    h_lo, h_hi = np.percentile(height, [5, 95]); vertical_extent = float(h_hi - h_lo)
    frame = {"point_count": int(len(finite)), "coordinate_origin_world": center.tolist(),
             "axes_world": {"e1": e1.tolist(), "e2": e2.tolist(), "up": up.tolist()},
             "vertical_extent_5_95": vertical_extent}
    if extent <= 0 or vertical_extent <= 1e-8:
        return _polygon_base({**frame, "status": "incomplete",
                              "incomplete_boundaries": [{"reason": "degenerate_point_cloud_extent"}]})

    # 1. Vertical plane candidates, refit as 2D lines from the points themselves.
    min_support = max(30, int(0.02 * len(finite)))
    lines = []
    planes = plane_data.get("planes", []) if isinstance(plane_data, dict) else []
    for index, plane in sorted(enumerate(planes), key=lambda item: -int(item[1].get("support_count", 0) or 0)):
        try:
            normal = _unit(np.asarray(plane["normal"], dtype=float)); offset = float(plane["offset"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(normal).all() and np.isfinite(offset)) or abs(float(normal @ up)) > VERTICAL_TILT:
            continue
        mask = np.abs(rel @ normal + (normal @ center + offset)) < tol
        if mask.sum() < min_support:
            continue
        m, c = _fit_line(xy[mask])
        for _ in range(2):
            mask = np.abs(xy @ m - c) < tol
            if mask.sum() < min_support:
                break
            m, c = _fit_line(xy[mask])
        if mask.sum() < min_support:
            continue
        lines.append({"plane_index": index, "m": m, "c": c, "support": int(mask.sum())})
    lines.sort(key=lambda line: -line["support"])
    merged = []
    for line in lines:
        duplicate = False
        for kept in merged:
            dot = float(line["m"] @ kept["m"])
            if abs(dot) > np.cos(np.radians(5)) and abs(line["c"] - np.sign(dot) * kept["c"]) < 2 * tol:
                duplicate = True; break
        if not duplicate:
            merged.append(line)

    # 2. Keep only structural walls; furniture planes are rejected whatever their support.
    walls, rejected = [], []
    for line in merged:
        m, c = line["m"], line["c"]
        signed = xy @ m - c
        on_line = np.abs(signed) < tol
        direction = np.array([-m[1], m[0]])
        # Floor/ceiling points that happen to lie on the line are not wall support: use the mid-height band.
        band = on_line & (height > h_lo + BAND_FRACTION * vertical_extent) & (height < h_hi - BAND_FRACTION * vertical_extent)
        t_on = xy[band] @ direction
        spans = _intervals(t_on, gap, 0.02 * extent)
        record = {"plane_index": line["plane_index"], "support_count": line["support"]}
        if not spans:
            rejected.append({**record, "reason": "no_continuous_support"}); continue
        t_lo, t_hi = spans[0][0], spans[-1][1]
        wall_h = height[on_line]; wall_lo, wall_hi = np.percentile(wall_h, [5, 95])
        if wall_hi - wall_lo < MIN_WALL_HEIGHT_FRACTION * vertical_extent:
            rejected.append({**record, "reason": "does_not_span_room_height",
                             "height_fraction": float((wall_hi - wall_lo) / vertical_extent)}); continue
        if wall_hi < h_hi - CEILING_REACH_FRACTION * vertical_extent:
            rejected.append({**record, "reason": "does_not_reach_ceiling_level"}); continue
        along = xy @ direction
        depth = BEHIND_DEPTH_FRACTION * extent
        # Each observed stretch of the line is judged on its own: interior and behind come from the shallow slab either
        # side of that stretch (across a notch or courtyard the far side of the plane can still hold other bars of the
        # plan), and collinear stretches may face opposite ways (S/Z layouts).
        kept = {+1: [], -1: []}
        for lo, hi, count in spans:
            margin = min(4 * tol, 0.25 * (hi - lo))                 # clear of the ends: re-entrant corners are not "behind"
            in_prism = (along >= lo + margin) & (along <= hi - margin)
            pos_mask, neg_mask = in_prism & (signed > 2 * tol) & (signed <= depth), in_prism & (signed < -2 * tol) & (signed >= -depth)
            positive, negative = int(pos_mask.sum()), int(neg_mask.sum())
            behind_mask = pos_mask if positive <= negative else neg_mask
            stretch_inliers = int((on_line & (along >= lo - tol) & (along <= hi + tol)).sum())
            behind = int(behind_mask.sum()) / max(1, stretch_inliers)
            # Surface behind a stretch means furniture in front of a wall only if the space behind is still room: floor
            # or ceiling points continue there. Behind a notch/alcove wall lies outside space with no floor or ceiling.
            in_band_b = (height[behind_mask] > h_lo + BAND_FRACTION * vertical_extent) & (height[behind_mask] < h_hi - BAND_FRACTION * vertical_extent)
            behind_horizontal = int((~in_band_b).sum())
            if behind > MAX_BEHIND_FRACTION and behind_horizontal >= BEHIND_ROOM_FRACTION * int(behind_mask.sum()):
                rejected.append({**record, "reason": "observed_surface_behind_patch", "behind_fraction": float(behind),
                                 "behind_floor_ceiling_points": behind_horizontal}); continue
            kept[+1 if negative >= positive else -1].append((lo, hi, count))   # +1: interior on the negative side already
        for orientation, stretches in kept.items():
            if not stretches:
                continue
            wm, wc, wd = (m, c, direction) if orientation > 0 else (-m, -c, np.array([m[1], -m[0]]))   # m is outward
            ivs = stretches if orientation > 0 else sorted((-hi, -lo, cnt) for lo, hi, cnt in stretches)
            mid = wc * wm + wd * (0.5 * (ivs[0][0] + ivs[-1][1]))
            walls.append({**record, "m": wm, "c": wc, "d": wd, "intervals": ivs,
                          "angle": float(np.arctan2(mid[1], mid[0]))})

    def segment_xy(wall, a, b):
        return [(wall["c"] * wall["m"] + wall["d"] * a).tolist(), (wall["c"] * wall["m"] + wall["d"] * b).tolist()]
    observed_walls = [{"plane_index": w["plane_index"], "support_count": w["support_count"],
                       "segments_xy": [segment_xy(w, a, b) for a, b, _ in w["intervals"]]} for w in walls]
    result_extra = {**frame, "observed_wall_segments": observed_walls, "rejected_candidates": rejected,
                    "wall_candidate_count": len(walls)}

    def incomplete(*reasons):
        return _polygon_base({**result_extra, "status": "incomplete", "incomplete_boundaries": list(reasons)})
    if len(walls) < 3:
        return incomplete({"reason": "fewer_than_three_supported_walls", "supported_walls": len(walls)})

    # 3. Boundary assembly: evidence-supported connectivity, not an angular sort. A directed link i -> j exists when
    # the lines of walls i and j meet at a corner that the observed extents support (the corner lies just beyond the
    # observed end of i and just before the observed start of j, within the reach/overshoot limits). Every closed
    # cycle through ALL wall nodes is a candidate layout; each is checked for edge support, simplicity,
    # counter-clockwise area and points outside. One surviving layout (or one clearly better supported) is accepted;
    # none or several near-equal layouts stay incomplete. No corner is invented and no hull or box is substituted.
    # A line observed in several separate stretches is tried both as one wall (a doorway or occlusion gap) and as
    # separate collinear walls (a notch), because both are legitimate boundaries (C, U, T, S layouts).
    reach, overshoot_limit = CORNER_REACH_FRACTION * extent, OVERSHOOT_FRACTION * extent

    def partitions(spans):
        """All ways to group consecutive observed stretches of one line into walls."""
        if len(spans) == 1:
            return [[spans]]
        out = []
        for cut in range(1, 1 << (len(spans) - 1)):
            groups, current = [], [spans[0]]
            for k in range(1, len(spans)):
                if cut >> (k - 1) & 1:
                    groups.append(current); current = [spans[k]]
                else:
                    current.append(spans[k])
            out.append(groups + [current])
        return [[spans]] + out

    def solve(nodes):
        n = len(nodes)

        def corner(i, j):
            a, b = nodes[i], nodes[j]
            det = float(a["m"][0] * b["m"][1] - a["m"][1] * b["m"][0])
            if abs(det) < PARALLEL_SIN:
                return None
            vertex = np.linalg.solve(np.array([a["m"], b["m"]]), np.array([a["c"], b["c"]]))
            end_i, start_j = a["intervals"][-1][1], b["intervals"][0][0]
            gap_i, gap_j = float(a["d"] @ vertex) - end_i, start_j - float(b["d"] @ vertex)
            if max(-gap_i, -gap_j) > overshoot_limit or max(gap_i, gap_j) > reach:
                return None
            return vertex, max(0.0, gap_i), max(0.0, gap_j)
        links = {(i, j): corner(i, j) for i in range(n) for j in range(n) if i != j}
        links = {k: v for k, v in links.items() if v is not None}
        successors = {i: [j for j in range(n) if (i, j) in links] for i in range(n)}
        cycles, budget = [], [SEARCH_STEP_BUDGET]

        def extend(path, seen):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            if len(path) == n:
                if (path[-1], path[0]) in links:
                    cycles.append(list(path))
                return
            for nxt in successors[path[-1]]:
                if nxt not in seen:
                    seen.add(nxt); path.append(nxt); extend(path, seen); path.pop(); seen.discard(nxt)
        extend([0], {0})

        def assemble(order):
            vertices = [links[(order[k], order[(k + 1) % n])][0] for k in range(n)]
            edges, problems, corner_gap = [], [], 0.0
            for k, wi in enumerate(order):
                wall = nodes[wi]; start, end = vertices[k - 1], vertices[k]
                t_start, t_end = float(wall["d"] @ start), float(wall["d"] @ end)
                a, b = wall["intervals"][0][0], wall["intervals"][-1][1]
                if t_end - t_start <= 1e-9:
                    problems.append({"reason": "misordered_or_degenerate_edge", "plane_index": wall["plane_index"]}); continue
                covered = sum(max(0.0, min(hi, t_end) - max(lo, t_start)) for lo, hi, _ in wall["intervals"])
                fraction = float(covered / (t_end - t_start))
                if fraction < MIN_EDGE_FRACTION:
                    problems.append({"reason": "edge_unsupported", "plane_index": wall["plane_index"], "observed_fraction": fraction}); continue
                corner_gap += max(0.0, a - t_start) + max(0.0, t_end - b)
                edges.append({"index": k, "plane_index": wall["plane_index"], "support_count": wall["support_count"],
                              "start_xy": start.tolist(), "end_xy": end.tolist(), "length": float(t_end - t_start),
                              "observed_fraction": fraction, "label": "observed" if fraction >= OBSERVED_EDGE_FRACTION else "inferred",
                              "observed_segments_xy": [segment_xy(wall, max(lo, t_start), min(hi, t_end))
                                                       for lo, hi, _ in wall["intervals"] if min(hi, t_end) > max(lo, t_start)],
                              "corner_gap_start": float(max(0.0, a - t_start)), "corner_gap_end": float(max(0.0, t_end - b))})
            poly = np.array(vertices)
            area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
            if not problems and (area <= 0 or not _is_simple(poly)):
                problems.append({"reason": "polygon_not_simple_counter_clockwise", "signed_area": area})
            outside_fraction = 0.0
            if not problems:
                outside = ~_inside(poly, xy)
                if outside.any():
                    boundary = np.min([_segment_distance(xy[outside], poly[k], poly[(k + 1) % n]) for k in range(n)], axis=0)
                    outside_fraction = float((boundary > 2 * tol).sum() / len(xy))
                if outside_fraction > MAX_OUTSIDE_FRACTION:
                    problems.append({"reason": "observed_points_outside_polygon_missing_boundary", "outside_fraction": outside_fraction})
            return {"order": order, "plane_order": [nodes[i]["plane_index"] for i in order], "poly": poly, "edges": edges,
                    "area": area, "outside_fraction": outside_fraction, "gap": corner_gap, "problems": problems}
        layouts = [assemble(c) for c in cycles]
        return {"nodes": nodes, "links": links, "successors": successors, "cycles": cycles, "layouts": layouts,
                "exhausted": budget[0] <= 0}

    variants = [[]]
    for w in walls:                                   # product over per-line groupings, bounded
        grouped = partitions(w["intervals"])
        variants = [v + [g] for v in variants for g in grouped][:MAX_GROUPING_VARIANTS]
    solved, nodes_seen = [], 0
    for v in variants:
        nodes = [{**w, "intervals": g} for w, groups in zip(walls, v) for g in groups]
        nodes_seen = max(nodes_seen, len(nodes))
        if len(nodes) > MAX_SEARCH_WALLS:
            continue
        solved.append(solve(nodes))
    if not solved:
        return incomplete({"reason": "too_many_wall_candidates_for_bounded_search", "wall_count": nodes_seen, "limit": MAX_SEARCH_WALLS})
    all_layouts = [l for s in solved for l in s["layouts"]]
    search = {"wall_lines": len(walls), "grouping_variants_searched": len(solved),
              "closed_cycles_through_all_walls": sum(len(s["cycles"]) for s in solved),
              "step_budget_exhausted": any(s["exhausted"] for s in solved)}
    if search["step_budget_exhausted"]:
        return incomplete({"reason": "boundary_search_budget_exhausted", **search})
    valid = sorted((l for l in all_layouts if not l["problems"]), key=lambda l: l["gap"])
    search["valid_layouts"] = len(valid)
    if not valid:
        if not all_layouts:
            s0 = solved[0]; nn = len(s0["nodes"])
            missing_out = [s0["nodes"][i]["plane_index"] for i in range(nn) if not s0["successors"][i]]
            missing_in = [s0["nodes"][j]["plane_index"] for j in range(nn) if not any((i, j) in s0["links"] for i in range(nn))]
            return incomplete({"reason": "no_closed_boundary_through_all_supported_walls",
                               "walls_without_supported_next_corner": missing_out,
                               "walls_without_supported_previous_corner": missing_in, **search})
        best = min(all_layouts, key=lambda l: (len(l["problems"]), l["gap"]))
        return incomplete(*best["problems"], {"reason": "closed_boundary_rejected_by_checks", **search})
    margin = max(0.03 * extent, 3 * tol)
    if len(valid) > 1 and valid[1]["gap"] - valid[0]["gap"] < margin:
        return incomplete({"reason": "ambiguous_boundary_order", "equally_supported_layouts": len(valid),
                           "plane_orders": [l["plane_order"] for l in valid[:4]],
                           "unobserved_corner_gap_totals": [l["gap"] for l in valid[:4]], "margin": float(margin), **search})
    chosen = valid[0]
    n = len(chosen["edges"])
    poly, edges, area, outside_fraction = chosen["poly"], chosen["edges"], chosen["area"], chosen["outside_fraction"]
    search["alternative_valid_layouts_rejected_on_evidence"] = len(valid) - 1
    floor_world = center + h_lo * up
    return _polygon_base({**result_extra, "status": "complete_polygon_hypothesis",
                          "vertices_xy": poly.tolist(),
                          "vertices_world": [(floor_world + v[0] * e1 + v[1] * e2).tolist() for v in poly],
                          "edges": edges, "area": area, "perimeter": float(sum(e["length"] for e in edges)),
                          "inferred_edge_count": sum(e["label"] == "inferred" for e in edges),
                          "outside_point_fraction": outside_fraction, "boundary_search": search, "shape_gate_eligible": True})


def _parse_up(text: str) -> np.ndarray:
    value = np.array([float(x) for x in text.replace(",", " ").split()], dtype=float)
    if value.shape != (3,):
        raise ValueError("--up needs three numbers")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path, help="generic RGB-derived prediction.npz")
    parser.add_argument("--planes", required=True, type=Path, help="matching plane-candidate JSON")
    parser.add_argument("--output", required=True, type=Path, help="new floor-plan polygon hypothesis JSON file")
    parser.add_argument("--up", type=_parse_up, help="declared up direction in the prediction frame, e.g. from camera motion")
    parser.add_argument("--cuboid-output", type=Path, help="optional separate file for the diagnostic cuboid baseline")
    args = parser.parse_args()
    for path in (args.output, args.cuboid_output):
        if path is not None and path.exists():
            parser.error(f"output already exists: {path}")
    try:
        plane_data = json.loads(args.planes.read_text(encoding="utf-8"))
        with np.load(args.prediction) as data:
            # Import the same RGB-only back-projection used by plane extraction.
            from extract_planes import points_from_prediction
            points = points_from_prediction(data)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(f"could not load generic prediction inputs: {exc}")
    outputs = [(args.output, estimate_polygon(plane_data, points, up=args.up))]
    if args.cuboid_output is not None:
        outputs.append((args.cuboid_output, estimate_cuboid_baseline(plane_data, points)))
    for path, result in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())