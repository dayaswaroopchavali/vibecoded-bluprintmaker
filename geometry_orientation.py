"""Gravity-free structural-axis hypotheses from observed plane candidates only.

A room's blueprint needs one distinguished axis (the "prism axis" the walls are parallel to). It is taken from the
geometry alone: a direction a is supported when most plane support is either perpendicular to a (wall-like) or
parallel to a (floor/ceiling-like), a parallel pair brackets the cloud, and the wall-like planes actually span that
pair. Nothing here reads camera image-up, camera path, camera poses, simulator data or ground truth, so captures may
roll, tilt and invert freely.

What the geometry can and cannot say is kept explicit:
 * The sign of an axis is never resolved (floor versus ceiling is a physical label, not a geometric fact).
 * A box-like room is symmetric: all three axes explain the planes equally, so they are reported as co-supported
   (status ambiguous_axes). Skew or pentagonal walls break the symmetry because only one direction is perpendicular
   to every wall normal.
 * Hypotheses are returned ranked with their evidence; a caller may run downstream stages once per co-supported
   hypothesis but must keep the label "hypothesis", never "gravity".
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

PARALLEL_COS = float(np.cos(np.radians(12)))   # |n.a| above this: plane is perpendicular to the axis (floor/ceiling-like)
PERPENDICULAR_SIN = float(np.sin(np.radians(12)))  # |n.a| below this: plane contains the axis (wall-like)
MIN_SUPPORT_FRACTION = 0.04                    # a plane must carry this share of total support to count as structure
CO_SUPPORT_MARGIN = 0.03                       # hypotheses this close to the best are equally supported
MIN_SPAN_FRACTION_OF_BRACKET = 0.6             # a wall must cover this much of the bracket separation along the axis
MAX_HYPOTHESES = 6
WELL_EXPLAINED = 0.9                           # share of structural support an axis must explain to be called supported


def _unit(v):
    v = np.asarray(v, dtype=float); n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _planes(plane_data):
    """Valid planes as (index, unit normal, offset normalised with it, support weight, record)."""
    out = []
    for i, p in enumerate(plane_data.get("planes", []) if isinstance(plane_data, dict) else []):
        try:
            raw = np.asarray(p["normal"], dtype=float); off = float(p["offset"]); w = max(0, int(p.get("support_count", 0) or 0))
            norm = float(np.linalg.norm(raw))
            if raw.shape != (3,) or not np.isfinite(raw).all() or not np.isfinite(off) or norm < 1e-10 or w <= 0:
                continue
            out.append((i, raw / norm, off / norm, w, p))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return out


def _candidate_axes(planes):
    """Plane normals and zone axes (perpendicular to two non-parallel normals), de-duplicated by direction."""
    axes = [n for _i, n, _o, _w, _p in planes]
    for (_, a, *_), (_, b, *_) in itertools.combinations(planes, 2):
        c = np.cross(a, b)
        if np.linalg.norm(c) > 0.3:       # normals more than ~17 degrees apart
            axes.append(_unit(c))
    kept = []
    for a in axes:
        if not any(abs(float(a @ k)) > np.cos(np.radians(3)) for k in kept):
            kept.append(a)
    return kept


def _axis_span(plane, axis):
    """Observed extent of a plane's surface patches along an axis (None without patch information)."""
    info = plane.get("surface_patches") if isinstance(plane, dict) else None
    try:
        u, v = (np.asarray(x, dtype=float) for x in info["axes_world"]); centre = np.asarray(plane["center"], dtype=float)
        spans = []
        for patch in info["patches"]:
            (u0, v0), (u1, v1) = patch["bounds_uv"]
            corners = np.asarray([centre + u * s + v * t for s, t in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]) @ axis
            spans.append(float(corners.max() - corners.min()))
        return max(spans) if spans else None
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def axis_hypotheses(plane_data: dict, points: np.ndarray | None = None) -> dict:
    """Rank structural-axis hypotheses from planes (and the cloud's centre) without any camera information."""
    planes = _planes(plane_data)
    base = {"model": "geometry_only_structural_axis_hypotheses", "units": "arbitrary", "gravity_known": False,
            "camera_orientation_used": False, "camera_path_used": False, "sign_resolved": False, "hypotheses": [],
            "limitations": ["The axis sign (floor vs ceiling) is never resolved from geometry.",
                            "A box-like room is symmetric: its axes stay co-supported until another cue exists.",
                            "A hypothesis is not gravity; do not report it as a known vertical.",
                            "Uses observed plane candidates only; furniture planes are not excluded here."]}
    total = sum(w for _i, _n, _o, w, _p in planes)
    structure = [pl for pl in planes if pl[3] >= MIN_SUPPORT_FRACTION * total] if total else []
    if len(structure) < 3:
        return {**base, "status": "insufficient_planes", "supported_plane_count": len(structure)}
    pts = np.asarray(points, dtype=float) if points is not None else np.empty((0, 3))
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.ndim == 2 and pts.shape[1:] == (3,) else np.empty((0, 3))
    middle = np.median(pts, axis=0) if len(pts) else np.zeros(3)
    structural_total = sum(pl[3] for pl in structure)
    scored = []
    for axis in _candidate_axes(structure):
        parallel, wall_like, other = [], [], []
        for pl in structure:
            c = abs(float(pl[1] @ axis))
            (parallel if c >= PARALLEL_COS else wall_like if c <= PERPENDICULAR_SIN else other).append(pl)
        explained = (sum(p[3] for p in parallel) + sum(p[3] for p in wall_like)) / structural_total
        # Parallel planes: positions along the axis through the cloud centre; a bracket needs one on each side.
        positions = sorted(((-(float(p[2]) + float(p[1] @ middle)) / float(p[1] @ axis)), p[3], p[0]) for p in parallel)
        bracket = None
        if len(positions) >= 2 and positions[-1][0] - positions[0][0] > 1e-9:
            ext = float(np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0))) if len(pts) else 0.0
            if ext == 0.0 or positions[-1][0] - positions[0][0] > 0.05 * ext:
                bracket = {"separation_unscaled": float(positions[-1][0] - positions[0][0]),
                           "plane_ids": [int(positions[0][2]), int(positions[-1][2])]}
        # Walls should span the bracket (floor to ceiling); unknown when patch info is missing.
        spans = [(_axis_span(p[4], axis), p[3]) for p in wall_like]
        known = [(s, w) for s, w in spans if s is not None]
        span_fraction = None
        if bracket is not None and known:
            span_fraction = sum(w for s, w in known if s >= MIN_SPAN_FRACTION_OF_BRACKET * bracket["separation_unscaled"]) / sum(w for _s, w in known)
        # Distinct wall azimuths: unsigned directions of the wall normals within the plane perpendicular to the axis.
        reference = _unit(np.cross(axis, [1, 0, 0] if abs(axis[0]) < .9 else [0, 1, 0])); other_axis = np.cross(axis, reference)
        azimuths = sorted(float(np.degrees(np.arctan2(p[1] @ other_axis, p[1] @ reference)) % 180) for p in wall_like)
        clusters = []
        for az in azimuths:
            if not clusters or az - clusters[-1][-1] > 8:
                clusters.append([az])
            else:
                clusters[-1].append(az)
        if len(clusters) > 1 and clusters[-1][-1] - clusters[0][0] > 172:        # wrap-around at 180 degrees
            clusters[0] = clusters.pop() + clusters[0]
        score = explained + (0.25 if bracket else 0.0) + (0.25 * span_fraction if span_fraction is not None else 0.0)
        scored.append({"axis_world": axis.tolist(), "alternative_sign_axis_world": (-axis).tolist(), "sign_resolved": False,
                       "explained_support_fraction": float(explained), "parallel_plane_ids": [int(p[0]) for p in parallel],
                       "wall_like_plane_ids": [int(p[0]) for p in wall_like], "unexplained_plane_ids": [int(p[0]) for p in other],
                       "bracketing_pair": bracket, "walls_span_bracket_fraction": span_fraction,
                       "wall_azimuth_count": len(clusters), "score": float(score)})
    # A usable hypothesis needs wall-like planes in at least two directions (a corner) and an explained majority.
    scored = [h for h in scored if h["wall_azimuth_count"] >= 2 and h["explained_support_fraction"] >= 0.5]
    if not scored:
        return {**base, "status": "no_supported_axis", "supported_plane_count": len(structure)}
    scored.sort(key=lambda h: -h["score"])
    best = scored[0]["score"]
    for rank, h in enumerate(scored[:MAX_HYPOTHESES]):
        h["rank"] = rank; h["co_supported_with_best"] = bool(best - h["score"] <= CO_SUPPORT_MARGIN)
    kept = scored[:MAX_HYPOTHESES]
    co = [h for h in kept if h["co_supported_with_best"]]
    # Axes within a few degrees of each other are one hypothesis (zone axes of near-parallel normals); report distinct ones.
    status = "single_axis_sign_ambiguous" if len(co) == 1 else "ambiguous_axes"
    top = kept[0]
    if top["bracketing_pair"] is None or top["explained_support_fraction"] < WELL_EXPLAINED:
        # No parallel pair bracketing the cloud, or too much structure unexplained: report, but do not call it resolved.
        status = "weakly_supported_axes"
    return {**base, "status": status, "supported_plane_count": len(structure), "hypotheses": kept,
            "co_supported_count": len(co), "best_axis_world": co[0]["axis_world"] if len(co) == 1 else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planes", type=Path, required=True); parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    points = None
    if args.prediction:
        from extract_planes import points_from_prediction
        with np.load(args.prediction) as data:
            points = points_from_prediction(data)
    result = axis_hypotheses(json.loads(args.planes.read_text(encoding="utf-8")), points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
