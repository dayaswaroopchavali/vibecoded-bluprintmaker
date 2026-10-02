"""RGB-derived, arbitrary-unit floor/ceiling separation estimate.

This module only pairs observed plane candidates using an explicitly declared
up direction and RGB-derived camera centers. It does not use scene dimensions,
simulator poses, or metric calibration. Incomplete and ambiguous evidence is
reported as such instead of selecting a convenient pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _unit(value):
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-10 else np.zeros(3, dtype=float)


def camera_centers(extrinsics):
    """Convert DA3 world-to-camera poses into camera centers in prediction frame."""
    poses = np.asarray(extrinsics, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics must have shape Nx3x4 or Nx4x4")
    poses = poses[:, :3, :]
    rotations, translations = poses[:, :, :3], poses[:, :, 3]
    centers = -np.einsum("nji,nj->ni", rotations, translations)
    valid = np.isfinite(centers).all(axis=1)
    return centers[valid]


def _plane_record(plane, index, up, centers, cloud_extent):
    try:
        raw_normal = np.asarray(plane["normal"], dtype=float)
        raw_offset = float(plane["offset"])
        normal_length = float(np.linalg.norm(raw_normal))
        if raw_normal.shape != (3,) or not np.isfinite(raw_normal).all() or normal_length < 1e-10:
            return None
        # Plane equation n.x + d=0 is scale-invariant only when d is
        # normalized with n as well.
        normal = raw_normal / normal_length
        offset = raw_offset / normal_length
        support_count = int(plane.get("support_count", 0))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(normal).all() or np.linalg.norm(normal) < .9 or not np.isfinite(offset):
        return None
    alignment = abs(float(normal @ up))
    if alignment < np.cos(np.radians(12)):
        return None
    # Plane equation is n.x + d = 0. Orient it toward declared up.
    if normal @ up < 0:
        normal, offset = -normal, -offset
    up_component = float(normal @ up)
    if abs(up_component) < 1e-8:
        return None
    camera_offsets = -(centers @ normal + offset) / up_component
    # Patch area and camera-in-patch support help reject small horizontal furniture.
    patch_areas, camera_coverages = [], []
    patches = plane.get("surface_patches", {})
    axes = np.asarray(patches.get("axes_world", []), dtype=float)
    if axes.shape == (2, 3):
        try:
            patch_origin = np.asarray(plane.get("center", -offset * normal), dtype=float)
            if patch_origin.shape != (3,) or not np.isfinite(patch_origin).all():
                patch_origin = -offset * normal
        except (TypeError, ValueError):
            patch_origin = -offset * normal
        patch_uv = (centers - patch_origin) @ axes.T
        for patch in patches.get("patches", []):
            try:
                bounds = np.asarray(patch["bounds_uv"], dtype=float)
                if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
                    continue
                spans = np.maximum(0., bounds[1] - bounds[0])
                patch_areas.append(float(spans[0] * spans[1]))
                inside = np.all((patch_uv >= bounds[0] - .03 * cloud_extent) &
                                (patch_uv <= bounds[1] + .03 * cloud_extent), axis=1)
                camera_coverages.append(float(inside.mean()) if len(inside) else 0.)
            except (KeyError, TypeError, ValueError):
                continue
    largest_area = max(patch_areas, default=0.)
    footprint = cloud_extent ** 2
    area_fraction = largest_area / max(footprint, 1e-12)
    cam_coverage = max(camera_coverages, default=0.)
    return {"plane_id": int(index), "normal_world": normal.tolist(), "offset": offset,
            "camera_vertical_intersections_unscaled": camera_offsets.tolist(),
            "normal_up_alignment": alignment,
            "support_count": support_count, "largest_patch_area": largest_area,
            "patch_area_fraction_of_cloud_footprint": float(area_fraction),
            "camera_centers_in_patch_fraction": float(cam_coverage)}


def estimate_height(plane_data, points, up, centers, *, min_area_fraction=.12,
                    min_camera_coverage=.5, min_support_fraction=.04):
    """Pair broad observed planes that bracket camera centers along declared up.

    The input cloud is used only for robust scene extent and support proportions.
    `up` must be declared by the caller; it is never inferred from plane labels.
    """
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    up = _unit(up)
    finite = points[np.isfinite(points).all(axis=1)] if points.ndim == 2 and points.shape[1:] == (3,) else np.empty((0, 3))
    centers = centers[np.isfinite(centers).all(axis=1)] if centers.ndim == 2 and centers.shape[1:] == (3,) else np.empty((0, 3))
    base = {"model": "rgb_derived_floor_ceiling_separation", "units": "arbitrary",
            "metric_scale_source": None, "scale_applied": False,
            "status": "incomplete", "floor_plane_id": None, "ceiling_plane_id": None,
            "separation_unscaled": None, "support_evidence": {}, "reasons": [],
            "limitations": ["Not a metric height estimate; no scale calibration is used.",
                            "Observed horizontal furniture may remain indistinguishable from structure.",
                            "The declared up direction is a prior, not independent validation of orientation."]}
    if np.linalg.norm(up) < .9:
        return {**base, "reasons": ["declared up direction is missing or degenerate"]}
    if len(finite) < 20 or len(centers) < 1:
        return {**base, "reasons": ["insufficient RGB-derived points or camera centers"]}
    spread = np.percentile(finite, 95, axis=0) - np.percentile(finite, 5, axis=0)
    cloud_extent = float(np.linalg.norm(spread))
    # A characteristic horizontal footprint length; this remains scale-equivariant.
    horiz = finite - np.outer(finite @ up, up)
    footprint_extent = float(np.linalg.norm(np.percentile(horiz, 95, axis=0) - np.percentile(horiz, 5, axis=0)))
    cloud_extent = max(footprint_extent, cloud_extent * .25, 1e-12)
    all_planes = plane_data.get("planes", []) if isinstance(plane_data, dict) else []
    support_total = sum(max(0, int(p.get("support_count", 0) or 0)) for p in all_planes if isinstance(p, dict))
    records = []
    for i, plane in enumerate(all_planes):
        if isinstance(plane, dict):
            rec = _plane_record(plane, i, up, centers, cloud_extent)
            if rec is not None:
                rec["support_fraction_of_candidates"] = rec["support_count"] / max(1, support_total)
                rec["support_accepted"] = bool(rec["patch_area_fraction_of_cloud_footprint"] >= min_area_fraction and
                                                rec["camera_centers_in_patch_fraction"] >= min_camera_coverage and
                                                rec["support_fraction_of_candidates"] >= min_support_fraction)
                records.append(rec)
    candidates = [r for r in records if r["support_accepted"]]
    if len(centers):
        ch = centers @ up
        c_lo, c_hi = float(ch.min()), float(ch.max())
    else:
        c_lo = c_hi = 0.
    # Require each plane to lie beyond all observed camera centers. The margin
    # guards against a camera sitting on a candidate surface due to noise.
    margin = .015 * cloud_extent
    floors = [p for p in candidates if max(p["camera_vertical_intersections_unscaled"]) < -margin]
    ceilings = [p for p in candidates if min(p["camera_vertical_intersections_unscaled"]) > margin]
    pairs = []
    for floor in floors:
        for ceiling in ceilings:
            floor_offsets = np.asarray(floor["camera_vertical_intersections_unscaled"])
            ceiling_offsets = np.asarray(ceiling["camera_vertical_intersections_unscaled"])
            if floor_offsets.shape != ceiling_offsets.shape or not len(floor_offsets):
                continue
            local_separations = ceiling_offsets - floor_offsets
            separation = float(np.median(local_separations))
            if separation <= .05 * cloud_extent or np.any(local_separations <= 0):
                continue
            angle = float(np.degrees(np.arccos(np.clip(np.dot(floor["normal_world"],
                                                                  ceiling["normal_world"]), -1., 1.))))
            separation_range = float(np.max(local_separations) - np.min(local_separations))
            # Prefer broad, well-supported candidates; retain a ranking so near ties can be refused.
            score = (floor["patch_area_fraction_of_cloud_footprint"] + ceiling["patch_area_fraction_of_cloud_footprint"]
                     + floor["camera_centers_in_patch_fraction"] + ceiling["camera_centers_in_patch_fraction"]
                     + floor["support_fraction_of_candidates"] + ceiling["support_fraction_of_candidates"])
            pairs.append((score, floor, ceiling, separation, separation_range, angle))
    evidence = {"declared_up_world": up.tolist(), "camera_center_count": len(centers),
                "camera_height_interval_unscaled": [c_lo, c_hi], "candidate_planes": records,
                "support_thresholds": {"minimum_patch_area_fraction": min_area_fraction,
                                       "minimum_camera_coverage": min_camera_coverage,
                                       "minimum_candidate_support_fraction": min_support_fraction,
                                       "camera_bracketing_margin_fraction": .015}}
    if not pairs:
        reasons = []
        if not candidates:
            reasons.append("no broad horizontal plane has adequate patch and camera-path support")
        else:
            if not floors:
                reasons.append("no supported floor candidate lies below all camera centers")
            if not ceilings:
                reasons.append("no supported ceiling candidate lies above all camera centers")
        return {**base, "support_evidence": evidence, "reasons": reasons or ["no plausible bracketing pair"]}
    pairs.sort(key=lambda p: p[0], reverse=True)
    best = pairs[0]
    if len(pairs) > 1 and best[0] - pairs[1][0] < max(.08, abs(best[0]) * .04):
        return {**base, "status": "ambiguous", "support_evidence": evidence,
                "reasons": ["multiple floor/ceiling pairs have comparable spatial support"],
                "competing_pairs": [{"floor_plane_id": int(p[1]["plane_id"]),
                                     "ceiling_plane_id": int(p[2]["plane_id"]),
                                     "separation_unscaled": float(p[3]), "support_score": float(p[0])}
                                    for p in pairs[:5]]}
    _, floor, ceiling, separation, separation_range, angle = best
    if angle > 2.0 or separation_range > max(.02 * separation, 2 * margin):
        return {**base, "status": "ambiguous", "floor_plane_id": int(floor["plane_id"]),
                "ceiling_plane_id": int(ceiling["plane_id"]),
                "support_evidence": evidence,
                "local_separation_median_unscaled": separation,
                "local_separation_range_unscaled": [separation - separation_range / 2,
                                                    separation + separation_range / 2],
                "pair_normal_angle_degrees": angle,
                "reasons": ["candidate planes are not parallel enough for one room-height value" if angle > 2.0
                            else "local plane separation varies materially across camera positions"]}
    return {**base, "status": "estimated_unscaled_separation",
            "floor_plane_id": int(floor["plane_id"]), "ceiling_plane_id": int(ceiling["plane_id"]),
            "separation_unscaled": separation, "support_evidence": evidence,
            "local_separation_range_unscaled": [separation - separation_range / 2,
                                                separation + separation_range / 2],
            "pair_normal_angle_degrees": angle,
            "reasons": ["selected supported near-parallel planes bracketing all supplied camera centers"],
            "pair_support_score": float(best[0])}


def _parse_up(text):
    value = np.array([float(v) for v in text.replace(",", " ").split()], dtype=float)
    if value.shape != (3,):
        raise argparse.ArgumentTypeError("--up requires three components")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--up", type=_parse_up, required=True,
                        help="declared up prior in prediction coordinates; e.g. 0 0 1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from extract_planes import points_from_prediction
    with np.load(args.prediction) as prediction:
        points = points_from_prediction(prediction)
        centers = camera_centers(prediction["extrinsics"])
    result = estimate_height(json.loads(args.planes.read_text(encoding="utf-8")), points, args.up, centers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
