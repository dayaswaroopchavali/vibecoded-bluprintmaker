"""Infer wall apertures from public depth dropouts and fitted wall geometry.

No simulator scene, semantic surface labels or private blueprint is read. Each
ray is assigned to its nearest candidate room wall. Missing depth is evidence
for an aperture only when multiple frames see the same wall cell and nearby
cells are observed as wall. Occluding foreground depth is ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _basis(boundary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axes = boundary["axes_world"]
    return (np.asarray(boundary["coordinate_origin_world"], float),
            np.asarray(axes["e1"], float), np.asarray(axes["e2"], float),
            np.asarray(axes["up"], float))


def _supported_bounds(component_mask: np.ndarray, u_min: np.ndarray, u_max: np.ndarray,
                      v_min: np.ndarray, v_max: np.ndarray,
                      fallback: tuple[float, float, float, float]
                      ) -> tuple[float, float, float, float]:
    """Return continuous bounds from projected dropout rays in coarse cells.

    Component membership is deliberately decided on the coarse, frame-vote
    grid. The sub-cell extrema only refine its measurement; they cannot create
    a component or bridge unsupported cells.
    """
    supported = component_mask & np.isfinite(u_min) & np.isfinite(u_max)
    if not np.any(supported):
        return fallback
    return (float(np.min(u_min[supported])), float(np.max(u_max[supported])),
            float(np.min(v_min[supported])), float(np.max(v_max[supported])))


def matching_axis_hypothesis(candidates: list[dict], axis: np.ndarray) -> dict | None:
    """The candidate whose unsigned axis is closest to the boundary's own axis.

    The boundary was built from exactly one candidate, so the closest candidate
    (|cos| > 0.99) is it. Other near-parallel candidates (e.g. a slightly tilted
    duplicate) must not make the height pair 'ambiguous'; only an exact tie at
    the top between candidates with different bracketing pairs does.
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    scored = sorted(((abs(float(np.dot(axis, np.asarray(h["axis_world"], float)
                                       / np.linalg.norm(h["axis_world"])))), i) for i, h in enumerate(candidates)),
                    reverse=True)
    if not scored or scored[0][0] <= .99:
        return None
    best = candidates[scored[0][1]]
    if (len(scored) > 1 and scored[0][0] - scored[1][0] < 1e-6
            and candidates[scored[1][1]].get("bracketing_pair") != best.get("bracketing_pair")):
        return None
    return best


def detect(prediction: Path, boundary_path: Path, axes_path: Path,
           planes_path: Path, *, grid_m: float = .01) -> dict:
    import cv2

    if not .005 <= grid_m <= .05:
        raise ValueError("grid_m must be between 0.005 and 0.05")
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    if boundary.get("status") != "complete_polygon_hypothesis":
        return {"schema": "depth_wall_openings_v1", "status": "incomplete_boundary", "openings": []}
    axes_doc = json.loads(axes_path.read_text(encoding="utf-8"))
    planes_doc = json.loads(planes_path.read_text(encoding="utf-8"))
    candidate_axis = np.asarray(boundary["axes_world"]["up"], float)
    candidates = axes_doc["hypotheses"]
    match = matching_axis_hypothesis(candidates, candidate_axis)
    if match is None or not match.get("bracketing_pair"):
        return {"schema": "depth_wall_openings_v1", "status": "ambiguous_height_pair", "openings": []}
    cap_ids = match["bracketing_pair"]["plane_ids"]
    cap_centers = [np.asarray(planes_doc["planes"][int(i)]["center"], float) for i in cap_ids]
    origin, e1, e2, up = _basis(boundary)
    vertices = np.asarray(boundary["vertices_xy"], float)
    walls = []
    for i in range(len(vertices)):
        a = origin + vertices[i, 0] * e1 + vertices[i, 1] * e2
        b = origin + vertices[(i + 1) % len(vertices), 0] * e1 + vertices[(i + 1) % len(vertices), 1] * e2
        tangent = b - a
        length = float(np.linalg.norm(tangent))
        tangent /= length
        normal = np.cross(tangent, up)
        normal /= np.linalg.norm(normal)
        cap_v = sorted(float(np.dot(c - a, up)) for c in cap_centers)
        nu = int(np.ceil(length / grid_m)) + 1
        nv = int(np.ceil((cap_v[1] - cap_v[0]) / grid_m)) + 1
        walls.append({"a": a, "tangent": tangent, "normal": normal, "length": length,
                      "v_min": cap_v[0], "v_max": cap_v[1], "shape": (nv, nu),
                      "hole": np.zeros((nv, nu), np.uint32), "wall": np.zeros((nv, nu), np.uint32),
                      "hole_frames": np.zeros((nv, nu), np.uint16),
                      "wall_frames": np.zeros((nv, nu), np.uint16),
                      "hole_u_min": np.full((nv, nu), np.inf),
                      "hole_u_max": np.full((nv, nu), -np.inf),
                      "hole_v_min": np.full((nv, nu), np.inf),
                      "hole_v_max": np.full((nv, nu), -np.inf)})
    with np.load(prediction) as data:
        depth = data["depth"].copy()
        extrinsics = data["extrinsics"].copy()
        intrinsics = data["intrinsics"].copy()
    n, h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    xx, yy = xx.ravel(), yy.ravel()
    for fi in range(n):
        K = intrinsics[fi]
        rays_cam = np.stack(((xx - K[0, 2]) / K[0, 0],
                             (yy - K[1, 2]) / K[1, 1], np.ones(len(xx))), axis=1)
        R = extrinsics[fi, :, :3]
        t = extrinsics[fi, :, 3]
        center = -R.T @ t
        rays = rays_cam @ R
        closest = np.full(len(xx), np.inf)
        closest_wall = np.full(len(xx), -1, np.int16)
        closest_u = np.zeros(len(xx))
        closest_v = np.zeros(len(xx))
        for wi, wall in enumerate(walls):
            den = rays @ wall["normal"]
            numerator = np.dot(wall["a"] - center, wall["normal"])
            z = np.divide(numerator, den, out=np.full(len(xx), np.inf), where=np.abs(den) > 1e-8)
            with np.errstate(invalid="ignore"):
                hit = center + rays * z[:, None]
                relative = hit - wall["a"]
                u = relative @ wall["tangent"]
                v = relative @ up
            valid = ((z > .02) & (z < closest) & (u >= 0) & (u <= wall["length"]) &
                     (v >= wall["v_min"]) & (v <= wall["v_max"]))
            closest[valid], closest_wall[valid] = z[valid], wi
            closest_u[valid], closest_v[valid] = u[valid], v[valid]
        observed = depth[fi].ravel()
        for wi, wall in enumerate(walls):
            assigned = closest_wall == wi
            if not np.any(assigned):
                continue
            u = closest_u[assigned]
            v = closest_v[assigned]
            ui = np.clip((u / grid_m).astype(int), 0, wall["shape"][1] - 1)
            vi = np.clip(((v - wall["v_min"]) / grid_m).astype(int), 0, wall["shape"][0] - 1)
            index = vi * wall["shape"][1] + ui
            z = observed[assigned]
            wall_hit = np.isfinite(z) & (z > 0) & (np.abs(z - closest[assigned]) < .035)
            hole_hit = ~np.isfinite(z) | (z <= 0)
            for key, frame_key, hit_mask in (("wall", "wall_frames", wall_hit),
                                             ("hole", "hole_frames", hole_hit)):
                ids = index[hit_mask]
                if len(ids):
                    counts = np.bincount(ids, minlength=wall[key].size).reshape(wall["shape"])
                    wall[key] += counts.astype(np.uint32)
                    wall[frame_key].flat[np.unique(ids)] += 1
            # Keep the true projected locations of dropout rays in each coarse
            # cell. These refine component edges later without changing the
            # robust coarse-grid voting that preserves sparse support.
            if np.any(hole_hit):
                hole_ids = index[hole_hit]
                for suffix, values, op in (("u_min", u[hole_hit], np.minimum.at),
                                           ("u_max", u[hole_hit], np.maximum.at),
                                           ("v_min", v[hole_hit], np.minimum.at),
                                           ("v_max", v[hole_hit], np.maximum.at)):
                    op(wall[f"hole_{suffix}"].ravel(), hole_ids, values)
    openings = []
    diagnostics = []
    for wi, wall in enumerate(walls):
        hole, solid = wall["hole_frames"], wall["wall_frames"]
        mask = ((hole >= 3) & (hole > 2 * solid)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = []
        for component in range(1, count):
            x, y, width, height, area = map(int, stats[component])
            if area * grid_m * grid_m < .08 or width * grid_m < .2 or height * grid_m < .2:
                continue
            component_mask = labels == component
            coarse_bounds = (x * grid_m, min(wall["length"], (x + width) * grid_m),
                             wall["v_min"] + y * grid_m,
                             min(wall["v_max"], wall["v_min"] + (y + height) * grid_m))
            u_min, u_max, v_min, v_max = _supported_bounds(
                component_mask, wall["hole_u_min"], wall["hole_u_max"],
                wall["hole_v_min"], wall["hole_v_max"], coarse_bounds)
            # A dropout ray has a finite pixel footprint. Extrema of all
            # dropout rays systematically extend an aperture by about one
            # grid cell in total; use the center of each edge cell instead.
            if u_max - u_min > grid_m:
                u_min += grid_m / 2
                u_max -= grid_m / 2
            frames = int(np.max(hole[component_mask]))
            item = {"wall_index": wi, "u_interval_m": [u_min, u_max],
                    "v_interval_from_room_axis_m": [v_min, v_max],
                    "width_m": u_max - u_min, "height_m": v_max - v_min,
                    "supporting_frame_count_at_best_cell": frames,
                    "type": "unknown_without_floor_sign_or_semantics",
                    "measurement_interval_m": [max(0, u_max - u_min - 2 * grid_m),
                                               u_max - u_min + 2 * grid_m]}
            components.append(item)
            openings.append(item)
        diagnostics.append({"wall_index": wi, "hole_cells": int(mask.sum()),
                            "wall_cells": int((solid > 0).sum()),
                            "component_count": len(components)})
    return {"schema": "depth_wall_openings_v1", "status": "candidates",
            "grid_m": grid_m, "input_frames": n, "wall_count": len(walls),
            "openings": openings, "diagnostics": diagnostics,
            "truth_free": True, "ground_truth_read": False,
            "limitations": ["Depth dropouts may be sensor failures, glass or openings; RGB semantics unresolved.",
                            "Axis sign is unresolved, so doorway/window labels are unknown.",
                            "Only complete supported room polygons can be processed."]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--boundary", type=Path, required=True)
    p.add_argument("--axes", type=Path, required=True)
    p.add_argument("--planes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid-m", type=float, default=.01)
    a = p.parse_args()
    result = detect(a.prediction, a.boundary, a.axes, a.planes, grid_m=a.grid_m)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "openings": result["openings"]}, indent=2))


if __name__ == "__main__":
    main()
