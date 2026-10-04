"""Truth-free robust edge refinement for depth-dropout opening candidates.

This module consumes only public depth predictions and frozen room geometry.
It reprojects depth-dropout rays onto the already selected wall planes, then
uses a per-frame consensus of observed aperture edges to reduce sensitivity to
one extreme ray. It does not inspect simulator scenes or ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _walls(boundary: dict) -> list[dict]:
    axes = boundary["axes_world"]
    origin = np.asarray(boundary["coordinate_origin_world"], dtype=float)
    e1, e2, up = (np.asarray(axes[k], dtype=float) for k in ("e1", "e2", "up"))
    vertices = np.asarray(boundary["vertices_xy"], dtype=float)
    result = []
    for i, xy in enumerate(vertices):
        a = origin + xy[0] * e1 + xy[1] * e2
        bxy = vertices[(i + 1) % len(vertices)]
        b = origin + bxy[0] * e1 + bxy[1] * e2
        tangent = b - a
        length = float(np.linalg.norm(tangent))
        tangent /= length
        normal = np.cross(tangent, up)
        normal /= np.linalg.norm(normal)
        result.append({"a": a, "tangent": tangent, "normal": normal,
                       "length": length})
    return result


def frame_edge_observations(prediction: Path, boundary: dict, opening: dict,
                            *, roi_margin_m: float = .10) -> list[tuple[float, float]]:
    """Return (left, right) dropout-ray bounds for each frame that sees the opening.

    The ROI is anchored to the public detector candidate with a fixed margin.
    Rays must be assigned to that wall as their nearest room wall and have
    missing/nonpositive public depth. Each frame contributes at most one pair,
    avoiding overweighting dense views.
    """
    walls = _walls(boundary)
    wi = int(opening["wall_index"])
    wall = walls[wi]
    u0, u1 = map(float, opening["u_interval_m"])
    v0, v1 = map(float, opening["v_interval_from_room_axis_m"])
    with np.load(prediction, allow_pickle=False) as data:
        depth = data["depth"]
        extrinsics = data["extrinsics"]
        intrinsics = data["intrinsics"]
        h, w = depth.shape[1:]
        yy, xx = np.mgrid[:h, :w]
        xx, yy = xx.ravel(), yy.ravel()
        observations = []
        for fi in range(depth.shape[0]):
            k = intrinsics[fi]
            rays_cam = np.stack(((xx-k[0, 2])/k[0, 0],
                                 (yy-k[1, 2])/k[1, 1], np.ones(len(xx))), axis=1)
            r = extrinsics[fi, :, :3]
            t = extrinsics[fi, :, 3]
            center = -r.T @ t
            rays = rays_cam @ r
            closest = np.full(len(xx), np.inf)
            closest_wall = np.full(len(xx), -1, dtype=np.int16)
            target_u = target_v = None
            for j, candidate in enumerate(walls):
                den = rays @ candidate["normal"]
                numerator = np.dot(candidate["a"] - center, candidate["normal"])
                z = np.divide(numerator, den, out=np.full(len(xx), np.inf), where=np.abs(den) > 1e-8)
                hit = center + rays * z[:, None]
                rel = hit - candidate["a"]
                u = rel @ candidate["tangent"]
                v = rel @ np.asarray(boundary["axes_world"]["up"], dtype=float)
                valid = ((z > .02) & (z < closest) & (u >= 0) & (u <= candidate["length"])
                         )
                closest[valid] = z[valid]
                closest_wall[valid] = j
                if j == wi:
                    target_u, target_v = u, v
            roi = ((closest_wall == wi) & (target_u >= u0-roi_margin_m)
                   & (target_u <= u1+roi_margin_m) & (target_v >= v0-roi_margin_m)
                   & (target_v <= v1+roi_margin_m))
            observed = depth[fi].ravel()
            holes = roi & (~np.isfinite(observed) | (observed <= 0))
            if np.count_nonzero(holes) >= 3:
                observations.append((float(np.min(target_u[holes])),
                                     float(np.max(target_u[holes]))))
    return observations


def consensus_interval(observations: list[tuple[float, float]], *, trim_fraction: float = .05,
                       minimum_frames: int = 3) -> tuple[float, float] | None:
    """Estimate edges from trimmed frame-level extrema; return None if unsupported."""
    if not 0 <= trim_fraction < .25:
        raise ValueError("trim_fraction must be in [0, .25)")
    if len(observations) < minimum_frames:
        return None
    bounds = np.asarray(observations, dtype=float)
    # Lower quantile of per-frame left edges and upper quantile of right edges
    # retain most aperture extent while suppressing isolated frame extremes.
    q = trim_fraction
    left = float(np.quantile(bounds[:, 0], q, method="linear"))
    right = float(np.quantile(bounds[:, 1], 1-q, method="linear"))
    return (left, right) if right > left else None


def refine(prediction: Path, boundary: dict, opening_doc: dict, *,
           trim_fraction: float = .05, roi_margin_m: float = .10) -> dict:
    """Return a separate refined document, preserving all original predictions."""
    result = json.loads(json.dumps(opening_doc))
    result["schema"] = "depth_wall_openings_robust_edges_v1"
    result["edge_refinement"] = {"method": "trimmed_per_frame_dropout_extrema",
                                 "trim_fraction": trim_fraction,
                                 "roi_margin_m": roi_margin_m,
                                 "truth_free": True}
    for item in result.get("openings", []):
        old = list(map(float, item["u_interval_m"]))
        obs = frame_edge_observations(prediction, boundary, item, roi_margin_m=roi_margin_m)
        interval = consensus_interval(obs, trim_fraction=trim_fraction)
        item["edge_refinement_diagnostic"] = {"frames_with_roi_dropout": len(obs),
                                              "original_u_interval_m": old,
                                              "refined": interval is not None}
        if interval is None:
            continue
        lo, hi = interval
        item["u_interval_m"] = [lo, hi]
        item["width_m"] = hi - lo
        item["measurement_interval_m"] = [max(0., hi-lo-.02), hi-lo+.02]
        item["edge_refinement_diagnostic"]["refined_u_interval_m"] = [lo, hi]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trim-fraction", type=float, default=.05)
    args = parser.parse_args()
    boundary = json.loads(args.boundary.read_text(encoding="utf-8"))
    openings = json.loads(args.openings.read_text(encoding="utf-8"))
    result = refine(args.prediction, boundary, openings, trim_fraction=args.trim_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
