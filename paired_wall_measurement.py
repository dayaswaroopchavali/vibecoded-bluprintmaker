"""Estimate rectangular-room dimensions from paired observed wall planes.

The measurement uses only boundary line fits and their reported fit
uncertainties. It is intended as an alternate metric measurement for a
four-sided, near-rectangular boundary, not as a replacement for its vertices.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def _unit_line(line: dict) -> tuple[np.ndarray, float]:
    normal = np.asarray(line["normal_xy"], dtype=float).reshape(2)
    offset = float(line["offset"])
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal).all() or not math.isfinite(offset) or norm <= 1e-12:
        raise ValueError("invalid wall line")
    return normal / norm, offset / norm


def _separation(normal_a: np.ndarray, offset_a: float,
                normal_b: np.ndarray, offset_b: float) -> float:
    """Distance between two nearly parallel lines along their shared normal."""
    sign_b = 1.0 if float(normal_a @ normal_b) >= 0.0 else -1.0
    mean_normal = normal_a + sign_b * normal_b
    norm = float(np.linalg.norm(mean_normal))
    if norm <= 1e-12:
        raise ValueError("opposite wall normals do not define a shared direction")
    mean_normal /= norm
    # c*n is the closest point on n.x=c to the shared coordinate origin.
    point_delta = offset_a * normal_a - offset_b * normal_b
    return abs(float(mean_normal @ point_delta))


def _separation_sd(line_a: dict, line_b: dict,
                   normal_a: np.ndarray, offset_a: float,
                   normal_b: np.ndarray, offset_b: float) -> float | None:
    ua, ub = line_a.get("uncertainty") or {}, line_b.get("uncertainty") or {}
    try:
        sigmas = [float(ua["offset_sd"]), float(ub["offset_sd"]),
                  math.radians(float(ua["angle_sd_deg"])),
                  math.radians(float(ub["angle_sd_deg"]))]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(x) and x >= 0 for x in sigmas):
        return None

    # Numerically propagate the producer's independent offset/angle standard
    # deviations through the same projection used for the point estimate.
    theta_a = math.atan2(float(normal_a[1]), float(normal_a[0]))
    theta_b = math.atan2(float(normal_b[1]), float(normal_b[0]))

    def value(ca: float, cb: float, aa: float, ab: float) -> float:
        na = np.array([math.cos(aa), math.sin(aa)])
        nb = np.array([math.cos(ab), math.sin(ab)])
        return _separation(na, ca, nb, cb)

    values = [offset_a, offset_b, theta_a, theta_b]
    steps = [1e-5, 1e-5, 1e-6, 1e-6]
    gradient = []
    for i, step in enumerate(steps):
        lo, hi = values.copy(), values.copy()
        lo[i] -= step
        hi[i] += step
        gradient.append((value(*hi) - value(*lo)) / (2.0 * step))
    return float(math.sqrt(sum((g * s) ** 2 for g, s in zip(gradient, sigmas))))


def estimate_paired_wall_measurements(
    walls: dict,
    boundary: dict,
    *,
    max_opposite_angle_deg: float = 5.0,
    max_axis_departure_deg: float = 10.0,
) -> dict:
    """Return four edge lengths from two pairs of opposite observed planes.

    The pairing is selected by maximum absolute normal alignment. Results are
    refused unless the selected pairs are near parallel and their mean normals
    are near orthogonal, so non-rectangular boundaries keep their original
    vertex-derived measurements.
    """
    if boundary.get("status") != "complete_polygon_hypothesis":
        return {"status": "unsupported_boundary", "reason": "boundary_not_complete"}
    edges = boundary.get("edges") or []
    if len(edges) != 4:
        return {"status": "unsupported_shape", "reason": "requires_four_edges"}
    edge_ids = [int(edge["line_id"]) for edge in edges]
    if len(set(edge_ids)) != 4:
        return {"status": "unsupported_shape", "reason": "requires_four_distinct_wall_lines"}
    line_by_id = {int(line["id"]): line for line in walls.get("lines") or []}
    if any(line_id not in line_by_id for line_id in edge_ids):
        return {"status": "invalid_input", "reason": "boundary_line_missing"}
    try:
        normalized = {line_id: _unit_line(line_by_id[line_id]) for line_id in edge_ids}
    except (KeyError, TypeError, ValueError):
        return {"status": "invalid_input", "reason": "invalid_wall_line"}

    pairings = []
    for (i, j), (k, l) in (
        (((0, 1), (2, 3))),
        (((0, 2), (1, 3))),
        (((0, 3), (1, 2))),
    ):
        first = (edge_ids[i], edge_ids[j])
        second = (edge_ids[k], edge_ids[l])
        align_first = abs(float(normalized[first[0]][0] @ normalized[first[1]][0]))
        align_second = abs(float(normalized[second[0]][0] @ normalized[second[1]][0]))
        pairings.append((align_first + align_second, (first, second)))
    _, pairs = max(pairings, key=lambda item: item[0])
    min_alignment = math.cos(math.radians(max_opposite_angle_deg))
    pair_alignments = [abs(float(normalized[a][0] @ normalized[b][0])) for a, b in pairs]
    if min(pair_alignments) < min_alignment:
        return {"status": "unsupported_shape", "reason": "opposite_walls_not_parallel_enough",
                "pair_alignments": pair_alignments}
    mean_normals = []
    for a, b in pairs:
        na, _ = normalized[a]
        nb, _ = normalized[b]
        if float(na @ nb) < 0:
            nb = -nb
        mean_normals.append((na + nb) / np.linalg.norm(na + nb))
    cross_alignment = abs(float(mean_normals[0] @ mean_normals[1]))
    if cross_alignment > math.sin(math.radians(max_axis_departure_deg)):
        return {"status": "unsupported_shape", "reason": "wall_axes_not_orthogonal_enough",
                "pair_axis_alignment": cross_alignment}

    measurements = []
    for (line_a_id, line_b_id), pair_alignment in zip(pairs, pair_alignments):
        line_a, line_b = line_by_id[line_a_id], line_by_id[line_b_id]
        normal_a, offset_a = normalized[line_a_id]
        normal_b, offset_b = normalized[line_b_id]
        distance = _separation(normal_a, offset_a, normal_b, offset_b)
        measurements.append({
            "line_ids": [line_a_id, line_b_id],
            "parallel_alignment": pair_alignment,
            "separation_m": distance,
            "separation_sd_m": _separation_sd(line_a, line_b, normal_a, offset_a,
                                                normal_b, offset_b),
        })
    measurements.sort(key=lambda item: item["separation_m"])
    lengths = [measurement["separation_m"] for measurement in measurements for _ in range(2)]
    return {
        "status": "paired_plane_measurement",
        "method": "opposite_plane_offsets_projected_on_pair_mean_normal",
        "opposite_pairing": [measurement["line_ids"] for measurement in measurements],
        "opposite_pair_alignments": [measurement["parallel_alignment"] for measurement in measurements],
        "pair_axis_alignment_abs_dot": cross_alignment,
        "pair_measurements": measurements,
        "edge_lengths_m_sorted": lengths,
        "edge_length_sd_m_sorted": [measurement["separation_sd_m"] for measurement in measurements
                                    for _ in range(2)],
    }
