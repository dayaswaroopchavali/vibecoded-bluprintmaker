"""Truth-free, guarded rectangle fit from a frozen four-wall hypothesis.

This is a measured output option for already rectangular rooms, never a shape
prior for arbitrary rooms. It refuses nonrectangular or weakly supported input.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def _dimension_for_edge(edge: dict, pairs: list[dict]) -> float | None:
    line_id = edge.get("line_id")
    if line_id is None:
        return None
    other = [p for p in pairs if line_id not in p.get("line_ids", [])]
    if len(other) != 1:
        return None
    value = float(other[0]["separation_m"])
    return value if math.isfinite(value) and value > 0 else None


def rectify(boundary: dict, paired: dict, openings: dict | None = None,
            *, max_corner_shift_m: float = .03,
            max_axis_deviation_deg: float = 2.0) -> dict:
    refusal = {"status": "refused", "reason": None, "truth_free": True,
               "ground_truth_read": False}
    if boundary.get("status") != "complete_polygon_hypothesis":
        return {**refusal, "reason": "incomplete_boundary"}
    edges = boundary.get("edges", [])
    if len(edges) != 4 or not boundary.get("shape_gate_eligible"):
        return {**refusal, "reason": "not_supported_four_edge_room"}
    if paired.get("status") != "paired_plane_measurement":
        return {**refusal, "reason": "missing_paired_plane_measurement"}
    alignments = paired.get("opposite_pair_alignments", [])
    if len(alignments) != 2 or min(alignments) < math.cos(math.radians(max_axis_deviation_deg)):
        return {**refusal, "reason": "opposite_walls_not_parallel_enough"}
    if paired.get("pair_axis_alignment_abs_dot", 1) > math.sin(math.radians(max_axis_deviation_deg)):
        return {**refusal, "reason": "wall_axes_not_perpendicular_enough"}
    pairs = paired.get("pair_measurements", [])
    if len(pairs) != 2:
        return {**refusal, "reason": "not_two_plane_pairs"}
    lengths = [_dimension_for_edge(edge, pairs) for edge in edges]
    if any(value is None for value in lengths):
        return {**refusal, "reason": "wall_plane_pair_mapping_missing"}
    if abs(lengths[0] - lengths[2]) > 1e-8 or abs(lengths[1] - lengths[3]) > 1e-8:
        return {**refusal, "reason": "opposite_dimensions_disagree"}
    vertices = np.asarray(boundary["vertices_xy"], dtype=float)
    if vertices.shape != (4, 2) or not np.isfinite(vertices).all():
        return {**refusal, "reason": "invalid_vertices"}
    steps = np.roll(vertices, -1, axis=0) - vertices
    if np.any(np.linalg.norm(steps, axis=1) < .01):
        return {**refusal, "reason": "degenerate_edge"}
    along = steps[0] - steps[2]
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    if np.dot(across, steps[1]) < 0:
        across = -across
    for step, axis in zip(steps, (along, across, -along, -across)):
        angle = math.degrees(math.acos(float(np.clip(np.dot(step / np.linalg.norm(step), axis), -1, 1))))
        if angle > max_axis_deviation_deg:
            return {**refusal, "reason": "edge_direction_not_rectangular", "maximum_axis_deviation_deg": angle}
    middle = vertices.mean(axis=0)
    half0, half1 = lengths[0] / 2, lengths[1] / 2
    new_vertices = np.stack((middle - half0 * along - half1 * across,
                             middle + half0 * along - half1 * across,
                             middle + half0 * along + half1 * across,
                             middle - half0 * along + half1 * across))
    shift = np.linalg.norm(new_vertices - vertices, axis=1)
    if float(shift.max()) > max_corner_shift_m:
        return {**refusal, "reason": "measured_rectangle_too_far_from_observed_corners",
                "max_corner_shift_m": float(shift.max())}
    new_edges = copy.deepcopy(edges)
    for i, edge in enumerate(new_edges):
        edge["start_xy"] = new_vertices[i].tolist()
        edge["end_xy"] = new_vertices[(i + 1) % 4].tolist()
        edge["length"] = lengths[i]
        edge["measurement_source"] = "perpendicular_paired_plane_separation"
    adjusted_openings = []
    for item in (openings or {}).get("openings", []):
        copy_item = copy.deepcopy(item)
        wall = int(item["wall_index"])
        old_tangent = steps[wall] / np.linalg.norm(steps[wall])
        new_tangent = (new_vertices[(wall + 1) % 4] - new_vertices[wall]) / lengths[wall]
        old_start = vertices[wall]
        copy_item["u_interval_m"] = [float(np.dot(old_start + float(u) * old_tangent - new_vertices[wall], new_tangent))
                                     for u in item["u_interval_m"]]
        copy_item["width_m"] = copy_item["u_interval_m"][1] - copy_item["u_interval_m"][0]
        copy_item["measurement_source"] = "original_dropout_projection_onto_rectified_edge"
        adjusted_openings.append(copy_item)
    new_boundary = copy.deepcopy(boundary)
    new_boundary["vertices_xy"] = new_vertices.tolist()
    new_boundary["edges"] = new_edges
    new_boundary["perimeter"] = 2 * (lengths[0] + lengths[1])
    new_boundary["area"] = lengths[0] * lengths[1]
    new_boundary["rectification"] = {"method": "guarded_paired_plane_rectangle",
                                      "max_corner_shift_m": float(shift.max()),
                                      "axis_deviation_limit_deg": max_axis_deviation_deg,
                                      "truth_free": True, "ground_truth_read": False}
    return {"status": "rectified", "boundary": new_boundary,
            "openings": adjusted_openings, "paired_edge_lengths_m": lengths,
            "max_corner_shift_m": float(shift.max()), "truth_free": True,
            "ground_truth_read": False}
