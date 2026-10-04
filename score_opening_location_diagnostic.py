"""Evaluator-only opening diagnostic with a physical-location check.

Predictions must already be frozen. Private scene geometry is read here only
to prevent a similarly sized opening elsewhere on a wall from scoring a hit.
This is not the assessment's stable-ID opening gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _predicted_world_interval(boundary: dict, opening: dict, wall_axis: int) -> tuple[float, float]:
    xy = np.asarray(boundary["vertices_xy"], float)
    idx = int(opening["wall_index"])
    origin = np.asarray(boundary["coordinate_origin_world"], float)
    axes = boundary["axes_world"]
    start = origin + xy[idx, 0] * np.asarray(axes["e1"]) + xy[idx, 1] * np.asarray(axes["e2"])
    end = origin + xy[(idx + 1) % len(xy), 0] * np.asarray(axes["e1"]) + xy[(idx + 1) % len(xy), 1] * np.asarray(axes["e2"])
    direction = (end - start) / np.linalg.norm(end - start)
    lo, hi = (float(v) for v in opening["u_interval_m"])
    return tuple(sorted((float(start[wall_axis] + lo * direction[wall_axis]),
                         float(start[wall_axis] + hi * direction[wall_axis]))))


def _truth_interval(opening: dict) -> tuple[float, float]:
    return (opening["center"] - opening["width"] / 2,
            opening["center"] + opening["width"] / 2)


def location_matches(pred: tuple[float, float], truth: tuple[float, float]) -> bool:
    overlap = max(0., min(pred[1], truth[1]) - max(pred[0], truth[0]))
    return overlap >= .5 * min(pred[1] - pred[0], truth[1] - truth[0])


def score(row: dict, batch_root: Path, geometry_root: Path | None = None) -> dict:
    house, path_index = row["house_id"], int(row["path_index"])
    scene = json.loads((batch_root / house / "private_ground_truth" / "scene.json").read_text())
    result = json.loads(Path(row["opening_file"]).read_text())
    label = f"{house.replace('_', '')}_path{path_index}"
    hypothesis = row.get("hypothesis_index")
    boundary = None
    if hypothesis is not None:
        path = (geometry_root or batch_root) / house / f"evaluation_{path_index:02d}" / "frozen" / label / f"hyp{hypothesis}_boundary.json"
        boundary = json.loads(path.read_text())
    edges = row.get("oracle_wall_mapping", {})
    candidates = []
    for pi, opening in enumerate(result.get("openings", [])):
        wall = edges.get(str(opening["wall_index"]))
        if wall is None or boundary is None:
            continue
        coordinate_axis = 0 if wall in (0, 2) else 1
        pred_interval = _predicted_world_interval(boundary, opening, coordinate_axis)
        for ti, target in enumerate(scene["openings"]):
            if int(target["wall"]) != wall or abs(opening["width_m"] - target["width"]) > .02:
                continue
            if location_matches(pred_interval, _truth_interval(target)):
                candidates.append((abs(opening["width_m"] - target["width"]), pi, ti))
    matched_p, matched_t = set(), set()
    for _, pi, ti in sorted(candidates):
        if pi not in matched_p and ti not in matched_t:
            matched_p.add(pi)
            matched_t.add(ti)
    truth_count = len(scene["openings"])
    phantom = len(result.get("openings", [])) - len(matched_p)
    denominator = truth_count + phantom
    return {"house_id": house, "path_index": path_index, "hits": len(matched_p),
            "missed": truth_count - len(matched_t), "phantom": phantom,
            "score": len(matched_p) / denominator if denominator else None,
            "diagnostic_pass": bool(denominator and len(matched_p) / denominator >= .85)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-evaluation", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.frozen_evaluation.read_text())["rows"]
    scored = [score(row, args.batch_root, args.geometry_root) for row in rows]
    report = {"schema": "opening_location_oracle_diagnostic_v1", "runs": len(scored),
              "diagnostic_passes": sum(row["diagnostic_pass"] for row in scored),
              "hits": sum(row["hits"] for row in scored),
              "missed": sum(row["missed"] for row in scored),
              "phantom": sum(row["phantom"] for row in scored),
              "rows": scored, "official_stage_completion": False,
              "pairing": "oracle wall location and 50% horizontal interval overlap, then <=2cm width"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("runs", "diagnostic_passes", "hits", "missed", "phantom")}, indent=2))


if __name__ == "__main__":
    main()
