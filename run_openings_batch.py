"""Freeze public-depth opening candidates, then score against private scenes.

The evaluator maps predicted edges to simulator wall IDs using private truth;
this is explicitly an oracle diagnostic, not the final semantic-ID contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import detect_openings_depth


def _truth_wall_for_edge(boundary: dict, edge_index: int, scene: dict) -> int | None:
    xy = np.asarray(boundary["vertices_xy"], float)
    midpoint = (xy[edge_index] + xy[(edge_index + 1) % len(xy)]) / 2
    origin = np.asarray(boundary["coordinate_origin_world"], float)
    axes = boundary["axes_world"]
    world = origin + midpoint[0] * np.asarray(axes["e1"]) + midpoint[1] * np.asarray(axes["e2"])
    distances = [abs(world[1] - scene["length"] / 2),
                 abs(world[0] - scene["width"] / 2),
                 abs(world[1] + scene["length"] / 2),
                 abs(world[0] + scene["width"] / 2)]
    best = int(np.argmin(distances))
    return best if distances[best] < .10 else None


def _score(candidates: dict, boundary: dict | None, scene: dict) -> dict:
    truth = list(scene["openings"])
    detected = list(candidates.get("openings", []))
    mapping = {}
    if boundary is not None and boundary.get("status") == "complete_polygon_hypothesis":
        mapping = {i: _truth_wall_for_edge(boundary, i, scene)
                   for i in range(len(boundary["edges"]))}
    pairs = []
    for pi, p in enumerate(detected):
        tw = mapping.get(p["wall_index"])
        for ti, t in enumerate(truth):
            if tw == t["wall"] and abs(p["width_m"] - t["width"]) <= .02:
                pairs.append((abs(p["width_m"] - t["width"]), pi, ti))
    used_p, used_t, matches = set(), set(), []
    for error, pi, ti in sorted(pairs):
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            matches.append({"predicted_index": pi, "truth_index": ti,
                            "width_error_m": float(error)})
    missed = len(truth) - len(used_t)
    phantom = len(detected) - len(used_p)
    denominator = len(truth) + phantom
    return {"truth_openings": len(truth), "predicted_openings": len(detected),
            "matched_within_2cm": len(matches), "missed": missed, "phantom": phantom,
            "score": len(matches) / denominator if denominator else None,
            "matches": matches, "oracle_wall_mapping": mapping,
            "official_gate_pass": False,
            "assessment_85_percent_diagnostic_pass":
                denominator > 0 and len(matches) / denominator >= .85}


def run(batch_root: Path, frozen_axis: Path, output: Path,
        geometry_root: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    selections = json.loads(frozen_axis.read_text(encoding="utf-8"))["rows"]
    geometry_root = geometry_root or batch_root
    frozen_rows = []
    for selection in selections:
        house = selection["house_id"]
        path_index = int(selection["path_index"])
        folder = batch_root / house
        label = f"{house.replace('_', '')}_path{path_index}"
        root = geometry_root / house / f"evaluation_{path_index:02d}" / "frozen" / label
        choice = selection.get("hypothesis_index") if selection["status"] == "selected" else None
        boundary = root / f"hyp{choice}_boundary.json" if choice is not None else None
        if boundary is not None:
            result = detect_openings_depth.detect(folder / f"capture_{path_index:02d}" / "prediction.npz",
                                                   boundary, root / "axes.json", root / "planes.json")
        else:
            result = {"schema": "depth_wall_openings_v1", "status": "axis_refused", "openings": []}
        destination = output / f"{house}_path{path_index}_openings.json"
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        frozen_rows.append({"house_id": house, "path_index": path_index,
                            "hypothesis_index": choice, "opening_file": str(destination.resolve()),
                            "status": result["status"]})
        print(f"{house} path{path_index}: {len(result['openings'])} opening candidates", flush=True)
    frozen = {"schema": "frozen_openings_batch_v1", "private_truth_read": False,
              "rows": frozen_rows}
    (output / "frozen_index.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    # Private scenes are read only after every candidate list has been frozen.
    scored = []
    for row in frozen_rows:
        house, path_index = row["house_id"], row["path_index"]
        scene = json.loads((batch_root / house / "private_ground_truth" / "scene.json").read_text())
        candidates = json.loads(Path(row["opening_file"]).read_text())
        label = f"{house.replace('_', '')}_path{path_index}"
        boundary_path = (geometry_root / house / f"evaluation_{path_index:02d}" / "frozen" /
                         label / f"hyp{row['hypothesis_index']}_boundary.json")
        boundary = json.loads(boundary_path.read_text()) if row["hypothesis_index"] is not None else None
        scored.append({**row, **_score(candidates, boundary, scene)})
    report = {"schema": "opening_batch_oracle_diagnostic_v1", "runs": len(scored),
              "official_stage_completion": False,
              "per_capture_85_percent_diagnostic_passes": sum(x["assessment_85_percent_diagnostic_pass"] for x in scored),
              "total_truth_openings": sum(x["truth_openings"] for x in scored),
              "total_hits": sum(x["matched_within_2cm"] for x in scored),
              "total_missed": sum(x["missed"] for x in scored),
              "total_phantom": sum(x["phantom"] for x in scored),
              "rows": scored,
              "limitations": ["Ground-truth wall location is used only for evaluator pairing.",
                              "Depth dropouts are not semantically classified as doorway or window.",
                              "No official opening gate claim without stable wall IDs and all capture tiers."]}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--frozen-axis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--geometry-root", type=Path,
                   help="Frozen geometry root when it differs from the sensor batch root")
    a = p.parse_args()
    report = run(a.batch_root, a.frozen_axis, a.output, a.geometry_root)
    print(json.dumps({k: report[k] for k in ("runs", "per_capture_85_percent_diagnostic_passes",
                                           "total_truth_openings", "total_hits", "total_missed", "total_phantom")}, indent=2))


if __name__ == "__main__":
    main()
