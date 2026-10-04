"""Frozen simulated-LiDAR single-room batch, with explicit diagnostic gates.

The sensor exporter alone reads the private scene/path before prediction is
frozen. Each evaluator reads the private scene only after that capture exists.
The current wall-ID and opening contract is unsolved, so sorted wall matching
is labelled oracle diagnostic and this runner never declares stage completion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import capture_lidar_sim
import capture_parallax
import evaluate_rgb_boundary
import simulator


def _metric_diagnostic(scene: dict, frozen_dir: Path, hypothesis: int) -> dict:
    boundary = json.loads((frozen_dir / f"hyp{hypothesis}_boundary.json").read_text())
    axes = json.loads((frozen_dir / "axes.json").read_text())
    candidate = axes["hypotheses"][hypothesis]
    if boundary.get("status") != "complete_polygon_hypothesis":
        return {"status": "incomplete", "reason": boundary.get("failure_reasons", [])}
    height = (candidate.get("bracketing_pair") or {}).get("separation_unscaled")
    lengths = sorted(float(e["length"]) for e in boundary["edges"])
    if len(lengths) != 4 or height is None:
        return {"status": "incomplete", "reason": "missing four edges or height pair"}
    # This correspondence is evaluator-oracle sorting, never a reconstruction
    # wall ID or a scored official gate.
    truth = sorted([float(scene["width"])] * 2 + [float(scene["length"])] * 2)
    errors = [abs(a - b) for a, b in zip(lengths, truth)]
    rel = [e / t for e, t in zip(errors, truth)]
    height_error = abs(float(height) - float(scene["height"]))
    return {"status": "oracle_sorted_diagnostic", "predicted_wall_lengths_m": lengths,
            "true_wall_lengths_m": truth, "max_wall_error_m": max(errors),
            "max_wall_relative_error": max(rel), "predicted_height_m": float(height),
            "true_height_m": float(scene["height"]), "height_error_m": height_error,
            "diagnostic_3_percent_wall_pass": max(rel) <= .03,
            "diagnostic_1p5cm_height_pass": height_error <= .015,
            "official_gate_pass": False}


def run(output: Path, scene_seeds: list[int], frames: int = 120,
        capture_seed: int = 3101) -> dict:
    if len(set(scene_seeds)) != len(scene_seeds):
        raise ValueError("scene seeds must be distinct")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for house_seed in scene_seeds:
        scene = simulator.make_scene(house_seed)
        house = output / f"house_{house_seed}"
        private = house / "private_ground_truth"
        private.mkdir(parents=True)
        scene_file = private / "scene.json"
        scene_file.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        path_seeds = simulator.path_seeds(scene, capture_seed)
        for path_index, path_seed in enumerate(path_seeds):
            path = capture_parallax.make_wide_path(scene, frames, path_seed, path_index)
            path_file = private / f"path_{path_index:02d}.json"
            path_file.write_text(json.dumps({"poses": path}, indent=2), encoding="utf-8")
            capture_dir = house / f"capture_{path_index:02d}"
            capture_lidar_sim.capture(scene_file, path_file, capture_dir,
                                      seed=house_seed * 100 + path_index)
            eval_dir = house / f"evaluation_{path_index:02d}"
            result = evaluate_rgb_boundary.run(capture_dir / "prediction.npz", scene_file,
                                                frames, eval_dir, f"house{house_seed}_path{path_index}")
            frozen = eval_dir / "frozen" / f"house{house_seed}_path{path_index}"
            candidate_rows = []
            for row in result["table"]:
                h = int(row["hypothesis"])
                metric = _metric_diagnostic(scene, frozen, h) if row["status"] != "skipped_not_co_supported" else {"status": "not_co_supported"}
                candidate_rows.append({"hypothesis": h, "boundary_status": row["status"],
                                       "metric_diagnostic": metric})
            complete = [r for r in candidate_rows if r["metric_diagnostic"]["status"] == "oracle_sorted_diagnostic"]
            selected = complete[0]["hypothesis"] if len(complete) == 1 else None
            rows.append({"stage": "single_room", "house_id": f"house_{house_seed}",
                         "capture_seed": int(path_seed), "path_index": path_index,
                         "candidate_count": len(candidate_rows), "complete_candidate_count": len(complete),
                         "truth_free_unique_complete_candidate": selected,
                         "candidates": candidate_rows,
                         "official_gate_pass": False,
                         "capture": str(capture_dir.resolve()), "evaluation": str(eval_dir.resolve())})
            print(f"house {house_seed} path {path_index}: {len(complete)} complete hypotheses", flush=True)
    chosen = [r for r in rows if r["truth_free_unique_complete_candidate"] is not None]
    metric_ok = sum(all(c["metric_diagnostic"].get("diagnostic_3_percent_wall_pass") and
                        c["metric_diagnostic"].get("diagnostic_1p5cm_height_pass")
                        for c in r["candidates"] if c["hypothesis"] == r["truth_free_unique_complete_candidate"])
                    for r in chosen)
    report = {"schema": "single_room_lidar_batch_diagnostic_v1", "scene_seeds": scene_seeds,
              "paths_per_scene": 3, "runs": len(rows), "unique_complete_candidates": len(chosen),
              "unique_complete_candidates_passing_two_metric_diagnostics": metric_ok,
              "official_stage_completion": False,
              "official_blockers": ["wall semantic IDs/correspondence unverified",
                                    "opening detection and width gate not implemented",
                                    "multi-tier and whole-property gates not implemented"],
              "rows": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene-seeds", type=int, nargs="+", required=True)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--capture-seed", type=int, default=3101)
    a = p.parse_args()
    report = run(a.output, a.scene_seeds, a.frames, a.capture_seed)
    print(json.dumps({k: report[k] for k in ("runs", "unique_complete_candidates",
                                           "unique_complete_candidates_passing_two_metric_diagnostics")}, indent=2))


if __name__ == "__main__":
    main()
