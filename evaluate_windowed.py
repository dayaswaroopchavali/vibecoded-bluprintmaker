"""Evaluation-only scoring of windowed reconstructions against private renderer truth (never fed back to inference).

Reports, for a frozen windowed run and an optional frozen baseline prediction, the two pose metrics separately:
  * centre-aligned: one Umeyama similarity fitted to camera centres; rotation error is measured after that alignment;
  * rotation-aligned: one Kabsch rotation fitted to the camera rotations themselves;
plus retained capture coverage. Scores use the largest connected component; nothing is dropped to look better: the
fraction of capture frames in it, the component sizes and the isolated windows are reported alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_camera_trajectory as A


def score_prediction(extrinsics, source_indices, gt_poses):
    centres = A.camera_centers(extrinsics)
    gt_centres = np.asarray([gt_poses[int(i)]["position"] for i in source_indices], dtype=float)
    fit = A.fit_similarity(centres, gt_centres)
    err = np.linalg.norm(np.asarray(fit["aligned"]) - gt_centres, axis=1)
    extent = float(np.linalg.norm(np.ptp(gt_centres, axis=0)))
    ori = A.orientation_audit(extrinsics, gt_poses, np.asarray(source_indices), center_alignment_rotation=np.asarray(fit["rotation"]))
    return {"frames": int(len(source_indices)), "center_rmse_fraction_of_extent": float(np.sqrt((err ** 2).mean()) / extent),
            "center_aligned_rotation": {k: ori["center_umeyama_aligned"][k] for k in ("geodesic_median_deg", "geodesic_p90_deg", "geodesic_max_deg")},
            "rotation_aligned_rotation": {k: ori["rotation_kabsch_aligned"][k] for k in ("geodesic_median_deg", "geodesic_p90_deg", "geodesic_max_deg")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windowed", type=Path, required=True, help="windowed_reconstruction output folder")
    parser.add_argument("--ground-truth", type=Path, required=True, help="private_ground_truth/path_XX.json (evaluation only)")
    parser.add_argument("--baseline", type=Path, help="frozen baseline prediction.npz scored the same way")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))["poses"]
    manifest = json.loads((args.windowed / "manifest.json").read_text(encoding="utf-8"))
    with np.load(args.windowed / "prediction.npz") as data:
        ids, E, src = data["component_ids"], data["extrinsics"], data["source_indices"]
    big = int(np.bincount(ids).argmax()); keep = ids == big
    merge = manifest["merge"]
    result = {"windowed": {"folder": str(args.windowed), "largest_component": score_prediction(E[keep], src[keep], gt),
                           "retained_capture_fraction": merge["retained_capture_fraction"], "components": merge["components"],
                           "component_frame_counts": merge["component_frame_counts"], "isolated_windows": len(merge.get("isolated_windows", [])),
                           "links_ok": merge["links_ok"], "pairs_linked": merge.get("pairs_linked"),
                           "cycle_consistency": merge.get("cycle_consistency"), "frames_available": merge["frames_available"],
                           "runtime_seconds": manifest["elapsed_seconds"], "inference_seconds": float(sum(manifest["inference"]["seconds_per_window"])),
                           "gpu_peak_memory_bytes": manifest["inference"]["gpu_peak_memory_bytes"]}}
    if args.baseline:
        with np.load(args.baseline) as b:
            result["baseline"] = {"file": str(args.baseline), **score_prediction(b["extrinsics"], b["source_indices"], gt)}
    result["note"] = "evaluation only; truth never reached inference; scores use the largest component, retention reported separately"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result["windowed"].items() if k in ("largest_component", "retained_capture_fraction")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
