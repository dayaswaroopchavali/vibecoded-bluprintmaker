"""RGB-only ablation: combine depth from one prediction with poses from another.

Both predictions must contain the same uniquely identified source frames. The
depth prediction's per-pixel intrinsics remain attached to its depth map; only
world-to-camera extrinsics are replaced. This does not estimate or correct a
possible scale difference between model runs, so it is a diagnostic baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import windowed_reconstruction as W


def run(depth_prediction: Path, pose_prediction: Path, output: Path, *, match_depth_scale: bool = False,
        intrinsics_source: str = "depth") -> dict:
    depth_prediction, pose_prediction, output = map(Path, (depth_prediction, pose_prediction, output))
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    with np.load(depth_prediction, allow_pickle=False) as data:
        fused = {key: data[key] for key in data.files}
    if intrinsics_source not in ("depth", "pose"):
        raise ValueError("intrinsics_source must be depth or pose")
    with np.load(pose_prediction, allow_pickle=False) as data:
        pose_indices = np.asarray(data["source_indices"], dtype=int)
        pose_extrinsics = np.asarray(data["extrinsics"])
        pose_intrinsics = np.asarray(data["intrinsics"]) if intrinsics_source == "pose" else None
        pose_depth = np.asarray(data["depth"]) if match_depth_scale else None
        pose_confidence = np.asarray(data["confidence"]) if match_depth_scale else None
    depth_indices = np.asarray(fused["source_indices"], dtype=int)
    if (len(set(depth_indices.tolist())) != len(depth_indices)
            or len(set(pose_indices.tolist())) != len(pose_indices)
            or set(depth_indices.tolist()) != set(pose_indices.tolist())):
        raise ValueError("predictions must contain the same unique source frames")
    if fused["depth"].shape[0] != len(depth_indices) or pose_extrinsics.shape != (len(pose_indices), 3, 4):
        raise ValueError("depth or pose rows do not match source indices")
    pose_row = {int(index): row for row, index in enumerate(pose_indices)}
    ordering = [pose_row[int(index)] for index in depth_indices]
    fused["extrinsics"] = pose_extrinsics[ordering]
    if intrinsics_source == "pose":
        fused["intrinsics"] = pose_intrinsics[ordering]
    calibration = []
    if match_depth_scale:
        if pose_depth.shape != fused["depth"].shape:
            raise ValueError("depth maps must have the same pixel grid for scale matching")
        scaled_depth = np.array(fused["depth"], copy=True)
        window_confidence = fused["confidence"]
        for row, source in enumerate(depth_indices):
            j = pose_row[int(source)]
            reference = pose_depth[j, ::8, ::8]
            local = scaled_depth[row, ::8, ::8]
            ref_conf = pose_confidence[j, ::8, ::8]
            local_conf = window_confidence[row, ::8, ::8]
            valid = (np.isfinite(reference) & np.isfinite(local) & (reference > 0) & (local > 0)
                     & (ref_conf > np.percentile(ref_conf, 50))
                     & (local_conf > np.percentile(local_conf, 50)))
            ratios = reference[valid] / local[valid]
            if len(ratios) < 100:
                raise ValueError(f"frame {source}: too few reliable shared depth pixels")
            scale = float(np.median(ratios))
            dispersion = float(np.median(np.abs(ratios - scale)) / scale)
            scaled_depth[row] *= scale
            calibration.append({"source_index": int(source), "scale": scale,
                                "relative_mad": dispersion, "support_pixels": int(len(ratios))})
        fused["depth"] = scaled_depth
    output.mkdir(parents=True)
    np.savez_compressed(output / "prediction.npz", **fused)
    manifest = {
        "kind": "rgb_pose_swap_ablation",
        "truth_free": True,
        "ground_truth_read": False,
        "depth_source": str(depth_prediction.resolve()),
        "depth_source_sha256": W.sha256(depth_prediction),
        "pose_source": str(pose_prediction.resolve()),
        "pose_source_sha256": W.sha256(pose_prediction),
        "source_frames": len(depth_indices),
        "intrinsics": f"{intrinsics_source} source, same pixel grid",
        "scale": "per-frame RGB-prediction depth ratio" if match_depth_scale else "unadjusted; model run scale consistency is unverified",
        "depth_scale_calibration": calibration,
        "prediction_sha256": W.sha256(output / "prediction.npz"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-prediction", type=Path, required=True)
    parser.add_argument("--pose-prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-depth-scale", action="store_true", help="match each depth map's scale to the pose prediction using shared high-confidence pixels")
    parser.add_argument("--intrinsics-source", choices=("depth", "pose"), default="depth")
    args = parser.parse_args()
    result = run(args.depth_prediction, args.pose_prediction, args.output,
                 match_depth_scale=args.match_depth_scale, intrinsics_source=args.intrinsics_source)
    print(json.dumps({key: result[key] for key in ("kind", "truth_free", "source_frames", "intrinsics", "scale", "prediction_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
