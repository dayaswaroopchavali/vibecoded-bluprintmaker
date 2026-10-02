"""Evaluation-only similarity audit of frozen RGB camera-center trajectories.

The fitted transform is an oracle diagnostic for locating upstream pose errors. It
must never be exported or supplied to reconstruction stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_centers(extrinsics):
    """Convert world-to-camera 3x4/4x4 poses to centers: C = -R.T @ t."""
    poses = np.asarray(extrinsics, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("extrinsics must have shape Nx3x4 or Nx4x4")
    rotations, translations = poses[:, :3, :3], poses[:, :3, 3]
    if not np.isfinite(poses).all():
        raise ValueError("extrinsics must be finite")
    if not np.allclose(np.einsum("nji,njk->nik", rotations, rotations), np.eye(3), atol=2e-2):
        raise ValueError("extrinsic rotations must be orthonormal")
    determinants = np.linalg.det(rotations)
    if np.any(determinants < 0.999) or np.any(determinants > 1.001):
        raise ValueError("extrinsic rotations must be proper rotations (determinant +1)")
    return -np.einsum("nji,nj->ni", rotations, translations)


def fit_similarity(source, target):
    """Fit proper rotation, positive uniform scale, and translation by Umeyama LS."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("paired trajectories must be matching Nx3 arrays with N >= 3")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("trajectory coordinates must be finite")
    xc, yc = x.mean(axis=0), y.mean(axis=0)
    xs, ys = x - xc, y - yc
    singular = np.linalg.svd(xs, compute_uv=False)
    target_singular = np.linalg.svd(ys, compute_uv=False)
    if singular[0] <= 1e-10 or singular[1] <= singular[0] * 1e-7:
        raise ValueError("predicted trajectory is degenerate (collinear or coincident)")
    if target_singular[0] <= 1e-10 or target_singular[1] <= target_singular[0] * 1e-7:
        raise ValueError("ground-truth trajectory is degenerate (collinear or coincident)")
    covariance = ys.T @ xs / len(x)
    u, values, vt = np.linalg.svd(covariance)
    covariance_condition = float(values[-1] / max(values[0], 1e-15))
    sign = np.linalg.det(u @ vt)
    correction = np.diag([1.0, 1.0, 1.0 if sign >= 0 else -1.0])
    rotation = u @ correction @ vt
    variance = float(np.sum(xs * xs) / len(x))
    scale = float(np.sum(values * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0 or np.linalg.det(rotation) < 0.999:
        raise ValueError("could not fit a positive-scale proper rotation")
    translation = yc - scale * (rotation @ xc)
    aligned = scale * (x @ rotation.T) + translation
    residual = aligned - y
    extent = float(np.linalg.norm(np.ptp(y, axis=0)))
    if extent <= 1e-10:
        raise ValueError("ground-truth trajectory extent is degenerate")
    return {"aligned": aligned, "rotation": rotation, "scale": scale, "translation": translation,
            "rmse": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
            "max_error": float(np.max(np.linalg.norm(residual, axis=1))), "gt_extent": extent,
            "position_singular_values": singular.tolist(), "alignment_covariance_singular_values": values.tolist(),
            "alignment_covariance_condition_ratio": covariance_condition,
            "alignment_rotation_well_conditioned": covariance_condition > 1e-3}


def gt_opencv_c2w(pose):
    """GT camera-to-world rotation in the OpenCV camera frame (x right, y down, z forward).

    Uses simulator.camera_basis (incl. roll, inverted views) so the audit can never drift from the renderer.
    """
    from simulator import camera_basis
    right, up, forward = camera_basis(pose)
    return np.column_stack((right, -up, forward))


def to_world_to_camera_rotations(poses, pose_kind="world_to_camera"):
    """Return Nx3x3 world-to-camera rotations from w2c or c2w pose arrays (Nx3x4 / Nx4x4)."""
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] not in ((3, 3), (3, 4), (4, 4)):
        raise ValueError("poses must have shape Nx3x3, Nx3x4 or Nx4x4")
    if pose_kind not in ("world_to_camera", "camera_to_world"):
        raise ValueError("pose_kind must be world_to_camera or camera_to_world")
    rot = arr[:, :3, :3]
    return rot if pose_kind == "world_to_camera" else np.transpose(rot, (0, 2, 1))


def geodesic_deg(relative):
    """Rotation angle (deg) of each 3x3 relative rotation; atan2 form stays accurate near 0 and 180."""
    skew = np.stack([relative[:, 2, 1] - relative[:, 1, 2], relative[:, 0, 2] - relative[:, 2, 0],
                     relative[:, 1, 0] - relative[:, 0, 1]], axis=1) / 2
    cosine = (np.trace(relative, axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arctan2(np.linalg.norm(skew, axis=1), cosine))


def gt_c2w_stack(gt_poses, indices):
    return np.asarray([gt_opencv_c2w(gt_poses[int(i)]) for i in indices])


def fit_rotation_alignment(pred_w2c_rotations, gt_c2w):
    """Kabsch: single world rotation Q minimizing sum ||gt_c2w_i - Q @ pred_c2w_i||_F (Q proper)."""
    pred_c2w = np.transpose(np.asarray(pred_w2c_rotations, float), (0, 2, 1))
    m = np.einsum("nij,nkj->ik", gt_c2w, pred_c2w)
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt)) or 1.0]) @ vt


def rotation_angles_deg(predicted_extrinsics, gt_poses, indices, alignment_rotation, pose_kind="world_to_camera"):
    """Per-frame full-rotation errors after ONE global world alignment rotation (evaluation only)."""
    pred_w2c = to_world_to_camera_rotations(predicted_extrinsics, pose_kind)
    pred_c2w_aligned = np.einsum("ij,njk->nik", np.asarray(alignment_rotation, float), np.transpose(pred_w2c, (0, 2, 1)))
    gt_c2w = gt_c2w_stack(gt_poses, indices)
    relative = np.einsum("nji,njk->nik", gt_c2w, pred_c2w_aligned)
    angle = geodesic_deg(relative)  # full SO(3) geodesic: includes roll and inverted views
    ang = lambda a, b: np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), np.sum(a * b, axis=1)))
    forward_deg = ang(pred_c2w_aligned[:, :, 2], gt_c2w[:, :, 2])
    up_deg = ang(-pred_c2w_aligned[:, :, 1], -gt_c2w[:, :, 1])
    roll_deg = None
    return {"per_frame_geodesic_deg": angle.tolist(),
            "geodesic_mean_deg": float(np.mean(angle)), "geodesic_median_deg": float(np.median(angle)),
            "geodesic_p90_deg": float(np.percentile(angle, 90)), "geodesic_max_deg": float(np.max(angle)),
            "forward_mean_deg": float(np.mean(forward_deg)), "forward_median_deg": float(np.median(forward_deg)),
            "forward_p90_deg": float(np.percentile(forward_deg, 90)), "forward_max_deg": float(np.max(forward_deg)),
            "up_mean_deg": float(np.mean(up_deg)), "up_median_deg": float(np.median(up_deg)),
            "up_p90_deg": float(np.percentile(up_deg, 90)), "up_max_deg": float(np.max(up_deg))}


def orientation_audit(predicted_extrinsics, gt_poses, indices, center_alignment_rotation=None, pose_kind="world_to_camera"):
    """Report orientation errors under (a) the camera-centre Umeyama rotation and (b) a Kabsch rotation fit.

    Evaluation-only: the fitted rotations are diagnostics and are never exported as corrected poses.
    """
    pred_w2c = to_world_to_camera_rotations(predicted_extrinsics, pose_kind)
    gt_c2w = gt_c2w_stack(gt_poses, indices)
    kabsch = fit_rotation_alignment(pred_w2c, gt_c2w)
    out = {"rotation_kabsch_aligned": rotation_angles_deg(pred_w2c, gt_poses, indices, kabsch),
           "kabsch_alignment_rotation": kabsch.tolist(), "exported_as_corrected_poses": False}
    if center_alignment_rotation is not None:
        out["center_umeyama_aligned"] = rotation_angles_deg(pred_w2c, gt_poses, indices, center_alignment_rotation)
    return out

def audit_path(prediction_npz: Path, ground_truth_json: Path, path_name: str):
    with np.load(prediction_npz, allow_pickle=False) as data:
        required = {"extrinsics", "source_indices"}
        if not required.issubset(data.files):
            raise ValueError(f"prediction archive missing arrays: {sorted(required - set(data.files))}")
        extrinsics = np.asarray(data["extrinsics"], dtype=float)
        indices = np.asarray(data["source_indices"], dtype=int)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError("source_indices must be unique nonnegative integers")
    gt_doc = json.loads(ground_truth_json.read_text(encoding="utf-8"))
    poses = gt_doc.get("poses")
    if not isinstance(poses, list):
        raise ValueError("ground truth must contain a poses array")
    if np.any(indices >= len(poses)):
        raise ValueError("source_indices refer beyond ground-truth pose array")
    gt_centers = np.asarray([poses[int(i)]["position"] for i in indices], dtype=float)
    pred_centers = camera_centers(extrinsics)
    if len(pred_centers) != len(indices):
        raise ValueError("extrinsics and source_indices lengths differ")
    fit = fit_similarity(pred_centers, gt_centers)
    orientation = rotation_angles_deg(extrinsics, poses, indices, fit["rotation"])
    orientation_extended = orientation_audit(extrinsics, poses, indices, fit["rotation"])
    return {
        "schema_version": 2, "diagnostic": "oracle_similarity_aligned_camera_pose_trajectory",
        "orientation_convention": "GT c2w columns [right,-up,forward] from simulator.camera_basis (roll: right -> right*cos r - up*sin r); absent roll means zero",
        "path": path_name, "frame_count": len(indices), "source_indices": indices.tolist(),
        "input_hashes": {"prediction_npz_sha256": sha256(prediction_npz),
                         "private_ground_truth_json_sha256": sha256(ground_truth_json)},
        "metrics": {"position_rmse_m": fit["rmse"], "position_rmse_fraction_of_gt_extent": fit["rmse"] / fit["gt_extent"],
                    "max_position_error_m": fit["max_error"], "max_position_error_fraction_of_gt_extent": fit["max_error"] / fit["gt_extent"],
                    "groundtruth_trajectory_extent_m": fit["gt_extent"],
                    "alignment_covariance_condition_ratio": fit["alignment_covariance_condition_ratio"],
                    "alignment_rotation_well_conditioned": fit["alignment_rotation_well_conditioned"],
                    "orientation": orientation, "orientation_extended": orientation_extended},
        "oracle_alignment": {"used_only_for_diagnostic": True, "proper_rotation_det": float(np.linalg.det(fit["rotation"])),
                             "uniform_scale": fit["scale"], "rotation_matrix": fit["rotation"].tolist(),
                             "translation": fit["translation"].tolist(), "position_singular_values": fit["position_singular_values"],
                             "alignment_covariance_singular_values": fit["alignment_covariance_singular_values"],
                             "exported_as_reconstruction_input": False},
        "metric_pass": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    pred_paths = {p.name: p for p in args.prediction_root.glob("path_*") if (p / "prediction.npz").is_file()}
    gt_paths = {p.stem: p for p in args.ground_truth_root.glob("path_*.json")}
    names = sorted(set(pred_paths) & set(gt_paths))
    if not names:
        raise SystemExit("no matching path_* prediction and ground-truth files")
    reports = [audit_path(pred_paths[name] / "prediction.npz", gt_paths[name], name) for name in names]
    args.output.mkdir(parents=True, exist_ok=False)
    for report in reports:
        destination = args.output / f"{report['path']}.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"schema_version": 2,
        "diagnostic": "evaluation-only oracle camera pose trajectory audit", "metric_pass": False,
        "paths": [{"path": r["path"], **r["metrics"]} for r in reports]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()



