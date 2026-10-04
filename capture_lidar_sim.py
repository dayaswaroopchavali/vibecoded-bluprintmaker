"""Generate a public LiDAR-like capture from a private simulator scene/path.

Only RGB, noisy metric camera-z depth, sensor confidence, intrinsics and noisy
sensor poses are exported. Surface labels, scene dimensions and true poses stay
in the private source files, as they would with a real phone capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import simulator


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sensor_extrinsic(pose: dict, rng: np.random.Generator,
                      position_noise_m: float, rotation_noise_deg: float) -> np.ndarray:
    right, up, forward = simulator.camera_basis(pose)
    camera_to_world = np.column_stack((right, -up, forward))
    if rotation_noise_deg:
        delta = rng.normal(0, np.deg2rad(rotation_noise_deg), 3)
        noise, _ = cv2.Rodrigues(delta.astype(float))
        camera_to_world = camera_to_world @ noise
    position = np.asarray(pose["position"], dtype=float) + rng.normal(0, position_noise_m, 3)
    R = camera_to_world.T
    return np.column_stack((R, -R @ position)).astype(np.float32)


def capture(scene_json: Path, poses_json: Path, output: Path, *, width: int = 240,
            height: int = 180, fov_deg: float = 78.0, seed: int = 0,
            depth_noise_m: float = 0.005, position_noise_m: float = 0.005,
            rotation_noise_deg: float = 0.1) -> dict:
    for name, value in (("depth_noise_m", depth_noise_m),
                        ("position_noise_m", position_noise_m),
                        ("rotation_noise_deg", rotation_noise_deg)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if min(width, height) <= 0 or not 10 < fov_deg < 150:
        raise ValueError("invalid image size or FOV")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    scene = json.loads(scene_json.read_text(encoding="utf-8"))
    poses = json.loads(poses_json.read_text(encoding="utf-8"))["poses"]
    rng = np.random.default_rng(seed)
    images, depths, confidences, extrinsics, intrinsics = [], [], [], [], []
    for pose in poses:
        with np.errstate(all="ignore"):
            rgb, ray_depth, camera = simulator.render(scene, pose, width, height, fov_deg, depth_out=True)
        fy, fx, cx, cy = camera["fy"], camera["fx"], camera["cx"], camera["cy"]
        yy, xx = np.mgrid[:height, :width]
        ray_norm = np.sqrt(1 + ((xx - cx) / fx)**2 + ((yy - cy) / fy)**2)
        valid = np.isfinite(ray_depth) & (ray_depth > 0)
        z = np.zeros((height, width), dtype=np.float32)
        z[valid] = (ray_depth[valid] / ray_norm[valid] +
                    rng.normal(0, depth_noise_m, int(valid.sum()))).astype(np.float32)
        valid &= z > 0
        z[~valid] = 0
        confidence = valid.astype(np.float32)
        images.append(rgb)
        depths.append(z)
        confidences.append(confidence)
        extrinsics.append(_sensor_extrinsic(pose, rng, position_noise_m, rotation_noise_deg))
        intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32))
    output.mkdir(parents=True, exist_ok=True)
    npz = output / "prediction.npz"
    np.savez_compressed(npz, images=np.asarray(images, dtype=np.uint8),
                        depth=np.asarray(depths, dtype=np.float32),
                        confidence=np.asarray(confidences, dtype=np.float32),
                        extrinsics=np.asarray(extrinsics, dtype=np.float32),
                        intrinsics=np.asarray(intrinsics, dtype=np.float32),
                        source_indices=np.arange(len(poses), dtype=np.int64))
    manifest = {"schema": "simulated_lidar_capture_v1", "capture_data": npz.name,
                "capture_sha256": _hash(npz), "frame_count": len(poses),
                "width": width, "height": height, "fov_deg": fov_deg, "seed": seed,
                "noise": {"depth_std_m": depth_noise_m, "position_std_m": position_noise_m,
                          "rotation_std_deg_per_axis": rotation_noise_deg},
                "public_arrays": ["images", "depth", "confidence", "extrinsics", "intrinsics", "source_indices"],
                "private_values_exported": False,
                "note": "Sensor poses/depth are observations, not the private scene blueprint or segmentation."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--poses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=240)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--fov", type=float, default=78.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth-noise-m", type=float, default=0.005)
    p.add_argument("--position-noise-m", type=float, default=0.005)
    p.add_argument("--rotation-noise-deg", type=float, default=0.1)
    a = p.parse_args()
    result = capture(a.scene, a.poses, a.output, width=a.width, height=a.height,
                     fov_deg=a.fov, seed=a.seed, depth_noise_m=a.depth_noise_m,
                     position_noise_m=a.position_noise_m,
                     rotation_noise_deg=a.rotation_noise_deg)
    print(json.dumps({"output": result["capture_data"], "frames": result["frame_count"]}))


if __name__ == "__main__":
    main()
