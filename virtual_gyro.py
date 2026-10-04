"""Export a simulated phone gyroscope stream from private simulator poses.

The private pose file is consumed only here. The public artifact contains
timestamps and consecutive relative camera rotations, never pose labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import simulator


SCHEMA = "virtual_gyro_v1"


def _opencv_camera_to_world(pose: dict) -> np.ndarray:
    """Return C with OpenCV camera axes (right, down, forward) as world columns."""
    right, up, forward = simulator.camera_basis(pose)
    return np.column_stack((right, -up, forward))


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Stable SO(3) logarithm, including near-zero and near-pi rotations."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return 0.5 * np.array([rotation[2, 1] - rotation[1, 2],
                              rotation[0, 2] - rotation[2, 0],
                              rotation[1, 0] - rotation[0, 1]])
    if math.pi - angle < 1e-6:
        # The symmetric part gives the axis even when sin(angle) is tiny.
        vals, vecs = np.linalg.eigh((rotation + np.eye(3)) * 0.5)
        axis = vecs[:, int(np.argmax(vals))]
        skew = np.array([rotation[2, 1] - rotation[1, 2],
                         rotation[0, 2] - rotation[2, 0],
                         rotation[1, 0] - rotation[0, 1]])
        if np.dot(axis, skew) < 0:
            axis = -axis
        return angle * axis
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2.0 * math.sin(angle))
    return angle * axis


def _rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        x, y, z = vector
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        return np.eye(3) + skew
    x, y, z = vector / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1. - math.cos(angle)) * (skew @ skew)


def make_stream(poses: Sequence[dict], *, timestamps_s: Sequence[float] | None = None,
                dt_s: float = 1.0 / 30.0, seed: int = 0,
                noise_std_rad_s: float = 0.0,
                bias_rad_s: Sequence[float] = (0.0, 0.0, 0.0)) -> dict:
    """Build a public gyro JSON value from private yaw/pitch/roll poses.

    Noise and bias are applied in OpenCV camera-axis rotation-vector space as
    an angular-rate sensor model. Each reported matrix remains in SO(3).
    """
    if len(poses) < 1:
        raise ValueError("at least one pose is required")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    if noise_std_rad_s < 0 or not math.isfinite(noise_std_rad_s):
        raise ValueError("noise_std_rad_s must be nonnegative and finite")
    bias = np.asarray(bias_rad_s, dtype=float)
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise ValueError("bias_rad_s must contain three finite values")
    if timestamps_s is None:
        times = np.arange(len(poses), dtype=float) * dt_s
    else:
        times = np.asarray(timestamps_s, dtype=float)
        if times.shape != (len(poses),) or not np.all(np.isfinite(times)):
            raise ValueError("timestamps_s must have one finite value per pose")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("timestamps_s must be strictly increasing")
    rotations = [_opencv_camera_to_world(p) for p in poses]
    rng = np.random.default_rng(seed)
    deltas = []
    for i in range(1, len(rotations)):
        sample_dt = float(times[i] - times[i - 1])
        if sample_dt <= 0:
            raise ValueError("timestamps_s must be strictly increasing")
        delta = rotations[i - 1].T @ rotations[i]
        rate = _rotation_vector(delta) / sample_dt
        rate = rate + bias + rng.normal(0.0, noise_std_rad_s, size=3)
        deltas.append(_rotation_exp(rate * sample_dt).tolist())
    return {
        "schema": SCHEMA,
        "frame_count": len(poses),
        "dt_s": float(dt_s),
        "timestamps_s": times.tolist(),
        "delta_camera_to_previous": deltas,
        "manifest": {
            "generator": "virtual_gyro.py",
            "seed": int(seed),
            "noise_model": "independent Gaussian angular-rate noise, rad/s",
            "noise_std_rad_s": float(noise_std_rad_s),
            "bias_rad_s": bias.tolist(),
            "rotation_convention": "OpenCV camera axes (right, down, forward); delta[i] = C[i-1].T @ C[i]",
        },
    }


def export(private_pose_json: Path, output_json: Path, *, fps: float = 30.0,
           seed: int = 0, noise_std_rad_s: float = 0.0,
           bias_rad_s: Sequence[float] = (0.0, 0.0, 0.0)) -> dict:
    """Read simulator private poses, emit only a public relative sensor stream."""
    raw = private_pose_json.read_bytes()
    private = json.loads(raw)
    poses = private["poses"]
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    dt = 1.0 / fps
    stream = make_stream(poses, dt_s=dt, seed=seed,
                         noise_std_rad_s=noise_std_rad_s, bias_rad_s=bias_rad_s)
    stream["manifest"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stream, indent=2) + "\n", encoding="utf-8")
    return stream


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses-json", type=Path, required=True, help="private simulator path JSON")
    parser.add_argument("--output", type=Path, required=True, help="public gyro stream JSON")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std-rad-s", type=float, default=0.0)
    parser.add_argument("--bias-rad-s", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    args = parser.parse_args(argv)
    stream = export(args.poses_json, args.output, fps=args.fps, seed=args.seed,
                    noise_std_rad_s=args.noise_std_rad_s, bias_rad_s=args.bias_rad_s)
    print(f"Wrote {len(stream['delta_camera_to_previous'])} relative gyro samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
