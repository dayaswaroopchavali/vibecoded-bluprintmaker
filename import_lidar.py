"""Import an extracted real depth-camera capture into the benchmark NPZ schema.

Input layout: depth/{frame:06d}.png, confidence/{frame:06d}.png, and
odometry.csv. Configuration is explicit; see ``REQUIRED_CONFIG`` below.
Depths are camera aligned with the supplied per-row intrinsics. RGB is not
registered, so ``images`` contains black placeholders for geometry only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


REQUIRED_CONFIG = {
    "depth_scale_to_meters", "depth_kind", "pose_translation_scale_to_meters",
    "pose_direction", "camera_axes", "intrinsics_width", "intrinsics_height",
    "min_confidence", "depth_intrinsics_aligned",
}
CSV_FIELDS = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw",
              "fx", "fy", "cx", "cy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_positive(config: dict, key: str) -> float:
    if isinstance(config[key], bool):
        raise ValueError(f"{key} must be a finite positive number")
    try:
        value = float(config[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite positive number") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive number")
    return value


def _read_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")
    missing = REQUIRED_CONFIG - config.keys()
    if missing:
        raise ValueError(f"configuration is missing required fields: {', '.join(sorted(missing))}")
    result = dict(config)
    result["depth_scale_to_meters"] = _finite_positive(config, "depth_scale_to_meters")
    result["pose_translation_scale_to_meters"] = _finite_positive(config, "pose_translation_scale_to_meters")
    for key in ("intrinsics_width", "intrinsics_height"):
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or int(v) != v or v <= 0:
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(v)
    if config["depth_kind"] not in ("axial", "ray"):
        raise ValueError("depth_kind must be 'axial' or 'ray'")
    if config["pose_direction"] not in ("camera_to_world", "world_to_camera"):
        raise ValueError("pose_direction must be 'camera_to_world' or 'world_to_camera'")
    if config["camera_axes"] not in ("opencv", "opengl"):
        raise ValueError("camera_axes must be 'opencv' or 'opengl'")
    mc = config["min_confidence"]
    if isinstance(mc, bool) or not isinstance(mc, (int, float)) or not np.isfinite(mc) or int(mc) != mc or mc not in (0, 1, 2):
        raise ValueError("min_confidence must be 0, 1, or 2")
    if config["depth_intrinsics_aligned"] is not True:
        raise ValueError("depth_intrinsics_aligned must be true: RGB registration is not assumed")
    return result


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError(f"odometry quaternion must be normalized (norm={norm:.8g})")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _extrinsic(row: dict[str, float], cfg: dict) -> np.ndarray:
    R = _quaternion_matrix(np.array([row[k] for k in ("qx", "qy", "qz", "qw")], dtype=float))
    t = np.array([row[k] for k in ("x", "y", "z")], dtype=float) * cfg["pose_translation_scale_to_meters"]
    # B maps OpenCV camera coordinates to the source camera basis. It is its
    # own inverse for the OpenGL/OpenCV axis flip.
    B = np.diag([1.0, -1.0, -1.0]) if cfg["camera_axes"] == "opengl" else np.eye(3)
    if cfg["pose_direction"] == "camera_to_world":
        Rwc = B.T @ R.T
        twc = -Rwc @ t
    else:
        Rwc = B.T @ R
        twc = B.T @ t
    return np.column_stack((Rwc, twc)).astype(np.float32)


def _read_odometry(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not set(CSV_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"odometry.csv must contain columns: {', '.join(CSV_FIELDS)}")
        previous = None
        seen = set()
        for line, raw in enumerate(reader, start=2):
            try:
                values = {k: float(raw[k]) for k in CSV_FIELDS}
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid numeric odometry field on CSV line {line}") from exc
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError(f"non-finite odometry value on CSV line {line}")
            frame_value = values["frame"]
            if frame_value < 0 or int(frame_value) != frame_value:
                raise ValueError(f"frame must be a nonnegative integer on CSV line {line}")
            frame = int(frame_value)
            if frame in seen or (previous is not None and frame <= previous):
                raise ValueError("frame IDs must be unique and strictly increasing in odometry.csv")
            values["frame"] = frame
            # Validate quaternions even when max_frames means rows later in the
            # file are not exported.
            _quaternion_matrix(np.array([values[k] for k in ("qx", "qy", "qz", "qw")]))
            if values["fx"] <= 0 or values["fy"] <= 0:
                raise ValueError(f"fx and fy must be positive on CSV line {line}")
            rows.append(values)
            seen.add(frame)
            previous = frame
    if not rows:
        raise ValueError("odometry.csv has no frame rows")
    return rows


def convert_capture(input_dir: str | Path, output_dir: str | Path, config: dict,
                    max_frames: int | None = None) -> dict:
    """Convert a capture folder, returning the written manifest dictionary."""
    src, out = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    cfg = _read_config(config)
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, (int, np.integer)) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer when supplied")
    if not src.is_dir():
        raise FileNotFoundError(f"input capture folder does not exist: {src}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    depth_dir, conf_dir = src / "depth", src / "confidence"
    csv_path = src / "odometry.csv"
    if not depth_dir.is_dir() or not conf_dir.is_dir() or not csv_path.is_file():
        raise ValueError("input must contain depth/, confidence/, and odometry.csv")
    rows = _read_odometry(csv_path)
    ids = [r["frame"] for r in rows]
    expected = {f"{i:06d}.png" for i in ids}
    for label, folder in (("depth", depth_dir), ("confidence", conf_dir)):
        actual = {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
        if actual != expected:
            missing, extra = sorted(expected - actual), sorted(actual - expected)
            raise ValueError(f"{label} PNGs must exactly match odometry frame IDs; missing={missing[:5]}, extra={extra[:5]}")
    if max_frames is not None:
        selected = np.linspace(0, len(rows) - 1, min(int(max_frames), len(rows)), dtype=int)
        rows = [rows[i] for i in selected]
    if not rows:
        raise ValueError("no frames selected")

    depths, confidences, intrinsics, extrinsics, input_hashes = [], [], [], [], []
    image_size = None
    for row in rows:
        frame = row["frame"]
        dp, cp = depth_dir / f"{frame:06d}.png", conf_dir / f"{frame:06d}.png"
        raw_depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        raw_conf = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
        if raw_depth is None or raw_conf is None:
            raise ValueError(f"could not decode matched PNGs for frame {frame}")
        if raw_depth.ndim != 2 or raw_conf.ndim != 2:
            raise ValueError(f"depth and confidence PNGs must be single-channel for frame {frame}")
        if raw_depth.shape != raw_conf.shape:
            raise ValueError(f"depth/confidence dimensions differ for frame {frame}")
        if image_size is None:
            image_size = raw_depth.shape
        elif raw_depth.shape != image_size:
            raise ValueError("all capture frames must have consistent image dimensions")
        h, w = raw_depth.shape
        sx, sy = w / cfg["intrinsics_width"], h / cfg["intrinsics_height"]
        fx, fy = row["fx"] * sx, row["fy"] * sy
        cx, cy = (row["cx"] + 0.5) * sx - 0.5, (row["cy"] + 0.5) * sy - 0.5
        d = raw_depth.astype(np.float64) * cfg["depth_scale_to_meters"]
        valid = np.isfinite(d) & (d > 0)
        if cfg["depth_kind"] == "ray":
            yy, xx = np.mgrid[:h, :w]
            ray_norm = np.sqrt(1.0 + ((xx - cx) / fx)**2 + ((yy - cy) / fy)**2)
            d = d / ray_norm
        c_raw = raw_conf.astype(np.float64)
        # Confidence format is explicitly the three ordinal levels 0, 1, 2.
        valid_conf = np.isfinite(c_raw) & np.isin(c_raw, [0, 1, 2])
        if not valid_conf.all():
            raise ValueError(f"confidence PNG must contain only levels 0, 1, 2 (frame {frame})")
        c = (c_raw / 2.0).astype(np.float32)
        valid &= np.isfinite(d) & (d > 0) & (c_raw >= cfg["min_confidence"])
        d[~valid] = 0.0
        c[~valid] = 0.0
        depths.append(d.astype(np.float32))
        confidences.append(c)
        intrinsics.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
        extrinsics.append(_extrinsic(row, cfg))
        input_hashes.extend([{"path": f"depth/{dp.name}", "sha256": _sha256(dp)},
                             {"path": f"confidence/{cp.name}", "sha256": _sha256(cp)}])
    h, w = image_size
    images = np.zeros((len(rows), h, w, 3), dtype=np.uint8)
    out.mkdir(parents=True, exist_ok=False)
    data_path = out / "prediction.npz"
    np.savez_compressed(data_path, images=images, depth=np.asarray(depths, np.float32),
                        confidence=np.asarray(confidences, np.float32),
                        extrinsics=np.asarray(extrinsics, np.float32),
                        intrinsics=np.asarray(intrinsics, np.float32),
                        source_indices=np.asarray([r["frame"] for r in rows], dtype=np.int64))
    manifest = {
        "schema": "real_lidar_capture_v1", "capture_data": str(data_path),
        "capture_sha256": _sha256(data_path), "frame_count": len(rows), "width": w, "height": h,
        "source_indices": [r["frame"] for r in rows], "images_placeholder": True,
        "accuracy_verified": False,
        "source": {"odometry_csv": "odometry.csv", "odometry_csv_sha256": _sha256(csv_path),
                   "input_hashes": input_hashes},
        "conventions": {
            "depth_kind_source": cfg["depth_kind"], "depth_output": "axial camera-z meters",
            "depth_scale_to_meters": cfg["depth_scale_to_meters"],
            "pose_translation_scale_to_meters": cfg["pose_translation_scale_to_meters"],
            "pose_direction_source": cfg["pose_direction"], "camera_axes_source": cfg["camera_axes"],
            "output_pose": "world-to-camera OpenCV 3x4 extrinsics",
            "intrinsics_source_resolution": [cfg["intrinsics_width"], cfg["intrinsics_height"]],
            "intrinsics_output_resolution": [w, h],
            "intrinsics_resize": "fx,cx scaled by width ratio; fy,cy by height ratio; pixel-center cx'=(cx+0.5)*sx-0.5, cy'=(cy+0.5)*sy-0.5",
            "depth_intrinsics_aligned": True, "confidence_levels": {"0": 0.0, "1": 0.5, "2": 1.0},
            "min_confidence_level": int(cfg["min_confidence"]),
            "masked_depth_and_confidence": "both are zero for invalid depth or confidence below min_confidence",
            "frame_sampling": "all frames in source order" if max_frames is None else "uniform indices across the full source sequence",
            "images": "black RGB placeholders; RGB registration is not assumed; geometry only",
        },
        "config": cfg,
        "limitations": ["Sensor accuracy and coordinate conventions are caller asserted and not independently verified.",
                        "RGB is unavailable or unregistered; placeholder images must not be used for appearance evaluation."],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="extracted capture folder")
    parser.add_argument("--output", type=Path, required=True, help="new output folder")
    parser.add_argument("--config", type=Path, required=True, help="JSON containing all required conventions/scales")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = convert_capture(args.input, args.output, config, args.max_frames)
    print(json.dumps({"output": manifest["capture_data"], "frames": manifest["frame_count"],
                      "accuracy_verified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
