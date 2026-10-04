"""Remove axial image twist using only relative gyroscope rotations.

The reference frame is the first camera, not gravity or simulator world-up.
Parallel transport keeps the image basis continuous through upside-down views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cross = np.cross(a, b)
    s = float(np.linalg.norm(cross))
    c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        raise ValueError("adjacent forward directions are antipodal")
    k = cross / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def transport_roll_angles(deltas: np.ndarray) -> np.ndarray:
    """Return per-frame physical axial twist relative to a transported basis.

    deltas[i] maps camera i+1 vectors into camera i coordinates. Only relative
    rotations are accepted; the initial attitude is the identity by gauge.
    """
    d = np.asarray(deltas, dtype=float)
    if d.ndim != 3 or d.shape[1:] != (3, 3):
        raise ValueError("delta rotations must be (N-1,3,3)")
    if not np.all(np.isfinite(d)) or not np.allclose(d @ np.transpose(d, (0, 2, 1)), np.eye(3), atol=2e-3):
        raise ValueError("delta rotations must be finite proper rotations")
    if np.any(np.linalg.det(d) < .998):
        raise ValueError("reflections are not valid gyroscope increments")
    attitude = np.eye(3)
    transported = np.eye(3)
    angles = [0.0]
    for delta in d:
        prev_forward = attitude[:, 2]
        attitude = attitude @ delta
        transported = _minimal_rotation(prev_forward, attitude[:, 2]) @ transported
        target_right = transported[:, 0]
        actual_right = attitude[:, 0]
        forward = attitude[:, 2]
        angles.append(float(np.arctan2(np.dot(forward, np.cross(target_right, actual_right)),
                                       np.dot(target_right, actual_right))))
    return np.unwrap(np.asarray(angles))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stabilize(image_dir: Path, gyro_json: Path, output: Path, *, crop_no_padding: bool = False) -> dict:
    data = json.loads(gyro_json.read_text(encoding="utf-8"))
    if data.get("schema") != "virtual_gyro_v1":
        raise ValueError("unsupported gyroscope stream schema")
    angles = transport_roll_angles(np.asarray(data["delta_camera_to_previous"], dtype=float))
    frames = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if len(frames) != len(angles) or len(frames) != data["frame_count"]:
        raise ValueError("gyroscope and RGB frame counts differ")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    output.mkdir(parents=True, exist_ok=True)
    hashes = []
    side = None
    for i, path in enumerate(frames):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read {path}")
        h, w = image.shape[:2]
        candidate = int(np.ceil(np.hypot(h, w)))
        if side is None:
            side = candidate
        if candidate != side:
            raise ValueError("input frames must have identical dimensions")
        # OpenCV's positive angle turns a screen-right feature upward. A
        # positive physical camera twist already turned it upward, so undo it
        # with the negative angle. The diagonal canvas retains the full FOV.
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -float(np.degrees(angles[i])), 1.0)
        m[:, 2] += ((side - w) / 2, (side - h) / 2)
        corrected = cv2.warpAffine(image, m, (side, side), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        if crop_no_padding:
            # The central inscribed square is valid at every possible roll.
            # Its half diagonal is at most half the shorter source dimension.
            crop = int(np.floor(min(h, w) / np.sqrt(2)))
            lo = (side - crop) // 2
            corrected = cv2.resize(corrected[lo:lo + crop, lo:lo + crop],
                                   (max(h, w), max(h, w)), interpolation=cv2.INTER_LINEAR)
        dst = output / f"{i:06d}.png"
        if not cv2.imwrite(str(dst), corrected):
            raise OSError(f"failed to write {dst}")
        hashes.append({"input": str(path.resolve()), "sha256": _sha(path)})
    manifest = {"schema": "gyro_stabilization_v1", "gyro_input": str(gyro_json.resolve()),
                "gyro_sha256": _sha(gyro_json), "frame_count": len(frames),
                "reference": "first camera; parallel-transported image basis; no gravity/up",
                "stabilization_deg": np.degrees(angles).tolist(), "source_images": hashes,
                "output_dir": str(output.resolve()), "truth_free": True,
                "crop_no_padding": bool(crop_no_padding)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--gyro", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crop-no-padding", action="store_true",
                   help="use a central inscribed square with no artificial borders")
    a = p.parse_args()
    result = stabilize(a.images, a.gyro, a.output, crop_no_padding=a.crop_no_padding)
    print(json.dumps({"output": result["output_dir"], "frame_count": result["frame_count"]}))


if __name__ == "__main__":
    main()
