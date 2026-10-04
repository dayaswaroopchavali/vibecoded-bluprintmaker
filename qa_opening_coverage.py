"""Capture-side QA of aperture coverage; never an inference input.

Uses the private simulator scene to determine whether each capture actually
observed the full extent of each aperture. Writes only under the QA output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import simulator


def evaluate(scene: dict, path: list[dict], *, width: int = 120, height: int = 90,
             bins: int = 10) -> list[dict]:
    counts = np.zeros((len(scene["openings"]), bins, bins), np.uint16)
    frames = np.zeros(len(scene["openings"]), np.uint16)
    pixels = np.zeros(len(scene["openings"]), np.uint32)
    for pose in path:
        with np.errstate(all="ignore"):
            _, _, _, kind, positions = simulator.render(scene, pose, width, height,
                                                         78., True, True)
        mask = kind == simulator.KIND_OPENING
        if not np.any(mask):
            continue
        points = positions[mask].reshape(-1, 3)
        ids = simulator._opening_at(scene, points)
        for oi, opening in enumerate(scene["openings"]):
            selected = points[ids == oi]
            if len(selected) == 0:
                continue
            frames[oi] += 1
            pixels[oi] += len(selected)
            along_axis = 0 if opening["wall"] in (0, 2) else 1
            u = (selected[:, along_axis] - (opening["center"] - opening["width"] / 2)) / opening["width"]
            v = (selected[:, 2] - opening["z_min"]) / (opening["z_max"] - opening["z_min"])
            ui = np.clip((u * bins).astype(int), 0, bins - 1)
            vi = np.clip((v * bins).astype(int), 0, bins - 1)
            ids_flat = np.unique(vi * bins + ui)
            counts[oi].flat[ids_flat] += 1
    rows = []
    for oi, opening in enumerate(scene["openings"]):
        grid = counts[oi]
        rows.append({"opening_index": oi, "type": opening["type"],
                     "frames_with_any_pixels": int(frames[oi]),
                     "pixel_fraction_all_frames": float(pixels[oi] / (len(path) * width * height)),
                     "spatial_bins_seen_once_fraction": float(np.mean(grid >= 1)),
                     "spatial_bins_seen_three_frames_fraction": float(np.mean(grid >= 3)),
                     "spatial_bins_seen_five_frames_fraction": float(np.mean(grid >= 5)),
                     "min_edge_strip_three_frame_fraction": float(min(
                         np.mean(grid[:, :1] >= 3), np.mean(grid[:, -1:] >= 3))),
                     "frame_support_grid": grid.tolist()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=90)
    args = parser.parse_args()
    rows = []
    for house in sorted(args.batch_root.glob("house_*")):
        private = house / "private_ground_truth"
        scene = json.loads((private / "scene.json").read_text())
        for pi in range(3):
            path = json.loads((private / f"path_{pi:02d}.json").read_text())["poses"]
            for item in evaluate(scene, path, width=args.width, height=args.height):
                rows.append({"house_id": house.name, "path_index": pi, **item})
    report = {"schema": "capture_side_opening_spatial_qa_v1", "rows": rows,
              "private_truth_read": True, "inference_input": False,
              "resolution": [args.width, args.height]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"captures": len(rows) // 2,
                      "openings": len(rows),
                      "mean_three_frame_spatial_fraction": float(np.mean(
                          [r["spatial_bins_seen_three_frames_fraction"] for r in rows])),
                      "min_three_frame_spatial_fraction": float(min(
                          r["spatial_bins_seen_three_frames_fraction"] for r in rows))}, indent=2))


if __name__ == "__main__":
    main()
