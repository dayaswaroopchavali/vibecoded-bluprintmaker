"""Freeze fresh simulator scenes and opening-aware capture paths without scoring.

The output is private capture-side material. Reconstruction receives only
sensor exports made from these files, never the scene or true poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import opening_capture_protocol
import simulator


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output: Path, scene_seeds: list[int], *, frames: int = 320,
            capture_seed: int = 3101) -> dict:
    if len(scene_seeds) < 10 or len(set(scene_seeds)) != len(scene_seeds):
        raise ValueError("requires at least ten distinct scene seeds")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty holdout: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in scene_seeds:
        scene = simulator.make_scene(seed)
        folder = output / f"scene_{seed}"
        folder.mkdir()
        scene_file = folder / "scene.json"
        scene_file.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        path_seeds = simulator.path_seeds(scene, capture_seed)
        if len(set(path_seeds)) != 3:
            raise RuntimeError(f"path seeds are not independent for scene {seed}")
        for index, path_seed in enumerate(path_seeds):
            poses = opening_capture_protocol.make_opening_path(scene, frames, path_seed, index)
            path_file = folder / f"path_{index:02d}.json"
            path_file.write_text(json.dumps({"path_seed": path_seed, "poses": poses}, indent=2), encoding="utf-8")
            rows.append({"house_id": f"scene_{seed}", "path_index": index,
                         "path_seed": int(path_seed), "frames": len(poses),
                         "scene_sha256": _sha(scene_file), "path_sha256": _sha(path_file)})
    manifest = {"schema": "private_lidar_holdout_capture_v1",
                "capture_protocol": "opening_capture_protocol.make_opening_path",
                "capture_protocol_sha256": _sha(Path(opening_capture_protocol.__file__)),
                "scene_seeds": scene_seeds, "path_count": len(rows),
                "scored": False, "rows": rows}
    (output / "PRIVATE_CAPTURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=320)
    parser.add_argument("--capture-seed", type=int, default=3101)
    args = parser.parse_args()
    report = prepare(args.output, args.scene_seeds, frames=args.frames,
                     capture_seed=args.capture_seed)
    print(json.dumps({"houses": len(report["scene_seeds"]),
                      "paths": report["path_count"], "scored": report["scored"]}))


if __name__ == "__main__":
    main()
