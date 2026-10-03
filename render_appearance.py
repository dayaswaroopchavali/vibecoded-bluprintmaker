"""Re-render a frozen capture path under a chosen APPEARANCE (original | textured | realistic) for a controlled ablation.

Lights, render settings, camera poses and the room shell are identical between appearances; original/textured change materials only,
realistic additionally adds thin decor/trim/rug objects (private_ground_truth/decor_manifest.json lists them for masking). The poses
come from the capture's private_ground_truth/path_0N.json and go through simulator.camera_basis exactly as in
blender_capture. Output layout (never overwrites; refuses a non-empty destination):

    OUT/rgb/path_XX/NNNNNN.png          the only thing a model may see (no ground truth inside)
    OUT/videos/path_XX.mp4, OUT/contact_sheets/path_XX.jpg
    OUT/appearance_manifest.json        appearance, texture seed, all material parameters, poses sha256
    OUT/private_ground_truth/           scene.json, path_XX.json (poses), seed_manifest.json  (kept separate, as in captures)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import blender_capture as bc  # noqa: E402


def parse_frames(spec: str | None, count: int, step: int) -> list[int]:
    if spec:
        idx = [int(s) for s in spec.split(",") if s.strip()]
        if not idx or any(i < 0 or i >= count for i in idx):
            raise ValueError(f"--frames must be indices in [0,{count})")
        return idx
    return list(range(0, count, step))


def render(capture: Path, output: Path, path_name: str = "path_00", appearance: str = "original", texture_seed: int = 1,
           width: int = 480, height: int = 360, fov: float = 78., fps: int = 6, frame_step: int = 1,
           frames: str | None = None, blender: Path = bc.DEFAULT_BLENDER) -> dict:
    capture, output = Path(capture), Path(output)
    private = capture / "private_ground_truth"
    scene_path, poses_path = private / "scene.json", private / f"{path_name}.json"
    for p in (scene_path, poses_path):
        if not p.is_file(): raise FileNotFoundError(p)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty {output}")
    if appearance not in bc.APPEARANCES: raise ValueError(f"appearance must be one of {bc.APPEARANCES}")
    pj = bc.load_poses_json(poses_path)
    idx = parse_frames(frames, len(pj["poses"]), frame_step)
    seeds = {}
    sm = private / "seed_manifest.json"
    if sm.is_file(): seeds = json.loads(sm.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    bc.run_worker(blender, scene_path, output, int(seeds.get("capture_seed", 0)), width, height, len(pj["poses"]), fov, fps,
                  False, appearance, texture_seed, poses_path, path_name, idx)
    bc.build_video_and_sheet(output, path_name, idx, width, height, fps)
    mpath = output / "appearance_manifest.json"
    man = json.loads(mpath.read_text(encoding="utf-8"))
    man.update({"source_capture": str(capture.resolve()), "source_poses_file": str(poses_path.resolve()),
                "source_poses_file_sha256": hashlib.sha256(poses_path.read_bytes()).hexdigest(),
                "frame_indices": idx, "frame_step": frame_step if not frames else None, "fps": fps,
                "layout": "rgb/ is model-visible; private_ground_truth/ must never be given to a model"})
    mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True, help="existing capture folder (contains private_ground_truth/)")
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--path", default="path_00")
    ap.add_argument("--appearance", choices=bc.APPEARANCES, default="original"); ap.add_argument("--texture-seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=480); ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fov", type=float, default=78.); ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--frame-step", type=int, default=1); ap.add_argument("--frames", help="comma list of frame indices (overrides step)")
    ap.add_argument("--blender", type=Path, default=bc.DEFAULT_BLENDER)
    a = ap.parse_args()
    if min(a.width, a.height, a.fps, a.frame_step) <= 0 or not 10 < a.fov < 150: ap.error("bad width/height/fps/step/fov")
    try:
        man = render(a.capture, a.output, a.path, a.appearance, a.texture_seed, a.width, a.height, a.fov, a.fps, a.frame_step, a.frames, a.blender)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        ap.error(str(e))
    print(json.dumps({"output": str(a.output), "appearance": man["appearance"], "poses_sha256": man["poses_sha256"], "frames": len(man["frame_indices"])}))


if __name__ == "__main__":
    main()
