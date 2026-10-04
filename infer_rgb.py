"""Run offline Depth Anything 3 inference on RGB images or a video.

This runner deliberately accepts no pose, depth, camera, or simulator labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="directory of RGB images or a video file")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--frames", type=int, default=24, help="number of evenly spaced frames to use (default: 24)")
    parser.add_argument("--resolution", type=int, default=420, help="maximum processing dimension in pixels (default: 420)")
    parser.add_argument("--seed", type=int, default=2026, help="random seed (default: 2026)")
    parser.add_argument("--assets", type=Path, help="Photogrammetry assets folder containing models/ and tools/da3-source/")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing output directory")
    return parser


def _even_indices(count: int, wanted: int) -> list[int]:
    import numpy as np

    n = min(count, wanted)
    if n == 1:
        return [0]
    return np.linspace(0, count - 1, n, dtype=int).tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gather_images(source: Path) -> tuple[list[Path], list[int], list[dict[str, str]]]:
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No supported RGB image files found in {source}")
    return files, list(range(len(files))), [{"path": str(p.resolve()), "sha256": _sha256(p)} for p in files]


def _video_frames(video: Path, wanted: int, temp_dir: Path) -> tuple[list[Path], list[int], list[dict[str, str]]]:
    # OpenCV is imported only for video inputs. Every decoded BGR frame is
    # converted to RGB before it is written and passed to the model.
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or not cap.isOpened():
            raise ValueError(f"Unable to read video or its frame count: {video}")
        indices = _even_indices(total, wanted)
        paths: list[Path] = []
        for ordinal, frame_index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Could not decode video frame {frame_index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            path = temp_dir / f"frame_{ordinal:05d}.png"
            Image.fromarray(rgb, mode="RGB").save(path)
            paths.append(path)
        return paths, indices, [{"path": str(video.resolve()), "sha256": _sha256(video)}]
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.frames < 1:
        raise SystemExit("--frames must be at least 1")
    if args.resolution < 1:
        raise SystemExit("--resolution must be at least 1")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists() or (not source.is_dir() and not source.is_file()):
        raise SystemExit(f"Input does not exist or is not a file/directory: {source}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists; choose a fresh folder or pass --overwrite: {output}")
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        raise SystemExit("--input must be a directory of images or a video file, not one image")

    # These paths follow the installed offline environment used by the
    # independent DA3 baseline; no model download is attempted.
    repo = Path(__file__).resolve().parents[2]
    photo = (args.assets or repo / "outputs" / "photogrammetry").expanduser().resolve()
    model_path = photo / "models" / "DA3-LARGE"
    source_roots = sorted((photo / "tools" / "da3-source").glob("*/src"))
    if not model_path.is_dir():
        raise SystemExit(f"Expected local DA3-LARGE model was not found: {model_path}")
    if not source_roots:
        raise SystemExit(f"DA3 source tree was not found under {photo / 'tools' / 'da3-source'}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(photo / "models" / "hf-cache")
    os.environ["MPLCONFIGDIR"] = str(photo / "tools" / "matplotlib-cache")
    sys.path.insert(0, str(source_roots[0]))

    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    output.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.time()
    temp_context = tempfile.TemporaryDirectory(prefix="da3-rgb-") if source.is_file() else None
    try:
        if source.is_dir():
            all_paths, all_indices, input_hashes = _gather_images(source)
            selected = _even_indices(len(all_paths), args.frames)
            paths = [all_paths[i] for i in selected]
            source_indices = [all_indices[i] for i in selected]
        else:
            paths, source_indices, input_hashes = _video_frames(source, args.frames, Path(temp_context.name))

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            device = "cuda"
        else:
            device = "cpu"
        model = DepthAnything3.from_pretrained(str(model_path)).to(device).eval()
        prediction = model.inference(
            [str(p) for p in paths],
            process_res=args.resolution,
            process_res_method="upper_bound_resize",
        )

        np.savez_compressed(
            output / "prediction.npz",
            depth=prediction.depth,
            confidence=prediction.conf,
            extrinsics=prediction.extrinsics,
            intrinsics=prediction.intrinsics,
            images=prediction.processed_images,
            source_indices=np.asarray(source_indices, dtype=np.int64),
        )
        manifest = {
            "model": "DA3-LARGE",
            "model_path": str(model_path),
            "input_type": "image_directory" if source.is_dir() else "video_rgb_frames",
            "input_path": str(source),
            "input_hashes": input_hashes,
            "source_indices": source_indices,
            "frames_requested": args.frames,
            "frames_used": len(paths),
            "resolution": args.resolution,
            "resolution_method": "upper_bound_resize",
            "seed": args.seed,
            "started_at_utc": started_at,
            "elapsed_seconds": time.time() - started,
            "device": device,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
            "unscaled": True,
            "scale_applied": False,
            "external_poses_depth_intrinsics_or_labels_used": False,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved {output / 'prediction.npz'}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
