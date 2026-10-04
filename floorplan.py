"""Local floor-plan CLI for RGB video/photos and existing simulated-LiDAR NPZ captures.

The output always states whether a usable polygon was actually recovered.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent


def call(*args: object) -> None:
    subprocess.run([sys.executable, *(str(arg) for arg in args)], check=True)


def polygon_from_rgb(run_dir: Path) -> dict:
    candidates = sorted(run_dir.glob("polygon_axis_hypothesis_*.json"))
    candidates += [run_dir / "polygon.json"]
    hypotheses = []
    for path in candidates:
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "complete_polygon_hypothesis":
            hypotheses.append({"floor_plan": data, "source": str(path)})
    if hypotheses:
        return {"status": "orientation_unresolved", "modality": "rgb",
                "floor_plan": None, "hypotheses": hypotheses,
                "reason": "Closed polygon hypotheses exist; the pipeline has not resolved the room orientation.",
                "diagnostics": str(run_dir / "pipeline_manifest.json")}
    return {"status": "incomplete", "modality": "rgb", "floor_plan": None,
            "reason": "No supported closed room polygon was recovered.",
            "diagnostics": str(run_dir / "pipeline_manifest.json")}


def rgb(source: Path, output: Path, frames: int, resolution: int, assets: Path | None = None) -> dict:
    inference = output / "inference"
    reconstruction = output / "reconstruction"
    call(HERE / "infer_rgb.py", "--input", source, "--output", inference,
         "--frames", frames, "--resolution", resolution,
         *(["--assets", assets] if assets else []))
    prediction = inference / "prediction.npz"
    if not prediction.is_file():
        raise RuntimeError(f"RGB inference did not write {prediction}")
    command = [HERE / "run_pipeline.py", "from-prediction", "--prediction", prediction,
               "--output", reconstruction]
    if source.is_file():
        command += ["--video", source]
    call(*command)
    return polygon_from_rgb(reconstruction)


def lidar_npz(source: Path, output: Path, modality: str = "simulated_lidar") -> dict:
    """Accept the public NPZ schema produced by capture_lidar_sim.py, not raw phone LiDAR."""
    import numpy as np
    import run_opening_lidar_batch as batch

    with np.load(source, allow_pickle=False) as archive:
        required = {"images", "depth", "confidence", "intrinsics", "extrinsics"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing LiDAR NPZ arrays: {sorted(missing)}")
        frames = len(archive["extrinsics"])
    item = batch._freeze_one({"label": "capture", "seed": 0, "path_index": 0,
                              "prediction": source, "frames": frames}, output / "lidar")
    boundary = item.get("boundary")
    if not boundary or not boundary.get("vertices_xy"):
        return {"status": "incomplete", "modality": modality,
                "floor_plan": None, "reason": "No supported closed boundary was recovered.",
                "diagnostics": item}
    return {"status": "complete_metric", "modality": modality, "accuracy_verified": False,
            "floor_plan": boundary, "height": item.get("height_estimate"),
            "openings": item.get("openings"), "paired_walls": item.get("paired_walls")}


def write_svg(result: dict, path: Path) -> None:
    """Draw the recovered closed polygon without inventing missing walls."""
    plan = result.get("floor_plan") or {}
    vertices = plan.get("vertices_xy") or []
    if len(vertices) < 3:
        return
    points = [(float(x), float(y)) for x, y in vertices]
    xs, ys = zip(*points)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    scale = 680 / span
    margin = 60
    mapped = [(margin + (x - min(xs)) * scale,
               margin + (max(ys) - y) * scale) for x, y in points]
    poly = " ".join(f"{x:.2f},{y:.2f}" for x, y in mapped)
    units = escape(str(plan.get("units", "unscaled")))
    labels = []
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(points, points[1:] + points[:1]), 1):
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** .5
        mx = (mapped[index - 1][0] + mapped[index % len(mapped)][0]) / 2
        my = (mapped[index - 1][1] + mapped[index % len(mapped)][1]) / 2
        labels.append(f'<text x="{mx:.1f}" y="{my - 8:.1f}" text-anchor="middle" '
                      f'font-size="14" fill="#184268">{length:.2f} {units}</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">'
           '<rect width="800" height="800" fill="white"/>'
           f'<polygon points="{poly}" fill="#edf5fa" stroke="#173e5c" '
           'stroke-width="5" stroke-linejoin="round"/>'
           + "".join(labels) + '</svg>')
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="RGB video, directory of photos, or simulated-LiDAR prediction.npz")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--modality", choices=["rgb", "simulated-lidar", "lidar"], default="rgb",
                        help="NPZ has no trustworthy unit tag; simulated metric LiDAR must be explicitly selected")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--assets", type=Path, help="Existing photogrammetry model/source assets folder for RGB")
    parser.add_argument("--sensor-config", type=Path, help="Explicit depth/pose/intrinsics conventions for extracted LiDAR logs")
    parser.add_argument("--lidar-max-frames", type=int, default=120, help="Maximum uniformly sampled depth frames")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if args.modality == "rgb" and source.suffix.lower() == ".npz":
        parser.error("RGB NPZ predictions are unscaled. Use RGB video/photos, or explicitly select --modality simulated-lidar for metric sensor captures.")
    if args.modality == "simulated-lidar" and source.suffix.lower() != ".npz":
        parser.error("Simulated LiDAR input must be a prediction.npz capture.")
    if args.modality == "lidar" and (not source.is_dir() or not args.sensor_config):
        parser.error("LiDAR requires an extracted capture folder and --sensor-config.")
    output.mkdir(parents=True)
    if args.modality == "lidar":
        from import_lidar import convert_capture
        config = json.loads(args.sensor_config.read_text(encoding="utf-8"))
        imported = output / "sensor_import"
        manifest = convert_capture(source, imported, config, max_frames=args.lidar_max_frames)
        result = lidar_npz(imported / "prediction.npz", output, "lidar_configured")
        result["sensor_import"] = manifest
    elif args.modality == "simulated-lidar":
        result = lidar_npz(source, output)
    else:
        result = rgb(source, output, args.frames, args.resolution, args.assets)
    plan = output / "floor_plan.json"
    plan.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_svg(result, output / "floor_plan.svg")
    for index, hypothesis in enumerate(result.get("hypotheses", [])):
        write_svg(hypothesis, output / f"floor_plan_hypothesis_{index}.svg")
    print(plan)
    return 0 if result["status"].startswith("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
