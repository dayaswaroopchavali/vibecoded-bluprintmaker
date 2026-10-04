"""Freeze and evaluate one RGB reconstruction with the existing boundary pipeline.

The reconstruction is loaded, converted to structural hypotheses, and frozen
before the private scene is opened for evaluation. This CLI is for assessment
only; private scene data never enters inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import boundary_viewer
import compare_wall_boundaries as C


def run(prediction: Path, private_scene: Path, frames_available: int, output: Path, label: str) -> dict:
    if frames_available < 1:
        raise ValueError("frames_available must be positive")
    if not Path(prediction).is_file() or not Path(private_scene).is_file():
        raise FileNotFoundError("prediction or private scene is missing")
    budget = [{
        "label": label,
        "description": "frozen RGB reconstruction; all native co-supported structural axes",
        "prediction": str(Path(prediction).resolve()),
        "scene": str(Path(private_scene).resolve()),
        "frames_available": frames_available,
    }]
    result = C.run_comparison(budget, Path(output), evaluate=True)
    payload = boundary_viewer.payload_from_comparison(Path(output))
    boundary_viewer.write_viewer(payload, Path(output) / "viewer.html")
    boundary_viewer.write_overlay_pngs(payload, Path(output) / "overlays")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--private-scene", type=Path, required=True)
    parser.add_argument("--frames-available", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    result = run(args.prediction, args.private_scene, args.frames_available, args.output, args.label)
    rows = result["table"]
    print(json.dumps({"output": str(args.output.resolve()), "rows": [{
        "hypothesis": row.get("hypothesis"), "status": row.get("status"),
        "score_status": (row.get("evaluation") or {}).get("status"),
        "max_abs_wall_error_cm": (row.get("evaluation") or {}).get("max_abs_wall_error_cm"),
        "max_rel_wall_error_percent": (row.get("evaluation") or {}).get("max_rel_wall_error_percent"),
        "custom_1_5cm_pass": (row.get("evaluation") or {}).get("custom_1_5cm_pass"),
        "three_percent_pass": (row.get("evaluation") or {}).get("three_percent_pass"),
    } for row in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
