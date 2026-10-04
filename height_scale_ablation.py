"""Evaluation-only sensitivity of floor-plan errors to RGB-derived height scales.

Reads frozen depth predictions and evaluator outputs. It never changes a reconstruction,
chooses a percentile by ground truth, or marks a benchmark stage complete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import compare_wall_boundaries as comparison
import estimate_height
import extract_planes


QUANTILES = {"q95_q05": (5.0, 95.0), "q98_q02": (2.0, 98.0), "q99_q01": (1.0, 99.0)}


def analyze(root: Path) -> dict:
    rows = []
    root = Path(root)
    frozen_index = json.loads((root / "frozen_index.json").read_text(encoding="utf-8"))
    for budget in frozen_index.get("budgets", []):
        label = budget["label"]
        prediction_path = Path(budget["prediction"])
        if not prediction_path.is_absolute():
            prediction_path = comparison.OUTPUTS / prediction_path
        frozen = root / "frozen" / label
        axes_data = json.loads((frozen / "axes.json").read_text(encoding="utf-8"))
        planes = json.loads((frozen / "planes.json").read_text(encoding="utf-8"))
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction = {key: archive[key] for key in archive.files}
        points = extract_planes.points_from_prediction(prediction)
        centers = estimate_height.camera_centers(prediction["extrinsics"])
        for index, hypothesis in enumerate(axes_data.get("hypotheses", [])):
            if not hypothesis.get("co_supported_with_best"):
                continue
            axis = np.asarray(hypothesis["axis_world"], dtype=float)
            estimate = estimate_height.estimate_height(planes, points, axis, centers)
            row = {"budget": label, "hypothesis": index, "axis_world": axis.tolist(),
                   "height_estimator_status": estimate.get("status"),
                   "plane_pair_height_unscaled": estimate.get("separation_unscaled"),
                   "quantile_spans_unscaled": {}, "evaluator_diagnostic": None}
            if estimate.get("status") != "estimated_unscaled_separation":
                row["height_estimator_reasons"] = estimate.get("reasons")
                rows.append(row)
                continue
            centered_heights = (points - np.median(points, axis=0)) @ axis
            for name, (lo, hi) in QUANTILES.items():
                row["quantile_spans_unscaled"][name] = float(
                    np.percentile(centered_heights, hi) - np.percentile(centered_heights, lo))
            eval_path = root / "evaluation" / f"{label}_hyp{index}.json"
            if eval_path.exists():
                evaluation = json.loads(eval_path.read_text(encoding="utf-8")).get("evaluation") or {}
                if evaluation.get("status") == "scored":
                    metrics = evaluation["metrics"]
                    base_height = float(estimate["separation_unscaled"])
                    methods = {}
                    for name, proposed_height in row["quantile_spans_unscaled"].items():
                        ratio = base_height / proposed_height
                        errors = [(edge["pred_length_m"] * ratio - edge["true_length_m"])
                                  for edge in metrics.get("edge_errors", [])]
                        methods[name] = {
                            "scale_multiplier_vs_plane_pair": ratio,
                            "max_abs_wall_error_cm": 100.0 * max(map(abs, errors)) if errors else None,
                            "max_relative_wall_error_percent": max(
                                (100.0 * abs(error) / edge["true_length_m"]
                                 for error, edge in zip(errors, metrics.get("edge_errors", []))), default=None),
                            "passes_1_5cm_diagnostic": bool(errors and max(map(abs, errors)) <= 0.015),
                        }
                    row["evaluator_diagnostic"] = {
                        "role": "post_freeze_sensitivity_only; contains isolated ground-truth comparison",
                        "plane_pair_max_abs_wall_error_cm": metrics.get("max_abs_wall_error_cm"),
                        "alternative_height_scale_results": methods,
                        "not_a_metric_estimator_validation": True,
                    }
            rows.append(row)
    return {
        "model": "height_scale_ablation",
        "stage_complete": False,
        "inputs": str(root.resolve()),
        "ground_truth_use": "evaluation files read only after reconstruction was frozen; never used to derive the candidate height spans",
        "quantile_choices": QUANTILES,
        "limitation": "Depth-cloud quantile spans can include furniture, openings, and outliers. This single-room sensitivity study does not validate any percentile across houses or captures.",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, default=comparison.OUTPUTS / "wall-boundary-ray-v2")
    parser.add_argument("--output", type=Path, required=True, help="new JSON path; existing files are not overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = analyze(args.comparison_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
