"""Select a structural axis from metric camera-center motion, without camera up.

This is a declared capture-protocol prior: the operator walks mainly on one
floor. A strongly planar trajectory supplies an *unsigned* normal. Rolling or
inverting the phone does not change camera centers. Nonplanar or short/linear
motion is refused, and no candidate is chosen from ground-truth scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    e = np.asarray(extrinsics, dtype=float)
    if e.ndim != 3 or e.shape[1:] != (3, 4) or not np.isfinite(e).all():
        raise ValueError("expected finite Nx3x4 world-to-camera poses")
    R, t = e[:, :, :3], e[:, :, 3]
    if not np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3), atol=.02):
        raise ValueError("invalid camera rotations")
    return -np.einsum("nji,nj->ni", R, t)


def select_motion_axis(extrinsics: np.ndarray, axis_hypotheses: list[dict], *,
                       max_planarity_ratio: float = .5,
                       min_axis_alignment: float = .9,
                       min_alignment_gap: float = .2,
                       duplicate_axis_cosine: float = .98) -> dict:
    centers = camera_centers(extrinsics)
    if len(centers) < 8:
        return {"status": "refused", "reason": "too_few_poses"}
    _, singular, vt = np.linalg.svd(centers - np.median(centers, axis=0), full_matrices=False)
    if singular[1] < .1 or singular[1] / max(singular[0], 1e-12) < .1:
        return {"status": "refused", "reason": "trajectory_nearly_linear",
                "singular_values": singular.tolist()}
    ratio = float(singular[2] / singular[1])
    if ratio > max_planarity_ratio:
        return {"status": "refused", "reason": "trajectory_not_planar",
                "planarity_ratio": ratio, "singular_values": singular.tolist()}
    normal = vt[2]
    alignments = []
    for h in axis_hypotheses:
        axis = np.asarray(h["axis_world"], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("invalid structural axis")
        alignments.append(float(abs(np.dot(normal, axis / np.linalg.norm(axis)))))
    if not alignments:
        return {"status": "refused", "reason": "no_axis_candidates"}
    order = np.argsort(alignments)[::-1]
    best = float(alignments[int(order[0])])
    # Hypotheses within ~11 degrees of the best axis (unsigned) are the same
    # direction, not a competing one; they must not trigger the gap refusal.
    best_axis = np.asarray(axis_hypotheses[int(order[0])]["axis_world"], dtype=float)
    best_axis /= np.linalg.norm(best_axis)
    rivals = [int(i) for i in order[1:]
              if abs(float(np.dot(best_axis, np.asarray(axis_hypotheses[int(i)]["axis_world"], dtype=float)
                                  / np.linalg.norm(axis_hypotheses[int(i)]["axis_world"])))) < duplicate_axis_cosine]
    second = float(alignments[rivals[0]]) if rivals else 0.0
    if best < min_axis_alignment or best - second < min_alignment_gap:
        return {"status": "refused", "reason": "axis_alignment_ambiguous",
                "best_alignment": best, "second_alignment": second,
                "planarity_ratio": ratio}
    return {"status": "selected", "hypothesis_index": int(order[0]),
            "best_alignment": best, "second_alignment": second,
            "planarity_ratio": ratio, "axis_sign_resolved": False,
            "prior": "camera-centre motion is planar; no camera-up or gravity input"}


def run_batch(batch_root: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for capture in sorted(batch_root.glob("house_*/capture_*/prediction.npz")):
        house = capture.parent.parent
        path_index = int(capture.parent.name.rsplit("_", 1)[1])
        house_seed = int(house.name.rsplit("_", 1)[1])
        label = f"house{house_seed}_path{path_index}"
        axes_path = house / f"evaluation_{path_index:02d}" / "frozen" / label / "axes.json"
        with np.load(capture) as pred:
            extrinsics = pred["extrinsics"].copy()
        axes = json.loads(axes_path.read_text(encoding="utf-8"))["hypotheses"]
        choice = select_motion_axis(extrinsics, axes)
        rows.append({"house_id": house.name, "path_index": path_index, **choice})
    frozen = {"schema": "motion_axis_selection_v1", "truth_free": True,
              "ground_truth_read": False, "rows": rows}
    (output / "frozen_selection.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    # Only after selection is frozen may the evaluation-only diagnostics be read.
    diagnostics = json.loads((batch_root / "report.json").read_text(encoding="utf-8"))
    indexed = {(r["house_id"], r["path_index"]): r for r in diagnostics["rows"]}
    evaluated = []
    for row in rows:
        original = indexed[(row["house_id"], row["path_index"])]
        candidate = (original["candidates"][row["hypothesis_index"]]
                     if row["status"] == "selected" else None)
        metric = candidate["metric_diagnostic"] if candidate else None
        evaluated.append({**row, "boundary_status": candidate["boundary_status"] if candidate else None,
                          "metric_diagnostic": metric})
    result = {"schema": "motion_axis_batch_evaluation_v1", "selection_frozen_first": True,
              "official_stage_completion": False, "rows": evaluated,
              "selected_count": sum(r["status"] == "selected" for r in evaluated),
              "selected_complete_count": sum(r["status"] == "selected" and
                                             r["boundary_status"] == "complete_polygon_hypothesis"
                                             for r in evaluated),
              "diagnostic_metric_pass_count": sum((r["metric_diagnostic"] or {}).get("diagnostic_3_percent_wall_pass", False)
                                                   and (r["metric_diagnostic"] or {}).get("diagnostic_1p5cm_height_pass", False)
                                                   for r in evaluated)}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run_batch(a.batch_root, a.output)
    print(json.dumps({k: result[k] for k in ("selected_count", "selected_complete_count",
                                           "diagnostic_metric_pass_count")}, indent=2))


if __name__ == "__main__":
    main()
