"""Aggregate run_pipeline.py manifests into a plain report, with an optional separate evaluation step.

collect() reads only reconstruction-side files. evaluate_runs() is the only place private truth is read: it takes
the generator scene file after predictions are frozen, builds the user's custom diagnostic (prediction and ground
truth each normalized to 3 m height), and keeps the official anchor-protocol gates "not_evaluated" because no
ceiling height, opening or wall-correspondence stage exists. Nothing here claims a stage gate, ten-house result
or assessment completion.

  python integration_report.py --runs <run_dir_or_parent>... --output <new prefix>
  python integration_report.py --runs ... --output <new prefix> --evaluate-scene <capture>/private_ground_truth/scene.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_pipeline import NAMED_BLOCKERS


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_runs(roots: list[Path]) -> list[Path]:
    """Run folders = any folder holding a path-level pipeline_manifest.json (a `full` top folder is expanded)."""
    runs: list[Path] = []
    for root in roots:
        manifest = _read(Path(root) / "pipeline_manifest.json")
        if manifest is not None and manifest.get("mode") in ("from_prediction", "full_path"):
            runs.append(Path(root))
        else:
            runs.extend(sorted(p.parent for p in Path(root).glob("*/pipeline_manifest.json")
                               if (_read(p) or {}).get("mode") in ("from_prediction", "full_path")))
    return runs


def collect_run(run: Path) -> dict:
    manifest = _read(run / "pipeline_manifest.json") or {}
    stages = {s["name"]: s for s in manifest.get("stages", [])}
    polygon, prior = _read(run / "polygon.json") or {}, _read(run / "up_prior.json") or {}
    orientation, cuboid = _read(run / "orientation.json") or {}, _read(run / "cuboid_baseline.json") or {}
    edges = polygon.get("edges") or []
    return {
        "run": str(run), "name": run.name, "mode": manifest.get("mode"),
        "prediction_sha256": manifest.get("prediction_sha256"),
        "input_video": manifest.get("input_video"),
        "stage_status": {k: v.get("status") for k, v in stages.items()},
        "stage_seconds": {k: v.get("duration_seconds") for k, v in stages.items()},
        "failure_reasons": {k: v.get("failure_reasons") for k, v in stages.items() if v.get("failure_reasons")},
        "output_hashes": {k: v.get("outputs") for k, v in stages.items() if v.get("outputs")},
        "up_prior": {"status": prior.get("status"), "up_world": prior.get("up_world"), "reason": prior.get("reason"),
                     "inlier_fraction": (prior.get("camera_up_diagnostics") or {}).get("inlier_fraction"),
                     "trajectory_check": (prior.get("trajectory_check") or {}).get("status")},
        "polygon": {"status": polygon.get("status"), "vertex_count": len(polygon.get("vertices_xy") or []),
                    "edge_labels": [e["label"] for e in edges], "inferred_edge_count": polygon.get("inferred_edge_count"),
                    "wall_candidate_count": polygon.get("wall_candidate_count"),
                    "incomplete_boundaries": polygon.get("incomplete_boundaries"),
                    "rejected_candidates": [{"plane": r.get("plane_index"), "reason": r.get("reason")}
                                            for r in polygon.get("rejected_candidates", [])]},
        "orientation": {"status": orientation.get("status"), "reason": orientation.get("reason"),
                        "wall_lengths_unscaled": orientation.get("wall_lengths_unscaled")},
        "cuboid_baseline": {"role": "diagnostic_baseline_only", "status": cuboid.get("status"),
                            "dimensions": cuboid.get("dimensions")},
        "viewer": str(run / "viewer.html") if (run / "viewer.html").is_file() else None,
    }


def collect(roots: list[Path]) -> dict:
    runs = [collect_run(r) for r in find_runs(roots)]
    return {"schema_version": 1, "runs": runs,
            "polygon_complete_count": sum(r["polygon"]["status"] == "complete_polygon_hypothesis" for r in runs),
            "oriented_count": sum((r["orientation"]["status"] or "").startswith("oriented") for r in runs),
            "run_count": len(runs),
            "named_blockers": [{"name": k, "status": "open", "detail": v} for k, v in NAMED_BLOCKERS.items()],
            "claims": "none: no stage gate, ten-house, official-gate or assessment completion is claimed; unit tests clear no blocker"}


def build_eval_documents(run_summaries: list[dict], run_dirs: list[Path], scene: dict, house_id: str) -> tuple[dict, dict]:
    """Prediction bundle (RGB-derived only) and ground-truth bundle (private scene) for the 3 m-normalized diagnostic."""
    pred_runs, gt_runs = [], []
    for summary, run in zip(run_summaries, run_dirs):
        box = _read(run / "box_orientation.json") or {}
        cuboid = _read(run / "cuboid_baseline.json") or {}
        if (box.get("status") or "").startswith("oriented"):
            dims, axis, pairs = box["shape_dimensions"], box["height_axis_index"], 3
        else:  # no independently predicted height axis: the evaluator labels any axis match as oracle
            ok = cuboid.get("status") == "complete_box_hypothesis" and cuboid.get("dimensions")
            dims, axis = (cuboid["dimensions"] if ok else [None, None, None]), None
            pairs = sum(bool((a.get("surface_pair") or {}).get("complete")) for a in cuboid.get("axes", []))
        room = {"room_id": "room-a", "shape_dimensions": dims, "wall_height_m": None, "walls": [], "openings": [],
                "complete_surface_pairs": pairs}
        if axis is not None:
            room["height_axis_index"] = axis
        key = {"stage": "single_room", "house_id": house_id, "capture_seed": summary["name"]}
        pred_runs.append({**key, "rooms": [room]})
        gt_runs.append({**key, "rooms": [{"room_id": "room-a", "shape_dimensions": [scene["width"], scene["length"], scene["height"]],
                                          "wall_height_m": scene["height"], "walls": [], "openings": []}]})
    return ({"schema_version": 1, "evaluation_mode": "height_normalized_3m", "runs": pred_runs},
            {"schema_version": 1, "runs": gt_runs})


def evaluate_runs(run_dirs: list[Path], summaries: list[dict], scene_path: Path) -> dict:
    """The only function that reads private truth. Official anchor protocols are deliberately not run."""
    from evaluation import evaluate
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8"))
    pred, gt = build_eval_documents(summaries, run_dirs, scene, f"house-{scene.get('seed')}")
    result = evaluate(pred, gt)
    custom = result["height_normalized_3m"]
    return {"scene_file": str(scene_path), "evaluator_mode": "height_normalized_3m",
            "custom_diagnostic": {"protocol": result["custom_protocol"], "per_run": custom["per_run"],
                                  "max_horizontal_absolute_error_cm": custom["max_horizontal_absolute_error_cm"],
                                  "custom_1_5cm_gate_full_dataset": custom["full_dataset_custom_gate_le_1_5cm"],
                                  "height_axis_oracle_used": custom["height_axis_oracle_used"],
                                  "evaluated_run_count": len(custom["per_run"]), "submitted_run_count": len(summaries),
                                  "evaluator_height_recovered_flag": custom["height_recovered"],
                                  "height_axis_source_note": "a declared camera-motion prior, unvalidated; the evaluator flag only says no oracle axis was used and covers only the runs it could score",
                                  "incomplete_support_room_count": custom["incomplete_support_room_count"],
                                  "note": "prediction and ground truth are both normalized to 3 m; not a metric-height result"},
            "official_anchor_protocols": {"status": "not_evaluated",
                "reason": "no ceiling-height estimate, opening detection or wall correspondence exists to build anchored predictions"},
            "completion": result["completion"],
            "prediction_documents_hash_input": [summary["prediction_sha256"] for summary in summaries]}


def to_markdown(report: dict) -> str:
    lines = ["# Pipeline integration report", "", f"Runs: {report['run_count']}; complete polygons: {report['polygon_complete_count']}; "
             f"oriented: {report['oriented_count']}. {report['claims']}.", "",
             "| run | up prior | polygon | edges | orientation | cuboid baseline (diagnostic) |", "|---|---|---|---|---|---|"]
    for r in report["runs"]:
        edges = ",".join(r["polygon"]["edge_labels"]) or "none"
        lines.append(f"| {r['name']} | {r['up_prior']['status']} | {r['polygon']['status']} | {edges} | {r['orientation']['status']} | {r['cuboid_baseline']['status']} |")
    lines += ["", "## Stage failures and incomplete reasons", ""]
    for r in report["runs"]:
        for stage, reasons in r["failure_reasons"].items():
            lines.append(f"- {r['name']} / {stage}: {'; '.join(map(str, reasons))}")
        for rej in r["polygon"]["rejected_candidates"]:
            lines.append(f"- {r['name']} / rejected plane {rej['plane']}: {rej['reason']}")
    lines += ["", "## Named blockers (open)", ""] + [f"- **{b['name']}**: {b['detail']}" for b in report["named_blockers"]]
    ev = report.get("evaluation")
    if ev:
        d = ev["custom_diagnostic"]
        lines += ["", "## Custom 3 m-normalized diagnostic (evaluator only)", "",
                  f"Max horizontal error: {d['max_horizontal_absolute_error_cm']} cm; full-dataset 1.5 cm gate: {d['custom_1_5cm_gate_full_dataset']}; "
                  f"oracle height axis used: {d['height_axis_oracle_used']}; official anchor protocols: {ev['official_anchor_protocols']['status']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new prefix; writes <prefix>.json and <prefix>.md")
    parser.add_argument("--evaluate-scene", type=Path, help="private scene.json; read only by the evaluation step")
    args = parser.parse_args()
    json_path, md_path = args.output.with_suffix(".json"), args.output.with_suffix(".md")
    for path in (json_path, md_path):
        if path.exists():
            parser.error(f"output already exists: {path}")
    runs = find_runs(args.runs)
    if not runs:
        parser.error("no pipeline run folders found")
    report = collect(args.runs)
    if args.evaluate_scene:
        report["evaluation"] = evaluate_runs(runs, report["runs"], args.evaluate_scene)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(json_path); print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
