"""Run axis_candidates on the frozen DA3 predictions, then (afterwards, evaluation only) score every candidate.

Stage 1 (truth-free, `run_extraction`): for each prediction enumerate the complete structural-axis candidate set, and for EVERY
candidate run height_diagnosis.estimate_room_height and wall_extraction + boundary_assembly.assemble_boundary_ray. Everything
is frozen (sha256) to <out>/frozen/<label>/ and a per-prediction JSON + markdown report is written. No scene argument exists
in this stage: truth cannot reach it.

Stage 2 (EVALUATION ONLY, `run_evaluation`): reads the frozen files back (hash-checked) and scores EVERY candidate with
evaluate_floorplan.py against the private scene. Truth is never used to choose a candidate; the truth-free ranking's preference
and whether it is the candidate that scores are reported side by side. Gates are not changed (1.5 cm, 3 %).

    python run_axis_candidates.py --output-dir <NEW folder under outputs/sim-benchmark>
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

import compare_wall_boundaries as cwb

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
RAY_K = 2.0
FRAMES_AVAILABLE = 120
COVERAGE_SCENE = "blender-coverage-v3/capture/private_ground_truth/scene.json"
BUDGETS = [
    {"label": "path_01_full120", "prediction": "blender-coverage-v3/path_01/da3_full120/prediction.npz", "scene": COVERAGE_SCENE},
    {"label": "b_stride3_40", "prediction": "backend-compare-v1/control_original/da3/prediction_source_indices_remapped.npz",
     "scene": "texture-ablation-v1/control_original/private_ground_truth/scene.json"},
    {"label": "c_stride2_60_DIFFERENT_BUDGET",
     "prediction": "backend-compare-v1/extra_control_original_step2_DIFFERENT_BUDGET/da3/prediction_source_indices_remapped.npz",
     "scene": "texture-ablation-v1/control_original/private_ground_truth/scene.json"},
    {"label": "d_textured40", "prediction": "backend-compare-v1/control_textured/da3/prediction_source_indices_remapped.npz",
     "scene": "texture-ablation-v1/control_textured/private_ground_truth/scene.json"},
    {"label": "path_02_36", "prediction": "blender-coverage-v3/path_02/da3/prediction.npz", "scene": COVERAGE_SCENE},
    {"label": "a_linspace36", "prediction": "blender-coverage-v3/path_00/da3/prediction.npz", "scene": COVERAGE_SCENE},
    {"label": "wide_seed17_path00_full120", "prediction": "wide-capture-seed17-path00/da3_full120/prediction.npz",
     "scene": "wide-capture-seed17-path00/private_ground_truth/scene.json"},
]


def _default_hooks():
    def load_prediction(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def per_view_points(pred):
        import height_diagnosis
        return height_diagnosis._per_view_points(pred)

    def normals_from_prediction(pred):
        import axis_candidates
        return axis_candidates.normals_from_prediction(pred)

    def extract_planes(points):
        import extract_planes
        return extract_planes.extract(points)

    def camera_centers(extrinsics):
        import estimate_height
        return estimate_height.camera_centers(extrinsics)

    def estimate_room_height(points, axis, ids):
        import height_diagnosis
        return height_diagnosis.estimate_room_height(points, axis, ids)

    def extract_wall_lines(pred, axis):
        import wall_extraction
        return wall_extraction.extract_wall_lines(pred, axis)

    def assemble_boundary_ray(pred, walls, k):
        import boundary_assembly
        return boundary_assembly.assemble_boundary_ray(pred, walls, k)

    def evaluate_floorplan(prediction, scene, height, manifest):
        import evaluate_floorplan as module
        return module.evaluate_floorplan(prediction, scene, height, manifest=manifest, axis_sign_resolved=False)

    def load_scene(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return {k: v for k, v in locals().items() if callable(v)}


def _key(axis):
    return tuple(round(float(x), 10) for x in axis)


def _freeze(out_dir, rel, obj):
    return {"path": rel, "sha256": cwb.write_new_json(Path(out_dir) / rel, obj)}


def _walls_summary(walls):
    """Walls without the per-frame observation arrays (3 MB each); enough to audit the polygon."""
    return {k: v for k, v in walls.items() if k != "observations"}


def run_extraction(budgets, out_dir, hooks=None, events=None, max_candidates=None):
    """Truth-free stage: no truth argument exists here. Returns the frozen index."""
    import axis_candidates as ac
    hooks = {**_default_hooks(), **(hooks or {})}
    events = events if events is not None else []
    out_dir = Path(out_dir)
    index = {"budgets": [], "ray_k": RAY_K}
    for budget in budgets:
        label = budget["label"]
        events.append(("extract_budget", label))
        rec = {"label": label, "prediction": str(budget["prediction"]), "frames_available": FRAMES_AVAILABLE, "files": {},
               "candidates": []}
        try:
            pred = hooks["load_prediction"](budget["prediction"])
            idx = np.asarray(pred.get("source_indices", np.arange(len(pred["depth"]))))
            rec["source_indices"] = [int(i) for i in idx]
            rec["retained_capture_fraction"] = float(len(idx) / FRAMES_AVAILABLE)
            points, ids = hooks["per_view_points"](pred)
            planes = hooks["extract_planes"](points)
            normals = hooks["normals_from_prediction"](pred)
            centers = hooks["camera_centers"](pred["extrinsics"])
        except Exception as exc:
            rec.update(status="failed_before_candidates",
                       error={"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc()[-1500:]})
            index["budgets"].append(rec)
            continue
        store = {}

        def runner(axis, pred=pred, points=points, ids=ids, store=store, label=label):
            events.append(("candidate_downstream", label, _key(axis)))
            item = {"height": None, "walls": None, "boundary": None, "errors": {}}
            axis = np.asarray(axis, float)
            try:
                item["height"] = hooks["estimate_room_height"](points, axis, ids)
            except Exception as exc:
                item["errors"]["height"] = f"{type(exc).__name__}: {exc}"
            summary = {"boundary_status": "not_run", "walls_status": None}
            try:
                walls = hooks["extract_wall_lines"](pred, axis)
                item["walls"] = walls
                summary.update(walls_status=walls.get("status"), wall_line_count=len(walls.get("lines", [])),
                               walls_reasons=list(walls.get("reasons", []) or []))
                if walls.get("status") == "ok":
                    item["boundary"] = hooks["assemble_boundary_ray"](pred, walls, RAY_K)
                    summary["boundary_status"] = item["boundary"].get("status")
                    v = item["boundary"].get("vertices_xy")
                    summary["polygon_vertex_count"] = None if v is None else len(v)
                else:
                    summary["boundary_status"] = "walls_unavailable"
            except Exception as exc:
                item["errors"]["walls_or_boundary"] = f"{type(exc).__name__}: {exc}"
                summary["boundary_status"] = "failed"
                summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc()[-1500:]}
            h = item["height"] or {}
            summary["height_status"] = h.get("status")
            summary["height_separation_unscaled"] = h.get("separation_unscaled")
            store[_key(axis)] = item
            return summary

        kwargs = {} if max_candidates is None else {"max_candidates": max_candidates}
        result = ac.enumerate_axis_candidates(planes, points, normals=normals, camera_centers=centers, polygon_runner=runner, **kwargs)
        schema = ac.to_axis_hypotheses(result)
        rec["status"] = "ok"
        rec["axis_status"] = result.get("status")
        rec["files"]["planes"] = _freeze(out_dir, f"frozen/{label}/planes.json", planes)
        rec["files"]["candidates"] = _freeze(out_dir, f"frozen/{label}/candidates.json", result)
        rec["files"]["axis_hypotheses_schema"] = _freeze(out_dir, f"frozen/{label}/axis_hypotheses_schema.json", schema)
        for c in result.get("candidates", []):
            item = store.get(_key(c["axis_world"]), {})
            row = {"candidate_id": c["id"], "axis_world": c["axis_world"], "rank": c["rank"], "score": c["score"],
                   "viable": c["viable"], "co_supported_with_best": c["co_supported_with_best"], "kinds": c["kinds"],
                   "polygon": c.get("polygon"), "files": {}}
            k = c["id"]
            if item.get("height") is not None:
                row["files"]["height"] = _freeze(out_dir, f"frozen/{label}/cand{k}_height.json", item["height"])
            if item.get("walls") is not None:
                row["files"]["walls_summary"] = _freeze(out_dir, f"frozen/{label}/cand{k}_walls_summary.json", _walls_summary(item["walls"]))
            if item.get("boundary") is not None:
                row["files"]["boundary"] = _freeze(out_dir, f"frozen/{label}/cand{k}_boundary.json", item["boundary"])
            if item["errors"] if item else False:
                row["errors"] = item["errors"]
            rec["candidates"].append(row)
        index["budgets"].append(rec)
    events.append(("frozen", None))
    cwb.write_new_json(out_dir / "frozen_index.json", index)
    return index


def write_reports(index, out_dir):
    """Per-prediction truth-free JSON + markdown (candidate count, polygon status, height)."""
    out_dir = Path(out_dir)
    for rec in index["budgets"]:
        if rec.get("status") != "ok":
            cwb.write_new_json(out_dir / "reports" / f"{rec['label']}.json", rec)
            continue
        cands = cwb.read_frozen(out_dir, rec["files"]["candidates"]["path"], rec["files"]["candidates"]["sha256"])
        rows = []
        for c in rec["candidates"]:
            poly = c.get("polygon") or {}
            rows.append({"candidate_id": c["candidate_id"], "axis_world": c["axis_world"], "rank": c["rank"], "score": c["score"],
                         "viable": c["viable"], "co_supported_with_best": c["co_supported_with_best"], "kinds": c["kinds"],
                         "polygon_status": poly.get("boundary_status"), "walls_status": poly.get("walls_status"),
                         "height_status": poly.get("height_status"), "height_separation_unscaled": poly.get("height_separation_unscaled")})
        report = {"label": rec["label"], "prediction": rec["prediction"], "axis_status": rec["axis_status"],
                  "candidate_count": len(rows), "co_supported_count": cands.get("co_supported_count"),
                  "unresolved_ambiguity": cands.get("unresolved_ambiguity"), "candidate_cap": cands.get("candidate_cap"),
                  "directions": cands.get("directions"), "candidates": rows, "truth_used": False}
        cwb.write_new_json(out_dir / "reports" / f"{rec['label']}.json", report)
        md = [f"# {rec['label']}", "", f"- prediction: `{rec['prediction']}`", f"- axis status: {rec['axis_status']}",
              f"- candidates: {len(rows)} (co-supported {cands.get('co_supported_count')}); cap {cands.get('candidate_cap')}",
              f"- unresolved ambiguity: {cands.get('unresolved_ambiguity')}", "- truth used: no (this report is truth-free)", "",
              "| id | axis (world, unsigned) | kinds | score | viable | co-sup | polygon | height status | height (unscaled) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ax = ", ".join(f"{x:.3f}" for x in r["axis_world"])
            sep = "" if r["height_separation_unscaled"] is None else f"{r['height_separation_unscaled']:.3f}"
            md.append(f"| {r['candidate_id']} | ({ax}) | {'+'.join(r['kinds'])} | {r['score']:.2f} | {r['viable']} | "
                      f"{r['co_supported_with_best']} | {r['polygon_status']} | {r['height_status']} | {sep} |")
        (out_dir / "reports" / f"{rec['label']}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _gate(ev, name):
    g = (ev or {}).get(name) or {}
    return g.get("pass")


def run_evaluation(budgets, out_dir, index, hooks=None, events=None):
    """EVALUATION ONLY. Reads the frozen files back, scores every candidate, never selects by truth."""
    hooks = {**_default_hooks(), **(hooks or {})}
    events = events if events is not None else []
    out_dir = Path(out_dir)
    if not (out_dir / "frozen_index.json").exists():
        raise RuntimeError("refusing to evaluate before results are frozen")
    scene_of = {b["label"]: b.get("scene") for b in budgets}
    rows = []
    for rec in index["budgets"]:
        if rec.get("status") != "ok":
            continue
        label = rec["label"]
        scene = hooks["load_scene"](scene_of[label])
        manifest = {"provenance": "frozen frame budget", "retained_capture_fraction": rec["retained_capture_fraction"],
                    "frames_registered": len(rec["source_indices"]), "frames_available": rec["frames_available"]}
        viable_scores = [c["score"] for c in rec["candidates"] if c["viable"]]
        best = max(viable_scores) if viable_scores else None
        tied = sum(1 for s in viable_scores if best is not None and abs(s - best) < 1e-9)
        for c in rec["candidates"]:
            row = {"label": label, "candidate_id": c["candidate_id"], "rank": c["rank"], "co_supported": c["co_supported_with_best"],
                   "score": c["score"], "tie_count_at_top": tied,
                   "truth_free_top_ranked": bool(c["viable"] and best is not None and abs(c["score"] - best) < 1e-9),
                   "axis_world": c["axis_world"],
                   "polygon_status": (c.get("polygon") or {}).get("boundary_status"), "evaluation": None}
            files = c["files"]
            if "boundary" not in files or "height" not in files:
                row["not_scored_reason"] = "no boundary" if "boundary" not in files else "no height estimate"
                row["status"] = "not_scored"
                if "height" in files:
                    h = cwb.read_frozen(out_dir, files["height"]["path"], files["height"]["sha256"])
                    row["height_status"] = h.get("status")
                rows.append(row)
                continue
            boundary = cwb.read_frozen(out_dir, files["boundary"]["path"], files["boundary"]["sha256"])
            height = cwb.read_frozen(out_dir, files["height"]["path"], files["height"]["sha256"])
            import height_diagnosis
            compat = height_diagnosis.evaluator_compatible(height)
            events.append(("evaluate", label, c["candidate_id"]))
            try:
                ev = hooks["evaluate_floorplan"](cwb.eval_copy(boundary), scene, compat, manifest)
                m = ev.get("metrics") or {}
                row.update(status=ev.get("status"), height_status=height.get("status"), failure_reasons=ev.get("failure_reasons"),
                           iou=m.get("iou"), max_abs_wall_error_cm=m.get("max_abs_wall_error_cm"),
                           max_rel_wall_error_percent=m.get("max_rel_wall_error_percent"),
                           predicted_edges=m.get("predicted_edge_count"), true_edges=m.get("true_edge_count"),
                           gate_1_5cm=_gate(ev, "custom_1_5cm_horizontal_diagnostic"),
                           gate_3pct=_gate(ev, "wall_length_3_percent_threshold_oracle_normalised"),
                           gate_failure_reasons=(ev.get("custom_1_5cm_horizontal_diagnostic") or {}).get("failure_reasons"))
                cwb.write_new_json(out_dir / "evaluation" / f"{label}_cand{c['candidate_id']}.json", {"evaluation": ev})
            except Exception as exc:
                row.update(status="evaluation_error", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    summary = {"rows": rows, "gates": {"custom_cm": 1.5, "official_relative_percent": 3.0, "changed": False},
               "oracle_choices": cwb.ORACLE_CHOICES, "selection_by_truth": False,
               "note": "Every candidate is scored; the evaluator never chose among candidates. No accuracy or completion claim."}
    cwb.write_new_json(out_dir / "evaluation_summary.json", summary)
    return summary


def results_markdown(summary):
    lines = ["# Axis candidates: oracle scoring of EVERY candidate (evaluation only)", "",
             "Truth is used only to score; it never selected a candidate. Gates unchanged: 1.5 cm and 3 %.", "",
             "top-tied = truth-free score equals the best viable score (tie count = how many viable candidates share it).", "",
             "| prediction | cand | rank | co-sup | top-tied | tie count | polygon | status | IoU | max err cm | max rel % | 1.5 cm | 3 % |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, fmt):
        return "" if v is None else format(v, fmt)
    for r in summary["rows"]:
        lines.append(f"| {r['label']} | {r['candidate_id']} | {r['rank']} | {r['co_supported']} | {r['truth_free_top_ranked']} | "
                     f"{r.get('tie_count_at_top')} | {r['polygon_status']} | {r.get('status')} | {f(r.get('iou'), '.3f')} | {f(r.get('max_abs_wall_error_cm'), '.2f')} | "
                     f"{f(r.get('max_rel_wall_error_percent'), '.2f')} | {r.get('gate_1_5cm')} | {r.get('gate_3pct')} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW folder (refuses to overwrite)")
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--no-evaluation", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output folder already exists: {args.output_dir}")
    budgets = []
    for b in BUDGETS:
        if args.labels and b["label"] not in args.labels:
            continue
        pred = args.outputs_root / b["prediction"]
        if not pred.exists():
            print(f"skipping {b['label']}: {pred} missing")
            continue
        budgets.append({**b, "prediction": str(pred), "scene": str(args.outputs_root / b["scene"])})
    args.output_dir.mkdir(parents=True)
    index = run_extraction(budgets, args.output_dir, max_candidates=args.max_candidates)
    write_reports(index, args.output_dir)
    if not args.no_evaluation:
        summary = run_evaluation(budgets, args.output_dir, index)
        (args.output_dir / "RESULTS.md").write_text(results_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "budgets": len(index["budgets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
