"""Wall-based height anchor on the six frozen axis-candidates-v1 predictions, then (afterwards) oracle-labelled evaluation.

Stage 1 (truth-free, `run_extraction`): for EVERY structural-axis candidate of a frozen prediction, run
wall_height_anchor.estimate_wall_height on the prediction's cloud with the candidate's axis; the frozen cap-plane height of the same
candidate (axis-candidates-v1 candN_height.json, hash-checked) is passed only as the agreement reference. Results are frozen (sha256).
No scene argument exists in this stage.

Stage 2 (EVALUATION ONLY, `run_evaluation`): only after the freeze, scores every candidate that has a frozen boundary with
evaluate_floorplan.py twice, side by side: (old) the frozen cap-plane height through height_diagnosis.evaluator_compatible, (new) the
wall-based height. The new height is passed through the DIAGNOSTIC adapter when its status is not complete (the strict adapter would
make the evaluator refuse), and the row says so. Truth never selects a candidate or an estimator; gates are unchanged.

    python run_wall_height_anchor.py --output-dir <NEW folder under outputs/sim-benchmark>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import compare_wall_boundaries as cwb
import run_axis_candidates as rac

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
SOURCE = OUTPUTS / "axis-candidates-v1"
LABELS = ["path_01_full120", "b_stride3_40", "c_stride2_60_DIFFERENT_BUDGET", "d_textured40", "path_02_36", "a_linspace36"]
UNIT_HEIGHT_BAND = (0.9, 1.1)   # cap-plane separation in cloud units; "about one unit" = the true-vertical candidates (truth-free flag)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_extraction(source_index, source_dir, out_dir, labels=LABELS, hooks=None, events=None):
    import wall_height_anchor as wha
    hooks = {"per_view_points": None, **(hooks or {})}
    if hooks["per_view_points"] is None:
        import height_diagnosis
        hooks["per_view_points"] = height_diagnosis._per_view_points
    events = events if events is not None else []
    out_dir = Path(out_dir)
    index = {"budgets": [], "model": wha.estimate_wall_height.__module__}
    for rec in source_index["budgets"]:
        label = rec["label"]
        if label not in labels:
            continue
        events.append(("extract_budget", label))
        pred_path = rec["prediction"]
        with np.load(pred_path, allow_pickle=False) as data:
            pred = {k: data[k] for k in ("depth", "confidence", "intrinsics", "extrinsics")}
        points, ids = hooks["per_view_points"](pred)
        out = {"label": label, "prediction": str(pred_path), "prediction_sha256": _sha_file(pred_path),
               "retained_capture_fraction": rec["retained_capture_fraction"], "source_indices": rec["source_indices"],
               "frames_available": rec["frames_available"], "candidates": []}
        for c in rec["candidates"]:
            row = {"candidate_id": c["candidate_id"], "axis_world": c["axis_world"], "co_supported_with_best": c["co_supported_with_best"],
                   "files": {}}
            cap = None
            if "height" in c["files"]:
                cap = cwb.read_frozen(source_dir, c["files"]["height"]["path"], c["files"]["height"]["sha256"])
                row["cap_plane_height_source"] = c["files"]["height"]
            events.append(("candidate_downstream", label, c["candidate_id"]))
            try:
                res = wha.estimate_wall_height(points, np.asarray(c["axis_world"], float), ids,
                                               reference=None if cap is None else cap, compute_reference=False)
            except Exception as exc:                                       # recorded, never dropped
                row["error"] = f"{type(exc).__name__}: {exc}"
                out["candidates"].append(row)
                continue
            row["status"] = res["status"]
            row["separation_unscaled"] = res.get("separation_unscaled")
            row["cap_plane_separation_unscaled"] = None if cap is None else cap.get("separation_unscaled")
            for name, obj in (("wall_height", res), ("wall_height_strict_adapter", wha.evaluator_compatible(res)),
                              ("wall_height_diagnostic_adapter", wha.diagnostic_unscaled_separation(res))):
                rel = f"frozen/{label}/cand{c['candidate_id']}_{name}.json"
                row["files"][name] = {"path": rel, "sha256": cwb.write_new_json(out_dir / rel, obj)}
            out["candidates"].append(row)
        index["budgets"].append(out)
    events.append(("frozen", None))
    cwb.write_new_json(out_dir / "frozen_index.json", index)
    return index


def _gate(ev, name):
    return ((ev or {}).get(name) or {}).get("pass")


def _score(hooks, boundary, scene, height, manifest):
    ev = hooks["evaluate_floorplan"](cwb.eval_copy(boundary), scene, height, manifest)
    m = ev.get("metrics") or {}
    return ev, {"status": ev.get("status"), "failure_reasons": ev.get("failure_reasons"), "iou": m.get("iou"),
                "max_abs_wall_error_cm": m.get("max_abs_wall_error_cm"), "max_rel_wall_error_percent": m.get("max_rel_wall_error_percent"),
                "gate_1_5cm": _gate(ev, "custom_1_5cm_horizontal_diagnostic"),
                "gate_3pct": _gate(ev, "wall_length_3_percent_threshold_oracle_normalised"),
                "prediction_scale": ((ev.get("scale") or {}).get("prediction_scale") if isinstance(ev.get("scale"), dict) else None)}


def run_evaluation(budgets, source_index, source_dir, out_dir, index, hooks=None, events=None):
    """EVALUATION ONLY: reads frozen files back (hash-checked); never selects by truth."""
    import height_diagnosis
    hooks = {**rac._default_hooks(), **(hooks or {})}
    events = events if events is not None else []
    out_dir = Path(out_dir)
    if not (out_dir / "frozen_index.json").exists():
        raise RuntimeError("refusing to evaluate before results are frozen")
    scene_of = {b["label"]: b.get("scene") for b in budgets}
    by_label = {r["label"]: r for r in source_index["budgets"]}
    rows = []
    for rec in index["budgets"]:
        label = rec["label"]
        src = by_label[label]
        scene = hooks["load_scene"](scene_of[label])
        manifest = {"provenance": "frozen frame budget", "retained_capture_fraction": rec["retained_capture_fraction"],
                    "frames_registered": len(rec["source_indices"]), "frames_available": rec["frames_available"]}
        src_c = {c["candidate_id"]: c for c in src["candidates"]}
        for row in rec["candidates"]:
            cid = row["candidate_id"]
            sc = src_c[cid]
            out = {"label": label, "candidate_id": cid, "co_supported": row["co_supported_with_best"],
                   "cap_plane_separation": row.get("cap_plane_separation_unscaled"),
                   "wall_separation": row.get("separation_unscaled"), "wall_status": row.get("status"),
                   "about_one_unit": (row.get("cap_plane_separation_unscaled") is not None
                                      and UNIT_HEIGHT_BAND[0] <= row["cap_plane_separation_unscaled"] <= UNIT_HEIGHT_BAND[1]),
                   "oracle_label": "EVALUATOR-ORACLE: truth only scores; it selected neither candidate nor estimator"}
            if "boundary" not in sc["files"] or "wall_height" not in row["files"] or "height" not in sc["files"]:
                out["status"] = "not_scored"
                out["not_scored_reason"] = "no frozen boundary" if "boundary" not in sc["files"] else "no height estimate"
                rows.append(out)
                continue
            boundary = cwb.read_frozen(source_dir, sc["files"]["boundary"]["path"], sc["files"]["boundary"]["sha256"])
            cap = cwb.read_frozen(source_dir, sc["files"]["height"]["path"], sc["files"]["height"]["sha256"])
            diag = cwb.read_frozen(out_dir, row["files"]["wall_height_diagnostic_adapter"]["path"],
                                   row["files"]["wall_height_diagnostic_adapter"]["sha256"])
            strict = cwb.read_frozen(out_dir, row["files"]["wall_height_strict_adapter"]["path"],
                                     row["files"]["wall_height_strict_adapter"]["sha256"])
            events.append(("evaluate", label, cid))
            record = {}
            for name, height in (("old_cap_plane", height_diagnosis.evaluator_compatible(cap)),
                                 ("new_wall_strict_adapter", strict), ("new_wall_diagnostic_adapter", diag)):
                try:
                    ev, summary = _score(hooks, boundary, scene, height, manifest)
                    record[name] = summary
                    cwb.write_new_json(out_dir / "evaluation" / f"{label}_cand{cid}_{name}.json", {"evaluation": ev})
                except Exception as exc:
                    record[name] = {"status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}"}
            out.update(status="scored", **record)
            rows.append(out)
    summary = {"rows": rows, "gates": {"custom_cm": 1.5, "official_relative_percent": 3.0, "changed": False},
               "oracle_choices": cwb.ORACLE_CHOICES, "selection_by_truth": False,
               "note": "Every candidate with a frozen boundary is scored with both heights. No accuracy or completion claim."}
    cwb.write_new_json(out_dir / "evaluation_summary.json", summary)
    return summary


def results_markdown(summary):
    def f(v, fmt):
        return "" if v is None else format(v, fmt)
    L = ["# Wall-based height anchor vs cap-plane height: oracle scoring (evaluation only)", "",
         "Truth is used only to score; it selected neither a candidate nor an estimator. Gates unchanged: 1.5 cm and 3 %.",
         "new-wall rows use the DIAGNOSTIC adapter (separation passed on although the wall-height status is not complete);",
         "the strict adapter makes the evaluator refuse every non-complete height (column `strict`).", "",
         "| prediction | cand | about 1 unit | cap sep | wall sep | wall status | old max cm | old rel % | old 1.5cm | old 3% | "
         "new max cm | new rel % | new 1.5cm | new 3% | strict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        if r.get("status") != "scored":
            L.append(f"| {r['label']} | {r['candidate_id']} | {r['about_one_unit']} | {f(r['cap_plane_separation'], '.3f')} | "
                     f"{f(r['wall_separation'], '.3f')} | {r['wall_status']} | not scored: {r.get('not_scored_reason')} | | | | | | | | |")
            continue
        o, n, s = r["old_cap_plane"], r["new_wall_diagnostic_adapter"], r["new_wall_strict_adapter"]
        L.append(f"| {r['label']} | {r['candidate_id']} | {r['about_one_unit']} | {f(r['cap_plane_separation'], '.3f')} | "
                 f"{f(r['wall_separation'], '.3f')} | {r['wall_status']} | {f(o.get('max_abs_wall_error_cm'), '.2f')} | "
                 f"{f(o.get('max_rel_wall_error_percent'), '.2f')} | {o.get('gate_1_5cm')} | {o.get('gate_3pct')} | "
                 f"{f(n.get('max_abs_wall_error_cm'), '.2f')} | {f(n.get('max_rel_wall_error_percent'), '.2f')} | "
                 f"{n.get('gate_1_5cm')} | {n.get('gate_3pct')} | {s.get('status')} |")
    L += ["", "## Oracle choices", ""] + [f"- {c}" for c in summary["oracle_choices"]]
    return "\n".join(L) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW folder (refuses to overwrite)")
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    parser.add_argument("--no-evaluation", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output folder already exists: {args.output_dir}")
    source_dir = args.outputs_root / "axis-candidates-v1"
    source_index = json.loads((source_dir / "frozen_index.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True)
    index = run_extraction(source_index, source_dir, args.output_dir)
    if not args.no_evaluation:
        budgets = [{**b, "scene": str(args.outputs_root / b["scene"])} for b in rac.BUDGETS if b["label"] in LABELS]
        summary = run_evaluation(budgets, source_index, source_dir, args.output_dir, index)
        (args.output_dir / "RESULTS.md").write_text(results_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "budgets": len(index["budgets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
