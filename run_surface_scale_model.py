"""CLI for surface_scale_model.py (truth-free correction) plus a clearly separated EVALUATION-ONLY oracle comparison.

  estimate     --prediction P.npz --output-dir NEW [--axis x,y,z ...]   truth-free: corrected npz + report JSON/markdown per axis
  frozen-six   --output-dir NEW                                         truth-free: the six frozen DA3 predictions of depth_bias_analysis,
                                                                         one corrected npz per co-supported frozen axis hypothesis;
                                                                         writes frozen_corrections.json (sha256) BEFORE anything else
  oracle-eval  --corrections DIR --output-dir NEW                       EVALUATION ONLY, after freezing: ORACLE-labelled before/after
                                                                         depth bias, ceiling-normalised wall-length ratio and downstream
                                                                         wall-boundary result. Nothing is tuned against it.

Output folders must be NEW; existing files are never overwritten. Estimation functions never see a scene, pose truth or simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import surface_scale_model as ssm

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
FROZEN_SIX = [("a_linspace36", "wall-boundaries-v1"), ("b_stride3_40", "wall-boundaries-v1"), ("c_stride2_60_DIFFERENT_BUDGET", "wall-boundaries-v1"),
              ("d_textured40", "wall-boundaries-v1"), ("path_01_36", "wall-boundary-paths-01-02-v1"), ("path_02_36", "wall-boundary-paths-01-02-v1")]
EVALUATION_ONLY_FUNCTIONS = ("oracle_vertical_like", "oracle_depth_metrics", "run_oracle_evaluation", "oracle_markdown")
ORACLE_CHOICES = [
    "ORACLE: which corrected hypothesis feeds the downstream rerun = the co-supported axis within 15 deg of the TRUE up seen through a "
    "similarity fitted from the prediction's camera centres to the TRUE camera centres",
    "ORACLE: TRUE per-pixel z-depth, TRUE surface labels and TRUE intrinsics (private scene + poses) in depth_bias_analysis helpers",
    "ORACLE: the downstream wall-boundary scoring (mirroring, alignment, edge matching) is the frozen evaluator's",
    "No parameter of surface_scale_model was chosen with these numbers; they are reported once, as produced.",
]


# ------------------------------------------------------------------------------------------------------------ io helpers
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_json(path: Path, obj) -> str:
    data = json.dumps(ssm.json_safe(obj), indent=1, allow_nan=False).encode("utf-8")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def write_new_npz(path: Path, arrays: dict) -> str:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in arrays.items() if isinstance(v, np.ndarray)})
    return sha256_file(path)


def parse_axis(text: str):
    v = [float(x) for x in text.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("--axis needs x,y,z")
    return v


# ------------------------------------------------------------------------------------------------------------ truth-free part
def correct_and_write(prediction_path, axes, out_dir: Path, label: str, **kwargs) -> dict:
    """Truth-free. One corrected npz + report per axis hypothesis (the axis-independent stages run once)."""
    pred = ssm.load_prediction(prediction_path)
    results = ssm.correct_prediction_axes(pred, [a["axis_world"] for a in axes], **kwargs)
    record = {"label": label, "input_prediction": str(prediction_path), "input_sha256": sha256_file(prediction_path), "hypotheses": []}
    for a, (corrected, report) in zip(axes, results):
        k = a["hypothesis_index"]
        npz = Path(out_dir) / label / f"corrected_hyp{k}.npz"
        rep = Path(out_dir) / label / f"report_hyp{k}.json"
        record["hypotheses"].append({"hypothesis_index": k, "axis_world": a["axis_world"], "npz": str(npz), "npz_sha256": write_new_npz(npz, corrected),
                                     "report": str(rep), "report_sha256": write_new_json(rep, report),
                                     "poses_identical": bool(np.array_equal(corrected["extrinsics"], pred["extrinsics"])),
                                     "summary": report["summary"], "not_identifiable": report["not_identifiable"]})
    return record


def frozen_axes(label: str, root: str):
    """Co-supported axis hypotheses of a frozen budget (truth-free artifacts of compare_wall_boundaries)."""
    base = OUTPUTS / root
    index = json.loads((base / "frozen_index.json").read_text(encoding="utf-8"))
    rec = next(b for b in index["budgets"] if b["label"] == label)
    return rec["prediction"], [{"hypothesis_index": h["hypothesis_index"], "axis_world": h["axis_world"]} for h in rec["hypotheses"]
                               if h.get("status") == "ran"]


def own_axes(prediction_path):
    """Co-supported structural-axis hypotheses from the cloud itself (extract_planes + geometry_orientation; no camera information)."""
    import extract_planes
    import geometry_orientation
    pred = ssm.load_prediction(prediction_path)
    pts = extract_planes.points_from_prediction(pred)
    axes = geometry_orientation.axis_hypotheses(extract_planes.extract(pts), pts)
    return [{"hypothesis_index": k, "axis_world": h["axis_world"]} for k, h in enumerate(axes.get("hypotheses", []) or [])
            if h.get("co_supported_with_best")]


def parameter_table(report: dict) -> list[str]:
    L = []
    sc = report.get("frame_scales") or {}
    if sc.get("status") == "ok":
        u = np.array(sc["log_scale_gauged"]); sd = np.array(sc["log_scale_sd_leave_one_view_out"], float); ap = np.array(sc["applied"])
        L.append(f"- frame scales: {len(u)} frames, sd of the estimated log-scale {100*np.std(u):.2f} %, median leave-one-view-out sd {100*np.nanmedian(sd):.2f} %, "
                 f"applied to {int(ap.sum())}/{len(ap)} frames (rest not identifiable), depth-vs-pose unit log-factor {100*sc['depth_vs_pose_unit_log_factor']:+.2f} % (NOT applied)")
    else:
        L.append(f"- frame scales: {sc.get('status', 'not run')} {sc.get('reason', '')}")
    f = report.get("intrinsics") or {}
    if f.get("status") == "ok":
        for name, est, sd in zip(f["parameters"], f["estimate"], f["sd_leave_one_view_out"]):
            L.append(f"- intrinsics {name}: {est:+.5f} +- {sd:.5f} -> {'applied' if f['applied'][name] else 'NOT IDENTIFIABLE (not applied)'}")
    else:
        L.append(f"- intrinsics: {f.get('status', 'not run')} {f.get('reason', '')}")
    for it in (report.get("surface_gains") or {}).get("iterations", []):
        b = it.get("baseline")
        L.append(f"- surface: status {it.get('status')}, wall baseline {b}" + (f", reason {it.get('reason')}" if it.get("reason") else ""))
        for g in it.get("gains", []):
            sw = g.get("sweep", {})
            L.append(f"  - cap plane {g['plane_index']} (angle to axis {g['angle_to_axis_deg']:.1f} deg): sweep eps {sw.get('eps')} sd {sw.get('sd_leave_one_view_out')}, "
                     f"relative-to-walls {g.get('eps_relative_to_walls')} sd {g.get('sd_relative')} -> {g['status']}"
                     + (f" ({g['reason']})" if g.get("reason") else "") + f"; seam cross-check {g.get('seam')}")
    return L


def markdown_estimate(record: dict, reports: dict) -> str:
    L = [f"# Truth-free depth correction: {record['label']}", "", f"input: {record['input_prediction']} (sha256 {record['input_sha256'][:12]})", ""]
    for h in record["hypotheses"]:
        L += [f"## axis hypothesis {h['hypothesis_index']} {np.round(h['axis_world'], 3).tolist()}", "", f"- corrected npz: {h['npz']} (poses identical: {h['poses_identical']})",
              f"- summary: {h['summary']}", f"- not identifiable: {h['not_identifiable'] or 'none'}"] + parameter_table(reports[h["hypothesis_index"]]) + [""]
    L += ["## Limitations", "", "- Offsets are relative to the wall-like baseline; global depth scale is left untouched.",
          "- A parameter is applied only if its leave-one-view-out sd <= its magnitude; many small corrections therefore stay unapplied.",
          "- No accuracy or completion claim; this is a consistency correction whose effect is judged elsewhere."]
    return "\n".join(L)


def cmd_estimate(args) -> int:
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    axes = [{"hypothesis_index": k, "axis_world": a} for k, a in enumerate(args.axis)] if args.axis else own_axes(args.prediction)
    if not axes:
        raise SystemExit("no axis hypotheses (give --axis or use a prediction with co-supported hypotheses)")
    label = Path(args.prediction).stem
    record = correct_and_write(args.prediction, axes, out, label)
    reports = {h["hypothesis_index"]: json.loads(Path(h["report"]).read_text(encoding="utf-8")) for h in record["hypotheses"]}
    write_new_json(out / "manifest.json", record)
    with open(out / "report.md", "x", encoding="utf-8") as f:
        f.write(markdown_estimate(record, reports))
    print(out.resolve())
    return 0


def cmd_frozen_six(args) -> int:
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    records, md = [], ["# Truth-free correction of the six frozen DA3 predictions", "",
                       "Estimation only: no scene, truth or simulator is read in this command. Axis hypotheses are the frozen, truth-free ones.", ""]
    for label, root in FROZEN_SIX:
        pred_path, axes = frozen_axes(label, root)
        print("correcting", label, "axes", [a["hypothesis_index"] for a in axes], flush=True)
        rec = correct_and_write(pred_path, axes, out, label)
        rec["frozen_root"] = root
        records.append(rec)
        reports = {h["hypothesis_index"]: json.loads(Path(h["report"]).read_text(encoding="utf-8")) for h in rec["hypotheses"]}
        md.append(markdown_estimate(rec, reports).replace("# ", "## ", 1))
    write_new_json(out / "frozen_corrections.json", {"model": ssm.MODEL, "truth_free": True, "corrections": records})
    with open(out / "report.md", "x", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(out.resolve())
    return 0


# ------------------------------------------------------------------------------------------------------------ EVALUATION ONLY
def oracle_vertical_like(sim, axes, max_deg=15.0):
    """ORACLE: index of the hypothesis axis closest to the TRUE up seen in the prediction frame (and its angle)."""
    up_pred = sim["rotation"].T @ np.array([0.0, 0.0, 1.0])
    best = None
    for a in axes:
        ax = np.asarray(a["axis_world"], float)
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(ax @ up_pred)) / np.linalg.norm(ax) / np.linalg.norm(up_pred)))))
        if best is None or ang < best[1]:
            best = (a["hypothesis_index"], ang)
    return best


def oracle_depth_metrics(dba, pred, true_z, labels, intr, sim, dims) -> dict:
    """ORACLE: per-surface depth offset (after ONE global scale), per-frame scale sd, fx/fy error and the ceiling-normalised wall ratios."""
    stats = dba.depth_bias_stats(true_z, pred["depth"], labels, pred["intrinsics"], intr["fx"], intr["fy"])
    pts, labs = dba.unproject_world(pred["depth"], pred["intrinsics"], pred["extrinsics"], labels, dba.CLOUD_STRIDE, 0.0, ~dba.edge_mask(labels))
    ext = dba.cloud_extents(dba.apply_similarity(sim, pts), labs, dims)
    eff = stats["effects_percent"]
    return {"s_g": stats["global_scale_s_g_median"], "floor_offset_percent_vs_wall": eff["surface_offset_percent_vs_wall"].get("floor"),
            "ceiling_offset_percent_vs_wall": eff["surface_offset_percent_vs_wall"].get("ceiling"),
            "per_frame_scale_sd_percent_adjusted": eff["per_frame_scale_sd_percent"], "per_frame_scale_sd_percent_raw": stats["per_frame_scale"]["sd_percent_of_s_g"],
            "fx_pred_over_true_percent": stats["intrinsics"]["fx_ratio_pred_over_true_mean_percent"],
            "fy_pred_over_true_percent": stats["intrinsics"]["fy_ratio_pred_over_true_mean_percent"],
            "fx_over_fy_pred": stats["intrinsics"]["fx_over_fy_pred_mean"],
            "height_ratio_pred_over_true": ext.get("height_ratio_pred_over_true"),
            "x_wall_ratio_under_ceiling_norm": ext.get("length_ratio_x_walls_under_ceiling_norm"),
            "y_wall_ratio_under_ceiling_norm": ext.get("length_ratio_y_walls_under_ceiling_norm"),
            "mean_wall_ratio_under_ceiling_norm": dba.mean_wall_ratio(ext)}


def _downstream_rows(comparison: dict, frozen_axes_by_label: dict, sims: dict) -> list[dict]:
    rows = []
    index = comparison["frozen_index"]
    for rec in index["budgets"]:
        for h in rec.get("hypotheses", []):
            row = next((r for r in comparison["table"] if r["budget"] == rec["label"] and r["hypothesis"] == h["hypothesis_index"]), None)
            if row is None or h.get("status") != "ran":
                continue
            v = oracle_vertical_like(sims[rec["label"]], [{"hypothesis_index": h["hypothesis_index"], "axis_world": h["axis_world"]}])
            ev = row.get("evaluation") or {}
            rows.append({"label": rec["label"], "hypothesis": h["hypothesis_index"], "oracle_angle_to_true_up_deg": v[1],
                         "vertical_like": bool(v[1] <= 15.0), "boundary_status": row["status"], "edges": row.get("edges"),
                         "scored": ev.get("status"), "iou": ev.get("iou"), "max_abs_wall_error_cm": ev.get("max_abs_wall_error_cm"),
                         "max_rel_wall_error_percent": ev.get("max_rel_wall_error_percent"), "three_percent_pass": ev.get("three_percent_pass")})
    return rows


def run_oracle_evaluation(corrections_dir: Path, out_dir: Path) -> dict:
    """EVALUATION ONLY, after the corrections were frozen to disk (sha256 verified before any truth is read)."""
    import compare_wall_boundaries as cwb
    import depth_bias_analysis as dba
    import estimate_height
    corrections_dir, out_dir = Path(corrections_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    doc = json.loads((corrections_dir / "frozen_corrections.json").read_text(encoding="utf-8"))
    for rec in doc["corrections"]:
        for h in rec["hypotheses"]:
            if sha256_file(h["npz"]) != h["npz_sha256"]:
                raise RuntimeError(f"corrected prediction changed after freezing: {h['npz']}")
    cfgs = {c["label"]: c for c in dba.CONFIGS}
    cache, results, sims = {}, [], {}
    budgets_before, budgets_after = [], []
    for rec in doc["corrections"]:
        cfg = cfgs[rec["label"]]
        before = ssm.load_prediction(rec["input_prediction"])
        src = [int(i) for i in before.get("source_indices", np.arange(len(before["depth"])))]
        truth = dba.load_truth(OUTPUTS / cfg["scene"], OUTPUTS / cfg["poses"])
        n, h, w = before["depth"].shape
        true_z, labels, intr, true_centres = dba.render_true_depth(truth, src, w, h, dba._fov_from(truth), cache)
        sim = dba.similarity_from_centres(estimate_height.camera_centers(before["extrinsics"]), true_centres)
        sims[rec["label"]] = sim
        scene = truth["scene"]
        dims = (float(scene["width"]), float(scene["length"]), float(scene["height"]))
        pick = oracle_vertical_like(sim, rec["hypotheses"])
        chosen = next(hh for hh in rec["hypotheses"] if hh["hypothesis_index"] == pick[0])
        after = ssm.load_prediction(chosen["npz"])
        m_before = oracle_depth_metrics(dba, before, true_z, labels, intr, sim, dims)
        m_after = oracle_depth_metrics(dba, after, true_z, labels, intr, sim, dims)
        results.append({"label": rec["label"], "oracle_vertical_like_hypothesis": pick[0], "oracle_angle_to_true_up_deg": pick[1],
                        "before": m_before, "after": m_after, "estimated_parameters": {
                            "summary": chosen["summary"], "not_identifiable": chosen["not_identifiable"], "report": chosen["report"]}})
        scene_path = str(OUTPUTS / cfg["scene"])
        budgets_before.append({"label": rec["label"], "description": "uncorrected frozen prediction", "prediction": rec["input_prediction"],
                               "scene": scene_path, "frames_available": 120})
        budgets_after.append({"label": rec["label"], "description": "truth-free corrected prediction (oracle-chosen vertical-like hypothesis)",
                              "prediction": chosen["npz"], "scene": scene_path, "frames_available": 120})
        print("depth metrics done", rec["label"], flush=True)
    out_dir.mkdir(parents=True)
    down = {}
    for name, budgets in (("before", budgets_before), ("after", budgets_after)):
        print("downstream", name, flush=True)
        comp = cwb.run_comparison(budgets, out_dir / f"downstream_{name}")
        down[name] = _downstream_rows(comp, None, sims)
    result = {"model": "surface_scale_model_oracle_evaluation", "evaluation_only": True, "stage_complete": False, "oracle_choices": ORACLE_CHOICES,
              "predictions": results, "downstream": down,
              "limitations": ["ONE room (house 17); the six predictions share a scene and are not independent houses.",
                              "Depth metrics reuse depth_bias_analysis helpers (TRUE depth rendered by simulator.render, labels from the simulator).",
                              "'before' and 'after' downstream runs both use the CURRENT pipeline code on the uncorrected / corrected npz.",
                              "No accuracy or completion claim; nothing was tuned against these numbers."]}
    write_new_json(out_dir / "results.json", result)
    with open(out_dir / "reading.md", "x", encoding="utf-8") as f:
        f.write(oracle_markdown(result))
    return result


def _f(x, p=2, sign=False):
    return "-" if x is None else (f"{x:+.{p}f}" if sign else f"{x:.{p}f}")


def oracle_markdown(result: dict) -> str:
    L = ["# ORACLE-labelled before/after of the truth-free surface/scale correction", "",
         "EVALUATION-ONLY, after the corrections were frozen (sha256 verified). Nothing was tuned against these numbers; no accuracy or completion claim.", "",
         "## Oracle choices", ""] + [f"- {c}" for c in result["oracle_choices"]]
    L += ["", "## Depth bias and ceiling-normalised wall-length ratio (1.000 = no horizontal-vs-vertical anisotropy)", "",
          "| prediction | hyp (oracle vertical-like) | ceiling offset % before -> after | floor offset % before -> after | per-frame scale sd % before -> after | fx pred/true % before -> after | fy pred/true % before -> after | x-wall ratio before -> after | y-wall ratio before -> after | mean ratio before -> after |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["predictions"]:
        b, a = r["before"], r["after"]
        pair = lambda k, p=2, s=False: f"{_f(b[k], p, s)} -> {_f(a[k], p, s)}"
        L.append(f"| {r['label']} | {r['oracle_vertical_like_hypothesis']} | {pair('ceiling_offset_percent_vs_wall', 2, True)} | {pair('floor_offset_percent_vs_wall', 2, True)} | "
                 f"{pair('per_frame_scale_sd_percent_adjusted')} | {pair('fx_pred_over_true_percent', 2, True)} | {pair('fy_pred_over_true_percent', 2, True)} | "
                 f"{pair('x_wall_ratio_under_ceiling_norm', 4)} | {pair('y_wall_ratio_under_ceiling_norm', 4)} | {pair('mean_wall_ratio_under_ceiling_norm', 4)} |")
    L += ["", "## Estimated parameters (truth-free) behind the chosen hypothesis", ""]
    for r in result["predictions"]:
        L.append(f"- {r['label']}: {r['estimated_parameters']['summary']}; not identifiable: {len(r['estimated_parameters']['not_identifiable'])} items "
                 f"(report {r['estimated_parameters']['report']})")
    L += ["", "## Downstream wall-boundary result (frozen evaluator; ORACLE-labelled scoring)", "",
          "| prediction | run | hyp | vertical-like (oracle) | boundary status | edges | scored | IoU | max wall err cm | max rel % | 3% pass |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["predictions"]:
        for name in ("before", "after"):
            for row in (x for x in result["downstream"][name] if x["label"] == r["label"]):
                L.append(f"| {row['label']} | {name} | {row['hypothesis']} | {row['vertical_like']} | {row['boundary_status']} | {row['edges']} | {row['scored']} | "
                         f"{_f(row['iou'], 3)} | {_f(row['max_abs_wall_error_cm'], 1)} | {_f(row['max_rel_wall_error_percent'], 2)} | {row['three_percent_pass']} |")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in result["limitations"]]
    return "\n".join(L)


def cmd_oracle_eval(args) -> int:
    run_oracle_evaluation(Path(args.corrections), Path(args.output_dir))
    print(Path(args.output_dir).resolve())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("estimate"); p.add_argument("--prediction", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--axis", type=parse_axis, action="append", default=[]); p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("frozen-six"); p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=cmd_frozen_six)
    p = sub.add_parser("oracle-eval"); p.add_argument("--corrections", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=cmd_oracle_eval)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
