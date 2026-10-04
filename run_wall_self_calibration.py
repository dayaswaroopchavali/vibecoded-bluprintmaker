"""Post-hoc method development: wall_self_calibration on the 60 FROZEN RGB (DA3) predictions, then the same chain and the same
evaluation stage as run_rgb_batch.py on the calibrated predictions.

LABEL: these 60 runs (rgb-batch-upright-v2 + rgb-batch-realistic-v1) were already seen (and the module was developed while
looking at some of them), so nothing here is a held-out result. A fresh-seed set is needed before any claim. No accuracy or
completion claim is made.

Stages (separate commands, serial semantics, resumable by refusing to overwrite):
  calibrate  TRUTH-FREE. For every run, every axis candidate that is co-supported or the vertical flag is calibrated
             (wall_self_calibration.calibrate); the calibrated prediction handed to the chain is the one of the vertical-flag
             axis (viable candidate with a cap pair and the largest cap support: truth-free). Writes calibrated predictions
             (no images) + per-axis reports, then freezes sha256 of everything (freeze_index.json).
  chain      TRUTH-FREE. Verifies the freeze, then the unchanged chain (run_axis_candidates.run_extraction, ray k = 2.0, current
             working-copy modules) on (a) the calibrated predictions and (b) the ORIGINAL predictions as a control with the same
             current modules, so before/after differ only by the calibration.
  evaluate   EVALUATION ONLY (reads private truth after the freeze): run_rgb_batch.evaluate_batch (same gates, same unique-top
             = selected, tie = failure rule) on both variants, plus oracle labels (vertical = axis within 15 deg of the true up).
  summarize  per-run and total before/after tables (RESULTS.md, tables.json).

    python run_wall_self_calibration.py calibrate --output-dir <NEW folder> [--workers 6]
    python run_wall_self_calibration.py chain     --output-dir <dir> [--workers 6]
    python run_wall_self_calibration.py evaluate  --output-dir <dir>
    python run_wall_self_calibration.py summarize --output-dir <dir>
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
BATCHES = ("rgb-batch-upright-v2", "rgb-batch-realistic-v1")
CODE_FILES = ("wall_self_calibration.py", "wall_extraction.py", "axis_candidates.py", "boundary_assembly.py", "fragment_association.py",
              "extract_planes.py", "height_diagnosis.py", "estimate_height.py", "geometry_orientation.py", "run_axis_candidates.py",
              "compare_wall_boundaries.py", "evaluate_floorplan.py", "evaluation.py", "run_rgb_batch.py", "diagnose_rgb_closure.py",
              "run_wall_self_calibration.py")
VARIANTS = ("control", "calibrated")
# stages that must stay free of truth (see test_wall_self_calibration.py)
TRUTH_FREE_FUNCTIONS = ("truth_free_axes", "calibrate_run", "stage_calibrate", "freeze_calibration", "chain_run", "stage_chain",
                        "verify_calibration_freeze")
PRIVATE_NAMES = ("private_ground_truth", "generator_private", "scene.json", "true_alignment", "line_offset_errors", "evaluate_batch")


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for chunk in iter(lambda: s.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes():
    return {n: (sha256_file(HERE / n) if (HERE / n).is_file() else None) for n in CODE_FILES}


def write_new_json(path, obj):
    data = json.dumps(obj, indent=1, allow_nan=False, default=_json_default).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as s:
        s.write(data)
    return hashlib.sha256(data).hexdigest()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean_floats(o):
    if isinstance(o, dict):
        return {str(k): _clean_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean_floats(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean_floats(o.tolist())
    if isinstance(o, np.generic):
        return _clean_floats(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ------------------------------------------------------------------------------------------------------------ run discovery
def list_runs(outputs_root=OUTPUTS, batches=BATCHES):
    import diagnose_rgb_closure as d
    runs = d.list_runs(outputs_root, batches)
    return sorted(runs, key=lambda r: (r["batch"], r["house"], r["path"]))


def indexed_runs(out_root):
    """The runs that stage `calibrate` processed (all 60 unless --limit was used)."""
    done = {r["run_id"] for r in read_json(Path(out_root) / "calibration_index.json")["runs"]}
    return [r for r in list_runs() if r["run_id"] in done]


def _rel(run):
    return Path("houses") / f"house_{run['house']}" / f"path_{run['path']:02d}"


def plan_spec(batch_root, run):
    cfg = read_json(Path(batch_root) / "batch_config.json")
    for spec in cfg["plan"]:
        if spec["house_seed"] == run["house"] and spec["path_index"] == run["path"]:
            return spec
    raise KeyError(run["run_id"])


def truth_free_axes(run_dir):
    """Frozen candidates of a run -> (metas, vertical_flag_id, ids to calibrate = co-supported or the flag). No truth."""
    import diagnose_rgb_closure as d
    run_dir = Path(run_dir)
    index = read_json(run_dir / "axis" / "frozen_index.json")["budgets"][0]
    cands = read_json(run_dir / "axis" / index["files"]["candidates"]["path"])
    evid = {c["id"]: c.get("evidence") for c in cands["candidates"]}
    metas = [d.candidate_meta(c, evid.get(c["candidate_id"])) for c in index["candidates"]]
    flag = d.vertical_flag_id(metas)
    ids = sorted({m["id"] for m in metas if m.get("co_supported_with_best")} | ({flag} if flag is not None else set()))
    return metas, flag, ids


# ------------------------------------------------------------------------------------------------------------ calibrate stage
def calibrate_run(task):
    """Worker: calibrate every selected axis of one run; write the calibrated prediction of the vertical-flag axis."""
    import diagnose_rgb_closure as d
    import wall_self_calibration as wsc
    run, out_root = task["run"], Path(task["out"])
    t0 = time.time()
    rec = {"run_id": run["run_id"], "batch": run["batch"], "house": run["house"], "path": run["path"], "status": "ok", "axes": []}
    try:
        run_dir = Path(run["run_dir"])
        metas, flag, ids = truth_free_axes(run_dir)
        rec["vertical_flag_id"] = flag
        pred = d._load_npz(run_dir / "da3" / "prediction.npz")
        rec["input_prediction_sha256"] = sha256_file(run_dir / "da3" / "prediction.npz")
        calibrated = None
        for cid in ids:
            meta = next(m for m in metas if m["id"] == cid)
            out, rep = wsc.calibrate(pred, meta["axis_world"], verify=True)
            rep_path = out_root / "calibration" / run["batch"] / f"house_{run['house']}" / f"path_{run['path']:02d}" / f"axis{cid}.json"
            rec["axes"].append({"candidate_id": cid, "is_vertical_flag": cid == flag, "co_supported": bool(meta.get("co_supported_with_best")),
                                "status": rep["status"], "reasons": rep.get("reasons"),
                                "report": str(rep_path.relative_to(out_root)).replace("\\", "/"),
                                "layered_before": (rep.get("verification") or {}).get("layered_before"),
                                "layered_after": (rep.get("verification") or {}).get("layered_after"),
                                "lines_before": (rep.get("verification") or {}).get("lines_before"),
                                "lines_after": (rep.get("verification") or {}).get("lines_after"),
                                "cost_drop_fraction": (rep.get("info") or {}).get("cost_drop_fraction")})
            write_new_json(rep_path, _clean_floats(rep))
            if cid == flag:
                calibrated = out
                rec["flag_status"] = rep["status"]
        rec["flag_applied"] = bool(calibrated is not None and rec.get("flag_status") == "calibrated")
        dest = out_root / "variants" / "calibrated" / run["batch"] / _rel(run) / "da3"
        dest.mkdir(parents=True, exist_ok=True)
        base = calibrated if calibrated is not None else pred
        keep = {k: base[k] for k in ("depth", "confidence", "extrinsics", "intrinsics", "source_indices") if k in base}
        with open(dest / "prediction.npz", "xb") as s:
            np.savez_compressed(s, **keep)
        rec["calibrated_prediction_sha256"] = sha256_file(dest / "prediction.npz")
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback_tail=traceback.format_exc()[-1500:])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def stage_calibrate(out_root, workers=6, limit=None):
    out_root = Path(out_root)
    if out_root.exists():
        raise FileExistsError(f"output folder already exists (new folders only): {out_root}")
    runs = list_runs()
    if limit:
        runs = runs[:limit]
    out_root.mkdir(parents=True)
    tasks = [{"run": r, "out": str(out_root)} for r in runs]
    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(calibrate_run, tasks):
            results.append(rec)
            print(f"[{_utc()}] {rec['run_id']} {rec['status']} flag={rec.get('vertical_flag_id')} applied={rec.get('flag_applied')} "
                  f"axes={[(a['candidate_id'], a['status']) for a in rec['axes']]} ({rec['seconds']} s)", flush=True)
    write_new_json(out_root / "calibration_index.json", {"written_utc": _utc(), "runs": results,
                                                         "label": "post-hoc method development on frozen predictions; not held-out"})
    freeze_calibration(out_root, results)
    return results


def freeze_calibration(out_root, results):
    """sha256 of every calibrated prediction and report + the input predictions + the code, before the chain or any truth."""
    out_root = Path(out_root)
    files = {}
    for p in sorted((out_root / "variants" / "calibrated").rglob("*.npz")) + sorted((out_root / "calibration").rglob("*.json")):
        files[str(p.relative_to(out_root)).replace("\\", "/")] = sha256_file(p)
    files["calibration_index.json"] = sha256_file(out_root / "calibration_index.json")
    freeze = {"frozen_utc": _utc(), "files": files, "input_prediction_sha256": {r["run_id"]: r.get("input_prediction_sha256") for r in results},
              "code_sha256": code_hashes(), "note": "written before the chain and before any evaluation / private scene read"}
    write_new_json(out_root / "calibration_freeze.json", freeze)
    return freeze


def verify_calibration_freeze(out_root):
    out_root = Path(out_root)
    freeze = read_json(out_root / "calibration_freeze.json")
    for rel, digest in freeze["files"].items():
        if sha256_file(out_root / rel) != digest:
            raise RuntimeError(f"frozen file changed after freezing: {rel}")
    return freeze


# ------------------------------------------------------------------------------------------------------------ chain stage
def _link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def chain_run(task):
    """Worker: the unchanged truth-free chain for one run of one variant ('calibrated' or 'control') + freeze.json."""
    import run_rgb_batch as rrb
    run, out_root, variant = task["run"], Path(task["out"]), task["variant"]
    t0 = time.time()
    rec = {"run_id": run["run_id"], "variant": variant, "status": "ok"}
    try:
        batch_root = out_root / "variants" / variant / run["batch"]
        spec = plan_spec(OUTPUTS / run["batch"], run)
        run_dir = batch_root / _rel(run)
        pred = run_dir / "da3" / "prediction.npz"
        if variant == "control":
            _link_or_copy(Path(run["run_dir"]) / "da3" / "prediction.npz", pred)
        (run_dir / "receipts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(run["run_dir"]) / "receipts" / "inference.json", run_dir / "receipts" / "inference.json")
        rec["prediction_sha256"] = sha256_file(pred)
        r = rrb.stage_chain(run_dir, spec, rrb._default_hooks())
        rec["chain"] = {"status": r["status"], "details": r["details"]}
        rrb.stage_freeze(run_dir, spec, code_hashes())
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback_tail=traceback.format_exc()[-1500:])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def stage_chain(out_root, workers=6, variants=VARIANTS):
    out_root = Path(out_root)
    freeze = verify_calibration_freeze(out_root)
    runs = indexed_runs(out_root)
    done = {}
    for variant in variants:
        for batch in BATCHES:
            root = out_root / "variants" / variant / batch
            root.mkdir(parents=True, exist_ok=True)
            if not (root / "batch_config.json").exists():
                shutil.copy2(OUTPUTS / batch / "batch_config.json", root / "batch_config.json")
        tasks = [{"run": r, "out": str(out_root), "variant": variant} for r in runs]
        recs = []
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for rec in pool.map(chain_run, tasks):
                recs.append(rec)
                print(f"[{_utc()}] {variant} {rec['run_id']} {rec['status']} {rec.get('error') or ''} ({rec['seconds']} s)", flush=True)
        done[variant] = recs
    write_new_json(out_root / f"chain_index_{int(time.time())}.json", {"written_utc": _utc(), "variants": done,
                                                                         "code_sha256_at_chain": code_hashes(),
                                                                         "calibration_freeze_code_sha256": freeze["code_sha256"]})
    return done


# ------------------------------------------------------------------------------------------------------------ evaluation (reads private truth)
def _prepare_private(out_root, variant):
    """EVALUATION ONLY: bring the generator-side files next to the variant runs (copies; the originals are never touched)."""
    for run in indexed_runs(out_root):
        src = Path(run["run_dir"])
        dst = Path(out_root) / "variants" / variant / run["batch"] / _rel(run)
        for rel in (Path("capture") / "private_ground_truth", Path("generator_private")):
            if (dst / rel).exists():
                continue
            if (src / rel).is_dir():
                shutil.copytree(src / rel, dst / rel)


def stage_evaluate(out_root):
    import run_rgb_batch as rrb
    out_root = Path(out_root)
    verify_calibration_freeze(out_root)
    summary = {}
    for variant in VARIANTS:
        _prepare_private(out_root, variant)
        for batch in BATCHES:
            root = out_root / "variants" / variant / batch
            partial = len(indexed_runs(out_root)) < 60
            if not any(r["batch"] == batch for r in indexed_runs(out_root)):
                continue
            summary[f"{variant}/{batch}"] = rrb.evaluate_batch(root, allow_partial=partial)
    write_new_json(out_root / f"evaluate_index_{int(time.time())}.json", {"written_utc": _utc(), "summary": summary,
                                                                            "code_sha256_at_evaluate": code_hashes()})
    return summary


# ------------------------------------------------------------------------------------------------------------ oracle labels + tables
VERTICAL_DEG, HORIZONTAL_DEG = 15.0, 75.0


def variant_run_metrics(out_root, variant, run, batch_results):
    """Everything about one run of one variant (EVALUATION ONLY: needs the true up for the vertical / horizontal labels)."""
    import diagnose_rgb_closure as d
    out_root = Path(out_root)
    batch_root = out_root / "variants" / variant / run["batch"]
    run_dir = batch_root / _rel(run)
    index = read_json(run_dir / "axis" / "frozen_index.json")["budgets"][0]
    cands = index["candidates"]
    cfile = read_json(run_dir / "axis" / index["files"]["candidates"]["path"])
    evid = {c["id"]: c.get("evidence") for c in cfile["candidates"]}
    metas = [d.candidate_meta(c, evid.get(c["candidate_id"])) for c in cands]
    pred = d._load_npz(run_dir / "da3" / "prediction.npz")
    poses = read_json(next((run_dir / "capture" / "private_ground_truth").glob("path_*.json")))["poses"]
    scene = read_json(run_dir / "capture" / "private_ground_truth" / "scene.json")
    align = d.true_alignment(pred["extrinsics"], poses, pred.get("source_indices"))
    centres = d.camera_centres(pred["extrinsics"])
    rows = {r["candidate_id"]: r for r in batch_results["candidates"] if r["run_id"] == run_run_id(run)}
    flag = d.vertical_flag_id(metas)
    per = []
    for c, meta in zip(cands, metas):
        files = c.get("files") or {}
        walls = read_json(run_dir / "axis" / files["walls_summary"]["path"]) if "walls_summary" in files else None
        boundary = read_json(run_dir / "axis" / files["boundary"]["path"]) if "boundary" in files else None
        an = d.analyze_candidate(walls, boundary, centres)
        ang = d.axis_angle_to_true_up(meta["axis_world"], align)
        kind = "vertical" if ang <= VERTICAL_DEG else "horizontal" if ang >= HORIZONTAL_DEG else "oblique"
        err = None
        if kind == "vertical" and walls and walls.get("status") == "ok":
            e = d.line_offset_errors(d.lines_compact(walls), align, scene)
            err = {"median_abs_cm": e["median_abs_offset_error_cm"], "max_abs_cm": e["max_abs_offset_error_cm"],
                   "walls_with_a_line": e["walls_with_a_line"], "n_matched": e["n_matched"]}
        row = rows.get(c["candidate_id"], {})
        per.append({"id": c["candidate_id"], "kind": kind, "angle_to_true_up_deg": ang, "closed": bool(an.get("closed")),
                    "walls_status": an.get("walls_status"), "lines": (an.get("line_stats") or {}).get("n_lines"),
                    "layered_drift_pairs": (an.get("layered") or {}).get("n_layered_drift_scale"),
                    "layered_pairs": (an.get("layered") or {}).get("n_layered"), "offset_error": err,
                    "eval_status": row.get("status"), "iou": row.get("iou"), "max_abs_wall_error_cm": row.get("max_abs_wall_error_cm"),
                    "max_rel_wall_error_percent": row.get("max_rel_wall_error_percent"), "gate_all": row.get("gate_all"),
                    "top": row.get("truth_free_top_ranked"), "vertical_flag": c["candidate_id"] == flag, "polygon_status": row.get("polygon_status")})
    run_rec = next(r for r in batch_results["runs"] if r["run_id"] == run_run_id(run))
    return {"run_id": run["run_id"], "batch": run["batch"], "house": run["house"], "path": run["path"], "vertical_flag_id": flag,
            "candidates": per, "run": run_rec,
            "alignment": {k: v for k, v in align.items() if k not in ("R", "t")}}


def run_run_id(run):
    return f"house{run['house']}_path{run['path']}"


def collect_tables(out_root):
    out_root = Path(out_root)
    tables = {}
    for variant in VARIANTS:
        per_batch = {}
        for batch in BATCHES:
            root = out_root / "variants" / variant / batch
            if not list(root.glob("results_*")):
                continue
            res = sorted(root.glob("results_*"))[-1]
            per_batch[batch] = {"runs": read_json(res / "runs.json"), "candidates": read_json(res / "candidates.json")}
        tables[variant] = {}
        for run in indexed_runs(out_root):
            tables[variant][run["run_id"]] = variant_run_metrics(out_root, variant, run, per_batch[run["batch"]])
    return tables


def _med(v):
    v = [x for x in v if x is not None]
    return float(statistics.median(v)) if v else None


def _selected(m):
    sel = [c for c in m["candidates"] if c["top"]]
    return sel[0] if len(sel) == 1 else None


def per_run_row(m):
    """One variant of one run reduced to the requested numbers."""
    cands = m["candidates"]
    vert = [c for c in cands if c["kind"] == "vertical"]
    hor = [c for c in cands if c["kind"] == "horizontal"]
    flag = next((c for c in cands if c["vertical_flag"]), None)
    sel = _selected(m)
    tops = [c for c in cands if c["top"]]
    if sel is not None:
        top_status = f"{sel['eval_status']}/{sel['kind']}"
    else:
        top_status = f"tie({len(tops)})" if len(tops) > 1 else "none"
    sel_run = m["run"]["selected"]
    gates = None if not sel_run else {"1.5cm": sel_run.get("gate_1_5cm"), "3pct": sel_run.get("gate_3pct"),
                                      "ceiling": sel_run.get("gate_ceiling_height_1_5cm")}
    errs = [c["offset_error"]["median_abs_cm"] for c in vert if c["offset_error"] and c["offset_error"]["median_abs_cm"] is not None]
    return {"run_id": m["run_id"], "candidates": len(cands), "vertical_candidates": len(vert),
            "vflag_kind": flag["kind"] if flag else None,
            "lines_per_vertical_median": _med([c["lines"] for c in vert if c["lines"] is not None]),
            "layered_drift_pairs_per_vertical_median": _med([c["layered_drift_pairs"] for c in vert if c["layered_drift_pairs"] is not None]),
            "layered_pairs_vertical_total": sum(c["layered_drift_pairs"] or 0 for c in vert),
            "closed_vertical": sum(c["closed"] for c in vert), "closed_horizontal": sum(c["closed"] for c in hor),
            "closed_other": sum(c["closed"] for c in cands if c["kind"] == "oblique"),
            "scoreable": sum(1 for c in cands if c["eval_status"] == "scored"),
            "top_status": top_status, "top_kind": sel["kind"] if sel else None,
            "iou": sel["iou"] if sel else None, "max_cm": sel["max_abs_wall_error_cm"] if sel else None,
            "max_pct": sel["max_rel_wall_error_percent"] if sel else None,
            "gates": gates,
            "run_pass": m["run"]["run_pass"], "fail_reasons": m["run"]["fail_reasons"],
            "wall_offset_error_vertical_median_cm": _med(errs),
            "vflag": None if not flag else {"lines": flag["lines"], "layered_drift_pairs": flag["layered_drift_pairs"], "closed": flag["closed"],
                                            "offset_median_cm": (flag["offset_error"] or {}).get("median_abs_cm")}}


def totals(rows):
    def s(key):
        return sum(r[key] or 0 for r in rows)

    sel = [r for r in rows if r["top_kind"] is not None]
    return {"runs": len(rows), "lines_per_vertical_median_of_runs": _med([r["lines_per_vertical_median"] for r in rows]),
            "layered_drift_pairs_per_vertical_median_of_runs": _med([r["layered_drift_pairs_per_vertical_median"] for r in rows]),
            "layered_pairs_vertical_total": s("layered_pairs_vertical_total"),
            "closed_vertical": s("closed_vertical"), "closed_horizontal": s("closed_horizontal"),
            "runs_with_a_closed_vertical": sum(1 for r in rows if r["closed_vertical"]),
            "scoreable_candidates": s("scoreable"), "runs_with_unique_top": len(sel),
            "runs_top_is_vertical": sum(1 for r in rows if r["top_kind"] == "vertical"),
            "runs_top_scored": sum(1 for r in rows if (r["top_status"] or "").startswith("scored")),
            "iou_median_selected": _med([r["iou"] for r in rows]), "max_cm_median_selected": _med([r["max_cm"] for r in rows]),
            "max_pct_median_selected": _med([r["max_pct"] for r in rows]),
            "wall_offset_error_vertical_median_cm_of_runs": _med([r["wall_offset_error_vertical_median_cm"] for r in rows]),
            "runs_pass_all_gates": sum(1 for r in rows if r["run_pass"]),
            "runs_gate_1_5cm": sum(1 for r in rows if r["gates"] and r["gates"]["1.5cm"]),
            "runs_gate_3pct": sum(1 for r in rows if r["gates"] and r["gates"]["3pct"]),
            "vflag_runs_with_vertical_axis": sum(1 for r in rows if r["vflag_kind"] == "vertical")}


def _f(v, spec=".2f"):
    return "" if v is None else (format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))


def stage_summarize(out_root):
    out_root = Path(out_root)
    tables = collect_tables(out_root)
    idx = read_json(out_root / "calibration_index.json")
    cal = {r["run_id"]: r for r in idx["runs"]}
    frozen_orig = {}
    for batch in BATCHES:
        res = sorted((OUTPUTS / batch).glob("results_*"))[-1]
        for r in read_json(res / "runs.json"):
            frozen_orig[f"{batch}:{r['run_id']}"] = r
    rows = {v: [per_run_row(tables[v][k]) for k in sorted(tables[v])] for v in VARIANTS}
    tot = {v: totals(rows[v]) for v in VARIANTS}
    write_new_json(out_root / "tables.json", {"label": "post-hoc method development on frozen, already-seen predictions; NOT held-out",
                                              "per_run": rows, "totals": tot, "oracle_threshold_deg": {"vertical": VERTICAL_DEG,
                                                                                                          "horizontal": HORIZONTAL_DEG}})
    flagstat = {}
    for rid, r in cal.items():
        flagstat[r.get("flag_status")] = flagstat.get(r.get("flag_status"), 0) + 1
    md = ["# wall_self_calibration on the 60 frozen RGB predictions: before / after", "",
          "**Post-hoc method development.** The 60 runs were already seen (and the module was developed while looking at some of them), so "
          "this is NOT a held-out result. A fresh-seed set is needed before any claim. No accuracy or completion claim. Gates unchanged "
          "(1.5 cm, 3 %, ceiling 1.5 cm; selected = unique truth-free top-ranked candidate, a tie fails the run).", "",
          "before = control: the original frozen predictions through the SAME chain with the CURRENT working-copy modules; "
          "after = calibrated predictions (vertical-flag axis calibration) through that chain. Vertical = axis within 15 deg of the true up "
          "(evaluator-only); horizontal >= 75 deg.", "",
          f"Calibration status of the vertical-flag axis over 60 runs: {flagstat}", "", "## Totals", "",
          "| metric | before (control) | after (calibrated) |", "|---|---|---|"]
    for k in tot["control"]:
        md.append(f"| {k} | {_f(tot['control'][k], '.3f')} | {_f(tot['calibrated'][k], '.3f')} |")
    md += ["", "## Per run (before -> after)", "",
           "| run | flag status | vflag vertical (b/a) | lines/vert (b->a) | layered/vert total (b->a) | closed vert (b->a) | closed horiz (b->a) | scoreable (b->a) | top status (b->a) | IoU (b->a) | max cm (b->a) | max % (b->a) | wall err cm (b->a) | pass (b->a) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, a in zip(rows["control"], rows["calibrated"]):
        r = cal.get(b["run_id"], {})
        md.append(f"| {b['run_id']} | {r.get('flag_status')} | {b['vflag_kind']}/{a['vflag_kind']} | {_f(b['lines_per_vertical_median'], '.0f')}->{_f(a['lines_per_vertical_median'], '.0f')} | "
                  f"{b['layered_pairs_vertical_total']}->{a['layered_pairs_vertical_total']} | {b['closed_vertical']}->{a['closed_vertical']} | "
                  f"{b['closed_horizontal']}->{a['closed_horizontal']} | {b['scoreable']}->{a['scoreable']} | {b['top_status']}->{a['top_status']} | "
                  f"{_f(b['iou'])}->{_f(a['iou'])} | {_f(b['max_cm'], '.1f')}->{_f(a['max_cm'], '.1f')} | {_f(b['max_pct'])}->{_f(a['max_pct'])} | "
                  f"{_f(b['wall_offset_error_vertical_median_cm'], '.1f')}->{_f(a['wall_offset_error_vertical_median_cm'], '.1f')} | "
                  f"{b['run_pass']}->{a['run_pass']} |")
    md += ["", "Frozen original results (earlier evaluation, earlier module state) run_pass count: "
           f"{sum(1 for v in frozen_orig.values() if v['run_pass'])} of {len(frozen_orig)}."]
    (out_root / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "chain", "evaluate", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--output-dir", type=Path, required=True)
        if name in ("calibrate", "chain"):
            p.add_argument("--workers", type=int, default=6)
        if name == "calibrate":
            p.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        stage_calibrate(a.output_dir, a.workers, a.limit)
    elif a.cmd == "chain":
        stage_chain(a.output_dir, a.workers)
    elif a.cmd == "evaluate":
        print(json.dumps(stage_evaluate(a.output_dir), indent=1, default=_json_default)[:3000])
    else:
        print(json.dumps(stage_summarize(a.output_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
