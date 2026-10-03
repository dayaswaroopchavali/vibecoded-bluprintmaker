"""POST-HOC chain variant: re-run ONLY the truth-free chain on the already-frozen RGB DA3 predictions with the CURRENT modules.

LABEL: POST-HOC VARIANT. The chain modules (boundary_assembly.py, estimate_height.py) were changed after the predeclared RGB batches
(rgb-batch-upright-v2, rgb-batch-realistic-v1) had been evaluated. This is NOT a replacement for those batches' predeclared results;
it answers one question: with everything else unchanged, does the current working-copy chain close polygons / make runs scoreable on the
same frozen predictions? Nothing is selected by truth; gates and the run-level selection rule (unique truth-free top rank; a tie at the top
is an ambiguity failure) are exactly those of run_rgb_batch.py.

Stages per batch (new folder, never overwritten):
  1. chain   (truth-free): planes -> axis_candidates -> height_diagnosis -> wall_extraction + assemble_boundary_ray (k=2.0) via
             run_rgb_batch.stage_chain / run_axis_candidates.run_extraction, reading the ORIGINAL frozen prediction in place.
  2. freeze  sha256 of the input prediction, every chain output and every chain module source -> freeze.json
  3. evaluate  ONLY after all runs are frozen: run_rgb_batch.evaluate_batch (unchanged evaluators and gates) against the original private scene.
  4. compare   per-run table original vs variant; truth-free axis-tie evidence report.

    python rerun_chain_variants.py --output-dir <NEW folder under outputs/sim-benchmark> [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import run_rgb_batch as rb

HERE = Path(__file__).resolve().parent
OUTPUTS = rb.OUTPUTS
BATCHES = ("rgb-batch-upright-v2", "rgb-batch-realistic-v1")
CHAIN_MODULES = ("run_axis_candidates.py", "compare_wall_boundaries.py", "axis_candidates.py", "height_diagnosis.py",
                 "wall_extraction.py", "boundary_assembly.py", "extract_planes.py", "estimate_height.py", "room_hypothesis.py",
                 "evaluate_floorplan.py", "run_rgb_batch.py")
LABEL = ("POST-HOC VARIANT: chain modules changed after seeing the predeclared batch results; not a replacement for them; "
         "no candidate selected by truth")


def module_hashes(names=CHAIN_MODULES, base: Path = HERE) -> dict:
    return {n: (rb.sha256_file(Path(base) / n) if (Path(base) / n).is_file() else None) for n in names}


def pinned_vs_current(cfg: dict, current: dict) -> dict:
    """Modules whose hash pinned in the original batch config differs from the current source (both hashes recorded)."""
    out = {}
    for name, pinned in cfg["code_sha256"].items():
        if name in current and pinned != current[name]:
            out[name] = {"pinned_in_original_batch": pinned, "current_working_copy": current[name]}
    return out


def init_variant_root(root: Path, batches=BATCHES, outputs: Path = OUTPUTS, current: dict | None = None) -> dict:
    """Create the NEW output folder. Refuses to touch an existing one. One batch-shaped sub-root per source batch."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"output folder already exists (never overwrite): {root}")
    current = current or module_hashes()
    root.mkdir(parents=True)
    summary = {"label": LABEL, "schema": "rerun_chain_variant_v1", "utc": rb._utc(), "current_module_sha256": current, "batches": {}}
    for b in batches:
        src = rb.read_json(Path(outputs) / b / "batch_config.json")
        diff = pinned_vs_current(src, current)
        cfg = dict(src)
        cfg["code_sha256"] = {k: current.get(k, v) for k, v in src["code_sha256"].items()}    # evaluate_batch only reads it for consistency
        cfg["variant_of"] = b
        cfg["variant_label"] = LABEL
        rb.write_new_json(root / b / "batch_config.json", cfg)
        summary["batches"][b] = {"original_pinned_code_sha256": src["code_sha256"], "modules_differing_from_pinned": diff}
    rb.write_new_json(root / "variant_config.json", summary)
    return summary


def source_run_dir(outputs: Path, batch: str, spec: dict) -> Path:
    return rb.run_dir_of(Path(outputs) / batch, spec)


def stage_variant_chain(variant_root: Path, batch: str, spec: dict, outputs: Path = OUTPUTS, hooks: dict | None = None) -> dict:
    """TRUTH-FREE: run the chain on the original frozen prediction (read in place), write to the new run folder."""
    run_dir = rb.run_dir_of(Path(variant_root) / batch, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    pred = source_run_dir(outputs, batch, spec) / "da3" / "prediction.npz"
    if not pred.is_file():
        raise FileNotFoundError(f"no frozen prediction: {pred}")
    sha = rb.sha256_file(pred)
    orig_freeze = rb.read_json(source_run_dir(outputs, batch, spec) / "freeze.json")
    pinned = orig_freeze["files"].get("da3/prediction.npz")
    if pinned is not None and pinned != sha:
        raise rb.CorruptRun(f"input prediction no longer matches its original freeze: {pred}")
    rb.write_new_json(run_dir / "inputs.json", {"prediction": str(pred), "prediction_sha256": sha, "matches_original_freeze": pinned == sha})
    hooks = dict(hooks or {})

    def run_extraction(budgets, out_dir):
        if "run_extraction" in hooks:
            return hooks["run_extraction"]([{**budgets[0], "prediction": str(pred)}], out_dir)
        import run_axis_candidates as rac
        return rac.run_extraction([{**budgets[0], "prediction": str(pred)}], out_dir)

    def write_reports(index, out_dir):
        if "write_reports" in hooks:
            return hooks["write_reports"](index, out_dir)
        import run_axis_candidates as rac
        return rac.write_reports(index, out_dir)

    return rb.stage_chain(run_dir, spec, {"run_extraction": run_extraction, "write_reports": write_reports})


def stage_variant_freeze(variant_root: Path, batch: str, spec: dict, code: dict) -> dict:
    """Hash input record + every chain output + module sources. Written once; evaluation refuses without it."""
    run_dir = rb.run_dir_of(Path(variant_root) / batch, spec)
    rels = ["inputs.json"] + [str(p.relative_to(run_dir)).replace("\\", "/") for p in sorted((run_dir / "axis").rglob("*")) if p.is_file()]
    inputs = rb.read_json(run_dir / "inputs.json")
    freeze = {"run_id": rb.run_id(spec["house_seed"], spec["path_index"]), "frozen_utc": rb._utc(), "label": LABEL,
              "files": rb._rel_hashes(run_dir, rels), "input_prediction_sha256": inputs["prediction_sha256"], "code_sha256": code,
              "reconstruction_inputs": ["frozen DA3 prediction (unchanged)"]}
    rb.write_new_json(run_dir / "freeze.json", freeze)
    return freeze


def _chain_one(args):
    variant_root, batch, spec, code = args
    try:
        rec = stage_variant_chain(Path(variant_root), batch, spec)
        stage_variant_freeze(Path(variant_root), batch, spec, code)
        return spec["run_id"], rec["status"], rec["details"].get("error")
    except Exception as exc:                                                    # report, never hide
        return spec["run_id"], "failed", f"{type(exc).__name__}: {exc}"


def run_chain_all(variant_root: Path, batches=BATCHES, workers: int = 4, code: dict | None = None) -> dict:
    code = code or module_hashes()
    jobs = []
    for b in batches:
        cfg = rb.read_json(Path(variant_root) / b / "batch_config.json")
        jobs += [(str(variant_root), b, spec, code) for spec in cfg["plan"]]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_chain_one, jobs))
    else:
        res = [_chain_one(j) for j in jobs]
    return {"runs": len(res), "ok": sum(1 for r in res if r[1] == "ok"), "failed": [r for r in res if r[1] != "ok"]}


# ------------------------------------------------------------------ evaluation (separate step; reads private truth only after all freezes)
def evaluate_variant(variant_root: Path, batch: str, outputs: Path = OUTPUTS, hooks: dict | None = None) -> dict:
    """Refuses unless every run of the batch is frozen; then copies generator-side QA / inference receipt (no truth) and runs the
    UNCHANGED run_rgb_batch.evaluate_batch with the original private scene substituted by a hook."""
    root = Path(variant_root) / batch
    cfg = rb.load_root(root)
    missing = [s["run_id"] for s in cfg["plan"] if not (rb.run_dir_of(root, s) / "freeze.json").is_file()]
    if missing:
        raise RuntimeError(f"refusing to evaluate: not frozen: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    for s in cfg["plan"]:
        rb.verify_freeze(rb.run_dir_of(root, s))
        src, dst = source_run_dir(outputs, batch, s), rb.run_dir_of(root, s)
        (dst / "generator_private").mkdir(exist_ok=True)
        (dst / "receipts").mkdir(exist_ok=True)
        shutil.copyfile(src / "generator_private" / "capture_qa.json", dst / "generator_private" / "capture_qa.json")
        shutil.copyfile(src / "receipts" / "inference.json", dst / "receipts" / "inference.json")

    def load_scene(path):
        rel = Path(path).relative_to(root)
        return json.loads((Path(outputs) / batch / rel).read_text(encoding="utf-8"))

    return rb.evaluate_batch(root, hooks={"load_scene": load_scene, **(hooks or {})})


# ------------------------------------------------------------------ comparison (post-evaluation, reporting only)
def closed_polygon(row: dict) -> bool:
    return row.get("polygon_status") == "complete_polygon_hypothesis"


def run_summary(run: dict, rows: list) -> dict:
    mine = [r for r in rows if r["run_id"] == run["run_id"]]
    tops = [r for r in mine if r["truth_free_top_ranked"]]
    if len(tops) > 1:
        top = f"tie({len(tops)})"
    elif len(tops) == 1:
        top = tops[0]["status"]
    else:
        top = "no_viable"
    sel = run.get("selected") or {}
    return {"run_id": run["run_id"], "candidates": len(mine), "closed_polygons": sum(closed_polygon(r) for r in mine),
            "scoreable": sum(r["status"] == "scored" for r in mine), "top_candidate_status": top,
            "iou": sel.get("iou"), "max_wall_error_cm": sel.get("max_abs_wall_error_cm"),
            "max_wall_error_percent": sel.get("max_rel_wall_error_percent"), "gate_1_5cm": sel.get("gate_1_5cm"),
            "gate_3pct": sel.get("gate_3pct"), "gate_ceiling": sel.get("gate_ceiling_height_1_5cm"),
            "selected_gate_all": bool(sel and sel.get("gate_1_5cm") and sel.get("gate_3pct") and sel.get("gate_ceiling_height_1_5cm")),
            "run_pass": run["run_pass"], "fail_reasons": list(run["fail_reasons"])}


def per_run_table(orig_runs: list, orig_rows: list, new_runs: list, new_rows: list) -> list:
    new_by = {r["run_id"]: r for r in new_runs}
    table = []
    for run in orig_runs:
        o = run_summary(run, orig_rows)
        n = run_summary(new_by[run["run_id"]], new_rows) if run["run_id"] in new_by else None
        table.append({"run_id": run["run_id"], "original": o, "variant": n})
    return table


def totals(table: list, key: str) -> dict:
    s = [t[key] for t in table if t[key]]
    return {"runs": len(s), "candidates": sum(x["candidates"] for x in s), "closed_polygons": sum(x["closed_polygons"] for x in s),
            "runs_with_closed_polygon": sum(x["closed_polygons"] > 0 for x in s), "scoreable": sum(x["scoreable"] for x in s),
            "runs_with_scoreable": sum(x["scoreable"] > 0 for x in s),
            "top_candidate_scored": sum(x["top_candidate_status"] == "scored" for x in s),
            "top_candidate_tie": sum(x["top_candidate_status"].startswith("tie") for x in s),
            "selected_gates_all_passed": sum(x["selected_gate_all"] for x in s), "runs_pass_incl_capture_qa": sum(bool(x["run_pass"]) for x in s)}


def _f(v, fmt=".3f"):
    return "" if v is None else (format(v, fmt) if isinstance(v, float) else str(v))


def table_markdown(table: list) -> str:
    head = ("| run | cand o>v | closed o>v | scoreable o>v | top status o | top status v | IoU o | IoU v | max cm o | max cm v | max % o | max % v | "
            "gates(1.5cm/3%/ceil) v | selected gates all o>v |")
    lines = [head, "|" + "---|" * 14]
    for t in table:
        o, v = t["original"], t["variant"] or {}
        g = "/".join(str(v.get(k)) for k in ("gate_1_5cm", "gate_3pct", "gate_ceiling")) if v else ""
        lines.append(f"| {t['run_id']} | {o['candidates']}>{v.get('candidates')} | {o['closed_polygons']}>{v.get('closed_polygons')} | "
                     f"{o['scoreable']}>{v.get('scoreable')} | {o['top_candidate_status']} | {v.get('top_candidate_status')} | "
                     f"{_f(o['iou'])} | {_f(v.get('iou'))} | {_f(o['max_wall_error_cm'], '.2f')} | {_f(v.get('max_wall_error_cm'), '.2f')} | "
                     f"{_f(o['max_wall_error_percent'], '.2f')} | {_f(v.get('max_wall_error_percent'), '.2f')} | {g} | "
                     f"{o['selected_gate_all']}>{v.get('selected_gate_all')} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ truth-free axis tie evidence
FEATURES = (("normal_mass_parallel", lambda e: e.get("normal_mass_parallel")),
            ("normal_mass_explained", lambda e: e.get("normal_mass_explained")),
            ("explained_support_fraction", lambda e: e.get("explained_support_fraction")),
            ("cap_support", lambda e: (e.get("cap_pair") or {}).get("cap_support")),
            ("cap_low_coverage", lambda e: ((e.get("cap_pair") or {}).get("low") or {}).get("coverage")),
            ("cap_high_coverage", lambda e: ((e.get("cap_pair") or {}).get("high") or {}).get("coverage")),
            ("cap_min_coverage", lambda e: min(((e.get("cap_pair") or {}).get(k) or {}).get("coverage", float("nan")) for k in ("low", "high"))
             if (e.get("cap_pair") or {}).get("pair_found") else None),
            ("camera_share_between_caps", lambda e: (e.get("cap_pair") or {}).get("camera_share_between_caps")))


def tied_top(candidates: list, tol: float = 1e-9) -> list:
    viable = [c for c in candidates if c.get("viable")]
    if not viable:
        return []
    best = max(c["score"] for c in viable)
    return [c for c in viable if abs(c["score"] - best) < tol]


def tie_evidence(label: str, candidates: list) -> dict | None:
    """Truth-free: for the candidates tied at the top, list generic evidence features and whether each separates them (exact inequality,
    no threshold; ties of feature values are 'not separating'). Reports the arg-max as information; nothing is selected."""
    tied = tied_top(candidates)
    if len(tied) < 2:
        return None
    out = {"run": label, "tied_candidate_ids": [c["id"] for c in tied], "scores": [c["score"] for c in tied],
           "kinds": [c.get("kinds") for c in tied], "features": {}}
    for name, fn in FEATURES:
        vals = [fn(c.get("evidence") or {}) for c in tied]
        usable = [v for v in vals if isinstance(v, (int, float)) and v == v]
        sep = len(usable) == len(vals) and len(set(round(float(v), 9) for v in usable)) > 1
        arg = None
        if sep:
            m = max(usable)
            arg = [c["id"] for c, v in zip(tied, vals) if v == m]
        out["features"][name] = {"values": vals, "separates": bool(sep), "argmax_candidate_ids": arg}
    return out


def collect_tie_evidence(root: Path, batch: str) -> list:
    root = Path(root) / batch
    cfg = rb.read_json(root / "batch_config.json")
    res = []
    for s in cfg["plan"]:
        rd = rb.run_dir_of(root, s)
        label = f"house{s['house_seed']}_path{s['path_index']}"
        f = rd / "axis" / "frozen" / label / "candidates.json"
        if not f.is_file():
            continue
        t = tie_evidence(rb.run_id(s["house_seed"], s["path_index"]), rb.read_json(f).get("candidates", []))
        if t:
            res.append(t)
    return res


def tie_oracle_note(tie: dict, rows: list) -> dict:
    """EVALUATION-ONLY information (truth-aware, after evaluation): did the favoured candidate of each separating feature score or pass gates?
    Not used for selection or thresholds."""
    by = {r["candidate_id"]: r for r in rows if r["run_id"] == tie["run"]}
    out = {}
    for name, d in tie["features"].items():
        if d["separates"]:
            out[name] = [{"candidate_id": i, "status": by[i]["status"], "gate_all": by[i]["gate_all"]} for i in d["argmax_candidate_ids"] if i in by]
    return out


# ------------------------------------------------------------------ driver
def write_comparison(variant_root: Path, outputs: Path = OUTPUTS, batches=BATCHES) -> dict:
    variant_root = Path(variant_root)
    summary = {"label": LABEL, "batches": {}}
    md = [f"# {LABEL}", "", "Original = predeclared batch results (frozen, unchanged). Variant = same frozen DA3 predictions, current chain modules.", ""]
    for b in batches:
        orig = Path(outputs) / b / "results_000"
        new_dir = sorted((variant_root / b).glob("results_*"))[-1]
        table = per_run_table(rb.read_json(orig / "runs.json"), rb.read_json(orig / "candidates.json"),
                              rb.read_json(new_dir / "runs.json"), rb.read_json(new_dir / "candidates.json"))
        new_rows = rb.read_json(new_dir / "candidates.json")
        ties_new = collect_tie_evidence(variant_root, b)
        for t in ties_new:
            t["evaluation_only_oracle_note"] = tie_oracle_note(t, new_rows)
        summary["batches"][b] = {"original_totals": totals(table, "original"), "variant_totals": totals(table, "variant"), "per_run": table,
                                 "variant_results_dir": str(new_dir), "variant_axis_ties_at_top": ties_new}
        md += [f"## {b}", "", f"original: {json.dumps(summary['batches'][b]['original_totals'])}", "",
               f"variant: {json.dumps(summary['batches'][b]['variant_totals'])}", "", table_markdown(table),
               f"### Truth-free axis ties at top rank in the variant ({len(ties_new)} runs)", ""]
        for t in ties_new:
            sep = [k for k, d in t["features"].items() if d["separates"]]
            md.append(f"- {t['run']}: tied {t['tied_candidate_ids']}, kinds {t['kinds']}; separating features (exact inequality, no threshold): {sep or 'none'}")
        md.append("")
    rb.write_new_json(variant_root / "comparison.json", summary)
    rb.write_new_text(variant_root / "COMPARISON.md", "\n".join(md))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    info = init_variant_root(a.output_dir)
    print(json.dumps({"differing_modules": {b: v["modules_differing_from_pinned"] for b, v in info["batches"].items()}}, indent=1), flush=True)
    chain = run_chain_all(a.output_dir, workers=a.workers)
    print(json.dumps(chain, indent=1), flush=True)
    if chain["failed"]:
        print("some chain runs failed; evaluating the frozen ones is not allowed until all are frozen", file=sys.stderr)
        return 1
    for b in BATCHES:                                                           # all 60 frozen BEFORE any truth is read
        print(json.dumps(evaluate_variant(a.output_dir, b), indent=1)[:600], flush=True)
    write_comparison(a.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
