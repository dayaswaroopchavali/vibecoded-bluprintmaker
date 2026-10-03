"""Run fragment association on frozen predictions for EVERY co-supported axis hypothesis (truth-free, no scoring).

  extract_wall_lines -> fragment_association.associate_fragments -> boundary_assembly.assemble_boundary_ray (real ray scorer,
  same settings as compare_wall_boundaries.py) on three line sets: all lines (baseline), room_boundary_candidate only, and
  candidates + undetermined (ambiguity kept). Writes JSON + a short markdown into a NEW folder (never overwrites).
  Nothing here reads ground truth or the simulator and nothing is scored; the coordinator evaluates after freezing.

  python run_fragment_association.py --output-dir <NEW folder> [--prediction label=npz[:axes.json]]...
"""
from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
RAY_K = 2.0                                   # compare_wall_boundaries default tolerance multiplier (grid member, fixed a priori)
DEFAULT_INPUTS = [
    {"label": "path_01_full120",
     "prediction": "blender-coverage-v3/path_01/da3_full120/prediction.npz",
     "axes": "wall-boundary-path01-full120-v1/frozen/path_01_full120/axes.json"},
    {"label": "path_01_refined_windowed120",
     "prediction": "refined-windowed-path01-v1/prediction.npz",
     "axes": "wall-boundary-path01-refined-windowed120-v1/frozen/path_01_refined_windowed120/axes.json"},
]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:                 # 'x': never overwrites
        f.write(text)


def write_json(path: Path, obj) -> None:
    write_new(path, json.dumps(_clean(obj), indent=1, allow_nan=False))


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def hypotheses_for(pred, axes_json):
    """Co-supported axis hypotheses: from a frozen axes.json, else recomputed with the same code as compare_wall_boundaries."""
    if axes_json:
        axes = json.loads(Path(axes_json).read_text(encoding="utf-8"))
        source = str(axes_json)
    else:
        import extract_planes
        import geometry_orientation
        points = extract_planes.points_from_prediction(pred)
        axes = geometry_orientation.axis_hypotheses(extract_planes.extract(points), points)
        source = "recomputed (extract_planes + geometry_orientation)"
    out = []
    for k, h in enumerate(axes.get("hypotheses") or []):
        out.append({"hypothesis_index": k, "axis_world": h.get("axis_world"), "co_supported": bool(h.get("co_supported_with_best")),
                    "rank": h.get("rank"), "score": h.get("score")})
    return out, {"source": source, "status": axes.get("status")}


def summarize_boundary(b) -> dict:
    if b is None:
        return {"status": "not_run"}
    out = {"status": b.get("status"), "n_edges": len(b.get("edges") or []),
           "edge_line_ids": [e.get("line_id") for e in b.get("edges") or []]}
    if b.get("vertices_xy"):
        out["vertices_xy"] = b["vertices_xy"]
    search = b.get("boundary_search") or {}
    out["search"] = {k: search.get(k) for k in ("nodes", "corner_links", "closed_cycles", "layouts_evaluated", "valid_layouts")}
    out["rejected_by_reason"] = search.get("rejected_by_reason")
    inc = b.get("incomplete_boundaries") or []
    if inc:
        out["incomplete_reason"] = inc[0].get("reason")
        out["rejected_by_reason"] = inc[0].get("layouts_rejected_by_reason") or out["rejected_by_reason"]
        out["closest_layout_problems"] = inc[0].get("closest_layout_problems")
    out["alternatives"] = len(b.get("alternatives") or [])
    out["ignored_lines"] = b.get("ignored_lines")
    return out


def assemble(pred, walls, k):
    import boundary_assembly as ba
    if not walls.get("lines"):
        return {"status": "incomplete", "incomplete_boundaries": [{"reason": "no_lines_after_association"}], "edges": []}
    return ba.assemble_boundary_ray(pred, walls, k)


def run_one(label, pred, hyp, out_dir: Path, k: float) -> dict:
    import fragment_association as fa
    import wall_extraction
    rec = {**hyp, "label": label}
    if not hyp["co_supported"]:
        rec["status"] = "skipped_not_co_supported"
        return rec
    idx = hyp["hypothesis_index"]
    try:
        walls = wall_extraction.extract_wall_lines(pred, np.asarray(hyp["axis_world"], float))
        rec["walls_status"] = walls.get("status")
        if walls.get("status") != "ok":
            rec["status"] = "unavailable"
            rec["walls_reasons"] = walls.get("reasons")
            return rec
        assoc = fa.associate_fragments(walls)
        rec["association_status"] = assoc["status"]
        rec["counts"] = assoc["report"]["counts"]
        rec["tolerance"] = assoc["report"]["tolerance"]
        rec["lines"] = assoc["report"]["lines"]
        rec["pairs"] = assoc["report"]["pairs"]
        rec["footprint_clusters"] = assoc["report"]["footprint_clusters"]
        slim = {k2: v for k2, v in assoc["walls_all"].items() if k2 != "observations"}
        write_json(out_dir / "association" / f"{label}_hyp{idx}_walls_all.json", slim)
        variants = {"all_lines": assoc["walls_all"], "boundary_candidates": assoc["walls_boundary_candidates"],
                    "candidates_plus_undetermined": assoc["walls_with_undetermined"],
                    "layer_representatives_HYPOTHESIS": assoc["walls_layer_representatives"],
                    "layer_merged_HYPOTHESIS": assoc["walls_layer_merged"]}
        rec["assembly"] = {}
        for name, w in variants.items():
            try:
                b = assemble(pred, w, k)
                rec["assembly"][name] = {"n_lines": len(w["lines"]), "line_ids": [l["id"] for l in w["lines"]], **summarize_boundary(b)}
                write_json(out_dir / "boundaries" / f"{label}_hyp{idx}_{name}.json", b)
            except Exception as exc:                                # reported verbatim
                rec["assembly"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                         "traceback_tail": traceback.format_exc()[-800:]}
        rec["status"] = "ran"
    except Exception as exc:
        rec["status"] = "failed"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc()[-1500:]}
    return rec


def markdown(result) -> str:
    L = ["# Fragment association run (truth-free; no scoring, no accuracy claim)", "",
         f"ray scorer: boundary_assembly.assemble_boundary_ray, k={result['ray_k']}; classes are evidence labels, not truth.", ""]
    for inp in result["inputs"]:
        L += [f"## {inp['label']}", f"prediction: `{inp['prediction']}`; axes: {inp['axes_info']}", "",
              "| hyp | co-supported | lines | classes | all lines | candidates only | candidates+undetermined | layer reps (hypothesis) | layer merged (hypothesis) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for h in inp["hypotheses"]:
            if h["status"] != "ran":
                L.append(f"| {h['hypothesis_index']} | {h['co_supported']} | - | {h['status']} | - | - | - | - | - |")
                continue
            c = {k: v for k, v in h["counts"].items() if v}
            cell = lambda n: f"{h['assembly'][n]['status']} ({h['assembly'][n]['n_lines']} lines)"
            L.append(f"| {h['hypothesis_index']} | yes | {len(h['lines'])} | {c} | {cell('all_lines')} | "
                     f"{cell('boundary_candidates')} | {cell('candidates_plus_undetermined')} | {cell('layer_representatives_HYPOTHESIS')} | {cell('layer_merged_HYPOTHESIS')} |")
        L.append("")
        for h in inp["hypotheses"]:
            if h["status"] != "ran":
                continue
            L.append(f"### hyp{h['hypothesis_index']}")
            for line in h["lines"]:
                alt = ", ".join(f"{a['class']}={a['score']}" for a in line["alternatives"][:2])
                L.append(f"- line {line['line_id']}: **{line['class']}** (alt {alt})")
            for name, a in h["assembly"].items():
                why = f"{a.get('incomplete_reason') or a.get('error') or ''} {a.get('rejected_by_reason') or ''} {a.get('search') or ''}"
                L.append(f"- {name}: {a['status']}, edges from lines {a.get('edge_line_ids')}; {why}")
            L.append("")
    return "\n".join(L) + "\n"


def parse_input(spec: str, root: Path) -> dict:
    """label=prediction.npz[:axes.json]  (Windows drive colons are handled by splitting on the last '.npz:')."""
    label, _, rest = spec.partition("=")
    pred, axes = (rest.split(".npz:", 1) + [None])[:2] if ".npz:" in rest else (rest, None)
    if ".npz:" in rest:
        pred += ".npz"
    return {"label": label, "prediction": str(Path(pred)), "axes": str(Path(axes)) if axes else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True, help="NEW folder (refuses to overwrite)")
    ap.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    ap.add_argument("--prediction", action="append", default=None, help="label=prediction.npz[:axes.json] (repeatable)")
    ap.add_argument("--recompute-axes", action="store_true", help="ignore frozen axes.json, recompute hypotheses")
    ap.add_argument("--k", type=float, default=RAY_K)
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    if args.prediction:
        inputs = [parse_input(s, args.outputs_root) for s in args.prediction]
    else:
        inputs = [{"label": d["label"], "prediction": str(args.outputs_root / d["prediction"]),
                   "axes": str(args.outputs_root / d["axes"])} for d in DEFAULT_INPUTS]
    result = {"model": "fragment_association_run", "ray_k": args.k, "scoring": "none (truth-free; coordinator evaluates after freezing)",
              "inputs": []}
    for inp in inputs:
        pred = load_prediction(inp["prediction"])
        hyps, info = hypotheses_for(pred, None if args.recompute_axes else inp.get("axes"))
        rec = {**inp, "axes_info": info, "hypotheses": [run_one(inp["label"], pred, h, out_dir, args.k) for h in hyps]}
        result["inputs"].append(rec)
        print(inp["label"], [(h["hypothesis_index"], h["status"], (h.get("assembly") or {}).get("all_lines", {}).get("status"),
                              (h.get("assembly") or {}).get("boundary_candidates", {}).get("status")) for h in rec["hypotheses"]])
    write_json(out_dir / "fragment_association.json", result)
    write_new(out_dir / "fragment_association.md", markdown(result))
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
