"""Wall-boundary budget x axis-hypothesis comparison (task 0012, worker C).

Pipeline per frozen DA3 prediction (RGB-derived only, UNCHANGED settings):
    extract_planes.extract -> geometry_orientation.axis_hypotheses -> for EVERY co-supported axis hypothesis:
    wall_extraction.extract_wall_lines (A) -> boundary_assembly.assemble_boundary (B).
No hypothesis and no budget is ever selected by ground truth; failures, ambiguous and incomplete results are reported as is.

Order is enforced in code:  1) extraction path (no scene argument exists)  2) results FROZEN to disk (+ sha256)
3) truth-free frame-selection stability between budgets  4) EVALUATION ONLY, reading the frozen files back:
estimate_height (3.0 m normalisation) and evaluate_floorplan with a provenance manifest. Truth is read only in step 4
(`load_scene` hook). Every oracle choice (mirroring, rigid alignment, edge matching, which hypothesis is the vertical
axis) belongs to evaluate_floorplan and is labelled EVALUATOR-ORACLE. This module never claims a benchmark stage complete.

  python compare_wall_boundaries.py --output-dir <NEW folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
FRAMES_AVAILABLE = 120
# Truth-free stability heuristics (NOT accuracy gates): fractions of the polygon diameter / edge length.
STABLE_CORNER_TOL = 0.03
STABLE_EDGE_TOL = 0.05
EXPLAINED_SIGMA = 2.0
ORACLE_CHOICES = [
    "EVALUATOR-ORACLE: mirroring of the predicted polygon (axis sign is unresolved) chosen by IoU against the truth",
    "EVALUATOR-ORACLE: rigid alignment (rotation + translation) chosen by IoU against the truth",
    "EVALUATOR-ORACLE: cyclic vertex start and edge matching chosen with the truth in hand",
    "EVALUATOR-ORACLE: which structural-axis hypothesis is the vertical axis is NOT chosen; every co-supported hypothesis is scored",
]
BUDGETS = [
    {"label": "a_linspace36", "description": "36 np.linspace frames of the 120-frame capture",
     "prediction": "blender-coverage-v3/path_00/da3/prediction.npz",
     "scene": "blender-coverage-v3/capture/private_ground_truth/scene.json"},
    {"label": "b_stride3_40", "description": "40 frames, stride 3",
     "prediction": "backend-compare-v1/control_original/da3/prediction_source_indices_remapped.npz",
     "scene": "texture-ablation-v1/control_original/private_ground_truth/scene.json"},
    {"label": "c_stride2_60_DIFFERENT_BUDGET", "description": "60 frames, stride 2 (different budget)",
     "prediction": "backend-compare-v1/extra_control_original_step2_DIFFERENT_BUDGET/da3/prediction_source_indices_remapped.npz",
     "scene": "texture-ablation-v1/control_original/private_ground_truth/scene.json"},
    {"label": "d_textured40", "description": "40 frames, textured variant",
     "prediction": "backend-compare-v1/control_textured/da3/prediction_source_indices_remapped.npz",
     "scene": "texture-ablation-v1/control_textured/private_ground_truth/scene.json"},
]


# ---------------------------------------------------------------------------------------------------- io helpers
def _clean(obj, counter=None):
    """JSON-safe copy: numpy -> python, non-finite floats -> None (counted)."""
    counter = counter if counter is not None else [0]
    if isinstance(obj, dict):
        return {str(k): _clean(v, counter) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v, counter) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist(), counter)
    if isinstance(obj, np.generic):
        return _clean(obj.item(), counter)
    if isinstance(obj, float) and not math.isfinite(obj):
        counter[0] += 1
        return None
    return obj


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new_json(path: Path, obj) -> str:
    """Write JSON to a NEW file (never overwrites); returns sha256 of the bytes written."""
    data = json.dumps(_clean(obj), indent=1, allow_nan=False).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        stream.write(data)
    return sha256_bytes(data)


def read_frozen(out_dir: Path, rel: str, sha: str):
    data = (Path(out_dir) / rel).read_bytes()
    if sha256_bytes(data) != sha:
        raise ValueError(f"frozen file changed after freezing: {rel}")
    return json.loads(data.decode("utf-8"))


def _default_hooks() -> dict:
    """Lazy imports: A and B (and the heavy deps) are only imported when actually used."""
    def load_prediction(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def extract_planes(points):
        import extract_planes
        return extract_planes.extract(points)

    def points_from_prediction(pred):
        import extract_planes
        return extract_planes.points_from_prediction(pred)

    def axis_hypotheses(planes, points):
        import geometry_orientation
        return geometry_orientation.axis_hypotheses(planes, points)

    def extract_wall_lines(pred, axis):
        import wall_extraction
        return wall_extraction.extract_wall_lines(pred, axis)

    def assemble_boundary(walls):
        import boundary_assembly
        return boundary_assembly.assemble_boundary(walls)

    def assemble_boundary_ray(pred, walls, k):
        import boundary_assembly
        return boundary_assembly.assemble_boundary_ray(pred, walls, k)

    def camera_centers(extrinsics):
        import estimate_height
        return estimate_height.camera_centers(extrinsics)

    def estimate_height(planes, points, up, centers):
        import estimate_height as module
        return module.estimate_height(planes, points, up, centers)

    def evaluate_floorplan(prediction, scene, height, manifest):
        import evaluate_floorplan as module
        return module.evaluate_floorplan(prediction, scene, height, manifest=manifest, axis_sign_resolved=False)

    def load_scene(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return {k: v for k, v in locals().items() if callable(v)}


# ---------------------------------------------------------------------------------------------------- 1. extraction
def provenance_blockers(retained, boundary_status) -> list:
    """Truth-free whole-room-claim blockers from capture provenance and the boundary status."""
    out = ["whole_room_claim_never_allowed_for_this_comparison"]
    if retained is None:
        out.append("capture_provenance_missing")
    elif retained < 1.0:
        out.append("retained_capture_below_full")
    if boundary_status != "complete_polygon_hypothesis":
        out.append(f"boundary_not_complete:{boundary_status}")
    return out


def run_extraction(budgets, out_dir: Path, hooks=None, events=None) -> dict:
    """Run A then B for every co-supported hypothesis of every budget; freeze to <out_dir>/frozen. NO scene argument exists
    here: truth cannot reach this path. Returns the frozen index (paths + sha256), never the scoring."""
    hooks = {**_default_hooks(), **(hooks or {})}
    events = events if events is not None else []
    out_dir = Path(out_dir)
    index = {"budgets": []}
    for budget in budgets:
        label = budget["label"]
        events.append(("extract_budget", label))
        rec = {"label": label, "description": budget.get("description"), "prediction": str(budget["prediction"]),
               "frames_available": int(budget.get("frames_available", FRAMES_AVAILABLE)), "hypotheses": [], "files": {}}
        try:
            pred = hooks["load_prediction"](budget["prediction"])
            idx = np.asarray(pred.get("source_indices", np.arange(len(pred["depth"]))))
            rec["source_indices"] = [int(i) for i in idx]
            rec["retained_capture_fraction"] = float(len(idx) / rec["frames_available"])
            points = hooks["points_from_prediction"](pred)
            planes = hooks["extract_planes"](points)
            axes = hooks["axis_hypotheses"](planes, points)
        except Exception as exc:                                         # reported verbatim, never dropped
            rec["status"] = "failed_before_hypotheses"
            rec["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc()[-1500:]}
            index["budgets"].append(rec)
            continue
        rec["status"] = "ok"
        rec["axis_status"] = axes.get("status")
        rec["files"]["planes"] = _freeze(out_dir, f"frozen/{label}/planes.json", planes)
        rec["files"]["axes"] = _freeze(out_dir, f"frozen/{label}/axes.json", axes)
        for k, hyp in enumerate(axes.get("hypotheses", []) or []):
            h = {"hypothesis_index": k, "axis_world": hyp.get("axis_world"), "rank": hyp.get("rank"),
                 "score": hyp.get("score"), "co_supported_with_best": bool(hyp.get("co_supported_with_best"))}
            if not h["co_supported_with_best"]:
                h["status"] = "skipped_not_co_supported"
                rec["hypotheses"].append(h)
                continue
            events.append(("extract_hypothesis", label, k))
            try:
                walls = hooks["extract_wall_lines"](pred, np.asarray(hyp["axis_world"], float))
                h["walls_status"] = walls.get("status")
                h["walls_reasons"] = list(walls.get("reasons", []) or [])
                h["files"] = {"walls": _freeze(out_dir, f"frozen/{label}/hyp{k}_walls.json", walls)}
                if walls.get("status") != "ok":
                    h["status"] = "unavailable"
                else:
                    events.append(("assemble_hypothesis", label, k))
                    boundary = hooks["assemble_boundary"](walls)
                    h["boundary_status"] = boundary.get("status")
                    h["files"]["boundary"] = _freeze(out_dir, f"frozen/{label}/hyp{k}_boundary.json", boundary)
                    h["status"] = "ran"
            except Exception as exc:
                h["status"] = "failed"
                h["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc()[-1500:]}
            rec["hypotheses"].append(h)
        if not rec["hypotheses"]:
            rec["no_hypotheses_reason"] = f"axis_hypotheses status={axes.get('status')}"
        index["budgets"].append(rec)
    events.append(("frozen", None))
    write_new_json(out_dir / "frozen_index.json", index)
    return index


def _freeze(out_dir: Path, rel: str, obj) -> dict:
    return {"path": rel, "sha256": write_new_json(Path(out_dir) / rel, obj)}


# ---------------------------------------------------------------------------------------------------- 2. stability
def _poly(boundary):
    v = boundary.get("vertices_xy") if isinstance(boundary, dict) else None
    if v is None:
        return None
    p = np.asarray(v, float)
    return p if p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3 and np.isfinite(p).all() else None


def _similarity(src: np.ndarray, dst: np.ndarray):
    """Rotation + uniform scale + translation (no reflection) of src onto dst by least squares (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d0.T @ s0)
    fix = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    rot = u @ fix @ vt
    var = float((s0 ** 2).sum())
    scale = float((sig * np.diag(fix)).sum() / var) if var > 0 else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def stability_pair(a, b) -> dict:
    """Truth-free comparison of two boundary results: topology and geometry after the best rigid + uniform-scale fit.
    Reflection and cyclic start are tried (axis sign is unresolved) and the minimum-residual one is kept; the choice
    uses only the two predictions."""
    pa, pb = _poly(a), _poly(b)
    out = {"status_a": (a or {}).get("status"), "status_b": (b or {}).get("status")}
    if pa is None or pb is None:
        return {**out, "verdict": "not_comparable", "reason": "a_or_b_has_no_polygon", "edge_count_a": None if pa is None else len(pa),
                "edge_count_b": None if pb is None else len(pb)}
    out.update(edge_count_a=len(pa), edge_count_b=len(pb))
    if len(pa) != len(pb):
        return {**out, "verdict": "unstable_edge_count", "reason": f"edge count {len(pa)} vs {len(pb)}"}
    n = len(pa)
    diameter = float(np.max(np.linalg.norm(pb[:, None] - pb[None], axis=2)))
    best = None
    for mirror in (False, True):
        src = pa * np.array([-1.0, 1.0]) if mirror else pa
        if mirror:
            src = src[::-1]
        for shift in range(n):
            dst = np.roll(pb, -shift, axis=0)
            scale, rot, t = _similarity(src, dst)
            fitted = scale * src @ rot.T + t
            resid = np.linalg.norm(fitted - dst, axis=1)
            if best is None or resid.max() < best["resid"].max():
                best = {"resid": resid, "mirror": mirror, "shift": shift, "scale": scale, "fitted": fitted, "dst": dst,
                        "rotation_deg": float(math.degrees(math.atan2(rot[1, 0], rot[0, 0])))}
    la = np.linalg.norm(np.roll(best["fitted"], -1, axis=0) - best["fitted"], axis=1)
    lb = np.linalg.norm(np.roll(best["dst"], -1, axis=0) - best["dst"], axis=1)
    change = 100.0 * np.abs(la - lb) / np.maximum(lb, 1e-12)
    corner_frac = float(best["resid"].max() / max(diameter, 1e-12))
    ok = corner_frac <= STABLE_CORNER_TOL and float(change.max()) <= 100 * STABLE_EDGE_TOL
    return {**out, "verdict": "stable" if ok else "unstable_geometry",
            "reason": "edge count equal; corners and edge lengths agree after rigid+uniform-scale fit" if ok
            else "edge count equal but corner positions / edge lengths move after the best fit",
            "fit": {"uniform_scale": best["scale"], "rotation_deg": best["rotation_deg"], "reflection_used": best["mirror"],
                    "cyclic_shift": best["shift"], "note": "reflection and cyclic start chosen by minimum residual between the two predictions only (no truth)"},
            "corner_residual_max_fraction_of_diameter": corner_frac,
            "corner_residual_rms_fraction_of_diameter": float(np.sqrt((best["resid"] ** 2).mean()) / max(diameter, 1e-12)),
            "edge_length_change_percent": [float(c) for c in change], "edge_length_change_max_percent": float(change.max()),
            "tolerances": {"corner_fraction_of_diameter": STABLE_CORNER_TOL, "edge_length_percent": 100 * STABLE_EDGE_TOL,
                           "note": "stability heuristics, not accuracy gates"}}


def stability_matrix(results) -> dict:
    """`results`: list of {budget, hypothesis_index, boundary}. All cross-budget pairs are compared and listed (no filtering
    of bad pairs); per-result summaries count the other-budget results it is stable/unstable/not comparable with."""
    pairs = []
    for i, ra in enumerate(results):
        for rb in results[i + 1:]:
            if ra["budget"] == rb["budget"]:
                continue
            pairs.append({"a": {"budget": ra["budget"], "hypothesis_index": ra["hypothesis_index"]},
                          "b": {"budget": rb["budget"], "hypothesis_index": rb["hypothesis_index"]},
                          **stability_pair(ra["boundary"], rb["boundary"])})
    summary = {}
    for r in results:
        key = f"{r['budget']}/hyp{r['hypothesis_index']}"
        mine = [p for p in pairs if (p["a"]["budget"], p["a"]["hypothesis_index"]) == (r["budget"], r["hypothesis_index"])
                or (p["b"]["budget"], p["b"]["hypothesis_index"]) == (r["budget"], r["hypothesis_index"])]
        summary[key] = {v: sum(1 for p in mine if p["verdict"] == v)
                        for v in ("stable", "unstable_edge_count", "unstable_geometry", "not_comparable")}
        summary[key]["compared"] = len(mine)
    return {"pairs": pairs, "per_result": summary,
            "note": "Hypothesis indices are not matched across budgets (each prediction has its own world frame); every pair is listed."}


# ---------------------------------------------------------------------------------------------------- 3. evaluation
def eval_copy(boundary: dict) -> dict:
    """evaluate_floorplan labels extrapolated edges 'inferred'."""
    copy = json.loads(json.dumps(boundary))
    by_vertex = vertex_indexed_edges(copy)
    for j, e in by_vertex.items():
        e["assembly_index"] = e.get("index")
        e["index"] = j                                                  # evaluate_floorplan: edge j runs vertex j -> j+1
        if e.get("label") == "extrapolated":
            e["label"] = "inferred"
    return copy


def vertex_indexed_edges(boundary: dict) -> dict:
    """{j: edge} where edge j starts at vertex j. B's edge `index` need not follow that convention (its edge k ends at
    vertex k), so edges are matched to vertices by coordinates."""
    verts = _poly(boundary)
    out = {}
    if verts is None:
        return out
    for e in boundary.get("edges", []) or []:
        d = np.linalg.norm(verts - np.asarray(e["start_xy"], float), axis=1)
        j = int(np.argmin(d))
        if d[j] <= 1e-6 * max(1.0, float(np.abs(verts).max())):
            out[j] = e
    return out


def _line_by_id(walls):
    return {ln["id"]: ln for ln in walls.get("lines", []) or []}


def residual_sources(boundary: dict, walls: dict, evaluation: dict) -> dict:
    """Where the measured wall-length residual could come from, using only numbers we have (EVALUATION ONLY)."""
    if not evaluation or evaluation.get("status") != "scored":
        return {"available": False, "reason": "evaluation not scored"}
    metrics = evaluation["metrics"]
    scale = evaluation["scale"]["prediction_scale"]
    lines = _line_by_id(walls)
    by_vertex = vertex_indexed_edges(boundary)
    n = len(_poly(boundary))
    mirrored = evaluation["alignment"]["mirrored"]
    rows = []
    for er in metrics["edge_errors"]:
        j = er["pred_edge"]
        k = (n - 2 - j) % n if mirrored else j
        if k not in by_vertex:
            continue
        e = by_vertex[k]
        ln = lines.get(e.get("line_id"), {})
        sd = (ln.get("uncertainty") or {}).get("offset_sd")
        d = np.asarray(e["end_xy"], float) - np.asarray(e["start_xy"], float)
        d = d / max(np.linalg.norm(d), 1e-12)
        prop = None
        if sd is not None:
            parts = []
            for nb in (by_vertex.get((k - 1) % n, {}), by_vertex.get((k + 1) % n, {})):
                nln = lines.get(nb.get("line_id"), {})
                nsd = (nln.get("uncertainty") or {}).get("offset_sd")
                nn = nln.get("normal_xy")
                if nsd is not None and nn is not None:
                    parts.append(float(nsd) / max(abs(float(np.dot(nn, d))), 0.05))
            prop = float(math.sqrt(sum(p * p for p in parts)) * scale * 100) if parts else None
        row = {"boundary_edge": e.get("index"), "vertex_edge": k, "line_id": e.get("line_id"), "true_edge": er["true_edge"], "label": e.get("label"),
               "abs_error_cm": er["abs_error_cm"], "rel_error_percent": er["rel_error_percent"],
               "pred_over_true_length": er["pred_length_m"] / er["true_length_m"],
               "extrapolated_fraction": None if e.get("observed_fraction") is None else 1.0 - float(e["observed_fraction"]),
               "line_offset_sd_cm": None if sd is None else float(sd) * scale * 100,
               "propagated_endpoint_length_sd_cm": prop,
               "along_coverage_fraction": (ln.get("support") or {}).get("along_coverage_fraction"),
               "distinct_frames": (ln.get("support") or {}).get("distinct_frames")}
        row["within_stated_line_uncertainty"] = None if prop is None else bool(er["abs_error_cm"] <= EXPLAINED_SIGMA * prop)
        rows.append(row)
    out = {"available": True, "rows": rows, "uncertainty_note": f"within_stated_line_uncertainty = |error| <= {EXPLAINED_SIGMA} x propagated offset sd of the two neighbouring walls; A's offset_sd only"}
    if rows:
        ratios = np.array([r["pred_over_true_length"] for r in rows])
        med = float(np.median(ratios))
        true_len = np.array([er["true_length_m"] for er in metrics["edge_errors"]])
        after = 100 * np.abs(ratios / med - 1) * true_len
        max_abs = float(max(r["abs_error_cm"] for r in rows))
        out.update(common_length_ratio_median=med, max_abs_error_cm=max_abs,
                   max_abs_error_cm_after_removing_common_scale=float(after.max()),
                   common_scale_component_cm=float(max_abs - after.max()))
        ex = [(r["extrapolated_fraction"], r["abs_error_cm"]) for r in rows if r["extrapolated_fraction"] is not None]
        if len(ex) >= 3 and np.std([a for a, _ in ex]) > 0 and np.std([b for _, b in ex]) > 0:
            out["pearson_extrapolated_fraction_vs_abs_error"] = float(np.corrcoef(*zip(*ex))[0, 1])
        known = [r for r in rows if r["within_stated_line_uncertainty"] is not None]
        out["edges_beyond_stated_uncertainty"] = [r["boundary_edge"] for r in known if not r["within_stated_line_uncertainty"]]
        parts = []
        if out["common_scale_component_cm"] > 0.5 * max_abs:
            parts.append(f"a common length-ratio factor of {med:.4f} (the 3.0 m height normalisation / DA3 scale) explains {out['common_scale_component_cm']:.1f} of {max_abs:.1f} cm")
        else:
            parts.append(f"common scale explains only {out['common_scale_component_cm']:.1f} of {max_abs:.1f} cm; the rest is edge-specific")
        if out["edges_beyond_stated_uncertainty"]:
            parts.append(f"edges {out['edges_beyond_stated_uncertainty']} exceed {EXPLAINED_SIGMA} x the stated line-offset uncertainty (bias, not noise)")
        out["statement"] = "; ".join(parts)
    if metrics["unmatched_true_edges"] or metrics["unmatched_predicted_edges"]:
        out["unmatched_note"] = (f"unmatched true {metrics['unmatched_true_edges']} / predicted {metrics['unmatched_predicted_edges']}: "
                                 f"edge-count or position mismatch fails the gates before any length error is considered")
    return out


def run_evaluation(budgets, out_dir: Path, index: dict, hooks=None, events=None, variant=None) -> dict:
    """EVALUATION ONLY. Reads the frozen files back from disk, then scores. Writes <out_dir>/evaluation/*.json.
    `variant` = key of a scorer variant under each hypothesis' files (default: the default-scorer boundary)."""
    bkey = variant or "boundary"
    suffix = "" if variant is None else f"_{variant}"
    hooks = {**_default_hooks(), **(hooks or {})}
    events = events if events is not None else []
    out_dir = Path(out_dir)
    if not (out_dir / "frozen_index.json").exists():
        raise RuntimeError("refusing to evaluate before results are frozen")
    scene_of = {b["label"]: b.get("scene") for b in budgets}
    out = {}
    for rec in index["budgets"]:
        label = rec["label"]
        if rec.get("status") != "ok":
            continue
        events.append(("evaluate_budget", label))
        pred = hooks["load_prediction"](rec["prediction"])
        points = hooks["points_from_prediction"](pred)
        centers = hooks["camera_centers"](pred["extrinsics"])
        planes = read_frozen(out_dir, rec["files"]["planes"]["path"], rec["files"]["planes"]["sha256"])
        scene = hooks["load_scene"](scene_of[label])
        manifest = {"provenance": "frozen frame budget", "retained_capture_fraction": rec["retained_capture_fraction"],
                    "frames_registered": len(rec["source_indices"]), "frames_available": rec["frames_available"]}
        for h in rec["hypotheses"]:
            if h.get("status") != "ran" or bkey not in h["files"]:
                continue
            key = f"{label}/hyp{h['hypothesis_index']}"
            walls = read_frozen(out_dir, h["files"]["walls"]["path"], h["files"]["walls"]["sha256"])
            boundary = read_frozen(out_dir, h["files"][bkey]["path"], h["files"][bkey]["sha256"])
            try:
                height = hooks["estimate_height"](planes, points, np.asarray(h["axis_world"], float), centers)
                events.append(("evaluate", key))
                ev = hooks["evaluate_floorplan"](eval_copy(boundary), scene, height, manifest)
                record = {"evaluation": ev, "height_estimate": {k: height.get(k) for k in
                          ("status", "separation_unscaled", "floor_plane_id", "ceiling_plane_id", "reasons")},
                          "residual_sources": residual_sources(boundary, walls, ev), "oracle_choices": ORACLE_CHOICES}
                if boundary.get("status") == "ambiguous":
                    record["alternatives"] = _score_alternatives(boundary, scene, height, manifest, hooks, events, key)
            except Exception as exc:
                record = {"evaluation": None, "error": {"type": type(exc).__name__, "message": str(exc),
                                                         "traceback_tail": traceback.format_exc()[-1500:]}}
            write_new_json(out_dir / "evaluation" / f"{label}_hyp{h['hypothesis_index']}{suffix}.json", record)
            out[key] = record
    return out


def _score_alternatives(boundary, scene, height, manifest, hooks, events, key):
    rows = []
    for i, alt in enumerate(boundary.get("alternatives") or []):
        try:
            if not isinstance(alt, dict) or alt.get("vertices_xy") is None:
                rows.append({"alternative": i, "evaluation": None, "reason": "alternative carries no vertices_xy"})
                continue
            cand = {**boundary, **alt, "status": "complete_polygon_hypothesis"}
            events.append(("evaluate", f"{key}/alt{i}"))
            rows.append({"alternative": i, "evaluation": hooks["evaluate_floorplan"](eval_copy(cand), scene, height, manifest),
                         "note": "scored independently; evaluator did not choose between alternatives"})
        except Exception as exc:
            rows.append({"alternative": i, "evaluation": None, "error": f"{type(exc).__name__}: {exc}"})
    return rows


# ---------------------------------------------------------------------------------------------------- 4. report
def table_rows(index: dict, stability: dict, evaluations: dict | None, out_dir: Path) -> list:
    rows = []
    for rec in index["budgets"]:
        if rec.get("status") != "ok":
            rows.append({"budget": rec["label"], "hypothesis": "-", "status": "failed_before_hypotheses",
                         "failure": json.dumps(rec.get("error"))})
            continue
        if not rec["hypotheses"]:
            rows.append({"budget": rec["label"], "hypothesis": "-", "status": "no_hypotheses", "failure": rec.get("no_hypotheses_reason")})
        for h in rec["hypotheses"]:
            key = f"{rec['label']}/hyp{h['hypothesis_index']}"
            row = {"budget": rec["label"], "hypothesis": h["hypothesis_index"], "retained_capture_fraction": rec["retained_capture_fraction"],
                   "status": h.get("boundary_status") or h["status"], "run_status": h["status"], "edges": None,
                   "observed_edges": None, "extrapolated_edges": None, "failure": None}
            if h["status"] == "failed":
                row["failure"] = f"{h['error']['type']}: {h['error']['message']}"
            elif h["status"] == "unavailable":
                row["failure"] = f"A status={h.get('walls_status')}: {h.get('walls_reasons')}"
            elif h["status"] == "ran":
                b = read_frozen(out_dir, h["files"]["boundary"]["path"], h["files"]["boundary"]["sha256"])
                edges = b.get("edges") or []
                row.update(edges=len(edges), observed_edges=sum(1 for e in edges if e.get("label") == "observed"),
                           extrapolated_edges=sum(1 for e in edges if e.get("label") == "extrapolated"))
                if b.get("status") != "complete_polygon_hypothesis":
                    row["failure"] = f"{b.get('status')}: {b.get('incomplete_boundaries')}"
                row["stability"] = stability["per_result"].get(key)
            if evaluations is not None and key in evaluations:
                ev = evaluations[key].get("evaluation")
                if ev is None:
                    row["evaluation"] = {"status": "error", "failure": evaluations[key].get("error")}
                else:
                    m = ev.get("metrics") or {}
                    row["evaluation"] = {"status": ev["status"], "failure_reasons": ev["failure_reasons"], "iou": m.get("iou"),
                                         "max_abs_wall_error_cm": m.get("max_abs_wall_error_cm"),
                                         "max_rel_wall_error_percent": m.get("max_rel_wall_error_percent"),
                                         "unmatched_true": m.get("unmatched_true_edges"), "unmatched_pred": m.get("unmatched_predicted_edges"),
                                         "pred_edges": m.get("predicted_edge_count"), "true_edges": m.get("true_edge_count"),
                                         "custom_1_5cm_pass": ev["custom_1_5cm_horizontal_diagnostic"]["pass"],
                                         "three_percent_pass": ev["wall_length_3_percent_threshold_oracle_normalised"]["pass"],
                                         "labels": m.get("edge_labels"),
                                         "whole_room_claim": ev.get("whole_room_claim")}
            rows.append(row)
    return rows


def markdown(result: dict) -> str:
    L = ["# Wall-boundary comparison (budgets x axis hypotheses)", "",
         "EVALUATION-ONLY scoring of frozen results. Not a benchmark-stage completion claim. Every budget and every co-supported",
         "hypothesis is listed, including failures; nothing is selected by truth.", "",
         "## Table", "", "| budget | hyp | retained | status | edges (obs/extrap) | stable/unstable-count/unstable-geom/n.c. | scored | max wall err cm | max rel % | unmatched true/pred | IoU | 1.5cm | 3% | failure |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["table"]:
        st = r.get("stability") or {}
        ev = r.get("evaluation") or {}
        f = lambda v, p=2: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
        fail = (r.get("failure") or "") + (" | EVAL: " + "; ".join(map(str, ev.get("failure_reasons") or [])) if ev.get("failure_reasons") else "")
        L.append(f"| {r['budget']} | {r['hypothesis']} | {f(r.get('retained_capture_fraction'))} | {r['status']} | "
                 f"{f(r.get('edges'))} ({f(r.get('observed_edges'))}/{f(r.get('extrapolated_edges'))}) | "
                 f"{st.get('stable', '-')}/{st.get('unstable_edge_count', '-')}/{st.get('unstable_geometry', '-')}/{st.get('not_comparable', '-')} | "
                 f"{ev.get('status', '-')} | {f(ev.get('max_abs_wall_error_cm'))} | {f(ev.get('max_rel_wall_error_percent'))} | "
                 f"{ev.get('unmatched_true', '-')}/{ev.get('unmatched_pred', '-')} | {f(ev.get('iou'), 3)} | "
                 f"{ev.get('custom_1_5cm_pass', '-')} | {ev.get('three_percent_pass', '-')} | {fail.replace('|', '/')} |")
    L += ["", "## Frame-selection stability (truth-free)", "",
          f"Heuristics (not gates): corner residual <= {STABLE_CORNER_TOL:.0%} of polygon diameter, edge length change <= {STABLE_EDGE_TOL:.0%} after rigid + uniform-scale fit; reflection and cyclic start chosen by minimum residual between predictions only.", "",
          "| a | b | verdict | edges a/b | corner resid (frac diameter) | max edge length change % | reflection |", "|---|---|---|---|---|---|---|"]
    for p in result["stability"]["pairs"]:
        L.append(f"| {p['a']['budget']}/hyp{p['a']['hypothesis_index']} | {p['b']['budget']}/hyp{p['b']['hypothesis_index']} | {p['verdict']} | "
                 f"{p.get('edge_count_a')}/{p.get('edge_count_b')} | {p.get('corner_residual_max_fraction_of_diameter', '-')} | "
                 f"{p.get('edge_length_change_max_percent', '-')} | {(p.get('fit') or {}).get('reflection_used', '-')} |")
    L += ["", "## Residual-source analysis (evaluation only)", ""]
    for key, rs in (result.get("residual_sources") or {}).items():
        if rs.get("available"):
            L.append(f"- {key}: {rs.get('statement', 'no matched edges')}" + (f"  [{rs['unmatched_note']}]" if rs.get("unmatched_note") else ""))
        else:
            L.append(f"- {key}: not available ({rs.get('reason')})")
    L += ["", "## Oracle choices", ""] + [f"- {c}" for c in ORACLE_CHOICES]
    L += ["", "## Limitations", ""] + [f"- {t}" for t in result["limitations"]]
    return "\n".join(L) + "\n"


LIMITATIONS = [
    "One room (seed 17), one scene; no claim about other rooms, texture classes or the benchmark stage.",
    "Budgets (b) and (c) share the same capture; (a) is a separate render of the same scene; (d) is a textured variant. They are not independent samples.",
    "Hypotheses are not matched across budgets; stability lists every pair. Stability thresholds are heuristics.",
    "Evaluation is a 3.0 m oracle-normalised custom diagnostic; whole_room_claim is never allowed (partial retained capture, oracle alignment).",
    "Residual analysis uses A's stated offset uncertainty only; unmodelled bias (DA3 depth scale, pose drift) shows up as 'beyond stated uncertainty'.",
]


def run_comparison(budgets, out_dir: Path, hooks=None, evaluate: bool = True, events=None) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output folder already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    events = events if events is not None else []
    index = run_extraction(budgets, out_dir, hooks, events)             # 1+2: extract, then freeze (inside)
    results = []
    for rec in index["budgets"]:
        for h in rec.get("hypotheses", []):
            if h.get("status") == "ran":
                results.append({"budget": rec["label"], "hypothesis_index": h["hypothesis_index"],
                                "boundary": read_frozen(out_dir, h["files"]["boundary"]["path"], h["files"]["boundary"]["sha256"])})
    stability = stability_matrix(results)
    evaluations = run_evaluation(budgets, out_dir, index, hooks, events) if evaluate else None
    result = {"model": "wall_boundary_budget_hypothesis_comparison", "benchmark_stage_complete": False,
              "evaluation_only_after_freeze": bool(evaluate), "frozen_index": index, "stability": stability,
              "table": table_rows(index, stability, evaluations, out_dir),
              "residual_sources": {k: v["residual_sources"] for k, v in (evaluations or {}).items() if "residual_sources" in v},
              "oracle_choices": ORACLE_CHOICES, "limitations": LIMITATIONS, "event_order": [list(e) for e in events]}
    write_new_json(out_dir / "comparison.json", result)
    (out_dir / "comparison.md").write_text(markdown(result), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------------------------------- ray-scorer sweep
RAY_K_GRID = (1.0, 2.0, 4.0, 8.0)       # tolerance multipliers, fixed BEFORE any result was looked at; never chosen with truth


def variant_name(k: float) -> str:
    return f"boundary_ray_k{k:g}"


def run_ray_variants(out_dir: Path, index: dict, ks, hooks=None, events=None) -> dict:
    """For every hypothesis whose wall lines exist: re-assemble with the real ray scorer at every k (tolerance = k * max(median
    line offset_sd, noise_scale)) and FREEZE each result. NO scene argument exists here. Adds files[variant] to `index`."""
    hooks = {**_default_hooks(), **(hooks or {})}
    events = events if events is not None else []
    out_dir = Path(out_dir)
    record = {"k_grid": [float(k) for k in ks], "tolerance_rule": "tolerance = k * max(median line offset_sd, noise_scale)",
              "k_grid_note": "fixed before any result was looked at; never selected with truth", "variants": []}
    for rec in index["budgets"]:
        if rec.get("status") != "ok":
            continue
        pred = None
        for h in rec["hypotheses"]:
            if h.get("status") != "ran":
                continue
            walls = read_frozen(out_dir, h["files"]["walls"]["path"], h["files"]["walls"]["sha256"])
            for k in ks:
                name = variant_name(k)
                events.append(("assemble_ray_variant", rec["label"], h["hypothesis_index"], float(k)))
                row = {"budget": rec["label"], "hypothesis_index": h["hypothesis_index"], "variant": name, "k": float(k)}
                try:
                    pred = pred if pred is not None else hooks["load_prediction"](rec["prediction"])
                    boundary = hooks["assemble_boundary_ray"](pred, walls, float(k))
                    h["files"][name] = _freeze(out_dir, f"frozen/{rec['label']}/hyp{h['hypothesis_index']}_{name}.json", boundary)
                    row.update(status=boundary.get("status"),
                               scorer={kk: v for kk, v in (boundary.get("free_space_scorer") or {}).items() if kk != "calls"})
                except Exception as exc:
                    row.update(status="failed", error={"type": type(exc).__name__, "message": str(exc),
                                                       "traceback_tail": traceback.format_exc()[-1500:]})
                record["variants"].append(row)
    events.append(("ray_variants_frozen", None))
    write_new_json(out_dir / "ray_variants_index.json", record)
    return record


def polygon_delta(default, variant) -> dict:
    """Truth-free: did the final polygon change between two assemblies of the same walls (same frame, no fitting)?"""
    pa, pb = _poly(default), _poly(variant)
    sa, sb = (default or {}).get("status"), (variant or {}).get("status")
    if pa is None and pb is None:
        return {"change": "unchanged_no_polygon" if sa == sb else "status_changed_no_polygon", "status": [sa, sb]}
    if pa is None or pb is None:
        return {"change": "polygon_appeared" if pa is None else "polygon_lost", "status": [sa, sb]}
    if len(pa) != len(pb):
        return {"change": "topology_changed", "vertices": [len(pa), len(pb)], "status": [sa, sb]}
    diameter = float(np.max(np.linalg.norm(pb[:, None] - pb[None], axis=2)))
    best = min(float(np.linalg.norm(np.roll(pb, -s, axis=0) - pa, axis=1).max()) for s in range(len(pa)))
    return {"change": "identical_polygon" if best <= 1e-6 * max(diameter, 1e-12) else "geometry_changed",
            "max_vertex_shift_fraction_of_diameter": best / max(diameter, 1e-12), "status": [sa, sb]}


def _scheme_counts(boundary) -> dict:
    """Edge / span counts of one assembly (truth-free)."""
    edges = (boundary or {}).get("edges") or []
    spans = [s for e in edges for s in (e.get("free_space") or {}).get("spans", [])]
    scorer = (boundary or {}).get("free_space_scorer") or {}
    rejected_layouts = ((boundary or {}).get("boundary_search") or {}).get("rejected_by_reason", {}).get("free_space_contradiction", 0)
    return {"edges": len(edges), "observed_edges": sum(1 for e in edges if e.get("label") == "observed"),
            "extrapolated_edges": sum(1 for e in edges if e.get("label") == "extrapolated"),
            "final_extrapolated_spans_checked": len(spans),
            "final_extrapolated_spans_accepted": sum(1 for s in spans if not s.get("free_space_contradiction")),
            "layouts_rejected_for_free_space": int(rejected_layouts),
            "candidate_spans_scored": scorer.get("unique_extrapolated_spans_scored"),
            "candidate_spans_accepted": scorer.get("spans_accepted"),
            "candidate_spans_rejected": scorer.get("spans_rejected"),
            "candidate_spans_conflict_diagnostic_only": scorer.get("spans_conflict_diagnostic_only"),
            "tolerance": scorer.get("tolerance")}


def localized_examples(boundary, limit: int = 8) -> list:
    """Which sub-spans the ray scorer rejected or treated as opening-like, and why (candidate spans, final or not)."""
    out = []
    for c in ((boundary or {}).get("free_space_scorer") or {}).get("calls", []):
        states = [col["state"] for col in c.get("columns", [])]
        if c.get("rejected_subspans") or "partial_height_conflict_opening_like" in states:
            out.append({"span": [c["start_xy"], c["end_xy"]], "rejected": bool(c["free_space_contradiction"]),
                        "opening_like_gap_of_line": c.get("interior_gap_of_line"),
                        "conflict_treated_as_diagnostic": bool(c.get("conflict_treated_as_diagnostic")),
                        "rejected_column_fraction": c.get("rejected_column_fraction"), "column_states": states,
                        "rejected_subspans": c.get("rejected_subspans"), "free_space_fraction": c.get("free_space_fraction"),
                        "fraction_domain": c.get("fraction_domain"),
                        "reason": ("rejected: >= reject_fraction of the span is full-height free-space conflict from >= min_views independent "
                                   "source frames, not an interior gap" if c["free_space_contradiction"] else
                                   "conflict kept diagnostic: span is an interior gap between observed spans (opening-like)"
                                   if c.get("conflict_treated_as_diagnostic") else
                                   "kept: conflicting columns cover less than reject_fraction or conflict is partial-height (window-like)")})
    return out[:limit]


def ray_table_rows(index: dict, out_dir: Path, ks, evaluations: dict) -> list:
    rows = []
    for rec in index["budgets"]:
        for h in rec.get("hypotheses", []):
            if h.get("status") != "ran":
                continue
            default = read_frozen(out_dir, h["files"]["boundary"]["path"], h["files"]["boundary"]["sha256"])
            for scheme in ["default"] + [variant_name(k) for k in ks]:
                key = "boundary" if scheme == "default" else scheme
                if key not in h["files"]:
                    rows.append({"budget": rec["label"], "hypothesis": h["hypothesis_index"], "scheme": scheme, "status": "failed"})
                    continue
                b = read_frozen(out_dir, h["files"][key]["path"], h["files"][key]["sha256"])
                row = {"budget": rec["label"], "hypothesis": h["hypothesis_index"], "scheme": scheme, "status": b.get("status"),
                       **_scheme_counts(b), "vs_default": polygon_delta(default, b) if scheme != "default" else None}
                ev = (evaluations.get(scheme) or {}).get(f"{rec['label']}/hyp{h['hypothesis_index']}")
                if ev is not None:
                    e = ev.get("evaluation")
                    m = (e or {}).get("metrics") or {}
                    row["evaluation"] = None if e is None else {
                        "status": e["status"], "iou": m.get("iou"), "max_abs_wall_error_cm": m.get("max_abs_wall_error_cm"),
                        "max_rel_wall_error_percent": m.get("max_rel_wall_error_percent"),
                        "unmatched_true": m.get("unmatched_true_edges"), "unmatched_pred": m.get("unmatched_predicted_edges")}
                rows.append(row)
    return rows


def ray_markdown(result: dict) -> str:
    f = lambda v, p=2: "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))
    L = ["# Wall-boundary comparison v2: default scorer vs real ray scorer (tolerance sweep)", "",
         "Evaluation-only scoring of frozen results; not a benchmark-stage completion claim. k grid "
         f"{result['k_grid']} fixed before any result was looked at; tolerance = k * max(median line offset_sd, noise_scale). "
         "Nothing (k, hypothesis, budget) is selected by truth. Ray fractions are over `wall_patch_area_within_supplied_height_band`, "
         "not along-segment coverage. Candidate spans = every extrapolated span scored while searching layouts (unique); "
         "'final' = spans of the returned polygon.", "",
         "| budget | hyp | scheme | status | edges (obs/extrap) | final spans checked/accepted | candidate spans scored/accepted/rejected/diag-only | layouts rejected (free space) | vs default | scored | max wall err cm | IoU |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["table"]:
        ev = r.get("evaluation") or {}
        vs = r.get("vs_default")
        L.append(f"| {r['budget']} | {r['hypothesis']} | {r['scheme']} | {r['status']} | {r.get('edges')} ({r.get('observed_edges')}/{r.get('extrapolated_edges')}) | "
                 f"{r.get('final_extrapolated_spans_checked')}/{r.get('final_extrapolated_spans_accepted')} | "
                 f"{f(r.get('candidate_spans_scored'))}/{f(r.get('candidate_spans_accepted'))}/{f(r.get('candidate_spans_rejected'))}/{f(r.get('candidate_spans_conflict_diagnostic_only'))} | "
                 f"{r.get('layouts_rejected_for_free_space')} | {'-' if vs is None else vs['change']} | {ev.get('status', '-')} | {f(ev.get('max_abs_wall_error_cm'))} | {f(ev.get('iou'), 3)} |")
    L += ["", "## Cross-budget stability by scheme (truth-free; pairs of other budgets)", "", "| scheme | stable | unstable edge count | unstable geometry | not comparable |", "|---|---|---|---|---|"]
    for scheme, counts in result["stability_by_scheme"].items():
        L.append(f"| {scheme} | {counts['stable']} | {counts['unstable_edge_count']} | {counts['unstable_geometry']} | {counts['not_comparable']} |")
    L += ["", "## Localized evidence examples (candidate spans; ray scorer)", ""]
    for key, ex in result["localized_examples"].items():
        L.append(f"- {key}: {len(ex)} example(s)" + ("" if ex else " (no localized conflicts)"))
        for e in ex[:4]:
            L.append(f"  - span {e['span']}: {'REJECTED' if e['rejected'] else 'kept'}; columns {e['column_states']}; {e['reason']}")
    L += ["", "## Limitations", ""] + [f"- {t}" for t in result["limitations"]]
    return "\n".join(L) + "\n"


RAY_LIMITATIONS = [
    "One room (seed 17); budgets (b)/(c) share a capture and (d) is a textured variant, so they are not independent samples.",
    "The extractor reports no explicit opening flag: 'opening-like' means (i) a gap between two observed spans of one line, or (ii) a conflict at only some heights of a column.",
    "Predicted depth and pose errors cause false free-space conflicts (Codex measured 5-27% per view on supported patches); the policy needs >= 3 independent source frames and full-height conflict.",
    "The 4x4 evidence grid of the ray module bounds sub-span resolution to a quarter of a candidate span.",
    "The tolerance multiplier is a declared assumption; the sweep reports sensitivity, not a calibrated value.",
    "Evaluation is the 3.0 m oracle-normalised custom diagnostic; whole_room_claim is never allowed here.",
]


def run_ray_comparison(budgets, out_dir: Path, ks=RAY_K_GRID, hooks=None, evaluate: bool = True, events=None) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output folder already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    events = events if events is not None else []
    index = run_extraction(budgets, out_dir, hooks, events)                 # default scorer, frozen
    variants = run_ray_variants(out_dir, index, ks, hooks, events)          # real ray scorer per k, frozen
    schemes = ["default"] + [variant_name(k) for k in ks]
    stability_by_scheme, examples = {}, {}
    for scheme in schemes:
        key = "boundary" if scheme == "default" else scheme
        results = [{"budget": rec["label"], "hypothesis_index": h["hypothesis_index"],
                    "boundary": read_frozen(out_dir, h["files"][key]["path"], h["files"][key]["sha256"])}
                   for rec in index["budgets"] for h in rec.get("hypotheses", []) if h.get("status") == "ran" and key in h["files"]]
        pairs = stability_matrix(results)["pairs"]
        stability_by_scheme[scheme] = {v: sum(1 for p in pairs if p["verdict"] == v)
                                       for v in ("stable", "unstable_edge_count", "unstable_geometry", "not_comparable")}
        if scheme != "default":
            for r in results:
                examples[f"{r['budget']}/hyp{r['hypothesis_index']}/{scheme}"] = localized_examples(r["boundary"])
    evaluations = {}
    if evaluate:
        evaluations["default"] = run_evaluation(budgets, out_dir, index, hooks, events)
        for k in ks:
            evaluations[variant_name(k)] = run_evaluation(budgets, out_dir, index, hooks, events, variant=variant_name(k))
    result = {"model": "wall_boundary_ray_scorer_sweep", "benchmark_stage_complete": False, "evaluation_only_after_freeze": bool(evaluate),
              "k_grid": [float(k) for k in ks], "frozen_index": index, "ray_variants": variants,
              "stability_by_scheme": stability_by_scheme, "localized_examples": examples,
              "table": ray_table_rows(index, out_dir, ks, evaluations), "oracle_choices": ORACLE_CHOICES,
              "limitations": RAY_LIMITATIONS, "event_order": [list(e) for e in events]}
    write_new_json(out_dir / "comparison.json", result)
    (out_dir / "comparison.md").write_text(ray_markdown(result), encoding="utf-8")
    return result


CONTROL_DISPLACEMENT_FRACTION = 0.1      # of the cloud extent; fixed in advance


def scorer_controls(out_dir: Path, ks=RAY_K_GRID, hooks=None, stride: int = 8) -> dict:
    """Truth-free control experiment on the frozen walls: is the ray scorer alive? For the longest observed span of every line the
    scorer is asked about (a) the observed span itself (a wall that exists), (b) the same span displaced by
    CONTROL_DISPLACEMENT_FRACTION*extent to each side of the line. Writes <out_dir>/scorer_controls/ (new folder)."""
    hooks = {**_default_hooks(), **(hooks or {})}
    out_dir = Path(out_dir)
    target = out_dir / "scorer_controls"
    if target.exists():
        raise FileExistsError(target)
    import boundary_assembly as ba
    comparison = json.loads((out_dir / "comparison.json").read_text(encoding="utf-8"))
    rows = []
    for rec in comparison["frozen_index"]["budgets"]:
        if rec.get("status") != "ok":
            continue
        pred = hooks["load_prediction"](rec["prediction"])
        for h in rec["hypotheses"]:
            if h.get("status") != "ran":
                continue
            walls = read_frozen(out_dir, h["files"]["walls"]["path"], h["files"]["walls"]["sha256"])
            extent = float(walls["frame"]["extent"])
            for k in ks:
                scorer = ba.real_ray_scorer(pred, walls, ba.derive_ray_tolerance(walls, k)["tolerance"], stride=stride)
                for ln in walls["lines"]:
                    m, d, c = np.asarray(ln["normal_xy"], float), np.asarray(ln["direction_xy"], float), float(ln["offset"])
                    a, b = max(ln["observed_spans"], key=lambda s: s[1] - s[0])
                    half = min(0.5 * (b - a), 0.1 * extent) / 2
                    mid = 0.5 * (a + b)
                    for kind, shift in (("observed_span", 0.0), ("displaced_positive_normal", CONTROL_DISPLACEMENT_FRACTION * extent),
                                        ("displaced_negative_normal", -CONTROL_DISPLACEMENT_FRACTION * extent)):
                        p0, p1 = (c + shift) * m + (mid - half) * d, (c + shift) * m + (mid + half) * d
                        r = scorer(p0, p1)
                        rows.append({"budget": rec["label"], "hypothesis": h["hypothesis_index"], "k": float(k), "line_id": ln["id"], "kind": kind,
                                     "span_length_fraction_of_extent": float(2 * half / extent), "rejected": bool(r["free_space_contradiction"]),
                                     "raw_scorer_boolean_diagnostic": r["raw_scorer_boolean_diagnostic"],
                                     "free_space_fraction": r["free_space_fraction"], "occupied_fraction": r["occupied_fraction"],
                                     "unobserved_fraction": r["unobserved_fraction"], "column_states": [x["state"] for x in r["columns"]],
                                     "fraction_domain": r["fraction_domain"]})
    summary = {}
    for k in ks:
        for kind in ("observed_span", "displaced_positive_normal", "displaced_negative_normal"):
            sel = [r for r in rows if r["k"] == float(k) and r["kind"] == kind]
            summary[f"k{k:g}/{kind}"] = {
                "n": len(sel), "rejected": sum(r["rejected"] for r in sel),
                "raw_scorer_boolean_true": sum(r["raw_scorer_boolean_diagnostic"] for r in sel),
                "partial_height_conflict_columns": sum(s == "partial_height_conflict_opening_like" for r in sel for s in r["column_states"]),
                "mostly_unobserved": sum(r["unobserved_fraction"] > 0.9 for r in sel),
                "mostly_occupied": sum(r["occupied_fraction"] > 0.9 for r in sel)}
    result = {"purpose": "truth-free control: observed wall spans should not be rejected, spans displaced into free space should be, spans behind a wall are unobserved",
              "displacement_fraction_of_extent": CONTROL_DISPLACEMENT_FRACTION, "summary": summary, "rows": rows}
    write_new_json(target / "controls.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scorer-controls", type=Path, default=None, help="existing ray-sweep folder: write scorer_controls/ into it and exit")
    parser.add_argument("--ray-sweep", action="store_true", help="also re-assemble with the real ray scorer over the fixed k grid")
    parser.add_argument("--output-dir", type=Path, default=None, help="NEW folder (refuses to overwrite)")
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS)
    parser.add_argument("--no-evaluation", action="store_true")
    args = parser.parse_args()
    if args.scorer_controls is not None:
        print(json.dumps(scorer_controls(args.scorer_controls)["summary"], indent=1))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.output_dir.exists():
        parser.error(f"output folder already exists: {args.output_dir}")
    budgets = [{**b, "prediction": str(args.outputs_root / b["prediction"]), "scene": str(args.outputs_root / b["scene"])} for b in BUDGETS]
    import boundary_viewer
    if args.ray_sweep:
        result = run_ray_comparison(budgets, args.output_dir, evaluate=not args.no_evaluation)
        for scheme in [None] + [variant_name(k) for k in RAY_K_GRID]:
            payload = boundary_viewer.payload_from_comparison(args.output_dir, scheme)
            tag = "default" if scheme is None else scheme
            boundary_viewer.write_viewer(payload, args.output_dir / f"viewer_{tag}.html")
    else:
        result = run_comparison(budgets, args.output_dir, evaluate=not args.no_evaluation)
        payload = boundary_viewer.payload_from_comparison(args.output_dir)
        boundary_viewer.write_viewer(payload, args.output_dir / "viewer.html")
        boundary_viewer.write_overlay_pngs(payload, args.output_dir / "overlays")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(result["table"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

