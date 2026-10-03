"""Global Sim(3) refinement of the accepted window-link graph (truth-free adapter + CLI).

The existing windowed reconstruction (windowed_reconstruction.merge_window_graph) keeps only a spanning forest of the accepted
window links and chains transforms along it, so noise accumulates along tree paths while every other accepted (cycle-closing)
link is merely reported. This module keeps that tree solution as the BASELINE and then feeds ALL accepted links, with
link-quality weights, to the generic robust Sim(3) pose-graph optimizer (sim3_pose_graph.optimize_sim3_graph):

  1. re-link window pairs from the cached DA3 window pool with the SAME frozen gates recorded in the baseline manifest
     (windowed_reconstruction.link_windows via merge_window_graph; nothing is retuned) and verify the re-derived tree
     solution numerically against the baseline prediction.npz;
  2. optimise one transform per window (root gauge, components kept separate, robust loss);
  3. apply the refined transforms to cameras AND depth (Xg = s Q Xl + b; Rg = R Q^T, tg = s t - Rg b, depth *= s,
     intrinsics unchanged) and assemble with the baseline rule (most central window per (frame, component));
  4. write prediction.npz, largest_component.npz and a diagnostic manifest.json (all truth-free).

Link weight (scale-free; every term is a link's own diagnostic divided by the frozen gate that accepted it, so each is in 0..1):
    u = rms_fraction_of_extent / max_rms_fraction          (fit residual)
    v = max_rotation_disagreement_deg / max_pose_rotation_deg  (shared-frame pose disagreement)
    w = (1 - inlier_fraction) / (1 - min_inlier_fraction)  (outlier share)
    weight = 1 / (WEIGHT_FLOOR + u^2 + v^2 + w^2), then divided by the median over all accepted links.
Translation residuals are normalised by the link's own point extent (trans_scale), so nothing depends on the arbitrary
per-window DA3 scale. No ground truth, simulator, audit or evaluation module is imported or read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import windowed_reconstruction as W

WEIGHT_FLOOR = 0.25
WEIGHT_FORMULA = ("weight = 1 / (0.25 + u^2 + v^2 + w^2); u = rms_fraction_of_extent / max_rms_fraction, "
                  "v = max_rotation_disagreement_deg / max_pose_rotation_deg, w = (1 - inlier_fraction) / (1 - min_inlier_fraction); "
                  "normalised to median 1 over accepted links. trans_scale = link point extent.")
IDENTITY = (1.0, np.eye(3), np.zeros(3))


# --------------------------------------------------------------------------- link graph
def params_from_manifest(manifest):
    p = dict(manifest["merge"]["params"])
    if "scale_range" in p:
        p["scale_range"] = tuple(p["scale_range"])
    return p


def link_weight(link, params):
    u = link["rms_fraction_of_extent"] / params["max_rms_fraction"]
    v = link["max_rotation_disagreement_deg"] / params["max_pose_rotation_deg"]
    w = (1.0 - link["inlier_fraction"]) / max(1.0 - params["min_inlier_fraction"], 1e-12)
    return 1.0 / (WEIGHT_FLOOR + u * u + v * v + w * w), {"u": float(u), "v": float(v), "w": float(w)}


def accepted_links(links):
    """{(i, j): link} for links that are ok and not pruned (j -> i, i < j)."""
    return {tuple(l["windows"]): l for l in links if l["ok"] and not l.get("pruned_by_cycle_check")}


def tree_solution(n, links):
    """Re-derive the baseline tree solution exactly as windowed_reconstruction.merge_window_graph does."""
    acc = accepted_links(links)
    attempted = {i for l in links for i in l["windows"]}
    isolated = sorted(attempted - {i for pair in acc for i in pair})
    tree = W._spanning_forest(n, acc)
    transforms, component, tparent, depth = W._orient_forest(n, tree, acc, isolated)
    return {"accepted": acc, "tree": tree, "isolated": isolated, "transforms": transforms, "component": component}


def _tuple(t):
    return float(t[0]), np.asarray(t[1], float), np.asarray(t[2], float)


def link_residual(link, Ti, Tj):
    """Disagreement of a link (j -> i) with the transforms (local -> global) of its two windows. Truth-free."""
    sp, Qp, bp = W._compose(W._invert(*Ti), *Tj)
    Q = np.asarray(link["Q"], float)
    return {"rotation_deg": W.rotation_angle_deg(Q, Qp), "scale_ratio": float(link["s"] / sp),
            "translation_fraction_of_extent": float(np.linalg.norm(np.asarray(link["b"]) - bp) / link["extent"])}


def _r2(res):
    return (np.radians(res["rotation_deg"]) ** 2 + np.log(res["scale_ratio"]) ** 2 + res["translation_fraction_of_extent"] ** 2)


def residual_summary(acc, transforms, weights, tree_edges=None):
    rows = {k: link_residual(l, transforms[k[0]], transforms[k[1]]) for k, l in acc.items()}

    def summ(keys):
        r = [rows[k] for k in keys]
        if not r:
            return {"count": 0}
        w = np.asarray([weights[k] for k in keys]); r2 = np.asarray([_r2(x) for x in r])
        return {"count": len(r), "max_rotation_deg": max(x["rotation_deg"] for x in r),
                "median_rotation_deg": float(np.median([x["rotation_deg"] for x in r])),
                "max_scale_ratio_deviation": max(abs(x["scale_ratio"] - 1) for x in r),
                "max_translation_fraction_of_extent": max(x["translation_fraction_of_extent"] for x in r),
                "median_translation_fraction_of_extent": float(np.median([x["translation_fraction_of_extent"] for x in r])),
                "weighted_sum_squared_residual": float((w * r2).sum()), "weighted_mean_squared_residual": float((w * r2).sum() / w.sum())}
    out = {"all_links": summ(list(rows))}
    if tree_edges is not None:
        out["non_tree_links"] = summ([k for k in rows if k not in tree_edges])
    return rows, out


# --------------------------------------------------------------------------- poses per window
def window_poses(windows, transforms, component, keep):
    """frame -> list of (window, rotation, centre) under per-window transforms (poses only; no depth needed)."""
    by_frame = {}
    for wi in keep:
        E = W.transform_extrinsics(windows[wi]["extrinsics"], *transforms[wi]); C = W.camera_center(E)
        for k, f in enumerate(windows[wi]["source_indices"]):
            by_frame.setdefault((int(f), component[wi]), []).append((wi, E[k, :3, :3], C[k]))
    return by_frame


def _extent_of(centers):
    centers = np.asarray(centers)
    return float(np.linalg.norm(np.percentile(centers, 95, axis=0) - np.percentile(centers, 5, axis=0))) or 1.0


def multiwindow_disagreement(by_frame, chosen_window, extents):
    """Per frame: largest rotation / centre disagreement between the selected window's pose and the other windows containing it."""
    out = {}
    for key, lst in by_frame.items():
        ref = next(x for x in lst if x[0] == chosen_window[key])
        rot = max(W.rotation_angle_deg(ref[1], x[1]) for x in lst)
        ctr = max(float(np.linalg.norm(ref[2] - x[2])) for x in lst) / extents[key[1]]
        out[key] = {"windows": len(lst), "max_rotation_deg": rot, "max_center_fraction_of_extent": ctr}
    return out


def _stats(values):
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "mean": float(v.mean()), "max": float(v.max())} if len(v) else {}


# --------------------------------------------------------------------------- refinement
def to_edges(acc, params):
    weights = {k: link_weight(l, params)[0] for k, l in acc.items()}
    med = float(np.median(list(weights.values()))) if weights else 1.0
    keys = sorted(acc)
    edges = [{"i": int(i), "j": int(j), "s": float(acc[(i, j)]["s"]), "Q": np.asarray(acc[(i, j)]["Q"], float),
              "b": np.asarray(acc[(i, j)]["b"], float), "weight": weights[(i, j)] / med, "trans_scale": float(acc[(i, j)]["extent"])}
             for i, j in keys]
    return keys, edges, {k: weights[k] / med for k in keys}


def default_optimizer():
    from sim3_pose_graph import optimize_sim3_graph
    return optimize_sim3_graph


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def refine_links(windows, links, params, optimizer=None, optimizer_kwargs=None):
    """Tree baseline + refined solution from an accepted-link list (each link: windows=[i,j], ok, s, Q, b, extent, diagnostics).

    Returns dict(prediction, tree_transforms, refined_transforms, component, report). Windows are not modified."""
    optimizer = optimizer or default_optimizer()
    n = len(windows); tr = tree_solution(n, links); acc = tr["accepted"]
    keys, edges, weights = to_edges(acc, params)
    t0 = time.time()
    init = [(float(t[0]), np.asarray(t[1]), np.asarray(t[2])) if t is not None else IDENTITY for t in tr["transforms"]]   # tree solution = start
    res = optimizer(n, edges, init=init, **{"loss": "cauchy", **(optimizer_kwargs or {})})
    opt_seconds = time.time() - t0
    keep = [k for k in range(n) if tr["component"][k] >= 0]
    tree_T = tr["transforms"]
    ref_raw = [_tuple((r["s"], r["Q"], r["b"])) for r in res["transforms"]]
    # component ids as in the baseline (lowest window first); the optimizer's partition must equal the tree's
    comp = tr["component"]
    opt_comp = res["component_of_node"]
    for a in keep:
        for b in keep:
            if (comp[a] == comp[b]) != (opt_comp[a] == opt_comp[b]):
                raise RuntimeError("optimizer components differ from the accepted-link graph components")
    # common gauge: the refined solution is re-expressed so that the tree root window of every component is the identity
    refined_T = [None] * n
    for c in sorted({comp[k] for k in keep}):
        members = [k for k in keep if comp[k] == c]; root = min(members)
        G = W._invert(*ref_raw[root])
        for k in members:
            refined_T[k] = W._compose(G, *ref_raw[k])
    weights_final = list(res["final_robust_weights"]) if res.get("final_robust_weights") is not None else [1.0] * len(keys)
    outliers = {int(e) for e in (res.get("outlier_edges") or [])}
    pairs_out = {(keys[e][0], keys[e][1]) for e in outliers if isinstance(e, int) and e < len(keys)}
    tree_edges = tr["tree"]
    rows_tree, sum_tree = residual_summary(acc, tree_T, weights, tree_edges)
    rows_ref, sum_ref = residual_summary(acc, refined_T, weights, tree_edges)

    # assembly with the baseline rule
    prediction = W._assemble([windows[k] for k in keep], [comp[k] for k in keep], [refined_T[k] for k in keep])
    # source_window keeps the baseline convention (index into the kept, non-isolated windows); `chosen` maps to original window ids
    chosen = {(int(f), int(c)): keep[int(w)] for f, c, w in zip(prediction["source_indices"], prediction["component_ids"], prediction["source_window"])}

    # truth-free per-frame diagnostics
    pf_tree = window_poses(windows, tree_T, comp, keep); pf_ref = window_poses(windows, refined_T, comp, keep)
    extents = {}
    for c in sorted({comp[k] for k in keep}):
        extents[c] = _extent_of([pf_tree[key][0][2] for key in pf_tree if key[1] == c])
    dis_tree = multiwindow_disagreement(pf_tree, chosen, extents); dis_ref = multiwindow_disagreement(pf_ref, chosen, extents)
    vs_tree = {}
    for key in chosen:
        a = next(x for x in pf_tree[key] if x[0] == chosen[key]); r = next(x for x in pf_ref[key] if x[0] == chosen[key])
        vs_tree[key] = {"rotation_deg": W.rotation_angle_deg(a[1], r[1]), "center_fraction_of_extent": float(np.linalg.norm(a[2] - r[2]) / extents[key[1]])}
    frame_keys = sorted(chosen, key=lambda kk: (kk[1], kk[0]))
    corrections = []
    for k in keep:
        s, Q, b = W._compose(refined_T[k], *W._invert(*tree_T[k]))      # tree global frame -> refined global frame
        corrections.append({"window": k, "component": comp[k], "rotation_deg": W.rotation_angle_deg(Q, np.eye(3)), "abs_log_scale": float(abs(np.log(s))),
                            "translation_fraction_of_extent": float(np.linalg.norm(b) / extents[comp[k]])})
    link_rows = []
    for e, key in enumerate(keys):
        q = link_weight(acc[key], params)
        link_rows.append({"windows": list(key), "tree_edge": key in tree_edges, "weight": weights[key], "quality_terms": q[1],
                          "robust_weight_final": float(weights_final[e]), "outlier": key in pairs_out,
                          "before": rows_tree[key], "after": rows_ref[key]})
    comp_counts = {int(c): int((prediction["component_ids"] == c).sum()) for c in sorted(set(prediction["component_ids"].tolist()))}
    total = len({int(f) for w in windows for f in w["source_indices"]})
    report = {
        "weights": {"formula": WEIGHT_FORMULA, "floor": WEIGHT_FLOOR},
        "links_accepted_used": len(keys), "tree_links": len(tree_edges), "non_tree_links": len(keys) - len(tree_edges),
        "isolated_windows": tr["isolated"], "windows_used": len(keep),
        "residuals_before_tree": sum_tree, "residuals_after_refined": sum_ref,
        "cycle_consistency_before": sum_tree.get("non_tree_links"), "cycle_consistency_after": sum_ref.get("non_tree_links"),
        "objective_before": res.get("objective_before", res.get("initial_objective")), "objective_after": res.get("objective_after", res.get("final_objective")),
        "optimizer": {k: _jsonable(v) for k, v in res.items() if k in ("converged", "iterations", "warnings", "components", "root_of_component", "outlier_edges",
                                                                       "objective_before", "objective_after", "initial_objective", "final_objective")},
        "optimizer_seconds": opt_seconds,
        "outlier_links": [list(k) for k in sorted(pairs_out)],
        "link_details": link_rows,
        "per_window_correction_vs_tree": corrections,
        "per_window_correction_summary": {"rotation_deg": _stats([c["rotation_deg"] for c in corrections]),
                                          "translation_fraction_of_extent": _stats([c["translation_fraction_of_extent"] for c in corrections]),
                                          "abs_log_scale": _stats([c["abs_log_scale"] for c in corrections])},
        "components": len(comp_counts), "component_frame_counts": comp_counts, "frames_available": total,
        "frames_in_largest_component": max(comp_counts.values()), "retained_capture_fraction": max(comp_counts.values()) / total,
        "component_of_window": comp,
        "per_frame": {"keys": [list(k) for k in frame_keys],
                      "refined_vs_tree_rotation_deg": [vs_tree[k]["rotation_deg"] for k in frame_keys],
                      "refined_vs_tree_center_fraction_of_extent": [vs_tree[k]["center_fraction_of_extent"] for k in frame_keys],
                      "multiwindow_rotation_deg_tree": [dis_tree[k]["max_rotation_deg"] for k in frame_keys],
                      "multiwindow_rotation_deg_refined": [dis_ref[k]["max_rotation_deg"] for k in frame_keys],
                      "multiwindow_center_fraction_tree": [dis_tree[k]["max_center_fraction_of_extent"] for k in frame_keys],
                      "multiwindow_center_fraction_refined": [dis_ref[k]["max_center_fraction_of_extent"] for k in frame_keys]},
        "per_frame_summary": {"refined_vs_tree_rotation_deg": _stats([vs_tree[k]["rotation_deg"] for k in frame_keys]),
                              "refined_vs_tree_center_fraction": _stats([vs_tree[k]["center_fraction_of_extent"] for k in frame_keys]),
                              "multiwindow_rotation_deg_tree": _stats([dis_tree[k]["max_rotation_deg"] for k in frame_keys]),
                              "multiwindow_rotation_deg_refined": _stats([dis_ref[k]["max_rotation_deg"] for k in frame_keys]),
                              "multiwindow_center_fraction_tree": _stats([dis_tree[k]["max_center_fraction_of_extent"] for k in frame_keys]),
                              "multiwindow_center_fraction_refined": _stats([dis_ref[k]["max_center_fraction_of_extent"] for k in frame_keys])},
    }
    return {"prediction": prediction, "tree_transforms": tree_T, "refined_transforms": refined_T, "component": comp, "report": report}


# --------------------------------------------------------------------------- cached pool and baseline
def load_pool_windows(pool_dir, groups, resolution):
    """Cached windows only; a missing file is an error (no inference is ever started here)."""
    pool = W.WindowPool.__new__(W.WindowPool); pool.dir = Path(pool_dir); pool.resolution = int(resolution)
    files, windows = [], []
    for g in groups:
        p = pool._file(g)
        if not p.exists():
            raise FileNotFoundError(f"window pool file missing (no inference allowed): {p}")
        files.append(p)
    for p in files:
        with np.load(p) as d:
            windows.append({k: d[k] for k in d.files})
    return windows, files


def verify_baseline(prediction, baseline_npz, links_report, manifest_links, atol=1e-6):
    """Numeric comparison of the re-derived tree prediction with the stored baseline; returns a report (raises on mismatch)."""
    rep = {}
    with np.load(baseline_npz) as d:
        for name in ("source_indices", "component_ids", "source_window"):
            rep[name + "_equal"] = bool(np.array_equal(d[name], prediction[name]))
        for name in ("extrinsics", "intrinsics"):
            rep[name + "_max_abs_diff"] = float(np.abs(d[name].astype(float) - prediction[name].astype(float)).max())
        a, b = d["depth"].astype(np.float64), np.asarray(prediction["depth"], np.float64)
        rep["depth_nonfinite_pattern_equal"] = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        fin = np.isfinite(a) & np.isfinite(b)
        rep["depth_max_rel_diff"] = float((np.abs(a[fin] - b[fin]) / np.maximum(np.abs(a[fin]), 1e-6)).max()) if fin.any() else 0.0
    new = {tuple(l["windows"]): l["ok"] for l in links_report}; old = {tuple(l["windows"]): l["ok"] for l in manifest_links}
    rep["link_accept_sets_equal"] = new == old
    ok = (all(rep[k] for k in rep if k.endswith("_equal") or k.endswith("_pattern_equal")) and rep["extrinsics_max_abs_diff"] < atol and rep["intrinsics_max_abs_diff"] < atol
          and rep["depth_max_rel_diff"] < 1e-5)
    rep["matches"] = bool(ok)
    return rep


def run(pool_dir, baseline_dir, output, optimizer=None, optimizer_kwargs=None):
    t_start = time.time(); baseline_dir = Path(baseline_dir); output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    manifest_path = baseline_dir / "manifest.json"; baseline_npz = baseline_dir / "prediction.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = params_from_manifest(manifest); groups = manifest["window_groups"]; merge = manifest["merge"]
    windows, pool_files = load_pool_windows(pool_dir, groups, manifest["resolution"])
    shared = merge.get("shared_intrinsics") is not None
    max_neighbors = params["window"] // 2                     # the baseline CLI's rule (windowed_reconstruction.main)
    prune = merge.get("cycle_pruning", {})
    gates = prune.get("gates", W.PRUNE) if prune.get("enabled") else W.PRUNE
    t0 = time.time()
    base_pred, base_report = W.merge_window_graph(windows, params, shared, max_neighbors=max_neighbors, prune_cycles=bool(prune.get("enabled")), prune_gates=gates)
    link_seconds = time.time() - t0
    verification = verify_baseline(base_pred, baseline_npz, base_report["links"], merge["links"])
    del base_pred
    if not verification["matches"]:
        raise RuntimeError(f"re-derived baseline does not match {baseline_npz}: {verification}")
    work = [dict(w, intrinsics=np.tile(W.consensus_intrinsics(windows)[0], (len(w["source_indices"]), 1, 1))) for w in windows] if shared else windows
    out = refine_links(work, base_report["links"], params, optimizer, optimizer_kwargs)
    pred = out["prediction"]; rep = out["report"]
    output.mkdir(parents=True)
    np.savez_compressed(output / "prediction.npz", **pred)
    big = int(np.bincount(pred["component_ids"]).argmax()); sel = pred["component_ids"] == big
    np.savez_compressed(output / "largest_component.npz", **{k: v[sel] for k, v in pred.items()})
    tree_T = out["tree_transforms"]; ref_T = out["refined_transforms"]
    ser = lambda T: [None if t is None else {"s": float(t[0]), "Q": np.asarray(t[1]).tolist(), "b": np.asarray(t[2]).tolist()} for t in T]
    out_manifest = {
        "kind": "refined_window_graph", "truth_free": True, "ground_truth_read": False, "external_poses_depth_intrinsics_or_labels_used": False,
        "inputs": {"baseline_dir": str(baseline_dir.resolve()), "baseline_manifest_sha256": W.sha256(manifest_path),
                   "baseline_prediction_sha256": W.sha256(baseline_npz), "baseline_prediction_note": "tree solution kept at this path (not duplicated)",
                   "pool_dir": str(Path(pool_dir).resolve()), "pool_files": {p.name: W.sha256(p) for p in pool_files}},
        "parameters": {"link_params_from_baseline_manifest": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                       "max_neighbors": max_neighbors, "shared_intrinsics": shared, "cycle_pruning": prune,
                       "optimizer_kwargs": {"loss": "cauchy", **(optimizer_kwargs or {})}, "weights": rep["weights"]},
        "window_groups": groups, "links_pairs_attempted": base_report["pairs_linked"], "links_accepted": base_report["links_ok"],
        "links_rejected": base_report["links_rejected"], "link_rejection_reasons": base_report["link_rejection_reasons"],
        "baseline_verification": verification, "tree_transforms": ser(tree_T), "refined_transforms": ser(ref_T),
        "tree_cycle_consistency_reported_by_baseline": merge.get("cycle_consistency"),
        "refinement": rep, "prediction_sha256": W.sha256(output / "prediction.npz"),
        "unscaled": True, "scale_applied": False,
        "runtime_seconds": {"linking_and_tree": link_seconds, "optimizer": rep["optimizer_seconds"], "total": time.time() - t_start}}
    (output / "manifest.json").write_text(json.dumps(_jsonable(out_manifest), indent=1), encoding="utf-8")
    return out_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window-pool", type=Path, required=True, help="cached DA3 window pool directory (read-only; no inference)")
    parser.add_argument("--baseline", type=Path, required=True, help="existing tree-merge output dir (manifest.json + prediction.npz)")
    parser.add_argument("--output", type=Path, required=True, help="NEW output directory (never overwritten)")
    parser.add_argument("--loss", default="cauchy")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.window_pool.is_dir() or not (args.baseline / "manifest.json").exists():
        parser.error("window pool or baseline manifest not found")
    m = run(args.window_pool, args.baseline, args.output, optimizer_kwargs={"loss": args.loss})
    r = m["refinement"]
    print(args.output / "prediction.npz")
    print(json.dumps({"links_used": r["links_accepted_used"], "components": r["components"], "retained_capture_fraction": r["retained_capture_fraction"],
                      "objective_before": r["objective_before"], "objective_after": r["objective_after"],
                      "runtime_seconds": m["runtime_seconds"]["total"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
