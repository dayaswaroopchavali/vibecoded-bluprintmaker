"""Short overlapping DA3 windows merged by robust single Sim(3) transforms (no gravity, no simulator orientation).

DA3 tracks poorly when the camera rolls continuously across a whole video. Short windows of temporally dense frames
see little roll change, so each window is reconstructed on its own (arbitrary similarity frame) and consecutive
windows are tied together through the frames they share:

  * every shared frame is unprojected in both windows (same pixels, own depth and pose), giving corresponding RGB-derived
    3D points; confident, mutually visible points feed a robust Umeyama fit of ONE proper Sim(3), s>0, rotation Q;
  * rank-deficient/collinear overlaps, high residuals, low inlier fractions, reflections and implausible scales are
    rejected, and shared-frame poses must still agree after the transform (disagreement is reported, never hidden);
  * a rejected link starts a new component; identity is never used to bridge a failed overlap and nothing is invented;
  * transform convention (local -> global): Xg = s Q Xl + b; world-to-camera Xc = R Xl + t becomes
    Rg = R Q^T, tg = s t - Rg b, depth *= s, pixel intrinsics unchanged, camera centre Cg = s Q Cl + b.

Nothing here reads ground truth; evaluation against private truth happens elsewhere, after results are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Frozen before held-out tests (see 0009). Calibrated on the single development scene only: window 6/stride 1 and the
# 10 deg rotation gate came from self-consistency (links accepted, cycle residuals) and a dev-scene audit; the camera-centre
# gate is off because in rotation-dominated windows the centres are baseline-limited noise (median disagreement ~10% of the
# scene extent) - centre disagreement is still reported per link. Further changes need a new frozen version.
PARAMS = {"window": 6, "stride": 1, "pixel_stride": 4, "confidence_percentile": 50.0, "min_shared_frames": 3,
          "min_points": 200, "max_rms_fraction": 0.03, "min_inlier_fraction": 0.5, "min_singular_ratio": 0.02,
          "scale_range": (0.2, 5.0), "max_pose_rotation_deg": 10.0, "max_pose_center_fraction": 1e9,
          "irls_iterations": 4, "irls_residual_mads": 3.0}


# --------------------------------------------------------------------------- geometry
def unproject(depth, conf, K, w2c, stride=4, conf_percentile=50.0):
    """World points of one frame (world = R^T (Xc - t)) with their pixel indices; invalid or low-confidence pixels dropped."""
    y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z, c = depth[y, x], conf[y, x]
    valid = np.isfinite(z) & (z > 0) & np.isfinite(c)
    if valid.any():
        valid &= c >= np.percentile(c[valid], conf_percentile)
    pix = np.stack([x[valid], y[valid], np.ones(int(valid.sum()))], axis=1)
    cam = (pix @ np.linalg.inv(K).T) * z[valid, None]
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (cam - t) @ R, np.stack([y[valid], x[valid]], axis=1)


def frame_points(window, k, params):
    return unproject(window["depth"][k], window["confidence"][k], window["intrinsics"][k], window["extrinsics"][k],
                     params["pixel_stride"], params["confidence_percentile"])


def umeyama(src, dst, weights=None):
    """Proper similarity dst ~ s Q src + b (rotation only, det=+1): returns (s, Q, b, singular values of the covariance)."""
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s, mu_d = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd * w[:, None]).T @ xs
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3); sign[2, 2] = np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0
    Q = u @ sign @ vt
    var = (w * (xs ** 2).sum(1)).sum()
    s = float(np.trace(np.diag(d) @ sign) / var) if var > 0 else 0.0
    return s, Q, mu_d - s * Q @ mu_s, d


def robust_sim3(src, dst, params=PARAMS):
    """IRLS/trimmed Umeyama with explicit rejection reasons (never returns a transform it does not trust)."""
    out = {"ok": False, "reasons": [], "n": int(len(src))}
    if len(src) < params["min_points"]:
        out["reasons"].append("too_few_correspondences"); return out
    extent = float(np.linalg.norm(np.percentile(dst, 95, axis=0) - np.percentile(dst, 5, axis=0)))
    centred = dst - dst.mean(0)
    sv = np.linalg.svd(centred, compute_uv=False)
    out["singular_ratio_dst"] = float(sv[2] / max(sv[0], 1e-12)); out["singular_ratio_mid"] = float(sv[1] / max(sv[0], 1e-12))
    if extent <= 0 or sv[1] / max(sv[0], 1e-12) < params["min_singular_ratio"]:
        out["reasons"].append("rank_deficient_or_collinear_overlap"); return out
    keep = np.ones(len(src), bool)
    for _ in range(params["irls_iterations"]):
        s, Q, b, _d = umeyama(src[keep], dst[keep])
        if not (s > 0 and np.isfinite(s)):
            out["reasons"].append("non_positive_scale"); return out
        res = np.linalg.norm(dst - (s * src @ Q.T + b), axis=1)
        mad = float(np.median(np.abs(res - np.median(res))) * 1.4826) + 1e-12
        new_keep = res <= np.median(res) + params["irls_residual_mads"] * mad
        if new_keep.sum() < params["min_points"]:
            out["reasons"].append("too_few_inliers"); return out
        if (new_keep == keep).all():
            break
        keep = new_keep
    s, Q, b, d = umeyama(src[keep], dst[keep])
    res = np.linalg.norm(dst - (s * src @ Q.T + b), axis=1)
    rms = float(np.sqrt((res[keep] ** 2).mean())); inlier_fraction = float(keep.mean())
    out.update({"s": float(s), "Q": Q.tolist(), "b": b.tolist(), "rms": rms, "rms_fraction_of_extent": rms / extent,
                "inlier_fraction": inlier_fraction, "extent": extent, "determinant": float(np.linalg.det(Q))})
    if abs(np.linalg.det(Q) - 1) > 1e-6:
        out["reasons"].append("not_a_proper_rotation")
    if not (params["scale_range"][0] <= s <= params["scale_range"][1]):
        out["reasons"].append("implausible_scale")
    if rms / extent > params["max_rms_fraction"]:
        out["reasons"].append("high_residual")
    if inlier_fraction < params["min_inlier_fraction"]:
        out["reasons"].append("low_inlier_fraction")
    out["ok"] = not out["reasons"]
    return out


def transform_extrinsics(extrinsics, s, Q, b):
    """Poses of a local window in the global frame (Xg = s Q Xl + b): Rg = R Q^T, tg = s t - Rg b."""
    Q = np.asarray(Q, float); b = np.asarray(b, float)
    E = np.asarray(extrinsics, float)[:, :3, :]
    R, t = E[:, :, :3], E[:, :, 3]
    Rg = R @ Q.T
    tg = s * t - np.einsum("nij,j->ni", Rg, b)
    return np.concatenate([Rg, tg[:, :, None]], axis=2)


def transform_window(win, s, Q, b):
    """Local -> global Sim(3) (Xg = s Q Xl + b) applied consistently to poses and depth; intrinsics untouched."""
    out = dict(win)
    out["extrinsics"] = transform_extrinsics(win["extrinsics"], s, Q, b)
    out["depth"] = np.asarray(win["depth"]) * s
    return out

def camera_center(E):
    E = np.asarray(E)
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def rotation_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# --------------------------------------------------------------------------- linking and merging
def shared_correspondences(pa, pixa, pb, pixb):
    """Corresponding 3D points of one shared frame seen in two windows: both windows sample the same pixel grid, so a
    correspondence is simply an identical (y, x) pixel kept as confident by both. Vectorised; ordered by pixel (raster)."""
    ka = pixa[:, 0].astype(np.int64) * 1_000_003 + pixa[:, 1]; kb = pixb[:, 0].astype(np.int64) * 1_000_003 + pixb[:, 1]
    _common, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    return pa[ia], pb[ib]


def link_windows(a, b, params=PARAMS, cache=None):
    """Sim(3) taking window b's frame into window a's frame, from the frames both contain.

    `cache` (optional dict) memoises per-(window, frame) unprojected points while one merge compares many pairs.
    """
    def points(win, k):
        if cache is None:
            return frame_points(win, k, params)
        key = (id(win), k)
        if key not in cache:
            cache[key] = frame_points(win, k, params)
        return cache[key]
    ia = {int(s): k for k, s in enumerate(a["source_indices"])}; ib = {int(s): k for k, s in enumerate(b["source_indices"])}
    shared = sorted(set(ia) & set(ib))
    report = {"shared_frames": shared, "ok": False, "reasons": []}
    if len(shared) < params["min_shared_frames"]:
        report["reasons"].append("insufficient_shared_frames"); return report
    src, dst = [], []
    for f in shared:
        pa, pixa = points(a, ia[f]); pb, pixb = points(b, ib[f])
        d, s_ = shared_correspondences(pa, pixa, pb, pixb)                       # same pixel, both confident
        if len(d):
            dst.append(d); src.append(s_)
    if not src:
        report["reasons"].append("no_mutually_visible_points"); return report
    fit = robust_sim3(np.concatenate(src), np.concatenate(dst), params)
    report.update({k: v for k, v in fit.items() if k not in ("ok", "reasons")}); report["reasons"] = list(fit["reasons"])
    if not fit["ok"]:
        return report
    moved = {"extrinsics": transform_extrinsics(b["extrinsics"], fit["s"], fit["Q"], fit["b"])}
    centers_a, extent = camera_center(a["extrinsics"]), fit["extent"]
    centers_b = camera_center(moved["extrinsics"])
    per_frame = []
    for f in shared:
        Ra, Rb = np.asarray(a["extrinsics"][ia[f]])[:3, :3], np.asarray(moved["extrinsics"][ib[f]])[:3, :3]
        per_frame.append({"frame": f, "rotation_disagreement_deg": rotation_angle_deg(Ra, Rb),
                          "center_disagreement_fraction_of_extent": float(np.linalg.norm(centers_a[ia[f]] - centers_b[ib[f]]) / extent)})
    report["shared_frame_pose_disagreement"] = per_frame
    worst_rot = max(p["rotation_disagreement_deg"] for p in per_frame); worst_ctr = max(p["center_disagreement_fraction_of_extent"] for p in per_frame)
    report["max_rotation_disagreement_deg"], report["max_center_disagreement_fraction"] = worst_rot, worst_ctr
    if worst_rot > params["max_pose_rotation_deg"]:
        report["reasons"].append("shared_frame_rotation_disagreement")
    if worst_ctr > params["max_pose_center_fraction"]:
        report["reasons"].append("shared_frame_center_disagreement")
    report["ok"] = not report["reasons"]
    return report


def _compose(parent, s, Q, b):
    """global(parent) o link: x_g = s_p Q_p (s Q x + b) + b_p."""
    sp, Qp, bp = parent
    return sp * s, Qp @ np.asarray(Q), sp * (Qp @ np.asarray(b)) + bp


def consensus_intrinsics(windows):
    """One camera model for the whole video: per-entry median of DA3's own per-frame intrinsics (RGB-derived).

    DA3 re-estimates focal length in every window (here 335-680 px for the same frame), which distorts unprojected
    geometry non-rigidly and makes a Sim(3) link impossible. A video comes from one camera, so the median is used.
    """
    K = np.concatenate([np.asarray(w["intrinsics"]) for w in windows])
    med = np.median(K, axis=0)
    spread = {name: float(np.percentile(K[:, i, j], 75) - np.percentile(K[:, i, j], 25)) for name, (i, j) in
              {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}.items()}
    return med, {"median_K": med.tolist(), "iqr": spread, "fx_min": float(K[:, 0, 0].min()), "fx_max": float(K[:, 0, 0].max()),
                 "frames": int(len(K))}


def merge_windows(windows, params=PARAMS, shared_intrinsics=False):
    """Chain consecutive windows; a failed link starts a new component. Returns (prediction arrays, report)."""
    intrinsics_report = None
    if shared_intrinsics:
        med, intrinsics_report = consensus_intrinsics(windows)
        windows = [dict(w, intrinsics=np.tile(med, (len(w["source_indices"]), 1, 1))) for w in windows]
    n = len(windows)
    transforms, component, links = [None] * n, [0] * n, []
    transforms[0] = (1.0, np.eye(3), np.zeros(3)); comp = 0
    for i in range(1, n):
        link = link_windows(windows[i - 1], windows[i], params); link["windows"] = [i - 1, i]; links.append(link)
        if link["ok"]:
            transforms[i] = _compose(transforms[i - 1], link["s"], link["Q"], link["b"]); component[i] = comp
        else:
            comp += 1; transforms[i] = (1.0, np.eye(3), np.zeros(3)); component[i] = comp   # own frame; never bridged
    moved = [transform_window(w, *t) for w, t in zip(windows, transforms)]
    chosen = {}                                                              # frame -> (window, local index), most central wins
    for wi, w in enumerate(windows):
        m = len(w["source_indices"])
        for k, f in enumerate(w["source_indices"]):
            centrality = min(k, m - 1 - k)
            key = (int(f), component[wi])
            if key not in chosen or centrality > chosen[key][2]:
                chosen[key] = (wi, k, centrality)
    keys = sorted(chosen, key=lambda kk: (kk[1], kk[0]))
    pick = lambda name: np.stack([np.asarray(moved[chosen[kk][0]][name][chosen[kk][1]]) for kk in keys])
    prediction = {"depth": pick("depth"), "confidence": pick("confidence"), "intrinsics": pick("intrinsics"),
                  "extrinsics": pick("extrinsics"), "images": pick("images"),
                  "source_indices": np.asarray([kk[0] for kk in keys], dtype=np.int64),
                  "component_ids": np.asarray([kk[1] for kk in keys], dtype=np.int64),
                  "source_window": np.asarray([chosen[kk][0] for kk in keys], dtype=np.int64)}
    comp_sizes = {int(c): int((prediction["component_ids"] == c).sum()) for c in sorted(set(component))}
    total = len({int(f) for w in windows for f in w["source_indices"]})
    report = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "window_count": n,
              "shared_intrinsics": intrinsics_report,
              "links": links, "links_ok": sum(l["ok"] for l in links), "links_rejected": sum(not l["ok"] for l in links),
              "component_of_window": component, "component_frame_counts": comp_sizes,
              "components": len(comp_sizes), "connected": len(comp_sizes) == 1,
              "frames_available": total, "frames_in_largest_component": max(comp_sizes.values()),
              "retained_capture_fraction": max(comp_sizes.values()) / total,
              "cumulative_scale_by_window": [float(t[0]) for t in transforms],
              "chain_length_max": int(max(np.bincount(component))) if n else 0,
              "drift_note": "transforms are chained from the first window of each component; residual drift is not corrected here"}
    return prediction, report


def _invert(s, Q, b):
    Q = np.asarray(Q)
    return 1.0 / s, Q.T, -(Q.T @ np.asarray(b)) / s


def _assemble(windows, component, transforms):
    """Pick one source window per (frame, component) - the most central one - and emit the merged prediction arrays."""
    moved = [transform_window(w, *t) for w, t in zip(windows, transforms)]
    chosen = {}
    for wi, w in enumerate(windows):
        m = len(w["source_indices"])
        for k, f in enumerate(w["source_indices"]):
            key = (int(f), component[wi]); centrality = min(k, m - 1 - k)
            if key not in chosen or centrality > chosen[key][2]:
                chosen[key] = (wi, k, centrality)
    keys = sorted(chosen, key=lambda kk: (kk[1], kk[0]))
    pick = lambda name: np.stack([np.asarray(moved[chosen[kk][0]][name][chosen[kk][1]]) for kk in keys])
    return {"depth": pick("depth"), "confidence": pick("confidence"), "intrinsics": pick("intrinsics"),
            "extrinsics": pick("extrinsics"), "images": pick("images"),
            "source_indices": np.asarray([kk[0] for kk in keys], dtype=np.int64),
            "component_ids": np.asarray([kk[1] for kk in keys], dtype=np.int64),
            "source_window": np.asarray([chosen[kk][0] for kk in keys], dtype=np.int64)}


PRUNE = {"max_rotation_deg": 10.0, "max_scale_deviation": 0.15, "max_iterations": 100}   # fixed, RGB-derived gates (not tuned on truth)


def _link_quality(l):
    return l["rms_fraction_of_extent"] + 0.002 * l["max_rotation_disagreement_deg"] + 0.2 * l["max_center_disagreement_fraction"]


def _spanning_forest(n, accepted):
    """Kruskal over the accepted links, best-supported first."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree = set()
    for (i, j), l in sorted(accepted.items(), key=lambda kv: (_link_quality(kv[1]), kv[0])):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj; tree.add((i, j))
    return tree


def _orient_forest(n, tree, accepted, isolated):
    """Chain transforms along the tree from the lowest window of each component. Returns transforms, component ids,
    tree-parent and depth (for tree paths)."""
    adjacency = {i: [] for i in range(n)}
    for i, j in tree:
        adjacency[i].append(j); adjacency[j].append(i)
    transforms, component, comp = [None] * n, [-1] * n, 0
    tparent, depth = [-1] * n, [0] * n
    for iso in isolated:
        component[iso] = -2
    for root in range(n):
        if component[root] != -1:
            continue
        transforms[root] = (1.0, np.eye(3), np.zeros(3)); component[root] = comp; stack = [root]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if component[v] != -1:
                    continue
                lo, hi = (u, v) if u < v else (v, u); l = accepted[(lo, hi)]
                link = (l["s"], l["Q"], l["b"])                                         # hi -> lo
                transforms[v] = _compose(transforms[u], *link) if v == hi else _compose(transforms[u], *_invert(*link))
                component[v] = comp; tparent[v] = u; depth[v] = depth[u] + 1; stack.append(v)
        comp += 1
    return transforms, component, tparent, depth


def _tree_path_edges(i, j, tparent, depth):
    edges = []
    while i != j:
        if depth[i] >= depth[j]:
            edges.append((min(i, tparent[i]), max(i, tparent[i]))); i = tparent[i]
        else:
            edges.append((min(j, tparent[j]), max(j, tparent[j]))); j = tparent[j]
    return edges


def _cycle_residuals(accepted, tree, transforms, component, tparent, depth):
    """Disagreement of every accepted non-tree link with the transform the spanning forest predicts for it."""
    cycle = []
    for (i, j), l in accepted.items():
        if (i, j) in tree or component[i] != component[j]:
            continue
        sp, Qp, bp = _compose(_invert(*transforms[i]), *transforms[j])                  # j -> i predicted by the tree
        cycle.append({"windows": [i, j], "rotation_deg": rotation_angle_deg(np.asarray(l["Q"]), Qp),
                      "scale_ratio": float(l["s"] / sp),
                      "translation_fraction_of_extent": float(np.linalg.norm(np.asarray(l["b"]) - bp) / l["extent"]),
                      "path": _tree_path_edges(i, j, tparent, depth)})
    return cycle


def _cycle_summary(cycle):
    return {"count": len(cycle), "max_rotation_deg": max((c["rotation_deg"] for c in cycle), default=None),
            "median_rotation_deg": float(np.median([c["rotation_deg"] for c in cycle])) if cycle else None,
            "max_translation_fraction_of_extent": max((c["translation_fraction_of_extent"] for c in cycle), default=None),
            "max_scale_ratio_deviation": max((abs(c["scale_ratio"] - 1) for c in cycle), default=None)}


def prune_inconsistent_links(n, accepted, links_isolated_fn, gates=PRUNE):
    """Cycle-consistency pruning to a fixed point (bounded). Each round builds the spanning forest, closes every non-tree
    link into a cycle through the tree and flags the cycle as violated when its rotation disagreement exceeds the gate or
    its scale ratio deviates by more than the gate. Blame is assigned by votes: every link on a violated cycle (the
    non-tree link and its tree path) gets +1, every link on a consistent cycle gets -1; the single link with the highest
    net blame (ties: more violated cycles, then worse individual link quality) is dropped and the forest rebuilt.
    Only RGB-derived link quantities are used. Returns (remaining accepted links, pruned-link report, converged flag)."""
    accepted = dict(accepted); pruned = []
    for iteration in range(gates["max_iterations"]):
        isolated = links_isolated_fn(accepted)
        tree = _spanning_forest(n, accepted)
        transforms, component, tparent, depth = _orient_forest(n, tree, accepted, isolated)
        cycle = _cycle_residuals(accepted, tree, transforms, component, tparent, depth)
        bad = [c for c in cycle if c["rotation_deg"] > gates["max_rotation_deg"] or abs(c["scale_ratio"] - 1) > gates["max_scale_deviation"]]
        if not bad:
            return accepted, pruned, True
        votes = {}
        for c in cycle:
            violated = c in bad
            for e in c["path"] + [tuple(c["windows"])]:
                v = votes.setdefault(e, [0, 0]); v[0 if violated else 1] += 1
        victim = max((e for e, v in votes.items() if v[0] > 0), key=lambda e: (votes[e][0] - votes[e][1], votes[e][0], _link_quality(accepted[e]), e))
        worst = max((c for c in bad if victim in c["path"] or tuple(c["windows"]) == victim), key=lambda c: c["rotation_deg"])
        reasons = []
        if worst["rotation_deg"] > gates["max_rotation_deg"]:
            reasons.append("cycle_rotation_disagreement")
        if abs(worst["scale_ratio"] - 1) > gates["max_scale_deviation"]:
            reasons.append("cycle_scale_disagreement")
        pruned.append({"windows": list(victim), "iteration": iteration, "reasons": reasons, "violated_cycles": votes[victim][0],
                       "consistent_cycles": votes[victim][1], "was_tree_link": victim in tree,
                       "worst_cycle_rotation_deg": worst["rotation_deg"], "worst_cycle_scale_ratio": worst["scale_ratio"],
                       "worst_cycle_translation_fraction_of_extent": worst["translation_fraction_of_extent"]})
        del accepted[victim]
    return accepted, pruned, False


def merge_window_graph(windows, params=PARAMS, shared_intrinsics=False, max_neighbors=8, prune_cycles=False, prune_gates=PRUNE):
    """Link every window pair that shares enough frames, keep the best accepted links as a spanning forest, and report
    how well the remaining accepted links agree (cycle residuals) instead of silently chaining drift.

    With prune_cycles=True (opt-in) accepted links that disagree with the rest of the graph beyond fixed gates are dropped
    first (see prune_inconsistent_links) and listed in report["cycle_pruning"]."""
    intrinsics_report = None
    if shared_intrinsics:
        med, intrinsics_report = consensus_intrinsics(windows)
        windows = [dict(w, intrinsics=np.tile(med, (len(w["source_indices"]), 1, 1))) for w in windows]
    n = len(windows); frames_of = [set(int(f) for f in w["source_indices"]) for w in windows]
    links, cache = {}, {}
    for i in range(n):
        for j in range(i + 1, min(n, i + max_neighbors + 1)):
            if len(frames_of[i] & frames_of[j]) >= params["min_shared_frames"]:
                links[(i, j)] = link_windows(windows[i], windows[j], params, cache)    # transform: window j -> window i
    cache.clear()
    accepted = {k: l for k, l in links.items() if l["ok"]}
    attempted = {i for pair in links for i in pair}
    isolated_of = lambda acc: sorted(attempted - {i for pair in acc for i in pair})   # overlapped others yet agreed with none: excluded, never bridged
    pruning = None
    if prune_cycles:
        before = accepted
        accepted, pruned, converged = prune_inconsistent_links(n, before, isolated_of, prune_gates)
        t0 = _spanning_forest(n, before); iso0 = isolated_of(before)
        pre = _cycle_summary(_cycle_residuals(before, t0, *_orient_forest(n, t0, before, iso0)))
        pruning = {"enabled": True, "gates": dict(prune_gates), "pruned": pruned, "pruned_count": len(pruned), "converged": converged,
                   "windows_isolated_by_pruning": sorted(set(isolated_of(accepted)) - set(iso0)), "pre_prune_cycle_consistency": pre}
    isolated = isolated_of(accepted)
    tree = _spanning_forest(n, accepted)
    transforms, component, tparent, depth = _orient_forest(n, tree, accepted, isolated)
    cycle = _cycle_residuals(accepted, tree, transforms, component, tparent, depth)
    keep = [k for k in range(n) if component[k] >= 0]
    prediction = _assemble([windows[k] for k in keep], [component[k] for k in keep], [transforms[k] for k in keep])
    comp_sizes = {int(c): int((prediction["component_ids"] == c).sum()) for c in sorted({component[k] for k in keep})}
    total = len({f for fs in frames_of for f in fs})
    covered = {int(f) for f in prediction["source_indices"]}
    pruned_keys = {tuple(p["windows"]) for p in pruning["pruned"]} if pruning else set()
    ok_links = sum(l["ok"] for l in links.values())
    report = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "mode": "graph",
              "isolated_windows": isolated, "frames_not_in_any_component": sorted(set().union(*frames_of) - covered),
              "window_count": n, "shared_intrinsics": intrinsics_report, "pairs_linked": len(links), "links_ok": ok_links,
              "links_rejected": len(links) - ok_links, "tree_links": len(tree),
              "link_rejection_reasons": {r: sum(r in l["reasons"] for l in links.values()) for r in sorted({r for l in links.values() for r in l["reasons"]})},
              "component_of_window": component, "component_frame_counts": comp_sizes, "components": len(comp_sizes),
              "connected": len(comp_sizes) == 1 and len(covered) == total, "frames_available": total, "frames_in_largest_component": max(comp_sizes.values()),
              "retained_capture_fraction": max(comp_sizes.values()) / total,
              "cycle_consistency": _cycle_summary(cycle), "cycle_pruning": pruning or {"enabled": False},
              "links": [dict(l, windows=list(k), **({"pruned_by_cycle_check": True} if k in pruned_keys else {})) for k, l in links.items()],
              "drift_note": "tree transforms accumulate along paths; cycle_consistency measures disagreement with the extra accepted links"}
    return prediction, report


GAP_SIZES = (4, 5, 7, 8)   # bridging window lengths, tried only where the base windows failed to connect


def uncovered_runs(prediction, count):
    """Contiguous runs of frames that are not in the largest connected component (the gaps to bridge)."""
    ids = prediction["component_ids"]
    big = int(np.bincount(ids).argmax())
    inside = {int(f) for f in prediction["source_indices"][ids == big]}
    runs, start = [], None
    for f in range(count + 1):
        out = f < count and f not in inside
        if out and start is None:
            start = f
        if not out and start is not None:
            runs.append((start, f - 1)); start = None
    return runs


def plan_gap_windows(runs, count, existing_groups, sizes=GAP_SIZES, margin=4):
    """New windows covering every gap run with a few frames of margin either side, skipping windows already run."""
    have = {tuple(g) for g in existing_groups}; new = []
    for a, b in runs:
        lo, hi = max(0, a - margin), min(count - 1, b + margin)
        for size in sizes:
            for s in range(lo, max(lo, hi - size + 1) + 1):
                g = tuple(range(s, min(count, s + size)))
                if len(g) == size and g not in have:
                    have.add(g); new.append(list(g))
    return new


def merge_with_gap_filling(paths, windows, groups, params, resolution, rounds=1, shared_intrinsics=False, pool=None, prune_cycles=False):
    """Base merge, then up to `rounds` of extra DA3 windows over the uncovered runs, re-merged under the SAME gates.

    Added windows have no special trust: a bridging link must pass the identical checks, and the cycle consistency of
    the final graph is reported so a wrongly accepted bridge shows up instead of being averaged away.
    """
    log, count = [], len(paths)
    windows, groups = list(windows), list(groups)
    order = lambda ws: sorted(range(len(ws)), key=lambda i: (int(ws[i]["source_indices"][0]), len(ws[i]["source_indices"])))
    pred, rep = merge_window_graph(windows, params, shared_intrinsics, max_neighbors=len(windows), prune_cycles=prune_cycles)
    log.append({"round": 0, "windows": len(windows), "components": rep["components"], "retained": rep["retained_capture_fraction"]})
    extra_runs = {"seconds": []}
    for r in range(1, rounds + 1):
        runs = uncovered_runs(pred, count)
        if not runs:
            break
        new = plan_gap_windows(runs, count, groups)
        if not new:
            break
        more, run = run_da3_windows(paths, new, resolution, pool)
        extra_runs["seconds"] += run["seconds_per_window"]; extra_runs["gpu_peak_memory_bytes"] = run["gpu_peak_memory_bytes"]
        windows += more; groups += new
        idx = order(windows); windows = [windows[i] for i in idx]; groups = [groups[i] for i in idx]
        pred, rep = merge_window_graph(windows, params, shared_intrinsics, max_neighbors=len(windows), prune_cycles=prune_cycles)
        log.append({"round": r, "uncovered_runs": runs, "added_windows": len(new), "windows": len(windows), "components": rep["components"],
                    "retained": rep["retained_capture_fraction"], "cycle_max_rotation_deg": rep["cycle_consistency"]["max_rotation_deg"]})
    rep["gap_filling"] = {"rounds": log, "gap_sizes": list(GAP_SIZES), "extra_inference": extra_runs}
    return pred, rep, windows, groups


# --------------------------------------------------------------------------- GPU inference and CLI
def make_windows(count, window, stride):
    starts = list(range(0, max(1, count - window + 1), stride))
    if starts[-1] + window < count:
        starts.append(count - window)
    return [list(range(s, min(count, s + window))) for s in starts]


# --------------------------------------------------------------------------- RGB-driven window planning
ADAPTIVE = {"max_inplane_deg": 30.0, "min_frames": 4, "max_frames": 12, "min_matches": 40}


def consecutive_inplane_angles(images_bgr):
    """In-plane rotation (deg, 2D similarity angle) and verified match count between consecutive RGB frames.

    A heuristic read of observed image motion only: yaw/pitch mostly shift the image and parallax adds noise, so this is
    used to size windows, never to alter images or poses. Unmatchable pairs return (nan, matches) and are treated as
    unknown (the planner then shortens the window).
    """
    import cv2
    orb = cv2.ORB_create(1500); matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    grey = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in images_bgr]
    feats = [orb.detectAndCompute(g, None) for g in grey]
    out = []
    for (ka, da), (kb, db) in zip(feats, feats[1:]):
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            out.append((float("nan"), 0)); continue
        matches = matcher.match(da, db)
        if len(matches) < 8:
            out.append((float("nan"), len(matches))); continue
        pa = np.float32([ka[m.queryIdx].pt for m in matches]); pb = np.float32([kb[m.trainIdx].pt for m in matches])
        M, inliers = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inliers is None:
            out.append((float("nan"), 0)); continue
        out.append((float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), int(inliers.sum())))
    return out


def plan_windows_adaptive(pair_angles, pair_matches, count, cfg=ADAPTIVE):
    """Window groups sized from observed image motion: grow while cumulative in-plane rotation stays small and
    consecutive frames still match; consecutive windows overlap by at least half. Pure function of RGB-derived inputs."""
    groups, start = [], 0
    while start < count:
        end, total = start + 1, 0.0
        while end < count and end - start < cfg["max_frames"]:
            a, m = pair_angles[end - 1], pair_matches[end - 1]
            if not np.isfinite(a) or m < cfg["min_matches"] or abs(total + a) > cfg["max_inplane_deg"]:
                break
            total += a; end += 1
        if end - start < cfg["min_frames"]:
            end = min(count, start + cfg["min_frames"])           # shortest allowed window even where motion is unmatchable
        groups.append(list(range(start, end)))
        if end >= count:
            break
        start += max(1, (end - start) // 2)                       # at least half-window overlap
    return groups


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_video(video, directory):
    import cv2
    cap = cv2.VideoCapture(str(video)); paths = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        p = Path(directory) / f"{len(paths):06d}.png"; cv2.imwrite(str(p), bgr); paths.append(p)
    cap.release()
    return paths


class WindowPool:
    """On-disk cache of DA3 window outputs keyed by (frame group, resolution), so merge experiments can be repeated and
    extended without re-running inference. Pure cache of model outputs; holds no truth and never changes a window."""
    def __init__(self, directory, resolution):
        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True); self.resolution = int(resolution)

    def _file(self, group):
        tag = hashlib.sha1(json.dumps([int(g) for g in group]).encode()).hexdigest()[:8]
        return self.dir / f"r{self.resolution}_{int(group[0]):04d}_{int(group[-1]):04d}_{len(group)}_{tag}.npz"

    def get(self, group):
        p = self._file(group)
        if not p.exists():
            return None
        with np.load(p) as d:
            return {k: d[k] for k in d.files}

    def put(self, group, window):
        p = self._file(group); tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **window); tmp.replace(p)


def run_da3_windows(paths, groups, resolution, pool=None):
    """One DA3 load, many windows, strictly sequential on the GPU. RGB only; no poses/depth/intrinsics supplied.
    With a WindowPool, cached windows are reused and only missing ones are run (the model is not loaded if none are)."""
    windows, seconds = [None] * len(groups), [0.0] * len(groups)
    missing = []
    for n, g in enumerate(groups):
        cached = pool.get(g) if pool is not None else None
        if cached is not None:
            windows[n] = cached
        else:
            missing.append(n)
    peak, device = None, ("cache" if not missing else None)
    if missing:
        repo = Path(__file__).resolve().parents[2]; photo = repo / "outputs" / "photogrammetry"
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(photo / "models" / "hf-cache"); os.environ["MPLCONFIGDIR"] = str(photo / "tools" / "matplotlib-cache")
        sys.path.insert(0, str(sorted((photo / "tools" / "da3-source").glob("*/src"))[0]))
        import torch
        from depth_anything_3.api import DepthAnything3
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DepthAnything3.from_pretrained(str(photo / "models" / "DA3-LARGE")).to(device).eval()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for n in missing:
            g = groups[n]; t0 = time.time()
            pred = model.inference([str(paths[i]) for i in g], process_res=resolution, process_res_method="upper_bound_resize")
            windows[n] = {"depth": np.asarray(pred.depth), "confidence": np.asarray(pred.conf), "extrinsics": np.asarray(pred.extrinsics),
                          "intrinsics": np.asarray(pred.intrinsics), "images": np.asarray(pred.processed_images),
                          "source_indices": np.asarray(g, dtype=np.int64)}
            seconds[n] = time.time() - t0
            if pool is not None:
                pool.put(g, windows[n])
        peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    return windows, {"device": device, "gpu_peak_memory_bytes": peak, "seconds_per_window": seconds,
                     "windows_from_cache": len(groups) - len(missing), "windows_inferred": len(missing)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=PARAMS["window"]); parser.add_argument("--stride", type=int, default=PARAMS["stride"])
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--fill-gaps", type=int, default=0, help="rounds of extra bridging windows over frames outside the largest component")
    parser.add_argument("--window-pool", type=Path, help="directory caching DA3 window outputs (reused/extended across merge experiments)")
    parser.add_argument("--prune-cycles", action="store_true", help="opt-in: drop accepted links that disagree with the rest of the graph (fixed gates)")
    parser.add_argument("--prune-rotation-deg", type=float, default=PRUNE["max_rotation_deg"])
    parser.add_argument("--prune-scale-deviation", type=float, default=PRUNE["max_scale_deviation"])
    parser.add_argument("--chain", action="store_true", help="legacy: only link consecutive windows (default links all overlapping pairs)")
    parser.add_argument("--shared-intrinsics", action="store_true", help="use one consensus camera model (median of DA3 intrinsics) for the whole video")
    parser.add_argument("--adaptive", action="store_true", help="size windows from observed RGB in-plane motion instead of a fixed length")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    params = {**PARAMS, "window": args.window, "stride": args.stride}
    args.output.mkdir(parents=True); frames_dir = args.output / "frames"; frames_dir.mkdir()
    started = time.time(); paths = decode_video(args.video, frames_dir)
    adaptive_report = None
    if args.adaptive:
        import cv2
        pairs = consecutive_inplane_angles([cv2.imread(str(p)) for p in paths])
        groups = plan_windows_adaptive([a for a, _ in pairs], [m for _, m in pairs], len(paths))
        adaptive_report = {"config": ADAPTIVE, "pair_inplane_angles_deg": [a for a, _ in pairs], "pair_matches": [m for _, m in pairs],
                           "window_lengths": [len(g) for g in groups]}
    else:
        groups = make_windows(len(paths), params["window"], params["stride"])
    pool = WindowPool(args.window_pool, args.resolution) if args.window_pool else None
    gates = {**PRUNE, "max_rotation_deg": args.prune_rotation_deg, "max_scale_deviation": args.prune_scale_deviation}
    windows, run = run_da3_windows(paths, groups, args.resolution, pool)
    if args.fill_gaps:
        prediction, report, windows, groups = merge_with_gap_filling(paths, windows, groups, params, args.resolution, args.fill_gaps, args.shared_intrinsics, pool, args.prune_cycles)
        run["seconds_per_window"] += report["gap_filling"]["extra_inference"]["seconds"]
    elif args.chain:
        prediction, report = merge_windows(windows, params, args.shared_intrinsics)
    else:
        prediction, report = merge_window_graph(windows, params, args.shared_intrinsics, max_neighbors=params["window"] // 2, prune_cycles=args.prune_cycles, prune_gates=gates)
    np.savez_compressed(args.output / "prediction.npz", **prediction)
    big = int(np.bincount(prediction["component_ids"]).argmax()); keep = prediction["component_ids"] == big
    np.savez_compressed(args.output / "largest_component.npz", **{k: v[keep] for k, v in prediction.items()})
    manifest = {"input_video": str(args.video.resolve()), "input_video_sha256": sha256(args.video), "frames_decoded": len(paths),
                "window_groups": groups, "adaptive_planning": adaptive_report, "resolution": args.resolution, "inference": run, "merge": report,
                "elapsed_seconds": time.time() - started, "unscaled": True, "scale_applied": False,
                "external_poses_depth_intrinsics_or_labels_used": False, "ground_truth_read": False,
                "prediction_sha256": sha256(args.output / "prediction.npz")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output / "prediction.npz"); print(json.dumps({k: report[k] for k in ("window_count", "links_ok", "links_rejected", "components", "retained_capture_fraction")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
