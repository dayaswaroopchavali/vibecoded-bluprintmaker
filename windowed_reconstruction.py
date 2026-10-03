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


def transform_window(win, s, Q, b):
    """Local -> global Sim(3) (Xg = s Q Xl + b) applied consistently to poses and depth; intrinsics untouched."""
    Q = np.asarray(Q, float); b = np.asarray(b, float)
    E = np.asarray(win["extrinsics"], float)[:, :3, :]
    R, t = E[:, :, :3], E[:, :, 3]
    Rg = R @ Q.T
    tg = s * t - np.einsum("nij,j->ni", Rg, b)
    out = dict(win)
    out["extrinsics"] = np.concatenate([Rg, tg[:, :, None]], axis=2)
    out["depth"] = np.asarray(win["depth"]) * s
    return out


def camera_center(E):
    E = np.asarray(E)
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def rotation_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# --------------------------------------------------------------------------- linking and merging
def link_windows(a, b, params=PARAMS):
    """Sim(3) taking window b's frame into window a's frame, from the frames both contain."""
    ia = {int(s): k for k, s in enumerate(a["source_indices"])}; ib = {int(s): k for k, s in enumerate(b["source_indices"])}
    shared = sorted(set(ia) & set(ib))
    report = {"shared_frames": shared, "ok": False, "reasons": []}
    if len(shared) < params["min_shared_frames"]:
        report["reasons"].append("insufficient_shared_frames"); return report
    src, dst = [], []
    for f in shared:
        pa, pixa = frame_points(a, ia[f], params); pb, pixb = frame_points(b, ib[f], params)
        common_a = {tuple(p): i for i, p in enumerate(pixa)}
        idx = [(common_a[tuple(p)], j) for j, p in enumerate(pixb) if tuple(p) in common_a]   # same pixel, both confident
        if idx:
            dst.append(pa[[i for i, _ in idx]]); src.append(pb[[j for _, j in idx]])
    if not src:
        report["reasons"].append("no_mutually_visible_points"); return report
    fit = robust_sim3(np.concatenate(src), np.concatenate(dst), params)
    report.update({k: v for k, v in fit.items() if k not in ("ok", "reasons")}); report["reasons"] = list(fit["reasons"])
    if not fit["ok"]:
        return report
    moved = transform_window(b, fit["s"], fit["Q"], fit["b"])
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


def merge_window_graph(windows, params=PARAMS, shared_intrinsics=False, max_neighbors=8):
    """Link every window pair that shares enough frames, keep the best accepted links as a spanning forest, and report
    how well the remaining accepted links agree (cycle residuals) instead of silently chaining drift."""
    intrinsics_report = None
    if shared_intrinsics:
        med, intrinsics_report = consensus_intrinsics(windows)
        windows = [dict(w, intrinsics=np.tile(med, (len(w["source_indices"]), 1, 1))) for w in windows]
    n = len(windows); frames_of = [set(int(f) for f in w["source_indices"]) for w in windows]
    links = {}
    for i in range(n):
        for j in range(i + 1, min(n, i + max_neighbors + 1)):
            if len(frames_of[i] & frames_of[j]) >= params["min_shared_frames"]:
                links[(i, j)] = link_windows(windows[i], windows[j], params)           # transform: window j -> window i

    def quality(l):
        return l["rms_fraction_of_extent"] + 0.002 * l["max_rotation_disagreement_deg"] + 0.2 * l["max_center_disagreement_fraction"]
    accepted = {k: l for k, l in links.items() if l["ok"]}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree = set()
    for (i, j), l in sorted(accepted.items(), key=lambda kv: quality(kv[1])):          # best-supported links first (Kruskal)
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj; tree.add((i, j))
    attempted = {i for pair in links for i in pair}
    linked = {i for pair in accepted for i in pair}
    isolated = sorted(attempted - linked)          # overlapped others yet agreed with none: reported and excluded, never bridged
    adjacency = {i: [] for i in range(n)}
    for i, j in tree:
        adjacency[i].append(j); adjacency[j].append(i)
    transforms, component, comp = [None] * n, [-1] * n, 0
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
                component[v] = comp; stack.append(v)
        comp += 1
    cycle = []
    for (i, j), l in accepted.items():
        if (i, j) in tree or component[i] != component[j]:
            continue
        sp, Qp, bp = _compose(_invert(*transforms[i]), *transforms[j])                  # j -> i predicted by the tree
        cycle.append({"windows": [i, j], "rotation_deg": rotation_angle_deg(np.asarray(l["Q"]), Qp),
                      "scale_ratio": float(l["s"] / sp),
                      "translation_fraction_of_extent": float(np.linalg.norm(np.asarray(l["b"]) - bp) / l["extent"])})
    keep = [k for k in range(n) if component[k] >= 0]
    prediction = _assemble([windows[k] for k in keep], [component[k] for k in keep], [transforms[k] for k in keep])
    comp_sizes = {int(c): int((prediction["component_ids"] == c).sum()) for c in sorted({component[k] for k in keep})}
    total = len({f for fs in frames_of for f in fs})
    covered = {int(f) for f in prediction["source_indices"]}
    report = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "mode": "graph",
              "isolated_windows": isolated, "frames_not_in_any_component": sorted(set().union(*frames_of) - covered),
              "window_count": n, "shared_intrinsics": intrinsics_report, "pairs_linked": len(links), "links_ok": len(accepted),
              "links_rejected": len(links) - len(accepted), "tree_links": len(tree),
              "link_rejection_reasons": {r: sum(r in l["reasons"] for l in links.values()) for r in sorted({r for l in links.values() for r in l["reasons"]})},
              "component_of_window": component, "component_frame_counts": comp_sizes, "components": len(comp_sizes),
              "connected": len(comp_sizes) == 1 and len(covered) == total, "frames_available": total, "frames_in_largest_component": max(comp_sizes.values()),
              "retained_capture_fraction": max(comp_sizes.values()) / total,
              "cycle_consistency": {"count": len(cycle), "max_rotation_deg": max((c["rotation_deg"] for c in cycle), default=None),
                                    "median_rotation_deg": float(np.median([c["rotation_deg"] for c in cycle])) if cycle else None,
                                    "max_translation_fraction_of_extent": max((c["translation_fraction_of_extent"] for c in cycle), default=None),
                                    "max_scale_ratio_deviation": max((abs(c["scale_ratio"] - 1) for c in cycle), default=None)},
              "links": [dict(l, windows=list(k)) for k, l in links.items()],
              "drift_note": "tree transforms accumulate along paths; cycle_consistency measures disagreement with the extra accepted links"}
    return prediction, report


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


def run_da3_windows(paths, groups, resolution):
    """One DA3 load, many windows, strictly sequential on the GPU. RGB only; no poses/depth/intrinsics supplied."""
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
    windows, seconds = [], []
    for g in groups:
        t0 = time.time()
        pred = model.inference([str(paths[i]) for i in g], process_res=resolution, process_res_method="upper_bound_resize")
        windows.append({"depth": np.asarray(pred.depth), "confidence": np.asarray(pred.conf), "extrinsics": np.asarray(pred.extrinsics),
                        "intrinsics": np.asarray(pred.intrinsics), "images": np.asarray(pred.processed_images),
                        "source_indices": np.asarray(g, dtype=np.int64)})
        seconds.append(time.time() - t0)
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
    return windows, {"device": device, "gpu_peak_memory_bytes": peak, "seconds_per_window": seconds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=PARAMS["window"]); parser.add_argument("--stride", type=int, default=PARAMS["stride"])
    parser.add_argument("--resolution", type=int, default=560)
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
    windows, run = run_da3_windows(paths, groups, args.resolution)
    if args.chain:
        prediction, report = merge_windows(windows, params, args.shared_intrinsics)
    else:
        prediction, report = merge_window_graph(windows, params, args.shared_intrinsics, max_neighbors=params["window"] // 2)
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
