"""Observed wall LINES from per-frame depth, kept as multi-view evidence (not planes, not a polygon).

Why not RANSAC planes: walls are sparse, oblique and noisy (5-15 cm on a 6 m room) and floor/ceiling contours truncate the
room. Here depth/confidence/intrinsics/extrinsics are back-projected per frame (every 3D point keeps its frame id and camera
centre), a mid-height band relative to the cloud's own extent along the given axis is projected to 2D, and candidate wall
lines are found by sequential robust multi-line RANSAC. Nothing assumes a wall count, right angles or that walls reach the
floor/ceiling (furniture and occlusion hide wall bottoms/tops).

Evidence rules (see WALL_BOUNDARY_INTERFACE.md for the output schema)
 * one 2D sample per (frame, small spatial cell): dense pixels of one frame are never independent votes;
 * a line is ranked by sum over along-line bins of sqrt(distinct frames in the bin): spatially distributed multi-view support;
 * tolerance is estimated from the data (local cross-view line-fit scatter) and is expressed relative to the cloud extent;
 * uncertainty = leave-one-view-out jackknife (frame bootstrap for a single view) combined with a per-view scatter prior
   that is pooled over the data, so few views (or one view) can never claim centimetre confidence;
 * lines seen from fewer than `weak_frames` frames, or over a short span, carry `weak: true` (single-frame tiny ones are
   dropped); collinear walls separated by a real gap stay separate spans of one line.
No ground truth, camera-up, camera path, or simulator data are read. Units are arbitrary; thresholds are scale-free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "observed_wall_lines"
UNC_METHODS = ("leave_view_out", "bootstrap")


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def common_frame(axis, origin):
    a = _unit(axis)
    seed = np.array([1., 0., 0.]) if abs(a[0]) < .9 else np.array([0., 1., 0.])
    e1 = _unit(np.cross(a, seed))
    e2 = np.cross(a, e1)
    return a, e1, e2


def _cloud(prediction, stride, conf_percentile):
    """Per-frame world points with frame position, plus camera centres. DA3 extrinsics are world-to-camera."""
    depth, extr, intr = prediction["depth"], np.asarray(prediction["extrinsics"], float), np.asarray(prediction["intrinsics"], float)
    conf = prediction.get("confidence")
    n = len(depth)
    ids = np.asarray(prediction["source_indices"]).astype(int) if prediction.get("source_indices") is not None else np.arange(n)
    pts, fpos, centres, frame_ids = [], [], [], []
    for k in range(n):
        d = np.asarray(depth[k], float)
        y, x = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[y, x]
        c = np.asarray(conf[k], float)[y, x] if conf is not None else np.ones_like(z)
        ok = np.isfinite(z) & (z > 0) & np.isfinite(c)
        if ok.sum() < 20:
            continue
        ok &= c >= np.percentile(c[ok], conf_percentile)
        K = intr[k] if intr.ndim == 3 else intr
        pix = np.stack([x[ok], y[ok], np.ones(int(ok.sum()))], axis=1)
        cam = (pix @ np.linalg.inv(K).T) * z[ok, None]
        R, t = extr[k][:3, :3], extr[k][:3, 3]
        pts.append((cam - t) @ R)
        fpos.append(np.full(len(cam), len(frame_ids), dtype=np.int64))
        centres.append(-R.T @ t)
        frame_ids.append(int(ids[k]))
    if not pts:
        return None
    return np.concatenate(pts), np.concatenate(fpos), np.asarray(centres), frame_ids


def _fit_line(P, w=None):
    """Total least squares line m.x = off through points; returns (m, off, centroid)."""
    if w is None:
        c = P.mean(axis=0)
        Q = P - c
        cov = Q.T @ Q / len(P)
    else:
        c = (P * w[:, None]).sum(axis=0) / w.sum()
        Q = P - c
        cov = (Q * w[:, None]).T @ Q / w.sum()
    vals, vecs = np.linalg.eigh(cov)
    m = vecs[:, 0]
    return m, float(m @ c), c


def _cap_score(t, fpos, bin_size):
    """sum over along-line bins of sqrt(#distinct frames in the bin); also (covered bins, distinct frames)."""
    if len(t) == 0:
        return 0.0, 0, 0
    b = np.floor((t - t.min()) / bin_size).astype(np.int64)
    width = int(b.max()) + 1
    keys = np.unique(fpos.astype(np.int64) * width + b)
    _, inv = np.unique(keys % width, return_inverse=True)
    cnt = np.bincount(inv)
    return float(np.sqrt(cnt).sum()), int(len(cnt)), int(len(np.unique(fpos)))


def _spans(t, gap, min_len, min_count=3):
    """Runs of sorted t separated by gaps > gap; keeps runs with length >= min_len and >= min_count samples."""
    if len(t) == 0:
        return [], np.zeros(0, bool)
    order = np.argsort(t)
    ts = t[order]
    cuts = np.flatnonzero(np.diff(ts) > gap) + 1
    keep = np.zeros(len(t), bool)
    spans = []
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(ts)]):
        if hi - lo >= min_count and ts[hi - 1] - ts[lo] >= min_len:
            spans.append([float(ts[lo]), float(ts[hi - 1])])
            keep[order[lo:hi]] = True
    return spans, keep


def _tall_mask(t, ht):
    """Along-line bins whose supporting samples cover enough of the band height: a window sill or door lintel strip inside the
    mid-height band is not a wall (openings leave gaps), while furniture hiding only the lower part keeps the wall."""
    q0, q1, bin_size, min_cov = ht
    ab = np.floor((t - t.min()) / bin_size).astype(np.int64)
    occ = np.zeros((int(ab.max()) + 1, 8), bool)
    for k in range(8):
        sel = (q1 >= k / 8) & (q0 <= (k + 1) / 8)
        occ[ab[sel], k] = True
    return occ.mean(axis=1)[ab] >= min_cov


def _noise_scale(xy, extent):
    """Local cross-view line-fit scatter: low quantile of sqrt(lambda_min) over locally straight neighbourhoods."""
    r = extent * 0.03
    cell = np.floor(xy / r).astype(np.int64)
    key = (cell[:, 0] - cell[:, 0].min()) * (int(cell[:, 1].max() - cell[:, 1].min()) + 1) + (cell[:, 1] - cell[:, 1].min())
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    vals = []
    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(ks)]):
        if hi - lo < 12:
            continue
        Q = xy[order[lo:hi]]
        Q = Q - Q.mean(axis=0)
        ev = np.linalg.eigvalsh(Q.T @ Q / len(Q))
        if ev[1] > (0.2 * r) ** 2 and ev[0] < 0.25 * ev[1]:
            vals.append(np.sqrt(max(ev[0], 0.0)))
    if len(vals) < 5:
        return float(extent * 0.01), len(vals)
    return float(np.clip(np.percentile(vals, 40), extent * 0.001, extent * 0.02)), len(vals)


def _ransac_line(P, fpos, active, tau, bin_size, extent, rng, trials, min_len):
    idx = np.flatnonzero(active)
    if len(idx) < 8:
        return None
    Q, F = P[idx], fpos[idx]
    best, best_score = None, 0.0
    dmin, dmax = 0.04 * extent, 0.5 * extent
    for _ in range(trials):
        i = int(rng.integers(len(Q)))
        dist = np.linalg.norm(Q - Q[i], axis=1)
        cand = np.flatnonzero((dist > dmin) & (dist < dmax))
        if len(cand) == 0:
            continue
        j = int(cand[rng.integers(len(cand))])
        d = Q[j] - Q[i]
        m = np.array([d[1], -d[0]]) / np.linalg.norm(d)
        off = float(m @ Q[i])
        mask = np.abs(Q @ m - off) < tau
        if mask.sum() < 8:
            continue
        direction = np.array([-m[1], m[0]])
        score, _, _ = _cap_score(Q[mask] @ direction, F[mask], bin_size)
        if score > best_score:
            best, best_score = (m, off), score
    if best is None:
        return None
    m, off = best
    for _ in range(3):                      # refine on inliers
        mask = np.abs(Q @ m - off) < tau
        if mask.sum() < 3:
            return None
        m, off, _ = _fit_line(Q[mask])
    return m, off


def _line_tolerance(res, tau):
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res)))) if len(res) else tau
    return float(np.clip(2.5 * mad, tau, 2.5 * tau))


def _refine(P, fpos, m, off, tau, gap, min_len, ht):
    """Iterate: inliers within tolerance (all points), spans, refit. Returns dict or None."""
    tl = tau
    for _ in range(3):
        d = np.array([-m[1], m[0]])
        res = P @ m - off
        cand = np.flatnonzero(np.abs(res) < tl)
        if len(cand) < 5:
            return None
        cand = cand[_tall_mask(P[cand] @ d, (ht[0][cand], ht[1][cand], ht[2], ht[3]))]
        if len(cand) < 5:
            return None
        spans, keep = _spans(P[cand] @ d, gap, min_len)
        if not spans:
            return None
        inl = cand[keep]
        m2, off2, _ = _fit_line(P[inl])
        if float(m2 @ m) < 0:
            m2, off2 = -m2, -off2
        m, off = m2, off2
        tl = _line_tolerance(P[inl] @ m - off, tau)
    d = np.array([-m[1], m[0]])
    cand = np.flatnonzero(np.abs(P @ m - off) < tl)
    if len(cand) < 5:
        return None
    cand = cand[_tall_mask(P[cand] @ d, (ht[0][cand], ht[1][cand], ht[2], ht[3]))]
    spans, keep = _spans(P[cand] @ d, gap, min_len)
    if not spans:
        return None
    return {"m": m, "off": off, "tol": tl, "spans": spans, "inl": cand[keep]}


def _orient(m, off):
    if off < -1e-12 or (abs(off) <= 1e-12 and (m[0] < 0 or (m[0] == 0 and m[1] < 0))):
        return -m, -off
    return m, off


def _leave_view_out(P, fpos, m, off, rng, min_cells=2):
    """Jackknife over frames (or cell bootstrap for a single frame) of line offset at a fixed reference and angle."""
    frames, counts = np.unique(fpos, return_counts=True)
    frames = frames[counts >= min_cells]
    n = max(len(frames), 1)
    c0 = P.mean(axis=0)
    ref = c0 - m * (m @ c0 - off)
    devs, angs = [], []
    if len(frames) >= 2:
        for f in frames:
            keep = fpos != f
            if keep.sum() < 3:
                continue
            m2, o2, _ = _fit_line(P[keep])
            if m2 @ m < 0:
                m2, o2 = -m2, -o2
            devs.append(float(m2 @ ref - o2))
            angs.append(float(np.arctan2(m2[0] * m[1] - m2[1] * m[0], m2 @ m)))
        k = len(devs)
        if k >= 2:
            jo = float(np.sqrt((k - 1) / k * np.sum((np.array(devs) - np.mean(devs)) ** 2)))
            ja = float(np.sqrt((k - 1) / k * np.sum((np.array(angs) - np.mean(angs)) ** 2)))
            return jo, ja, n, "leave_view_out"
    for _ in range(40):                      # single view: bootstrap its spatial cells
        pick = rng.integers(0, len(P), len(P))
        if len(np.unique(P[pick][:, 0])) < 3:
            continue
        m2, o2, _ = _fit_line(P[pick])
        if m2 @ m < 0:
            m2, o2 = -m2, -o2
        devs.append(float(m2 @ ref - o2))
        angs.append(float(np.arctan2(m2[0] * m[1] - m2[1] * m[0], m2 @ m)))
    if len(devs) >= 2:
        return float(np.std(devs, ddof=1)), float(np.std(angs, ddof=1)), n, "bootstrap"
    return 0.0, 0.0, n, "bootstrap"


def _describe(P, H, fpos, cams, line, tau, extent, bin_size, rng, all_xy, all_h, params):
    m, off = _orient(line["m"], line["off"])
    d = np.array([-m[1], m[0]])
    if float(m @ line["m"]) < 0:                      # direction flipped: spans were measured along the old direction
        line = {**line, "spans": [[-b, -a] for a, b in reversed(line["spans"])]}
    inl = line["inl"]
    t = P[inl] @ d
    F = fpos[inl]
    score, covered_bins, _ = _cap_score(t, F, bin_size)
    frames, counts = np.unique(F, return_counts=True)
    good_frames = frames[counts >= 2]
    nfr = int(len(good_frames)) if len(good_frames) else int(len(frames))
    span_total = float(sum(b - a for a, b in line["spans"]))
    # view spread: incidence angle of each frame's rays on the line (spread of per-frame median angles)
    ang = []
    for f in (good_frames if len(good_frames) else frames):
        sel = F == f
        r = P[inl][sel] - cams[f][:2]
        ang.append(float(np.median(np.degrees(np.arctan2(r @ d, np.abs(r @ m))))))
    spread = float(max(ang) - min(ang)) if len(ang) > 1 else 0.0
    bins = np.unique(np.floor((t - t.min()) / bin_size).astype(np.int64))
    total_bins = int(np.floor((t.max() - t.min()) / bin_size)) + 1
    # heights of the supporting surface at ANY height (band-limited detection must not limit the reported range)
    near = np.flatnonzero((np.abs(all_xy @ m - off) < line["tol"]) &
                          np.any([(all_xy @ d >= a - tau) & (all_xy @ d <= b + tau) for a, b in line["spans"]], axis=0))
    hh = all_h[near] if len(near) >= 10 else H[inl]
    jo, ja, n_unc, method = _leave_view_out(P[inl], F, m, off, rng)
    return {"normal_xy": m.tolist(), "offset": float(off), "direction_xy": d.tolist(),
            "observed_spans": [[float(a), float(b)] for a, b in line["spans"]],
            "support": {"distinct_frames": nfr, "frame_ids": [int(f) for f in (good_frames if len(good_frames) else frames)],
                        "view_spread_deg": spread, "along_coverage_fraction": float(len(bins) / max(total_bins, 1)),
                        "weight": float(score * bin_size / extent)},
            "height_range": [float(np.percentile(hh, 2)), float(np.percentile(hh, 98))],
            "_jk": (jo, ja, n_unc, method), "_span_total": span_total, "_tol": float(line["tol"]),
            "_frame_positions": [int(f) for f in (good_frames if len(good_frames) else frames)]}


def _merge(lines, P, fpos, tau, gap, min_len, ht):
    """Merge near-coincident lines (same orientation, offset within tolerance of each other)."""
    lines = list(lines)
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                if abs(float(a["m"] @ b["m"])) < np.cos(np.radians(4.0)):
                    continue
                tol = 1.5 * max(a["tol"], b["tol"])
                da = np.median(np.abs(P[b["inl"]] @ a["m"] - a["off"]))
                db = np.median(np.abs(P[a["inl"]] @ b["m"] - b["off"]))
                if max(da, db) > tol:
                    continue
                inl = np.union1d(a["inl"], b["inl"])
                m, off, _ = _fit_line(P[inl])
                if m @ a["m"] < 0:
                    m, off = -m, -off
                merged = _refine(P, fpos, m, off, tau, gap, min_len, ht)
                if merged is None:
                    continue
                merged["inl"] = np.union1d(merged["inl"], inl[np.abs(P[inl] @ merged["m"] - merged["off"]) < merged["tol"]])
                lines[i] = merged
                del lines[j]
                changed = True
                break
            if changed:
                break
    return lines


def extract_wall_lines(prediction, axis, *, stride=4, conf_percentile=40.0, band=(0.25, 0.75), max_lines=30, trials=600,
                       seed=0, min_cover_fraction=0.06, weak_frames=3, weak_cover_fraction=0.10, hits_per_frame=300,
                       min_height_coverage=0.35):
    """Observed 2D wall lines (common frame of WALL_BOUNDARY_INTERFACE.md) from a DA3-format prediction dict."""
    a = _unit(axis)
    base = {"model": MODEL, "status": "insufficient", "reasons": [], "axis_world": a.tolist(), "frame": None, "noise_scale": None,
            "lines": [], "observations": {"frame_ids": [], "camera_xy": [], "hits_xy": [], "hit_heights": []},
            "params": {"stride": stride, "conf_percentile": conf_percentile, "band": list(band), "trials": trials, "seed": seed}}
    if not np.isfinite(a).all() or np.linalg.norm(a) < .5:
        return {**base, "reasons": ["invalid axis"]}
    cloud = _cloud(prediction, stride, conf_percentile)
    if cloud is None or len(cloud[0]) < 200:
        return {**base, "reasons": ["too few valid depth samples"]}
    pts, fpos, centres, frame_ids = cloud
    finite = np.isfinite(pts).all(axis=1)
    pts, fpos = pts[finite], fpos[finite]
    origin = np.median(pts, axis=0)
    extent = float(np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)))
    if extent <= 0:
        return {**base, "reasons": ["degenerate cloud"]}
    a, e1, e2 = common_frame(a, origin)
    rel = pts - origin
    xy_all = np.stack([rel @ e1, rel @ e2], axis=1)
    h_all = rel @ a
    cams = (np.asarray(centres) - origin)
    cam_xy = np.stack([cams @ e1, cams @ e2], axis=1)
    frame = {"origin_world": origin.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "extent": extent}
    obs = {"frame_ids": [int(f) for f in frame_ids], "camera_xy": cam_xy.tolist(), "hits_xy": [], "hit_heights": []}
    for f in range(len(frame_ids)):
        idx = np.flatnonzero(fpos == f)
        if len(idx) > hits_per_frame:
            idx = idx[np.linspace(0, len(idx) - 1, hits_per_frame).astype(int)]
        obs["hits_xy"].append(xy_all[idx].tolist())
        obs["hit_heights"].append(h_all[idx].tolist())
    base = {**base, "frame": frame, "observations": obs}

    # Mid-height band relative to the cloud's own extent along the axis (walls need not reach floor or ceiling).
    lo, hi = np.percentile(h_all, [2, 98])
    in_band = (h_all >= lo + band[0] * (hi - lo)) & (h_all <= lo + band[1] * (hi - lo))
    if in_band.sum() < 100:
        return {**base, "reasons": ["too few samples in the mid-height band"]}
    # One sample per (frame, spatial cell): dense pixels of one frame are not independent votes.
    c0 = extent * 0.004
    cell = np.floor(xy_all[in_band] / c0).astype(np.int64)
    cell -= cell.min(axis=0)
    key = (fpos[in_band] * (int(cell[:, 0].max()) + 1) + cell[:, 0]) * (int(cell[:, 1].max()) + 1) + cell[:, 1]
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    P = np.stack([np.bincount(inv, weights=xy_all[in_band][:, k]) / cnt for k in range(2)], axis=1)
    H = np.bincount(inv, weights=h_all[in_band]) / cnt
    hb = (lo + band[0] * (hi - lo), lo + band[1] * (hi - lo))
    hmin, hmax = np.full(len(P), np.inf), np.full(len(P), -np.inf)
    np.minimum.at(hmin, inv, h_all[in_band]); np.maximum.at(hmax, inv, h_all[in_band])
    q0, q1 = (hmin - hb[0]) / (hb[1] - hb[0]), (hmax - hb[0]) / (hb[1] - hb[0])
    pf = fpos[in_band][first]
    sigma, n_cells = _noise_scale(P, extent)
    tau = float(np.clip(3.0 * sigma, extent * 0.003, extent * 0.03))
    bin_size = max(extent * 0.02, 2 * tau)
    gap = max(extent * 0.04, 6 * tau)
    min_len = extent * 0.04
    ht = (q0, q1, bin_size, min_height_coverage)
    rng = np.random.default_rng(seed)
    active = np.ones(len(P), bool)
    found = []
    min_bins = max(2, int(np.ceil(min_cover_fraction * extent / bin_size)))
    for _ in range(max_lines):
        cand = _ransac_line(P, pf, active, tau, bin_size, extent, rng, trials, min_len)
        if cand is None:
            break
        line = _refine(P, pf, cand[0], cand[1], tau, gap, min_len, ht)
        if line is None:                     # no span long enough: discard these samples and keep looking
            active &= ~(np.abs(P @ cand[0] - cand[1]) < tau)
            continue
        act = line["inl"][active[line["inl"]]]   # ranking evidence = samples not already explained by a found line
        _, covered, _ = _cap_score(P[act] @ np.array([-line["m"][1], line["m"][0]]), pf[act], bin_size)
        if len(act) < 5 or covered < min_bins:
            break
        found.append(line)
        active &= ~(np.abs(P @ line["m"] - line["off"]) < line["tol"])
    found = _merge(found, P, pf, tau, gap, min_len, ht)

    described = [_describe(P, H, pf, cams, ln, tau, extent, bin_size, np.random.default_rng(seed + 1 + k), xy_all, h_all, base["params"])
                 for k, ln in enumerate(found)]
    # Pooled per-view scatter of line offset (jackknife sd * sqrt(n) for lines with >= 3 views), never below tau / 2.
    per_view = [d["_jk"][0] * np.sqrt(d["_jk"][2]) for d in described if d["_jk"][3] == "leave_view_out" and d["_jk"][2] >= 3]
    s_view = max(float(np.median(per_view)) if per_view else tau, tau / 2.0)
    lines = []
    for d in described:
        jo, ja, n, method = d.pop("_jk")
        span_total = d.pop("_span_total")
        d.pop("_tol")
        d.pop("_frame_positions")
        infl = 3.0 if n <= 1 else 1.5 if n == 2 else 1.0
        offset_sd = float(np.sqrt(jo ** 2 + s_view ** 2 / n) * infl)
        angle_sd = float(np.degrees(np.sqrt(ja ** 2 + (s_view * np.sqrt(12.0) / (max(span_total, min_len) * np.sqrt(n))) ** 2) * infl))
        d["uncertainty"] = {"offset_sd": offset_sd, "angle_sd_deg": angle_sd, "method": method}
        sup = d["support"]
        reasons = []
        if sup["distinct_frames"] < weak_frames:
            reasons.append(f"fewer than {weak_frames} distinct frames")
        if span_total < weak_cover_fraction * extent:
            reasons.append("short observed span")
        d["weak"] = bool(reasons)
        d["weak_reasons"] = reasons
        if sup["distinct_frames"] <= 1 and span_total < weak_cover_fraction * extent:
            continue                          # single-frame tiny patch: dropped
        lines.append(d)
    lines.sort(key=lambda d: (-d["support"]["weight"], d["offset"], d["normal_xy"][0]))
    for k, d in enumerate(lines):
        d["id"] = k
    out = {**base, "noise_scale": float(sigma), "lines": [{"id": d["id"], **{k: v for k, v in d.items() if k != "id"}} for d in lines]}
    out["params"] = {**base["params"], "tolerance": float(tau), "along_bin": float(bin_size), "gap": float(gap),
                     "per_view_offset_scatter": float(s_view), "noise_cells": int(n_cells), "band_samples": int(len(P)),
                     "height_band": [float(lo + band[0] * (hi - lo)), float(lo + band[1] * (hi - lo))]}
    out["status"] = "ok" if lines else "insufficient"
    out["reasons"] = [] if lines else ["no line passed the multi-view support test"]
    return out


def validate_walls_schema(d):
    """Return a list of schema problems for an observed_wall_lines dict (empty list = valid). Producers other than
    extract_wall_lines (synthetic fixtures) may use this too."""
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)

    if not isinstance(d, dict):
        return ["not a dict"]
    need(d.get("model") == MODEL, "model must be 'observed_wall_lines'")
    need(isinstance(d.get("status"), str), "status must be a string")
    need(isinstance(d.get("reasons"), list), "reasons must be a list")
    fr = d.get("frame")
    if d.get("status") == "ok" or fr is not None:
        try:
            need(all(len(fr[k]) == 3 for k in ("origin_world", "e1", "e2")) and float(fr["extent"]) > 0, "bad frame")
            need(np.isfinite(np.asarray(fr["origin_world"], float)).all(), "frame origin not finite")
        except (TypeError, KeyError, ValueError):
            errs.append("bad frame")
    lines = d.get("lines")
    if not isinstance(lines, list):
        return errs + ["lines must be a list"]
    need(d.get("status") != "ok" or len(lines) > 0, "status ok requires lines")
    ids = set()
    for ln in lines:
        try:
            lid = ln["id"]
            need(lid not in ids, f"duplicate line id {lid}")
            ids.add(lid)
            m, dr = np.asarray(ln["normal_xy"], float), np.asarray(ln["direction_xy"], float)
            need(m.shape == (2,) and abs(np.linalg.norm(m) - 1) < 1e-6, f"line {lid}: normal not unit")
            need(dr.shape == (2,) and abs(np.linalg.norm(dr) - 1) < 1e-6 and abs(float(m @ dr)) < 1e-6,
                 f"line {lid}: direction not unit/perpendicular")
            need(np.isfinite(float(ln["offset"])), f"line {lid}: offset not finite")
            spans = ln["observed_spans"]
            need(len(spans) >= 1 and all(len(s) == 2 and np.isfinite(s).all() and s[0] < s[1] for s in spans),
                 f"line {lid}: bad spans")
            need(all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), f"line {lid}: spans overlap/unsorted")
            s = ln["support"]
            need(int(s["distinct_frames"]) >= 1 and len(s["frame_ids"]) >= 1, f"line {lid}: support frames")
            need(0 <= float(s["along_coverage_fraction"]) <= 1 and float(s["view_spread_deg"]) >= 0 and float(s["weight"]) >= 0,
                 f"line {lid}: support values")
            u = ln["uncertainty"]
            need(u["method"] in UNC_METHODS and float(u["offset_sd"]) >= 0 and float(u["angle_sd_deg"]) >= 0,
                 f"line {lid}: uncertainty")
            hr = ln["height_range"]
            need(len(hr) == 2 and hr[0] <= hr[1], f"line {lid}: height_range")
        except (TypeError, KeyError, ValueError, IndexError):
            errs.append(f"line malformed: {ln!r:.80}")
    o = d.get("observations")
    try:
        n = len(o["frame_ids"])
        need(len(o["camera_xy"]) == n and len(o["hits_xy"]) == n and len(o["hit_heights"]) == n, "observations length mismatch")
        need(all(len(a) == len(b) for a, b in zip(o["hits_xy"], o["hit_heights"])), "hits/heights shape mismatch")
    except (TypeError, KeyError):
        errs.append("observations missing")
    return errs


def check_walls_schema(d):
    errs = validate_walls_schema(d)
    if errs:
        raise ValueError("; ".join(errs))
    return d


def _summary(result):
    return {"status": result["status"], "axis_world": result["axis_world"], "noise_scale": result["noise_scale"],
            "extent": result["frame"]["extent"] if result["frame"] else None, "line_count": len(result["lines"]),
            "weak_line_count": sum(1 for ln in result["lines"] if ln.get("weak")),
            "lines": [{"id": ln["id"], "weak": ln["weak"], "span_count": len(ln["observed_spans"]),
                       "span_lengths": [round(b - a, 4) for a, b in ln["observed_spans"]],
                       "distinct_frames": ln["support"]["distinct_frames"], "view_spread_deg": round(ln["support"]["view_spread_deg"], 1),
                       "coverage": round(ln["support"]["along_coverage_fraction"], 3), "weight": round(ln["support"]["weight"], 4),
                       "offset_sd": round(ln["uncertainty"]["offset_sd"], 5), "angle_sd_deg": round(ln["uncertainty"]["angle_sd_deg"], 3)}
                      for ln in result["lines"]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new folder for walls_axis<k>.json and summary.json")
    parser.add_argument("--axis", type=float, nargs=3, help="explicit structural axis (world); default: ranked hypotheses")
    parser.add_argument("--planes", type=Path, help="planes.json from extract_planes (else extracted from the prediction)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output folder already exists: {args.output_dir}")
    with np.load(args.prediction) as data:
        pred = {k: data[k] for k in data.files if k != "images"}
    axes = []
    if args.axis:
        axes = [list(args.axis)]
    else:
        from extract_planes import extract, points_from_prediction
        from geometry_orientation import axis_hypotheses
        points = points_from_prediction(pred)
        planes = json.loads(args.planes.read_text(encoding="utf-8")) if args.planes else extract(points)
        hyp = axis_hypotheses(planes, points)
        axes = [h["axis_world"] for h in hyp["hypotheses"][:args.top_k]]
    args.output_dir.mkdir(parents=True)
    summaries = []
    for k, axis in enumerate(axes):
        result = extract_wall_lines(pred, axis)
        check_walls_schema(result)
        path = args.output_dir / f"walls_axis{k}.json"
        path.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
        summaries.append({"file": path.name, **_summary(result)})
        print(path)
    (args.output_dir / "summary.json").write_text(json.dumps({"prediction": str(args.prediction), "axes": summaries}, indent=1,
                                                           allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


