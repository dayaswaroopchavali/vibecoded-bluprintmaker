"""Complete, ranked, truth-free enumeration of structural-axis candidates from a reconstructed cloud.

Why this exists: geometry_orientation.axis_hypotheses only tries plane normals and zone axes of the (>=4% support) planes
that survive its filters, and on path_01 (full 120-frame DA3 prediction) it returned ONE hypothesis that was not the true
vertical. This module enumerates a COMPLETE candidate set and never selects among them:

 * dominant directions = plane-normal families from extract_planes output  UNION  Gauss-map clusters of per-pixel normals
   of the depth-derived cloud (or of kNN normals of a bare point cloud);
 * candidate axes = every dominant direction (floor/ceiling-like: parallel plane families), plus every zone axis, i.e. an
   axis perpendicular to two (or more) dominant non-parallel directions (wall-prism axis; includes the third member of any
   near-orthogonal triple). Nothing assumes a Manhattan frame: skew / pentagonal rooms and non-orthogonal single
   directions with parallel plane pairs are kept;
 * for every candidate, truth-free evidence: how much of the normal / plane mass it explains, floor/ceiling-like cap pair
   (two broad height peaks at opposite ends spanning the footprint, reported for both axis signs, with the share of camera
   centres between them as evidence ONLY), wall-azimuth corner count, orthogonal-triple membership and, when a runner is
   supplied, closed-polygon feasibility from wall_extraction + boundary_assembly.assemble_boundary_ray on that axis.

Forbidden inputs (enforced by an AST test): camera up / camera path used as gravity, ground truth, simulator, evaluators.
The ranking is a documented heuristic, not a selector: a box-like room is symmetric, so several candidates stay
co-supported and the result says so. The downstream stage must run every candidate that is co-supported (or simply all).
"""
from __future__ import annotations

import itertools

import numpy as np

COS_PARALLEL = float(np.cos(np.radians(12)))       # |n.a| above: plane is perpendicular to the axis (cap-like)
SIN_PERPENDICULAR = float(np.sin(np.radians(12)))  # |n.a| below: plane contains the axis (wall-like)
MERGE_DEG = 5.0                                    # directions closer than this are one direction
CANDIDATE_MERGE_DEG = 3.0
CLUSTER_CONE_DEG = 8.0
MIN_DIRECTION_SHARE = 0.03                         # a direction needs this share of plane support or of normals
MIN_CLUSTER_SHARE = 0.02
MAX_DIRECTIONS = 14
MAX_CANDIDATES = 24
MIN_ZONE_ANGLE_DEG = 20.0                          # a zone axis needs two directions at least this far apart
ZONE_PERPENDICULAR_DEG = 10.0                      # directions this close to perpendicular refine / vote for a zone axis
TRIPLE_ORTHOGONAL_DEG = 10.0
MAX_NORMALS = 40000
CO_SUPPORT_MARGIN = 0.25
# cap-pair evidence (all scale-free)
BIN_FRACTION = 0.03
CELL_FRACTION = 0.03
MIN_CELL_POINTS = 2
MIN_COVER = 0.12
COVERAGE_RATIO = 0.5
MIN_PEAK_SEPARATION = 0.08
MIN_SEPARATION_FRACTION = 0.10
MAX_POINTS = 150000

LIMITATIONS = [
    "The axis sign (floor vs ceiling) is never resolved; both signs are reported and are mirror images.",
    "A box-like room is symmetric: wall pairs look like floor/ceiling pairs, so several candidates stay co-supported.",
    "The ranking is a heuristic over cloud evidence only; it is not validated as a selector and no candidate is called gravity.",
    "Furniture planes are not excluded; the candidate set is capped (see candidate_cap) and the cap is recorded.",
    "Camera centres, if supplied, are evidence only (cameras between the caps); they never define an axis or a sign.",
]


# ------------------------------------------------------------------------------------------------ small helpers
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _angle_deg(a, b):
    """Unsigned angle between two axes (0..90)."""
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0))))


def _basis(axis):
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    a = _unit(np.cross(axis, seed))
    return a, np.cross(axis, a)


def _extent(points):
    return float(np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))


def _finite_points(points):
    pts = np.asarray(points, dtype=float) if points is not None else np.empty((0, 3))
    if pts.ndim != 2 or pts.shape[1:] != (3,):
        return np.empty((0, 3))
    return pts[np.isfinite(pts).all(axis=1)]


def _principal_axis(vectors, weights=None):
    """Unsigned dominant direction of unit vectors (leading eigenvector of the weighted scatter)."""
    v = np.asarray(vectors, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    scatter = (v * w[:, None]).T @ v
    _vals, vecs = np.linalg.eigh(scatter)
    return _unit(vecs[:, -1])


# ------------------------------------------------------------------------------------------------ normals
def normals_from_prediction(prediction, stride: int = 6, conf_percentile: float = 40.0, max_jump: float = 0.04,
                            max_normals: int = MAX_NORMALS, seed: int = 0):
    """Per-pixel surface normals (world frame, unit, sign arbitrary) from a DA3-format prediction (depth, confidence,
    intrinsics, extrinsics). Uses only the depth maps and the predicted poses; neighbouring samples across a depth
    discontinuity (relative jump above `max_jump`) are dropped."""
    out = []
    for depth, conf, K, pose in zip(prediction["depth"], prediction["confidence"], prediction["intrinsics"],
                                    prediction["extrinsics"]):
        y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[y, x].astype(float)
        c = conf[y, x]
        valid = np.isfinite(z) & (z > 0) & np.isfinite(c)
        if valid.sum() < 16:
            continue
        valid &= c >= np.percentile(c[valid], conf_percentile)
        pix = np.stack([x, y, np.ones_like(x)], axis=-1).astype(float)
        cam = (pix @ np.linalg.inv(K).T) * z[..., None]
        dx = cam[:-1, 1:] - cam[:-1, :-1]
        dy = cam[1:, :-1] - cam[:-1, :-1]
        n = np.cross(dx, dy)
        norm = np.linalg.norm(n, axis=-1)
        ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & (norm > 1e-12)
        zc = z[:-1, :-1]
        ok &= (np.abs(z[:-1, 1:] - zc) <= max_jump * zc) & (np.abs(z[1:, :-1] - zc) <= max_jump * zc)
        if not ok.any():
            continue
        n_cam = n[ok] / norm[ok][:, None]
        out.append(n_cam @ pose[:3, :3])        # world = (cam - t) @ R, so a camera-frame direction maps with @ R
    if not out:
        return np.empty((0, 3))
    normals = np.concatenate(out)
    if len(normals) > max_normals:
        normals = normals[np.random.default_rng(seed).choice(len(normals), max_normals, replace=False)]
    return normals


def normals_from_points(points, k: int = 16, max_normals: int = 6000, seed: int = 0):
    """kNN-PCA normals of a bare point cloud (sign arbitrary). Used when no depth maps exist (synthetic clouds)."""
    from scipy.spatial import cKDTree
    pts = _finite_points(points)
    if len(pts) < 4 * k:
        return np.empty((0, 3))
    rng = np.random.default_rng(seed)
    tree = cKDTree(pts)
    idx = rng.choice(len(pts), min(max_normals, len(pts)), replace=False)
    _d, nn = tree.query(pts[idx], k=k)
    nb = pts[nn] - pts[nn].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    vals, vecs = np.linalg.eigh(cov)
    ok = (vals[:, 1] > 1e-18) & (vals[:, 0] < 0.2 * vals[:, 1])              # planar, non-degenerate neighbourhoods only
    return vecs[ok, :, 0]


def cluster_directions(normals, cone_deg: float = CLUSTER_CONE_DEG, min_share: float = MIN_CLUSTER_SHARE,
                       max_clusters: int = 12, max_seeds: int = 1500, seed: int = 0):
    """Gauss-map clustering of unsigned normals: greedy densest cone, refined by principal axis, then removed."""
    n = np.asarray(normals, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3 or len(n) == 0:
        return []
    n = n[np.isfinite(n).all(axis=1)]
    norm = np.linalg.norm(n, axis=1)
    n = n[norm > 1e-9] / norm[norm > 1e-9][:, None]
    if len(n) > MAX_NORMALS:
        n = n[np.random.default_rng(seed).choice(len(n), MAX_NORMALS, replace=False)]
    total = len(n)
    if total < 50:
        return []
    cos = float(np.cos(np.radians(cone_deg)))
    rng = np.random.default_rng(seed)
    remaining = np.ones(total, bool)
    out = []
    for _ in range(max_clusters):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_share * total:
            break
        seeds = idx if len(idx) <= max_seeds else rng.choice(idx, max_seeds, replace=False)
        counts = (np.abs(n[idx] @ n[seeds].T) >= cos).sum(axis=0)
        a = n[seeds[int(np.argmax(counts))]]
        inl = idx
        for _r in range(4):
            inl = idx[np.abs(n[idx] @ a) >= cos]
            if len(inl) < 3:
                break
            a = _principal_axis(n[inl])
        inl = idx[np.abs(n[idx] @ a) >= cos]
        if len(inl) < min_share * total:
            break
        out.append({"axis": a, "share": float(len(inl) / total), "count": int(len(inl))})
        remaining[inl] = False
    return out


# ------------------------------------------------------------------------------------------------ directions
def _plane_records(plane_data):
    out = []
    for i, p in enumerate(plane_data.get("planes", []) if isinstance(plane_data, dict) else []):
        try:
            raw = np.asarray(p["normal"], dtype=float)
            off = float(p["offset"])
            w = max(0, int(p.get("support_count", 0) or 0))
            nn = float(np.linalg.norm(raw))
            if raw.shape != (3,) or not np.isfinite(raw).all() or not np.isfinite(off) or nn < 1e-10 or w <= 0:
                continue
            out.append({"id": i, "normal": raw / nn, "offset": off / nn, "weight": w})
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return out


def _plane_families(planes, merge_deg=MERGE_DEG):
    """Group planes whose (unsigned) normals agree within merge_deg; each family has a support-weighted axis."""
    total = float(sum(p["weight"] for p in planes)) or 1.0
    fams = []
    for p in sorted(planes, key=lambda q: -q["weight"]):
        for f in fams:
            if _angle_deg(f["axis"], p["normal"]) <= merge_deg:
                f["members"].append(p)
                s = np.sign(float(f["axis"] @ p["normal"])) or 1.0
                f["_acc"] = f["_acc"] + s * p["weight"] * p["normal"]
                f["axis"] = _unit(f["_acc"])
                break
        else:
            fams.append({"axis": p["normal"].copy(), "members": [p], "_acc": p["weight"] * p["normal"]})
    for f in fams:
        f["share"] = float(sum(m["weight"] for m in f["members"]) / total)
        f["plane_ids"] = [m["id"] for m in f["members"]]
    return fams


def dominant_directions(plane_data, normals=None, *, merge_deg=MERGE_DEG):
    """Merge plane-normal families and Gauss-map clusters into one list of dominant (unsigned) directions."""
    planes = _plane_records(plane_data)
    families = _plane_families(planes, merge_deg)
    clusters = cluster_directions(normals) if normals is not None and len(normals) else []
    used, dirs = set(), []
    for f in families:
        match = None
        for ci, c in enumerate(clusters):
            if ci not in used and _angle_deg(f["axis"], c["axis"]) <= merge_deg + 1.0:
                match = ci
                break
        if match is not None:
            used.add(match)
        dirs.append({"axis": f["axis"], "plane_share": f["share"], "plane_ids": f["plane_ids"],
                     "normal_share": clusters[match]["share"] if match is not None else 0.0,
                     "sources": ["plane_family"] + (["gauss_cluster"] if match is not None else [])})
    for ci, c in enumerate(clusters):
        if ci not in used:
            dirs.append({"axis": c["axis"], "plane_share": 0.0, "plane_ids": [], "normal_share": c["share"],
                         "sources": ["gauss_cluster"]})
    # clusters can match each other's plane families only once; merge any remaining near-duplicates
    merged = []
    for d in sorted(dirs, key=lambda q: -max(q["plane_share"], q["normal_share"])):
        for m in merged:
            if _angle_deg(m["axis"], d["axis"]) <= merge_deg:
                m["plane_share"] += d["plane_share"]
                m["normal_share"] += d["normal_share"]
                m["plane_ids"] += d["plane_ids"]
                m["sources"] = sorted(set(m["sources"]) | set(d["sources"]))
                break
        else:
            merged.append(d)
    keep = [d for d in merged if max(d["plane_share"], d["normal_share"]) >= MIN_DIRECTION_SHARE]
    keep.sort(key=lambda q: -max(q["plane_share"], q["normal_share"]))
    keep = keep[:MAX_DIRECTIONS]
    for i, d in enumerate(keep):
        d["id"] = i
        d["weight"] = float(max(d["plane_share"], d["normal_share"]))
    return keep, planes


# ------------------------------------------------------------------------------------------------ candidates
def _zone_axis(dirs, a, b):
    """Axis perpendicular to dominant directions a and b, refined by every dominant direction near-perpendicular to it."""
    z = _unit(np.cross(a["axis"], b["axis"]))
    sin_tol = np.sin(np.radians(ZONE_PERPENDICULAR_DEG))
    for _ in range(3):
        members = [d for d in dirs if abs(float(d["axis"] @ z)) <= sin_tol]
        if len(members) < 2:
            break
        # axis best perpendicular to all members: smallest eigenvector of the weighted scatter
        scatter = sum(d["weight"] * np.outer(d["axis"], d["axis"]) for d in members)
        _vals, vecs = np.linalg.eigh(scatter)
        z_new = vecs[:, 0]
        if abs(float(z_new @ z)) < np.cos(np.radians(15)):
            break
        z = z_new if float(z_new @ z) > 0 else -z_new
    return z


def generate_candidates(dirs):
    """All dominant directions + all zone axes, de-duplicated within CANDIDATE_MERGE_DEG, with provenance."""
    raw = []
    for d in dirs:
        raw.append((d["axis"], "dominant_direction", [d["id"]]))
    for a, b in itertools.combinations(dirs, 2):
        ang = _angle_deg(a["axis"], b["axis"])
        if ang < MIN_ZONE_ANGLE_DEG:
            continue
        z = _zone_axis(dirs, a, b)
        kind = "orthogonal_completion" if ang >= 90 - TRIPLE_ORTHOGONAL_DEG else "zone_axis"
        raw.append((z, kind, [a["id"], b["id"]]))
    cands = []
    for axis, kind, src in raw:
        for c in cands:
            if _angle_deg(c["axis"], axis) <= CANDIDATE_MERGE_DEG:
                if kind not in c["kinds"]:
                    c["kinds"].append(kind)
                c["source_direction_ids"] = sorted(set(c["source_direction_ids"]) | set(src))
                break
        else:
            cands.append({"axis": _unit(axis), "kinds": [kind], "source_direction_ids": sorted(set(src))})
    return cands


# ------------------------------------------------------------------------------------------------ evidence
def _cap_evidence(points, axis, extent, camera_centers=None):
    """Two broad height peaks (floor/ceiling-like) along `axis`: coverage = share of footprint cells occupied in the slab."""
    a = _unit(axis)
    u, v = _basis(a)
    h = points @ a
    uv = np.column_stack([points @ u, points @ v])
    hext = float(np.linalg.norm(np.percentile(uv, 95, axis=0) - np.percentile(uv, 5, axis=0)))
    cell = max(CELL_FRACTION * hext, 1e-9)
    ij = np.floor((uv - uv.min(axis=0)) / cell).astype(np.int64)
    cell_id = ij[:, 0] * (int(ij[:, 1].max()) + 1) + ij[:, 1]
    _u, inv_cell, cnt = np.unique(cell_id, return_inverse=True, return_counts=True)
    foot = cnt >= MIN_CELL_POINTS
    if foot.sum() < 4:
        return {"pair_found": False, "reason": "footprint too small", "peaks": []}
    keep = foot[inv_cell]
    binw = max(BIN_FRACTION * extent, 1e-9)
    hb = np.floor((h - h.min()) / binw).astype(np.int64)[keep]
    cid = inv_cell[keep]
    n_cells = len(cnt)
    nb = int(hb.max()) + 1
    pairs = []
    for s in (-1, 0, 1):
        b = hb + s
        ok = (b >= 0) & (b < nb)
        pairs.append(b[ok] * n_cells + cid[ok])
    up = np.unique(np.concatenate(pairs))
    cov = np.bincount(up // n_cells, minlength=nb) / float(foot.sum())
    sep = max(1, int(np.ceil(MIN_PEAK_SEPARATION * extent / binw)))
    work = cov.copy()
    peaks = []
    for _ in range(8):
        b = int(np.argmax(work))
        if work[b] < MIN_COVER:
            break
        peaks.append({"height": float(h.min() + (b + 0.5) * binw), "coverage": float(cov[b])})
        work[max(0, b - sep):b + sep + 1] = -1.0
    out = {"pair_found": False, "peaks": sorted(peaks, key=lambda p: p["height"]), "bin_width": float(binw)}
    if not peaks:
        out["reason"] = "no broad height peak"
        return out
    best = max(p["coverage"] for p in peaks)
    broad = [p for p in out["peaks"] if p["coverage"] >= max(MIN_COVER, COVERAGE_RATIO * best)]
    lo, hi = broad[0], broad[-1]
    out["broad_peak_count"] = len(broad)
    if hi["height"] - lo["height"] < MIN_SEPARATION_FRACTION * extent:
        out["reason"] = "broad peaks not separated"
        return out
    out.update(pair_found=True, low=lo, high=hi, separation_unscaled=float(hi["height"] - lo["height"]),
               cap_support=float(min(lo["coverage"], hi["coverage"])))
    if camera_centers is not None and len(camera_centers):
        ch = np.asarray(camera_centers, float) @ a
        out["camera_share_between_caps"] = float(np.mean((ch >= lo["height"]) & (ch <= hi["height"])))
    return out


def _azimuth_count(wall_axes, axis, cluster_deg=8.0):
    if not wall_axes:
        return 0
    ref = _unit(np.cross(axis, [1, 0, 0] if abs(axis[0]) < .9 else [0, 1, 0]))
    other = np.cross(axis, ref)
    az = sorted(float(np.degrees(np.arctan2(w @ other, w @ ref)) % 180) for w in wall_axes)
    clusters = []
    for x in az:
        if not clusters or x - clusters[-1][-1] > cluster_deg:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    if len(clusters) > 1 and clusters[-1][-1] - clusters[0][0] > 180 - cluster_deg:
        clusters[0] = clusters.pop() + clusters[0]
    return len(clusters)


def _evidence(cand, dirs, planes, normals, pts, middle, extent, camera_centers):
    axis = cand["axis"]
    total_w = sum(d["weight"] for d in dirs) or 1.0
    par, wall, other = [], [], []
    for d in dirs:
        c = abs(float(d["axis"] @ axis))
        (par if c >= COS_PARALLEL else wall if c <= SIN_PERPENDICULAR else other).append(d)
    explained = (sum(d["weight"] for d in par) + sum(d["weight"] for d in wall)) / total_w
    nm_par = nm_perp = None
    if normals is not None and len(normals):
        dots = np.abs(normals @ axis)
        nm_par, nm_perp = float(np.mean(dots >= COS_PARALLEL)), float(np.mean(dots <= SIN_PERPENDICULAR))
    # planes (schema fields of geometry_orientation): positions of parallel planes along the axis through the cloud centre
    pl_par, pl_wall, pl_other = [], [], []
    for p in planes:
        c = abs(float(p["normal"] @ axis))
        (pl_par if c >= COS_PARALLEL else pl_wall if c <= SIN_PERPENDICULAR else pl_other).append(p)
    positions = sorted((-(p["offset"] + float(p["normal"] @ middle)) / float(p["normal"] @ axis), p["weight"], p["id"])
                       for p in pl_par)
    bracket = None
    if len(positions) >= 2 and positions[-1][0] - positions[0][0] > max(1e-9, 0.05 * extent):
        bracket = {"separation_unscaled": float(positions[-1][0] - positions[0][0]),
                   "plane_ids": [int(positions[0][2]), int(positions[-1][2])]}
    cap = _cap_evidence(pts, axis, extent, camera_centers) if len(pts) else {"pair_found": False, "peaks": []}
    azimuths = _azimuth_count([d["axis"] for d in wall], axis)
    triple = False
    for d1, d2 in itertools.combinations(dirs, 2):
        lim = np.sin(np.radians(TRIPLE_ORTHOGONAL_DEG))
        if (abs(float(d1["axis"] @ axis)) <= lim and abs(float(d2["axis"] @ axis)) <= lim
                and abs(float(d1["axis"] @ d2["axis"])) <= lim):
            triple = True
            break
    # both signs: the cap pair seen from +axis and from -axis are mirror images (low/high swap); sign is never resolved
    signs = None
    if cap.get("pair_found"):
        signs = {"plus": {"floor_like_height": cap["low"]["height"], "ceiling_like_height": cap["high"]["height"]},
                 "minus": {"floor_like_height": -cap["high"]["height"], "ceiling_like_height": -cap["low"]["height"]},
                 "separation_agrees": True}
    return {"explained_support_fraction": float(explained),
            "parallel_direction_ids": [d["id"] for d in par], "wall_like_direction_ids": [d["id"] for d in wall],
            "unexplained_direction_ids": [d["id"] for d in other],
            "normal_mass_parallel": nm_par, "normal_mass_perpendicular": nm_perp,
            "normal_mass_explained": None if nm_par is None else float(nm_par + nm_perp),
            "parallel_plane_ids": [p["id"] for p in pl_par], "wall_like_plane_ids": [p["id"] for p in pl_wall],
            "unexplained_plane_ids": [p["id"] for p in pl_other], "bracketing_pair": bracket,
            "wall_azimuth_count": int(azimuths), "orthogonal_triple_of_observed_directions": bool(triple),
            "cap_pair": cap, "axis_sign_views": signs}


def _fast_score(ev):
    cap = ev["cap_pair"].get("cap_support", 0.0) if ev["cap_pair"].get("pair_found") else 0.0
    return float(ev["explained_support_fraction"] + 0.25 * min(1.0, cap / 0.5) + 0.25 * (ev["wall_azimuth_count"] >= 2))


def _polygon_bonus(status):
    return {"complete_polygon_hypothesis": 0.5, "ambiguous": 0.25}.get(status, 0.0)


def _viable(ev):
    reasons = []
    if ev["wall_azimuth_count"] < 2:
        reasons.append("fewer than two wall-like azimuths (no corner)")
    if ev["explained_support_fraction"] < 0.5:
        reasons.append("explains under half of the dominant direction mass")
    return not reasons, reasons


# ------------------------------------------------------------------------------------------------ public API
def enumerate_axis_candidates(plane_data, points, *, normals=None, camera_centers=None, polygon_runner=None,
                              max_candidates: int = MAX_CANDIDATES) -> dict:
    """Complete ranked candidate set. `polygon_runner(axis) -> dict` (optional) runs wall extraction + ray assembly for one
    candidate and returns at least {"boundary_status": str}; its result is stored and feeds the final ranking.
    No candidate is dropped for being unlikely; only the explicit cap `max_candidates` applies and is recorded."""
    pts = _finite_points(points)
    base = {"model": "geometry_only_structural_axis_candidates_complete", "units": "arbitrary", "gravity_known": False,
            "camera_orientation_used": False, "camera_path_used": False, "sign_resolved": False,
            "truth_used": False, "selection": "none: ranked evidence only; every co-supported candidate must be run downstream",
            "limitations": list(LIMITATIONS), "candidates": [], "directions": []}
    if len(pts) < 200:
        return {**base, "status": "insufficient_points"}
    if len(pts) > MAX_POINTS:
        pts = pts[np.random.default_rng(0).choice(len(pts), MAX_POINTS, replace=False)]
    extent = _extent(pts)
    if extent <= 1e-12:
        return {**base, "status": "insufficient_points"}
    middle = np.median(pts, axis=0)
    nrm = None if normals is None else np.asarray(normals, float)
    if nrm is not None and nrm.size:
        nrm = nrm[np.isfinite(nrm).all(axis=1)]
        nn = np.linalg.norm(nrm, axis=1)
        nrm = nrm[nn > 1e-9] / nn[nn > 1e-9][:, None]
    dirs, planes = dominant_directions(plane_data, nrm)
    base["directions"] = [{"id": d["id"], "axis_world": d["axis"].tolist(), "plane_share": d["plane_share"],
                           "normal_share": d["normal_share"], "plane_ids": d["plane_ids"], "sources": d["sources"]}
                          for d in dirs]
    if len(dirs) < 2:
        return {**base, "status": "insufficient_directions"}
    cands = generate_candidates(dirs)
    for c in cands:
        c["evidence"] = _evidence(c, dirs, planes, nrm, pts, middle, extent, camera_centers)
        c["fast_score"] = _fast_score(c["evidence"])
        c["viable"], c["viability_reasons"] = _viable(c["evidence"])
    cands.sort(key=lambda c: (-c["viable"], -c["fast_score"]))
    cap_dropped = max(0, len(cands) - max_candidates)
    cands = cands[:max_candidates]
    for c in cands:
        if polygon_runner is not None:
            try:
                c["polygon"] = polygon_runner(c["axis"])
            except Exception as exc:                                   # never dropped: recorded verbatim
                c["polygon"] = {"boundary_status": "runner_failed", "error": f"{type(exc).__name__}: {exc}"}
        else:
            c["polygon"] = None
        bonus = _polygon_bonus(c["polygon"].get("boundary_status")) if c["polygon"] else 0.0
        c["score"] = float(c["fast_score"] + bonus)
    cands.sort(key=lambda c: (-c["viable"], -c["score"]))
    viable_best = max((c["score"] for c in cands if c["viable"]), default=None)
    for rank, c in enumerate(cands):
        c["rank"] = rank
        c["co_supported_with_best"] = bool(c["viable"] and viable_best is not None and viable_best - c["score"] <= CO_SUPPORT_MARGIN)
        c["axis_world"] = c["axis"].tolist()
        c["alternative_sign_axis_world"] = (-c["axis"]).tolist()
        c["sign_resolved"] = False
        c["id"] = rank
    co = [c for c in cands if c["co_supported_with_best"]]
    ambiguity = ["axis sign unresolved for every candidate"]
    if len(co) > 1:
        ambiguity.append(f"{len(co)} candidates are co-supported; ranking cannot separate them from cloud evidence")
    if len(co) >= 3 and any(c["evidence"]["orthogonal_triple_of_observed_directions"] for c in co):
        ambiguity.append("co-supported candidates include a near-orthogonal triple: box symmetry, floor/ceiling vs wall pairs")
    for c in cands:
        c["axis"] = c["axis"].tolist()
    status = "no_viable_candidate" if not co else ("single_axis_sign_ambiguous" if len(co) == 1 else "ambiguous_axes")
    return {**base, "status": status, "candidates": cands, "candidate_count": len(cands),
            "candidate_cap": {"max_candidates": max_candidates, "dropped_by_cap": cap_dropped},
            "co_supported_count": len(co), "co_support_margin": CO_SUPPORT_MARGIN,
            "unresolved_ambiguity": ambiguity,
            "ranking_note": "score = explained direction mass + 0.25*cap-pair coverage + 0.25*corner (+0.5 complete / +0.25 "
                            "ambiguous polygon when run); heuristic, never used to drop a candidate"}


def to_axis_hypotheses(result: dict, *, only_viable: bool = False) -> dict:
    """Emit geometry_orientation.axis_hypotheses' schema so compare_wall_boundaries / run_pipeline code can consume it.

    Same top-level keys (model, units, gravity_known, camera_orientation_used, camera_path_used, sign_resolved,
    limitations, status, supported_plane_count, hypotheses, co_supported_count, best_axis_world) and same per-hypothesis
    keys (axis_world, alternative_sign_axis_world, sign_resolved, explained_support_fraction, parallel_plane_ids,
    wall_like_plane_ids, unexplained_plane_ids, bracketing_pair, walls_span_bracket_fraction, wall_azimuth_count, score,
    rank, co_supported_with_best). Differences: (1) ALL candidates are listed (not capped at 6; non-viable ones are
    included with co_supported_with_best False unless only_viable); (2) co_supported_with_best means viable and within
    CO_SUPPORT_MARGIN (0.25) of the best viable score, a wider margin than 0.03 because the evidence is coarser; (3)
    walls_span_bracket_fraction is always None (patch spans are not used); (4) extra keys: id, kinds, evidence, polygon,
    viable, viability_reasons; (5) `model` differs and `truth_used` is False; (6) status additionally may be
    no_viable_candidate / insufficient_points / insufficient_directions."""
    hyps = []
    for c in result.get("candidates", []):
        if only_viable and not c["viable"]:
            continue
        ev = c["evidence"]
        hyps.append({"axis_world": c["axis_world"], "alternative_sign_axis_world": c["alternative_sign_axis_world"],
                     "sign_resolved": False, "explained_support_fraction": ev["explained_support_fraction"],
                     "parallel_plane_ids": ev["parallel_plane_ids"], "wall_like_plane_ids": ev["wall_like_plane_ids"],
                     "unexplained_plane_ids": ev["unexplained_plane_ids"], "bracketing_pair": ev["bracketing_pair"],
                     "walls_span_bracket_fraction": None, "wall_azimuth_count": ev["wall_azimuth_count"],
                     "score": c["score"], "rank": c["rank"], "co_supported_with_best": c["co_supported_with_best"],
                     "id": c["id"], "kinds": c["kinds"], "viable": c["viable"], "viability_reasons": c["viability_reasons"],
                     "evidence": ev, "polygon": c.get("polygon")})
    co = [h for h in hyps if h["co_supported_with_best"]]
    status = result.get("status")
    out = {"model": result.get("model"), "units": "arbitrary", "gravity_known": False, "camera_orientation_used": False,
           "camera_path_used": False, "sign_resolved": False, "truth_used": False, "limitations": result.get("limitations", []),
           "status": status, "supported_plane_count": len(result.get("directions", [])), "hypotheses": hyps,
           "co_supported_count": len(co), "best_axis_world": co[0]["axis_world"] if len(co) == 1 else None}
    return out
