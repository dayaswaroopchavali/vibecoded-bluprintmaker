"""Truth-free association of wall-line fragments across views (stage 1 of the simulator -> blueprint benchmark).

Input is `wall_extraction.extract_wall_lines` output (WALL_BOUNDARY_INTERFACE.md): lines + per-frame observations (camera
centres, subsampled hits, hit heights). Every line gets an evidence-based `association`; nothing is dropped silently.

Classes (never forced; evidence values and the alternative are always reported):
  room_boundary_candidate   no stronger explanation than "this is a wall of the room"
  interior_object           two-sided visibility (cameras on both faces), member of a closed footprint seen from outside
                            (the intersection of the far-side half-planes of >= 3 lines is a bounded polygon with no camera
                            in it), free space observed directly behind it, short stub in front of a longer parallel line,
                            height range that does not span the floor-to-ceiling band of the same cloud, short extent
  duplicate_of(line_id)     near-parallel within a drift allowance of a better supported line seen from the same side, and the
                            two are (mostly) seen by DISJOINT frame sets (one frame sees one copy of a surface: per-frame /
                            window offsets), or coincide within the noise
  exterior_or_one_sided     camera->hit rays of its supporting frames cross an observed span / opening of another candidate
                            line first (it lies beyond the room boundary that those cameras saw)
  undetermined              evidence conflicts or is too weak: kept, with the competing alternatives

Class scores are noisy-OR combinations of independent evidence terms (weights below are policy, not data fits). A class is
assigned only when its score >= `accept` and it beats every other non-boundary class by `margin`; the boundary class is the
complement of the strongest contradiction. Everything else is `undetermined`.

Outputs: `walls_all` (every line + `association`), `walls_boundary_candidates` (class room_boundary_candidate only),
`walls_with_undetermined` (candidates + undetermined), `walls_layer_representatives` (candidates + the best supported line of
each unresolved layered group: an explicit HYPOTHESIS variant, never a conclusion) - all still A->B schema dicts - and a
`report`. "Layered": near-parallel same-side lines within the drift allowance seen by the same frames; they cannot be
separated into duplicate / object in front / exterior, so they stay undetermined.
All thresholds are derived from `noise_scale` / `extent`; no instance constants, no forced wall count, rectangle or Manhattan
assumption; nothing outside the walls dict is read.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np

import boundary_assembly as ba

MODEL = "fragment_association"
CLASSES = ("room_boundary_candidate", "interior_object", "duplicate_of", "exterior_or_one_sided", "undetermined")

DEFAULTS = {
    "tol_noise_multiple": 3.0,          # tol = max(this * noise_scale, this * median line offset_sd)
    "min_frame_hits": 3,                # hits near a line for a frame to count as seeing it
    "accept": 0.6,                      # minimum class score to assign it
    "margin": 0.2,                      # minimum lead over the best competing non-boundary class
    "full_height_fraction": 0.75,       # height_range / cloud band at which height deficit evidence reaches zero
    "short_fraction": 0.2,              # observed length / extent below which "short" evidence starts
    "drift_fraction": 0.12,             # max separation of duplicate lines over their overlap / extent
    "dup_max_angle_deg": 20.0,
    "dup_min_overlap": 0.3,             # lateral overlap / shorter hull
    "behind_offset_multiple": 2.5,      # free-space probe distance behind a line, in tol
    "max_object_area_fraction": 0.2,    # closed footprint area / extent^2
    "ray_frames": 12,                   # supporting frames sampled for ray-order tests
    "through_fraction": 0.5,            # fraction of rays crossing another line that saturates exterior evidence
    "observed_cross_weight": 0.25,      # exterior evidence weight of a ray crossing an observed span (opening gap = 1)
    "layered_support_ratio": 0.6,       # a layered partner with < this share of the frames is the one in doubt, not the canonical line
    "layered_score": 0.5,              # blocking score of a non-separable layered pair (< accept: stays undetermined)
    "max_lines": 40,
    "max_footprint_faces": 5,
    "footprint_budget": 200000,
    "weights": {"two_sided": 0.9, "cluster": 0.9, "free_behind": 0.55, "height_deficit": 0.5, "short": 0.25,
                "front_stub": 0.55, "exterior": 0.8},
}


# ---------------------------------------------------------------------------------------------------------------- utils
def _clip01(x) -> float:
    return float(min(1.0, max(0.0, x)))


def _noisy_or(terms) -> float:
    prod = 1.0
    for w, e in terms:
        prod *= 1.0 - w * _clip01(e)
    return 1.0 - prod


def _merge_spans(spans, eps=0.0):
    out = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in spans):
        if out and a <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _prepare(walls, p):
    """Normalised line records (own copy, independent of the assembler) + the shared observation arrays."""
    obs = walls.get("observations") or {}
    cams = np.asarray(obs.get("camera_xy") or [], dtype=float).reshape(-1, 2)
    hits = [np.asarray(h, dtype=float).reshape(-1, 2) for h in obs.get("hits_xy") or []]
    heights = [np.asarray(h, dtype=float).reshape(-1) for h in obs.get("hit_heights") or []]
    lines, bad = [], []
    for k, raw in enumerate(walls.get("lines") or []):
        try:
            m = np.asarray(raw["normal_xy"], dtype=float)
            c = float(raw["offset"])
            d = np.asarray(raw["direction_xy"], dtype=float)
            spans = _merge_spans([(float(a), float(b)) for a, b in raw.get("observed_spans") or []])
            nm = float(np.linalg.norm(m))
            if not (np.isfinite(m).all() and np.isfinite(c) and np.isfinite(d).all()) or nm < 1e-9:
                raise ValueError("non finite")
        except (KeyError, TypeError, ValueError):
            bad.append(raw.get("id", k) if isinstance(raw, dict) else k)
            continue
        m, c = m / nm, c / nm
        d = d - m * float(d @ m)
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([-m[1], m[0]])
        hr = raw.get("height_range")
        sup = raw.get("support") or {}
        unc = raw.get("uncertainty") or {}
        lines.append({"k": k, "id": raw.get("id", k), "m": m, "c": c, "d": d, "spans": spans,
                      "hr": [float(hr[0]), float(hr[1])] if hr and len(hr) == 2 else None,
                      "frames": int(sup.get("distinct_frames", 0) or 0), "weight": float(sup.get("weight", 0.0) or 0.0),
                      "offset_sd": float(unc.get("offset_sd", 0.0) or 0.0)})
    return lines, bad, cams, hits, heights


def _clip_halfplane(poly, m, c, sign):
    """Keep the part of convex polygon `poly` with sign*(m.x - c) <= 0 (Sutherland-Hodgman)."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        fa, fb = sign * (float(m @ a) - c), sign * (float(m @ b) - c)
        if fa <= 0:
            out.append(a)
        if (fa < 0 < fb) or (fb < 0 < fa):
            out.append(a + (b - a) * (fa / (fa - fb)))
    return out


def _poly_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    q = np.asarray(poly)
    return 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))


# --------------------------------------------------------------------------------------------------- per-line evidence
def _line_frame_evidence(L, cams, hits, tol, p):
    """Per-frame hit counts near the line (inside its spans), camera side of every frame that sees it."""
    nframes = len(hits)
    counts = np.zeros(nframes, dtype=int)
    for f, h in enumerate(hits):
        if len(h) == 0:
            continue
        r = h @ L["m"] - L["c"]
        t = h @ L["d"]
        inside = np.zeros(len(h), dtype=bool)
        for a, b in L["spans"]:
            inside |= (t >= a - tol) & (t <= b + tol)
        counts[f] = int(((np.abs(r) < tol) & inside).sum())
    seen = counts >= p["min_frame_hits"]
    side = (cams @ L["m"] - L["c"]) if len(cams) else np.zeros(0)
    pos = int((seen & (side > tol)).sum()) if len(cams) else 0
    neg = int((seen & (side < -tol)).sum()) if len(cams) else 0
    L["counts"], L["seen"] = counts, seen
    L["n_seen"] = int(seen.sum())
    L["side_pos"], L["side_neg"] = pos, neg
    total = pos + neg
    if total == 0:
        L["facing"] = 0                                  # +1: cameras on the + side of m; -1: on the - side; 0: unresolved
    elif min(pos, neg) >= max(2, 0.1 * total):
        L["facing"] = 0
    else:
        L["facing"] = 1 if pos > neg else -1
    hull = (min(a for a, _ in L["spans"]), max(b for _, b in L["spans"])) if L["spans"] else (0.0, 0.0)
    L["hull"] = hull
    L["length"] = float(sum(b - a for a, b in L["spans"]))


def _free_behind(L, scorer, tol, p):
    """Free space observed directly behind the line (away from the cameras that see it), over each observed span."""
    if L["facing"] == 0 or scorer is None:
        return {"free_fraction": None, "frames": 0, "available": False}
    behind = -L["facing"] * L["m"]
    delta = p["behind_offset_multiple"] * tol
    num = den = 0.0
    frames = 0
    for a, b in L["spans"]:
        pad = 0.1 * (b - a)
        a, b = a + pad, b - pad
        if b - a < 2 * tol:
            continue
        shift = L["c"] * L["m"] + delta * behind
        res = scorer(shift + a * L["d"], shift + b * L["d"])
        num += (b - a) * res["free_space_fraction"]
        den += b - a
        frames = max(frames, int(res.get("contradicting_frames", 0)))
    if den <= 0:
        return {"free_fraction": None, "frames": 0, "available": False}
    return {"free_fraction": float(num / den), "frames": frames, "available": True}


def _two_sided(L, p):
    pos, neg = L["side_pos"], L["side_neg"]
    total = pos + neg
    both = total > 0 and min(pos, neg) >= max(2, 0.1 * total)
    return {"frames_pos_side": pos, "frames_neg_side": neg, "evidence": 1.0 if both else 0.0}


def _height_evidence(L, band, p):
    if L["hr"] is None or band is None:
        return {"coverage": None, "deficit": 0.0, "available": False}
    cov = (L["hr"][1] - L["hr"][0]) / band[2]
    lo_gap = (L["hr"][0] - band[0]) / band[2]
    hi_gap = (band[1] - L["hr"][1]) / band[2]
    deficit = _clip01((p["full_height_fraction"] - cov) / (p["full_height_fraction"] - 0.4))
    return {"coverage": float(cov), "gap_to_floor": float(lo_gap), "gap_to_ceiling": float(hi_gap), "deficit": deficit,
            "available": True}


# --------------------------------------------------------------------------------------------------- pair evidence
def _dup_pair(Li, Lj, extent, tol, p):
    """Evidence that Li (lower ranked) is a second copy of Lj. Returns dict or None when not even geometrically comparable."""
    if Li["facing"] == 0 or Lj["facing"] == 0:
        return None
    # facing normals (pointing to the cameras) must agree
    fi, fj = Li["facing"] * Li["m"], Lj["facing"] * Lj["m"]
    ang = math.degrees(math.acos(float(np.clip(fi @ fj, -1, 1))))
    if ang > p["dup_max_angle_deg"]:
        return None
    d = Lj["d"]
    # lateral overlap of the hulls, in Lj's direction
    ends_i = np.array([Li["c"] * Li["m"] + t * Li["d"] for t in Li["hull"]])
    ti = sorted(float(d @ q) for q in ends_i)
    tj = list(Lj["hull"])
    ov = _overlap(ti, tj)
    shorter = min(ti[1] - ti[0], tj[1] - tj[0])
    if shorter <= 0 or ov < p["dup_min_overlap"] * shorter:
        return None
    # separation of Li's overlapping stretch from Lj's line
    lo, hi = max(ti[0], tj[0]), min(ti[1], tj[1])
    sep = []
    for t in (lo, hi):
        # point of Li whose Lj-direction coordinate is t
        denom = float(d @ Li["d"])
        if abs(denom) < 1e-9:
            return None
        s = (t - float(d @ (Li["c"] * Li["m"]))) / denom
        q = Li["c"] * Li["m"] + s * Li["d"]
        sep.append(float(Lj["m"] @ q) - Lj["c"])
    sep_max = float(max(abs(x) for x in sep))
    sep_mean = float(np.mean(sep))
    D = p["drift_fraction"] * extent
    if sep_max > D:
        return None
    seen_i, seen_j = Li["seen"], Lj["seen"]
    n_min = min(int(seen_i.sum()), int(seen_j.sum()))
    co = int((seen_i & seen_j).sum())
    exclusivity = 1.0 - (co / n_min if n_min else 1.0)
    if sep_max <= tol:
        score, why = 1.0, "coincident within the noise tolerance"
    else:
        closeness = _clip01(1.0 - (sep_max - tol) / max(D - tol, 1e-12))
        score = float(closeness * exclusivity)
        why = "disjoint frame sets at offset" if exclusivity > 0.5 else "same frames see both: separate surfaces"
    return {"target": Lj["id"], "score": score, "angle_deg": float(ang), "separation_max": sep_max,
            "separation_mean_signed": sep_mean, "overlap": float(ov), "co_visible_frames": co,
            "frames_i": int(seen_i.sum()), "frames_j": int(seen_j.sum()), "exclusivity": float(exclusivity),
            "support_ratio": float(int(seen_i.sum()) / max(int(seen_j.sum()), 1)), "reason": why}


def _front_stub(Li, Lj, extent, tol, p):
    """Li is a shorter near-parallel line lying in front of (camera side of) Lj and inside Lj's lateral hull,
    seen together with it by the same frames (an object in front of a wall, not the wall itself)."""
    if Li["facing"] == 0 or Lj["facing"] == 0 or Li["facing"] * Lj["facing"] * float(Li["m"] @ Lj["m"]) < math.cos(
            math.radians(p["dup_max_angle_deg"])):
        return None
    d = Lj["d"]
    ends = np.array([Li["c"] * Li["m"] + t * Li["d"] for t in Li["hull"]])
    ti = sorted(float(d @ q) for q in ends)
    tj = Lj["hull"]
    ilen, jlen = ti[1] - ti[0], tj[1] - tj[0]
    if ilen <= 0 or jlen <= 0 or ilen > 0.6 * jlen:
        return None
    inside = _overlap(ti, tj) / ilen
    if inside < 0.8:
        return None
    mid = Li["c"] * Li["m"] + 0.5 * sum(Li["hull"]) * Li["d"]
    offset = (float(Lj["m"] @ mid) - Lj["c"]) * Lj["facing"]        # >0: Li is on the camera side of Lj
    if offset < 2 * tol or offset > p["drift_fraction"] * extent * 2:
        return None
    n_min = min(int(Li["seen"].sum()), int(Lj["seen"].sum()))
    co = int((Li["seen"] & Lj["seen"]).sum())
    covis = co / n_min if n_min else 0.0
    if covis < 0.3:
        return None
    return {"in_front_of": Lj["id"], "inside_fraction": float(inside), "offset_in_front": float(offset),
            "co_visibility": float(covis), "length_ratio": float(ilen / jlen), "evidence": float(min(1.0, covis))}


def _through(Li, others, cams, hits, tol, p, extent):
    """For rays from Li's supporting cameras to points on Li: fraction crossing another line's observed span or the opening
    between its spans before arriving. Returns per other line {obs, gap, rays}."""
    frames = np.flatnonzero(Li["seen"])
    if len(frames) == 0 or len(cams) == 0:
        return {}
    if len(frames) > p["ray_frames"]:
        frames = frames[np.linspace(0, len(frames) - 1, p["ray_frames"]).astype(int)]
    out = {}
    per_frame = {}
    for f in frames:
        h = hits[f]
        r = h @ Li["m"] - Li["c"]
        t = h @ Li["d"]
        inside = np.zeros(len(h), dtype=bool)
        for a, b in Li["spans"]:
            inside |= (t >= a - tol) & (t <= b + tol)
        sel = h[(np.abs(r) < tol) & inside]
        if len(sel) > 24:
            sel = sel[np.linspace(0, len(sel) - 1, 24).astype(int)]
        per_frame[int(f)] = sel                      # real camera->hit rays only (the frame's own hits on this line)
    for Lj in others:
        if Lj is Li:
            continue
        obs = gap = total = 0
        for f in frames:
            cam = cams[f]
            pts = per_frame[int(f)]
            if len(pts) == 0:
                continue
            sc = float(Lj["m"] @ cam) - Lj["c"]
            sq = pts @ Lj["m"] - Lj["c"]
            cross = (sc * sq < 0) & (np.abs(sq) > tol) & (abs(sc) > tol)
            total += len(pts)
            if not cross.any():
                continue
            u = sc / (sc - sq[cross])
            x = cam + u[:, None] * (pts[cross] - cam)
            t = x @ Lj["d"]
            in_obs = np.zeros(len(t), dtype=bool)
            fam = Lj.get("fam_spans") or Lj["spans"]
            for a, b in fam:
                in_obs |= (t >= a - tol) & (t <= b + tol)
            in_gap = np.zeros(len(t), dtype=bool)
            for (a0, a1), (b0, b1) in zip(fam[:-1], fam[1:]):
                in_gap |= (t > a1 + tol) & (t < b0 - tol)
            obs += int(in_obs.sum())
            gap += int((~in_obs & in_gap).sum())
        if total:
            out[Lj["id"]] = {"obs": obs / total, "gap": gap / total, "rays": total}
    return out


def _footprints(lines, cams, extent, tol, p):
    """Closed convex footprints seen from outside: a subset (3..max_footprint_faces) of lines whose far-side half-planes
    intersect in a BOUNDED polygon (every line contributes an edge, observed span on it) containing no camera. Subsets are
    grown depth-first with pruning by emptiness. Returns ({line id: cluster id}, clusters, accepted subsets)."""
    cand = [L for L in lines if L["facing"] != 0 and L["spans"]]
    B = 4.0 * extent
    box = [np.array(v) for v in ((-B, -B), (B, -B), (B, B), (-B, B))]
    found, budget = [], [int(p["footprint_budget"])]

    def accept(sub, poly):
        q = np.asarray(poly)
        area = abs(_poly_area(poly))
        if np.abs(q).max() > B * (1 - 1e-6) or area < 4 * tol * tol or area > p["max_object_area_fraction"] * extent ** 2:
            return False
        if len(cams):
            inside = np.ones(len(cams), dtype=bool)
            ctr = q.mean(axis=0)
            for k in range(len(q)):
                a, b = q[k], q[(k + 1) % len(q)]
                e = b - a
                s = e[0] * (cams[:, 1] - a[1]) - e[1] * (cams[:, 0] - a[0])
                s_ref = e[0] * (ctr[1] - a[1]) - e[1] * (ctr[0] - a[0])
                inside &= s * s_ref >= 0
            if inside.any():
                return False
        for i in sub:
            L = cand[i]
            on = [v for v in q if abs(float(L["m"] @ v) - L["c"]) < 1e-6 * extent + 1e-9]
            if len(on) < 2:
                return False
            ts = sorted(float(L["d"] @ v) for v in on)
            edge = (ts[0], ts[-1])
            hull = (L["hull"][0] - tol, L["hull"][1] + tol)
            if _overlap(edge, hull) < 0.3 * min(edge[1] - edge[0], hull[1] - hull[0]):
                return False
        found.append({"line_ids": [cand[i]["id"] for i in sub], "area": float(area),
                      "centroid": [float(x) for x in q.mean(axis=0)]})
        return True

    def grow(sub, poly, start):
        for i in range(start, len(cand)):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            L = cand[i]
            nxt = _clip_halfplane(poly, L["m"], L["c"], L["facing"])      # far side: facing * (m.x - c) <= 0
            if len(nxt) < 3 or abs(_poly_area(nxt)) < 4 * tol * tol:
                continue
            sub2 = sub + [i]
            if len(sub2) >= 3 and accept(sub2, nxt):
                continue                                                   # minimal closed footprint: do not grow further
            if len(sub2) < p["max_footprint_faces"]:
                grow(sub2, nxt, i + 1)

    grow([], box, 0)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for rec in found:
        ids = rec["line_ids"]
        for x in ids[1:]:
            parent[find(ids[0])] = find(x)
    members, clusters = {}, {}
    for rec in found:
        for x in rec["line_ids"]:
            members[x] = find(x)
    for x, root in members.items():
        clusters.setdefault(root, []).append(x)
    return members, [{"cluster_id": k, "line_ids": sorted(v, key=str)} for k, v in sorted(clusters.items(), key=lambda kv: str(kv[0]))], found


def associate_fragments(walls: dict, **kw) -> dict:
    p = {**DEFAULTS, **kw}
    p["weights"] = {**DEFAULTS["weights"], **(kw.get("weights") or {})}
    base = {"model": MODEL, "status": "insufficient", "reasons": [], "params": copy.deepcopy(p)}
    if not isinstance(walls, dict) or walls.get("status") != "ok" or not walls.get("lines"):
        reason = "no ok wall lines" if isinstance(walls, dict) else "input is not a dict"
        return {**base, "reasons": [reason], "walls_all": walls, "walls_boundary_candidates": walls,
                "walls_with_undetermined": walls, "walls_layer_representatives": walls, "walls_layer_merged": walls, "report": {"counts": {}, "lines": [], "pairs": []}}
    lines, bad, cams, hits, heights = _prepare(walls, p)
    reasons = []
    if bad:
        reasons.append(f"malformed lines left unclassified: {bad}")
    if len(lines) > p["max_lines"]:
        reasons.append(f"{len(lines)} lines > max_lines {p['max_lines']}: lowest-support lines unclassified")
        keep = sorted(lines, key=lambda L: (-L["frames"], -L["weight"]))[:p["max_lines"]]
        lines = [L for L in lines if L in keep]
    extent = float((walls.get("frame") or {}).get("extent") or 0.0)
    if not np.isfinite(extent) or extent <= 0:
        pts = np.concatenate([h for h in hits if len(h)] or [np.zeros((1, 2))])
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0
        reasons.append("frame.extent missing: derived from hits")
    sds = [L["offset_sd"] for L in lines if L["offset_sd"] > 0]
    noise = float(walls.get("noise_scale") or 0.0)
    tol = p["tol_noise_multiple"] * max(noise if np.isfinite(noise) else 0.0, float(np.median(sds)) if sds else 0.0)
    if tol <= 0:
        tol = 0.005 * extent
        reasons.append("no noise_scale/offset_sd: tolerance = 0.5% of extent")
    for L in lines:                          # collinear lines (one wall split over several ids): shared span family
        spans = list(L["spans"])
        for O in lines:
            if O is L:
                continue
            if abs(float(L["m"] @ O["m"])) >= math.cos(math.radians(3.0)) and \
                    abs(float(L["m"] @ (O["c"] * O["m"])) - L["c"]) <= tol:
                for a, b in O["spans"]:
                    ends = sorted(float(L["d"] @ (O["c"] * O["m"] + x * O["d"])) for x in (a, b))
                    spans.append((ends[0], ends[1]))
        L["fam_spans"] = _merge_spans(spans, eps=0.0)
    has_obs = len(cams) > 0 and len(hits) == len(cams)
    if not has_obs:
        reasons.append("observations missing or inconsistent: camera/ray based evidence unavailable")
    # height band from the same cloud
    allh = np.concatenate([h for h in heights if len(h)]) if heights and any(len(h) for h in heights) else None
    band = None
    if allh is not None and len(allh) >= 20:
        lo, hi = np.percentile(allh, [2, 98])
        if hi - lo > 1e-9:
            band = (float(lo), float(hi), float(hi - lo))
    for L in lines:
        _line_frame_evidence(L, cams, hits, tol, p) if has_obs else L.update(
            counts=np.zeros(0, int), seen=np.zeros(0, bool), n_seen=0, side_pos=0, side_neg=0, facing=0,
            hull=(min((a for a, _ in L["spans"]), default=0.0), max((b for _, b in L["spans"]), default=0.0)),
            length=float(sum(b - a for a, b in L["spans"])))
    scorer = None
    if has_obs:
        scorer = ba.FreeSpaceScorer({"camera_xy": cams.tolist(), "hits_xy": [h.tolist() for h in hits]},
                                    min_frames=3, tolerance=tol)
    # rank: better supported lines are canonical for duplicates
    order = sorted(lines, key=lambda L: (-L["n_seen"], -L["weight"], str(L["id"])))
    rank = {L["id"]: i for i, L in enumerate(order)}
    members, clusters, triples = _footprints(lines, cams, extent, tol, p) if has_obs else ({}, [], [])

    W = p["weights"]
    ev = {}
    pairs = []
    for L in lines:
        e = {"two_sided": _two_sided(L, p) if has_obs else {"evidence": 0.0, "available": False},
             "height": _height_evidence(L, band, p),
             "free_behind": _free_behind(L, scorer, tol, p),
             "short": {"length_fraction": L["length"] / extent,
                       "evidence": _clip01((p["short_fraction"] - L["length"] / extent) / (0.75 * p["short_fraction"]))},
             "cluster": {"cluster_id": members.get(L["id"]), "evidence": 1.0 if L["id"] in members else 0.0},
             "camera_side": {"frames_pos": L["side_pos"], "frames_neg": L["side_neg"], "facing": L["facing"],
                             "distinct_frames_seen": L["n_seen"]}}
        dups = []
        stubs = []
        for Lj in lines:
            if Lj is L or not has_obs:
                continue
            if rank[Lj["id"]] < rank[L["id"]]:
                r = _dup_pair(L, Lj, extent, tol, p)
                if r:
                    dups.append(r)
                    pairs.append({"kind": "duplicate_candidate", "line": L["id"], **r})
            s = _front_stub(L, Lj, extent, tol, p)
            if s:
                stubs.append(s)
        e["duplicate"] = max(dups, key=lambda r: r["score"]) if dups else None
        e["duplicate_candidates"] = sorted(dups, key=lambda r: -r["score"])
        e["front_stub"] = max(stubs, key=lambda r: r["evidence"]) if stubs else None
        ev[L["id"]] = e

    def dup_score(i):
        d = ev[i]["duplicate"]
        return d["score"] if d else 0.0

    layered = {L["id"]: set() for L in lines}
    for L in lines:
        for r in ev[L["id"]]["duplicate_candidates"]:
            if r["score"] < p["accept"]:
                layered[L["id"]].add(r["target"])
                layered[r["target"]].add(L["id"])

    # first pass: object / duplicate scores without exterior
    scores = {}
    for L in lines:
        e = ev[L["id"]]
        s_dup = dup_score(L["id"])
        explained = s_dup >= p["accept"] or any(           # as copy or as the canonical line of a duplicate pair
            ev[o["id"]]["duplicate"] and ev[o["id"]]["duplicate"]["target"] == L["id"]
            and ev[o["id"]]["duplicate"]["score"] >= p["accept"] for o in lines) or any(
            r["exclusivity"] >= 0.3 and L["id"] in (o["id"], r["target"])    # partly disjoint frames: a drifted layer explains it
            for o in lines for r in ev[o["id"]]["duplicate_candidates"])
        fb = e["free_behind"]
        free_ev = 0.0 if (fb["free_fraction"] is None or fb["frames"] < 3) else _clip01(fb["free_fraction"] / 0.3)
        terms = [(W["two_sided"], e["two_sided"]["evidence"]), (W["cluster"], e["cluster"]["evidence"]),
                 (W["free_behind"], 0.0 if explained else free_ev), (W["height_deficit"], e["height"]["deficit"]),
                 (W["short"], e["short"]["evidence"]),
                 (W["front_stub"], e["front_stub"]["evidence"] if e["front_stub"] else 0.0)]
        e["free_behind"]["effective_evidence"] = 0.0 if explained else free_ev
        e["free_behind"]["explained_by_duplicate"] = bool(explained and free_ev > 0)
        scores[L["id"]] = {"interior_object": _noisy_or(terms), "duplicate_of": s_dup, "exterior_or_one_sided": 0.0,
                           "layered_ambiguity": 0.0}

    # layered pairs: near-parallel, same side, within the drift allowance, laterally overlapping, but NOT separable as
    # duplicates (the same frames see both). Copies of one surface, an object in front of a wall and an exterior face are
    # then all consistent: neither line is promoted, both stay undetermined unless other evidence decides.
    blocked = set()
    for L in lines:
        for r in ev[L["id"]]["duplicate_candidates"]:
            if r["score"] < p["accept"]:
                blocked.add(L["id"])                                  # the lower ranked line of the pair is always in doubt
                if r["support_ratio"] > p["layered_support_ratio"]:
                    blocked.add(r["target"])                          # comparable support: neither can be preferred
    for L in lines:
        if L["id"] in blocked and scores[L["id"]]["duplicate_of"] < p["accept"]:
            scores[L["id"]]["layered_ambiguity"] = p["layered_score"]
        ev[L["id"]]["layered_with"] = sorted(layered[L["id"]], key=str)

    def tentative_nonboundary(i):
        return scores[i]["interior_object"] >= p["accept"] or scores[i]["duplicate_of"] >= p["accept"]

    dup_pairs = {(L["id"], ev[L["id"]]["duplicate"]["target"]) for L in lines if ev[L["id"]]["duplicate"]
                 and ev[L["id"]]["duplicate"]["score"] >= p["accept"]}
    # second pass: exterior = rays to the line cross another candidate-like line first
    for L in lines:
        if not has_obs:
            break
        cands = [Lj for Lj in lines if Lj is not L and not tentative_nonboundary(Lj["id"])
                 and (L["id"], Lj["id"]) not in dup_pairs and (Lj["id"], L["id"]) not in dup_pairs
                 and Lj["id"] not in layered[L["id"]]]
        th = _through(L, cands, cams, hits, tol, p, extent)
        best = None
        for jid, r in th.items():
            tot = r["gap"] + p["observed_cross_weight"] * r["obs"]     # rays through an OBSERVED span are contradictory, not proof
            if best is None or tot > best[1]:
                best = (jid, tot, r)
        if best:
            jid, tot, r = best
            e = ev[L["id"]]
            e["exterior"] = {"beyond_line": jid, "through_fraction": float(tot), "through_observed_span": float(r["obs"]),
                             "through_opening_gap": float(r["gap"]), "rays": r["rays"],
                             "conflict_note": ("rays cross an OBSERVED span of the other line: it may be non-opaque, a "
                                               "drifted copy or this line may be the real boundary") if r["obs"] > r["gap"] else None}
            scores[L["id"]]["exterior_or_one_sided"] = _noisy_or([(W["exterior"], tot / p["through_fraction"])])
        else:
            ev[L["id"]]["exterior"] = None

    out_lines, summary = [], []
    for L in lines:
        i = L["id"]
        s = scores[i]
        strongest = max(s.values())
        s_bnd = 1.0 - strongest
        allc = {"room_boundary_candidate": s_bnd, **s}
        ranked = sorted(allc.items(), key=lambda kv: -kv[1])
        nb = sorted(((k, v) for k, v in s.items() if k != "layered_ambiguity"), key=lambda kv: -kv[1])
        label, why = "undetermined", ""
        if nb[0][1] >= p["accept"] and (nb[0][1] - nb[1][1]) >= p["margin"]:
            label, why = nb[0][0], "score >= accept and leads the other non-boundary classes by the margin"
        elif s_bnd >= p["accept"]:
            label, why = "room_boundary_candidate", "no contradicting evidence above the acceptance level"
        else:
            blocker = max(s.items(), key=lambda kv: kv[1])
            why = "evidence conflicts or is insufficient (strongest competing evidence: %s=%.2f)" % blocker
            if blocker[0] == "layered_ambiguity":
                why += ("; near-parallel same-side lines %s within the drift allowance are seen by the same frames: copies "
                        "of one surface, an object in front of a wall or an exterior face are all consistent" % ev[i]["layered_with"])
        alt = [{"class": k, "score": round(float(v), 4)} for k, v in ranked if k != label][:3]
        a = {"class": label, "label": label, "scores": {k: round(float(v), 4) for k, v in allc.items()},
             "duplicate_of": ev[i]["duplicate"]["target"] if (label == "duplicate_of" and ev[i]["duplicate"]) else None,
             "beyond_line": (ev[i].get("exterior") or {}).get("beyond_line") if label == "exterior_or_one_sided" else None,
             "alternatives": alt, "reason": why, "evidence": _jsonable(ev[i])}
        if label == "duplicate_of":
            a["label"] = "duplicate_of(%s)" % a["duplicate_of"]
        a["ambiguous_alternative_score_gap"] = round(float(ranked[0][1] - ranked[1][1]), 4)
        raw = copy.deepcopy(walls["lines"][L["k"]])
        raw["association"] = a
        out_lines.append(raw)
        summary.append({"line_id": i, "class": a["label"], "scores": a["scores"], "alternatives": alt, "reason": why})

    def variant(sel):
        w = copy.deepcopy({k: v for k, v in walls.items() if k != "lines"})
        w["lines"] = [copy.deepcopy(l) for l in out_lines if sel(l["association"]["class"])]
        w["association_filter"] = MODEL
        return w

    def variant_ids(sel):
        w = copy.deepcopy({k: v for k, v in walls.items() if k != "lines"})
        w["lines"] = [copy.deepcopy(l) for l in out_lines if sel(l)]
        w["association_filter"] = MODEL + ":layer_representatives(HYPOTHESIS: layered lines are one surface)"
        return w

    # layered groups (connected components of layered pairs): an explicit HYPOTHESIS variant keeps the best supported member of
    # each group that has no accepted boundary line ("the layers are one surface"); never used as a conclusion.
    label_of = {l["id"]: l["association"]["class"] for l in out_lines}
    seen_ids, groups = set(), []
    for L in lines:
        if L["id"] in seen_ids or not layered[L["id"]]:
            continue
        comp, stack = set(), [L["id"]]
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(layered.get(x, ()))
        seen_ids |= comp
        groups.append(sorted(comp, key=str))
    by_id = {L["id"]: L for L in lines}
    reps = []
    for g in groups:
        pool = [i for i in g if label_of[i] == "undetermined"]
        if pool and not any(label_of[i] == "room_boundary_candidate" for i in g):
            reps.append(max(pool, key=lambda i: (by_id[i]["n_seen"], by_id[i]["weight"])))
    # merged variant (HYPOTHESIS): an undetermined line that is the lower ranked member of a layered pair is a drifted FRAGMENT
    # of its better supported partner: its observed extent is transplanted onto that line (partner geometry unchanged, offset
    # uncertainty widened by the layer separation, support frames united). Each line merges once; chains resolve to a root.
    root = {}
    for L in sorted(lines, key=lambda L: -rank[L["id"]]):
        cand = [r for r in ev[L["id"]]["duplicate_candidates"] if r["score"] < p["accept"]]
        if label_of[L["id"]] == "undetermined" and cand:
            root[L["id"]] = max(cand, key=lambda r: (r["overlap"], -r["separation_max"]))["target"]

    def final_root(i):
        seen_r = set()
        while i in root and i not in seen_r:
            seen_r.add(i)
            i = root[i]
        return i

    merged_into = {i: final_root(i) for i in root}
    merged = {}
    for i, r in merged_into.items():
        if r == i:
            continue
        merged.setdefault(r, []).append(i)

    def merged_variant():
        w = copy.deepcopy({k: v for k, v in walls.items() if k != "lines"})
        w["association_filter"] = MODEL + ":layer_merged(HYPOTHESIS: weaker layers are drifted fragments of the stronger line)"
        out = []
        for l in out_lines:
            i = l["id"]
            if i in merged_into and merged_into[i] != i:
                continue
            if label_of[i] not in ("room_boundary_candidate", "undetermined"):
                continue
            l = copy.deepcopy(l)
            if i in merged:
                R = by_id[i]
                spans = list(R["spans"])
                seen_all = R["seen"].copy()
                sep = 0.0
                for j in merged[i]:
                    Lj = by_id[j]
                    for a, b in Lj["spans"]:
                        ends = sorted(float(R["d"] @ (Lj["c"] * Lj["m"] + x * Lj["d"])) for x in (a, b))
                        spans.append((ends[0], ends[1]))
                    seen_all |= Lj["seen"]
                    for r in ev[j]["duplicate_candidates"]:
                        if r["target"] in (i, *merged[i]):
                            sep = max(sep, r["separation_max"])
                l["observed_spans"] = [[a, b] for a, b in _merge_spans(spans)]
                fids = [int(x) for x in (walls.get("observations") or {}).get("frame_ids") or []]
                l["support"] = {**(l.get("support") or {}), "distinct_frames": int(seen_all.sum()),
                                "frame_ids": [fids[k] if k < len(fids) else int(k) for k in np.flatnonzero(seen_all)]}
                unc = dict(l.get("uncertainty") or {})
                unc["offset_sd"] = float(max(float(unc.get("offset_sd", 0.0) or 0.0), 0.5 * sep))
                l["uncertainty"] = unc
                l["association"] = {**l["association"], "merged_from": sorted(merged[i], key=str), "hypothesis": "layer_merge"}
            out.append(l)
        w["lines"] = out
        return w

    walls_all = variant(lambda c: True)
    if bad:
        walls_all["lines"].extend(copy.deepcopy(l) for l in walls["lines"] if isinstance(l, dict) and l.get("id") in bad)
    counts = {c: sum(1 for l in out_lines if l["association"]["class"] == c) for c in CLASSES}
    return {"model": MODEL, "status": "ok", "reasons": reasons, "params": copy.deepcopy(p),
            "walls_all": walls_all,
            "walls_boundary_candidates": variant(lambda c: c == "room_boundary_candidate"),
            "walls_with_undetermined": variant(lambda c: c in ("room_boundary_candidate", "undetermined")),
            "walls_layer_representatives": variant_ids(
                lambda l: l["association"]["class"] == "room_boundary_candidate" or l["id"] in reps),
            "walls_layer_merged": merged_variant(),
            "report": {"tolerance": float(tol), "extent": extent, "noise_scale": noise,
                       "height_band": list(band) if band else None, "counts": counts, "lines": summary,
                       "pairs": _jsonable(pairs), "layered_groups": groups, "layer_representatives": reps,
                       "layer_merges_hypothesis": {str(k): v for k, v in merged.items()},
                       "footprint_clusters": clusters, "footprint_triples": len(triples),
                       "note": "classes are evidence labels, not truth; undetermined lines are kept in walls_all and "
                               "walls_with_undetermined"}}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o



