"""Assemble a closed 2D floor-plan boundary from observed wall lines (task 0012, worker B).

Input is the `observed_wall_lines` dict of WALL_BOUNDARY_INTERFACE.md. Output is a boundary hypothesis whose edges keep
OBSERVED and EXTRAPOLATED spans apart. Nothing here reads ground truth, simulator data, camera-up or camera path.

Method (generalises room_hypothesis.estimate_polygon's evidence-supported search):
* every wall stretch is a node; one line observed in several stretches may be one wall (the gaps between stretches are
  extrapolated) or several collinear walls; each node can be walked in either direction (opposite-facing collinear walls);
* a corner u -> v exists only when the two lines are not near-parallel and both walls' observed ends plus a bounded
  extrapolation (relative to the extent and to the corner's own propagated uncertainty) reach the intersection;
* every simple directed cycle of corners is a candidate layout. It is kept only if it is a simple counter-clockwise
  polygon, every observed stretch left out of it lies inside it (partition/furniture) and none outside, no extrapolated
  span is contradicted by free-space rays or crosses supported occupied geometry, cameras are inside and observed hits
  are not outside. Unobserved/occluded space is ambiguity, never evidence.
* the least-extrapolated layout wins; near-equal layouts are reported as `ambiguous` with `alternatives`; nothing
  valid is `incomplete` (no corner or polygon is fabricated).
"""

from __future__ import annotations

import math

import numpy as np

from room_hypothesis import _inside, _is_simple, _proper_cross, _segment_distance

DEFAULTS = {
    "reach_fraction": 0.25,            # normal corner extrapolation limit / extent
    "fallback_reach_fraction": 0.40,    # retry limit only when the normal search finds no closed cycle
    "overshoot_fraction": 0.03,        # tolerated observed stretch beyond a corner / extent
    "min_corner_angle_deg": 8.0,       # lines closer to parallel than this make no corner
    "max_corner_sd_fraction": 0.10,    # corner position uncertainty / extent
    "max_gap_fraction": 0.5,           # largest unobserved gap bridged inside one wall / extent
    "min_line_frames": 2,              # lines supported by fewer distinct frames are ignored
    "min_edge_observed_fraction": 0.15,
    "min_edge_observed_length_fraction": 0.03,
    "min_edge_length_fraction": 0.02,
    "tie_margin": 0.03,                # extrapolated-length difference / extent that still counts as a tie
    "max_outside_hit_fraction": 0.04,
    "max_cameras_outside_fraction": 0.15,
    "search_budget": 200000,
    "max_cycle_length": 14,
    "max_nodes": 80,
    "min_frames": 3,
    "min_side_frames": 2,              # frames needed on one side (none on the other) to call a stretch a boundary face
    "observed_label_fraction": 0.6,
}
MAX_ALTERNATIVES = 6


# ---------------------------------------------------------------------------
# Free-space scorer (default) and adapter
# ---------------------------------------------------------------------------
def _unobserved_result() -> dict:
    return {"free_space_contradiction": False, "free_space_fraction": 0.0, "occupied_fraction": 0.0,
            "unobserved_fraction": 1.0, "supporting_frames": 0, "contradicting_frames": 0}


class FreeSpaceScorer:
    """Camera-to-hit rays in 2D. Per frame the hits form a polar range profile; a point is observed EMPTY by that frame
    when it lies in front of the (angularly dense enough) neighbouring hits. A point is free when >= `min_frames`
    distinct frames see it empty (several rays of one frame are never independent), occupied when a hit lies on it, and
    otherwise unobserved. Scale-free: the tolerance defaults to 1% of the hit bounding diagonal."""

    def __init__(self, observations: dict, min_frames: int = 3, tolerance: float | None = None,
                 contradiction_fraction: float = 0.15, max_ray_gap_deg: float = 4.0, max_samples: int = 256):
        obs = observations or {}
        self.min_frames = int(min_frames)
        self.contradiction_fraction = float(contradiction_fraction)
        self.max_gap = math.radians(max_ray_gap_deg)
        self.max_samples = int(max_samples)
        frames, everything = [], []
        for frame_index, (cam, hits) in enumerate(zip(obs.get("camera_xy") or [], obs.get("hits_xy") or [])):
            hits = np.asarray(hits, dtype=float).reshape(-1, 2)
            cam = np.asarray(cam, dtype=float).reshape(2)
            ok = np.isfinite(hits).all(axis=1) & np.isfinite(cam).all()
            hits = hits[ok]
            if len(hits) < 3:
                continue
            rel = hits - cam
            rng = np.hypot(rel[:, 0], rel[:, 1])
            keep = rng > 1e-12
            hits, rel, rng = hits[keep], rel[keep], rng[keep]
            if len(hits) < 3:
                continue
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            order = np.argsort(ang, kind="stable")
            ang, rng = ang[order], rng[order]
            frames.append({"id": frame_index, "cam": cam, "hits": hits,
                           "ang": np.concatenate([[ang[-1] - 2 * np.pi], ang, [ang[0] + 2 * np.pi]]),
                           "rng": np.concatenate([[rng[-1]], rng, [rng[0]]])})
            everything.append(hits)
        self.frames = frames
        if tolerance is None:
            if everything:
                allp = np.concatenate(everything)
                tolerance = 0.01 * float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0)))
            else:
                tolerance = 1e-6
        self.tolerance = max(float(tolerance), 1e-12)

    def __call__(self, p0_xy, p1_xy) -> dict:
        p0, p1 = np.asarray(p0_xy, dtype=float), np.asarray(p1_xy, dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        if not self.frames or length < 1e-12 or not np.isfinite(length):
            return _unobserved_result()
        n = int(np.clip(math.ceil(length / (0.5 * self.tolerance)), 8, self.max_samples))
        pts = p0 + ((np.arange(n) + 0.5) / n)[:, None] * (p1 - p0)
        occupied = np.zeros(n, dtype=bool)
        free_by_frame, near_frames = [], set()
        for frame in self.frames:
            d2 = ((pts[:, None, :] - frame["hits"][None, :, :]) ** 2).sum(axis=2).min(axis=1)
            near = d2 <= (1.5 * self.tolerance) ** 2
            if near.any():
                near_frames.add(frame["id"])
            occupied |= near
            rel = pts - frame["cam"]
            dist = np.hypot(rel[:, 0], rel[:, 1])
            idx = np.clip(np.searchsorted(frame["ang"], np.arctan2(rel[:, 1], rel[:, 0])), 1, len(frame["ang"]) - 1)
            gap = frame["ang"][idx] - frame["ang"][idx - 1]
            nearest = np.minimum(frame["rng"][idx - 1], frame["rng"][idx])
            free_by_frame.append((frame["id"], (gap <= self.max_gap) & (dist > 1e-12) & (dist < nearest - 2 * self.tolerance)))
        count = np.sum([f for _, f in free_by_frame], axis=0)
        free = (count >= self.min_frames) & ~occupied
        crossing = {fid for fid, f in free_by_frame if (f & free).any()}
        free_fraction, occupied_fraction = float(free.mean()), float(occupied.mean())
        return {"free_space_contradiction": bool(free_fraction >= self.contradiction_fraction),
                "free_space_fraction": free_fraction, "occupied_fraction": occupied_fraction,
                "unobserved_fraction": float(max(0.0, 1.0 - free_fraction - occupied_fraction)),
                "supporting_frames": int(len(near_frames | crossing)), "contradicting_frames": int(len(crossing))}


def adapt_ray_scorer(module_scorer, contradiction_fraction: float = 0.15):
    """Wrap any callable `(p0_xy, p1_xy) -> dict` following only the protocol keys, normalising what it returns. A scorer
    that raises or returns a non-dict is treated as 'no evidence' (never as a contradiction) and the error is kept."""
    def scorer(p0_xy, p1_xy) -> dict:
        try:
            raw = module_scorer(p0_xy, p1_xy)
            if not isinstance(raw, dict):
                raise TypeError("scorer returned %s, expected dict" % type(raw).__name__)
        except Exception as exc:  # noqa: BLE001 - external scorer; failure means no evidence, recorded
            return {**_unobserved_result(), "scorer_error": f"{type(exc).__name__}: {exc}"}
        free = float(np.clip(raw.get("free_space_fraction", 0.0) or 0.0, 0.0, 1.0))
        occupied = float(np.clip(raw.get("occupied_fraction", 0.0) or 0.0, 0.0, 1.0))
        unobserved = raw.get("unobserved_fraction")
        unobserved = float(np.clip(1.0 - free - occupied, 0.0, 1.0)) if unobserved is None else float(np.clip(unobserved, 0.0, 1.0))
        contradiction = raw.get("free_space_contradiction")
        contradiction = free >= contradiction_fraction if contradiction is None else bool(contradiction)
        return {"free_space_contradiction": contradiction, "free_space_fraction": free, "occupied_fraction": occupied,
                "unobserved_fraction": unobserved, "supporting_frames": int(raw.get("supporting_frames", 0) or 0),
                "contradicting_frames": int(raw.get("contradicting_frames", 0) or 0)}
    return scorer


def ray_evidence_scorer(prediction, frame: dict, axes_world, height_range, tolerance: float, stride: int = 8,
                        sub_patches: int = 8, min_frames: int = 3, conflict_fraction: float = 0.5,
                        contradiction_fraction: float = 0.15, rays=None):
    """Scorer-protocol adapter over ray_boundary_evidence (Codex). A 2D segment in the common frame becomes a vertical
    patch over the mid-height band (15%..85% of `height_range`, heights along the structural axis `axes_world`), cut into
    short sub-patches so a local opening does not condemn a whole wall. Per sub-patch: contradicted when >= `min_frames`
    views, and more than `conflict_fraction` of the visible views, see rays crossing it before their endpoint; supported
    when visible and not contradicted; otherwise unobserved (occluded endpoints and missed rays are NOT evidence).
    `tolerance` is the caller-declared depth/pose uncertainty (use the lines' own offset_sd). `rays` may supply
    (origins, endpoints, view_ids) instead of a prediction."""
    import ray_boundary_evidence as rbe
    if rays is None:
        origins, endpoints, views = rbe.rays_from_prediction(prediction, stride)
    else:
        origins, endpoints, views = rays
    o = np.asarray(frame["origin_world"], dtype=float)
    e1, e2 = np.asarray(frame["e1"], dtype=float), np.asarray(frame["e2"], dtype=float)
    up = np.asarray(axes_world, dtype=float)
    up = up / np.linalg.norm(up)
    lo, hi = float(height_range[0]), float(height_range[1])
    half_h = max(0.35 * (hi - lo), 1e-9)
    h_mid = 0.5 * (lo + hi)

    def world(xy, h):
        return o + xy[0] * e1 + xy[1] * e2 + h * up

    def scorer(p0_xy, p1_xy) -> dict:
        p0, p1 = np.asarray(p0_xy, dtype=float), np.asarray(p1_xy, dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        if not len(endpoints) or length < 1e-12:
            return _unobserved_result()
        d2 = (p1 - p0) / length
        d3 = d2[0] * e1 + d2[1] * e2
        axes = np.array([d3, up])
        conflicted = supported = 0
        views_support, views_conflict = set(), set()
        step = length / sub_patches
        for k in range(sub_patches):
            mid = p0 + (k + 0.5) * step * d2
            res = rbe.score_patch(origins, endpoints, views, world(mid, h_mid), axes,
                                  [[-0.5 * step, -half_h], [0.5 * step, half_h]], tolerance)
            visible = [v for v in res["per_view"] if v["conflict_fraction_visible"] is not None]
            if not visible:
                continue
            bad = [v["view_id"] for v in visible if v["conflict_fraction_visible"] >= 0.5]
            if len(bad) >= min_frames and len(bad) > conflict_fraction * len(visible):
                conflicted += 1
                views_conflict.update(bad)
            else:
                supported += int(res["counts"]["observed_support"] > 0)
                views_support.update(v["view_id"] for v in visible if v["observed_support"] > 0)
        free_fraction, occupied_fraction = conflicted / sub_patches, supported / sub_patches
        return {"free_space_contradiction": bool(free_fraction >= contradiction_fraction),
                "free_space_fraction": free_fraction, "occupied_fraction": occupied_fraction,
                "unobserved_fraction": max(0.0, 1.0 - free_fraction - occupied_fraction),
                "supporting_frames": len(views_support), "contradicting_frames": len(views_conflict)}
    return scorer


# ---------------------------------------------------------------------------
# Preferred real-data scorer: localized ray evidence (Codex make_boundary_scorer) with an explicit policy
# ---------------------------------------------------------------------------
RAY_FRACTION_DOMAIN = "wall_patch_area_within_supplied_height_band"
DEFAULT_SEGMENT_DOMAIN = "along_segment_samples"
DEFAULT_K_GRID = (1.0, 2.0, 4.0, 8.0)      # fixed before any result was looked at; never chosen with truth


def derive_ray_tolerance(walls: dict, k: float) -> dict:
    """Caller-declared depth/pose tolerance from the producer's own data (no truth):
    tolerance = k * max(median over supported lines of offset_sd, noise_scale), in the common-frame units."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive")
    sds = [float((l.get("uncertainty") or {}).get("offset_sd") or 0.0) for l in walls.get("lines") or []]
    sds = [s for s in sds if np.isfinite(s) and s > 0]
    median_sd = float(np.median(sds)) if sds else 0.0
    noise = float(walls.get("noise_scale") or 0.0)
    base = max(median_sd, noise if np.isfinite(noise) else 0.0)
    if base <= 0:
        raise ValueError("no offset_sd or noise_scale to derive a tolerance from")
    return {"tolerance": float(k * base), "k": float(k), "median_line_offset_sd": median_sd, "noise_scale": noise,
            "rule": "tolerance = k * max(median line offset_sd, noise_scale)"}


def split_prediction_by_component(prediction) -> dict:
    """{component_id: sub-prediction}. Components have unrelated coordinate frames and are never pooled."""
    if "component_ids" not in prediction:
        return {None: prediction}
    ids = np.asarray(prediction["component_ids"])
    n = len(ids)
    out = {}
    for cid in np.unique(ids):
        sel = ids == cid
        out[int(cid)] = {key: (np.asarray(prediction[key])[sel] if np.ndim(prediction[key]) and len(prediction[key]) == n
                               else prediction[key]) for key in prediction}
    return out


def _interior_gaps(walls: dict) -> list:
    """Unobserved stretches bounded by observed spans on BOTH sides of one line: where the extractor saw a wall on each side
    but none between (height-coverage gaps/opening-like or occluded)."""
    gaps = []
    for raw in walls.get("lines") or []:
        try:
            m = np.asarray(raw["normal_xy"], float)
            d = np.asarray(raw["direction_xy"], float)
            spans = _merge([(float(a), float(b)) for a, b in raw.get("observed_spans") or []])
            nm = float(np.linalg.norm(m))
            c = float(raw["offset"]) / nm
            m = m / nm
        except (KeyError, TypeError, ValueError):
            continue
        d = d - m * float(d @ m)
        if np.linalg.norm(d) < 1e-9:
            continue
        d = d / np.linalg.norm(d)
        for (a0, a1), (b0, b1) in zip(spans[:-1], spans[1:]):
            gaps.append({"line_id": raw.get("id"), "m": m, "c": c, "d": d, "t": (a1, b0)})
    return gaps


def real_ray_scorer(prediction, walls: dict, tolerance: float, *, component_id=None, min_views: int = 3,
                    reject_fraction: float = 0.5, stride: int = 8, height_range=None):
    """Preferred real-data scorer: Codex's make_boundary_scorer over the frame / explicit structural axis / height band of
    A's output (explicit structural axis only), plus the policy that makes its localized evidence usable:
    * fractions are over `wall_patch_area_within_supplied_height_band` (label carried in every result), not the along-segment
      coverage;
    * predicted depth causes false conflicts, so there is no zero-conflict rule: a cell conflicts only when >= `min_views`
      independent source frames (grouped by source_indices) cross it before their endpoint and at least half of its visible
      views do;
    * a sub-span (grid column) is REJECTED only when every visible height cell of the column conflicts (a free-space
      contradiction holds at all heights; a partial-height conflict is window/door-like) and the span is not an interior gap
      between observed spans of the same line (the extractor's height-coverage gap / opening). Opening-like conflicts are
      DIAGNOSTIC. The span's boolean is true only when rejected columns cover >= `reject_fraction` of it: one cell never
      rejects a wall;
    * several component_ids: pass `component_id`; one scorer per component, identity kept in every result."""
    import ray_boundary_evidence as rbe
    if component_id is not None:
        components = split_prediction_by_component(prediction)
        if component_id not in components:
            raise ValueError(f"unknown component_id {component_id}")
        prediction = components[component_id]
    frame, axis = walls["frame"], walls["axis_world"]
    if height_range is None:
        band = (walls.get("params") or {}).get("height_band")
        if band is None:
            ranges = [l["height_range"] for l in walls.get("lines") or [] if l.get("height_range")]
            if not ranges:
                raise ValueError("no height band available")
            band = [float(np.median([r[0] for r in ranges])), float(np.median([r[1] for r in ranges]))]
        height_range = band
    base = rbe.make_boundary_scorer(prediction, frame, axis, height_range, tolerance, min_views=min_views,
                                    reject_fraction=reject_fraction, stride=stride)
    gaps = _interior_gaps(walls)
    cache, log = {}, []

    def in_interior_gap(p0, p1):
        for g in gaps:
            if max(abs(float(g["m"] @ q) - g["c"]) for q in (p0, p1)) > 2.0 * tolerance:
                continue
            t0, t1 = sorted((float(g["d"] @ p0), float(g["d"] @ p1)))
            if t0 >= g["t"][0] - tolerance and t1 <= g["t"][1] + tolerance:
                return g["line_id"]
        return None

    def scorer(p0_xy, p1_xy) -> dict:
        p0, p1 = np.asarray(p0_xy, float), np.asarray(p1_xy, float)
        key = tuple(np.round(np.concatenate([p0, p1]), 9).tolist())
        if key in cache:
            return cache[key]
        raw = base(p0, p1)
        grid = raw["ray_evidence"]["spatial_grid"]
        size = int(grid["size"])
        columns, rejected = [], []
        for x in range(size):
            cells = [c for c in grid["cells"] if c["cell"][0] == x]
            visible = [c for c in cells if c["status"] == "visible_evidence"]
            bad = [c for c in visible if c["conflict_view_count"] >= min_views and (c["conflict_fraction_equal_view"] or 0) >= 0.5]
            if visible and len(bad) == len(visible) and len(visible) >= max(2, int(np.ceil(size / 2))):
                state = "full_height_conflict"
            elif bad:
                state = "partial_height_conflict_opening_like"
            elif visible:
                state = "visible_not_conflicting"
            else:
                state = "unknown"
            columns.append({"column": x, "state": state, "visible_cells": len(visible), "conflicting_cells": len(bad),
                            "max_conflict_views": max([c["conflict_view_count"] for c in cells] or [0])})
            if state == "full_height_conflict":
                rejected.append(x)
        gap_line = in_interior_gap(p0, p1)
        fraction_rejected = len(rejected) / size
        subspans = [{"start_xy": (p0 + (p1 - p0) * x / size).tolist(), "end_xy": (p0 + (p1 - p0) * (x + 1) / size).tolist(),
                     "column": x} for x in rejected]
        contradiction = bool(fraction_rejected >= reject_fraction and gap_line is None)
        result = {"free_space_contradiction": contradiction,
                  "free_space_fraction": float(raw["free_space_fraction"]), "occupied_fraction": float(raw["occupied_fraction"]),
                  "unobserved_fraction": float(raw["unobserved_fraction"]),
                  "supporting_frames": int(raw["supporting_frames"]), "contradicting_frames": int(raw["contradicting_frames"]),
                  "fraction_domain": RAY_FRACTION_DOMAIN,
                  "contradiction_policy": {"min_views": int(min_views), "reject_fraction": float(reject_fraction),
                                           "rule": "reject a column only if all visible height cells conflict; reject the span only if "
                                                   "rejected columns cover >= reject_fraction and the span is not an interior gap",
                                           "tolerance": float(tolerance)},
                  "raw_scorer_boolean_diagnostic": bool(raw["free_space_contradiction"]),
                  "rejected_column_fraction": float(fraction_rejected), "rejected_subspans": subspans,
                  "columns": columns, "interior_gap_of_line": gap_line,
                  "conflict_treated_as_diagnostic": bool(fraction_rejected >= reject_fraction and gap_line is not None),
                  "component_id": component_id, "tolerance": float(tolerance)}
        cache[key] = result
        log.append({"start_xy": p0.tolist(), "end_xy": p1.tolist(), **result})
        return result
    scorer.log = log
    scorer.component_id = component_id
    return scorer


def assemble_boundary_ray(prediction, walls: dict, k: float = 2.0, *, component_id=None, scorer_kw=None, **kw) -> dict:
    """assemble_boundary with real_ray_scorer; tolerance derived by `derive_ray_tolerance` (rule recorded in the output)."""
    tol = derive_ray_tolerance(walls, k)
    scorer = real_ray_scorer(prediction, walls, tol["tolerance"], component_id=component_id, **(scorer_kw or {}))
    result = assemble_boundary(walls, scorer=scorer, **kw)
    calls = scorer.log
    result["free_space_scorer"] = {
        "name": "ray_boundary_evidence.make_boundary_scorer + boundary_assembly policy", "fraction_domain": RAY_FRACTION_DOMAIN,
        "component_id": component_id, **tol,
        "unique_extrapolated_spans_scored": len(calls),
        "spans_rejected": sum(c["free_space_contradiction"] for c in calls),
        "spans_conflict_diagnostic_only": sum(c["conflict_treated_as_diagnostic"] for c in calls),
        "spans_accepted": sum(not c["free_space_contradiction"] for c in calls), "calls": calls}
    result["component_id"] = component_id
    return result


def assemble_boundaries_by_component(prediction, walls_by_component: dict, k: float = 2.0, **kw) -> dict:
    """One scorer and one boundary per component; `walls_by_component` is {component_id: A's dict for that component}."""
    return {cid: assemble_boundary_ray(prediction, walls, k, component_id=cid, **kw)
            for cid, walls in sorted(walls_by_component.items(), key=lambda kv: str(kv[0]))}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _f(value) -> float:
    return float(value)


def _pt(p) -> list:
    return [float(p[0]), float(p[1])]


def _merge(spans: list[tuple[float, float]], eps: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b - a <= 0:
            continue
        if out and a <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _prepare_lines(walls: dict, min_frames: int) -> tuple[list[dict], list[dict]]:
    lines, ignored = [], []
    for raw in walls.get("lines") or []:
        try:
            m = np.asarray(raw["normal_xy"], dtype=float)
            c = float(raw["offset"])
            d = np.asarray(raw["direction_xy"], dtype=float)
            spans = _merge([(float(a), float(b)) for a, b in raw.get("observed_spans") or []])
        except (KeyError, TypeError, ValueError):
            ignored.append({"line_id": raw.get("id") if isinstance(raw, dict) else None, "reason": "malformed_line"})
            continue
        if not (np.isfinite(m).all() and np.isfinite(d).all() and np.isfinite(c)) or np.linalg.norm(m) < 1e-9:
            ignored.append({"line_id": raw.get("id"), "reason": "non_finite_line"})
            continue
        norm = float(np.linalg.norm(m))
        m, c = m / norm, c / norm
        d = d - m * float(d @ m)                      # direction orthogonal to the normal, sign kept
        if np.linalg.norm(d) < 1e-9:
            d = np.array([-m[1], m[0]])
        d = d / np.linalg.norm(d)
        support = raw.get("support") or {}
        frames = int(support.get("distinct_frames", 0) or 0)
        if frames < min_frames:
            ignored.append({"line_id": raw.get("id"), "reason": "too_few_distinct_frames", "distinct_frames": frames})
            continue
        if not spans:
            ignored.append({"line_id": raw.get("id"), "reason": "no_observed_span"})
            continue
        unc = raw.get("uncertainty") or {}
        height_range = raw.get("height_range") or []
        height_span = (float(height_range[1]) - float(height_range[0])) if len(height_range) == 2 else None
        lines.append({"id": raw.get("id", len(lines)), "m": m, "c": c, "d": d, "spans": spans,
                      "offset_sd": float(unc.get("offset_sd", 0.0) or 0.0),
                      "angle_sd_deg": float(unc.get("angle_sd_deg", 0.0) or 0.0),
                      "distinct_frames": frames, "frame_ids": list(support.get("frame_ids") or []),
                      "height_span": height_span})
    return lines, ignored


def _extent_of(walls: dict, lines: list[dict]) -> float:
    value = (walls.get("frame") or {}).get("extent")
    if value is not None and np.isfinite(value) and value > 0:
        return float(value)
    pts = [l["c"] * l["m"] + t * l["d"] for l in lines for a, b in l["spans"] for t in (a, b)]
    for hits in (walls.get("observations") or {}).get("hits_xy") or []:
        pts.extend(np.asarray(hits, dtype=float).reshape(-1, 2).tolist())
    if not pts:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    return float(np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)))


def _build_nodes(lines: list[dict], max_gap: float) -> list[dict]:
    """Every contiguous run of stretches of a line, in both walking directions."""
    nodes = []
    for li, line in enumerate(lines):
        k = len(line["spans"])
        for i in range(k):
            for j in range(i, k):
                if any(line["spans"][q + 1][0] - line["spans"][q][1] > max_gap for q in range(i, j)):
                    continue
                for s in (1, -1):
                    run = line["spans"][i:j + 1]
                    spans_t = [(a, b) for a, b in run] if s > 0 else [(-b, -a) for a, b in reversed(run)]
                    nodes.append({"line": li, "i": i, "j": j, "s": s, "spans": spans_t,
                                  "lo": spans_t[0][0], "hi": spans_t[-1][1], "tdir": s * line["d"]})
    return nodes


def _corner(u: dict, v: dict, lines: list[dict], extent: float, p: dict) -> dict | None:
    lu, lv = lines[u["line"]], lines[v["line"]]
    det = float(lu["m"][0] * lv["m"][1] - lu["m"][1] * lv["m"][0])
    if abs(det) < math.sin(math.radians(p["min_corner_angle_deg"])):
        return None
    vertex = np.linalg.solve(np.array([lu["m"], lv["m"]]), np.array([lu["c"], lv["c"]]))
    gap_u = float(u["tdir"] @ vertex) - u["hi"]
    gap_v = v["lo"] - float(v["tdir"] @ vertex)
    pos_sd = math.hypot(lu["offset_sd"], lv["offset_sd"]) / abs(det)
    extra_sd = math.hypot(max(gap_u, 0.0) * math.tan(math.radians(lu["angle_sd_deg"])),
                          max(gap_v, 0.0) * math.tan(math.radians(lv["angle_sd_deg"])))
    sd = math.hypot(pos_sd, extra_sd)
    if sd > p["max_corner_sd_fraction"] * extent:
        return None
    overshoot = p["overshoot_fraction"] * extent + 2.0 * pos_sd
    reach = p["reach_fraction"] * extent
    if min(gap_u, gap_v) < -overshoot or max(gap_u, gap_v) > reach:
        return None
    return {"vertex": vertex, "gap_u": gap_u, "gap_v": gap_v, "sd": sd}


def _find_cycles(nodes: list[dict], succ: dict, budget: int, max_len: int) -> tuple[list[list[int]], bool]:
    cycles: list[list[int]] = []
    steps = [budget]

    def compatible(a: int, b: int) -> bool:
        na, nb = nodes[a], nodes[b]
        return na["line"] != nb["line"] or na["j"] < nb["i"] or nb["j"] < na["i"]

    def dfs(start: int, path: list[int]) -> None:
        if steps[0] <= 0:
            return
        steps[0] -= 1
        for nxt in succ[path[-1]]:
            if nxt == start and len(path) >= 3:
                cycles.append(list(path))
            elif nxt > start and nxt not in path and len(path) < max_len and all(compatible(nxt, q) for q in path):
                path.append(nxt)
                dfs(start, path)
                path.pop()

    for start in range(len(nodes)):
        dfs(start, [start])
    return cycles, steps[0] <= 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _base(walls: dict, status: str, **extra) -> dict:
    frame = walls.get("frame") if isinstance(walls, dict) else None
    frame = frame or {}
    return {"model": "assembled_wall_boundary_hypothesis", "status": status, "vertices_xy": None, "edges": [],
            "alternatives": [], "incomplete_boundaries": [], "ignored_lines": [], "frame": frame or None,
            "axes_world": ({"e1": frame.get("e1"), "e2": frame.get("e2"), "up": (walls or {}).get("axis_world")}
                           if frame else None),
            "coordinate_origin_world": frame.get("origin_world"), "units": "arbitrary", "area": None,
            "perimeter": None, "shape_gate_eligible": False, **extra}


def assemble_boundary(walls: dict, scorer=None, **kw) -> dict:
    p = {**DEFAULTS, **{k: v for k, v in kw.items() if k in DEFAULTS}}
    if not isinstance(walls, dict):
        return _base({}, "incomplete", incomplete_boundaries=[{"reason": "no_wall_input"}])
    if walls.get("status") not in (None, "ok"):
        return _base(walls, "incomplete", incomplete_boundaries=[
            {"reason": "wall_extraction_not_ok", "producer_status": walls.get("status"), "producer_reasons": walls.get("reasons")}])
    lines, ignored = _prepare_lines(walls, p["min_line_frames"])
    extent = _extent_of(walls, lines)
    if extent <= 0 or len(lines) < 3:
        return _base(walls, "incomplete", ignored_lines=ignored, incomplete_boundaries=[
            {"reason": "fewer_than_three_supported_wall_lines" if extent > 0 else "degenerate_extent", "supported_lines": len(lines)}])
    observations = walls.get("observations") or {}
    noise = float(walls.get("noise_scale") or 0.0)
    tol = max(2.0 * noise, 0.01 * extent)
    if scorer is None:
        scorer = FreeSpaceScorer(observations, min_frames=p["min_frames"], tolerance=tol)
    nodes = _build_nodes(lines, p["max_gap_fraction"] * extent)
    if len(nodes) > p["max_nodes"] * 2:
        return _base(walls, "incomplete", ignored_lines=ignored, incomplete_boundaries=[
            {"reason": "too_many_wall_candidates_for_bounded_search", "nodes": len(nodes)}])
    links: dict = {}
    for a in range(len(nodes)):
        for b in range(len(nodes)):
            if a != b and nodes[a]["line"] != nodes[b]["line"]:
                link = _corner(nodes[a], nodes[b], lines, extent, p)
                if link is not None:
                    links[(a, b)] = link
    succ = {a: [b for b in range(len(nodes)) if (a, b) in links] for a in range(len(nodes))}
    cycles, exhausted = _find_cycles(nodes, succ, p["search_budget"], p["max_cycle_length"])
    search = {"nodes": len(nodes), "corner_links": len(links), "closed_cycles": len(cycles), "step_budget_exhausted": exhausted}
    if not exhausted and not cycles and p["fallback_reach_fraction"] > p["reach_fraction"]:
        # A partial wall may stop well before a genuine room corner. Keep the
        # normal bounded search as the default, and expand the corner reach
        # only when that search cannot form any cycle. The ordinary evidence
        # checks below still have to accept every expanded closure.
        fallback = {**p, "reach_fraction": p["fallback_reach_fraction"]}
        links = {}
        for a in range(len(nodes)):
            for b in range(len(nodes)):
                if a != b and nodes[a]["line"] != nodes[b]["line"]:
                    link = _corner(nodes[a], nodes[b], lines, extent, fallback)
                    if link is not None:
                        links[(a, b)] = link
        succ = {a: [b for b in range(len(nodes)) if (a, b) in links] for a in range(len(nodes))}
        cycles, exhausted = _find_cycles(nodes, succ, p["search_budget"], p["max_cycle_length"])
        search.update({"corner_links": len(links), "closed_cycles": len(cycles),
                       "fallback_reach_fraction": p["fallback_reach_fraction"],
                       "fallback_used": bool(cycles), "step_budget_exhausted": exhausted})
    if exhausted:
        return _base(walls, "incomplete", ignored_lines=ignored, boundary_search=search,
                     incomplete_boundaries=[{"reason": "boundary_search_budget_exhausted"}])
    if not cycles:
        has_next = {nodes[a]["line"] for a in succ if succ[a]}
        has_prev = {nodes[b]["line"] for (a, b) in links}
        no_next = sorted(l["id"] for i, l in enumerate(lines) if i not in has_next)
        no_prev = sorted(l["id"] for i, l in enumerate(lines) if i not in has_prev)
        return _base(walls, "incomplete", ignored_lines=ignored, boundary_search=search, incomplete_boundaries=[
            {"reason": "no_closed_boundary_through_supported_walls", "walls_without_supported_next_corner": no_next,
             "walls_without_supported_previous_corner": no_prev,
             "detail": "a corner needs both walls' observed spans plus bounded extrapolation to reach it; "
                       "a missing wall or an over-long unobserved stretch leaves the boundary open"}])

    cams = np.asarray(observations.get("camera_xy") or [], dtype=float).reshape(-1, 2)
    hit_sets = [np.asarray(h, dtype=float).reshape(-1, 2) for h in observations.get("hits_xy") or []]
    hit_sets = hit_sets[:len(cams)] + [np.empty((0, 2))] * max(0, len(cams) - len(hit_sets))
    hits = np.concatenate(hit_sets) if hit_sets else np.empty((0, 2))
    hits = hits[np.isfinite(hits).all(axis=1)] if len(hits) else hits
    stretch_segments = [(li, a, b, lines[li]["c"] * lines[li]["m"] + a * lines[li]["d"], lines[li]["c"] * lines[li]["m"] + b * lines[li]["d"])
                        for li, l in enumerate(lines) for a, b in l["spans"]]
    def side_frames(li, a, b):
        """Distinct frames whose hits lie on this stretch, split by which side of the line their camera is on."""
        line = lines[li]
        s0, s1 = line["c"] * line["m"] + a * line["d"], line["c"] * line["m"] + b * line["d"]
        pos = neg = 0
        for cam, h in zip(cams, hit_sets):
            if len(h) and (_segment_distance(h, s0, s1) <= 1.5 * tol).any():
                if float(line["m"] @ cam) - line["c"] > 0:
                    pos += 1
                else:
                    neg += 1
        return pos, neg
    one_sided = {}
    for li, line in enumerate(lines):
        for q, (a, b) in enumerate(line["spans"]):
            pos, neg = side_frames(li, a, b)
            one_sided[(li, q)] = (pos >= p["min_side_frames"] and neg == 0) or (neg >= p["min_side_frames"] and pos == 0)

    def point(node, t):
        line = lines[node["line"]]
        return line["c"] * line["m"] + node["s"] * t * line["d"]

    def evaluate(order: list[int]) -> dict:
        n = len(order)
        vertices = np.array([links[(order[k], order[(k + 1) % n])]["vertex"] for k in range(n)])
        problems, edges, extrapolated_length = [], [], 0.0
        for k, ni in enumerate(order):
            node = nodes[ni]
            start, end = vertices[k - 1], vertices[k]
            t0, t1 = float(node["tdir"] @ start), float(node["tdir"] @ end)
            if t1 - t0 < p["min_edge_length_fraction"] * extent:
                problems.append(("degenerate_edge", lines[node["line"]]["id"]))
                continue
            observed = _merge([(max(a, t0), min(b, t1)) for a, b in node["spans"] if min(b, t1) > max(a, t0)])
            observed_length = sum(b - a for a, b in observed)
            fraction = observed_length / (t1 - t0)
            if fraction < p["min_edge_observed_fraction"] or observed_length < p["min_edge_observed_length_fraction"] * extent:
                problems.append(("edge_unsupported", lines[node["line"]]["id"]))
                continue
            extrapolated, cursor = [], t0
            for a, b in observed:
                if a - cursor > 1e-9:
                    extrapolated.append((cursor, a))
                cursor = b
            if t1 - cursor > 1e-9:
                extrapolated.append((cursor, t1))
            extrapolated_length += sum(b - a for a, b in extrapolated)
            line = lines[node["line"]]
            sd_start, sd_end = links[(order[k - 1], ni)]["sd"], links[(ni, order[(k + 1) % n])]["sd"]
            spans = sorted([(a, b, "observed") for a, b in observed] + [(a, b, "extrapolated") for a, b in extrapolated])
            edges.append({
                "index": k, "line_id": line["id"], "start_xy": _pt(start), "end_xy": _pt(end), "length": _f(t1 - t0),
                "label": "observed" if fraction >= p["observed_label_fraction"] else "extrapolated",
                "observed_fraction": _f(fraction), "has_extrapolated_span": bool(extrapolated),
                "observed_spans_xy": [[_pt(point(node, a)), _pt(point(node, b))] for a, b in observed],
                "extrapolated_spans_xy": [[_pt(point(node, a)), _pt(point(node, b))] for a, b in extrapolated],
                "spans": [{"start_xy": _pt(point(node, a)), "end_xy": _pt(point(node, b)), "label": lab} for a, b, lab in spans],
                "uncertainty": {"offset_sd": line["offset_sd"], "angle_sd_deg": line["angle_sd_deg"],
                                "start_corner_sd": _f(sd_start), "end_corner_sd": _f(sd_end),
                                "extrapolated_length": _f(sum(b - a for a, b in extrapolated)),
                                "total_sd": _f(max(line["offset_sd"], sd_start, sd_end))},
                "supporting_frames": line["distinct_frames"], "supporting_frame_ids": line["frame_ids"],
                "free_space": {"checked_spans": 0, "contradiction": False, "spans": [], "fraction_domains": []},
                "_extrap_t": extrapolated, "_node": ni})
        area = 0.5 * float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1]))
        if not problems and (area <= 0 or not _is_simple(vertices)):
            problems.append(("polygon_not_simple_counter_clockwise", area))
        if not problems:
            used = {(nodes[ni]["line"], q) for ni in order for q in range(nodes[ni]["i"], nodes[ni]["j"] + 1)}
            for li, line in enumerate(lines):
                for q, (a, b) in enumerate(line["spans"]):
                    if (li, q) in used:
                        continue
                    samples = np.array([line["c"] * line["m"] + (a + (b - a) * w) * line["d"] for w in (0.0, 0.25, 0.5, 0.75, 1.0)])
                    outside = ~_inside(vertices, samples)
                    if outside.any():
                        dist = np.min([_segment_distance(samples[outside], vertices[k], vertices[(k + 1) % n]) for k in range(n)], axis=0)
                        if (dist > 3 * tol).any():
                            problems.append(("unused_supported_stretch_outside_boundary", line["id"]))
                            continue
                    # A tall furnishing or partition can give a one-sided
                    # return too. Require most of its observed stretch to lie
                    # against the proposed perimeter before treating it as a
                    # missed boundary face; a single near-corner sample is not
                    # enough to invalidate an otherwise supported room cycle.
                    near_perimeter = sum(
                        float(np.min([_segment_distance(sample.reshape(1, 2), vertices[k],
                                                         vertices[(k + 1) % n])[0]
                                      for k in range(n)])) <= 3 * tol
                        for sample in samples
                    )
                    # A long interior stretch can instead be an indentation
                    # wall omitted by a too-simple outer cycle. Keep that
                    # contradiction; the short, isolated interior returns
                    # common on tall furniture do not invalidate the cycle.
                    boundary_adjacent = near_perimeter >= 3 or (b - a) >= .15 * extent
                    if one_sided[(li, q)] and boundary_adjacent:
                        problems.append(("unused_one_sided_stretch_is_a_boundary_face", line["id"]))
            for edge in edges:   # extrapolated spans: not through supported occupied geometry, not contradicted by free space
                node = nodes[edge["_node"]]
                for a, b in edge["_extrap_t"]:
                    p0, p1 = point(node, a), point(node, b)
                    if np.linalg.norm(p1 - p0) <= 2 * tol:
                        continue
                    u = (p1 - p0) / np.linalg.norm(p1 - p0)
                    q0, q1 = p0 + u * tol, p1 - u * tol
                    for li, _a, _b, s0, s1 in stretch_segments:
                        if li != node["line"]:
                            w = s1 - s0
                            ln = float(np.linalg.norm(w))
                            if ln > 2 * tol and _proper_cross(q0, q1, s0 + w / ln * tol, s1 - w / ln * tol):
                                problems.append(("extrapolation_crosses_supported_geometry", lines[node["line"]]["id"]))
                    result = scorer(_pt(p0), _pt(p1))
                    edge["free_space"]["spans"].append({"start_xy": _pt(p0), "end_xy": _pt(p1), **result})
                    edge["free_space"]["checked_spans"] += 1
                    domain = str(result.get("fraction_domain", DEFAULT_SEGMENT_DOMAIN))
                    if domain not in edge["free_space"]["fraction_domains"]:
                        edge["free_space"]["fraction_domains"].append(domain)
                    if result.get("free_space_contradiction"):
                        edge["free_space"]["contradiction"] = True
                        problems.append(("free_space_contradiction", lines[node["line"]]["id"]))
            if len(cams) and (~_inside(vertices, cams)).mean() > p["max_cameras_outside_fraction"]:
                problems.append(("cameras_outside_boundary", None))
            if len(hits):
                out = ~_inside(vertices, hits)
                if out.any():
                    dist = np.min([_segment_distance(hits[out], vertices[k], vertices[(k + 1) % n]) for k in range(n)], axis=0)
                    if float((dist > 3 * tol).sum() / len(hits)) > p["max_outside_hit_fraction"]:
                        problems.append(("hits_outside_boundary_missing_wall", None))
        for edge in edges:
            edge.pop("_extrap_t", None)
            edge.pop("_node", None)
        if len(edges) == n:
            # edge i starts at vertex i and ends at vertex i+1 (evaluate_floorplan's convention). Edge k was built from
            # vertex k-1 to vertex k, so the list rotates left by one.
            edges = edges[1:] + edges[:1]
            for i, e in enumerate(edges):
                e["index"] = i
        return {"order": order, "line_order": [lines[nodes[i]["line"]]["id"] for i in order], "vertices": vertices,
                "edges": edges, "area": area, "cost": extrapolated_length / extent, "problems": problems}

    layouts = [evaluate(c) for c in cycles]
    reasons: dict = {}
    for layout in layouts:
        for reason, _detail in layout["problems"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    search["layouts_evaluated"] = len(layouts)
    search["rejected_by_reason"] = dict(sorted(reasons.items()))
    valid = [l for l in layouts if not l["problems"]]
    valid.sort(key=lambda l: (l["cost"], l["line_order"]))
    unique: list[dict] = []
    for layout in valid:       # the same polygon reached through different groupings counts once
        if not any(len(layout["vertices"]) == len(u["vertices"]) and
                   all(np.linalg.norm(u["vertices"] - v, axis=1).min() <= 3 * tol for v in layout["vertices"]) and
                   all(np.linalg.norm(layout["vertices"] - v, axis=1).min() <= 3 * tol for v in u["vertices"]) for u in unique):
            unique.append(layout)
    search["valid_layouts"] = len(unique)

    def describe(layout: dict) -> dict:
        return {"vertices_xy": layout["vertices"].tolist(), "edges": layout["edges"], "area": _f(layout["area"]),
                "extrapolated_length_fraction": _f(layout["cost"]), "line_order": layout["line_order"]}
    if not unique:
        closest = min(layouts, key=lambda l: (len(l["problems"]), l["cost"]))
        return _base(walls, "incomplete", ignored_lines=ignored, boundary_search=search, incomplete_boundaries=[
            {"reason": "no_closed_boundary_survives_evidence_checks", "layouts_rejected_by_reason": search["rejected_by_reason"],
             "closest_layout_problems": [{"reason": r, "line_id": d} for r, d in closest["problems"]]}])
    tied = [l for l in unique if l["cost"] - unique[0]["cost"] < p["tie_margin"]]
    if len(tied) > 1:
        return _base(walls, "ambiguous", ignored_lines=ignored, boundary_search=search,
                     alternatives=[describe(l) for l in tied[:MAX_ALTERNATIVES]],
                     incomplete_boundaries=[{"reason": "equally_supported_layouts", "count": len(tied),
                                             "tie_margin": _f(p["tie_margin"])}])
    best = unique[0]
    search["alternative_valid_layouts_rejected_on_evidence"] = len(unique) - 1
    return _base(walls, "complete_polygon_hypothesis", ignored_lines=ignored, boundary_search=search,
                 vertices_xy=best["vertices"].tolist(), edges=best["edges"], area=_f(best["area"]),
                 perimeter=_f(sum(e["length"] for e in best["edges"])),
                 extrapolated_length_fraction=_f(best["cost"]), line_order=best["line_order"],
                 inferred_edge_count=sum(e["has_extrapolated_span"] for e in best["edges"]), shape_gate_eligible=True)
