# Wall-boundary pipeline interface (task 0012)

Shared by workers A (`wall_extraction.py`), B (`boundary_assembly.py`), C (`compare_wall_boundaries.py`,
`boundary_viewer.py`) and Codex's `ray_boundary_evidence.py` (do not edit Codex's files). No module may read ground truth,
simulator data, camera-up or camera-path orientation; scale-free thresholds; arbitrary wall angles; no forced rectangle,
fixed wall count, Manhattan assumption or forced corner.

## Common 2D frame (same as room_hypothesis / footprint)
Given a unit structural axis `a` (sign unresolved) and the cloud median `o`:
`seed = [1,0,0] if |a[0]| < .9 else [0,1,0]`, `e1 = unit(cross(a, seed))`, `e2 = cross(a, e1)`,
`xy = ((p-o)@e1, (p-o)@e2)`, height `h = (p-o)@a`. DA3 extrinsics are world-to-camera (`world = (cam - t) @ R`).
`extent` = norm(p95 - p5) of the 3D cloud. Units arbitrary.

## A: `extract_wall_lines(prediction, axis, **kw) -> dict`  (`prediction` = dict of npz arrays: depth, confidence,
intrinsics, extrinsics, source_indices[, images])
```
{"model": "observed_wall_lines", "status": "ok"|"insufficient"|..., "reasons": [...],
 "axis_world": [..], "frame": {"origin_world": [3], "e1": [3], "e2": [3], "extent": float},
 "noise_scale": float,                       # data-derived, in cloud units
 "lines": [{"id": int,
            "normal_xy": [mx, my],           # unit, line is  m . x = offset ; no outward/inward convention implied
            "offset": float, "direction_xy": [dx, dy],
            "observed_spans": [[t0, t1], ...],        # intervals along direction_xy (t = d . x) that have support
            "support": {"distinct_frames": int, "frame_ids": [source frame indices], "view_spread_deg": float,
                        "along_coverage_fraction": float, "weight": float},   # weight is NOT a raw point count
            "uncertainty": {"offset_sd": float, "angle_sd_deg": float, "method": "leave_view_out"|"bootstrap"},
            "height_range": [lo, hi]}],
 "observations": {"frame_ids": [...], "camera_xy": [[x, y], ...],            # camera centres in the common frame
                  "hits_xy": [[[x, y], ...], ...],                          # per frame: subsampled 2D surface hits
                  "hit_heights": [[h, ...], ...]}}                          # per frame, same shape as hits
```
## B: `assemble_boundary(walls, scorer=None, **kw) -> dict`
Takes A's dict (any producer that satisfies the schema, incl. synthetic fixtures). Output (compatible with
`room_hypothesis.estimate_polygon` where possible):
`status` in `complete_polygon_hypothesis | ambiguous | incomplete`, `vertices_xy` (CCW) or None, `edges[{index,
line_id, start_xy, end_xy, length, label: observed|extrapolated, observed_fraction, observed_spans_xy, uncertainty,
supporting_frames, free_space: {...}}]`, `alternatives` (when ambiguous: the competing layouts), `incomplete_boundaries
[{reason}]`, `frame` (copied from A), `axes_world`, `coordinate_origin_world`, `units`.
Observed vs extrapolated is kept per edge/span; floor/ceiling are NOT mandatory edges.

### free-space scorer protocol (B defines a default; Codex supplies ray_boundary_evidence later via an adapter)
`scorer(p0_xy, p1_xy) -> {"free_space_contradiction": bool, "free_space_fraction": float,   # share of the segment crossed
 by camera-to-surface rays of >= `min_frames` independent frames (observed EMPTY)
 "occupied_fraction": float,                      # share supported by surface hits
 "unobserved_fraction": float,                    # share neither crossed by rays nor hit: occlusion/unknown
 "supporting_frames": int, "contradicting_frames": int}`
A proposed extension is rejected only when contradicted by observed free space or crossing supported occupied geometry;
unobserved is ambiguity, never evidence for or against.

## C: stability and evaluation
`compare_wall_boundaries.py` runs A then B on frozen predictions (36 linspace, 40 stride-3, 60 stride-2, textured 40),
for EVERY co-supported axis hypothesis from `geometry_orientation.axis_hypotheses`, then (evaluation only, after freezing)
scores with `evaluate_floorplan.py`. `boundary_viewer.py` writes an offline HTML (no network) showing observed patches,
extrapolated edges, final polygon, true outline (labelled evaluator-oracle), per budget.
