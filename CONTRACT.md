# Simulation reconstruction benchmark contract

This benchmark scores submitted predictions against separately held ground truth. Keep `groundtruth.json` in a restricted evaluation location; reconstruction receives prediction schema/examples only and must not read ground truth. Evaluate only after predictions are frozen. Do not place ground-truth values, derived labels, or evaluation feedback in reconstruction inputs.

## Coverage required for completion

Stages are `single_room`, `multiroom`, and `multifloor`. Each stage requires at least 10 distinct `house_id` values and at least 3 distinct `capture_seed` values for every represented house/stage pair. A run is uniquely identified by `(stage, house_id, capture_seed)`; repeated keys are invalid. Extra runs are allowed.

## Files and records

Both UTF-8 JSON files have `schema_version: 1` and a `runs` array. Keep prediction and ground-truth files separate. Each run has:

```json
{
  "stage": "single_room",
  "house_id": "house-001",
  "capture_seed": 41,
  "rooms": [{
    "room_id": "room-a",
    "walls": [{"wall_id": "wall-north", "video_length_m": 4.0}],
    "wall_height_m": 2.7,
    "openings": [{"opening_id": "door-1", "width_m": 0.9, "height_m": 2.0}],
    "shape_dimensions": [4.0, 3.0, 2.7]
  }]
}
```

Prediction documents may set `evaluation_mode` to `metric` (default) or `shape_only`. Metric-mode runs require `scale_source`, a nonempty provenance string describing that prediction's metric-scale source (for example, a documented sensor measurement or external calibration). Do not expose ground-truth-derived metadata to reconstruction. A learned scale prior is not metrology and must be disclosed as such. Missing provenance fails metric evaluation. The evaluator never estimates or fits prediction scale from ground truth in metric mode; it does not judge whether stated provenance is honest, so audit that separately. Shape-only mode accepts arbitrary-unit measurements and does not require metric-scale provenance.

In metric mode all distances are finite, positive metres. A prediction may use `null` for an unavailable wall length, wall height, or opening width; the evaluator reports it as an incomplete measurement and the relevant gate fails. Malformed present values remain invalid. `room_id`, `wall_id`, and `opening_id` are stable semantic identifiers corresponding between prediction and ground truth. IDs must be unique in their scope. Each wall record measures one individual wall; a room-level aggregate cannot conceal compensating wall errors. Metric-mode `shape_dimensions` is a positive three-element vector of corresponding extents in fixed order. No arbitrary axis permutation or geometric correspondence search is performed in metric mode.

In shape-only mode, `shape_dimensions` are arbitrary-unit extents with arbitrary axis order. A single-room comparison pairs the only predicted and ground-truth room by count, so no GT room label is supplied to inference. A room must include `complete_surface_pairs: 3`, derived from three complete opposing surface pairs (six supported planes), for the provisional shape score to be eligible. This does not establish that axes represent horizontal walls and vertical height, and does not establish room orientation. Axis permutation is selected by the evaluator as an oracle correspondence for the shape diagnostic only.

## Scoring and gates

Rooms, walls, and openings are joined by stable IDs. This bounded first version does not solve geometric matching or robust correspondence under changed labels. Unmatched IDs count as missed or phantom objects. Topology, inter-room placement, and floor-to-floor drift checks are not implemented; `multiroom` and `multifloor` stages are unsupported and cannot pass overall completion, even when records are present.

Metric gates use unaligned absolute metre measurements. For every matched wall capture, relative wall-length error against ground truth must be at most 3%. Every room's absolute wall-height error must be at most 0.015 m. For every room across captures of one house/stage, predicted wall-height spread (max minus min) must be at most 0.01 m.

An opening is a hit when its stable ID exists in both files and width error is at most 0.02 m. Height error is reported diagnostically, not gated. In metric mode and each anchor protocol, the per-capture penalized opening score is hits divided by (ground-truth held-out opening count plus phantom prediction count), so missed and phantom openings both penalize it. The 85% gate requires every scored capture to meet the threshold; the pooled score remains diagnostic and cannot hide a failing capture. A capture with no ground-truth held-out openings and no predicted phantom openings is reported as `not_measured`, never as a pass; if any capture is not measured, the opening gate fails. In anchor protocols, the declared anchor opening is excluded from that protocol's held-out opening score.

Wall repeatability is gated and reported under both interpretations of “0.01 m or 0.005”: pairwise per-wall differences must be at most 0.01 m, and separately at most 0.5% of the reference wall length (`0.005 × ground-truth wall length`). For every matched wall and capture pair of a house/stage, both pass rates must be 100%. This is separate from absolute wall accuracy.

Shape-only evaluation is an explicitly named `unordered_cuboid` diagnostic. It applies one evaluator-only uniform scale to the three extents of the whole scene/run and chooses one axis permutation for that run. This scale alignment and correspondence are evaluation only and must never be supplied to reconstruction. The provisional 3% target applies to all three cuboid extents; it is not a wall-accuracy, height-accuracy, or floorplan-orientation result. Three-run proportion repeatability is reported separately. Missing runs, rooms, extents, or any opposing surface pair suppress the provisional shape pass. Metric gates are explicitly `not_evaluated`, height recovery and floorplan orientation remain false, and shape-only output always has `completion: false`.

Overall completion requires coverage, metric and repeatability gates, and supported stage checks. Since only `single_room` geometry gates exist initially, the unsupported-stage gate remains false until topology, placement, and floor-drift evaluation is added. This contract makes no claim of benchmark results; publish only actual evaluator output from real predictions and held ground truth.
