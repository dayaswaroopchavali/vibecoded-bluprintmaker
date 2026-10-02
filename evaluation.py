"""Bounded evaluator for the simulation reconstruction benchmark (stdlib only).`n`nModes: metric, shape_only, height_normalized_3m (custom diagnostic) and anchor_protocols`n(three separate single-anchor scale protocols; see evaluate_anchor_protocols)."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path

STAGES = ("single_room", "multiroom", "multifloor")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def positive(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    value = float(value)
    require(math.isfinite(value) and value > 0, f"{label} must be finite and positive")
    return value


def optional_positive(value, label, prediction):
    """Allow absent estimates in predictions; malformed present values remain invalid."""
    if prediction and value is None:
        return None
    return positive(value, label)


def load_document(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def index_document(doc, prediction):
    require(isinstance(doc, dict) and doc.get("schema_version") == 1, "schema_version must be 1")
    require(isinstance(doc.get("runs"), list), "runs must be an array")
    indexed = {}
    for ri, run in enumerate(doc["runs"]):
        require(isinstance(run, dict), f"run {ri} must be an object")
        stage, house, seed = run.get("stage"), run.get("house_id"), run.get("capture_seed")
        require(stage in STAGES, f"run {ri} has unknown stage")
        require(isinstance(house, str) and house.strip(), f"run {ri} has invalid house_id")
        require(isinstance(seed, (int, str)) and not isinstance(seed, bool), f"run {ri} has invalid capture_seed")
        key = (stage, house, str(seed))
        require(key not in indexed, f"duplicate run key: {key}")
        mode = doc.get("evaluation_mode", "metric")
        if prediction and mode == "metric":
            require(isinstance(run.get("scale_source"), str) and run["scale_source"].strip(),
                    f"prediction {key} missing scale_source; metric evaluation requires scale provenance")
        require(isinstance(run.get("rooms"), list), f"run {key} rooms must be an array")
        require(bool(run["rooms"]) or prediction, f"ground-truth run {key} must have nonempty rooms")
        rooms = {}
        for room in run["rooms"]:
            require(isinstance(room, dict), f"room in {key} must be an object")
            rid = room.get("room_id")
            require(isinstance(rid, str) and rid.strip() and rid not in rooms, f"invalid/duplicate room_id in {key}")
            optional_positive(room.get("wall_height_m"), f"{key}/{rid}/wall_height_m", prediction)
            dims = room.get("shape_dimensions")
            if dims is None and (prediction or room.get("layout") == "non_cuboid_polygon"):
                pass
            else:
                require(isinstance(dims, list) and len(dims) == 3, f"{key}/{rid}/shape_dimensions must have 3 values")
                for d in dims:
                    if prediction and d is None:
                        continue
                    positive(d, f"{key}/{rid}/shape_dimensions")
            if prediction and mode in ("shape_only", "height_normalized_3m"):
                pairs = room.get("complete_surface_pairs")
                require(isinstance(pairs, int) and not isinstance(pairs, bool) and 0 <= pairs <= 3,
                        f"{key}/{rid}/complete_surface_pairs must be an integer from 0 to 3")
            require(isinstance(room.get("walls"), list), f"{key}/{rid}/walls must be an array")
            walls = {}
            for wall in room["walls"]:
                require(isinstance(wall, dict), "wall must be an object")
                wid = wall.get("wall_id")
                require(isinstance(wid, str) and wid.strip() and wid not in walls, f"invalid/duplicate wall_id in {key}/{rid}")
                optional_positive(wall.get("video_length_m"), f"wall {wid} video_length_m", prediction)
                walls[wid] = wall
            require(isinstance(room.get("openings"), list), f"{key}/{rid}/openings must be an array")
            openings = {}
            for opening in room["openings"]:
                require(isinstance(opening, dict), "opening must be an object")
                oid = opening.get("opening_id")
                require(isinstance(oid, str) and oid.strip() and oid not in openings, f"invalid/duplicate opening_id in {key}/{rid}")
                optional_positive(opening.get("width_m"), f"opening {oid} width_m", prediction)
                optional_positive(opening.get("height_m"), f"opening {oid} height_m", prediction)
                openings[oid] = opening
            room["_walls"], room["_openings"] = walls, openings
            rooms[rid] = room
        run["_rooms"] = rooms
        indexed[key] = run
    return indexed


def evaluate(prediction, groundtruth):
    """Evaluate loaded JSON objects. Raises ValueError for invalid schema or values."""
    mode = prediction.get("evaluation_mode", "metric")
    require(mode in ("metric", "shape_only", "height_normalized_3m", "anchor_protocols"),
            "evaluation_mode must be metric, shape_only, height_normalized_3m, or anchor_protocols")
    if mode == "anchor_protocols":
        require(prediction.get("prediction_units") == "arbitrary_unscaled",
                "anchor_protocols needs prediction_units == arbitrary_unscaled (raw, unscaled predictions)")
    pred = index_document(prediction, True)
    gt = index_document(groundtruth, False)
    if mode == "anchor_protocols":
        return evaluate_anchor_protocols(pred, gt)
    if mode == "shape_only":
        return evaluate_shape_only(pred, gt)
    if mode == "height_normalized_3m":
        return evaluate_height_normalized_3m(pred, gt)
    coverage, coverage_ok = {}, True
    for stage in STAGES:
        houses = sorted({h for (s, h, _seed) in pred if s == stage})
        per_house = {h: sorted(seed for (s, hh, seed) in pred if s == stage and hh == h) for h in houses}
        passed = len(houses) >= 10 and all(len(set(seeds)) >= 3 for seeds in per_house.values())
        coverage[stage] = {"distinct_houses": len(houses), "seeds_per_house": per_house, "pass": passed}
        coverage_ok &= passed
    missing, extra = sorted(set(gt) - set(pred)), sorted(set(pred) - set(gt))
    wall_errors, height_errors, opening_hits, opening_total = [], [], 0, 0
    missed_openings = phantom_openings = unmatched_rooms = unmatched_walls = 0
    missing_wall_measurements = missing_height_measurements = missing_opening_widths = 0
    shape_errors, height_by_room = [], defaultdict(list)
    wall_captures = defaultdict(list)
    for key in sorted(set(pred) & set(gt)):
        prun, grun = pred[key], gt[key]
        for rid in set(prun["_rooms"]) | set(grun["_rooms"]):
            if rid not in prun["_rooms"] or rid not in grun["_rooms"]:
                unmatched_rooms += 1
                continue
            p, g = prun["_rooms"][rid], grun["_rooms"][rid]
            if p["wall_height_m"] is None:
                missing_height_measurements += 1
            else:
                height_errors.append(abs(p["wall_height_m"] - g["wall_height_m"]))
                height_by_room[(key[0], key[1], rid)].append(p["wall_height_m"])
            for wid in set(p["_walls"]) | set(g["_walls"]):
                if wid not in p["_walls"] or wid not in g["_walls"]:
                    unmatched_walls += 1
                    continue
                pl, gl = p["_walls"][wid]["video_length_m"], g["_walls"][wid]["video_length_m"]
                if pl is None:
                    missing_wall_measurements += 1
                    continue
                wall_errors.append(abs(pl - gl) / gl)
                wall_captures[(key[0], key[1], rid, wid)].append((pl, gl))
            po, go = p["_openings"], g["_openings"]
            opening_total += len(go)
            for oid, obj in go.items():
                pred_obj = po.get(oid)
                if pred_obj and pred_obj["width_m"] is not None and abs(pred_obj["width_m"] - obj["width_m"]) <= .02:
                    opening_hits += 1
                    # Opening height is diagnostic only.
                else:
                    missed_openings += 1
                    if pred_obj is not None and pred_obj["width_m"] is None:
                        missing_opening_widths += 1
            phantom_openings += len(set(po) - set(go))
            pd, gd = p["shape_dimensions"], g["shape_dimensions"]
            if pd is not None and gd is not None:
                shape_errors.append(max(abs(pd[i] / max(pd) - gd[i] / max(gd)) for i in range(3)))
    height_spreads = {"|".join(k): max(v) - min(v) for k, v in height_by_room.items()}
    max_height_spread = max(height_spreads.values(), default=None)
    abs_repeat, rel_repeat = [], []
    for vals in wall_captures.values():
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diff = abs(vals[i][0] - vals[j][0])
                reference = (vals[i][1] + vals[j][1]) / 2
                abs_repeat.append(diff)
                rel_repeat.append(diff / reference)
    recall_score = opening_hits / (opening_total + phantom_openings) if opening_total + phantom_openings else None
    max_wall = max(wall_errors, default=None)
    max_height = max(height_errors, default=None)
    rep_1cm = sum(x <= .01 for x in abs_repeat) / len(abs_repeat) if abs_repeat else None
    rep_half_percent = sum(x <= .005 for x in rel_repeat) / len(rel_repeat) if rel_repeat else None
    # Initial evaluator has no topology, placement, or floor-drift checks for these stages.
    unsupported_stage_gate = not any((s in ("multiroom", "multifloor")) for s, _, _ in pred)
    gates = {
        "coverage": coverage_ok,
        "matched_run_coverage": not missing and not extra,
        "room_id_coverage": unmatched_rooms == 0,
        "wall_id_coverage": unmatched_walls == 0,
        "wall_length_error_le_3_percent": max_wall is not None and max_wall <= .03 and missing_wall_measurements == 0,
        "wall_height_error_le_0_015m": max_height is not None and max_height <= .015 and missing_height_measurements == 0,
        "height_spread_le_0_01m": max_height_spread is not None and max_height_spread <= .01 and missing_height_measurements == 0,
        "opening_score_ge_85_percent": recall_score is not None and recall_score >= .85,
        "repeatability_le_0_01m": rep_1cm is not None and rep_1cm == 1.0,
        "repeatability_le_0_5_percent_reference": rep_half_percent is not None and rep_half_percent == 1.0,
        "supported_stage_checks": unsupported_stage_gate,
    }
    return {
        "schema_version": 1, "completion": all(gates.values()), "coverage": coverage,
        "missing_prediction_runs": [list(k) for k in missing], "unmatched_prediction_runs": [list(k) for k in extra],
        "metric": {"max_relative_wall_length_error": max_wall, "max_wall_height_error_m": max_height,
                   "max_height_spread_across_captures_m": max_height_spread, "height_spread_by_stage_house_room_m": height_spreads,
                   "opening_hits": opening_hits, "groundtruth_openings": opening_total,
                   "missed_openings": missed_openings, "phantom_openings": phantom_openings,
                   "opening_penalized_score": recall_score, "unmatched_room_count": unmatched_rooms,
                   "unmatched_wall_count": unmatched_walls,
                   "missing_wall_measurement_count": missing_wall_measurements,
                   "missing_height_measurement_count": missing_height_measurements,
                   "missing_opening_width_count": missing_opening_widths},
        "wall_repeatability": {"pair_count": len(abs_repeat), "pass_rate_le_0_01m": rep_1cm,
                               "pass_rate_le_0_5_percent_reference": rep_half_percent},
        "per_room_normalized_shape_diagnostic_no_gate": {"room_count": len(shape_errors),
            "max_normalized_extent_error": max(shape_errors, default=None),
            "mean_normalized_extent_error": sum(shape_errors) / len(shape_errors) if shape_errors else None,
            "metric_gate": False, "does_not_test_inter_room_scale_drift": True},
        "gates": gates,
    }


def evaluate_shape_only(pred, gt):
    """Unordered single-room cuboid diagnostic; never a blueprint/metric pass."""
    import itertools
    errors, run_scales, axis_choices = [], {}, {}
    seed_ratios = defaultdict(list)
    missing, extra = sorted(set(gt) - set(pred)), sorted(set(pred) - set(gt))
    unmatched_rooms = missing_dimensions = incomplete_surface_pairs = 0
    for key in sorted(set(pred) & set(gt)):
        prun, grun = pred[key], gt[key]
        if len(prun["_rooms"]) == len(grun["_rooms"]) == 1:
            room_pairs = [(next(iter(prun["_rooms"].values())), next(iter(grun["_rooms"].values())))]
        else:
            shared = sorted(set(prun["_rooms"]) & set(grun["_rooms"]))
            room_pairs = [(prun["_rooms"][rid], grun["_rooms"][rid]) for rid in shared]
            unmatched_rooms += len(set(prun["_rooms"]) ^ set(grun["_rooms"]))
        pairs = [(p["shape_dimensions"], g["shape_dimensions"]) for p, g in room_pairs
                 if p["shape_dimensions"] is not None and all(x is not None for x in p["shape_dimensions"])
                 and g["shape_dimensions"] is not None]
        missing_dimensions += sum(p["shape_dimensions"] is None or any(x is None for x in p["shape_dimensions"]) for p, _g in room_pairs)
        incomplete_surface_pairs += sum(p.get("complete_surface_pairs", 0) != 3 for p, _g in room_pairs)
        if not pairs:
            continue
        # One permutation and one uniform scale for the full run; correspondence is evaluator-only.
        candidates = []
        for permutation in itertools.permutations(range(3)):
            pv = [p[i] for p, _g in pairs for i in permutation]
            gv = [g[i] for _p, g in pairs for i in range(3)]
            scale = sum(a * b for a, b in zip(pv, gv)) / sum(a * a for a in pv)
            residuals = [abs(scale * a - b) / b for a, b in zip(pv, gv)]
            candidates.append((sum((scale * a - b) ** 2 for a, b in zip(pv, gv)), permutation, scale, residuals))
        _loss, permutation, scale, residuals = min(candidates, key=lambda item: item[0])
        run_scales["|".join(key)] = scale
        axis_choices["|".join(key)] = list(permutation)
        errors.extend(residuals)
        for room_index, (p, _g) in enumerate(room_pairs):
            if p["shape_dimensions"] is not None:
                dims = [p["shape_dimensions"][i] for i in permutation]
                norm = max(dims)
                for axis_index, extent in enumerate(dims):
                    seed_ratios[(key[0], key[1], room_index, axis_index)].append((str(key[2]), extent / norm))
    spread = []
    for values in seed_ratios.values():
        if len(values) >= 3:
            proportions = [v for _seed, v in values]
            spread.append(max(proportions) - min(proportions))
    max_wall_error = max(errors, default=None)
    max_extent_error = max(errors, default=None)
    # Full benchmark coverage is required even for this provisional shape threshold.
    per_stage = {}
    for stage in STAGES:
        houses = {h for s, h, _seed in pred if s == stage}
        per_stage[stage] = len(houses) >= 10 and all(
            len({seed for s, hh, seed in pred if s == stage and hh == house}) >= 3 for house in houses
        )
    complete_runs = not missing and not extra
    complete_geometry = unmatched_rooms == 0 and missing_dimensions == 0 and incomplete_surface_pairs == 0
    data_complete = complete_runs and complete_geometry and all(per_stage.values())
    return {
        "schema_version": 1,
        "evaluation_mode": "shape_only",
        "completion": False,
        "missing_prediction_runs": [list(k) for k in missing],
        "unmatched_prediction_runs": [list(k) for k in extra],
        "metric_gates": "not_evaluated",
        "shape_only": {
            "diagnostic": "unordered_cuboid; axes may be permuted by evaluator; does not establish up, wall labels, or floorplan orientation",
            "scale_fit": "one global uniform least-squares scale per whole scene/run; evaluation only",
            "oracle_axis_permutation_by_run": axis_choices,
            "global_scale_by_run": run_scales,
            "max_cuboid_extent_relative_error": max_extent_error,
            "provisional_cuboid_shape_gate_le_3_percent": data_complete and max_extent_error is not None and max_extent_error <= .03,
            "max_scaled_extent_ratio_error": max_extent_error,
            "three_run_shape_ratio_repeatability_max_range": max(spread, default=None),
            "three_run_wall_proportion_repeatability_values": spread,
            "stage_coverage": per_stage,
            "matched_run_coverage": complete_runs,
            "unmatched_room_count": unmatched_rooms,
            "missing_shape_dimension_count": missing_dimensions,
            "incomplete_surface_pair_room_count": incomplete_surface_pairs,
            "data_complete_for_provisional_gate": data_complete,
            "height_recovered": False,
            "floorplan_orientation_solved": False,
            "provisional_only": True,
        },
    }


def evaluate_height_normalized_3m(pred, gt):
    """Compare shape after evaluator-only normalization to an assumed 3 m height."""
    import itertools
    missing, extra = sorted(set(gt) - set(pred)), sorted(set(pred) - set(gt))
    per_run, max_horizontal_error = [], None
    unmatched_rooms = incomplete_support = 0
    horizontal_errors = []
    for key in sorted(set(pred) & set(gt)):
        prun, grun = pred[key], gt[key]
        if len(prun["_rooms"]) == len(grun["_rooms"]) == 1:
            p = next(iter(prun["_rooms"].values()))
            g = next(iter(grun["_rooms"].values()))
            room_pairs = [(p, g)]
        else:
            shared = sorted(set(prun["_rooms"]) & set(grun["_rooms"]))
            room_pairs = [(prun["_rooms"][rid], grun["_rooms"][rid]) for rid in shared]
            unmatched_rooms += len(set(prun["_rooms"]) ^ set(grun["_rooms"]))
        for p, g in room_pairs:
            pd, gd = p["shape_dimensions"], g["shape_dimensions"]
            if pd is None or any(x is None for x in pd) or gd is None:
                incomplete_support += 1
                continue
            if p.get("complete_surface_pairs", 0) != 3:
                incomplete_support += 1
            supplied_height_axis = p.get("height_axis_index")
            if supplied_height_axis is not None:
                require(isinstance(supplied_height_axis, int) and not isinstance(supplied_height_axis, bool)
                        and supplied_height_axis in (0, 1, 2), f"{key} height_axis_index must be 0, 1, or 2")
                candidate_height_axes = [supplied_height_axis]
                mapping_kind = "independently_predicted_height_axis"
            else:
                candidate_height_axes = [0, 1, 2]
                mapping_kind = "oracle_height_axis_and_horizontal_swap"
            options = []
            for hidx in candidate_height_axes:
                ph, gh = pd[hidx], gd[2]
                pn = [3.0 * x / ph for x in pd]
                gn = [3.0 * x / gh for x in gd]
                pred_horizontal = [pn[i] for i in range(3) if i != hidx]
                for gt_order in ((0, 1), (1, 0)):
                    errs = [abs(pred_horizontal[i] - gn[gt_order[i]]) for i in range(2)]
                    rels = [errs[i] / gn[gt_order[i]] for i in range(2)]
                    options.append((max(errs), hidx, gt_order, errs, rels, pn, gn))
            _maxerr, hidx, gt_order, errs, rels, pn, gn = min(options, key=lambda o: o[0])
            result = {"run": list(key), "axis_matching": mapping_kind,
                      "predicted_height_axis_index": hidx, "groundtruth_horizontal_axis_order": list(gt_order),
                      "predicted_dimensions_normalized_to_3m": pn,
                      "groundtruth_dimensions_normalized_to_3m": gn,
                      "horizontal_absolute_errors_m": errs,
                      "horizontal_absolute_errors_cm": [100.0 * x for x in errs],
                      "horizontal_relative_errors": rels,
                      "height_normalization_forced": True,
                      "custom_1_5cm_pass": max(errs) <= .015}
            per_run.append(result)
            horizontal_errors.extend(errs)
    max_horizontal_error = max(horizontal_errors, default=None)
    per_stage = {}
    for stage in STAGES:
        houses = {h for s, h, _seed in pred if s == stage}
        per_stage[stage] = len(houses) >= 10 and all(
            len({seed for s, hh, seed in pred if s == stage and hh == house}) >= 3 for house in houses
        )
    complete = (not missing and not extra and unmatched_rooms == 0 and incomplete_support == 0
                and all(per_stage.values()) and len(per_run) == len(pred))
    return {"schema_version": 1, "evaluation_mode": "height_normalized_3m", "completion": False,
            "metric_gates": "not_evaluated", "official_height_accuracy": "not_evaluated_height_forced_to_3m",
            "custom_protocol": "normalize prediction and GT independently to 3.0m height, then compare horizontal extents",
            "height_normalized_3m": {"per_run": per_run,
                "max_horizontal_absolute_error_m": max_horizontal_error,
                "max_horizontal_absolute_error_cm": 100.0 * max_horizontal_error if max_horizontal_error is not None else None,
                "max_horizontal_relative_error": max((r for result in per_run for r in result["horizontal_relative_errors"]), default=None),
                "tolerance_m": .015,
                "full_dataset_custom_gate_le_1_5cm": complete and max_horizontal_error is not None and max_horizontal_error <= .015,
                "stage_coverage": per_stage, "matched_run_coverage": not missing and not extra,
                "missing_prediction_runs": [list(k) for k in missing],
                "unmatched_prediction_runs": [list(k) for k in extra],
                "unmatched_room_count": unmatched_rooms,
                "incomplete_support_room_count": incomplete_support,
                "height_recovered": False if any(r["axis_matching"] == "oracle_height_axis_and_horizontal_swap" for r in per_run) else True,
                "floorplan_orientation_solved": False,
                "height_axis_oracle_used": any(r["axis_matching"] == "oracle_height_axis_and_horizontal_swap" for r in per_run),
                "provisional_only": True}}


CEILING_ANCHOR_M = 3.0
ANCHOR_VALUE_TOLERANCE_M = 1e-3
ANCHOR_PROTOCOLS = ("ceiling_height_3m", "known_horizontal_wall", "known_door_width")
COMPLETE_LAYOUTS = ("complete_polygon_hypothesis", "complete_box_hypothesis")
CUSTOM_DIAGNOSTIC_TOLERANCE_M = 0.015
ANCHOR_REASONS = {"anchor_missing", "anchor_malformed", "ambiguous_anchor", "anchor_not_in_prediction",
                  "anchor_measurement_unavailable", "anchor_not_in_groundtruth", "anchor_value_conflicts_with_groundtruth",
                  "height_anchor_unavailable", "ceiling_anchor_must_be_3_0_m",
                  "ceiling_anchor_not_applicable_groundtruth_height_differs_from_3_0_m",
                  "wall_height_disagrees_with_declared_height_axis"}


def _num(value):
    """Positive finite number or None."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    return float(value) if ok else None


def _predicted_height(room):
    """Unscaled predicted height from the declared height axis / wall_height_m; never from ground truth."""
    dims, axis, height = room.get("shape_dimensions"), room.get("height_axis_index"), _num(room.get("wall_height_m"))
    if isinstance(dims, list) and len(dims) == 3 and isinstance(axis, int) and not isinstance(axis, bool) and axis in (0, 1, 2):
        axis_value = _num(dims[axis])
        if height is None:
            return axis_value, None
        if axis_value is not None and abs(axis_value - height) > 1e-6 * max(height, 1.0):
            return None, "wall_height_disagrees_with_declared_height_axis"
    return height, None


def _anchor_scale(protocol, run, p, g):
    """The single uniform scale for one prediction, from a declared anchor only.

    Ground truth is consulted solely to reject an anchor whose declared value contradicts the scene;
    it never contributes to the scale. Returns (scale, info, excluded, reasons).
    """
    anchors = run.get("anchors")
    if protocol == "ceiling_height_3m":
        if isinstance(anchors, dict) and "ceiling_height_m" in anchors and _num(anchors["ceiling_height_m"]) != CEILING_ANCHOR_M:
            return None, None, {}, ["ceiling_anchor_must_be_3_0_m"]
        height, problem = _predicted_height(p)
        if problem:
            return None, None, {}, [problem]
        if height is None:
            return None, None, {}, ["height_anchor_unavailable"]
        info = {"kind": "ceiling_height", "declared_m": CEILING_ANCHOR_M, "predicted_unscaled": height}
        gt_height = _num(g.get("wall_height_m"))
        if gt_height is None or abs(gt_height - CEILING_ANCHOR_M) > ANCHOR_VALUE_TOLERANCE_M:
            return None, info, {}, ["ceiling_anchor_not_applicable_groundtruth_height_differs_from_3_0_m"]
        return CEILING_ANCHOR_M / height, info, {"height": True}, []
    key, id_key, field, kind = (("wall", "wall_id", "video_length_m", "known_horizontal_wall") if protocol == "known_horizontal_wall"
                                else ("door", "opening_id", "width_m", "known_door_width"))
    value_key = "length_m" if key == "wall" else "width_m"
    if not isinstance(anchors, dict) or key not in anchors:
        return None, None, {}, ["anchor_missing"]
    if not isinstance(anchors[key], dict) or any(k in anchors for k in (key + "s", "walls", "doors") if k != key):
        return None, None, {}, ["ambiguous_anchor"]
    entry = anchors[key]; ident, declared = entry.get(id_key), _num(entry.get(value_key))
    if not isinstance(ident, str) or not ident.strip() or declared is None:
        return None, None, {}, ["anchor_malformed"]
    info = {"kind": kind, "id": ident, "declared_m": declared, "source": entry.get("source")}
    table_p, table_g = (p["_walls"], g["_walls"]) if key == "wall" else (p["_openings"], g["_openings"])
    if ident not in table_p:
        return None, info, {}, ["anchor_not_in_prediction"]
    measured = _num(table_p[ident].get(field if key == "wall" else "width_m"))
    if measured is None:
        return None, info, {}, ["anchor_measurement_unavailable"]
    info["predicted_unscaled"] = measured
    if ident not in table_g:
        return None, info, {}, ["anchor_not_in_groundtruth"]
    truth = _num(table_g[ident].get(field if key == "wall" else "width_m"))
    if truth is None or abs(truth - declared) > ANCHOR_VALUE_TOLERANCE_M:
        return None, info, {}, ["anchor_value_conflicts_with_groundtruth"]
    return declared / measured, info, {key: ident}, []


def _score_anchor_run(protocol, key, run, grun):
    out = {"run": list(key), "status": "failed", "failure_reasons": [], "scale": None, "anchor": None,
           "scale_provenance": {"source": "declared_anchor", "groundtruth_used_for_scale": False,
                                "groundtruth_used_only_to_reject_contradicting_anchor": True},
           "walls": [], "height": None, "opening_hits": 0, "opening_total": 0, "opening_phantoms": 0, "scored_wall_ids": []}
    reasons = out["failure_reasons"]
    if len(run["_rooms"]) != 1 or len(grun["_rooms"]) != 1:
        reasons.append("single_room_pairing_requires_exactly_one_room_each")
        return out
    p, g = next(iter(run["_rooms"].values())), next(iter(grun["_rooms"].values()))
    if p.get("layout_status") not in COMPLETE_LAYOUTS:
        reasons.append("incomplete_layout")
    if p.get("up_source") != "camera_motion_prior":
        reasons.append("up_direction_not_declared_camera_motion_prior")
    if run.get("scale_source") is not None:
        reasons.append("prediction_already_carries_scale_source")
    scale, info, excluded, anchor_reasons = _anchor_scale(protocol, run, p, g)
    out["anchor"] = info; reasons.extend(anchor_reasons)
    if reasons:
        return out
    out["scale"] = scale
    skipped_wall, skipped_opening = excluded.get("wall"), excluded.get("door")
    for wid in sorted(set(p["_walls"]) | set(g["_walls"])):
        if wid == skipped_wall:
            continue
        if wid not in p["_walls"] or wid not in g["_walls"]:
            reasons.append("wall_ids_unmatched"); continue
        length = _num(p["_walls"][wid].get("video_length_m"))
        if length is None:
            reasons.append("wall_length_missing"); continue
        out["walls"].append((wid, scale * length, float(g["_walls"][wid]["video_length_m"])))
        out["scored_wall_ids"].append(wid)
    if not excluded.get("height"):
        height, problem = _predicted_height(p)
        if problem or height is None:
            reasons.append(problem or "height_unavailable")
        else:
            out["height"] = (scale * height, float(g["wall_height_m"]))
    for oid, truth in g["_openings"].items():
        if oid == skipped_opening:
            continue
        out["opening_total"] += 1
        got = p["_openings"].get(oid)
        width = _num(got.get("width_m")) if got else None
        out["opening_hits"] += int(width is not None and abs(scale * width - float(truth["width_m"])) <= .02)
    out["opening_phantoms"] = len([oid for oid in p["_openings"] if oid not in g["_openings"] and oid != skipped_opening])
    if not reasons:
        out["status"] = "scored"
    return out


def _oracle_axis_diagnostic(key, p, g):
    """Evaluator-only: which axis would match ground truth. Never feeds any pass/fail result."""
    dims, gdims = p.get("shape_dimensions"), g.get("shape_dimensions")
    if not (isinstance(dims, list) and isinstance(gdims, list) and len(dims) == len(gdims) == 3
            and all(_num(x) is not None for x in dims + gdims)):
        return None
    best = None
    for axis in range(3):
        norm = [3.0 * x / dims[axis] for x in dims]; target = [3.0 * x / gdims[2] for x in gdims]
        horizontal = [norm[i] for i in range(3) if i != axis]
        error = min(max(abs(horizontal[0] - target[a]), abs(horizontal[1] - target[b])) for a, b in ((0, 1), (1, 0)))
        if best is None or error < best[0]:
            best = (error, axis)
    declared = p.get("height_axis_index")
    smallest = min(range(3), key=lambda i: dims[i])
    return {"run": list(key), "oracle_height_axis_index": best[1], "predicted_height_axis_index": declared,
            "predicted_axis_matches_oracle": declared == best[1],
            "smallest_extent_axis_index": smallest, "smallest_extent_experiment_matches_oracle": smallest == best[1],
            "smallest_extent_experiment_is_independent_recovery": False,
            "used_for_pass_results": False, "independently_recovered": False}


def evaluate_anchor_protocols(pred, gt):
    """Three separate single-anchor protocols; each applies one uniform scale to the whole prediction.

    Predictions are raw and unscaled. The anchor is excluded from scored dimensions, a missing or
    ambiguous anchor fails only its own protocol, and official gates stay apart from the custom
    1.5 cm diagnostic. Orientation must be the declared camera-motion prior; oracle axis matching
    is reported separately and never changes a pass result. Never claims the benchmark complete.
    """
    missing, extra = sorted(set(gt) - set(pred)), sorted(set(pred) - set(gt))
    single = {k: v for k, v in pred.items() if k[0] == "single_room"}
    houses = {}
    for stage, house, seed in single:
        houses.setdefault(house, set()).add(seed)
    coverage = len(houses) >= 10 and all(len(seeds) >= 3 for seeds in houses.values())
    other_stage_runs = sorted(k for k in pred if k[0] != "single_room")
    protocols = {}
    for protocol in ANCHOR_PROTOCOLS:
        results = [_score_anchor_run(protocol, key, pred[key], gt[key]) for key in sorted(single) if key in gt]
        scored = [r for r in results if r["status"] == "scored"]
        reasons = [reason for r in results for reason in r["failure_reasons"]]
        counts = {reason: reasons.count(reason) for reason in sorted(set(reasons))}
        wall_errors = [abs(pl - gl) / gl for r in scored for _w, pl, gl in r["walls"]]
        abs_errors = [abs(pl - gl) for r in scored for _w, pl, gl in r["walls"]]
        height_errors = [abs(r["height"][0] - r["height"][1]) for r in scored if r["height"]]
        by_house = {}
        for r in scored:
            if r["height"]:
                by_house.setdefault(r["run"][1], []).append(r["height"][0])
        spread = max((max(v) - min(v) for v in by_house.values()), default=None)
        hits, total, phantoms = (sum(r[k] for r in scored) for k in ("opening_hits", "opening_total", "opening_phantoms"))
        opening_score = hits / (total + phantoms) if total + phantoms else None
        captures = {}
        for r in scored:
            for wid, pl, gl in r["walls"]:
                captures.setdefault((r["run"][1], wid), []).append((pl, gl))
        abs_repeat, rel_repeat = [], []
        for values in captures.values():
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    diff = abs(values[i][0] - values[j][0]); abs_repeat.append(diff)
                    rel_repeat.append(diff / ((values[i][1] + values[j][1]) / 2))
        all_scored = bool(results) and len(scored) == len(results)
        height_scored = protocol != "ceiling_height_3m"
        gates = {
            "coverage_single_room": coverage,
            "matched_run_coverage": not missing and not extra,
            "supported_stage_checks": not other_stage_runs,
            "anchors_valid_all_runs": bool(results) and not any(r in ANCHOR_REASONS for r in reasons),
            "declared_up_prior_all_runs": bool(results) and "up_direction_not_declared_camera_motion_prior" not in reasons,
            "complete_layout_all_runs": bool(results) and "incomplete_layout" not in reasons,
            "all_runs_scorable": all_scored,
            "wall_length_error_le_3_percent": all_scored and bool(wall_errors) and max(wall_errors) <= .03,
            "wall_height_error_le_0_015m": (all_scored and bool(height_errors) and max(height_errors) <= .015) if height_scored else None,
            "height_spread_le_0_01m": (all_scored and spread is not None and spread <= .01) if height_scored else None,
            "opening_score_ge_85_percent": all_scored and opening_score is not None and opening_score >= .85,
            "repeatability_le_0_01m": all_scored and bool(abs_repeat) and all(x <= .01 for x in abs_repeat),
            "repeatability_le_0_5_percent_reference": all_scored and bool(rel_repeat) and all(x <= .005 for x in rel_repeat),
        }
        max_abs = max(abs_errors, default=None)
        protocols[protocol] = {
            "scale_rule": {"ceiling_height_3m": "scale = 3.0 m / predicted ceiling height (fixed declared anchor)",
                           "known_horizontal_wall": "scale = declared known wall length / that wall's predicted length",
                           "known_door_width": "scale = declared known door width / that door's predicted width"}[protocol],
            "exactly_one_uniform_scale_per_prediction": True,
            "anchor_excluded_from_scored_dimensions": {"ceiling_height_3m": "ceiling height",
                "known_horizontal_wall": "the anchor wall", "known_door_width": "the anchor opening"}[protocol],
            "official_gates": gates,
            "official_gates_pass": all(v for v in gates.values() if v is not None),
            "official_gate_notes": ({"wall_height_error_le_0_015m": "not scored: height is the anchor",
                                     "height_spread_le_0_01m": "not scored: height is the anchor"} if not height_scored else {}),
            "metrics": {"run_count": len(results), "scored_run_count": len(scored),
                        "max_relative_wall_length_error": max(wall_errors, default=None),
                        "max_wall_height_error_m": max(height_errors, default=None),
                        "height_spread_across_captures_m": spread, "opening_hits": hits, "opening_total": total,
                        "opening_phantoms": phantoms, "opening_penalized_score": opening_score,
                        "wall_repeatability_pair_count": len(abs_repeat),
                        "wall_repeatability_pass_rate_le_0_01m": (sum(x <= .01 for x in abs_repeat) / len(abs_repeat)) if abs_repeat else None,
                        "wall_repeatability_pass_rate_le_0_5_percent": (sum(x <= .005 for x in rel_repeat) / len(rel_repeat)) if rel_repeat else None},
            "custom_1_5cm_diagnostic": {"official_gate": False, "tolerance_m": CUSTOM_DIAGNOSTIC_TOLERANCE_M,
                "max_scored_wall_absolute_error_cm": 100.0 * max_abs if max_abs is not None else None,
                "pass_all_scored_walls": all_scored and max_abs is not None and max_abs <= CUSTOM_DIAGNOSTIC_TOLERANCE_M,
                "note": "user-defined diagnostic over scored wall lengths; separate from the assessment gates"},
            "failure_reason_counts": counts,
            "runs": [{"run": r["run"], "status": r["status"], "failure_reasons": r["failure_reasons"], "scale": r["scale"],
                      "anchor": r["anchor"], "scale_provenance": r["scale_provenance"],
                      "scored_wall_ids": r["scored_wall_ids"]} for r in results],
        }
    oracle = []
    for key in sorted(single):
        if key in gt and len(pred[key]["_rooms"]) == 1 and len(gt[key]["_rooms"]) == 1:
            entry = _oracle_axis_diagnostic(key, next(iter(pred[key]["_rooms"].values())), next(iter(gt[key]["_rooms"].values())))
            if entry:
                oracle.append(entry)
    declared = sum(1 for r in single.values() for room in r["_rooms"].values() if room.get("up_source") == "camera_motion_prior")
    return {
        "schema_version": 1, "evaluation_mode": "anchor_protocols", "completion": False,
        "benchmark_claim": "none: single_room diagnostic protocols only; multiroom and multifloor are unsupported",
        "prediction_units": "arbitrary_unscaled",
        "protocols": protocols,
        "orientation": {"required_source": "camera_motion_prior", "oracle_axis_used_in_pass_results": False,
                        "runs_declaring_camera_motion_prior": declared, "single_room_runs": len(single)},
        "oracle_diagnostics": {"note": "evaluator-only ground-truth comparisons; not recovered orientation and not used for any pass",
                               "used_for_pass_results": False, "independently_recovered": False, "per_run": oracle},
        "missing_prediction_runs": [list(k) for k in missing], "unmatched_prediction_runs": [list(k) for k in extra],
        "unsupported_stage_runs": [list(k) for k in other_stage_runs],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("groundtruth", type=Path, help="restricted evaluation input; keep outside reconstruction environment")
    args = parser.parse_args()
    result = evaluate(load_document(args.prediction), load_document(args.groundtruth))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["completion"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
