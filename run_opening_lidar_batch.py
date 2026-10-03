"""Run the seeded simulated-LiDAR opening benchmark with a prediction freeze.

The truth-free stages export capture observations, freeze all structural
candidates, select an axis from camera-centre motion, and detect depth-dropout
openings. Private scene JSON is opened only after the batch freeze marker is
written. All reported scores are diagnostics; this runner never marks a stage
complete.

Example (a single pilot path):
  python run_opening_lidar_batch.py --private-root opening_capture_seeded/private_ground_truth \
      --output-dir ../../outputs/sim-benchmark/opening-lidar-smoke --max-paths 1

Full benchmark (explicit invocation):
  python run_opening_lidar_batch.py --private-root opening_capture_seeded/private_ground_truth \
      --output-dir ../../outputs/sim-benchmark/opening-lidar-full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _discover(private_root: Path, max_paths: int | None, seeds: list[int] | None = None) -> list[dict]:
    runs = []
    for scene_dir in sorted(Path(private_root).glob("scene_*"), key=lambda p: int(p.name.removeprefix("scene_"))):
        if seeds and int(scene_dir.name.removeprefix("scene_")) not in set(seeds):
            continue
        scene_file = scene_dir / "scene.json"
        if not scene_file.is_file():
            continue
        for path_file in sorted(scene_dir.glob("path_*.json")):
            # baseline_path files are deliberately not inputs to this benchmark.
            if path_file.name.startswith("baseline_"):
                continue
            try:
                path_index = int(path_file.stem.split("_")[-1])
            except ValueError:
                continue
            runs.append({"seed": int(scene_dir.name.removeprefix("scene_")), "path_index": path_index,
                         "scene_file": scene_file, "path_file": path_file})
    runs.sort(key=lambda r: (r["seed"], r["path_index"]))
    if max_paths is not None:
        runs = runs[:max_paths]
    if not runs:
        raise FileNotFoundError(f"no scene_<seed>/scene.json and path_XX.json inputs found under {private_root}")
    return runs


def _capture(run: dict, public_dir: Path, args) -> Path:
    import capture_lidar_sim

    seed = int(run["seed"] * 100 + run["path_index"])
    capture_lidar_sim.capture(run["scene_file"], run["path_file"], public_dir,
                              width=args.width, height=args.height, fov_deg=args.fov,
                              seed=seed, depth_noise_m=args.depth_noise,
                              position_noise_m=args.position_noise,
                              rotation_noise_deg=args.rotation_noise)
    return public_dir / "prediction.npz"


def _freeze_one(run: dict, output: Path) -> dict:
    """Extract and freeze candidates, axis, height, paired walls and openings for ONE capture.

    Reads only the public prediction.npz. Resumable: a finished capture leaves
    freeze_row.json under its own frozen directory and is reloaded verbatim.
    """
    import time

    import compare_wall_boundaries as cwb
    import detect_openings_depth
    import estimate_height
    import extract_planes
    import motion_plane_axis
    import paired_wall_measurement

    frozen_root = output / "frozen" / run["label"]
    row_file = frozen_root / "freeze_row.json"
    if row_file.is_file():
        return _read(row_file)
    t0 = time.perf_counter()
    budgets = [{"label": run["label"], "prediction": str(run["prediction"]), "frames_available": run["frames"]}]
    index = cwb.run_extraction(budgets, frozen_root)
    rec = index["budgets"][0]
    item = {"label": run["label"], "seed": run["seed"], "path_index": run["path_index"],
            "prediction_sha256": _sha(run["prediction"]), "candidate_status": rec.get("status"),
            "candidates": rec.get("hypotheses", []), "selected_axis": None,
            "boundary": None, "openings": None, "paired_walls": None, "height_estimate": None,
            "frozen_index_sha256": _sha(frozen_root / "frozen_index.json")}

    def finish() -> dict:
        item["inference_seconds"] = time.perf_counter() - t0
        _write_json(row_file, item)
        return item

    if rec.get("status") != "ok":
        return finish()
    axes_ref = rec.get("files", {}).get("axes")
    planes_ref = rec.get("files", {}).get("planes")
    axes = cwb.read_frozen(frozen_root, axes_ref["path"], axes_ref["sha256"])
    with np.load(run["prediction"], allow_pickle=False) as archive:
        pred_arrays = {k: archive[k] for k in archive.files}
    choice = motion_plane_axis.select_motion_axis(pred_arrays["extrinsics"], axes.get("hypotheses", []))
    if choice.get("status") == "selected":
        choice["axis_world"] = axes["hypotheses"][choice["hypothesis_index"]]["axis_world"]
    item["selected_axis"] = choice
    if choice.get("status") != "selected":
        return finish()
    hyp = rec["hypotheses"][choice["hypothesis_index"]]
    if hyp.get("status") != "ran" or not hyp.get("files", {}).get("boundary"):
        return finish()
    boundary_ref = hyp["files"]["boundary"]
    boundary = cwb.read_frozen(frozen_root, boundary_ref["path"], boundary_ref["sha256"])
    walls_ref = hyp["files"]["walls"]
    walls = cwb.read_frozen(frozen_root, walls_ref["path"], walls_ref["sha256"])
    item["boundary"] = boundary
    item["paired_walls"] = paired_wall_measurement.estimate_paired_wall_measurements(walls, boundary)
    points = extract_planes.points_from_prediction(pred_arrays)
    height_planes = extract_planes.extract(points)
    centers = estimate_height.camera_centers(pred_arrays["extrinsics"])
    item["height_estimate"] = estimate_height.estimate_height(
        height_planes, points, np.asarray(choice["axis_world"], float), centers)
    selected_dir = frozen_root / run["label"]
    item["height_prediction_sha256"] = _write_json(selected_dir / f"{run['label']}_height.json",
                                                   item["height_estimate"])
    opening_doc = detect_openings_depth.detect(run["prediction"], frozen_root / boundary_ref["path"],
                                               frozen_root / axes_ref["path"], frozen_root / planes_ref["path"])
    item["opening_file"] = f"{run['label']}_openings.json"
    item["opening_prediction_sha256"] = _write_json(selected_dir / item["opening_file"], opening_doc)
    item["openings"] = opening_doc
    _write_json(selected_dir / f"{run['label']}_selection.json", choice)
    return finish()


def _freeze_truth_free(run_records: list[dict], output: Path) -> dict:
    """Freeze every capture's predictions, then write the batch freeze marker."""
    if (output / "prediction_freeze.json").is_file():   # resumed after the freeze was already written
        return _read(output / "prediction_freeze.json")
    rows = []
    for run in run_records:
        row = _freeze_one(run, output)
        print(f"[freeze] {run['label']} {row.get('inference_seconds', 0):.1f}s", flush=True)
        rows.append(row)
    manifest = {"schema": "opening_lidar_batch_prediction_freeze_v1", "truth_free": True,
                "ground_truth_read": False, "benchmark_stage_complete": False,
                "run_count": len(rows), "runs": rows}
    digest = _write_json(output / "prediction_freeze.json", manifest)
    (output / "prediction_freeze.sha256").write_text(digest + "\n", encoding="utf-8")
    return manifest


def _truth_opening_fraction(scene: dict, wall: int, center: float, width: float) -> tuple[float, float]:
    """Location/width fraction along a rectangular simulator wall, independent of wall naming in predictions."""
    wall_length = float(scene["width"] if int(wall) in (0, 2) else scene["length"])
    # Simulator openings store center relative to the wall centre.
    return ((center + wall_length / 2) / wall_length, width / wall_length)


def _opening_metrics(frozen_run: dict, scene: dict, wall_edge_map: dict | None) -> dict:
    """One-to-one location diagnostic: >=50% horizontal overlap and width error <=2 cm."""
    truth = scene.get("openings", [])
    pred_doc = frozen_run.get("openings") or {}
    found = pred_doc.get("openings", []) if pred_doc.get("status") == "candidates" else []
    # The selected boundary edge mapping is evaluator-oracle information and is
    # retained as an explicit flag. Without it, report all detections as unmatched.
    candidates, used = [], set()
    if wall_edge_map:
        for pi, p in enumerate(found):
            true_wall = wall_edge_map.get(int(p.get("wall_index", -1)), None)
            if true_wall is None:
                continue
            coordinate_axis = 0 if true_wall in (0, 2) else 1
            pred_interval = _predicted_world_interval(frozen_run["boundary"], p, coordinate_axis)
            pwidth = float(p.get("width_m", 0.0))
            choices = []
            for ti, t in enumerate(truth):
                if ti in used or int(t.get("wall", -1)) != true_wall:
                    continue
                width_error_m = abs(pwidth - float(t["width"]))
                truth_interval = (float(t["center"]) - float(t["width"]) / 2,
                                  float(t["center"]) + float(t["width"]) / 2)
                overlap = max(0.0, min(pred_interval[1], truth_interval[1]) - max(pred_interval[0], truth_interval[0]))
                location_ok = overlap >= .5 * min(pred_interval[1] - pred_interval[0],
                                                  truth_interval[1] - truth_interval[0])
                if width_error_m <= .02 and location_ok:
                    choices.append((width_error_m, pi, ti, pred_interval, truth_interval))
            if choices:
                candidates.extend(choices)
    matches, matched_predictions, used = [], set(), set()
    for width_error_m, pi, ti, pred_interval, truth_interval in sorted(candidates):
        if pi in matched_predictions or ti in used:
            continue
        matched_predictions.add(pi)
        used.add(ti)
        matches.append({"prediction_index": pi, "truth_index": ti, "width_error_m": width_error_m,
                        "predicted_horizontal_interval_m": list(pred_interval),
                        "truth_horizontal_interval_m": list(truth_interval),
                        "vertical_interval_diagnostic_m": found[pi].get("v_interval_from_room_axis_m")})
    missed = [i for i in range(len(truth)) if i not in used]
    phantom = [i for i in range(len(found)) if i not in {m["prediction_index"] for m in matches}]
    denominator = len(truth) + len(phantom)
    score = len(matches) / denominator if denominator else None
    return {"status": pred_doc.get("status"), "truth_openings": len(truth), "predicted_openings": len(found),
            "matches": matches, "missed_truth_indices": missed, "phantom_prediction_indices": phantom,
            "penalized_score": score, "pass_85_percent": bool(score is not None and score >= .85),
            "width_error_tolerance_m": .02, "minimum_horizontal_interval_overlap": .5,
            "oracle_wall_id_mapping_used": bool(wall_edge_map),
            "oracle_wall_id_mapping_note": "wall correspondence comes from evaluator alignment/edge matching; vertical intervals are diagnostic only"}


def _edge_world_endpoints(boundary: dict, idx: int) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(boundary["vertices_xy"], float)
    origin = np.asarray(boundary["coordinate_origin_world"], float)
    axes = boundary["axes_world"]
    pts = [origin + xy[i % len(xy), 0] * np.asarray(axes["e1"]) + xy[i % len(xy), 1] * np.asarray(axes["e2"])
           for i in (idx, idx + 1)]
    return pts[0], pts[1]


def _geometric_truth_wall(boundary: dict, idx: int, scene: dict) -> int | None:
    """Evaluator-side oracle: simulator wall whose plane the predicted edge lies on.

    Capture poses are exported in the scene world frame (x east, y north), so no
    alignment is needed: a predicted edge parallel to x lies on a north/south
    wall, one parallel to y on an east/west wall; the sign picks the side.
    Returns None when the edge is not within 0.5 m of any wall plane.
    """
    a, b = _edge_world_endpoints(boundary, idx)
    d = np.abs(b[:2] - a[:2])
    mid = (a[:2] + b[:2]) / 2
    half_w, half_l = float(scene["width"]) / 2, float(scene["length"]) / 2
    if d[0] >= d[1]:       # runs along x -> north (y=+L/2) / south
        wall, dist = (0, abs(mid[1] - half_l)) if mid[1] >= 0 else (2, abs(mid[1] + half_l))
    else:                  # runs along y -> east (x=+W/2) / west
        wall, dist = (1, abs(mid[0] - half_w)) if mid[0] >= 0 else (3, abs(mid[0] + half_w))
    return wall if dist <= .5 else None


def _predicted_world_interval(boundary: dict, opening: dict, wall_axis: int) -> tuple[float, float]:
    xy = np.asarray(boundary["vertices_xy"], float)
    idx = int(opening["wall_index"])
    origin = np.asarray(boundary["coordinate_origin_world"], float)
    axes = boundary["axes_world"]
    start = origin + xy[idx, 0] * np.asarray(axes["e1"]) + xy[idx, 1] * np.asarray(axes["e2"])
    end = origin + xy[(idx + 1) % len(xy), 0] * np.asarray(axes["e1"]) + xy[(idx + 1) % len(xy), 1] * np.asarray(axes["e2"])
    direction = (end - start) / max(float(np.linalg.norm(end - start)), 1e-12)
    lo, hi = (float(v) for v in opening["u_interval_m"])
    return tuple(sorted((float(start[wall_axis] + lo * direction[wall_axis]),
                         float(start[wall_axis] + hi * direction[wall_axis]))))


def _evaluate(freeze: dict, records: list[dict], output: Path) -> dict:
    import compare_wall_boundaries as cwb
    import evaluate_floorplan

    by_label = {r["label"]: r for r in records}
    evaluated = []
    # Private scene files are opened only after the freeze file exists and its
    # recorded hash still matches the bytes on disk.
    freeze_sha = _sha(output / "prediction_freeze.json")
    recorded = (output / "prediction_freeze.sha256").read_text(encoding="utf-8").strip()
    if freeze_sha != recorded:
        raise RuntimeError("prediction_freeze.json changed after freezing; refusing to read private truth")
    for row in freeze["runs"]:
        source = by_label[row["label"]]
        scene = _read(source["scene_file"])
        height = row.get("height_estimate")
        evaluation = None
        wall_edge_map = None
        if row.get("boundary") and height is not None:
            try:
                evaluation = evaluate_floorplan.evaluate_floorplan(cwb.eval_copy(row["boundary"]), scene, height,
                                                                     manifest={"retained_capture_fraction": 1.0})
            except Exception as exc:
                evaluation = {"status": "evaluation_error", "error": f"{type(exc).__name__}: {exc}"}
        if row.get("boundary"):
            # Evaluator-oracle geometric wall association (scene world frame == capture frame).
            wall_edge_map = {i: _geometric_truth_wall(row["boundary"], i, scene)
                             for i in range(len(row["boundary"]["vertices_xy"]))}
        # Height is assessed against the private true height; the estimate itself remains frozen.
        height_m = (height or {}).get("separation_unscaled")
        true_height = scene.get("height")
        height_error_cm = (100 * abs(float(height_m) - float(true_height))) if height_m is not None and true_height else None
        paired = row.get("paired_walls") or {}
        paired_values = paired.get("edge_lengths_m_sorted") if paired.get("status") == "paired_plane_measurement" else None
        truth_values = sorted([float(scene["width"]), float(scene["width"]),
                               float(scene["length"]), float(scene["length"])]) if "width" in scene and "length" in scene else None
        if paired_values is not None and truth_values is not None and len(paired_values) == len(truth_values):
            wall_errors = [abs(float(p) - t) for p, t in zip(paired_values, truth_values)]
            wall_rel_errors = [100 * e / max(t, 1e-9) for e, t in zip(wall_errors, truth_values)]
            paired_wall_evaluation = {"status": "scored_raw_metric", "prediction_lengths_m_sorted": paired_values,
                                      "truth_lengths_m_sorted": truth_values,
                                      "max_abs_error_cm": 100 * max(wall_errors),
                                      "max_relative_error_percent": max(wall_rel_errors),
                                      "pass_3_percent": max(wall_rel_errors) <= 3.0,
                                      "pass_1_5cm": max(wall_errors) <= .015,
                                      "oracle_sorted_dimension_correspondence": True}
        else:
            paired_wall_evaluation = {"status": paired.get("status", "unavailable"),
                                      "reason": "paired wall measurements unsupported or missing"}
        openings = _opening_metrics(row, scene, wall_edge_map)
        evaluated.append({"label": row["label"], "seed": row["seed"], "path_index": row["path_index"],
                          "selected_axis": row.get("selected_axis"), "height_estimate": height,
                          "true_height_m": true_height,
                          "height_abs_error_cm": height_error_cm,
                          "height_le_1_5cm": bool(height_error_cm is not None and height_error_cm <= 1.5),
                          "floorplan_evaluation": evaluation,
                          "oracle_normalized_floorplan_diagnostic": evaluation,
                          "paired_wall_measurement": row.get("paired_walls"),
                          "raw_paired_wall_evaluation": paired_wall_evaluation,
                          "wall_3pct_pass": paired_wall_evaluation.get("pass_3_percent", False),
                          "wall_1_5cm_pass": paired_wall_evaluation.get("pass_1_5cm", False),
                          "openings": openings,
                          "oracle_wall_id_mapping": openings["oracle_wall_id_mapping_used"]})
    # Repeatability across independent paths for the same scene; compare successful
    # estimates and selected edge lengths, without using truth to choose matches.
    repeat = _repeatability(evaluated)
    report = {"schema": "opening_lidar_batch_evaluation_v1", "prediction_freeze_sha256": _sha(output / "prediction_freeze.json"),
              "benchmark_stage_complete": False, "oracle_choices": [
                  "motion-plane axis hypothesis is chosen without truth",
                  "wall alignment and opening wall-ID association are evaluator-oracle diagnostics"],
              "gates": {"wall_relative_percent": 3.0, "height_absolute_cm": 1.5,
                        "repeatability_absolute_cm": 1.0, "repeatability_relative_percent": .5,
                        "opening_location_tolerance_cm": 2.0, "opening_per_capture_score": .85},
              "runs": evaluated, "repeatability": repeat,
              "summary": _summary(evaluated, repeat)}
    _write_json(output / "evaluation.json", report)
    return report


def _edge_to_sim_wall(scene: dict) -> dict:
    # evaluate_floorplan's rectangle is CCW from southwest: south/east/north/west.
    if not all(k in scene for k in ("width", "length")):
        return {}
    # simulator.make_scene labels: 0 north, 1 east, 2 south, 3 west.
    return {0: 2, 1: 1, 2: 0, 3: 3}


def _repeatability(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row["seed"], []).append(row)
    results = []
    for seed, group in sorted(groups.items()):
        heights = [(r.get("height_estimate") or {}).get("separation_unscaled") for r in group
                   if (r.get("height_estimate") or {}).get("separation_unscaled") is not None]
        spread_cm = 100 * (max(heights) - min(heights)) if len(heights) >= 2 and len(heights) == len(group) else None
        true_height = next((r.get("true_height_m") for r in group if r.get("true_height_m") is not None), None)
        denom_cm = 100 * float(true_height) if true_height is not None else None
        rel_pct = 100 * spread_cm / denom_cm if spread_cm is not None and denom_cm else None
        wall_vectors = [np.asarray(r["paired_wall_measurement"]["edge_lengths_m_sorted"], float)
                        for r in group if (r.get("paired_wall_measurement") or {}).get("status") =="paired_plane_measurement"]
        wall_spread_cm = (100 * float(np.max(np.ptp(np.stack(wall_vectors), axis=0)))
                          if len(wall_vectors) >= 2 and len(wall_vectors) == len(group) and len({len(v) for v in wall_vectors}) == 1 else None)
        wall_denom_cm = (100 * float(np.mean(np.stack(wall_vectors)))
                         if wall_vectors and len({len(v) for v in wall_vectors}) == 1 else None)
        wall_rel_pct = (100 * wall_spread_cm / wall_denom_cm
                        if wall_spread_cm is not None and wall_denom_cm else None)
        results.append({"seed": seed, "path_count": len(group), "height_spread_cm": spread_cm,
                        "height_spread_le_1cm": bool(spread_cm is not None and spread_cm <= 1.0),
                        "height_spread_percent": rel_pct, "height_spread_le_0_5_percent": bool(rel_pct is not None and rel_pct <= .5),
                        "wall_length_spread_cm": wall_spread_cm,
                        "wall_length_spread_le_1cm": bool(wall_spread_cm is not None and wall_spread_cm <= 1.0),
                        "wall_length_spread_percent": wall_rel_pct,
                        "wall_length_spread_le_0_5_percent": bool(wall_rel_pct is not None and wall_rel_pct <= .5),
                        "wall_repeatability_status": "measured" if wall_spread_cm is not None else "unsupported_or_missing_pair"})
    return {"per_house": results, "note": "Unmatched or unsupported paired-wall measurements are reported, never imputed."}


def _summary(rows: list[dict], repeat: dict) -> dict:
    n = len(rows)
    return {"captures": n,
            "wall_3pct_pass_count": sum(r["wall_3pct_pass"] for r in rows),
            "height_1_5cm_pass_count": sum(r["height_le_1_5cm"] for r in rows),
            "opening_85pct_pass_count": sum(r["openings"]["pass_85_percent"] for r in rows),
            "opening_scored_count": sum(r["openings"]["penalized_score"] is not None for r in rows),
            "houses_height_repeatability_le_1cm": sum(x["height_spread_le_1cm"] for x in repeat["per_house"]),
            "houses_height_repeatability_le_0_5pct": sum(x["height_spread_le_0_5_percent"] for x in repeat["per_house"]),
            "houses_wall_repeatability_le_1cm": sum(x["wall_length_spread_le_1cm"] for x in repeat["per_house"]),
            "houses_wall_repeatability_le_0_5pct": sum(x["wall_length_spread_le_0_5_percent"] for x in repeat["per_house"]),
            "stage_completion_claim": False}


def run(args) -> dict:
    private_root = args.private_root.resolve()
    output = args.output_dir.resolve()
    import time

    if output.exists() and any(output.iterdir()) and not getattr(args, "resume", False):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output} (use --resume)")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError("evaluation.json already exists; refusing to re-score into the same folder")
    runs = _discover(private_root, args.max_paths, args.seeds)
    records = []
    timing = {}
    for run in runs:
        label = f"scene{run['seed']}_path{run['path_index']:02d}"
        public_dir = output / "public" / label
        t0 = time.perf_counter()
        if args.reuse_public_root:
            root = args.reuse_public_root.resolve()
            prediction = root / "public" / label / "prediction.npz"   # a previous run folder of this script
            if not prediction.is_file():
                prediction = root / f"house_{run['seed']}" / f"capture_{run['path_index']:02d}" / "prediction.npz"
            if not prediction.is_file():
                raise FileNotFoundError(f"missing reusable public prediction: {prediction}")
        elif (public_dir / "manifest.json").is_file():   # manifest is written last: capture complete
            prediction = public_dir / "prediction.npz"
        else:
            if public_dir.exists():
                import shutil
                shutil.rmtree(public_dir)   # incomplete capture from an interrupted run of this folder
            prediction = _capture(run, public_dir, args)
        timing[label] = {"capture_seconds": time.perf_counter() - t0}
        print(f"[capture] {label} {timing[label]['capture_seconds']:.1f}s", flush=True)
        with np.load(prediction, allow_pickle=False) as archive:
            frames = int(len(archive["depth"]))
        records.append({**run, "label": label, "prediction": prediction, "frames": frames})
    freeze = _freeze_truth_free(records, output)
    for row in freeze["runs"]:
        timing[row["label"]]["inference_seconds"] = row.get("inference_seconds")
    (output / f"timing_{int(time.time())}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    return _evaluate(freeze, records, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-public-root", type=Path, help="reuse prediction.npz files under house_<seed>/capture_XX")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run in the same output dir")
    parser.add_argument("--max-paths", type=int, help="limit to first N discovered captures for a smoke run")
    parser.add_argument("--seeds", type=int, nargs="*", help="restrict to selected scene seeds")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fov", type=float, default=78.0)
    parser.add_argument("--depth-noise", type=float, default=.005)
    parser.add_argument("--position-noise", type=float, default=.005)
    parser.add_argument("--rotation-noise", type=float, default=.1)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
