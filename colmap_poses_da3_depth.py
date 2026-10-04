"""COLMAP poses + frozen DA3 depth (post-hoc method development on already-seen captures; NOT a held-out result).

Hypothesis: classical SfM (COLMAP) gives more reliable camera poses than whole-video DA3 on the textured (realistic) captures.
Pipeline per capture (stages 1-4 are TRUTH-FREE: they read only the video and the frozen DA3 prediction):

  1 colmap  : the 120 video frames (same cv2 decoding and indices as infer_rgb.py, 480x360) -> SIFT (CPU, one shared SIMPLE_PINHOLE
              camera) -> exhaustive matcher (CPU) -> mapper, exactly the invocation of compare_backends.run_colmap.
  2 build   : largest COLMAP model; for every registered frame the frozen DA3 depth map is multiplied by ONE scale, the robust median
              of (COLMAP depth of a sparse point) / (DA3 depth at that point's pixel) over the points that frame observes. Frames with
              too few points are refused and dropped. Poses = COLMAP world-to-camera, intrinsics = COLMAP's (rescaled to the DA3 map).
              Output prediction.npz has the DA3 schema; source_indices lists exactly the frames kept. COLMAP's scale is arbitrary;
              the 3 m ceiling-normalisation protocol absorbs it (no true scale is used anywhere).
  3 chain   : run_axis_candidates.run_extraction (same truth-free chain and k as run_rgb_batch.py).
  4 freeze  : sha256 of video, DA3 prediction, COLMAP model, new prediction and chain outputs, written before any truth is read.
  evaluate  : EVALUATION ONLY, separate command. Verifies every freeze, then scores poses (Sim(3) alignment, audit_camera_trajectory via
              compare_backends.score_pose_set) and the chain candidates with run_rgb_batch's candidate/run-level logic (gates unchanged,
              a tie at the top = failure).

  python colmap_poses_da3_depth.py run      --source-root <batch> --output-dir <NEW dir> [--houses 17] [--paths 0] [--max-hours 2.5]
  python colmap_poses_da3_depth.py evaluate --output-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_COLMAP = REPO / "outputs" / "photogrammetry" / "tools" / "colmap" / "bin" / "colmap.exe"
MIN_POINTS = 30                 # a frame needs at least this many usable sparse points to get a depth scale (fixed before any run)
RAY_K = 2.0                     # unchanged from run_rgb_batch
STAGES = ("colmap", "build", "chain", "freeze")
TRUTH_FREE_FUNCTIONS = ("align_depth_scale", "depth_pixels", "sparse_points_in_frame", "build_prediction", "write_prediction",
                        "extract_video_frames", "stage_colmap", "stage_build", "stage_chain", "stage_freeze", "run_truth_free_stages",
                        "verify_run_freeze", "parse_sparse_model", "camera_intrinsics")
FORBIDDEN_TOKENS = ("private_ground_truth", "scene", "evaluate_floorplan", "oracle", "truth", "load_scene", "simulator",
                    "audit_camera_trajectory", "score_pose_set")


# ------------------------------------------------------------------------------------------------ helpers
def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return sha256_file(path)


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ------------------------------------------------------------------------------------------------ depth-scale alignment (pure)
def depth_pixels(xy, image_wh, depth_wh):
    """COLMAP pixel coordinates (centre of the top-left pixel = 0.5, 0.5, in the image the model was built from) -> integer
    (column, row) of the DA3 depth map, which covers the same field of view at another resolution."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    sx, sy = depth_wh[0] / float(image_wh[0]), depth_wh[1] / float(image_wh[1])
    cols = np.clip(np.floor(xy[:, 0] * sx).astype(int), 0, depth_wh[0] - 1)
    rows = np.clip(np.floor(xy[:, 1] * sy).astype(int), 0, depth_wh[1] - 1)
    return cols, rows


def align_depth_scale(depth, cols, rows, z, *, min_points: int = MIN_POINTS) -> dict:
    """Scale s with COLMAP_depth ~= s * DA3_depth: the median of z / depth[rows, cols] over valid track pixels (robust to outliers).

    Refuses (status 'refused_too_few_points') if fewer than `min_points` valid ratios exist; never guesses a scale."""
    depth = np.asarray(depth, float)
    z = np.asarray(z, float).ravel()
    d = depth[np.asarray(rows, int), np.asarray(cols, int)] if z.size else np.zeros(0)
    valid = np.isfinite(d) & (d > 0) & np.isfinite(z) & (z > 0)
    n_valid = int(valid.sum())
    out = {"n_points_offered": int(z.size), "n_valid": n_valid, "min_points": int(min_points)}
    if n_valid < max(int(min_points), 1):
        return {**out, "status": "refused_too_few_points", "scale": None, "rel_mad": None}
    ratios = z[valid] / d[valid]
    med = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - med)))
    if not np.isfinite(med) or med <= 0:
        return {**out, "status": "refused_non_positive_scale", "scale": None, "rel_mad": None}
    return {**out, "status": "ok", "scale": med, "rel_mad": mad / med}


# ------------------------------------------------------------------------------------------------ COLMAP text model parsing
def _qvec_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def parse_sparse_model(model_dir) -> dict:
    """COLMAP TXT model -> cameras, images keyed by source index (name stem), 3D points. Keeps 2D observations with their point ids."""
    model_dir = Path(model_dir)
    cams = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            t = line.split()
            cams[int(t[0])] = {"model": t[1], "width": int(t[2]), "height": int(t[3]), "params": [float(v) for v in t[4:]]}
    lines = [ln for ln in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    images = {}
    for k in range(0, len(lines) - 1, 2):
        t = lines[k].split()
        if len(t) < 10:
            continue
        rot = _qvec_rot([float(v) for v in t[1:5]])
        w2c = np.concatenate([rot, np.asarray([[float(v) for v in t[5:8]]]).T], axis=1)
        obs = np.asarray(lines[k + 1].split(), float).reshape(-1, 3) if lines[k + 1].strip() else np.zeros((0, 3))
        stem = Path(t[9]).stem
        if stem.isdigit():
            images[int(stem)] = {"name": t[9], "camera_id": int(t[8]), "w2c": w2c, "xy": obs[:, :2], "pid": obs[:, 2].astype(np.int64)}
    points, errors, tracks = {}, [], []
    pfile = model_dir / "points3D.txt"
    if pfile.exists():
        for line in pfile.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                t = line.split()
                points[int(t[0])] = np.asarray([float(v) for v in t[1:4]])
                errors.append(float(t[7]))
                tracks.append((len(t) - 8) // 2)
    errors, tracks = np.asarray(errors, float), np.asarray(tracks, float)
    obs_n = float(tracks.sum()) if len(tracks) else 0.0
    return {"cameras": cams, "images": images, "points": points,
            "stats": {"registered_images": len(images), "num_points3D": len(points), "observations": int(obs_n),
                      "mean_reprojection_error_px": float((errors * tracks).sum() / obs_n) if obs_n else None,
                      "mean_track_length": float(tracks.mean()) if len(tracks) else None}}


def camera_intrinsics(camera: dict, depth_wh) -> np.ndarray:
    """3x3 K of a COLMAP SIMPLE_PINHOLE / PINHOLE camera, rescaled to the DA3 depth-map resolution."""
    p, model = camera["params"], camera["model"]
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    sx, sy = depth_wh[0] / float(camera["width"]), depth_wh[1] / float(camera["height"])
    return np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]])


def sparse_points_in_frame(image: dict, points: dict):
    """(xy, z): pixel positions and COLMAP camera-frame depths of the 3D points this registered image observes."""
    keep = [k for k, pid in enumerate(image["pid"]) if int(pid) in points]
    if not keep:
        return np.zeros((0, 2)), np.zeros(0)
    xyz = np.stack([points[int(image["pid"][k])] for k in keep])
    cam = xyz @ image["w2c"][:, :3].T + image["w2c"][:, 3]
    return image["xy"][keep], cam[:, 2]


# ------------------------------------------------------------------------------------------------ prediction builder
def build_prediction(model: dict, da3: dict, *, min_points: int = MIN_POINTS) -> tuple[dict, dict]:
    """COLMAP poses + intrinsics with the frozen DA3 depth scaled per frame onto the COLMAP sparse points.

    `da3`: dict with depth (N,H,W), confidence, images, source_indices (the frozen DA3 prediction). Returns (arrays, report); the arrays
    carry ONLY registered frames that got a scale, source_indices lists them, the report names every dropped frame and why."""
    src = [int(i) for i in np.asarray(da3["source_indices"])]
    depth = np.asarray(da3["depth"], np.float32)
    h, w = depth.shape[1:]
    keep, scales, frames = [], [], []
    for k, si in enumerate(src):
        im = model["images"].get(si)
        if im is None:
            frames.append({"source_index": si, "status": "dropped_unregistered"})
            continue
        cam = model["cameras"][im["camera_id"]]
        xy, z = sparse_points_in_frame(im, model["points"])
        cols, rows = depth_pixels(xy, (cam["width"], cam["height"]), (w, h))
        al = align_depth_scale(depth[k], cols, rows, z, min_points=min_points)
        frames.append({**al, "source_index": si, "alignment_status": al["status"], "status": "kept" if al["status"] == "ok" else "dropped_" + al["status"]})
        if al["status"] == "ok":
            keep.append(k)
            scales.append(al["scale"])
    order = sorted(range(len(keep)), key=lambda j: src[keep[j]])
    keep = [keep[j] for j in order]
    scales = [scales[j] for j in order]
    k0 = np.asarray(keep, int)
    arrays = {"depth": (depth[k0] * np.asarray(scales, np.float32)[:, None, None]) if len(keep) else np.zeros((0, h, w), np.float32),
              "confidence": np.asarray(da3["confidence"], np.float32)[k0],
              "intrinsics": np.stack([camera_intrinsics(model["cameras"][model["images"][src[k]]["camera_id"]], (w, h)) for k in keep]).astype(np.float32)
              if keep else np.zeros((0, 3, 3), np.float32),
              "extrinsics": np.stack([model["images"][src[k]]["w2c"] for k in keep]).astype(np.float32) if keep else np.zeros((0, 3, 4), np.float32),
              "images": np.asarray(da3["images"])[k0],
              "source_indices": np.asarray([src[k] for k in keep], np.int64)}
    n_total = len(src)
    sc = np.asarray(scales, float)
    report = {"frames_in_da3_prediction": n_total, "frames_kept": len(keep),
              "frames_dropped_unregistered": sum(f["status"] == "dropped_unregistered" for f in frames),
              "frames_dropped_too_few_points": sum(f["status"] == "dropped_refused_too_few_points" for f in frames),
              "min_points": int(min_points), "depth_map_wh": [int(w), int(h)],
              "scale_median": float(np.median(sc)) if len(sc) else None,
              "scale_spread_p10_p90_ratio": float(np.percentile(sc, 90) / np.percentile(sc, 10)) if len(sc) else None,
              "frames": frames}
    return arrays, report


def write_prediction(path, arrays: dict) -> str:
    """Deterministic npz (fixed key order, uncompressed) with the DA3 prediction schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, depth=arrays["depth"], confidence=arrays["confidence"], extrinsics=arrays["extrinsics"],
             intrinsics=arrays["intrinsics"], images=arrays["images"], source_indices=arrays["source_indices"])
    return sha256_file(path)


# ------------------------------------------------------------------------------------------------ frames from the video (as infer_rgb.py)
def extract_video_frames(video: Path, out_dir: Path, wanted: int = 120) -> list:
    """Decode the same frames infer_rgb.py feeds DA3 (cv2, np.linspace over the frame count) and save them as <video-frame-index>.png."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or not cap.isOpened():
            raise ValueError(f"cannot read video {video}")
        indices = np.linspace(0, total - 1, min(total, wanted), dtype=int).tolist() if min(total, wanted) > 1 else [0]
        out_dir.mkdir(parents=True, exist_ok=False)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"cannot decode frame {i}")
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), bgr)
        return indices
    finally:
        cap.release()


# ------------------------------------------------------------------------------------------------ stages (receipts)
def receipt_file(run_dir: Path, stage: str) -> Path:
    return Path(run_dir) / "receipts" / f"{stage}.json"


def receipt_state(run_dir: Path, stage: str) -> str:
    f = receipt_file(run_dir, stage)
    return "missing" if not f.is_file() else ("done" if read_json(f).get("status") == "ok" else "failed")


def write_receipt(run_dir: Path, stage: str, status: str, started: float, outputs: dict, details: dict) -> dict:
    rec = {"stage": stage, "status": status, "finished_utc": _utc(), "duration_seconds": round(time.time() - started, 2),
           "outputs": outputs, "details": _clean(details)}
    write_json(receipt_file(run_dir, stage), rec)
    return rec


def _hashes(base: Path, rels) -> dict:
    return {r: sha256_file(Path(base) / r) for r in rels}


def _move_aside(path: Path) -> None:
    n = 0
    while (path.parent / f"{path.name}.incomplete-{n}").exists():
        n += 1
    path.rename(path.parent / f"{path.name}.incomplete-{n}")


def largest_model_dir(colmap_dir: Path):
    """Text model directory with the most registered images (ties: lowest model id)."""
    best, best_n = None, -1
    for d in sorted(Path(colmap_dir).glob("text_*"), key=lambda p: (len(p.name), p.name)):
        if (d / "images.txt").is_file():
            n = sum(1 for ln in (d / "images.txt").read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#") and ln.split()[-1].lower().endswith((".png", ".jpg")))
            if n > best_n:
                best, best_n = d, n
    return best


# COLMAP settings variants. "default" is exactly compare_backends.run_colmap (SIFT defaults). The others only change classical SIFT/matching
# thresholds (no label of any kind is involved); the variant is recorded in colmap_report.json and chosen by registered fraction only.
SIFT_VARIANTS = {
    "default": {"feature": [], "matcher": []},
    "lowthresh": {"feature": ["--SiftExtraction.peak_threshold", "0.0013", "--SiftExtraction.edge_threshold", "20"], "matcher": []},
    "lowthresh_loose": {"feature": ["--SiftExtraction.peak_threshold", "0.0013", "--SiftExtraction.edge_threshold", "20",
                                    "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"],
                        "matcher": ["--SiftMatching.max_ratio", "0.9", "--SiftMatching.max_distance", "0.8"]},
}


def run_colmap_variant(colmap: Path, images: Path, work: Path, variant: str = "default", min_model_size: int = 3):
    """compare_backends.run_colmap with extra SIFT/matcher flags (default variant = identical command). Returns (report, parsed models)."""
    import compare_backends as cb
    v = SIFT_VARIANTS[variant]
    work.mkdir(parents=True, exist_ok=False)
    db, sparse = work / "database.db", work / "sparse"
    sparse.mkdir()
    plan = [("feature_extractor", [colmap, "feature_extractor", "--database_path", db, "--image_path", images, "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                                   "--ImageReader.single_camera", "1", "--FeatureExtraction.use_gpu", "0", "--log_target", "stderr"] + v["feature"]),
            ("exhaustive_matcher", [colmap, "exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "0", "--log_target", "stderr"] + v["matcher"]),
            ("mapper", [colmap, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse, "--Mapper.min_model_size",
                        str(min_model_size), "--log_target", "stderr"])]
    steps = []
    for name, argv in plan:
        rec = cb.run_command(argv)
        rec["name"] = name
        steps.append(rec)
        if rec.get("returncode") != 0:
            break
    parsed = []
    model_dirs = sorted((p for p in sparse.iterdir() if p.is_dir() and (p / "images.bin").exists()), key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30)
    for md in model_dirs:
        txt = work / f"text_{md.name}"
        txt.mkdir()
        cb.run_command([colmap, "model_converter", "--input_path", md, "--output_path", txt, "--output_type", "TXT"])
        parsed.append((md.name, cb.parse_text_model(txt)))
    report = {"colmap_executable": str(colmap), "variant": variant, "variant_flags": v, "camera_model": "SIMPLE_PINHOLE", "single_camera": True,
              "matcher": "exhaustive, CPU", "model_count": len(model_dirs), "pipeline_completed": all(s.get("returncode") == 0 for s in steps),
              "database_stats": cb.database_stats(db), "runtime_seconds": round(sum(s["duration_seconds"] for s in steps), 3),
              "steps": [cb._public(s) for s in steps], "commands": [s["command"] for s in steps]}
    return report, parsed


def stage_colmap(run_dir: Path, source_run: Path, colmap: Path, variant: str = "default") -> dict:
    """Frames from the video -> COLMAP (CPU) -> text models; frames and database are removed afterwards."""
    started = time.time()
    import compare_backends as cb
    video = next(iter(sorted((Path(source_run) / "capture" / "videos").glob("*.mp4"))))
    frames, work = Path(run_dir) / "colmap_frames", Path(run_dir) / "colmap"
    for p in (frames, work):
        if p.exists():
            _move_aside(p)
    indices = extract_video_frames(video, frames)
    report, parsed = run_colmap_variant(Path(colmap), frames, work, variant)
    best = max(parsed, key=lambda mp: len(mp[1]["images"]), default=None)
    models = [{"model_id": mid, "registered_images": pm["registered_images"], "num_points3D": pm["num_points3D"],
               "mean_reprojection_error_px": pm["mean_reprojection_error_px_observation_weighted"], "track_length": pm["track_length"]}
              for mid, pm in parsed]
    keep_rel = [str(p.relative_to(run_dir)).replace("\\", "/") for p in sorted(work.rglob("*"))
                if p.is_file() and p.suffix in (".txt", ".bin") and p.parent.name != "sparse_unused"]
    shutil.rmtree(frames, ignore_errors=True)
    db = work / "database.db"
    if db.is_file():
        db.unlink()
    write_json(Path(run_dir) / "colmap_report.json", _clean({k: v for k, v in report.items() if k != "steps"} | {
        "frame_indices": indices, "video_sha256": sha256_file(video), "models_summary": models,
        "largest_model_id": None if best is None else best[0],
        "step_seconds": {s["name"]: s["duration_seconds"] for s in report.get("steps", [])}, "frames_and_database_removed_after_run": True}))
    status = "ok" if best is not None and best[1]["registered_images"] >= 3 else "failed"
    return write_receipt(run_dir, "colmap", status, started, _hashes(run_dir, keep_rel + ["colmap_report.json"]),
                         {"registered_images": 0 if best is None else best[1]["registered_images"], "model_count": len(parsed),
                          "num_points3D": 0 if best is None else best[1]["num_points3D"],
                          "mean_reprojection_error_px": None if best is None else best[1]["mean_reprojection_error_px_observation_weighted"],
                          "colmap_seconds": report.get("runtime_seconds"), "error": None if status == "ok" else "no COLMAP model with >= 3 images"})


def stage_build(run_dir: Path, source_run: Path, min_points: int = MIN_POINTS) -> dict:
    started = time.time()
    model_dir = largest_model_dir(Path(run_dir) / "colmap")
    if model_dir is None:
        raise RuntimeError("no COLMAP text model to build from")
    model = parse_sparse_model(model_dir)
    with np.load(Path(source_run) / "da3" / "prediction.npz", allow_pickle=False) as data:
        da3 = {k: data[k] for k in data.files}
    arrays, report = build_prediction(model, da3, min_points=min_points)
    report.update(colmap_model_dir=str(model_dir.relative_to(run_dir)).replace("\\", "/"), colmap_stats=model["stats"],
                  da3_prediction_sha256=sha256_file(Path(source_run) / "da3" / "prediction.npz"),
                  note="depth scaled per frame onto COLMAP sparse points; arbitrary COLMAP scale; unregistered/refused frames dropped")
    sha = write_prediction(Path(run_dir) / "prediction" / "prediction.npz", arrays)
    write_json(Path(run_dir) / "prediction" / "alignment_report.json", report)
    ok = report["frames_kept"] >= 3
    return write_receipt(run_dir, "build", "ok" if ok else "failed", started,
                         _hashes(run_dir, ["prediction/prediction.npz", "prediction/alignment_report.json"]),
                         {"frames_kept": report["frames_kept"], "frames_dropped_unregistered": report["frames_dropped_unregistered"],
                          "frames_dropped_too_few_points": report["frames_dropped_too_few_points"], "prediction_sha256": sha,
                          "error": None if ok else "fewer than 3 frames kept"})


def stage_chain(run_dir: Path, label: str) -> dict:
    """planes -> axis candidates -> height -> walls -> ray boundary, via run_axis_candidates.run_extraction (as run_rgb_batch)."""
    started = time.time()
    import run_axis_candidates as rac
    tmp, final = Path(run_dir) / "axis.tmp", Path(run_dir) / "axis"
    for p in (tmp, final):
        if p.exists():
            _move_aside(p)
    index = rac.run_extraction([{"label": label, "prediction": str(Path(run_dir) / "prediction" / "prediction.npz")}], tmp)
    rac.write_reports(index, tmp)
    tmp.rename(final)
    rec = index["budgets"][0]
    cands = rec.get("candidates", [])
    outs = _hashes(run_dir, [str(p.relative_to(run_dir)).replace("\\", "/") for p in sorted(final.rglob("*")) if p.is_file()])
    return write_receipt(run_dir, "chain", "ok" if rec.get("status") == "ok" else "failed", started, outs,
                         {"label": label, "chain_status": rec.get("status"), "axis_status": rec.get("axis_status"),
                          "candidate_count": len(cands), "co_supported_count": sum(1 for c in cands if c["co_supported_with_best"]),
                          "error": rec.get("error")})


def freeze_files(run_dir: Path, source_run: Path) -> dict:
    """Absolute-path -> sha256 of everything the truth-free part consumed or produced."""
    files = {}
    srcs = [Path(source_run) / "capture" / "videos", Path(source_run) / "da3"]
    for base in srcs:
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files["source:" + str(p.relative_to(source_run)).replace("\\", "/")] = sha256_file(p)
    for sub in ("colmap", "prediction", "axis"):
        for p in sorted((Path(run_dir) / sub).rglob("*")):
            if p.is_file():
                files[str(p.relative_to(run_dir)).replace("\\", "/")] = sha256_file(p)
    files["colmap_report.json"] = sha256_file(Path(run_dir) / "colmap_report.json")
    return files


def stage_freeze(run_dir: Path, source_run: Path, run_label: str, code: dict) -> dict:
    started = time.time()
    files = freeze_files(run_dir, source_run)
    freeze = {"run_id": run_label, "frozen_utc": _utc(), "source_run": str(source_run), "files": files, "code_sha256": code,
              "min_points": MIN_POINTS, "ray_k": RAY_K,
              "reconstruction_inputs": ["capture video", "frozen DA3 prediction (depth only)", "COLMAP sparse model from the video frames"]}
    write_json(Path(run_dir) / "freeze.json", freeze)
    return write_receipt(run_dir, "freeze", "ok", started, {"freeze.json": sha256_file(Path(run_dir) / "freeze.json")}, {"files": len(files)})


def verify_run_freeze(run_dir: Path) -> dict:
    freeze = read_json(Path(run_dir) / "freeze.json")
    src = Path(freeze["source_run"])
    for rel, digest in freeze["files"].items():
        f = src / rel[len("source:"):] if rel.startswith("source:") else Path(run_dir) / rel
        if not f.is_file() or sha256_file(f) != digest:
            raise RuntimeError(f"frozen file changed or missing after freezing: {f}")
    return freeze


def run_truth_free_stages(run_dir: Path, source_run: Path, label: str, colmap: Path, code: dict, retry_failed: bool = False, variant: str = "default") -> dict:
    """colmap -> build -> chain -> freeze for one capture, resuming finished stages from receipts. Returns per-stage states."""
    fns = (("colmap", lambda: stage_colmap(run_dir, source_run, colmap, variant)), ("build", lambda: stage_build(run_dir, source_run)),
           ("chain", lambda: stage_chain(run_dir, label)), ("freeze", lambda: stage_freeze(run_dir, source_run, label, code)))
    states = {}
    for stage, fn in fns:
        st = receipt_state(run_dir, stage)
        if st == "failed" and retry_failed:
            receipt_file(run_dir, stage).rename(receipt_file(run_dir, stage).with_suffix(".failed.json"))
            st = "missing"
        if st == "done":
            states[stage] = "skipped"
            continue
        if st == "failed":
            states[stage] = "failed_previously"
            break
        try:
            rec = fn()
        except Exception as exc:  # recorded, the batch continues with the next capture
            write_receipt(run_dir, stage, "failed", time.time(), {}, {"error": f"{type(exc).__name__}: {exc}"})
            states[stage] = f"failed: {type(exc).__name__}: {exc}"
            break
        states[stage] = rec["status"]
        if rec["status"] != "ok":
            break
    return states


# ------------------------------------------------------------------------------------------------ batch driver
def code_hashes() -> dict:
    import run_rgb_batch as rb
    out = rb.code_hashes()
    for name in ("colmap_poses_da3_depth.py", "compare_backends.py", "audit_camera_trajectory.py", "run_rgb_batch.py"):
        out[name] = sha256_file(HERE / name)
    return out


def run_dir_for(root: Path, spec: dict) -> Path:
    return Path(root) / "houses" / f"house_{spec['house_seed']}" / f"path_{spec['path_index']:02d}"


def run_batch(source_root: Path, out_root: Path, houses=None, paths=None, max_hours: float = 2.5, colmap: Path = DEFAULT_COLMAP,
              retry_failed: bool = False, variant: str = "default") -> dict:
    source_root, out_root = Path(source_root), Path(out_root)
    cfg_src = read_json(source_root / "batch_config.json")
    cfg_file = out_root / "colmap_da3_config.json"
    if out_root.exists() and not cfg_file.is_file():
        raise SystemExit(f"{out_root} exists and is not a colmap_poses_da3_depth output folder (never overwrite)")
    if not cfg_file.is_file():
        write_json(cfg_file, {"schema": "colmap_poses_da3_depth_v1", "created_utc": _utc(), "source_root": str(source_root),
                              "source_batch_config_sha256": sha256_file(source_root / "batch_config.json"), "min_points": MIN_POINTS, "sift_variant": variant,
                              "ray_k": RAY_K, "label": "post-hoc method development on already-seen captures; not held-out; no oracle tuning; gates unchanged"})
    elif read_json(cfg_file)["source_root"] != str(source_root):
        raise SystemExit("output folder belongs to a different source root")
    code = code_hashes()
    t0, results = time.time(), []
    for spec in cfg_src["plan"]:
        if (houses is not None and spec["house_seed"] not in houses) or (paths is not None and spec["path_index"] not in paths):
            continue
        if (time.time() - t0) / 3600.0 >= max_hours:
            print(f"[{_utc()}] time budget {max_hours} h reached before {spec['run_id']}", flush=True)
            break
        run_dir = run_dir_for(out_root, spec)
        source_run = run_dir_for(source_root, spec)
        run_dir.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        print(f"[{_utc()}] {spec['run_id']}", flush=True)
        states = run_truth_free_stages(run_dir, source_run, spec["run_id"], colmap, code, retry_failed, variant)
        res = {"run_id": spec["run_id"], "states": states, "seconds_this_invocation": round(time.time() - t1, 1)}
        results.append(res)
        print(f"[{_utc()}]   -> {states} ({res['seconds_this_invocation']} s)", flush=True)
    return {"runs_processed": len(results), "results": results}


# ------------------------------------------------------------------------------------------------ EVALUATION ONLY (never called by stages)
def _pose_row(src_run: Path, run_dir: Path) -> dict:
    """Pose errors of COLMAP, frozen DA3 and DA3-on-the-same-frames against the private poses. Sim(3) alignment, per
    compare_backends.score_pose_set (audit_camera_trajectory). Alignments are diagnostics and never exported."""
    import compare_backends as cb
    gt = read_json(next(iter((src_run / "capture" / "private_ground_truth").glob("path_*.json"))))["poses"]
    rep = read_json(run_dir / "colmap_report.json")
    row = {"colmap_models": rep.get("model_count"), "colmap_registered_in_largest": None, "colmap_runtime_s": rep.get("runtime_seconds")}
    model = parse_sparse_model(largest_model_dir(run_dir / "colmap")) if largest_model_dir(run_dir / "colmap") else None
    with np.load(src_run / "da3" / "prediction.npz", allow_pickle=False) as d:
        da3_idx, da3_ex = np.asarray(d["source_indices"], int), np.asarray(d["extrinsics"], float)
    strip = lambda s: {k: v for k, v in s.items() if not k.startswith("_")}

    def summary(s):
        if s["status"] != "scored":
            return {"status": s["status"], "reason": s.get("reason")}
        c = s["centre_aligned_umeyama"]
        return {"status": "scored", "frames": s["frames_scored"], "centre_rmse_m": c["centre_rmse_m"],
                "rot_median_deg": c["rotation_error_after_centre_alignment"]["geodesic_median_deg"],
                "rot_p90_deg": c["rotation_error_after_centre_alignment"]["geodesic_p90_deg"],
                "rot_max_deg": c["rotation_error_after_centre_alignment"]["geodesic_max_deg"],
                "rot_kabsch_median_deg": s["rotation_aligned_kabsch"]["geodesic_median_deg"]}
    row["da3_all120"] = summary(strip(cb.score_pose_set(da3_ex, da3_idx, gt)))
    if model is not None and model["images"]:
        ids = sorted(model["images"])
        row["colmap_registered_in_largest"] = len(ids)
        row["colmap_registered_fraction"] = len(ids) / len(da3_idx)
        row["colmap"] = summary(strip(cb.score_pose_set(np.stack([model["images"][i]["w2c"] for i in ids]), ids, gt)))
        pos = {int(i): k for k, i in enumerate(da3_idx)}
        common = [i for i in ids if i in pos]
        row["da3_same_frames"] = summary(strip(cb.score_pose_set(da3_ex[[pos[i] for i in common]], common, gt)))
        pred = read_json(run_dir / "prediction" / "alignment_report.json")
        kept = [f["source_index"] for f in pred["frames"] if f["status"] == "kept"]
        row["frames_kept_after_depth_alignment"] = len(kept)
        if len(kept) >= 8:
            row["colmap_kept_frames"] = summary(strip(cb.score_pose_set(np.stack([model["images"][i]["w2c"] for i in kept]), kept, gt)))
        row["colmap_stats"] = model["stats"]
    else:
        row["colmap_registered_in_largest"], row["colmap_registered_fraction"] = 0, 0.0
        row["colmap"] = {"status": "not_comparable", "reason": "no registered frames"}
    return row


def evaluate_root(out_root: Path, hooks: dict | None = None) -> dict:
    """EVALUATION ONLY. Refuses unless every processed run is frozen and verifies; writes the freeze index BEFORE any private file is read."""
    import run_rgb_batch as rb
    out_root = Path(out_root)
    cfg = read_json(out_root / "colmap_da3_config.json")
    source_root = Path(cfg["source_root"])
    src_cfg = read_json(source_root / "batch_config.json")
    plan = [s for s in src_cfg["plan"] if (run_dir_for(out_root, s) / "freeze.json").is_file()]
    if not plan:
        raise SystemExit("no frozen runs")
    n = 0
    while (out_root / f"results_{n:03d}").exists():
        n += 1
    out = out_root / f"results_{n:03d}"
    write_json(out / "freeze_index_before_truth.json", {"written_utc": _utc(), "runs": {s["run_id"]: sha256_file(run_dir_for(out_root, s) / "freeze.json") for s in plan}})
    for s in plan:
        verify_run_freeze(run_dir_for(out_root, s))
        rb.verify_freeze(run_dir_for(source_root, s))          # the frozen DA3 baseline must also be intact
    hk = {**rb._default_eval_hooks(), **(hooks or {})}
    gates = src_cfg["gates"]
    runs, cand_rows, poses = [], [], []
    for spec in plan:
        run_dir, src_run = run_dir_for(out_root, spec), run_dir_for(source_root, spec)
        index = read_json(run_dir / "axis" / "frozen_index.json")
        rec = index["budgets"][0]
        scene = hk["load_scene"](src_run / "capture" / "private_ground_truth" / "scene.json")
        qa = read_json(src_run / "generator_private" / "capture_qa.json")
        n_kept = len(rec.get("source_indices", []))
        chain = {**read_json(receipt_file(run_dir, "chain"))["details"]}
        manifest = {"provenance": "frozen frame budget (COLMAP-registered frames with a depth scale)", "retained_capture_fraction": n_kept / rb.SETTINGS["frames_available"],
                    "frames_registered": n_kept, "frames_available": rb.SETTINGS["frames_available"]}
        cands = rec.get("candidates", [])
        tie = rb.tie_info_of([{"candidate_id": c["candidate_id"], "viable": c["viable"], "score": c["score"]} for c in cands])
        rows = []
        for c in cands:
            files = c["files"]
            cand = {"candidate_id": c["candidate_id"], "rank": c["rank"], "co_supported_with_best": c["co_supported_with_best"], "score": c["score"],
                    "viable": c["viable"], "polygon": c.get("polygon")}
            hstatus, ev = None, None
            if "height" in files:
                hstatus = hk["read_frozen"](run_dir / "axis", files["height"]["path"], files["height"]["sha256"]).get("status")
            if "boundary" in files and "height" in files:
                boundary = hk["read_frozen"](run_dir / "axis", files["boundary"]["path"], files["boundary"]["sha256"])
                height = hk["read_frozen"](run_dir / "axis", files["height"]["path"], files["height"]["sha256"])
                try:
                    ev = hk["evaluate_floorplan"](hk["eval_copy"](boundary), scene, hk["evaluator_compatible"](height), manifest)
                except Exception as exc:  # noqa: BLE001
                    ev = {"status": "evaluation_error", "failure_reasons": [f"{type(exc).__name__}: {exc}"], "metrics": None}
            rows.append(rb.candidate_row(spec, cand, tie, ev, hstatus, float(scene["height"]), gates))
        wall = sum(read_json(receipt_file(run_dir, s)).get("duration_seconds", 0.0) for s in STAGES if receipt_file(run_dir, s).is_file())
        run = rb.run_level(spec, qa, {}, {"status": "ok", "frames_used": n_kept, "da3_elapsed_seconds": None}, chain, rows, wall)
        run["stage_seconds"] = {s: read_json(receipt_file(run_dir, s)).get("duration_seconds") for s in STAGES if receipt_file(run_dir, s).is_file()}
        runs.append(run)
        cand_rows += rows
        pr = {"run_id": spec["run_id"], "house_seed": spec["house_seed"], "path_index": spec["path_index"]}
        try:
            pr.update(_pose_row(src_run, run_dir))
        except Exception as exc:  # noqa: BLE001
            pr["error"] = f"{type(exc).__name__}: {exc}"
        pr["stage_seconds"] = run["stage_seconds"]
        poses.append(pr)
    write_json(out / "runs.json", runs)
    write_json(out / "candidates.json", cand_rows)
    write_json(out / "pose_table.json", _clean(poses))
    (out / "POSES.md").write_text(poses_markdown(poses), encoding="utf-8")
    (out / "RESULTS.md").write_text("# COLMAP poses + DA3 depth: chain results (EVALUATION ONLY; post-hoc method development on already-seen captures, not held-out)\n\n"
                                    + rb.runs_markdown(runs) + "\n## Every candidate\n\n" + rb.candidates_markdown(cand_rows), encoding="utf-8")
    return {"results_dir": str(out), "runs": len(runs), "runs_pass": sum(r["run_pass"] for r in runs)}


def poses_markdown(poses: list) -> str:
    f = lambda v, nd=2: "-" if v is None else format(v, f".{nd}f")
    head = ["| run | COLMAP reg/120 | models | pts | reproj px | COLMAP s | COLMAP RMSE m | COLMAP rot med/p90/max deg | DA3 RMSE m (120) | DA3 rot med/p90/max | DA3 same frames RMSE m | frames kept (depth) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in poses:
        c, d, s = p.get("colmap") or {}, p.get("da3_all120") or {}, p.get("da3_same_frames") or {}
        st = p.get("colmap_stats") or {}
        rot = lambda x: "-" if x.get("status") != "scored" else f"{f(x['rot_median_deg'], 1)}/{f(x['rot_p90_deg'], 1)}/{f(x['rot_max_deg'], 1)}"
        head.append(f"| {p['run_id']} | {p.get('colmap_registered_in_largest')} | {p.get('colmap_models')} | {st.get('num_points3D')} | {f(st.get('mean_reprojection_error_px'))} | "
                    f"{f(p.get('colmap_runtime_s'), 0)} | {f(c.get('centre_rmse_m'), 3)} | {rot(c)} | {f(d.get('centre_rmse_m'), 3)} | {rot(d)} | "
                    f"{f(s.get('centre_rmse_m'), 3)} | {p.get('frames_kept_after_depth_alignment')} |")
    return "\n".join(head) + "\n"


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="truth-free: COLMAP + depth alignment + chain + freeze, serial and resumable")
    r.add_argument("--source-root", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--houses", type=int, nargs="*", default=None)
    r.add_argument("--paths", type=int, nargs="*", default=None)
    r.add_argument("--max-hours", type=float, default=2.5)
    r.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--sift-variant", choices=sorted(SIFT_VARIANTS), default="default")
    e = sub.add_parser("evaluate", help="EVALUATION ONLY, after every run is frozen")
    e.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        print(json.dumps(run_batch(a.source_root, a.output_dir, a.houses, a.paths, a.max_hours, a.colmap, a.retry_failed, a.sift_variant), indent=1))
    else:
        print(json.dumps(evaluate_root(a.output_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

