"""Backend comparison on ONE appearance variant: whole-video DA3 versus COLMAP SfM, on IDENTICAL selected RGB frames.

Inference inputs are only the selected RGB frames copied into a scratch image folder inside the output directory.
Private ground truth (<variant>/private_ground_truth) is read ONLY by the scoring functions below (evaluation only) and
by evaluate_floorplan.py; it is never passed to DA3, COLMAP or run_pipeline.py.

Per variant, one comparison.json records: the explicit frame budget, the exact commands, sha256 prefixes of inputs and
outputs, runtimes, VRAM (DA3 manifest), failures verbatim, per-backend full-capture pose scores and a head-to-head line on
the COMMON registered source_indices.

Pose scoring uses ONE joint protocol per backend (audit_camera_trajectory functions): (i) one centre-aligned Umeyama
similarity with the rotation error measured after it, and (ii) separately labelled one Kabsch rotation fit. A COLMAP result
can contain several disconnected models (each with its own arbitrary frame): every model is scored separately and
reported; the headline is the largest model and the registered/retained fractions count ALL models. Too few or degenerate
common frames give "not_comparable", never success. COLMAP sparse points are not a floor plan; the dense stages are not run
in this protocol, so no floor plan is produced from COLMAP. Nothing here claims a benchmark stage complete.

  python compare_backends.py run --variant <variant folder> --output <NEW dir> [--frame-step 3]
  python compare_backends.py summarize --comparisons a.json b.json ... --output <NEW dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import audit_camera_trajectory as A

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHOTO = REPO / "outputs" / "photogrammetry"
DEFAULT_COLMAP = PHOTO / "tools" / "colmap" / "bin" / "colmap.exe"
PRIVATE_DIR = "private_ground_truth"
MIN_COMMON_FRAMES = 8          # fewer registered frames than this => not_comparable
DEFAULT_FRAME_STEP = 3
DA3_RESOLUTION = 560
DA3_SEED = 2026
HASH_LEN = 16


# ------------------------------------------------------------------------------------------------ small helpers
def sha_prefix(path, n: int = HASH_LEN):
    p = Path(path)
    if not p.is_file():
        return None
    return A.sha256(p)[:n]


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(obj):
    """JSON-safe copy (numpy scalars/arrays, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_command(argv, cwd=None, timeout=None):
    """Run a subprocess; record the exact command, runtime, return code and verbatim output tails."""
    argv = [str(a) for a in argv]
    rec = {"command": argv, "started_utc": _utc()}
    t0 = time.time()
    try:
        done = subprocess.run(argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
        keep = 3000 if done.returncode else 800
        rec.update(returncode=done.returncode, stdout_tail=done.stdout[-keep:], stderr_tail=done.stderr[-keep:])
        rec["_stdout"], rec["_stderr"] = done.stdout, done.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        rec.update(returncode=None, error=f"{type(exc).__name__}: {exc}", stdout_tail="", stderr_tail="")
        rec["_stdout"], rec["_stderr"] = "", ""
    rec["duration_seconds"] = round(time.time() - t0, 3)
    return rec


def _public(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# ------------------------------------------------------------------------------------------------ frame selection
def select_frames(frames_available: int, frame_step: int = DEFAULT_FRAME_STEP, offset: int = 0):
    """Deterministic frame budget: every `frame_step`-th source frame starting at `offset`. Returns the record to store."""
    if frames_available < 1:
        raise ValueError("no frames available")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if not 0 <= offset < frame_step:
        raise ValueError("offset must satisfy 0 <= offset < frame_step")
    indices = list(range(offset, frames_available, frame_step))
    default = (frame_step == DEFAULT_FRAME_STEP and offset == 0)
    return {"frames_available": int(frames_available), "frame_step": int(frame_step), "offset": int(offset),
            "frames_selected": len(indices), "selected_indices": indices,
            "budget_label": (f"default_every_{frame_step}rd_frame" if default else
                             f"NON_DEFAULT_BUDGET_every_{frame_step}_frame_offset_{offset}: not comparable with the default budget"),
            "is_default_budget": bool(default),
            "selected_fraction_of_capture": len(indices) / frames_available}


def write_scratch_frames(rgb_dir: Path, indices, scratch: Path):
    """Copy the selected frames under their ORIGINAL names; the name encodes the source index (no other label)."""
    if PRIVATE_DIR in {p.lower() for p in Path(scratch).resolve().parts}:
        raise ValueError("scratch frames must not live inside private ground truth")
    scratch.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in Path(rgb_dir).iterdir() if p.suffix.lower() == ".png")
    by_index = {int(p.stem): p for p in files}
    out = []
    for i in indices:
        if int(i) not in by_index:
            raise FileNotFoundError(f"frame {i} missing in {rgb_dir}")
        dest = scratch / by_index[int(i)].name
        shutil.copy2(by_index[int(i)], dest)
        out.append({"source_index": int(i), "file": dest.name, "sha256_prefix": sha_prefix(dest)})
    return out


def remap_source_indices(prediction_npz: Path, selected_indices, out_npz: Path):
    """infer_rgb numbers images 0..N-1 inside the scratch folder; write a copy whose source_indices are the ORIGINAL frame
    indices (raw file untouched) so scoring and the structural stages see true frame numbers."""
    with np.load(prediction_npz, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    raw = arrays["source_indices"].tolist()
    ordered = sorted(int(i) for i in selected_indices)
    if raw != list(range(len(ordered))):
        raise ValueError(f"unexpected infer_rgb source_indices {raw[:5]}...: expected 0..{len(ordered) - 1}")
    arrays["source_indices"] = np.asarray(ordered, dtype=np.int64)
    np.savez_compressed(out_npz, **arrays)
    return ordered


# ------------------------------------------------------------------------------------------------ scoring (evaluation only)
def gt_centres_all(gt_poses):
    return np.asarray([p["position"] for p in gt_poses], dtype=float)


def score_pose_set(w2c, indices, gt_poses, *, min_frames: int = MIN_COMMON_FRAMES):
    """Joint pose score of ONE reconstruction frame (one model): centre-aligned Umeyama + separate Kabsch rotation fit.

    `w2c`: Nx3x4 / Nx4x4 world-to-camera (OpenCV) poses; `indices`: their source frame indices. Evaluation only.
    """
    indices = np.asarray(indices, dtype=int)
    n = len(indices)
    all_centres = gt_centres_all(gt_poses)
    full_extent = float(np.linalg.norm(np.ptp(all_centres, axis=0)))
    base = {"frames_scored": int(n), "source_indices": indices.tolist(), "gt_full_trajectory_extent_m": full_extent}
    if n < min_frames:
        return {"status": "not_comparable", "reason": f"only {n} registered frames (< {min_frames})", **base}
    if len(np.unique(indices)) != n:
        return {"status": "not_comparable", "reason": "duplicate source indices", **base}
    try:
        w2c = np.asarray(w2c, float)
        centres = A.camera_centers(w2c)
        gt_c = np.asarray([gt_poses[int(i)]["position"] for i in indices], float)
        fit = A.fit_similarity(centres, gt_c)
        audit = A.orientation_audit(w2c, gt_poses, indices, center_alignment_rotation=np.asarray(fit["rotation"]))
    except (ValueError, np.linalg.LinAlgError, IndexError) as exc:
        return {"status": "not_comparable", "reason": f"degenerate or invalid pose set: {exc}", **base}
    pick = lambda d: {k: d[k] for k in ("geodesic_median_deg", "geodesic_p90_deg", "geodesic_max_deg")}
    return {"status": "scored", **base,
            "centre_aligned_umeyama": {"rotation_error_after_centre_alignment": pick(audit["center_umeyama_aligned"]),
                                       "centre_rmse_m": fit["rmse"],
                                       "centre_rmse_fraction_of_full_gt_extent": fit["rmse"] / full_extent,
                                       "centre_rmse_fraction_of_subset_gt_extent": fit["rmse"] / fit["gt_extent"],
                                       "uniform_scale": fit["scale"],
                                       "alignment_rotation_well_conditioned": fit["alignment_rotation_well_conditioned"]},
            "rotation_aligned_kabsch": {"label": "separate rotation-only Kabsch fit to the camera rotations (not centre-aligned)",
                                        **pick(audit["rotation_kabsch_aligned"])},
            "oracle_alignment_exported_as_corrected_poses": False,
            "_similarity": {"scale": fit["scale"], "rotation": np.asarray(fit["rotation"]).tolist(),
                            "translation": np.asarray(fit["translation"]).tolist()}}


def score_backend(models, gt_poses, selected_indices, frames_available):
    """models: list of {"model_id", "indices", "w2c"}; every model scored separately; retained fractions use ALL models."""
    selected = sorted(int(i) for i in selected_indices)
    registered = sorted({int(i) for m in models for i in m["indices"]})
    scored = []
    for m in models:
        s = score_pose_set(m["w2c"], m["indices"], gt_poses)
        scored.append({"model_id": m["model_id"], "registered_frames": len(m["indices"]), **s})
    order = sorted(range(len(models)), key=lambda k: -len(models[k]["indices"]))
    out = {"frames_selected": len(selected), "frames_registered_all_models": len(registered),
           "registered_fraction_of_selected": len(registered) / len(selected) if selected else None,
           "retained_capture_fraction_of_full_capture": len(registered) / frames_available if frames_available else None,
           "full_capture_frames": int(frames_available), "model_count": len(models),
           "protocol": "per-model joint protocol (own Umeyama + own Kabsch); headline = largest model; fractions count all models",
           "models": scored}
    if models:
        head = scored[order[0]]
        out["headline_model_id"] = head["model_id"]
        out["headline"] = {k: v for k, v in head.items() if not k.startswith("_")}
        out["headline_frames_fraction_of_selected"] = len(models[order[0]]["indices"]) / len(selected)
    else:
        out["headline"] = {"status": "not_comparable", "reason": "no registered frames"}
    return out


def head_to_head(model_a, model_b, gt_poses, min_common: int = MIN_COMMON_FRAMES):
    """Rescore both backends on the common source_indices only. model_x = {"indices","w2c"} (largest model for COLMAP)."""
    ia = {int(i): k for k, i in enumerate(model_a["indices"])}
    ib = {int(i): k for k, i in enumerate(model_b["indices"])}
    common = sorted(set(ia) & set(ib))
    out = {"common_source_indices": common, "common_count": len(common),
           "registered_a": len(ia), "registered_b": len(ib), "min_common_required": min_common}
    if len(common) < min_common:
        out.update(status="not_comparable", reason=f"only {len(common)} common registered frames (< {min_common})")
        return out
    wa = np.asarray(model_a["w2c"], float)[[ia[i] for i in common]]
    wb = np.asarray(model_b["w2c"], float)[[ib[i] for i in common]]
    sa, sb = score_pose_set(wa, common, gt_poses, min_frames=min_common), score_pose_set(wb, common, gt_poses, min_frames=min_common)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    out.update(status="scored" if sa["status"] == sb["status"] == "scored" else "not_comparable", a=strip(sa), b=strip(sb))
    if out["status"] != "scored":
        out["reason"] = "; ".join(f"{n}: {s.get('reason')}" for n, s in (("a", sa), ("b", sb)) if s["status"] != "scored")
    return out


# ------------------------------------------------------------------------------------------------ COLMAP parsing
_GLOG_PREFIX = re.compile(r"^[IWEF]\d{8}\s+[\d:.]+\s+\d+\s+[^\]]*\]\s*")


def parse_model_analyzer(text: str) -> dict:
    """Parse `colmap model_analyzer` output ("Registered images: 12", "Mean reprojection error: 0.6px", ...)."""
    out = {}
    for raw in text.splitlines():
        line = _GLOG_PREFIX.sub("", raw.strip())
        m = re.match(r"^([A-Za-z][A-Za-z _/]*?):\s*(-?[0-9.]+(?:[eE][-+]?\d+)?)\s*(?:px)?\s*$", line)
        if m:
            out[m.group(1).strip().lower().replace(" ", "_")] = float(m.group(2))
    return out


def _qvec_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def parse_text_model(model_dir: Path) -> dict:
    """Parse a COLMAP TXT model (cameras/images/points3D .txt): registered images + poses, point and track statistics."""
    model_dir = Path(model_dir)
    cams = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            t = line.split()
            cams[int(t[0])] = {"model": t[1], "width": int(t[2]), "height": int(t[3]), "params": [float(v) for v in t[4:]]}
    lines = [l for l in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    images = []
    # images.txt alternates a pose line and a 2D-points line (the latter may be empty): do not drop blank lines.
    for k in range(0, len(lines) - 1, 2):
        t = lines[k].split()
        if len(t) < 10:
            continue
        q = [float(v) for v in t[1:5]]
        rot = _qvec_to_rot(q)
        w2c = np.concatenate([rot, np.asarray([[float(v) for v in t[5:8]]]).T], axis=1)
        name = t[9]
        stem = Path(name).stem
        images.append({"image_id": int(t[0]), "name": name, "source_index": int(stem) if stem.isdigit() else None,
                       "camera_id": int(t[8]), "w2c": w2c, "num_points2D": len(lines[k + 1].split()) // 3})
    errors, tracks = [], []
    pfile = model_dir / "points3D.txt"
    if pfile.exists():
        for line in pfile.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                t = line.split()
                errors.append(float(t[7])); tracks.append((len(t) - 8) // 2)
    errors, tracks = np.asarray(errors, float), np.asarray(tracks, float)
    obs = float(tracks.sum()) if len(tracks) else 0.0
    return {"cameras": cams, "images": images, "registered_images": len(images), "num_points3D": int(len(errors)),
            "observations": int(obs),
            "mean_reprojection_error_px_observation_weighted": float((errors * tracks).sum() / obs) if obs else None,
            "mean_point_error_px": float(errors.mean()) if len(errors) else None,
            "track_length": ({"mean": float(tracks.mean()), "median": float(np.median(tracks)), "min": int(tracks.min()),
                              "max": int(tracks.max()), "p90": float(np.percentile(tracks, 90))} if len(tracks) else None)}


def models_for_scoring(parsed_models):
    out = []
    for mid, pm in parsed_models:
        ok = [im for im in pm["images"] if im["source_index"] is not None]
        out.append({"model_id": mid, "indices": [im["source_index"] for im in ok], "w2c": np.stack([im["w2c"] for im in ok]) if ok else np.zeros((0, 3, 4))})
    return out


def colmap_failure_lines(text: str):
    keys = ("initial pair", "no good", "failed", "error", "fatal", "check failed", "cannot", "could not")
    return [l.strip() for l in text.splitlines() if any(k in l.lower() for k in keys)][-20:]


def database_stats(db: Path) -> dict:
    """Feature counts per image and number of geometrically verified pairs, straight from the COLMAP database."""
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
        try:
            counts = np.asarray([r[0] for r in con.execute("select rows from keypoints")], float)
            images = con.execute("select count(*) from images").fetchone()[0]
            verified = con.execute("select count(*) from two_view_geometries where rows > 0").fetchone()[0]
            matched = con.execute("select count(*) from matches where rows > 0").fetchone()[0]
            inl = [r[0] for r in con.execute("select rows from two_view_geometries where rows > 0")]
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"images": int(images), "features_per_image": {"min": float(counts.min()), "median": float(np.median(counts)), "max": float(counts.max()),
                                                          "images_with_zero_features": int((counts == 0).sum())} if len(counts) else None,
            "pairs_with_raw_matches": int(matched), "pairs_geometrically_verified": int(verified),
            "verified_pair_inliers": {"median": float(np.median(inl)), "max": int(max(inl))} if inl else None}


def run_colmap(colmap: Path, images: Path, work: Path, *, min_model_size: int = 3):
    """SIFT (CPU, single SIMPLE_PINHOLE camera) -> exhaustive matcher (CPU) -> mapper; every produced model is exported to TXT."""
    work.mkdir(parents=True, exist_ok=False)
    db = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir()
    steps, report = [], {"colmap_executable": str(colmap), "settings": {
        "features": "SIFT, CPU, default extraction thresholds (no contrast added)", "camera_model": "SIMPLE_PINHOLE",
        "single_camera": True, "matcher": "exhaustive, CPU", "mapper_min_model_size": min_model_size,
        "note": "mapper default min_model_size (10) lowered so small disconnected models are reported rather than hidden"}}
    version = run_command([colmap, "version"])
    report["version_output"] = (version.get("_stdout", "") + version.get("_stderr", "")).strip()[-400:]
    report["cuda_build"] = "with CUDA" in report["version_output"]
    report["dense_stage"] = {"cuda_build_present": report["cuda_build"], "run": False,
                             "status": "not run in this protocol: no dense floor plan pipeline exists for COLMAP output; "
                                       "COLMAP sparse points alone are not a completed floor plan",
                             "floor_plan_from_colmap": None}
    plan = [("feature_extractor", [colmap, "feature_extractor", "--database_path", db, "--image_path", images,
                                   "--ImageReader.camera_model", "SIMPLE_PINHOLE", "--ImageReader.single_camera", "1",
                                   "--FeatureExtraction.use_gpu", "0", "--log_target", "stderr"]),
            ("exhaustive_matcher", [colmap, "exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "0",
                                    "--log_target", "stderr"]),
            ("mapper", [colmap, "mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                        "--Mapper.min_model_size", str(min_model_size), "--log_target", "stderr"])]
    ok = True
    for name, argv in plan:
        rec = run_command(argv)
        rec["name"] = name
        rec["failure_lines"] = colmap_failure_lines(rec["_stdout"] + "\n" + rec["_stderr"]) if rec.get("returncode") != 0 or name == "mapper" else []
        steps.append(rec)
        if rec.get("returncode") != 0:
            ok = False
            break
    report["steps"] = [_public(s) for s in steps]
    report["mapper_initial_pair_lines"] = [l for l in (steps[-1].get("failure_lines", []) if steps else []) if "initial pair" in l.lower() or "no good" in l.lower()]
    report["database_sha256_prefix"] = sha_prefix(db)
    model_dirs = sorted((p for p in sparse.iterdir() if p.is_dir() and (p / "images.bin").exists()), key=lambda p: int(p.name) if p.name.isdigit() else 1 << 30)
    parsed = []
    for md in model_dirs:
        txt = work / f"text_{md.name}"
        txt.mkdir()
        conv = run_command([colmap, "model_converter", "--input_path", md, "--output_path", txt, "--output_type", "TXT"])
        ana = run_command([colmap, "model_analyzer", "--path", md])
        analyzer = parse_model_analyzer(ana["_stdout"] + "\n" + ana["_stderr"])
        entry = {"model_id": md.name, "model_dir": str(md), "convert_returncode": conv.get("returncode"),
                 "model_analyzer": analyzer, "model_analyzer_command": ana["command"],
                 "model_analyzer_raw_tail": (ana["_stdout"] + ana["_stderr"])[-1500:]}
        try:
            pm = parse_text_model(txt)
            entry.update({k: v for k, v in pm.items() if k not in ("images", "cameras")})
            entry["registered_names"] = sorted(im["name"] for im in pm["images"])
            entry["camera"] = list(pm["cameras"].values())
            parsed.append((md.name, pm))
        except (OSError, ValueError, IndexError) as exc:
            entry["parse_error"] = f"{type(exc).__name__}: {exc}"
        entry["images_bin_sha256_prefix"] = sha_prefix(md / "images.bin")
        report.setdefault("models", []).append(entry)
    report["pipeline_completed"] = ok
    report["model_count"] = len(model_dirs)
    report["database_stats"] = database_stats(db)
    if not model_dirs:
        last = steps[-1] if steps else {}
        lines = last.get("failure_lines", [])
        no_pair = sum("no good initial image pair" in l.lower() for l in lines)
        final = [l for l in lines if "failed to create any sparse model" in l.lower()]
        report["no_model_reason"] = ((f"COLMAP could not start: 'No good initial image pair found' ({no_pair} of the last logged attempts); " if no_pair else "")
                                     + ("; ".join(final) if final else "; ".join(lines[-3:]) or (last.get("stderr_tail") or "")[-400:]))
        report["no_model_reason_verbatim_lines"] = lines
    report["runtime_seconds"] = round(sum(s["duration_seconds"] for s in steps), 3)
    return report, parsed


# ------------------------------------------------------------------------------------------------ DA3 + structural stages
def run_da3(frames_dir: Path, out_dir: Path, n_frames: int):
    argv = [sys.executable, HERE / "infer_rgb.py", "--input", frames_dir, "--output", out_dir, "--frames", n_frames,
            "--resolution", DA3_RESOLUTION, "--seed", DA3_SEED]
    rec = run_command(argv, cwd=HERE)
    info = {"command": rec["command"], "returncode": rec.get("returncode"), "wall_seconds_incl_load": rec["duration_seconds"],
            "stderr_tail": rec.get("stderr_tail", ""), "stdout_tail": rec.get("stdout_tail", "")}
    man = out_dir / "manifest.json"
    if rec.get("returncode") == 0 and man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        info.update(inference_manifest_elapsed_seconds=m.get("elapsed_seconds"), gpu_peak_memory_bytes=m.get("gpu_peak_memory_bytes"),
                    gpu_peak_memory_gib=(m["gpu_peak_memory_bytes"] / 2 ** 30 if m.get("gpu_peak_memory_bytes") else None),
                    device=m.get("device"), model=m.get("model"), resolution=m.get("resolution"), seed=m.get("seed"),
                    frames_used=m.get("frames_used"), manifest_sha256_prefix=sha_prefix(man))
    return info


def write_provenance(path: Path, *, selected, registered, available, note):
    """Small provenance JSON for evaluate_floorplan --manifest: retained fraction is vs the FULL capture."""
    prov = {"kind": "compare_backends_provenance", "frames_available": int(available), "frames_selected": int(selected),
            "frames_registered": int(registered), "selected_fraction_of_capture": selected / available,
            "retained_capture_fraction": registered / available, "note": note}
    path.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


def run_structural(prediction: Path, variant: Path, out: Path, provenance: Path, height_scale_hint=None):
    """run_pipeline.py from-prediction, then evaluate_floorplan.py for EVERY complete footprint hypothesis (none selected vs truth)."""
    pipe = out / "pipeline"
    res = {"pipeline_command": None, "hypotheses": []}
    rec = run_command([sys.executable, HERE / "run_pipeline.py", "from-prediction", "--prediction", prediction, "--output", pipe], cwd=HERE)
    res["pipeline_command"] = rec["command"]
    res["pipeline_returncode"] = rec.get("returncode")
    res["pipeline_duration_seconds"] = rec["duration_seconds"]
    if rec.get("returncode") != 0:
        res["pipeline_stderr_tail"] = rec.get("stderr_tail")
        return res
    man_path = pipe / "pipeline_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    res["pipeline_manifest_sha256_prefix"] = sha_prefix(man_path)
    summ = man.get("summary", {})
    res["structural_axis_hypotheses_status"] = summ.get("structural_axis_hypotheses")
    res["stage_status"] = summ.get("stage_status")
    res["polygon_complete"] = summ.get("polygon_complete")
    res["footprint_complete"] = summ.get("footprint_complete")
    res["stage_failure_reasons"] = summ.get("failure_reasons")
    ks = sorted(int(m.group(1)) for s in man["stages"] for m in [re.match(r"footprint_axis_hypothesis_(\d+)$", s["name"])] if m)
    for k in ks:
        fp, hh = pipe / f"footprint_axis_hypothesis_{k}.json", pipe / f"height_axis_hypothesis_{k}.json"
        row = {"hypothesis": k, "footprint_status": None, "footprint_sha256_prefix": sha_prefix(fp), "height_sha256_prefix": sha_prefix(hh)}
        fdoc = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
        hdoc = json.loads(hh.read_text(encoding="utf-8")) if hh.exists() else None
        row["footprint_status"] = fdoc.get("status") if fdoc else "missing"
        row["footprint_incomplete_reasons"] = [b.get("reason") for b in (fdoc or {}).get("incomplete_boundaries", [])]
        row["height_status"] = hdoc.get("status") if hdoc else "missing"
        row["height_separation_unscaled"] = (hdoc or {}).get("separation_unscaled")
        if fdoc and fdoc.get("status") == "complete_polygon_hypothesis" and hh.exists():
            ev_dir = out / f"floorplan_eval_hyp{k}"
            ev = run_command([sys.executable, HERE / "evaluate_floorplan.py", "--prediction", fp, "--capture", variant, "--height", hh,
                              "--manifest", provenance, "--output-dir", ev_dir], cwd=HERE)
            row["evaluate_command"] = ev["command"]
            row["evaluate_returncode"] = ev.get("returncode")
            ej = ev_dir / "evaluation.json"
            if ev.get("returncode") == 0 and ej.exists():
                doc = json.loads(ej.read_text(encoding="utf-8"))
                m = doc.get("metrics") or {}
                row["evaluation"] = {"path": str(ej), "sha256_prefix": sha_prefix(ej), "status": doc.get("status"),
                                     "failure_reasons": doc.get("failure_reasons"),
                                     "iou": m.get("iou"), "area_ratio": m.get("area_ratio"),
                                     "predicted_edge_count": m.get("predicted_edge_count"), "true_edge_count": m.get("true_edge_count"),
                                     "max_abs_wall_error_cm": m.get("max_abs_wall_error_cm"), "max_rel_wall_error_percent": m.get("max_rel_wall_error_percent"),
                                     "edge_errors": m.get("edge_errors"),
                                     "prediction_scale": (doc.get("scale") or {}).get("prediction_scale"),
                                     "mirrored_by_evaluator_oracle": (doc.get("alignment") or {}).get("mirrored"),
                                     "custom_1_5cm_pass": doc["custom_1_5cm_horizontal_diagnostic"]["pass"],
                                     "wall_3pct_pass_oracle_normalised": doc["wall_length_3_percent_threshold_oracle_normalised"]["pass"],
                                     "whole_room_claim": doc.get("whole_room_claim"), "retained_capture": doc.get("retained_capture")}
            else:
                row["evaluate_stderr_tail"] = ev.get("stderr_tail")
        else:
            row["evaluation"] = {"status": "not_run", "reason": "footprint not complete" if not (fdoc and fdoc.get("status") == "complete_polygon_hypothesis") else "height json missing"}
        res["hypotheses"].append(row)
    return res


def height_ratio_oracle(height_separation, similarity_scale, true_height):
    """Oracle diagnostic: estimated floor-ceiling separation after the camera-centre similarity scale, over the true height."""
    if not (height_separation and similarity_scale and true_height):
        return None
    return float(height_separation * similarity_scale / true_height)


# ------------------------------------------------------------------------------------------------ driver
def compare_variant(variant: Path, output: Path, *, frame_step: int = DEFAULT_FRAME_STEP, run_da3_stage: bool = True,
                    run_colmap_stage: bool = True, colmap: Path = DEFAULT_COLMAP, log=print):
    variant, output = Path(variant).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists (never overwrite): {output}")
    rgb = variant / "rgb" / "path_00"
    gt_path = variant / PRIVATE_DIR / "path_00.json"
    available = len(list(rgb.glob("*.png")))
    budget = select_frames(available, frame_step)
    output.mkdir(parents=True)
    frames_dir = output / "selected_frames"
    frame_rows = write_scratch_frames(rgb, budget["selected_indices"], frames_dir)
    doc = {"schema_version": 1, "kind": "backend_comparison", "variant": variant.name, "variant_path": str(variant),
           "started_utc": _utc(), "evaluation_only_scoring": True, "benchmark_stage_complete": False,
           "frame_budget": budget, "selected_frames_sha256_prefix": {r["file"]: r["sha256_prefix"] for r in frame_rows},
           "inputs": {"video_sha256_prefix": sha_prefix(variant / "videos" / "path_00.mp4"),
                      "private_pose_file_sha256_prefix_evaluation_only": sha_prefix(gt_path),
                      "appearance_manifest_sha256_prefix": sha_prefix(variant / "appearance_manifest.json")},
           "note_inputs": "DA3 and COLMAP receive only the selected RGB frames in selected_frames/; truth is read only for scoring"}
    gt_poses = json.loads(gt_path.read_text(encoding="utf-8"))["poses"]
    scene = json.loads((variant / PRIVATE_DIR / "scene.json").read_text(encoding="utf-8"))
    selected = budget["selected_indices"]
    da3_models = colmap_models = None

    # (a)+(c)+(d) DA3
    da3 = {"status": "skipped"}
    if run_da3_stage:
        log(f"[{variant.name}] DA3 on {len(selected)} frames")
        da3_dir = output / "da3"
        da3 = {"inference": run_da3(frames_dir, da3_dir, len(selected))}
        raw = da3_dir / "prediction.npz"
        if da3["inference"]["returncode"] == 0 and raw.exists():
            fixed = da3_dir / "prediction_source_indices_remapped.npz"
            remap_source_indices(raw, selected, fixed)
            da3["prediction_raw_sha256_prefix"], da3["prediction_remapped_sha256_prefix"] = sha_prefix(raw), sha_prefix(fixed)
            da3["remap_note"] = "infer_rgb numbers scratch images 0..N-1; source_indices rewritten to original frame indices in the copy"
            with np.load(fixed) as d:
                ex, si = np.asarray(d["extrinsics"], float), np.asarray(d["source_indices"], int)
            da3["registered_frames_note"] = "DA3 predicts a pose for every supplied frame (no registration failure mode); this does not imply correct poses"
            da3_models = [{"model_id": "whole_video", "indices": si.tolist(), "w2c": ex}]
            da3["pose_score"] = score_backend(da3_models, gt_poses, selected, available)
            prov = output / "da3" / "provenance.json"
            da3["provenance"] = write_provenance(prov, selected=len(selected), registered=len(si), available=available,
                                                 note="frames present in the DA3 prediction; says nothing about pose correctness. retained fraction is vs the full capture")
            head = next((m for m in da3["pose_score"]["models"] if m["model_id"] == "whole_video"), {})
            scale = (head.get("_similarity") or {}).get("scale")
            log(f"[{variant.name}] structural pipeline")
            da3["structural"] = run_structural(fixed, variant, output / "da3", prov)
            h_true = scene.get("height")
            for row in da3["structural"].get("hypotheses", []):
                row["oracle_height_ratio_under_centre_similarity"] = height_ratio_oracle(row.get("height_separation_unscaled"), scale, h_true)
            for m in da3["pose_score"]["models"]:
                m.pop("_similarity", None)
        else:
            da3["failure_verbatim"] = da3["inference"].get("stderr_tail")
    doc["da3"] = da3

    # (b)+(c) COLMAP
    colmap_doc = {"status": "skipped"}
    if run_colmap_stage:
        log(f"[{variant.name}] COLMAP on {len(selected)} frames")
        report, parsed = run_colmap(Path(colmap), frames_dir, output / "colmap")
        colmap_models = models_for_scoring(parsed)
        colmap_doc = {**report, "pose_score": score_backend(colmap_models, gt_poses, selected, available)}
        for m in colmap_doc["pose_score"]["models"]:
            m.pop("_similarity", None)
        colmap_doc["pose_score"].get("headline", {}).pop("_similarity", None)
        colmap_doc["registered_image_fraction"] = colmap_doc["pose_score"]["registered_fraction_of_selected"]
        colmap_doc["floor_plan"] = {"produced": False, "reason": "dense stage not run; COLMAP sparse points alone are not a completed floor plan"}
    doc["colmap"] = colmap_doc

    # head-to-head on common source indices (COLMAP: largest model only)
    if da3_models and colmap_models:
        big = max(colmap_models, key=lambda m: len(m["indices"]))
        if len(big["indices"]):
            doc["head_to_head"] = {"a": "da3_whole_video", "b": f"colmap_largest_model_{big['model_id']}", **head_to_head(da3_models[0], big, gt_poses)}
        else:
            doc["head_to_head"] = {"status": "not_comparable", "reason": "COLMAP registered no frames"}
    else:
        doc["head_to_head"] = {"status": "not_comparable", "reason": "one backend produced no pose set"}
    doc["finished_utc"] = _utc()
    out_json = output / "comparison.json"
    out_json.write_text(json.dumps(_clean(doc), indent=2, allow_nan=False), encoding="utf-8")
    log(f"wrote {out_json}")
    return doc


# ------------------------------------------------------------------------------------------------ summary
def _fmt(v, nd=2, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.{nd}f}"


def summary_rows(docs):
    rows = []
    for d in docs:
        for backend in ("da3", "colmap"):
            b = d.get(backend) or {}
            ps = b.get("pose_score")
            row = {"variant": d["variant"], "backend": backend, "frame_budget": d["frame_budget"]["budget_label"],
                   "frames_selected": d["frame_budget"]["frames_selected"], "status": "skipped"}
            if ps:
                h = ps.get("headline", {})
                row.update(frames_registered=ps["frames_registered_all_models"], registered_fraction=ps["registered_fraction_of_selected"],
                           retained_of_full_capture=ps["retained_capture_fraction_of_full_capture"], models=ps["model_count"],
                           headline_model_frames=h.get("registered_frames"), headline_status=h.get("status"), headline_reason=h.get("reason"))
                if h.get("status") == "scored":
                    c, r = h["centre_aligned_umeyama"], h["rotation_aligned_kabsch"]
                    row.update(centre_rot_median=c["rotation_error_after_centre_alignment"]["geodesic_median_deg"],
                               centre_rot_p90=c["rotation_error_after_centre_alignment"]["geodesic_p90_deg"],
                               centre_rot_max=c["rotation_error_after_centre_alignment"]["geodesic_max_deg"],
                               centre_rmse_frac=c["centre_rmse_fraction_of_full_gt_extent"],
                               kabsch_rot_median=r["geodesic_median_deg"], kabsch_rot_p90=r["geodesic_p90_deg"], kabsch_rot_max=r["geodesic_max_deg"])
                row["status"] = h.get("status", "n/a")
            if backend == "colmap" and b:
                ms = b.get("models", [])
                row["mean_reproj_px"] = ms[0].get("mean_reprojection_error_px_observation_weighted") if ms else None
                row["points3D"] = [m.get("num_points3D") for m in ms]
                row["track_mean"] = [(m.get("track_length") or {}).get("mean") for m in ms]
                row["no_model_reason"] = b.get("no_model_reason")
                row["floor_plan"] = "none: " + b.get("floor_plan", {}).get("reason", "")
                row["runtime_s"] = b.get("runtime_seconds")
            if backend == "da3" and b.get("inference"):
                row["runtime_s"] = b["inference"].get("inference_manifest_elapsed_seconds")
                row["vram_gib"] = b["inference"].get("gpu_peak_memory_gib")
                st = b.get("structural") or {}
                row["axis_status"] = st.get("structural_axis_hypotheses_status")
                row["floor_plan_hypotheses"] = [
                    {"k": h["hypothesis"], "footprint": h["footprint_status"], "eval": (h.get("evaluation") or {}).get("status"),
                     "iou": (h.get("evaluation") or {}).get("iou"), "max_wall_cm": (h.get("evaluation") or {}).get("max_abs_wall_error_cm"),
                     "max_wall_pct": (h.get("evaluation") or {}).get("max_rel_wall_error_percent"),
                     "height_ratio_oracle": h.get("oracle_height_ratio_under_centre_similarity"),
                     "blockers": ((h.get("evaluation") or {}).get("whole_room_claim") or {}).get("blockers"),
                     "why_not": h.get("footprint_incomplete_reasons") or (h.get("evaluation") or {}).get("failure_reasons")}
                    for h in st.get("hypotheses", [])]
            rows.append(row)
        h2h = d.get("head_to_head") or {}
        rows.append({"variant": d["variant"], "backend": "head_to_head_common_frames", "status": h2h.get("status"),
                     "common_count": h2h.get("common_count"), "reason": h2h.get("reason")})
    return rows


def render_markdown(rows):
    lines = ["# Backend comparison (DA3 whole-video vs COLMAP SfM), identical selected frames", "",
             "Evaluation-only scores against private truth. Pose errors: C = rotation error after centre-aligned Umeyama; K = separate Kabsch rotation fit (median/p90/max deg). "
             "Registered fraction counts all COLMAP models; pose numbers are for the largest model (headline). No stage is claimed complete.", "",
             "| variant | backend | budget | registered/selected | models | status | C rot med/p90/max | K rot med/p90/max | centre RMSE / GT extent | reproj px | runtime s | VRAM GiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["backend"] == "head_to_head_common_frames":
            continue
        rot = lambda a, b, c: f"{_fmt(r.get(a), 1)}/{_fmt(r.get(b), 1)}/{_fmt(r.get(c), 1)}" if r.get(a) is not None else "n/a"
        lines.append(f"| {r['variant']} | {r['backend']} | {r['frame_budget']} | {r.get('frames_registered', 'n/a')}/{r['frames_selected']} ({_fmt(r.get('registered_fraction'), pct=True)}) "
                     f"| {r.get('models', 'n/a')} | {r.get('status')} {('- ' + str(r.get('headline_reason'))) if r.get('headline_reason') else ''} "
                     f"| {rot('centre_rot_median', 'centre_rot_p90', 'centre_rot_max')} | {rot('kabsch_rot_median', 'kabsch_rot_p90', 'kabsch_rot_max')} "
                     f"| {_fmt(r.get('centre_rmse_frac'), 3)} | {_fmt(r.get('mean_reproj_px'), 2)} | {_fmt(r.get('runtime_s'), 1)} | {_fmt(r.get('vram_gib'), 2)} |")
    lines += ["", "## Head-to-head on common registered frames", ""]
    for r in rows:
        if r["backend"] == "head_to_head_common_frames":
            lines.append(f"- {r['variant']}: {r['status']} (common frames {r.get('common_count')}) {r.get('reason') or ''}")
    lines += ["", "## COLMAP details", ""]
    for r in rows:
        if r["backend"] == "colmap":
            lines.append(f"- {r['variant']}: models {r.get('models')}, points3D per model {r.get('points3D')}, mean track per model {r.get('track_mean')}; "
                         f"{r.get('no_model_reason') or ''} floor plan: {r.get('floor_plan')}")
    lines += ["", "## DA3 structural pipeline (every co-supported axis hypothesis, none chosen against truth)", ""]
    for r in rows:
        if r["backend"] == "da3":
            lines.append(f"- {r['variant']}: axis status {r.get('axis_status')}")
            for h in r.get("floor_plan_hypotheses", []):
                lines.append(f"    - hyp {h['k']}: footprint {h['footprint']}, eval {h['eval']}, IoU {_fmt(h['iou'], 3)}, max wall error {_fmt(h['max_wall_cm'], 1)} cm / {_fmt(h['max_wall_pct'], 1)} %, "
                             f"est/true height under centre similarity (oracle) {_fmt(h['height_ratio_oracle'], 3)}, whole_room_claim blockers {h['blockers']}, why-not {h['why_not']}")
    return "\n".join(lines) + "\n"


def summarize(paths, output: Path):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    rows = summary_rows(docs)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(_clean({"sources": [{"path": str(p), "sha256_prefix": sha_prefix(p)} for p in paths], "rows": rows,
                                                           "benchmark_stage_complete": False}), indent=2, allow_nan=False), encoding="utf-8")
    (output / "summary.md").write_text(render_markdown(rows), encoding="utf-8")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="CLI-only: runs DA3 (GPU) and COLMAP (CPU)")
    run.add_argument("--variant", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--frame-step", type=int, default=DEFAULT_FRAME_STEP,
                     help="every Nth frame (default 3 => 40 of 120). 1 = dense budget, labelled as a different budget")
    run.add_argument("--skip-da3", action="store_true")
    run.add_argument("--skip-colmap", action="store_true")
    run.add_argument("--colmap", type=Path, default=DEFAULT_COLMAP)
    summ = sub.add_parser("summarize")
    summ.add_argument("--comparisons", type=Path, nargs="+", required=True)
    summ.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        compare_variant(args.variant, args.output, frame_step=args.frame_step, run_da3_stage=not args.skip_da3,
                        run_colmap_stage=not args.skip_colmap, colmap=args.colmap)
    else:
        summarize(args.comparisons, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
