"""Runner for track_ba: `infer` (truth-free, writes + freezes outputs) and `evaluate` (post-freeze, oracle only).

infer    python run_track_ba.py infer --prediction PRED.npz --candidates CANDIDATES.json --out NEW_DIR [--label L]
evaluate python run_track_ba.py evaluate --run NEW_DIR --gt-poses path_00.json --scene scene.json --out NEW_DIR2

`infer` never reads ground truth. `evaluate` first verifies the sha256 freeze manifest, then compares DA3-only vs
refined camera trajectories (audit_camera_trajectory API) and ceiling-normalised room extents against the private
scene. Oracle alignment is diagnostic only and is never exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import track_ba as tb


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as s:
        for block in iter(lambda: s.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_prediction(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def infer(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    pred = _load_prediction(args.prediction)
    n = len(pred["extrinsics"])
    t0 = time.time()
    tcfg = tb.TrackConfig()
    bcfg = tb.BAConfig(depth_prior_weight=args.depth_prior_weight, refine_focal=args.refine_focal, loss=args.loss,
                       rot_prior_deg=args.rot_prior_deg, pos_prior_frac=args.pos_prior_frac)
    h, w = pred["images"].shape[1:3]
    if args.video:
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA))
        full = tb.track_features(frames, tcfg)
        stats_full = tb.track_statistics(full, len(frames))
        tracks = tb.restrict_to_frames(full, pred["source_indices"])
        stats = tb.track_statistics(tracks, n)
        stats["full_video_tracking"] = stats_full
        per_frame = full["per_frame"]
        stats["video_frames_tracked"] = len(frames)
    else:
        tracks = tb.track_features(pred["images"], tcfg)
        stats = tb.track_statistics(tracks, n)
        per_frame = tracks["per_frame"]
    stats["per_frame_active_tracked_sequence"] = [p["active"] for p in per_frame]
    stats["loss_reasons_total"] = {k: int(sum(p["lost"][k] for p in per_frame)) for k in per_frame[-1]["lost"]}
    t_track = time.time() - t0
    sampler = tb.sample_depth_prior(pred) if args.depth_prior_weight > 0 else None
    ba_input = dict(pred)
    if args.square_pixels:   # truth-free prior: square pixels, f = mean(fx, fy) of the DA3 intrinsics
        Kq = np.array(pred["intrinsics"], dtype=float)
        f_sq = 0.5 * (Kq[:, 0, 0] + Kq[:, 1, 1])
        Kq[:, 0, 0] = f_sq
        Kq[:, 1, 1] = f_sq
        ba_input["intrinsics"] = Kq
    ref = tb.refine_prediction(ba_input, tracks, bcfg, depth_sampler=sampler)
    report = {"label": args.label, "prediction": str(args.prediction), "prediction_sha256": sha256(Path(args.prediction)),
              "config": tb.config_dict(tcfg, bcfg), "seconds_tracking": t_track, "seconds_total": time.time() - t0,
              "track_statistics": stats, "ba_ok": bool(ref["ok"]), "square_pixels": bool(args.square_pixels), "labels": {
                  "truth_used": False, "uses_da3_depth": bool(args.depth_prior_weight > 0),
                  "note": "poses/intrinsics from DA3 initialise BA; tracking runs on the images stored in the prediction"}}
    if not ref["ok"]:
        report["ba_failure"] = {"reason": ref.get("reason"), "info": ref.get("info")}
        (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
        _freeze(out)
        return report
    report["ba"] = {k: ref[k] for k in ("info", "rounds", "residual_stats_before_first_round_px",
                                        "residual_stats_final_px", "gauge", "prior", "camera_centre_shift",
                                        "rotation_shift_deg", "focal_scale", "final_gate", "point_quality")}
    report["depth_consistency_with_da3"] = tb.depth_consistency(pred, ref["points"], ref["point_view_ids"])
    # refined prediction: same npz schema; DA3 depth/confidence are carried over unchanged (not re-rendered)
    refined = dict(pred)
    refined["extrinsics"] = tb.refined_extrinsics(ref)
    refined["intrinsics"] = ref["K_refined"].astype(np.float32)
    refined["sparse_points"] = ref["points"].astype(np.float32)
    refined["sparse_point_view_ids"] = ref["point_view_ids"].astype(np.int64)
    refined["sparse_point_track_lengths"] = ref["point_track_lengths"].astype(np.int64)
    np.savez_compressed(out / "refined_prediction.npz", **refined)
    # scale-bias test (truth-free): every co-supported structural-axis candidate, sparse vs DA3 dense
    cand = json.loads(Path(args.candidates).read_text(encoding="utf-8"))["candidates"]
    dense_pts, dense_ids = tb.dense_points_per_view(pred)
    sb = []
    for ci, c in enumerate(cand):
        rep = tb.scale_bias_report(ref["points"], ref["point_view_ids"], dense_pts, dense_ids, c["axis_world"])
        rep["candidate_id"] = c.get("id", ci)
        rep["axis_world"] = c["axis_world"]
        sb.append(rep)
    report["scale_bias"] = sb
    (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    _freeze(out)
    return report


def _freeze(out: Path):
    files = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "freeze.json"}
    (out / "freeze.json").write_text(json.dumps({"frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                 "files": files}, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(args):
    import audit_camera_trajectory as audit
    run = Path(args.run)
    frozen = json.loads((run / "freeze.json").read_text(encoding="utf-8"))["files"]
    for name, digest in frozen.items():
        if sha256(run / name) != digest:
            raise SystemExit(f"frozen file changed: {name}")
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    base_pred = Path(report["prediction"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    gt = Path(args.gt_poses)
    res = {"label": report["label"], "freeze_verified": True, "oracle_only": True}
    a0 = audit.audit_path(base_pred, gt, "da3_only")
    a1 = audit.audit_path(run / "refined_prediction.npz", gt, "refined")

    def brief(a):
        m = a["metrics"]
        return {"position_rmse_m": m["position_rmse_m"], "position_rmse_fraction_of_extent": m["position_rmse_fraction_of_gt_extent"],
                "max_position_error_m": m["max_position_error_m"], "gt_extent_m": m["groundtruth_trajectory_extent_m"],
                "geodesic_mean_deg_centre_aligned": m["orientation"]["geodesic_mean_deg"],
                "geodesic_p90_deg_centre_aligned": m["orientation"]["geodesic_p90_deg"],
                "geodesic_mean_deg_kabsch": m["orientation_extended"]["rotation_kabsch_aligned"]["geodesic_mean_deg"],
                "up_mean_deg": m["orientation"]["up_mean_deg"], "oracle_scale": a["oracle_alignment"]["uniform_scale"]}
    res["camera_pose"] = {"da3_only": brief(a0), "refined": brief(a1)}
    rot = np.asarray(a1["oracle_alignment"]["rotation_matrix"])  # refined-pred -> truth world
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    truth_dims = {"width_x": scene["width"], "length_y": scene["length"], "height_z": scene["height"]}
    res["truth_dims"] = truth_dims
    ex, ey, ez = (rot.T @ np.eye(3)).T  # truth axes expressed in the prediction frame
    rows = []
    for s in report.get("scale_bias", []):
        axis = np.asarray(s["axis_world"], float)
        axis /= np.linalg.norm(axis)
        vertical_deg = float(np.degrees(np.arccos(min(1.0, abs(axis @ ez)))))
        row = {"candidate_id": s["candidate_id"], "angle_to_true_vertical_deg": vertical_deg,
               "is_true_vertical": vertical_deg < 10.0}
        for name in ("sparse", "dense_da3"):
            m = s[name]
            entry = {"status": m["status"], "height": m.get("separation")}
            if m.get("separation") and m.get("walls"):
                d = [np.asarray(v) for v in s["horizontal_dirs"]]
                entry["height_error_vs_truth_ratio_unscaled"] = None
                for key in ("extent_mode", "extent_p2_p98"):
                    ratios = {}
                    for dd, w in zip(d, m["walls"]):
                        if w.get(key) is None:
                            continue
                        # the true horizontal axis best aligned with this direction
                        ax_name, ax_vec, true_len = max((("width_x", ex, truth_dims["width_x"]),
                                                         ("length_y", ey, truth_dims["length_y"])),
                                                        key=lambda a_: abs(a_[1] @ dd))
                        truth_ratio = true_len / truth_dims["height_z"]
                        ratios[ax_name] = {"est_ratio": w[key] / m["separation"], "truth_ratio": truth_ratio,
                                           "rel_error": (w[key] / m["separation"]) / truth_ratio - 1.0,
                                           "alignment_cos": float(abs(ax_vec @ dd))}
                    entry["ratio_" + key] = ratios
            row[name] = entry
        rows.append(row)
    res["ceiling_normalised_wall_ratio_vs_truth"] = rows
    res["note"] = ("ratio = wall extent / floor-to-ceiling separation from the cloud itself; truth ratio = room width|length "
                   "/ height. Only the candidate with is_true_vertical is a floor/ceiling measurement; evaluate_floorplan "
                   "semantic scoring needs a polygon and was not run (no footprint is produced by this tracker).")
    (out / "evaluation.json").write_text(json.dumps(_jsonable(res), indent=2), encoding="utf-8")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("infer")
    a.add_argument("--prediction", required=True)
    a.add_argument("--candidates", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--label", default="run")
    a.add_argument("--video", default=None, help="track over every video frame; keep observations at source_indices")
    a.add_argument("--depth-prior-weight", type=float, default=0.0)
    a.add_argument("--refine-focal", action="store_true")
    a.add_argument("--loss", default="huber")
    a.add_argument("--square-pixels", action="store_true")
    a.add_argument("--rot-prior-deg", type=float, default=2.0)
    a.add_argument("--pos-prior-frac", type=float, default=0.05)
    b = sub.add_parser("evaluate")
    b.add_argument("--run", required=True)
    b.add_argument("--gt-poses", required=True)
    b.add_argument("--scene", required=True)
    b.add_argument("--out", required=True)
    args = p.parse_args(argv)
    res = infer(args) if args.cmd == "infer" else evaluate(args)
    print(json.dumps(_jsonable(res), indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
