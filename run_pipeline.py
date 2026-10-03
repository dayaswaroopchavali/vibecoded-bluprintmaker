"""Repeatable RGB-only pipeline: capture -> DA3 inference -> planes -> polygon (arbitrary world frame) -> viewer.
Room orientation is unresolved: camera image-up and the camera path are never used as gravity (captures roll and invert).

Every stage is a subprocess of the project CLIs so the recorded command is exactly what ran. Stages write a
per-stage manifest entry (input/output hashes, command, duration, status, failure reasons) and never overwrite an
existing output folder. Reconstruction stages receive only RGB-derived files: private_ground_truth paths are
refused, no seeds, poses or dimensions are passed, and nothing here reads private truth (evaluation does that
separately, see integration_report.py). A failed or incomplete stage is recorded and downstream stages that need
it are marked blocked rather than rescued.

  python run_pipeline.py from-prediction --prediction <prediction.npz> --output <new dir>   [--legacy-upright-diagnostic]
  python run_pipeline.py full --scene-seed 17 --capture-seed 3101 --output <new dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
PRIVATE_DIR = "private_ground_truth"

# Stages the pipeline does not have. They stay named blockers whatever the unit tests say.
NAMED_BLOCKERS = {
    "orientation_resolution": "No geometry-only stage assigns floor/ceiling or a room up; camera image-up and camera path are inadmissible (free roll/tilt/inversion).",
    "orientation_resolution": "No geometry-only stage assigns floor/ceiling or a room up; camera image-up and camera path are inadmissible (free roll/tilt/inversion).",
    "ceiling_height_estimate": "No stage estimates ceiling height; polygons carry floor-plan edges only.",
    "opening_detection": "Doors and windows are not detected; opening widths are unavailable.",
    "wall_correspondence": "Polygon wall ids follow polygon order and are not matched to named ground-truth walls.",
    "metric_scale": "No scale source: every output is in arbitrary units.",
    "light_field": "Not produced by this pipeline.",
    "segmentation": "No semantic or furniture segmentation stage exists.",
    "completion_mesh": "No completed/inpainted mesh is produced; occluded geometry is not filled in.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(paths) -> dict:
    return {str(p): (sha256(p) if Path(p).is_file() else None) for p in paths}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _assert_reconstruction_input(path: Path) -> None:
    if PRIVATE_DIR in {part.lower() for part in Path(path).resolve().parts}:
        raise ValueError(f"refusing private ground truth as a reconstruction input: {path}")


class Recorder:
    """Accumulates stage records and rewrites the manifest after every stage."""

    def __init__(self, manifest_path: Path, header: dict):
        self.path, self.data = manifest_path, {**header, "started_utc": _utc(), "stages": [],
                                               "named_blockers": [{"name": k, "status": "open", "detail": v}
                                                                  for k, v in NAMED_BLOCKERS.items()]}
        self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def add(self, record: dict) -> dict:
        self.data["stages"].append(record); self.flush(); return record

    def status(self, name: str):
        return next((s["status"] for s in self.data["stages"] if s["name"] == name), None)


def run_stage(rec: Recorder, name: str, argv: list, inputs: list, outputs: list, classify=None,
              reconstruction: bool = True, cwd: Path = HERE) -> dict:
    """Run one stage; classify(returncode) may refine ok/incomplete/failed and supply failure reasons."""
    for item in inputs:
        if reconstruction:
            _assert_reconstruction_input(Path(item))
    record = {"name": name, "command": [str(a) for a in argv], "started_utc": _utc(),
              "inputs": _hashes(inputs), "outputs": {}, "status": "failed", "failure_reasons": []}
    started = time.time()
    try:
        done = subprocess.run([str(a) for a in argv], cwd=str(cwd), capture_output=True, text=True)
        record["returncode"] = done.returncode
        record["stdout_tail"], record["stderr_tail"] = done.stdout[-1500:], done.stderr[-1500:]
    except OSError as exc:
        record["failure_reasons"].append(f"could not start process: {exc}")
        done = None
    record["duration_seconds"] = round(time.time() - started, 3)
    if done is not None:
        if done.returncode != 0:
            record["failure_reasons"].append(f"exit code {done.returncode}: " + (done.stderr.strip().splitlines() or ["no stderr"])[-1])
        else:
            record["status"] = "ok"
            if classify is not None:
                status, reasons = classify()
                record["status"] = status; record["failure_reasons"].extend(reasons)
    record["outputs"] = _hashes(outputs)
    for out, digest in record["outputs"].items():
        if digest is None and record["status"] in ("ok", "incomplete"):
            record["status"] = "failed"; record["failure_reasons"].append(f"expected output missing: {out}")
    return rec.add(record)


def blocked(rec: Recorder, name: str, because: str) -> dict:
    return rec.add({"name": name, "status": "blocked", "failure_reasons": [because], "command": None,
                    "inputs": {}, "outputs": {}, "duration_seconds": 0.0})


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dependency_stage(rec: Recorder) -> dict:
    code = ("import sys,cv2,numpy,torch;print(sys.version.split()[0],cv2.__version__,numpy.__version__,"
            "torch.__version__,torch.cuda.is_available())")
    record = run_stage(rec, "dependencies", [sys.executable, "-c", code], [], [], reconstruction=False)
    record["interpreter"] = sys.executable; record["platform"] = platform.platform()
    record["versions"] = (record.get("stdout_tail") or "").split()
    rec.flush()
    return record


MAX_AXIS_HYPOTHESES = 3
ORIENTATION_UNRESOLVED = ("orientation_unresolved: no scene-derived orientation stage exists; camera-up consensus and the camera-path "
                          "plane are inadmissible as gravity because captures may roll, tilt and invert freely")


def reconstruct(rec: Recorder, run_dir: Path, prediction: Path, video: Path | None = None, legacy_upright: bool = False) -> dict:
    """Planes -> polygon -> viewer for one RGB-derived prediction, in arbitrary world coordinates.

    No room "up" is derived from camera image-up or the camera path, so the production polygon stage runs without
    one and reports incomplete/up_axis_unavailable until a geometry-only orientation stage exists. The old upright-phone
    branch is kept as clearly labelled legacy_* diagnostic stages, off by default, never counted as a result and not
    valid for unrestricted (rolled/inverted) captures.
    """
    py = sys.executable
    planes, polygon, cuboid = run_dir / "planes.json", run_dir / "polygon.json", run_dir / "cuboid_baseline.json"

    def classify_planes():
        count = len(_load(planes).get("planes", []))
        return ("ok", []) if count else ("incomplete", ["no plane candidates extracted"])
    run_stage(rec, "planes", [py, HERE / "extract_planes.py", "--prediction", prediction, "--output", planes],
              [prediction, HERE / "extract_planes.py"], [planes], classify_planes)

    # Geometry-only structural axis hypotheses (never camera up / camera path); sign (floor vs ceiling) stays open.
    axes_json = run_dir / "axis_hypotheses.json"

    def classify_axes():
        data = _load(axes_json); status = data.get("status")
        if status == "single_axis_sign_ambiguous":
            return "ok", ["one supported structural axis; its sign (floor vs ceiling) is unresolved and it is not gravity"]
        return "incomplete", [f"{status}: " + ("co-supported axes (symmetric room)" if status == "ambiguous_axes" else "no adequately supported axis")]
    eligible = []
    if rec.status("planes") in ("ok", "incomplete"):
        run_stage(rec, "structural_axis_hypotheses", [py, HERE / "geometry_orientation.py", "--planes", planes, "--prediction", prediction,
                                                      "--output", axes_json], [prediction, planes, HERE / "geometry_orientation.py"], [axes_json], classify_axes)
        if rec.status("structural_axis_hypotheses") in ("ok", "incomplete"):
            axes = _load(axes_json)
            if axes.get("status") in ("single_axis_sign_ambiguous", "ambiguous_axes"):
                eligible = [h for h in axes["hypotheses"] if h["co_supported_with_best"]][:MAX_AXIS_HYPOTHESES]
    else:
        blocked(rec, "structural_axis_hypotheses", "plane stage failed")

    def classify_polygon():
        data = _load(polygon)
        if data.get("status") == "complete_polygon_hypothesis":
            return "ok", []
        reasons = [b.get("reason", "unspecified") for b in data.get("incomplete_boundaries", [])] or [data.get("status", "incomplete")]
        return "incomplete", reasons
    if rec.status("planes") in ("ok", "incomplete"):
        # No --up: a gravity direction is never taken from camera orientation, camera path, or ground truth.
        run_stage(rec, "polygon", [py, HERE / "room_hypothesis.py", "--prediction", prediction, "--planes", planes, "--output", polygon,
                                   "--cuboid-output", cuboid],
                  [prediction, planes, HERE / "room_hypothesis.py"], [polygon, cuboid], classify_polygon)
    else:
        blocked(rec, "polygon", "plane stage failed")
    for k, hyp in enumerate(eligible):
        # Downstream stages run once per co-supported axis hypothesis and keep that label; they never claim gravity.
        up = ",".join(repr(float(x)) for x in hyp["axis_world"])
        hp, hh = run_dir / f"polygon_axis_hypothesis_{k}.json", run_dir / f"height_axis_hypothesis_{k}.json"

        def classify_hyp_polygon(path=hp):
            data = _load(path)
            if data.get("status") == "complete_polygon_hypothesis":
                return "ok", [f"under axis hypothesis {k} (sign unresolved)"]
            return "incomplete", [b.get("reason", "unspecified") for b in data.get("incomplete_boundaries", [])] or [data.get("status", "incomplete")]
        run_stage(rec, f"polygon_axis_hypothesis_{k}", [py, HERE / "room_hypothesis.py", "--prediction", prediction, "--planes", planes, "--output", hp, f"--up={up}"],
                  [prediction, planes, axes_json, HERE / "room_hypothesis.py"], [hp], classify_hyp_polygon)

        fp = run_dir / f"footprint_axis_hypothesis_{k}.json"

        def classify_fp(path=fp):
            data = _load(path)
            return ("ok", [f"floor/ceiling observed-support outline under axis hypothesis {k} (sign unresolved)"]) if data.get("status") == "complete_polygon_hypothesis" \
                else ("incomplete", [b.get("reason", "unspecified") for b in data.get("incomplete_boundaries", [])] or [data.get("status", "incomplete")])
        run_stage(rec, f"footprint_axis_hypothesis_{k}", [py, HERE / "footprint.py", "--prediction", prediction, "--planes", planes, "--orientation", axes_json,
                  "--hypothesis", k, "--output", fp], [prediction, planes, axes_json, HERE / "footprint.py"], [fp], classify_fp)
        def classify_hyp_height(path=hh):
            data = _load(path)
            return ("ok", ["arbitrary-unit separation between the axis-bracketing planes; floor/ceiling labels unresolved"]) if data.get("status") == "estimated_unscaled_separation" \
                else ("incomplete", data.get("reasons", [data.get("status", "incomplete")]))
        run_stage(rec, f"height_axis_hypothesis_{k}", [py, HERE / "estimate_height.py", "--prediction", prediction, "--planes", planes, f"--up={up}", "--output", hh],
                  [prediction, planes, axes_json, HERE / "estimate_height.py"], [hh], classify_hyp_height)
    rec.add({"name": "orientation", "status": "incomplete", "command": None, "inputs": {}, "outputs": {},
             "failure_reasons": [("axis hypotheses exist but floor vs ceiling (sign) is unresolved and symmetric axes stay co-supported" if eligible
                                  else ORIENTATION_UNRESOLVED)], "duration_seconds": 0.0})

    if legacy_upright:
        legacy_diagnostics(rec, run_dir, prediction, planes)

    viewer_out = run_dir / "viewer.html"
    run_stage(rec, "viewer", [py, HERE / "viewer.py", "--pipeline-run", run_dir, "--output", viewer_out] + (["--video", video] if video else []),
              [prediction, HERE / "viewer.py"], [viewer_out])
    return rec.data


def legacy_diagnostics(rec: Recorder, run_dir: Path, prediction: Path, planes: Path) -> None:
    """Upright-phone camera-up branch. LEGACY DIAGNOSTIC ONLY: invalid for rolled/inverted captures, never a result."""
    py = sys.executable
    prior, polygon = run_dir / "legacy_up_prior.json", run_dir / "legacy_polygon.json"
    box_orientation, orientation = run_dir / "legacy_box_orientation.json", run_dir / "legacy_orientation.json"
    label = ["LEGACY upright-phone assumption: not valid for unrestricted capture; not a pipeline result"]

    def classify_prior():
        data = _load(prior)
        return ("ok", label) if data.get("status") == "declared_prior" else ("incomplete", label + [data.get("reason", "up prior refused")])
    run_stage(rec, "legacy_camera_up_prior_diagnostic", [py, HERE / "orient_room.py", "--prior-only", "--prediction", prediction, "--output", prior],
              [prediction, HERE / "orient_room.py"], [prior], classify_prior)
    if rec.status("legacy_camera_up_prior_diagnostic") != "ok":
        return
    up = ",".join(repr(float(x)) for x in _load(prior)["up_world"])

    def classify_polygon():
        data = _load(polygon)
        return ("ok" if data.get("status") == "complete_polygon_hypothesis" else "incomplete",
                label + [b.get("reason", "?") for b in data.get("incomplete_boundaries", [])])
    run_stage(rec, "legacy_upright_polygon_diagnostic", [py, HERE / "room_hypothesis.py", "--prediction", prediction, "--planes", planes,
                                                       "--output", polygon, f"--up={up}"],
              [prediction, planes, HERE / "room_hypothesis.py"], [polygon], classify_polygon)

    def classify_orient(path):
        def inner():
            data = _load(path)
            return ("ok" if data.get("status", "").startswith("oriented") else "incomplete", label + ([data["reason"]] if data.get("reason") else []))
        return inner
    run_stage(rec, "legacy_orientation_diagnostic", [py, HERE / "orient_room.py", "--hypothesis", polygon, "--prediction", prediction, "--output", orientation],
              [prediction, polygon], [orientation], classify_orient(orientation))
    cuboid = run_dir / "cuboid_baseline.json"
    if cuboid.is_file():
        run_stage(rec, "legacy_box_orientation_diagnostic", [py, HERE / "orient_room.py", "--hypothesis", cuboid, "--prediction", prediction,
                  "--output", box_orientation, "--svg", run_dir / "legacy_box_orientation.svg"], [prediction, cuboid], [box_orientation], classify_orient(box_orientation))


def summarize(rec: Recorder) -> dict:
    stages = {s["name"]: s for s in rec.data["stages"]}
    legacy = {k: v["status"] for k, v in stages.items() if k.startswith("legacy_")}
    hyp_polygons = {k: v["status"] for k, v in stages.items() if k.startswith("polygon_axis_hypothesis_")}
    hyp_heights = {k: v["status"] for k, v in stages.items() if k.startswith("height_axis_hypothesis_")}
    hyp_footprints = {k: v["status"] for k, v in stages.items() if k.startswith("footprint_axis_hypothesis_")}
    axes_status = stages.get("structural_axis_hypotheses", {}).get("status")
    rec.data["summary"] = {
        "stage_status": {k: v["status"] for k, v in stages.items() if not k.startswith("legacy_")}, "structural_axis_hypotheses": axes_status,
        "legacy_diagnostic_stage_status": legacy,
        "polygon_complete": any(s == "ok" for s in hyp_polygons.values()),
        "footprint_complete": any(s == "ok" for s in hyp_footprints.values()),
        "polygon_basis": "structural axis hypothesis (sign unresolved, not gravity)" if hyp_polygons else "none: no supported axis hypothesis",
        "axis_hypothesis_polygons": hyp_polygons, "axis_hypothesis_heights": hyp_heights, "axis_hypothesis_footprints": hyp_footprints,
        "orientation_status": ("axis_hypothesis_sign_ambiguous" if len(hyp_polygons) == 1 else "axis_hypotheses_co_supported" if hyp_polygons else "unresolved"),
        "floor_ceiling_labels": "unresolved",
        "failure_reasons": {k: v["failure_reasons"] for k, v in stages.items() if v["failure_reasons"] and not k.startswith("legacy_")},
        "claims": "none: diagnostic run; no stage gate, ten-house, or assessment completion is claimed",
    }
    rec.data["finished_utc"] = _utc(); rec.flush()
    return rec.data["summary"]


def cmd_from_prediction(args) -> int:
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"output already exists (never overwrite frozen results): {out}")
    prediction = args.prediction.resolve()
    _assert_reconstruction_input(prediction)
    out.mkdir(parents=True)
    rec = Recorder(out / "pipeline_manifest.json", {"mode": "from_prediction", "inference": "reused_frozen_prediction_not_rerun",
                                                   "prediction": str(prediction), "prediction_sha256": sha256(prediction),
                                                   "interpreter": sys.executable})
    dependency_stage(rec)
    rec.add({"name": "inference", "status": "ok", "command": None, "reused": True, "inputs": {}, "outputs": _hashes([prediction]),
             "failure_reasons": [], "duration_seconds": 0.0, "note": "frozen prediction reused; not copied or modified"})
    reconstruct(rec, out, prediction, args.video, args.legacy_upright_diagnostic)
    print(json.dumps(summarize(rec)["stage_status"]))
    return 0


def cmd_full(args) -> int:
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"output already exists (never overwrite frozen results): {out}")
    out.mkdir(parents=True)
    capture, private_in = out / "capture", out / "generator_private"
    private_in.mkdir()
    py = sys.executable
    top = Recorder(out / "pipeline_manifest.json", {"mode": "full", "scene_seed_note": "generator-side only; never passed to reconstruction",
                                                   "capture_seed": args.capture_seed, "interpreter": sys.executable,
                                                   "parameters": {"frames": args.frames, "width": args.width, "height": args.height,
                                                                  "fps": args.fps, "fov": args.fov, "infer_frames": args.infer_frames,
                                                                  "resolution": args.resolution}})
    dependency_stage(top)
    scene_path = private_in / "scene_input.json"
    gen = ("import json,sys;sys.path.insert(0,r'%s');import simulator;"
           "open(sys.argv[1],'w',encoding='utf-8').write(json.dumps(simulator.make_scene(int(sys.argv[2])),indent=2))" % HERE)
    run_stage(top, "scene_generation", [py, "-c", gen, scene_path, str(args.scene_seed)], [], [scene_path], reconstruction=False)
    if top.status("scene_generation") != "ok":
        summarize(top); return 1
    run_stage(top, "capture", [py, HERE / "blender_capture.py", "--blender", args.blender, "--scene", scene_path, "--output", capture,
                               "--capture-seed", args.capture_seed, "--width", args.width, "--height", args.height,
                               "--frames", args.frames, "--fov", args.fov, "--fps", args.fps],
              [scene_path], [capture / "videos" / f"path_{i:02d}.mp4" for i in range(3)] + [capture / PRIVATE_DIR / "coverage_diagnostics.json"],
              reconstruction=False)
    if top.status("capture") != "ok":
        summarize(top); return 1
    for i in range(3):
        name = f"path_{i:02d}"; run_dir = out / name; run_dir.mkdir()
        video = capture / "videos" / f"{name}.mp4"
        rec = Recorder(run_dir / "pipeline_manifest.json", {"mode": "full_path", "path": name, "interpreter": sys.executable,
                                                           "input_video": str(video)})
        rec.add({**top.data["stages"][0], "note": "copied from run-level dependency check"})
        # GPU inference: strictly sequential, one path at a time, RGB video only.
        run_stage(rec, "inference", [py, HERE / "infer_rgb.py", "--input", video, "--output", run_dir / "da3", "--frames", args.infer_frames,
                                     "--resolution", args.resolution],
                  [video, HERE / "infer_rgb.py"], [run_dir / "da3" / "prediction.npz", run_dir / "da3" / "manifest.json"])
        if rec.status("inference") != "ok":
            for stage in ("planes", "polygon", "orientation"):
                blocked(rec, stage, "inference failed")
        else:
            reconstruct(rec, run_dir, run_dir / "da3" / "prediction.npz", video, args.legacy_upright_diagnostic)
        summarize(rec)
        top.add({"name": f"path:{name}", "status": "ok" if rec.status("polygon") == "ok" else "incomplete", "command": None,
                 "inputs": {}, "outputs": {str(run_dir / "pipeline_manifest.json"): sha256(run_dir / "pipeline_manifest.json")},
                 "failure_reasons": rec.data["summary"]["failure_reasons"].get("polygon", []), "duration_seconds": 0.0})
    summarize(top)
    print(json.dumps({s["name"]: s["status"] for s in top.data["stages"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("from-prediction", help="reuse a frozen prediction.npz; inference is not rerun")
    one.add_argument("--prediction", type=Path, required=True); one.add_argument("--output", type=Path, required=True)
    one.add_argument("--video", type=Path, help="optional input video to show in the viewer")
    one.add_argument("--legacy-upright-diagnostic", action="store_true", help="also run the upright-phone camera-up branch as labelled legacy diagnostics")
    one.set_defaults(func=cmd_from_prediction)
    full = sub.add_parser("full", help="render captures, run RGB-only inference sequentially, reconstruct")
    full.add_argument("--scene-seed", type=int, required=True); full.add_argument("--capture-seed", type=int, required=True)
    full.add_argument("--output", type=Path, required=True); full.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    full.add_argument("--frames", type=int, default=120); full.add_argument("--width", type=int, default=480)
    full.add_argument("--height", type=int, default=360); full.add_argument("--fov", type=float, default=78.)
    full.add_argument("--fps", type=int, default=6); full.add_argument("--infer-frames", type=int, default=36)
    full.add_argument("--resolution", type=int, default=560)
    full.add_argument("--legacy-upright-diagnostic", action="store_true", help="also run the upright-phone camera-up branch as labelled legacy diagnostics")
    full.set_defaults(func=cmd_full)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
