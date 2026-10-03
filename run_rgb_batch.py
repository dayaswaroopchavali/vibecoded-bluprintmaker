"""Stage-1 simulator -> blueprint RGB benchmark batch: 10 houses x 3 independently seeded UPRIGHT camera videos.

Per (house, path) run, strictly serial and resumable:
  capture   Blender EEVEE render of the wide perimeter path with roll forced to 0 (upright protocol) + capture-side QA
  inference DA3-LARGE whole-video inference (infer_rgb.py, frozen settings, one GPU process at a time)
  chain     TRUTH-FREE: planes -> axis_candidates (EVERY candidate) -> estimate_room_height -> wall_extraction +
            assemble_boundary_ray (k=2.0)   [run_axis_candidates.run_extraction]
  freeze    sha256 of the video, prediction and every chain output -> freeze.json
and only afterwards, as a separate command, the EVALUATION-ONLY stage (private scene read, evaluator-oracle 3.0 m
normalised protocol, every candidate scored, gates exactly as coded in evaluate_floorplan.py, no gate weakened).

    python run_rgb_batch.py init     --output-dir <NEW folder>              # writes the predeclared plan/split
    python run_rgb_batch.py run      --output-dir <dir> [--houses 17] [--paths 0] [--max-hours 3]
    python run_rgb_batch.py evaluate --output-dir <dir> [--allow-partial]   # separate stage, reads private truth

Nothing here fits or applies a calibration. The cap-vs-wall scale ratio is only recorded (evaluation part) for a later,
separately labelled cross-house experiment. See RGB_BATCH.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "outputs" / "sim-benchmark"
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")

# ------------------------------------------------------------------ predeclared plan (fixed before any run)
HOUSE_SEEDS = (17, 101, 102, 103, 104, 105, 106, 107, 108, 109)   # simulator.make_scene seeds: 17 + the next nine used by the QA/LiDAR batches
SPLIT_SIZE = 5
PATHS_PER_HOUSE = 3
CAPTURE_SEED = 3101                       # root of simulator.path_seeds -> 3 independent path seeds per house
PROTOCOLS = ("upright", "rolling")        # rolling keeps the generator's roll sweep; implemented, NOT run in stage 1
SETTINGS = {"frames": 120, "width": 480, "height": 360, "fov": 78.0, "fps": 6, "appearance": "original",
            "da3_frames": 120, "da3_resolution": 480, "da3_seed": 2026, "ray_k": 2.0, "frames_available": 120,
            "ceiling_normalised_m": 3.0}
DISTINCT_MIN_DIFF_M = 0.20               # two houses are distinct when some of (width, length, height) differs by >= this
MAX_HOURS_DEFAULT = 3.0
STAGES = ("capture", "inference", "chain", "freeze")
# modules whose source hashes pin "same frozen settings for every run"
CODE_FILES = ("run_axis_candidates.py", "compare_wall_boundaries.py", "axis_candidates.py", "height_diagnosis.py",
              "wall_extraction.py", "boundary_assembly.py", "extract_planes.py", "estimate_height.py", "infer_rgb.py",
              "simulator.py", "blender_capture.py", "capture_parallax.py", "evaluate_floorplan.py")
# Functions that must stay free of truth: no scene, private_ground_truth, evaluator or oracle names (see test_run_rgb_batch.py).
TRUTH_FREE_FUNCTIONS = ("stage_inference", "stage_chain", "stage_freeze", "run_truth_free_stages", "verify_freeze")
FORBIDDEN_TOKENS = ("private_ground_truth", "scene", "evaluate_floorplan", "oracle", "truth", "load_scene")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new_json(path, obj) -> str:
    """Write JSON to a NEW file (FileExistsError if it exists). Returns the sha256 of the bytes written."""
    data = json.dumps(obj, indent=1, allow_nan=False, sort_keys=False).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        stream.write(data)
    return sha256_bytes(data)


def write_new_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(text)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hashes() -> dict:
    return {name: (sha256_file(HERE / name) if (HERE / name).is_file() else None) for name in CODE_FILES}


# ------------------------------------------------------------------ plan / split
def predeclared_split(seeds=HOUSE_SEEDS, size: int = SPLIT_SIZE) -> dict:
    """Houses sorted ascending by seed id: first `size` = calibration split, last `size` = held-out. No other inputs."""
    ordered = sorted(int(s) for s in seeds)
    if len(set(ordered)) != len(ordered):
        raise ValueError("house seeds must be distinct")
    if len(ordered) != 2 * size:
        raise ValueError(f"need exactly {2 * size} houses, got {len(ordered)}")
    return {"rule": "houses sorted ascending by seed id; first 5 = calibration split, last 5 = held-out; fixed before any run, "
                    "no selection after seeing results",
            "calibration": ordered[:size], "held_out": ordered[size:]}


def check_distinct_houses(scenes: dict, min_diff: float = DISTINCT_MIN_DIFF_M) -> dict:
    """scenes: {seed: scene dict}. Distinct = (width, length, height) differ by >= min_diff m in at least one component."""
    keys = sorted(scenes)
    dims = {k: (scenes[k]["width"], scenes[k]["length"], scenes[k]["height"]) for k in keys}
    clashes = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
               if max(abs(x - y) for x, y in zip(dims[a], dims[b])) < min_diff]
    return {"distinct": not clashes, "clashes": clashes, "min_diff_m": min_diff,
            "dimensions_m": {str(k): [round(v, 3) for v in dims[k]] for k in keys}}


def run_id(house: int, path: int) -> str:
    return f"house{int(house)}_path{int(path)}"


def run_plan(seeds=HOUSE_SEEDS, paths: int = PATHS_PER_HOUSE, capture_seed: int = CAPTURE_SEED, scene_maker=None) -> list:
    """House-major order (house ascending, then path index). Path seeds come from simulator.path_seeds(scene, capture_seed)."""
    if scene_maker is None:
        import simulator
        scene_maker = simulator.make_scene
        seeder = simulator.path_seeds
    else:
        seeder = lambda scene, cs: [int(cs * 1009 + i) for i in range(PATHS_PER_HOUSE)]
    split = predeclared_split(seeds)
    plan = []
    for house in sorted(int(s) for s in seeds):
        pseeds = seeder(scene_maker(house), capture_seed)
        for p in range(paths):
            plan.append({"run_id": run_id(house, p), "house_seed": house, "path_index": p, "path_seed": int(pseeds[p]),
                         "split": "calibration" if house in split["calibration"] else "held_out"})
    return plan


def build_config(protocol: str = "upright") -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    return {"schema": "rgb_batch_config_v1", "stage": "stage_1_single_room_rgb", "protocol": protocol,
            "protocol_note": "upright = make_wide_path poses with roll forced to 0; rolling = generator roll sweep (not run in stage 1)",
            "house_seeds": list(HOUSE_SEEDS), "house_rule": "seed 17 plus the next nine seeds 101..109 of simulator.make_scene "
            "(the ten houses of wide-capture-qa-10houses-widefix-v1); distinct = some footprint/height component differs by >= 0.20 m",
            "capture_seed": CAPTURE_SEED, "paths_per_house": PATHS_PER_HOUSE, "settings": SETTINGS,
            "split": predeclared_split(), "plan": run_plan(),
            "gates": {"custom_1_5cm": 1.5, "wall_relative_percent": 3.0, "ceiling_height_error_cm": 1.5,
                      "height_spread_cm": 1.0, "repeatability_abs_m": 0.01, "repeatability_rel_of_true_wall": 0.005,
                      "changed": False},
            "code_sha256": code_hashes()}


def config_fingerprint(cfg: dict) -> str:
    """Hash of everything that must not change between runs of one batch (not the selection filters)."""
    return sha256_bytes(json.dumps({k: cfg[k] for k in ("protocol", "house_seeds", "capture_seed", "paths_per_house", "settings",
                                                        "split", "plan", "gates", "code_sha256")}, sort_keys=True).encode())


def init_root(root: Path, protocol: str = "upright") -> dict:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"output folder already exists (never overwrite): {root}")
    cfg = build_config(protocol)
    root.mkdir(parents=True)
    write_new_json(root / "batch_config.json", cfg)
    write_new_text(root / "PREDECLARED.md", predeclaration_markdown(cfg))
    return cfg


def predeclaration_markdown(cfg: dict) -> str:
    sp = cfg["split"]
    lines = ["# Predeclared before any run (rgb batch, stage 1)", "", f"- written: {_utc()}",
             f"- protocol: {cfg['protocol']} ({cfg['protocol_note']})",
             f"- houses: {cfg['house_seeds']} - {cfg['house_rule']}",
             f"- capture seed {cfg['capture_seed']} -> path seeds per house via simulator.path_seeds",
             f"- split rule: {sp['rule']}", f"- calibration split: {sp['calibration']}", f"- held-out split: {sp['held_out']}",
             f"- frozen settings: {json.dumps(cfg['settings'])}", f"- gates (unchanged): {json.dumps(cfg['gates'])}",
             "- no calibration is fitted or applied in this batch; the cap-vs-wall ratio is only recorded in the evaluation part.", "",
             "| run | house | path | path seed | split |", "|---|---|---|---|---|"]
    lines += [f"| {p['run_id']} | {p['house_seed']} | {p['path_index']} | {p['path_seed']} | {p['split']} |" for p in cfg["plan"]]
    return "\n".join(lines) + "\n"


def load_root(root: Path) -> dict:
    """Open an existing batch folder; refuse if the config's frozen fingerprint no longer matches the code on disk."""
    cfg_path = Path(root) / "batch_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"not a batch folder (no batch_config.json): {root}")
    cfg = read_json(cfg_path)
    return cfg


def assert_same_code(cfg: dict) -> None:
    now = code_hashes()
    changed = sorted(k for k in cfg["code_sha256"] if cfg["code_sha256"][k] != now.get(k))
    if changed:
        raise RuntimeError("frozen-settings violation: source files changed since the batch was initialised: "
                           + ", ".join(changed) + " (re-init a new batch folder; settings must be identical for all 30 runs)")


# ------------------------------------------------------------------ receipts (resume)
class CorruptRun(RuntimeError):
    pass


def _rel_hashes(base: Path, rels) -> dict:
    return {r: sha256_file(Path(base) / r) for r in rels}


def receipt_path(run_dir: Path, stage: str) -> Path:
    return Path(run_dir) / "receipts" / f"{stage}.json"


def receipt_state(run_dir: Path, stage: str) -> str:
    """'done' (ok receipt, all output hashes match), 'failed', or 'missing'. A hash mismatch raises CorruptRun."""
    p = receipt_path(run_dir, stage)
    if not p.is_file():
        return "missing"
    rec = read_json(p)
    if rec.get("status") != "ok":
        return "failed"
    for rel, digest in rec.get("outputs", {}).items():
        f = Path(run_dir) / rel
        if not f.is_file() or sha256_file(f) != digest:
            raise CorruptRun(f"{stage} output changed or missing after it was recorded: {f}")
    return "done"


def write_receipt(run_dir: Path, stage: str, status: str, started: float, outputs: dict, details: dict) -> dict:
    rec = {"stage": stage, "status": status, "finished_utc": _utc(), "duration_seconds": round(time.time() - started, 3),
           "outputs": outputs, "details": details}
    write_new_json(receipt_path(run_dir, stage), rec)
    return rec


def move_aside(path: Path) -> Path | None:
    """Keep (never delete) a partial folder from an interrupted attempt."""
    path = Path(path)
    if not path.exists():
        return None
    k = 1
    while path.with_name(f"{path.name}.incomplete-{k}").exists():
        k += 1
    target = path.with_name(f"{path.name}.incomplete-{k}")
    path.rename(target)
    return target


def archive_failed_receipt(run_dir: Path, stage: str) -> None:
    p = receipt_path(run_dir, stage)
    k = 1
    while p.with_name(f"{stage}.failed-{k}.json").exists():
        k += 1
    p.rename(p.with_name(f"{stage}.failed-{k}.json"))


# ------------------------------------------------------------------ run paths
def run_dir_of(root: Path, spec: dict) -> Path:
    return Path(root) / "houses" / f"house_{spec['house_seed']}" / f"path_{spec['path_index']:02d}"


def path_name(spec: dict) -> str:
    return f"path_{spec['path_index']:02d}"


# ------------------------------------------------------------------ capture (generator side; may use the scene it generates)
def upright_poses(poses: list) -> list:
    return [{**p, "roll": 0.0} for p in poses]


def make_capture_inputs(spec: dict, protocol: str, cfg_settings: dict) -> tuple[dict, list]:
    import capture_parallax
    import simulator
    scene = simulator.make_scene(spec["house_seed"])
    poses = capture_parallax.make_wide_path(scene, cfg_settings["frames"], spec["path_seed"], spec["path_index"])
    return scene, (upright_poses(poses) if protocol == "upright" else poses)


def compute_capture_qa(scene: dict, poses: list, settings: dict) -> dict:
    """Capture-side QA from the generator's own scene (never an input of the truth-free chain). Failures are reported, not hidden."""
    import simulator
    n = len(poses)
    ang = simulator.angular_coverage(poses, settings["fov"], settings["width"], settings["height"], selected=list(range(n)))
    motion = simulator.camera_motion_summary(scene, poses)
    surf = simulator.coverage_diagnostics(scene, poses, settings["fov"], width=40, height=30)
    checks = {"angular_full_sphere": bool(ang["passes_full_sphere"]), "surface_coverage_complete": bool(surf["complete"]),
              "no_furniture_collisions": surf["furniture_collisions"] == 0,
              "no_clearance_violations": motion["furniture_or_wall_violations"] == 0,
              "upright": motion["inverted_fraction"] == 0.0}
    return {"frames": n, "checks": checks, "qa_pass": bool(all(checks.values())),
            "angular": {"full_fraction": ang["full_video"]["fraction"], "polar_up": ang["full_video"]["polar_up_fraction"],
                        "polar_down": ang["full_video"]["polar_down_fraction"],
                        "max_hole_deg": ang["full_video"]["max_uncovered_hole_radius_deg"]},
            "motion": motion, "surface_observed": surf["observed"], "furniture_collisions": surf["furniture_collisions"],
            "frame_overlap_min": surf["frame_overlap_min"]}


def render_capture(blender: Path, scene_path: Path, poses_path: Path, capture_dir: Path, spec: dict, settings: dict, log: Path) -> None:
    cmd = [sys.executable, str(HERE / "blender_capture.py"), "--blender", str(blender), "--scene", str(scene_path),
           "--output", str(capture_dir), "--capture-seed", str(CAPTURE_SEED), "--width", str(settings["width"]),
           "--height", str(settings["height"]), "--fov", str(settings["fov"]), "--fps", str(settings["fps"]),
           "--poses-json", str(poses_path), "--path-name", path_name(spec)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as stream:
        done = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise RuntimeError(f"blender_capture exit code {done.returncode} (see {log})")


def stage_capture(run_dir: Path, spec: dict, protocol: str, settings: dict, blender: Path, hooks: dict) -> dict:
    """Generator-side stage: it owns the scene it makes; its QA is capture-side and not consumed by the truth-free chain."""
    started = time.time()
    gp = run_dir / "generator_private"
    if gp.exists():
        move_aside(gp)
    scene, poses = hooks["make_capture_inputs"](spec, protocol, settings)
    qa = hooks["compute_capture_qa"](scene, poses, settings)
    write_new_json(gp / "scene_input.json", scene)
    write_new_json(gp / "poses.json", {"path_seed": spec["path_seed"], "poses": poses})
    write_new_json(gp / "capture_qa.json", qa)
    capture_dir = run_dir / "capture"
    if capture_dir.exists():
        move_aside(capture_dir)
    hooks["render_capture"](blender, gp / "scene_input.json", gp / "poses.json", capture_dir, spec, settings, run_dir / "logs" / "capture.log")
    video = capture_dir / "videos" / f"{path_name(spec)}.mp4"
    outs = _rel_hashes(run_dir, ["generator_private/scene_input.json", "generator_private/poses.json", "generator_private/capture_qa.json",
                                 f"capture/videos/{path_name(spec)}.mp4", "capture/private_ground_truth/scene.json"])
    if not video.is_file():
        raise RuntimeError(f"capture produced no video: {video}")
    return write_receipt(run_dir, "capture", "ok", started, outs,
                         {"qa_pass": qa["qa_pass"], "qa_checks": qa["checks"], "frames_rendered": len(list((capture_dir / "rgb" / path_name(spec)).glob("*.png")))})


# ------------------------------------------------------------------ TRUTH-FREE stages (inputs: video / prediction only)
def da3_command(video: Path, out_dir: Path, settings: dict) -> list:
    return [sys.executable, str(HERE / "infer_rgb.py"), "--input", str(video), "--output", str(out_dir), "--frames", str(settings["da3_frames"]),
            "--resolution", str(settings["da3_resolution"]), "--seed", str(settings["da3_seed"])]


def run_da3(cmd: list, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as stream:
        done = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    if done.returncode != 0:
        raise RuntimeError(f"infer_rgb exit code {done.returncode} (see {log})")


def stage_inference(run_dir: Path, spec: dict, settings: dict, hooks: dict) -> dict:
    started = time.time()
    video = run_dir / "capture" / "videos" / f"{path_name(spec)}.mp4"
    if not video.is_file():
        raise RuntimeError("no video: capture stage has not produced one")
    out = run_dir / "da3"
    if out.exists():
        move_aside(out)
    hooks["run_da3"](da3_command(video, out, settings), run_dir / "logs" / "inference.log")
    manifest = read_json(out / "manifest.json")
    outs = _rel_hashes(run_dir, ["da3/prediction.npz", "da3/manifest.json"])
    return write_receipt(run_dir, "inference", "ok", started, outs,
                         {"device": manifest.get("device"), "frames_used": manifest.get("frames_used"),
                          "source_indices_count": len(manifest.get("source_indices", [])),
                          "resolution": manifest.get("resolution"), "da3_elapsed_seconds": manifest.get("elapsed_seconds"),
                          "gpu_peak_memory_bytes": manifest.get("gpu_peak_memory_bytes"), "input_video_sha256": sha256_file(video)})


def stage_chain(run_dir: Path, spec: dict, hooks: dict) -> dict:
    """planes -> every axis candidate -> height -> walls -> ray boundary, all via run_axis_candidates.run_extraction."""
    started = time.time()
    label = run_id(spec["house_seed"], spec["path_index"])
    prediction = run_dir / "da3" / "prediction.npz"
    tmp, final = run_dir / "axis.tmp", run_dir / "axis"
    for p in (tmp, final):
        if p.exists():
            move_aside(p)
    index = hooks["run_extraction"]([{"label": label, "prediction": str(prediction)}], tmp)
    hooks["write_reports"](index, tmp)
    os.rename(tmp, final)
    rec = index["budgets"][0]
    outs = _rel_hashes(run_dir, [str(p.relative_to(run_dir)).replace("\\", "/") for p in sorted(final.rglob("*")) if p.is_file()])
    cands = rec.get("candidates", [])
    return write_receipt(run_dir, "chain", "ok" if rec.get("status") == "ok" else "failed", started, outs,
                         {"label": label, "chain_status": rec.get("status"), "axis_status": rec.get("axis_status"),
                          "candidate_count": len(cands), "co_supported_count": sum(1 for c in cands if c["co_supported_with_best"]),
                          "error": rec.get("error")})


def collect_freeze_files(run_dir: Path) -> list:
    keep = [run_dir / "capture" / "videos", run_dir / "da3", run_dir / "axis"]
    files = [p for base in keep if base.exists() for p in sorted(base.rglob("*")) if p.is_file()]
    return [str(p.relative_to(run_dir)).replace("\\", "/") for p in files]


def stage_freeze(run_dir: Path, spec: dict, code: dict) -> dict:
    """Hash everything the truth-free part produced and consumed. Written once, before any evaluation may run."""
    started = time.time()
    rels = collect_freeze_files(run_dir)
    freeze = {"run_id": run_id(spec["house_seed"], spec["path_index"]), "frozen_utc": _utc(), "files": _rel_hashes(run_dir, rels),
              "code_sha256": code, "reconstruction_inputs": ["capture video", "DA3 prediction"]}
    write_new_json(run_dir / "freeze.json", freeze)
    return write_receipt(run_dir, "freeze", "ok", started, {"freeze.json": sha256_file(run_dir / "freeze.json")}, {"files": len(rels)})


def verify_freeze(run_dir: Path) -> dict:
    freeze = read_json(Path(run_dir) / "freeze.json")
    for rel, digest in freeze["files"].items():
        f = Path(run_dir) / rel
        if not f.is_file() or sha256_file(f) != digest:
            raise CorruptRun(f"frozen file changed or missing after freezing: {f}")
    return freeze


def run_truth_free_stages(run_dir: Path, spec: dict, settings: dict, hooks: dict, events: list) -> None:
    """inference -> chain -> freeze for one capture that already exists. Resumes finished stages by receipt."""
    for stage, fn in (("inference", lambda: stage_inference(run_dir, spec, settings, hooks)),
                      ("chain", lambda: stage_chain(run_dir, spec, hooks)),
                      ("freeze", lambda: stage_freeze(run_dir, spec, hooks["code_hashes"]()))):
        state = receipt_state(run_dir, stage)
        if state == "done":
            events.append(("skip", stage))
            continue
        if state == "failed":
            raise RuntimeError(f"{stage} previously failed for {run_dir.name}; pass --retry-failed to try again")
        events.append(("start", stage))
        rec = fn()
        events.append(("done", stage))
        if rec["status"] != "ok":
            raise RuntimeError(f"{stage} failed: {rec['details']}")


def process_run(root: Path, spec: dict, cfg: dict, hooks: dict, blender: Path = DEFAULT_BLENDER, retry_failed: bool = False,
                events: list | None = None) -> dict:
    """One (house, path): capture (+QA) then the truth-free stages. Returns a small status dict; never reads truth for stages 2-4."""
    events = events if events is not None else []
    run_dir = run_dir_of(root, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {"run_id": spec["run_id"], "state": "ok", "error": None}
    try:
        for stage in STAGES:
            if retry_failed and receipt_state(run_dir, stage) == "failed":
                archive_failed_receipt(run_dir, stage)
        state = receipt_state(run_dir, "capture")
        if state == "failed":
            raise RuntimeError(f"capture previously failed for {run_dir.name}; pass --retry-failed to try again")
        if state == "done":
            events.append(("skip", "capture"))
        else:
            events.append(("start", "capture"))
            t0 = time.time()
            try:
                stage_capture(run_dir, spec, cfg["protocol"], cfg["settings"], blender, hooks)
            except Exception as exc:
                write_receipt(run_dir, "capture", "failed", t0, {}, {"error": f"{type(exc).__name__}: {exc}",
                                                                      "traceback_tail": traceback.format_exc()[-1500:]})
                raise
            events.append(("done", "capture"))
        try:
            run_truth_free_stages(run_dir, spec, cfg["settings"], hooks, events)
        except CorruptRun:
            raise
        except Exception as exc:
            for stage in ("inference", "chain"):
                if receipt_state(run_dir, stage) == "missing":
                    write_receipt(run_dir, stage, "failed", started, {}, {"error": f"{type(exc).__name__}: {exc}",
                                                                           "traceback_tail": traceback.format_exc()[-1500:]})
                    break
            raise
    except CorruptRun:
        raise
    except Exception as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
    status["wall_seconds_this_invocation"] = round(time.time() - started, 2)
    return status


def _default_hooks() -> dict:
    def run_extraction(budgets, out_dir):
        import run_axis_candidates as rac
        return rac.run_extraction(budgets, out_dir)

    def write_reports(index, out_dir):
        import run_axis_candidates as rac
        rac.write_reports(index, out_dir)

    return {"make_capture_inputs": make_capture_inputs, "compute_capture_qa": compute_capture_qa, "render_capture": render_capture,
            "run_da3": run_da3, "run_extraction": run_extraction, "write_reports": write_reports, "code_hashes": code_hashes}


def run_batch(root: Path, houses=None, paths=None, max_hours: float = MAX_HOURS_DEFAULT, blender: Path = DEFAULT_BLENDER,
              retry_failed: bool = False, hooks: dict | None = None, clock=time.time) -> dict:
    """Serial loop over the predeclared plan (house-major). Stops starting new runs after `max_hours` and reports exactly what finished."""
    root = Path(root)
    cfg = load_root(root)
    assert_same_code(cfg)
    hooks = {**_default_hooks(), **(hooks or {})}
    lock = root / ".gpu_serial.lock"
    try:
        with open(lock, "x") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"another batch process holds {lock} (GPU jobs are strictly serial); remove it only if that process is dead")
    t_start = clock()
    results, stopped = [], None
    try:
        for spec in cfg["plan"]:
            if houses is not None and spec["house_seed"] not in houses:
                continue
            if paths is not None and spec["path_index"] not in paths:
                continue
            if (clock() - t_start) / 3600.0 >= max_hours:
                stopped = f"time budget {max_hours} h reached before {spec['run_id']}"
                break
            print(f"[{_utc()}] run {spec['run_id']}", flush=True)
            status = process_run(root, spec, cfg, hooks, blender, retry_failed)
            print(f"[{_utc()}]   -> {status['state']} {status.get('error') or ''} ({status['wall_seconds_this_invocation']} s)", flush=True)
            results.append(status)
    finally:
        lock.unlink(missing_ok=True)
    summary = {"attempted": len(results), "ok": sum(r["state"] == "ok" for r in results),
               "failed": [r for r in results if r["state"] == "failed"], "stopped_for_budget": stopped,
               "completed_runs_total": count_frozen_runs(root, cfg)}
    return summary


def count_frozen_runs(root: Path, cfg: dict) -> int:
    return sum(1 for spec in cfg["plan"] if (run_dir_of(root, spec) / "freeze.json").is_file())


# ------------------------------------------------------------------ EVALUATION ONLY (reads private truth; never imported by the stages above)
def _pairwise(values):
    return [(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def scale_diagnostic(ev: dict, true_height_m: float) -> dict:
    """Cap-vs-wall scale ratio, EVALUATION ONLY. In the 3.0 m normalised frame the evaluator's edge ratio
    r = pred_len / true_len equals s_wall / s_cap, with s_cap = (estimated floor-ceiling separation, unscaled) / (true height)
    and s_wall = unscaled predicted wall length / true wall length. Recorded for a later labelled calibration
    experiment; nothing is fitted or applied here."""
    edges = ((ev.get("metrics") or {}).get("edge_errors")) or []
    ratios = [e["pred_length_m"] / e["true_length_m"] for e in edges if e.get("true_length_m")]
    sep = ((ev.get("scale") or {}).get("prediction_height") or {}).get("separation_unscaled")
    out = {"separation_unscaled": sep, "true_height_m": true_height_m, "matched_walls": len(ratios),
           "wall_to_cap_scale_ratio_median": None, "cap_to_wall_scale_ratio": None, "cap_to_wall_scale_ratio_spread": None,
           "ceiling_height_wall_scaled_m": None, "ceiling_height_error_cm": None, "s_cap_units_per_m": None}
    if sep is not None and true_height_m:
        out["s_cap_units_per_m"] = sep / true_height_m
    if ratios:
        r = statistics.median(ratios)
        out.update(wall_to_cap_scale_ratio_median=r, cap_to_wall_scale_ratio=1.0 / r,
                   cap_to_wall_scale_ratio_spread=(max(ratios) - min(ratios)) / r,
                   ceiling_height_wall_scaled_m=true_height_m / r, ceiling_height_error_cm=100.0 * abs(true_height_m / r - true_height_m))
    return out


def candidate_row(spec: dict, cand: dict, tie_info: dict, ev: dict | None, height_status, true_height_m: float, gates: dict) -> dict:
    row = {"run_id": spec["run_id"], "house_seed": spec["house_seed"], "path_index": spec["path_index"], "split": spec["split"],
           "candidate_id": cand["candidate_id"], "rank": cand["rank"], "co_supported": cand["co_supported_with_best"],
           "score": cand["score"], "viable": cand["viable"], "truth_free_top_ranked": tie_info["top"].get(cand["candidate_id"], False),
           "tie_count_at_top": tie_info["tie_count"], "polygon_status": (cand.get("polygon") or {}).get("boundary_status"),
           "height_status": height_status, "status": "not_scored", "iou": None, "max_abs_wall_error_cm": None,
           "max_rel_wall_error_percent": None, "gate_1_5cm": None, "gate_3pct": None, "gate_ceiling_height_1_5cm": None,
           "gate_all": False, "matched_edges": None, "scale": None, "failure_reasons": None}
    if ev is None:
        return row
    m = ev.get("metrics") or {}
    row.update(status=ev.get("status"), failure_reasons=ev.get("failure_reasons"), iou=m.get("iou"),
               max_abs_wall_error_cm=m.get("max_abs_wall_error_cm"), max_rel_wall_error_percent=m.get("max_rel_wall_error_percent"),
               gate_1_5cm=(ev.get("custom_1_5cm_horizontal_diagnostic") or {}).get("pass"),
               gate_3pct=(ev.get("wall_length_3_percent_threshold_oracle_normalised") or {}).get("pass"))
    if ev.get("status") == "scored":
        sc = scale_diagnostic(ev, true_height_m)
        row["scale"] = sc
        row["matched_edges"] = [{"true_edge": e["true_edge"], "pred_length_m": e["pred_length_m"], "true_length_m": e["true_length_m"]}
                                for e in m.get("edge_errors", [])]
        err = sc["ceiling_height_error_cm"]
        row["gate_ceiling_height_1_5cm"] = None if err is None else bool(err <= gates["ceiling_height_error_cm"])
    row["gate_all"] = bool(row["gate_1_5cm"] and row["gate_3pct"] and row["gate_ceiling_height_1_5cm"])
    return row


def tie_info_of(candidates: list) -> dict:
    viable = [c["score"] for c in candidates if c["viable"]]
    best = max(viable) if viable else None
    tops = {c["candidate_id"]: bool(c["viable"] and best is not None and abs(c["score"] - best) < 1e-9) for c in candidates}
    return {"best": best, "top": tops, "tie_count": sum(tops.values())}


def run_level(spec: dict, qa: dict | None, receipts: dict, da3: dict | None, chain: dict, rows: list, wall_seconds: float | None) -> dict:
    """Run-level record. Selected candidate = the UNIQUE truth-free top-ranked viable candidate (never chosen by truth);
    a tie at the top is an unresolved axis ambiguity and fails the run."""
    tops = [r for r in rows if r["truth_free_top_ranked"]]
    selected = tops[0] if len(tops) == 1 else None
    scored = [r for r in rows if r["status"] == "scored"]
    cosup = [r for r in rows if r["co_supported"]]
    reasons = []
    if not qa or not qa.get("qa_pass"):
        reasons.append("capture_qa_failed")
    if da3 is None or da3.get("status") != "ok":
        reasons.append("da3_not_ok")
    if not rows:
        reasons.append("no_axis_candidates")
    elif selected is None:
        reasons.append("unresolved_axis_ambiguity" if len(tops) > 1 else "no_viable_candidate")
    elif not selected["gate_all"]:
        why = [g for g, v in (("1.5cm", selected["gate_1_5cm"]), ("3pct", selected["gate_3pct"]),
                              ("ceiling_height_1.5cm", selected["gate_ceiling_height_1_5cm"])) if not v]
        reasons.append("selected_candidate_fails:" + ",".join(why) if scored and selected["status"] == "scored"
                       else f"selected_candidate_{selected['status']}")
    return {"run_id": spec["run_id"], "house_seed": spec["house_seed"], "path_index": spec["path_index"], "split": spec["split"],
            "capture_qa_pass": None if qa is None else qa.get("qa_pass"), "capture_qa_failed_checks": None if qa is None else
            [k for k, v in qa["checks"].items() if not v], "retained_frames": None if da3 is None else da3.get("frames_used"),
            "retained_capture_fraction": None if da3 is None else (da3.get("frames_used") or 0) / SETTINGS["frames_available"],
            "da3_status": None if da3 is None else da3.get("status"), "da3_seconds": None if da3 is None else da3.get("da3_elapsed_seconds"),
            "axis_status": chain.get("axis_status"), "axis_candidate_count": chain.get("candidate_count"),
            "co_supported_count": chain.get("co_supported_count"),
            "candidate_cap": chain.get("candidate_cap"),
            "candidate_cap_hit": bool((chain.get("candidate_cap") or {}).get("dropped_by_cap")),
            "polygon_statuses": {str(r["candidate_id"]): r["polygon_status"] for r in rows},
            "height_statuses": {str(r["candidate_id"]): r["height_status"] for r in rows},
            "scored_candidates": len(scored), "any_candidate_gate_all_oracle_info_only": any(r["gate_all"] for r in rows),
            "co_supported_gate_all_count": sum(1 for r in cosup if r["gate_all"]),
            "selected_candidate_id": None if selected is None else selected["candidate_id"],
            "selected": None if selected is None else {k: selected[k] for k in ("iou", "max_abs_wall_error_cm", "max_rel_wall_error_percent",
                                                                               "gate_1_5cm", "gate_3pct", "gate_ceiling_height_1_5cm", "status")},
            "selected_scale": None if selected is None else selected["scale"],
            "run_pass": not reasons, "fail_reasons": reasons, "wall_seconds": wall_seconds}


def repeatability(selected_rows: list, gates: dict) -> dict:
    """Wall repeatability across a house's paths per CONTRACT.md: for every true wall matched in two paths, the difference of the
    predicted lengths (3.0 m normalised frame) must be <= 0.01 m AND (separately) <= 0.005 x true length; both pass rates 100%.
    Ceiling-height spread (wall-scaled estimate across paths) must be <= 1 cm."""
    pairs = []
    by_edge = {}
    for r in selected_rows:
        for e in r["matched_edges"] or []:
            by_edge.setdefault(e["true_edge"], []).append((r["path_index"], e["pred_length_m"], e["true_length_m"]))
    for edge, vals in by_edge.items():
        for a, b in _pairwise(vals):
            d = abs(a[1] - b[1])
            pairs.append({"true_edge": edge, "paths": [a[0], b[0]], "abs_diff_m": d, "rel_diff_of_true": d / a[2]})
    heights = [r["scale"]["ceiling_height_wall_scaled_m"] for r in selected_rows if r["scale"] and r["scale"]["ceiling_height_wall_scaled_m"] is not None]
    rate_abs = sum(p["abs_diff_m"] <= gates["repeatability_abs_m"] for p in pairs) / len(pairs) if pairs else None
    rate_rel = sum(p["rel_diff_of_true"] <= gates["repeatability_rel_of_true_wall"] for p in pairs) / len(pairs) if pairs else None
    spread_cm = 100.0 * (max(heights) - min(heights)) if len(heights) >= 2 else None
    return {"pair_count": len(pairs), "pass_rate_abs_1cm": rate_abs, "pass_rate_rel_0_5pct": rate_rel,
            "max_abs_diff_cm": 100 * max((p["abs_diff_m"] for p in pairs), default=0) if pairs else None,
            "max_rel_diff_percent": 100 * max((p["rel_diff_of_true"] for p in pairs), default=0) if pairs else None,
            "gate_repeatability_1cm": rate_abs == 1.0, "gate_repeatability_0_5pct": rate_rel == 1.0,
            "height_spread_cm": spread_cm, "gate_height_spread_1cm": spread_cm is not None and spread_cm <= gates["height_spread_cm"]}


def house_level(house: int, runs: list, selected_rows_by_run: dict, gates: dict, expected_paths: int = PATHS_PER_HOUSE) -> dict:
    mine = [r for r in runs if r["house_seed"] == house]
    present = len(mine)
    sel = [selected_rows_by_run[r["run_id"]] for r in mine if r["run_id"] in selected_rows_by_run and selected_rows_by_run[r["run_id"]]]
    rep = repeatability(sel, gates) if len(sel) == expected_paths else None
    all_pass = present == expected_paths and all(r["run_pass"] for r in mine)
    reasons = []
    if present != expected_paths:
        reasons.append(f"only_{present}_of_{expected_paths}_runs_present")
    reasons += [f"{r['run_id']}:{'+'.join(r['fail_reasons'])}" for r in mine if not r["run_pass"]]
    if all_pass:
        if rep is None or not (rep["gate_repeatability_1cm"] and rep["gate_repeatability_0_5pct"] and rep["gate_height_spread_1cm"]):
            reasons.append("repeatability_or_height_spread_gate_fails")
    return {"house_seed": house, "split": mine[0]["split"] if mine else None, "runs_present": present, "runs_pass": sum(r["run_pass"] for r in mine),
            "repeatability": rep, "house_pass": bool(all_pass and rep and rep["gate_repeatability_1cm"] and rep["gate_repeatability_0_5pct"]
                                                      and rep["gate_height_spread_1cm"]), "fail_reasons": reasons}


def stage_one(plan: list, runs: list, houses: list, code_consistent: bool) -> dict:
    expected = len(plan)
    by_id = {r["run_id"] for r in runs}
    complete = (len(runs) == expected and all(p["run_id"] in by_id for p in plan) and all(r["run_pass"] for r in runs)
                and len(houses) == len({p["house_seed"] for p in plan}) and all(h["house_pass"] for h in houses) and code_consistent)
    return {"stage_1_complete": bool(complete), "runs_expected": expected, "runs_evaluated": len(runs),
            "runs_pass": sum(r["run_pass"] for r in runs), "houses_pass": sum(h["house_pass"] for h in houses),
            "houses_expected": len({p["house_seed"] for p in plan}), "frozen_code_consistent": code_consistent,
            "rule": "stage_1_complete is false unless ALL 30 runs are present, pass every per-run gate, every house passes repeatability and height-spread, "
                    "and all runs used identical frozen code; no accuracy or completion claim otherwise"}


def _fmt(v, f=".3f"):
    return "" if v is None else (format(v, f) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v))


def runs_markdown(runs: list) -> str:
    lines = ["| run | split | QA | frames | DA3 | wall s | cand | co-sup | selected | IoU | max cm | max % | 1.5cm | 3% | ceil cm | cap/wall | run pass | why not |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s, sc = r["selected"] or {}, r["selected_scale"] or {}
        lines.append(f"| {r['run_id']} | {r['split']} | {r['capture_qa_pass']} | {r['retained_frames']} | {r['da3_status']} | {_fmt(r['wall_seconds'], '.0f')} | "
                     f"{r['axis_candidate_count']} | {r['co_supported_count']} | {r['selected_candidate_id']} | {_fmt(s.get('iou'))} | "
                     f"{_fmt(s.get('max_abs_wall_error_cm'), '.2f')} | {_fmt(s.get('max_rel_wall_error_percent'), '.2f')} | {s.get('gate_1_5cm')} | {s.get('gate_3pct')} | "
                     f"{_fmt(sc.get('ceiling_height_error_cm'), '.2f')} | {_fmt(sc.get('cap_to_wall_scale_ratio'), '.4f')} | {r['run_pass']} | {'; '.join(r['fail_reasons'])} |")
    return "\n".join(lines) + "\n"


def candidates_markdown(rows: list) -> str:
    lines = ["| run | cand | rank | co-sup | top | ties | polygon | height | status | IoU | max cm | max % | 1.5cm | 3% | ceil cm | cap/wall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sc = r["scale"] or {}
        lines.append(f"| {r['run_id']} | {r['candidate_id']} | {r['rank']} | {r['co_supported']} | {r['truth_free_top_ranked']} | {r['tie_count_at_top']} | "
                     f"{r['polygon_status']} | {r['height_status']} | {r['status']} | {_fmt(r['iou'])} | {_fmt(r['max_abs_wall_error_cm'], '.2f')} | "
                     f"{_fmt(r['max_rel_wall_error_percent'], '.2f')} | {r['gate_1_5cm']} | {r['gate_3pct']} | {_fmt(sc.get('ceiling_height_error_cm'), '.2f')} | "
                     f"{_fmt(sc.get('cap_to_wall_scale_ratio'), '.4f')} |")
    return "\n".join(lines) + "\n"


def houses_markdown(houses: list) -> str:
    lines = ["| house | split | runs | runs pass | repeat pairs | abs<=1cm rate | rel<=0.5% rate | max abs diff cm | height spread cm | house pass | why not |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in houses:
        rep = h["repeatability"] or {}
        lines.append(f"| {h['house_seed']} | {h['split']} | {h['runs_present']} | {h['runs_pass']} | {rep.get('pair_count')} | {_fmt(rep.get('pass_rate_abs_1cm'))} | "
                     f"{_fmt(rep.get('pass_rate_rel_0_5pct'))} | {_fmt(rep.get('max_abs_diff_cm'), '.2f')} | {_fmt(rep.get('height_spread_cm'), '.2f')} | "
                     f"{h['house_pass']} | {'; '.join(h['fail_reasons'])} |")
    return "\n".join(lines) + "\n"


def _default_eval_hooks() -> dict:
    def load_scene(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def evaluate_floorplan(boundary, scene, height, manifest):
        import evaluate_floorplan as module
        return module.evaluate_floorplan(boundary, scene, height, manifest=manifest, axis_sign_resolved=False)

    def read_frozen(out_dir, rel, sha):
        import compare_wall_boundaries as cwb
        return cwb.read_frozen(out_dir, rel, sha)

    def eval_copy(boundary):
        import compare_wall_boundaries as cwb
        return cwb.eval_copy(boundary)

    def evaluator_compatible(height):
        import height_diagnosis
        return height_diagnosis.evaluator_compatible(height)

    return {"load_scene": load_scene, "evaluate_floorplan": evaluate_floorplan, "read_frozen": read_frozen, "eval_copy": eval_copy,
            "evaluator_compatible": evaluator_compatible}


def evaluate_run(root: Path, spec: dict, cfg: dict, hooks: dict, events: list | None = None) -> tuple[dict, list]:
    """EVALUATION ONLY for one frozen run. Verifies the freeze, then reads the private scene and scores EVERY candidate."""
    events = events if events is not None else []
    run_dir = run_dir_of(root, spec)
    freeze = verify_freeze(run_dir)                                 # freeze must exist and match BEFORE truth is read
    events.append(("freeze_verified", spec["run_id"]))
    index = read_json(run_dir / "axis" / "frozen_index.json")
    rec = index["budgets"][0]
    gates = cfg["gates"]
    scene = hooks["load_scene"](run_dir / "capture" / "private_ground_truth" / "scene.json")
    events.append(("scene_read", spec["run_id"]))
    qa = read_json(run_dir / "generator_private" / "capture_qa.json")
    inf = read_json(receipt_path(run_dir, "inference"))
    da3 = {"status": inf["status"], **inf["details"]}
    chain_rec = read_json(receipt_path(run_dir, "chain"))
    chain = {**chain_rec["details"]}
    rep_path = run_dir / "axis" / "reports" / f"{rec['label']}.json"
    if rep_path.is_file():
        rep = read_json(rep_path)
        chain.update(candidate_cap=rep.get("candidate_cap"), unresolved_ambiguity=rep.get("unresolved_ambiguity"))
    manifest = {"provenance": "frozen frame budget", "retained_capture_fraction": len(rec.get("source_indices", [])) / SETTINGS["frames_available"],
                "frames_registered": len(rec.get("source_indices", [])), "frames_available": SETTINGS["frames_available"]}
    cands = rec.get("candidates", [])
    tie = tie_info_of([{"candidate_id": c["candidate_id"], "viable": c["viable"], "score": c["score"]} for c in cands])
    rows, evals = [], {}
    true_h = float(scene["height"])
    for c in cands:
        files = c["files"]
        cand = {"candidate_id": c["candidate_id"], "rank": c["rank"], "co_supported_with_best": c["co_supported_with_best"], "score": c["score"],
                "viable": c["viable"], "polygon": c.get("polygon")}
        hstatus, ev = None, None
        if "height" in files:
            hstatus = hooks["read_frozen"](run_dir / "axis", files["height"]["path"], files["height"]["sha256"]).get("status")
        if "boundary" in files and "height" in files:
            boundary = hooks["read_frozen"](run_dir / "axis", files["boundary"]["path"], files["boundary"]["sha256"])
            height = hooks["read_frozen"](run_dir / "axis", files["height"]["path"], files["height"]["sha256"])
            events.append(("evaluate", spec["run_id"], c["candidate_id"]))
            try:
                ev = hooks["evaluate_floorplan"](hooks["eval_copy"](boundary), scene, hooks["evaluator_compatible"](height), manifest)
                evals[c["candidate_id"]] = ev
            except Exception as exc:
                ev = {"status": "evaluation_error", "failure_reasons": [f"{type(exc).__name__}: {exc}"], "metrics": None}
        rows.append(candidate_row(spec, cand, tie, ev, hstatus, true_h, gates))
    wall = sum(read_json(receipt_path(run_dir, s)).get("duration_seconds", 0.0) for s in STAGES if receipt_path(run_dir, s).is_file())
    run = run_level(spec, qa, {}, da3, chain, rows, wall)
    run["stage_seconds"] = {s: read_json(receipt_path(run_dir, s)).get("duration_seconds") for s in STAGES if receipt_path(run_dir, s).is_file()}
    run["private_scene_dims_m"] = [scene["width"], scene["length"], scene["height"]]
    run["evaluation_only_labels"] = ["EVALUATOR-ORACLE 3.0 m ceiling-normalised protocol", "mirroring/alignment/edge matching chosen with truth in hand",
                                     "cap-vs-wall scale ratio recorded for a later labelled calibration experiment; no calibration fitted or applied"]
    return run, rows


def evaluate_batch(root: Path, allow_partial: bool = False, hooks: dict | None = None) -> dict:
    root = Path(root)
    cfg = load_root(root)
    hooks = {**_default_eval_hooks(), **(hooks or {})}
    plan = cfg["plan"]
    frozen = [s for s in plan if (run_dir_of(root, s) / "freeze.json").is_file()]
    if len(frozen) != len(plan) and not allow_partial:
        raise RuntimeError(f"refusing to evaluate: only {len(frozen)} of {len(plan)} runs are frozen (use --allow-partial to evaluate what exists)")
    if not frozen:
        raise RuntimeError("no frozen runs to evaluate")
    n = 0
    while (root / f"results_{n:03d}").exists():
        n += 1
    out = root / f"results_{n:03d}"
    freeze_index = {f"{s['run_id']}": sha256_file(run_dir_of(root, s) / "freeze.json") for s in frozen}
    write_new_json(out / "freeze_index_before_truth.json", {"written_utc": _utc(), "runs": freeze_index})   # BEFORE any scene is read
    code_sets = {json.dumps(read_json(run_dir_of(root, s) / "freeze.json")["code_sha256"], sort_keys=True) for s in frozen}
    runs, cand_rows, selected_by_run, events = [], [], {}, []
    for spec in frozen:
        run, rows = evaluate_run(root, spec, cfg, hooks, events)
        run_json = {"run": run, "candidates": rows}
        write_new_json(run_dir_of(root, spec) / f"evaluation_{out.name}.json", run_json)
        runs.append(run)
        cand_rows += rows
        sel = [r for r in rows if r["truth_free_top_ranked"]]
        selected_by_run[run["run_id"]] = sel[0] if len(sel) == 1 else None
    houses = [house_level(h, runs, selected_by_run, cfg["gates"]) for h in sorted({s["house_seed"] for s in plan})]
    houses = [h for h in houses if h["runs_present"] or not allow_partial]
    stage = stage_one(plan, runs, houses, len(code_sets) == 1)
    split_summary = {sp: {"runs": sum(1 for r in runs if r["split"] == sp), "runs_pass": sum(1 for r in runs if r["split"] == sp and r["run_pass"])}
                     for sp in ("calibration", "held_out")}
    write_new_json(out / "runs.json", runs)
    write_new_json(out / "candidates.json", cand_rows)
    write_new_json(out / "houses.json", houses)
    write_new_json(out / "stage1.json", {**stage, "split_summary": split_summary, "predeclared_split": cfg["split"],
                                         "allow_partial": allow_partial, "no_calibration_fitted_or_applied": True})
    md = ["# RGB batch stage 1 - evaluation (EVALUATION ONLY, evaluator-oracle 3.0 m normalised protocol)", "",
          f"**stage 1 complete: {str(stage['stage_1_complete']).lower()}** ({stage['runs_pass']}/{stage['runs_expected']} runs pass; "
          f"{stage['runs_evaluated']} evaluated; {stage['houses_pass']}/{stage['houses_expected']} houses pass)", "",
          "Truth never selected a candidate; the selected candidate is the unique truth-free top-ranked one (a tie is an unresolved ambiguity and fails the run). "
          "Gates unchanged: 1.5 cm, 3 %, plus ceiling-height 1.5 cm / 1 cm spread and 1 cm / 0.5 % repeatability.", "",
          f"calibration split {cfg['split']['calibration']}: {split_summary['calibration']}; held-out {cfg['split']['held_out']}: {split_summary['held_out']}", "",
          "## Per run", "", runs_markdown(runs), "## Per house", "", houses_markdown(houses), "## Every candidate", "", candidates_markdown(cand_rows)]
    write_new_text(out / "RESULTS.md", "\n".join(md))
    return {"results_dir": str(out), **stage, "split_summary": split_summary, "order_events": events}


# ------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="create a NEW batch folder with the predeclared plan and split")
    p_init.add_argument("--output-dir", type=Path, required=True)
    p_init.add_argument("--protocol", choices=PROTOCOLS, default="upright", help="rolling is kept as a flag but is NOT run in stage 1")
    p_run = sub.add_parser("run", help="capture + DA3 + truth-free chain + freeze, serial and resumable")
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument("--houses", type=int, nargs="*", default=None)
    p_run.add_argument("--paths", type=int, nargs="*", default=None)
    p_run.add_argument("--max-hours", type=float, default=MAX_HOURS_DEFAULT)
    p_run.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    p_run.add_argument("--retry-failed", action="store_true")
    p_ev = sub.add_parser("evaluate", help="EVALUATION ONLY: separate stage that reads private truth after everything is frozen")
    p_ev.add_argument("--output-dir", type=Path, required=True)
    p_ev.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        cfg = init_root(a.output_dir, a.protocol)
        print(json.dumps({"output_dir": str(a.output_dir), "split": cfg["split"], "runs": len(cfg["plan"])}))
    elif a.cmd == "run":
        cfg = load_root(a.output_dir)
        if cfg["protocol"] != "upright":
            raise SystemExit("stage 1 runs the upright protocol only; rolling is not run here")
        print(json.dumps(run_batch(a.output_dir, a.houses, a.paths, a.max_hours, a.blender, a.retry_failed), indent=1))
    else:
        print(json.dumps(evaluate_batch(a.output_dir, a.allow_partial), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
