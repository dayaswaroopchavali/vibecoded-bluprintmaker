"""Small seeded single-room RGB benchmark generator.

This is a procedural geometry and texture prototype, not a physically based
renderer. It uses CPU ray/box intersections and synthetic textures; lighting,
materials, lens distortion, sensor noise, and real-world clutter are not modeled.
Only RGB and video are ordinary inputs. Camera calibration/poses are written
under private_ground_truth and should never be supplied to a reconstruction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def make_scene(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    w, length, height = rng.uniform(4.0, 7.0), rng.uniform(5.0, 9.0), rng.uniform(2.5, 3.6)
    # Openings are rectangular and attached to room boundaries.
    door_wall = int(rng.integers(0, 4))
    door_width = float(rng.uniform(.82, 1.08))
    door_height = float(rng.uniform(2.0, min(2.3, height - .12)))
    door = {"wall": door_wall, "center": float(rng.uniform(-.25, .25)), "width": door_width,
            "z_min": 0.0, "z_max": door_height, "type": "doorway"}
    win_wall = (door_wall + int(rng.integers(1, 4))) % 4
    wall_span = w if win_wall in (0, 2) else length
    win_width = float(min(rng.uniform(1.0, 1.8), wall_span * .42))
    window = {"wall": win_wall, "center": float(rng.uniform(-.2, .2)), "width": win_width,
              "z_min": float(rng.uniform(1.0, 1.35)), "z_max": 2.2, "type": "window"}
    window["z_max"] = min(window["z_max"], height - .35)
    # Keep sparse occluders inside the navigable center, leaving a continuous loop.
    furniture = []
    for i in range(3):
        bw, bd = rng.uniform(.55, 1.15), rng.uniform(.45, 1.0)
        cx = float(rng.uniform(-w * .27, w * .27)); cy = float(rng.uniform(-length * .27, length * .27))
        furniture.append({"name": ["table", "cabinet", "sofa"][i], "center": [cx, cy],
                          "size": [float(bw), float(bd), float(rng.uniform(.65, 1.15))],
                          "color": [float(x) for x in rng.uniform(.25, .72, 3)]})
    palette = rng.uniform(.42, .78, size=(4, 3)).tolist()
    return {"seed": int(seed), "width": float(w), "length": float(length), "height": float(height),
            "walls": ["north", "east", "south", "west"], "openings": [door, window],
            "furniture": furniture, "wall_colors": palette,
            "floor_color": [float(x) for x in rng.uniform(.34, .65, 3)],
            "ceiling_color": [float(x) for x in rng.uniform(.68, .86, 3)]}


PATH_COUNT = 3
CAMERA_WALL_MARGIN = .45       # metres between camera and any wall
CAMERA_FURNITURE_MARGIN = .42  # metres between camera and any furniture footprint
MAX_PITCH_RAD = math.radians(36)  # bounded pitch sweep; never looks straight up/down


def path_seeds(scene: dict, capture_seed: int | None = None) -> list[int]:
    """One independent seed per walkthrough, derived from the capture seed."""
    root = scene["seed"] if capture_seed is None else capture_seed
    return [int(root * 1009 + i) for i in range(PATH_COUNT)]


def camera_is_clear(scene: dict, x: float, y: float, wall_margin: float = CAMERA_WALL_MARGIN,
                    furniture_margin: float = CAMERA_FURNITURE_MARGIN) -> bool:
    """Footprint test shared by path generation and tests; the single place to swap in a polygon room."""
    if abs(x) >= scene["width"] / 2 - wall_margin or abs(y) >= scene["length"] / 2 - wall_margin:
        return False
    for f in scene["furniture"]:
        fx, fy = f["center"]; fw, fd = f["size"][:2]
        if abs(x - fx) < fw / 2 + furniture_margin and abs(y - fy) < fd / 2 + furniture_margin:
            return False
    return True


def _find_loop(scene: dict, rng, legacy: bool = False) -> tuple[float, float, float, float, float, float]:
    """Return (cx, cy, rx, ry, rot, phase) for a small elliptical loop clear of walls and furniture.

    The check is dense and slightly padded, so every frame position of any frame count clears the nominal margins.
    legacy=True reproduces the original 48-sample, unpadded check (it let some loops clip furniture between samples);
    it exists only so the retired bounded-pitch mode stays bit-identical to the paths the independent audit measured.
    """
    check_angles = np.arange(48 if legacy else 360) / (48 if legacy else 360) * 2 * math.pi; pad = 0. if legacy else .02
    for _ in range(1000):
        rot = float(rng.uniform(0, 2 * math.pi)); phase = float(rng.uniform(0, 2 * math.pi))
        rx = float(rng.uniform(.3, min(.9, scene["width"] * .18)))
        ry = float(rng.uniform(.3, min(1.0, scene["length"] * .18)))
        cx = float(rng.uniform(-.7, .7)); cy = float(rng.uniform(-.8, .8))
        ca = check_angles + phase; lx, ly = rx * np.cos(ca), ry * np.sin(ca)
        xs = cx + lx * math.cos(rot) - ly * math.sin(rot); ys = cy + lx * math.sin(rot) + ly * math.cos(rot)
        if all(camera_is_clear(scene, float(x), float(y), CAMERA_WALL_MARGIN + pad, CAMERA_FURNITURE_MARGIN + pad) for x, y in zip(xs, ys)):
            return cx, cy, rx, ry, rot, phase
    # A tiny orbit around a free grid point is the safe fallback for a very cluttered seed.
    for cx in np.arange(-scene["width"] / 2 + .7, scene["width"] / 2 - .7, .2):
        for cy in np.arange(-scene["length"] / 2 + .7, scene["length"] / 2 - .7, .2):
            if camera_is_clear(scene, float(cx) + .12, float(cy)) and camera_is_clear(scene, float(cx) - .12, float(cy)) \
                    and camera_is_clear(scene, float(cx), float(cy) + .12) and camera_is_clear(scene, float(cx), float(cy) - .12):
                return float(cx), float(cy), .12, .12, 0., 0.
    return 0., 0., .12, .12, 0., 0.


PATH_MODES = ("spherical", "bounded")  # "bounded" is the legacy +-36 degree pitch loop, kept for regression/QA only
SPHERE_PITCH_RANGE_RAD = (math.radians(62), math.radians(64.5))  # amplitude: below ~60 the poles thin out, above ~66 the middle bands do
# (yaw revolutions, pitch cycles) per loop. Coprime pairs only: when pitch cycles divide the yaw revolutions the two
# sweeps phase-lock and trace the same bands (4 revs x 2 cycles covered only ~83% of the sphere). Slower sweeps keep more
# overlap between the 36 frames actually used for inference; (3,4)/(4,3)/(5,x) cover slightly more but retain only ~10% of a
# frame in its successor, which is poor for tracking, and (2,3) drops under the selected-frame gate, so they are excluded.
SPHERE_SWEEP_PATTERNS = ((3, 2), (4, 1))
INFERENCE_FRAME_COUNT = 36  # frames infer_rgb.py samples (evenly) from a video by default
SPHERE_FULL_VIDEO_MIN, SPHERE_SELECTED_MIN, SPHERE_POLAR_MIN = 0.99, 0.97, 0.97
SPHERE_POLAR_ELEVATION_DEG = 60.0
MAX_RELATIVE_ROTATION_DEG = 25.0  # largest SO(3) rotation between consecutive frames of a 120-frame loop
SPHERE_DIRECTIONS = 30000


def make_path(scene: dict, frames: int, path_seed: int, index: int = 0, mode: str = "spherical") -> list[dict]:
    """One continuous collision-free translation loop with a seeded orientation sweep.

    mode "spherical": yaw advances monotonically through several revolutions while pitch oscillates over
    +-62..64.5 degrees (positive looks down), so the camera looks up at the ceiling/pole and down at the floor/nadir
    as well as around the horizon, and roll turns through one or two full revolutions about the optical axis, so
    the camera is upside-down for roughly half the capture. Nothing assumes the camera or its path has a "world up":
    orientation is a smooth curve on SO(3) (see camera_basis for the convention). Revolutions/pitch cycles come from SPHERE_SWEEP_PATTERNS; start, direction,
    amplitude, phases and roll are seeded. Orientation changes smoothly (no jumps) and the loop closes.
    mode "bounded": the legacy single-revolution, +-36 degree pitch loop (about 84-86% of the sphere).
    """
    if mode not in PATH_MODES:
        raise ValueError(f"mode must be one of {PATH_MODES}")
    n = max(2, frames); rng = np.random.default_rng(path_seed)
    cx, cy, rx, ry, rot, phase = _find_loop(scene, rng, legacy=(mode == "bounded"))
    yaw0 = float(rng.uniform(0, 2 * math.pi)); direction = 1. if rng.random() < .5 else -1.
    wobble_phase = float(rng.uniform(0, 2 * math.pi)); wobble = float(rng.uniform(.03, .07))
    if mode == "bounded":
        pitch_amp = float(rng.uniform(.8, 1.) * MAX_PITCH_RAD); pitch_cycles = int(rng.integers(2, 4))
        pitch_phase = float(rng.uniform(0, 2 * math.pi)); revolutions = 1
    else:
        revolutions, pitch_cycles = SPHERE_SWEEP_PATTERNS[int(rng.integers(len(SPHERE_SWEEP_PATTERNS)))]
        pitch_amp = float(rng.uniform(*SPHERE_PITCH_RANGE_RAD)); pitch_phase = float(rng.uniform(0, 2 * math.pi))
        roll_revolutions = int(rng.integers(1, 3)); roll_direction = 1. if rng.random() < .5 else -1.
        roll0 = float(rng.uniform(0, 2 * math.pi)); roll_wobble = float(rng.uniform(.05, .15)); roll_wobble_phase = float(rng.uniform(0, 2 * math.pi))
    path = []
    for i in range(n):
        s = i / n; u = phase + 2 * math.pi * s
        lx, ly = rx * math.cos(u), ry * math.sin(u)
        x = cx + lx * math.cos(rot) - ly * math.sin(rot); y = cy + lx * math.sin(rot) + ly * math.cos(rot)
        z = float(1.43 + .18 * math.sin(2 * u + index))
        yaw = yaw0 + direction * (2 * math.pi * revolutions * s + wobble * math.sin(2 * math.pi * 3 * s + wobble_phase))
        if mode == "bounded":
            pitch = pitch_amp * math.sin(2 * math.pi * pitch_cycles * s + pitch_phase)
        else:
            pitch = pitch_amp * math.cos(2 * math.pi * pitch_cycles * s + pitch_phase)
        roll = 0.0 if mode == "bounded" else roll0 + roll_direction * (2 * math.pi * roll_revolutions * s + roll_wobble * math.sin(2 * math.pi * 2 * s + roll_wobble_phase))
        path.append({"position": [float(x), float(y), z], "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)})
    return path


def make_paths(scene: dict, frames: int, capture_seed: int | None = None, mode: str = "spherical") -> list[list[dict]]:
    """Three independently seeded continuous loops, clear of furniture footprints."""
    return [make_path(scene, frames, s, i, mode) for i, s in enumerate(path_seeds(scene, capture_seed))]


def selected_frame_indices(count: int, wanted: int = INFERENCE_FRAME_COUNT) -> list[int]:
    """The evenly spaced frames infer_rgb.py feeds to the model; independent of any scene knowledge."""
    n = min(count, wanted)
    return [0] if n == 1 else np.linspace(0, count - 1, n, dtype=int).tolist()


def sphere_directions(count: int = SPHERE_DIRECTIONS) -> np.ndarray:
    """Deterministic Fibonacci-sphere unit vectors (equal area, z up like the simulator world)."""
    i = np.arange(count) + .5
    z = 1 - 2 * i / count; phi = math.pi * (3 - math.sqrt(5)) * i; r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def camera_basis(pose: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) world vectors: THE orientation convention, shared by the CPU renderer, Blender and QA.

    World is x east, y north, z up. Yaw turns the camera about +z (yaw 0 faces +y, positive turns toward +x), pitch
    tilts it (positive looks down, so pitch beyond 90 degrees is a legitimate inverted view), then roll turns it about
    its own forward axis by the right-hand rule (positive roll: right -> right*cos(r) - up*sin(r)). The basis is
    right-handed (right x forward = up) and any (yaw, pitch, roll), including upside-down, is a valid orientation.
    """
    yaw, pitch, roll = pose["yaw"], pose["pitch"], pose.get("roll", 0.0)
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.])
    forward = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
    up = np.cross(right, forward)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        right, up = c * right - s * up, s * right + c * up
    return right, up, forward


def rotation_matrix(pose: dict) -> np.ndarray:
    """Proper (det +1) camera-to-world rotation with columns (right, forward, up)."""
    right, up, forward = camera_basis(pose)
    return np.stack([right, forward, up], axis=1)


def relative_rotation_angles_deg(poses: list[dict], closed: bool = True) -> np.ndarray:
    """Geodesic angle of R_i^T R_{i+1} on SO(3): wrap-around and gimbal safe, unlike raw Euler differences."""
    mats = [rotation_matrix(p) for p in poses]
    pairs = list(zip(mats, mats[1:] + ([mats[0]] if closed else [])))
    traces = np.array([np.trace(a.T @ b) for a, b in pairs])
    return np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1)))


def inverted_fraction(poses: list[dict]) -> float:
    """Share of frames whose camera-up has a negative world-z component (upside-down views)."""
    return float(np.mean([camera_basis(p)[1][2] < 0 for p in poses]))


def _seen_masks(poses: list[dict], fov: float, width: int, height: int, directions: np.ndarray) -> np.ndarray:
    half_h = math.tan(math.radians(fov) / 2); half_v = half_h * height / width  # square pixels: real vertical FOV
    masks = np.zeros((len(poses), len(directions)), dtype=bool)
    for k, pose in enumerate(poses):
        right, up, forward = camera_basis(pose)
        f = directions @ forward
        masks[k] = (f > 1e-9) & (np.abs(directions @ right) <= half_h * f) & (np.abs(directions @ up) <= half_v * f)
    return masks


def _coverage_summary(seen: np.ndarray, directions: np.ndarray) -> dict:
    elevation = np.degrees(np.arcsin(np.clip(directions[:, 2], -1, 1)))
    up_cap, down_cap = elevation >= SPHERE_POLAR_ELEVATION_DEG, elevation <= -SPHERE_POLAR_ELEVATION_DEG
    azimuth = (np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) + 360) % 360
    equator = np.abs(elevation) < SPHERE_POLAR_ELEVATION_DEG
    wedges = [float(seen[equator & (azimuth >= a) & (azimuth < a + 45)].mean()) for a in range(0, 360, 45)]
    hole = 0.0
    if not seen.all():
        covered, missing = directions[seen], directions[~seen]
        if len(covered) == 0:
            hole = 180.0
        else:
            nearest = np.concatenate([(missing[i:i + 2000] @ covered.T).max(axis=1) for i in range(0, len(missing), 2000)])
            hole = float(np.degrees(np.arccos(np.clip(nearest, -1, 1))).max())
    return {"fraction": float(seen.mean()), "polar_up_fraction": float(seen[up_cap].mean()),
            "polar_down_fraction": float(seen[down_cap].mean()), "equatorial_azimuth_wedge_min_fraction": min(wedges),
            "max_uncovered_hole_radius_deg": hole}


def angular_coverage(poses: list[dict], fov: float = 78., width: int = 480, height: int = 360,
                     count: int = SPHERE_DIRECTIONS, selected: list[int] | None = None) -> dict:
    """Orientation-only sphere coverage on an equal-area direction set, with the real horizontal and vertical FOV.

    A direction counts as seen when it lies in front of some frame's camera and inside that frame's horizontal and
    vertical half-angles. This IGNORES OCCLUSION and camera translation: it measures where the camera pointed, not
    which surfaces were visible or reconstructable (see coverage_diagnostics for surface hits). The full video and
    the frames actually selected for inference are reported independently.
    """
    directions = sphere_directions(count)
    masks = _seen_masks(poses, fov, width, height, directions)
    chosen = selected_frame_indices(len(poses)) if selected is None else list(selected)
    full, picked = masks.any(axis=0), masks[chosen].any(axis=0)

    def overlap(rows):
        neighbours = list(zip(rows, np.roll(rows, -1, axis=0)))
        jaccard = [(a & b).sum() / max(1, (a | b).sum()) for a, b in neighbours]
        retained = [(a & b).sum() / max(1, a.sum()) for a, b in neighbours]  # share of a frame's view still seen by the next frame
        return {"mean": float(np.mean(jaccard)), "min": float(np.min(jaccard)),
                "retained_mean": float(np.mean(retained)), "retained_min": float(np.min(retained))}
    full_summary, picked_summary = _coverage_summary(full, directions), _coverage_summary(picked, directions)
    full_summary["consecutive_frame_view_overlap"] = overlap(masks)
    picked_summary["consecutive_frame_view_overlap"] = overlap(masks[chosen])
    picked_summary["frame_count"] = len(chosen); picked_summary["indices"] = [int(i) for i in chosen]
    full_summary["frame_count"] = len(poses)
    full_summary["passes_full_sphere"] = bool(full_summary["fraction"] >= SPHERE_FULL_VIDEO_MIN
                                              and min(full_summary["polar_up_fraction"], full_summary["polar_down_fraction"]) >= SPHERE_POLAR_MIN)
    picked_summary["passes_full_sphere"] = bool(picked_summary["fraction"] >= SPHERE_SELECTED_MIN
                                                and min(picked_summary["polar_up_fraction"], picked_summary["polar_down_fraction"]) >= SPHERE_POLAR_MIN)
    return {"orientation_only": True, "ignores_occlusion": True, "ignores_translation": True,
            "note": "angular diagnostic only: where the camera pointed, not visibility of occluded surfaces or reconstruction quality",
            "direction_count": count, "fov_horizontal_deg": fov, "frame_aspect_height_over_width": height / width,
            "fov_vertical_deg": float(math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) * height / width))),
            "gates": {"full_video_min": SPHERE_FULL_VIDEO_MIN, "selected_frames_min": SPHERE_SELECTED_MIN,
                      "polar_min": SPHERE_POLAR_MIN, "polar_elevation_deg": SPHERE_POLAR_ELEVATION_DEG},
            "full_video": full_summary, "selected_inference_frames": picked_summary,
            "passes_full_sphere": bool(full_summary["passes_full_sphere"] and picked_summary["passes_full_sphere"])}


def camera_motion_summary(scene: dict, path: list[dict]) -> dict:
    """Translation baseline (parallax), clearance and orientation smoothness for one path."""
    pos = np.asarray([p["position"] for p in path]); yaw = np.unwrap([p["yaw"] for p in path]); pitch = np.asarray([p["pitch"] for p in path])
    diffs = pos[:, None, :2] - pos[None, :, :2]
    step = relative_rotation_angles_deg(path); chosen = [path[i] for i in selected_frame_indices(len(path))]
    chosen_step = relative_rotation_angles_deg(chosen); roll = np.unwrap([p.get("roll", 0.0) for p in path])
    return {"horizontal_baseline_m": float(np.sqrt((diffs ** 2).sum(-1)).max()),
            "max_relative_rotation_deg": float(step.max()), "mean_relative_rotation_deg": float(step.mean()),
            "selected_max_relative_rotation_deg": float(chosen_step.max()), "selected_mean_relative_rotation_deg": float(chosen_step.mean()),
            "inverted_fraction": inverted_fraction(path), "roll_range_deg": float(np.degrees(roll.max() - roll.min())),
            "max_yaw_step_deg": float(np.degrees(np.abs(np.diff(yaw)).max())), "max_pitch_step_deg": float(np.degrees(np.abs(np.diff(pitch)).max())),
            "pitch_range_deg": [float(np.degrees(pitch.min())), float(np.degrees(pitch.max()))],
            "furniture_or_wall_violations": int(sum(not camera_is_clear(scene, p["position"][0], p["position"][1]) for p in path)),
            "min_wall_clearance_m": float(min(min(scene["width"] / 2 - abs(p["position"][0]), scene["length"] / 2 - abs(p["position"][1])) for p in path))}


def capture_qa(scene_seeds, capture_seed: int = 3101, frames: int = 120, fov: float = 78., width: int = 480, height: int = 360,
               modes=("spherical", "bounded")) -> dict:
    """CPU QA over many scenes x three path seeds; never a statement about reconstruction."""
    rows = []; budgets = (24, 36, 48, 72, 120); directions = sphere_directions()
    for scene_seed in scene_seeds:
        scene = make_scene(scene_seed)
        for mode in modes:
            for pi, path in enumerate(make_paths(scene, frames, capture_seed, mode)):
                report = angular_coverage(path, fov, width, height); motion = camera_motion_summary(scene, path)
                masks = _seen_masks(path, fov, width, height, directions); budget = {}
                for b in budgets:
                    rows_b = masks[selected_frame_indices(len(path), b)]
                    kept = [(x & y).sum() / max(1, x.sum()) for x, y in zip(rows_b, np.roll(rows_b, -1, axis=0))]
                    budget[str(b)] = {"coverage": float(rows_b.any(axis=0).mean()), "retained_min": float(np.min(kept)), "retained_mean": float(np.mean(kept))}
                rows.append({"scene_seed": scene_seed, "path": pi, "mode": mode, "room_m": [round(scene[k], 3) for k in ("width", "length", "height")],
                             "full_video_fraction": report["full_video"]["fraction"],
                             "selected_fraction": report["selected_inference_frames"]["fraction"],
                             "polar_up_full": report["full_video"]["polar_up_fraction"], "polar_down_full": report["full_video"]["polar_down_fraction"],
                             "polar_up_selected": report["selected_inference_frames"]["polar_up_fraction"],
                             "polar_down_selected": report["selected_inference_frames"]["polar_down_fraction"],
                             "max_hole_deg_full": report["full_video"]["max_uncovered_hole_radius_deg"],
                             "max_hole_deg_selected": report["selected_inference_frames"]["max_uncovered_hole_radius_deg"],
                             "overlap_mean_full": report["full_video"]["consecutive_frame_view_overlap"]["mean"],
                             "overlap_min_full": report["full_video"]["consecutive_frame_view_overlap"]["min"],
                             "overlap_mean_selected": report["selected_inference_frames"]["consecutive_frame_view_overlap"]["mean"],
                             "overlap_min_selected": report["selected_inference_frames"]["consecutive_frame_view_overlap"]["min"],
                             "retained_min_selected": report["selected_inference_frames"]["consecutive_frame_view_overlap"]["retained_min"],
                             "retained_mean_selected": report["selected_inference_frames"]["consecutive_frame_view_overlap"]["retained_mean"],
                             "frame_budget": budget,
                             "passes_full_sphere": report["passes_full_sphere"], **motion})
    summary = {}
    for mode in modes:
        sel = [r for r in rows if r["mode"] == mode]
        summary[mode] = {"paths": len(sel), "passes_full_sphere": sum(r["passes_full_sphere"] for r in sel),
                         "full_video_fraction_min": min(r["full_video_fraction"] for r in sel),
                         "full_video_fraction_mean": float(np.mean([r["full_video_fraction"] for r in sel])),
                         "selected_fraction_min": min(r["selected_fraction"] for r in sel),
                         "selected_fraction_mean": float(np.mean([r["selected_fraction"] for r in sel])),
                         "polar_min_selected": min(min(r["polar_up_selected"], r["polar_down_selected"]) for r in sel),
                         "max_hole_deg_selected": max(r["max_hole_deg_selected"] for r in sel),
                         "overlap_min_selected": min(r["overlap_min_selected"] for r in sel),
                         "overlap_mean_selected": float(np.mean([r["overlap_mean_selected"] for r in sel])),
                         "retained_min_selected": min(r["retained_min_selected"] for r in sel),
                         "retained_mean_selected": float(np.mean([r["retained_mean_selected"] for r in sel])),
                         "frame_budget_tradeoff": {str(b): {"coverage_min": min(r["frame_budget"][str(b)]["coverage"] for r in sel),
                                                           "coverage_mean": float(np.mean([r["frame_budget"][str(b)]["coverage"] for r in sel])),
                                                           "retained_min": min(r["frame_budget"][str(b)]["retained_min"] for r in sel),
                                                           "retained_mean": float(np.mean([r["frame_budget"][str(b)]["retained_mean"] for r in sel]))}
                                                   for b in budgets},
                         "collision_or_wall_violations": sum(r["furniture_or_wall_violations"] for r in sel),
                         "max_yaw_step_deg": max(r["max_yaw_step_deg"] for r in sel),
                         "max_relative_rotation_deg": max(r["max_relative_rotation_deg"] for r in sel),
                         "selected_max_relative_rotation_deg": max(r["selected_max_relative_rotation_deg"] for r in sel),
                         "inverted_fraction_min": min(r["inverted_fraction"] for r in sel),
                         "inverted_fraction_max": max(r["inverted_fraction"] for r in sel),
                         "roll_range_deg_min": min(r["roll_range_deg"] for r in sel),
                         "min_horizontal_baseline_m": min(r["horizontal_baseline_m"] for r in sel)}
    return {"schema": "capture_qa_v1", "parameters": {"scene_seeds": list(scene_seeds), "capture_seed": capture_seed, "frames": frames,
            "selected_frames": INFERENCE_FRAME_COUNT, "fov_horizontal_deg": fov, "width": width, "height": height, "direction_count": SPHERE_DIRECTIONS},
            "gates": {"full_video_min": SPHERE_FULL_VIDEO_MIN, "selected_frames_min": SPHERE_SELECTED_MIN, "polar_min": SPHERE_POLAR_MIN},
            "summary": summary, "rows": rows,
            "claims": "none: orientation coverage and collision checks only; nothing here says any room was reconstructed or that a gate passed"}


# Surface kinds returned by render(kind_out=True).
KIND_FLOOR, KIND_CEILING, KIND_OPENING, KIND_FURNITURE0 = 0, 1, 7, 8
WALL_NAMES = ["north", "east", "south", "west"]
MIN_PIXEL_FRACTION = {"wall": .02, "floor": .02, "ceiling": .02, "doorway": .004, "window": .004}
MIN_FRAMES = 5


def _opening_at(scene: dict, pos: np.ndarray) -> np.ndarray:
    """Index of the scene opening containing each aperture hit point, -1 where none matches."""
    w, length = scene["width"], scene["length"]
    which = np.full(pos.shape[0], -1, int)
    for oi, o in enumerate(scene["openings"]):
        wi = o["wall"]; axis, side = [(1, 1), (0, 1), (1, -1), (0, -1)][wi]
        boundary = (length / 2 if axis == 1 else w / 2) * side
        along = pos[:, 0] if axis == 1 else pos[:, 1]
        hit = (np.abs(pos[:, axis] - boundary) < 1e-2) & (np.abs(along - o["center"]) <= o["width"] / 2 + 1e-3) \
            & (pos[:, 2] >= o["z_min"] - 1e-3) & (pos[:, 2] <= o["z_max"] + 1e-3)
        which[hit & (which < 0)] = oi
    return which


def coverage_diagnostics(scene: dict, path: list[dict], fov: float = 78., width: int = 64, height: int = 48) -> dict:
    """Ray-cast coverage of one path against the private scene (capture-side QA, never reconstruction input).

    Reports which surfaces the camera actually sees: four wall directions, floor, ceiling, doorway,
    window, per-wall/floor/ceiling spatial bins, frame-to-frame overlap and camera clearance.
    """
    w, length = scene["width"], scene["length"]
    names = [f"wall_{n}" for n in WALL_NAMES] + ["floor", "ceiling", "doorway", "window", "furniture"]
    pixels = {n: 0 for n in names}; frames_seen = {n: 0 for n in names}; total = 0
    wall_bins = {n: np.zeros(4, int) for n in WALL_NAMES}
    fc_cells = {"floor": np.zeros((3, 3), int), "ceiling": np.zeros((3, 3), int)}
    cell_sets = []; nframes = len(path)
    for pose in path:
        with np.errstate(all="ignore"):  # axis-parallel rays divide by zero in the plane tests; those hits are masked out
            _, _, _, kind, pos = render(scene, pose, width, height, fov, True, True)
        flat_kind = kind.reshape(-1); flat_pos = pos.reshape(-1, 3); total += flat_kind.size
        counts = {n: 0 for n in names}; keys = []
        counts["floor"] = int((flat_kind == KIND_FLOOR).sum()); counts["ceiling"] = int((flat_kind == KIND_CEILING).sum())
        for wi, wn in enumerate(WALL_NAMES):
            m = flat_kind == 2 + wi; counts[f"wall_{wn}"] = int(m.sum())
            if m.any():
                p = flat_pos[m]; axis = 0 if wi in (0, 2) else 1; span = w if axis == 0 else length
                b = np.clip(((p[:, axis] + span / 2) / span * 4).astype(int), 0, 3)
                wall_bins[wn] += np.bincount(b, minlength=4)
        op = flat_kind == KIND_OPENING
        if op.any():
            which = _opening_at(scene, flat_pos[op]); types = np.array([o["type"] for o in scene["openings"]] + ["?"])
            for t in ("doorway", "window"): counts[t] = int((types[which] == t).sum())
        counts["furniture"] = int((flat_kind >= KIND_FURNITURE0).sum())
        for n in names: pixels[n] += counts[n]; frames_seen[n] += int(counts[n] >= MIN_PIXEL_FRACTION.get("wall" if n.startswith("wall_") else n, .004) * flat_kind.size)
        for fc, k in (("floor", KIND_FLOOR), ("ceiling", KIND_CEILING)):
            m = flat_kind == k
            if m.any():
                p = flat_pos[m]; cx = np.clip(((p[:, 0] / w + .5) * 3).astype(int), 0, 2); cy = np.clip(((p[:, 1] / length + .5) * 3).astype(int), 0, 2)
                np.add.at(fc_cells[fc], (cx, cy), 1)
        # Coarse world cells (0.5 m) of everything seen, for frame overlap.
        valid = flat_kind >= 0
        q = np.floor(flat_pos[valid][::7] / .5).astype(int) + 40
        cell_sets.append(set(((flat_kind[valid][::7].astype(int) * 100 + q[:, 0]) * 100 + q[:, 1]) * 100 + q[:, 2]))
    overlaps = [len(a & b) / max(1, len(a | b)) for a, b in zip(cell_sets, cell_sets[1:] + cell_sets[:1])]
    yaw = np.array([p["yaw"] for p in path]); pitch = np.array([p["pitch"] for p in path])
    yaw_step = np.abs(np.diff(yaw)) if nframes > 1 else np.zeros(1)
    clearance = min(min(w / 2 - abs(p["position"][0]), length / 2 - abs(p["position"][1])) for p in path)
    observed = {}
    for n in names:
        if n == "furniture": continue
        frac = MIN_PIXEL_FRACTION["wall" if n.startswith("wall_") else n]
        observed[n] = bool(frames_seen[n] >= MIN_FRAMES and pixels[n] >= frac * total * .02)
    fc_frac = {k: float((v > 0).mean()) for k, v in fc_cells.items()}
    observed["floor_spread"] = fc_frac["floor"] >= 5 / 9; observed["ceiling_spread"] = fc_frac["ceiling"] >= 5 / 9
    return {"frames": nframes, "pixel_fraction": {n: pixels[n] / max(1, total) for n in names},
            "frames_observing": frames_seen, "observed": observed,
            "wall_along_bins_pixels": {n: v.tolist() for n, v in wall_bins.items()},
            "floor_cells_observed_fraction": fc_frac["floor"], "ceiling_cells_observed_fraction": fc_frac["ceiling"],
            "frame_overlap_mean": float(np.mean(overlaps)), "frame_overlap_min": float(np.min(overlaps)),
            "yaw_range_deg": float(math.degrees(yaw.max() - yaw.min())), "max_yaw_step_deg": float(math.degrees(yaw_step.max())),
            "pitch_min_deg": float(math.degrees(pitch.min())), "pitch_max_deg": float(math.degrees(pitch.max())),
            "min_wall_clearance_m": float(clearance),
            "furniture_collisions": int(sum(not camera_is_clear(scene, p["position"][0], p["position"][1], 0., 0.) for p in path)),
            "complete": bool(all(observed.values()))}


def union_coverage(per_path: list[dict]) -> dict:
    """OR of per-path observations: the capture set (not any single path) must cover every surface."""
    keys = per_path[0]["observed"].keys()
    observed = {k: any(d["observed"][k] for d in per_path) for k in keys}
    return {"paths": len(per_path), "observed": observed, "complete": bool(all(observed.values()))}


def render(scene: dict, pose: dict, width: int, height: int, fov: float, depth_out: bool = False, kind_out: bool = False):
    w, length, room_h = scene["width"], scene["length"], scene["height"]
    fx = width / (2 * math.tan(math.radians(fov) / 2)); fy = fx
    cx, cy = (width - 1) / 2, (height - 1) / 2
    yy, xx = np.mgrid[0:height, 0:width]
    cam = np.stack(((xx-cx)/fx, -(yy-cy)/fy, np.ones_like(xx)), axis=-1).astype(np.float32)
    right, up, forward = (v.astype(np.float32) for v in camera_basis(pose))
    dirs = cam[..., 0, None]*right + cam[..., 1, None]*up + cam[..., 2, None]*forward
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origin = np.array(pose["position"], np.float32)
    best = np.full((height, width), np.inf, np.float32); rgb = np.zeros((height, width, 3), np.float32)
    hit_kind = np.full((height, width), -1, np.int8); hitpos = np.zeros_like(dirs)

    def accept(t, mask, kind, color_fn):
        nonlocal best, rgb, hit_kind, hitpos
        valid = mask & (t > .02) & (t < best)
        if not np.any(valid): return
        pos = origin + dirs * t[..., None]
        col = color_fn(pos)
        rgb[valid] = col[valid]; best[valid] = t[valid]; hit_kind[valid] = kind; hitpos[valid] = pos[valid]

    # Floor and ceiling.
    for z, kind, base in [(0., 0, scene["floor_color"]), (room_h, 1, scene["ceiling_color"])]:
        t = (z-origin[2]) / dirs[..., 2]
        pos = origin + dirs*t[..., None]
        mask = (np.abs(pos[..., 0]) <= w/2) & (np.abs(pos[..., 1]) <= length/2)
        def plane_color(pos, base=np.array(base, np.float32), kind=kind):
            grain = 0.035*np.sin(pos[..., 0]*9 + pos[..., 1]*5) + 0.025*np.sin(pos[..., 0]*23-pos[..., 1]*13)
            checker = .045*((np.floor((pos[...,0]+w/2)*2)+np.floor((pos[...,1]+length/2)*2))%2)
            v = grain + (checker if kind == 0 else 0)
            return np.clip(base + v[...,None], .04, .96)
        accept(t, mask, kind, plane_color)

    # Four walls; opening apertures expose a deterministic outdoor/corridor background.
    for wi, (axis, side, span, name) in enumerate([(1, 1, w, "north"), (0, 1, length, "east"),
                                                    (1, -1, w, "south"), (0, -1, length, "west")]):
        boundary = (length/2 if axis == 1 else w/2) * side
        comp = dirs[..., axis]
        t = (boundary-origin[axis]) / comp
        pos = origin + dirs*t[...,None]
        along = pos[..., 0] if axis == 1 else pos[...,1]
        mask = (pos[...,2] >= 0) & (pos[...,2] <= room_h) & (np.abs(along) <= span/2)
        opening = np.zeros((height,width), bool)
        for o in scene["openings"]:
            if o["wall"] == wi:
                opening |= (np.abs(along-o["center"]) < o["width"]/2) & (pos[...,2] >= o["z_min"]) & (pos[...,2] <= o["z_max"])
        wallmask = mask & ~opening
        base = np.array(scene["wall_colors"][wi], np.float32)
        def wall_color(pos, base=base, wi=wi, axis=axis):
            u = pos[..., 0] if axis == 1 else pos[...,1]
            pattern = .06*np.sin(u*8 + wi) + .025*np.sin(pos[...,2]*21 + u*3)
            seam = np.where((np.floor(pos[...,2]*3)%3)==0, -.035, 0)
            return np.clip(base + (pattern+seam)[...,None], .05,.95)
        accept(t, wallmask, 2+wi, wall_color)
        # Apertures are visual backgrounds rather than simulated adjacent rooms.
        def outside_color(pos, wi=wi):
            z = pos[...,2]
            sky = np.array([.43,.63,.76],np.float32)
            dark = np.array([.15,.18,.2],np.float32)
            horizon = np.clip((1.7-z)*.45, 0, 1)[...,None]
            col = sky*(1-horizon)+dark*horizon
            return np.clip(col + .06*np.sin(pos[...,0:1]*4+pos[...,1:2]*3),0,1)
        accept(t, mask & opening, 7, outside_color)

    # Furniture as axis-aligned boxes with top and side hits.
    for fi, box in enumerate(scene["furniture"]):
        c=np.array(box["center"],np.float32); size=np.array(box["size"],np.float32)
        lo=np.array([c[0]-size[0]/2,c[1]-size[1]/2,0],np.float32)
        hi=np.array([c[0]+size[0]/2,c[1]+size[1]/2,size[2]],np.float32)
        inv=1/np.where(np.abs(dirs)<1e-8, np.sign(dirs)*1e-8+1e-8, dirs)
        t0=(lo-origin)*inv; t1=(hi-origin)*inv
        near=np.maximum(np.minimum(t0,t1).max(axis=-1), .02); far=np.minimum(np.maximum(t0,t1).min(axis=-1), np.inf)
        t=np.where(near<far,near,far)
        mask=(near<far)&(t>0)
        color=np.array(box["color"],np.float32)
        def furniture_color(pos,color=color,fi=fi):
            grain=.035*np.sin(pos[...,0]*13+pos[...,1]*17+fi)
            return np.clip(color+grain[...,None],.02,.95)
        accept(t,mask,8+fi,furniture_color)

    # Gentle distance falloff preserves contrast without pretending to be a sensor model.
    rgb = np.clip(rgb * (.94 - .025*np.minimum(best, 8)[...,None]), 0, 1)
    image = (rgb*255).astype(np.uint8)
    best[hit_kind == 7] = np.nan
    if depth_out:
        intrinsics = {"fx":fx,"fy":fy,"cx":cx,"cy":cy}
        if kind_out: return image, best.astype(np.float32), intrinsics, hit_kind, hitpos
        return image, best.astype(np.float32), intrinsics
    return image


def write_coverage(private: Path, scene: dict, paths: list[list[dict]], fov: float, seeds: list[int],
                   width: int = 480, height: int = 360, mode: str = "spherical") -> dict:
    """Write capture-coverage diagnostics under private_ground_truth (they derive from the private scene).

    Per path: surface-hit coverage (coverage_diagnostics) and orientation-only sphere coverage (angular_coverage)
    for the full video and the evenly selected inference frames. Neither says anything about reconstruction.
    """
    per_path = [coverage_diagnostics(scene, p, fov) for p in paths]
    angular = [angular_coverage(p, fov, width, height) for p in paths]
    report = {"note": "capture diagnostics from the private scene; a capture input check, not reconstruction accuracy",
              "path_mode": mode, "path_seeds": seeds, "paths": per_path, "union": union_coverage(per_path),
              "angular_coverage": angular, "angular_all_paths_pass_full_sphere": all(a["passes_full_sphere"] for a in angular),
              "camera_motion": [camera_motion_summary(scene, p) for p in paths]}
    (private / "coverage_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate(output: Path, seed=1, width=320, height=240, frames=12, fov=78., fps=12, capture_seed=None, diagnostics_only=False,
             mode="spherical"):
    scene=make_scene(seed); seeds=path_seeds(scene,capture_seed)
    output.mkdir(parents=True,exist_ok=True)
    private=output/"private_ground_truth"; private.mkdir(exist_ok=True)
    (private/"scene.json").write_text(json.dumps(scene,indent=2),encoding="utf-8")
    # Seeds can regenerate the room and paths, so they live with the private truth, not beside the RGB.
    (private/"seed_manifest.json").write_text(json.dumps({"seed":seed,"capture_seed":seed if capture_seed is None else capture_seed,
        "path_seeds":seeds},indent=2),encoding="utf-8")
    (output/"capture_manifest.json").write_text(json.dumps({"width":width,"height":height,"frames_per_path":frames,"paths":PATH_COUNT,
        "fov_degrees":fov,"fps":fps,"renderer":"procedural_cpu_ray_box_v1","rgb_only":True,"path_mode":mode},indent=2),encoding="utf-8")
    paths=make_paths(scene,frames,capture_seed,mode)
    write_coverage(private,scene,paths,fov,seeds,width,height,mode)
    if diagnostics_only: return
    import cv2  # deferred so path generation and coverage diagnostics need only numpy
    (output/"rgb").mkdir(exist_ok=True); (output/"videos").mkdir(exist_ok=True)
    for pi,path in enumerate(paths):
        name=f"path_{pi:02d}"; rgbdir=output/"rgb"/name; rgbdir.mkdir(exist_ok=True)
        video=None; poses=[]; k=None
        for fi,pose in enumerate(path):
            image,depth,k=render(scene,pose,width,height,fov,True)
            cv2.imwrite(str(rgbdir/f"{fi:06d}.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            poses.append(pose)
            if video is None:
                video=cv2.VideoWriter(str(output/"videos"/f"{name}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
                if not video.isOpened(): raise RuntimeError("OpenCV could not open the MP4 video writer")
            video.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        video.release()
        (private/f"{name}.json").write_text(json.dumps({"path_seed":seeds[pi],"intrinsics":k,"poses":poses},indent=2),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=Path("outputs/sim-benchmark/smoke")); ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--width",type=int,default=320); ap.add_argument("--height",type=int,default=240)
    ap.add_argument("--frames",type=int,default=12); ap.add_argument("--fov",type=float,default=78); ap.add_argument("--fps",type=int,default=12)
    ap.add_argument("--capture-seed",type=int,default=None)
    ap.add_argument("--diagnostics-only",action="store_true",help="write coverage diagnostics without rendering frames or video")
    ap.add_argument("--path-mode",choices=PATH_MODES,default="spherical")
    ap.add_argument("--sphere-qa",type=Path,help="write a versioned CPU capture-QA JSON (orientation coverage, collisions, overlap) and exit")
    ap.add_argument("--qa-scenes",type=int,default=10,help="number of scene seeds (1..N) for --sphere-qa")
    a=ap.parse_args()
    if min(a.width,a.height,a.frames,a.fps)<=0 or not (10<a.fov<150): ap.error("dimensions, frames, fps must be positive and fov must be between 10 and 150 degrees")
    if a.sphere_qa is not None:
        if a.sphere_qa.exists(): ap.error(f"QA output already exists: {a.sphere_qa}")
        qa=capture_qa(range(1,a.qa_scenes+1),3101 if a.capture_seed is None else a.capture_seed,a.frames,a.fov,a.width,a.height)
        a.sphere_qa.parent.mkdir(parents=True,exist_ok=True); a.sphere_qa.write_text(json.dumps(qa,indent=2),encoding="utf-8")
        print(a.sphere_qa); print(json.dumps(qa["summary"],indent=2)); return
    generate(a.output,a.seed,a.width,a.height,a.frames,a.fov,a.fps,a.capture_seed,a.diagnostics_only,a.path_mode)

if __name__=="__main__": main()
