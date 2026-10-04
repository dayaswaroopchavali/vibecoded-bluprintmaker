"""Independent wide-baseline capture-path search and capture-side QA.

Writes poses accepted by blender_capture.py --poses-json.  The optimizer uses
only scene geometry for capture-side collision checking; it is not an inference
component.  Orientation follows the simulator's continuous spherical sweep.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import simulator


def _ellipse(scene: dict, seed: int, tries: int = 12000) -> tuple[float, float, float, float, float, float]:
    """Find a large valid ellipse using dense continuous-path samples.

    Sampling includes padding beyond the camera clearance test, making the
    eventual 120-frame path safely clear between samples as well.
    """
    rng = np.random.default_rng(seed)
    margin = simulator.CAMERA_WALL_MARGIN + .055
    fm = simulator.CAMERA_FURNITURE_MARGIN + .055
    max_rx = scene["width"] / 2 - margin
    max_ry = scene["length"] / 2 - margin
    theta = np.arange(720) * (2 * math.pi / 720)
    cands = []
    for _ in range(tries):
        # Broad radii are preferred; random center and rotation let the orbit
        # route around the three furniture footprints.
        rx = float(rng.uniform(max(.55, max_rx * .48), max_rx))
        ry = float(rng.uniform(max(.60, max_ry * .48), max_ry))
        rot = float(rng.uniform(0, 2 * math.pi))
        # Bound the ellipse's rotated extents to the interior room rectangle.
        ex = math.sqrt((rx * math.cos(rot)) ** 2 + (ry * math.sin(rot)) ** 2)
        ey = math.sqrt((rx * math.sin(rot)) ** 2 + (ry * math.cos(rot)) ** 2)
        if ex > max_rx or ey > max_ry:
            continue
        cx = float(rng.uniform(-max_rx + ex, max_rx - ex)) if ex < max_rx else 0.
        cy = float(rng.uniform(-max_ry + ey, max_ry - ey)) if ey < max_ry else 0.
        phase = float(rng.uniform(0, 2 * math.pi))
        a = theta + phase
        lx, ly = rx * np.cos(a), ry * np.sin(a)
        xs = cx + lx * math.cos(rot) - ly * math.sin(rot)
        ys = cy + lx * math.sin(rot) + ly * math.cos(rot)
        clear = (np.abs(xs) < scene["width"] / 2 - margin) & (np.abs(ys) < scene["length"] / 2 - margin)
        for f in scene["furniture"]:
            fx, fy = f["center"]; fw, fd = f["size"][:2]
            clear &= ~((np.abs(xs - fx) < fw / 2 + fm) & (np.abs(ys - fy) < fd / 2 + fm))
        if clear.all():
            # Broad area is the main objective, with longest baseline a small
            # tie breaker. This retains different valid ellipses by seed.
            score = rx * ry + .08 * max(rx, ry)
            cands.append((score, cx, cy, rx, ry, rot, phase))
    if not cands:
        # Structured fallback: sweep a center grid and descending radii.  This
        # is intentionally exhaustive over a modest grid to avoid giving up
        # when random ellipse orientations happen to hit furniture.
        centers_x = np.arange(-max_rx + .35, max_rx - .34, .25)
        centers_y = np.arange(-max_ry + .35, max_ry - .34, .25)
        radii_x = np.linspace(max_rx, .28, 20)
        radii_y = np.linspace(max_ry, .28, 24)
        best_fallback = None
        for cx in centers_x:
            for cy in centers_y:
                if not simulator.camera_is_clear(scene, float(cx), float(cy), margin, fm):
                    continue
                for rx in radii_x:
                    if abs(cx) + rx >= max_rx:
                        continue
                    for ry in radii_y:
                        if abs(cy) + ry >= max_ry:
                            continue
                        score = float(rx * ry + .08 * max(rx, ry))
                        if best_fallback is not None and score <= best_fallback[0]:
                            continue
                        xs, ys = cx + rx * np.cos(theta), cy + ry * np.sin(theta)
                        clear = (np.abs(xs) < scene["width"] / 2 - margin) & (np.abs(ys) < scene["length"] / 2 - margin)
                        for f in scene["furniture"]:
                            fx, fy = f["center"]; fw, fd = f["size"][:2]
                            clear &= ~((np.abs(xs - fx) < fw / 2 + fm) & (np.abs(ys - fy) < fd / 2 + fm))
                        if clear.all():
                            best_fallback = (score, float(cx), float(cy), float(rx), float(ry))
        if best_fallback is not None:
            _, cx, cy, rx, ry = best_fallback
            return cx, cy, rx, ry, 0., float(rng.uniform(0, 2 * math.pi))
        raise RuntimeError(f"no collision-free ellipse found for scene seed {scene.get('seed')}")
    _, cx, cy, rx, ry, rot, phase = max(cands)
    return cx, cy, rx, ry, rot, phase


def _wide_positions(scene: dict, seed: int, frames: int) -> np.ndarray:
    """Find a broad, collision-free rounded-rectangle loop, then sample it at
    nearly uniform travel distance. Such loops can pass outside central
    furniture even when no large ellipse can.
    """
    rng = np.random.default_rng(seed)
    theta = np.arange(1440) * (2 * math.pi / 1440)
    pad = .045
    candidates = []
    for margin in (.50, .56, .64, .74, .86):
        ax, ay = scene["width"] / 2 - margin, scene["length"] / 2 - margin
        if min(ax, ay) <= .25:
            continue
        xs0 = np.arange(-min(.6, ax * .3), min(.6, ax * .3) + 1e-6, .3)
        ys0 = np.arange(-min(.6, ay * .3), min(.6, ay * .3) + 1e-6, .3)
        for power in (2, 3, 4, 5, 6, 8, 10):
            cs = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2 / power)
            sn = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2 / power)
            for cx in xs0:
                for cy in ys0:
                    xs, ys = float(cx) + ax * cs, float(cy) + ay * sn
                    clear = (np.abs(xs) < scene["width"] / 2 - simulator.CAMERA_WALL_MARGIN - pad) & \
                            (np.abs(ys) < scene["length"] / 2 - simulator.CAMERA_WALL_MARGIN - pad)
                    for f in scene["furniture"]:
                        fx, fy = f["center"]; fw, fd = f["size"][:2]
                        clear &= ~((np.abs(xs - fx) < fw / 2 + simulator.CAMERA_FURNITURE_MARGIN + pad) &
                                   (np.abs(ys - fy) < fd / 2 + simulator.CAMERA_FURNITURE_MARGIN + pad))
                    if clear.all():
                        area = abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))) / 2
                        baseline = float(np.hypot(np.ptp(xs), np.ptp(ys)))
                        candidates.append((area + .03 * baseline + float(rng.uniform(0, .02)), xs, ys))
    if candidates:
        _, xs, ys = max(candidates, key=lambda r: r[0])
        shift = int(rng.integers(len(xs)))
        xs, ys = np.roll(xs, -shift), np.roll(ys, -shift)
        if rng.random() < .5:
            xs, ys = xs[::-1], ys[::-1]
        ds = np.hypot(np.diff(np.r_[xs, xs[0]]), np.diff(np.r_[ys, ys[0]]))
        arc = np.r_[0., np.cumsum(ds[:-1])]
        targets = np.arange(frames) * (arc[-1] + ds[-1]) / frames
        arc = np.r_[arc, arc[-1] + ds[-1]]
        return np.stack([np.interp(targets, arc, np.r_[v, v[0]]) for v in (xs, ys)], axis=1)
    cx, cy, rx, ry, rot, phase = _ellipse(scene, seed)
    a = phase + np.arange(frames) * 2 * math.pi / frames
    lx, ly = rx * np.cos(a), ry * np.sin(a)
    return np.stack([cx + lx * math.cos(rot) - ly * math.sin(rot),
                     cy + lx * math.sin(rot) + ly * math.cos(rot)], axis=1)


def make_wide_path(scene: dict, frames: int, path_seed: int, index: int = 0) -> list[dict]:
    """Large smooth room-perimeter loop plus a periodic SO(3) sphere sweep."""
    n = max(2, int(frames))
    rng = np.random.default_rng(path_seed)
    xy = _wide_positions(scene, path_seed ^ 0x5A17, n)
    yaw0 = float(rng.uniform(0, 2 * math.pi)); direction = 1. if rng.random() < .5 else -1.
    wobble_phase = float(rng.uniform(0, 2 * math.pi)); wobble = float(rng.uniform(.03, .07))
    revolutions, pitch_cycles = simulator.SPHERE_SWEEP_PATTERNS[int(rng.integers(len(simulator.SPHERE_SWEEP_PATTERNS)))]
    pitch_amp = float(rng.uniform(*simulator.SPHERE_PITCH_RANGE_RAD)); pitch_phase = float(rng.uniform(0, 2 * math.pi))
    roll_revolutions = int(rng.integers(1, 3)); roll_direction = 1. if rng.random() < .5 else -1.
    roll0 = float(rng.uniform(0, 2 * math.pi)); roll_wobble = float(rng.uniform(.05, .15))
    roll_phase = float(rng.uniform(0, 2 * math.pi))
    result = []
    for i in range(n):
        s = i / n; u = 2 * math.pi * s
        x, y = xy[i]
        z = 1.43 + .18 * math.sin(2 * u + index)
        yaw = yaw0 + direction * (2 * math.pi * revolutions * s + wobble * math.sin(2 * math.pi * 3 * s + wobble_phase))
        pitch = pitch_amp * math.cos(2 * math.pi * pitch_cycles * s + pitch_phase)
        roll = roll0 + roll_direction * (2 * math.pi * roll_revolutions * s + roll_wobble * math.sin(2 * math.pi * 2 * s + roll_phase))
        result.append({"position": [float(x), float(y), float(z)], "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)})
    return result


def _row(scene: dict, path: list[dict], name: str) -> dict:
    motion = simulator.camera_motion_summary(scene, path)
    angular = simulator.angular_coverage(path, count=6000)
    # Coverage ray casts are deliberately low-resolution QA, not input to pose
    # generation or reconstruction.
    surface = simulator.coverage_diagnostics(scene, path, width=40, height=30)
    return {"path": name, "frames": len(path), "horizontal_baseline_m": motion["horizontal_baseline_m"],
            "max_adjacent_so3_deg": motion["max_relative_rotation_deg"],
            "selected_max_adjacent_so3_deg": motion["selected_max_relative_rotation_deg"],
            "inverted_fraction": motion["inverted_fraction"], "roll_range_deg": motion["roll_range_deg"],
            "violations_with_margins": motion["furniture_or_wall_violations"],
            "min_wall_clearance_m": motion["min_wall_clearance_m"],
            "angular_coverage_full": angular["full_video"]["fraction"],
            "angular_coverage_selected": angular["selected_inference_frames"]["fraction"],
            "polar_up_selected": angular["selected_inference_frames"]["polar_up_fraction"],
            "polar_down_selected": angular["selected_inference_frames"]["polar_down_fraction"],
            "surface_coverage_complete": surface["complete"], "furniture_collisions": surface["furniture_collisions"],
            "surface_observed": surface["observed"]}


def run(output: Path, scene_seeds: list[int], frames: int = 120, capture_seed: int = 3101) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private_ground_truth"; private.mkdir(exist_ok=True)
    rows = []
    for ss in scene_seeds:
        scene = simulator.make_scene(int(ss))
        if int(ss) == int(scene_seeds[0]):
            (private / f"scene_seed_{int(ss)}.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
        seeds = simulator.path_seeds(scene, capture_seed)
        old_paths = simulator.make_paths(scene, frames, capture_seed)
        new_paths = [make_wide_path(scene, frames, ps, i) for i, ps in enumerate(seeds)]
        for i, (old, new) in enumerate(zip(old_paths, new_paths)):
            for label, path in (("old", old), ("wide", new)):
                row = _row(scene, path, f"{ss}:{i}:{label}")
                row.update(scene_seed=int(ss), path_index=i, protocol=label)
                rows.append(row)
            if int(ss) == int(scene_seeds[0]):
                (private / f"path_{i:02d}.json").write_text(json.dumps({"poses": new_paths[i]}, indent=2), encoding="utf-8")
    summary = {}
    for protocol in ("old", "wide"):
        subset = [r for r in rows if r["protocol"] == protocol]
        summary[protocol] = {
            "paths": len(subset),
            "baseline_mean_m": float(np.mean([r["horizontal_baseline_m"] for r in subset])),
            "baseline_min_m": float(min(r["horizontal_baseline_m"] for r in subset)),
            "max_adjacent_so3_deg": float(max(r["max_adjacent_so3_deg"] for r in subset)),
            "collision_count": int(sum(r["furniture_collisions"] for r in subset)),
            "clearance_violation_count": int(sum(r["violations_with_margins"] for r in subset)),
            "coverage_pass_count": int(sum(r["angular_coverage_selected"] >= .97 and
                min(r["polar_up_selected"], r["polar_down_selected"]) >= .97 for r in subset)),
            "surface_complete_count": int(sum(r["surface_coverage_complete"] for r in subset)),
        }
    report = {"schema": "wide_capture_path_qa_v1", "capture_seed": capture_seed, "frames": frames,
              "scene_seeds": [int(x) for x in scene_seeds], "summary": summary, "rows": rows,
              "scope": "capture-side geometry, motion and orientation coverage diagnostics only; no rendered frames or reconstruction claim"}
    (output / "wide_capture_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "wide_capture_seed17")
    ap.add_argument("--scene-seeds", type=int, nargs="*", default=[17, *range(101, 111)])
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--capture-seed", type=int, default=3101)
    a = ap.parse_args()
    print(json.dumps(run(a.output, a.scene_seeds, a.frames, a.capture_seed), indent=2))


if __name__ == "__main__":
    main()
