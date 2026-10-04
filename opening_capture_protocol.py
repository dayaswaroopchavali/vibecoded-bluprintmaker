"""Capture protocol that keeps a smooth full-room sweep and explicitly aims at openings.

The scene's opening geometry is used only to choose capture viewpoints and aim
directions. It is never serialized into the public RGB input. QA and private
camera poses stay under the selected output directory's private_ground_truth.

Operator protocol: walk one broad, smooth perimeter loop while continuously
sweeping yaw, pitch, and roll over the full sphere. As each visually noticed
doorway or window comes into view, keep walking and smoothly turn the camera
toward its center for several frames, then ease back into the room sweep. Get
the opening at more than one point along the loop when possible; do not stop,
jerk the camera, or cross furniture. The simulator adapts this by briefly
blending its broad sweep toward each opening from a nearby collision-free
viewpoint. No opening dimensions, labels, or locations belong in reconstruction
inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import capture_parallax
import qa_opening_coverage
import simulator


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def _opening_target(scene: dict, opening: dict) -> np.ndarray:
    """Center of a visible aperture in the renderer's world coordinates."""
    wall, center = opening["wall"], opening["center"]
    z = (opening["z_min"] + opening["z_max"]) / 2
    if wall == 0:  # north, along x
        return np.array([center, scene["length"] / 2, z])
    if wall == 1:  # east, along y
        return np.array([scene["width"] / 2, center, z])
    if wall == 2:  # south, along x
        return np.array([center, -scene["length"] / 2, z])
    return np.array([-scene["width"] / 2, center, z])


def _aim_angles(position: list[float], target: np.ndarray) -> tuple[float, float]:
    delta = target - np.asarray(position, float)
    yaw = math.atan2(delta[0], delta[1])
    pitch = -math.atan2(delta[2], math.hypot(delta[0], delta[1]))
    return yaw, pitch


def _event_centers(scene: dict, path: list[dict]) -> list[int]:
    """Pick one nearby broad-sweep viewpoint for each visible opening."""
    n = len(path)
    indices = np.arange(n)
    positions = np.asarray([p["position"] for p in path])
    selected: list[int] = []
    for opening in scene["openings"]:
        target = _opening_target(scene, opening)
        wall, center = opening["wall"], opening["center"]
        # Stand just inside the room, looking almost squarely at the aperture.
        if wall == 0:
            ideal = np.array([center, scene["length"] / 2 - 2.0, 1.45])
        elif wall == 1:
            ideal = np.array([scene["width"] / 2 - 2.0, center, 1.45])
        elif wall == 2:
            ideal = np.array([center, -scene["length"] / 2 + 2.0, 1.45])
        else:
            ideal = np.array([-scene["width"] / 2 + 2.0, center, 1.45])
        # Prefer the nearest loop sample; discourage both nearby event windows
        # from landing on top of one another while retaining useful viewpoints.
        score = np.linalg.norm(positions[:, :2] - ideal[:2], axis=1)
        for prior in selected:
            d = np.minimum((indices - prior) % n, (prior - indices) % n)
            score += np.maximum(0., 22 - d) * .075
        order = np.argsort(score)
        chosen = next((int(i) for i in order if simulator.camera_is_clear(
            scene, *positions[i, :2], simulator.CAMERA_WALL_MARGIN, simulator.CAMERA_FURNITURE_MARGIN)), int(order[0]))
        selected.append(chosen)
    return selected


def _clear_viewpoint(scene: dict, opening: dict, start: list[float], target_z: float) -> tuple[list[float], list[list[float]]]:
    """Find a direct, collision-free inward approach that frames the aperture."""
    wall, center = opening["wall"], opening["center"]
    candidates = []

    def clear_segment(begin: np.ndarray, end: np.ndarray) -> bool:
        route_pad = .05
        count = max(2, int(np.linalg.norm(end - begin) / .01) + 1)
        for t in np.linspace(0, 1, count):
            point = begin + (end - begin) * t
            if not simulator.camera_is_clear(scene, float(point[0]), float(point[1]),
                    simulator.CAMERA_WALL_MARGIN + route_pad,
                    simulator.CAMERA_FURNITURE_MARGIN + route_pad):
                return False
        return True
    for distance in np.arange(1.45, 2.81, .15):
        for lateral in np.arange(-.75, .751, .15):
            if wall == 0:
                xy = [center + lateral, scene["length"] / 2 - distance]
            elif wall == 1:
                xy = [scene["width"] / 2 - distance, center + lateral]
            elif wall == 2:
                xy = [center + lateral, -scene["length"] / 2 + distance]
            else:
                xy = [-scene["width"] / 2 + distance, center + lateral]
            begin = np.asarray(start[:2])
            end = np.asarray(xy)
            if clear_segment(begin, end):
                route_xy = [end]
            else:
                # Walk parallel to the wall at the current safe clearance,
                # then approach inward at the candidate's lateral offset.
                staging = np.array([end[0], begin[1]]) if wall in (0, 2) else np.array([begin[0], end[1]])
                if not clear_segment(begin, staging) or not clear_segment(staging, end):
                    continue
                route_xy = [staging, end]
            score = abs(distance - 2.0) + .25 * abs(lateral)
            score += .1 * sum(float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
                              for a, b in zip([begin, *route_xy[:-1]], route_xy))
            route = [[float(p[0]), float(p[1]), float(start[2])] for p in route_xy[:-1]]
            route.append([float(end[0]), float(end[1]), float(target_z)])
            candidates.append((score, [float(end[0]), float(end[1]), float(target_z)], route))
    if candidates:
        _, destination, route = min(candidates, key=lambda item: item[0])
        return destination, route
    return list(start), [list(start)]


def _along_route(route: list[list[float]], fraction: float) -> list[float]:
    points = np.asarray(route, float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 1e-12:
        return points[-1].tolist()
    remaining = float(np.clip(fraction, 0, 1)) * total
    for i, length in enumerate(lengths):
        if remaining <= length or i == len(lengths) - 1:
            t = 0. if length <= 1e-12 else remaining / length
            return (points[i] + t * (points[i + 1] - points[i])).tolist()
        remaining -= length
    return points[-1].tolist()


def make_opening_path(scene: dict, frames: int, path_seed: int, index: int = 0,
                      hold_frames: int = 12, max_turn_step_deg: float = 18.) -> list[dict]:
    """Insert smooth turn, hold, and return observations into the wide sweep.

    Every original sweep pose is preserved verbatim and in order. Added poses
    dwell at a collision-free loop viewpoint while quaternion-slerping toward
    the visually identifiable aperture and back. Only this capture-side
    generator reads the private opening geometry.
    """
    base = capture_parallax.make_wide_path(scene, frames, path_seed, index)
    centers = _event_centers(scene, base)
    events = {}
    for opening, center in zip(scene["openings"], centers):
        target = _opening_target(scene, opening)
        view_position = list(base[center]["position"])
        # A lower doorway viewpoint includes its threshold in the vertical
        # field of view; a window uses its center height.
        view_position[2] = 1.05 if opening["type"] == "doorway" else float(target[2])
        view_position, intermediate_route = _clear_viewpoint(scene, opening, view_position, view_position[2])
        route = [list(base[center]["position"]), *intermediate_route]
        yaw, pitch = _aim_angles(view_position, target)
        target_pose = {"position": view_position, "yaw": yaw,
                       "pitch": pitch, "roll": base[center].get("roll", 0.)}
        start_q = _matrix_quaternion(simulator.rotation_matrix(base[center]))
        target_q = _matrix_quaternion(simulator.rotation_matrix(target_pose))
        angle = _quaternion_angle(start_q, target_q)
        travel = float(np.linalg.norm(np.diff(np.asarray(route), axis=0), axis=1).sum())
        # The eased position interpolation peaks at 1.5× its average step.
        # .05 m spacing therefore bounds each adjacent translation below .08 m.
        steps = max(1, int(math.ceil(math.degrees(angle) / max_turn_step_deg)), int(math.ceil(travel / .05)))
        events[center] = (target_q, steps, view_position, route)
    output = []
    for i, pose in enumerate(base):
        output.append(pose)
        if i not in events:
            continue
        target_q, steps, view_position, route = events[i]
        start_q = _matrix_quaternion(simulator.rotation_matrix(pose))
        for k in range(1, steps + 1):
            t = k / steps
            eased = t * t * (3 - 2 * t)
            q = _slerp(start_q, target_q, t)
            pos = _along_route(route, eased)
            output.append(_pose_from_quaternion(pos, q))
        output.extend([dict(output[-1], position=list(view_position)) for _ in range(max(1, hold_frames - 1))])
        for k in range(1, steps + 1):
            t = k / steps
            eased = t * t * (3 - 2 * t)
            q = _slerp(target_q, start_q, t)
            pos = _along_route(list(reversed(route)), eased)
            output.append(_pose_from_quaternion(pos, q))
    return output


def _matrix_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation to normalized [w,x,y,z]."""
    m = np.asarray(matrix, float)
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.) * 2
        q = np.array([.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, .25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          .25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, .25 * s])
    return q / np.linalg.norm(q)


def _quaternion_angle(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * math.acos(float(np.clip(abs(np.dot(a, b)), -1, 1)))


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > .9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta)


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def _pose_from_quaternion(position: list[float], q: np.ndarray) -> dict:
    matrix = _quaternion_matrix(q)
    right, forward, up = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    pitch = math.asin(float(np.clip(-forward[2], -1, 1)))
    yaw = math.atan2(float(forward[0]), float(forward[1]))
    right0 = np.array([math.cos(yaw), -math.sin(yaw), 0.])
    roll = math.atan2(float(np.dot(up, right0)), float(np.dot(right, right0)))
    return {"position": list(position), "yaw": yaw, "pitch": pitch, "roll": roll}


def _aperture_spatial_coverage(scene: dict, path: list[dict], fov: float = 78.,
                               width: int = 36, height: int = 27) -> dict:
    """Private low-resolution aperture coverage: 3 across x 2 high cells/opening."""
    bins = [np.zeros((2, 3), bool) for _ in scene["openings"]]
    hit_pixels = np.zeros(len(bins), dtype=int)
    frames = np.zeros(len(bins), dtype=int)
    for pose in path:
        with np.errstate(all="ignore"):
            _, _, _, kind, pos = simulator.render(scene, pose, width, height, fov, True, True)
        mask = kind.reshape(-1) == simulator.KIND_OPENING
        if not mask.any():
            continue
        hits = pos.reshape(-1, 3)[mask]
        which = simulator._opening_at(scene, hits)
        for oi, opening in enumerate(scene["openings"]):
            p = hits[which == oi]
            if not len(p):
                continue
            frames[oi] += 1
            hit_pixels[oi] += len(p)
            wall = opening["wall"]
            along = p[:, 0] if wall in (0, 2) else p[:, 1]
            xbin = np.clip(((along - (opening["center"] - opening["width"] / 2)) / opening["width"] * 3).astype(int), 0, 2)
            zbin = np.clip(((p[:, 2] - opening["z_min"]) / max(.01, opening["z_max"] - opening["z_min"]) * 2).astype(int), 0, 1)
            bins[oi][zbin, xbin] = True
    return {"openings": [{"type": o["type"], "visible_frames": int(frames[i]), "hit_pixels": int(hit_pixels[i]),
                          "spatial_cells_observed": int(bins[i].sum()), "spatial_cells_total": 6,
                          "spatial_coverage": float(bins[i].mean())}
                         for i, o in enumerate(scene["openings"])]}


def _metrics(scene: dict, path: list[dict], name: str, fov: float, width: int, height: int,
             base_path: list[dict] | None = None) -> dict:
    motion = simulator.camera_motion_summary(scene, path)
    angular = simulator.angular_coverage(path, fov, width, height, count=6000)
    surface = simulator.coverage_diagnostics(scene, path, fov, width=36, height=27)
    aperture = qa_opening_coverage.evaluate(scene, path, width=120, height=90, bins=10)
    positions = np.asarray([p["position"] for p in path], float)
    translation_steps = np.linalg.norm(np.diff(np.vstack([positions, positions[0]]), axis=0), axis=1)
    added_step_max = 0.
    if base_path is not None and len(path) > len(base_path):
        base_indices = []
        cursor = 0
        for base_pose in base_path:
            target_rot = simulator.rotation_matrix(base_pose)
            for j in range(cursor, len(path)):
                if np.allclose(path[j]["position"], base_pose["position"], atol=1e-7) and \
                        np.allclose(simulator.rotation_matrix(path[j]), target_rot, atol=1e-7):
                    base_indices.append(j)
                    cursor = j + 1
                    break
        for left, right in zip(base_indices, base_indices[1:]):
            if right - left > 1:
                local = np.linalg.norm(np.diff(positions[left:right + 1], axis=0), axis=1)
                added_step_max = max(added_step_max, float(local.max()))
    return {"path": name, "frames": len(path),
            "duration_seconds_at_12_fps": len(path) / 12,
            "full_sphere_fraction": angular["full_video"]["fraction"],
            "selected_sphere_fraction": angular["selected_inference_frames"]["fraction"],
            "selected_polar_up_fraction": angular["selected_inference_frames"]["polar_up_fraction"],
            "selected_polar_down_fraction": angular["selected_inference_frames"]["polar_down_fraction"],
            "surface_coverage_complete": surface["complete"],
            "surface_observed": surface["observed"],
            "max_adjacent_so3_deg": motion["max_relative_rotation_deg"],
            "selected_max_adjacent_so3_deg": motion["selected_max_relative_rotation_deg"],
            "horizontal_baseline_m": motion["horizontal_baseline_m"],
            "max_adjacent_translation_step_m": float(translation_steps.max()),
            "max_added_approach_translation_step_m": added_step_max,
            "collision_count": surface["furniture_collisions"],
            "clearance_violations": motion["furniture_or_wall_violations"],
            "minimum_wall_clearance_m": motion["min_wall_clearance_m"],
            "aperture_spatial_coverage": aperture}


def render_saved_videos(output: Path, scene_seeds: list[int], width: int = 160,
                        height: int = 120, fps: int = 12) -> int:
    """Render videos from already QA-frozen private scenes and pose files."""
    import cv2
    written = 0
    for ss in scene_seeds:
        private_scene = output / "private_ground_truth" / f"scene_{ss}"
        scene = json.loads((private_scene / "scene.json").read_text(encoding="utf-8"))
        video_dir = output / "videos" / f"scene_{ss}"
        video_dir.mkdir(parents=True, exist_ok=True)
        for pi in range(simulator.PATH_COUNT):
            path = json.loads((private_scene / f"path_{pi:02d}.json").read_text(encoding="utf-8"))["poses"]
            video_path = video_dir / f"seeded_path_{pi:02d}.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not open the MP4 video writer: {video_path}")
            for pose in path:
                image = simulator.render(scene, pose, width, height, 78.)
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            writer.release()
            written += 1
    return written


def run(output: Path, scene_seeds: list[int], frames: int = 320, capture_seed: int = 3101,
        fov: float = 78., width: int = 160, height: int = 120, render_videos: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private_ground_truth"
    private.mkdir(exist_ok=True)
    rows = []
    for ss in scene_seeds:
        scene = simulator.make_scene(int(ss))
        seeds = simulator.path_seeds(scene, capture_seed)
        private_scene = private / f"scene_{ss}"
        private_scene.mkdir(exist_ok=True)
        (private_scene / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
        (private_scene / "seed_manifest.json").write_text(json.dumps({"scene_seed": int(ss),
            "capture_seed": int(capture_seed), "path_seeds": seeds}, indent=2), encoding="utf-8")
        for pi, path_seed in enumerate(seeds):
            old = capture_parallax.make_wide_path(scene, frames, path_seed, pi)
            new = make_opening_path(scene, frames, path_seed, pi)
            (private_scene / f"baseline_path_{pi:02d}.json").write_text(
                json.dumps({"path_seed": path_seed, "poses": old}, indent=2), encoding="utf-8")
            (private_scene / f"path_{pi:02d}.json").write_text(
                json.dumps({"path_seed": path_seed, "poses": new}, indent=2), encoding="utf-8")
            for protocol, path in (("smooth_sweep", old), ("opening_observation", new)):
                row = _metrics(scene, path, f"{ss}:{pi}:{protocol}", fov, width, height, old)
                row.update(scene_seed=int(ss), path_index=pi, path_seed=path_seed, protocol=protocol)
                rows.append(row)
                if render_videos and protocol == "opening_observation":
                    video_dir = output / "videos" / f"scene_{ss}"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    for fi, pose in enumerate(path):
                        image = simulator.render(scene, pose, width, height, fov)
                        # Use the simulator's video renderer lazily, only when explicitly requested.
                        import cv2
                        video_path = video_dir / f"seeded_path_{pi:02d}.mp4"
                        if fi == 0:
                            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (width, height))
                            if not writer.isOpened():
                                raise RuntimeError("OpenCV could not open the MP4 video writer")
                        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    writer.release()
    protocols = ("smooth_sweep", "opening_observation")
    summary = {}
    for protocol in protocols:
        subset = [r for r in rows if r["protocol"] == protocol]
        all_openings = [o for r in subset for o in r["aperture_spatial_coverage"]]
        covered = [o["spatial_bins_seen_three_frames_fraction"] >= .9 for o in all_openings]
        summary[protocol] = {
            "paths": len(subset),
            "full_sphere_fraction_mean": float(np.mean([r["full_sphere_fraction"] for r in subset])),
            "full_sphere_fraction_min": float(min(r["full_sphere_fraction"] for r in subset)),
            "selected_sphere_fraction_mean": float(np.mean([r["selected_sphere_fraction"] for r in subset])),
            "selected_sphere_fraction_min": float(min(r["selected_sphere_fraction"] for r in subset)),
            "surface_complete_paths": int(sum(r["surface_coverage_complete"] for r in subset)),
            "apertures_observed": int(sum(o["frames_with_any_pixels"] > 0 for o in all_openings)),
            "apertures_total": len(all_openings),
            "apertures_with_90pct_bins_supported_three_frames": int(sum(covered)),
            "aperture_10x10_three_frame_coverage_mean": float(np.mean([o["spatial_bins_seen_three_frames_fraction"] for o in all_openings])),
            "aperture_10x10_three_frame_coverage_min": float(min(o["spatial_bins_seen_three_frames_fraction"] for o in all_openings)),
            "max_adjacent_so3_deg": float(max(r["max_adjacent_so3_deg"] for r in subset)),
            "max_adjacent_translation_step_m": float(max(r["max_adjacent_translation_step_m"] for r in subset)),
            "max_added_approach_translation_step_m": float(max(r["max_added_approach_translation_step_m"] for r in subset)),
            "max_video_duration_seconds": float(max(r["duration_seconds_at_12_fps"] for r in subset)),
            "collision_count": int(sum(r["collision_count"] for r in subset)),
            "clearance_violation_count": int(sum(r["clearance_violations"] for r in subset)),
        }
    report = {"schema": "opening_capture_protocol_qa_v1", "capture_seed": capture_seed,
              "scene_seeds": scene_seeds, "frames": frames, "fov_degrees": fov,
              "resolution_for_qa": [width, height], "summary": summary, "rows": rows,
              "human_followable_protocol": __doc__,
              "scope": "capture-side motion and coverage diagnostics only; scene opening metadata is private and never a reconstruction input"}
    (output / "opening_capture_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "opening_capture_seeded")
    ap.add_argument("--scene-seeds", type=int, nargs="*", default=[17, *range(101, 110)])
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--capture-seed", type=int, default=3101)
    ap.add_argument("--render-videos", action="store_true", help="render three seeded opening-aware MP4 paths per room")
    ap.add_argument("--videos-only", action="store_true", help="render from existing QA-frozen private scenes and paths")
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    a = ap.parse_args()
    if a.videos_only:
        count = render_saved_videos(a.output, a.scene_seeds, a.width, a.height)
        print(json.dumps({"videos_written": count, "output": str(a.output / "videos")}))
        return
    report = run(a.output, a.scene_seeds, a.frames, a.capture_seed, width=a.width, height=a.height,
                 render_videos=a.render_videos)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
