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


def _find_loop(scene: dict, rng) -> tuple[float, float, float, float, float, float]:
    """Return (cx, cy, rx, ry, rot, phase) for a small elliptical loop clear of walls and furniture."""
    check_angles = np.arange(48) / 48 * 2 * math.pi
    for _ in range(1000):
        rot = float(rng.uniform(0, 2 * math.pi)); phase = float(rng.uniform(0, 2 * math.pi))
        rx = float(rng.uniform(.3, min(.9, scene["width"] * .18)))
        ry = float(rng.uniform(.3, min(1.0, scene["length"] * .18)))
        cx = float(rng.uniform(-.7, .7)); cy = float(rng.uniform(-.8, .8))
        ca = check_angles + phase; lx, ly = rx * np.cos(ca), ry * np.sin(ca)
        xs = cx + lx * math.cos(rot) - ly * math.sin(rot); ys = cy + lx * math.sin(rot) + ly * math.cos(rot)
        if all(camera_is_clear(scene, float(x), float(y)) for x, y in zip(xs, ys)): return cx, cy, rx, ry, rot, phase
    # A tiny orbit around a free grid point is the safe fallback for a very cluttered seed.
    for cx in np.arange(-scene["width"] / 2 + .7, scene["width"] / 2 - .7, .2):
        for cy in np.arange(-scene["length"] / 2 + .7, scene["length"] / 2 - .7, .2):
            if camera_is_clear(scene, float(cx) + .12, float(cy)) and camera_is_clear(scene, float(cx) - .12, float(cy)) \
                    and camera_is_clear(scene, float(cx), float(cy) + .12) and camera_is_clear(scene, float(cx), float(cy) - .12):
                return float(cx), float(cy), .12, .12, 0., 0.
    return 0., 0., .12, .12, 0., 0.


def make_path(scene: dict, frames: int, path_seed: int, index: int = 0) -> list[dict]:
    """One continuous walkthrough: a small collision-free loop with a full 360-degree yaw sweep.

    Yaw advances monotonically through one revolution (direction and start chosen by the seed), so
    every wall direction is faced regardless of room size. Pitch oscillates inside +-MAX_PITCH_RAD
    (positive looks down) at a seed-chosen number of cycles so floor and ceiling are both swept.
    """
    n = max(2, frames); rng = np.random.default_rng(path_seed)
    cx, cy, rx, ry, rot, phase = _find_loop(scene, rng)
    yaw0 = float(rng.uniform(0, 2 * math.pi)); direction = 1. if rng.random() < .5 else -1.
    wobble_phase = float(rng.uniform(0, 2 * math.pi)); wobble = float(rng.uniform(.03, .07))
    pitch_amp = float(rng.uniform(.8, 1.) * MAX_PITCH_RAD); pitch_cycles = int(rng.integers(2, 4))
    pitch_phase = float(rng.uniform(0, 2 * math.pi))
    path = []
    for i in range(n):
        s = i / n; u = phase + 2 * math.pi * s
        lx, ly = rx * math.cos(u), ry * math.sin(u)
        x = cx + lx * math.cos(rot) - ly * math.sin(rot); y = cy + lx * math.sin(rot) + ly * math.cos(rot)
        z = float(1.43 + .18 * math.sin(2 * u + index))
        yaw = yaw0 + direction * (2 * math.pi * s + wobble * math.sin(2 * math.pi * 3 * s + wobble_phase))
        pitch = pitch_amp * math.sin(2 * math.pi * pitch_cycles * s + pitch_phase)
        path.append({"position": [float(x), float(y), z], "yaw": float(yaw), "pitch": float(pitch)})
    return path


def make_paths(scene: dict, frames: int, capture_seed: int | None = None) -> list[list[dict]]:
    """Three independently seeded continuous loops, clear of furniture footprints."""
    return [make_path(scene, frames, s, i) for i, s in enumerate(path_seeds(scene, capture_seed))]


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
    yaw, pitch = pose["yaw"], pose["pitch"]
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.], np.float32)
    forward = np.array([math.sin(yaw)*math.cos(pitch), math.cos(yaw)*math.cos(pitch), -math.sin(pitch)], np.float32)
    up = np.cross(right, forward)
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


def write_coverage(private: Path, scene: dict, paths: list[list[dict]], fov: float, seeds: list[int]) -> dict:
    """Write capture-coverage diagnostics under private_ground_truth (they derive from the private scene)."""
    per_path = [coverage_diagnostics(scene, p, fov) for p in paths]
    report = {"note": "capture diagnostics from the private scene; a capture input check, not reconstruction accuracy",
              "path_seeds": seeds, "paths": per_path, "union": union_coverage(per_path)}
    (private / "coverage_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def generate(output: Path, seed=1, width=320, height=240, frames=12, fov=78., fps=12, capture_seed=None, diagnostics_only=False):
    scene=make_scene(seed); seeds=path_seeds(scene,capture_seed)
    output.mkdir(parents=True,exist_ok=True)
    private=output/"private_ground_truth"; private.mkdir(exist_ok=True)
    (private/"scene.json").write_text(json.dumps(scene,indent=2),encoding="utf-8")
    # Seeds can regenerate the room and paths, so they live with the private truth, not beside the RGB.
    (private/"seed_manifest.json").write_text(json.dumps({"seed":seed,"capture_seed":seed if capture_seed is None else capture_seed,
        "path_seeds":seeds},indent=2),encoding="utf-8")
    (output/"capture_manifest.json").write_text(json.dumps({"width":width,"height":height,"frames_per_path":frames,"paths":PATH_COUNT,
        "fov_degrees":fov,"fps":fps,"renderer":"procedural_cpu_ray_box_v1","rgb_only":True},indent=2),encoding="utf-8")
    paths=make_paths(scene,frames,capture_seed)
    write_coverage(private,scene,paths,fov,seeds)
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
    a=ap.parse_args()
    if min(a.width,a.height,a.frames,a.fps)<=0 or not (10<a.fov<150): ap.error("dimensions, frames, fps must be positive and fov must be between 10 and 150 degrees")
    generate(a.output,a.seed,a.width,a.height,a.frames,a.fov,a.fps,a.capture_seed,a.diagnostics_only)

if __name__=="__main__": main()
