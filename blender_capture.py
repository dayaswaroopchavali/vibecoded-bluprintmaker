"""Render the bounded procedural room in headless Blender's EEVEE renderer.

No asset files are downloaded. Room geometry and camera poses (360-degree yaw sweep, bounded pitch sweep)
derive from simulator.py, which also writes the coverage diagnostics. This is
still a simple synthetic scene, not a photorealistic or sensor-faithful renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")


def path_seeds(scene: dict, capture_seed: int | None = None) -> list[int]:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import simulator
    return simulator.path_seeds(scene, capture_seed)


def camera_poses(scene: dict, frames: int, capture_seed: int | None = None) -> list[list[dict]]:
    """Same collision-checked loops as simulator, each with a full 360-degree yaw sweep and bounded pitch sweep."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import simulator  # numpy-only at import time (cv2 is deferred), so it also loads inside Blender
    return simulator.make_paths(scene, frames, capture_seed)


def write_coverage(scene: dict, output: Path, frames: int, fov: float, capture_seed: int) -> dict:
    """Coverage diagnostics for the poses Blender renders (private: derived from the private scene)."""
    import simulator
    private = output / "private_ground_truth"; private.mkdir(parents=True, exist_ok=True)
    return simulator.write_coverage(private, scene, camera_poses(scene, frames, capture_seed), fov, path_seeds(scene, capture_seed))

APPEARANCES = ("original", "textured", "realistic")


def poses_sha256(poses: list[dict]) -> str:
    """Canonical hash of a pose list (independent of file whitespace), recorded in appearance manifests."""
    return hashlib.sha256(json.dumps(poses, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_poses_json(path: Path) -> dict:
    """Read a capture's private_ground_truth/path_0N.json; every pose needs position, yaw, pitch (roll optional)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = data["poses"]
    if not poses:
        raise ValueError("poses json has no poses")
    for i, p in enumerate(poses):
        if len(p["position"]) != 3 or any(k not in p for k in ("yaw", "pitch")):
            raise ValueError(f"pose {i} needs position[3], yaw, pitch")
    return data


def texture_params(texture_seed: int, scene: dict) -> dict:
    """All TEXTURED-appearance material parameters, drawn from a private RNG that depends ONLY on texture_seed.

    The RNG is a SeedSequence([texture_seed, tag]) stream: independent of the scene seed and of the capture/path
    seeds (changing --texture-seed changes materials only; changing scene/capture seeds never changes the draw).
    scene is read only for base albedo colours and furniture *names* (painted walls keep the scene's paint colour);
    no room dimension, opening or furniture position is consulted, so texture features are never placed from geometry.
    Texture-space units are metres of object-local coordinates.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(texture_seed), 0x7E57A9]))
    u = lambda a, b: float(rng.uniform(a, b))
    off = lambda: [float(v) for v in rng.uniform(0, 200, 3)]
    walls = []
    for c in scene["wall_colors"]:
        walls.append({"base_color": [float(v) for v in c], "roughness": u(.86, .95),
                      "plaster": {"scale": u(55, 85), "amp": u(.070, .100), "offset": off()},
                      "blotch": {"scale": u(.9, 1.6), "amp": u(.080, .110), "offset": off()},
                      "brush": {"scale": [u(18, 26), u(18, 26), u(1.6, 2.6)], "amp": u(.050, .070), "offset": off()},
                      "bump_strength": u(.015, .035)})
    wood = [u(.17, .23), u(.10, .14), u(.05, .08)]
    floor = {"plank_width_m": u(.11, .16), "plank_length_m": u(1.1, 1.7), "stagger": .5, "gap_m": u(.0025, .004),
             "wood_tone_a": wood, "wood_tone_b": [c * u(.78, .90) for c in wood],
             "gap_color": [c * .45 for c in wood], "roughness": u(.52, .68),
             "grain": {"scale": [u(1.5, 3), u(45, 70), u(2, 4)], "amp": u(.05, .08), "offset": off()},
             "plank_tone_bias": u(.35, .65), "bump_strength": u(.03, .06)}
    ceiling = {"base_color": [float(v) for v in scene["ceiling_color"]], "roughness": .95,
               "plaster": {"scale": u(40, 60), "amp": u(.035, .060), "offset": off()},
               "blotch": {"scale": u(.6, 1.1), "amp": u(.035, .060), "offset": off()}}
    furniture = {}
    for i, f in enumerate(scene["furniture"]):
        if f["name"] == "sofa":
            furniture[f["name"]] = {"kind": "fabric", "base_color": [float(v) * .5 for v in f["color"]], "albedo_scale_vs_scene": .5, "roughness": .92,
                                    "weave": {"scale": u(95, 150), "amp": u(.05, .08), "offset": off()},
                                    "mottle": {"scale": u(5, 9), "amp": u(.03, .06), "offset": off()},
                                    "bump_strength": u(.10, .20)}
        else:
            w = [u(.14, .22), u(.08, .12), u(.04, .07)]
            furniture[f["name"]] = {"kind": "wood", "base_color": w, "roughness": u(.45, .6),
                                    "bands": {"scale": u(4, 8), "distortion": u(3, 6), "amp": u(.07, .12), "offset": off()},
                                    "fine": {"scale": [u(1.5, 3), u(50, 80), u(2, 4)], "amp": u(.03, .05), "offset": off()},
                                    "bump_strength": u(.03, .06)}
    return {"texture_seed": int(texture_seed), "rng": "numpy SeedSequence([texture_seed, 0x7E57A9])",
            "walls": walls, "floor": floor, "ceiling": ceiling, "furniture": furniture,
            "coverage": {"wall_area_with_structured_pattern_fraction": 0.0,
                         "wall_area_with_plaster_blotch_brush_noise_fraction": 1.0,
                         "wall_noise_amp_parameter_sum_max": round(max(w["plaster"]["amp"] + w["blotch"]["amp"] + w["brush"]["amp"] for w in walls), 4),
                         "note": "No wallpaper/mural/frames: every wall is plain paint; plaster/blotch/brush noise modulates luminance by only a few percent rms (amp parameters are map-range half-widths of a noise that rarely leaves its centre; measured wall luminance std is checked in tests); patterned surfaces are floor planks and furniture only."}}


def _textured_materials(bpy, params: dict, scene_data: dict) -> dict:
    """Build the procedural node materials of the textured appearance (local Blender nodes only, no images)."""
    def new(name, color, rough):
        m = bpy.data.materials.new(name); m.diffuse_color = (*[min(max(c, 0), 1) for c in color], 1); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear(); out = nt.nodes.new("ShaderNodeOutputMaterial")
        bs = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.inputs["Roughness"].default_value = rough
        nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"]); return m, nt, bs

    def vec(nt, scale, offset):
        c = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = offset; mp.inputs["Scale"].default_value = tuple(scale) if isinstance(scale, (list, tuple)) else (scale,) * 3
        nt.links.new(c.outputs["Object"], mp.inputs["Vector"]); return mp.outputs["Vector"]

    def noise(nt, cfg, detail=4., rough=.55):
        n = nt.nodes.new("ShaderNodeTexNoise"); n.inputs["Scale"].default_value = 1.
        n.inputs["Detail"].default_value = detail; n.inputs["Roughness"].default_value = rough
        nt.links.new(vec(nt, cfg["scale"], cfg["offset"]), n.inputs["Vector"]); return n.outputs["Fac"]

    def centred(nt, fac, amp):
        r = nt.nodes.new("ShaderNodeMapRange"); r.inputs["From Min"].default_value = .2; r.inputs["From Max"].default_value = .8
        r.inputs["To Min"].default_value = -amp; r.inputs["To Max"].default_value = amp
        nt.links.new(fac, r.inputs["Value"]); return r.outputs["Result"]

    def tone(nt, terms):  # 1 + sum(centred noise terms)
        acc = None
        for fac, amp in terms:
            t = centred(nt, fac, amp)
            if acc is None: acc = t; continue
            a = nt.nodes.new("ShaderNodeMath"); a.operation = "ADD"; nt.links.new(acc, a.inputs[0]); nt.links.new(t, a.inputs[1]); acc = a.outputs[0]
        one = nt.nodes.new("ShaderNodeMath"); one.operation = "ADD"; one.inputs[1].default_value = 1.; nt.links.new(acc, one.inputs[0])
        return one.outputs[0]

    def tint(nt, color, tone_out):
        if isinstance(color, (list, tuple)):
            c = nt.nodes.new("ShaderNodeRGB"); c.outputs[0].default_value = (*color, 1); color = c.outputs[0]
        s = nt.nodes.new("ShaderNodeVectorMath"); s.operation = "SCALE"
        nt.links.new(color, s.inputs[0]); nt.links.new(tone_out, s.inputs[3]); return s.outputs[0]

    def finish(nt, bs, color_out, bump_src, strength, dist=.02):
        nt.links.new(color_out, bs.inputs["Base Color"])
        b = nt.nodes.new("ShaderNodeBump"); b.inputs["Strength"].default_value = strength; b.inputs["Distance"].default_value = dist
        nt.links.new(bump_src, b.inputs["Height"]); nt.links.new(b.outputs["Normal"], bs.inputs["Normal"])

    mats = {"walls": [], "furniture": {}}
    for i, p in enumerate(params["walls"]):
        m, nt, bs = new(f"wall {i+1} painted", p["base_color"], p["roughness"])
        pl = noise(nt, p["plaster"], 5, .6); bl = noise(nt, p["blotch"], 2, .5); br = noise(nt, p["brush"], 2, .5)
        finish(nt, bs, tint(nt, p["base_color"], tone(nt, [(pl, p["plaster"]["amp"]), (bl, p["blotch"]["amp"]), (br, p["brush"]["amp"])])), pl, p["bump_strength"], .01)
        mats["walls"].append(m)
    f = params["floor"]
    m, nt, bs = new("wood plank floor", f["wood_tone_a"], f["roughness"])
    brick = nt.nodes.new("ShaderNodeTexBrick"); brick.offset = f["stagger"]; brick.offset_frequency = 2; brick.squash = 1.; brick.squash_frequency = 2
    nt.links.new(vec(nt, 1., [0, 0, 0]), brick.inputs["Vector"])
    for k, v in (("Color1", f["wood_tone_a"]), ("Color2", f["wood_tone_b"]), ("Mortar", f["gap_color"])): brick.inputs[k].default_value = (*v, 1)
    for k, v in (("Scale", 1.), ("Mortar Size", f["gap_m"]), ("Mortar Smooth", .1), ("Bias", f["plank_tone_bias"] - .5),
                 ("Brick Width", f["plank_length_m"]), ("Row Height", f["plank_width_m"])): brick.inputs[k].default_value = v
    gr = noise(nt, f["grain"], 4, .6)
    finish(nt, bs, tint(nt, brick.outputs["Color"], tone(nt, [(gr, f["grain"]["amp"])])), gr, f["bump_strength"], .01)
    mats["floor"] = m
    c = params["ceiling"]
    m, nt, bs = new("ceiling painted", c["base_color"], c["roughness"])
    pl = noise(nt, c["plaster"], 5, .6); bl = noise(nt, c["blotch"], 2, .5)
    finish(nt, bs, tint(nt, c["base_color"], tone(nt, [(pl, c["plaster"]["amp"]), (bl, c["blotch"]["amp"])])), pl, .02, .01)
    mats["ceiling"] = m
    for name, p in params["furniture"].items():
        m, nt, bs = new(name + (" fabric" if p["kind"] == "fabric" else " wood"), p["base_color"], p["roughness"])
        if p["kind"] == "fabric":
            wv = noise(nt, p["weave"], 2, .5); mo = noise(nt, p["mottle"], 3, .5)
            terms = [(wv, p["weave"]["amp"]), (mo, p["mottle"]["amp"])]; bump_src = wv
        else:
            w = nt.nodes.new("ShaderNodeTexWave"); w.wave_type = "BANDS"; w.bands_direction = "X"
            w.inputs["Scale"].default_value = 1.; w.inputs["Distortion"].default_value = p["bands"]["distortion"]; w.inputs["Detail"].default_value = 2.
            nt.links.new(vec(nt, [p["bands"]["scale"], p["bands"]["scale"] * .25, p["bands"]["scale"] * .25], p["bands"]["offset"]), w.inputs["Vector"])
            fn = noise(nt, p["fine"], 4, .6); terms = [(w.outputs["Fac"], p["bands"]["amp"]), (fn, p["fine"]["amp"])]; bump_src = fn
        finish(nt, bs, tint(nt, p["base_color"], tone(nt, terms)), bump_src, p["bump_strength"], .01)
        mats["furniture"][name] = m
    return mats


# ---------------------------------------------------------------------------------------------------------------------
# REALISTIC appearance: furnished-room look (paint, stains, decor, trim, rug). Everything is real mesh/material.
# Parameters depend ONLY on texture_seed (realistic_params). Where decor goes is decided by realistic_layout from the
# seed-drawn slots + the wall's own local (u, z) coordinates; the scene is consulted only to reject slots that would leave
# the wall or overlap an opening (and to size skirting/trim/rug), never to draw random numbers.
# ---------------------------------------------------------------------------------------------------------------------
REALISTIC_SCHEMA = "realistic_decor_v1"
_ART_STYLES = ("blocks", "stripes", "gradient", "abstract")


def _hsv(h: float, s: float, v: float) -> list[float]:
    import colorsys
    return [float(c) for c in colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.), 1.), min(max(v, 0.), 1.))]


def realistic_params(texture_seed: int, scene: dict) -> dict:
    """All REALISTIC-appearance parameters: a function of texture_seed only (scene is read for furniture *names* only,
    through texture_params, exactly like the textured appearance). Decor slots are fixed-length draws (96 uniforms per
    slot) so the random stream never depends on which slots are later accepted or rejected."""
    rng = np.random.default_rng(np.random.SeedSequence([int(texture_seed), 0x2EA115]))
    u = lambda a, b: float(rng.uniform(a, b))
    off = lambda: [float(v) for v in rng.uniform(0, 200, 3)]
    h0 = u(0, 1); s0 = u(.07, .26); v0 = u(.52, .80); accent = int(rng.integers(0, 4))
    walls = []
    for i in range(4):
        if i == accent: col = _hsv(h0 + u(.08, .5), u(.18, .38), u(.40, .62))
        else: col = _hsv(h0 + u(-.02, .02), s0 * u(.85, 1.15), v0 * u(.94, 1.06))
        lo = u(.70, .85)
        walls.append({"base_color": col, "accent": i == accent, "roughness": u(.86, .95),
                      "lowfreq": {"scale": u(.10, .28), "amp": u(.25, .42), "offset": off()},
                      "vertical": {"bottom": lo, "top": u(1.0, 1.07), "z_span_m": u(2.4, 3.2)},
                      "plaster": {"scale": u(55, 85), "amp": u(.04, .06), "offset": off()},
                      "stain": {"scale": [u(.5, .9), u(.5, .9), u(.14, .30)], "offset": off(), "lo": u(.50, .56), "width": u(.06, .12),
                                "strength": u(.35, .65), "tint": [u(.70, .85), u(.58, .72), u(.35, .50)]},
                      "bump_strength": u(.012, .03)})
    tv = u(.72, .90); trim_color = [tv, tv, tv * u(.94, 1.0)]
    wood = [u(.18, .32), u(.10, .19), u(.05, .11)]
    floor = {"plank_width_m": u(.11, .17), "plank_length_m": u(1.0, 1.7), "stagger": .5, "gap_m": u(.0025, .004),
             "wood_tone_a": wood, "wood_tone_b": [c * u(.55, .85) for c in wood], "gap_color": [c * .35 for c in wood],
             "roughness": u(.45, .62), "bias": u(-.3, .3),
             "grain": {"scale": [u(1.5, 3), u(45, 70), u(2, 4)], "amp": u(.08, .13), "offset": off()},
             "lowfreq": {"scale": u(.2, .5), "amp": u(.06, .10), "offset": off()}, "bump_strength": u(.03, .06)}
    cv = u(.74, .90)
    ceiling = {"base_color": [cv, cv, cv * u(.95, 1.0)], "roughness": .95,
               "plaster": {"scale": u(40, 60), "amp": u(.03, .05), "offset": off()},
               "lowfreq": {"scale": u(.2, .5), "amp": u(.12, .20), "offset": off()}}
    rh = u(0, 1)
    rug = {"width_m": u(1.6, 2.4), "length_m": u(1.1, 1.7), "offset_m": [u(-.3, .3), u(-.3, .3)], "thickness_m": .008,
           "border_m": u(.07, .12), "stripe_m": u(.025, .045),
           "colors": {"border": _hsv(rh, u(.35, .6), u(.25, .45)), "stripe": _hsv(rh + .5, u(.2, .4), u(.6, .8)),
                      "field": _hsv(rh + u(-.05, .05), u(.15, .4), u(.55, .8)), "medallion": _hsv(rh + u(.1, .4), u(.3, .6), u(.35, .6))}}
    fixtures = {"skirting_height_m": u(.09, .14), "skirting_depth_m": u(.016, .026), "trim_width_m": u(.05, .08),
                "trim_depth_m": u(.015, .025), "cornice_size_m": u(.05, .09), "trim_color": trim_color,
                "ceiling_light_diameter_m": u(.30, .50), "ceiling_light_thickness_m": .04}
    frame_cols = [[.03, .03, .03], [.12, .07, .035], trim_color, [.45, .33, .12]]
    slots = []
    kinds_big = ["frame", "poster", "shelf", "clock", "frame"]
    for wi in range(4):
        order = [kinds_big[j] for j in rng.permutation(len(kinds_big))]
        specs = [(k, (j + .15 + .7 * float(rng.random())) / len(order)) for j, k in enumerate(order)]
        specs += [("switch", u(.1, .9)), ("socket", u(.1, .9)), ("socket", u(.1, .9))]
        for kind, uf in specs:
            r = iter(rng.random(96)); U = lambda a, b: a + (b - a) * float(next(r))
            art = {"style": _ART_STYLES[int(U(0, 4)) % 4], "scale": U(2, 6), "offset": [U(0, 50) for _ in range(3)],
                   "colors": [_hsv(U(0, 1), U(.35, .9), U(.25, .9)) for _ in range(4)], "circle_r": U(.12, .32)}
            s = {"wall": wi, "kind": kind, "u_frac": uf, "art": art}
            if kind == "frame":
                w = U(.35, .95); s.update(width=w, height=min(w * U(.7, 1.4), 1.1), z_center=U(1.3, 1.75), bar=U(.025, .045), depth=U(.025, .04),
                                         frame_color=frame_cols[int(U(0, 4)) % 4], mat_color=[U(.82, .93)] * 3)
            elif kind == "poster":
                w = U(.5, .9); s.update(width=w, height=w * U(1.2, 1.5), z_center=U(1.2, 1.6), border=U(.015, .03), border_color=[U(.85, .95)] * 3)
            elif kind == "shelf":
                sd = U(.18, .26); w = U(.6, 1.2)
                s.update(width=w, height=.12 + .03 + .28, z_center=U(1.25, 1.9), depth=sd, board_color=frame_cols[1 + int(U(0, 3)) % 3],
                         items=[{"w": U(.05, .2), "h": U(.08, .28), "d": U(.05, .15), "gap": U(0, .05), "color": _hsv(U(0, 1), U(.2, .8), U(.25, .8))} for _ in range(5)],
                         n_items=3 + int(U(0, 3)) % 3)
            elif kind == "clock":
                d = U(.25, .4); s.update(width=d, height=d, z_center=U(1.9, 2.2), rim_color=frame_cols[int(U(0, 4)) % 4], face_color=[U(.85, .95)] * 3,
                                         hand_angles_deg=[U(0, 360), U(0, 360)])
            elif kind == "switch": s.update(width=.085, height=.085, z_center=U(1.1, 1.25), color=[.88, .88, .86])
            else: s.update(width=.085, height=.085, z_center=U(.22, .40), color=[.88, .88, .86])
            slots.append(s)
    return {"texture_seed": int(texture_seed), "rng": "numpy SeedSequence([texture_seed, 0x2EA115]); decor slots use fixed 96-uniform draws",
            "walls": walls, "floor": floor, "ceiling": ceiling, "rug": rug, "fixtures": fixtures, "decor_slots": slots,
            "furniture": texture_params(texture_seed, scene)["furniture"]}


def _wall_frame(scene: dict, wi: int) -> dict:
    w, l = scene["width"], scene["length"]
    return {"span": [w, l, w, l][wi], "plane": [l / 2, w / 2, -l / 2, -w / 2][wi], "axis": "y" if wi in (0, 2) else "x", "sign": [-1., -1., 1., 1.][wi]}


def _wbox(scene: dict, wi: int, ua: float, ub: float, d0: float, d1: float, z0: float, z1: float) -> list:
    """World AABB [lo, hi] of a box given in the wall's own coordinates: u along the wall, d inward from the wall plane, z up."""
    f = _wall_frame(scene, wi); n0, n1 = sorted((f["plane"] + f["sign"] * d0, f["plane"] + f["sign"] * d1))
    lo, hi = ([ua, n0, z0], [ub, n1, z1]) if f["axis"] == "y" else ([n0, ua, z0], [n1, ub, z1])
    return [[float(v) for v in lo], [float(v) for v in hi]]


def _part(shape, aabb, mat, name, axis=None, roll_deg=0.):
    return {"shape": shape, "aabb": aabb, "center": [(a + b) / 2 for a, b in zip(*aabb)], "size": [b - a for a, b in zip(*aabb)],
            "mat": mat, "name": name, "axis": axis, "roll_deg": float(roll_deg)}


def _solid(color, rough=.7, fabric=0.): return {"type": "solid", "color": [float(c) for c in color], "rough": rough, "fabric": fabric}


def _rects_overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def realistic_layout(params: dict, scene: dict) -> dict:
    """Deterministic scene construction: slots -> accepted decor + skirting/trim/cornice/ceiling light/rug as box/disc parts.
    Slots outside the wall, within 0.25 m of a corner, too high/low, or overlapping an opening (+trim +0.10 m) or an
    earlier slot are DROPPED (no re-draw, so the seed-only parameters never depend on openings)."""
    w, l, h = scene["width"], scene["length"], scene["height"]; fx = params["fixtures"]
    sk_h, sk_d, tw, td, cs = fx["skirting_height_m"], fx["skirting_depth_m"], fx["trim_width_m"], fx["trim_depth_m"], fx["cornice_size_m"]
    trim_m = _solid(fx["trim_color"], .55); items = []
    def add(kind, wall, parts, **extra):
        lo = [min(p["aabb"][0][k] for p in parts) for k in range(3)]; hi = [max(p["aabb"][1][k] for p in parts) for k in range(3)]
        items.append({"id": f"decor_{len(items):03d}", "kind": kind, "wall": wall, "aabb": [lo, hi], "parts": parts, **extra})
    margin = .10 + tw
    occupied = {wi: [] for wi in range(4)}
    for wi in range(4):
        for o in scene["openings"]:
            if o["wall"] == wi:
                occupied[wi].append((o["center"] - o["width"] / 2 - margin, o["center"] + o["width"] / 2 + margin, o["z_min"] - margin, o["z_max"] + margin))
    # trim, skirting, cornice (scene construction from the openings / walls themselves)
    for wi in range(4):
        half = _wall_frame(scene, wi)["span"] / 2
        for o in (o for o in scene["openings"] if o["wall"] == wi):
            u0, u1 = max(-half, o["center"] - o["width"] / 2), min(half, o["center"] + o["width"] / 2)
            z0, z1 = max(0., o["z_min"]), min(h, o["z_max"]); a, b = max(-half, u0 - tw), min(half, u1 + tw); sill = o["type"] == "window"
            zl = z0 - tw if sill else z0
            ps = [_part("box", _wbox(scene, wi, a, u0, 0, td, zl, z1 + tw), trim_m, "trim left"), _part("box", _wbox(scene, wi, u1, b, 0, td, zl, z1 + tw), trim_m, "trim right"),
                  _part("box", _wbox(scene, wi, u0, u1, 0, td, z1, z1 + tw), trim_m, "trim top")]
            if sill: ps.append(_part("box", _wbox(scene, wi, u0, u1, 0, td * 1.6, z0 - tw, z0), trim_m, "sill"))
            add("opening_trim", wi, ps, opening_type=o["type"])
        span = (-half, half) if wi in (0, 2) else (-half + sk_d, half - sk_d)
        cuts = sorted((o["center"] - o["width"] / 2 - tw, o["center"] + o["width"] / 2 + tw) for o in scene["openings"] if o["wall"] == wi and o["z_min"] < sk_h)
        segs, cur = [], span[0]
        for a, b in cuts:
            if a > cur: segs.append((cur, min(a, span[1])))
            cur = max(cur, b)
        if cur < span[1]: segs.append((cur, span[1]))
        ps = [_part("box", _wbox(scene, wi, a, b, 0, sk_d, 0, sk_h), trim_m, "skirting") for a, b in segs if b - a > .05]
        if ps: add("skirting", wi, ps)
        cspan = (-half, half) if wi in (0, 2) else (-half + cs, half - cs)
        add("cornice", wi, [_part("box", _wbox(scene, wi, cspan[0], cspan[1], 0, cs, h - cs, h), _solid(params["ceiling"]["base_color"], .9), "cornice")])
    d = fx["ceiling_light_diameter_m"]; t = fx["ceiling_light_thickness_m"]
    add("ceiling_light", None, [_part("disc", [[-d * .6, -d * .6, h - t * .5], [d * .6, d * .6, h]], _solid(trim_m["color"], .6), "ceiling rose", "z"),
                                _part("disc", [[-d / 2, -d / 2, h - t - .01], [d / 2, d / 2, h - t * .5]], {"type": "emit", "color": [1., .96, .88], "strength": 2.}, "ceiling light", "z")])
    rg = params["rug"]; rw, rl = min(rg["width_m"], w - 1.0), min(rg["length_m"], l - 1.0); ox, oy = rg["offset_m"]; th = rg["thickness_m"]; c = rg["colors"]; b1 = rg["border_m"]; b2 = b1 + rg["stripe_m"]
    if rw > 3 * b2 and rl > 3 * b2:
        box = lambda hx, hy, z: [[ox - hx, oy - hy, 0.], [ox + hx, oy + hy, z]]
        add("rug", None, [_part("box", box(rw / 2, rl / 2, th), _solid(c["border"], .95, .1), "rug border"), _part("box", box(rw / 2 - b1, rl / 2 - b1, th * 1.25), _solid(c["stripe"], .95, .1), "rug stripe"),
                          _part("box", box(rw / 2 - b2, rl / 2 - b2, th * 1.5), _solid(c["field"], .95, .12), "rug field"), _part("box", box(rw * .17, rl * .17, th * 1.75), _solid(c["medallion"], .95, .12), "rug medallion")])
    # wall decor from seed slots
    caps = {"frame": 10, "poster": 4, "shelf": 3, "clock": 1, "switch": 2, "socket": 3}; taken: dict = {}  # per-house caps, applied in slot order
    for s in params["decor_slots"]:
        if taken.get(s["kind"], 0) >= caps[s["kind"]]: continue
        wi = s["wall"]; half = _wall_frame(scene, wi)["span"] / 2; wd, ht, zc = s["width"], s["height"], s["z_center"]
        uc = -half + s["u_frac"] * 2 * half; rect = (uc - wd / 2, uc + wd / 2, zc - ht / 2, zc + ht / 2)
        if s["kind"] == "shelf": rect = (rect[0], rect[1], zc - .15, zc + .30)
        if rect[0] < -half + .25 or rect[1] > half - .25 or rect[2] < sk_h + .03 or rect[3] > h - cs - .10: continue
        grown = (rect[0] - .08, rect[1] + .08, rect[2] - .08, rect[3] + .08)
        if any(_rects_overlap(grown, o) for o in occupied[wi]): continue
        occupied[wi].append(rect); taken[s["kind"]] = taken.get(s["kind"], 0) + 1
        u0, u1, z0, z1 = uc - wd / 2, uc + wd / 2, zc - ht / 2, zc + ht / 2; B = lambda *a: _wbox(scene, wi, *a); art = s["art"]; ps = []; uax = "x" if _wall_frame(scene, wi)["axis"] == "y" else "y"
        if s["kind"] == "frame":
            bw, fd = s["bar"], s["depth"]; fm = _solid(s["frame_color"], .5)
            ps = [_part("box", B(u0, u0 + bw, 0, fd, z0, z1), fm, "frame bar"), _part("box", B(u1 - bw, u1, 0, fd, z0, z1), fm, "frame bar"),
                  _part("box", B(u0 + bw, u1 - bw, 0, fd, z1 - bw, z1), fm, "frame bar"), _part("box", B(u0 + bw, u1 - bw, 0, fd, z0, z0 + bw), fm, "frame bar"),
                  _part("box", B(u0 + bw, u1 - bw, 0, .018, z0 + bw, z1 - bw), _solid(s["mat_color"], .9), "frame mat")]
            m = bw + .045
            if wd - 2 * m > .08 and ht - 2 * m > .08: ps.append(_part("box", B(u0 + m, u1 - m, 0, .022, z0 + m, z1 - m), {"type": "art", **art, "rough": .4, "uaxis": uax}, "frame art"))
        elif s["kind"] == "poster":
            bd = s["border"]; ps = [_part("box", B(u0, u1, 0, .004, z0, z1), _solid(s["border_color"], .8), "poster border"),
                                    _part("box", B(u0 + bd, u1 - bd, 0, .0065, z0 + bd, z1 - bd), {"type": "art", **art, "rough": .55, "uaxis": uax}, "poster art")]
        elif s["kind"] == "shelf":
            sd = s["depth"]; bm = _solid(s["board_color"], .5); zb = zc
            ps = [_part("box", B(u0, u1, 0, sd, zb - .015, zb + .015), bm, "shelf board")] + [_part("box", B(uu - .01, uu + .01, 0, sd * .8, zb - .135, zb - .015), bm, "shelf bracket") for uu in (u0 + .08, u1 - .08)]
            cur = u0 + .04
            for it in s["items"][:s["n_items"]]:
                if cur + it["w"] > u1 - .04: break
                ps.append(_part("box", B(cur, cur + it["w"], .01, .01 + min(it["d"], sd - .03), zb + .015, zb + .015 + it["h"]), _solid(it["color"], .7), "shelf box")); cur += it["w"] + it["gap"]
        elif s["kind"] == "clock":
            ps = [_part("disc", B(uc - wd / 2, uc + wd / 2, 0, .04, zc - wd / 2, zc + wd / 2), _solid(s["rim_color"], .4), "clock rim", _wall_frame(scene, wi)["axis"]),
                  _part("disc", B(uc - wd * .45, uc + wd * .45, 0, .045, zc - wd * .45, zc + wd * .45), _solid(s["face_color"], .6), "clock face", _wall_frame(scene, wi)["axis"])]
            for nm, ln, ang in (("hour", .30, s["hand_angles_deg"][0]), ("minute", .42, s["hand_angles_deg"][1])):
                ps.append(_part("box", B(uc - ln * wd / 2, uc + ln * wd / 2, .045, .05, zc - .006, zc + .006), _solid([.04, .04, .04], .5), f"clock {nm} hand", roll_deg=ang))
                ps[-1]["aabb"] = B(uc - wd * .45, uc + wd * .45, .045, .05, zc - wd * .45, zc + wd * .45)  # conservative: rotated hand stays inside the face
        elif s["kind"] == "switch":
            ps = [_part("box", B(u0, u1, 0, .012, z0, z1), _solid(s["color"], .45), "switch plate"), _part("box", B(uc - .015, uc + .015, .012, .02, zc - .025, zc + .025), _solid([.82, .82, .8], .4), "switch rocker")]
        else:
            ps = [_part("box", B(u0, u1, 0, .012, z0, z1), _solid(s["color"], .45), "socket plate")] + [_part("box", B(uc + dx - .006, uc + dx + .006, .012, .0135, zc - .012, zc + .012), _solid([.03, .03, .03], .6), "socket hole") for dx in (-.014, .014)]
        add(s["kind"], wi, ps, u_center=float(uc), z_center=float(zc), width=float(wd), height=float(ht))
    shell = {"width": w, "length": l, "height": h, "floor_plane_z": 0.0, "ceiling_plane_z": h,
             "wall_planes": [{"wall": wi, "axis": _wall_frame(scene, wi)["axis"], "coordinate": _wall_frame(scene, wi)["plane"],
                              "inward_normal": ([0., _wall_frame(scene, wi)["sign"], 0.] if wi in (0, 2) else [_wall_frame(scene, wi)["sign"], 0., 0.])} for wi in range(4)],
             "unchanged_by_decor": True, "note": "floor/ceiling slabs, wall cell meshes and opening apertures are built by the same code for every appearance; decor only adds separate objects on the inward side of the planes"}
    return {"schema": REALISTIC_SCHEMA, "room_shell": shell, "items": items}


def decor_manifest(params: dict, layout: dict) -> dict:
    """Private per-house manifest (private_ground_truth/decor_manifest.json): evaluators may mask these AABBs."""
    items = [{"id": it["id"], "kind": it["kind"], "wall": it["wall"], "aabb_min": it["aabb"][0], "aabb_max": it["aabb"][1],
              **{k: it[k] for k in ("u_center", "z_center", "width", "height", "opening_type") if k in it},
              "parts": [{"name": p["name"], "shape": p["shape"], "aabb_min": p["aabb"][0], "aabb_max": p["aabb"][1]} for p in it["parts"]]} for it in layout["items"]]
    counts: dict = {}
    for it in items: counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    return {"schema": REALISTIC_SCHEMA, "texture_seed": params["texture_seed"], "room_shell": layout["room_shell"], "counts": counts, "items": items,
            "mask_note": "Depth/geometry ground truth of the room shell (z=0, z=H, wall planes) is unchanged; these items add real geometry on the inward side (max protrusion listed by aabb). Mask them before comparing shell planes."}


def _realistic_materials(bpy, params: dict) -> dict:
    """Procedural node materials for the realistic appearance (no images). Returns shell materials and a cached part-material factory."""
    def new(name, color, rough):
        m = bpy.data.materials.new(name); m.diffuse_color = (*[min(max(c, 0), 1) for c in color], 1); m.use_nodes = True
        nt = m.node_tree; nt.nodes.clear(); out = nt.nodes.new("ShaderNodeOutputMaterial")
        bs = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.inputs["Roughness"].default_value = rough
        nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"]); return m, nt, bs

    def vec(nt, scale, offset, src="Object"):
        c = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Location"].default_value = offset; mp.inputs["Scale"].default_value = tuple(scale) if isinstance(scale, (list, tuple)) else (scale,) * 3
        nt.links.new(c.outputs[src], mp.inputs["Vector"]); return mp.outputs["Vector"]

    def noise(nt, scale, offset, detail=4., rough=.55, src="Object"):
        n = nt.nodes.new("ShaderNodeTexNoise"); n.inputs["Scale"].default_value = 1.; n.inputs["Detail"].default_value = detail; n.inputs["Roughness"].default_value = rough
        nt.links.new(vec(nt, scale, offset, src), n.inputs["Vector"]); return n.outputs["Fac"]

    def maprange(nt, fac, a, b, c, d):
        r = nt.nodes.new("ShaderNodeMapRange"); r.inputs["From Min"].default_value = a; r.inputs["From Max"].default_value = b
        r.inputs["To Min"].default_value = c; r.inputs["To Max"].default_value = d; nt.links.new(fac, r.inputs["Value"]); return r.outputs["Result"]

    def math(nt, op, a, b):
        n = nt.nodes.new("ShaderNodeMath"); n.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, float): n.inputs[i].default_value = v
            else: nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    def tone(nt, terms):  # 1 + sum(centred noise terms)
        acc = None
        for fac, amp in terms:
            t = maprange(nt, fac, .2, .8, -amp, amp); acc = t if acc is None else math(nt, "ADD", acc, t)
        return math(nt, "ADD", acc, 1.)

    def scale_color(nt, color, k):
        if isinstance(color, (list, tuple)):
            c = nt.nodes.new("ShaderNodeRGB"); c.outputs[0].default_value = (*color, 1); color = c.outputs[0]
        s = nt.nodes.new("ShaderNodeVectorMath"); s.operation = "SCALE"; nt.links.new(color, s.inputs[0]); nt.links.new(k, s.inputs[3]); return s.outputs[0]

    def mix(nt, fac, a, b):
        m = nt.nodes.new("ShaderNodeMix"); m.data_type = "RGBA"; nt.links.new(fac, m.inputs[0]); nt.links.new(a, m.inputs[6]); nt.links.new(b, m.inputs[7]); return m.outputs[2]

    def rgb(nt, color):
        c = nt.nodes.new("ShaderNodeRGB"); c.outputs[0].default_value = (*color, 1); return c.outputs[0]

    def finish(nt, bs, color_out, bump_src, strength, dist=.01):
        nt.links.new(color_out, bs.inputs["Base Color"])
        b = nt.nodes.new("ShaderNodeBump"); b.inputs["Strength"].default_value = strength; b.inputs["Distance"].default_value = dist
        nt.links.new(bump_src, b.inputs["Height"]); nt.links.new(b.outputs["Normal"], bs.inputs["Normal"])

    mats = {"walls": []}
    for i, p in enumerate(params["walls"]):
        m, nt, bs = new(f"realistic wall {i+1}", p["base_color"], p["roughness"])
        lf = noise(nt, p["lowfreq"]["scale"], p["lowfreq"]["offset"], 3, .5); pl = noise(nt, p["plaster"]["scale"], p["plaster"]["offset"], 5, .6)
        co = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(co.outputs["Object"], sep.inputs["Vector"])
        vg = maprange(nt, sep.outputs["Z"], 0., p["vertical"]["z_span_m"], p["vertical"]["bottom"], p["vertical"]["top"])
        t = math(nt, "MULTIPLY", tone(nt, [(lf, p["lowfreq"]["amp"]), (pl, p["plaster"]["amp"])]), vg)
        col = scale_color(nt, p["base_color"], t)
        st = p["stain"]; sn = noise(nt, st["scale"], st["offset"], 4, .55)
        mask = maprange(nt, sn, st["lo"], st["lo"] + st["width"], 0., st["strength"])
        stain = nt.nodes.new("ShaderNodeVectorMath"); stain.operation = "MULTIPLY"; nt.links.new(col, stain.inputs[0]); nt.links.new(rgb(nt, st["tint"]), stain.inputs[1])
        finish(nt, bs, mix(nt, mask, col, stain.outputs[0]), pl, p["bump_strength"], .01); mats["walls"].append(m)
    f = params["floor"]; m, nt, bs = new("realistic wood floor", f["wood_tone_a"], f["roughness"])
    brick = nt.nodes.new("ShaderNodeTexBrick"); brick.offset = f["stagger"]; brick.offset_frequency = 2; brick.squash = 1.; brick.squash_frequency = 2
    nt.links.new(vec(nt, 1., [0, 0, 0]), brick.inputs["Vector"])
    for k, v in (("Color1", f["wood_tone_a"]), ("Color2", f["wood_tone_b"]), ("Mortar", f["gap_color"])): brick.inputs[k].default_value = (*v, 1)
    for k, v in (("Scale", 1.), ("Mortar Size", f["gap_m"]), ("Mortar Smooth", .1), ("Bias", f["bias"]), ("Brick Width", f["plank_length_m"]), ("Row Height", f["plank_width_m"])): brick.inputs[k].default_value = v
    gr = noise(nt, f["grain"]["scale"], f["grain"]["offset"], 4, .6); lf = noise(nt, f["lowfreq"]["scale"], f["lowfreq"]["offset"], 2, .5)
    finish(nt, bs, scale_color(nt, brick.outputs["Color"], tone(nt, [(gr, f["grain"]["amp"]), (lf, f["lowfreq"]["amp"])])), gr, f["bump_strength"]); mats["floor"] = m
    c = params["ceiling"]; m, nt, bs = new("realistic ceiling", c["base_color"], c["roughness"])
    pl = noise(nt, c["plaster"]["scale"], c["plaster"]["offset"], 5, .6); lf = noise(nt, c["lowfreq"]["scale"], c["lowfreq"]["offset"], 3, .5)
    finish(nt, bs, scale_color(nt, c["base_color"], tone(nt, [(pl, c["plaster"]["amp"]), (lf, c["lowfreq"]["amp"])])), pl, .02); mats["ceiling"] = m
    cache = {}

    def part_material(spec: dict):
        key = json.dumps(spec, sort_keys=True)
        if key in cache: return cache[key]
        if spec["type"] == "solid":
            m, nt, bs = new("decor solid", spec["color"], spec["rough"])
            if spec.get("fabric"):
                nz = noise(nt, 90., [3., 7., 11.], 3, .5); finish(nt, bs, scale_color(nt, spec["color"], tone(nt, [(nz, spec["fabric"])])), nz, .1, .01)
            else: bs.inputs["Base Color"].default_value = (*spec["color"], 1)
        elif spec["type"] == "emit":
            m, nt, bs = new("ceiling light", spec["color"], .5); bs.inputs["Base Color"].default_value = (*spec["color"], 1)
            bs.inputs["Emission Color"].default_value = (*spec["color"], 1); bs.inputs["Emission Strength"].default_value = spec["strength"]
        else:  # art: abstract colour-block / stripe / gradient / noise picture, coordinates = the box's own Generated (0..1) coords
            m, nt, bs = new("decor art " + spec["style"], spec["colors"][0], spec["rough"]); cs = spec["colors"]; sc = spec["scale"]; ofs = spec["offset"]
            def planar(scale, offset):  # (u, z, .37) of the box face from Generated coords, so the picture lies in the wall plane
                co = nt.nodes.new("ShaderNodeTexCoord"); sp_ = nt.nodes.new("ShaderNodeSeparateXYZ"); cb = nt.nodes.new("ShaderNodeCombineXYZ")
                nt.links.new(co.outputs["Generated"], sp_.inputs["Vector"]); nt.links.new(sp_.outputs[spec["uaxis"].upper()], cb.inputs["X"]); nt.links.new(sp_.outputs["Z"], cb.inputs["Y"]); cb.inputs["Z"].default_value = .37
                mp_ = nt.nodes.new("ShaderNodeMapping"); mp_.inputs["Scale"].default_value = scale if isinstance(scale, tuple) else (scale,) * 3; mp_.inputs["Location"].default_value = offset
                nt.links.new(cb.outputs["Vector"], mp_.inputs["Vector"]); return mp_.outputs["Vector"]
            g = planar(sc, (0., 0., 0.))
            if spec["style"] == "blocks":
                ck = nt.nodes.new("ShaderNodeTexChecker"); ck.inputs["Color1"].default_value = (*cs[0], 1); ck.inputs["Color2"].default_value = (*cs[1], 1)
                nt.links.new(g, ck.inputs["Vector"]); ck.inputs["Scale"].default_value = 1.
                sp = nt.nodes.new("ShaderNodeTexGradient"); sp.gradient_type = "SPHERICAL"
                nt.links.new(planar((2., 2., 2.), (-1., -1., -1.)), sp.inputs["Vector"])
                disc = maprange(nt, sp.outputs["Fac"], 1 - 2 * spec["circle_r"], 1 - 2 * spec["circle_r"] + .02, 0., 1.)
                col = mix(nt, disc, ck.outputs["Color"], rgb(nt, cs[2]))
            elif spec["style"] == "stripes":
                wv = nt.nodes.new("ShaderNodeTexWave"); wv.wave_type = "BANDS"; wv.bands_direction = "DIAGONAL"; wv.inputs["Scale"].default_value = 1.; wv.inputs["Distortion"].default_value = 1.5
                nt.links.new(g, wv.inputs["Vector"]); col = mix(nt, maprange(nt, wv.outputs["Fac"], .45, .55, 0., 1.), rgb(nt, cs[0]), rgb(nt, cs[1]))
            else:
                n_ = nt.nodes.new("ShaderNodeTexNoise"); n_.inputs["Scale"].default_value = 1.; n_.inputs["Detail"].default_value = 3. if spec["style"] == "gradient" else 6.; n_.inputs["Roughness"].default_value = .5
                nt.links.new(planar(sc * .5, tuple(ofs)), n_.inputs["Vector"]); nz = n_.outputs["Fac"]; ramp = nt.nodes.new("ShaderNodeValToRGB")
                ramp.color_ramp.interpolation = "LINEAR" if spec["style"] == "gradient" else "CONSTANT"
                for k, pos in enumerate((.0, .35, .55, .8)):
                    el = ramp.color_ramp.elements[k] if k < 2 else ramp.color_ramp.elements.new(pos)
                    el.position = pos; el.color = (*cs[k], 1)
                nt.links.new(nz, ramp.inputs["Fac"]); col = ramp.outputs["Color"]
            nt.links.new(col, bs.inputs["Base Color"])
        cache[key] = m; return m

    mats["part"] = part_material
    return mats


def _worker(scene_path: Path, output: Path, capture_seed: int, width: int, height: int,
            frames: int, fov: float, fps: int, preview: bool, opts: dict | None = None) -> None:
    import bpy
    from mathutils import Matrix, Vector
    opts = opts or {}
    appearance = opts.get("appearance", "original"); texture_seed = int(opts.get("texture_seed", 1))
    poses_json = opts.get("poses_json")
    scene_data=json.loads(scene_path.read_text(encoding="utf-8"))
    if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
    import simulator
    if poses_json:  # render EXACTLY the supplied poses (frozen from a previous capture) for a single named path
        pj=load_poses_json(Path(poses_json)); poses=[pj["poses"]]; path_names=[opts.get("path_name","path_00")]
        per_path_seeds=[pj.get("path_seed")]
    else:
        poses=camera_poses(scene_data,frames,capture_seed); path_names=[f"path_{i:02d}" for i in range(len(poses))]
        per_path_seeds=path_seeds(scene_data,capture_seed)
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    material_log=[]
    scene=bpy.context.scene
    scene.render.engine="BLENDER_EEVEE"
    scene.eevee.taa_render_samples=32
    scene.render.resolution_x=width; scene.render.resolution_y=height; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"; scene.render.image_settings.color_mode="RGB"
    scene.view_settings.view_transform="AgX"; scene.view_settings.look="AgX - Medium High Contrast"
    scene.render.film_transparent=False
    scene.world.color=(.10,.12,.15)

    def material(name,color,scale=4.0,rough=.8):
        material_log.append({"name":name,"color":[float(c) for c in color],"noise_scale":scale,"roughness":rough})
        m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
        nt=m.node_tree; nt.nodes.clear(); out=nt.nodes.new("ShaderNodeOutputMaterial")
        bs=nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.inputs["Roughness"].default_value=rough
        tex=nt.nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value=scale
        tex.inputs["Detail"].default_value=3; tex.inputs["Roughness"].default_value=.72
        coord=nt.nodes.new("ShaderNodeTexCoord"); nt.links.new(coord.outputs["Generated"],tex.inputs["Vector"])
        ramp=nt.nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.elements[0].position=.19; ramp.color_ramp.elements[1].position=.82
        ramp.color_ramp.elements[0].color=tuple(float(c)*.58 for c in color)+(1,)
        ramp.color_ramp.elements[1].color=tuple(min(float(c)*1.36,1) for c in color)+(1,)
        nt.links.new(tex.outputs["Fac"],ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"],bs.inputs["Base Color"])
        bump=nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value=.11; bump.inputs["Distance"].default_value=.035
        nt.links.new(tex.outputs["Fac"],bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"],bs.inputs["Normal"])
        nt.links.new(bs.outputs["BSDF"],out.inputs["Surface"]); return m

    def cube(name,center,size,mat,bevel=.03):
        bpy.ops.mesh.primitive_cube_add(size=1,location=center); obj=bpy.context.object; obj.name=name
        obj.dimensions=size; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        obj.data.materials.append(mat)
        if bevel:
            mod=obj.modifiers.new("soft edges","BEVEL"); mod.width=bevel; mod.segments=2
            obj.modifiers.new("weighted corners","WEIGHTED_NORMAL")
        return obj

    w,l,h=scene_data["width"],scene_data["length"],scene_data["height"]
    tparams=None
    if appearance=="textured":
        tparams=texture_params(texture_seed,scene_data); tm=_textured_materials(bpy,tparams,scene_data)
        wall_mats=tm["walls"]; floor_mat=tm["floor"]; ceil_mat=tm["ceiling"]
    elif appearance=="realistic":
        tparams=realistic_params(texture_seed,scene_data); rm=_realistic_materials(bpy,tparams)
        tm={"furniture":_textured_materials(bpy,texture_params(texture_seed,scene_data),scene_data)["furniture"]}
        wall_mats=rm["walls"]; floor_mat=rm["floor"]; ceil_mat=rm["ceiling"]
    else:
        wall_mats=[material(f"wall {i+1}",c,5.5) for i,c in enumerate(scene_data["wall_colors"])]
        floor_mat=material("subtle floor grain",scene_data["floor_color"],3.0)
        ceil_mat=material("ceiling",scene_data["ceiling_color"],7.0)
    # Floor and ceiling.
    cube("floor",(0,0,-.07),(w,l,.14),floor_mat,.01)
    cube("ceiling",(0,0,h+.045),(w,l,.09),ceil_mat,.01)

    openings=scene_data["openings"]
    # Partition each wall into cells; cells inside an opening are omitted, leaving true apertures.
    spans=[w,l,w,l]
    for wi in range(4):
        half=spans[wi]/2
        relevant=[o for o in openings if o["wall"]==wi]
        u_edges={-half,half}; z_edges={0.,h}
        for o in relevant:
            u_edges.update([max(-half,o["center"]-o["width"]/2),min(half,o["center"]+o["width"]/2)])
            z_edges.update([max(0.,o["z_min"]),min(h,o["z_max"])])
        us=sorted(u_edges); zs=sorted(z_edges); verts=[]; faces=[]
        for ua,ub in zip(us,us[1:]):
            for za,zb in zip(zs,zs[1:]):
                um,zm=(ua+ub)/2,(za+zb)/2
                if any(abs(um-o["center"])<o["width"]/2 and o["z_min"]<zm<o["z_max"] for o in relevant): continue
                base=len(verts)
                if wi==0: coords=[(ua,l/2,za),(ub,l/2,za),(ub,l/2,zb),(ua,l/2,zb)]
                elif wi==1: coords=[(w/2,ua,za),(w/2,ub,za),(w/2,ub,zb),(w/2,ua,zb)]
                elif wi==2: coords=[(ua,-l/2,za),(ub,-l/2,za),(ub,-l/2,zb),(ua,-l/2,zb)]
                else: coords=[(-w/2,ua,za),(-w/2,ub,za),(-w/2,ub,zb),(-w/2,ua,zb)]
                verts.extend(coords); faces.append(tuple(range(base,base+4)))
        mesh=bpy.data.meshes.new(f"wall {wi} cutout mesh"); mesh.from_pydata(verts,[],faces); mesh.materials.append(wall_mats[wi])
        wall=bpy.data.objects.new(f"wall {wi} with openings",mesh); scene.collection.objects.link(wall)
        # Simple colored panels outside openings provide a visible background through apertures.
        for oi,o in enumerate(relevant):
            bg=material(f"opening backdrop {wi}-{oi}",[.36,.53,.63] if o["type"]=="window" else [.20,.25,.27],7)
            along=o["center"]; zc=(o["z_min"]+o["z_max"])/2; t=.16
            if wi==0: center=(along,l/2+t,zc); size=(o["width"],.025,o["z_max"]-o["z_min"])
            elif wi==1: center=(w/2+t,along,zc); size=(.025,o["width"],o["z_max"]-o["z_min"])
            elif wi==2: center=(along,-l/2-t,zc); size=(o["width"],.025,o["z_max"]-o["z_min"])
            else: center=(-w/2-t,along,zc); size=(.025,o["width"],o["z_max"]-o["z_min"])
            cube(f"{o['type']} backdrop",center,size,bg,0)

    decor_man=None
    if appearance=="realistic":  # decor = separate thin objects on the inward side of the unchanged shell; see decor_manifest
        layout=realistic_layout(tparams,scene_data); decor_man=decor_manifest(tparams,layout)
        for it in layout["items"]:
            for p in it["parts"]:
                mat=rm["part"](p["mat"]); c=p["center"]; sz=p["size"]
                if p["shape"]=="disc":
                    bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=.5,depth=1,location=c); o=bpy.context.object
                    o.rotation_euler={"z":(0,0,0),"y":(math.pi/2,0,0),"x":(0,math.pi/2,0)}[p["axis"]]
                    ax={"z":(0,1,2),"y":(0,2,1),"x":(2,1,0)}[p["axis"]]; o.scale=(sz[ax[0]],sz[ax[1]],sz[ax[2]])
                else:
                    bpy.ops.mesh.primitive_cube_add(size=1,location=c); o=bpy.context.object; o.scale=tuple(sz)
                    if p["roll_deg"]: o.rotation_euler[1 if it["wall"] in (0,2) else 0]=math.radians(p["roll_deg"])
                o.name=it["id"]+" "+p["name"]; o.data.materials.append(mat)
    # A few simple, seed-defined occluders. Shape and footprint stay inside the source boxes.
    for box in scene_data["furniture"]:
        x,y=box["center"]; bw,bd,bh=box["size"]; mat=tm["furniture"][box["name"]] if tparams else material(box["name"]+" fabric or wood",box["color"],8.0)
        if box["name"]=="table":
            top=max(.10,bh*.14); leg=max(.06,bw*.09)
            cube("table top",(x,y,bh-top/2),(bw,bd,top),mat,.025)
            for sx in (-1,1):
                for sy in (-1,1): cube("table leg",(x+sx*(bw/2-leg/2),y+sy*(bd/2-leg/2),(bh-top)/2),(leg,leg,bh-top),mat,.012)
        elif box["name"]=="sofa":
            cube("sofa base",(x,y,bh*.3),(bw,bd,bh*.6),mat,.1)
            cube("sofa back",(x,y-bd*.38,bh*.72),(bw,bd*.23,bh*.56),mat,.08)
            for sx in (-1,1): cube("sofa arm",(x+sx*bw*.42,y,bh*.49),(bw*.16,bd,bh*.74),mat,.07)
        else: cube(box["name"],(x,y,bh/2),(bw,bd,bh),mat,.025)

    # Soft broad lights.
    def area(name,location,power,size):
        bpy.ops.object.light_add(type="AREA",location=location); o=bpy.context.object; o.name=name
        o.data.energy=power; o.data.shape="DISK"; o.data.size=size
        o.rotation_euler=(Vector((0,0,h*.45))-o.location).to_track_quat("-Z","Y").to_euler()
    area("soft key",(0,0,h-.12),750,max(w,l)*.62)
    area("window fill",(0,0,h*.75),260,max(w,l)*.35)

    cam_data=bpy.data.cameras.new("RGB camera"); cam=bpy.data.objects.new("RGB camera",cam_data); scene.collection.objects.link(cam); scene.camera=cam
    cam_data.lens=36/(2*math.tan(math.radians(fov)/2)); cam_data.sensor_width=36; cam_data.clip_start=.05; cam_data.clip_end=max(w,l)*4
    root=output/"rgb"; private=output/"private_ground_truth"; root.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
    indices=[0] if preview else None
    if opts.get("frame_indices") is not None: indices=[int(i) for i in opts["frame_indices"]]
    rendered={}
    for pi,path in enumerate(poses):
        name=path_names[pi]; rgbdir=root/name; rgbdir.mkdir(exist_ok=True)
        selected=indices if indices is not None else range(len(path)); stored=[]; rendered[name]=list(selected)
        for fi in selected:
            pose=path[fi]; x,y,z=pose["position"]
            r_,u_,f_=simulator.camera_basis(pose)  # same convention (incl. roll) as the CPU renderer
            right,up,forward=Vector(tuple(r_)),Vector(tuple(u_)),Vector(tuple(f_))
            cam.location=(x,y,z); cam.rotation_euler=Matrix((right,up,-forward)).transposed().to_quaternion().to_euler()
            scene.render.filepath=str(rgbdir/f"{fi:06d}.png"); bpy.ops.render.render(write_still=True)
            stored.append(pose)
        (private/f"{name}.json").write_text(json.dumps({"path_seed":per_path_seeds[pi],"poses":stored,"intrinsics":{"fx":width*cam_data.lens/36,"fy":width*cam_data.lens/36,"cx":(width-1)/2,"cy":(height-1)/2}},indent=2),encoding="utf-8")
    (private/"scene.json").write_text(json.dumps(scene_data,indent=2),encoding="utf-8")
    (private/"seed_manifest.json").write_text(json.dumps({"scene_seed":scene_data["seed"],"capture_seed":capture_seed,
        "path_seeds":per_path_seeds,"paths":len(poses) if poses_json else 3,"frames_per_path":1 if preview else (len(indices) if indices is not None else frames),"width":width,"height":height,
        "renderer":"Blender EEVEE synthetic prototype"},indent=2),encoding="utf-8")
    manifest={"appearance":appearance,"texture_seed":texture_seed if appearance in ("textured","realistic") else None,
              "capture_seed":capture_seed,"scene_seed":scene_data["seed"],
              "poses_source":str(Path(poses_json).resolve()) if poses_json else "regenerated from scene + capture seed",
              "poses_sha256":{n:poses_sha256(poses[i]) for i,n in enumerate(path_names)},
              "frames_rendered":rendered,"width":width,"height":height,"fov":fov,
              "render_settings":{"engine":"BLENDER_EEVEE","taa_render_samples":32,"view_transform":"AgX","look":"AgX - Medium High Contrast"},
              "lights":"unchanged between appearances","geometry":"unchanged between appearances (materials only; no baseboards/frames added)" if appearance!="realistic" else "realistic adds thin decor/trim/rug objects listed in private_ground_truth/decor_manifest.json; shell planes unchanged",
              "materials":material_log if appearance=="original" else {"opening_backdrops":[m for m in material_log if m["name"].startswith("opening backdrop")],"procedural":tparams}}
    if decor_man is not None: (private/"decor_manifest.json").write_text(json.dumps(decor_man,indent=2),encoding="utf-8")
    (output/"appearance_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def run_worker(blender: Path, scene: Path, output: Path, capture_seed: int, width: int, height: int, frames: int,
               fov: float, fps: int, preview: bool = False, appearance: str = "original", texture_seed: int = 1,
               poses_json: Path | None = None, path_name: str = "path_00", frame_indices: list[int] | None = None) -> None:
    opts = {"appearance": appearance, "texture_seed": int(texture_seed), "poses_json": str(Path(poses_json).resolve()) if poses_json else None,
            "path_name": path_name, "frame_indices": frame_indices}
    cmd = [str(blender), "--background", "--python", str(Path(__file__).resolve()), "--", "--worker",
           str(Path(scene).resolve()), str(Path(output).resolve()), str(capture_seed), str(width), str(height), str(frames), str(fov), str(fps),
           str(int(preview)), json.dumps(opts)]
    subprocess.run(cmd, check=True)


def build_video_and_sheet(output: Path, name: str, indices: list[int], width: int, height: int, fps: int) -> None:
    """mp4 of all rendered frames of one path plus a 3x4 contact sheet of 12 evenly spaced ones."""
    import cv2
    from PIL import Image, ImageDraw
    video_dir = output / "videos"; contact_dir = output / "contact_sheets"
    video_dir.mkdir(parents=True, exist_ok=True); contact_dir.mkdir(parents=True, exist_ok=True)
    rgbdir = output / "rgb" / name
    writer = cv2.VideoWriter(str(video_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened(): raise RuntimeError(f"Could not open video writer for {name}")
    picks = set(int(i) for i in np.array(indices)[np.linspace(0, len(indices) - 1, 12, dtype=int)]); thumbs = []
    for fi in indices:
        path = rgbdir / f"{fi:06d}.png"; frame = cv2.imread(str(path))
        if frame is None: writer.release(); raise RuntimeError(f"Missing rendered frame: {path}")
        writer.write(frame)
        if fi in picks:
            thumbs.append((fi, Image.fromarray(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width // 2, height // 2)))))
    writer.release()
    sheet = Image.new("RGB", (3 * (width // 2), 4 * (height // 2)), (30, 30, 30)); draw = ImageDraw.Draw(sheet)
    for k, (fi, im) in enumerate(thumbs):
        x = (k % 3) * (width // 2); y = (k // 3) * (height // 2)
        sheet.paste(im, (x, y)); draw.rectangle((x, y, x + 48, y + 16), fill=(0, 0, 0)); draw.text((x + 3, y + 2), f"frame {fi}", fill=(255, 255, 255))
    sheet.save(contact_dir / f"{name}.jpg", quality=92)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blender",type=Path,default=DEFAULT_BLENDER); ap.add_argument("--scene",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--capture-seed",type=int,default=2026)
    ap.add_argument("--width",type=int,default=320); ap.add_argument("--height",type=int,default=240)
    ap.add_argument("--frames",type=int,default=12); ap.add_argument("--fov",type=float,default=78); ap.add_argument("--fps",type=int,default=6)
    ap.add_argument("--appearance",choices=APPEARANCES,default="original",help="materials only; geometry/lights/poses are identical")
    ap.add_argument("--texture-seed",type=int,default=1,help="RNG for textured materials only; independent of scene and capture seeds")
    ap.add_argument("--poses-json",type=Path,help="render EXACTLY these poses (private_ground_truth/path_0N.json format) instead of regenerating")
    ap.add_argument("--path-name",default=None,help="output path folder name in --poses-json mode (default: json file stem)")
    ap.add_argument("--frame-step",type=int,default=1,help="--poses-json mode: render every Nth supplied pose")
    ap.add_argument("--preview",action="store_true"); a=ap.parse_args()
    if min(a.width,a.height,a.frames,a.fps,a.frame_step)<=0 or not (10<a.fov<150): ap.error("dimensions, frames, fps and step must be positive; fov in (10,150)")
    if not a.blender.exists(): ap.error(f"Blender not found at {a.blender}")
    if a.poses_json:
        if a.output.exists() and any(a.output.iterdir()): ap.error(f"refusing to overwrite non-empty {a.output}")
        n=len(load_poses_json(a.poses_json)["poses"]); idx=list(range(0,n,a.frame_step)); name=a.path_name or a.poses_json.stem
        run_worker(a.blender,a.scene,a.output,a.capture_seed,a.width,a.height,n,a.fov,a.fps,a.preview,a.appearance,a.texture_seed,a.poses_json,name,idx[:1] if a.preview else idx)
        if not a.preview: build_video_and_sheet(a.output,name,idx,a.width,a.height,a.fps)
        return
    run_worker(a.blender,a.scene,a.output,a.capture_seed,a.width,a.height,a.frames,a.fov,a.fps,a.preview,a.appearance,a.texture_seed)
    if not a.preview:
        write_coverage(json.loads(a.scene.read_text(encoding="utf-8")),a.output,a.frames,a.fov,a.capture_seed)
        for pi in range(3): build_video_and_sheet(a.output,f"path_{pi:02d}",list(range(a.frames)),a.width,a.height,a.fps)
        manifest=a.output/"private_ground_truth"/"seed_manifest.json"
        data=json.loads(manifest.read_text(encoding="utf-8")); data["fps"]=a.fps; data["capture_sampling_seconds_per_frame"]=1/a.fps
        data["loop_nominal_duration_seconds"]=a.frames/a.fps; data["contact_sheet_samples_per_path"]=12
        manifest.write_text(json.dumps(data,indent=2),encoding="utf-8")


if "--worker" in sys.argv:
    args=sys.argv[sys.argv.index("--worker")+1:]
    _worker(Path(args[0]),Path(args[1]),int(args[2]),int(args[3]),int(args[4]),int(args[5]),float(args[6]),int(args[7]),bool(int(args[8])),
            json.loads(args[9]) if len(args)>9 else None)
elif __name__=="__main__":
    main()