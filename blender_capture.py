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

APPEARANCES = ("original", "textured")


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
    manifest={"appearance":appearance,"texture_seed":texture_seed if appearance=="textured" else None,
              "capture_seed":capture_seed,"scene_seed":scene_data["seed"],
              "poses_source":str(Path(poses_json).resolve()) if poses_json else "regenerated from scene + capture seed",
              "poses_sha256":{n:poses_sha256(poses[i]) for i,n in enumerate(path_names)},
              "frames_rendered":rendered,"width":width,"height":height,"fov":fov,
              "render_settings":{"engine":"BLENDER_EEVEE","taa_render_samples":32,"view_transform":"AgX","look":"AgX - Medium High Contrast"},
              "lights":"unchanged between appearances","geometry":"unchanged between appearances (materials only; no baseboards/frames added)",
              "materials":material_log if appearance=="original" else {"opening_backdrops":[m for m in material_log if m["name"].startswith("opening backdrop")],"procedural":tparams}}
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