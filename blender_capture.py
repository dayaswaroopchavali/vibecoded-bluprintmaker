"""Render the bounded procedural room in headless Blender's EEVEE renderer.

No asset files are downloaded. Room geometry and camera poses (360-degree yaw sweep, bounded pitch sweep)
derive from simulator.py, which also writes the coverage diagnostics. This is
still a simple synthetic scene, not a photorealistic or sensor-faithful renderer.
"""
from __future__ import annotations

import argparse
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

def _worker(scene_path: Path, output: Path, capture_seed: int, width: int, height: int,
            frames: int, fov: float, fps: int, preview: bool) -> None:
    import bpy
    from mathutils import Matrix, Vector
    scene_data=json.loads(scene_path.read_text(encoding="utf-8"))
    poses=camera_poses(scene_data,frames,capture_seed)
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.render.engine="BLENDER_EEVEE"
    scene.eevee.taa_render_samples=32
    scene.render.resolution_x=width; scene.render.resolution_y=height; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"; scene.render.image_settings.color_mode="RGB"
    scene.view_settings.view_transform="AgX"; scene.view_settings.look="AgX - Medium High Contrast"
    scene.render.film_transparent=False
    scene.world.color=(.10,.12,.15)

    def material(name,color,scale=4.0,rough=.8):
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
        x,y=box["center"]; bw,bd,bh=box["size"]; mat=material(box["name"]+" fabric or wood",box["color"],8.0)
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
    per_path_seeds=path_seeds(scene_data,capture_seed)
    for pi,path in enumerate(poses):
        name=f"path_{pi:02d}"; rgbdir=root/name; rgbdir.mkdir(exist_ok=True)
        selected=indices if indices is not None else range(len(path)); stored=[]
        for fi in selected:
            pose=path[fi]; x,y,z=pose["position"]; yaw,pitch=pose["yaw"],pose["pitch"]
            right=Vector((math.cos(yaw),-math.sin(yaw),0))
            forward=Vector((math.sin(yaw)*math.cos(pitch),math.cos(yaw)*math.cos(pitch),-math.sin(pitch)))
            up=right.cross(forward)
            cam.location=(x,y,z); cam.rotation_euler=Matrix((right,up,-forward)).transposed().to_quaternion().to_euler()
            scene.render.filepath=str(rgbdir/f"{fi:06d}.png"); bpy.ops.render.render(write_still=True)
            stored.append(pose)
        (private/f"{name}.json").write_text(json.dumps({"path_seed":per_path_seeds[pi],"poses":stored,"intrinsics":{"fx":width*cam_data.lens/36,"fy":width*cam_data.lens/36,"cx":(width-1)/2,"cy":(height-1)/2}},indent=2),encoding="utf-8")
    (private/"scene.json").write_text(json.dumps(scene_data,indent=2),encoding="utf-8")
    (private/"seed_manifest.json").write_text(json.dumps({"scene_seed":scene_data["seed"],"capture_seed":capture_seed,
        "path_seeds":per_path_seeds,"paths":3,"frames_per_path":1 if preview else frames,"width":width,"height":height,
        "renderer":"Blender EEVEE synthetic prototype"},indent=2),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blender",type=Path,default=DEFAULT_BLENDER); ap.add_argument("--scene",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--capture-seed",type=int,default=2026)
    ap.add_argument("--width",type=int,default=320); ap.add_argument("--height",type=int,default=240)
    ap.add_argument("--frames",type=int,default=12); ap.add_argument("--fov",type=float,default=78); ap.add_argument("--fps",type=int,default=6)
    ap.add_argument("--preview",action="store_true"); a=ap.parse_args()
    if min(a.width,a.height,a.frames,a.fps)<=0 or not (10<a.fov<150): ap.error("dimensions, frames and fps must be positive; fov in (10,150)")
    if not a.blender.exists(): ap.error(f"Blender not found at {a.blender}")
    cmd=[str(a.blender),"--background","--python",str(Path(__file__).resolve()),"--","--worker",
         str(a.scene.resolve()),str(a.output.resolve()),str(a.capture_seed),str(a.width),str(a.height),str(a.frames),str(a.fov),str(a.fps),str(int(a.preview))]
    subprocess.run(cmd,check=True)
    if not a.preview:
        write_coverage(json.loads(a.scene.read_text(encoding="utf-8")),a.output,a.frames,a.fov,a.capture_seed)
        import cv2
        from PIL import Image, ImageDraw
        video_dir=a.output/"videos"; contact_dir=a.output/"contact_sheets"
        video_dir.mkdir(parents=True,exist_ok=True); contact_dir.mkdir(parents=True,exist_ok=True)
        for pi in range(3):
            name=f"path_{pi:02d}"; rgbdir=a.output/"rgb"/name
            writer=cv2.VideoWriter(str(video_dir/f"{name}.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),a.fps,(a.width,a.height))
            if not writer.isOpened(): raise RuntimeError(f"Could not open video writer for {name}")
            indices=np.linspace(0,a.frames-1,12,dtype=int); thumbs=[]
            for fi in range(a.frames):
                path=rgbdir/f"{fi:06d}.png"; frame=cv2.imread(str(path))
                if frame is None: writer.release(); raise RuntimeError(f"Missing rendered frame: {path}")
                writer.write(frame)
                if fi in indices:
                    thumb=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(a.width//2,a.height//2))
                    thumbs.append((fi,Image.fromarray(thumb)))
            writer.release()
            sheet=Image.new("RGB",(3*(a.width//2),4*(a.height//2)),(30,30,30)); draw=ImageDraw.Draw(sheet)
            for k,(fi,im) in enumerate(thumbs):
                x=(k%3)*(a.width//2); y=(k//3)*(a.height//2)
                sheet.paste(im,(x,y)); draw.rectangle((x,y,x+48,y+16),fill=(0,0,0)); draw.text((x+3,y+2),f"frame {fi}",fill=(255,255,255))
            sheet.save(contact_dir/f"{name}.jpg",quality=92)
        manifest=a.output/"private_ground_truth"/"seed_manifest.json"
        data=json.loads(manifest.read_text(encoding="utf-8")); data["fps"]=a.fps; data["capture_sampling_seconds_per_frame"]=1/a.fps
        data["loop_nominal_duration_seconds"]=a.frames/a.fps; data["contact_sheet_samples_per_path"]=12
        manifest.write_text(json.dumps(data,indent=2),encoding="utf-8")


if "--worker" in sys.argv:
    args=sys.argv[sys.argv.index("--worker")+1:]
    _worker(Path(args[0]),Path(args[1]),int(args[2]),int(args[3]),int(args[4]),int(args[5]),float(args[6]),int(args[7]),bool(int(args[8])))
elif __name__=="__main__":
    main()
