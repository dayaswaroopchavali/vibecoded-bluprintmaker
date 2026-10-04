"""Build a self-contained viewer for unscaled box hypotheses and plane candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_html(planes: dict | None, three_source: str, hypothesis: dict | None = None) -> str:
    data = json.dumps({"planes": planes.get("planes", []) if planes else [],
                       "hypothesis": hypothesis}, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    js = r"""
'use strict';
const DATA=__DATA__;
const $=id=>document.getElementById(id);
const status=$('status');
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x17232f);
const viewport=$('view');
const renderer=new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2)); viewport.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(68,1,.001,100000);
const patchGroup=new THREE.Group(),boxGroup=new THREE.Group(); scene.add(patchGroup,boxGroup);
const colors=[0x53c5e8,0xffa94d,0xa78bfa,0x7dd87d,0xf472b6,0xfacc15,0x22d3b8,0xfb7185];
const planes=Array.isArray(DATA.planes)?DATA.planes:[];
const hypothesis=DATA.hypothesis;
const box=new THREE.Box3();
for(let i=0;i<planes.length;i++){
 const p=planes[i];
 if(!Array.isArray(p.center)||!Array.isArray(p.normal)||!Array.isArray(p.observed_bounds_world)||p.observed_bounds_world.length!==2) continue;
 const center=new THREE.Vector3(...p.center), normal=new THREE.Vector3(...p.normal).normalize();
 if(normal.lengthSq()<1e-10) continue;
 let u=new THREE.Vector3(0,1,0).cross(normal);
 if(u.lengthSq()<1e-8) u=new THREE.Vector3(1,0,0).cross(normal);
 u.normalize(); const v=normal.clone().cross(u).normalize();
 const corners=[];
 for(let mask=0;mask<8;mask++) corners.push(new THREE.Vector3(
   p.observed_bounds_world[(mask>>0)&1][0],p.observed_bounds_world[(mask>>1)&1][1],p.observed_bounds_world[(mask>>2)&1][2]));
 const pu=corners.map(q=>q.clone().sub(center).dot(u)),pv=corners.map(q=>q.clone().sub(center).dot(v));
 const w=Math.max(...pu)-Math.min(...pu),h=Math.max(...pv)-Math.min(...pv);
 if(!(w>0&&h>0&&Number.isFinite(w+h))) continue;
 const mesh=new THREE.Mesh(new THREE.PlaneGeometry(w,h),new THREE.MeshBasicMaterial({color:colors[i%colors.length],side:THREE.DoubleSide,transparent:true,opacity:.55,depthWrite:false}));
 mesh.position.copy(center).addScaledVector(u,(Math.max(...pu)+Math.min(...pu))/2).addScaledVector(v,(Math.max(...pv)+Math.min(...pv))/2);
 mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(u,v,normal));
 mesh.userData.candidate=i; patchGroup.add(mesh); box.expandByObject(mesh);
}
const boxCorners=Array.isArray(hypothesis?.corners)&&hypothesis.corners.length===8?hypothesis.corners.map(p=>new THREE.Vector3(...p)):[];
if(boxCorners.length===8){
 const faces=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[1,3,7,5],[3,2,6,7],[2,0,4,6]];
 for(let i=0;i<faces.length;i++){
  const [a,b,c,d]=faces[i];
  const positions=[...boxCorners[a].toArray(),...boxCorners[b].toArray(),...boxCorners[c].toArray(),
    ...boxCorners[a].toArray(),...boxCorners[c].toArray(),...boxCorners[d].toArray()];
  const faceGeometry=new THREE.BufferGeometry();
  faceGeometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));
  faceGeometry.computeVertexNormals();
  boxGroup.add(new THREE.Mesh(faceGeometry,new THREE.MeshBasicMaterial({
    color:i===0?0x708a9a:i===1?0x67849c:0x6bb5c9,
    side:THREE.DoubleSide,transparent:true,opacity:i===0?.42:.16,depthWrite:false
  })));
 }
 const vertices=[];
 for(let i=0;i<8;i++)for(const bit of [1,2,4])if((i&bit)===0){vertices.push(...boxCorners[i].toArray(),...boxCorners[i|bit].toArray());}
 const edgeGeometry=new THREE.BufferGeometry();edgeGeometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
 boxGroup.add(new THREE.LineSegments(edgeGeometry,new THREE.LineBasicMaterial({color:0xffe08a,linewidth:2})));
 for(const point of boxCorners){const marker=new THREE.Mesh(new THREE.SphereGeometry(.012,8,6),new THREE.MeshBasicMaterial({color:0xffe08a}));marker.position.copy(point);boxGroup.add(marker);box.expandByPoint(point);}
}
const center=box.isEmpty()?new THREE.Vector3():box.getCenter(new THREE.Vector3());
const radius=box.isEmpty()?5:Math.max(box.getSize(new THREE.Vector3()).length()*.55,.05);
let mode='first-person',orbitYaw=0,orbitPitch=.35,orbitDistance=radius*2.4;
const keys=new Set(); let dragging=false,lastX=0,lastY=0,fpYaw=0,fpPitch=0;
function firstPose(){ camera.position.set(center.x,center.y,center.z+radius*.45); camera.lookAt(center); fpYaw=Math.atan2(center.x-camera.position.x,center.z-camera.position.z); fpPitch=0; }
function updateOrbit(){orbitPitch=Math.max(-1.48,Math.min(1.48,orbitPitch));const cp=Math.cos(orbitPitch);camera.position.set(center.x+orbitDistance*cp*Math.sin(orbitYaw),center.y+orbitDistance*Math.sin(orbitPitch),center.z+orbitDistance*cp*Math.cos(orbitYaw));camera.lookAt(center);}
function setMode(next){mode=next;$('mode').textContent=mode==='first-person'?'Switch to orbit overview':'Switch to first-person';if(mode==='first-person')firstPose();else updateOrbit();status.textContent=mode==='first-person'?'First-person · WASD move · drag to look':'Orbit overview · drag to rotate · wheel to zoom';}
function reset(){orbitYaw=0;orbitPitch=.35;orbitDistance=radius*2.4;setMode('first-person');}
firstPose();
const hasBox=boxCorners.length===8;
status.textContent=hasBox?'Unscaled box hypothesis · vertical axis unknown · arbitrary units':(planes.length?`Unscaled plane candidates only · ${patchGroup.children.length} patches · arbitrary units`:'No complete box or plane candidates in input.');
if($('togglePlanes'))$('togglePlanes').onclick=()=>{patchGroup.visible=!patchGroup.visible;$('togglePlanes').textContent='Plane patches: '+(patchGroup.visible?'on':'off');};
$('mode').onclick=()=>setMode(mode==='first-person'?'orbit':'first-person'); $('reset').onclick=reset; $('overview').onclick=()=>setMode('orbit');
renderer.domElement.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;renderer.domElement.setPointerCapture(e.pointerId)});
renderer.domElement.addEventListener('pointerup',()=>dragging=false);
renderer.domElement.addEventListener('pointermove',e=>{if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;if(mode==='first-person'){fpYaw-=dx*.004;fpPitch=Math.max(-1.5,Math.min(1.5,fpPitch-dy*.004));}else{orbitYaw-=dx*.006;orbitPitch-=dy*.006;updateOrbit();}});
renderer.domElement.addEventListener('wheel',e=>{if(mode==='orbit'){orbitDistance=Math.max(radius*.2,Math.min(radius*30,orbitDistance*Math.exp(e.deltaY*.001)));updateOrbit();}},{passive:true});
window.addEventListener('keydown',e=>{if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.code)){keys.add(e.code);e.preventDefault();}});
window.addEventListener('keyup',e=>keys.delete(e.code));
function move(dt){if(mode!=='first-person')return;const speed=radius*.45*dt,forward=new THREE.Vector3(Math.sin(fpYaw),0,Math.cos(fpYaw)),right=new THREE.Vector3(forward.z,0,-forward.x);if(keys.has('KeyW')||keys.has('ArrowUp'))camera.position.addScaledVector(forward,speed);if(keys.has('KeyS')||keys.has('ArrowDown'))camera.position.addScaledVector(forward,-speed);if(keys.has('KeyD')||keys.has('ArrowRight'))camera.position.addScaledVector(right,speed);if(keys.has('KeyA')||keys.has('ArrowLeft'))camera.position.addScaledVector(right,-speed);camera.rotation.order='YXZ';camera.rotation.y=fpYaw;camera.rotation.x=fpPitch;}
let last=performance.now();function resize(){const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/Math.max(h,1);camera.updateProjectionMatrix();}new ResizeObserver(resize).observe(viewport);resize();
function animate(now){requestAnimationFrame(animate);const dt=Math.min((now-last)/1000,.05);last=now;move(dt);renderer.render(scene,camera);}requestAnimationFrame(animate);
""".replace("__DATA__", data)
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Unscaled room-box hypothesis</title>
<style>*{box-sizing:border-box}body{margin:0;background:#17232f;color:#eef3f8;font:15px system-ui}header{position:absolute;z-index:2;top:12px;left:12px;padding:12px;background:#202d3beF;border-radius:8px;max-width:370px}button{font:inherit;color:white;background:#314961;border:1px solid #71859a;border-radius:5px;padding:8px 10px;margin:2px;cursor:pointer}#view{width:100vw;height:100vh}canvas{display:block;width:100%;height:100%;touch-action:none}#status{margin-top:8px;color:#b9d8e8}p{font-size:13px;line-height:1.4;margin:8px 0 0;color:#c4d0de}</style>
<header><b>Unscaled room-box hypothesis</b><div><button id="mode">Switch to orbit overview</button><button id="overview">Orbit overview</button><button id="reset">Reset view</button><button id="togglePlanes">Plane patches: on</button></div><div id="status">Unscaled box hypothesis · vertical axis unknown</div><p>Axes are unordered and the vertical axis is unknown. WASD to move; drag to look. Plane patches use observed point bounds. This is not a blueprint and does not validate boundaries or openings.</p></header><div id="view"></div>
<script>""" + three_source + "</script><script>" + js + "</script></html>"


def build_projection_svg(hypothesis: dict) -> str:
    corners = np.asarray(hypothesis.get("corners"), dtype=float)
    axes = np.asarray([axis["direction_world"] for axis in hypothesis.get("axes", [])], dtype=float)
    if corners.shape != (8, 3) or axes.shape != (3, 3):
        raise ValueError("SVG projections require a complete 8-corner hypothesis with 3 axes")
    center = corners.mean(axis=0)
    pairs = ((0, 1), (0, 2), (1, 2))
    panels = []
    for panel, (ia, ib) in enumerate(pairs):
        projected = np.column_stack(((corners-center) @ axes[ia], (corners-center) @ axes[ib]))
        lo, hi = projected.min(axis=0), projected.max(axis=0)
        span = max(float((hi-lo).max()), 1e-8)
        scale = 175.0 / span
        xoff, yoff = 35 + panel*260, 220
        points = [(xoff+20+(p[0]-lo[0])*scale, yoff-20-(p[1]-lo[1])*scale) for p in projected]
        lines = []
        for i, (x, y) in enumerate(points):
            for bit in (1, 2, 4):
                if i & bit == 0:
                    x2, y2 = points[i|bit]
                    edge_axis = {1: 2, 2: 1, 4: 0}[bit]
                    color = "#ffcf68" if edge_axis in (ia, ib) else "#637789"
                    lines.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="2"/>')
        panels.append(f'<text x="{xoff+105}" y="25" fill="#eaf2f8" text-anchor="middle">Axis {ia} × axis {ib}</text>'+"".join(lines))
    return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="250" viewBox="0 0 800 250"><rect width="100%" height="100%" fill="#17232f"/><text x="400" y="15" fill="#b9d8e8" text-anchor="middle" font-family="sans-serif" font-size="13">Unordered cuboid projections · arbitrary units · vertical axis unknown</text>'+"".join(panels)+'</svg>'


STATUS_LABELS = ("observed", "inferred", "incomplete", "unavailable")
UNAVAILABLE_STAGES = {
    "light_field": "No light-field stage exists in this pipeline; the point cloud is not a light field.",
    "segmentation": "No semantic or furniture segmentation stage exists; plane patches may include furniture.",
    "completion_mesh": "No completed mesh is produced; occluded geometry is not filled in.",
    "ceiling_height": "No ceiling-height estimate exists.",
    "openings": "Doors and windows are not detected.",
}


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _up_rotation(up) -> np.ndarray:
    """Rotation taking the declared up vector to +Y; identity when no prior is available."""
    if up is None:
        return np.eye(3)
    up = np.asarray(up, dtype=float); up = up / np.linalg.norm(up); target = np.array([0., 1., 0.])
    v = np.cross(up, target); c = float(up @ target)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def sample_points(prediction: Path, max_points: int = 24000, stride: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB-derived back-projected points, colours and camera centres (same convention as extract_planes)."""
    with np.load(prediction) as data:
        depth, conf, K, E = data["depth"], data["confidence"], data["intrinsics"], data["extrinsics"]
        images = data["images"] if "images" in data else None
    chunks, colors, centers = [], [], []
    for i in range(len(depth)):
        y, x = np.mgrid[0:depth.shape[1]:stride, 0:depth.shape[2]:stride]
        z, c = depth[i][y, x], conf[i][y, x]
        valid = np.isfinite(z) & (z > 0) & np.isfinite(c)
        if not valid.any():
            continue
        valid &= c >= np.percentile(c[valid], 50)
        pixels = np.stack([x[valid], y[valid], np.ones(valid.sum())], axis=1)
        camera = (pixels @ np.linalg.inv(K[i]).T) * z[valid, None]
        chunks.append((camera - E[i][:3, 3]) @ E[i][:3, :3])
        centers.append(-E[i][:3, :3].T @ E[i][:3, 3])
        if images is not None and images[i].shape[:2] == depth[i].shape:
            colors.append(images[i][y, x][valid].reshape(-1, 3) if images[i].dtype == np.uint8 else (images[i][y, x][valid] * 255))
        else:
            colors.append(np.full((int(valid.sum()), 3), 200, dtype=np.uint8))
    if not chunks:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    points, colors = np.concatenate(chunks), np.concatenate(colors).astype(np.uint8)
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return points, colors, np.asarray(centers)


def locate_prediction(run_dir: Path, manifest: dict | None) -> Path | None:
    candidates = [Path(manifest["prediction"])] if manifest and manifest.get("prediction") else []
    candidates += [run_dir / "da3" / "prediction.npz", run_dir / "prediction.npz"]
    return next((p for p in candidates if p.is_file()), None)


def plan_svg(polygon: dict | None, cameras_xy: np.ndarray | None) -> tuple[str, str, str]:
    """Floor-plan SVG, a status label and a one-line explanation. Incomplete results draw no closed polygon."""
    if not polygon or "status" not in polygon:
        return "", "unavailable", "No polygon stage output."
    status = polygon["status"]
    edges, walls = polygon.get("edges") or [], polygon.get("observed_wall_segments") or []
    segments = [s for w in walls for s in w.get("segments_xy", [])]
    pts = [p for e in edges for p in (e["start_xy"], e["end_xy"])] + [p for s in segments for p in s]
    if not pts:
        reasons = ", ".join(b.get("reason", "?") for b in polygon.get("incomplete_boundaries", [])) or status
        return "", "incomplete" if status != "insufficient_geometry" else "unavailable", f"No plan drawn: {reasons}."
    arr = np.asarray(pts, dtype=float); lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = max(float((hi - lo).max()), 1e-9); scale = 420 / span; pad = 40

    def tf(p):
        return pad + (p[0] - lo[0]) * scale, pad + 420 - (p[1] - lo[1]) * scale
    body = []
    if status == "complete_polygon_hypothesis":
        for e in edges:
            (x1, y1), (x2, y2) = tf(e["start_xy"]), tf(e["end_xy"])
            inferred = e["label"] == "inferred"; dash = ' stroke-dasharray="12 8"' if inferred else ""
            body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{"#ffb347" if inferred else "#5fd38d"}" '
                        f'stroke-width="5"{dash}><title>edge {e["index"]}: {e["label"]}, '
                        f'{e["observed_fraction"]:.0%} observed, length {e["length"]:.3f} (unscaled)</title></line>')
            for seg in e.get("observed_segments_xy", []):
                (a, b), (c, d) = tf(seg[0]), tf(seg[1])
                body.append(f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="#ffffff" stroke-width="1.5" opacity=".8"/>')
        label = "inferred" if any(e["label"] == "inferred" for e in edges) else "observed"
        note = f'{len(edges)} edges, {sum(e["label"] == "observed" for e in edges)} observed, {sum(e["label"] == "inferred" for e in edges)} inferred; arbitrary units.'
    else:
        for seg in segments:
            (x1, y1), (x2, y2) = tf(seg[0]), tf(seg[1])
            body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#5aa9e6" stroke-width="4"><title>observed wall segment</title></line>')
        reasons = ", ".join(b.get("reason", "?") for b in polygon.get("incomplete_boundaries", [])) or status
        label, note = "incomplete", f"INCOMPLETE: no polygon closed. Only observed wall segments are drawn. Reasons: {reasons}."
        body.append(f'<text x="{pad}" y="{pad - 14}" fill="#ff7b7b" font-size="13" font-family="sans-serif">INCOMPLETE - not a room outline</text>')
    for c in (cameras_xy if cameras_xy is not None else []):
        x, y = tf(c); body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#d7dce2" opacity=".7"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2 * pad + 420} {2 * pad + 420}" role="img" aria-label="floor plan, {label}">'
           f'<rect width="100%" height="100%" fill="#101a24"/>{"".join(body)}</svg>')
    return svg, label, note


def build_pipeline_data(run_dir: Path, video: Path | None, html_dir: Path) -> dict:
    """Everything the pipeline viewer shows, gathered from one run directory; absent stages stay absent."""
    manifest = _read_json(run_dir / "pipeline_manifest.json")
    planes, polygon = _read_json(run_dir / "planes.json"), _read_json(run_dir / "polygon.json")
    hypothesis_note = ""
    candidates = sorted(run_dir.glob("polygon_axis_hypothesis_*.json"))
    loaded = [(p.stem, _read_json(p)) for p in candidates]
    loaded = [(n, d) for n, d in loaded if d]
    if loaded:   # show the best axis-hypothesis polygon (complete first); it is a hypothesis, never gravity
        name, polygon = sorted(loaded, key=lambda item: item[1].get("status") != "complete_polygon_hypothesis")[0]
        hypothesis_note = f" Shown under {name.replace('polygon_', '')}: a structural axis hypothesis, sign (floor vs ceiling) unresolved, not gravity; {len(loaded)} co-supported hypothesis polygon(s) exist."
    cuboid = _read_json(run_dir / "cuboid_baseline.json")
    rot = np.eye(3)  # arbitrary DA3 world frame: no gravity from camera up or camera path; orientation is unresolved
    prediction = locate_prediction(run_dir, manifest)
    out = {"run": run_dir.name, "frame": "arbitrary DA3 world frame; room orientation unresolved (floor/ceiling not labelled)",
           "stages": (manifest or {}).get("stages", []), "blockers": (manifest or {}).get("named_blockers", []),
           "unavailable": UNAVAILABLE_STAGES}
    video = video or (Path(manifest["input_video"]) if manifest and manifest.get("input_video") else None)
    if video is not None and Path(video).is_file():
        out["video"] = {"status": "observed", "src": Path(os_relpath(Path(video).resolve(), html_dir.resolve())).as_posix(), "name": Path(video).name}
    else:
        out["video"] = {"status": "unavailable", "reason": "no input video was supplied or found"}
    cameras_xy = None
    if prediction is not None:
        points, colors, centers = sample_points(prediction)
        out["points"] = {"status": "observed", "count": int(len(points)), "positions": np.round(points @ rot.T, 4).flatten().tolist(),
                         "colors": colors.flatten().tolist(), "cameras": np.round(centers @ rot.T, 4).flatten().tolist(),
                         "note": "RGB-derived DA3 point geometry, unscaled; a diagnostic point cloud, not a light field."}
        if polygon and polygon.get("axes_world") and polygon.get("coordinate_origin_world") and len(centers):
            ax = polygon["axes_world"]; rel = centers - np.asarray(polygon["coordinate_origin_world"])
            cameras_xy = np.stack([rel @ np.asarray(ax["e1"]), rel @ np.asarray(ax["e2"])], axis=1)
    else:
        out["points"] = {"status": "unavailable", "reason": "prediction.npz not found"}
    quads = []
    for i, plane in enumerate((planes or {}).get("planes", [])):
        info = plane.get("surface_patches")
        if not info:
            continue
        u, v = (np.asarray(a) for a in info["axes_world"]); center = np.asarray(plane["center"])
        for patch in info["patches"]:
            (u0, v0), (u1, v1) = patch["bounds_uv"]
            corners = np.asarray([center + u * a + v * b for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))]) @ rot.T
            quads.append({"plane": i, "support": int(patch["support_count"]), "corners": np.round(corners, 4).tolist()})
    out["patches"] = ({"status": "observed", "quads": quads, "note": "Observed plane patches (candidates). Furniture is not excluded here; they are not walls."}
                      if quads else {"status": "unavailable", "reason": "planes.json missing or has no surface patches"})
    svg, label, note = plan_svg(polygon, cameras_xy)
    out["polygon"] = {"status": label, "svg": svg, "note": note + hypothesis_note, "model": (polygon or {}).get("model"),
                      "rejected": [{"plane": r.get("plane_index"), "reason": r.get("reason")} for r in (polygon or {}).get("rejected_candidates", [])]}
    out["cuboid_baseline"] = ({"status": cuboid.get("status"), "note": "diagnostic baseline only; never shown as the room"} if cuboid else None)
    return out


def os_relpath(target: Path, start: Path) -> str:
    import os
    try:
        return os.path.relpath(target, start)
    except ValueError:  # different drive
        return target.as_uri()


def build_pipeline_html(data: dict, three_source: str) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    css = ("*{box-sizing:border-box}body{margin:0;background:#17232f;color:#eef3f8;font:15px system-ui}header{padding:12px 16px;background:#202d3b}"
           "nav{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px;background:#1b2836}nav button{font:inherit;color:#fff;background:#314961;border:1px solid #71859a;border-radius:6px;padding:7px 11px;cursor:pointer}"
           "nav button.on{background:#4a6f94;border-color:#b9d8e8}.chip{font-size:11px;text-transform:uppercase;letter-spacing:.04em;border-radius:9px;padding:1px 7px;margin-left:6px}"
           ".observed{background:#1f6b45}.inferred{background:#8a5a12}.incomplete{background:#8a2a2a}.unavailable{background:#4a4f57}"
           "section{display:none;padding:14px 16px}section.on{display:block}#view{width:100%;height:68vh;min-height:320px;background:#101a24;border-radius:8px}canvas{width:100%;height:100%;display:block;touch-action:none}"
           "table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #31465a;padding:5px 8px;text-align:left;vertical-align:top}p.note{color:#c4d0de;font-size:13px;max-width:70em}"
           "video{max-width:100%;border-radius:8px;background:#000}svg{max-width:560px;width:100%;border-radius:8px}")
    js = r"""
'use strict';
const D=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);
const chip=(s)=>`<span class="chip ${s}">${s}</span>`;
const tabs=[['video','Input video',D.video.status],['points','Point geometry',D.points.status],['patches','Plane patches',D.patches.status],
 ['polygon','Floor-plan polygon',D.polygon.status],['stages','Stages & blockers','observed'],['unavailable','Unavailable stages','unavailable']];
$('nav').innerHTML=tabs.map(t=>`<button data-t="${t[0]}">${t[1]}${chip(t[2])}</button>`).join('');
function show(t){document.querySelectorAll('section').forEach(s=>s.classList.toggle('on',s.id==='s-'+t));
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));
 const three=(t==='points'||t==='patches');$('s-3d').classList.toggle('on',three);if(three){setLayer(t);resize();}}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>show(b.dataset.t));
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('s-video').innerHTML=D.video.status==='observed'?`<p class="note">Input RGB walkthrough ${chip('observed')} ${esc(D.video.name)}</p><video controls src="${esc(D.video.src)}"></video>`:`<p class="note">${chip('unavailable')} ${esc(D.video.reason)}</p>`;
$('s-polygon').innerHTML=`<p class="note">${chip(D.polygon.status)} ${esc(D.polygon.note)}</p>`+(D.polygon.svg||'<p class="note">No plan to draw.</p>')+
 `<p class="note">Green solid = observed edge, amber dashed = inferred edge (labelled continuation), blue = observed wall segment of an incomplete result. A cuboid is never drawn as the room.</p>`+
 (D.polygon.rejected.length?`<p class="note">Rejected as non-wall candidates: ${D.polygon.rejected.map(r=>'plane '+r.plane+' ('+esc(r.reason)+')').join(', ')}</p>`:'')+
 (D.cuboid_baseline?`<p class="note">Cuboid baseline (diagnostic only, not shown as a room): ${esc(D.cuboid_baseline.status)}</p>`:'');
$('s-stages').innerHTML='<table><tr><th>stage</th><th>status</th><th>seconds</th><th>reasons</th></tr>'+D.stages.map(s=>`<tr><td>${esc(s.name)}</td><td>${chip(s.status==='ok'?'observed':(s.status==='incomplete'?'incomplete':(s.status==='blocked'?'unavailable':'incomplete')))} ${esc(s.status)}</td><td>${s.duration_seconds??''}</td><td>${esc((s.failure_reasons||[]).join('; '))}</td></tr>`).join('')+
 '</table><h3>Named blockers (open)</h3><table>'+D.blockers.map(b=>`<tr><td>${esc(b.name)}</td><td>${esc(b.detail)}</td></tr>`).join('')+'</table>';
$('s-unavailable').innerHTML=Object.entries(D.unavailable).map(([k,v])=>`<p class="note">${chip('unavailable')} <b>${esc(k)}</b>: ${esc(v)}</p>`).join('');
const view=$('view');let renderer=null,scene,camera,pointsObj,patchGroup,camLine;
function initThree(){if(renderer||typeof THREE==='undefined')return;
 renderer=new THREE.WebGLRenderer({antialias:true});view.appendChild(renderer.domElement);scene=new THREE.Scene();scene.background=new THREE.Color(0x101a24);
 camera=new THREE.PerspectiveCamera(60,1,.001,100000);
 const P=D.points;if(P.status==='observed'){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(P.positions,3));
  g.setAttribute('color',new THREE.Float32BufferAttribute(P.colors.map(c=>c/255),3));pointsObj=new THREE.Points(g,new THREE.PointsMaterial({size:.012,vertexColors:true,sizeAttenuation:true}));scene.add(pointsObj);
  if(P.cameras.length>5){const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.Float32BufferAttribute(P.cameras,3));camLine=new THREE.Line(cg,new THREE.LineBasicMaterial({color:0xffe08a}));scene.add(camLine);}}
 patchGroup=new THREE.Group();scene.add(patchGroup);const col=[0x53c5e8,0xffa94d,0xa78bfa,0x7dd87d,0xf472b6,0xfacc15,0x22d3b8,0xfb7185];
 if(D.patches.status==='observed')for(const q of D.patches.quads){const g=new THREE.BufferGeometry();const c=q.corners;
  g.setAttribute('position',new THREE.Float32BufferAttribute([...c[0],...c[1],...c[2],...c[0],...c[2],...c[3]],3));
  patchGroup.add(new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:col[q.plane%8],side:THREE.DoubleSide,transparent:true,opacity:.5,depthWrite:false})));}
 const bb=new THREE.Box3();if(pointsObj){pointsObj.geometry.computeBoundingBox();bb.copy(pointsObj.geometry.boundingBox);}else patchGroup.children.forEach(m=>{m.geometry.computeBoundingBox();bb.union(m.geometry.boundingBox);});
 ctr=bb.isEmpty()?new THREE.Vector3():bb.getCenter(new THREE.Vector3());rad=bb.isEmpty()?5:Math.max(bb.getSize(new THREE.Vector3()).length()*.5,.05);dist=rad*1.6;
 const cv=renderer.domElement;cv.onpointerdown=e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId)};cv.onpointerup=()=>drag=false;
 cv.onpointermove=e=>{if(!drag)return;yaw-=(e.clientX-lx)*.006;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-ly)*.006));lx=e.clientX;ly=e.clientY;orbit();};
 cv.addEventListener('wheel',e=>{dist=Math.max(rad*.1,Math.min(rad*20,dist*Math.exp(e.deltaY*.001)));orbit();e.preventDefault();},{passive:false});orbit();}
let ctr,rad,dist,yaw=.6,pitch=.4,drag=false,lx=0,ly=0;
function orbit(){const cp=Math.cos(pitch);camera.position.set(ctr.x+dist*cp*Math.sin(yaw),ctr.y+dist*Math.sin(pitch),ctr.z+dist*cp*Math.cos(yaw));camera.lookAt(ctr);renderer.render(scene,camera);}
function resize(){if(!renderer)return;const w=view.clientWidth,h=view.clientHeight;renderer.setSize(w,h);camera.aspect=w/Math.max(h,1);camera.updateProjectionMatrix();orbit();}
function setLayer(t){initThree();if(!renderer){$('s3note').textContent='3D view unavailable: WebGL or the bundled renderer is missing.';return;}
 if(pointsObj)pointsObj.visible=(t==='points');if(camLine)camLine.visible=(t==='points');patchGroup.visible=(t==='patches');
 $('s3note').innerHTML=t==='points'?chip(D.points.status)+' '+esc(D.points.note||D.points.reason||'')+' Frame: '+esc(D.frame)+'. Yellow line = predicted camera path.':chip(D.patches.status)+' '+esc(D.patches.note||D.patches.reason||'')+' Frame: '+esc(D.frame)+'.';orbit();}
window.addEventListener('resize',resize);show('video');
"""
    return ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Pipeline run viewer</title><style>{css}</style>"
            f"<header><b>Pipeline run: {data['run']}</b> &middot; unscaled arbitrary units &middot; no completion or accuracy claim</header><nav id=\"nav\"></nav>"
            "<section id=\"s-video\"></section><section id=\"s-3d\"><p class=\"note\" id=\"s3note\"></p><div id=\"view\"></div></section>"
            "<section id=\"s-polygon\"></section><section id=\"s-stages\"></section><section id=\"s-unavailable\"></section>"
            f"<script id=\"data\" type=\"application/json\">{payload}</script><script>{three_source}</script><script>{js}</script></html>")


def _main_pipeline(args, parser) -> int:
    run_dir, output = args.pipeline_run.expanduser().resolve(), args.output.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"pipeline run folder does not exist: {run_dir}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    three_path = Path(__file__).resolve().parents[2] / "outputs" / "re3sim-trial" / "viewer-assets" / "three.js"
    if not three_path.is_file():
        parser.error(f"local Three.js bundle not found: {three_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = build_pipeline_data(run_dir, args.video.resolve() if args.video else None, output.parent)
    output.write_text(build_pipeline_html(data, three_path.read_text(encoding="utf-8")), encoding="utf-8")
    print(output)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planes", type=Path, help="optional plane-candidate JSON from extract_planes.py")
    parser.add_argument("--hypothesis", type=Path, help="optional room_hypothesis.json with actual eight corners")
    parser.add_argument("--output", type=Path, required=True, help="path for the standalone HTML viewer")
    parser.add_argument("--svg", type=Path, help="optional unordered-cuboid projection SVG output")
    parser.add_argument("--pipeline-run", type=Path, help="run_pipeline.py output folder: build the tabbed pipeline viewer instead")
    parser.add_argument("--video", type=Path, help="input video to show in the pipeline viewer")
    args = parser.parse_args()
    if args.pipeline_run is not None:
        return _main_pipeline(args, parser)
    if args.planes is None and args.hypothesis is None:
        parser.error("provide at least one of --planes or --hypothesis")
    output = args.output.expanduser().resolve()
    plane_source = args.planes.expanduser().resolve() if args.planes else None
    hypothesis_source = args.hypothesis.expanduser().resolve() if args.hypothesis else None
    for source in (plane_source, hypothesis_source):
        if source is not None and not source.is_file():
            parser.error(f"input JSON does not exist: {source}")
    svg_path = args.svg.expanduser().resolve() if args.svg else None
    if svg_path is not None and svg_path.exists():
        parser.error(f"SVG output already exists: {svg_path}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    repo = Path(__file__).resolve().parents[2]
    three_path = repo / "outputs" / "re3sim-trial" / "viewer-assets" / "three.js"
    if not three_path.is_file():
        parser.error(f"local Three.js bundle not found: {three_path}")
    try:
        planes = json.loads(plane_source.read_text(encoding="utf-8")) if plane_source else None
        hypothesis = json.loads(hypothesis_source.read_text(encoding="utf-8")) if hypothesis_source else None
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read input JSON: {exc}")
    if planes is not None and (not isinstance(planes, dict) or not isinstance(planes.get("planes"), list)):
        parser.error("expected plane-candidate JSON object containing a planes array")
    if hypothesis is not None and not isinstance(hypothesis, dict):
        parser.error("expected room-hypothesis JSON object")
    html = build_html(planes, three_path.read_text(encoding="utf-8"), hypothesis)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    if svg_path is not None:
        if hypothesis is None:
            parser.error("--svg requires --hypothesis")
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(build_projection_svg(hypothesis), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
