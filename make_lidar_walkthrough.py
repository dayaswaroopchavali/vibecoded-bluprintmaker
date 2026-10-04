"""Make one offline, walkable viewer from truth-free frozen LiDAR predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rectify_rectangle import rectify


def _room(row: dict) -> dict:
    boundary = row.get("boundary") or {}
    height = row.get("height_estimate") or {}
    if boundary.get("status") != "complete_polygon_hypothesis" or height.get("status") != "estimated_unscaled_separation":
        return {"label": row["label"], "status": "incomplete"}
    corrected = rectify(boundary, row.get("paired_walls") or {}, row.get("openings") or {})
    if corrected["status"] == "rectified":
        boundary = corrected["boundary"]
        openings = corrected["openings"]
        geometry_source = "guarded paired-plane rectangle"
    else:
        openings = (row.get("openings") or {}).get("openings", [])
        geometry_source = f"raw boundary (rectification refused: {corrected['reason']})"
    pairs = row.get("paired_walls", {}).get("pair_measurements", [])
    return {
        "label": row["label"], "status": "complete_polygon_hypothesis",
        "vertices": boundary["vertices_xy"],
        "height": height["separation_unscaled"],
        "paired_dimensions": [p["separation_m"] for p in pairs],
        "geometry_source": geometry_source,
        "openings": [{"wall": o["wall_index"], "u": o["u_interval_m"],
                       "width": o["width_m"]} for o in openings],
    }


def build_html(freeze: dict, three_source: str) -> str:
    if freeze.get("truth_free") is not True or freeze.get("ground_truth_read") is not False:
        raise ValueError("viewer accepts only a truth-free prediction freeze")
    rooms = [_room(row) for row in freeze.get("runs", [])]
    data = json.dumps(rooms, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiDAR room walkthrough</title><style>
html,body{margin:0;height:100%;background:#101d29;color:#e9f2f5;font:14px system-ui}
#view{position:absolute;inset:0}canvas{display:block}#ui{position:absolute;top:16px;left:16px;
background:#132636e8;border:1px solid #426175;border-radius:12px;padding:14px;max-width:310px;
box-shadow:0 12px 30px #0007}h1{font-size:18px;margin:0 0 9px}select,button{background:#244257;
color:white;border:1px solid #688aa1;border-radius:6px;padding:7px;margin:3px 0;cursor:pointer}
select{width:100%}button{margin-right:5px}#meta{line-height:1.55;margin-top:8px}
.note{color:#bacdd7;font-size:12px;margin-top:9px}#help{position:absolute;right:16px;bottom:16px;
background:#132636d9;border-radius:8px;padding:8px 11px;color:#d4e5ed}
</style></head><body><div id="view"></div><div id="ui"><h1>Simulated LiDAR room</h1>
<select id="room"></select><div><button id="overview">Overview</button><button id="walk">Walk inside</button></div>
<div id="meta"></div><div class="note">Reconstructed geometry only. Yellow lines mark possible opening widths;
their vertical position and door/window type remain unresolved. No textures or furniture were reconstructed.</div></div>
<div id="help">Drag to look · WASD move · Q/E height · wheel zoom in overview</div>
<script>__THREE__</script><script>
'use strict';
const DATA=__DATA__, scene=new THREE.Scene(); scene.background=new THREE.Color(0x101d29);
const camera=new THREE.PerspectiveCamera(70,1,.01,1000); camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
document.getElementById('view').appendChild(renderer.domElement);
const group=new THREE.Group();scene.add(group);
const grid=new THREE.GridHelper(20,20,0x365268,0x284459);grid.rotation.x=Math.PI/2;scene.add(grid);
let room=null,mode='overview',center=new THREE.Vector3(),radius=5,yaw=0,pitch=0,zoom=1;
let drag=false,lastX=0,lastY=0;const keys=new Set();
function line(points,color=0xffdc75){const g=new THREE.BufferGeometry().setFromPoints(points);
 return new THREE.Line(g,new THREE.LineBasicMaterial({color}));}
function quad(a,b,c,d,color,opacity){const g=new THREE.BufferGeometry();
 const p=[...a.toArray(),...b.toArray(),...c.toArray(),...a.toArray(),...c.toArray(),...d.toArray()];
 g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));g.computeVertexNormals();
 return new THREE.Mesh(g,new THREE.MeshBasicMaterial({color,side:THREE.DoubleSide,transparent:true,opacity,depthWrite:false}));}
function setMode(next){mode=next;const H=room?.height||3;if(next==='walk'){
 camera.position.set(center.x,center.y,Math.max(.35,Math.min(H*.52,H-.2)));
 yaw=0;pitch=0;} else {zoom=1;yaw=0;pitch=-.55;}updateCamera();}
function updateCamera(){if(!room)return;if(mode==='overview'){
 const d=radius*2.3*zoom,c=Math.cos(pitch);camera.position.set(center.x+d*c*Math.sin(yaw),
 center.y-d*c*Math.cos(yaw),room.height*.48-d*Math.sin(pitch));
 camera.lookAt(center.x,center.y,room.height*.48);
 }else{const direction=new THREE.Vector3(Math.sin(yaw)*Math.cos(pitch),
 Math.cos(yaw)*Math.cos(pitch),Math.sin(pitch));camera.lookAt(camera.position.clone().add(direction));}}
function show(i){while(group.children.length){const o=group.children[0];group.remove(o);o.geometry?.dispose();o.material?.dispose();}
 room=DATA[i];if(room.status!=='complete_polygon_hypothesis'){
 document.getElementById('meta').textContent='Incomplete room hypothesis';return;}
 const v=room.vertices.map(p=>new THREE.Vector3(p[0],p[1],0));const H=room.height;
 center.set(v.reduce((s,p)=>s+p.x,0)/v.length,v.reduce((s,p)=>s+p.y,0)/v.length,0);
 radius=Math.max(2,...v.map(p=>p.distanceTo(center)));
 for(let k=0;k<v.length;k++){const a=v[k],b=v[(k+1)%v.length],at=a.clone().add(new THREE.Vector3(0,0,H)),
 bt=b.clone().add(new THREE.Vector3(0,0,H));
 group.add(quad(a,b,bt,at,0x70bbcb,.31));group.add(line([a,b],0xb4effb));group.add(line([at,bt],0x83bed0));
 group.add(line([a,at],0x83bed0));}
 if(v.length>=3){for(let k=1;k<v.length-1;k++)group.add(quad(v[0],v[k],v[k+1],v[0],0x66859a,.28));}
 for(const o of room.openings){const a=v[o.wall],b=v[(o.wall+1)%v.length];if(!a||!b)continue;
 const t=b.clone().sub(a).normalize(),z=new THREE.Vector3(0,0,.045);
 const p=a.clone().addScaledVector(t,o.u[0]).add(z),q=a.clone().addScaledVector(t,o.u[1]).add(z);
 group.add(line([p,q]));for(const x of [p,q])group.add(line([x.clone().add(new THREE.Vector3(0,0,-.04)),
 x.clone().add(new THREE.Vector3(0,0,.18))]));}
 document.getElementById('meta').innerHTML=`<b>${room.label}</b><br>Paired wall dimensions: `+
 room.paired_dimensions.map(x=>x.toFixed(3)+' m').join(' × ')+`<br>Height: ${H.toFixed(3)} m`+
 `<br>Possible openings: ${room.openings.length}<br>Geometry: ${room.geometry_source}`;
 setMode('overview');}
const select=document.getElementById('room');DATA.forEach((r,i)=>{let o=document.createElement('option');
 o.value=i;o.textContent=r.label;select.appendChild(o)});select.onchange=()=>show(Number(select.value));
document.getElementById('overview').onclick=()=>setMode('overview');
document.getElementById('walk').onclick=()=>setMode('walk');
renderer.domElement.addEventListener('pointerdown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;renderer.domElement.setPointerCapture(e.pointerId)});
renderer.domElement.addEventListener('pointerup',()=>drag=false);
renderer.domElement.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;
 lastX=e.clientX;lastY=e.clientY;yaw-=dx*.005;pitch=Math.max(-1.45,Math.min(1.45,pitch+dy*.005));updateCamera()});
renderer.domElement.addEventListener('wheel',e=>{if(mode==='overview'){e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(e.deltaY*.001)));updateCamera()}},{passive:false});
window.addEventListener('keydown',e=>{keys.add(e.key.toLowerCase());if('wasdqe'.includes(e.key.toLowerCase()))e.preventDefault()});
window.addEventListener('keyup',e=>keys.delete(e.key.toLowerCase()));
function resize(){const w=innerWidth,h=innerHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}
window.addEventListener('resize',resize);resize();if(DATA.length)show(0);
let last=performance.now();function frame(now){const dt=Math.min(.05,(now-last)/1000);last=now;
 if(mode==='walk'){const speed=Math.max(.5,radius*.55)*dt,forward=new THREE.Vector3(Math.sin(yaw),Math.cos(yaw),0),
 right=new THREE.Vector3(Math.cos(yaw),-Math.sin(yaw),0);
 if(keys.has('w'))camera.position.addScaledVector(forward,speed);if(keys.has('s'))camera.position.addScaledVector(forward,-speed);
 if(keys.has('d'))camera.position.addScaledVector(right,speed);if(keys.has('a'))camera.position.addScaledVector(right,-speed);
 if(keys.has('e'))camera.position.z+=speed;if(keys.has('q'))camera.position.z-=speed;updateCamera();}
 renderer.render(scene,camera);requestAnimationFrame(frame)}requestAnimationFrame(frame);
</script></body></html>'''
    return html.replace("__THREE__", three_source).replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--three", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    html = build_html(freeze, args.three.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
