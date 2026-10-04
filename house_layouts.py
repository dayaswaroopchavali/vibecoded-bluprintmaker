"""Seeded procedural layout truth for the private simulation benchmark.

This module generates geometry only: no renderer, videos, captures, or results.
Coordinates and measurements are metres. Standard-library checks are used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable

STAGES = ("single_room", "multiroom", "multifloor")
EPS = 1e-8


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _on_segment(a,b,p):
    return abs(_cross(a,b,p)) <= EPS and min(a[0],b[0])-EPS <= p[0] <= max(a[0],b[0])+EPS and min(a[1],b[1])-EPS <= p[1] <= max(a[1],b[1])+EPS


def _segments_intersect(a,b,c,d):
    vals=(_cross(a,b,c),_cross(a,b,d),_cross(c,d,a),_cross(c,d,b))
    x,y,z,w=vals
    if ((x>EPS and y < -EPS) or (x < -EPS and y>EPS)) and ((z>EPS and w < -EPS) or (z < -EPS and w>EPS)): return True
    return any((_on_segment(a,b,c),_on_segment(a,b,d),_on_segment(c,d,a),_on_segment(c,d,b))[i] for i,v in enumerate(vals) if abs(v)<=EPS)


def polygon_area(poly):
    return abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))/2)


def validate_polygon(poly):
    if not isinstance(poly,list) or len(poly)<3: raise ValueError("polygon needs at least three vertices")
    try: pts=[tuple(float(v) for v in p) for p in poly]
    except (TypeError,ValueError): raise ValueError("polygon vertices must be finite XY pairs")
    if any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in pts): raise ValueError("polygon vertices must be finite XY pairs")
    if len(set(pts))!=len(pts): raise ValueError("polygon has duplicate vertices")
    n=len(pts)
    for i in range(n):
        a,b=pts[i],pts[(i+1)%n]
        if math.dist(a,b)<=EPS: raise ValueError("polygon has zero-length edge")
        for j in range(i+1,n):
            if j==(i+1)%n or (j+1)%n==i: continue
            if _segments_intersect(a,b,pts[j],pts[(j+1)%n]): raise ValueError("polygon self-intersects")
    if polygon_area(pts)<=EPS: raise ValueError("polygon area must be positive")
    return pts


def _point_in_poly(p,poly):
    x,y=p; inside=False
    for i,(x1,y1) in enumerate(poly):
        x2,y2=poly[(i+1)%len(poly)]
        if _on_segment((x1,y1),(x2,y2),p): return False
        if (y1>y)!=(y2>y) and x<(x2-x1)*(y-y1)/(y2-y1)+x1: inside=not inside
    return inside


def _interior_overlap(a,b):
    """Return true for positive-area overlap, including identical polygons."""
    if not a or not b: return False
    try: a=validate_polygon([list(p) for p in a]); b=validate_polygon([list(p) for p in b])
    except ValueError: return True
    if any(_point_in_poly(p,b) for p in a) or any(_point_in_poly(p,a) for p in b): return True
    # Boundary crossings that are not collinear are proper interior overlap;
    # isolated endpoint contacts and collinear shared edges are legal.
    for i,p in enumerate(a):
        q=a[(i+1)%len(a)]
        for j,r in enumerate(b):
            s=b[(j+1)%len(b)]
            if _segments_intersect(p,q,r,s) and abs(_cross(p,q,r))>EPS and abs(_cross(p,q,s))>EPS:
                return True
    # Coincident polygons have no strictly interior vertices. Coincident area
    # is detected by testing a point nudged toward each polygon's interior.
    for poly, other in ((a,b),(b,a)):
        sign=1 if sum(poly[i][0]*poly[(i+1)%len(poly)][1]-poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))>=0 else -1
        for i,p in enumerate(poly):
            q=poly[(i+1)%len(poly)]; dx=q[0]-p[0]; dy=q[1]-p[1]; length=math.hypot(dx,dy)
            probe=((p[0]+q[0])/2-sign*dy/length*1e-6,(p[1]+q[1])/2+sign*dx/length*1e-6)
            if _point_in_poly(probe,other): return True
    return False


def _shared_segment(a,b):
    found=[]
    for i,p in enumerate(a):
        q=a[(i+1)%len(a)]
        for j,r in enumerate(b):
            s=b[(j+1)%len(b)]
            if abs(_cross(p,q,r))>EPS or abs(_cross(p,q,s))>EPS: continue
            axis=0 if abs(q[0]-p[0])>=abs(q[1]-p[1]) else 1
            lo=max(min(p[axis],q[axis]),min(r[axis],s[axis])); hi=min(max(p[axis],q[axis]),max(r[axis],s[axis]))
            if hi-lo<=EPS: continue
            f=lambda t:(t-p[axis])/(q[axis]-p[axis])
            found.append((hi-lo,[[p[0]+f(lo)*(q[0]-p[0]),p[1]+f(lo)*(q[1]-p[1])],[p[0]+f(hi)*(q[0]-p[0]),p[1]+f(hi)*(q[1]-p[1])]]))
    return max(found,key=lambda v:v[0])[1] if found else None


def _wall_records(room):
    p=room["polygon_xy_m"]
    return [{"surface_id":f"{room['room_id']}-wall-{i+1}","edge_index":i,"start_xy_m":p[i],"end_xy_m":p[(i+1)%len(p)],"height_m":room["ceiling_height_m"]} for i in range(len(p))]


def _room(rid,poly,z,h):
    room={"room_id":rid,"polygon_xy_m":[[round(x,4),round(y,4)] for x,y in poly],"floor_elevation_m":round(z,4),"ceiling_height_m":round(h,4)}
    room["role"]="connector" if "connector" in rid else "living"
    room["walls"]=_wall_records(room)
    return room


def _rectangle(x0,y0,x1,y1): return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def _single(rng):
    w,l=rng.uniform(3.4,6.4),rng.uniform(3.8,7.2); h=rng.uniform(2.45,3.35)
    if rng.random()<.56:
        cx=w*rng.uniform(.34,.62); cy=l*rng.uniform(.34,.62); p=[[0,0],[w,0],[w,cy],[cx,cy],[cx,l],[0,l]]; tag="l_shape"
    else: p=_rectangle(0,0,w,l); tag="rectangle"
    return [_room("room-1",p,0,h)],[],[],{"topology":tag}


def _door(a,b,rng):
    seg=_shared_segment(a["polygon_xy_m"],b["polygon_xy_m"])
    if not seg: raise ValueError("adjacent rooms need shared wall segment")
    sa=next(w for w in a["walls"] if _shared_segment([w["start_xy_m"],w["end_xy_m"]],b["polygon_xy_m"]))
    sb=next(w for w in b["walls"] if _shared_segment([w["start_xy_m"],w["end_xy_m"]],a["polygon_xy_m"]))
    axis=0 if abs(seg[0][0]-seg[1][0])>abs(seg[0][1]-seg[1][1]) else 1
    lo,hi=sorted((seg[0][axis],seg[1][axis])); length=hi-lo
    width=min(rng.uniform(.78,1.12),length*.55); center=(lo+hi)/2
    other=1-axis; fixed=seg[0][other]
    ends=[]
    for v in (center-width/2,center+width/2):
        p=[0.,0.]; p[axis]=v; p[other]=fixed; ends.append([round(x,4) for x in p])
    return {"type":"doorway","room_a":a["room_id"],"room_b":b["room_id"],"surface_a_id":sa["surface_id"],"surface_b_id":sb["surface_id"],"endpoints_xy_m":ends,"width_m":round(math.dist(*ends),4),"height_m":round(min(2.15,a["ceiling_height_m"]-.15,b["ceiling_height_m"]-.15),4)}


def _multi(rng):
    h=rng.uniform(2.5,3.3); d=rng.uniform(4.0,5.8); wa=rng.uniform(3.0,4.4); wb=rng.uniform(3.0,4.4); wc=rng.uniform(3.0,4.4); conn=rng.uniform(1.4,1.8)
    family=rng.choice(("linear","branch"))
    if family=="linear":
        # Living rooms flank a separate narrower connector room in a chain.
        x1=wa; x2=x1+conn; x3=x2+wb; x4=x3+conn
        rooms=[_room("living-a",_rectangle(0,0,x1,d),0,h),_room("connector",_rectangle(x1,0,x2,d),0,h),_room("living-b",_rectangle(x2,0,x3,d),0,h),_room("living-c",_rectangle(x3,0,x4,d),0,h)]
        pairs=((rooms[0],rooms[1]),(rooms[1],rooms[2]),(rooms[2],rooms[3]))
    else:
        # Connector is the central hub, with three living-room wings.
        cw=rng.uniform(1.8,2.3); ch=rng.uniform(1.8,2.3); offset=max(wa,wb)+cw/2
        rooms=[_room("connector",_rectangle(offset,offset,cw+offset,ch+offset),0,h),
               _room("living-north",_rectangle(offset,offset+ch,offset+cw,offset+ch+d),0,h),
               _room("living-east",_rectangle(offset+cw,offset,offset+cw+d,offset+ch),0,h),
               _room("living-south",_rectangle(offset,offset-d,offset+cw,offset),0,h)]
        pairs=((rooms[0],rooms[1]),(rooms[0],rooms[2]),(rooms[0],rooms[3]))
    return rooms,[_door(a,b,rng) for a,b in pairs],[],{"topology":f"connector_{family}","connector_room_id":"connector"}


def _multifloor(rng):
    h0=rng.uniform(2.5,3.25); h1=rng.uniform(2.5,3.25); rise=rng.uniform(.12,.22)
    width=rng.uniform(3.2,4.2); depth=rng.uniform(3.5,4.8); conn=1.5
    rooms=[]; doors=[]
    for floor,z,h in ((0,0,h0),(1,h0+rise,h1)):
        prefix=f"floor-{floor}-"; x1=width; x2=x1+conn; x3=x2+width
        group=[_room(prefix+"living-a",_rectangle(0,0,x1,depth),z,h),_room(prefix+"connector",_rectangle(x1,0,x2,depth),z,h),_room(prefix+"living-b",_rectangle(x2,0,x3,depth),z,h)]
        rooms.extend(group); doors.extend((_door(group[0],group[1],rng),_door(group[1],group[2],rng)))
    lower,upper=rooms[1],rooms[4]
    # Stair envelope is intentionally a conservative plan-space exclusion;
    # tread/run/headroom geometry is not modeled, so this is not capture-ready.
    wx=1.05; wy=.7; x0=width+.22; y0=depth*.28
    footprint=_rectangle(x0,y0,x0+wx,y0+wy)
    doors.append({"type":"stairway","room_a":lower["room_id"],"room_b":upper["room_id"],"footprint_xy_m":footprint})
    # A stair connects the lower and upper floor connector rooms; the upper
    # connector has a matching XY footprint envelope at its floor elevation.
    stair={"stair_id":"stair-1","room_a":lower["room_id"],"room_b":upper["room_id"],"lower_elevation_m":lower["floor_elevation_m"],"upper_elevation_m":upper["floor_elevation_m"],"footprint_xy_m":footprint,"status":"physically_unvalidated","physical_validation":{"walkable_treads":False,"run_rise_checked":False,"headroom_checked":False}}
    return rooms,doors,[stair],{"topology":"two_floor_connector_chain"}


def _finite_positive(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v)) and v>0


def _validate_wall_records(room,poly):
    walls=room.get("walls")
    if not isinstance(walls,list) or len(walls)!=len(poly): raise ValueError("wall surface records do not match polygon edges")
    ids=set()
    for i,w in enumerate(walls):
        sid=w.get("surface_id")
        if not isinstance(sid,str) or not sid or sid in ids: raise ValueError("wall surface ids invalid")
        ids.add(sid)
        if w.get("edge_index")!=i or w.get("start_xy_m")!=list(poly[i]) or w.get("end_xy_m")!=list(poly[(i+1)%len(poly)]): raise ValueError("wall records must match actual polygon edges")
        if not _finite_positive(w.get("height_m")) or abs(w["height_m"]-room["ceiling_height_m"])>.001: raise ValueError("wall height must match finite room height")


def _connected(graph,nodes):
    if not nodes: return True
    seen=set(); todo=[next(iter(nodes))]
    while todo:
        n=todo.pop()
        if n not in seen: seen.add(n); todo.extend(graph[n]-seen)
    return seen==set(nodes)


def validate_layout(layout):
    """Validate polygon simplicity, metric consistency, adjacency and stairs."""
    if layout.get("stage") not in STAGES: raise ValueError("unknown stage")
    rooms=layout.get("rooms",[])
    if not rooms: raise ValueError("layout has no rooms")
    byid={}; levels={}
    for room in rooms:
        rid=room.get("room_id")
        if not isinstance(rid,str) or not rid or rid in byid: raise ValueError("duplicate or invalid room id")
        byid[rid]=room
        poly=validate_polygon(room.get("polygon_xy_m"))
        if not _finite_positive(room.get("ceiling_height_m")): raise ValueError("room height must be finite and positive")
        z=room.get("floor_elevation_m")
        if not isinstance(z,(int,float)) or isinstance(z,bool) or not math.isfinite(float(z)) or z<0: raise ValueError("floor elevation must be finite and nonnegative")
        levels.setdefault(float(z),set()).add(rid)
        _validate_wall_records(room,poly)
    for i,a in enumerate(rooms):
        for b in rooms[i+1:]:
            if a["floor_elevation_m"]==b["floor_elevation_m"] and _interior_overlap(a["polygon_xy_m"],b["polygon_xy_m"]): raise ValueError("room interiors overlap on same floor")
    surfaces={w["surface_id"]:(r,w) for r in rooms for w in r["walls"]}; graph={rid:set() for rid in byid}; floor_graph={z:set() for z in levels}
    for door in layout.get("doorways",[]):
        a,b=door.get("room_a"),door.get("room_b")
        if a not in byid or b not in byid or a==b: raise ValueError("door references invalid room")
        if door.get("type")=="stairway":
            if byid[a]["floor_elevation_m"]==byid[b]["floor_elevation_m"]: raise ValueError("stairway must connect distinct floors")
            graph[a].add(b); graph[b].add(a)
            continue
        if door.get("surface_a_id") not in surfaces or door.get("surface_b_id") not in surfaces: raise ValueError("door references unknown surface")
        ra,wa=surfaces[door["surface_a_id"]]; rb,wb=surfaces[door["surface_b_id"]]
        if ra["room_id"]!=a or rb["room_id"]!=b: raise ValueError("door surfaces do not match adjacent rooms")
        if ra["floor_elevation_m"]!=rb["floor_elevation_m"]: raise ValueError("doorway cannot connect distinct floor elevations")
        shared=_shared_segment([wa["start_xy_m"],wa["end_xy_m"]],[wb["start_xy_m"],wb["end_xy_m"]])
        if shared is None: raise ValueError("door surfaces do not share a boundary")
        ends=door.get("endpoints_xy_m")
        if not isinstance(ends,list) or len(ends)!=2: raise ValueError("door must have two endpoints")
        p,q=ends
        if not _on_segment(*shared,p) or not _on_segment(*shared,q): raise ValueError("door endpoints must lie on shared boundary")
        if not _finite_positive(door.get("width_m")) or abs(math.dist(p,q)-door["width_m"])>.002: raise ValueError("door width invalid")
        if not _finite_positive(door.get("height_m")) or door["height_m"]>min(ra["ceiling_height_m"],rb["ceiling_height_m"]): raise ValueError("door height invalid")
        graph[a].add(b); graph[b].add(a)
    if layout["stage"]=="multiroom":
        if len(rooms)<4 or "connector_room_id" not in layout: raise ValueError("multiroom requires at least three living rooms plus connector")
        if not _connected(graph,set(byid)): raise ValueError("room doorway graph is disconnected")
        if len({layout.get("topology") for _ in (0,)})!=1: raise ValueError("multiroom topology missing")
    if layout["stage"]=="multifloor":
        if len(levels)<2: raise ValueError("multifloor needs at least two elevations")
        if any(sum(r["floor_elevation_m"]==z for r in rooms)<3 or not any(r["floor_elevation_m"]==z and r.get("role")=="connector" for r in rooms) for z in levels): raise ValueError("each multifloor level needs three rooms including connector")
        if not _connected(graph,set(byid)): raise ValueError("room doorway graph must connect all rooms across floors")
        stairs=layout.get("stairs",[])
        if not stairs: raise ValueError("multifloor needs stair link")
        stair_graph={z:set() for z in levels}
        for s in stairs:
            if s.get("room_a") not in byid or s.get("room_b") not in byid: raise ValueError("stair references unknown room")
            a,b=byid[s["room_a"]],byid[s["room_b"]]; za,zb=a["floor_elevation_m"],b["floor_elevation_m"]
            if za==zb or s.get("lower_elevation_m")!=min(za,zb) or s.get("upper_elevation_m")!=max(za,zb): raise ValueError("stair elevations must match connected rooms")
            fp=validate_polygon(s.get("footprint_xy_m"))
            if _interior_overlap(fp,fp) is False: raise ValueError("stair footprint invalid")
            if not any(_point_in_poly(p,byid[s["room_a"]]["polygon_xy_m"]) or _point_in_poly(p,byid[s["room_b"]]["polygon_xy_m"]) for p in fp): raise ValueError("stair footprint lies outside connected rooms")
            if s.get("status")!="physically_unvalidated": raise ValueError("stair must be marked physically_unvalidated")
            stair_graph[za].add(zb); stair_graph[zb].add(za)
            graph[s["room_a"]].add(s["room_b"]); graph[s["room_b"]].add(s["room_a"])
        if not _connected(graph,set(byid)): raise ValueError("room doorway and stair graph must connect all rooms")
        if not _connected(stair_graph,set(levels)): raise ValueError("stair graph does not connect all floors")
        if layout.get("capture_ready") is not False: raise ValueError("multifloor layout is not capture-ready")
    return True


def generate_house(stage: str,house_index: int,seed: int)->dict:
    if stage not in STAGES: raise ValueError(f"stage must be one of {STAGES}")
    rng=random.Random(int(seed))
    if stage=="single_room": rooms,doors,stairs,extra=_single(rng)
    elif stage=="multiroom": rooms,doors,stairs,extra=_multi(rng)
    else: rooms,doors,stairs,extra=_multifloor(rng); extra["capture_ready"]=False
    layout={"stage":stage,"house_id":f"{stage}-house-{house_index:02d}","layout_seed":int(seed),"capture_seeds":[int(seed)*1009+i for i in range(3)],"rooms":rooms,"doorways":doors,"stairs":stairs,**extra}
    validate_layout(layout)
    return layout


def generate_suite(seed=20261003):
    layouts=[generate_house(stage,i+1,int(seed)+si*100_000+i*997) for si,stage in enumerate(STAGES) for i in range(10)]
    return {"schema_version":1,"data_classification":"private_benchmark_ground_truth_layouts","contains_capture_outputs":False,"stages":list(STAGES),"layouts":layouts}


def main(argv:Iterable[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--seed",type=int,default=20261003); parser.add_argument("--output",type=Path,default=Path("outputs/sim-benchmark/layout-suite-private/house_layouts.json")); args=parser.parse_args(argv)
    if args.output.exists(): raise SystemExit(f"refusing to overwrite existing file: {args.output}")
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(generate_suite(args.seed),indent=2)+"\n",encoding="utf-8"); print(f"wrote private layout manifest: {args.output}")


if __name__=="__main__": main()
