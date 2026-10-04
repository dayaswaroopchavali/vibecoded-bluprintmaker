"""Public-only opening candidates from wall-space depth occupancy ratios.

Unlike the depth-vote detector, this partitions each wall into a 2-D metric
grid and segments cells by the fraction of *observed wall-or-dropout evidence*
that is missing depth. It does not bridge gaps morphologically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def detect(prediction: Path, boundary_path: Path, axes_path: Path,
           planes_path: Path, *, grid_m: float = .01,
           min_frames: int = 3, dropout_fraction: float = .65) -> dict:
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    empty = {"schema": "occupancy_wall_openings_v1", "openings": []}
    if boundary.get("status") != "complete_polygon_hypothesis":
        return {**empty, "status": "incomplete_boundary"}
    axes_doc = json.loads(axes_path.read_text(encoding="utf-8"))
    planes_doc = json.loads(planes_path.read_text(encoding="utf-8"))
    up = np.asarray(boundary["axes_world"]["up"], float)
    matches = [h for h in axes_doc["hypotheses"]
               if abs(np.dot(up, np.asarray(h["axis_world"], float))) > .99]
    if len(matches) != 1 or not matches[0].get("bracketing_pair"):
        return {**empty, "status": "ambiguous_height_pair"}
    caps = [np.asarray(planes_doc["planes"][int(i)]["center"], float)
            for i in matches[0]["bracketing_pair"]["plane_ids"]]
    origin = np.asarray(boundary["coordinate_origin_world"], float)
    e1, e2 = (np.asarray(boundary["axes_world"][k], float) for k in ("e1", "e2"))
    verts = np.asarray(boundary["vertices_xy"], float)
    walls = []
    for wi in range(len(verts)):
        a = origin + verts[wi, 0] * e1 + verts[wi, 1] * e2
        b = origin + verts[(wi + 1) % len(verts), 0] * e1 + verts[(wi + 1) % len(verts), 1] * e2
        tangent = (b - a) / np.linalg.norm(b - a)
        normal = np.cross(tangent, up); normal /= np.linalg.norm(normal)
        cv = sorted(float(np.dot(c - a, up)) for c in caps)
        shape = (int(np.ceil((cv[1] - cv[0]) / grid_m)) + 1,
                 int(np.ceil(np.linalg.norm(b - a) / grid_m)) + 1)
        walls.append(dict(a=a, tangent=tangent, normal=normal, length=float(np.linalg.norm(b-a)),
                          vlo=cv[0], vhi=cv[1], shape=shape,
                          dropout=np.zeros(shape, np.uint16), observed=np.zeros(shape, np.uint16),
                          frames=np.zeros(shape, np.uint16),
                          umin=np.full(shape, np.inf), umax=np.full(shape, -np.inf),
                          vmin=np.full(shape, np.inf), vmax=np.full(shape, -np.inf)))
    with np.load(prediction) as data:
        depth, extr, intr = (data[k].copy() for k in ("depth", "extrinsics", "intrinsics"))
    n, height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]; xx, yy = xx.ravel(), yy.ravel()
    for fi in range(n):
        K = intr[fi]
        rc = np.stack(((xx-K[0,2])/K[0,0], (yy-K[1,2])/K[1,1], np.ones(len(xx))), axis=1)
        R, t = extr[fi, :, :3], extr[fi, :, 3]
        center = -R.T @ t; rays = rc @ R
        nearest = np.full(len(xx), np.inf); which = np.full(len(xx), -1, np.int16)
        us = np.zeros(len(xx)); vs = np.zeros(len(xx))
        for wi, wall in enumerate(walls):
            den = rays @ wall["normal"]
            z = np.divide(np.dot(wall["a"]-center, wall["normal"]), den,
                          out=np.full(len(xx), np.inf), where=np.abs(den)>1e-8)
            hit = center + rays*z[:,None]; rel = hit-wall["a"]
            u, v = rel@wall["tangent"], rel@up
            good = (z>.02)&(z<nearest)&(u>=0)&(u<=wall["length"])&(v>=wall["vlo"])&(v<=wall["vhi"])
            nearest[good], which[good], us[good], vs[good] = z[good], wi, u[good], v[good]
        obs = depth[fi].ravel()
        for wi, wall in enumerate(walls):
            sel = which == wi
            if not np.any(sel): continue
            u, v, d, z = us[sel], vs[sel], obs[sel], nearest[sel]
            ui = np.clip((u/grid_m).astype(int), 0, wall["shape"][1]-1)
            vi = np.clip(((v-wall["vlo"])/grid_m).astype(int), 0, wall["shape"][0]-1)
            ids = vi*wall["shape"][1]+ui
            dropout = (~np.isfinite(d)) | (d<=0)
            hitwall = np.isfinite(d)&(d>0)&(np.abs(d-z)<.035)
            evidence = dropout|hitwall
            wall["observed"] += np.bincount(ids[evidence], minlength=wall["observed"].size).reshape(wall["shape"]).astype(np.uint16)
            wall["dropout"] += np.bincount(ids[dropout], minlength=wall["dropout"].size).reshape(wall["shape"]).astype(np.uint16)
            active = np.unique(ids[evidence]); wall["frames"].flat[active] += 1
            dh = dropout
            if np.any(dh):
                did = ids[dh]
                for key, values, op in (("umin",u[dh],np.minimum.at),("umax",u[dh],np.maximum.at),
                                        ("vmin",v[dh],np.minimum.at),("vmax",v[dh],np.maximum.at)):
                    op(wall[key].ravel(), did, values)
    openings, diagnostics = [], []
    for wi, wall in enumerate(walls):
        fraction = np.divide(wall["dropout"], wall["observed"],
                             out=np.zeros(wall["shape"], float), where=wall["observed"]>0)
        mask = ((wall["frames"]>=min_frames)&(fraction>=dropout_fraction)).astype(np.uint8)
        # Label only positive occupancy cells. A small Python flood fill keeps
        # this detector dependent on NumPy alone and avoids morphology closing.
        remaining = set(map(tuple, np.argwhere(mask)))
        components = []
        while remaining:
            seed = remaining.pop(); stack = [seed]; cells = [seed]
            while stack:
                yy, xx = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = (yy + dy, xx + dx)
                        if q in remaining:
                            remaining.remove(q); stack.append(q); cells.append(q)
            components.append(cells)
        nc = 0
        for cells in components:
            coords = np.asarray(cells, dtype=int); ys, xs = coords[:, 0], coords[:, 1]
            y, x = int(ys.min()), int(xs.min()); h, w = int(ys.max()-y+1), int(xs.max()-x+1)
            area = len(cells)
            if area*grid_m**2 < .08 or w*grid_m < .2 or h*grid_m < .2: continue
            cm = np.zeros(wall["shape"], bool); cm[ys, xs] = True
            ok=cm&np.isfinite(wall["umin"])&np.isfinite(wall["umax"])
            if not ok.any(): continue
            lo,hi=float(wall["umin"][ok].min()),float(wall["umax"][ok].max())
            vlo,vhi=float(wall["vmin"][ok].min()),float(wall["vmax"][ok].max())
            openings.append({"wall_index":wi,"u_interval_m":[lo,hi],
                "v_interval_from_room_axis_m":[vlo,vhi],"width_m":hi-lo,"height_m":vhi-vlo,
                "supporting_frame_count_at_best_cell":int(wall["frames"][cm].max()),
                "type":"unknown_without_floor_sign_or_semantics",
                "measurement_interval_m":[max(0.,hi-lo-2*grid_m),hi-lo+2*grid_m]})
            nc+=1
        diagnostics.append({"wall_index":wi,"supported_cells":int((wall["frames"]>=min_frames).sum()),
                            "candidate_cells":int(mask.sum()),"component_count":nc})
    return {"schema":"occupancy_wall_openings_v1","status":"candidates","grid_m":grid_m,
            "min_frames":min_frames,"dropout_fraction":dropout_fraction,"input_frames":n,
            "openings":openings,"diagnostics":diagnostics,"truth_free":True,"ground_truth_read":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("prediction","boundary","axes","planes","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(); result=detect(a.prediction,a.boundary,a.axes,a.planes)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"openings":result["openings"]},indent=2))

if __name__=="__main__": main()
