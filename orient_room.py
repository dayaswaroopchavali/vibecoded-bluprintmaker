"""Orient unscaled room hypotheses using an explicit, declared camera-motion up prior.

Inputs are a frozen RGB-derived box hypothesis and DA3 world-to-camera poses.
The camera-up prior is not an IMU measurement and can be wrong for rotated or
landscape captures. Ambiguous cases are refused rather than guessed. Openings
are deliberately left unknown; no simulator annotations or dimensions are read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n=float(np.linalg.norm(v))
    return v/n if n > 1e-10 else np.zeros(3,dtype=float)


def _camera_up(extrinsics: np.ndarray, min_poses: int=3) -> tuple[np.ndarray,dict,np.ndarray]:
    poses=np.asarray(extrinsics,dtype=float)
    if poses.ndim!=3 or poses.shape[1:] not in ((3,4),(4,4)):
        raise ValueError("extrinsics must have shape Nx3x4 or Nx4x4")
    if poses.shape[1:]==(4,4): poses=poses[:,:3,:]
    rotations=poses[:,:,:3]; translations=poses[:,:,3]
    finite=np.isfinite(poses).all(axis=(1,2))
    orth_err=np.full(len(poses),np.inf)
    for i,R in enumerate(rotations):
        if finite[i]: orth_err[i]=np.linalg.norm(R@R.T-np.eye(3),ord="fro")
    valid=finite&(orth_err<.12)
    if int(valid.sum())<min_poses:
        return np.zeros(3),{"valid_pose_count":int(valid.sum()),"input_pose_count":len(poses),
            "status":"insufficient_valid_camera_poses"},np.empty((0,3))
    R=rotations[valid]; t=translations[valid]
    ups=-R[:,1,:]
    ups/=np.maximum(np.linalg.norm(ups,axis=1,keepdims=True),1e-12)
    # A component median supplies a robust signed center for upright camera-up vectors.
    med=np.median(ups,axis=0); med=_unit(med)
    angles=np.degrees(np.arccos(np.clip(ups@med,-1.,1.)))
    med_angle=float(np.median(angles)); mad=float(np.median(np.abs(angles-med_angle)))
    cutoff=min(48.,max(22.,med_angle+3*1.4826*mad))
    inliers=angles<=cutoff
    fraction=float(inliers.mean())
    resultant=float(np.linalg.norm(ups[inliers].mean(axis=0))) if inliers.any() else 0.
    mean=_unit(ups[inliers].mean(axis=0)) if inliers.any() else np.zeros(3)
    consensus=bool(inliers.sum()>=min_poses and fraction>=.60 and resultant>=.72)
    centers=-np.einsum("nji,nj->ni",R,t)
    diagnostics={"status":"strong_consensus" if consensus else "weak_or_inconsistent_consensus",
        "prior":"OpenCV upright-phone camera up = -R_world_to_camera[1,:]",
        "valid_pose_count":int(valid.sum()),"input_pose_count":int(len(poses)),
        "inlier_pose_count":int(inliers.sum()),"inlier_fraction":fraction,
        "robust_resultant_length":resultant,"median_angular_deviation_degrees":med_angle,
        "p90_angular_deviation_degrees":float(np.percentile(angles,90)),
        "inlier_cutoff_degrees":cutoff,"rotation_orthogonality_error_median":float(np.median(orth_err[valid])),
        "assumption_is_imu_measurement":False}
    return mean,diagnostics,centers


TRAJECTORY_PLANARITY_RATIO = 0.5     # motion plane check applies when the path is this flat
TRAJECTORY_AGREEMENT_DEGREES = 25.0  # camera-up and motion-plane normal must agree this closely


def camera_up_prior(extrinsics: np.ndarray) -> dict:
    """Up direction inferred from RGB-derived camera motion, declared as a prior and nothing more.

    Camera-up consensus from the poses (upright-phone assumption) gives the signed direction;
    the principal plane of the camera path, when the path is flat enough, must agree with it.
    No scene geometry, simulator data, or ground-truth axis is consulted; the result is never
    evidence that orientation was "recovered" and a wrong prior (landscape or rolled capture,
    vertical-sweep path) is refused only when the two cues disagree.
    """
    base = {"declared_prior": True, "prior_kind": "camera_motion_upright_phone",
            "independent_of_scene_geometry": True, "oracle_axis_matching_used": False, "up_world": None}
    try:
        up, camdiag, centers = _camera_up(extrinsics)
    except (TypeError, ValueError) as exc:
        return {**base, "status": "refused", "reason": str(exc), "camera_up_diagnostics": {"status": "invalid_extrinsics"}}
    if camdiag.get("status") != "strong_consensus":
        return {**base, "status": "refused", "reason": "camera-up consensus is not strong", "camera_up_diagnostics": camdiag}
    trajectory = {"status": "not_checked_too_few_centers"}
    finite = centers[np.isfinite(centers).all(axis=1)] if len(centers) else centers
    if len(finite) >= 6:
        singular = np.linalg.svd(finite - finite.mean(axis=0), full_matrices=False)
        values, normal = singular[1], singular[2][-1]
        ratio = float(values[2] / max(values[1], 1e-12))
        if ratio < TRAJECTORY_PLANARITY_RATIO:
            angle = float(np.degrees(np.arccos(np.clip(abs(normal @ up), 0., 1.))))
            trajectory = {"status": "checked", "plane_normal_world": normal.tolist(), "flatness_ratio": ratio,
                          "angle_to_camera_up_degrees": angle}
            if angle > TRAJECTORY_AGREEMENT_DEGREES:
                return {**base, "status": "refused", "reason": "camera path plane disagrees with camera-up consensus",
                        "camera_up_diagnostics": camdiag, "trajectory_check": trajectory}
        else:
            trajectory = {"status": "not_checked_path_not_planar", "flatness_ratio": ratio}
    return {**base, "status": "declared_prior", "up_world": up.tolist(), "camera_up_diagnostics": camdiag,
            "trajectory_check": trajectory,
            "limitations": ["Not an IMU measurement; wrong for rotated, landscape, or strongly rolled captures.",
                            "Agreement of two camera-derived cues is not independent validation of the room axes."]}


def orient_polygon(polygon: dict, extrinsics: np.ndarray) -> dict:
    """Attach the declared up prior to a floor-plan polygon hypothesis; refuse anything incomplete."""
    from room_hypothesis import _inside
    prior = camera_up_prior(extrinsics)
    base = {"model": "upright_prior_oriented_polygon_floorplan", "units": "arbitrary", "scale_applied": False,
            "metric_scale_source": None, "up_source": "camera_motion_prior", "up_prior": prior,
            "floorplan": None, "walls": [], "height_unscaled": None,
            "height_status": "unavailable_not_estimated", "openings": None,
            "wall_id_semantics": "arbitrary_polygon_order_not_matched_to_any_ground_truth_label",
            "limitations": ["Up is a declared camera-motion prior, not an IMU measurement.",
                            "Wall ids follow polygon order; correspondence to named walls is not solved.",
                            "No ceiling height, openings, or metric scale."]}
    def refuse(reason, **extra):
        return {**base, "status": "orientation_refused", "reason": reason, **extra}
    if prior["status"] != "declared_prior":
        return refuse("camera-motion up prior unavailable: " + prior["reason"])
    if polygon.get("status") != "complete_polygon_hypothesis" or not polygon.get("vertices_xy") or not polygon.get("edges"):
        return refuse("polygon hypothesis is incomplete", polygon_status=polygon.get("status"),
                      incomplete_boundaries=polygon.get("incomplete_boundaries"))
    try:
        axes = polygon["axes_world"]
        declared_up = _unit(np.asarray(axes["up"], dtype=float)); origin = np.asarray(polygon["coordinate_origin_world"], dtype=float)
        e1, e2 = np.asarray(axes["e1"], dtype=float), np.asarray(axes["e2"], dtype=float)
        vertices = np.asarray(polygon["vertices_xy"], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all() or origin.shape != (3,):
            raise ValueError("invalid polygon geometry")
    except (KeyError, TypeError, ValueError) as exc:
        return refuse(f"polygon hypothesis lacks its coordinate frame: {exc}")
    agreement = float(np.degrees(np.arccos(np.clip(declared_up @ np.asarray(prior["up_world"]), -1., 1.))))
    if agreement > 10.0:
        return refuse("polygon was built with a different up direction than the camera-motion prior",
                      up_disagreement_degrees=agreement)
    poses = np.asarray(extrinsics, dtype=float)
    if poses.shape[1:] == (4, 4):
        poses = poses[:, :3, :]
    centers = -np.einsum("nji,nj->ni", poses[:, :, :3], poses[:, :, 3])
    centers = centers[np.isfinite(centers).all(axis=1)]
    xy = np.stack([(centers - origin) @ e1, (centers - origin) @ e2], axis=1) if len(centers) else np.empty((0, 2))
    inside = float(_inside(vertices, xy).mean()) if len(xy) else 0.0
    if len(xy) < 3 or inside < 0.7:
        return refuse("camera centers do not lie mostly inside the floor-plan polygon", camera_inside_fraction=inside)
    walls = [{"wall_id": f"wall_{i + 1}", "edge_index": int(e["index"]), "length_unscaled": float(e["length"]),
              "boundary_label": e["label"], "observed_fraction": float(e["observed_fraction"])}
             for i, e in enumerate(polygon["edges"])]
    return {**base, "status": "oriented_unscaled_polygon_floorplan", "up_agreement_degrees": agreement,
            "camera_inside_fraction": inside, "walls": walls, "wall_lengths_unscaled": [w["length_unscaled"] for w in walls],
            "inferred_edge_count": int(sum(w["boundary_label"] == "inferred" for w in walls)),
            "floorplan": {"coordinate_system": "polygon_local_xy_unscaled", "units": "arbitrary",
                          "polygon_2d": vertices.tolist(), "origin_world": origin.tolist(),
                          "axis_a_world": e1.tolist(), "axis_b_world": e2.tolist(), "up_world": declared_up.tolist()}}


def orient(hypothesis: dict, extrinsics: np.ndarray) -> dict:
    """Select vertical axis and return an unscaled local 2D box floor plan."""
    base={"model":"upright_prior_oriented_single_room_box","units":"arbitrary","scale_applied":False,
        "metric_scale_source":None,"openings":None,"opening_status":"unknown_not_estimated",
        "limitations":["Vertical orientation uses an upright-phone camera-up prior, not an IMU.",
            "Camera-up can be wrong for rotated, landscape, or strongly rolled captures.",
            "Unscaled box dimensions are geometric hypotheses, not measured room dimensions.",
            "Openings are not detected; no door or window measurements are claimed."]}
    try:
        up,camdiag,centers=_camera_up(extrinsics)
    except (TypeError,ValueError) as exc:
        return {**base,"status":"orientation_refused","reason":str(exc),"floorplan":None,
            "camera_up_diagnostics":{"status":"invalid_extrinsics"}}
    if camdiag.get("status")!="strong_consensus":
        return {**base,"status":"orientation_refused","reason":"camera-up consensus is not strong",
            "camera_up_diagnostics":camdiag,"floorplan":None,"shape_dimensions":hypothesis.get("dimensions"),
            "height_axis_index":None,"complete_surface_pairs":_complete_pairs(hypothesis)}
    if hypothesis.get("status")!="complete_box_hypothesis" or not isinstance(hypothesis.get("axes"),list) or len(hypothesis["axes"])!=3:
        return {**base,"status":"orientation_refused","reason":"input is not a complete three-axis box hypothesis",
            "camera_up_diagnostics":camdiag,"floorplan":None,"shape_dimensions":hypothesis.get("dimensions"),
            "height_axis_index":None,"complete_surface_pairs":_complete_pairs(hypothesis)}
    try:
        origin=np.asarray(hypothesis["coordinate_origin_world"],dtype=float)
        axes=np.asarray([a["direction_world"] for a in hypothesis["axes"]],dtype=float)
        dims=np.asarray(hypothesis["dimensions"],dtype=float)
        pairs=[a["surface_pair"] for a in hypothesis["axes"]]
        if origin.shape!=(3,) or axes.shape!=(3,3) or dims.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(axes).all() or not np.isfinite(dims).all() or np.any(dims<=0):
            raise ValueError("invalid box origin, directions, or dimensions")
        axes=np.asarray([_unit(a) for a in axes])
        if not all(p.get("complete") and np.isfinite([p["low"],p["high"]]).all() and p["high"]>p["low"] for p in pairs):
            raise ValueError("box does not have three complete opposing surface pairs")
    except (KeyError,TypeError,ValueError) as exc:
        return {**base,"status":"orientation_refused","reason":str(exc),"camera_up_diagnostics":camdiag,
            "floorplan":None,"shape_dimensions":hypothesis.get("dimensions"),"height_axis_index":None,
            "complete_surface_pairs":_complete_pairs(hypothesis)}
    scores=np.abs(axes@up); order=np.argsort(scores)[::-1]
    best,second=int(order[0]),float(scores[order[1]])
    gap=float(scores[best]-second)
    selection={"camera_up_world":up.tolist(),"axis_alignment_abs_cosines":scores.tolist(),
        "selected_raw_height_axis_index":best,"best_alignment":float(scores[best]),
        "runner_up_alignment":second,"alignment_gap":gap}
    if scores[best]<.80 or gap<.16:
        return {**base,"status":"orientation_refused","reason":"no unique box axis aligns strongly with camera-up consensus",
            "camera_up_diagnostics":camdiag,"axis_selection_diagnostics":selection,"floorplan":None,
            "shape_dimensions":dims.tolist(),"height_axis_index":None,"complete_surface_pairs":3}

    # Orient the selected box direction upward so its lower supported plane is the floor.
    vertical=axes[best].copy(); vp=pairs[best]; vlo,vhi=float(vp["low"]),float(vp["high"])
    if float(vertical@up)<0:
        vertical=-vertical; vlo,vhi=-vhi,-vlo
    span=vhi-vlo
    center_coord=(centers-origin)@vertical
    finite=np.isfinite(center_coord)
    frac=(center_coord[finite]-vlo)/span
    interior=(frac>=.05)&(frac<=.95)
    above_floor=frac>.05
    inside_fraction=float(interior.mean()) if len(frac) else 0.
    above_fraction=float(above_floor.mean()) if len(frac) else 0.
    median_height=float(np.median(frac)) if len(frac) else float("nan")
    camera_checks={"valid_camera_center_count":int(len(frac)),"between_floor_ceiling_fraction":inside_fraction,
        "above_floor_fraction":above_fraction,"median_camera_height_fraction":median_height,
        "floor_plane_coordinate":vlo,"ceiling_plane_coordinate":vhi}
    if len(frac)<3 or inside_fraction<.70 or above_fraction<.75 or not (.05<median_height<.95):
        return {**base,"status":"orientation_refused","reason":"predicted camera centers do not mostly lie above the supported floor and below the ceiling",
            "camera_up_diagnostics":camdiag,"axis_selection_diagnostics":selection,
            "camera_center_plane_check":camera_checks,"floorplan":None,
            "shape_dimensions":dims.tolist(),"height_axis_index":None,"complete_surface_pairs":3}

    horizontal=[j for j in range(3) if j!=best]
    a,b=horizontal
    axis_a,axis_b=axes[a].copy(),axes[b].copy()
    len_a=float(pairs[a]["high"]-pairs[a]["low"]); len_b=float(pairs[b]["high"]-pairs[b]["low"])
    floor_origin=origin+sum(float(p["low"])*ax for ax,p,ax in zip(pairs,range(3),axes)) if False else origin.copy()
    # Coordinates in the hypothesis use pair offsets from coordinate_origin_world.
    for j in range(3):
        if j==best:
            floor_origin += vlo*vertical
        else:
            floor_origin += float(pairs[j]["low"])*axes[j]
    poly=[[0.,0.],[len_a,0.],[len_a,len_b],[0.,len_b]]
    world_poly=[(floor_origin+u*axis_a+v*axis_b).tolist() for u,v in poly]
    wall_lengths=[len_a,len_b,len_a,len_b]
    plan={"coordinate_system":"local_unscaled_floor_axes","units":"arbitrary",
        "origin_world":floor_origin.tolist(),"axis_a_world":axis_a.tolist(),"axis_b_world":axis_b.tolist(),
        "polygon_2d":poly,"polygon_world":world_poly,
        "walls":[{"wall_id":f"wall_{i+1}","length_unscaled":length} for i,length in enumerate(wall_lengths)],
        "height_unscaled":span,"openings":None}
    ordered_dims=[len_a,len_b,span]
    return {**base,"status":"oriented_unscaled_box_floorplan","up_source":"camera_motion_prior","complete_surface_pairs":3,
        "shape_dimensions":ordered_dims,"height_axis_index":2,
        "raw_hypothesis_dimensions":dims.tolist(),"raw_height_axis_index":best,
        "wall_lengths_unscaled":wall_lengths,"height_unscaled":span,
        "camera_up_diagnostics":camdiag,"axis_selection_diagnostics":selection,
        "camera_center_plane_check":camera_checks,"floorplan":plan}


def _complete_pairs(hypothesis: dict) -> int:
    axes=hypothesis.get("axes",[])
    return sum(bool(a.get("surface_pair",{}).get("complete")) for a in axes) if isinstance(axes,list) else 0


def write_svg(result: dict, path: Path) -> None:
    plan=result.get("floorplan")
    if not plan:
        raise ValueError("cannot draw an SVG plan when orientation was refused")
    a,b=result["shape_dimensions"][:2]; pad=48.; scale=360/max(a,b)
    width=2*pad+a*scale; height=2*pad+b*scale
    points=[(pad+u*scale,pad+v*scale) for u,v in plan["polygon_2d"]]
    pts=" ".join(f"{x:.2f},{height-y:.2f}" for x,y in points)
    mid_a=(pad+a*scale/2,height-pad+18); mid_b=(pad-24,height-pad-b*scale/2)
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.2f} {height:.2f}">'
         '<rect width="100%" height="100%" fill="#fafafa"/>'
         f'<polygon points="{pts}" fill="#e8eef4" stroke="#20334a" stroke-width="4"/>'
         f'<text x="{mid_a[0]:.2f}" y="{mid_a[1]:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">wall length {a:.3f} (unscaled)</text>'
         f'<text x="{mid_b[0]:.2f}" y="{mid_b[1]:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 {mid_b[0]:.2f} {mid_b[1]:.2f})">wall length {b:.3f} (unscaled)</text>'
         f'<text x="{pad:.2f}" y="{height-12:.2f}" font-family="sans-serif" font-size="12">Floor plan orientation estimated from upright-phone prior; openings unknown</text></svg>')
    path.write_text(svg,encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis",type=Path,help="box or floor-plan polygon hypothesis JSON")
    parser.add_argument("--prediction",required=True,type=Path,help="DA3 RGB-only prediction.npz containing extrinsics")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--prior-only",action="store_true",help="write only the camera-motion up prior (feed it to room_hypothesis --up)")
    parser.add_argument("--svg",type=Path,help="optional SVG plan path; defaults beside output when a box orientation succeeds")
    args=parser.parse_args()
    if args.output.exists(): parser.error(f"output already exists: {args.output}")
    if not args.prior_only and args.hypothesis is None: parser.error("--hypothesis is required unless --prior-only")
    try:
        with np.load(args.prediction) as data: extrinsics=np.asarray(data["extrinsics"])
        if args.prior_only:
            result=camera_up_prior(extrinsics)
        else:
            hypothesis=json.loads(args.hypothesis.read_text(encoding="utf-8"))
            result=orient_polygon(hypothesis,extrinsics) if hypothesis.get("model")=="floor_plan_polygon_hypothesis" else orient(hypothesis,extrinsics)
    except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
        parser.error(f"could not load orientation inputs: {exc}")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    plan=result.get("floorplan")
    if plan is not None and "polygon_2d" in plan and result.get("shape_dimensions") is not None:
        write_svg(result,args.svg or args.output.with_suffix(".svg"))
    print(args.output)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
