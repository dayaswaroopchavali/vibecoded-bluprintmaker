"""Visibility evidence for finite candidate wall patches; never a room-completion gate.

Depth is a prediction, so free-space contradictions are evidence, not sensor truth.
No gravity, camera-upright, simulator, or ground-truth input is used.
"""
import numpy as np


def audit_prediction(prediction_path, planes_path, output_path, stride=8, tolerance_fraction=.01):
    """Score extracted patches using predicted rays; write reproducible diagnostic."""
    import hashlib
    import json
    from pathlib import Path
    prediction_path, planes_path, output_path = map(Path, (prediction_path, planes_path, output_path))
    if output_path.exists():
        raise FileExistsError(output_path)
    if not np.isfinite(tolerance_fraction) or tolerance_fraction <= 0:
        raise ValueError('positive tolerance fraction required')
    with np.load(prediction_path, allow_pickle=False) as data:
        origins, endpoints, views = rays_from_prediction(data, stride)
    if not len(endpoints):
        raise ValueError('no valid rays')
    # Quantiles of radius preserve rotation and translation invariance.
    reference = endpoints.mean(axis=0)
    extent = float(np.percentile(np.linalg.norm(endpoints-reference, axis=1), 90))
    if extent <= 0:
        raise ValueError('degenerate point geometry')
    planes = json.loads(planes_path.read_text(encoding='utf-8'))['planes']
    rows = []
    for i, plane in enumerate(planes):
        patches = plane.get('surface_patches', {})
        if not patches.get('patches'):
            rows.append({'plane_id': i, 'status': 'patch_bounds_unavailable'})
        for j, patch in enumerate(patches.get('patches', [])):
            row = score_patch(origins, endpoints, views, plane['center'],
                              patches['axes_world'], patch['bounds_uv'], extent*tolerance_fraction)
            rows.append({'plane_id': i, 'patch_id': j, **row})
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                h.update(chunk)
        return h.hexdigest()
    result = {'role': 'predicted_ray_visibility_diagnostic', 'ground_truth_used': False,
              'input_hashes': {str(p): digest(p) for p in (prediction_path, planes_path)},
              'sampled_ray_count': len(endpoints), 'sampled_view_count': len(np.unique(views)),
              'stride': stride, 'tolerance_fraction': tolerance_fraction,
              'tolerance_is_calibrated': False, 'patches': rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def rays_from_prediction(data, stride=8, confidence_percentile=50):
    if stride < 1 or not 0 <= confidence_percentile <= 100:
        raise ValueError('invalid sampling parameters')
    arrays = [data[k] for k in ('depth', 'confidence', 'intrinsics', 'extrinsics')]
    if len({len(a) for a in arrays}) != 1:
        raise ValueError('frame counts differ')
    frame_ids = np.asarray(data['source_indices']) if 'source_indices' in data else np.arange(len(arrays[0]))
    if frame_ids.shape != (len(arrays[0]),) or frame_ids.dtype.kind not in 'iu' or np.any(frame_ids < 0):
        raise ValueError('source_indices must be nonnegative integer frame identities')
    if 'component_ids' in data and len(np.unique(data['component_ids'])) > 1:
        raise ValueError('disconnected components have different coordinate frames; score separately')
    origins, endpoints, views = [], [], []
    for i, (depth, conf, K, pose) in enumerate(zip(*arrays)):
        depth, conf, K, pose = map(np.asarray, (depth, conf, K, pose))
        if depth.ndim != 2 or conf.shape != depth.shape or K.shape != (3, 3) or pose.shape not in ((3, 4), (4, 4)):
            raise ValueError('invalid prediction shapes')
        R, t = pose[:3, :3], pose[:3, 3]
        if not np.isfinite(pose).all() or not np.isfinite(K).all() or not np.allclose(R @ R.T, np.eye(3), atol=1e-4) or np.linalg.det(R) < .99:
            raise ValueError('invalid camera calibration or rotation')
        y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z, c = depth[y, x], conf[y, x]
        valid = np.isfinite(z) & (z > 0) & np.isfinite(c)
        if not valid.any():
            continue
        valid &= c >= np.percentile(c[valid], confidence_percentile)
        pixels = np.stack([x[valid], y[valid], np.ones(valid.sum())], axis=1)
        camera = (pixels @ np.linalg.inv(K).T) * z[valid, None]
        endpoints.append((camera - t) @ R)
        origins.append(np.broadcast_to(-R.T @ t, camera.shape).copy())
        views.append(np.full(len(camera), frame_ids[i], dtype=int))
    if not endpoints:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0, dtype=int)
    return np.concatenate(origins), np.concatenate(endpoints), np.concatenate(views)


def score_patch(origins, endpoints, view_ids, center, axes, bounds, tolerance, grid_size=4):
    """Classify ray intersections with a finite plane rectangle.

    axes: orthonormal 2x3 patch axes; bounds: local UV min/max (2x2).
    tolerance: caller-declared depth/pose uncertainty in the same units as points.
    A hit beyond the measured endpoint is occluded, never support for that wall.
    No intersection means unobserved. Counts are also grouped per view to avoid
    treating thousands of correlated pixels as independent confirmations.
    """
    o, p = np.asarray(origins, float), np.asarray(endpoints, float)
    ids, c, axes, bounds = np.asarray(view_ids), np.asarray(center, float), np.asarray(axes, float), np.asarray(bounds, float)
    if o.ndim != 2 or o.shape[1:] != (3,) or p.shape != o.shape or ids.shape != (len(o),):
        raise ValueError('rays must be matching Nx3 arrays with N view ids')
    if c.shape != (3,) or axes.shape != (2, 3) or bounds.shape != (2, 2):
        raise ValueError('invalid patch geometry')
    if not all(np.isfinite(a).all() for a in (o, p, c, axes, bounds)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('finite geometry and positive uncertainty required')
    if not np.allclose(axes @ axes.T, np.eye(2), atol=1e-6) or np.any(bounds[1] <= bounds[0]):
        raise ValueError('patch axes must be orthonormal and bounds nonempty')
    if not isinstance(grid_size, int) or not 1 <= grid_size <= 64:
        raise ValueError('grid_size must be an integer between1 and64')
    normal = np.cross(*axes)
    ray = p - o
    length = np.linalg.norm(ray, axis=1)
    denominator = ray @ normal
    valid = (length > 0) & (abs(denominator) > 1e-8 * length)
    t = np.zeros(len(o))
    np.divide((c - o) @ normal, denominator, out=t, where=valid)
    uv = (o + t[:, None] * ray - c) @ axes.T
    intersects = valid & (t > 0) & np.all((uv >= bounds[0]) & (uv <= bounds[1]), axis=1)
    gap = (1 - t) * length
    masks = {
        'observed_support': intersects & (abs(gap) <= tolerance),
        'free_space_conflict': intersects & (gap > tolerance),
        'occluded': intersects & (gap < -tolerance),
        'unobserved': ~intersects,
    }
    per_view = []
    for view in np.unique(ids):
        chosen = ids == view
        counts = {k: int((v & chosen).sum()) for k, v in masks.items()}
        tested = counts['observed_support'] + counts['free_space_conflict']
        per_view.append({'view_id': int(view), **counts,
                         'conflict_fraction_visible': counts['free_space_conflict'] / tested if tested else None})
    fractions = [v['conflict_fraction_visible'] for v in per_view if v['conflict_fraction_visible'] is not None]
    # Only rays intersecting this finite patch belong in its cells. Empty cells
    # remain unknown; occluded-only cells provide no positive wall evidence.
    cell_ids = np.full(len(o), -1, dtype=int)
    coordinates = (uv[intersects]-bounds[0])/(bounds[1]-bounds[0])
    cells = np.minimum((coordinates*grid_size).astype(int), grid_size-1)
    cell_ids[intersects] = cells[:, 0] + grid_size*cells[:, 1]
    spatial = []
    for y in range(grid_size):
        for x in range(grid_size):
            selected = cell_ids == x+grid_size*y
            counts = {k: int((v & selected).sum()) for k, v in masks.items() if k != 'unobserved'}
            visible = selected & (masks['observed_support'] | masks['free_space_conflict'])
            visible_views = np.unique(ids[visible])
            ratios = [float((masks['free_space_conflict'] & selected & (ids == v)).sum() /
                            (visible & (ids == v)).sum()) for v in visible_views]
            lo = bounds[0] + np.array([x, y])*(bounds[1]-bounds[0])/grid_size
            hi = lo+(bounds[1]-bounds[0])/grid_size
            spatial.append({'cell': [x, y], 'bounds_uv': [lo.tolist(), hi.tolist()], **counts,
                            'support_view_count': len(np.unique(ids[selected & masks['observed_support']])),
                            'conflict_view_count': len(np.unique(ids[selected & masks['free_space_conflict']])),
                            'visible_view_count': len(visible_views),
                            'conflict_fraction_equal_view': float(np.mean(ratios)) if ratios else None,
                            'status': 'visible_evidence' if ratios else 'unknown'})
    return {'counts': {k: int(v.sum()) for k, v in masks.items()}, 'per_view': per_view,
            'spatial_grid': {'size': grid_size, 'cells': spatial},
            'mean_per_view_conflict_fraction': float(np.mean(fractions)) if fractions else None,
            'visible_view_count': len(fractions), 'tolerance': float(tolerance),
            'status': 'visibility_evidence' if fractions else 'no_visible_evidence',
            'limitation': 'Predicted depth and poses may be wrong; occlusion does not prove a wall exists.'}


def make_boundary_scorer(prediction, frame, axis, height_range, tolerance,
                         min_views=3, reject_fraction=.5, stride=8):
    """Adapter for boundary_assembly's scorer(p0_xy,p1_xy) protocol.

    Fractions refer to finite wall-patch area over the supplied observed height
    band, not calibrated probabilities. An opening alone is not room free space;
    the caller must account for openings before using rejection as a hard gate.
    """
    origins, endpoints, ids = rays_from_prediction(prediction, stride)
    origin = np.asarray(frame['origin_world'], float)
    e1, e2, axis = map(lambda v: np.asarray(v, float), (frame['e1'], frame['e2'], axis))
    basis = np.stack([e1, e2, axis])
    if not np.allclose(basis@basis.T, np.eye(3), atol=1e-6):
        raise ValueError('boundary frame must be orthonormal')
    lo, hi = map(float, height_range)
    if not np.isfinite([lo, hi]).all() or hi <= lo or min_views < 1 or not 0 < reject_fraction <= 1:
        raise ValueError('invalid height band or evidence policy')

    def scorer(p0_xy, p1_xy):
        p0, p1 = np.asarray(p0_xy, float), np.asarray(p1_xy, float)
        if p0.shape != (2,) or p1.shape != (2,) or not np.isfinite([p0,p1]).all():
            raise ValueError('finite 2D segment required')
        delta = p1-p0
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            raise ValueError('nonzero boundary segment required')
        direction = (delta[0]*e1+delta[1]*e2)/length
        midpoint = (p0+p1)/2
        center = origin+midpoint[0]*e1+midpoint[1]*e2+(lo+hi)/2*axis
        report = score_patch(origins, endpoints, ids, center, [direction, axis],
                             [[-length/2, -(hi-lo)/2], [length/2, (hi-lo)/2]], tolerance)
        cells = report['spatial_grid']['cells']
        conflicts = [c['conflict_view_count'] >= min_views and
                     (c['conflict_fraction_equal_view'] or 0) >= .5 for c in cells]
        support = [c['support_view_count'] >= min_views and not conflict
                   for c, conflict in zip(cells, conflicts)]
        fraction = float(np.mean(conflicts))
        return {'free_space_contradiction': fraction >= reject_fraction,
                'free_space_fraction': fraction, 'occupied_fraction': float(np.mean(support)),
                'unobserved_fraction': float(np.mean([not (a or b) for a,b in zip(conflicts,support)])),
                'supporting_frames': sum(v['observed_support'] > 0 for v in report['per_view']),
                'contradicting_frames': sum(v['free_space_conflict'] > 0 for v in report['per_view']),
                'fraction_domain': 'wall_patch_area_within_supplied_height_band',
                'ray_evidence': report, 'policy': {'min_views': min_views, 'reject_fraction': reject_fraction}}
    return scorer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--planes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--tolerance-fraction', type=float, default=.01)
    args = parser.parse_args()
    result = audit_prediction(args.prediction, args.planes, args.output, args.stride, args.tolerance_fraction)
    print(f"Scored {len(result['patches'])} patches across {result['sampled_view_count']} views")
