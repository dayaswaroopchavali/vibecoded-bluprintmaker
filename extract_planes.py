"""Unscaled observed plane candidates from DA3; not a completed blueprint.

Reads predicted RGB-derived geometry only. No simulator imports or annotations.
Planes carry connected surface patches with observed in-plane bounds.
Planes carry connected surface patches with observed in-plane bounds.
Plane support may include furniture: downstream semantic/boundary reasoning is
required before calling any candidate a wall. Thresholds are scale-relative.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def points_from_prediction(data, stride=6):
    chunks = []
    for depth, conf, K, pose in zip(data['depth'], data['confidence'],
                                     data['intrinsics'], data['extrinsics']):
        y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[y, x]
        c = conf[y, x]
        valid = np.isfinite(z) & (z > 0) & np.isfinite(c)
        if not valid.any():
            continue
        valid &= c >= np.percentile(c[valid], 50)
        pixels = np.stack([x[valid], y[valid], np.ones(valid.sum())], axis=1)
        camera = (pixels @ np.linalg.inv(K).T) * z[valid, None]
        # DA3 extrinsics are world-to-camera, column-vector convention.
        world = (camera - pose[:3, 3]) @ pose[:3, :3]
        chunks.append(world)
    if not chunks:
        raise ValueError('No valid predicted depth samples')
    return np.concatenate(chunks)


def inplane_axes(normal):
    """Deterministic orthonormal in-plane axes (u, v) for a unit normal."""
    seed = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    u = np.cross(normal, seed)
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def split_patches(support, normal, center, cell, min_count):
    """Split a plane's inliers into 8-connected occupied-cell patches.

    One RANSAC plane can join disjoint coplanar surfaces (two wall segments
    either side of a doorway or a column). Each patch keeps its own observed
    in-plane bounds so missing stretches stay missing.
    """
    u, v = inplane_axes(normal)
    uv = np.stack([(support - center) @ u, (support - center) @ v], axis=1)
    cells = np.floor(uv / cell).astype(np.int64)
    members = {}
    for index, key in enumerate(map(tuple, cells)):
        members.setdefault(key, []).append(index)
    seen, patches = set(), []
    for start in members:
        if start in seen:
            continue
        stack, indices = [start], []
        seen.add(start)
        while stack:
            i, j = stack.pop()
            indices.extend(members[(i, j)])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    neighbour = (i + di, j + dj)
                    if neighbour in members and neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
        if len(indices) >= min_count:
            lo, hi = np.percentile(uv[indices], [2, 98], axis=0)
            patches.append({'bounds_uv': [lo.tolist(), hi.tolist()], 'support_count': len(indices)})
    patches.sort(key=lambda patch: -patch['support_count'])
    return {'axes_world': [u.tolist(), v.tolist()], 'cell_size': float(cell), 'patches': patches}


def extract(points, seed=0, max_planes=16, trials=300):
    rng = np.random.default_rng(seed)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 100:
        raise ValueError('Insufficient finite geometry')
    extent = np.linalg.norm(np.percentile(finite, 95, axis=0) -
                            np.percentile(finite, 5, axis=0))
    if extent <= 0:
        raise ValueError('Degenerate point cloud')
    threshold = extent * .006
    remaining = finite[rng.choice(len(finite), min(len(finite), 25000), replace=False)]
    minimum = max(80, int(len(remaining) * .025))
    planes = []
    for _ in range(max_planes):
        if len(remaining) < minimum:
            break
        best = None
        best_count = 0
        for _ in range(trials):
            a, b, c = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            norm = np.linalg.norm(normal)
            if norm < extent * extent * 1e-9:
                continue
            normal /= norm
            mask = np.abs((remaining-a) @ normal) < threshold
            count = int(mask.sum())
            if count > best_count:
                best, best_count = mask, count
        if best_count < minimum:
            break
        support = remaining[best]
        center = support.mean(axis=0)
        _, _, axes = np.linalg.svd(support-center, full_matrices=False)
        normal = axes[-1]
        residual = np.abs((support-center) @ normal)
        planes.append({'normal': normal.tolist(), 'offset': float(-normal @ center),
                       'center': center.tolist(), 'support_count': best_count,
                       'median_residual': float(np.median(residual)),
                       'observed_bounds_world': np.percentile(support, [2,98], axis=0).tolist(),
                       'surface_patches': split_patches(support, normal, center, extent * .04, max(20, int(best_count * .05))),
                       'semantic_class': 'unknown', 'measurement_interval': None})
        remaining = remaining[~best]
    return {'status': 'plane_candidates_only', 'units': 'arbitrary',
            'metric_scale_source': None, 'threshold': float(threshold),
            'input_point_count': len(finite), 'planes': planes,
            'limitations': ['Furniture can be a plane candidate.',
                           'No calibrated intervals or opening detection.',
                           'No complete room boundary or metric gate claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.prediction) as data:
        result = extract(points_from_prediction(data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
