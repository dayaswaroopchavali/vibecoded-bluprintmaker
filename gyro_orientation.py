"""Truth-free gyro-orientation prior for a rolling / full-sphere capture.

Public inputs only: a ``virtual_gyro_v1`` stream (timestamps + consecutive relative camera
rotations), a frozen raw DA3 prediction (depth, intrinsics, images, extrinsics) and nothing else.
No simulator, private pose, scene dimension or ground-truth module is imported here.

Variants (all frozen to a NEW folder with sha256 before any evaluator touches them):

``gyro_rot_da3_centres``
    Replace the DA3 rotations by the gyro chain and keep DA3 camera centres unchanged. Needs one
    global world rotation Q (gauge), estimated robustly from DA3-vs-gyro agreement. Baseline only:
    centres computed by a rotation-confused DA3 are not made consistent by this swap.
``gyro_rot_depth_centres``
    Gyro rotations are a HARD constraint; DA3 contributes depth and intrinsics only. Camera centres
    are re-solved by translation-only Gauss-Newton on multi-view depth consistency (the z-depth of
    frame i's points, reprojected into frame j, must equal frame j's depth map at that pixel).
    The equation is linear in the centres given the rotations, so the problem is convex per
    association step and is solved by continuation over the frame-offset window (near pairs first).

Gauge: the world frame is camera 0 (R_0 = I); an optional gauge Q expresses it in DA3's world.
A global rotation never changes the geometry, so Q only matters for comparing with DA3 poses.
Gyro chain drift is reported by noise injection (bias + white noise) on the same public stream:
that measures sensitivity to sensor error, not accuracy, and needs no truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "gyro_orientation_v1"
DEPTH_VARIANTS = ("gyro_rot_depth_centres", "gyro_rot_depth_centres_smooth", "gyro_rot_depth_centres_smooth_scale")


# ------------------------------------------------------------------------------------------ SO(3) helpers
def rotation_exp(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, float)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.eye(3) + np.array([[0., -vec[2], vec[1]], [vec[2], 0., -vec[0]], [-vec[1], vec[0], 0.]])
    x, y, z = vec / angle
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * skew + (1. - math.cos(angle)) * (skew @ skew)


def project_to_so3(m: np.ndarray) -> np.ndarray:
    """Nearest proper rotation (polar/SVD); never a reflection."""
    u, _, vt = np.linalg.svd(np.asarray(m, float))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def geodesic_deg(rel: np.ndarray) -> np.ndarray:
    """Rotation angle (deg) of each relative rotation (N,3,3); atan2 form is exact near 0 and 180."""
    rel = np.asarray(rel, float)
    skew = np.stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], 1) / 2
    cosine = (np.trace(rel, axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arctan2(np.linalg.norm(skew, axis=1), cosine))


def validate_deltas(deltas) -> np.ndarray:
    d = np.asarray(deltas, float)
    if d.ndim != 3 or d.shape[1:] != (3, 3) or len(d) < 1:
        raise ValueError("delta rotations must be (N-1,3,3)")
    if not np.all(np.isfinite(d)):
        raise ValueError("delta rotations must be finite")
    if not np.allclose(d @ np.transpose(d, (0, 2, 1)), np.eye(3), atol=2e-3):
        raise ValueError("delta rotations must be orthonormal")
    if np.any(np.linalg.det(d) < .998):
        raise ValueError("reflections are not valid gyroscope increments")
    return d


def integrate_gyro(deltas) -> np.ndarray:
    """Per-frame camera-to-world rotations relative to frame 0 (R_0 = I).

    ``delta[i]`` maps camera i+1 vectors into camera i coordinates (virtual_gyro_v1 convention), so
    R_{i+1} = R_i @ delta[i]. Full SO(3): arbitrary smooth roll and inversion are preserved because
    no up direction or gravity is ever introduced.
    """
    d = validate_deltas(deltas)
    out = np.empty((len(d) + 1, 3, 3))
    out[0] = np.eye(3)
    for i, delta in enumerate(d):
        out[i + 1] = project_to_so3(out[i] @ delta)
    return out


def load_gyro(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "virtual_gyro_v1":
        raise ValueError("unsupported gyroscope stream schema")
    deltas = validate_deltas(data["delta_camera_to_previous"])
    if len(deltas) + 1 != int(data["frame_count"]):
        raise ValueError("frame_count disagrees with the number of gyro increments")
    return {"deltas": deltas, "timestamps_s": np.asarray(data["timestamps_s"], float), "manifest": data.get("manifest", {})}


# ------------------------------------------------------------------------------------------ gyro-chain drift
def perturb_deltas(deltas, timestamps_s, noise_std_rad_s=0.0, bias_rad_s=(0, 0, 0), seed=0) -> np.ndarray:
    """Inject bias + white angular-rate noise into a public gyro stream (same model as virtual_gyro)."""
    d = validate_deltas(deltas)
    t = np.asarray(timestamps_s, float)
    if len(t) != len(d) + 1 or np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing, one per frame")
    rng = np.random.default_rng(seed)
    out = np.empty_like(d)
    for i, delta in enumerate(d):
        dt = float(t[i + 1] - t[i])
        extra = (np.asarray(bias_rad_s, float) + rng.normal(0.0, noise_std_rad_s, 3)) * dt
        out[i] = delta @ rotation_exp(extra)
    return out


def drift_vs_noise(deltas, timestamps_s, noise_levels_rad_s=(0.0, 0.002, 0.005, 0.01, 0.02), bias_rad_s=(0, 0, 0),
                   seeds=range(5)) -> list:
    """Truth-free sensitivity: geodesic deviation of a perturbed chain from the supplied chain.

    This is the drift the pipeline WOULD suffer if the supplied stream carried that much error; it is
    not a measurement of the real stream's accuracy.
    """
    base = integrate_gyro(deltas)
    rows = []
    for sigma in noise_levels_rad_s:
        ends, means = [], []
        for s in seeds:
            chain = integrate_gyro(perturb_deltas(deltas, timestamps_s, sigma, bias_rad_s, s))
            g = geodesic_deg(np.einsum("nji,njk->nik", base, chain))
            ends.append(float(g[-1]))
            means.append(float(g.mean()))
        rows.append({"noise_std_rad_s": float(sigma), "bias_rad_s": list(map(float, bias_rad_s)),
                     "mean_deviation_deg": float(np.mean(means)), "final_deviation_deg": float(np.mean(ends))})
    return rows


# ------------------------------------------------------------------------------------------ gauge
def robust_gauge(gyro_c2w, da3_c2w, iterations=20, trim=0.5) -> dict:
    """Single global rotation Q with da3_c2w[i] ~= Q @ gyro_c2w[i], from DA3-vs-gyro agreement only.

    Iteratively re-weighted Kabsch on the best ``trim`` fraction of frames (geodesic residual).
    Truth-free: both inputs are public. If DA3 orientation is unreliable (as in rolling video) the
    inlier fraction and the residual say so; the result is then NOT a reliable gauge and is reported.
    """
    g, d = np.asarray(gyro_c2w, float), np.asarray(da3_c2w, float)
    if g.shape != d.shape or g.ndim != 3:
        raise ValueError("rotation stacks must have identical shape (N,3,3)")
    weights = np.ones(len(g))
    q = np.eye(3)
    for _ in range(iterations):
        m = np.einsum("n,nij,nkj->ik", weights, d, g)
        q = project_to_so3(m)
        res = geodesic_deg(np.einsum("nji,njk->nik", d, np.einsum("ij,njk->nik", q, g)))
        keep = max(3, int(math.ceil(trim * len(g))))
        cut = np.sort(res)[keep - 1]
        weights = (res <= cut).astype(float)
    return {"Q": q, "residual_deg": res, "median_residual_deg": float(np.median(res)),
            "inlier_fraction_below_10deg": float(np.mean(res < 10.0)), "trimmed_fraction": trim}


# ------------------------------------------------------------------------------------------ black borders
def valid_pixel_mask(image: np.ndarray, margin: int = 2, black_threshold: int = 3) -> np.ndarray:
    """True for pixels carrying scene content. Black padding is pure-black area connected to the border.

    Interior dark scene regions are kept (they do not touch the frame edge as an exact-black region).
    """
    import cv2
    img = np.asarray(image)
    gray = img.max(axis=2) if img.ndim == 3 else img
    black = (gray <= black_threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=4)
    h, w = gray.shape
    pad = np.zeros((h, w), bool)
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        # padding is a large exact-black region on the border; a few stray black pixels are not
        if touches and area >= 0.002 * h * w:
            pad |= labels == k
    if margin > 0 and pad.any():
        pad = cv2.dilate(pad.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)).astype(bool)
    return ~pad


# ------------------------------------------------------------------------------------------ depth sampling
def _bilinear(img, u, v):
    h, w = img.shape
    uu = np.clip(u, 0, w - 1.001)
    vv = np.clip(v, 0, h - 1.001)
    u0, v0 = uu.astype(int), vv.astype(int)
    fu, fv = uu - u0, vv - v0
    return ((img[v0, u0] * (1 - fu) + img[v0, u0 + 1] * fu) * (1 - fv)
            + (img[v0 + 1, u0] * (1 - fu) + img[v0 + 1, u0 + 1] * fu) * fv)


class _Frames:
    """Per-frame precomputation for the depth-consistency solver."""

    def __init__(self, depth, K, R, masks, stride):
        n, h, w = depth.shape
        self.n, self.h, self.w = n, h, w
        self.depth, self.K, self.R = depth.astype(np.float64), K.astype(np.float64), R
        us, vs = np.meshgrid(np.arange(stride // 2, w, stride), np.arange(stride // 2, h, stride))
        self.u, self.v = us.ravel().astype(float), vs.ravel().astype(float)
        self.masks = masks
        self.grad_u, self.grad_v, self.samples = [], [], []
        for i in range(n):
            d = self.depth[i]
            gv, gu = np.gradient(_box(d))
            self.grad_u.append(gu)
            self.grad_v.append(gv)
            ok = masks[i][self.v.astype(int), self.u.astype(int)] & np.isfinite(d[self.v.astype(int), self.u.astype(int)]) \
                & (d[self.v.astype(int), self.u.astype(int)] > 1e-6)
            z = d[self.v.astype(int), self.u.astype(int)]
            ray = np.stack([(self.u - K[i, 0, 2]) / K[i, 0, 0], (self.v - K[i, 1, 2]) / K[i, 1, 1], np.ones_like(self.u)], 1)
            a = (ray * z[:, None]) @ R[i].T           # world-frame offset of each sample from its centre
            self.samples.append((a[ok], ok))


def _box(d, k=5):
    pad = k // 2
    p = np.pad(d, pad, mode="edge")
    out = np.zeros_like(d)
    for dy in range(k):
        for dx in range(k):
            out += p[dy:dy + d.shape[0], dx:dx + d.shape[1]]
    return out / (k * k)


def _normal_equations(fr, centres, scales, window, gate, huber, fit_scale, min_pairs=15):
    """Gauss-Newton normal equations; unknowns are [C_0..C_{n-1} (3n), log-scale_0..log-scale_{n-1} (n)]."""
    n = fr.n
    m = 4 * n
    H = np.zeros((m, m))
    b = np.zeros(m)
    cost, count = 0.0, 0
    for i in range(n):
        a_i = fr.samples[i][0] * scales[i]
        if len(a_i) == 0:
            continue
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            dx = centres[i] - centres[j] + a_i
            pc = dx @ fr.R[j]
            z = pc[:, 2]
            ok = z > 1e-4
            u = fr.K[j, 0, 0] * pc[:, 0] / np.where(ok, z, 1.0) + fr.K[j, 0, 2]
            v = fr.K[j, 1, 1] * pc[:, 1] / np.where(ok, z, 1.0) + fr.K[j, 1, 2]
            ok &= (u >= 1) & (u <= fr.w - 2) & (v >= 1) & (v <= fr.h - 2)
            if ok.sum() < min_pairs:
                continue
            ui, vi = u[ok], v[ok]
            ok_idx = np.flatnonzero(ok)
            ok2 = fr.masks[j][vi.astype(int), ui.astype(int)]
            dj = _bilinear(fr.depth[j], ui, vi) * scales[j]
            e = z[ok_idx] - dj
            ok2 &= np.isfinite(dj) & (dj > 1e-6) & (np.abs(e) < gate * dj)
            if ok2.sum() < min_pairs:
                continue
            sel = ok_idx[ok2]
            e, dj, ui, vi = e[ok2], dj[ok2], ui[ok2], vi[ok2]
            zz = z[sel]
            gu = _bilinear(fr.grad_u[j], ui, vi) * scales[j]
            gv = _bilinear(fr.grad_v[j], ui, vi) * scales[j]
            fx, fy = fr.K[j, 0, 0], fr.K[j, 1, 1]
            pcx, pcy = pc[sel, 0], pc[sel, 1]
            wc = np.stack([-gu * fx / zz, -gv * fy / zz, 1.0 + (gu * fx * pcx + gv * fy * pcy) / zz ** 2], 1)
            wc = np.clip(wc, -5, 5)
            wv = wc @ fr.R[j].T                         # d e / d (C_i - C_j) in world coordinates
            scale = huber * dj
            rw = np.where(np.abs(e) <= scale, 1.0, scale / np.maximum(np.abs(e), 1e-12)) / dj ** 2
            cost += float(np.sum(rw * e * e)); count += len(e)
            # Jacobian rows over [C_i, C_j, ls_i, ls_j]: e = z(C_i - C_j + s_i a_i) - s_j D_j(uv)
            cols = [slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)]
            jac = [wv, -wv]
            if fit_scale:
                cols += [slice(3 * n + i, 3 * n + i + 1), slice(3 * n + j, 3 * n + j + 1)]
                jac += [np.sum(wv * a_i[sel], 1, keepdims=True), -dj[:, None]]
            for ca, ja in zip(cols, jac):
                b[ca] -= np.einsum("s,s,sa->a", rw, e, ja)
                for cb, jb in zip(cols, jac):
                    H[ca, cb] += np.einsum("s,sa,sb->ab", rw, ja, jb)
    return H, b, cost, count


def solve_centres(depth, intrinsics, rotations_c2w, images=None, stride=14, windows=(1, 2, 4, 8, 16, 1000),
                  gates=(0.30, 0.2, 0.15, 0.1, 0.08, 0.06), iterations=6, huber=0.03, damping=1e-6, anchor=0,
                  fit_scale=False, scale_sigma=0.2, smooth_weights=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003),
                  initial_centres=None) -> dict:
    """Translation-only Gauss-Newton on multi-view depth consistency with HARD (given) rotations.

    Cameras are unknown only through their centres C_i; rotations, depths and intrinsics are fixed.
    ``fit_scale`` additionally estimates one positive depth multiplier per frame (monocular depth
    is only defined up to scale per frame; DA3 multi-view depth is not guaranteed coherent).
    Continuation: start with adjacent-frame pairs and widen the pair window (and tighten the outlier
    gate) so each association step starts inside the basin of the previous one. Camera ``anchor`` is
    pinned at the origin with scale 1 (centre and global-scale gauge).
    """
    depth = np.asarray(depth)
    n = len(depth)
    if images is not None:
        masks = [valid_pixel_mask(images[i]) for i in range(n)]
    else:
        masks = [np.ones(depth.shape[1:], bool) for _ in range(n)]
    fr = _Frames(depth, np.asarray(intrinsics), np.asarray(rotations_c2w, float), masks, stride)
    centres = np.zeros((n, 3)) if initial_centres is None else np.asarray(initial_centres, float) - np.asarray(initial_centres, float)[anchor]
    scales = np.ones(n)
    history = []
    for stage, (window, gate) in enumerate(zip(windows, gates)):
        for it in range(iterations):
            H, b, cost, count = _normal_equations(fr, centres, scales, window, gate, huber, fit_scale)
            history.append({"window": int(window), "gate": float(gate), "iteration": it,
                            "weighted_cost_per_residual": cost / max(count, 1), "residuals": int(count)})
            if fit_scale:
                # Gaussian prior on log-scale (sigma_s): without it the all-zero depth scale is a trivial minimum.
                prior = (huber / scale_sigma) ** 2 * count / n   # strength: one frame's share of the data
                for k in range(n):
                    H[3 * n + k, 3 * n + k] += prior
                    b[3 * n + k] -= prior * np.log(scales[k])
            lam = smooth_weights[min(stage, len(smooth_weights) - 1)] if smooth_weights else 0.0
            if lam > 0 and n >= 3:
                # Smooth-path prior (second differences of the centres), annealed away stage by stage.
                d2 = np.zeros((n - 2, n))
                for k in range(n - 2):
                    d2[k, k:k + 3] = (1.0, -2.0, 1.0)
                pen = lam * np.mean(np.diag(H)[:3 * n]) * np.kron(d2.T @ d2, np.eye(3))
                H[:3 * n, :3 * n] += pen
                b[:3 * n] -= pen @ centres.reshape(-1)
            free = [k for k in range(3 * n) if k // 3 != anchor]
            if fit_scale:
                free += [3 * n + k for k in range(n) if k != anchor]
            keep = np.array(free)
            Hk = H[np.ix_(keep, keep)] + damping * np.trace(H) / len(H) * np.eye(len(keep))
            step = np.zeros(4 * n)
            step[keep] = np.linalg.solve(Hk, b[keep])
            centres = centres + step[:3 * n].reshape(n, 3)
            scales = scales * np.exp(np.clip(step[3 * n:], -0.5, 0.5))
            if np.max(np.abs(step)) < 1e-6:
                break
    # Truth-free observability: eigenvalues of the final data Hessian over the free centres. In planar rooms the
    # in-plane translation of a camera that only sees parallel walls is weakly constrained by depth alone.
    h_fin, _, _, _ = _normal_equations(fr, centres, scales, windows[-1], gates[-1], huber, False)
    free_c = np.array([k for k in range(3 * n) if k // 3 != anchor])
    eig = np.linalg.eigvalsh(h_fin[np.ix_(free_c, free_c)])
    observability = {"min_eig_over_max_eig": float(eig[0] / eig[-1]), "weak_modes_below_1e-3_of_max": int(np.sum(eig < 1e-3 * eig[-1])),
                     "free_unknowns": int(len(free_c))}
    return {"centres": centres, "scales": scales, "history": history, "observability": observability,
            "mask_valid_fraction": [float(m.mean()) for m in masks]}


# ------------------------------------------------------------------------------------------ prediction assembly
def extrinsics_from(rotations_c2w, centres) -> np.ndarray:
    """World-to-camera 3x4 (DA3 convention): R_w2c = R_c2w.T, t = -R_w2c C."""
    r = np.transpose(np.asarray(rotations_c2w, float), (0, 2, 1))
    t = -np.einsum("nij,nj->ni", r, np.asarray(centres, float))
    return np.concatenate([r, t[:, :, None]], axis=2).astype(np.float32)


def da3_centres_and_c2w(extrinsics):
    e = np.asarray(extrinsics, float)
    r_w2c = e[:, :3, :3]
    return -np.einsum("nji,nj->ni", r_w2c, e[:, :3, 3]), np.transpose(r_w2c, (0, 2, 1))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_variants(prediction_npz: Path, gyro_json: Path, out_dir: Path, variants=("gyro_rot_da3_centres", "gyro_rot_depth_centres", "gyro_rot_depth_centres_smooth",
                   "gyro_rot_depth_centres_smooth_scale"),
                   stride=14, apply_gauge=False) -> dict:
    """Write each variant to <out_dir>/<variant>/path_00/prediction.npz (+manifest). Truth is never opened."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    with np.load(prediction_npz, allow_pickle=False) as z:
        pred = {k: z[k] for k in z.files}
    gyro = load_gyro(gyro_json)
    r_chain = integrate_gyro(gyro["deltas"])
    n = len(pred["depth"])
    if len(r_chain) != n:
        raise ValueError("gyro stream and prediction frame counts differ")
    da3_c, da3_r = da3_centres_and_c2w(pred["extrinsics"])
    gauge = robust_gauge(r_chain, da3_r)
    q = gauge["Q"] if apply_gauge else np.eye(3)
    rot = np.einsum("ij,njk->nik", q, r_chain)
    report = {"schema": SCHEMA, "truth_free": True, "inputs": {"prediction": str(prediction_npz), "prediction_sha256": _sha(prediction_npz),
              "gyro": str(gyro_json), "gyro_sha256": _sha(gyro_json)},
              "gauge_da3_vs_gyro": {"median_residual_deg": gauge["median_residual_deg"],
                                    "inlier_fraction_below_10deg": gauge["inlier_fraction_below_10deg"], "applied": bool(apply_gauge),
                                    "note": "agreement diagnostic; low inlier fraction means DA3 orientation cannot define the gauge"},
              "gyro_drift_sensitivity": drift_vs_noise(gyro["deltas"], gyro["timestamps_s"]), "variants": {}}
    out_dir.mkdir(parents=True)
    for name in variants:
        if name == "gyro_rot_da3_centres":
            # DA3 centres live in DA3's world, so the gyro chain must be expressed there as well.
            centres, rotations, extra = da3_c, np.einsum("ij,njk->nik", gauge["Q"], r_chain), {}
        elif name in DEPTH_VARIANTS:
            sol = solve_centres(pred["depth"], pred["intrinsics"], rot, pred.get("images"), stride=stride,
                                fit_scale=name.endswith("_scale"),
                                smooth_weights=(1.0, 0.3, 0.1, 0.03, 0.01, 0.003) if "_smooth" in name else None)
            centres, rotations = sol["centres"], rot
            extra = {"solver_history": sol["history"], "depth_scales": sol["scales"].tolist(), "observability": sol["observability"], "mask_valid_fraction_min": float(min(sol["mask_valid_fraction"]))}
        else:
            raise ValueError(f"unknown variant {name}")
        folder = out_dir / name / "path_00"
        folder.mkdir(parents=True)
        out = dict(pred)
        if name.endswith("_scale"):
            out["depth"] = (pred["depth"] * sol["scales"][:, None, None]).astype(pred["depth"].dtype)
        out["extrinsics"] = extrinsics_from(rotations, centres)
        np.savez_compressed(folder / "prediction.npz", **out)
        entry = {"prediction": str(folder / "prediction.npz"), "sha256": _sha(folder / "prediction.npz"), **extra}
        (folder / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "variant": name, **entry,
                                                         "truth_free": True}, indent=2), encoding="utf-8")
        report["variants"][name] = entry
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--gyro", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="NEW folder")
    p.add_argument("--stride", type=int, default=14)
    p.add_argument("--apply-gauge", action="store_true")
    a = p.parse_args()
    r = build_variants(a.prediction, a.gyro, a.output, stride=a.stride, apply_gauge=a.apply_gauge)
    print(json.dumps({k: v["sha256"] for k, v in r["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
