"""TRUTH-FREE removal of depth inconsistencies from a frozen DA3 prediction (opt-in; poses are never touched).

Evaluation-only diagnosis of frozen DA3 predictions found (after one global scale) a ceiling that reads a few percent too near,
per-frame depth scales that differ by 1-3 % and fx != fy intrinsics. This module estimates and removes such inconsistencies from
the prediction's OWN geometry only:

  (a) per-frame depth scale factors      robust multi-view reprojection depth agreement over overlapping view pairs
                                         (poses fixed; gauge fixed to mean log-scale = 0, so the global scale is untouched);
  (b) per-cap-plane depth gain           plane-to-plane CONTACT closure: wherever a floor/ceiling-like plane meets a wall-like plane in
                                         an image, the two surfaces must have the same depth at the seam. The depth ratio of the two
                                         fitted planes along the seam rays is the cap's multiplicative offset relative to the wall
                                         baseline. Classes come from the cloud's own plane fits and the caller's structural axis
                                         (geometry_orientation.axis_hypotheses); never from camera up or the camera path;
  (c) fx, fy (optionally cx, cy)         self-calibration: minimise the multi-view depth-reprojection residual.

Every parameter is reported with a leave-one-view-out (jackknife) standard deviation. A correction whose uncertainty exceeds its
magnitude is NOT applied and is reported as 'not_identifiable'. Assumptions are labelled in the report: walls are the depth baseline
(only RELATIVE offsets are identifiable); the surface offset is multiplicative (an additive offset is not separated from it by a
seam); cameras are assumed to see the seams.

Nothing here reads ground truth, simulator data, camera-up, camera-path gravity or any instance constant.
"""
from __future__ import annotations

import math

import numpy as np

MODEL = "surface_scale_model_v1"
SAMPLE_STRIDE = 12             # pixel grid used for the multi-view residuals
NEIGHBOURS = 5                 # overlapping views kept per view
MIN_PAIR_INLIERS = 30
CONF_PERCENTILE = 30.0         # drop the least confident share of pixels per frame
GRADIENT_TOL = 0.06            # log-depth change over +-2 px above which a pixel is a discontinuity
INLIER_LOG = 0.12              # |log depth ratio| gate on candidate correspondences
TUKEY_C = 4.685
FOCAL_DELTA = 0.015            # scale of the bounded (Cauchy) loss in the focal self-calibration
REFUSE_FACTOR = 1.0            # apply a parameter only if its jackknife sd <= REFUSE_FACTOR * |value|
CAP_COS = float(np.cos(np.radians(12)))      # same thresholds as geometry_orientation
WALL_SIN = float(np.sin(np.radians(12)))
LABEL_STRIDE = 3
SEAM_OFFSETS = ((0, 1), (1, 0), (0, 2), (2, 0))    # label-grid steps between a cap pixel and a wall pixel
MIN_SEAM_SAMPLES_PER_VIEW = 15
MIN_SEAM_VIEWS = 4
SEAM_MAX_LOG = 0.15
SEAM_MIN_GRAZING = 0.2         # |n . unit ray| below this: plane seen too obliquely for a stable depth along the ray
FEATHER_PX = 7


class SurfaceScaleError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------- basics
def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def json_safe(obj):
    return _clean(obj)


def camera_centres(E):
    E = np.asarray(E, float)
    R, t = E[:, :, :3], E[:, :, 3]
    return -np.einsum("nji,nj->ni", R, t)


def _check(prediction):
    for key in ("depth", "confidence", "intrinsics", "extrinsics"):
        if key not in prediction:
            raise SurfaceScaleError(f"prediction lacks '{key}'")
    depth = np.asarray(prediction["depth"])
    if depth.ndim != 3:
        raise SurfaceScaleError("depth must be N x H x W")
    n = depth.shape[0]
    if np.asarray(prediction["intrinsics"]).shape != (n, 3, 3) or np.asarray(prediction["extrinsics"]).shape != (n, 3, 4):
        raise SurfaceScaleError("intrinsics / extrinsics must be N x 3 x 3 / N x 3 x 4")
    return n, depth.shape[1], depth.shape[2]


def _valid_masks(depth, conf):
    """(valid, smooth_valid): finite positive confident depth; and additionally free of depth discontinuities."""
    n, h, w = depth.shape
    ok = np.isfinite(depth) & (depth > 0) & np.isfinite(conf)
    for i in range(n):
        if ok[i].any():
            ok[i] &= conf[i] >= np.percentile(conf[i][ok[i]], CONF_PERCENTILE)
    ld = np.log(np.where(ok, depth, 1.0).astype(np.float32))
    smooth = ok.copy()
    for ax, shift in ((1, 2), (1, -2), (2, 2), (2, -2)):
        sh = np.roll(ld, shift, axis=ax); okr = np.roll(ok, shift, axis=ax)
        smooth &= okr & (np.abs(ld - sh) < GRADIENT_TOL)
    smooth[:, :3, :] = False; smooth[:, -3:, :] = False; smooth[:, :, :3] = False; smooth[:, :, -3:] = False
    return ok, smooth


def _scaled_K(K, theta):
    """K with fx, fy multiplied by exp(theta[0]), exp(theta[1]) and (optionally) cx, cy shifted by theta[2], theta[3] pixels."""
    K = np.array(K, float)
    K[:, 0, 0] *= math.exp(theta[0]); K[:, 1, 1] *= math.exp(theta[1])
    if len(theta) > 2:
        K[:, 0, 2] += theta[2]; K[:, 1, 2] += theta[3]
    return K


# ---------------------------------------------------------------------------------------------------------------- samples
class _Samples:
    """Flat list of source pixels (frame si, pixel x,y, source depth) paired with destination frames sj."""

    def __init__(self, depth, smooth, E):
        self.depth = np.asarray(depth, np.float64)
        self.n, self.h, self.w = self.depth.shape
        self.smooth = smooth
        self.ld = np.log(np.where(smooth, self.depth, 1.0))
        E = np.asarray(E, float)
        self.R, self.t = E[:, :, :3], E[:, :, 3]
        self.C = camera_centres(E)

    def grid(self, stride):
        pts = []
        for i in range(self.n):
            yy, xx = np.mgrid[stride // 2:self.h:stride, stride // 2:self.w:stride]
            m = self.smooth[i, yy, xx]
            pts.append((yy[m], xx[m]))
        return pts

    def project(self, K, si, xs, ys, sj, logscale):
        """Source pixels (frame si) seen by frames sj: returns z (dest depth of the scaled point), b0 (z per unit scale), px, py."""
        Ki = K[si]
        d = self.depth[si, ys.astype(int), xs.astype(int)]
        rays = np.stack([(xs - Ki[:, 0, 2]) / Ki[:, 0, 0], (ys - Ki[:, 1, 2]) / Ki[:, 1, 1], np.ones(len(xs))], axis=1)
        Rj = self.R[sj]
        dir_w = np.einsum("sji,sj->si", self.R[si], rays)
        base = np.einsum("sij,sj->si", Rj, self.C[si]) + self.t[sj]
        m = np.einsum("sij,sj->si", Rj, dir_w)
        s = np.exp(logscale[si])
        Xc = base + (s * d)[:, None] * m
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = K[sj, 0, 0] * Xc[:, 0] / z + K[sj, 0, 2]
            py = K[sj, 1, 1] * Xc[:, 1] / z + K[sj, 1, 2]
        return z, d * m[:, 2], base[:, 2], px, py

    def sample_dest(self, sj, px, py):
        """Bilinear log-depth at (px, py) of frame sj; ok only if all four corners are smooth valid pixels."""
        with np.errstate(invalid="ignore"):
            inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (py >= 0) & (px <= self.w - 1.001) & (py <= self.h - 1.001)
        x0 = np.floor(np.where(inside, px, 0)).astype(int); y0 = np.floor(np.where(inside, py, 0)).astype(int)
        fx = np.where(inside, px, 0) - x0; fy = np.where(inside, py, 0) - y0
        ok = inside.copy(); val = np.zeros(len(px))
        for dy, dx, wgt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
            ok &= self.smooth[sj, y0 + dy, x0 + dx]
            val += wgt * self.ld[sj, y0 + dy, x0 + dx]
        return val, ok


def _select_pairs(S, K, grid, neighbours, min_inliers):
    """Overlap by depth-consistent correspondences at unit scale; keep the best `neighbours` views per view."""
    n = S.n
    counts = np.zeros((n, n))
    zero = np.zeros(n)
    for i in range(n):
        ys, xs = grid[i]
        if len(xs) == 0:
            continue
        js = np.array([j for j in range(n) if j != i])
        si = np.repeat(i, len(xs) * len(js))
        xx = np.tile(xs, len(js)).astype(float); yy = np.tile(ys, len(js)).astype(float)
        sj = np.repeat(js, len(xs))
        z, _b, _a, px, py = S.project(K, si, xx, yy, sj, zero)
        val, ok = S.sample_dest(sj, px, py)
        ok &= np.isfinite(z) & (z > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ok &= np.abs(np.log(np.where(z > 0, z, 1.0)) - val) < INLIER_LOG
        counts[i] = np.bincount(sj[ok], minlength=n)
    pairs = set()
    for i in range(n):
        order = np.argsort(-counts[i], kind="stable")
        for j in order[:neighbours]:
            if counts[i, j] >= min_inliers:
                pairs.add((i, int(j)))
                pairs.add((int(j), i))
    return sorted(pairs), counts


def _flat_samples(grid, pairs):
    si, sj, xs, ys = [], [], [], []
    for i, j in pairs:
        y, x = grid[i]
        si.append(np.full(len(x), i)); sj.append(np.full(len(x), j)); xs.append(x.astype(float)); ys.append(y.astype(float))
    if not si:
        raise SurfaceScaleError("no overlapping view pairs")
    return np.concatenate(si), np.concatenate(sj), np.concatenate(xs), np.concatenate(ys)


def _tukey(r, scale):
    x = np.abs(r) / (TUKEY_C * scale)
    return np.where(x < 1, (1 - x * x) ** 2, 0.0)


def _robust_scale(r):
    return max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.003) if len(r) else 0.003


# ---------------------------------------------------------------------------------------------------------------- (a) scales
def estimate_frame_scales(prediction, K=None, sample_stride=SAMPLE_STRIDE, neighbours=NEIGHBOURS, outer=4, inner=4,
                          refuse_factor=REFUSE_FACTOR, loo=True) -> dict:
    """Per-frame log depth scale u_i (depth_i * exp(u_i) is multi-view consistent with the fixed poses).

    Model: z_j(X_i(u_i)) = a + exp(u_i) b with the source depth scaled, residual r = log z_j - log D_j - u_j, robust (Tukey) GN.
    Gauge: the reported (applied) values have mean log-scale 0 over the frames; the ungauged mean is reported as
    'depth_vs_pose_unit_log_factor' (it is NOT applied). Uncertainty: leave-one-view-out Newton re-solve on the final
    correspondences, jackknife sd per frame."""
    n, h, w = _check(prediction)
    depth = np.asarray(prediction["depth"], float)
    conf = np.asarray(prediction["confidence"], float)
    K = np.array(prediction["intrinsics"], float) if K is None else np.asarray(K, float)
    _ok, smooth = _valid_masks(depth, conf)
    S = _Samples(depth, smooth, prediction["extrinsics"])
    grid = S.grid(sample_stride)
    pairs, _counts = _select_pairs(S, K, grid, neighbours, MIN_PAIR_INLIERS)
    if n < 3 or len(pairs) < 2:
        return {"status": "not_identifiable", "reason": "too_few_overlapping_views", "n_frames": n, "n_pairs": len(pairs)}
    si, sj, xs, ys = _flat_samples(grid, pairs)
    u = np.zeros(n)
    for _ in range(outer):
        z, b0, a0, px, py = S.project(K, si, xs, ys, sj, u)
        val, ok = S.sample_dest(sj, px, py)
        ok &= np.isfinite(z) & (z > 1e-9)
        a = a0
        base_ok = ok.copy()
        for _inner in range(inner):
            e = np.exp(u[si])
            zz = a + e * b0
            good = base_ok & (zz > 1e-9)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(good, np.log(np.where(zz > 0, zz, 1.0)) - val - u[sj], 0.0)
            good &= np.abs(r) < INLIER_LOG
            wgt = np.where(good, _tukey(r, _robust_scale(r[good])), 0.0)
            Ji = np.where(good, e * b0 / np.where(zz > 0, zz, 1.0), 0.0)
            H, g = _normal_equations(si, sj, wgt, Ji, r, n)
            lam = 1e-9 * max(np.trace(H) / n, 1e-12)
            du = np.linalg.solve(H + lam * np.eye(n), -g)
            u = u + np.clip(du, -0.2, 0.2)
        # re-associate with the new scales (pixel correspondences move slightly)
    e = np.exp(u[si]); zz = a + e * b0
    good = base_ok & (zz > 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(good, np.log(np.where(zz > 0, zz, 1.0)) - val - u[sj], 0.0)
    good &= np.abs(r) < INLIER_LOG
    wgt = np.where(good, _tukey(r, _robust_scale(r[good])), 0.0)
    Ji = np.where(good, e * b0 / np.where(zz > 0, zz, 1.0), 0.0)
    n_samples = np.bincount(si[wgt > 0], minlength=n) + np.bincount(sj[wgt > 0], minlength=n)
    mean_u = float(np.mean(u))
    u_g = u - mean_u
    out = {"status": "ok", "n_frames": n, "n_pairs": len(pairs), "n_samples": int(len(si)), "n_inlier_samples": int((wgt > 0).sum()),
           "residual_robust_sd_log": _robust_scale(r[good]) if good.any() else None,
           "depth_vs_pose_unit_log_factor": mean_u, "log_scale_gauged": u_g.tolist(), "inlier_samples_per_frame": n_samples.tolist(),
           "gauge": "mean log-scale over frames = 0 (global depth scale left unchanged)"}
    if loo:
        pid = np.unique(np.stack([si, sj], axis=1), axis=0, return_inverse=True)
        pair_ids = pid[1].reshape(-1)
        ij = pid[0]
        npair = len(ij)
        agg = _pair_aggregates(pair_ids, npair, wgt, Ji, r)
        Hfull, gfull = _assemble(ij, agg, n, np.ones(npair, bool))
        lam = 1e-9 * max(np.trace(Hfull) / n, 1e-12)
        du_full = np.linalg.solve(Hfull + lam * np.eye(n), -gfull)
        est = np.full((n, n), np.nan)               # est[k, i]: gauge-fixed u_i with view k left out
        for k in range(n):
            keep = (ij[:, 0] != k) & (ij[:, 1] != k)
            Hk, gk = _assemble(ij, agg, n, keep)
            lamk = 1e-9 * max(np.trace(Hk) / n, 1e-12)
            uk = u + np.linalg.solve(Hk + lamk * np.eye(n), -gk)
            others = np.arange(n) != k
            est[k, others] = uk[others] - np.mean(uk[others])
        sd = np.full(n, np.nan)
        for i in range(n):
            col = est[np.arange(n) != i, i]
            col = col[np.isfinite(col)]
            if len(col) >= 3:
                sd[i] = math.sqrt((len(col) - 1) / len(col) * float(np.sum((col - col.mean()) ** 2)))
        out["log_scale_sd_leave_one_view_out"] = sd.tolist()
        gm = []
        for k in range(n):
            uk = est[k, np.arange(n) != k]
            gm.append(float(np.mean(u[np.arange(n) != k] + (uk - np.mean(uk)))))
        applied = np.isfinite(sd) & (sd <= refuse_factor * np.abs(u_g)) & (n_samples >= 30)
        out["applied"] = applied.tolist()
        out["log_scale_to_apply"] = np.where(applied, u_g, 0.0).tolist()
        out["not_identifiable_frames"] = [int(i) for i in np.nonzero(~applied)[0]]
    else:
        out["log_scale_to_apply"] = u_g.tolist()
        out["applied"] = [True] * n
        out["not_identifiable_frames"] = []
    return out


def _normal_equations(si, sj, w, Ji, r, n):
    idx = lambda a, b: a * n + b
    Hf = np.zeros(n * n)
    Hf += np.bincount(idx(si, si), weights=w * Ji * Ji, minlength=n * n)
    Hf += np.bincount(idx(sj, sj), weights=w, minlength=n * n)
    cross = np.bincount(idx(si, sj), weights=-w * Ji, minlength=n * n)
    H = Hf.reshape(n, n) + cross.reshape(n, n) + cross.reshape(n, n).T
    g = np.bincount(si, weights=w * Ji * r, minlength=n) + np.bincount(sj, weights=-w * r, minlength=n)
    return H, g


def _pair_aggregates(pair_ids, npair, w, Ji, r):
    bc = lambda x: np.bincount(pair_ids, weights=x, minlength=npair)
    return {"hii": bc(w * Ji * Ji), "hjj": bc(w), "hij": bc(-w * Ji), "gi": bc(w * Ji * r), "gj": bc(-w * r)}


def _assemble(ij, agg, n, keep):
    H = np.zeros((n, n)); g = np.zeros(n)
    for p in np.nonzero(keep)[0]:
        i, j = ij[p]
        H[i, i] += agg["hii"][p]; H[j, j] += agg["hjj"][p]; H[i, j] += agg["hij"][p]; H[j, i] += agg["hij"][p]
        g[i] += agg["gi"][p]; g[j] += agg["gj"][p]
    return H, g


# ---------------------------------------------------------------------------------------------------------------- (c) focal
def _focal_cost_factory(S, K0, flat, logscale, fit_principal_point):
    si, sj, xs, ys = flat

    def residuals(theta):
        K = _scaled_K(K0, theta)
        z, _b, _a, px, py = S.project(K, si, xs, ys, sj, logscale)
        val, ok = S.sample_dest(sj, px, py)
        ok &= np.isfinite(z) & (z > 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.log(np.where(z > 0, z, 1.0)) - val - logscale[sj], 0.0)
        return r, ok

    def cost(theta, keep=None):
        r, ok = residuals(theta)
        if keep is not None:
            ok = ok & keep
        c = np.log1p((r / FOCAL_DELTA) ** 2)
        return float(np.sum(c[ok]) / max(int(ok.sum()), 1)) + 1.0 * float(not ok.any())

    return residuals, cost


def estimate_intrinsics(prediction, log_scale=None, fit_principal_point=False, sample_stride=SAMPLE_STRIDE + 4, neighbours=4,
                        refuse_factor=REFUSE_FACTOR, loo=True, max_samples=60000) -> dict:
    """Self-calibrate a common relative factor on every view's fx and fy (and optionally a common cx, cy shift in pixels) by
    minimising the multi-view depth-reprojection residual (bounded loss). Leave-one-view-out re-optimisation gives the sd."""
    from scipy.optimize import minimize
    n, h, w = _check(prediction)
    depth = np.asarray(prediction["depth"], float); conf = np.asarray(prediction["confidence"], float)
    K0 = np.array(prediction["intrinsics"], float)
    _ok, smooth = _valid_masks(depth, conf)
    S = _Samples(depth, smooth, prediction["extrinsics"])
    grid = S.grid(sample_stride)
    pairs, _c = _select_pairs(S, K0, grid, neighbours, MIN_PAIR_INLIERS)
    if n < 3 or len(pairs) < 2:
        return {"status": "not_identifiable", "reason": "too_few_overlapping_views", "n_frames": n}
    flat = _flat_samples(grid, pairs)
    if len(flat[0]) > max_samples:                       # deterministic cap keeps the optimiser fast
        keep = np.sort(np.random.default_rng(0).choice(len(flat[0]), max_samples, replace=False))
        flat = tuple(a[keep] for a in flat)
    ls = np.zeros(n) if log_scale is None else np.asarray(log_scale, float)
    _res, cost = _focal_cost_factory(S, K0, flat, ls, fit_principal_point)
    dim = 4 if fit_principal_point else 2
    x0 = np.zeros(dim)
    names = ["log_fx_factor", "log_fy_factor"] + (["cx_shift_px", "cy_shift_px"] if fit_principal_point else [])
    steps = np.array([0.01, 0.01] + ([2.0, 2.0] if fit_principal_point else []))

    def optimise(keep, start, step_scale, maxiter):
        simplex = np.vstack([start] + [start + np.eye(dim)[i] * steps[i] * step_scale for i in range(dim)])
        res = minimize(lambda t: cost(t, keep), start, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 2e-4, "fatol": 1e-7, "maxiter": maxiter})
        return res.x, float(res.fun)

    theta, fbest = optimise(None, x0, 1.0, 120)
    f0 = cost(x0)
    out = {"status": "ok", "n_frames": n, "n_pairs": len(pairs), "n_samples": int(len(flat[0])), "parameters": names,
           "estimate": theta.tolist(), "cost_at_identity": f0, "cost_at_estimate": fbest,
           "fx_over_fy_factor": math.exp(theta[0] - theta[1]),
           "assumptions": "one common relative factor on every view's fx and fy; poses and per-frame scales as given"}
    if loo:
        si_ = flat[0]; sj_ = flat[1]
        est = []
        for k in range(n):
            keep = (si_ != k) & (sj_ != k)
            tk, _f = optimise(keep, theta, 0.4, 40)
            est.append(tk)
        est = np.array(est)
        sd = np.sqrt((n - 1) / n * np.sum((est - est.mean(axis=0)) ** 2, axis=0))
        out["sd_leave_one_view_out"] = sd.tolist()
        out["leave_one_out_estimates"] = est.tolist()
        applied = [bool(sd[i] <= refuse_factor * abs(theta[i])) for i in range(dim)]
    else:
        out["sd_leave_one_view_out"] = [None] * dim
        applied = [True] * dim
    out["applied"] = dict(zip(names, applied))
    out["to_apply"] = [float(theta[i]) if applied[i] else 0.0 for i in range(dim)]
    out["not_identifiable"] = [names[i] for i in range(dim) if not applied[i]]
    return out


# ---------------------------------------------------------------------------------------------------------------- (b) surface
def _unproject(depth, K, E, mask, stride, offset=None):
    n, h, w = depth.shape
    yy, xx = np.mgrid[(offset if offset is not None else stride // 2):h:stride, (offset if offset is not None else stride // 2):w:stride]
    pts, fid = [], []
    for i in range(n):
        m = mask[i, yy, xx]
        z = depth[i, yy, xx][m].astype(float)
        if not len(z):
            continue
        x = (xx[m] - K[i, 0, 2]) / K[i, 0, 0] * z; y = (yy[m] - K[i, 1, 2]) / K[i, 1, 1] * z
        cam = np.stack([x, y, z], axis=1)
        R, t = np.asarray(E[i], float)[:, :3], np.asarray(E[i], float)[:, 3]
        pts.append((cam - t) @ R); fid.append(np.full(len(z), i))
    return np.concatenate(pts), np.concatenate(fid)


def classify_planes(planes, axis, min_fraction=0.04):
    """cap (floor/ceiling-like, normal parallel to axis) / side (wall-like, normal perpendicular) / other, by the caller's axis."""
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    total = sum(max(0, int(p.get("support_count", 0))) for p in planes) or 1
    rows = []
    for k, p in enumerate(planes):
        nrm = np.asarray(p["normal"], float); nrm = nrm / np.linalg.norm(nrm)
        c = abs(float(nrm @ axis))
        share = p.get("support_count", 0) / total
        kind = "cap" if c >= CAP_COS else "side" if c <= WALL_SIN else "other"
        if share < min_fraction:
            kind = "minor_" + kind
        rows.append({"plane_index": k, "class": kind, "normal": nrm.tolist(), "offset": float(p["offset"]) / float(np.linalg.norm(p["normal"])),
                     "support_fraction": float(share), "angle_to_axis_deg": float(np.degrees(np.arccos(min(1.0, c))))})
    return rows


def _structure_check(rows):
    """Diagnostic only (not used for estimation): orthogonality / parallelism of the room's own planes."""
    caps = [np.array(r["normal"]) for r in rows if r["class"] == "cap"]
    sides = [np.array(r["normal"]) for r in rows if r["class"] == "side"]
    ang = lambda a, b: float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))
    out = {"cap_cap_angle_deg": [ang(a, b) for i, a in enumerate(caps) for b in caps[i + 1:]],
           "cap_side_deviation_from_90_deg": [abs(90.0 - ang(a, b)) for a in caps for b in sides]}
    dev = []
    for i, a in enumerate(sides):
        for b in sides[i + 1:]:
            x = ang(a, b)
            if 60 <= x <= 120 or x >= 80:
                dev.append(abs(90.0 - x) if x > 45 else None)
    out["wall_wall_deviation_from_90_deg_for_near_orthogonal_pairs"] = [d for d in dev if d is not None]
    return out


def _label_frames(depth, valid, K, E, planes, rows, tau):
    """Plane label (index into planes, -1 none) on the stride-LABEL_STRIDE grid for every frame."""
    n, h, w = depth.shape
    s = LABEL_STRIDE
    yy, xx = np.mgrid[s // 2:h:s, s // 2:w:s]
    labels = np.full((n, yy.shape[0], yy.shape[1]), -1, np.int32)
    normals = np.array([r["normal"] for r in rows]); offs = np.array([r["offset"] for r in rows])
    use = np.array([not r["class"].startswith("minor_") and r["class"] in ("cap", "side") for r in rows])
    for i in range(n):
        z = depth[i, yy, xx].astype(float)
        m = valid[i, yy, xx]
        x = (xx - K[i, 0, 2]) / K[i, 0, 0] * z; y = (yy - K[i, 1, 2]) / K[i, 1, 1] * z
        R, t = np.asarray(E[i], float)[:, :3], np.asarray(E[i], float)[:, 3]
        X = (np.stack([x, y, z], axis=-1) - t) @ R
        dist = np.abs(X @ normals.T + offs)                       # h x w x P
        dist[..., ~use] = np.inf
        best = np.argmin(dist, axis=-1)
        bd = np.take_along_axis(dist, best[..., None], axis=-1)[..., 0]
        labels[i] = np.where(m & (bd < tau), best, -1)
    return labels, (yy, xx)


def _seam_samples(labels, grid, K, E, rows):
    """log(depth_cap_plane / depth_side_plane) along seam rays, per view and cap plane."""
    yy, xx = grid
    s = LABEL_STRIDE
    cls = np.array([r["class"] for r in rows])
    normals = np.array([r["normal"] for r in rows]); offs = np.array([r["offset"] for r in rows])
    out = {}
    for i in range(labels.shape[0]):
        L = labels[i]
        R, t = np.asarray(E[i], float)[:, :3], np.asarray(E[i], float)[:, 3]
        C = -R.T @ t
        for dy, dx in SEAM_OFFSETS:
            A = L[:L.shape[0] - dy, :L.shape[1] - dx]; B = L[dy:, dx:]
            ya = yy[:L.shape[0] - dy, :L.shape[1] - dx]; xa = xx[:L.shape[0] - dy, :L.shape[1] - dx]
            ok = (A >= 0) & (B >= 0)
            ca = np.where(ok, cls[np.where(A >= 0, A, 0)], ""); cb = np.where(ok, cls[np.where(B >= 0, B, 0)], "")
            for first_cap in (True, False):
                m = ok & ((ca == "cap") & (cb == "side") if first_cap else (ca == "side") & (cb == "cap"))
                if not m.any():
                    continue
                cap_id = (A if first_cap else B)[m]; side_id = (B if first_cap else A)[m]
                u = xa[m] + dx * s / 2.0; v = ya[m] + dy * s / 2.0
                rays = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0], (v - K[i, 1, 2]) / K[i, 1, 1], np.ones(len(u))], axis=1)
                dirw = rays @ R                                   # R^T ray for row vectors
                def plane_depth(pid):
                    nr = normals[pid]; of = offs[pid]
                    den = np.einsum("ij,ij->i", nr, dirw)
                    graz = np.abs(den) / np.linalg.norm(dirw, axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lam = -((nr @ C) + of) / den
                    return lam, graz
                dc, gc = plane_depth(cap_id); ds, gs = plane_depth(side_id)
                good = (dc > 0) & (ds > 0) & (gc >= SEAM_MIN_GRAZING) & (gs >= SEAM_MIN_GRAZING) & np.isfinite(dc) & np.isfinite(ds)
                with np.errstate(invalid="ignore", divide="ignore"):
                    lr = np.log(dc / ds)
                good &= np.abs(lr) < SEAM_MAX_LOG
                for pid in np.unique(cap_id[good]):
                    sel = good & (cap_id == pid)
                    out.setdefault(int(pid), {}).setdefault(i, []).append(lr[sel])
    return {p: {v: np.concatenate(a) for v, a in d.items()} for p, d in out.items()}


def _jackknife_median(view_values):
    vals = np.array(view_values)
    est = float(np.median(vals))
    if len(vals) < 3:
        return est, float("nan")
    loo = np.array([np.median(np.delete(vals, k)) for k in range(len(vals))])
    sd = math.sqrt((len(vals) - 1) / len(vals) * float(np.sum((loo - loo.mean()) ** 2)))
    return est, sd


def _seam_diagnostic(labels, grid, K, E, rows, refuse_factor):
    """Seam-closure cross-check (diagnostic only, never applied): depth ratio cap-plane / wall-plane along seam rays."""
    seams = _seam_samples(labels, grid, K, E, rows)
    out = {}
    for pid, per_view in sorted(seams.items()):
        if rows[pid]["class"] != "cap":
            continue
        views = [v for v, a in per_view.items() if len(a) >= MIN_SEAM_SAMPLES_PER_VIEW]
        rec = {"n_views_with_seam": len(views), "n_seam_samples": int(sum(len(per_view[v]) for v in views))}
        if len(views) >= MIN_SEAM_VIEWS:
            est, sd = _jackknife_median([float(np.median(per_view[v])) for v in views])
            rec.update(log_ratio_cap_over_wall=est, ratio_cap_over_wall=math.exp(est), sd_leave_one_view_out=sd,
                       would_pass_identifiability=bool(np.isfinite(sd) and sd <= refuse_factor * abs(est)))
        else:
            rec.update(log_ratio_cap_over_wall=None, would_pass_identifiability=False)
        out[pid] = rec
    return out


# ---- plane sweep (photometric multi-view consistency of a fitted plane) ------------------------------------------------
SWEEP_EPS = np.round(np.linspace(-0.08, 0.08, 41), 6)      # plane moved away from the cameras by eps * mean camera-plane distance
SWEEP_MAX_PER_VIEW = 300
SWEEP_DESTINATIONS = 4
SWEEP_MIN_VIEWS = 4
SWEEP_MIN_BASELINE = 0.05                                   # camera baseline as a fraction of the mean camera-plane distance
SWEEP_CLIP = 2.0
SWEEP_WEIGHT_SIGMA = 6.0
SWEEP_NORM_WINDOW = 7


def normalised_images(images):
    """Locally contrast-normalised gray images and a texture weight in [0, 1] (flat patches carry no photometric information)."""
    from scipy.ndimage import uniform_filter
    img = np.asarray(images, np.float32)
    gray = img.mean(axis=-1) if img.ndim == 4 else img
    if gray.max() <= 1.5:
        gray = gray * 255.0
    mu = np.stack([uniform_filter(g, SWEEP_NORM_WINDOW) for g in gray]); m2 = np.stack([uniform_filter(g * g, SWEEP_NORM_WINDOW) for g in gray])
    sigma = np.sqrt(np.maximum(m2 - mu * mu, 0.0))
    return ((gray - mu) / (sigma + 2.0)).astype(np.float32), np.clip(sigma / SWEEP_WEIGHT_SIGMA, 0.0, 1.0).astype(np.float32)


def _bilinear_plain(img, si, px, py):
    n, h, w = img.shape
    inside = np.isfinite(px) & np.isfinite(py) & (px >= 0) & (py >= 0) & (px <= w - 1.001) & (py <= h - 1.001)
    x0 = np.floor(np.where(inside, px, 0)).astype(int); y0 = np.floor(np.where(inside, py, 0)).astype(int)
    fx = np.where(inside, px, 0) - x0; fy = np.where(inside, py, 0) - y0
    val = ((1 - fx) * (1 - fy) * img[si, y0, x0] + fx * (1 - fy) * img[si, y0, x0 + 1]
           + (1 - fx) * fy * img[si, y0 + 1, x0] + fx * fy * img[si, y0 + 1, x0 + 1])
    return val, inside


def _plane_side(rows_plane, C):
    nrm = np.asarray(rows_plane["normal"], float); off = float(rows_plane["offset"])
    h = C @ nrm + off
    sigma = 1.0 if np.median(h) >= 0 else -1.0
    same = float(np.mean(sigma * h > 0))
    return nrm, off, sigma, float(np.mean(np.abs(h))), same


def plane_sweep(rows_plane, labels_p, grid, K, E, norm_img, wt_img, refuse_factor=REFUSE_FACTOR) -> dict:
    """Photometric plane sweep of ONE fitted plane. The plane is moved along its normal (away from the cameras by eps times the
    mean camera-plane distance); each labelled source pixel is placed on the moved plane, projected into neighbouring views and
    compared (locally contrast-normalised intensity). eps minimising the robust photometric cost is the plane's offset in the units
    the POSES define. Leave-one-view-out gives the sd."""
    C = camera_centres(E)
    nrm, off, sigma, hbar, same = _plane_side(rows_plane, C)
    if same < 0.9:
        return {"status": "not_identifiable", "reason": "cameras_on_both_sides_of_plane"}
    yy, xx = grid
    n = labels_p.shape[0]
    R = np.asarray(E, float)[:, :, :3]; T = np.asarray(E, float)[:, :, 3]
    srcs = []
    for i in range(n):
        m = labels_p[i]
        v, u = yy[m], xx[m]
        if len(u) > SWEEP_MAX_PER_VIEW:
            sel = np.linspace(0, len(u) - 1, SWEEP_MAX_PER_VIEW).astype(int)
            v, u = v[sel], u[sel]
        srcs.append((u.astype(float), v.astype(float)))
    views = [i for i in range(n) if len(srcs[i][0]) >= 20]
    if len(views) < SWEEP_MIN_VIEWS:
        return {"status": "not_identifiable", "reason": "too_few_views_see_the_plane", "n_views": len(views)}

    def place(i, u, v, eps):
        rays = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0], (v - K[i, 1, 2]) / K[i, 1, 1], np.ones(len(u))], axis=1)
        dirw = rays @ R[i]
        den = dirw @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = (-sigma * eps * hbar - (nrm @ C[i] + off)) / den
        lam = np.where(lam > 0, lam, np.nan)
        return C[i] + lam[:, None] * dirw

    def project(j, X):
        Xc = X @ R[j].T + T[j]
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            return K[j, 0, 0] * Xc[:, 0] / z + K[j, 0, 2], K[j, 1, 1] * Xc[:, 1] / z + K[j, 1, 2], z

    # destinations: most overlapping views with a useful baseline (decided at eps = 0)
    pairs = []
    for i in views:
        X0 = place(i, srcs[i][0], srcs[i][1], 0.0)
        score = []
        for j in views:
            base = float(np.linalg.norm(C[i] - C[j]))
            if j == i or base < SWEEP_MIN_BASELINE * hbar:
                continue
            px, py, z = project(j, X0)
            inside = np.isfinite(px) & (z > 0) & (px >= 0) & (py >= 0) & (px <= K[j, 0, 2] * 2 - 1) & (py <= K[j, 1, 2] * 2 - 1)
            frac = float(np.mean(inside))
            if frac >= 0.3:
                score.append((frac * min(base / hbar, 1.0), j))
        for _s, j in sorted(score, reverse=True)[:SWEEP_DESTINATIONS]:
            pairs.append((i, j))
    if len(pairs) < SWEEP_MIN_VIEWS:
        return {"status": "not_identifiable", "reason": "no_pairs_with_baseline_and_overlap"}
    P = len(pairs)
    pid_list, wlist = [], []
    eps_grid = SWEEP_EPS
    cost = np.zeros((len(eps_grid), 0)); valid = np.ones((len(eps_grid), 0), bool)
    costs, valids, pids, weights = [], [], [], []
    for p, (i, j) in enumerate(pairs):
        u, v = srcs[i]
        src_n, ok_s = _bilinear_plain(norm_img, np.full(len(u), i), u, v)
        wgt = _bilinear_plain(wt_img, np.full(len(u), i), u, v)[0]
        c = np.zeros((len(eps_grid), len(u))); vv = np.zeros((len(eps_grid), len(u)), bool)
        for e, eps in enumerate(eps_grid):
            X = place(i, u, v, eps)
            px, py, z = project(j, X)
            dst_n, ok_d = _bilinear_plain(norm_img, np.full(len(u), j), px, py)
            vv[e] = ok_d & np.isfinite(z) & (z > 0)
            c[e] = np.minimum(np.abs(src_n - dst_n), SWEEP_CLIP)
        costs.append(c); valids.append(vv); pids.append(np.full(len(u), p)); weights.append(wgt)
    cost = np.concatenate(costs, axis=1); valid = np.concatenate(valids, axis=1).all(axis=0)
    pid = np.concatenate(pids); wgt = np.concatenate(weights) * valid
    if wgt.sum() < 100:
        return {"status": "not_identifiable", "reason": "too_little_texture_or_overlap", "n_samples": int(valid.sum())}
    A = np.stack([np.bincount(pid, weights=wgt * cost[e], minlength=P) for e in range(len(eps_grid))])      # [E, P]
    B = np.bincount(pid, weights=wgt, minlength=P)
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])

    def argmin_curve(mask):
        curve = A[:, mask].sum(axis=1) / max(B[mask].sum(), 1e-9)
        k = int(np.argmin(curve))
        if k == 0 or k == len(eps_grid) - 1:
            return float(eps_grid[k]), curve, True
        y0, y1, y2 = curve[k - 1], curve[k], curve[k + 1]
        den = y0 - 2 * y1 + y2
        step = eps_grid[1] - eps_grid[0]
        shift = 0.5 * (y0 - y2) / den if den > 1e-12 else 0.0
        return float(eps_grid[k] + np.clip(shift, -1, 1) * step), curve, False

    est, curve, at_edge = argmin_curve(np.ones(P, bool))
    out = {"status": "ok", "eps": est, "n_views": len(views), "n_pairs": P, "n_samples": int((wgt > 0).sum()),
           "mean_camera_plane_distance": hbar, "sigma_side": sigma, "min_at_grid_edge": at_edge,
           "curve_contrast": float(curve.max() - curve.min()), "cost_at_zero": float(curve[len(eps_grid) // 2]), "cost_at_min": float(curve.min())}
    loo = []
    for k in views:
        e_k, _c, edge_k = argmin_curve((pi != k) & (pj != k))
        loo.append(e_k)
    loo = np.array(loo)
    out["sd_leave_one_view_out"] = float(math.sqrt((len(loo) - 1) / len(loo) * np.sum((loo - loo.mean()) ** 2)))
    out["identifiable"] = bool(not at_edge and out["sd_leave_one_view_out"] <= refuse_factor * abs(est))
    return out


def estimate_surface_gains(prediction, axis, K=None, depth=None, planes=None, refuse_factor=REFUSE_FACTOR, seed=0) -> dict:
    """Depth offset of every cap (floor/ceiling-like) plane relative to the wall-like baseline.

    Primary evidence: photometric plane sweep (needs prediction['images']); wall-like planes define the baseline (their median
    sweep offset, which also absorbs any depth-vs-pose unit mismatch). Cross-check only: seam closure."""
    import extract_planes
    n, h, w = _check(prediction)
    depth = np.asarray(prediction["depth"] if depth is None else depth, float)
    conf = np.asarray(prediction["confidence"], float)
    K = np.array(prediction["intrinsics"], float) if K is None else np.asarray(K, float)
    E = np.asarray(prediction["extrinsics"], float)
    valid, smooth = _valid_masks(depth, conf)
    if planes is None:
        pts, _f = _unproject(depth, K, E, smooth, 6)
        if len(pts) < 500:
            return {"status": "not_identifiable", "reason": "too_few_points", "gains": []}
        try:
            planes = extract_planes.extract(pts, seed=seed)
        except ValueError as exc:
            return {"status": "not_identifiable", "reason": f"plane_extraction_failed:{exc}", "gains": []}
    rows = classify_planes(planes["planes"], axis)
    tau = 1.5 * float(planes["threshold"])
    labels, grid = _label_frames(depth, valid, K, E, planes["planes"], rows, tau)
    seam = _seam_diagnostic(labels, grid, K, E, rows, refuse_factor)
    base = {"status": "ok", "plane_classes": rows, "structure_check": _structure_check(rows), "n_planes": len(rows),
            "label_stride": LABEL_STRIDE, "wall_baseline_assumption": True, "seam_closure_cross_check_not_applied": seam,
            "_labels": labels, "_planes": planes, "_grid": grid}
    if "images" not in prediction or prediction["images"] is None or np.asarray(prediction["images"]).shape[:3] != (n, h, w):
        return {**base, "gains": [], "baseline": None, "reason": "no_images_for_photometric_plane_sweep"}
    norm_img, wt_img = normalised_images(prediction["images"])
    sweeps = {}
    for r in rows:
        if r["class"] in ("cap", "side"):
            sweeps[r["plane_index"]] = {**plane_sweep(r, labels == r["plane_index"], grid, K, E, norm_img, wt_img, refuse_factor),
                                        "class": r["class"]}
    walls = [s for s in sweeps.values() if s["class"] == "side" and s.get("status") == "ok" and not s.get("min_at_grid_edge")]
    if walls:
        b_est = float(np.median([s["eps"] for s in walls]))
        b_sd = float(1.25 * np.median([s["sd_leave_one_view_out"] for s in walls]) / math.sqrt(len(walls)))
        baseline = {"eps_wall_median": b_est, "sd": b_sd, "n_wall_planes": len(walls)}
    else:
        baseline = None
    gains = []
    for pid, s in sorted(sweeps.items()):
        if s["class"] != "cap":
            continue
        rec = {"plane_index": pid, "angle_to_axis_deg": rows[pid]["angle_to_axis_deg"], "support_fraction": rows[pid]["support_fraction"],
               "sweep": {k: v for k, v in s.items()}, "seam": seam.get(pid), "applied": False, "shift_fraction_to_apply": 0.0}
        if s.get("status") != "ok" or baseline is None:
            rec.update(status="not_identifiable", reason=s.get("reason") or "no_wall_baseline")
        else:
            rel = s["eps"] - baseline["eps_wall_median"]
            sd = math.sqrt(s["sd_leave_one_view_out"] ** 2 + baseline["sd"] ** 2)
            ok = bool(s["identifiable"] and sd <= refuse_factor * abs(rel))
            rec.update(eps_relative_to_walls=rel, sd_relative=sd, applied=ok, status="applied" if ok else "not_identifiable",
                       reason=None if ok else ("sweep_minimum_at_grid_edge" if s.get("min_at_grid_edge") else "uncertainty_exceeds_magnitude"),
                       shift_fraction_to_apply=float(rel) if ok else 0.0)
        gains.append(rec)
    return {**base, "gains": gains, "baseline": baseline, "sweeps_by_plane": {int(k): v for k, v in sweeps.items()}}


def apply_surface_gains(depth, labels, gains, planes_rows, grid, K, E):
    """Move every gained cap plane away from the cameras by shift_fraction * mean camera-plane distance and rescale the depth of
    its labelled pixels (grown by one label cell into UNLABELLED neighbours only) by the ratio of the ray's distances to the
    moved and the fitted plane. Pixels labelled with any other plane are never touched; poses are untouched."""
    from scipy.ndimage import binary_dilation
    n, h, w = depth.shape
    out = np.array(depth, float)
    use = [g for g in gains if g.get("applied") and abs(g.get("shift_fraction_to_apply", 0.0)) > 0]
    if not use:
        return out
    C = camera_centres(E)
    s = LABEL_STRIDE
    for i in range(n):
        cell = np.zeros(labels[i].shape)
        for g in use:
            pid = g["plane_index"]
            m = labels[i] == pid
            grown = binary_dilation(m, iterations=1) & ((labels[i] < 0) | m)
            if not grown.any():
                continue
            nrm, off, sigma, hbar, _same = _plane_side(planes_rows[pid], C)
            num0 = -(float(C[i] @ nrm) + off)           # the distance ratio of moved/fitted plane is the same along every ray of a camera
            ratio = (num0 - sigma * g["shift_fraction_to_apply"] * hbar) / num0 if abs(num0) > 1e-12 else 1.0
            cell[grown & (cell == 0)] = float(np.log(np.clip(ratio, 0.8, 1.25)))
        if not np.any(cell):
            continue
        full = np.kron(cell, np.ones((s, s)))
        full = np.pad(full, ((0, max(h - full.shape[0], 0)), (0, max(w - full.shape[1], 0))), mode="edge")[:h, :w]
        out[i] = out[i] * np.exp(full)
    return out


# ---------------------------------------------------------------------------------------------------------------- driver
def correct_prediction(prediction, axis, **kwargs):
    """Opt-in correction for ONE structural-axis hypothesis. Returns (corrected_prediction, report). See correct_prediction_axes."""
    return correct_prediction_axes(prediction, [axis], **kwargs)[0]


def correct_prediction_axes(prediction, axes, estimate_scales=True, estimate_surfaces=True, estimate_focal=True,
                            fit_principal_point=False, surface_iterations=1, refuse_factor=REFUSE_FACTOR, loo=True,
                            sample_stride=SAMPLE_STRIDE):
    """Opt-in correction; one (corrected_prediction, report) per axis hypothesis. The focal and per-frame-scale stages do not
    depend on the axis and are computed once; only the surface stage uses the axis (to class planes as cap / wall-like).
    Poses, confidence and images are returned unchanged; only 'depth' and 'intrinsics' can change, and only by parameters that
    pass the identifiability rule (jackknife sd <= refuse_factor * |value|)."""
    n, h, w = _check(prediction)
    unit_axes = []
    for axis in axes:
        axis = np.asarray(axis, float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise SurfaceScaleError("axis must be a finite 3-vector (a geometry_orientation hypothesis)")
        unit_axes.append(axis / np.linalg.norm(axis))
    depth = np.array(prediction["depth"], np.float64)
    K = np.array(prediction["intrinsics"], float)
    common = {"model": MODEL, "uses_ground_truth": False, "uses_simulator": False, "uses_camera_up": False,
              "uses_camera_path_gravity": False, "poses_modified": False, "n_frames": n, "refuse_factor": refuse_factor,
              "assumptions": ["walls (side-class planes) are the depth baseline; only relative cap offsets are identifiable",
                              "a cap plane's offset is a rigid move of the fitted plane along its normal, measured photometrically in the "
                              "units the (fixed) poses define; the seam-closure ratio is only a cross-check",
                              "surface offsets need image texture and poses accurate to ~1 %: otherwise they are reported not identifiable",
                              "one common relative factor on fx, fy for all views",
                              "global depth scale is left untouched (mean per-frame log-scale = 0)"],
              "not_identifiable": []}
    E_ = np.asarray(prediction["extrinsics"], float)
    log_scale = np.zeros(n)
    if estimate_scales:                        # preliminary scales with the given K feed the focal stage
        prelim = estimate_frame_scales(prediction, K=K, sample_stride=sample_stride, loo=False)
        if prelim.get("status") == "ok":
            log_scale = np.array(prelim["log_scale_to_apply"])
    if estimate_focal:
        foc = estimate_intrinsics(prediction, log_scale=log_scale, fit_principal_point=fit_principal_point, refuse_factor=refuse_factor,
                                  loo=loo, sample_stride=sample_stride + 4)
        common["intrinsics"] = foc
        if foc.get("status") == "ok":
            K = _scaled_K(K, foc["to_apply"])
            common["not_identifiable"] += [f"intrinsics:{p}" for p in foc["not_identifiable"]]
        else:
            common["not_identifiable"].append("intrinsics")
    if estimate_scales:
        sc = estimate_frame_scales(prediction, K=K, sample_stride=sample_stride, refuse_factor=refuse_factor, loo=loo)
        common["frame_scales"] = sc
        if sc.get("status") == "ok":
            depth = depth * np.exp(np.array(sc["log_scale_to_apply"]))[:, None, None]
            common["not_identifiable"] += [f"frame_scale:{i}" for i in sc["not_identifiable_frames"]]
        else:
            common["not_identifiable"].append("frame_scales")
    before = np.asarray(prediction["depth"], float)
    results = []
    for axis in unit_axes:
        d_axis = depth.copy()
        report = json_safe(common)
        report["axis_input"] = axis.tolist()
        if estimate_surfaces:
            iters = []
            for it in range(max(1, surface_iterations)):
                surf = estimate_surface_gains(prediction, axis, K=K, depth=d_axis, planes=None, refuse_factor=refuse_factor)
                rec = {k: v for k, v in surf.items() if not k.startswith("_")}
                rec["iteration"] = it
                iters.append(rec)
                if surf.get("status") != "ok":
                    report["not_identifiable"].append("surface_gains")
                    break
                if not any(g.get("applied") for g in surf["gains"]):
                    break
                d_axis = apply_surface_gains(d_axis, surf["_labels"], surf["gains"], surf["plane_classes"], surf["_grid"], K, E_)
            residual = None
            if iters and iters[-1].get("status") == "ok" and any(g.get("applied") for g in iters[-1]["gains"]):
                chk = estimate_surface_gains(prediction, axis, K=K, depth=d_axis, planes=None, refuse_factor=refuse_factor)
                residual = {k: v for k, v in chk.items() if not k.startswith("_")}
            report["surface_gains"] = {"iterations": json_safe(iters), "wall_baseline": True,
                                       "re_estimate_after_correction_diagnostic_only_not_applied": json_safe(residual)}
            report["not_identifiable"] += [f"surface_gain:plane{g['plane_index']}:iter{r['iteration']}" for r in iters
                                           for g in r.get("gains", []) if not g.get("applied")]
            if iters and not iters[0].get("gains") and iters[0].get("reason"):
                report["not_identifiable"].append(f"surface_gains:{iters[0]['reason']}")
        corrected = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in prediction.items()}
        corrected["depth"] = d_axis.astype(before.dtype if np.asarray(prediction["depth"]).dtype.kind == "f" else np.float32)
        corrected["intrinsics"] = K.astype(np.asarray(prediction["intrinsics"]).dtype)
        both = np.isfinite(before) & (before > 0) & np.isfinite(d_axis)
        report["summary"] = {
            "median_abs_log_depth_change": float(np.median(np.abs(np.log(d_axis[both] / before[both])))) if both.any() else 0.0,
            "intrinsics_changed": bool(np.any(corrected["intrinsics"] != np.asarray(prediction["intrinsics"]))),
            "surface_gains_applied": [g["plane_index"] for r in report.get("surface_gains", {}).get("iterations", []) for g in r.get("gains", [])
                                      if g.get("applied")]}
        results.append((corrected, report))
    return results


def load_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}
