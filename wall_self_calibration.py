"""Truth-free wall-consistency self-calibration of a DA3-format prediction (depth scale + yaw + in-plane translation per frame).

Why: DA3 frames of one video are mutually inconsistent (per-frame depth scale and pose drift). The wall lines extracted by
wall_extraction are tight (noise ~0.3 % of the extent) but the same physical wall shows up as several same-side near-parallel
"layered" lines several percent of the extent apart (36 x the noise); that breaks polygon closure (rgb-closure-diagnosis-v1).

Method (everything from the prediction and one structural axis; no ground truth, no camera-up, no simulator)
  1. wall_extraction.extract_wall_lines gives the observed lines. The calibration points are re-sampled from the same frames:
     per frame the pixels in the mid-height band whose depth-map normal is wall-like (a floor/ceiling patch leaking into the band
     under pitch error is dropped), one sample per frame and cell, with their in-plane normals.
  2. Surfaces. Every extracted line is a surface; lines that are same-side, near-parallel, laterally overlapping in their OBSERVED
     spans and separated by more than the noise but less than the drift allowance (fragment_association's definition of a
     layered pair) are linked into one hypothesised surface (single linkage). A linked group is kept only when every frame is
     internally single-layer (a frame that itself sees two layers at once - furniture in front of a wall, a pilaster - shows a
     within-frame scatter far above the noise, so the group is split back into separate surfaces and nothing is collapsed).
     Interlocking spans (an alcove, an L-shaped step) never link because their observed spans do not overlap.
  3. Parameters per frame (the identifiable ones only): depth scale s_i = exp(ls_i) about the camera centre, yaw theta_i about
     the structural axis (pivot at the camera centre) and a horizontal translation u_i of the camera centre. The wall points
     constrain exactly these; vertical translation, pitch and roll are almost invisible to horizontal wall lines and stay at the
     DA3 values (documented, not estimated). Gauge: mean log s = 0 (hard); the prior also pins the global yaw / translation.
  4. Objective: Cauchy loss of the signed distance of every wall point to its (iteratively refitted, robustly weighted) surface
     line, annealed from the drift scale to a few noise units, plus a Gaussian prior toward the DA3 values and a random-walk
     smoothness along the video for yaw / translation / scale. Alternating: refit lines -> Gauss-Newton on all frames -> refit.
  5. Uncertainty: leave-one-view-out jackknife (each other view removed in turn, lines refitted) combined with the Laplace
     posterior of the joint problem. A parameter whose uncertainty exceeds its magnitude is REFUSED (set back to the DA3 value,
     as surface_scale_model does) and the rest are re-solved. If the robust cost does not drop, nothing is changed.
  6. Output: same npz schema; depth multiplied per frame, extrinsics (world-to-camera) re-posed per frame by a proper rotation
     about the axis plus a horizontal shift (a reflection is refused: every input rotation must have det > 0). With
     verify=True the calibrated prediction is re-extracted and the result is reverted unless layered pairs and lines do not
     increase.
Thresholds are scale-free (fractions of the cloud extent or multiples of the measured noise scale). Deterministic.
"""
from __future__ import annotations

import math

import numpy as np

import wall_extraction as we

MODEL = "wall_self_calibration_v1"

DEFAULTS = {
    "stride": 4, "conf_percentile": 40.0, "cell_fraction": 0.004, "max_jump": 0.04, "wall_like_deg": 20.0,
    "link_max_angle_deg": 10.0,        # layered pair: near-parallel
    "link_min_overlap": 0.3,           # observed-span overlap / shorter observed length
    "drift_fraction": 0.12,            # layered pair: separation <= this x extent (fragment_association drift allowance)
    "tol_noise_multiple": 3.0,         # layered pair: separation > this x max(noise_scale, median offset_sd)
    "normal_cos": float(np.cos(np.radians(30.0))),   # a point supports a line only if its in-plane normal agrees
    "gate_tau_multiple": 2.0,          # corridor half width around a line, x the extraction tolerance
    "min_line_frames": 5,              # a line is wall evidence only if this many frames support it with wall-like samples
    "min_frame_points": 12,            # a frame supports a group with at least this many samples
    "single_layer_noise": 4.0,         # a frame is single-layer if its own line-fit scatter <= this x noise_scale
    "single_layer_sep_fraction": 0.25, # ... or this fraction of the median layer separation (two layers in one frame scatter ~ 0.5 x it)
    "single_layer_fraction": 0.7,      # share of the frames (point weighted) that must be single-layer to keep a group
    "frame_cap": 60.0,                 # max effective points of one frame in one surface
    "sigma_theta": 0.2,                # prior sd of yaw, rad (~11 deg)
    "sigma_scale": 0.08,               # prior sd of log depth scale
    "sigma_shift": 0.08,               # prior sd of the shift, x extent
    "step_theta": 0.03, "step_scale": 0.02, "step_shift": 0.015,   # random-walk sd per frame step (robust: jumps allowed)
    "smooth_robust": 3.0,              # Cauchy scale of a step difference, x its random-walk sd
    "c0_fraction": 0.05, "c_anneal": 0.7, "c_final_noise": 2.5,    # Cauchy scale: start x extent, factor, final x noise
    "sigma_noise_multiple": 1.5,       # data sd in the cost, x noise_scale
    "knot_every": 6,                   # a parameter knot every this many active frames (piecewise-linear in time)
    "iterations": 14, "gauge_weight": 1e6, "refuse_factor": 1.0,
    "min_cost_drop": 0.1,              # accept only if the data cost falls by this fraction
}


class SelfCalibrationError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------- basics
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _pose_rotations_ok(extr):
    R = np.asarray(extr, float)[:, :3, :3]
    return bool(np.isfinite(R).all() and (np.linalg.det(R) > 0).all())


def camera_centres(extr):
    E = np.asarray(extr, float)
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def _rodrigues(axis, theta):
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------------------------------------------- points
def band_points(prediction, walls, *, stride=4, conf_percentile=40.0, cell_fraction=0.004, max_jump=0.04, wall_like_deg=20.0):
    """Per-frame WALL-LIKE band samples in the plane frame of `walls`. Returns P (n,2), N2 (n,2) unit in-plane normals, F (n,),
    C (N,2) camera plane xy, valid (N,)."""
    fr = walls["frame"]
    o = np.asarray(fr["origin_world"], float)
    e1, e2 = np.asarray(fr["e1"], float), np.asarray(fr["e2"], float)
    a = _unit(walls["axis_world"])
    extent = float(fr["extent"])
    hb = walls["params"]["height_band"]
    depth, extr, intr = prediction["depth"], np.asarray(prediction["extrinsics"], float), np.asarray(prediction["intrinsics"], float)
    conf = prediction.get("confidence")
    N = len(depth)
    Ps, Ns, Fs = [], [], []
    C = np.zeros((N, 2))
    valid = np.zeros(N, bool)
    c0 = extent * cell_fraction
    sin_wall = float(np.sin(np.radians(wall_like_deg)))
    for k in range(N):
        R, t = extr[k][:3, :3], extr[k][:3, 3]
        cen = -R.T @ t
        C[k] = [(cen - o) @ e1, (cen - o) @ e2]
        d = np.asarray(depth[k], float)
        y, x = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        z = d[y, x]
        c = np.asarray(conf[k], float)[y, x] if conf is not None else np.ones_like(z)
        ok = np.isfinite(z) & (z > 0) & np.isfinite(c)
        if ok.sum() < 20:
            continue
        ok &= c >= np.percentile(c[ok], conf_percentile)
        K = intr[k] if intr.ndim == 3 else intr
        pixg = np.stack([x, y, np.ones_like(x)], axis=-1).astype(float)
        cam = (pixg @ np.linalg.inv(K).T) * z[..., None]
        dx = cam[:-1, 1:] - cam[:-1, :-1]
        dy = cam[1:, :-1] - cam[:-1, :-1]
        nrm = np.cross(dx, dy)
        nn = np.linalg.norm(nrm, axis=-1)
        g = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & (nn > 1e-12)
        zc = z[:-1, :-1]
        g &= (np.abs(z[:-1, 1:] - zc) <= max_jump * zc) & (np.abs(z[1:, :-1] - zc) <= max_jump * zc)
        if not g.any():
            continue
        n_world = (nrm[g] / nn[g][:, None]) @ R
        pts = (cam[:-1, :-1][g] - t) @ R - o
        h = pts @ a
        sel = (h >= hb[0]) & (h <= hb[1]) & (np.abs(n_world @ a) <= sin_wall) & np.isfinite(pts).all(axis=1)
        if sel.sum() < 5:
            continue
        xy = np.stack([pts[sel] @ e1, pts[sel] @ e2], axis=1)
        n2 = np.stack([n_world[sel] @ e1, n_world[sel] @ e2], axis=1)
        n2 = n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-12)
        cell = np.floor(xy / c0).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        Ps.append(xy[first])
        Ns.append(n2[first])
        Fs.append(np.full(len(first), k, dtype=np.int64))
        valid[k] = True
    P = np.concatenate(Ps) if Ps else np.zeros((0, 2))
    N2 = np.concatenate(Ns) if Ns else np.zeros((0, 2))
    F = np.concatenate(Fs) if Fs else np.zeros(0, dtype=np.int64)
    return {"P": P, "N2": N2, "F": F, "C": C, "n_frames": N, "valid": valid}


# ---------------------------------------------------------------------------------------------------------------- lines
def _merge_intervals(iv):
    out = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _line_records(walls):
    recs = []
    for ln in walls.get("lines") or []:
        m = np.asarray(ln["normal_xy"], float)
        nm = float(np.linalg.norm(m))
        if nm < 1e-9 or not np.isfinite(m).all():
            continue
        m = m / nm
        c = float(ln["offset"]) / nm
        d = np.array([-m[1], m[0]])
        spans = [[float(a), float(b)] for a, b in ln.get("observed_spans") or []]
        if not spans:
            continue
        sd = float((ln.get("uncertainty") or {}).get("offset_sd") or 0.0)
        recs.append({"id": ln.get("id"), "m": m, "c": c, "d": d, "spans": _merge_intervals(spans), "sd": sd})
    return recs


def _span_world(rec, ref_d):
    """Observed spans of a line expressed along the direction ref_d (a coordinate measured from the origin)."""
    iv = []
    for a, b in rec["spans"]:
        pa = rec["c"] * rec["m"] + a * rec["d"]
        pb = rec["c"] * rec["m"] + b * rec["d"]
        iv.append((float(ref_d @ pa), float(ref_d @ pb)))
    return _merge_intervals(iv)


def _overlap_len(ia, ib):
    tot = 0.0
    for a0, a1 in ia:
        for b0, b1 in ib:
            tot += max(0.0, min(a1, b1) - max(a0, b0))
    return tot


def layered_pairs(walls, *, max_angle_deg=10.0, min_overlap=0.3, drift_fraction=0.12, noise_multiple=3.0):
    """Truth-free same-side layered pairs of an observed_wall_lines dict (observed-span overlap, camera side from the
    observations): near-parallel, separation beyond noise_multiple x noise and within drift_fraction x extent."""
    recs = _line_records(walls)
    fr = walls.get("frame") or {}
    extent = float(fr.get("extent") or 0.0)
    noise = float(walls.get("noise_scale") or 0.0)
    tol = noise_multiple * max(noise, float(np.median([r["sd"] for r in recs])) if recs else 0.0)
    cams = np.asarray((walls.get("observations") or {}).get("camera_xy") or [], float).reshape(-1, 2)
    cen = cams.mean(axis=0) if len(cams) else None
    out = []
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            dot = float(a["m"] @ b["m"])
            ang = math.degrees(math.acos(min(1.0, abs(dot))))
            if ang > max_angle_deg:
                continue
            mb, cb = (b["m"], b["c"]) if dot >= 0 else (-b["m"], -b["c"])
            ia, ib = _span_world(a, a["d"]), _span_world({**b, "m": mb, "c": cb}, a["d"])
            shorter = min(sum(y - x for x, y in ia), sum(y - x for x, y in ib))
            if shorter <= 0:
                continue
            ov = _overlap_len(ia, ib)
            if ov < min_overlap * shorter:
                continue
            lo = max(min(x for x, _ in ia), min(x for x, _ in ib))
            hi = min(max(y for _, y in ia), max(y for _, y in ib))
            tm = 0.5 * (lo + hi)
            base = a["c"] * a["m"]
            p = base + (tm - float(a["d"] @ base)) * a["d"]
            sep = abs(float(mb @ p) - cb)
            same = None if cen is None else bool((float(a["m"] @ cen) - a["c"]) * (float(mb @ cen) - cb) > 0)
            if same is False or sep <= tol or (extent and sep > drift_fraction * extent):
                continue
            out.append({"line_a": a["id"], "line_b": b["id"], "separation": sep, "angle_deg": ang,
                        "separation_over_extent": sep / extent if extent else None, "overlap_over_shorter": ov / shorter})
    return out


def _line_support(B, recs, tau, p):
    """Number of frames with at least min_frame_points wall-like samples (normal agreeing, inside the corridor and the observed
    spans) of every line: an extracted line that few frames support with wall-like pixels is not wall evidence."""
    out = []
    P, N2, F = B["P"], B["N2"], B["F"]
    gate = p["gate_tau_multiple"] * tau
    for r in recs:
        t = P @ r["d"]
        along = np.zeros(len(P), bool)
        for a, b in r["spans"]:
            along |= (t >= a - tau) & (t <= b + tau)
        sel = (np.abs(P @ r["m"] - r["c"]) < gate) & along & (np.abs(N2 @ r["m"]) >= p["normal_cos"])
        cnt = np.bincount(F[sel], minlength=B["n_frames"])
        out.append(int((cnt >= p["min_frame_points"]).sum()))
    return out


def _link_groups(recs, walls, p, pair_ok=None):
    """Single-linkage groups of layered same-side pairs that pass `pair_ok` (a lone line is its own group).
    Returns (groups of indices into recs, all layered pairs, rejected pairs)."""
    pairs = layered_pairs(walls, max_angle_deg=p["link_max_angle_deg"], min_overlap=p["link_min_overlap"],
                          drift_fraction=p["drift_fraction"], noise_multiple=p["tol_noise_multiple"])
    index = {r["id"]: k for k, r in enumerate(recs)}
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rejected = []
    for pr in pairs:
        if pair_ok is not None and not pair_ok(index[pr["line_a"]], index[pr["line_b"]], pr):
            rejected.append([pr["line_a"], pr["line_b"]])
            continue
        ra, rb = find(index[pr["line_a"]]), find(index[pr["line_b"]])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for k in range(len(recs)):
        groups.setdefault(find(k), []).append(k)
    return [sorted(v) for _, v in sorted(groups.items())], pairs, rejected


def _assign(B, recs, group_members, tau, p):
    """Group index of every band point (-1 = none). A point is in a group when it lies inside the strip spanned by the group's
    lines (fan of layers +/- gate), inside the lines' observed spans and its in-plane normal agrees with the line."""
    P, N2 = B["P"], B["N2"]
    best = np.full(len(P), np.inf)
    gid = np.full(len(P), -1, dtype=np.int64)
    gate = p["gate_tau_multiple"] * tau
    for g, members in enumerate(group_members):
        r = np.stack([P @ recs[k]["m"] - recs[k]["c"] for k in members], axis=1)
        inside = (r.min(axis=1) < gate) & (r.max(axis=1) > -gate)
        # a point must also be near the strip of a member line in the SIGNED sense: between the extreme lines, +/- gate
        along = np.zeros(len(P), bool)
        agree = np.zeros(len(P), bool)
        for k in members:
            t = P @ recs[k]["d"]
            for a, b in recs[k]["spans"]:
                along |= (t >= a - tau) & (t <= b + tau)
            agree |= np.abs(N2 @ recs[k]["m"]) >= p["normal_cos"]
        ok = inside & along & agree
        score = np.abs(r).min(axis=1)
        score = np.where(r.min(axis=1) * r.max(axis=1) < 0, 0.0, score)      # between the layers: distance 0
        better = ok & (score < best)
        gid[better] = g
        best[better] = score[better]
    return gid


# ---------------------------------------------------------------------------------------------------------------- fitting
def _fit_lines(P, w, gid, G, prev=None):
    """Weighted TLS line (unit normal m, offset c) of every group. Returns arrays M (G,2), Cc (G,) and group weights."""
    M = np.zeros((G, 2))
    Cc = np.zeros(G)
    W = np.zeros(G)
    for g in range(G):
        s = gid == g
        ws = w[s]
        if ws.sum() <= 1e-12 or s.sum() < 3:
            if prev is not None:
                M[g], Cc[g] = prev[0][g], prev[1][g]
            continue
        q = P[s]
        mu = (q * ws[:, None]).sum(axis=0) / ws.sum()
        d = q - mu
        cov = (d * ws[:, None]).T @ d / ws.sum()
        vals, vecs = np.linalg.eigh(cov)
        m = vecs[:, 0]
        if prev is not None and float(m @ prev[0][g]) < 0:
            m = -m
        M[g], Cc[g], W[g] = m, float(m @ mu), ws.sum()
    return M, Cc, W


def _apply(P, F, C, x):
    """Corrected plane points: p' = c + s R(theta) (p - c) + u with x columns (theta, ls, u1, u2)."""
    th, ls, u = x[F, 0], x[F, 1], x[F, 2:4]
    c, s = np.cos(th), np.sin(th)
    q = P - C[F]
    rq = np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]], axis=1) * np.exp(ls)[:, None]
    return C[F] + rq + u, rq


def _jac(rq, m):
    """d r / d (theta, ls, u1, u2) for r = m . p' - c, with rq = s R q (p' = c + rq + u)."""
    jt = m[:, 0] * (-rq[:, 1]) + m[:, 1] * rq[:, 0]
    jl = m[:, 0] * rq[:, 0] + m[:, 1] * rq[:, 1]
    return np.stack([jt, jl, m[:, 0], m[:, 1]], axis=1)


def _cauchy_w(r, c):
    return 1.0 / (1.0 + (r / c) ** 2)


def _rho(r, c):
    return c * c * np.log1p((r / c) ** 2)


def _blocks(J, w, F, n):
    """Per-frame 4x4 normal blocks sum w J^T J and the 4-vectors sum w J^T r (r supplied separately)."""
    H = np.zeros((n, 4, 4))
    for a in range(4):
        for b in range(a, 4):
            v = np.bincount(F, weights=w * J[:, a] * J[:, b], minlength=n)
            H[:, a, b] = v
            H[:, b, a] = v
    return H


def _knot_basis(n_frames, order, every):
    """Hat-function interpolation (n_frames x m) of knot parameters placed every `every` active frames along the video;
    inactive frames get a zero row. Piecewise-linear in time: a short window shares its parameters, so a frame that sees a single
    wall is constrained by its neighbours (which see other walls)."""
    L = len(order)
    knots = list(range(0, L, every))
    if knots[-1] != L - 1:
        knots.append(L - 1)
    Phi = np.zeros((n_frames, len(knots)))
    for q, fi in enumerate(order):
        j = int(np.searchsorted(knots, q, side="right") - 1)
        j = min(j, len(knots) - 2) if len(knots) > 1 else 0
        if len(knots) == 1:
            Phi[fi, 0] = 1.0
            continue
        a, b = knots[j], knots[j + 1]
        w = (q - a) / float(b - a)
        Phi[fi, j], Phi[fi, j + 1] = 1.0 - w, w
    return Phi, knots


class _Problem:
    def __init__(self, B, gid, G, active, extent, noise, p):
        self.p = p
        keep = (gid >= 0) & active[B["F"]]
        self.P, self.N2, self.F, self.gid = B["P"][keep], B["N2"][keep], B["F"][keep], gid[keep]
        self.C = B["C"]
        self.n = B["n_frames"]
        self.G = G
        self.extent, self.noise = extent, noise
        self.active = active
        # per (frame, group) cap so that a frame never counts as more than frame_cap independent samples in a surface
        key = self.F * max(G, 1) + self.gid
        cnt = np.bincount(key, minlength=self.n * max(G, 1)).astype(float)
        self.w0 = np.minimum(1.0, p["frame_cap"] / np.maximum(cnt[key], 1.0))
        order = np.flatnonzero(active).tolist()
        self.order = order
        self.Phi, self.knots = _knot_basis(self.n, order, max(1, int(p["knot_every"])))
        self.m = self.Phi.shape[1]
        self.Phi4 = np.kron(self.Phi, np.eye(4))
        spacing = np.diff(np.asarray(self.knots, dtype=float)) if self.m > 1 else np.zeros(0)
        self.edge_gap = spacing
        self.sig = [p["sigma_theta"], p["sigma_scale"], p["sigma_shift"] * extent, p["sigma_shift"] * extent]
        self.step = [p["step_theta"], p["step_scale"], p["step_shift"] * extent, p["step_shift"] * extent]
        self.sigma_n = max(p["sigma_noise_multiple"] * noise, 1e-12)
        self.kappa = p["smooth_robust"]
        A = np.zeros((4 * self.m, 4 * self.m))
        for k in range(4):
            A[np.arange(self.m) * 4 + k, np.arange(self.m) * 4 + k] += 1.0 / self.sig[k] ** 2
        # gauge: mean log depth scale over the active frames is zero (hard)
        g = np.zeros(4 * self.m)
        g[1::4] = self.Phi[active].mean(axis=0) if active.any() else 0.0
        A += p["gauge_weight"] * np.outer(g, g)
        self.abs_prior = A

    def expand(self, xk):
        return (self.Phi4 @ xk.reshape(-1)).reshape(self.n, 4)

    def _edge_terms(self, xk):
        d = xk[1:] - xk[:-1]
        s2 = np.array(self.step)[None, :] ** 2 * np.maximum(self.edge_gap, 1.0)[:, None]
        return d, s2

    def prior_matrix(self, xk):
        """Absolute prior + gauge + ROBUST (Cauchy) random-walk smoothness between knots, linearised at xk (IRLS)."""
        A = self.abs_prior.copy()
        if self.m > 1:
            d, s2 = self._edge_terms(xk)
            w = 1.0 / (s2 * (1.0 + d ** 2 / (self.kappa ** 2 * s2)))
            for k in range(4):
                ii, jj = np.arange(self.m - 1) * 4 + k, np.arange(1, self.m) * 4 + k
                A[ii, ii] += w[:, k]
                A[jj, jj] += w[:, k]
                A[ii, jj] -= w[:, k]
                A[jj, ii] -= w[:, k]
        return A

    def prior_cost(self, xk):
        v = xk.reshape(-1)
        cost = float(v @ self.abs_prior @ v)
        if self.m > 1:
            d, s2 = self._edge_terms(xk)
            cost += float((self.kappa ** 2 * np.log1p(d ** 2 / (self.kappa ** 2 * s2))).sum())
        return cost

    def residuals(self, xk, M, Cc):
        x = self.expand(xk)
        Pp, rq = _apply(self.P, self.F, self.C, x)
        m = M[self.gid]
        r = np.einsum("ij,ij->i", Pp, m) - Cc[self.gid]
        return r, rq, m

    def data_cost(self, r, c, w=None):
        w = self.w0 if w is None else w
        return float((w * _rho(r, c)).sum() / self.sigma_n ** 2)

    def line_fit(self, xk, c, prev=None, extra_w=None):
        Pp, _ = _apply(self.P, self.F, self.C, self.expand(xk))
        r0 = np.zeros(len(Pp))
        if prev is not None:
            r0 = np.einsum("ij,ij->i", Pp, prev[0][self.gid]) - prev[1][self.gid]
        w = self.w0 * _cauchy_w(r0, c) if prev is not None else self.w0.copy()
        if extra_w is not None:
            w = w * extra_w
        return _fit_lines(Pp, w, self.gid, self.G, prev)

    def information(self, xk, M, Cc, c, scale=None, extra_w=None):
        """Data normal equations (4m x 4m) and gradient (4m) of the IRLS surrogate at xk (frame blocks mapped through Phi)."""
        r, rq, m = self.residuals(xk, M, Cc)
        sig2 = self.sigma_n ** 2 if scale is None else scale ** 2
        w = self.w0 * _cauchy_w(r, c) / sig2
        if extra_w is not None:
            w = w * extra_w
        J = _jac(rq, m)
        n = self.n
        H = _blocks(J, w, self.F, n)
        g = np.stack([np.bincount(self.F, weights=w * J[:, a] * r, minlength=n) for a in range(4)], axis=1).reshape(-1)
        Hbd = np.zeros((4 * n, 4 * n))
        for i in range(n):
            Hbd[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = H[i]
        return self.Phi4.T @ Hbd @ self.Phi4, self.Phi4.T @ g

    def gn_step(self, xk, M, Cc, c, fixed=None, extra_w=None):
        Hk, gk = self.information(xk, M, Cc, c, None, extra_w)
        Pm = self.prior_matrix(xk)
        A = Pm + Hk
        rhs = -(gk + Pm @ xk.reshape(-1))
        if fixed is not None and fixed.any():
            fidx = np.flatnonzero(fixed.reshape(-1))
            A[fidx, :] = 0.0
            A[:, fidx] = 0.0
            A[fidx, fidx] = 1.0
            rhs[fidx] = 0.0
        try:
            d = np.linalg.solve(A + 1e-12 * np.eye(len(A)), rhs)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(A, rhs, rcond=None)[0]
        return d.reshape(self.m, 4)


def _optimise(prob, x0, c_schedule, fixed=None, extra_w=None):
    """Alternate (refit lines, Gauss-Newton on the knot parameters) over the annealed Cauchy scale. Returns x, M, Cc, history."""
    x = x0.copy()
    M, Cc, _ = prob.line_fit(x, c_schedule[0])
    hist = []
    for c in c_schedule:
        M, Cc, _ = prob.line_fit(x, c, (M, Cc), extra_w)
        M, Cc, _ = prob.line_fit(x, c, (M, Cc), extra_w)
        for _ in range(2):
            d = prob.gn_step(x, M, Cc, c, fixed, extra_w)
            base = prob.data_cost(prob.residuals(x, M, Cc)[0], c) + prob.prior_cost(x)
            step = 1.0
            for _t in range(5):
                xn = x + step * d
                if fixed is not None:
                    xn = np.where(fixed, 0.0, xn)
                cost = prob.data_cost(prob.residuals(xn, M, Cc)[0], c) + prob.prior_cost(xn)
                if cost <= base:
                    x = xn
                    break
                step *= 0.5
        hist.append(float(prob.data_cost(prob.residuals(x, M, Cc)[0], c) + prob.prior_cost(x)))
    return x, M, Cc, hist


# ---------------------------------------------------------------------------------------------------------------- uncertainty
def _uncertainty(prob, x, M, Cc, c):
    """Leave-one-view-out jackknife of every knot parameter (each view removed in turn, lines refitted, one Gauss-Newton step)
    combined with the Laplace posterior sd of the joint problem (empirical residual scale, prior included)."""
    r, _rq, _m = prob.residuals(x, M, Cc)
    scale = max(1.4826 * float(np.median(np.abs(r))), prob.sigma_n)
    Hk, _g = prob.information(x, M, Cc, c, scale)
    A = prob.prior_matrix(x) + Hk
    try:
        cov = np.linalg.inv(A + 1e-12 * np.eye(len(A)))
        lap = np.sqrt(np.maximum(np.diag(cov), 0.0)).reshape(prob.m, 4)
    except np.linalg.LinAlgError:
        lap = np.full((prob.m, 4), np.inf)
    frames = [int(i) for i in np.unique(prob.F)]
    devs = []
    for j in frames:
        extra = (prob.F != j).astype(float)
        M2, C2, _ = prob.line_fit(x, c, (M, Cc), extra)
        devs.append(prob.gn_step(x, M2, C2, c, None, extra))
    devs = np.asarray(devs)                      # (k, m, 4): knot parameter change when view j is removed
    k = len(frames)
    jk = np.zeros((prob.m, 4))
    if k >= 3:
        jk = np.sqrt((k - 1) / k * np.sum((devs - devs.mean(axis=0)) ** 2, axis=0))
    return jk, lap, scale


# ---------------------------------------------------------------------------------------------------------------- output
def _pose_apply(prediction, x, centre_axis, frame):
    """Calibrated copy of the prediction: depth scaled per frame, extrinsics (world-to-camera) re-posed."""
    out = dict(prediction)
    depth = np.asarray(prediction["depth"])
    extr = np.asarray(prediction["extrinsics"])
    nd = depth.astype(np.float64)
    ne = extr.astype(np.float64).copy()
    a = _unit(frame["axis"])
    e1, e2 = np.asarray(frame["e1"], float), np.asarray(frame["e2"], float)
    for k in range(len(depth)):
        th, ls, u1, u2 = x[k]
        if th == 0.0 and ls == 0.0 and u1 == 0.0 and u2 == 0.0:
            continue
        s = math.exp(ls)
        R, t = ne[k][:3, :3], ne[k][:3, 3]
        cen = -R.T @ t
        Q = _rodrigues(a, th)
        R2 = R @ Q.T
        cen2 = cen + u1 * e1 + u2 * e2
        ne[k][:3, :3] = R2
        ne[k][:3, 3] = -R2 @ cen2
        nd[k] = nd[k] * s
    out["depth"] = nd.astype(depth.dtype)
    out["extrinsics"] = ne.astype(extr.dtype)
    return out


def _identity(prediction, status, reasons, extra=None):
    rep = {"model": MODEL, "status": status, "reasons": list(reasons), "truth_used": False, "applied": False,
           "parameters": {"estimated": [], "refused": [], "kept_at_da3": ["vertical translation", "pitch", "roll", "intrinsics"]}}
    if extra:
        rep.update(extra)
    return dict(prediction), rep


# ---------------------------------------------------------------------------------------------------------------- public API
def calibrate(prediction, axis, *, walls=None, verify=True, **overrides):
    """Truth-free per-frame calibration for one structural axis. Returns (calibrated_prediction, report); the prediction is
    returned unchanged (status != 'calibrated') whenever the evidence is insufficient, unidentifiable or does not reduce the cost."""
    p = {**DEFAULTS, **overrides}
    extr = np.asarray(prediction["extrinsics"], float)
    if not _pose_rotations_ok(extr):
        return _identity(prediction, "refused_reflection", ["an input pose rotation has det <= 0 (reflection) or is not finite"])
    a = _unit(axis)
    if not np.isfinite(a).all() or np.linalg.norm(a) < 0.5:
        return _identity(prediction, "invalid_axis", ["invalid axis"])
    w0 = walls if walls is not None else we.extract_wall_lines(prediction, a)
    if w0.get("status") != "ok" or not w0.get("lines"):
        return _identity(prediction, "no_lines", ["wall extraction found no lines: " + "; ".join(w0.get("reasons") or [])])
    extent = float(w0["frame"]["extent"])
    noise = float(w0["noise_scale"])
    tau = float(w0["params"]["tolerance"])
    B = band_points(prediction, w0, stride=p["stride"], conf_percentile=p["conf_percentile"], cell_fraction=p["cell_fraction"],
                    max_jump=p["max_jump"], wall_like_deg=p["wall_like_deg"])
    recs_all = _line_records(w0)
    support = _line_support(B, recs_all, tau, p)
    keep_ids = {r["id"] for r, s in zip(recs_all, support) if s >= p["min_line_frames"]}
    recs = [r for r in recs_all if r["id"] in keep_ids]
    w_sub = {**w0, "lines": [ln for ln in w0["lines"] if ln.get("id") in keep_ids]}
    def pair_ok(ia, ib, pr):
        """A layered pair is linked only when every frame is internally single-layer there: a frame that itself sees both lines
        (furniture in front of a wall, a pilaster) scatters about the layer separation, while a drift layer is a straight line
        inside each frame (scatter at the noise)."""
        gid2 = _assign(B, [recs[ia], recs[ib]], [[0, 1]], tau, p)
        s = gid2 == 0
        thr = max(p["single_layer_noise"] * noise, p["single_layer_sep_fraction"] * pr["separation"])
        ok_w = tot_w = 0.0
        for f in np.unique(B["F"][s]):
            q = B["P"][s & (B["F"] == f)]
            if len(q) < p["min_frame_points"]:
                continue
            mu = q.mean(axis=0)
            vals, vecs = np.linalg.eigh(np.cov((q - mu).T))
            res = (q - mu) @ vecs[:, 0]
            sc = 1.4826 * float(np.median(np.abs(res - np.median(res))))
            wf = min(len(q), p["frame_cap"])
            tot_w += wf
            ok_w += wf * (sc <= thr)
        return tot_w > 0 and ok_w / tot_w >= p["single_layer_fraction"]

    final_groups, pairs, split = _link_groups(recs, w_sub, p, pair_ok)
    gid = _assign(B, recs, final_groups, tau, p)
    G = len(final_groups)
    n_linked = sum(1 for m in final_groups if len(m) > 1)
    info = {"lines": len(recs_all), "lines_with_wall_like_support": len(recs), "layered_pairs": len(pairs), "groups": G, "linked_groups": n_linked,
            "rejected_multilayer_pairs": split, "band_samples": int(len(B["P"])), "assigned_samples": int((gid >= 0).sum()),
            "extent": extent, "noise_scale": noise, "tolerance": tau}
    if n_linked == 0:
        return _identity(prediction, "no_layered_groups", ["no layered same-side group of lines: nothing to reconcile"], {"info": info})
    frames_in = np.unique(B["F"][gid >= 0])
    active = np.zeros(B["n_frames"], bool)
    active[frames_in] = True
    if active.sum() < 6:
        return _identity(prediction, "too_few_frames", [f"only {int(active.sum())} frames see the surfaces"], {"info": info})
    prob = _Problem(B, gid, G, active, extent, noise, p)
    c_final = p["c_final_noise"] * noise
    c0 = max(p["c0_fraction"] * extent, c_final)
    sched = []
    c = c0
    while True:
        sched.append(max(c, c_final))
        if c <= c_final or len(sched) >= p["iterations"]:
            break
        c *= p["c_anneal"]
    sched += [c_final] * 3
    x0 = np.zeros((prob.m, 4))
    r0, _, _ = prob.residuals(x0, *_fit_lines(prob.P, prob.w0, prob.gid, G)[:2])
    cost0 = prob.data_cost(r0, c_final)
    x, M, Cc, hist = _optimise(prob, x0, sched)
    rf, _, _ = prob.residuals(x, M, Cc)
    cost1 = prob.data_cost(rf, c_final)
    info.update(cost_before=cost0, cost_after_unrefused=cost1, median_abs_residual_before_over_noise=float(np.median(np.abs(r0)) / noise),
                median_abs_residual_after_over_noise=float(np.median(np.abs(rf)) / noise))
    # uncertainty + refusal
    jk, lap, scale = _uncertainty(prob, x, M, Cc, c_final)
    unc = np.sqrt(jk ** 2 + lap ** 2)
    refused = unc > p["refuse_factor"] * np.abs(x)
    x_ref = np.where(refused, 0.0, x)
    if refused.all():
        return _identity(prediction, "unidentifiable", ["every parameter is refused (uncertainty exceeds magnitude)"], {"info": info})
    x2, M2, C2, _ = _optimise(prob, x_ref, [c_final] * 4, fixed=refused)
    rf2, _, _ = prob.residuals(x2, M2, C2)
    cost2 = prob.data_cost(rf2, c_final)
    info.update(cost_after=cost2, median_abs_residual_after_refusal_over_noise=float(np.median(np.abs(rf2)) / noise))
    drop = 1.0 - cost2 / cost0 if cost0 > 0 else 0.0
    info["cost_drop_fraction"] = float(drop)
    if drop < p["min_cost_drop"]:
        return _identity(prediction, "no_gain", [f"data cost fell by only {drop:.3f} (< {p['min_cost_drop']})"], {"info": info})
    xo = prob.expand(np.where(refused, 0.0, x2))
    frame = {"axis": a, "e1": w0["frame"]["e1"], "e2": w0["frame"]["e2"]}
    out = _pose_apply(prediction, xo, a, frame)
    if not _pose_rotations_ok(out["extrinsics"]):
        return _identity(prediction, "refused_reflection", ["calibrated rotation is not proper"], {"info": info})
    names = ("yaw_rad", "log_depth_scale", "shift_e1", "shift_e2")
    table = [{"frame_position": int(i), "yaw_deg": float(math.degrees(xo[i, 0])), "depth_scale": float(math.exp(xo[i, 1])),
              "shift_over_extent": [float(xo[i, 2] / extent), float(xo[i, 3] / extent)],
              } for i in np.flatnonzero(active)]
    knot_table = [{"knot_frame_position": int(prob.order[kf]), "values": {names[k]: float(x2[j, k]) for k in range(4)},
                   "uncertainty": {names[k]: float(unc[j, k]) for k in range(4)},
                   "refused": {names[k]: bool(refused[j, k]) for k in range(4)}} for j, kf in enumerate(prob.knots)]
    report = {"model": MODEL, "status": "calibrated", "reasons": [], "truth_used": False, "applied": True, "info": info,
              "parameters": {"estimated": list(names), "refused_count": int(refused.sum()), "active_frames": int(active.sum()),
                             "knots": int(prob.m), "total_params": int(4 * prob.m), "kept_at_da3": ["vertical translation", "pitch", "roll", "intrinsics"],
                             "gauge": "mean log depth scale over the active frames = 0",
                             "max_abs_yaw_deg": float(np.degrees(np.abs(xo[:, 0]).max())),
                             "max_abs_shift_over_extent": float(np.abs(xo[:, 2:4]).max() / extent),
                             "max_abs_log_scale": float(np.abs(xo[:, 1]).max())}, "knots": knot_table, "frames": table}
    if verify:
        w1 = we.extract_wall_lines(out, a)
        before = len(layered_pairs(w0)), len(w0["lines"])
        after = (len(layered_pairs(w1)), len(w1["lines"])) if w1.get("status") == "ok" else (None, 0)
        report["verification"] = {"layered_before": before[0], "layered_after": after[0], "lines_before": before[1],
                                  "lines_after": after[1], "rule": "revert unless layered pairs and lines do not increase"}
        if w1.get("status") != "ok" or after[0] > before[0] or after[1] > before[1]:
            pred, rep = _identity(prediction, "reverted_by_verification", ["re-extraction shows more layered pairs or lines"],
                                  {"info": info, "verification": report["verification"], "rejected_parameters": report["parameters"]})
            return pred, rep
    return out, report


