"""Truth-free feature tracking + robust bundle adjustment on top of a frozen DA3 prediction.

Pipeline (CPU only, deterministic):
  1. `track_features`: Shi-Tomasi corners seeded on every frame, pyramidal KLT with forward-backward check and
     RANSAC fundamental-matrix verification between consecutive frames; tracks are chained recursively
     (a track lives while it stays visible and consistent; new corners fill the gaps).
  2. `triangulate_tracks`: vectorised multi-view midpoint triangulation from DA3 poses/intrinsics.
  3. `bundle_adjust`: robust (Huber/Cauchy) sparse least squares over camera poses (camera 0 pose + one
     scale constraint fix the gauge; an optional weak pose prior to DA3 stabilises poorly observed frames),
     3D points, and optionally one shared focal scale. An optional weak DA3-depth prior is available but OFF by default.
  4. `scale_bias_report`: floor-to-ceiling separation vs horizontal wall-to-wall extents measured from the
     sparse triangulated cloud only, against the same quantity from DA3 dense depth. Uses
     height_diagnosis.estimate_room_height (points + view ids + structural axis; no camera-up).

Ground truth, the simulator and evaluators are never imported here (enforced by an AST test). Poses are DA3
world-to-camera extrinsics, X_c = R X_w + t. Units are DA3's arbitrary units.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------------------------------- tracking
@dataclass
class TrackConfig:
    max_corners_per_frame: int = 1500
    max_active: int = 8000
    quality_level: float = 0.001
    min_distance: float = 6.0
    block_size: int = 5
    border: int = 6
    lk_window: int = 31
    lk_levels: int = 4
    fb_threshold_px: float = 1.0
    ransac_px: float = 1.5
    min_pairs_for_f: int = 12
    seed_every: int = 1
    seed: int = 0


def _gray(frame):
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def track_features(frames, cfg: TrackConfig | None = None):
    """Track corners through `frames` (list/array of RGB or gray uint8 images).

    Returns dict(track_id, frame, xy) observation arrays sorted by (track, frame), plus per-frame diagnostics.
    """
    cfg = cfg or TrackConfig()
    cv2.setRNGSeed(cfg.seed)
    grays = [_gray(f) for f in frames]
    n = len(grays)
    h, w = grays[0].shape
    lk = dict(winSize=(cfg.lk_window, cfg.lk_window), maxLevel=cfg.lk_levels,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pts = np.zeros((0, 2), np.float32)
    ids = np.zeros(0, np.int64)
    next_id = 0
    o_id, o_fr, o_xy = [], [], []
    per_frame = []
    for t in range(n):
        lost = {"fb": 0, "ransac": 0, "border": 0, "klt": 0, "unverifiable": 0}
        if t > 0 and len(pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[t - 1], grays[t], pts.reshape(-1, 1, 2), None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[t - 1], nxt, None, **lk)
            nxt, back = nxt.reshape(-1, 2), back.reshape(-1, 2)
            ok = (st.ravel() == 1) & (st2.ravel() == 1)
            lost["klt"] = int((~ok).sum())
            fb = np.linalg.norm(back - pts, axis=1)
            fb_ok = ok & (fb < cfg.fb_threshold_px)
            lost["fb"] = int((ok & ~fb_ok).sum())
            ok = fb_ok
            inside = ((nxt[:, 0] >= cfg.border) & (nxt[:, 0] < w - cfg.border) &
                      (nxt[:, 1] >= cfg.border) & (nxt[:, 1] < h - cfg.border))
            lost["border"] = int((ok & ~inside).sum())
            ok &= inside
            if ok.sum() >= cfg.min_pairs_for_f:
                sel = np.flatnonzero(ok)
                _f, mask = cv2.findFundamentalMat(pts[sel], nxt[sel], cv2.FM_RANSAC, cfg.ransac_px, 0.999, 2000)
                if mask is None:
                    lost["ransac"] = int(ok.sum())
                    ok[:] = False
                else:
                    good = mask.ravel() > 0
                    lost["ransac"] = int((~good).sum())
                    ok[sel[~good]] = False
            else:
                lost["unverifiable"] = int(ok.sum())
                ok[:] = False
            pts, ids = nxt[ok].astype(np.float32), ids[ok]
        n_continued = len(pts)
        n_new = 0
        if t % cfg.seed_every == 0 and len(pts) < cfg.max_active:
            mask = np.full((h, w), 255, np.uint8)
            mask[:cfg.border, :] = 0
            mask[-cfg.border:, :] = 0
            mask[:, :cfg.border] = 0
            mask[:, -cfg.border:] = 0
            for p in pts:
                cv2.circle(mask, (int(round(p[0])), int(round(p[1]))), int(cfg.min_distance), 0, -1)
            budget = min(cfg.max_corners_per_frame, cfg.max_active - len(pts))
            corners = cv2.goodFeaturesToTrack(grays[t], budget, cfg.quality_level, cfg.min_distance, mask=mask,
                                              blockSize=cfg.block_size)
            if corners is not None and len(corners):
                corners = corners.reshape(-1, 2).astype(np.float32)
                n_new = len(corners)
                pts = np.concatenate([pts, corners])
                ids = np.concatenate([ids, np.arange(next_id, next_id + n_new, dtype=np.int64)])
                next_id += n_new
        per_frame.append({"frame": t, "continued": int(n_continued), "new": int(n_new), "active": int(len(pts)),
                          "lost": lost})
        o_id.append(ids.copy())
        o_fr.append(np.full(len(ids), t, np.int64))
        o_xy.append(pts.astype(np.float64).copy())
    track_id = np.concatenate(o_id) if o_id else np.zeros(0, np.int64)
    frame = np.concatenate(o_fr) if o_fr else np.zeros(0, np.int64)
    xy = np.concatenate(o_xy) if o_xy else np.zeros((0, 2))
    order = np.lexsort((frame, track_id))
    return {"track_id": track_id[order], "frame": frame[order], "xy": xy[order], "per_frame": per_frame,
            "n_frames": n, "image_size": (w, h)}


def compact_tracks(obs, min_len=2):
    """Keep tracks with >= min_len observations; relabel ids to 0..T-1 (order preserved)."""
    tid = obs["track_id"]
    uniq, inv, counts = np.unique(tid, return_inverse=True, return_counts=True)
    keep_t = counts >= min_len
    keep_o = keep_t[inv]
    new_id = (np.cumsum(keep_t) - 1)[inv]
    return {"track_id": new_id[keep_o], "frame": obs["frame"][keep_o], "xy": obs["xy"][keep_o]}


def restrict_to_frames(obs, source_indices):
    """Keep observations at the frames listed in `source_indices` (video index per prediction frame) and
    renumber frames to prediction indices. Tracking over ALL video frames then keeps longer chains."""
    lookup = {int(s): i for i, s in enumerate(source_indices)}
    keep = np.array([int(f) in lookup for f in obs["frame"]], bool)
    frame = np.array([lookup[int(f)] for f in obs["frame"][keep]], np.int64)
    return {"track_id": obs["track_id"][keep], "frame": frame, "xy": obs["xy"][keep]}


def track_statistics(obs, n_frames=None):
    """Summary counts / lengths (all tracks incl. length-1 seeds, and verified length>=2)."""
    tid = obs["track_id"]
    if len(tid) == 0:
        return {"seeded_tracks": 0, "tracks_len_ge2": 0}
    _u, counts = np.unique(tid, return_counts=True)
    n_frames = int(n_frames or (obs["frame"].max() + 1))
    per_frame = np.bincount(obs["frame"], minlength=n_frames)
    multi = counts[counts >= 2]
    out = {"seeded_tracks": int(len(counts)), "observations": int(len(tid)),
           "tracks_len_ge2": int(len(multi)), "tracks_len_ge3": int((counts >= 3).sum()),
           "tracks_len_ge5": int((counts >= 5).sum()), "tracks_len_ge10": int((counts >= 10).sum()),
           "max_length": int(counts.max()), "mean_length_ge2": float(multi.mean()) if len(multi) else 0.0,
           "median_length_ge2": float(np.median(multi)) if len(multi) else 0.0,
           "frac_seeded_surviving_ge3": float((counts >= 3).mean()),
           "frac_seeded_surviving_ge5": float((counts >= 5).mean()),
           "active_per_frame_mean": float(per_frame.mean()), "active_per_frame_min": int(per_frame.min()),
           "active_per_frame_median": float(np.median(per_frame)),
           "frames_with_ge50_active": int((per_frame >= 50).sum()), "n_frames": n_frames}
    return out


# ----------------------------------------------------------------------------------------- geometry helpers
def camera_centres(R, t):
    return -np.einsum("nji,nj->ni", R, t)


def _rays(obs, K, R):
    """World-frame unit rays for each observation. K is Nx3x3 per frame."""
    f = obs["frame"]
    uv1 = np.concatenate([obs["xy"], np.ones((len(f), 1))], axis=1)
    cam = np.einsum("nij,nj->ni", np.linalg.inv(K)[f], uv1)
    d = np.einsum("nji,nj->ni", R[f], cam)  # R^T cam
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def triangulate_tracks(obs, K, R, t, n_tracks=None):
    """Midpoint multi-view triangulation (vectorised). Returns points (T,3), valid mask, parallax_deg (T,)."""
    tid, f = obs["track_id"], obs["frame"]
    n_tracks = int(n_tracks if n_tracks is not None else tid.max() + 1)
    C = camera_centres(R, t)
    d = _rays(obs, K, R)
    proj = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = np.zeros((n_tracks, 3, 3))
    b = np.zeros((n_tracks, 3))
    np.add.at(A, tid, proj)
    np.add.at(b, tid, np.einsum("nij,nj->ni", proj, C[f]))
    det = np.linalg.det(A)
    valid = np.isfinite(det) & (np.abs(det) > 1e-9)
    X = np.full((n_tracks, 3), np.nan)
    X[valid] = np.linalg.solve(A[valid], b[valid][:, :, None])[:, :, 0]
    # parallax: largest angle between a track's rays
    first = np.zeros(n_tracks, np.int64)
    first[tid[::-1]] = np.arange(len(tid))[::-1]
    ang = np.degrees(np.arccos(np.clip(np.sum(d * d[first[tid]], axis=1), -1, 1)))
    par = np.zeros(n_tracks)
    np.maximum.at(par, tid, ang)
    # cheirality: positive depth in every observing camera
    Xc = np.einsum("nij,nj->ni", R[f], X[tid]) + t[f]
    behind = np.zeros(n_tracks, bool)
    np.logical_or.at(behind, tid, ~(Xc[:, 2] > 1e-6))
    valid &= ~behind
    return X, valid, par


def reprojection_errors(obs, K, R, t, X, focal_scale=1.0):
    """Per-observation pixel reprojection error vectors (n,2) and depths."""
    f, tid = obs["frame"], obs["track_id"]
    Xc = np.einsum("nij,nj->ni", R[f], X[tid]) + t[f]
    z = np.maximum(Xc[:, 2], 1e-6)
    u = focal_scale * K[f, 0, 0] * Xc[:, 0] / z + K[f, 0, 2]
    v = focal_scale * K[f, 1, 1] * Xc[:, 1] / z + K[f, 1, 2]
    return np.stack([u, v], axis=1) - obs["xy"], Xc[:, 2]


# ------------------------------------------------------------------------------------------ bundle adjustment
@dataclass
class BAConfig:
    loss: str = "huber"            # "huber" | "cauchy" | "soft_l1" | "linear"
    f_scale_px: float = 1.0
    min_track_len: int = 3
    min_parallax_deg: float = 2.0
    final_min_parallax_deg: float = 3.0   # output gate on refined geometry (depth precision ~ 0.5px / (f * parallax))
    max_range_factor: float = 4.0         # drop points farther than this x median range from the camera centroid
    max_init_reproj_px: float = 10.0
    prune_px: float = 3.0
    prune_rounds: int = 2
    max_points: int = 6000
    refine_focal: bool = False
    rot_prior_deg: float = 2.0     # weak prior to the DA3 pose (stabilises weakly observed frames)
    pos_prior_frac: float = 0.05   # fraction of camera-centre extent
    depth_prior_weight: float = 0.0  # 0 = OFF (DA3 depth is NOT used)
    max_nfev: int = 25
    irls_iterations: int = 5
    coarse_f_scale_factor: float = 8.0   # round r uses f_scale * max(1, factor / 2**r): coarse-to-fine robust loss
    seed: int = 0


def _build(obs, n_cam, n_pts, cfg, refine_focal, have_depth):
    f = obs["frame"]
    m = len(f)
    rows, cols = [], []
    free = f > 0
    for k in range(6):
        rows.append(np.flatnonzero(free) * 2), cols.append((f[free] - 1) * 6 + k)
        rows.append(np.flatnonzero(free) * 2 + 1), cols.append((f[free] - 1) * 6 + k)
    base = 6 * (n_cam - 1)
    for k in range(3):
        rows.append(np.arange(m) * 2), cols.append(base + obs["track_id"] * 3 + k)
        rows.append(np.arange(m) * 2 + 1), cols.append(base + obs["track_id"] * 3 + k)
    if refine_focal:
        col = base + 3 * n_pts
        rows.append(np.arange(2 * m)), cols.append(np.full(2 * m, col))
    r0 = 2 * m
    prior_rows = np.arange(6 * (n_cam - 1)) + r0
    rows.append(prior_rows), cols.append(np.arange(6 * (n_cam - 1)))
    r1 = r0 + 6 * (n_cam - 1)
    # scale-gauge row touches all centre params of the far camera: filled by caller via `far`
    return rows, cols, r1


def bundle_adjust(obs, K, R0, t0, X0, cfg: BAConfig | None = None, depth_obs=None, init=None):
    """Robust BA. obs: track_id (0..P-1 matching X0 rows), frame, xy. Returns dict with refined poses/points."""
    cfg = cfg or BAConfig()
    n_cam, n_pts = len(R0), len(X0)
    C0 = camera_centres(R0, t0)
    extent = float(np.linalg.norm(np.ptp(C0, axis=0))) or 1.0
    far = int(np.argmax(np.linalg.norm(C0 - C0[0], axis=1)))
    d0 = float(np.linalg.norm(C0[far] - C0[0])) or 1.0
    sig_rot = np.radians(cfg.rot_prior_deg)
    sig_pos = cfg.pos_prior_frac * extent
    fx0 = K[:, 0, 0]
    m = len(obs["frame"])
    f, tid = obs["frame"], obs["track_id"]
    focal = bool(cfg.refine_focal)
    n_free_cam = 6 * (n_cam - 1)
    use_depth = depth_obs is not None and cfg.depth_prior_weight > 0 and np.isfinite(depth_obs).any()
    depth_rows = np.flatnonzero(np.isfinite(depth_obs)) if use_depth else np.zeros(0, np.int64)

    def unpack(x):
        dr = np.zeros((n_cam, 3))
        dc = np.zeros((n_cam, 3))
        cam = x[:n_free_cam].reshape(-1, 6)
        dr[1:], dc[1:] = cam[:, :3], cam[:, 3:]
        X = x[n_free_cam:n_free_cam + 3 * n_pts].reshape(-1, 3)
        s = float(np.exp(x[-1])) if focal else 1.0
        return dr, dc, X, s

    def poses(dr, dc):
        R = Rotation.from_rotvec(dr).as_matrix() @ R0
        C = C0 + dc
        return R, C

    def residual(x):
        dr, dc, X, s = unpack(x)
        R, C = poses(dr, dc)
        Xc = np.einsum("nij,nj->ni", R[f], X[tid] - C[f])
        z = np.maximum(Xc[:, 2], 1e-3)
        u = s * K[f, 0, 0] * Xc[:, 0] / z + K[f, 0, 2]
        v = s * K[f, 1, 1] * Xc[:, 1] / z + K[f, 1, 2]
        res = np.empty(2 * m)
        res[0::2] = u - obs["xy"][:, 0]
        res[1::2] = v - obs["xy"][:, 1]
        prior = np.concatenate([dr[1:] / sig_rot, dc[1:] / sig_pos], axis=1).ravel()
        gauge = np.array([(np.linalg.norm(C[far] - C[0]) - d0) / (1e-3 * d0)])
        parts = [res, prior, gauge]
        if use_depth:
            parts.append(cfg.depth_prior_weight * (Xc[depth_rows, 2] / depth_obs[depth_rows] - 1.0))
        return np.concatenate(parts)

    rows, cols, r1 = _build(obs, n_cam, n_pts, cfg, focal, use_depth)
    # prior rows: free cam param j has row r0+j; param layout per cam is [rot3, centre3] -> diagonal
    rows_all = np.concatenate(rows)
    cols_all = np.concatenate(cols)
    # gauge row
    g_cols = []
    for cam in (far, 0):
        if cam > 0:
            g_cols.extend(range((cam - 1) * 6 + 3, (cam - 1) * 6 + 6))
    rows_all = np.concatenate([rows_all, np.full(len(g_cols), r1, np.int64)])
    cols_all = np.concatenate([cols_all, np.asarray(g_cols, np.int64)])
    n_rows = r1 + 1
    if use_depth:
        # depth residual of observation j depends on the same cam/point columns as that observation
        for k in range(6):
            sel = depth_rows[f[depth_rows] > 0]
            rows_all = np.concatenate([rows_all, n_rows + np.searchsorted(depth_rows, sel)])
            cols_all = np.concatenate([cols_all, (f[sel] - 1) * 6 + k])
        for k in range(3):
            rows_all = np.concatenate([rows_all, n_rows + np.arange(len(depth_rows))])
            cols_all = np.concatenate([cols_all, n_free_cam + tid[depth_rows] * 3 + k])
        if focal:
            pass  # depth does not depend on focal scale
        n_rows += len(depth_rows)
    n_cols = n_free_cam + 3 * n_pts + (1 if focal else 0)
    sparsity = coo_matrix((np.ones(len(rows_all)), (rows_all, cols_all)), shape=(n_rows, n_cols)).tocsr()

    cam0 = np.zeros((n_cam - 1, 6))
    if init is not None:   # warm start; the DA3 pose R0/t0 stays the prior anchor
        cam0[:, :3] = Rotation.from_matrix(init["R"][1:] @ np.transpose(R0[1:], (0, 2, 1))).as_rotvec()
        cam0[:, 3:] = init["C"][1:] - C0[1:]
    x0 = np.concatenate([cam0.ravel(), np.asarray(X0, float).ravel(), [0.0] if focal else []])
    before = residual(x0)[:2 * m]
    # Robust loss by iteratively reweighted least squares (each inner solve is plain L2, which converges
    # far more reliably with scipy's sparse TRF than its built-in robust losses at multi-pixel initial errors).
    wrow = np.ones(2 * m + n_free_cam + 1 + len(depth_rows))
    x = x0.copy()
    sol = None
    for _it in range(1 if cfg.loss == "linear" else cfg.irls_iterations):
        sol = least_squares(lambda z: residual(z) * wrow, x, jac_sparsity=sparsity, loss="linear", method="trf",
                            x_scale="jac", tr_solver="lsmr", max_nfev=cfg.max_nfev, ftol=1e-8, xtol=1e-9, gtol=1e-9)
        x = sol.x
        if cfg.loss == "linear":
            break
        e = np.linalg.norm(residual(x)[:2 * m].reshape(-1, 2), axis=1)
        k = cfg.f_scale_px
        w = np.where(e <= k, 1.0, k / np.maximum(e, 1e-12)) if cfg.loss == "huber" else 1.0 / (1.0 + (e / k) ** 2)
        wrow[:2 * m] = np.repeat(np.sqrt(w), 2)
    dr, dc, X, s = unpack(x)
    R, C = poses(dr, dc)
    t = -np.einsum("nij,nj->ni", R, C)
    after = residual(x)[:2 * m].reshape(-1, 2)
    return {"R": R, "t": t, "C": C, "X": X, "focal_scale": s, "residual_before": before.reshape(-1, 2),
            "residual_after": after, "cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status),
            "message": str(sol.message), "gauge": {"fixed_camera": 0, "scale_camera": far, "scale_distance": d0},
            "prior": {"rot_sigma_deg": cfg.rot_prior_deg, "pos_sigma": sig_pos, "depth_prior_weight": cfg.depth_prior_weight}}


def _err_stats(res):
    if len(res) == 0:
        return {"n": 0}
    e = np.linalg.norm(res, axis=1)
    return {"n": int(len(e)), "rms_px": float(np.sqrt(np.mean(e ** 2))), "median_px": float(np.median(e)),
            "p90_px": float(np.percentile(e, 90)), "p99_px": float(np.percentile(e, 99)), "max_px": float(e.max())}


def refine_prediction(prediction, obs, cfg: BAConfig | None = None, depth_sampler=None):
    """Triangulate from DA3 poses then run robust BA with outlier pruning. prediction: dict with
    extrinsics (Nx3x4 w2c) and intrinsics (Nx3x3). Returns dict with refined arrays + diagnostics."""
    cfg = cfg or BAConfig()
    E = np.asarray(prediction["extrinsics"], float)
    K = np.asarray(prediction["intrinsics"], float)
    R0, t0 = E[:, :3, :3], E[:, :3, 3]
    n_cam = len(R0)
    obs = compact_tracks(obs, 2)
    n_tr = int(obs["track_id"].max() + 1) if len(obs["track_id"]) else 0
    info = {"tracks_input_len_ge2": n_tr}
    if n_tr == 0:
        return {"ok": False, "reason": "no tracks", "info": info}
    X, valid, par = triangulate_tracks(obs, K, R0, t0, n_tr)
    err, _z = reprojection_errors(obs, K, R0, t0, np.nan_to_num(X))
    e = np.linalg.norm(err, axis=1)
    med = np.full(n_tr, np.inf)
    order = np.argsort(obs["track_id"], kind="stable")
    bounds = np.flatnonzero(np.diff(obs["track_id"][order], prepend=-1)) if len(order) else []
    for a, b in zip(bounds, list(bounds[1:]) + [len(order)]):
        med[obs["track_id"][order][a]] = np.median(e[order[a:b]])
    lengths = np.bincount(obs["track_id"], minlength=n_tr)
    ok = valid & (lengths >= cfg.min_track_len) & (par >= cfg.min_parallax_deg) & (med <= cfg.max_init_reproj_px)
    info.update({"tracks_len_ge_min": int((lengths >= cfg.min_track_len).sum()),
                 "tracks_parallax_ok": int(((par >= cfg.min_parallax_deg) & (lengths >= cfg.min_track_len)).sum()),
                 "tracks_triangulable": int(ok.sum())})
    cand = np.flatnonzero(ok)
    if len(cand) > cfg.max_points:
        rng = np.random.default_rng(cfg.seed)
        keep = np.sort(rng.choice(cand, cfg.max_points, replace=False, p=None))
        cand = keep
    if len(cand) < 20:
        return {"ok": False, "reason": f"only {len(cand)} triangulable tracks", "info": info}
    remap = np.full(n_tr, -1)
    remap[cand] = np.arange(len(cand))
    sel = remap[obs["track_id"]] >= 0
    sub = {"track_id": remap[obs["track_id"]][sel], "frame": obs["frame"][sel], "xy": obs["xy"][sel]}
    Xs = X[cand]
    orig_track = cand.copy()
    first_before = None
    rounds = []
    for rnd in range(cfg.prune_rounds + 1):
        depth_obs = None
        if depth_sampler is not None and cfg.depth_prior_weight > 0:
            depth_obs = depth_sampler(sub)
        rcfg = replace(cfg, f_scale_px=cfg.f_scale_px * max(1.0, cfg.coarse_f_scale_factor / 2 ** rnd))
        res = bundle_adjust(sub, K, R0, t0, Xs, rcfg, depth_obs, init=None if rnd == 0 else res)
        if first_before is None:
            first_before = res["residual_before"]
        e = np.linalg.norm(res["residual_after"], axis=1)
        rounds.append({"round": rnd, "obs": int(len(e)), "points": int(len(Xs)), "after": _err_stats(res["residual_after"]),
                       "cost": res["cost"], "nfev": res["nfev"]})
        Xs = res["X"]
        if cfg.refine_focal:
            K = K.copy()
            K[:, 0, 0] *= res["focal_scale"]
            K[:, 1, 1] *= res["focal_scale"]
        if rnd == cfg.prune_rounds:
            break
        keep_obs = e <= cfg.prune_px
        cnt = np.bincount(sub["track_id"][keep_obs], minlength=len(Xs))
        good_tr = cnt >= cfg.min_track_len
        keep_obs &= good_tr[sub["track_id"]]
        new_map = np.full(len(Xs), -1)
        new_map[good_tr] = np.arange(good_tr.sum())
        sub = {"track_id": new_map[sub["track_id"][keep_obs]], "frame": sub["frame"][keep_obs], "xy": sub["xy"][keep_obs]}
        Xs, orig_track = Xs[good_tr], orig_track[good_tr]
        if len(Xs) < 20:
            return {"ok": False, "reason": "too few points after pruning", "info": info, "rounds": rounds}
    # final quality gate on the refined geometry: parallax, range sanity (outputs only; BA already ran)
    Rf = res["R"]
    d_final = _rays(sub, K, Rf)
    first_i = np.zeros(len(Xs), np.int64)
    first_i[sub["track_id"][::-1]] = np.arange(len(sub["track_id"]))[::-1]
    ang = np.degrees(np.arccos(np.clip(np.sum(d_final * d_final[first_i[sub["track_id"]]], axis=1), -1, 1)))
    par_final = np.zeros(len(Xs))
    np.maximum.at(par_final, sub["track_id"], ang)
    Cf = camera_centres(Rf, res["t"])
    rng_pt = np.linalg.norm(Xs - Cf.mean(axis=0), axis=1)
    keep_pt = (par_final >= cfg.final_min_parallax_deg) & (rng_pt <= cfg.max_range_factor * np.median(rng_pt))
    final_gate = {"points_before_gate": int(len(Xs)), "points_after_gate": int(keep_pt.sum()),
                  "final_min_parallax_deg": cfg.final_min_parallax_deg, "max_range_factor": cfg.max_range_factor}
    if keep_pt.sum() < 20:
        return {"ok": False, "reason": "too few points after final gate", "info": info, "rounds": rounds,
                "final_gate": final_gate}
    new_map = np.full(len(Xs), -1)
    new_map[keep_pt] = np.arange(keep_pt.sum())
    sel_o = keep_pt[sub["track_id"]]
    res_after_all = res["residual_after"]
    res["residual_after"] = res_after_all[sel_o]
    sub = {"track_id": new_map[sub["track_id"][sel_o]], "frame": sub["frame"][sel_o], "xy": sub["xy"][sel_o]}
    Xs, orig_track = Xs[keep_pt], orig_track[keep_pt]
    # point-view ids: first observing frame per final track
    first = np.full(len(Xs), 10 ** 9)
    np.minimum.at(first, sub["track_id"], sub["frame"])
    lengths = np.bincount(sub["track_id"], minlength=len(Xs))
    par_kept = par_final[keep_pt]
    sig_px = float(np.median(np.linalg.norm(res["residual_after"], axis=1)))
    f_mean = float(np.mean(K[:, 0, 0]))
    sig_rel = sig_px / (f_mean * np.radians(np.maximum(par_kept, 1e-3)))
    quality = {"parallax_deg_p10_p50_p90": [float(v) for v in np.percentile(par_kept, [10, 50, 90])],
               "median_reproj_px": sig_px,
               "implied_relative_depth_sigma_p10_p50_p90": [float(v) for v in np.percentile(sig_rel, [10, 50, 90])],
               "note": "sigma_z/z ~ sigma_px / (f * parallax); first-order, ignores pose error"}
    final = {"ok": True, "point_quality": quality, "R": res["R"], "t": res["t"], "C": res["C"], "points": Xs, "point_view_ids": first,
             "point_track_lengths": lengths, "point_track_index": orig_track, "focal_scale": res["focal_scale"],
             "K_refined": K, "obs": sub, "rounds": rounds,
             "residual_stats_before_first_round_px": _err_stats(first_before),
             "residual_stats_final_px": _err_stats(res["residual_after"]), "gauge": res["gauge"], "prior": res["prior"], "final_gate": final_gate,
             "info": {**info, "points_final": int(len(Xs)), "observations_final": int(len(sub["frame"]))}}
    C0 = camera_centres(R0, t0)
    final["camera_centre_shift"] = {"rms": float(np.sqrt(np.mean(np.sum((res["C"] - C0) ** 2, axis=1)))),
                                    "max": float(np.max(np.linalg.norm(res["C"] - C0, axis=1))),
                                    "centre_extent": float(np.linalg.norm(np.ptp(C0, axis=0)))}
    cosang = (np.einsum("nij,nij->n", res["R"], R0) - 1) / 2
    final["rotation_shift_deg"] = {"mean": float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).mean()),
                                   "max": float(np.degrees(np.arccos(np.clip(cosang, -1, 1))).max())}
    return final


def refined_extrinsics(result):
    return np.concatenate([result["R"], result["t"][:, :, None]], axis=2).astype(np.float32)


def sample_depth_prior(prediction):
    """Observation sampler giving DA3 depth at the first observation of each track (NaN elsewhere)."""
    depth, conf = np.asarray(prediction["depth"]), np.asarray(prediction["confidence"])

    def sampler(obs):
        out = np.full(len(obs["frame"]), np.nan)
        first = np.zeros(obs["track_id"].max() + 1, np.int64)
        first[obs["track_id"][::-1]] = np.arange(len(obs["track_id"]))[::-1]
        for j in first:
            fr = obs["frame"][j]
            x, y = np.round(obs["xy"][j]).astype(int)
            if 0 <= y < depth.shape[1] and 0 <= x < depth.shape[2]:
                z = depth[fr, y, x]
                if np.isfinite(z) and z > 0 and np.isfinite(conf[fr, y, x]):
                    out[j] = z
        return out
    return sampler


# ---------------------------------------------------------------------------------------- scale-bias report
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def horizontal_directions(points, up, k=16, seed=0, max_normals=6000):
    """Dominant Manhattan yaw from wall-like normals of a (dense) cloud. Returns two orthonormal horizontal dirs."""
    import axis_candidates as ac
    up = _unit(up)
    n = ac.normals_from_points(points, k=k, max_normals=max_normals, seed=seed)
    n = n[np.abs(n @ up) < 0.25] if len(n) else n
    seed_vec = np.array([1., 0, 0]) if abs(up[0]) < 0.9 else np.array([0., 1, 0])
    e1 = _unit(np.cross(up, seed_vec))
    e2 = np.cross(up, e1)
    if len(n) < 20:
        return e1, e2, 0
    th = np.arctan2(n @ e2, n @ e1)
    mean4 = np.angle(np.mean(np.exp(4j * th))) / 4
    d1 = np.cos(mean4) * e1 + np.sin(mean4) * e2
    return d1, np.cross(up, d1), int(len(n))


def _mode_extreme(x, side, bins=100):
    q = np.percentile(x, 10 if side == "lo" else 90)
    sel = x[x <= q] if side == "lo" else x[x >= q]
    if len(sel) < 5:
        return float(q)
    hist, edges = np.histogram(sel, bins=max(10, bins // 5))
    j = int(np.argmax(hist))
    return float(0.5 * (edges[j] + edges[j + 1]))


def wall_extents(points, up, floor_h, ceil_h, dirs, margin=0.12):
    """Wall-to-wall extents along `dirs` using points between floor and ceiling (trimmed by `margin` of height)."""
    up = _unit(up)
    h = points @ up
    lo, hi = min(floor_h, ceil_h), max(floor_h, ceil_h)
    band = (h > lo + margin * (hi - lo)) & (h < hi - margin * (hi - lo))
    out = []
    for d in dirs:
        x = points[band] @ d
        if len(x) < 50:
            out.append({"n_band": int(len(x)), "extent_mode": None, "extent_p2_p98": None})
            continue
        out.append({"n_band": int(len(x)), "extent_mode": _mode_extreme(x, "hi") - _mode_extreme(x, "lo"),
                    "extent_p2_p98": float(np.percentile(x, 98) - np.percentile(x, 2))})
    return out


def _fit_cap(points, up, a, b, h0, slab, rounds=4):
    """Robust plane h = c0 + c1 u + c2 v near height h0 (slab follows the plane). Returns dict or None."""
    u, v, h = points @ a, points @ b, points @ up
    A_all = np.column_stack([np.ones(len(h)), u, v])
    coef = np.array([h0, 0., 0.])
    keep = None
    for _ in range(rounds):
        sel = np.abs(h - A_all @ coef) <= slab
        if sel.sum() < 30:
            return None
        A, y = A_all[sel], h[sel]
        keep = np.ones(len(y), bool)
        for _ in range(4):
            if keep.sum() < 30:
                return None
            coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            res = y - A @ coef
            mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
            keep = np.abs(res) <= max(2.5 * mad, 0.01 * slab)
    return {"coef": coef, "inliers": int(keep.sum()), "mad": float(mad)}


def plane_pair_height(points, up, view_ids=None, *, bin_fraction=0.015, prominence=4.0, min_share=0.03):
    """Own floor/ceiling estimator for SPARSE clouds (height_diagnosis' interior mask assumes few wall points).

    Height histogram along `up`; walls give a flat background, floor/ceiling give spikes. The lowest and
    highest prominent spikes are refined by robust plane fits (tilt allowed) and separated along the mean
    normal at the cloud's horizontal centroid. Leave-one-view-out spread is reported when view ids are given.
    """
    up = _unit(up)
    pts = np.asarray(points, float)
    pts = pts[np.isfinite(pts).all(axis=1)] if view_ids is None else pts
    out = {"status": "incomplete", "reasons": [], "separation": None}
    if len(pts) < 150:
        out["reasons"].append("fewer than 150 points")
        return out
    h = pts @ up
    lo, hi = np.percentile(h, [1, 99])
    rng_h = hi - lo
    if rng_h <= 0:
        out["reasons"].append("degenerate height range")
        return out
    bw = bin_fraction * rng_h
    nb = int(np.ceil((hi - lo) / bw)) + 1
    counts, edges = np.histogram(h, bins=nb, range=(lo, lo + nb * bw))
    background = max(float(np.median(counts[counts > 0])) if (counts > 0).any() else 1.0, 1.0)
    cand = [i for i in range(nb) if counts[i] >= prominence * background and counts[i] >= min_share * len(h) * 0.25 and
            counts[i] >= counts[max(0, i - 1):i + 2].max()]
    if len(cand) < 2:
        out["reasons"].append(f"fewer than two prominent height spikes ({len(cand)})")
        return out
    # merge spikes closer than 8% of the height range, keep the strongest of each cluster
    cand.sort(key=lambda i: -counts[i])
    kept = []
    for i in cand:
        if all(abs(i - j) * bw >= 0.08 * rng_h for j in kept):
            kept.append(i)
    kept.sort()
    floor_i, ceil_i = kept[0], kept[-1]
    if (ceil_i - floor_i) * bw < 0.3 * rng_h:
        out["reasons"].append("floor-like and ceiling-like spikes too close")
        return out
    seed = np.array([1., 0, 0]) if abs(up[0]) < 0.9 else np.array([0., 1, 0])
    a = _unit(np.cross(up, seed))
    b = np.cross(up, a)
    slab = 1.5 * bw

    def solve(P):
        hh = P @ up
        fits = []
        for i in (floor_i, ceil_i):
            f = _fit_cap(P, up, a, b, float(lo + (i + 0.5) * bw), slab)
            if f is None:
                return None
            fits.append(f)
        c = P.mean(axis=0)
        u0, v0 = c @ a, c @ b
        hs = [f["coef"][0] + f["coef"][1] * u0 + f["coef"][2] * v0 for f in fits]
        n = [_unit(up - f["coef"][1] * a - f["coef"][2] * b) for f in fits]
        mean_n = _unit(n[0] + n[1])
        p = [u0 * a + v0 * b + hh_ * up for hh_ in hs]
        ang = float(np.degrees(np.arccos(np.clip(n[0] @ n[1], -1, 1))))
        return float(mean_n @ (p[1] - p[0])), hs, ang, fits

    full = solve(pts)
    if full is None:
        out["reasons"].append("plane fit failed")
        return out
    sep, hs, ang, fits = full
    out.update(separation=sep, floor_h=float(hs[0]), ceiling_h=float(hs[1]), parallel_angle_deg=ang,
               floor_inliers=fits[0]["inliers"], ceiling_inliers=fits[1]["inliers"], n_points=int(len(pts)),
               floor_mad=fits[0]["mad"], ceiling_mad=fits[1]["mad"], status="complete")
    if ang > 6.0:
        out["status"], out["reasons"] = "ambiguous", [f"floor/ceiling fits {ang:.1f} deg apart"]
    if view_ids is not None:
        ids = np.asarray(view_ids)
        seps = []
        for v in np.unique(ids):
            r = solve(pts[ids != v])
            if r is not None:
                seps.append(r[0])
        if len(seps) >= 3:
            out["lovo_relative_range"] = float((max(seps) - min(seps)) / abs(sep))
    return out


def cloud_room_metrics(points, view_ids, up, dirs):
    """Floor-to-ceiling separation (own plane-pair estimator + height_diagnosis when it completes) and wall extents."""
    import height_diagnosis as hd
    est = plane_pair_height(points, up, view_ids)
    out = {"status": est["status"], "reasons": est["reasons"], "separation": est["separation"],
           "n_points": int(len(points)), "plane_pair": est}
    try:
        hdr = hd.estimate_room_height(points, up, view_ids)
        out["height_diagnosis"] = {"status": hdr.get("status"), "separation": hdr.get("separation_unscaled"),
                                   "reasons": hdr.get("reasons")}
    except Exception as exc:  # estimator refusals must not break the report
        out["height_diagnosis"] = {"status": "error", "reasons": [str(exc)]}
    if est["separation"] is None:
        return out
    out["floor_h"], out["ceiling_h"] = est["floor_h"], est["ceiling_h"]
    out["lovo_relative_range"] = est.get("lovo_relative_range")
    walls = wall_extents(points, up, out["floor_h"], out["ceiling_h"], dirs)
    out["walls"] = walls
    H = out["separation"]
    for w in walls:
        for key in ("extent_mode", "extent_p2_p98"):
            w["ratio_" + key] = (w[key] / H) if w[key] else None
    return out

def scale_bias_report(sparse_points, sparse_view_ids, dense_points, dense_view_ids, up):
    """Truth-free comparison: wall extent / floor-to-ceiling separation from sparse triangulated points vs DA3
    dense depth, along identical horizontal directions (Manhattan yaw taken from the dense cloud's normals)."""
    up = _unit(up)
    d1, d2, n_normals = horizontal_directions(dense_points, up)
    dirs = [d1, d2]
    sparse = cloud_room_metrics(sparse_points, sparse_view_ids, up, dirs)
    dense = cloud_room_metrics(dense_points, dense_view_ids, up, dirs)
    comp = {"horizontal_dirs": [d1.tolist(), d2.tolist()], "n_wall_normals": n_normals,
            "label": "truth-free; arbitrary units; ratio = wall extent / floor-to-ceiling separation",
            "sparse": sparse, "dense_da3": dense, "ratio_sparse_over_dense": None}
    if sparse.get("walls") and dense.get("walls"):
        comp["ratio_sparse_over_dense"] = [
            {key: (s["ratio_" + key] / d["ratio_" + key]) if s.get("ratio_" + key) and d.get("ratio_" + key) else None
             for key in ("extent_mode", "extent_p2_p98")} for s, d in zip(sparse["walls"], dense["walls"])]
        comp["height_sparse_over_dense"] = (sparse["separation"] / dense["separation"]
                                            if dense.get("separation") else None)
    return comp


def dense_points_per_view(prediction, stride=6):
    """DA3 dense points (extract_planes sampling) with the originating view index."""
    from extract_planes import points_from_prediction
    chunks, ids = [], []
    for i in range(len(prediction["depth"])):
        part = {k: np.asarray(prediction[k])[i:i + 1] for k in ("depth", "confidence", "intrinsics", "extrinsics")}
        try:
            p = points_from_prediction(part, stride=stride)
        except ValueError:
            continue
        chunks.append(p)
        ids.append(np.full(len(p), i))
    return np.concatenate(chunks), np.concatenate(ids)


def depth_consistency(prediction, points, view_ids):
    """Truth-free check: triangulated depth in the point's first view / DA3 depth at the same pixel."""
    E = np.asarray(prediction["extrinsics"], float)
    K = np.asarray(prediction["intrinsics"], float)
    D = np.asarray(prediction["depth"])
    v = np.asarray(view_ids)
    Xc = np.einsum("nij,nj->ni", E[v, :, :3], points) + E[v, :, 3]
    uv = np.einsum("nij,nj->ni", K[v], Xc)
    x = np.round(uv[:, 0] / uv[:, 2]).astype(int)
    y = np.round(uv[:, 1] / uv[:, 2]).astype(int)
    ok = (x >= 0) & (x < D.shape[2]) & (y >= 0) & (y < D.shape[1])
    z_da3 = np.full(len(points), np.nan)
    z_da3[ok] = D[v[ok], y[ok], x[ok]]
    ratio = Xc[:, 2] / z_da3
    good = np.isfinite(ratio) & (z_da3 > 0)
    per_cam = {int(c): float(np.median(ratio[good & (v == c)])) for c in np.unique(v[good])
               if (good & (v == c)).sum() >= 20}
    return {"n": int(good.sum()), "ratio_p5_p25_p50_p75_p95": [float(q) for q in np.percentile(ratio[good], [5, 25, 50, 75, 95])],
            "per_camera_median_ratio_min_max": [min(per_cam.values()), max(per_cam.values())] if per_cam else None,
            "note": "z_triangulated / z_DA3 at the first observation; 1.0 = DA3 depth and DA3-pose triangulation agree"}


def config_dict(track_cfg, ba_cfg):
    return {"track": asdict(track_cfg), "ba": asdict(ba_cfg)}
