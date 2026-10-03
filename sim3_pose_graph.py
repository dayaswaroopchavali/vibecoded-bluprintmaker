"""Generic robust Sim(3) pose-graph optimiser (numpy only; no ground truth, no simulator, no camera-up, no scene constants).

Model
  node k has an unknown local->global similarity T_k = (s_k, Q_k, b_k):  x_g = s_k Q_k x_local + b_k  (s>0, Q proper).
  edge (i, j) carries a measured relative similarity (s, Q, b) with  x_i = s Q x_j + b,  i.e. the prediction is
  T_i^{-1} T_j = (s_j/s_i,  Q_i^T Q_j,  Q_i^T (b_j - b_i)/s_i).

Residual per edge (7 numbers, whitened):
  [ Log(Q^T Q_pred) / sigma_rot,  log(s_pred / s) / sigma_logscale,  (b_pred - b) / (trans_scale * sigma_trans) ]
  then multiplied by sqrt(weight / mean weight).  u = norm of that 7-vector is fed to the robust loss rho(u).

Optimisation
  * Updates live on the Lie algebra: per node  d = (w[3], log-scale[1], dv[3]);  Q <- Q Exp(w) (always proper),
    s <- s exp(ls) (always > 0), b <- b + dv.  Jacobians are central differences of the vectorised residual (each edge
    only touches its two nodes, so the Gauss-Newton Hessian is block sparse).
  * Levenberg-Marquardt with IRLS weights w_e = rho'(u)/u (Cauchy, Huber, Tukey biweight, or plain L2).
  * Gauge: one root node per connected component is held fixed (lowest node id, or a `fixed` node); with no `init`
    the root is the identity.  Components never interact and never get bridged.
  * Initialisation: `init`, else a maximum-weight spanning forest chained from each root (reported as baseline).
  * Loss scale (whitened units): given, or estimated once from the initial residuals of the non-tree (cycle) edges as
    k * 1.4826 * median(u) (k = 2.385 Cauchy, 1.345 Huber, 4.685 Tukey), floored at 0.1; no cycles -> no robust scale
    is estimable, the loss degrades to L2 (warning).  Never uses truth.
Complexity: residual/Jacobian O(E); normal equations are dense (7 n_c)^2 per component with an O((7 n_c)^3) solve per
LM trial, so fine to ~300 nodes (115 nodes / 330 edges: well under a second per solve, see tests).
Defaults sigma_rot=sigma_logscale=sigma_trans=0.01 are nominal noise normalisers that only fix the relative weighting of
the three residual blocks; `loss_scale` is expressed in these whitened units.
"""
from __future__ import annotations

import math

import numpy as np

SIGMA_DEFAULT = 0.01
OUTLIER_WEIGHT = 0.3        # final robust weight below this flags an edge as an outlier
MIN_LOSS_SCALE = 0.1
_K = {"cauchy": 2.385, "huber": 1.345, "tukey": 4.685}


# ----------------------------------------------------------------------------- Lie group helpers (vectorised)
def _hat(w):
    z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([z, -w[..., 2], w[..., 1]], -1), np.stack([w[..., 2], z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def _exp_so3(w):
    w = np.asarray(w, float)
    th = np.linalg.norm(w, axis=-1)[..., None, None]
    K = _hat(w)
    small = th < 1e-8
    ths = np.where(small, 1.0, th)
    A = np.where(small, 1.0 - th ** 2 / 6, np.sin(ths) / ths)
    B = np.where(small, 0.5 - th ** 2 / 24, (1 - np.cos(ths)) / ths ** 2)
    return np.eye(3) + A * K + B * (K @ K)


def _log_so3(R):
    R = np.asarray(R, float)
    flat = R.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 3))
    v = np.stack([flat[:, 2, 1] - flat[:, 1, 2], flat[:, 0, 2] - flat[:, 2, 0], flat[:, 1, 0] - flat[:, 0, 1]], -1) / 2
    c = np.clip((np.trace(flat, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    s = np.sin(th)
    mid = th < math.pi - 1e-3
    fac = np.where(th < 1e-8, 1.0, th / np.where(s < 1e-12, 1.0, s))
    out[:] = v * fac[:, None]
    for n in np.nonzero(~mid)[0]:                        # near pi: axis from (R + I)/2 = a a^T
        M = (flat[n] + np.eye(3)) / 2
        k = int(np.argmax(np.diag(M)))
        a = M[:, k] / math.sqrt(max(M[k, k], 1e-300))
        a /= np.linalg.norm(a)
        if np.dot(a, v[n]) < 0:
            a = -a
        out[n] = a * th[n]
    return out.reshape(R.shape[:-2] + (3,))


def _angle_deg(R):
    return np.degrees(np.linalg.norm(_log_so3(R), axis=-1))


def _proper(Q):
    U, _, Vt = np.linalg.svd(np.asarray(Q, float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def _residual(si, Qi, bi, sj, Qj, bj, s, Q, b, ts, sig):
    """Whitened residual (m, 7) and raw parts, all arrays batched over edges."""
    sP = sj / si
    QP = np.swapaxes(Qi, -1, -2) @ Qj
    bP = (np.swapaxes(Qi, -1, -2) @ (bj - bi)[..., None])[..., 0] / si[:, None]
    rot = _log_so3(np.swapaxes(Q, -1, -2) @ QP)
    ls = np.log(sP / s)
    tr = (bP - b) / ts[:, None]
    r = np.concatenate([rot / sig[:, 0:1], ls[:, None] / sig[:, 1:2], tr / sig[:, 2:3]], 1)
    return r, rot, ls, tr


def _apply(s, Q, b, d):
    return s * np.exp(d[..., 3]), Q @ _exp_so3(d[..., 0:3]), b + d[..., 4:7]


def _rho(u, loss, c):
    if loss == "l2":
        return 0.5 * u * u
    if loss == "cauchy":
        return 0.5 * c * c * np.log1p((u / c) ** 2)
    if loss == "huber":
        return np.where(u <= c, 0.5 * u * u, c * (u - 0.5 * c))
    z = np.minimum(u / c, 1.0)                                   # tukey
    return (c * c / 6) * (1 - (1 - z * z) ** 3)


def _irls(u, loss, c):
    if loss == "l2":
        return np.ones_like(u)
    if loss == "cauchy":
        return 1.0 / (1.0 + (u / c) ** 2)
    if loss == "huber":
        return np.where(u <= c, 1.0, c / np.maximum(u, 1e-300))
    return np.where(u < c, (1 - (u / c) ** 2) ** 2, 0.0)


# ----------------------------------------------------------------------------- graph helpers
def _components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = sorted({find(k) for k in range(n)})
    cid = {r: c for c, r in enumerate(roots)}
    return [cid[find(k)] for k in range(n)]


def _tree_init(n, E, comp_of, root_of, w):
    """Maximum-weight spanning forest, chained from each root. Returns transforms list and tree edge index set."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    order = sorted(range(len(E["i"])), key=lambda e: (-w[e], E["i"][e], E["j"][e], e))
    adj = {k: [] for k in range(n)}
    tree = set()
    for e in order:
        i, j = E["i"][e], E["j"][e]
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            tree.add(e)
            adj[i].append((j, e))
            adj[j].append((i, e))
    T = [None] * n
    for r in root_of:
        T[r] = (1.0, np.eye(3), np.zeros(3))
        stack = [r]
        while stack:
            u = stack.pop()
            for v, e in adj[u]:
                if T[v] is not None:
                    continue
                s, Q, b = E["s"][e], E["Q"][e], E["b"][e]
                su, Qu, bu = T[u]
                if v == E["j"][e]:                       # u = i, v = j:  T_j = T_i (s,Q,b)
                    T[v] = (su * s, Qu @ Q, su * (Qu @ b) + bu)
                else:                                    # u = j, v = i:  T_i = T_j (s,Q,b)^-1
                    si, Qi, bi = 1.0 / s, Q.T, -(Q.T @ b) / s
                    T[v] = (su * si, Qu @ Qi, su * (Qu @ bi) + bu)
                stack.append(v)
    return T, tree


def _edge_arrays(edges, n):
    E = {"i": [], "j": [], "s": [], "Q": [], "b": [], "w": [], "ts": [], "sr": [], "sl": []}
    for k, e in enumerate(edges):
        i, j = int(e["i"]), int(e["j"])
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"edge {k}: node index out of range")
        if i == j:
            raise ValueError(f"edge {k}: self loop")
        s = float(e["s"])
        Q = np.asarray(e["Q"], float).reshape(3, 3)
        b = np.asarray(e["b"], float).reshape(3)
        w = float(e.get("weight", 1.0))
        if not (np.isfinite(s) and s > 0 and np.all(np.isfinite(Q)) and np.all(np.isfinite(b)) and w > 0):
            raise ValueError(f"edge {k}: non-finite, s<=0 or weight<=0")
        if np.linalg.det(Q) < 0 or np.abs(Q.T @ Q - np.eye(3)).max() > 1e-3:
            raise ValueError(f"edge {k}: Q is not a proper rotation")
        E["i"].append(i); E["j"].append(j); E["s"].append(s); E["Q"].append(_proper(Q)); E["b"].append(b); E["w"].append(w)
        E["ts"].append(e.get("trans_scale")); E["sr"].append(e.get("sigma_rot_rad")); E["sl"].append(e.get("sigma_logscale"))
    return E


def _edge_res_report(T, E, ts, sig, idx):
    S = np.array([t[0] for t in T]); Qs = np.stack([t[1] for t in T]); B = np.stack([t[2] for t in T])
    i, j = np.array(E["i"])[idx], np.array(E["j"])[idx]
    _, rot, ls, tr = _residual(S[i], Qs[i], B[i], S[j], Qs[j], B[j], np.array(E["s"])[idx], np.stack(E["Q"])[idx],
                               np.stack(E["b"])[idx], ts[idx], sig[idx])
    return rot, ls, tr


def optimize_sim3_graph(num_nodes, edges, *, init=None, loss="cauchy", loss_scale=None, max_iters=50, tol=1e-9,
                        damping=1e-3, trans_scale=None, fixed=None):
    """Robust Sim(3) pose-graph optimisation. See module docstring.

    edges: list of {i, j, s, Q, b, weight=1, trans_scale=None, sigma_rot_rad=None, sigma_logscale=None}.
    fixed: dict {node: (s,Q,b)} or iterable of nodes (held at their init value). Returns a dict (keys in the tests).
    """
    if loss not in ("cauchy", "huber", "tukey", "l2"):
        raise ValueError("unknown loss")
    n = int(num_nodes)
    warnings = []
    E = _edge_arrays(edges, n)
    m = len(E["i"])
    w_in = np.array(E["w"], float)
    comp_of = _components(n, zip(E["i"], E["j"]))
    ncomp = (max(comp_of) + 1) if n else 0
    components = [[k for k in range(n) if comp_of[k] == c] for c in range(ncomp)]

    fixed_map = {}
    if fixed is not None:
        if isinstance(fixed, dict):
            fixed_map = {int(k): v for k, v in fixed.items()}
        else:
            fixed_map = {int(k): None for k in fixed}
        for k in fixed_map:
            if not 0 <= k < n:
                raise ValueError("fixed node out of range")
    root_of = []
    for c, nodes in enumerate(components):
        fx = [k for k in nodes if k in fixed_map]
        root_of.append(fx[0] if fx else nodes[0])

    # per-edge normalisation
    gts = trans_scale
    if gts is None:
        nb = np.array([np.linalg.norm(b) for b in E["b"]]) if m else np.array([])
        nb = nb[nb > 0]
        gts = float(np.median(nb)) if nb.size else 1.0
    ts = np.array([gts if t is None else float(t) for t in E["ts"]], float)
    if np.any(ts <= 0):
        raise ValueError("trans_scale must be > 0")
    sig = np.array([[SIGMA_DEFAULT if a is None else float(a), SIGMA_DEFAULT if bb is None else float(bb), SIGMA_DEFAULT]
                    for a, bb in zip(E["sr"], E["sl"])], float).reshape(m, 3)
    wn = w_in / w_in.mean() if m else w_in
    sw = np.sqrt(wn)

    # initial solution
    tree_T, tree = _tree_init(n, E, comp_of, root_of, w_in)
    if init is not None:
        if len(init) != n:
            raise ValueError("init length != num_nodes")
        T0 = [(float(s), _proper(Q), np.asarray(b, float).reshape(3)) for s, Q, b in init]
    else:
        T0 = tree_T
    for k, v in fixed_map.items():
        if v is not None:
            T0[k] = (float(v[0]), np.array(v[1], float).reshape(3, 3), np.array(v[2], float).reshape(3))
    init_T = [(t[0], t[1].copy(), t[2].copy()) for t in T0]

    cur = [(t[0], t[1].copy(), t[2].copy()) for t in T0]
    final_w = np.zeros(m)
    objs0 = np.zeros(ncomp); objs1 = np.zeros(ncomp)
    iters = [0] * ncomp; conv = [True] * ncomp
    scales = [None] * ncomp
    tree_mask = np.zeros(m, bool)
    for e in tree:
        tree_mask[e] = True
    if m:
        eI, eJ = np.array(E["i"]), np.array(E["j"])
        eS, eQ, eB = np.array(E["s"]), np.stack(E["Q"]), np.stack(E["b"])

    for c, nodes in enumerate(components):
        eidx = np.array([e for e in range(m) if comp_of[E["i"][e]] == c], dtype=int)
        if eidx.size == 0:
            continue
        loc = {k: a for a, k in enumerate(nodes)}
        nc = len(nodes)
        li = np.array([loc[k] for k in eI[eidx]]); lj = np.array([loc[k] for k in eJ[eidx]])
        S = np.array([cur[k][0] for k in nodes]); Qs = np.stack([cur[k][1] for k in nodes]); Bs = np.stack([cur[k][2] for k in nodes])
        frozen = np.array([k == root_of[c] or k in fixed_map for k in nodes])
        free = np.nonzero(~frozen)[0]
        fidx = {a: p for p, a in enumerate(free)}
        es, eq, eb, ets, esig, esw = eS[eidx], eQ[eidx], eB[eidx], ts[eidx], sig[eidx], sw[eidx]

        def evalr(S_, Q_, B_):
            r = _residual(S_[li], Q_[li], B_[li], S_[lj], Q_[lj], B_[lj], es, eq, eb, ets, esig)[0] * esw[:, None]
            return r, np.linalg.norm(r, axis=1)

        r, u = evalr(S, Qs, Bs)
        if loss == "l2":
            cl = 1.0
        elif loss_scale is not None:
            cl = float(loss_scale)
        else:
            nt = ~tree_mask[eidx]
            if nt.any():
                cl = max(_K[loss] * 1.4826 * float(np.median(u[nt])), MIN_LOSS_SCALE)
            else:
                cl = 1e12
                warnings.append(f"component {c}: no cycles, robust loss scale not estimable; behaves as L2")
        scales[c] = cl
        obj = float(_rho(u, loss, cl).sum()); objs0[c] = obj
        lam = float(damping)
        done = False
        for it in range(int(max_iters)):
            iters[c] = it
            wt = _irls(u, loss, cl)
            # numeric Jacobians of the weighted residual wrt right-perturbation of node i and node j (m, 7, 7)
            Ji = np.zeros((len(eidx), 7, 7)); Jj = np.zeros_like(Ji)
            h = 1e-6
            for p in range(7):
                d = np.zeros((len(eidx), 7)); d[:, p] = h
                for J, side in ((Ji, li), (Jj, lj)):
                    sp, Qp, bp = _apply(S[side], Qs[side], Bs[side], d)
                    sm, Qm, bm = _apply(S[side], Qs[side], Bs[side], -d)
                    if side is li:
                        rp = _residual(sp, Qp, bp, S[lj], Qs[lj], Bs[lj], es, eq, eb, ets, esig)[0]
                        rm = _residual(sm, Qm, bm, S[lj], Qs[lj], Bs[lj], es, eq, eb, ets, esig)[0]
                    else:
                        rp = _residual(S[li], Qs[li], Bs[li], sp, Qp, bp, es, eq, eb, ets, esig)[0]
                        rm = _residual(S[li], Qs[li], Bs[li], sm, Qm, bm, es, eq, eb, ets, esig)[0]
                    J[:, :, p] = (rp - rm) / (2 * h) * esw[:, None]
            Hf = np.zeros((nc, nc, 7, 7)); g = np.zeros((nc, 7))
            Wj = wt[:, None, None]
            Hii = np.einsum("eki,ekl->eil", Ji, Wj * Ji); Hjj = np.einsum("eki,ekl->eil", Jj, Wj * Jj)
            Hij = np.einsum("eki,ekl->eil", Ji, Wj * Jj)
            gi = np.einsum("eki,ek->ei", Ji, wt[:, None] * r); gj = np.einsum("eki,ek->ei", Jj, wt[:, None] * r)
            np.add.at(Hf, (li, li), Hii); np.add.at(Hf, (lj, lj), Hjj)
            np.add.at(Hf, (li, lj), Hij); np.add.at(Hf, (lj, li), np.swapaxes(Hij, 1, 2))
            np.add.at(g, li, gi); np.add.at(g, lj, gj)
            if free.size == 0:
                done = True
                break
            sel = (free[:, None] * 7 + np.arange(7)[None, :]).ravel()
            H = Hf.transpose(0, 2, 1, 3).reshape(nc * 7, nc * 7)[np.ix_(sel, sel)]
            gv = g.reshape(-1)[sel]
            accepted = False
            for _ in range(12):
                A = H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(H.shape[0])
                try:
                    dx = np.linalg.solve(A, -gv)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                d = np.zeros((nc, 7)); d[free] = dx.reshape(-1, 7)
                S2, Q2, B2 = _apply(S, Qs, Bs, d)
                r2, u2 = evalr(S2, Q2, B2)
                obj2 = float(_rho(u2, loss, cl).sum())
                if np.isfinite(obj2) and obj2 <= obj:
                    accepted = True
                    break
                lam *= 4
            if not accepted:
                done = True                                    # no descent direction left: local minimum
                break
            dec = obj - obj2
            S, Qs, Bs, r, u, obj = S2, Q2, B2, r2, u2, obj2
            lam = max(lam / 3, 1e-12)
            if dec <= tol * (1 + obj) or np.linalg.norm(dx) < tol:
                done = True
                break
        else:
            iters[c] = int(max_iters)
        conv[c] = bool(done)
        if not done:
            warnings.append(f"component {c}: not converged in {max_iters} iterations")
        objs1[c] = obj
        final_w[eidx] = _irls(u, loss, cl)
        for a, k in enumerate(nodes):
            cur[k] = (float(S[a]), Qs[a].copy(), Bs[a].copy())

    # reports
    idx = np.arange(m)
    out_edges_before, out_edges_after = [], []
    if m:
        rb = _edge_res_report(init_T, E, ts, sig, idx); ra = _edge_res_report(cur, E, ts, sig, idx)
        for rr, dst in ((rb, out_edges_before), (ra, out_edges_after)):
            for e in range(m):
                dst.append({"rot_deg": float(np.degrees(np.linalg.norm(rr[0][e]))), "log_scale": float(rr[1][e]),
                            "trans_norm": float(np.linalg.norm(rr[2][e]))})
    nts = {k: [] for k in range(n)}
    for e in range(m):
        nts[E["i"][e]].append(ts[e]); nts[E["j"][e]].append(ts[e])
    corr = []
    for k in range(n):
        s0, Q0, b0 = init_T[k]; s1, Q1, b1 = cur[k]
        t = float(np.median(nts[k])) if nts[k] else 1.0
        corr.append({"rot_deg": float(_angle_deg(Q0.T @ Q1)), "scale_ratio": float(s1 / s0),
                     "trans": float(np.linalg.norm(b1 - b0) / t)})
    outliers = [e for e in range(m) if final_w[e] < OUTLIER_WEIGHT]
    return {
        "transforms": [{"s": t[0], "Q": t[1], "b": t[2]} for t in cur],
        "init_transforms": [{"s": t[0], "Q": t[1], "b": t[2]} for t in init_T],
        "component_of_node": comp_of, "components": components, "root_of_component": root_of,
        "edge_residuals_before": out_edges_before, "edge_residuals_after": out_edges_after,
        "initial_objective": float(objs0.sum()), "final_objective": float(objs1.sum()),
        "iterations": int(max(iters)) if iters else 0, "converged": bool(all(conv)),
        "final_robust_weights": [float(x) for x in final_w], "per_node_correction": corr,
        "outlier_edges": outliers, "outlier_weight_threshold": OUTLIER_WEIGHT,
        "loss": loss, "loss_scale_per_component": scales, "warnings": warnings,
    }
