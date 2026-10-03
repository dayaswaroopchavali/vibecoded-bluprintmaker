import ast
import time
import unittest
from pathlib import Path

import numpy as np

import sim3_pose_graph as pg
from sim3_pose_graph import optimize_sim3_graph


def rand_rot(rng, max_deg=180.0):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    return pg._exp_so3(ax * np.radians(rng.uniform(0, max_deg)))


def rel(Ti, Tj):
    """T_i^-1 T_j as (s,Q,b)."""
    si, Qi, bi = Ti; sj, Qj, bj = Tj
    return sj / si, Qi.T @ Qj, Qi.T @ (bj - bi) / si


def truth(rng, n, smin=1 / 30, smax=30, spread=5.0, max_deg=180.0):
    return [(float(np.exp(rng.uniform(np.log(smin), np.log(smax)))), rand_rot(rng, max_deg), rng.normal(size=3) * spread)
            for _ in range(n)]


def make_edges(rng, G, pairs, rot_noise=0.0, log_noise=0.0, tr_noise=0.0, ts=1.0, weight=1.0):
    edges = []
    for i, j in pairs:
        s, Q, b = rel(G[i], G[j])
        s = s * np.exp(rng.normal() * log_noise)
        Q = Q @ pg._exp_so3(rng.normal(size=3) * rot_noise)
        b = b + rng.normal(size=3) * tr_noise * ts
        edges.append({"i": i, "j": j, "s": s, "Q": Q, "b": b, "weight": weight, "trans_scale": ts})
    return edges


def pair_error(G, T):
    """Gauge-invariant error over all node pairs: rot deg, |log s|, normalised translation (mean)."""
    n = len(G); rots, lss, trs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            sg, Qg, bg = rel(G[i], G[j]); st, Qt, bt = rel(T[i], T[j])
            rots.append(np.degrees(np.linalg.norm(pg._log_so3(Qt.T @ Qg)))); lss.append(abs(np.log(sg / st)))
            trs.append(np.linalg.norm(bg - bt))
    return float(np.mean(rots)), float(np.mean(lss)), float(np.mean(trs))


def tup(res, key="transforms"):
    return [(t["s"], t["Q"], t["b"]) for t in res[key]]


def loop_graph(n, extra, rng):
    pairs = [(k, k + 1) for k in range(n - 1)]
    seen = set(pairs)
    while len(pairs) < n - 1 + extra:
        i, j = sorted(rng.choice(n, 2, replace=False))
        if (i, j) not in seen and j - i > 1:
            seen.add((i, j)); pairs.append((int(i), int(j)))
    return pairs


class Sim3Tests(unittest.TestCase):
    def test_a_tree_exact(self):
        rng = np.random.default_rng(1); G = truth(rng, 8)
        pairs = [(0, 1), (1, 2), (1, 3), (3, 4), (0, 5), (5, 6), (5, 7)]
        r = optimize_sim3_graph(8, make_edges(rng, G, pairs))
        self.assertLess(pair_error(G, tup(r))[0], 1e-6)
        self.assertLess(pair_error(G, tup(r))[2], 1e-6)
        self.assertLess(max(e["rot_deg"] + abs(e["log_scale"]) + e["trans_norm"] for e in r["edge_residuals_after"]), 1e-6)
        self.assertLess(r["final_objective"], 1e-9)
        for a, b in zip(r["init_transforms"], r["transforms"]):
            self.assertTrue(np.allclose(a["Q"], b["Q"], atol=1e-6))

    def test_b_loops_improve_over_tree(self):
        wins = 0; ratios = []
        for seed in range(8):
            rng = np.random.default_rng(100 + seed); n = 20
            G = truth(rng, n, smin=0.5, smax=2.0, spread=3.0)
            edges = make_edges(rng, G, loop_graph(n, 30, rng), rot_noise=0.02, log_noise=0.02, tr_noise=0.02)
            r = optimize_sim3_graph(n, edges, loss="l2", trans_scale=1.0)
            e0 = pair_error(G, tup(r, "init_transforms")); e1 = pair_error(G, tup(r))
            self.assertLess(r["final_objective"], r["initial_objective"])
            tot0 = sum(x["rot_deg"] + x["trans_norm"] for x in r["edge_residuals_before"])
            tot1 = sum(x["rot_deg"] + x["trans_norm"] for x in r["edge_residuals_after"])
            self.assertLess(tot1, tot0)
            ratios.append(e1[2] / e0[2]); wins += e1[0] < e0[0] and e1[2] < e0[2]
        self.assertEqual(wins, 8, ratios)
        self.assertLess(float(np.mean(ratios)), 0.8)

    def test_c_outlier_downweighted(self):
        good = {"cauchy": 0, "l2": 0}; errs = {"cauchy": [], "l2": []}
        for seed in range(5):
            rng = np.random.default_rng(200 + seed); n = 16
            G = truth(rng, n, smin=0.5, smax=2.0, spread=3.0)
            pairs = loop_graph(n, 25, rng)
            edges = make_edges(rng, G, pairs, rot_noise=0.005, log_noise=0.005, tr_noise=0.005)
            bad = 5 + 3                                           # a chain edge (tree edge when weights tie)
            e = edges[bad]
            e["Q"] = e["Q"] @ pg._exp_so3(np.array([0.0, 0.0, np.radians(90)])); e["s"] *= 3.0; e["weight"] = 1.0
            edges[bad] = e
            for loss in ("cauchy", "l2"):
                r = optimize_sim3_graph(n, edges, loss=loss, trans_scale=1.0)
                errs[loss].append(pair_error(G, tup(r))[0])
                if loss == "cauchy":
                    self.assertIn(bad, r["outlier_edges"])
                    self.assertLess(r["final_robust_weights"][bad], 0.2)
                    self.assertEqual(len(r["outlier_edges"]), 1, r["outlier_edges"])
        self.assertLess(max(errs["cauchy"]), 0.5)
        print(errs); self.assertGreater(min(errs["l2"]), 3 * max(errs["cauchy"]))

    def test_d_components_separate(self):
        rng = np.random.default_rng(3); G = truth(rng, 9)
        pairs = [(0, 1), (1, 2), (0, 2), (4, 5), (5, 6), (4, 6), (6, 7)]          # nodes 3 and 8 isolated
        r = optimize_sim3_graph(9, make_edges(rng, G, pairs, 0.01, 0.01, 0.01))
        self.assertEqual(r["components"], [[0, 1, 2], [3], [4, 5, 6, 7], [8]])
        self.assertEqual(r["component_of_node"], [0, 0, 0, 1, 2, 2, 2, 2, 3])
        self.assertEqual(r["root_of_component"], [0, 3, 4, 8])
        for root in r["root_of_component"]:
            t = r["transforms"][root]
            self.assertTrue(t["s"] == 1.0 and np.allclose(t["Q"], np.eye(3)) and np.allclose(t["b"], 0))
        for comp in [c for c in r["components"] if len(c) > 1]:   # within-component consistency; no cross-component transform
            sub = [tup(r)[k] for k in comp]
            self.assertLess(pair_error([G[k] for k in comp], sub)[2], 0.1)
        self.assertEqual(len(r["transforms"]), 9)

    def test_e_scale_rotation_proper(self):
        rng = np.random.default_rng(5); n = 12
        G = [(s, rand_rot(rng, 180), rng.normal(size=3) * 4) for s in np.geomspace(1 / 30, 30, n)]
        G[3] = (G[3][0], pg._exp_so3(np.array([np.pi, 0, 0])), G[3][2])           # exactly 180 deg
        pairs = [(k, k + 1) for k in range(n - 1)] + [(0, 5), (2, 9), (1, 11), (4, 8)]
        for loss in ("cauchy", "huber", "tukey", "l2"):
            edges = make_edges(rng, G, pairs, 0.002, 0.002, 0.002, ts=4.0)
            r = optimize_sim3_graph(n, edges, loss=loss)
            for t in r["transforms"] + r["init_transforms"]:
                self.assertGreater(t["s"], 0)
                self.assertAlmostEqual(np.linalg.det(t["Q"]), 1.0, places=8)
                self.assertLess(np.abs(t["Q"].T @ t["Q"] - np.eye(3)).max(), 1e-8)
            rel_err = [abs(np.log(rel(tup(r)[i], tup(r)[j])[0] / rel(G[i], G[j])[0])) for i, j in pairs]
            self.assertLess(max(rel_err), 0.05, loss)
            self.assertLess(pair_error(G, tup(r))[0], 1.5, loss)

    def test_f_gauge_invariance(self):
        rng = np.random.default_rng(7); n = 12
        G = truth(rng, n, 0.5, 2, 3)
        edges = make_edges(rng, G, loop_graph(n, 25, rng), 0.01, 0.01, 0.01)
        res = [optimize_sim3_graph(n, edges, loss="l2", trans_scale=1.0, fixed={root: G[root]}) for root in (0, 5, 11)]
        base = res[0]
        self.assertLess(pair_error(tup(base), tup(res[1]))[2], 1e-6)
        self.assertLess(pair_error(tup(base), tup(res[2]))[0], 1e-5)
        self.assertLess(pair_error(tup(base), tup(res[2]))[1], 1e-6)

    def test_g_deterministic(self):
        rng = np.random.default_rng(9); n = 10
        G = truth(rng, n, 0.5, 2, 3)
        edges = make_edges(rng, G, loop_graph(n, 10, rng), 0.01, 0.01, 0.01)
        a = optimize_sim3_graph(n, edges); b = optimize_sim3_graph(n, edges)
        for x, y in zip(a["transforms"], b["transforms"]):
            self.assertTrue(x["s"] == y["s"] and np.array_equal(x["Q"], y["Q"]) and np.array_equal(x["b"], y["b"]))
        self.assertEqual(a["final_robust_weights"], b["final_robust_weights"])

    def test_h_fixed_nodes(self):
        rng = np.random.default_rng(11); n = 8
        G = truth(rng, n, 0.5, 2, 3)
        edges = make_edges(rng, G, loop_graph(n, 6, rng), 0.02, 0.02, 0.02)
        fx = {3: G[3], 6: G[6]}
        r = optimize_sim3_graph(n, edges, fixed=fx, loss="l2", trans_scale=1.0)
        for k, v in fx.items():
            t = r["transforms"][k]
            self.assertTrue(t["s"] == v[0] and np.array_equal(t["Q"], v[1]) and np.array_equal(t["b"], v[2]))
        self.assertEqual(r["root_of_component"], [3])
        r2 = optimize_sim3_graph(n, edges, init=[(g[0], g[1], g[2]) for g in G], fixed=[2], loss="l2", trans_scale=1.0)
        self.assertTrue(np.array_equal(r2["transforms"][2]["b"], G[2][2]))

    def test_i_degenerate(self):
        r = optimize_sim3_graph(1, [])
        self.assertEqual(r["components"], [[0]]); self.assertEqual(r["root_of_component"], [0]); self.assertTrue(r["converged"])
        r = optimize_sim3_graph(3, [])
        self.assertEqual(r["components"], [[0], [1], [2]])
        self.assertEqual(optimize_sim3_graph(0, [])["transforms"], [])
        rng = np.random.default_rng(13); G = truth(rng, 3)
        e = make_edges(rng, G, [(0, 1), (1, 2)])
        dup = e + [dict(e[0]), dict(e[1], weight=2.0)]                   # exact duplicates (parallel edges)
        r = optimize_sim3_graph(3, dup, loss="l2")
        self.assertLess(pair_error(G, tup(r))[2], 1e-6)
        par = e + [make_edges(rng, G, [(0, 1)], 0.05, 0.05, 0.05)[0]]    # noisy parallel edge: compromise, no crash
        r = optimize_sim3_graph(3, par, loss="l2", trans_scale=1.0)
        self.assertLess(r["final_objective"], r["initial_objective"] + 1e-12)
        self.assertEqual(len(r["final_robust_weights"]), 3)
        with self.assertRaises(ValueError):
            optimize_sim3_graph(2, [{"i": 0, "j": 1, "s": 1.0, "Q": np.diag([1, 1, -1.0]), "b": np.zeros(3)}])

    def test_j_scale_timing_and_purity(self):
        rng = np.random.default_rng(17); n = 115
        G = truth(rng, n, 0.5, 2, 3)
        pairs = [(k, k + d) for k in range(n) for d in (1, 2, 3) if k + d < n]
        edges = make_edges(rng, G, pairs, 0.01, 0.01, 0.01, ts=1.0)
        t = time.time(); r = optimize_sim3_graph(n, edges, loss="cauchy"); dt = time.time() - t
        print(f"\n115 nodes / {len(edges)} edges: {dt:.2f}s, {r['iterations']} iterations")
        self.assertLess(dt, 60)
        self.assertLess(pair_error(G, tup(r))[2], pair_error(G, tup(r, "init_transforms"))[2])
        tree = ast.parse(Path(pg.__file__).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add(node.module.split(".")[0])
        self.assertLessEqual(mods, {"numpy", "math", "typing", "json", "__future__"})


if __name__ == "__main__":
    unittest.main()


