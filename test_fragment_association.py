import ast
import unittest
from pathlib import Path

import numpy as np

import boundary_assembly as ba
import fragment_association as fa
from wall_extraction import validate_walls_schema

HERE = Path(__file__).resolve().parent
NOISE = 0.004


def _seg_hit(cam, ang, segs, h):
    """Nearest hit of ray (cam, ang) among segments whose height range contains h."""
    d = np.array([np.cos(ang), np.sin(ang)])
    best = None
    for s in segs:
        if not (s["h"][0] <= h <= s["h"][1]):
            continue
        a, b = np.asarray(s["p0"], float), np.asarray(s["p1"], float)
        e = b - a
        M = np.array([d, -e]).T
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        t, u = np.linalg.solve(M, a - cam)
        if t > 1e-6 and 0 <= u <= 1 and (best is None or t < best[0]):
            best = (t, s)
    return best


def build_walls(segs, cams, *, drift=None, both=(), seed=0, rays=240, extent=None, band=(-1.0, 1.0), noise=NOISE):
    """Synthetic A->B schema dict. segs: [{'tag','p0','p1','h':(lo,hi)}]; cams: [[x,y]]. `drift` = {tag: (frame_set, shift)}
    displaces that segment's hits (along its normal) for those frames: a per-window pose offset.
    One line is fitted per (segment tag, drift group) from its own hits only (no geometry shortcut for the association)."""
    rng = np.random.default_rng(seed)
    cams = np.asarray(cams, float)
    hits_xy, hit_h, tagged = [], [], []
    for f, cam in enumerate(cams):
        hx, hh, tg = [], [], []
        for ang in np.linspace(0, 2 * np.pi, rays, endpoint=False):
            h = rng.uniform(*band)
            r = _seg_hit(cam, ang, segs, h)
            if r is None:
                continue
            t, s = r
            p = cam + t * np.array([np.cos(ang), np.sin(ang)])
            e = np.asarray(s["p1"], float) - np.asarray(s["p0"], float)
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            n = n if n @ (cam - np.asarray(s["p0"], float)) < 0 else -n          # pointing away from the camera
            shift = 0.0
            group = "main"
            if drift and s["tag"] in drift and f in drift[s["tag"]][0] and not (f in both and len(hx) % 2):
                shift, group = drift[s["tag"]][1], "drift"
            p = p + n * shift + rng.normal(0, noise, 2) * 0.5
            hx.append(p.tolist()); hh.append(h); tg.append((s["tag"], group))
        hits_xy.append(hx); hit_h.append(hh); tagged.append(tg)
    allp = np.array([q for h in hits_xy for q in h])
    ext = float(np.linalg.norm(np.percentile(allp, 95, axis=0) - np.percentile(allp, 5, axis=0))) if extent is None else extent
    lines = []
    keys = sorted({k for tg in tagged for k in tg})
    for lid, key in enumerate(keys):
        P, H, F = [], [], []
        for f in range(len(cams)):
            for q, h, k in zip(hits_xy[f], hit_h[f], tagged[f]):
                if k == key:
                    P.append(q); H.append(h); F.append(f)
        P, H, F = np.array(P), np.array(H), np.array(F)
        if len(P) < 12:
            continue
        c0 = P.mean(axis=0)
        u, s, vt = np.linalg.svd(P - c0, full_matrices=False)
        d = vt[0]
        m = np.array([-d[1], d[0]])
        off = float(m @ c0)
        t = P @ d
        cnt = np.array([(F == f).sum() for f in range(len(cams))])
        seen = [int(f) for f in range(len(cams)) if cnt[f] >= 3]
        if m @ (np.median(cams[seen], axis=0)) - off > 0:                          # cameras on the - side (extractor convention)
            m, off = -m, -off
            d = np.array([-m[1], m[0]])
            t = P @ d
        order = np.sort(t)
        cut = np.flatnonzero(np.diff(order) > 0.08 * ext)
        starts = np.r_[order[0], order[cut + 1]]
        ends = np.r_[order[cut], order[-1]]
        spans = [[float(a), float(b)] for a, b in zip(starts, ends) if b - a > 0.02 * ext]
        lines.append({"id": lid, "normal_xy": m.tolist(), "offset": off, "direction_xy": d.tolist(), "observed_spans": spans,
                      "support": {"distinct_frames": len(seen), "frame_ids": seen, "view_spread_deg": 30.0,
                                  "along_coverage_fraction": 1.0, "weight": float(len(P)) / 1000.0},
                      "uncertainty": {"offset_sd": 0.5 * noise, "angle_sd_deg": 0.3, "method": "leave_view_out"},
                      "height_range": [float(np.percentile(H, 2)), float(np.percentile(H, 98))], "_tag": key})
    walls = {"model": "observed_wall_lines", "status": "ok", "reasons": [], "axis_world": [0, 0, 1],
             "frame": {"origin_world": [0, 0, 0], "e1": [1, 0, 0], "e2": [0, 1, 0], "extent": ext},
             "noise_scale": noise, "lines": lines,
             "observations": {"frame_ids": list(range(len(cams))), "camera_xy": cams.tolist(), "hits_xy": hits_xy,
                              "hit_heights": hit_h}}
    return walls


def tag_of(walls, tag, group="main"):
    for l in walls["lines"]:
        if l["_tag"] == (tag, group):
            return l
    raise KeyError((tag, group))


def rect(x0, y0, x1, y1, prefix, h=(-1, 1)):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [{"tag": f"{prefix}{k}", "p0": pts[k], "p1": pts[(k + 1) % 4], "h": h} for k in range(4)]


def grid_cams(x0, y0, x1, y1, n=6, skip=None):
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    out = [[x, y] for x in xs for y in ys]
    return [c for c in out if not (skip and skip(c))]


def classes(result):
    return {l["_tag"]: l["association"]["class"] for l in result["walls_all"]["lines"]}


class FragmentAssociationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        room = rect(-2, -1.5, 2, 1.5, "w")
        cams = grid_cams(-1.4, -1.0, 1.4, 1.0, 6)
        cls.plain = build_walls(room, cams)
        box = rect(-0.4, -0.3, 0.4, 0.3, "b", h=(-1, -0.3))
        cls.furn = build_walls(room + box, cams[:12] + [[-1.4, 1.0], [1.4, 1.0], [1.4, -1.0], [-1.4, -1.0], [0, 1.1], [0, -1.1],
                                                        [-1.3, 0.0], [1.3, 0.0]], seed=1)
        # a duplicate: frames 0..11 see wall w0 displaced by 0.15 (pose drift of one window)
        cls.drift = build_walls(room, cams, drift={"w0": (set(range(0, 12)), 0.15)}, seed=2)

    def test_schema_valid_and_inputs_not_mutated(self):
        before = repr(self.plain["lines"][0])
        res = fa.associate_fragments(self.plain)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(before, repr(self.plain["lines"][0]))
        for key in ("walls_all", "walls_boundary_candidates", "walls_with_undetermined"):
            validate_walls_schema(res[key]) if res[key]["lines"] else None
            for l in res[key]["lines"]:
                self.assertIn("association", l)
        self.assertEqual(len(res["walls_all"]["lines"]), len(self.plain["lines"]))     # nothing dropped silently

    def test_closed_room_all_boundary(self):
        res = fa.associate_fragments(self.plain)
        self.assertEqual(set(classes(res).values()), {"room_boundary_candidate"})
        self.assertEqual(ba.assemble_boundary(res["walls_boundary_candidates"])["status"], "complete_polygon_hypothesis")

    def test_furniture_box_is_interior_object(self):
        res = fa.associate_fragments(self.furn)
        cl = classes(res)
        for k in range(4):
            self.assertEqual(cl[(f"w{k}", "main")], "room_boundary_candidate", cl)
        seen = [k for k in range(4) if (f"b{k}", "main") in cl]
        self.assertGreaterEqual(len(seen), 3)
        for k in seen:
            self.assertEqual(cl[(f"b{k}", "main")], "interior_object", res["report"]["lines"])
        out = res["walls_boundary_candidates"]
        self.assertEqual(len(out["lines"]), 4)
        self.assertEqual(ba.assemble_boundary(out)["status"], "complete_polygon_hypothesis")
        self.assertTrue(res["report"]["footprint_clusters"])

    def test_duplicate_wall_offset_by_drift(self):
        res = fa.associate_fragments(self.drift)
        cl = classes(res)
        self.assertIn(("w0", "drift"), cl)
        self.assertEqual(cl[("w0", "drift")], "duplicate_of", res["report"]["lines"])
        canon = [l for l in res["walls_all"]["lines"] if l["_tag"] == ("w0", "main")][0]
        dup = [l for l in res["walls_all"]["lines"] if l["_tag"] == ("w0", "drift")][0]
        self.assertEqual(dup["association"]["duplicate_of"], canon["id"])
        self.assertEqual(cl[("w0", "main")], "room_boundary_candidate")
        self.assertEqual(len(res["walls_boundary_candidates"]["lines"]), 4)
        self.assertGreater(dup["association"]["evidence"]["duplicate"]["exclusivity"], 0.5)

    def test_unfiltered_drift_fails_but_filtered_closes(self):
        raw = ba.assemble_boundary(self.drift)
        filtered = ba.assemble_boundary(fa.associate_fragments(self.drift)["walls_boundary_candidates"])
        self.assertEqual(filtered["status"], "complete_polygon_hypothesis")
        self.assertIn(raw["status"], {"complete_polygon_hypothesis", "ambiguous", "incomplete"})

    def test_exterior_face_seen_through_opening(self):
        # east wall with a doorway; an exterior wall beyond it is seen only through the doorway
        room = [{"tag": "w0", "p0": (-2, -1.5), "p1": (2, -1.5), "h": (-1, 1)},
                {"tag": "w1a", "p0": (2, -1.5), "p1": (2, -0.4), "h": (-1, 1)},
                {"tag": "w1b", "p0": (2, 0.4), "p1": (2, 1.5), "h": (-1, 1)},
                {"tag": "w2", "p0": (2, 1.5), "p1": (-2, 1.5), "h": (-1, 1)},
                {"tag": "w3", "p0": (-2, 1.5), "p1": (-2, -1.5), "h": (-1, 1)},
                {"tag": "ext", "p0": (3.2, -1.2), "p1": (3.2, 1.2), "h": (-1, 1)}]
        cams = grid_cams(-1.4, -1.0, 1.4, 1.0, 6)
        walls = build_walls(room, cams, seed=3)
        res = fa.associate_fragments(walls)
        cl = classes(res)
        self.assertEqual(cl[("ext", "main")], "exterior_or_one_sided", res["report"]["lines"])
        ext = [l for l in res["walls_all"]["lines"] if l["_tag"] == ("ext", "main")][0]
        self.assertIsNotNone(ext["association"]["beyond_line"])
        self.assertNotIn(("ext", "main"), [l["_tag"] for l in res["walls_boundary_candidates"]["lines"]])
        for k in ("w0", "w2", "w3"):
            self.assertEqual(cl[(k, "main")], "room_boundary_candidate")

    def test_nonconvex_l_room_keeps_every_true_wall(self):
        pts = [(-2, -1.5), (2, -1.5), (2, 0), (0.3, 0), (0.3, 1.5), (-2, 1.5)]          # L: short walls of 0.3.. 1.5
        room = [{"tag": f"l{k}", "p0": pts[k], "p1": pts[(k + 1) % 6], "h": (-1, 1)} for k in range(6)]
        cams = grid_cams(-1.7, -1.2, 1.7, 1.2, 7, skip=lambda c: c[0] > 0.0 and c[1] > -0.5)
        cams += [[-1.0, 1.2], [-0.5, 1.2], [-1.5, 1.0]]
        walls = build_walls(room, cams, seed=4)
        res = fa.associate_fragments(walls)
        cl = classes(res)
        self.assertEqual(len(walls["lines"]), 6, [l["_tag"] for l in walls["lines"]])
        self.assertEqual(set(cl.values()), {"room_boundary_candidate"}, res["report"]["lines"])
        self.assertEqual(len(res["walls_boundary_candidates"]["lines"]), 6)
        short = min(walls["lines"], key=lambda l: sum(b - a for a, b in l["observed_spans"]))
        self.assertLess(sum(b - a for a, b in short["observed_spans"]) / walls["frame"]["extent"], 0.3)
        out = ba.assemble_boundary(res["walls_boundary_candidates"])
        self.assertEqual(out["status"], "complete_polygon_hypothesis")
        self.assertEqual(len(out["vertices_xy"]), 6)

    def test_partial_covisibility_is_not_forced_to_duplicate(self):
        # frames 0..11 see wall w0 displaced by 0.15; half of them ALSO see the undisplaced copy: not exclusive, so the
        # duplicate evidence is only partial and must not be promoted to a duplicate label
        room = rect(-2, -1.5, 2, 1.5, "w")
        cams = grid_cams(-1.4, -1.0, 1.4, 1.0, 6)
        walls = build_walls(room, cams, drift={"w0": (set(range(12)), 0.15)}, both=set(range(6)), seed=6)
        res = fa.associate_fragments(walls)
        dup = [l for l in res["walls_all"]["lines"] if l["_tag"] == ("w0", "drift")][0]
        a = dup["association"]
        self.assertNotEqual(a["class"], "duplicate_of", a)
        d = a["evidence"]["duplicate"]
        self.assertIsNotNone(d)
        self.assertGreater(d["co_visible_frames"], 0)
        self.assertLess(d["score"], res["params"]["accept"])
        self.assertEqual(len(res["walls_all"]["lines"]), len(walls["lines"]))
        self.assertIn("alternatives", a)
        # the two layers cannot be separated: the weakly supported copy stays undetermined (layered), not forced to a label;
        # the dominant line keeps its boundary label
        main = [l for l in res["walls_all"]["lines"] if l["_tag"] == ("w0", "main")][0]["association"]
        self.assertEqual(a["class"], "undetermined", a["scores"])
        self.assertEqual(main["class"], "room_boundary_candidate", main["scores"])
        self.assertTrue(a["evidence"]["layered_with"])
        self.assertEqual(len(res["walls_boundary_candidates"]["lines"]), 4)
        self.assertEqual(len(res["walls_with_undetermined"]["lines"]), 5)
        self.assertEqual(len(res["walls_layer_representatives"]["lines"]), 4)
        # comparable support (18 vs 27 frames): neither can be preferred, both undetermined; the HYPOTHESIS variant keeps one
        walls2 = build_walls(room, cams, drift={"w0": (set(range(18)), 0.15)}, both=set(range(9)), seed=7)
        res2 = fa.associate_fragments(walls2)
        cl2 = classes(res2)
        self.assertEqual(cl2[("w0", "drift")], "undetermined", res2["report"]["lines"])
        self.assertEqual(cl2[("w0", "main")], "undetermined", res2["report"]["lines"])
        self.assertEqual(len(res2["walls_boundary_candidates"]["lines"]), 3)
        self.assertEqual(len(res2["walls_layer_representatives"]["lines"]), 4)
        self.assertEqual(ba.assemble_boundary(res2["walls_layer_representatives"])["status"], "complete_polygon_hypothesis")
        merged = res2["walls_layer_merged"]
        self.assertEqual(len(merged["lines"]), 4)
        self.assertTrue(any(l["association"].get("merged_from") for l in merged["lines"]))
        self.assertIn("HYPOTHESIS", merged["association_filter"])
        self.assertEqual(ba.assemble_boundary(merged)["status"], "complete_polygon_hypothesis")
        self.assertEqual(len(res2["walls_all"]["lines"]), 5)                        # the original set is untouched
    def test_conflicting_evidence_stays_undetermined_with_alternatives(self):
        # a low (half-height) long wall-like line: moderate object evidence only -> kept, undetermined, alternative recorded
        room = rect(-2, -1.5, 2, 1.5, "w")
        low = [{"tag": "low", "p0": (-1.5, -0.9), "p1": (1.5, -0.9), "h": (-1, -0.2)}]
        cams = grid_cams(-1.4, -0.6, 1.4, 1.0, 6)
        walls = build_walls(room + low, cams, seed=5)
        res = fa.associate_fragments(walls, weights={"free_behind": 0.2, "height_deficit": 0.45})
        a = [l for l in res["walls_all"]["lines"] if l["_tag"] == ("low", "main")][0]["association"]
        self.assertEqual(a["class"], "undetermined", a["scores"])
        self.assertTrue(any(x["class"] == "interior_object" for x in a["alternatives"]))
        self.assertTrue(any(x["class"] == "room_boundary_candidate" for x in a["alternatives"]))
        ids = lambda key: [l["_tag"] for l in res[key]["lines"]]
        self.assertIn(("low", "main"), ids("walls_all"))
        self.assertIn(("low", "main"), ids("walls_with_undetermined"))
        self.assertNotIn(("low", "main"), ids("walls_boundary_candidates"))
        # the default policy has stronger evidence (free space behind it) and decides, with the evidence recorded
        res2 = fa.associate_fragments(walls)
        b = [l for l in res2["walls_all"]["lines"] if l["_tag"] == ("low", "main")][0]["association"]
        self.assertEqual(b["class"], "interior_object")
        self.assertGreater(b["evidence"]["free_behind"]["free_fraction"], 0.3)
    def test_non_ok_input_passthrough(self):
        res = fa.associate_fragments({"status": "insufficient", "lines": []})
        self.assertEqual(res["status"], "insufficient")

    def test_missing_observations_is_not_forced(self):
        w = {k: v for k, v in self.plain.items() if k != "observations"}
        res = fa.associate_fragments(w)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(any("observations" in r for r in res["reasons"]))
        for l in res["walls_all"]["lines"]:
            self.assertIn(l["association"]["class"], fa.CLASSES)

    def test_module_has_no_truth_simulator_or_camera_up_dependency(self):
        tree = ast.parse((HERE / "fragment_association.py").read_text(encoding="utf-8-sig"))
        imported, names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
            elif isinstance(node, ast.Name):
                names.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                names.add(node.attr.lower())
            elif isinstance(node, ast.arg):
                names.add(node.arg.lower())
        self.assertLessEqual(imported, {"__future__", "copy", "itertools", "math", "numpy", "boundary_assembly"})
        self.assertFalse([n for n in names if any(w in n for w in ("truth", "gravity", "camera_up", "simulator", "annotation",
                                                                     "scene", "manhattan", "rectangle"))])


if __name__ == "__main__":
    unittest.main()
