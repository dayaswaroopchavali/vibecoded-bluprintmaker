"""Fast, non-rendering tests of the REALISTIC appearance: seed-only parameters, original/textured unchanged, decor placement."""
import copy
import hashlib
import inspect
import json
import unittest

import blender_capture as bc
import simulator

# sha256 of json.dumps(texture_params(seed, make_scene(17)), sort_keys=True), recorded BEFORE the realistic mode was added.
TEXTURED_BASELINE = {1: "0bd2583b0a75d7120a48c358da7fb0dce1f1da6a2e4c0a40a53e6948b2e6cf10",
                     5: "ea3fa2082882380840aed7dad0de0cd58c72e19ae4ae1dd28b935836c3e57744"}
ORIGINAL_SOURCE_LINES = (
    'wall_mats=[material(f"wall {i+1}",c,5.5) for i,c in enumerate(scene_data["wall_colors"])]',
    'floor_mat=material("subtle floor grain",scene_data["floor_color"],3.0)',
    'ceil_mat=material("ceiling",scene_data["ceiling_color"],7.0)',
    'tex.inputs["Detail"].default_value=3; tex.inputs["Roughness"].default_value=.72',
    'ramp.color_ramp.elements[0].color=tuple(float(c)*.58 for c in color)+(1,)',
    'bump.inputs["Strength"].default_value=.11; bump.inputs["Distance"].default_value=.035',
)


def _layout(texture_seed, scene):
    p = bc.realistic_params(texture_seed, scene)
    return p, bc.realistic_layout(p, scene)


def _opening_rect(o):
    return (o["center"] - o["width"] / 2, o["center"] + o["width"] / 2, o["z_min"], o["z_max"])


def _wall_uz(scene, wi, aabb):
    f = bc._wall_frame(scene, wi)
    k = 0 if f["axis"] == "y" else 1
    return aabb[0][k], aabb[1][k], aabb[0][2], aabb[1][2]


class RealisticAppearanceTests(unittest.TestCase):
    def test_modes_listed_and_original_textured_unchanged(self):
        self.assertEqual(bc.APPEARANCES[:2], ("original", "textured"))
        self.assertIn("realistic", bc.APPEARANCES)
        scene = simulator.make_scene(17)
        for seed, digest in TEXTURED_BASELINE.items():
            got = hashlib.sha256(json.dumps(bc.texture_params(seed, scene), sort_keys=True).encode()).hexdigest()
            self.assertEqual(got, digest, "textured material params changed")
        src = inspect.getsource(bc._worker).replace(" ", "")
        for line in ORIGINAL_SOURCE_LINES:  # the original-appearance material construction must be untouched
            self.assertIn(line.replace(" ", ""), src)
        self.assertIn('if appearance=="textured":'.replace(" ", ""), src)

    def test_realistic_params_deterministic_and_seed_sensitive(self):
        scene = simulator.make_scene(17)
        a = bc.realistic_params(5, scene)
        self.assertEqual(a, bc.realistic_params(5, scene))
        self.assertNotEqual(a["walls"], bc.realistic_params(6, scene)["walls"])
        self.assertNotEqual(a["decor_slots"], bc.realistic_params(6, scene)["decor_slots"])
        json.dumps(a)  # fully serialisable

    def test_realistic_params_depend_only_on_texture_seed(self):
        scene = simulator.make_scene(17); a = bc.realistic_params(5, scene)
        moved = json.loads(json.dumps(scene))
        moved["seed"] = 999; moved["width"] *= 1.7; moved["length"] *= .6; moved["height"] += 1
        for f in moved["furniture"]: f["center"] = [0., 0.]; f["size"] = [1., 1., 1.]
        moved["openings"] = [dict(o, center=o["center"] + .3, width=o["width"] * .5) for o in moved["openings"]] + [dict(moved["openings"][0], wall=2)]
        moved["wall_colors"] = [[.1, .2, .3]] * 4; moved["ceiling_color"] = [.1, .1, .1]; moved["floor_color"] = [.2, .2, .2]
        self.assertEqual(a, bc.realistic_params(5, moved))   # no scene dimension/opening/position/seed/colour enters the draw
        other = simulator.make_scene(3)
        self.assertEqual(a["walls"], bc.realistic_params(5, other)["walls"])
        self.assertEqual(a["decor_slots"], bc.realistic_params(5, other)["decor_slots"])
        bc.camera_poses(scene, 8, 1); bc.camera_poses(scene, 8, 2)  # capture seed never reaches it
        self.assertEqual(a, bc.realistic_params(5, scene))

    def test_layout_deterministic_and_scene_not_mutated(self):
        scene = simulator.make_scene(17); before = copy.deepcopy(scene)
        p, l1 = _layout(4, scene)
        self.assertEqual(l1, bc.realistic_layout(p, scene)); self.assertEqual(scene, before)

    def test_decor_does_not_overlap_openings(self):
        n = 0
        for house in range(1, 41):
            scene = simulator.make_scene(house)
            for ts in (1, 2, 3):
                _, lay = _layout(ts, scene)
                for it in lay["items"]:
                    if it["kind"] in ("frame", "poster", "shelf", "clock", "switch", "socket"):
                        n += 1; ur = _wall_uz(scene, it["wall"], it["aabb"])
                        for o in (o for o in scene["openings"] if o["wall"] == it["wall"]):
                            r = _opening_rect(o)
                            self.assertFalse(ur[0] < r[1] and r[0] < ur[1] and ur[2] < r[3] and r[2] < ur[3], (house, ts, it["id"], it["kind"]))
        self.assertGreater(n, 500)

    def test_decor_within_wall_bounds_and_shell_unchanged(self):
        for house in range(1, 41):
            scene = simulator.make_scene(house); w, l, h = scene["width"], scene["length"], scene["height"]
            _, lay = _layout(2, scene); sh = lay["room_shell"]
            self.assertEqual((sh["width"], sh["length"], sh["height"], sh["floor_plane_z"], sh["ceiling_plane_z"]), (w, l, h, 0.0, h))
            self.assertEqual([p["coordinate"] for p in sh["wall_planes"]], [l / 2, w / 2, -l / 2, -w / 2])
            self.assertTrue(sh["unchanged_by_decor"])
            for it in lay["items"]:
                for p in it["parts"]:
                    lo, hi = p["aabb"]
                    # everything stays inside the room volume: nothing crosses z=0 / z=H or any wall plane
                    self.assertGreaterEqual(lo[2], 0.0, (house, it["kind"], p["name"])); self.assertLessEqual(hi[2], h + 1e-9, (house, it["kind"], p["name"]))
                    self.assertGreaterEqual(lo[0], -w / 2 - 1e-9); self.assertLessEqual(hi[0], w / 2 + 1e-9)
                    self.assertGreaterEqual(lo[1], -l / 2 - 1e-9); self.assertLessEqual(hi[1], l / 2 + 1e-9)
                    if it["wall"] is not None:
                        f = bc._wall_frame(scene, it["wall"]); k = 1 if f["axis"] == "y" else 0
                        depth = max(abs(lo[k] - f["plane"]), abs(hi[k] - f["plane"]))
                        self.assertLessEqual(depth, .30 + 1e-9)  # thin objects on the wall; no part reaches behind the plane (checked by room bounds above)
                if it["kind"] in ("frame", "poster", "shelf", "clock", "switch", "socket"):
                    u0, u1, z0, z1 = _wall_uz(scene, it["wall"], it["aabb"]); half = bc._wall_frame(scene, it["wall"])["span"] / 2
                    self.assertGreaterEqual(u0, -half + .25 - 1e-9); self.assertLessEqual(u1, half - .25 + 1e-9)
                    self.assertGreater(z0, 0.1); self.assertLess(z1, h)

    def test_decor_items_do_not_overlap_each_other_on_a_wall(self):
        for house in (17, 5, 9):
            scene = simulator.make_scene(house); _, lay = _layout(7, scene)
            by = {}
            for it in lay["items"]:
                if it["kind"] in ("frame", "poster", "shelf", "clock", "switch", "socket"): by.setdefault(it["wall"], []).append(_wall_uz(scene, it["wall"], it["aabb"]))
            for rects in by.values():
                for i, a in enumerate(rects):
                    for b in rects[i + 1:]:
                        self.assertFalse(a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3])

    def test_manifest_schema(self):
        scene = simulator.make_scene(17); p, lay = _layout(3, scene); m = bc.decor_manifest(p, lay)
        self.assertEqual(m["schema"], "realistic_decor_v1"); self.assertEqual(m["texture_seed"], 3)
        self.assertEqual(set(m), {"schema", "texture_seed", "room_shell", "counts", "items", "mask_note"})
        ids = [i["id"] for i in m["items"]]; self.assertEqual(len(ids), len(set(ids)))
        for it in m["items"]:
            self.assertTrue({"id", "kind", "wall", "aabb_min", "aabb_max", "parts"} <= set(it))
            self.assertEqual(len(it["aabb_min"]), 3); self.assertTrue(all(a <= b for a, b in zip(it["aabb_min"], it["aabb_max"])))
            self.assertTrue(it["parts"] and all(set(q) == {"name", "shape", "aabb_min", "aabb_max"} for q in it["parts"]))
        for kind in ("skirting", "cornice", "ceiling_light", "rug"): self.assertIn(kind, m["counts"])
        self.assertEqual(sum(m["counts"].values()), len(m["items"]))
        json.dumps(m)
        # decor manifest is private: it must not be part of the seed-only parameter dict that goes to the public appearance manifest
        self.assertNotIn("items", p); self.assertNotIn("layout", p)

    def test_trim_and_rug_rules(self):
        scene = simulator.make_scene(17); p, lay = _layout(3, scene)
        door = [o for o in scene["openings"] if o["z_min"] < .05][0]
        for it in lay["items"]:
            if it["kind"] == "skirting" and it["wall"] == door["wall"]:
                for q in it["parts"]:  # skirting is broken across the doorway
                    u0, u1, _, _ = _wall_uz(scene, it["wall"], q["aabb"])
                    self.assertFalse(u0 < door["center"] + door["width"] / 2 and door["center"] - door["width"] / 2 < u1)
        rug = [i for i in lay["items"] if i["kind"] == "rug"][0]
        self.assertLessEqual(rug["aabb"][1][2], .02); self.assertEqual(rug["aabb"][0][2], 0.0)


if __name__ == "__main__":
    unittest.main()
