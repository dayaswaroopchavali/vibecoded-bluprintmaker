import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import house_layouts as hl
import layout_mesh as lm

L_SHAPE = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 5], [0, 5]]
U_SHAPE = [[0, 0], [6, 0], [6, 4], [4, 4], [4, 1.5], [2, 1.5], [2, 4], [0, 4]]
# deeper non-star polygon: no single interior point sees every edge (two deep, offset pockets)
COMB = [[0, 0], [9, 0], [9, 6], [8, 6], [8, 1], [7, 1], [7, 6], [6, 6], [6, 1], [5, 1], [5, 6],
        [4, 6], [4, 1], [3, 1], [3, 6], [0, 6]]
SPIRAL = [[0, 0], [6, 0], [6, 6], [1, 6], [1, 2], [4, 2], [4, 4], [3, 4], [3, 3], [2, 3], [2, 5], [5, 5],
          [5, 1], [0, 1]]


def _room(poly, h=2.6, z=0.0, rid="r"):
    return hl._room(rid, poly, z, h)


def suite(seed=20261003):
    return hl.generate_suite(seed)["layouts"]


class TriangulationTests(unittest.TestCase):
    def check(self, poly):
        tris = lm.triangulate_polygon(poly)
        self.assertEqual(len(tris), len(poly) - 2)
        self.assertAlmostEqual(lm.triangulated_area(poly, tris), hl.polygon_area(poly), places=9)
        for a, b, c in tris:
            self.assertGreater(lm._cross(poly[a], poly[b], poly[c]), 1e-9)  # CCW and non-degenerate
        return tris

    def test_convex_l_u_and_deep_nonstar_polygons(self):
        for poly in ([[0, 0], [3, 0], [3, 2], [0, 2]], L_SHAPE, U_SHAPE, COMB, SPIRAL):
            self.check(poly)
            self.check(poly[::-1])  # clockwise input

    def test_triangles_stay_inside_concave_polygon(self):
        for poly in (L_SHAPE, U_SHAPE, COMB, SPIRAL):
            for a, b, c in lm.triangulate_polygon(poly):
                cen = tuple(np.mean([poly[a], poly[b], poly[c]], axis=0))
                self.assertTrue(hl._point_in_poly(cen, [tuple(p) for p in poly]))

    def test_triangulation_detects_wrong_area(self):
        tris = lm.triangulate_polygon(L_SHAPE)
        self.assertNotAlmostEqual(lm.triangulated_area(L_SHAPE, tris[:-1]), hl.polygon_area(L_SHAPE), places=3)

    def test_collinear_vertices_do_not_create_degenerate_triangles(self):
        poly = [[0, 0], [2, 0], [4, 0], [4, 3], [0, 3]]
        tris = lm.triangulate_polygon(poly)
        for a, b, c in tris:
            self.assertGreater(abs(lm._cross(poly[a], poly[b], poly[c])), 1e-9)
        self.assertAlmostEqual(lm.triangulated_area(poly, tris), 12.0)

    def test_degenerate_polygon_rejected(self):
        with self.assertRaises(ValueError):
            lm.triangulate_polygon([[0, 0], [1, 0], [2, 0]])


class RoomMeshTests(unittest.TestCase):
    def test_floor_ceiling_area_normals_and_elevation(self):
        for poly in (L_SHAPE, U_SHAPE, COMB, SPIRAL):
            room = _room(poly, 2.7, 0.35)
            mesh = lm.build_room_mesh(room)
            lm.validate_room_mesh(room, mesh)
            floor, ceil = mesh.select(kind="floor"), mesh.select(kind="ceiling")
            self.assertTrue(np.allclose(floor.vertices[floor.faces][:, :, 2], 0.35))
            self.assertTrue(np.allclose(ceil.vertices[ceil.faces][:, :, 2], 3.05))
            self.assertTrue(np.all(floor.normals()[:, 2] > .999))
            self.assertTrue(np.all(ceil.normals()[:, 2] < -.999))

    def test_walls_face_interior_for_either_winding(self):
        for poly in (L_SHAPE, L_SHAPE[::-1], U_SHAPE):
            room = _room(poly)
            mesh = lm.build_room_mesh(room).select(kind="wall")
            cen = mesh.vertices[mesh.faces].mean(axis=1)
            probe = cen + mesh.normals() * 1e-3
            self.assertTrue(all(hl._point_in_poly((p[0], p[1]), [tuple(q) for q in poly]) for p in probe))

    def test_closed_room_without_openings_is_watertight(self):
        room = _room(U_SHAPE)
        boundary, max_use = lm.mesh_boundary_edges(lm.build_room_mesh(room))
        self.assertEqual(boundary, [])
        self.assertEqual(max_use, 2)


def two_room_layout(door_h=2.1, extra=False):
    a = _room([[0, 0], [4, 0], [4, 3], [0, 3]], 2.6, 0, "a")
    b = _room([[4, 0], [7, 0], [7, 3], [4, 3]], 2.6, 0, "b")
    sa, sb = a["walls"][1]["surface_id"], b["walls"][3]["surface_id"]
    doors = [{"type": "doorway", "room_a": "a", "room_b": "b", "surface_a_id": sa, "surface_b_id": sb,
              "endpoints_xy_m": [[4, 1.0], [4, 2.0]], "width_m": 1.0, "height_m": door_h}]
    if extra:
        doors[0]["endpoints_xy_m"] = [[4, 0.3], [4, 0.9]]
        doors.append({**doors[0], "endpoints_xy_m": [[4, 1.6], [4, 2.5]], "width_m": 0.9, "height_m": 1.9})
    return {"stage": "multiroom", "house_id": "t", "rooms": [a, b], "doorways": doors}


class OpeningTests(unittest.TestCase):
    def test_wall_area_and_boundary_around_real_opening(self):
        layout = two_room_layout()
        mesh = lm.build_layout_mesh(layout)["a"]
        sid = layout["rooms"][0]["walls"][1]["surface_id"]
        wall = mesh.select(kind="wall", surface_id=sid)
        self.assertAlmostEqual(wall.triangle_areas().sum(), 3 * 2.6 - 1.0 * 2.1, places=9)
        self.assertGreater(wall.triangle_areas().min(), 1e-9)
        boundary, max_use = lm.mesh_boundary_edges(wall)
        self.assertEqual(max_use, 2)
        perim = sum(np.linalg.norm(wall.vertices[a] - wall.vertices[b]) for a, b in boundary)
        self.assertAlmostEqual(perim, 2 * (3 + 2.6) + 2 * 2.1, places=9)  # rectangle + two jambs net of the sill gap
        deg = {}
        for a, b in boundary:
            deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
        self.assertTrue(all(v == 2 for v in deg.values()))  # closed outline: no cracks / dangling edges

    def test_ray_passes_through_opening_but_hits_beside_it(self):
        layout = two_room_layout()
        mesh = lm.combine(lm.build_layout_mesh(layout).values())
        wall_only = mesh.select(kind="wall")
        through = lm.ray_hits(wall_only, [3.0, 1.5, 1.0], [1, 0, 0], tmax=2.0)
        beside = lm.ray_hits(wall_only, [3.0, 0.5, 1.0], [1, 0, 0], tmax=2.0)
        above = lm.ray_hits(wall_only, [3.0, 1.5, 2.3], [1, 0, 0], tmax=2.0)
        self.assertEqual(through, [])
        self.assertEqual(len(beside), 2)  # wall of room a and wall of room b (both are real surfaces)
        self.assertEqual(len(above), 2)
        # just inside the jamb passes, just outside hits
        self.assertEqual(lm.ray_hits(wall_only, [3, 1.02, .1], [1, 0, 0], 2.0), [])
        self.assertNotEqual(lm.ray_hits(wall_only, [3, .98, .1], [1, 0, 0], 2.0), [])
        # headroom: head at 2.1
        self.assertEqual(lm.ray_hits(wall_only, [3, 1.5, 2.09], [1, 0, 0], 2.0), [])
        self.assertNotEqual(lm.ray_hits(wall_only, [3, 1.5, 2.11], [1, 0, 0], 2.0), [])

    def test_opening_removed_would_be_detected(self):
        layout = two_room_layout()
        layout["doorways"] = []
        mesh = lm.combine(lm.build_layout_mesh(layout).values()).select(kind="wall")
        self.assertNotEqual(lm.ray_hits(mesh, [3.0, 1.5, 1.0], [1, 0, 0], tmax=2.0), [])

    def test_two_openings_one_wall_and_validation(self):
        layout = two_room_layout(extra=True)
        meshes = lm.build_layout_mesh(layout)
        openings = lm.wall_openings(layout)
        for room in layout["rooms"]:
            lm.validate_room_mesh(room, meshes[room["room_id"]], openings)
        wall = meshes["a"].select(kind="wall", surface_id=layout["rooms"][0]["walls"][1]["surface_id"])
        self.assertAlmostEqual(wall.triangle_areas().sum(), 3 * 2.6 - .6 * 2.1 - .9 * 1.9, places=9)
        boundary, max_use = lm.mesh_boundary_edges(wall)
        self.assertEqual(max_use, 2)
        wo = wall
        self.assertEqual(lm.ray_hits(wo, [3, 0.6, 1.0], [1, 0, 0], 2.0), [])
        self.assertEqual(lm.ray_hits(wo, [3, 2.0, 1.0], [1, 0, 0], 2.0), [])
        self.assertNotEqual(lm.ray_hits(wo, [3, 1.25, 1.0], [1, 0, 0], 2.0), [])

    def test_validate_catches_missing_wall_area(self):
        layout = two_room_layout()
        meshes = lm.build_layout_mesh(layout)
        broken = lm.Mesh(meshes["a"].vertices, meshes["a"].faces[:-1], meshes["a"].labels[:-1])
        with self.assertRaises(ValueError):
            lm.validate_room_mesh(layout["rooms"][0], broken, lm.wall_openings(layout))


class SuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layouts = suite()

    def test_all_single_and_multiroom_layouts_build_and_validate(self):
        count = 0
        for layout in self.layouts:
            if layout["stage"] == "multifloor":
                continue
            summary = lm.summarize_layout(layout)
            self.assertGreater(summary["triangles"], 0)
            count += 1
        self.assertEqual(count, 20)

    def test_every_layout_doorway_has_a_real_clear_opening_and_solid_neighbours(self):
        doors_checked = 0
        for layout in self.layouts:
            if layout["stage"] != "multiroom":
                continue
            meshes = lm.build_layout_mesh(layout)
            walls = lm.combine(meshes.values()).select(kind="wall")
            rooms = {r["room_id"]: r for r in layout["rooms"]}
            for door in layout["doorways"]:
                p, q = np.array(door["endpoints_xy_m"], float)
                mid, width = (p + q) / 2, np.linalg.norm(q - p)
                axis = (q - p) / width
                normal = np.array([-axis[1], axis[0]])
                z = min(1.0, door["height_m"] / 2)
                o = np.array([*(mid - normal * .5), z])
                direction = np.array([*normal, 0.])
                self.assertEqual(lm.ray_hits(walls, o, direction, tmax=1.0), [], door)
                above = o.copy(); above[2] = door["height_m"] + .04
                self.assertGreaterEqual(len(lm.ray_hits(walls, above, direction, tmax=1.0)), 1)
                # beside: step along the wall past the jamb but inside the shared wall segment
                seg = hl._shared_segment(rooms[door["room_a"]]["polygon_xy_m"], rooms[door["room_b"]]["polygon_xy_m"])
                lo, hi = sorted((np.dot(np.array(seg[0]), axis), np.dot(np.array(seg[1]), axis)))
                free = min(np.dot(p, axis) - lo, hi - np.dot(q, axis))  # q = p + width*axis
                self.assertGreater(free, .05)
                side = o.copy(); side[:2] += axis * (width / 2 + free / 2)
                self.assertGreaterEqual(len(lm.ray_hits(walls, side, direction, tmax=1.0)), 1)
                doors_checked += 1
        self.assertEqual(doors_checked, 30)

    def test_multifloor_is_never_claimed_capture_ready(self):
        floors = [l for l in self.layouts if l["stage"] == "multifloor"]
        self.assertEqual(len(floors), 10)
        for layout in floors:
            s = lm.summarize_layout(layout)
            self.assertFalse(s["capture_ready"])
            self.assertIn("unvalidated", s["multifloor_capture_note"])
        self.assertIn("NOT capture-ready", lm.__doc__ + lm.MULTIFLOOR_CAPTURE_NOTE)

    def test_write_outputs_is_versioned_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "layout-mesh-v1"
            doc = lm.write_outputs(self.layouts[:2] + self.layouts[20:21], out)
            data = json.loads((out / "summary.json").read_text())
            self.assertEqual(data["mesh_version"], "layout-mesh-v1")
            self.assertFalse(data["multifloor_capture_ready"])
            self.assertEqual(len(doc["layouts"]), 3)
            self.assertTrue((out / f"{self.layouts[0]['house_id']}.obj").read_text().startswith("#"))
            with self.assertRaises(FileExistsError):
                lm.write_outputs(self.layouts[:1], out)


if __name__ == "__main__":
    unittest.main()

