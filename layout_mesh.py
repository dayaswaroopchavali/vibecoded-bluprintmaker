"""Generic triangle meshes from the PRIVATE benchmark house layouts (house_layouts.py).

Evaluation / simulation side only. Never reconstruction input: the layouts are private ground truth.

Per room: a triangulated floor (normal +z) and ceiling (normal -z) from an ear-clipped, possibly concave
polygon, plus one wall panel per polygon edge extruded from the floor to the ceiling. Doorways are REAL holes:
the wall panel is built from solid columns plus a lintel above each opening, so rays pass through the gap
(nothing is painted). Both rooms sharing a doorway get the matching opening in their own wall.

Normals face the room interior. Meshes are per room (a "room mesh"), and a layout mesh is the room dict.

CAPTURE READINESS: single_room and multiroom meshes are geometry only; this module does NOT claim any
benchmark stage complete. Multifloor stair links in house_layouts are physically unvalidated (no treads,
run/rise or headroom modelling; stair openings are NOT cut into floors/ceilings), so multifloor meshes are
emitted for inspection with ``capture_ready: false`` and must not be used for capture.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import house_layouts as hl

MESH_VERSION = "layout-mesh-v1"
MIN_TRIANGLE_AREA = 1e-9
MULTIFLOOR_CAPTURE_NOTE = ("multifloor stair links are physically unvalidated (no treads, run/rise, headroom, "
                           "and no stairwell opening in floor/ceiling meshes); NOT capture-ready")


# ----------------------------------------------------------------------------- triangulation
def _signed_area2(poly):
    return sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
               for i in range(len(poly)))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _in_triangle(p, a, b, c, eps):
    return _cross(a, b, p) >= -eps and _cross(b, c, p) >= -eps and _cross(c, a, p) >= -eps


def triangulate_polygon(points):
    """Ear-clip a simple (possibly concave) polygon given as 2D points.

    Returns a list of (i, j, k) index triples into ``points``, all counter-clockwise, covering the polygon
    exactly. Vertices may be given in either winding. Collinear vertices are kept (no T-junctions) whenever a
    proper ear is available; only if none is, a zero-area collinear vertex is dropped.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 3:
        raise ValueError("polygon needs at least three vertices")
    area2 = _signed_area2(pts)
    if abs(area2) <= 1e-12:
        raise ValueError("polygon has zero area")
    order = list(range(n)) if area2 > 0 else list(range(n - 1, -1, -1))
    scale = max(max(abs(c) for p in pts for c in p), 1.0)
    eps = 1e-12 * scale * scale
    tris = []
    guard = 0
    while len(order) > 3:
        guard += 1
        if guard > 10 * n * n:
            raise ValueError("ear clipping failed (polygon not simple?)")
        m = len(order)
        clipped = False
        for pass_ in (0, 1):
            for k in range(m):
                ia, ib, ic = order[k - 1], order[k], order[(k + 1) % m]
                a, b, c = pts[ia], pts[ib], pts[ic]
                cr = _cross(a, b, c)
                if pass_ == 0:
                    if cr <= eps:
                        continue
                    if any(_in_triangle(pts[o], a, b, c, eps) for o in order
                           if o not in (ia, ib, ic) and pts[o] not in (a, b, c)):
                        continue
                    tris.append((ia, ib, ic))
                else:
                    if abs(cr) > eps:
                        continue
                order.pop(k)  # collinear fallback drops the vertex without emitting a degenerate triangle
                clipped = True
                break
            if clipped:
                break
        if not clipped:
            raise ValueError("ear clipping failed (polygon not simple?)")
    ia, ib, ic = order
    if _cross(pts[ia], pts[ib], pts[ic]) > eps:
        tris.append((ia, ib, ic))
    return tris


def triangulated_area(points, tris):
    return sum(abs(_cross(points[a], points[b], points[c])) / 2 for a, b, c in tris)


# ----------------------------------------------------------------------------- mesh container
@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray  # Fx3 int
    labels: list = field(default_factory=list)  # per face: {"room_id", "kind", "surface_id"}

    def triangle_areas(self):
        v = self.vertices[self.faces]
        return np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) / 2

    def normals(self):
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    def select(self, kind=None, surface_id=None):
        keep = [i for i, l in enumerate(self.labels)
                if (kind is None or l["kind"] == kind) and (surface_id is None or l["surface_id"] == surface_id)]
        return Mesh(self.vertices, self.faces[keep], [self.labels[i] for i in keep])


class _Builder:
    def __init__(self, room_id):
        self.room_id = room_id
        self.verts, self.index, self.faces, self.labels = [], {}, [], []

    def vertex(self, p):
        key = tuple(round(float(c), 9) for c in p)  # weld shared corners so edges are shared
        if key not in self.index:
            self.index[key] = len(self.verts)
            self.verts.append(key)
        return self.index[key]

    def add(self, tri3d, kind, surface_id):
        ids = [self.vertex(p) for p in tri3d]
        self.faces.append(ids)
        self.labels.append({"room_id": self.room_id, "kind": kind, "surface_id": surface_id})

    def mesh(self):
        return Mesh(np.array(self.verts, dtype=float), np.array(self.faces, dtype=int).reshape(-1, 3), self.labels)


# ----------------------------------------------------------------------------- openings
def wall_openings(layout):
    """Map wall surface_id -> list of (s0, s1, height_m) along the wall edge, from the layout doorways."""
    surfaces = {w["surface_id"]: w for r in layout["rooms"] for w in r["walls"]}
    result = {}
    for door in layout.get("doorways", []):
        if door.get("type") == "stairway":
            continue  # stairs are not cut into walls (and are physically unvalidated)
        for key in ("surface_a_id", "surface_b_id"):
            wall = surfaces[door[key]]
            a, b = np.array(wall["start_xy_m"], float), np.array(wall["end_xy_m"], float)
            length = float(np.linalg.norm(b - a))
            d = (b - a) / length
            s = sorted(float((np.array(p, float) - a) @ d) for p in door["endpoints_xy_m"])
            result.setdefault(door[key], []).append((max(s[0], 0.0), min(s[1], length), float(door["height_m"])))
    for sid, ops in result.items():
        ops.sort()
        for (a0, a1, _), (b0, b1, _) in zip(ops, ops[1:]):
            if b0 < a1 - 1e-9:
                raise ValueError(f"overlapping openings on {sid}")
    return result


def _wall_pieces(length, height, openings):
    """Polygons (s, z) tiling a wall of given size minus openings: solid columns plus lintels.

    Columns carry an extra vertex at each jamb at the opening head height, so the lintel and the column share
    that vertex (no T-junction -> the wall mesh is watertight around every opening).
    """
    ops = [(s0, s1, min(h, height)) for s0, s1, h in openings if s1 - s0 > 1e-9]
    pieces = []
    cursor, prev_h = 0.0, None  # prev_h: head height of the opening whose right jamb is at cursor
    for s0, s1, h in ops + [(length, length, height)]:
        if s0 - cursor > 1e-9:
            poly = [(cursor, 0.0), (s0, 0.0)]
            if h < height - 1e-9:
                poly.append((s0, h))
            poly += [(s0, height), (cursor, height)]
            if prev_h is not None and prev_h < height - 1e-9:
                poly.append((cursor, prev_h))
            pieces.append(poly)
        if s1 > s0 and h < height - 1e-9:
            pieces.append([(s0, h), (s1, h), (s1, height), (s0, height)])
        cursor, prev_h = s1, h
    return pieces

# ----------------------------------------------------------------------------- room / layout meshes
def build_room_mesh(room, openings_by_surface=None):
    openings_by_surface = openings_by_surface or {}
    poly = hl.validate_polygon(room["polygon_xy_m"])
    z0, height = float(room["floor_elevation_m"]), float(room["ceiling_height_m"])
    b = _Builder(room["room_id"])
    tris = triangulate_polygon(poly)
    for i, j, k in tris:  # triangulate_polygon is CCW: normal +z
        b.add([(*poly[i], z0), (*poly[j], z0), (*poly[k], z0)], "floor", f"{room['room_id']}-floor")
        b.add([(*poly[i], z0 + height), (*poly[k], z0 + height), (*poly[j], z0 + height)], "ceiling",
              f"{room['room_id']}-ceiling")
    ccw = _signed_area2(poly) > 0
    for wall in room["walls"]:
        a = np.array(wall["start_xy_m"], float)
        e = np.array(wall["end_xy_m"], float)
        length = float(np.linalg.norm(e - a))
        d = (e - a) / length
        for piece in _wall_pieces(length, height, openings_by_surface.get(wall["surface_id"], [])):
            for i, j, k in triangulate_polygon(piece):  # CCW in (s, z): normal points to the right of the edge
                tri = [(*(a + piece[idx][0] * d), z0 + piece[idx][1]) for idx in (i, j, k)]
                # right of a CCW edge is outside; flip so normals face the room interior
                b.add(tri[::-1] if ccw else tri, "wall", wall["surface_id"])
    return b.mesh()


def build_layout_mesh(layout):
    """Return {room_id: Mesh} for a layout dict from house_layouts."""
    openings = wall_openings(layout)
    return {room["room_id"]: build_room_mesh(room, openings) for room in layout["rooms"]}


def combine(meshes):
    verts, faces, labels, off = [], [], [], 0
    for m in meshes:
        verts.append(m.vertices); faces.append(m.faces + off); labels += m.labels; off += len(m.vertices)
    return Mesh(np.vstack(verts), np.vstack(faces), labels)


# ----------------------------------------------------------------------------- queries / validation
def ray_hits(mesh, origin, direction, tmax=math.inf):
    """Moller-Trumbore: sorted distances t in (1e-9, tmax] at which the ray hits mesh triangles (two-sided)."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    v = mesh.vertices[mesh.faces]
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    t_vec = o - v[:, 0]
    u = np.einsum("ij,ij->i", t_vec, p) * inv
    q = np.cross(t_vec, e1)
    w = np.einsum("j,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-9) & (t <= tmax)
    return np.sort(t[hit]).tolist()


def mesh_boundary_edges(mesh):
    """Undirected edges (welded vertex ids) used by exactly one triangle, plus max edge multiplicity."""
    counts = {}
    for f in mesh.faces:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    return [e for e, c in counts.items() if c == 1], (max(counts.values()) if counts else 0)


def validate_room_mesh(room, mesh, openings_by_surface=None):
    """Raise ValueError unless the mesh is consistent with the room; return numeric diagnostics."""
    openings_by_surface = openings_by_surface or {}
    areas = mesh.triangle_areas()
    if areas.size == 0 or areas.min() <= MIN_TRIANGLE_AREA:
        raise ValueError(f"{room['room_id']}: degenerate triangle")
    poly_area = hl.polygon_area(room["polygon_xy_m"])
    h = float(room["ceiling_height_m"])
    for kind in ("floor", "ceiling"):
        got = float(mesh.select(kind=kind).triangle_areas().sum())
        if abs(got - poly_area) > 1e-7 * max(1, poly_area):
            raise ValueError(f"{room['room_id']}: {kind} area {got} != polygon area {poly_area}")
    for wall in room["walls"]:
        length = math.dist(wall["start_xy_m"], wall["end_xy_m"])
        holes = sum((s1 - s0) * min(oh, h) for s0, s1, oh in openings_by_surface.get(wall["surface_id"], []))
        got = float(mesh.select(kind="wall", surface_id=wall["surface_id"]).triangle_areas().sum())
        if abs(got - (length * h - holes)) > 1e-7 * max(1, length * h):
            raise ValueError(f"{wall['surface_id']}: wall area {got} != {length * h - holes}")
    _, max_use = mesh_boundary_edges(mesh)
    if max_use > 2:
        raise ValueError(f"{room['room_id']}: non-manifold edge")
    return {"triangles": int(len(mesh.faces)), "min_triangle_area_m2": float(areas.min()), "floor_area_m2": poly_area}


# ----------------------------------------------------------------------------- summaries / output
def summarize_layout(layout):
    meshes = build_layout_mesh(layout)
    openings = wall_openings(layout)
    rooms = []
    for room in layout["rooms"]:
        diag = validate_room_mesh(room, meshes[room["room_id"]], openings)
        rooms.append({"room_id": room["room_id"], "vertices": int(len(meshes[room["room_id"]].vertices)), **diag,
                      "openings": sum(len(openings.get(w["surface_id"], [])) for w in room["walls"])})
    out = {"house_id": layout["house_id"], "stage": layout["stage"], "rooms": rooms,
           "triangles": sum(r["triangles"] for r in rooms),
           "doorway_openings_cut": sum(r["openings"] for r in rooms),
           "capture_ready": False, "benchmark_stage_complete": False}
    if layout["stage"] == "multifloor":
        out["multifloor_capture_note"] = MULTIFLOOR_CAPTURE_NOTE
    else:
        out["capture_ready_note"] = "geometry only; no capture or renderer validation performed here"
    return out


def to_obj(meshes):
    lines, off = [f"# {MESH_VERSION} private benchmark geometry (evaluation only)"], 0
    for rid, m in meshes.items():
        lines.append(f"o {rid}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.vertices]
        lines += [f"f {a + off + 1} {b + off + 1} {c + off + 1}" for a, b, c in m.faces]
        off += len(m.vertices)
    return "\n".join(lines) + "\n"


def write_outputs(layouts, output_dir):
    """Write summary.json and one OBJ per layout into a NEW directory (never overwrites)."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    summaries = []
    for layout in layouts:
        summaries.append(summarize_layout(layout))
        (output_dir / f"{layout['house_id']}.obj").write_text(to_obj(build_layout_mesh(layout)), encoding="utf-8")
    doc = {"schema_version": 1, "mesh_version": MESH_VERSION,
           "data_classification": "private_benchmark_ground_truth_geometry", "reconstruction_input": False,
           "multifloor_capture_ready": False, "multifloor_capture_note": MULTIFLOOR_CAPTURE_NOTE,
           "layouts": summaries}
    (output_dir / "summary.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, default=Path("../../outputs/sim-benchmark") / MESH_VERSION)
    args = parser.parse_args(argv)
    doc = write_outputs(hl.generate_suite(args.seed)["layouts"], args.output)
    print(f"wrote {len(doc['layouts'])} layout meshes to {args.output}")


if __name__ == "__main__":
    main()

