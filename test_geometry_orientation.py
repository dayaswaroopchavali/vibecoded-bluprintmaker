"""Geometry-only structural-axis hypotheses: no camera image-up, camera path, poses or ground truth are involved."""
import inspect
import unittest

import numpy as np

import geometry_orientation as G


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def prism_planes(outline, height=2.7, rotation=None, shift=(0., 0., 0.), spanning=True, extra=(), scale=1.0, support=700):
    """Planes of a prism room (walls + floor + ceiling) in an arbitrary world frame, like extract_planes output.

    The true prism axis is rotation[:, 2]; an upside-down capture is just a rotation that points it down.
    """
    R = np.eye(3) if rotation is None else rotation
    verts = np.asarray(outline, dtype=float) * scale; h = height * scale; shift = np.asarray(shift, dtype=float)
    planes = []

    def add(normal, point, patch_axes=None, bounds=None, count=support):
        n = R @ normal; p = R @ point + shift
        record = {"normal": n.tolist(), "offset": float(-n @ p), "center": p.tolist(), "support_count": count}
        if patch_axes is not None:
            record["surface_patches"] = {"axes_world": [(R @ a).tolist() for a in patch_axes], "cell_size": .1,
                                         "patches": [{"bounds_uv": bounds, "support_count": count}]}
        planes.append(record)
    for k in range(len(verts)):
        a, b = verts[k], verts[(k + 1) % len(verts)]
        d = (b - a) / np.linalg.norm(b - a); m = np.array([d[1], -d[0], 0.]); length = np.linalg.norm(b - a)
        top = h if spanning else .3 * h
        add(m, np.array([*(a + b) / 2, h / 2]), [np.array([d[0], d[1], 0.]), np.array([0., 0., 1.])],
            [[-length / 2, -top / 2 + (h / 2 - top / 2) * 0], [length / 2, top / 2]])
    c = np.array([*verts.mean(axis=0), 0.]); big = 4 * support
    add(np.array([0., 0., 1.]), c, count=big); add(np.array([0., 0., -1.]), c + np.array([0, 0, h]), count=big)
    for spec in extra:
        planes.append(spec)
    return {"planes": planes}


PENTAGON = [(0, 0), (5, 0), (6.5, 3), (2.5, 5.5), (-1, 3)]
SKEW_L = [(0, 0), (6, 0), (6.8, 2.2), (3, 3.1), (2.2, 6), (0, 6)]
BOX = [(0, 0), (6, 0), (6, 4), (0, 4)]


def best(result):
    return np.asarray(result["hypotheses"][0]["axis_world"])


class StructuralAxisTests(unittest.TestCase):
    def test_pentagon_and_skew_walls_resolve_the_axis_up_to_sign_from_geometry_alone(self):
        for outline in (PENTAGON, SKEW_L):
            for seed in range(8):
                R = _rotation(seed)
                result = G.axis_hypotheses(prism_planes(outline, rotation=R, shift=(30., -5., 8.)))
                self.assertEqual(result["status"], "single_axis_sign_ambiguous", (outline, seed, result.get("hypotheses", [{}])[0]))
                self.assertAlmostEqual(abs(float(best(result) @ R[:, 2])), 1.0, places=6)
                self.assertFalse(result["sign_resolved"]); self.assertFalse(result["hypotheses"][0]["sign_resolved"])
                np.testing.assert_allclose(result["hypotheses"][0]["alternative_sign_axis_world"], -best(result))
                self.assertGreaterEqual(result["hypotheses"][0]["wall_azimuth_count"], 4)
                self.assertEqual(result["hypotheses"][0]["co_supported_with_best"], True)
                self.assertEqual(result["co_supported_count"], 1)

    def test_upside_down_and_rolled_scenes_give_the_same_axis_with_the_sign_left_open(self):
        R = _rotation(3)
        up = G.axis_hypotheses(prism_planes(PENTAGON, rotation=R))
        flip = np.diag([1., -1., -1.])                      # roll by 180 degrees: the whole scene is upside-down
        down = G.axis_hypotheses(prism_planes(PENTAGON, rotation=R @ flip))
        self.assertAlmostEqual(abs(float(best(up) @ best(down))), 1.0, places=6)
        for result in (up, down):
            self.assertFalse(result["sign_resolved"]); self.assertFalse(result["gravity_known"])

    def test_box_room_is_symmetric_so_all_axes_stay_co_supported(self):
        for seed in range(5):
            R = _rotation(seed)
            result = G.axis_hypotheses(prism_planes(BOX, rotation=R))
            self.assertEqual(result["status"], "ambiguous_axes")
            self.assertEqual(result["co_supported_count"], 3)
            self.assertIsNone(result["best_axis_world"])
            axes = [np.asarray(h["axis_world"]) for h in result["hypotheses"] if h["co_supported_with_best"]]
            for k in range(3):
                self.assertTrue(any(abs(float(a @ R[:, k])) > .999 for a in axes), "a true box axis is missing from the hypotheses")

    def test_no_camera_or_ground_truth_inputs_exist_and_flags_say_so(self):
        params = set(inspect.signature(G.axis_hypotheses).parameters)
        self.assertEqual(params, {"plane_data", "points"})
        import ast
        tree = ast.parse(inspect.getsource(G))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imported - {"__future__"}, {"argparse", "itertools", "json", "pathlib", "numpy", "extract_planes"})
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for forbidden in ("extrinsics", "camera_up_prior", "camera_centers", "private_ground_truth"):
            self.assertNotIn(forbidden, names)
        result = G.axis_hypotheses(prism_planes(PENTAGON))
        self.assertFalse(result["camera_orientation_used"]); self.assertFalse(result["camera_path_used"])
        self.assertFalse(result["gravity_known"])

    def test_walls_that_do_not_span_the_bracket_score_lower(self):
        full = G.axis_hypotheses(prism_planes(PENTAGON, rotation=_rotation(2), spanning=True), np.zeros((3, 3)))
        short = G.axis_hypotheses(prism_planes(PENTAGON, rotation=_rotation(2), spanning=False), np.zeros((3, 3)))
        a, b = full["hypotheses"][0], short["hypotheses"][0]
        self.assertGreater(a["walls_span_bracket_fraction"], b["walls_span_bracket_fraction"])
        self.assertGreater(a["score"], b["score"])

    def test_tilted_clutter_plane_is_unexplained_but_does_not_move_the_axis(self):
        R = _rotation(6)
        n = R @ _unit_vec([.5, .1, .85]); clutter = {"normal": n.tolist(), "offset": 1.3, "center": [0, 0, 0], "support_count": 600}
        result = G.axis_hypotheses(prism_planes(PENTAGON, rotation=R, extra=[clutter]))
        self.assertAlmostEqual(abs(float(best(result) @ R[:, 2])), 1.0, places=6)
        self.assertTrue(result["hypotheses"][0]["unexplained_plane_ids"])
        self.assertLess(result["hypotheses"][0]["explained_support_fraction"], 1.0)

    def test_scale_translation_and_rotation_do_not_change_the_hypotheses(self):
        base = G.axis_hypotheses(prism_planes(SKEW_L, rotation=np.eye(3)))
        moved = G.axis_hypotheses(prism_planes(SKEW_L, rotation=_rotation(9), shift=(500., 20., -70.), scale=7.5))
        self.assertEqual(base["status"], moved["status"])
        self.assertAlmostEqual(base["hypotheses"][0]["score"], moved["hypotheses"][0]["score"], places=6)
        self.assertEqual(base["hypotheses"][0]["wall_azimuth_count"], moved["hypotheses"][0]["wall_azimuth_count"])

    def test_too_few_planes_or_a_single_wall_direction_is_refused(self):
        self.assertEqual(G.axis_hypotheses({"planes": []})["status"], "insufficient_planes")
        two = prism_planes(BOX)["planes"][:2]
        self.assertEqual(G.axis_hypotheses({"planes": two})["status"], "insufficient_planes")
        parallel = {"planes": [dict(normal=[0, 1, 0], offset=-k, support_count=500) for k in (0, 3, 6, 9)]}
        self.assertIn(G.axis_hypotheses(parallel)["status"], ("no_supported_axis", "insufficient_planes"))

    def test_axes_without_a_bracketing_pair_are_weak_not_resolved(self):
        planes = prism_planes(PENTAGON, rotation=_rotation(5))
        walls_only = {"planes": planes["planes"][:-2]}                     # floor and ceiling planes were not observed
        result = G.axis_hypotheses(walls_only)
        self.assertEqual(result["status"], "weakly_supported_axes")
        self.assertIsNone(result["hypotheses"][0]["bracketing_pair"])
        self.assertFalse(result["sign_resolved"])

    def test_output_is_deterministic(self):

        planes = prism_planes(PENTAGON, rotation=_rotation(4))
        self.assertEqual(G.axis_hypotheses(planes), G.axis_hypotheses(planes))


def _unit_vec(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


if __name__ == "__main__":
    unittest.main()
