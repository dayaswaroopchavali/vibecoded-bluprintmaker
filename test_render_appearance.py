"""Real Blender renders (small) for the appearance ablation: original vs textured, frozen poses, seeds and statistics."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import blender_capture as bc
import render_appearance as ra
import simulator

W, H, FOV = 160, 120, 78.
EXISTING = Path(r"Path(__file__).resolve().parents[2] / "outputs/sim-benchmark/blender-coverage-v3/capture"")


def _poses():
    base = {"position": [1.2, -1.0, 1.4], "yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    a = {"position": [1.2, -1.0, 1.4], "yaw": 0.0, "pitch": float(np.pi - .3), "roll": 0.0}   # beyond 90 degrees: inverted
    b = {"position": [1.2, -1.0, 1.4], "yaw": float(np.pi), "pitch": .3, "roll": float(np.pi)}  # same orientation, other route
    c = dict(base, roll=float(np.pi))
    return [base, a, b, c]


def _img(folder, name="path_00", i=0):
    return cv2.imread(str(Path(folder) / "rgb" / name / f"{i:06d}.png"))


class AppearanceRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); root = Path(cls.tmp.name)
        cls.scene = simulator.make_scene(17)
        cap = root / "capture"; priv = cap / "private_ground_truth"; priv.mkdir(parents=True)
        (priv / "scene.json").write_text(json.dumps(cls.scene), encoding="utf-8")
        cls.poses = _poses()
        (priv / "path_00.json").write_text(json.dumps({"path_seed": 5, "poses": cls.poses}), encoding="utf-8")
        (priv / "seed_manifest.json").write_text(json.dumps({"capture_seed": 3101}), encoding="utf-8")
        cls.cap = cap
        cls.out = {}
        for key, app, seed in (("orig", "original", 1), ("t1", "textured", 1), ("t1b", "textured", 1), ("t2", "textured", 2)):
            cls.out[key] = root / key
            ra.render(cap, cls.out[key], "path_00", app, seed, W, H, FOV, 6, 1)
        cls.manifest = {k: json.loads((v / "appearance_manifest.json").read_text(encoding="utf-8")) for k, v in cls.out.items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_layout_and_private_truth_separation(self):
        o = self.out["t1"]
        self.assertEqual(sorted(p.name for p in o.iterdir()), ["appearance_manifest.json", "contact_sheets", "private_ground_truth", "rgb", "videos"])
        self.assertEqual({p.suffix for p in (o / "rgb").rglob("*") if p.is_file()}, {".png"})
        self.assertTrue((o / "contact_sheets" / "path_00.jpg").is_file() and (o / "videos" / "path_00.mp4").is_file())
        self.assertTrue((o / "private_ground_truth" / "scene.json").is_file())
        with self.assertRaises(FileExistsError):
            ra.render(self.cap, o, "path_00", "textured", 1, W, H, FOV)

    def test_original_manifest_matches_original_material_calls(self):
        mats = {m["name"]: m for m in self.manifest["orig"]["materials"]}
        s = self.scene
        for i, c in enumerate(s["wall_colors"]):
            self.assertEqual(mats[f"wall {i+1}"], {"name": f"wall {i+1}", "color": [float(v) for v in c], "noise_scale": 5.5, "roughness": .8})
        self.assertEqual(mats["subtle floor grain"]["noise_scale"], 3.0); self.assertEqual(mats["ceiling"]["noise_scale"], 7.0)
        for f in s["furniture"]:
            self.assertEqual(mats[f["name"] + " fabric or wood"]["noise_scale"], 8.0)
        self.assertIsNone(self.manifest["orig"]["texture_seed"])

    @unittest.skipUnless((EXISTING / "rgb/path_00/000000.png").is_file(), "existing capture not present")
    def test_original_appearance_reproduces_existing_capture_pixels(self):
        import tempfile as tf
        with tf.TemporaryDirectory() as t:
            out = Path(t) / "o"
            ra.render(EXISTING, out, "path_00", "original", 1, 480, 360, 78., 6, 1, frames="0,37")
            for f in (0, 37):
                d = np.abs(_img(out, i=f).astype(int) - _img(EXISTING, i=f).astype(int))
                self.assertLessEqual(int(d.max()), 2)   # bit-exact PNG bytes are not stable across processes; pixels are within 1 LSB

    def test_same_poses_geometry_across_appearances(self):
        hs = {k: m["poses_sha256"]["path_00"] for k, m in self.manifest.items()}
        self.assertEqual(len(set(hs.values())), 1)
        self.assertEqual(hs["orig"], bc.poses_sha256(self.poses))
        pj = {k: (v / "private_ground_truth" / "path_00.json").read_bytes() for k, v in self.out.items()}
        self.assertEqual(len(set(hashlib.sha256(b).hexdigest() for b in pj.values())), 1)
        sc = {(v / "private_ground_truth" / "scene.json").read_bytes() for v in self.out.values()}
        self.assertEqual(len(sc), 1)
        # image-space geometry proxy: dark ceiling mask and coarse structural edges agree between appearances
        for i in range(4):
            a, b = _img(self.out["orig"], i=i), _img(self.out["t1"], i=i)
            ma = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) < 60; mb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) < 60
            union = (ma | mb).sum()
            if union > 200: self.assertGreater((ma & mb).sum() / union, .9, f"ceiling mask IoU frame {i}")
            ea = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (0, 0), 2), 10, 30)
            eb = cv2.dilate(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (0, 0), 2), 10, 30), np.ones((7, 7), np.uint8))
            if (ea > 0).sum() > 30: self.assertGreater(((ea > 0) & (eb > 0)).sum() / (ea > 0).sum(), .75, f"structure edges frame {i}")
        self.assertGreater(np.abs(_img(self.out["orig"]).astype(int) - _img(self.out["t1"]).astype(int)).mean(), .5)  # pixels do change

    def test_texture_seed_determinism_and_difference(self):
        for i in range(4):
            self.assertTrue(np.array_equal(_img(self.out["t1"], i=i), _img(self.out["t1b"], i=i)))
        self.assertGreater(np.abs(_img(self.out["t1"], i=1).astype(int) - _img(self.out["t2"], i=1).astype(int)).mean(), .05)
        m1, m2 = self.manifest["t1"], self.manifest["t2"]
        self.assertEqual(m1["poses_sha256"], m2["poses_sha256"])
        self.assertNotEqual(m1["materials"]["procedural"], m2["materials"]["procedural"])
        self.assertEqual((self.out["t1"] / "private_ground_truth" / "path_00.json").read_bytes(), (self.out["t2"] / "private_ground_truth" / "path_00.json").read_bytes())
        self.assertEqual(m1["texture_seed"], 1); self.assertEqual(m2["texture_seed"], 2)

    def test_exact_roll_and_inverted_poses(self):
        r_a, u_a, f_a = simulator.camera_basis(self.poses[1]); r_b, u_b, f_b = simulator.camera_basis(self.poses[2])
        self.assertLess(u_a[2], 0)                       # frame 1 is genuinely upside down
        for k in ("orig", "t1"):
            a, b = _img(self.out[k], i=1).astype(int), _img(self.out[k], i=2).astype(int)
            self.assertLess(np.abs(a - b).mean(), 1.5)  # same basis reached via pitch>90 vs yaw+pi & roll pi
            base, rolled = _img(self.out[k], i=0), _img(self.out[k], i=3)
            self.assertLess(np.abs(cv2.rotate(rolled, cv2.ROTATE_180).astype(int) - base.astype(int)).mean(), 1.5)  # roll = pi turns image 180 degrees
            self.assertGreater(np.abs(rolled.astype(int) - base.astype(int)).mean(), 2.)

    def test_no_forbidden_appearance_features(self):
        # frame 0 looks at a plain wall: central crop must be low-contrast, desaturated and free of saturated pure colours
        for k in ("orig", "t1", "t2"):
            im = _img(self.out[k], i=0); h, w = im.shape[:2]
            crop = im[int(h * .35):int(h * .6), int(w * .3):int(w * .7)].astype(np.float32) / 255
            g = crop.mean(2); hp = g - cv2.blur(g, (5, 5))
            hsv = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
            self.assertLess(hsv[..., 1].max() / 255, .5)
            self.assertLess(g.std() / g.mean(), .06)          # luminance variation of plain painted wall: a few percent
            self.assertLess(hp.std() / g.mean(), .02)
        # whole-frame: no pure primary/secondary colours anywhere (no fiducials / coloured codes)
        for k in ("t1", "t2"):
            for i in range(4):
                hsv = cv2.cvtColor(_img(self.out[k], i=i), cv2.COLOR_BGR2HSV)
                self.assertFalse(((hsv[..., 1] > 200) & (hsv[..., 2] > 150)).any())
        # no high-contrast checker structure: nothing in the textured parameter set is a checker/marker
        blob = json.dumps(self.manifest["t1"]["materials"]).lower()
        for bad in ("checker", "aruco", "marker", "qr", "wallpaper", "image"):
            self.assertNotIn(bad, blob.replace("no wallpaper", ""))
        cov = self.manifest["t1"]["materials"]["procedural"]["coverage"]
        self.assertEqual(cov["wall_area_with_structured_pattern_fraction"], 0.0)


class CliTests(unittest.TestCase):
    def test_parse_frames_and_refusal_without_blender(self):
        self.assertEqual(ra.parse_frames(None, 10, 3), [0, 3, 6, 9]); self.assertEqual(ra.parse_frames("2,4", 10, 1), [2, 4])
        with self.assertRaises(ValueError): ra.parse_frames("11", 10, 1)
        with self.assertRaises(FileNotFoundError): ra.render(Path("nonexistent_capture"), Path("x_out_never"))


if __name__ == "__main__":
    unittest.main()
