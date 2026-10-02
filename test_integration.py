"""Integration checks for run_pipeline.py, the pipeline viewer and integration_report.py.

Fixtures are perfect-depth predictions rendered from the procedural simulator, written in DA3 npz format. They
exercise the plumbing (stage order, manifests, labels, refusals); passing them says nothing about real DA3 output.
"""
import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import integration_report
import run_pipeline
import simulator

HERE = Path(__file__).resolve().parent


def synthetic_prediction(path, scene_seed=17, capture_seed=5, frames=36, width=160, height=120, fov=78., roll_all_90=False):
    scene = simulator.make_scene(scene_seed)
    poses = simulator.make_paths(scene, 120, capture_seed)[0]
    poses = [poses[i] for i in np.linspace(0, 119, frames, dtype=int)]
    depth, conf, K, E, imgs = [], [], [], [], []
    for pose in poses:
        image, best, intr = simulator.render(scene, pose, width, height, fov, True)
        yy, xx = np.mgrid[0:height, 0:width]
        zd = best / np.sqrt(1 + ((xx - intr["cx"]) / intr["fx"]) ** 2 + ((yy - intr["cy"]) / intr["fy"]) ** 2)
        yaw, pitch = pose["yaw"], pose["pitch"]
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.])
        forward = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), -math.sin(pitch)])
        up = np.cross(right, forward)
        if roll_all_90:  # phone held sideways: camera-up now points along the camera x axis
            right, up = up, -right
        R = np.stack([right, -up, forward])
        E.append(np.c_[R, -R @ np.asarray(pose["position"])])
        K.append(np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.]]))
        depth.append(zd.astype(np.float32)); conf.append(np.isfinite(zd).astype(np.float32)); imgs.append(image)
    np.savez_compressed(path, depth=np.array(depth), confidence=np.array(conf), intrinsics=np.array(K),
                        extrinsics=np.array(E), images=np.array(imgs))
    return scene


def run_cli(*args):
    return subprocess.run([sys.executable, str(HERE / "run_pipeline.py"), *map(str, args)], capture_output=True, text=True)


def stages(run_dir):
    manifest = json.loads((Path(run_dir) / "pipeline_manifest.json").read_text(encoding="utf-8"))
    return manifest, {s["name"]: s for s in manifest["stages"]}


_FX = {}


def fx():
    """Build the three synthetic runs once per test process."""
    if _FX:
        return _FX
    import atexit
    tmp = tempfile.TemporaryDirectory(); atexit.register(tmp.cleanup)
    root = Path(tmp.name)
    _FX.update(tmp=tmp, root=root, runs={})
    _FX["pred_ok"] = root / "ok.npz"; _FX["scene"] = synthetic_prediction(_FX["pred_ok"], 17)
    _FX["pred_incomplete"] = root / "incomplete.npz"; synthetic_prediction(_FX["pred_incomplete"], 3)
    _FX["pred_sideways"] = root / "sideways.npz"; synthetic_prediction(_FX["pred_sideways"], 17, roll_all_90=True)
    _FX["hash_before"] = run_pipeline.sha256(_FX["pred_ok"])
    for name in ("ok", "incomplete", "sideways"):
        _FX["runs"][name] = root / f"run_{name}"
        done = run_cli("from-prediction", "--prediction", _FX[f"pred_{name}"], "--output", _FX["runs"][name])
        assert done.returncode == 0, done.stderr
    return _FX


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = fx()
        cls.tmp, cls.runs, cls.scene = f["tmp"], f["runs"], f["scene"]
        cls.pred_ok, cls.hash_before = f["pred_ok"], f["hash_before"]
    def test_stages_recorded_with_hashes_commands_durations_and_status(self):
        manifest, st = stages(self.runs["ok"])
        self.assertEqual([s["name"] for s in manifest["stages"]],
                         ["dependencies", "inference", "planes", "up_prior", "polygon", "orientation", "box_orientation_diagnostic", "viewer"])
        for name in ("planes", "up_prior", "polygon", "orientation", "viewer"):
            record = st[name]
            self.assertEqual(record["status"], "ok", record["failure_reasons"])
            self.assertIsInstance(record["command"], list)
            self.assertGreaterEqual(record["duration_seconds"], 0)
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in record["outputs"].values()))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in st["polygon"]["inputs"].values()))
        self.assertEqual(manifest["prediction_sha256"], self.hash_before)
        self.assertEqual(run_pipeline.sha256(self.pred_ok), self.hash_before)  # frozen input untouched
        self.assertTrue(st["inference"]["reused"])

    def test_perfect_depth_fixture_gives_a_labelled_polygon_and_orientation(self):
        polygon = json.loads((self.runs["ok"] / "polygon.json").read_text(encoding="utf-8"))
        self.assertEqual(polygon["status"], "complete_polygon_hypothesis")
        self.assertEqual(len(polygon["edges"]), 4)
        self.assertTrue(all(e["label"] in ("observed", "inferred") for e in polygon["edges"]))
        orientation = json.loads((self.runs["ok"] / "orientation.json").read_text(encoding="utf-8"))
        self.assertEqual(orientation["status"], "oriented_unscaled_polygon_floorplan")
        self.assertIsNone(orientation["height_unscaled"])
        self.assertEqual(orientation["height_status"], "unavailable_not_estimated")
        self.assertTrue((self.runs["ok"] / "cuboid_baseline.json").exists())

    def test_no_seeds_poses_or_dimensions_reach_reconstruction(self):
        manifest, st = stages(self.runs["ok"])
        for record in manifest["stages"]:
            joined = " ".join(record.get("command") or [])
            self.assertNotIn("private_ground_truth", joined)
            self.assertNotRegex(joined, r"--(scene|capture|seed)")
        for path in ("planes.json", "polygon.json", "up_prior.json"):
            text = (self.runs["ok"] / path).read_text(encoding="utf-8")
            self.assertNotIn('"scene_seed"', text)
        self.assertNotIn("scene", " ".join(st["polygon"]["command"]))
        for path in ("planes.json", "polygon.json", "up_prior.json", "cuboid_baseline.json"):
            text = (self.runs["ok"] / path).read_text(encoding="utf-8")
            for key in ("width", "length", "height"):  # simulator dimensions must not appear as exact values
                self.assertNotIn(repr(float(self.scene[key])), text)

    def test_private_truth_inputs_and_existing_outputs_are_refused(self):
        private = Path(self.tmp.name) / "private_ground_truth"; private.mkdir(exist_ok=True)
        pred = private / "prediction.npz"; pred.write_bytes(self.pred_ok.read_bytes())
        out = Path(self.tmp.name) / "refused"
        done = run_cli("from-prediction", "--prediction", pred, "--output", out)
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(out.exists())
        again = run_cli("from-prediction", "--prediction", self.pred_ok, "--output", self.runs["ok"])
        self.assertNotEqual(again.returncode, 0)  # never overwrites a frozen/previous result

    def test_incomplete_polygon_is_explicit_and_downstream_stages_say_why(self):
        manifest, st = stages(self.runs["incomplete"])
        self.assertEqual(st["polygon"]["status"], "incomplete")
        self.assertIn("fewer_than_three_supported_walls", st["polygon"]["failure_reasons"])
        self.assertEqual(st["orientation"]["status"], "incomplete")
        self.assertTrue(st["orientation"]["failure_reasons"])
        self.assertFalse(manifest["summary"]["polygon_complete"])
        polygon = json.loads((self.runs["incomplete"] / "polygon.json").read_text(encoding="utf-8"))
        self.assertIsNone(polygon["vertices_xy"]); self.assertEqual(polygon["edges"], [])

    def test_failed_orientation_prior_is_not_rescued(self):
        manifest, st = stages(self.runs["sideways"])
        self.assertEqual(st["up_prior"]["status"], "incomplete")
        polygon = json.loads((self.runs["sideways"] / "polygon.json").read_text(encoding="utf-8"))
        self.assertEqual(polygon["incomplete_boundaries"], [{"reason": "up_axis_unavailable"}])
        self.assertNotIn("--up", " ".join(st["polygon"]["command"]))
        self.assertNotEqual(st["orientation"]["status"], "ok")

    def test_named_blockers_stay_open_in_every_manifest(self):
        for run in self.runs.values():
            manifest, _ = stages(run)
            names = {b["name"] for b in manifest["named_blockers"]}
            self.assertEqual(names, set(run_pipeline.NAMED_BLOCKERS))
            self.assertTrue(all(b["status"] == "open" for b in manifest["named_blockers"]))
            self.assertIn("none", manifest["summary"]["claims"])


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = fx()["runs"]

    def html(self, name):
        return (self.runs[name] / "viewer.html").read_text(encoding="utf-8")

    def data(self, name):
        text = self.html(name)
        return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S).group(1).replace("<\\/", "</"))

    def test_viewer_is_offline_and_has_every_required_tab_and_label(self):
        text = self.html("ok")
        self.assertNotRegex(text, r'(?:src|href)\s*=\s*["\']https?://')
        self.assertNotIn("@import", text); self.assertNotIn("cdn.", text.lower())
        for tab in ("Input video", "Point geometry", "Plane patches", "Floor-plan polygon", "Stages &amp; blockers", "Unavailable stages"):
            self.assertIn(tab.replace("&amp;", "&"), text)
        for label in ("observed", "inferred", "incomplete", "unavailable"):
            self.assertIn(label, text)
        data = self.data("ok")
        self.assertEqual(data["points"]["status"], "observed"); self.assertGreater(data["points"]["count"], 1000)
        self.assertEqual(data["patches"]["status"], "observed"); self.assertGreater(len(data["patches"]["quads"]), 0)
        self.assertEqual(data["polygon"]["status"], "observed")
        self.assertIn("<svg", data["polygon"]["svg"])
        self.assertEqual(set(data["unavailable"]), {"light_field", "segmentation", "completion_mesh", "ceiling_height", "openings"})
        self.assertIn("not a light field", data["points"]["note"])

    def test_missing_video_is_unavailable_not_invented(self):
        self.assertEqual(self.data("ok")["video"]["status"], "unavailable")

    def test_incomplete_polygon_is_never_drawn_as_a_room(self):
        data = self.data("incomplete")
        self.assertEqual(data["polygon"]["status"], "incomplete")
        self.assertNotIn("<polygon", data["polygon"]["svg"])
        self.assertIn("INCOMPLETE", data["polygon"]["note"] + data["polygon"]["svg"])
        self.assertTrue(data["polygon"]["rejected"])
        self.assertEqual(data["cuboid_baseline"]["note"], "diagnostic baseline only; never shown as the room")
        self.assertNotIn("corners", json.dumps(data["cuboid_baseline"]))

    def test_up_axis_unavailable_draws_no_plan(self):
        data = self.data("sideways")
        self.assertEqual(data["polygon"]["svg"], "")
        self.assertEqual(data["polygon"]["status"], "incomplete")
        self.assertIn("up_axis_unavailable", data["polygon"]["note"])
        self.assertIn("no declared up", data["frame"])

    def test_video_path_is_relative_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clips" / "walk.mp4"; video.parent.mkdir(); video.write_bytes(b"\x00")
            out = Path(tmp) / "viewer.html"
            done = subprocess.run([sys.executable, str(HERE / "viewer.py"), "--pipeline-run", str(self.runs["ok"]), "--video", str(video),
                                   "--output", str(out)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', out.read_text(encoding="utf-8"), re.S).group(1))
            self.assertEqual(data["video"], {"status": "observed", "src": "clips/walk.mp4", "name": "walk.mp4"})


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs, cls.scene = fx()["runs"], fx()["scene"]

    def test_collect_reports_failures_and_keeps_blockers_open(self):
        report = integration_report.collect([self.runs["ok"], self.runs["incomplete"], self.runs["sideways"]])
        self.assertEqual(report["run_count"], 3)
        self.assertEqual(report["polygon_complete_count"], 1)
        self.assertEqual(report["oriented_count"], 1)
        by_name = {r["name"]: r for r in report["runs"]}
        self.assertIn("polygon", by_name["run_incomplete"]["failure_reasons"])
        self.assertEqual(by_name["run_sideways"]["up_prior"]["status"], "refused")
        self.assertEqual(len(report["named_blockers"]), len(run_pipeline.NAMED_BLOCKERS))
        self.assertIn("none", report["claims"])
        markdown = integration_report.to_markdown(report)
        for needle in ("Named blockers", "ceiling_height_estimate", "fewer_than_three_supported_walls", "up_axis_unavailable"):
            self.assertIn(needle, markdown)

    def test_parent_folder_expands_to_path_runs_and_evaluation_is_a_separate_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"; parent.mkdir()
            for name in ("path_00", "path_01"):
                done = run_cli("from-prediction", "--prediction", fx()["pred_ok"], "--output", parent / name)
                self.assertEqual(done.returncode, 0, done.stderr)
            runs = integration_report.find_runs([parent])
            self.assertEqual([r.name for r in runs], ["path_00", "path_01"])
            report = integration_report.collect([parent])
            self.assertNotIn("evaluation", report)
            scene_path = Path(tmp) / "scene.json"; scene_path.write_text(json.dumps(self.scene), encoding="utf-8")
            result = integration_report.evaluate_runs(runs, report["runs"], scene_path)
            self.assertEqual(result["evaluator_mode"], "height_normalized_3m")
            self.assertEqual(result["official_anchor_protocols"]["status"], "not_evaluated")
            self.assertFalse(result["completion"])
            self.assertFalse(result["custom_diagnostic"]["custom_1_5cm_gate_full_dataset"])
            self.assertEqual(len(result["custom_diagnostic"]["per_run"]), 2)
            # Ground truth never appears in the prediction document built for the evaluator.
            pred, gt = integration_report.build_eval_documents(report["runs"], runs, self.scene, "h")
            self.assertNotIn(str(round(self.scene["height"], 6)), json.dumps(pred))

    def test_report_cli_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "report"
            args = [sys.executable, str(HERE / "integration_report.py"), "--runs", str(self.runs["ok"]), "--output", str(prefix)]
            self.assertEqual(subprocess.run(args, capture_output=True, text=True).returncode, 0)
            self.assertNotEqual(subprocess.run(args, capture_output=True, text=True).returncode, 0)
            self.assertTrue(prefix.with_suffix(".md").exists())


class InterpreterTests(unittest.TestCase):
    def test_dependency_stage_checks_cv2_numpy_and_torch(self):
        _, st = stages(fx()["runs"]["ok"])
        self.assertEqual(st["dependencies"]["status"], "ok", st["dependencies"]["failure_reasons"])
        self.assertEqual(len(st["dependencies"]["versions"]), 5)


if __name__ == "__main__":
    unittest.main()
