"""Integration checks for run_pipeline.py, the pipeline viewer and integration_report.py.

Fixtures are perfect-depth predictions rendered from the procedural simulator, written in DA3 npz format. They
exercise the plumbing (stage order, manifests, labels, refusals); passing them says nothing about real DA3 output.
Captures roll, tilt and invert freely, so production runs have no gravity direction and orientation stays unresolved;
the upright-phone camera-up branch exists only as labelled legacy diagnostics, exercised here on upright fixtures.
"""
import atexit
import json
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
import viewer
from room_hypothesis import estimate_polygon
from test_room_hypothesis import L_SHAPE, _polygon_room

HERE = Path(__file__).resolve().parent


def synthetic_prediction(path, scene_seed=17, capture_seed=5, mode="spherical", frames=36, width=160, height=120, fov=78.):
    """Perfect-depth DA3-format prediction; mode 'bounded' gives the legacy upright, roll-free camera."""
    scene = simulator.make_scene(scene_seed)
    poses = simulator.make_paths(scene, 120, capture_seed, mode)[0]
    poses = [poses[i] for i in np.linspace(0, 119, frames, dtype=int)]
    depth, conf, K, E, imgs = [], [], [], [], []
    for pose in poses:
        image, best, intr = simulator.render(scene, pose, width, height, fov, True)
        yy, xx = np.mgrid[0:height, 0:width]
        zd = best / np.sqrt(1 + ((xx - intr["cx"]) / intr["fx"]) ** 2 + ((yy - intr["cy"]) / intr["fy"]) ** 2)
        right, up, forward = simulator.camera_basis(pose)            # includes roll / inversion
        R = np.stack([right, -up, forward])                          # OpenCV world-to-camera: x right, y down, z forward
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
    """Build the synthetic runs once per test process."""
    if _FX:
        return _FX
    tmp = tempfile.TemporaryDirectory(); atexit.register(tmp.cleanup)
    root = Path(tmp.name)
    _FX.update(tmp=tmp, root=root, runs={}, preds={})
    cases = {"rolled": (17, "spherical", False), "upright_ok": (17, "bounded", True), "upright_incomplete": (3, "bounded", True)}
    for name, (scene_seed, mode, legacy) in cases.items():
        pred = root / f"{name}.npz"; scene = synthetic_prediction(pred, scene_seed, mode=mode)
        _FX["preds"][name] = pred
        if name == "rolled":
            _FX["scene"] = scene
        _FX["runs"][name] = root / f"run_{name}"
        done = run_cli("from-prediction", "--prediction", pred, "--output", _FX["runs"][name], *(["--legacy-upright-diagnostic"] if legacy else []))
        assert done.returncode == 0, done.stderr
    _FX["hash_before"] = run_pipeline.sha256(_FX["preds"]["rolled"])
    return _FX


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = fx(); cls.runs, cls.preds, cls.scene, cls.root = f["runs"], f["preds"], f["scene"], f["root"]

    def test_stages_recorded_with_hashes_commands_durations_and_status(self):
        manifest, st = stages(self.runs["rolled"])
        self.assertEqual([s["name"] for s in manifest["stages"]],
                         ["dependencies", "inference", "planes", "orientation_resolution", "polygon", "orientation", "viewer"])
        for name in ("planes", "polygon", "viewer"):
            record = st[name]
            self.assertIn(record["status"], ("ok", "incomplete"))
            self.assertIsInstance(record["command"], list)
            self.assertGreaterEqual(record["duration_seconds"], 0)
            self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in record["outputs"].values()))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for h in st["polygon"]["inputs"].values()))
        self.assertEqual(manifest["prediction_sha256"], _FX["hash_before"])
        self.assertEqual(run_pipeline.sha256(self.preds["rolled"]), _FX["hash_before"])   # frozen input untouched
        self.assertTrue(st["inference"]["reused"])

    def test_rolled_inverted_capture_leaves_orientation_unresolved_and_no_gravity_is_used(self):
        manifest, st = stages(self.runs["rolled"])
        self.assertEqual(st["polygon"]["status"], "incomplete")
        self.assertEqual(st["polygon"]["failure_reasons"], ["up_axis_unavailable"])
        self.assertFalse(any(a.startswith("--up") and a != "--up-output" for a in st["polygon"]["command"]))
        for name in ("orientation_resolution", "orientation"):
            self.assertEqual(st[name]["status"], "incomplete")
            self.assertIn("orientation_unresolved", st[name]["failure_reasons"][0])
        self.assertEqual(manifest["summary"]["orientation_status"], "unresolved")
        self.assertFalse(manifest["summary"]["polygon_complete"])
        self.assertFalse((self.runs["rolled"] / "legacy_up_prior.json").exists())      # legacy branch is opt-in
        self.assertFalse(any(k.startswith("legacy_") for k in st))
        polygon = json.loads((self.runs["rolled"] / "polygon.json").read_text(encoding="utf-8"))
        self.assertIsNone(polygon["vertices_xy"]); self.assertEqual(polygon["edges"], [])

    def test_legacy_upright_branch_is_labelled_and_never_counts_as_a_result(self):
        manifest, st = stages(self.runs["upright_ok"])
        legacy = {k: v for k, v in st.items() if k.startswith("legacy_")}
        self.assertEqual(set(legacy), {"legacy_camera_up_prior_diagnostic", "legacy_upright_polygon_diagnostic",
                                       "legacy_orientation_diagnostic", "legacy_box_orientation_diagnostic"})
        self.assertTrue(all("LEGACY" in v["failure_reasons"][0] for v in legacy.values()))
        self.assertEqual(legacy["legacy_upright_polygon_diagnostic"]["status"], "ok")      # perfect upright fixture completes
        self.assertTrue(any(a.startswith("--up=") for a in legacy["legacy_upright_polygon_diagnostic"]["command"]))
        summary = manifest["summary"]
        self.assertFalse(summary["polygon_complete"])                                      # production result stays incomplete
        self.assertEqual(summary["orientation_status"], "unresolved")
        self.assertEqual(set(summary["legacy_diagnostic_stage_status"]), set(legacy))
        self.assertNotIn("legacy", " ".join(summary["stage_status"]))
        _, bad = stages(self.runs["upright_incomplete"])
        self.assertEqual(bad["legacy_upright_polygon_diagnostic"]["status"], "incomplete")

    def test_legacy_prior_is_refused_on_the_rolled_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            done = run_cli("from-prediction", "--prediction", self.preds["rolled"], "--output", out, "--legacy-upright-diagnostic")
            self.assertEqual(done.returncode, 0, done.stderr)
            _, st = stages(out)
            self.assertEqual(st["legacy_camera_up_prior_diagnostic"]["status"], "incomplete")   # no consensus when the camera rolls
            self.assertNotIn("legacy_upright_polygon_diagnostic", st)

    def test_no_seeds_poses_or_dimensions_reach_reconstruction(self):
        manifest, st = stages(self.runs["rolled"])
        for record in manifest["stages"]:
            joined = " ".join(record.get("command") or [])
            self.assertNotIn("private_ground_truth", joined)
            self.assertNotRegex(joined, r"--(scene|capture|seed)")
        for path in ("planes.json", "polygon.json", "cuboid_baseline.json"):
            text = (self.runs["rolled"] / path).read_text(encoding="utf-8")
            self.assertNotIn('"scene_seed"', text)
            for key in ("width", "length", "height"):
                self.assertNotIn(repr(float(self.scene[key])), text)

    def test_private_truth_inputs_and_existing_outputs_are_refused(self):
        private = self.root / "private_ground_truth"; private.mkdir(exist_ok=True)
        pred = private / "prediction.npz"; pred.write_bytes(self.preds["rolled"].read_bytes())
        out = self.root / "refused"
        done = run_cli("from-prediction", "--prediction", pred, "--output", out)
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(out.exists())
        again = run_cli("from-prediction", "--prediction", self.preds["rolled"], "--output", self.runs["rolled"])
        self.assertNotEqual(again.returncode, 0)  # never overwrites a frozen/previous result

    def test_named_blockers_stay_open_in_every_manifest(self):
        for run in self.runs.values():
            manifest, _ = stages(run)
            self.assertEqual({b["name"] for b in manifest["named_blockers"]}, set(run_pipeline.NAMED_BLOCKERS))
            self.assertTrue(all(b["status"] == "open" for b in manifest["named_blockers"]))
            self.assertIn("none", manifest["summary"]["claims"])
        self.assertIn("orientation_resolution", run_pipeline.NAMED_BLOCKERS)

    def test_dependency_stage_checks_cv2_numpy_and_torch(self):
        _, st = stages(self.runs["rolled"])
        self.assertEqual(st["dependencies"]["status"], "ok", st["dependencies"]["failure_reasons"])
        self.assertEqual(len(st["dependencies"]["versions"]), 5)


def _viewer_data(run):
    text = (Path(run) / "viewer.html").read_text(encoding="utf-8")
    return text, json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S).group(1).replace("<\\/", "</"))


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = fx()["runs"]

    def test_viewer_is_offline_and_has_every_required_tab_and_label(self):
        text, data = _viewer_data(self.runs["rolled"])
        self.assertNotRegex(text, r'(?:src|href)\s*=\s*["\']https?://')
        self.assertNotIn("@import", text); self.assertNotIn("cdn.", text.lower())
        for tab in ("Input video", "Point geometry", "Plane patches", "Floor-plan polygon", "Stages & blockers", "Unavailable stages"):
            self.assertIn(tab, text)
        for label in ("observed", "inferred", "incomplete", "unavailable"):
            self.assertIn(label, text)
        self.assertEqual(data["points"]["status"], "observed"); self.assertGreater(data["points"]["count"], 1000)
        self.assertEqual(data["patches"]["status"], "observed"); self.assertGreater(len(data["patches"]["quads"]), 0)
        self.assertEqual(set(data["unavailable"]), {"light_field", "segmentation", "completion_mesh", "ceiling_height", "openings"})
        self.assertIn("not a light field", data["points"]["note"])
        self.assertIn("arbitrary DA3 world frame", data["frame"])
        self.assertIn("orientation unresolved", data["frame"])

    def test_missing_video_is_unavailable_not_invented(self):
        self.assertEqual(_viewer_data(self.runs["rolled"])[1]["video"]["status"], "unavailable")

    def test_unresolved_orientation_draws_no_plan_and_labels_incomplete(self):
        _, data = _viewer_data(self.runs["rolled"])
        self.assertEqual(data["polygon"]["status"], "incomplete")
        self.assertEqual(data["polygon"]["svg"], "")
        self.assertIn("up_axis_unavailable", data["polygon"]["note"])
        self.assertEqual(data["cuboid_baseline"]["note"], "diagnostic baseline only; never shown as the room")
        self.assertNotIn("corners", json.dumps(data["cuboid_baseline"]))

    def test_complete_polygon_edges_are_labelled_observed_or_inferred(self):
        data, points, up, _ = _polygon_room(L_SHAPE, seed=3, gaps=((0, .25, .85),))
        polygon = estimate_polygon(data, points, up=up)
        svg, label, note = viewer.plan_svg(polygon, None)
        self.assertEqual(polygon["status"], "complete_polygon_hypothesis")
        self.assertEqual(label, "inferred")                    # one wall is only partly observed
        self.assertIn("#5fd38d", svg); self.assertIn("#ffb347", svg); self.assertIn("stroke-dasharray", svg)
        self.assertIn("inferred", note)

    def test_incomplete_polygon_with_observed_segments_is_never_closed(self):
        data, points, up, _ = _polygon_room(L_SHAPE, seed=3, missing=(4,))
        polygon = estimate_polygon(data, points, up=up)
        svg, label, note = viewer.plan_svg(polygon, None)
        self.assertEqual(label, "incomplete")
        self.assertIn("INCOMPLETE", svg + note); self.assertNotIn("<polygon", svg)
        self.assertIn("#5aa9e6", svg)                          # observed wall segments only

    def test_video_path_is_relative_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clips" / "walk.mp4"; video.parent.mkdir(); video.write_bytes(b"\x00")
            out = Path(tmp) / "viewer.html"
            done = subprocess.run([sys.executable, str(HERE / "viewer.py"), "--pipeline-run", str(self.runs["rolled"]), "--video", str(video),
                                   "--output", str(out)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', out.read_text(encoding="utf-8"), re.S).group(1))
            self.assertEqual(data["video"], {"status": "observed", "src": "clips/walk.mp4", "name": "walk.mp4"})


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs, cls.scene = fx()["runs"], fx()["scene"]

    def test_collect_reports_failures_orientation_unresolved_and_blockers_open(self):
        report = integration_report.collect(list(self.runs.values()))
        self.assertEqual(report["run_count"], 3)
        self.assertEqual(report["polygon_complete_count"], 0)       # production polygons all incomplete; legacy ones do not count
        self.assertEqual(report["oriented_count"], 0); self.assertEqual(report["orientation_status"], "unresolved")
        by_name = {r["name"]: r for r in report["runs"]}
        self.assertIn("polygon", by_name["run_rolled"]["failure_reasons"])
        self.assertTrue(by_name["run_upright_ok"]["legacy_upright_diagnostic"]["ran"])
        self.assertEqual(by_name["run_upright_ok"]["legacy_upright_diagnostic"]["polygon_status"], "complete_polygon_hypothesis")
        self.assertFalse(by_name["run_rolled"]["legacy_upright_diagnostic"]["ran"])
        self.assertIn("none", report["claims"])
        markdown = integration_report.to_markdown(report)
        for needle in ("Named blockers", "ceiling_height_estimate", "orientation_unresolved", "legacy upright polygon"):
            self.assertIn(needle, markdown)

    def test_parent_folder_expands_to_path_runs_and_evaluation_is_a_separate_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"; parent.mkdir()
            for name in ("path_00", "path_01"):
                done = run_cli("from-prediction", "--prediction", fx()["preds"]["rolled"], "--output", parent / name)
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
            pred, _gt = integration_report.build_eval_documents(report["runs"], runs, self.scene, "h")
            self.assertNotIn(str(round(self.scene["height"], 6)), json.dumps(pred))
            self.assertTrue(all("height_axis_index" not in r["rooms"][0] for r in pred["runs"]))   # no independent axis: evaluator labels oracle

    def test_report_cli_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "report"
            args = [sys.executable, str(HERE / "integration_report.py"), "--runs", str(self.runs["rolled"]), "--output", str(prefix)]
            self.assertEqual(subprocess.run(args, capture_output=True, text=True).returncode, 0)
            self.assertNotEqual(subprocess.run(args, capture_output=True, text=True).returncode, 0)
            self.assertTrue(prefix.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
