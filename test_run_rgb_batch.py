"""CPU-only, fast tests for run_rgb_batch: split determinism, resume, freeze-before-truth ordering, tables, no overwrite."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import run_rgb_batch as rb

HERE = Path(__file__).resolve().parent


def _cfg_for(root: Path) -> dict:
    return rb.load_root(root)


def _fake_hooks(log, scene_reads=None):
    """Stub heavy stages: render writes a fake video + private scene; DA3 writes a fake prediction; chain writes a frozen index."""
    def make_capture_inputs(spec, protocol, settings):
        log.append(("make_inputs", spec["run_id"]))
        return {"width": 5.0, "length": 6.0, "height": 2.7}, [{"position": [0, 0, 1.4], "yaw": 0.0, "pitch": 0.0, "roll": 0.0}]

    def compute_capture_qa(scene, poses, settings):
        return {"qa_pass": True, "checks": {"angular_full_sphere": True}, "frames": 1}

    def render_capture(blender, scene_path, poses_path, capture_dir, spec, settings, logpath):
        log.append(("render", spec["run_id"]))
        name = rb.path_name(spec)
        (capture_dir / "videos").mkdir(parents=True)
        (capture_dir / "videos" / f"{name}.mp4").write_bytes(b"video-" + spec["run_id"].encode())
        (capture_dir / "rgb" / name).mkdir(parents=True)
        (capture_dir / "private_ground_truth").mkdir()
        (capture_dir / "private_ground_truth" / "scene.json").write_text(json.dumps({"width": 5.0, "length": 6.0, "height": 2.7}))

    def run_da3(cmd, logpath):
        log.append(("da3", cmd[cmd.index("--input") + 1]))
        out = Path(cmd[cmd.index("--output") + 1])
        out.mkdir(parents=True)
        (out / "prediction.npz").write_bytes(b"pred")
        (out / "manifest.json").write_text(json.dumps({"frames_used": 120, "source_indices": list(range(120)), "device": "cuda",
                                                       "resolution": 480, "elapsed_seconds": 1.5}))

    def run_extraction(budgets, out_dir):
        log.append(("chain", budgets[0]["label"]))
        label = budgets[0]["label"]
        (out_dir / "frozen" / label).mkdir(parents=True)
        (out_dir / "frozen" / label / "planes.json").write_text("{}")
        index = {"budgets": [{"label": label, "status": "ok", "axis_status": "ambiguous_axes", "source_indices": list(range(120)),
                              "candidates": [{"candidate_id": 0, "rank": 0, "score": 1.0, "viable": True, "co_supported_with_best": True, "files": {}}]}]}
        (out_dir / "frozen_index.json").write_text(json.dumps(index))
        return index

    def write_reports(index, out_dir):
        (out_dir / "reports").mkdir()
        (out_dir / "reports" / "r.json").write_text("{}")

    return {"make_capture_inputs": make_capture_inputs, "compute_capture_qa": compute_capture_qa, "render_capture": render_capture,
            "run_da3": run_da3, "run_extraction": run_extraction, "write_reports": write_reports, "code_hashes": lambda: {"x": "1"}}


class SplitAndPlanTests(unittest.TestCase):
    def test_split_is_deterministic_sorted_and_order_independent(self):
        a = rb.predeclared_split()
        b = rb.predeclared_split(tuple(reversed(rb.HOUSE_SEEDS)))
        self.assertEqual(a, b)
        self.assertEqual(a["calibration"], [17, 101, 102, 103, 104])
        self.assertEqual(a["held_out"], [105, 106, 107, 108, 109])
        self.assertFalse(set(a["calibration"]) & set(a["held_out"]))
        self.assertEqual(a["calibration"], sorted(a["calibration"]))

    def test_split_rejects_duplicates_and_wrong_count(self):
        with self.assertRaises(ValueError):
            rb.predeclared_split((1, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        with self.assertRaises(ValueError):
            rb.predeclared_split((1, 2, 3))

    def test_plan_is_house_major_with_independent_path_seeds(self):
        plan = rb.run_plan()
        self.assertEqual(len(plan), 30)
        self.assertEqual([p["house_seed"] for p in plan][:6], [17, 17, 17, 101, 101, 101])
        for h in rb.HOUSE_SEEDS:
            seeds = [p["path_seed"] for p in plan if p["house_seed"] == h]
            self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(plan, rb.run_plan())
        self.assertEqual({p["split"] for p in plan[:15]}, {"calibration"})
        self.assertEqual({p["split"] for p in plan[15:]}, {"held_out"})

    def test_the_ten_real_houses_are_distinct(self):
        import simulator
        scenes = {s: simulator.make_scene(s) for s in rb.HOUSE_SEEDS}
        self.assertTrue(rb.check_distinct_houses(scenes)["distinct"])
        clash = rb.check_distinct_houses({1: {"width": 5, "length": 6, "height": 2.7}, 2: {"width": 5.1, "length": 6.1, "height": 2.7}})
        self.assertFalse(clash["distinct"])

    def test_upright_forces_zero_roll_only(self):
        poses = [{"position": [0, 0, 1], "yaw": 0.1, "pitch": 0.2, "roll": 1.5}]
        up = rb.upright_poses(poses)
        self.assertEqual(up[0]["roll"], 0.0)
        self.assertEqual((up[0]["yaw"], up[0]["pitch"]), (0.1, 0.2))
        self.assertEqual(poses[0]["roll"], 1.5)

    def test_upright_poses_match_the_wide_path_apart_from_roll(self):
        spec = rb.run_plan()[0]
        scene, up = rb.make_capture_inputs(spec, "upright", rb.SETTINGS)
        _, rolled = rb.make_capture_inputs(spec, "rolling", rb.SETTINGS)
        self.assertEqual(len(up), 120)
        self.assertTrue(all(p["roll"] == 0.0 for p in up))
        self.assertTrue(any(abs(p["roll"]) > 0 for p in rolled))
        self.assertEqual([p["position"] for p in up], [p["position"] for p in rolled])

    def test_frozen_inference_settings_match_da3_full120(self):
        cmd = rb.da3_command(Path("v.mp4"), Path("out"), rb.SETTINGS)
        self.assertEqual(cmd[cmd.index("--frames") + 1], "120")
        self.assertEqual(cmd[cmd.index("--resolution") + 1], "480")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "2026")


class OverwriteTests(unittest.TestCase):
    def test_init_refuses_existing_folder_and_writes_predeclaration(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "batch"
            cfg = rb.init_root(root)
            self.assertTrue((root / "PREDECLARED.md").is_file())
            self.assertEqual(rb.read_json(root / "batch_config.json")["split"], cfg["split"])
            with self.assertRaises(FileExistsError):
                rb.init_root(root)

    def test_write_new_json_never_overwrites(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a" / "x.json"
            rb.write_new_json(p, {"a": 1})
            with self.assertRaises(FileExistsError):
                rb.write_new_json(p, {"a": 2})
            self.assertEqual(rb.read_json(p), {"a": 1})

    def test_changed_source_blocks_the_batch(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "b"
            rb.init_root(root)
            cfg = rb.load_root(root)
            rb.assert_same_code(cfg)
            cfg["code_sha256"]["boundary_assembly.py"] = "0" * 64
            with self.assertRaises(RuntimeError):
                rb.assert_same_code(cfg)


class ResumeTests(unittest.TestCase):
    def _setup(self, t):
        root = Path(t) / "b"
        rb.init_root(root)
        return root, rb.load_root(root)

    def test_run_then_resume_skips_finished_stages(self):
        with tempfile.TemporaryDirectory() as t:
            root, cfg = self._setup(t)
            spec = cfg["plan"][0]
            log, events = [], []
            st = rb.process_run(root, spec, cfg, _fake_hooks(log), events=events)
            self.assertEqual(st["state"], "ok")
            self.assertEqual([e for e in log if e[0] in ("render", "da3", "chain")], [("render", spec["run_id"]), ("da3", log[2][1]), ("chain", spec["run_id"])])
            self.assertTrue((rb.run_dir_of(root, spec) / "freeze.json").is_file())
            log2, events2 = [], []
            st2 = rb.process_run(root, spec, cfg, _fake_hooks(log2), events=events2)
            self.assertEqual(st2["state"], "ok")
            self.assertEqual(log2, [])                                   # nothing re-ran
            self.assertEqual({e[1] for e in events2 if e[0] == "skip"}, {"capture", "inference", "chain", "freeze"})

    def test_modified_output_is_detected_not_silently_redone(self):
        with tempfile.TemporaryDirectory() as t:
            root, cfg = self._setup(t)
            spec = cfg["plan"][0]
            rb.process_run(root, spec, cfg, _fake_hooks([]))
            (rb.run_dir_of(root, spec) / "da3" / "prediction.npz").write_bytes(b"tampered")
            with self.assertRaises(rb.CorruptRun):
                rb.process_run(root, spec, cfg, _fake_hooks([]))

    def test_interrupted_stage_keeps_partial_and_restarts_it_only(self):
        with tempfile.TemporaryDirectory() as t:
            root, cfg = self._setup(t)
            spec = cfg["plan"][0]
            hooks = _fake_hooks([])
            calls = {"n": 0}
            real = hooks["run_da3"]

            def flaky(cmd, logpath):
                calls["n"] += 1
                if calls["n"] == 1:
                    Path(cmd[cmd.index("--output") + 1]).mkdir(parents=True)
                    raise RuntimeError("gpu died")
                return real(cmd, logpath)
            hooks["run_da3"] = flaky
            st = rb.process_run(root, spec, cfg, hooks)
            self.assertEqual(st["state"], "failed")
            run_dir = rb.run_dir_of(root, spec)
            self.assertEqual(rb.receipt_state(run_dir, "capture"), "done")
            self.assertEqual(rb.receipt_state(run_dir, "inference"), "failed")
            st_no_retry = rb.process_run(root, spec, cfg, hooks)                  # a failed stage is not silently retried
            self.assertEqual(st_no_retry["state"], "failed")
            st2 = rb.process_run(root, spec, cfg, hooks, retry_failed=True)
            self.assertEqual(st2["state"], "ok")
            self.assertTrue((run_dir / "da3.incomplete-1").exists())            # partial kept, not deleted
            self.assertTrue(list((run_dir / "receipts").glob("inference.failed-*.json")))

    def test_batch_filters_budget_and_lock(self):
        with tempfile.TemporaryDirectory() as t:
            root, cfg = self._setup(t)
            log = []
            out = rb.run_batch(root, houses=[17], paths=[0, 1], hooks=_fake_hooks(log))
            self.assertEqual(out["attempted"], 2)
            self.assertEqual(out["completed_runs_total"], 2)
            self.assertFalse((root / ".gpu_serial.lock").exists())
            ticks = iter([0, 0, 99999, 99999])
            out2 = rb.run_batch(root, houses=[17], paths=[2], hooks=_fake_hooks(log), max_hours=1, clock=lambda: next(ticks))
            self.assertEqual(out2["attempted"], 1)
            ticks = iter([0, 7200, 7200, 7200])
            out3 = rb.run_batch(root, houses=[101], hooks=_fake_hooks(log), max_hours=1, clock=lambda: next(ticks))
            self.assertEqual(out3["attempted"], 0)
            self.assertIn("time budget", out3["stopped_for_budget"])
            (root / ".gpu_serial.lock").write_text("1")
            with self.assertRaises(RuntimeError):
                rb.run_batch(root, houses=[17], hooks=_fake_hooks(log))


class FreezeBeforeTruthTests(unittest.TestCase):
    def _function_source_nodes(self, name):
        tree = ast.parse((HERE / "run_rgb_batch.py").read_text(encoding="utf-8"))
        return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    def _tokens(self, fn):
        """Every identifier, attribute and string constant in the function body (docstring excluded)."""
        body = list(fn.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        out = []
        for stmt in body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name):
                    out.append(node.id)
                elif isinstance(node, ast.Attribute):
                    out.append(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    out.append(node.value)
                elif isinstance(node, ast.arg):
                    out.append(node.arg)
        return out

    def test_truth_free_functions_name_no_scene_truth_or_evaluator(self):
        for name in rb.TRUTH_FREE_FUNCTIONS:
            toks = " ".join(self._tokens(self._function_source_nodes(name))).lower()
            for bad in rb.FORBIDDEN_TOKENS:
                self.assertNotIn(bad, toks, f"{name} mentions {bad!r}")

    def test_no_module_level_import_of_evaluator_or_truth_modules(self):
        tree = ast.parse((HERE / "run_rgb_batch.py").read_text(encoding="utf-8"))
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name for n in top for a in n.names} | {n.module for n in top if isinstance(n, ast.ImportFrom) and n.module}
        for bad in ("evaluate_floorplan", "evaluation", "compare_wall_boundaries", "run_axis_candidates", "simulator", "capture_parallax"):
            self.assertNotIn(bad, names)
        # the truth-free chain gets the extraction function only through the hook table (lazy import)
        extraction = self._tokens(self._function_source_nodes("stage_chain"))
        self.assertNotIn("run_evaluation", extraction)

    def test_runtime_order_freeze_precedes_any_scene_read_and_stages_never_read_it(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "b"
            rb.init_root(root)
            cfg = rb.load_root(root)
            scene_reads = []
            hooks = _fake_hooks([])
            events = []
            for spec in cfg["plan"][:2]:
                rb.process_run(root, spec, cfg, hooks, events=events)
            self.assertFalse(scene_reads)
            self.assertFalse([e for e in events if e[0] in ("scene_read", "evaluate")])
            for spec in cfg["plan"][:2]:
                self.assertTrue((rb.run_dir_of(root, spec) / "freeze.json").is_file())
            eval_events = []
            seen = []

            def load_scene(path):
                seen.append(path)
                # at the instant truth is read every frozen run already carries its freeze file
                for spec in cfg["plan"][:2]:
                    assert (rb.run_dir_of(root, spec) / "freeze.json").is_file()
                assert (root / "results_000" / "freeze_index_before_truth.json").is_file()
                return {"width": 5.0, "length": 6.0, "height": 2.7}
            with self.assertRaises(RuntimeError):                         # 2 of 30 frozen: refuses unless partial is explicit
                rb.evaluate_batch(root, hooks={"load_scene": load_scene})
            self.assertFalse(seen)
            out = rb.evaluate_batch(root, allow_partial=True, hooks={"load_scene": load_scene})
            self.assertEqual(len(seen), 2)
            self.assertFalse(out["stage_1_complete"])
            order = [e[0] for e in out["order_events"]]
            self.assertLess(order.index("freeze_verified"), order.index("scene_read"))

    def test_evaluation_refuses_when_a_frozen_file_changed(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "b"
            rb.init_root(root)
            cfg = rb.load_root(root)
            spec = cfg["plan"][0]
            rb.process_run(root, spec, cfg, _fake_hooks([]))
            (rb.run_dir_of(root, spec) / "axis" / "frozen_index.json").write_text("{\"changed\": true}")
            seen = []
            with self.assertRaises(rb.CorruptRun):
                rb.evaluate_batch(root, allow_partial=True, hooks={"load_scene": lambda p: seen.append(p)})
            self.assertFalse(seen)                                         # truth never read after tampering


def _edge(i, pred, true):
    return {"true_edge": i, "pred_length_m": pred, "true_length_m": true, "abs_error_cm": 100 * abs(pred - true),
            "rel_error_percent": 100 * abs(pred - true) / true}


def _ev(edges, sep=10.0, ok1=True, ok3=True, status="scored"):
    return {"status": status, "failure_reasons": [], "scale": {"prediction_height": {"separation_unscaled": sep}},
            "metrics": {"iou": 0.99, "max_abs_wall_error_cm": max(e["abs_error_cm"] for e in edges),
                        "max_rel_wall_error_percent": max(e["rel_error_percent"] for e in edges), "edge_errors": edges},
            "custom_1_5cm_horizontal_diagnostic": {"pass": ok1}, "wall_length_3_percent_threshold_oracle_normalised": {"pass": ok3}}


GATES = rb.build_config()["gates"]


def _row(spec, cid, ev, top=True, ties=1, cos=True):
    cand = {"candidate_id": cid, "rank": cid, "co_supported_with_best": cos, "score": 1.0, "viable": True, "polygon": {"boundary_status": "complete_polygon_hypothesis"}}
    return rb.candidate_row(spec, cand, {"top": {cid: top}, "tie_count": ties}, ev, "complete", 3.0, GATES)


def _spec(house, path, split="calibration"):
    return {"run_id": rb.run_id(house, path), "house_seed": house, "path_index": path, "split": split}


QA_OK = {"qa_pass": True, "checks": {"a": True}}
DA3_OK = {"status": "ok", "frames_used": 120, "da3_elapsed_seconds": 50.0}
CHAIN = {"axis_status": "ambiguous_axes", "candidate_count": 3, "co_supported_count": 2, "candidate_cap": {"max_candidates": 24, "dropped_by_cap": 0}}


class TableTests(unittest.TestCase):
    def test_scale_diagnostic_ratio_and_height(self):
        ev = _ev([_edge(0, 4.0, 4.0), _edge(1, 4.04, 4.0)], sep=10.0)
        sc = rb.scale_diagnostic(ev, 3.0)
        self.assertAlmostEqual(sc["wall_to_cap_scale_ratio_median"], 1.005)
        self.assertAlmostEqual(sc["cap_to_wall_scale_ratio"], 1 / 1.005)
        self.assertAlmostEqual(sc["ceiling_height_wall_scaled_m"], 3.0 / 1.005)
        self.assertAlmostEqual(sc["ceiling_height_error_cm"], 100 * abs(3.0 / 1.005 - 3.0))
        self.assertAlmostEqual(sc["s_cap_units_per_m"], 10.0 / 3.0)

    def test_ceiling_height_gate_is_1_5_cm_and_other_gates_unchanged(self):
        spec = _spec(17, 0)
        good = _row(spec, 0, _ev([_edge(0, 4.0, 4.0), _edge(1, 3.0, 3.0)]))
        self.assertTrue(good["gate_all"])
        off = _row(spec, 0, _ev([_edge(0, 4.1, 4.0), _edge(1, 3.075, 3.0)]))              # r = 1.025 -> 7.3 cm
        self.assertFalse(off["gate_ceiling_height_1_5cm"])
        self.assertFalse(off["gate_all"])
        self.assertEqual(GATES["custom_1_5cm"], 1.5)
        self.assertEqual(GATES["wall_relative_percent"], 3.0)
        self.assertEqual(GATES["ceiling_height_error_cm"], 1.5)
        self.assertEqual(GATES["height_spread_cm"], 1.0)
        self.assertEqual(GATES["repeatability_abs_m"], 0.01)
        self.assertEqual(GATES["repeatability_rel_of_true_wall"], 0.005)
        self.assertFalse(GATES["changed"])

    def test_unscored_candidate_never_passes(self):
        row = _row(_spec(17, 0), 0, None)
        self.assertEqual(row["status"], "not_scored")
        self.assertFalse(row["gate_all"])

    def test_run_selection_is_truth_free_and_ties_fail(self):
        spec = _spec(17, 0)
        rows = [_row(spec, 0, _ev([_edge(0, 4.0, 4.0)]), top=True, ties=1), _row(spec, 1, _ev([_edge(0, 4.5, 4.0)], ok1=False, ok3=False), top=False, ties=1)]
        run = rb.run_level(spec, QA_OK, {}, DA3_OK, CHAIN, rows, 100.0)
        self.assertEqual(run["selected_candidate_id"], 0)
        self.assertTrue(run["run_pass"])
        tied = [_row(spec, 0, _ev([_edge(0, 4.0, 4.0)]), top=True, ties=2), _row(spec, 1, _ev([_edge(0, 4.0, 4.0)]), top=True, ties=2)]
        run2 = rb.run_level(spec, QA_OK, {}, DA3_OK, CHAIN, tied, 100.0)
        self.assertFalse(run2["run_pass"])
        self.assertIn("unresolved_axis_ambiguity", run2["fail_reasons"])
        wrong_top = [_row(spec, 0, _ev([_edge(0, 4.5, 4.0)], ok1=False, ok3=False), top=True), _row(spec, 1, _ev([_edge(0, 4.0, 4.0)]), top=False)]
        run3 = rb.run_level(spec, QA_OK, {}, DA3_OK, CHAIN, wrong_top, 100.0)
        self.assertFalse(run3["run_pass"])                                  # a passing non-selected candidate does not rescue the run
        self.assertTrue(run3["any_candidate_gate_all_oracle_info_only"])

    def test_qa_failure_and_da3_failure_are_reported_not_hidden(self):
        spec = _spec(17, 0)
        rows = [_row(spec, 0, _ev([_edge(0, 4.0, 4.0)]))]
        bad_qa = {"qa_pass": False, "checks": {"angular_full_sphere": False, "upright": True}}
        run = rb.run_level(spec, bad_qa, {}, DA3_OK, CHAIN, rows, 1.0)
        self.assertFalse(run["run_pass"])
        self.assertIn("capture_qa_failed", run["fail_reasons"])
        self.assertEqual(run["capture_qa_failed_checks"], ["angular_full_sphere"])
        run2 = rb.run_level(spec, QA_OK, {}, {"status": "failed"}, CHAIN, rows, 1.0)
        self.assertIn("da3_not_ok", run2["fail_reasons"])

    def _house(self, lengths_by_path, house=17):
        runs, sel = [], {}
        for p, lens in enumerate(lengths_by_path):
            spec = _spec(house, p)
            row = _row(spec, 0, _ev([_edge(i, v, t) for i, (v, t) in enumerate(zip(lens, (4.0, 3.0, 4.0, 3.0)))]))
            run = rb.run_level(spec, QA_OK, {}, DA3_OK, CHAIN, [row], 10.0)
            runs.append(run)
            sel[run["run_id"]] = row
        return runs, sel

    def test_repeatability_gates(self):
        same = [(4.0, 3.0, 4.0, 3.0)] * 3
        runs, sel = self._house(same)
        h = rb.house_level(17, runs, sel, GATES)
        self.assertTrue(h["house_pass"], h["fail_reasons"])
        self.assertEqual(h["repeatability"]["pair_count"], 12)
        jitter = [(4.0, 3.0, 4.0, 3.0), (4.005, 3.0, 4.0, 3.0), (4.0, 3.0, 4.0, 3.0)]     # 0.5 cm apart: within 1 cm and 0.5 % (2 cm of 4 m)
        h2 = rb.house_level(17, *self._house(jitter), GATES)
        self.assertTrue(h2["repeatability"]["gate_repeatability_1cm"])
        big = [(4.0, 3.0, 4.0, 3.0), (4.012, 3.0, 4.0, 3.0), (4.0, 3.0, 4.0, 3.0)]        # 1.2 cm apart: fails the 1 cm gate
        h3 = rb.house_level(17, *self._house(big), GATES)
        self.assertFalse(h3["repeatability"]["gate_repeatability_1cm"])
        self.assertFalse(h3["house_pass"])
        short = [(3.0, 3.0, 4.0, 3.0), (3.0, 3.0, 4.0, 3.0), (3.0, 3.0, 4.0, 3.0)]
        rel = rb.repeatability([_row(_spec(17, p), 0, _ev([_edge(0, 3.0 + 0.0 * p, 3.0)])) for p in range(3)], GATES)
        self.assertTrue(rel["gate_repeatability_0_5pct"])
        two = rb.house_level(17, self._house(same)[0][:2], self._house(same)[1], GATES)
        self.assertFalse(two["house_pass"])                                  # a house with 2 of 3 runs cannot pass
        self.assertIn("only_2_of_3_runs_present", two["fail_reasons"][0])

    def test_stage_1_complete_is_false_unless_all_30_pass(self):
        plan = rb.run_plan()
        runs, houses = [], []
        sel = {}
        for spec in plan:
            row = _row(spec, 0, _ev([_edge(i, v, v) for i, v in enumerate((4.0, 3.0, 4.0, 3.0))]))
            run = rb.run_level(spec, QA_OK, {}, DA3_OK, CHAIN, [row], 10.0)
            runs.append(run)
            sel[run["run_id"]] = row
        houses = [rb.house_level(h, runs, sel, GATES) for h in sorted(rb.HOUSE_SEEDS)]
        done = rb.stage_one(plan, runs, houses, True)
        self.assertTrue(done["stage_1_complete"])
        self.assertEqual(done["runs_pass"], 30)
        self.assertFalse(rb.stage_one(plan, runs[:29], houses, True)["stage_1_complete"])          # a missing run
        self.assertFalse(rb.stage_one(plan, runs, houses, False)["stage_1_complete"])              # inconsistent code
        bad = [dict(r) for r in runs]
        bad[7]["run_pass"] = False
        bad_houses = [rb.house_level(h, bad, sel, GATES) for h in sorted(rb.HOUSE_SEEDS)]
        res = rb.stage_one(plan, bad, bad_houses, True)
        self.assertFalse(res["stage_1_complete"])
        self.assertEqual(res["runs_pass"], 29)
        self.assertEqual(res["houses_pass"], 9)

    def test_markdown_tables_render_with_missing_values(self):
        spec = _spec(17, 0)
        rows = [_row(spec, 0, None), _row(spec, 1, _ev([_edge(0, 4.0, 4.0)]), top=False)]
        run = rb.run_level(spec, QA_OK, {}, DA3_OK, CHAIN, rows, 5.0)
        self.assertIn("house17_path0", rb.runs_markdown([run]))
        self.assertIn("not_scored", rb.candidates_markdown(rows))
        self.assertIn("17", rb.houses_markdown([rb.house_level(17, [run], {}, GATES)]))


if __name__ == "__main__":
    unittest.main()
