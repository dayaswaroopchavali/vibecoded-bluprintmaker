"""Fast CPU tests for rerun_chain_variants: no-overwrite, hash recording, freeze-before-evaluate ordering, per-run table, tie evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import rerun_chain_variants as rv
import run_rgb_batch as rb

SPEC = {"run_id": "house1_path0", "house_seed": 1, "path_index": 0, "path_seed": 5, "split": "calibration"}
BATCH = "fake-batch"


def make_source(outputs: Path, pinned=None):
    root = outputs / BATCH
    cfg = {"protocol": "upright", "plan": [SPEC], "gates": {"ceiling_height_error_cm": 1.5}, "split": {"calibration": [1], "held_out": []}, "settings": {},
           "code_sha256": pinned or {"boundary_assembly.py": "OLD", "axis_candidates.py": rb.sha256_file(rv.HERE / "axis_candidates.py")}}
    rb.write_new_json(root / "batch_config.json", cfg)
    rd = rb.run_dir_of(root, SPEC)
    (rd / "da3").mkdir(parents=True)
    (rd / "da3" / "prediction.npz").write_bytes(b"frozen-prediction")
    rb.write_new_json(rd / "freeze.json", {"files": {"da3/prediction.npz": rb.sha256_file(rd / "da3" / "prediction.npz")}, "code_sha256": {}})
    rb.write_new_json(rd / "generator_private" / "capture_qa.json", {"qa_pass": True, "checks": {"a": True}})
    rb.write_new_json(rd / "receipts" / "inference.json", {"status": "ok", "details": {"frames_used": 120}})
    rb.write_new_json(rd / "capture" / "private_ground_truth" / "scene.json", {"width": 5.0, "length": 6.0, "height": 2.7})
    return root


def fake_chain_hooks(log):
    def run_extraction(budgets, out_dir):
        log.append(("chain", budgets[0]["prediction"]))
        label = budgets[0]["label"]
        (out_dir / "frozen" / label).mkdir(parents=True)
        (out_dir / "frozen" / label / "planes.json").write_text("{}")
        index = {"budgets": [{"label": label, "status": "ok", "axis_status": "x", "source_indices": list(range(120)),
                              "candidates": [{"candidate_id": 0, "rank": 0, "score": 1.0, "viable": True,
                                              "co_supported_with_best": True, "files": {}, "polygon": {"boundary_status": "incomplete"}}]}]}
        (out_dir / "frozen_index.json").write_text(json.dumps(index))
        return index

    def write_reports(index, out_dir):
        (out_dir / "reports").mkdir()
        (out_dir / "reports" / "r.json").write_text("{}")
    return {"run_extraction": run_extraction, "write_reports": write_reports}


def eval_hooks(log):
    return {"read_frozen": lambda *a: {}, "eval_copy": lambda b: b, "evaluator_compatible": lambda h: h,
            "evaluate_floorplan": lambda *a: {"status": "failed", "metrics": None}}


class VariantRoot(unittest.TestCase):
    def test_refuses_overwrite_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "out"
            make_source(outputs)
            cur = {"boundary_assembly.py": "NEW", "axis_candidates.py": rb.sha256_file(rv.HERE / "axis_candidates.py")}
            root = Path(tmp) / "variant"
            info = rv.init_variant_root(root, batches=(BATCH,), outputs=outputs, current=cur)
            self.assertEqual(info["batches"][BATCH]["modules_differing_from_pinned"],
                             {"boundary_assembly.py": {"pinned_in_original_batch": "OLD", "current_working_copy": "NEW"}})
            saved = rb.read_json(root / "variant_config.json")
            self.assertEqual(saved["current_module_sha256"], cur)
            self.assertIn("POST-HOC", saved["label"])
            with self.assertRaises(FileExistsError):
                rv.init_variant_root(root, batches=(BATCH,), outputs=outputs, current=cur)

    def test_module_hashes_are_sha256_of_sources(self):
        h = rv.module_hashes()
        self.assertEqual(h["boundary_assembly.py"], rb.sha256_file(rv.HERE / "boundary_assembly.py"))
        self.assertTrue(all(len(v) == 64 for v in h.values()))

    def test_chain_reads_original_prediction_in_place_and_freeze_records_input_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "out"
            src = make_source(outputs)
            root = Path(tmp) / "variant"
            rv.init_variant_root(root, batches=(BATCH,), outputs=outputs, current={})
            log = []
            rv.stage_variant_chain(root, BATCH, SPEC, outputs, fake_chain_hooks(log))
            fr = rv.stage_variant_freeze(root, BATCH, SPEC, {"m": "h"})
            rd = rb.run_dir_of(root / BATCH, SPEC)
            self.assertEqual(log[0][1], str(rb.run_dir_of(src, SPEC) / "da3" / "prediction.npz"))
            self.assertFalse((rd / "da3").exists())                      # nothing copied
            self.assertEqual(fr["input_prediction_sha256"], rb.sha256_file(rb.run_dir_of(src, SPEC) / "da3" / "prediction.npz"))
            self.assertIn("inputs.json", fr["files"])
            self.assertTrue(any(k.startswith("axis/") for k in fr["files"]))
            rb.verify_freeze(rd)

    def test_changed_input_prediction_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "out"
            src = make_source(outputs)
            (rb.run_dir_of(src, SPEC) / "da3" / "prediction.npz").write_bytes(b"tampered")
            root = Path(tmp) / "variant"
            rv.init_variant_root(root, batches=(BATCH,), outputs=outputs, current={})
            with self.assertRaises(rb.CorruptRun):
                rv.stage_variant_chain(root, BATCH, SPEC, outputs, fake_chain_hooks([]))


class EvaluationOrdering(unittest.TestCase):
    def _frozen(self, tmp):
        outputs = Path(tmp) / "out"
        make_source(outputs)
        root = Path(tmp) / "variant"
        rv.init_variant_root(root, batches=(BATCH,), outputs=outputs, current={})
        rv.stage_variant_chain(root, BATCH, SPEC, outputs, fake_chain_hooks([]))
        return outputs, root

    def test_evaluate_refuses_before_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, root = self._frozen(tmp)
            with self.assertRaises(RuntimeError):
                rv.evaluate_variant(root, BATCH, outputs, eval_hooks([]))
            self.assertFalse(list((root / BATCH).glob("results_*")))      # no results, no truth read

    def test_freeze_verified_before_scene_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, root = self._frozen(tmp)
            rv.stage_variant_freeze(root, BATCH, SPEC, {})
            res = rv.evaluate_variant(root, BATCH, outputs, eval_hooks([]))
            kinds = [e[0] for e in res["order_events"]]
            self.assertLess(kinds.index("freeze_verified"), kinds.index("scene_read"))
            fi = rb.read_json(Path(res["results_dir"]) / "freeze_index_before_truth.json")
            self.assertIn("house1_path0", fi["runs"])

    def test_frozen_file_tampering_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs, root = self._frozen(tmp)
            rv.stage_variant_freeze(root, BATCH, SPEC, {})
            rd = rb.run_dir_of(root / BATCH, SPEC)
            (rd / "axis" / "frozen" / "house1_path0" / "planes.json").write_text("{\"x\": 1}")
            with self.assertRaises(rb.CorruptRun):
                rv.evaluate_variant(root, BATCH, outputs, eval_hooks([]))


def _row(run, cid, polygon, status, top, gate=False):
    return {"run_id": run, "candidate_id": cid, "polygon_status": polygon, "status": status, "truth_free_top_ranked": top,
            "gate_all": gate}


def _run(run, selected=None, run_pass=False, reasons=()):
    return {"run_id": run, "selected": selected, "run_pass": run_pass, "fail_reasons": list(reasons)}


class Tables(unittest.TestCase):
    def test_per_run_table_counts_and_totals(self):
        orows = [_row("r", 0, "incomplete", "failed", True), _row("r", 1, "incomplete", "failed", False)]
        nrows = [_row("r", 0, "complete_polygon_hypothesis", "scored", True, True), _row("r", 1, "complete_polygon_hypothesis", "failed", False)]
        sel = {"iou": 0.98, "max_abs_wall_error_cm": 1.0, "max_rel_wall_error_percent": 0.4, "gate_1_5cm": True, "gate_3pct": True,
               "gate_ceiling_height_1_5cm": True, "status": "scored"}
        t = rv.per_run_table([_run("r", reasons=["selected_candidate_failed"])], orows, [_run("r", sel, True)], nrows)
        self.assertEqual(t[0]["original"]["closed_polygons"], 0)
        self.assertEqual(t[0]["variant"]["closed_polygons"], 2)
        self.assertEqual(t[0]["variant"]["scoreable"], 1)
        self.assertEqual(t[0]["variant"]["top_candidate_status"], "scored")
        self.assertTrue(t[0]["variant"]["selected_gate_all"])
        self.assertFalse(t[0]["original"]["selected_gate_all"])
        self.assertEqual(rv.totals(t, "variant")["selected_gates_all_passed"], 1)
        self.assertEqual(rv.totals(t, "original")["runs_with_closed_polygon"], 0)
        self.assertIn("r |", rv.table_markdown(t))

    def test_tie_at_top_is_reported_as_tie(self):
        rows = [_row("r", 0, "incomplete", "failed", True), _row("r", 1, "incomplete", "failed", True)]
        t = rv.per_run_table([_run("r")], rows, [_run("r")], rows)
        self.assertEqual(t[0]["original"]["top_candidate_status"], "tie(2)")
        self.assertEqual(rv.totals(t, "original")["top_candidate_tie"], 1)


class TieEvidence(unittest.TestCase):
    def _cand(self, i, score, parallel, cap_lo=0.8, cap_hi=0.8):
        return {"id": i, "score": score, "viable": True, "kinds": ["k"],
                "evidence": {"normal_mass_parallel": parallel, "cap_pair": {"pair_found": True, "cap_support": cap_lo,
                                                                           "low": {"coverage": cap_lo}, "high": {"coverage": cap_hi},
                                                                           "camera_share_between_caps": 1.0}}}

    def test_separating_feature_and_argmax_without_threshold(self):
        c = [self._cand(0, 2.0, 0.40), self._cand(1, 2.0, 0.55), self._cand(2, 1.0, 0.9)]
        t = rv.tie_evidence("r", c)
        self.assertEqual(t["tied_candidate_ids"], [0, 1])
        self.assertTrue(t["features"]["normal_mass_parallel"]["separates"])
        self.assertEqual(t["features"]["normal_mass_parallel"]["argmax_candidate_ids"], [1])
        self.assertFalse(t["features"]["cap_support"]["separates"])        # identical values: not separating

    def test_no_tie_returns_none(self):
        self.assertIsNone(rv.tie_evidence("r", [self._cand(0, 2.0, 0.4), self._cand(1, 1.5, 0.5)]))


if __name__ == "__main__":
    unittest.main()
