"""Tests for boundary_viewer on schema-conforming synthetic payloads."""
import re
import tempfile
import unittest
from pathlib import Path

import boundary_viewer as V
from test_compare_wall_boundaries import boundary_for, rect, walls_for

STATUSES = ("observed", "extrapolated", "incomplete", "ambiguous", "unavailable")


def hyp(index, status="ran", boundary=None, walls=None, evaluation=None, **kw):
    return {"index": index, "axis_world": [0, 0, 1], "status": status, "walls": walls, "boundary": boundary,
            "evaluation": evaluation, "blockers": ["retained_capture_below_full"], **kw}


def payload(evaluation=None):
    poly = rect(2.0, 4 / 3)
    ok = hyp(0, boundary=boundary_for(poly), walls=walls_for(poly), evaluation=evaluation)
    amb = hyp(1, boundary={"status": "ambiguous", "vertices_xy": None, "edges": [], "alternatives": [{}, {}]})
    inc = hyp(2, boundary={"status": "incomplete", "vertices_xy": None, "edges": [], "incomplete_boundaries": [{"reason": "gap"}]})
    una = hyp(3, status="failed", error={"type": "RuntimeError", "message": "boom <b>"})
    return {"title": "t", "evaluation_supplied": evaluation is not None,
            "budgets": [{"label": "b1", "description": "d", "status": "ok", "retained_capture_fraction": 0.3, "hypotheses": [ok, amb, inc, una]},
                        {"label": "b2", "description": "d", "status": "failed_before_hypotheses", "retained_capture_fraction": None,
                         "error": {"type": "OSError", "message": "x"}, "hypotheses": []}]}


def scored_evaluation():
    import evaluate_floorplan
    poly = rect(2.0, 4 / 3)
    b = boundary_for(poly)
    for e in b["edges"]:
        if e["label"] == "extrapolated":
            e["label"] = "inferred"
    return evaluate_floorplan.evaluate_floorplan(b, {"width": 6.0, "length": 4.0, "height": 3.0},
                                                 {"status": "estimated_unscaled_separation", "separation_unscaled": 1.0},
                                                 manifest={"retained_capture_fraction": 0.3})


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def html(self, p):
        path = V.write_viewer(p, self.root / "v.html")
        return path.read_text(encoding="utf-8")

    def test_offline_no_network_references(self):
        text = self.html(payload())
        self.assertIsNone(re.search(r"https?://", text))
        self.assertIsNone(re.search(r"(src|href)\s*=\s*[\"'](?!#)", text))      # only in-page anchors
        self.assertNotIn("<link", text)
        self.assertIn("<svg", text)
        self.assertIn('name="viewport"', text)

    def test_all_status_labels_and_blockers_and_failures_shown(self):
        text = self.html(payload())
        for s in STATUSES:
            self.assertIn(f'chip {s}">{s}<', text)
        self.assertIn("retained_capture_below_full", text)
        self.assertIn("whole_room_claim blockers", text)
        self.assertIn("RuntimeError: boom &lt;b&gt;", text)                 # verbatim, escaped
        self.assertIn("no hypotheses ran", text)
        self.assertIn("incomplete boundaries", text)
        self.assertIn("stroke-dasharray", text)                              # extrapolated spans dashed
        self.assertIn("observed wall patch", text)
        self.assertIn("offset sd", text)

    def test_oracle_label_only_when_evaluation_supplied(self):
        without = self.html(payload())
        self.assertNotIn("oracle", without.lower())
        self.assertNotIn("layer-truth", without)
        (self.root / "v.html").unlink()
        ev = scored_evaluation()
        self.assertEqual(ev["status"], "scored")
        with_eval = self.html(payload(ev))
        self.assertIn("EVALUATOR-ORACLE", with_eval)
        self.assertIn('class="layer-truth"', with_eval)

    def test_scored_evaluation_without_matched_edges_renders(self):
        ev = scored_evaluation()
        ev["metrics"]["max_abs_wall_error_cm"] = None
        ev["metrics"]["max_rel_wall_error_percent"] = None
        self.assertIn("no matched edges", self.html(payload(ev)))

    def test_refuses_to_overwrite(self):
        V.write_viewer(payload(), self.root / "v.html")
        with self.assertRaises(FileExistsError):
            V.write_viewer(payload(), self.root / "v.html")

    def test_overlay_png_per_budget_new_folder(self):
        paths = V.write_overlay_pngs(payload(scored_evaluation()), self.root / "ov")
        self.assertEqual(sorted(paths), ["b1", "b2"])
        for p in paths.values():
            self.assertGreater(Path(p).stat().st_size, 500)
        with self.assertRaises(FileExistsError):
            V.write_overlay_pngs(payload(), self.root / "ov")

    def test_extrapolated_spans_complement_observed(self):
        g = V.geometry(payload()["budgets"][0]["hypotheses"][0], use_truth=False)
        e = g["edges"][0]
        total = sum(((a - b) ** 2).sum() ** 0.5 for a, b in e["obs"] + e["ext"])
        self.assertAlmostEqual(total, e["length"], places=6)
        self.assertEqual(len(e["obs"]), 1)
        self.assertEqual(len(e["ext"]), 2)

    def test_free_space_domain_label_and_diagnostic_spans_shown(self):
        p = payload()
        edge = p["budgets"][0]["hypotheses"][0]["boundary"]["edges"][0]
        edge["free_space"] = {"checked_spans": 2, "contradiction": False, "fraction_domains": ["wall_patch_area_within_supplied_height_band"],
                              "spans": [{"free_space_contradiction": False, "conflict_treated_as_diagnostic": True, "free_space_fraction": 0.5,
                                         "fraction_domain": "wall_patch_area_within_supplied_height_band"},
                                        {"free_space_contradiction": True, "free_space_fraction": 0.75}]}
        text = self.html(p)
        self.assertIn("wall_patch_area_within_supplied_height_band", text)
        self.assertIn("2 span(s) checked, 1 rejected", text)
        self.assertIn("opening-like (diagnostic only)", text)

    def test_overall_labels(self):
        labels = [V.overall_label(h) for h in payload()["budgets"][0]["hypotheses"]]
        self.assertEqual(labels, ["extrapolated", "ambiguous", "incomplete", "unavailable"])


if __name__ == "__main__":
    unittest.main()


