"""Side-by-side table of two evaluated rgb batches (e.g. original v2 vs realistic v1). Reads only their results_NNN/ folders; evaluation-side only.

    python compare_rgb_batches.py --a <results_dir_A> --b <results_dir_B> --label-a original --label-b realistic --output <NEW dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLOSED = "complete_polygon_hypothesis"


def _load(d: Path):
    d = Path(d)
    return ({r["run_id"]: r for r in json.loads((d / "runs.json").read_text(encoding="utf-8"))},
            json.loads((d / "candidates.json").read_text(encoding="utf-8")))


def run_summary(run: dict | None, cands: list) -> dict:
    if run is None:
        return {"present": False}
    mine = [c for c in cands if c["run_id"] == run["run_id"]]
    closed = sum(1 for c in mine if c.get("polygon_status") == CLOSED)
    scored = sum(1 for c in mine if c.get("status") == "scored")
    sel = run.get("selected") or {}
    if run.get("selected_candidate_id") is None:
        top = "tie_unresolved"
    else:
        top = sel.get("status", "unknown")
    return {"present": True, "candidates": len(mine), "closed": closed, "scored": scored, "top": top, "iou": sel.get("iou"),
            "max_cm": sel.get("max_abs_wall_error_cm"), "max_pct": sel.get("max_rel_wall_error_percent"),
            "g15": sel.get("gate_1_5cm"), "g3": sel.get("gate_3pct"), "gceil": sel.get("gate_ceiling_height_1_5cm"),
            "cap_wall": (run.get("selected_scale") or {}).get("cap_to_wall_scale_ratio"), "qa": run.get("capture_qa_pass"),
            "pass": run.get("run_pass"), "any_gate_all": run.get("any_candidate_gate_all_oracle_info_only")}


def f(v, fmt=".3f"):
    return "-" if v is None else (format(v, fmt) if isinstance(v, float) else str(v))


def totals(rows: list) -> dict:
    rs = [r for r in rows if r["present"]]
    top = {}
    for r in rs:
        top[r["top"]] = top.get(r["top"], 0) + 1
    scored_top = [r for r in rs if r["top"] == "scored"]
    ious = [r["iou"] for r in scored_top if r["iou"] is not None]
    cm = [r["max_cm"] for r in scored_top if r["max_cm"] is not None]
    cw = [r["cap_wall"] for r in scored_top if r["cap_wall"] is not None]
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    return {"runs": len(rs), "candidates": sum(r["candidates"] for r in rs), "closed_polygons": sum(r["closed"] for r in rs),
            "scored_candidates": sum(r["scored"] for r in rs), "runs_with_any_closed": sum(1 for r in rs if r["closed"] > 0),
            "top_status": top, "runs_pass": sum(1 for r in rs if r["pass"]), "top_scored_runs": len(scored_top),
            "median_iou_top": med(ious), "median_max_cm_top": med(cm), "median_cap_wall_top": med(cw),
            "top_pass_1_5cm": sum(1 for r in scored_top if r["g15"]), "top_pass_3pct": sum(1 for r in scored_top if r["g3"]),
            "any_candidate_gate_all_runs": sum(1 for r in rs if r["any_gate_all"])}


def build(a_dir, b_dir, la, lb):
    ra, ca = _load(a_dir)
    rb, cb = _load(b_dir)
    ids = sorted(set(ra) | set(rb), key=lambda s: (int(s.split("_")[0][5:]), s))
    rows = []
    for i in ids:
        rows.append({"run_id": i, "house": (ra.get(i) or rb.get(i))["house_seed"], la: run_summary(ra.get(i), ca), lb: run_summary(rb.get(i), cb)})
    houses = {}
    for r in rows:
        houses.setdefault(r["house"], []).append(r)
    hsum = {h: {la: totals([x[la] for x in v]), lb: totals([x[lb] for x in v])} for h, v in houses.items()}
    return rows, hsum, {la: totals([r[la] for r in rows]), lb: totals([r[lb] for r in rows])}


def markdown(rows, hsum, tot, la, lb) -> str:
    def cells(s):
        if not s["present"]:
            return ["n/a"] * 8
        return [f"{s['closed']}/{s['candidates']}", str(s["scored"]), s["top"], f(s["iou"]), f(s["max_cm"], ".2f"),
                f(s["max_pct"], ".2f"), "/".join("-" if g is None else ("Y" if g else "N") for g in (s["g15"], s["g3"], s["gceil"])), f(s["cap_wall"], ".4f")]
    head = ["closed/cand", "scoreable", "top status", "IoU", "max cm", "max %", "gates 1.5/3%/ceil", "cap/wall"]
    out = [f"# {la} vs {lb} (same houses, split, path seeds, poses, DA3 settings, chain, gates)", "", "## Per run", "",
           "| run | " + " | ".join(f"{la} {h}" for h in head) + " | " + " | ".join(f"{lb} {h}" for h in head) + " |",
           "|---|" + "---|" * (2 * len(head))]
    for r in rows:
        out.append(f"| {r['run_id']} | " + " | ".join(cells(r[la])) + " | " + " | ".join(cells(r[lb])) + " |")
    out += ["", "## Per house", "", f"| house | {la} closed polygons | {la} scoreable | {la} top statuses | {la} pass | {lb} closed polygons | {lb} scoreable | {lb} top statuses | {lb} pass |",
            "|---|---|---|---|---|---|---|---|---|"]
    for h, s in sorted(hsum.items()):
        a, b = s[la], s[lb]
        out.append(f"| {h} | {a['closed_polygons']}/{a['candidates']} | {a['scored_candidates']} | {a['top_status']} | {a['runs_pass']}/{a['runs']} | "
                   f"{b['closed_polygons']}/{b['candidates']} | {b['scored_candidates']} | {b['top_status']} | {b['runs_pass']}/{b['runs']} |")
    out += ["", "## Overall", "", "```", json.dumps(tot, indent=1), "```", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    rows, hsum, tot = build(a.a, a.b, a.label_a, a.label_b)
    a.output.mkdir(parents=True)
    (a.output / "COMPARISON.md").write_text(markdown(rows, hsum, tot, a.label_a, a.label_b), encoding="utf-8")
    (a.output / "comparison.json").write_text(json.dumps({"runs": rows, "houses": hsum, "overall": tot}, indent=1), encoding="utf-8")
    print(json.dumps(tot, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
