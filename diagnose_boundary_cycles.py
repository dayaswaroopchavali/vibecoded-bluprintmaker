"""Truth-free audit of candidate cycles in frozen wall-boundary outputs.

This utility intentionally reports evidence and geometric rejection signals; it
does not rank room shapes or import evaluation/ground-truth data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import boundary_assembly as ba


def inspect(walls: dict) -> dict:
    p = ba.DEFAULTS
    lines, ignored = ba._prepare_lines(walls, p["min_line_frames"])
    extent = ba._extent_of(walls, lines)
    if extent <= 0 or len(lines) < 3:
        return {"status": "no_graph", "supported_lines": len(lines), "ignored": ignored}
    nodes = ba._build_nodes(lines, p["max_gap_fraction"] * extent)
    links = {}
    for a in range(len(nodes)):
        for b in range(len(nodes)):
            if a != b and nodes[a]["line"] != nodes[b]["line"]:
                q = ba._corner(nodes[a], nodes[b], lines, extent, p)
                if q is not None:
                    links[a, b] = q
    succ = {a: [b for b in range(len(nodes)) if (a, b) in links] for a in range(len(nodes))}
    cycles, exhausted = ba._find_cycles(nodes, succ, p["search_budget"], p["max_cycle_length"])
    results = []
    for order in cycles:
        vertices = np.asarray([links[order[k], order[(k + 1) % len(order)]]["vertex"] for k in range(len(order))])
        area = .5 * float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1]))
        simple = bool(ba._is_simple(vertices))
        cycle = {"line_order": [lines[nodes[i]["line"]]["id"] for i in order],
                 "vertices_xy": np.round(vertices, 5).tolist(), "signed_area": round(area, 6),
                 "simple": simple, "ccw": area > 0, "corner_gaps": [
                     {"from_line": lines[nodes[order[k]]["line"]]["id"], "to_line": lines[nodes[order[(k+1)%len(order)]]["line"]]["id"],
                      "outgoing_gap": round(float(links[order[k], order[(k+1)%len(order)]]["gap_u"]), 4),
                      "incoming_gap": round(float(links[order[k], order[(k+1)%len(order)]]["gap_v"]), 4),
                      "corner_sd": round(float(links[order[k], order[(k+1)%len(order)]]["sd"]), 5)}
                     for k in range(len(order))]}
        # Independent geometric trace of unused stretches. The production gate
        # also considers side-of-view evidence, reported below via assembler.
        used = {(nodes[i]["line"], q) for i in order for q in range(nodes[i]["i"], nodes[i]["j"] + 1)}
        unused = []
        for li, line in enumerate(lines):
            for qi, (a, b) in enumerate(line["spans"]):
                if (li, qi) in used:
                    continue
                sample = np.array([line["c"]*line["m"] + (a+(b-a)*w)*line["d"] for w in (0,.25,.5,.75,1)])
                outside = ~ba._inside(vertices, sample)
                unused.append({"line_id": line["id"], "span": qi,
                               "outside_sample_count": int(outside.sum()),
                               "max_outside_distance": round(float(np.max(np.min([
                                   ba._segment_distance(sample[outside], vertices[k], vertices[(k+1)%len(vertices)])
                                   for k in range(len(vertices))], axis=0))) if outside.any() else 0., 5)})
        cycle["unused_stretches"] = unused
        results.append(cycle)
    assembled = ba.assemble_boundary(walls)
    return {"status": "ok", "supported_lines": len(lines), "node_count": len(nodes),
            "corner_links": len(links), "cycle_count": len(cycles), "budget_exhausted": exhausted,
            "cycles": results, "production_rejection_counts": assembled.get("boundary_search", {}).get("rejected_by_reason", {}),
            "production_closest_problems": (assembled.get("incomplete_boundaries") or [{}])[0].get("closest_layout_problems", []),
            "ignored": ignored}


def run(root: Path) -> dict:
    frozen = root / "frozen"
    wall_files = sorted(frozen.glob("**/hyp*_walls.json"))
    out = {"result_root": str(root), "hypotheses": []}
    for wf in wall_files:
        try:
            walls = json.loads(wf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            out["hypotheses"].append({"file": str(wf), "error": str(e)})
            continue
        out["hypotheses"].append({"file": wf.name, "axis_world": walls.get("axis_world"), **inspect(walls)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=Path, help="baseline and/or refined output result roots")
    ap.add_argument("--output", type=Path, help="optional JSON report path")
    args = ap.parse_args()
    report = {r.name: run(r) for r in args.roots}
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
