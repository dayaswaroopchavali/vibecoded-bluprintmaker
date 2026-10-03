"""Offline HTML + PNG viewer for wall-boundary comparisons (task 0012, worker C). No network, no CDN: inline SVG and JS.

Per budget and per axis hypothesis it shows observed wall patches (A's observed_spans), the boundary edges with observed
spans solid and extrapolated spans dashed in a different colour, the final polygon, per-edge uncertainty bars, status labels
(observed / extrapolated / incomplete / ambiguous / unavailable) and the provenance / whole_room_claim blockers.

The true outline is drawn ONLY when a frozen evaluation is supplied, in the evaluator's aligned 3.0 m frame, and is always
labelled EVALUATOR-ORACLE (the alignment, mirroring and edge matching were chosen with the truth in hand). Without an
evaluation nothing here mentions or draws truth.

  python boundary_viewer.py --comparison <wall-boundaries folder> [--output-html <NEW file>] [--overlay-dir <NEW dir>]
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np

STATUS_LABELS = {
    "observed": "edge or polygon fully supported by observed wall spans",
    "extrapolated": "polygon contains edge spans extended beyond observed support (dashed)",
    "incomplete": "boundary could not be closed; reasons listed",
    "ambiguous": "competing layouts; none preferred",
    "unavailable": "wall extraction or assembly failed / insufficient data (error shown verbatim)",
}
COL = {"patch": "#0b7a75", "obs": "#0b3d91", "ext": "#e07b00", "poly": "#222222", "truth": "#000000", "unc": "#b3006e"}
BGR = {k: tuple(int(v[i:i + 2], 16) for i in (5, 3, 1)) for k, v in COL.items()}


# ---------------------------------------------------------------------------------------------------- payload
def overall_label(h: dict) -> str:
    b = h.get("boundary")
    if h.get("status") in ("failed", "unavailable", "skipped_not_co_supported") or b is None:
        return "unavailable"
    st = b.get("status")
    if st == "ambiguous":
        return "ambiguous"
    if st != "complete_polygon_hypothesis":
        return "incomplete"
    return "extrapolated" if any(e.get("label") == "extrapolated" for e in b.get("edges", []) or []) else "observed"


def _blockers(h: dict, retained) -> list:
    out = ["whole_room_claim_never_allowed_for_this_comparison"]
    out.append("capture_provenance_missing" if retained is None else "retained_capture_below_full" if retained < 1.0 else "")
    out = [x for x in out if x]
    if (h.get("boundary") or {}).get("status") != "complete_polygon_hypothesis":
        out.append("boundary_not_complete")
    return out


def payload_from_comparison(folder: Path) -> dict:
    """Rebuild the viewer payload from a comparison folder (frozen files and, when present, evaluation files)."""
    folder = Path(folder)
    cmp = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    from compare_wall_boundaries import provenance_blockers, read_frozen
    budgets, any_eval = [], False
    for rec in cmp["frozen_index"]["budgets"]:
        b = {"label": rec["label"], "description": rec.get("description"), "status": rec.get("status"),
             "retained_capture_fraction": rec.get("retained_capture_fraction"), "error": rec.get("error"), "hypotheses": []}
        for h in rec.get("hypotheses", []):
            item = {"index": h["hypothesis_index"], "axis_world": h.get("axis_world"), "status": h["status"], "error": h.get("error"),
                    "walls": None, "boundary": None, "evaluation": None, "reasons": h.get("walls_reasons")}
            if "files" in h:
                item["walls"] = read_frozen(folder, h["files"]["walls"]["path"], h["files"]["walls"]["sha256"])
                if "boundary" in h["files"]:
                    item["boundary"] = read_frozen(folder, h["files"]["boundary"]["path"], h["files"]["boundary"]["sha256"])
            ev_path = folder / "evaluation" / f"{rec['label']}_hyp{h['hypothesis_index']}.json"
            if ev_path.exists():
                item["evaluation"] = json.loads(ev_path.read_text(encoding="utf-8")).get("evaluation")
                any_eval = any_eval or item["evaluation"] is not None
            item["blockers"] = provenance_blockers(rec.get("retained_capture_fraction"), (item["boundary"] or {}).get("status"))
            b["hypotheses"].append(item)
        budgets.append(b)
    return {"title": "Wall-boundary comparison", "evaluation_supplied": any_eval, "budgets": budgets}


# ---------------------------------------------------------------------------------------------------- geometry
def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def oracle_transform(evaluation):
    """Callable mapping common-frame points into the evaluator's aligned 3.0 m frame, or None (not scored)."""
    if not evaluation or evaluation.get("status") != "scored":
        return None
    s = evaluation["scale"]["prediction_scale"]
    a = evaluation["alignment"]
    th = math.radians(a["rotation_deg"])
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.asarray(a["translation_m"], float)
    flip = np.array([-1.0, 1.0]) if a["mirrored"] else np.array([1.0, 1.0])
    return lambda p: (np.asarray(p, float) * s * flip) @ rot.T + t, s


def _intervals_minus(lo, hi, spans):
    spans = sorted((max(lo, a), min(hi, b)) for a, b in spans if min(hi, b) > max(lo, a))
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps, cur = [], lo
    for a, b in merged:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        gaps.append((cur, hi))
    return [tuple(m) for m in merged], gaps


def geometry(h: dict, use_truth: bool = True) -> dict:
    """Everything to draw for one hypothesis, in one frame (oracle-aligned 3 m frame when a scored evaluation is supplied
    and use_truth, else the common frame)."""
    tf = oracle_transform(h.get("evaluation")) if use_truth else None
    f, s = (tf if tf else (lambda p: np.asarray(p, float), 1.0))
    walls = h.get("walls") or {}
    lines = {ln["id"]: ln for ln in walls.get("lines", []) or []}
    out = {"frame": "oracle_aligned_3m" if tf else "common_arbitrary_units", "scale": s, "patches": [], "edges": [], "polygon": None, "truth": None}
    for ln in lines.values():
        m, d, off = np.asarray(ln["normal_xy"], float), _unit(ln["direction_xy"]), float(ln["offset"])
        for t0, t1 in ln.get("observed_spans", []) or []:
            out["patches"].append((f(off * m + t0 * d), f(off * m + t1 * d), ln["id"]))
    b = h.get("boundary") or {}
    for e in b.get("edges", []) or []:
        p0, p1 = np.asarray(e["start_xy"], float), np.asarray(e["end_xy"], float)
        ln = lines.get(e.get("line_id"))
        obs, ext = [], []
        if ln is not None:
            d = _unit(ln["direction_xy"])
            ts, te = float(d @ p0), float(d @ p1)
            lo, hi = min(ts, te), max(ts, te)
            spans = ([(sorted((float(d @ np.asarray(a, float)), float(d @ np.asarray(c, float)))))
                      for a, c in e["observed_spans_xy"]] if e.get("observed_spans_xy") else ln.get("observed_spans", []) or [])
            o, g = _intervals_minus(lo, hi, spans)
            at = lambda t: p0 + (t - ts) / (te - ts) * (p1 - p0) if abs(te - ts) > 1e-12 else p0
            obs = [(f(at(a)), f(at(c))) for a, c in o]
            ext = [(f(at(a)), f(at(c))) for a, c in g]
        elif e.get("label") == "observed":
            obs = [(f(p0), f(p1))]
        else:
            ext = [(f(p0), f(p1))]
        sd = ((ln or {}).get("uncertainty") or {}).get("offset_sd")
        if sd is None and isinstance(e.get("uncertainty"), dict):
            sd = e["uncertainty"].get("offset_sd")
        mid = f((p0 + p1) / 2)
        normal = _unit(np.asarray(f(p1)) - np.asarray(f(p0)))[::-1] * np.array([-1.0, 1.0])
        out["edges"].append({"index": e.get("index"), "label": e.get("label"), "obs": obs, "ext": ext, "mid": mid, "normal": normal,
                             "sd": None if sd is None else float(sd) * s, "observed_fraction": e.get("observed_fraction"),
                             "length": float(e.get("length", np.linalg.norm(p1 - p0))) * s, "line_id": e.get("line_id"),
                             "frames": e.get("supporting_frames")})
    if b.get("vertices_xy") is not None:
        out["polygon"] = np.array([f(v) for v in b["vertices_xy"]])
    ev = h.get("evaluation")
    if tf and use_truth:
        out["truth"] = np.asarray(ev["true_outline_normalised_m"], float)
    return out


def _bounds(g):
    pts = [p for a, b, _ in g["patches"] for p in (a, b)]
    for e in g["edges"]:
        pts += [p for seg in e["obs"] + e["ext"] for p in seg]
    for k in ("polygon", "truth"):
        if g[k] is not None:
            pts += list(g[k])
    if not pts:
        return np.zeros(2), np.ones(2)
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.06 * max(float(np.max(hi - lo)), 1e-9)
    return lo - pad, hi + pad


def _num(v):
    return "n/a (no matched edges)" if v is None else f"{v:.1f}"


def _fmt_len(v, g):
    return f"{100 * v:.1f} cm" if g["frame"] == "oracle_aligned_3m" else f"{v:.4g} u"


# ---------------------------------------------------------------------------------------------------- SVG / HTML
def svg_for(h: dict, use_truth: bool = True) -> str:
    g = geometry(h, use_truth)
    lo, hi = _bounds(g)
    size = max(float(np.max(hi - lo)), 1e-9)
    w = size / 190
    X = lambda p: (float(p[0] - lo[0]), float(hi[1] - p[1]))               # y up in the plan
    q = lambda p: "%.4f,%.4f" % X(p)
    out = [f'<svg viewBox="0 0 {hi[0] - lo[0]:.4f} {hi[1] - lo[1]:.4f}" role="img" aria-label="plan view">']
    out.append('<g class="layer-patches" stroke="%s" stroke-width="%.4f" stroke-linecap="round" opacity="0.55">' % (COL["patch"], w * 4))
    out += [f'<line x1="{X(a)[0]:.4f}" y1="{X(a)[1]:.4f}" x2="{X(b)[0]:.4f}" y2="{X(b)[1]:.4f}"><title>observed wall patch (line {lid})</title></line>' for a, b, lid in g["patches"]]
    out.append("</g>")
    if g["polygon"] is not None:
        out.append(f'<polygon class="layer-polygon" points="{" ".join(q(p) for p in g["polygon"])}" fill="rgba(34,34,34,0.05)" stroke="{COL["poly"]}" stroke-width="{w * 1.2:.4f}"/>')
    out.append('<g class="layer-edges">')
    for e in g["edges"]:
        for a, b in e["obs"]:
            out.append(f'<line x1="{X(a)[0]:.4f}" y1="{X(a)[1]:.4f}" x2="{X(b)[0]:.4f}" y2="{X(b)[1]:.4f}" stroke="{COL["obs"]}" stroke-width="{w * 2.6:.4f}"><title>edge {e["index"]} observed span</title></line>')
        for a, b in e["ext"]:
            out.append(f'<line x1="{X(a)[0]:.4f}" y1="{X(a)[1]:.4f}" x2="{X(b)[0]:.4f}" y2="{X(b)[1]:.4f}" stroke="{COL["ext"]}" stroke-width="{w * 2.6:.4f}" stroke-dasharray="{w * 7:.4f} {w * 4:.4f}"><title>edge {e["index"]} extrapolated span</title></line>')
    out.append("</g>")
    out.append(f'<g class="layer-unc" stroke="{COL["unc"]}" stroke-width="{w * 1.4:.4f}">')
    for e in g["edges"]:
        if e["sd"] is not None:
            r = max(e["sd"], w * 0.8)
            a, b = e["mid"] - r * e["normal"], e["mid"] + r * e["normal"]
            out.append(f'<line x1="{X(a)[0]:.4f}" y1="{X(a)[1]:.4f}" x2="{X(b)[0]:.4f}" y2="{X(b)[1]:.4f}"><title>edge {e["index"]} offset sd {_fmt_len(e["sd"], g)}</title></line>')
    out.append("</g>")
    if g["truth"] is not None:
        out.append(f'<polygon class="layer-truth" points="{" ".join(q(p) for p in g["truth"])}" fill="none" stroke="{COL["truth"]}" stroke-width="{w * 1.6:.4f}" stroke-dasharray="{w * 2:.4f} {w * 2:.4f}"><title>true outline (EVALUATOR-ORACLE)</title></polygon>')
    out.append("</svg>")
    return "".join(out)


def _bar(e, g, extent):
    if e["sd"] is None:
        return "n/a"
    pct = min(100.0, 100.0 * e["sd"] / max(0.03 * extent, 1e-12))
    return f'<span class="bar"><i style="width:{pct:.1f}%"></i></span> {html.escape(_fmt_len(e["sd"], g))}'


def _card(budget: dict, h: dict, use_truth: bool) -> str:
    g = geometry(h, use_truth)
    lab = overall_label(h)
    b = h.get("boundary") or {}
    ev = h.get("evaluation") if use_truth else None
    rows = []
    lo, hi = _bounds(g)
    extent = float(np.max(hi - lo))
    for e in g["edges"]:
        rows.append(f'<tr><td>{e["index"]}</td><td><span class="chip {html.escape(str(e["label"]))}">{html.escape(str(e["label"]))}</span></td>'
                    f'<td>{html.escape(_fmt_len(e["length"], g))}</td><td>{"-" if e["observed_fraction"] is None else format(float(e["observed_fraction"]), ".2f")}</td>'
                    f'<td>{_bar(e, g, extent)}</td></tr>')
    notes = []
    if h.get("error"):
        notes.append(f'<p class="err">error (verbatim): {html.escape(h["error"].get("type", ""))}: {html.escape(str(h["error"].get("message")))}</p>')
    if h.get("status") == "unavailable":
        notes.append(f'<p class="err">unavailable: {html.escape(json.dumps(h.get("reasons")))}</p>')
    if b.get("incomplete_boundaries"):
        notes.append(f'<p class="err">incomplete boundaries: {html.escape(json.dumps(b["incomplete_boundaries"]))}</p>')
    if b.get("status") == "ambiguous":
        notes.append(f'<p class="err">ambiguous: {len(b.get("alternatives") or [])} competing layouts, none preferred</p>')
    if ev is not None:
        if ev.get("status") == "scored":
            m = ev["metrics"]
            notes.append(f'<p class="evalnote">EVALUATOR-ORACLE score: IoU {m["iou"]:.3f}, max wall error {_num(m["max_abs_wall_error_cm"])} cm / {_num(m["max_rel_wall_error_percent"])} % (matched edges only), '
                         f'unmatched true {m["unmatched_true_edges"]} / predicted {m["unmatched_predicted_edges"]}; mirrored {ev["alignment"]["mirrored"]}. '
                         f'Alignment, mirroring and edge matching were chosen with the truth.</p>')
        else:
            notes.append(f'<p class="evalnote">EVALUATOR-ORACLE evaluation not scored: {html.escape("; ".join(map(str, ev.get("failure_reasons", []))))}</p>')
    blockers = (ev or {}).get("whole_room_claim", {}).get("blockers") if ev else h.get("blockers", _blockers(h, budget.get("retained_capture_fraction")))
    frame_note = "plan frame: evaluator-aligned, 3.0 m ceiling normalised" if g["frame"] == "oracle_aligned_3m" else "plan frame: reconstruction frame, arbitrary units (not metric)"
    return (f'<article class="card" id="{html.escape(budget["label"])}-h{h["index"]}"><h3>hypothesis {h["index"]} <span class="chip {lab}">{lab}</span></h3>'
            f'<p class="sub">axis_world {html.escape(json.dumps([round(x, 3) for x in h["axis_world"]]) if h.get("axis_world") else "n/a")}; {frame_note}</p>'
            f'<div class="plot">{svg_for(h, use_truth)}</div>'
            f'<table><thead><tr><th>edge</th><th>label</th><th>length</th><th>observed fraction</th><th>offset uncertainty</th></tr></thead><tbody>{"".join(rows) or "<tr><td colspan=5>no edges</td></tr>"}</tbody></table>'
            f'{"".join(notes)}<p class="blockers">whole_room_claim blockers: {html.escape(", ".join(blockers))}</p></article>')


def write_viewer(payload: dict, out_path: Path) -> Path:
    """One offline HTML file. Refuses to overwrite."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"output exists: {out_path}")
    use_truth = bool(payload.get("evaluation_supplied"))
    legend = "".join(f'<li><span class="chip {k}">{k}</span> {html.escape(v)}</li>' for k, v in STATUS_LABELS.items())
    legend += (f'<li><span class="sw" style="background:{COL["patch"]}"></span> observed wall patch (A observed_spans)</li>'
               f'<li><span class="sw" style="background:{COL["obs"]}"></span> boundary span with observed support (solid)</li>'
               f'<li><span class="sw dash" style="border-color:{COL["ext"]}"></span> extrapolated boundary span (dashed)</li>'
               f'<li><span class="sw" style="background:{COL["unc"]}"></span> per-edge offset uncertainty bar (leave-view-out / bootstrap sd)</li>')
    toggles = ["patches", "polygon", "edges", "unc"]
    if use_truth:
        legend += '<li><span class="sw dash" style="border-color:#000"></span> true outline (EVALUATOR-ORACLE: drawn only because a frozen evaluation was supplied)</li>'
        toggles.append("truth")
    boxes = "".join(f'<label><input type="checkbox" checked data-layer="{t}"> {t}</label> ' for t in toggles)
    sections = []
    for b in payload.get("budgets", []):
        frac = b.get("retained_capture_fraction")
        head = (f'<section id="{html.escape(b["label"])}"><h2>{html.escape(b["label"])}</h2><p class="sub">{html.escape(str(b.get("description") or ""))}; '
                f'retained capture fraction {"n/a" if frac is None else format(frac, ".3f")}</p>')
        if b.get("status") not in (None, "ok"):
            head += f'<p class="err">budget {html.escape(str(b["status"]))} (error verbatim): {html.escape(json.dumps(b.get("error")))}</p>'
        cards = "".join(_card(b, h, use_truth) for h in b.get("hypotheses", []))
        sections.append(head + f'<div class="grid">{cards or "<p class=err>no hypotheses ran for this budget</p>"}</div></section>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(payload.get("title", "Wall-boundary comparison"))}</title>
<style>
:root{{--bg:#fafafa;--fg:#1b1b1b;--card:#fff;--line:#d8d8d8}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--card:#20232a;--line:#3a3f48}}}}
body{{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}}
h1{{font-size:20px}} h2{{font-size:17px;margin:22px 0 2px}} h3{{font-size:15px;margin:0 0 4px}}
.sub{{opacity:.75;margin:2px 0 8px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto}}
.plot{{background:#fff;border:1px solid var(--line);border-radius:6px}} svg{{width:100%;height:auto;display:block;max-height:420px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-top:8px}} td,th{{border-bottom:1px solid var(--line);padding:3px 4px;text-align:left}}
.chip{{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;border:1px solid var(--line)}}
.chip.observed{{background:#cfe3ff;color:#0b3d91}} .chip.extrapolated{{background:#ffe3c2;color:#8a4b00}} .chip.incomplete{{background:#eee;color:#333}}
.chip.ambiguous{{background:#f6d4ea;color:#7a0050}} .chip.unavailable{{background:#ddd;color:#111;text-decoration:line-through}}
.bar{{display:inline-block;width:60px;height:7px;background:var(--line);vertical-align:middle}} .bar i{{display:block;height:100%;background:{COL["unc"]}}}
.err{{color:#b00020;font-size:12px;word-break:break-word}} .evalnote{{font-size:12px;border-left:3px solid #000;padding-left:6px}} .blockers{{font-size:12px;opacity:.85}}
.sw{{display:inline-block;width:22px;height:0;border-top:4px solid transparent;vertical-align:middle}} .sw[style*=background]{{height:4px;border:0}} .sw.dash{{border-top:3px dashed}}
ul.legend{{list-style:none;padding:0;columns:300px}} nav a{{margin-right:10px}}
</style></head><body>
<h1>{html.escape(payload.get("title", "Wall-boundary comparison"))}</h1>
<p class="sub">Offline page (no network). Frozen reconstruction results; not a benchmark-stage completion claim.
{"A true outline and scores are shown because a frozen evaluation was supplied." if use_truth else "No evaluation supplied: no truth is drawn or referenced."}</p>
<nav>{"".join(f'<a href="#{html.escape(b["label"])}">{html.escape(b["label"])}</a>' for b in payload.get("budgets", []))}</nav>
<p>{boxes}</p><ul class="legend">{legend}</ul>
{"".join(sections)}
<script>
document.querySelectorAll('input[data-layer]').forEach(function(cb){{cb.addEventListener('change',function(){{
document.querySelectorAll('.layer-'+cb.dataset.layer).forEach(function(el){{el.style.display=cb.checked?'':'none';}});}});}});
</script></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "x", encoding="utf-8") as stream:
        stream.write(page)
    return out_path


# ---------------------------------------------------------------------------------------------------- PNG
def _dashed(img, a, b, color, thick):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    for s0 in np.arange(0, length, 14):
        s1 = min(s0 + 8, length)
        p0, p1 = a + (b - a) * s0 / max(length, 1e-9), a + (b - a) * s1 / max(length, 1e-9)
        cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color, thick, cv2.LINE_AA)


def _panel(budget, h, use_truth, size=520):
    g = geometry(h, use_truth)
    img = np.full((size + 70, size, 3), 255, np.uint8)
    lo, hi = _bounds(g)
    ppm = (size - 20) / max(float(np.max(hi - lo)), 1e-9)
    P = lambda p: (int(round(10 + (p[0] - lo[0]) * ppm)), int(round(10 + (hi[1] - p[1]) * ppm)))
    for a, b, _ in g["patches"]:
        cv2.line(img, P(a), P(b), BGR["patch"], 6, cv2.LINE_AA)
    if g["polygon"] is not None:
        cv2.polylines(img, [np.array([P(p) for p in g["polygon"]])], True, BGR["poly"], 1, cv2.LINE_AA)
    for e in g["edges"]:
        for a, b in e["obs"]:
            cv2.line(img, P(a), P(b), BGR["obs"], 3, cv2.LINE_AA)
        for a, b in e["ext"]:
            _dashed(img, P(a), P(b), BGR["ext"], 3)
        if e["sd"] is not None:
            r = max(e["sd"], 2 / ppm)
            cv2.line(img, P(e["mid"] - r * e["normal"]), P(e["mid"] + r * e["normal"]), BGR["unc"], 2, cv2.LINE_AA)
    if g["truth"] is not None:
        pts = [P(p) for p in g["truth"]]
        for i in range(len(pts)):
            _dashed(img, pts[i], pts[(i + 1) % len(pts)], BGR["truth"], 2)
    lab = overall_label(h)
    lines = [f"{budget['label']} hyp{h['index']}: {lab}", "solid blue observed / dashed orange extrapolated / teal patches"]
    if g["truth"] is not None:
        lines.append("black dashed: true outline (EVALUATOR-ORACLE)")
    if h.get("error"):
        lines.append(f"error: {h['error'].get('type')}: {str(h['error'].get('message'))[:60]}")
    for i, text in enumerate(lines[:4]):
        cv2.putText(img, text, (6, size + 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
    return img


def write_overlay_pngs(payload: dict, out_dir: Path) -> dict:
    """One PNG per budget (a panel per hypothesis) in a NEW folder."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output folder already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    use_truth = bool(payload.get("evaluation_supplied"))
    paths = {}
    for b in payload.get("budgets", []):
        panels = [_panel(b, h, use_truth) for h in b.get("hypotheses", [])]
        if not panels:
            panels = [np.full((120, 520, 3), 255, np.uint8)]
            cv2.putText(panels[0], f"{b['label']}: no hypotheses ran ({b.get('status')})", (6, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 160), 1, cv2.LINE_AA)
        height = max(p.shape[0] for p in panels)
        panels = [np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in panels]
        path = out_dir / f"{b['label']}.png"
        cv2.imwrite(str(path), np.hstack(panels))
        paths[b["label"]] = str(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    args = parser.parse_args()
    if args.output_html.exists() or (args.overlay_dir and args.overlay_dir.exists()):
        parser.error("output exists; pass NEW paths")
    payload = payload_from_comparison(args.comparison)
    write_viewer(payload, args.output_html)
    if args.overlay_dir:
        write_overlay_pngs(payload, args.overlay_dir)
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


