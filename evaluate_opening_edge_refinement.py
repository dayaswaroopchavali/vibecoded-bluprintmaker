"""Freeze and diagnose public-only opening edge refinements on development runs.

All refinements and their hashes are written before any private scene JSON is
opened. Evaluation follows the existing batch scorer's frozen prediction API.
This is a development diagnostic; it cannot establish held-out performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import refine_opening_edges_public as edge


HERE = Path(__file__).resolve().parent
TRIMS = (0.0, .025, .05, .10)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze(freeze_path: Path, public_root: Path, out_dir: Path) -> dict:
    source = json.loads(freeze_path.read_text(encoding="utf-8"))
    variants = {str(trim): [] for trim in TRIMS}
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite experiment output {out_dir}")
    for run in source["runs"]:
        label = run["label"]
        prediction = public_root / label / "prediction.npz"
        if sha256(prediction) != run["prediction_sha256"]:
            raise RuntimeError(f"public prediction hash mismatch for {label}")
        openings = run.get("openings") or {}
        if openings.get("status") != "candidates" or not run.get("boundary"):
            for trim in TRIMS:
                variants[str(trim)].append({"label": label, "seed": run["seed"],
                                            "path_index": run["path_index"],
                                            "openings": openings, "boundary": run.get("boundary")})
            continue
        frame_obs = [edge.frame_edge_observations(prediction, run["boundary"], item)
                     for item in openings.get("openings", [])]
        for trim in TRIMS:
            refined = json.loads(json.dumps(openings))
            refined["schema"] = "depth_wall_openings_robust_edges_v1"
            refined["edge_refinement"] = {"method": "trimmed_per_frame_dropout_extrema",
                                          "trim_fraction": trim, "roi_margin_m": .10,
                                          "truth_free": True}
            for item, obs in zip(refined.get("openings", []), frame_obs):
                interval = edge.consensus_interval(obs, trim_fraction=trim)
                original = list(map(float, item["u_interval_m"]))
                item["edge_refinement_diagnostic"] = {"frames_with_roi_dropout": len(obs),
                                                       "original_u_interval_m": original,
                                                       "refined": interval is not None}
                if interval is None:
                    continue
                lo, hi = interval
                item["u_interval_m"] = [lo, hi]
                item["width_m"] = hi - lo
                item["measurement_interval_m"] = [max(0., hi-lo-.02), hi-lo+.02]
                item["edge_refinement_diagnostic"]["refined_u_interval_m"] = [lo, hi]
            variants[str(trim)].append({"label": label, "seed": run["seed"],
                                        "path_index": run["path_index"],
                                        "boundary": run["boundary"], "openings": refined})
        print(f"[public freeze] {label}: {len(frame_obs)} opening components", flush=True)
    manifest = {"schema": "public_opening_edge_refinement_freeze_v1",
                "truth_free": True, "ground_truth_read": False,
                "source_freeze_sha256": sha256(freeze_path),
                "source_run_count": len(source["runs"]), "trim_fractions": list(TRIMS),
                "variants": variants}
    out_dir.mkdir(parents=True, exist_ok=False)
    freeze_file = out_dir / "refinement_freeze.json"
    freeze_file.write_bytes(canonical_bytes(manifest))
    (out_dir / "refinement_freeze.sha256").write_text(sha256(freeze_file) + "\n", encoding="utf-8")
    return manifest


def evaluate(manifest: dict, freeze_path: Path, private_root: Path, out_dir: Path) -> dict:
    import run_opening_lidar_batch as batch

    freeze_file = out_dir / "refinement_freeze.json"
    digest = sha256(freeze_file)
    if digest != (out_dir / "refinement_freeze.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("refinement freeze hash mismatch; refusing to read private scenes")
    if digest != hashlib.sha256(canonical_bytes(manifest)).hexdigest():
        raise RuntimeError("in-memory refinement differs from frozen bytes")
    source = json.loads(freeze_path.read_text(encoding="utf-8"))
    by_label = {r["label"]: r for r in source["runs"]}
    reports = {}
    # The frozen variants and integrity marker exist before this first truth read.
    for trim, predictions in manifest["variants"].items():
        results = []
        for candidate in predictions:
            seed = int(candidate["seed"])
            scene = json.loads((private_root / f"scene_{seed}" / "scene.json").read_text(encoding="utf-8"))
            row = {"boundary": candidate.get("boundary"), "openings": candidate["openings"]}
            wall_map = ({i: batch._geometric_truth_wall(row["boundary"], i, scene)
                         for i in range(len(row["boundary"]["vertices_xy"]))}
                        if row["boundary"] else None)
            scored = batch._opening_metrics(row, scene, wall_map)
            results.append({"label": candidate["label"], "opening_metrics": scored})
        reports[trim] = results
    # Write only after all private evaluation, with hashes tying it to the freeze.
    report = {"schema": "opening_edge_refinement_development_evaluation_v1",
              "development_only": True, "heldout_data_used": False,
              "refinement_freeze_sha256": digest, "runs_per_variant": manifest["source_run_count"],
              "variants": reports}
    (out_dir / "evaluation.json").write_bytes(canonical_bytes(report))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-freeze", type=Path, required=True)
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    manifest = freeze(args.source_freeze, args.public_root, args.output_dir)
    report = evaluate(manifest, args.source_freeze, args.private_root, args.output_dir)
    for trim, rows in report["variants"].items():
        matched = sum(bool(r["opening_metrics"]["pass_85_percent"]) for r in rows)
        failures = [r["label"] for r in rows if not r["opening_metrics"]["pass_85_percent"]]
        print(json.dumps({"trim_fraction": float(trim), "passing_captures": matched,
                          "run_count": len(rows), "failures": failures}), flush=True)


if __name__ == "__main__":
    main()
