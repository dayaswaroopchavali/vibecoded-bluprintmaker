"""Rebuild a cached window graph with one RGB-inferred camera model.

The same physical camera recorded every frame. DA3 estimates a separate focal
length in each six-frame window, so linking those windows with their disparate
intrinsics can warp both the poses and the wall point cloud. This experiment
reuses the existing RGB-only window cache and estimates a consensus K from it.
No simulator pose, depth, dimensions, gravity, or evaluation files are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import refine_window_graph as R
import windowed_reconstruction as W


def run(pool_dir: Path, baseline_dir: Path, output_dir: Path) -> dict:
    pool_dir, baseline_dir, output_dir = map(Path, (pool_dir, baseline_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    manifest_path = baseline_dir / "manifest.json"
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = R.params_from_manifest(baseline)
    groups = baseline["window_groups"]
    windows, files = R.load_pool_windows(pool_dir, groups, baseline["resolution"])
    merge = baseline["merge"]
    pruning = merge.get("cycle_pruning") or {}
    prediction, report = W.merge_window_graph(
        windows,
        params,
        shared_intrinsics=True,
        max_neighbors=params["window"] // 2,
        prune_cycles=bool(pruning.get("enabled")),
        prune_gates=pruning.get("gates", W.PRUNE),
    )
    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "prediction.npz", **prediction)
    components, counts = np.unique(prediction["component_ids"], return_counts=True)
    largest = int(components[np.argmax(counts)])
    selected = prediction["component_ids"] == largest
    np.savez_compressed(
        output_dir / "largest_component.npz",
        **{key: value[selected] for key, value in prediction.items()},
    )
    result = {
        "kind": "cached_window_graph_shared_intrinsics",
        "truth_free": True,
        "ground_truth_read": False,
        "input_manifest_sha256": W.sha256(manifest_path),
        "pool_files_sha256": {path.name: W.sha256(path) for path in files},
        "source_rows": int(len(prediction["source_indices"])),
        "unique_source_frames": int(len(np.unique(prediction["source_indices"]))),
        "duplicate_source_rows_across_components": int(
            len(prediction["source_indices"]) - len(np.unique(prediction["source_indices"]))
        ),
        "components": {str(int(k)): int(v) for k, v in zip(components, counts)},
        "largest_component": largest,
        "shared_intrinsics": report.get("shared_intrinsics"),
        "merge": R._jsonable(report),
        "prediction_sha256": W.sha256(output_dir / "prediction.npz"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(R._jsonable(result), indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-pool", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.window_pool, args.baseline, args.output)
    print(json.dumps({key: result[key] for key in (
        "source_rows", "unique_source_frames", "duplicate_source_rows_across_components",
        "components", "largest_component", "shared_intrinsics")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
