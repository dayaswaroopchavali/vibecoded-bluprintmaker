"""Export dimensioned room polygons from a truth-free prediction freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rectify_rectangle import rectify


def export(freeze_path: Path) -> dict:
    raw = freeze_path.read_bytes()
    freeze = json.loads(raw)
    if freeze.get("truth_free") is not True or freeze.get("ground_truth_read") is not False:
        raise ValueError("refusing non-public or post-truth freeze")
    rooms = []
    for row in freeze.get("runs", []):
        result = rectify(row.get("boundary") or {}, row.get("paired_walls") or {},
                         row.get("openings") or {})
        rooms.append({"label": row["label"], "seed": row["seed"],
                      "path_index": row["path_index"], "height_estimate": row.get("height_estimate"),
                      **result})
    return {"schema": "rectified_single_room_blueprints_v1", "truth_free": True,
            "ground_truth_read": False, "source_freeze_sha256": hashlib.sha256(raw).hexdigest(),
            "rooms": rooms,
            "summary": {"captures": len(rooms),
                        "rectified": sum(x["status"] == "rectified" for x in rooms),
                        "refused": sum(x["status"] != "rectified" for x in rooms)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
