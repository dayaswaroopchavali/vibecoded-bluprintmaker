"""Development diagnostic for camera-patch support threshold on one capture."""
import json
import sys

import numpy as np

import estimate_height
import extract_planes


def main():
    capture, freeze = sys.argv[1:3]
    with np.load(capture, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    points = extract_planes.points_from_prediction(arrays)
    planes = extract_planes.extract(points)
    centers = estimate_height.camera_centers(arrays["extrinsics"])
    label = capture.replace("\\", "/").split("/")[-2]
    row = next(row for row in json.load(open(freeze, encoding="utf-8"))["runs"]
               if row["label"] == label)
    axis = np.asarray(row["selected_axis"]["axis_world"])
    for coverage in (0.5, 0.45, 0.4, 0.35):
        height = estimate_height.estimate_height(
            planes, points, axis, centers, min_camera_coverage=coverage)
        print(coverage, height["status"], height["separation_unscaled"],
              height["floor_plane_id"], height["ceiling_plane_id"])


if __name__ == "__main__":
    main()
