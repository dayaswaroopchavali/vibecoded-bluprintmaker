import numpy as np

import simulator
from virtual_gyro import make_stream


def _poses(count=20):
    return [
        {"position": [0.1 * i, 0.0, 1.4], "yaw": 0.3 * i,
         "pitch": -0.2 + 0.03 * i, "roll": -0.4 + 0.17 * i}
        for i in range(count)
    ]


def _opencv_rotation(pose):
    right, up, forward = simulator.camera_basis(pose)
    return np.column_stack((right, -up, forward))


def test_relative_rotations_are_proper_and_compose_to_endpoint_rotation():
    poses = _poses()
    stream = make_stream(poses, dt_s=0.02)
    assert stream["schema"] == "virtual_gyro_v1"
    assert stream["frame_count"] == len(poses)
    deltas = [np.asarray(m) for m in stream["delta_camera_to_previous"]]
    assert len(deltas) == len(poses) - 1
    for delta in deltas:
        np.testing.assert_allclose(delta.T @ delta, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(delta) - 1.0) < 1e-10
    composed = np.eye(3)
    for delta in deltas:
        composed = composed @ delta
    expected = _opencv_rotation(poses[0]).T @ _opencv_rotation(poses[-1])
    np.testing.assert_allclose(composed, expected, atol=1e-9)


def test_noise_and_bias_are_reproducible_for_a_seed():
    kwargs = dict(dt_s=1 / 60, seed=187, noise_std_rad_s=0.025,
                  bias_rad_s=(0.01, -0.02, 0.03))
    first = make_stream(_poses(), **kwargs)
    second = make_stream(_poses(), **kwargs)
    assert first == second
    clean = make_stream(_poses(), dt_s=kwargs["dt_s"])
    assert first["delta_camera_to_previous"] != clean["delta_camera_to_previous"]


def test_public_stream_contains_no_private_pose_or_scene_labels():
    stream = make_stream(_poses(), dt_s=0.03, seed=4)
    forbidden = {"position", "yaw", "pitch", "roll", "poses", "scene", "ground_truth",
                 "absolute_orientation", "up", "width", "height", "length"}

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.lower()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert forbidden.isdisjoint(set(keys(stream)))
    assert "delta_camera_to_previous" in stream
    assert "timestamps_s" in stream


def test_gradual_roll_and_inversion_remain_valid_relative_rotations():
    # Pitch crosses the inverted-view boundary while roll makes a full turn.
    poses = [{"yaw": 0.4, "pitch": float(p), "roll": float(r)}
             for p, r in zip(np.linspace(0.25, 2.85, 101), np.linspace(0., 2 * np.pi, 101))]
    ups = [simulator.camera_basis(p)[1] for p in poses]
    assert any(up[2] < 0 for up in ups)
    assert any(up[2] > 0 for up in ups)
    deltas = np.asarray(make_stream(poses, dt_s=0.01)["delta_camera_to_previous"])
    assert deltas.shape == (100, 3, 3)
    np.testing.assert_allclose(np.linalg.det(deltas), np.ones(100), atol=1e-9)
    assert np.max([np.linalg.norm(m - np.eye(3)) for m in deltas]) < 0.2
