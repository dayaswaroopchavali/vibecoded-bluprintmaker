# Supplied capture schema inspection

Inspected directly from the supplied `assessment/single_room.zip`.
Archive prefix: `c00a170fe1/`.

- `rgb.mp4`
- `depth/<six-digit-frame>.png`
- `confidence/<six-digit-frame>.png`
- `odometry.csv`: `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y`
- `imu.csv`: `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`
- `camera_matrix.csv`: numeric 3 by 3 matrix without a header.

Prior inventory established 256 by 192 depth frames and confidence values 0/1/2. Empty distortion-center fields occur. Per-frame focal/principal-point values differ from the standalone camera matrix.

Not established from file names or headers: depth unit, whether depth is axial or ray range, pose direction/world axes, RGB-to-depth registration, IMU units and timestamp synchronization convention. These must be verified before claiming a compatible synthetic sensor bundle. Column-name compatibility alone is insufficient. The first-stage simulator's internal poses are not yet this capture format, and its ideal ray intersections are not an iPhone LiDAR model.
