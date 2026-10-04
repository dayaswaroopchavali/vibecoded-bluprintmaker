# Room capture protocol

Use this procedure for each of the three independently seeded walkthroughs of a room.

1. Walk one broad loop around the room at a steady pace. Keep the camera orientation moving smoothly through the room: look around, up toward the ceiling, and down toward the floor. Keep the camera and operator clear of walls and furniture.
2. When a doorway or window is visible, approach a clear viewing position with enough distance to fit its full outline in the frame. If furniture blocks a direct approach, continue parallel to the wall until the route around it is clear.
3. Turn smoothly toward the opening center, hold the view briefly so the full opening stays in frame, then ease back into the broad room sweep. For doorways, lower the view enough to include the threshold; for windows, frame the opening around its center height.
4. Continue the loop and repeat this observation for each visible opening. Keep walking and turning steady; avoid sudden pivots or crossing through furniture.

The simulator uses the opening centers to choose capture poses in the same way an operator aims at visible openings. Scene geometry and camera poses remain under `private_ground_truth`; reconstruction receives only the rendered RGB videos.

The supplied videos use 12 fps. The QA-frozen baseline loop has 320 frames. Opening-aware paths add the smooth approach, view hold, and return segments while preserving the baseline sweep poses.
