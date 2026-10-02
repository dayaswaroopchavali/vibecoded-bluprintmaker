# Simulator-to-blueprint benchmark

Status: implementation started; no reconstruction stage has passed.

## Constraints

Local laptop only, no model training, no paid cloud. Existing pretrained DA3 is an initial baseline, not a promise of measurement accuracy. Coding tasks use Luna agents with main-agent review.

## Stage gates

1. Single room: recover floor boundary, wall lengths, openings and ceiling height from RGB.
2. Multiple rooms: at least three rooms and a connector, recover adjacency and placement; measure drift with correction enabled and disabled.
3. Multiple floors: recover floor assignments and vertical placement; measure accumulated height drift and correction ablation.

Require at least ten distinct house geometries and three independently seeded camera captures for each house in each stage. A smoke test, oracle geometry test, or successful rendering is not reconstruction success. Development scenes and final evaluation scenes must have separate seeds, with configuration frozen before final evaluation. Rectangles are only the initial smoke-test scene family; passing those cannot establish general-house success. Include nonrectangular layouts, furnishings, occlusions and varied textures in the final set.

## Measurement rules

Assessment video wall error is at most 3%; ceiling height error at most 1.5 cm per room, repeat height spread at most 1 cm; opening width error at most 2 cm for at least 85%, counting false and missed openings. Report both wall repeatability thresholds (1 cm and 0.5%) because the assessment says 'or'. Do not silently choose a favorable interpretation.

RGB has an unknown global metric scale. Report similarity-aligned proportions as a distinct diagnostic, never as a metric gate pass. Metric runs require an explicitly documented input scale source. Simulator dimensions, exact camera trajectories and ground-truth intrinsics are unavailable to RGB reconstruction. Ground truth must not select segmentation, wall planes, frames or parameters. Known real-room measurements such as 335 cm must not enter this pipeline.

## Architecture

Generator -> RGB-only capture folder -> generic pretrained inference -> observed structural plane extraction -> room boundary hypotheses -> blueprint prediction -> separate evaluator.

Generator stores geometry, camera poses, depth and measurement truth in a separate private tree. The evaluator can read that tree; reconstruction cannot. For later depth-tier tests, export a separate input bundle with documented noise and coordinate conventions. Synthetic depth is not validation of an actual phone LiDAR sensor.

Keep furniture features for camera tracking; exclude furniture from structural plane fitting. Do not paint in occluded walls and treat them as observations. Structural continuation is an inference and must carry that label and uncertainty. Unsupported geometry should cause incomplete output or a failed gate, rather than a fabricated confident blueprint.

## Viewing

Use a browser walkthrough with mouse look, keyboard movement, reset view, floor selection and a plan overlay. Keep rendering and reconstruction separate. A lightweight procedural renderer is an initial integration test; realistic sensor behavior and photorealistic furnished scenes remain separate requirements. Blender may be used headlessly later if useful, without requiring the user to navigate its editor.

## Assessment scope still outstanding

Simulation cannot replace the required real tape/laser benchmark, three input tiers, consumer-app comparison, damage evaluation, or live unseen-property defense. No simulator-only completion claim implies full assessment compliance.
