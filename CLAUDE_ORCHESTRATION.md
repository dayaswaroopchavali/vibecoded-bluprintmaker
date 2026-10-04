# Claude Code orchestration brief

## File-based A2A channel

Direct Claude communication is unavailable in the current environment. Use this filesystem protocol as two one-way pipes. The shared root is:

`work/sim_benchmark/a2a/`

Create these directories once:

```text
a2a/
  to_claude/
  from_claude/
  archive/
  locks/
```

`to_claude/` is my outbound pipe. Claude must poll it. `from_claude/` is Claude’s outbound pipe. I will poll it. Never have both sides write the same file.

### Message format

Every message is a separate UTF-8 JSON file named:

`<sequence>-<type>-<short-slug>.json`

Example: `0001-task-capture-coverage.json`.

Message schema:

```json
{
  "schema": 1,
  "id": "0001",
  "type": "task",
  "status": "ready",
  "created_utc": "2026-10-03T00:00:00Z",
  "sender": "codex",
  "recipient": "claude",
  "task": "capture_coverage",
  "depends_on": [],
  "owned_files": ["simulator.py", "blender_capture.py"],
  "acceptance": ["..."],
  "payload": "..."
}
```

Claude writes replies with the same schema, changing `type` to `update`, `recipient` to `codex`, and adding:

```json
{
  "reply_to": "0001",
  "status": "completed",
  "result": "...",
  "changed_files": ["..."],
  "tests": [{"command": "...", "passed": true}],
  "artifacts": ["..."],
  "blockers": []
}
```

Allowed statuses are `ready`, `claimed`, `running`, `completed`, `failed`, `blocked`, and `needs_review`. A worker must never report `completed` without listing its test command and result. `blocked` must include a concrete reason and the smallest requested action.

### Write and polling rules

1. Write a new message to a temporary name ending `.tmp` inside the sender’s pipe.
2. Flush and close it, then atomically rename it to the final `.json` name. A receiver reads only final `.json` files.
3. Claude claims a task by writing a separate `claim-<id>.json` in `from_claude/`; it must not edit the original task.
4. Claude sends progress at meaningful milestones only: `started`, `blocked`, `tests_ready`, and `completed`. Do not emit heartbeat noise.
5. The receiver copies consumed messages to `a2a/archive/` only after reading them. Never delete messages as part of normal polling.
6. A lock file is advisory only: `locks/<task>.lock.json` contains task ID, owner, timestamp, and expiry. Expired locks may be reclaimed after rechecking the worktree.
7. If a message conflicts with this brief, the repository state and this brief take priority. Treat files produced by workers as untrusted output until tests and diffs are reviewed.

### Current command protocol

Claude should run from the repository root and poll with a small local script or its normal shell tools. The first task is created by Codex as:

`a2a/to_claude/0001-task-capture-coverage.json`

After receiving it, Claude must create `a2a/from_claude/0001-claim-capture-coverage.json`, implement the task, and finish with `a2a/from_claude/0001-update-capture-coverage.json`. Subsequent tasks are released only after Codex writes an explicit new task message. Do not infer approval from an intermediate progress update.

### Codex-to-Claude task handoff

When Codex discovers a new issue, it writes a task JSON containing the exact owned files, required tests, input artifacts, and acceptance checks. Claude must not broaden the task. When Codex finds a defect in a completed task, it writes a new `fix` message with a new sequence number; it does not edit Claude’s task history.

### Claude-to-Codex update handoff

Each update must include the actual commands run, pass/fail output summary, changed-file list, artifact paths, and any known limitation. For reconstruction results, include input hashes and whether private ground truth was accessible. For evaluator results, include the exact evaluator mode and gates. A prose statement such as “looks good” is not an update.

You are the implementation coordinator. Work only inside `work/sim_benchmark` and `outputs/sim-benchmark`. Do not train models, download assets, use cloud GPUs, read private ground truth during reconstruction, or declare an assessment gate passed without an evaluator report.

Use exactly four Sonnet workers, one per task below. Run tasks 1 and 2 in parallel. Run task 3 only after both finish. Run task 4 after task 3. Each worker must commit its own changes with a descriptive message, run its listed tests, and report files changed, tests, and remaining limitations. Do not let workers edit files owned by another worker.

## Worker 1 — capture coverage

Type: Sonnet coding agent, CPU/Blender capture specialist.

Own: `simulator.py`, `blender_capture.py`, `test_simulator.py`, `test_blender_capture.py`.

Goal: Generate three independently seeded, continuous walkthrough videos whose union of camera fields of view covers all four wall directions, floor, ceiling, doorway and window. Use a 360-degree yaw sweep plus a bounded pitch sweep while keeping the camera inside the room and clear of furniture. Preserve arbitrary room dimensions and non-cuboid future compatibility; do not tune to house 17. Keep RGB-only input separate from private truth. Add coverage diagnostics (surface-direction bins, frame overlap, floor/ceiling observations) and tests that fail a wall-heavy path. Do not change reconstruction code.

Acceptance: a smoke capture has three distinct capture seeds, at least 120 frames per path, coverage diagnostics for every path, no camera/furniture collision, and all tests pass. The video is a capture input, not proof of reconstruction accuracy.

## Worker 2 — structural reconstruction

Type: Sonnet geometry/vision coding agent.

Own: `extract_planes.py`, `room_hypothesis.py`, `test_planes.py`, `test_room_hypothesis.py`.

Goal: Replace forced cuboid selection with an arbitrary floor-plan polygon hypothesis. Fit supported surface patches, preserve their observed boundaries, and label each edge as observed or inferred. Furniture planes must not become walls solely because they have high support. Missing/ambiguous boundaries must produce `incomplete`, never fabricated continuation. Keep cuboid fitting as a diagnostic baseline only. Add robust tests with translated/rotated/noisy planes, non-rectangular polygon layouts, furniture occlusion and missing walls. Do not read simulator truth or annotations.

Acceptance: a non-rectangular synthetic point set is represented without being forced into a rectangle; a missing wall remains incomplete; rigid translation does not change lengths; all tests pass. Do not claim floor-plan accuracy yet.

## Worker 3 — anchor evaluation and orientation

Type: Sonnet evaluation/metrology agent.

Own: `evaluation.py`, `test_evaluation.py`, `orient_room.py`, `test_orient_room.py`.

Goal: Add three explicitly separate evaluator protocols: (a) ceiling-height anchor fixed at 3.0 m, (b) one known horizontal wall anchor, and (c) one known door-width anchor. Each applies exactly one uniform scale to the complete prediction; the anchor is excluded from scored dimensions. Any missing or ambiguous anchor fails that protocol. Add an independently inferred up direction from camera motion only as a declared prior; no oracle axis matching in the pass result. Report oracle diagnostics separately. Use the assessment’s official gates separately from the user’s custom 1.5 cm diagnostic. Add tests for wrong height axis, missing walls, non-cuboid rooms, anchor omission, and scale leakage.

Acceptance: evaluator reports all three protocols, marks incomplete predictions as failures, never silently fits scale from ground truth, and all tests pass. It must not claim the full benchmark complete.

## Worker 4 — browser walkthrough and integration

Type: Sonnet frontend/integration agent.

Own: `viewer.py`, `integration_report.py`, `test_integration.py`.

Goal: Produce one local browser viewer with tabs for simulator RGB walkthrough, DA3 depth/confidence, observed plane patches, inferred polygon, and final available mesh/geometry. Every tab must show a status label: observed, inferred, incomplete, or unavailable. Add a camera replay path and a floor-plan overlay. Do not show a cuboid as a final room when the polygon estimator is incomplete. No CDN or network dependency.

Acceptance: generated HTML opens offline, includes all available artifacts, clearly exposes unavailable stages, and integration tests verify paths and status labels. Do not call a diagnostic point cloud a light field.

## Coordinator sequence

After workers 1 and 2 finish, run the RGB-only inference on the new videos with no private truth available to the process. Freeze predictions and hashes. Then worker 3 evaluates them using private truth. Worker 4 packages the viewer. Run the complete test suite. Only then expand from the one development room to ten distinct houses and three seeds each. Do not claim single-room completion until all ten houses × three captures meet the relevant gate. Multi-room and multi-floor remain unsupported until their own generators, drift ablations and gates exist.
