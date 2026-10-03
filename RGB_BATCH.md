# RGB batch, stage 1 (simulator -> blueprint, single room, UPRIGHT protocol)

Files: `run_rgb_batch.py`, `test_run_rgb_batch.py`, this file. Output root (new folder, never overwritten):
`outputs/sim-benchmark/rgb-batch-upright-v2/`. (`rgb-batch-upright-v1/` is the labelled 1-run pilot of an earlier code state; its distinctness threshold was 0.25 m, which two of the ten predeclared houses (104, 108) miss by 1 cm, so the threshold was set to 0.20 m before any evaluation result was used and the batch re-initialised as v2. The houses and the split did not change.) A copy of the predeclaration below is in that folder (`batch_config.json`, `PREDECLARED.md`),
written by `init` before any capture was rendered.

## Predeclared before running anything

- **Houses (10, fixed):** `simulator.make_scene` seeds 17 and 101..109 (the ten houses of `wide-capture-qa-10houses-widefix-v1`, so a later
  comparison with the LiDAR batches uses the same rooms). Distinct = some component of (width, length, height) differs by at least 0.20 m
  between every pair (checked in the tests). Single-room box houses; `house_layouts.py` is not used in stage 1.
- **Paths:** 3 per house, seeds from the existing scheme `simulator.path_seeds(scene, 3101)` (3128909..3128911; the seed depends on the
  capture seed only, the room geometry differs per house). Poses: `capture_parallax.make_wide_path` (perimeter loop plus sphere sweep),
  with **roll forced to 0** = upright protocol. `--protocol rolling` keeps the generator's roll sweep; it is implemented but not run in stage 1.
- **Train/held-out split (by house id order, no selection after results):** houses sorted ascending by seed id; first 5 = calibration split
  (17, 101, 102, 103, 104); last 5 = held-out (105, 106, 107, 108, 109). No calibration is fitted or applied in this batch.
- **Frozen settings (all runs):** Blender EEVEE, original appearance, 120 frames, 480x360, fov 78, 6 fps; DA3-LARGE via `infer_rgb.py`
  (`--frames 120 --resolution 480 --seed 2026`, the settings of `da3_full120` in `wide-capture-seed17-path00-roll0`); truth-free chain
  `run_axis_candidates.run_extraction` = planes -> `axis_candidates.enumerate_axis_candidates` (every candidate, cap 24 recorded)
  -> `height_diagnosis.estimate_room_height` -> `wall_extraction` + `boundary_assembly.assemble_boundary_ray` (k = 2.0).
  Source sha256 of every module involved is recorded in `batch_config.json`; `run` refuses to continue if any changed.
- **Serial:** one process at a time (lock file `.gpu_serial.lock`); DA3 is one inference process per capture.
- **Budget:** `--max-hours 3` (default) stops starting new runs; the report states exactly how many completed.

## Stages and the freeze-before-truth ordering

1. `capture` (generator side): scene and poses from the seeds, capture-side QA (`simulator.angular_coverage`, `camera_motion_summary`,
   `coverage_diagnostics`; failures are reported, never hidden), Blender render via `blender_capture.py --poses-json`.
2. `inference`, `chain`, `freeze` (**truth-free**): read only the video / prediction. `freeze.json` stores the sha256 of the video, the
   prediction and every chain output. These functions import and name no scene, `private_ground_truth` or evaluator
   (`TRUTH_FREE_FUNCTIONS`; AST test).
3. `evaluate` (separate CLI command, EVALUATION ONLY): refuses unless all runs are frozen (`--allow-partial` evaluates what exists), writes
   `freeze_index_before_truth.json`, verifies each run's freeze, and only then reads the private scene. Every candidate is scored by
   `evaluate_floorplan.evaluate_floorplan` (3.0 m ceiling-normalised, evaluator-oracle alignment/mirroring/edge matching, `axis_sign_resolved=False`).

## Gates (none weakened)

- Per candidate: `custom_1_5cm_horizontal_diagnostic` (max wall error <= 1.5 cm) and `wall_length_3_percent_threshold_oracle_normalised`
  (<= 3 %), exactly as coded in `evaluate_floorplan.py`.
- Ceiling-height diagnostic <= 1.5 cm: under the 3.0 m normalisation the cap separation defines the scale, so height error is zero by
  construction. The diagnostic therefore re-expresses the height with the wall scale: with r = median(predicted/true wall length) in the
  normalised frame (= s_wall / s_cap), the wall-scaled ceiling height is `h_true / r` and its error is `100 |h_true / r - h_true|` cm. Gate: <= 1.5 cm.
  Height spread across the three paths of a house (of the wall-scaled heights) <= 1 cm.
- Repeatability across the 3 paths of a house (CONTRACT.md): for every true wall matched in two paths the normalised predicted lengths differ by
  <= 0.01 m and, separately, by <= 0.5 % of the true length; both pass rates must be 100 %.
- **Selected candidate** for the run-level gate = the unique truth-free top-ranked viable candidate (score from `axis_candidates`). A tie at the top
  is an unresolved axis ambiguity and fails the run; truth never picks. Oracle information about the other candidates (`any_candidate_gate_all`,
  `co_supported_gate_all_count`) is listed but is not a pass criterion.
- A run passes only if capture QA passes, DA3 is ok, and the selected candidate passes all three per-candidate gates.
  A house passes only if its three runs pass and its repeatability and height-spread gates pass.
- **`stage_1_complete` is false unless ALL 30 runs are present and pass, every house passes, and every run used identical frozen code.**

## Recorded for a later labelled cross-house calibration experiment (evaluation part only)

Per candidate and per run: estimated floor-ceiling separation (unscaled), true height, `s_cap = separation / h_true`, median wall-to-cap scale
ratio, its inverse (cap-vs-wall scale ratio), spread over walls, wall-scaled ceiling height and its error. Nothing is fitted; the split above
is the only declared use (fit on the calibration houses, check on the held-out houses, later, with a separate label).

## Commands

```
python run_rgb_batch.py init     --output-dir <NEW folder under outputs/sim-benchmark>
python run_rgb_batch.py run      --output-dir <dir> [--houses 17] [--paths 0] [--max-hours 3] [--retry-failed]   # resumable
python run_rgb_batch.py evaluate --output-dir <dir> [--allow-partial]                                           # EVALUATION ONLY
python -m unittest test_run_rgb_batch
```

Resuming: each stage writes `receipts/<stage>.json` (status, duration, sha256 of its outputs). A finished stage whose output hashes still match is
skipped; a mismatch raises (no silent redo); an interrupted partial folder is renamed `*.incomplete-N` (kept), a failed stage needs `--retry-failed`.
Results go to a new `results_NNN/` folder per `evaluate` call (`RESULTS.md`, `runs.json`, `houses.json`, `candidates.json`, `stage1.json`).

## Limits

Single-room box houses, one renderer, original (untextured-noise) appearance, no sensor noise. DA3 output scale is arbitrary; the 3.0 m
normalisation and alignment are evaluator-oracle. Nothing here is an accuracy or completion claim.

## Run record (v2)

All 30 captures were rendered, inferred and frozen serially (one interrupted process was resumed from receipts). Evaluation: outputs/sim-benchmark/rgb-batch-upright-v2/results_000/ (RESULTS.md, stage1.json). Result at the time of writing: stage_1_complete false (0 of 30 runs pass). Capture QA: the angular full-video coverage gate (0.99) is missed by path 0 of every house, so those 10 runs are QA failures by the unchanged gate. Per-run wall times are in uns.json.
