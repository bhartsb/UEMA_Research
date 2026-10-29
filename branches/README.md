# Unmerged branch work

Each patch holds one development branch's own changes to `Source/Python/lem`, `tokenizer` and `mimo_bridge` (from its merge base with the release's source commit `9916126d1` to the branch tip), written as the difference between the release versions of each file: vendor names removed and the copyright and license lines added by the same rules as the release tree, and without the files the release leaves out. None is applied. `applies` is the result of `git apply --check <patch>` run in the release folder; `no` means the release's source commit changed the same lines after the branch started.

| Branch | Last commit | Commits since Oct 16 | Files | Lines + / - | Applies |
| --- | --- | --- | --- | --- | --- |
| `build/pose-reader-repair2-2026-10-16` | 2026-10-17 01:31:25 -0700 0ec1c6886 | 2 | 38 | +11529 / -39 | yes |
| `build/s4r-measure-recording-2026-10-22` | 2026-10-22 10:32:50 -0700 d6a12e317 | 26 | 18 | +7958 / -78 | yes |
| `prep/d1-correction-2026-10-16` | 2026-10-16 17:40:04 -0700 1b91eda76 | 5 | 8 | +3850 / -64 | yes |
| `prep/d1-correction-handoff-2026-10-20` | 2026-10-21 02:13:47 -0700 8862cedb5 | 11 | 8 | +4793 / -64 | yes |
| `prep/d1-d2-package-2026-10-16` | 2026-10-16 14:17:58 -0700 6f03f1a7e | 3 | 8 | +3242 / -64 | yes |
| `prep/d1-identity-diagnostic-2026-10-20` | 2026-10-20 17:53:22 -0700 2b13dfac8 | 6 | 9 | +4278 / -64 | yes |
| `prep/d2-correction-2026-10-16` | 2026-10-16 17:02:58 -0700 6b24287dd | 4 | 9 | +3546 / -64 | yes |
| `prep/finger-revision-2026-10-16` | 2026-10-17 01:14:43 -0700 baf75d3f5 | 8 | 10 | +6768 / -64 | yes |
| `prep/goal-value-b8-design-2026-10-16` | 2026-10-17 03:40:46 -0700 169d89bef | 3 | 8 | +3242 / -64 | yes |
| `prep/goal-value-correction-2026-10-16` | 2026-10-16 17:43:14 -0700 231f0344c | 3 | 8 | +3242 / -64 | yes |
| `prep/tests-page-terminal-block-display-2026-10-16` | 2026-10-16 20:30:52 -0700 3f7aa8da6 | 1 | 5 | +541 / -8 | yes |
| `prep/tests-page-terminal-block-display-repair-2026-10-17` | 2026-10-17 00:30:25 -0700 015adcd67 | 1 | 5 | +920 / -14 | yes |
| `prep/tests-page-terminal-block-display-repair-r2-2026-10-17` | 2026-10-17 01:49:12 -0700 477a15130 | 1 | 5 | +920 / -14 | yes |
| `prep/tests-page-terminal-block-display-successor-2026-10-16` | 2026-10-16 23:11:14 -0700 23a299847 | 1 | 5 | +747 / -9 | yes |
| `repair/a2-arm-consistency-2026-10-23` | 2026-10-23 07:51:44 -0700 71841320b | 66 | 32 | +14655 / -137 | no: Source/Python/mimo_bridge/tools/training_preflight.py |
| `repair/persistent-goal-20261025` | 2026-10-25 11:59:41 -0700 2d7718add | 71 | 36 | +15487 / -143 | no: Source/Python/mimo_bridge/tools/training_preflight.py |
| `repair/s4-learning-repairs-2026-10-21` | 2026-10-22 02:27:09 -0700 fd4236974 | 25 | 17 | +7880 / -78 | yes |
| `repair/s4-planner-records-2026-10-21` | 2026-10-22 00:17:34 -0700 fc6ab185f | 17 | 13 | +6818 / -72 | yes |
| `repair/s4r-arm-ik-fk-2026-10-22` | 2026-10-23 06:19:10 -0700 00bda4712 | 59 | 26 | +13206 / -123 | yes |
| `repair/s4r-pose-fields-s1-2026-10-22` | 2026-10-22 23:52:38 -0700 7dd0eabda | 57 | 29 | +11299 / -133 | no: Source/Python/mimo_bridge/tools/training_preflight.py |
| `repair/s4r-pose-objective-2026-10-22` | 2026-10-22 15:24:47 -0700 f0183150b | 48 | 21 | +9356 / -104 | yes |
| `repair/target-candidate-20261023` | 2026-10-24 18:27:41 -0700 46e0c8363 | 63 | 35 | +14273 / -138 | no: Source/Python/mimo_bridge/tools/training_preflight.py |
| `repair/whole-body-request-2026-10-21` | 2026-10-21 16:14:09 -0700 2e4a5e52c | 18 | 31 | +13094 / -73 | no: Source/Python/mimo_bridge/tests/test_movement_viewer.py, Source/Python/mimo_bridge/tools/movement_viewer.css, Source/Python/mimo_bridge/tools/movement_viewer.js, Source/Python/mimo_bridge/tools/movement_viewer/__init__.py |
| `working-tree-uncommitted` | uncommitted | 0 | 3 | +472 / -13 | yes |
