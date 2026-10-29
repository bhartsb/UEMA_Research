# Latent body hierarchy: review checklist (Sep 30, 2026)

Written before the review. Each item is checked by reading the code and, where marked (N), by a numeric test.

## A. Data layout, units and conventions
1. `bones` [H,11]: quaternion order (xyzw claimed), world cm, cm/s, capacity. Match the producer (env/datasource), including MuJoCo's wxyz order. (N)
2. Executed command [H,9]: cm target, quaternion order, radius, strength. Match the env's order format. (N)
3. Contact: patch → handle mapping (skin_owner), bit values.
4. `duration_s` 0.05 against the real model step (action repeat x physics timestep).
5. Action contract string against the frames' contract.

## B. Dimensions
6. STATE 12 = relative position 3 + quaternion 4 + velocity 3 + capacity 1 + contact 1. (N)
7. CHANGE 6 and ACTION 8, per handle; owned vs subtree handle counts.
8. Encoder input order equals the constructor's size sum; decoder output split order.
9. Peer messages and context widths; child code widths.

## C. Calculations
10. `quaternion()` canonicalisation (sign, half turn). (N)
11. `rotation_delta(start, end)`: frame (world vs body), multiplication order, small-angle limit, angle near pi. (N vs scipy)
12. `state_inputs`: anchor choice per chain, history delta.
13. `make_batch`: change, action and scaling; scale divide in training vs multiply in `handle_targets`. (N round trip)
14. `handle_targets`: position and orientation composition order consistent with `rotation_delta`. (N round trip)
15. `through_synergy`: unit conversions (/200), which rows it replaces.
16. Losses: every term, weight, reduction; latent penalty.
17. Search: candidate sampling, zero code, argmin, re-decode.
18. Metrics: `paired_gain`, `prediction_ranking`, `intervention_spread`, `acceptance` statistics.
19. Checkpoint: saved/restored fields (model, optimizer, RNG, extractor), fingerprint checks.

## D. Trainer integration
20. Call site: which frames and transitions, executed (not proposed) command, frame index alignment.
21. Extractor: history/start/end pairing, boundary drops.
22. No global RNG draw, no gradient into production, failure disables only the shadow.
23. Record cap and auto resume (Sep 29 additions).

## E. The causal-control test tool (Sep 30)
24. `decode_with` equals `decode`; override semantics.
25. Predicted vs actual change in the same units and frame, same horizon.
26. Code range from the encoder; sampling; hold order for other handles.
27. Statistics: spread, repeat, correlation and the shuffle control.

## F. Tests
28. Each test checks its stated claim; would it fail with the defect planted?

## Results (Sep 30, 2026)

Verified correct: every layer size (encoder and decoder widths per part), cm and xyzw conventions from
proprio_adapter.py and action_decoder.py, rotation_delta against scipy (world frame, end times start inverse),
the command round trip (under 0.000001 cm), the skin-to-part map (38 arm and hand patches; trunk, legs and head
unmapped by design), frame alignment at the trainer call site, no gradient or random draw into production.

Defects found and repaired (commits dea93f310, b1c2dd0e8, 9b6d45795):
1. Loss dominated by the parent-predicts-child term (97.8% of the live shadow's loss; typical code size 43 per
   dimension; gradient cap reached on all of the last 2,999 updates). Repair: code_bound "tanh".
2. Provisional unit scales: the command-target term outweighed the movement term 94 to 1 and turning was 0.57% of
   the movement term. Repair: scales measured by mimo_bridge.tools.latent_body_split (root mean square per channel;
   future loss 1.0 = "nothing moves").
3. Finger commands learned on the requested wrist: median 14.15 cm from the finger, against 0.85 cm once placed on
   the actual wrist (which matches the first physics frame within 0.12 cm). Repair: hand_action_frame
   "actual_wrist" (trajectory.move_digits; handle_targets places them back on the issued wrist, where the hand
   controller solves them: hand_target_control.project_world_orders).
4. Offline fit read the file in time order (eight bodies per moment). Repair: --shuffle-buffer, --epochs.
5. Tests: 7 of 8 planted defects passed the suite; the off-hook test failed in a full run. Repair: calculation
   tests; all 8 planted defects now fail the suite.
6. Causal test centred trial codes on zero, about half a spread from the live shadow's real codes. Repair:
   --centre mean (default), the offset reported per part.
Consequence: the live shadow's movement predictions were worse than "nothing moves" (median miss 1.24 cm against
1.02 cm), so the Sep 30 morning causal test (11 of 24 parts) is not evidence for the design. The refit and its
causal test run in the pause of Sep 30 (pause_jobs_2026-09-30_latent_refit.json); the in-training shadow then
continues from the refit (latent_shadow_v2_switch_2026-09-30.py).
