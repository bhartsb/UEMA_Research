# Movement prediction: real progress, narrower than reported

**A separate predictor learns physical consequences. Its claimed all-world
advantage did not survive a fresh confirmation, and the live brain has not
acquired this capability.** This result supports continuing action-conditioned
learning. It does not establish purposeful action or remove the other
developmental bottlenecks.

The reported model was frozen before confirmation: no new fit, adjustment or
sample extension. Six fresh seeded copies of the same crib supplied 96 decision
states, four commands each, and an exact-command replay. The world, candidate
and bout random streams were all new. All 96 states completed; all six worlds
are reported. This tests generalization within the crib setup, not transfer
to different scenes, bodies or UE Avatar.

This is kernel regression on raw body measurements, not a newly trained part
of the live transformer. Inputs include current pose, velocity, capacity and
touch. Its five history entries encode position changes, velocity, previous
command position offsets and strength; they do not encode previous command
orientation, radius or past touch. Outputs are body positions, not future
touch, fatigue or wellness. Success with these raw inputs does not demonstrate
that the live model's pooled sensory tokens preserve the required information.

## What the fresh test measured

Action-effect error is a squared-error ratio: 0 is exact and 1 predicts no
difference between actions. Lower is better. The history-free comparison was
selected using old training worlds only, before these outcomes were scored.

| New world | Reported predictor | Linear | Without history | Original comparison |
|---|---:|---:|---:|---|
| 0 | 0.710 | 0.792 | 0.689 | Pass |
| 1 | 0.622 | 0.651 | 0.581 | Pass |
| 2 | 0.803 | 0.763 | 0.772 | Fail |
| 3 | 0.716 | 0.701 | 0.718 | Fail |
| 4 | 0.658 | 0.766 | 0.680 | Pass |
| 5 | 0.738 | 0.797 | 0.717 | Pass |

Every world beats the no-effect reference and every one of eight shuffled
fits. Every world also beats no-movement and constant-velocity forecasts of
absolute final position. Physical replay MSE is at most 3.81e-11 cm squared per
coordinate, against action-effect MSE of 3.58-6.28, so replay disagreement does
not explain this learning. Eight shuffles are a comparison, not a calibrated
5% significance test.

The declared requirement to beat linear in *every* world fails in worlds 2
and 3. The additional final-frame action-effect comparison fails there too.
Absolute final-position forecasts beat linear in all six: being better at
where the body ends up does not automatically mean being better at telling
the candidate moves apart.

Remaining error at 250 ms is 3.49-5.19 cm RMS per body point, and 4.38-6.39 cm
per wrist. The improvement is real; the remaining error is material for contact.
The history-free model beats the reported model in four worlds. This does not
prove that memory can never help. It disproves treating this particular added
history as the established explanation for the improvement.

Scheduled-rest starts were slower in every world: observed body-point speed
was 4.05-11.38 cm/s, versus 15.24-34.28 during scheduled activity. They were
not motionless. This collection does not isolate whether rest improves learning
or helps wellness; both schedules remain in the reported results.

## Repairs and corrections

- **Wrong body/time indexing:** the old "final frame" slice mixed times and
  selected only the last body points. I repaired it and added a test where
  wrist errors cannot be hidden by stationary feet. Correct old absolute MSEs
  are 7.293/4.868/5.765 cm squared per coordinate. **The old numerical pass
  survives this repair.** Its tiny old absolute errors were invalid.
- **Reused evaluation evidence:** the old three worlds guided two decisions
  to add training data. Their outcomes were excluded from fitting, but they
  were no longer untouched confirmation. The six-world test resolves this.
- **Unsupported mechanism claim:** more examples helped this recipe, but no
  earlier ablation isolated history. This kernel regression also fits output
  columns independently; five output frames are not stepwise integration.
- **Missing safeguards:** gates now reject empty, nonfinite, incomplete-control
  and replay-unresolved evidence, and explicitly require beating no action
  difference. New collections record history timestamps; confirmation verifies
  that every command precedes the candidate and observations end by decision
  time. New fits require a new output directory instead of overwriting results.
- **Promised next test:** the repository had a sentence promising wellness
  choice, not an implemented sequence-predictor choice assay. Predicted
  positions alone do not provide future touch comfort or capacity.

The changed behavior passes 21 targeted tests across sequence scoring,
body-state prediction and action-effect collection. The earlier intermediate
negative fit receipts were overwritten by the original fitter; their prose
reports remain. This review archives the latest original result and source,
corrected old scores, frozen confirmation and all control scores.

## Decision

Keep this predictor offline. Close this confirmation with its failed all-world
comparison, preserving the real learning it measured. Proceed to the
[grounded wellness-choice experiment](next_wellness_test.md),
keeping current-state and predicted-path alternatives as offline comparisons.
The requirement is improved realized wellness from its own selection. The
body forecast is an input candidate, not itself the goal. Only then transfer
the successful learning target to a copy of the live model and test retention
and choice again. Responsive caregiving and spatial vision remain separate
developmental work; this body experiment has not tested them.

Production training continued during this review; no live model or execution
configuration changed. Current activity is recorded in
[training_status.json](training_status.json). This is a current-state check,
not an independent reconstruction of the report's "all day without a gap."

Receipts: [confirmation](confirmation.json), [control preparation](control_preparation.json),
[corrected old scores](corrected_old_worlds.json), [original reported result](reported_result.json),
[declaration](declaration.md), [reproduction](reproduce.md).
