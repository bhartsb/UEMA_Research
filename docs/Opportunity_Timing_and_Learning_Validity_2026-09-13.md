# The current audit does not measure the learner's complete gain of function

The audit measures whether the continuing predictor still works with the frozen
actor's encoder. It is not a valid measurement of the complete learner's retained
ability, because native training updates its own encoder. It also checks only the
first 50 ms of a held command whose self-choice forecast covers 250 ms. Neither
limitation establishes why the present model has not acquired a useful ability.
The present reading must not license a new chooser or establish forgetting.

## Verified timing and inputs

The native world advances 10 ms per source call. Five source calls make one
model observation, and a body order is held for five observations. Both the
order change and audit preparation use the same global-step modulus of five.
Consequently every audit starts at order onset and is completed by the next
observation. Increasing the spacing between audits does not increase their
forecast duration.

| Event | Elapsed physical time from audited order start |
|---|---:|
| Current raw observation and actor history captured | 0 ms |
| Actual command, including averaged executed gaze, recorded | 50 ms |
| Next actor call encodes the target and completes the audit | 50 ms |
| Same held order reaches the forecast's intended endpoint | 250 ms |

Preparation runs after the physical burst but before either learner or actor
consumes that next observation. It uses the prior actor tokens, achieved state,
context and memory bias. The command is the one executed, including scheduled
holds and gaze overrides. The target uses raw encoder tokens, not top-down
blended predictions. These parts of the causal ordering are correct.

However, `OpportunityAudit.predict` supplies the frozen actor encoder and inputs
to the continuing learner's `forward_model` and `prediction_heads`. Native
next-frame training re-encodes inputs and targets through the continuing
learner's own encoder. No alignment map or explicit fixed-coordinate training
objective makes the hybrid the complete trained learner. Encoder changes can
therefore lower the hybrid's retention score even if complete-learner prediction
remains useful. This is a verified interface mismatch; its numerical contribution
to H80's result has not been isolated.

The self-function predictor is another component. Its native training uses real
executed sequences over horizons 1, 5, 20 and occasionally 100 observations.
Quarter-second self-function learning could improve without improving the
one-observation main-transformer audit. Healthy function indices alone can stay
constant while the body learns a new movement or contact distinction.

## What H80 actually reported

The bounded read covers six logged snapshots through step 4,017,400. These are
sparse observations of 105 completed audits, not all intervening outcomes and
not independent counterfactual worlds. All logged learning-opportunity scores
were zero. At the last snapshot, all eight bodies met the declared health
criteria, but every body/sense pair failed the fixed-panel retention check.
Health was therefore not the immediate blocker at that snapshot.

The table counts correct-action forecasts beating both the shuffled-action and
unchanged-observation comparators. A perfect comparison would win every case;
winning these comparisons still does not establish purposeful behavior.

| Sense | Wins among 44 logged health-eligible body comparisons | Latest snapshot, out of 8 |
|---|---:|---:|
| Vision | 15 | 2 |
| Touch | 6 | 1 |
| Audio | 15 | 4 |
| Body position sense | 10 | 2 |

The meter also requires every one of eight retained fresh comparisons to beat
both controls, plus improvement beyond prior anchor best and preserved panel
accuracy. The source calibration's floor is exact-replay disagreement plus
float32 epsilon squared. That is a numerical repeatability floor, not a measured
tolerance for complete-learner representation changes. No threshold was changed
by this review.

Earlier corrected physical trials found resolvable action differences at both
50 and 250 ms. Their findings rebut the claim that first-frame consequences are
unlearnable: smaller movement is not irreducible noise. They support testing both
times on matched trajectories, not declaring that the longer time must succeed.
See [the corrected body-prediction review](body_prediction_review_2026-09-10/review.md).

## Minimum valid replacement

Build one auxiliary causal sequence predictor and a copied assay. Preserve every
existing learner tensor and optimizer state. Its inputs are re-encoded by the
same learner snapshot that supplies its representation; its targets have fixed
physical meanings. Do not splice a continuing predictor onto another encoder.

An immutable packet must contain the pre-action raw sensory window, source
checkpoint and world identity, every actual executed command and duration,
order identity/phase, masks, and each observed endpoint. Targets include achieved
body displacement/orientation and spatial touch pressure/location. They must
vary during healthy exploration; intact quality indices alone are insufficient.
Packets contain no future data in their predictor inputs.

Use a bounded causal readout of the current learner's own encoded window and
the complete command sequence. Train only the added parameters initially. The
same immutable raw packet must be re-encoded under each evaluated snapshot;
old cached embeddings cannot stand in for a newly changed encoder. Fixed
grounded targets make retained performance comparable. If a frozen sensory
target encoder is used for an additional output, train that output explicitly
in those coordinates and qualify it separately.

Before fitting, verify exact copied replay and actual action distinctions at
both 50 and 250 ms. Include every phase of the held order, scheduled and
model-authored control, and body-only action controls that preserve gaze.
Compare with unchanged state, constant motion where appropriate, action-blind
prediction and shuffled commands. Split by world before fitting; keep a fixed
retention set and an untouched confirmation set. Calibrate target scales from
training worlds only. Report pressure and position errors against the real
between-command effect and repeatability, including quiet or no-effect cases.

Only after predictions resolve useful physical distinctions should a copied
choice comparison ask whether the model chooses and retains newly controllable
outcomes while preserving bodily function. This experiment supplies no named
rolling or toy reward. It cannot promote itself through a favorable training
loss or the current hybrid meter.

## Reproduction and limits

`mimo_bridge.tools.opportunity_contract_assessment` traces the native audit's
interface calls with constructed, distinct owners and targets, then combines
that result with located native arguments and actor log metadata. It loads no
checkpoint, runs no training or physics, and changes no gate. Six tests passed,
including refusal to certify the hybrid even when supplied scores say available.

```powershell
$env:PYTHONPATH='Source/Python'
.venv_mimo/Scripts/python.exe -m mimo_bridge.tools.opportunity_contract_assessment `
  --run Saved/mimo_bridge/experiments/H80_local_shadow_20260913 `
  --through-step 4017400 `
  --out Saved/mimo_bridge/opportunity_contract_assessment_20260913.json
```

The contract trace identifies the currently imported implementation. It is not
a retrospective attestation of every inner function loaded at H80's birth.
Full numeric records are in
`Saved/mimo_bridge/opportunity_timing_audit_20260913.json` and
`Saved/mimo_bridge/opportunity_contract_assessment_20260913.json`.
No production control, checkpoint load or scientific gate change occurred.
