# Wellness-choice study — declaration (committed before any fit)

2026-09-11, implementing the reviewer's six-step design verbatim
(docs/sequence_confirmation_2026-09-11/next_wellness_test.md). Corrections
owned from their round-8 verification, restated so this study's claims start
clean: my "passed everything" held only on the original three evaluation
worlds (4 of 6 fresh worlds on their confirmation run); my "memory solved it"
was wrong (removing history IMPROVED 4 of 6 worlds - the active ingredients
were examples and state-action interactions); and the live brain has gained
nothing from any of this yet.

## Sample count and uncertainty method (their step 5, published up front)

Measured on the 27 existing worlds (world-level units, never action pairs):
the planner's chosen action leaves a mean 0.0592 of bodily wellness unclaimed
per decision versus the best available candidate (0-3 scale; the full
candidate spread is 0.1424), with world-to-world standard deviation 0.0211.
Arithmetic: a selector capturing 25% of that improvement needs ~8 worlds for
a world-level 95% confidence interval above zero (1.96 x 0.0211 / 0.0148)^2;
50% capture needs ~2. DECLARED: 9 confirmation worlds. Replay disagreement
(~1e-6 on this scale) is far below any margin in play. If realized
comfort/capacity do not differ between candidates beyond replay error in a
world, that world is reported as unable to answer, per their step 4.

## Fits (their steps 1-2), on the 24 existing training worlds only

Outcome predictor targets: END tactile comfort, capacity, posture (stored per
arm), plus INTERMEDIATE tactile and posture reconstructed offline from the
retained per-frame patches and bones through the same state_wellness
preparation - reconstruction must show exact parity with the stored end
wellness on every arm or the row is rejected. Intermediate capacity is not
reconstructible historically and is not fabricated; the extended collector
(--component-paths, parity by construction) supplies it for confirmation
worlds only. Future touch is a target, never an input. Two equal-budget
alternatives selected on training worlds only, then frozen: (a) sensing +
prior commands + candidate command; (b) the same plus the body predictor's
forecast path, produced for training examples by leave-their-world-out body
fits (never the recorded actual path, never an in-sample forecast).

## Confirmation (their steps 3-5), only after both alternatives freeze

9 fresh worlds (collector: --component-paths on, history 5, rest on). Each
selector ranks the same four executed candidates; its choice is recorded
before outcomes are revealed; all four arms + duplicate replay execute from
the identical snapshot. Score realized comfort/capacity/posture AND the worst
intermediate comfort. Comparators: the planner's own choice (arm 0), hold,
uniform-random, and alternative (a) without path features. The best available
candidate is reported as the ceiling only. PASS = positive paired benefit
over BOTH the planner's choice and hold, world-level 95% CI above zero,
benefit above replay disagreement; and the path-features claim is made only
if (b) beats (a). Worst intermediate comfort must not be measurably worse.
All worlds and failures preserved.

## After a pass (their step 6, not part of this study's gates)

Transfer into a COPY of the live model and repeat behaviorally; adoption is a
checkpointed boundary change with a rollback receipt. Production training
never pauses for this study.

## Confirmation result (9 sealed worlds): GATE FAIL for both selectors, kept

Both selectors clearly beat HOLD (B: +0.0321, world-level 95% CI +0.0237 to
+0.0404 on the 0-3 scale) and neither makes choices with worse intermediate
comfort - but neither beats the PLANNER'S OWN CHOICE: B +0.0067 with CI
-0.0076 to +0.0211 (straddles zero); capture of the available 0.059 headroom
is indistinguishable from zero. Replay floor 7e-7 - measurement is not the
limit.

Located reading, consistent with the freeze-time warning: the component
forecasts themselves were barely better than "no difference between actions"
(CV 0.971-0.978), so ranking by them adds nothing the live planner does not
already achieve. The chain holds through movement prediction and breaks at
COMFORT prediction: touch is contact-driven and discontinuous, and these
inputs at this data scale do not forecast which candidate touches what.
No step-6 transfer occurs. The concrete next lever, surfaced for
prioritization: a contact-aware comfort predictor (geometry between the
predicted hand path and object positions), or the learning-curve route at
larger scale - stated as options, not launched.
