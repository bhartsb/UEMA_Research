# Visual prediction repair

> **Status at 2026-09-24 9:58 PM PDT.** The text below was committed at 3:44 AM PDT (ecfc9060f),
> before production resumed; its "paused", "stopped" and "now running" lines describe that time. The
> final qualification passed at 3:48 AM PDT and production resumed from step 380,200 at 4:03 AM PDT.
> Vision then regressed again in production. On the same 8 saved moments the model at step 397,125
> misses the next picture by 3.09 times the error of predicting that the picture stays unchanged; the
> qualified model at step 380,500 misses by 1.24 times, so both lose to the unchanged picture, and the
> distance to a goal for eye planning is unmeasured
> ([evening audit](measurements/visual_regression_397125_2026-09-24.json)). Its rerun with the training
> arms (9:28-9:29 PM PDT) stopped without a result: the conditioning arm does not account for the
> input scale this morning's repair installed. Training resumed on this model as segment 40 at 9:46 PM PDT (from step 397,325; supervisor status 'training' at 9:58 PM PDT).

**The complete 300-update private continuation reduced visual prediction error
by 81.2% on the same 64 saved examples.** It finished below the error measured
before the regression. Production remains paused while a body-layer response
check is repeated over the reporting interval used in production.
[Full-continuation measurements](measurements/visual_continuation_2026-09-24.json).

Visual prediction did deteriorate. On the same 64 saved examples, error rose from
0.00974 at update 375906 to 0.04371 at update 380200: **4.49 times worse**.
Simply predicting that the picture would stay unchanged scored 0.00626.
Training is stopped, with update 380200 and all eight simulated bodies preserved.

The main fault is an unstable update in the visual prediction network. Of its
192 hidden units, 191 already normalize their inputs; one retained unit does not.
Its input magnitude grew more than fourfold. Small changes to that unit's weights
therefore caused much larger changes to the prediction. The visual network and
the separate eye-movement correction also learned large opposing outputs that
must cancel accurately. A second defect penalized the visual network's output
before adding that eye correction, although prediction training grades their sum.

The repair preserves the learned prediction while changing the retained unit's
input units. Its weight and existing optimizer history are converted together;
the initial prediction changes by less than 0.000004 from floating-point rounding.
The penalty now grades the complete prediction, including the eye correction.
Neither network is reset or disabled.

We tested 100 private updates from the stopped checkpoint, using 32 saved examples
for training and 32 separate examples for evaluation. The transformer and sensory
encoders stayed frozen to isolate the visual output update.

| Treatment | Error on the 32 evaluation examples, averaged over the last 20 updates |
| --- | ---: |
| Existing update and penalty | 0.02340 |
| Corrected penalty alone | 0.02339 |
| Corrected penalty and input units | **0.00589** |

That is a **74.8% reduction against the existing update**. The penalty correction
alone did not repair the regression. Error here means mean squared difference
between predicted and observed encoded visual features; it is not a percentage
of correct pictures or proof of useful eye control.

Past training is retained. This fault reduced prediction accuracy and can mislead
eye planning; it does not erase the learned body, sensory or motor networks.

The separate eye-correction layer also lost responsive units, and the general
health report missed it. A unit is responsive when its input leaves the flat
range of its activation curve, allowing input changes to affect its output.
Correcting the visual input units restored prediction
accuracy in the private test, but only 42% of this layer responded afterward;
the required minimum is 50%. We therefore tested an additional calibration that
moves inactive units into a range where they can learn. A fixed correction keeps
their existing predictions intact. No learned weight or optimizer history is
discarded.

The first two activation treatments produced unstable errors and were rejected.
Scaling the revived units to the sensitivity of the layer's existing responsive
units kept every unit responsive, with evaluation error averaging 0.00670 over
the final 20 private updates. The same 32 examples scored 0.00844 before the
regression; the simpler repair scored 0.00589. These evaluation examples also
helped select the treatment. The full continuation below tests the treatment
with the complete learner and its saved simulated bodies.
The initial eye-correction output and input gradients match exactly on the actual
saved head at production precision, and saving/reloading preserves the result.
[Activation comparisons and rejected trials](measurements/visual_activation_trials_2026-09-24.json).

The full continuation preserved every saved learning component and advanced all
eight bodies without resetting them. During its final 100 updates, at least 90.6%
of the eye-correction units responded in every report. The live visual predictions
still averaged 18.6% more error than the running unchanged-picture reference;
the repeat continuation checks both saved-example accuracy and ongoing prediction
quality before approval.

The run failed its response check because a body-prediction layer fell below the
50% requirement in 11 of 20 short reports. Those reports covered five updates;
production counts whether each unit responded at any point in 100 updates.
The final qualification is now running from the original preserved checkpoint
with that production reporting interval and the same 50% requirement. It also
includes the tested remembered-hand-goal stage and faster inverse hand calculation.
A pass will qualify that complete candidate; it will not isolate reporting-window
length as the sole cause of the earlier short-window dips. The failed result is
retained. Production resumes only after all qualification checks pass.

Review also found that the continuous monitor could hide low body-lesson
responses behind higher responses to shuffled test commands. The monitor now
checks these separately and requests a saved checkpoint and stop after two
complete low-response reports. Nine regression tests passed. This reporting
defect did not change learned weights or physical observations.

[Measurements](measurements/visual_regression_2026-09-24.json),
[measured calibration](measurements/visual_input_conditioning_2026-09-24.json), and
[executable audit](audit_visual_regression_2026_09_24.py) preserve the comparison.
The full local output is
`Saved/gof_takeover_20260924/repair/visual_audit64/result.json`.
