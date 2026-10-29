# Gain of function: repairs and capability tests

**September 24 update:** this report records the September 23 investigation.
The body-goal connection and moving-wrist hand controls are now implemented.
Training is saved and paused at step 380200 while their qualification and the
visual repair are completed. Follow the [current results and active experiments](RESUME_HERE.md)
and [hand-control measurements](HAND_CONTROL_QUALIFICATION_2026-09-24.md).

**The planner was discarding better commands. That is repaired.**
On September 23, training resumed with the learner, optimizer and all eight worlds preserved.
The repaired runtime passed 300 learning updates on private copies. Production
continued from step 372375, saved another 200 updates during a startup-monitor
repair, and resumed from that new checkpoint in segment 36. Nothing was reset.
[Qualification and restart record](measurements/planner_repair_qualification_2026-09-23.json).

## What the planner investigation established

The old search could throw away its best command and return an unevaluated final
update. It also predicted a smoothed eye command while requesting an unsmoothed
one. Both are repaired. A separate exploration bug overwrote part of a rotation
before finishing the calculation; that is repaired too.

We reran 30 decisions for eight bodies, giving each alternative the same model,
state and goal. All commands were scored through the corrected eye-command path.

| Search | Choices with more predicted goal error than the starting command |
| --- | ---: |
| Old search and old eye-command assumption | 118 / 240 |
| Correct eye-command assumption | 14 / 240 |
| Also retain the best evaluated command | 2 / 240 |

The remaining two choices saved commanded strength at the cost of 2.6% and 0.9%
more predicted goal error. Removing that penalty produced zero such choices.
Once body-guided choices pass a physical test, compare effort weights on the same
reachable goals, recording arrival and contact forces before changing the penalty. These 240 comparisons are a new controlled experiment;
they do not replace or reclassify the earlier 485/800 observations.

**The planner is not using the dedicated body-movement predictor to choose commands.**
It uses the dedicated eye predictor, but scores body goals through an internal
sensory representation instead. We tested whether its forecasts could guide commands to reachable body poses.

| Private test | Average position miss | Average angle miss | Decision |
| --- | --- | --- | --- |
| Use body forecasts directly, eight starts | 1.27 to 1.26 cm | 7.17 to 7.07 degrees | Improvement varied across bodies; keep testing before adoption. |
| Remove forecast bias using the actual future endpoint, eight starts | 1.27 to 1.00 cm | 7.17 to 6.34 degrees | Identifies forecast bias as a control problem; this future information is unavailable during live choice. |
| Correct using only the last completed movement, 24 trials | 1.14 to 1.18 cm | 7.13 to 7.52 degrees | Worse than the original controller; reject this correction. |

Each row compares its candidate with the starting controller from identical states
and goals. At step 375250, recheck forecasts after stored practice movements from before the arm repair have been replaced.
If bias persists, fit a correction that accounts for the current state using
completed earlier movements, then test on separate later starts. A successful path
also needs remembered physical goals connected to the planner; more training alone
will not install that connection.
[Body-choice experiments](measurements/body_prediction_choice_2026-09-23.json).

Every five minutes, the live fault watcher checks the retained planning records.
A worse total score for any body, missing scores or a non-finite search will trigger
a snapshot and stop.
Goal-versus-strength tradeoffs remain visible as an investigation, rather than
being hidden in the average. [Comparison data](measurements/planner_repair_comparison_2026-09-23.json).

**Synergy did not pass its movement comparison, so both control switches stay off.**
We tested planned and exploratory use from the same eight saved bodies, including
comparisons with matched command sizes. Neither met the required improvement.
The movement map keeps its learned weights. Its next adoption test depends on the
body-prediction correction or compatible-hand interface passing a physical test.
Then compare synergy against the existing controller on the same attainable goal.
The step-375250 recovery check is the next scheduled review of this dependency.
[Physical comparison](measurements/synergy_physical_2026-09-23.json).

## Hands: correct the commands, then test useful object handling

**The motors can close and reopen the hands. The requested hand poses are the
immediate coordination problem.** The learned controller independently requests
wrist and finger positions. Every finger/wrist pair in three inspected snapshots
was too far apart to reach. Planning and filtering retained those conflicts.
The decoder compromises between the requested finger position and orientation;
it cannot satisfy both by moving the wrist.

We built and ran a replacement target calculation that derives all finger targets
from one possible hand pose, including the tendons that couple neighbouring
fingers. It removed all 90 inspected geometry conflicts. Adoption requires the
predictor and planner to use the same compatible
commands; silently changing commands only at the motors would create another
prediction mismatch.

Removing the geometry conflicts alone did not improve following of the original,
unreachable finger positions. We then tested one feasible hand posture while the
wrist moved. Updating the finger targets from the moving wrist closed 20.3 degrees
of the joint-angle error in one second, against 4.5 degrees with the original
commands, averaged across the eight bodies. That supplies a useful private control
reference. The wrist itself remained about 28.9 cm from its requested position;
the body-goal work must address that separate reaching problem. Production adoption
also requires the predictor and planner to use the new command meaning.
[Moving-wrist comparison](measurements/hand_reference_frames_2026-09-23.json).

Longer opening tests reached within 1.5 degrees of straight on average, and within
0.8 degrees of the full extension limits. Removing finger contacts did not change
the result. The earlier short test was underestimating opening, so the hand tests
now record the movement over time rather than grading only a two-second endpoint.

The [hand programme](HAND_CAPABILITY_PROGRAM_2026-09-23.md) covers 18 categories:
opening and closing, selective fingers, thumb opposition, spread, wrist and forearm
control, reach shaping, touch, grasp, force and slip control, release, manipulation,
transfer and two-hand use. The free-object test has now run. Closing around a
3 cm cube moved it onto the forearm; the open-hand control carried it instead.
That identifies the next control problem: fit the thumb and fingers to the object,
then test whether the grip survives a palm tilt and opening releases it. This
scripted physical test used the left hand; it did not grade learned grasping.
[Object-test results](measurements/grasp_fixture_2026-09-23.json).

## Rolling and faster learning

The 96-sequence search reached 116.8 degrees of turn, extending the earlier
64.5-degree result. It did not reach the goal of 150 degrees followed by half a
second supported near belly-down. Seven follow-up trials tested longer execution,
shorter command holds and removal of smoothing; none completed that roll.
The next comparison changes the command interface to coherent joint poses in a
private copy, to test whether world-position requests are limiting the movement.

A [2026 MIMo study](https://arxiv.org/html/2606.17456v1) learned rolling with explicit
progress rewards and direct joint control. Its bodies and control timing differ
from ours, but it supplies a concrete experimental reference. Human developmental
ages guide our ability catalogue; they are not a waiting period for this learner.

## Impact on accumulated training

| Defect | Effect and recovery |
| --- | --- |
| Wrong search selection and eye-command assumption | Could choose poorer commands. Preserve learned weights and recorded outcomes; qualify corrected choices, then measure physical results. |
| Exploration rotation overwritten during calculation | Distorted the sampled movement directions. Actual command/outcome records remain useful. Retest exploration and synergy using the corrected baseline. |
| Contradictory hand targets | The requested pose cannot be achieved. Preserve experience, compare compatible targets, and train predictions on the adopted command interface. |
| Earlier arm equations and torque scaling | Omitted shoulder effects and distorted some joint forces until segment 34. Keep the learner and test adaptation under the repaired mechanics. |
| Report omitted a plan when a later decision chose no action | Hid completed planning activity; did not erase learning. Reports now retain the executed plan and count each decision. |
| Rounded comparison scores | Could make a reported choice appear inconsistent with its score. Reports now retain full precision; command selection used the unrounded values. |

These repairs preserve the existing learner. Changed controls require fresh
experience and repeated capability tests. The dashboard separates physical capacity,
chosen ability, repeatability, retention after further learning, and performance
from new starts. Its repair timeline records which measurements preceded each fix.

The first arm-recovery check is the saved checkpoint at or above step **375250**.
By then, each body's 6,000-observation rehearsal memory should contain only
experience after the arm repair, provided recording stayed continuous. The milestone
is 2,875 updates after the saved stop; the dashboard shows the remaining count.
Rerun body prediction, chosen movement
and hand-coordination tests there; the dashboard calculates the training time from
the measured update rate. Learned weights and recorded history remain preserved.

Open hand and rolling work is tracked in
[the action register](HAND_PROGRAM_ACTIONS_2026-09-23.json), with evidence, training
impact and explicit next-test triggers. The
[planner action register](PLANNER_ACTIONS_2026-09-23.json) tracks the missing body-goal
connection, rejected calibration, synergy adoption and the recovery checkpoint.
