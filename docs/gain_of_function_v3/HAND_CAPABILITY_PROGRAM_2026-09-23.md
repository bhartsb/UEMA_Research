# Hand use: findings and experiments

**A private control correction makes the fingers follow a chosen hand pose more
accurately while the wrist moves.** Across eight saved bodies, average finger-joint
error after one second fell from **29.1° to 13.3°**. The correction keeps the chosen
finger posture relative to the actual wrist as it moves. It preserves the wrist's
command. The first object test found that a generic curl pushed the cube onto
the forearm. The next correction must shape the thumb and fingers for the object.
Production adoption also requires prediction and command records to use the same
wrist-relative finger goal.

The programme covers useful hand actions through the range seen in children up to
24 months. Their ages supply examples of abilities, not a training timetable.

**What the new tests settled**

| Finding | Meaning and action |
| --- | --- |
| Both hands closed to within 1.28° of the requested bend angles on average. | The normal controls can produce a coordinated curl when targets follow the wrist. Use that as a physical reference for learned choices. |
| Both hands reopened to within 1.48° of straight on average after six simulated seconds; the largest joint error was 4.79°. | Extension works. The earlier two-second trial ended with the little fingers still bent, so a single short endpoint was an inadequate extension test. Score the full motion and settling time. |
| Extension to the joint limits averaged 0.79° error through normal controls. Removing finger collisions gave exactly the same final angles. | Contact obstruction did not cause this fixture's delayed opening. Keep collisions enabled in function tests. |
| A full individual index curl moved other fingers by 5.65° on average; middle-finger curls moved them by 14.17°. | The model has explicit tendons coupling neighbouring knuckles. Test selective movement within those limits rather than demanding that a fully curled finger leave all neighbours straight. |

These are **physical command tests**: the researcher supplied the targets. A learned
ability requires the model to choose the action, achieve it repeatedly, keep it
through further learning, and use it from changed starts. Save/restore separately
checks that storage preserves the behaviour. Dashboard progress should show those
stages separately.

**Who requests the impossible positions, and what happens?**

The learned motor controller proposes separate world positions and rotations for
the wrist and each finger. On planning events, the planner adjusts those numbers.
Exploration can also perturb each target separately. Smoothing limits sudden changes
but does not enforce the hand's geometry.

In three saved moments, every finger/wrist pair was too far apart already in the
controller proposal. All remained incompatible in the filtered and sent commands:
30 pairs at each of three processing stages. The sent positions demanded 7.1–40.8 cm
of separation; the fingers' link lengths bound it to 5.3–6.0 cm.
These targets track the base of each finger's final segment, not its skin tip.

The decoder solves the arm and each finger separately. Each finger receives a
compromise between its position and rotation requests, clipped to joint and motor
limits. It cannot move the wrist to reconcile their disagreement. The body therefore
executes a compromise. The correction must first improve physical control;
an equal-experience learning comparison will then measure its effect on acquisition.

The private correction finds one hand posture within joint and coupled-knuckle
limits, then builds all finger targets relative to the requested wrist. It reduced
the geometric conflicts from 90 to zero across the three recorded processing
stages. The median finger target moved 14.6 cm to become compatible: this would be
a material change in the command, so it must also appear inside prediction and
planning, with a new command-version tag. A last-second hidden correction would
teach the model a different action from the one it chose.

The first physical comparison used eight saved bodies for 250 ms each. Compatible
fixed targets left **36.71 cm** of average error from the original finger requests,
versus **36.56 cm** for the original commands. The correction therefore failed
this physical gate. The wrist had its own movement to complete; fixing the fingers
at the *requested* future wrist position may preserve a moving-hand conflict.
The follow-up kept the same finger joint goal and wrist command but rebuilt finger
world targets from the **actual moving wrist** every 10 ms. In one second, it
closed **20.3°** of the gap to that hand posture, versus **4.5°** with the original
commands. Holding the compatible finger targets fixed in the world instead moved
**0.7° farther away** from the joint goal. This makes tracking the moving wrist
the candidate to develop. Wrist arrival remains a separate control problem:
the wrist still averaged about 29 cm from its requested position in each condition of
the comparison. The grasp fixture therefore uses a coherent whole-arm reference,
with wrist and arm targets generated from one attainable set of joint angles.
[Physical comparison receipt](../../Saved/gof_takeover_20260923/planner_investigation/synergy_physical_372375_v2/result.json).
[Moving-wrist comparison](measurements/hand_reference_frames_2026-09-23.json).

**Training impact.** Existing records still describe movements that happened and
remain useful for learning perception and actual outcomes. They do not demonstrate
accurate execution of the requested hand pose. Keep the learned state. The
moving-wrist path improved hand-posture accuracy; now measure how quickly the
action predictor adapts under that new command meaning. Do not assign an invented percentage
of “lost training.”

**The hand programme**

The table defines the test catalogue. Components can be exercised in parallel;
an object-handling test depends on its required movements and contacts working.

| Movement or function | What the test must observe |
| --- | --- |
| Finger flexion and extension | Close, reopen and extend; record each joint and motion duration. |
| Individual fingers and finger combinations | Selected fingers move; measure intended and unintended neighbouring motion. |
| Finger spreading and closing sideways | Increase and decrease separation, separately from bending. |
| Thumb flexion, extension and positioning | Move the thumb across, away from and alongside the palm. |
| Thumb opposition and return | Turn the thumb to face a fingertip, make controlled contact, then separate. |
| Palm shaping | Cup and flatten the hand within the model's available joints. |
| Wrist bending and sideways movement | Flex/extend and move toward thumb/little-finger sides while keeping the chosen grip. |
| Forearm rotation | Turn the palm up/down while preserving the finger goal. |
| Reach and hand shaping | Approach an object with the opening and orientation suited to it. |
| Touch, press, stroke, rake and point | Select the contacting surface or finger and control contact force. |
| Whole-hand grasp | Hold an object using fingers and palm, then lift it without support. |
| Thumb–finger precision grips | Hold between thumb and index; progress to other finger combinations. |
| Grip adjustment | Change force or finger placement when the object slips or changes load. |
| Release and placement | Open deliberately and leave the object in a target area or container. |
| Object exploration and manipulation | Turn, reposition, shake, bang or slide the object for a chosen effect. |
| Transfer between hands | The receiving hand takes the load before the first releases it. |
| Two hands with different jobs | One stabilizes an object while the other turns, removes or inserts a part. |
| Object use | Operate a button/knob, stack or insert objects, and orient a simple tool. |

Infant research distinguishes whole-hand use from specialised fingertip use,
object-dependent hand opening, manipulation and two-hand coordination. These
findings motivate the separate tests above. Sources:
[Hohlstein's longitudinal grasp study](https://journals.sagepub.com/doi/abs/10.5014/ajot.36.3.170),
[Geerts et al., manipulation and insertion](https://pubmed.ncbi.nlm.nih.gov/14652162/),
[Nelson et al., different roles for the two hands](https://pmc.ncbi.nlm.nih.gov/articles/PMC3615031/),
[targeted self-touch and opposing-thumb grips](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.00009/full).
The CDC catalogue includes thumb–index pickup and placing objects in containers
[by one year](https://www.cdc.gov/act-early/milestones/1-year.html), and holding with
one hand while using the other, switches, knobs and spoon use
[by two years](https://www.cdc.gov/act-early/milestones/2-years.html).

For broader coverage, the [GRASP taxonomy](https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf)
distinguishes 33 static, stable, one-hand grasp types. It supplies an expansion
catalogue; those 33 types are not all infant milestones. The current simulator's
joint inventory determines which can be attempted without changing its anatomy.

**Object handling: what ran and what changes next**

The left-hand fixture compared closing/lifting/opening, holding the fingers open,
keeping them closed, and an exact repeat. It used a free **3 cm cube**, half the
production cube's side length, with no attachment or caregiver. The reachable arm
reference and collision-free start passed their setup checks.

Closing moved the cube onto the forearm. That contact remained through the lift,
so the hand did not pass its grip test. The open-hand control carried the cube;
its contact pattern also shows why contact counts alone cannot qualify a grip.
The full closing trajectory repeated exactly. Opening did not cause a release.
[Measurements and raw-source hashes](measurements/grasp_fixture_2026-09-23.json).

The next work has three concrete steps:

1. Integrate the successful moving-wrist hand reference into a versioned command
   path used consistently by prediction, planning, execution and saved experience.
   Compare equal amounts of learning against the current path without resetting
   the learner.
2. Fit opposing thumb and finger contact to the object's shape. Compare that grip
   with open fingers through the same lift and palm tilt. Require hand-only support
   and retention when gravity pulls the object off the palm before grading release.
3. Repeat both hands and the production 6 cm cube, then test learned selection,
   pickup, placement, further-learning retention and changed starts.

The first step has priority: it connects a measured control improvement to the
existing learner. [The action register](HAND_PROGRAM_ACTIONS_2026-09-23.json)
records its acceptance checks and follow-up triggers. Physical scripts are
`compare_hand_reference_frames.py` and `audit_grasp_release.py`.

**Why pursue faster learning now?**

Fast useful learning is an engineering target. Smith, Kostrikov and Levine trained
a real quadruped to walk in 20 minutes, with action-space design and implementation
central to the result. That makes our incompatible action requests a concrete
algorithm issue to test, rather than a reason to wait for an infant-like schedule.
[Their experiment](https://www.roboticsproceedings.org/rss19/p056.pdf).
Zhu et al. reported several real-hand manipulation tasks learned in 4–7 hours,
using direct task objectives. The relevant comparison for UEMA is acquisition of
a specified ability per unit of experience; those robots do not supply a deadline
for this model. [Their experiment](https://arxiv.org/abs/1810.06045).

**Rolling: an active physical reference**

The 96 additional command sequences reached **116.75 degrees**, up from 64.52 degrees, with
exact replay of the best final state. The goal remains a 150 degrees turn followed by
half a second supported near belly-down. Extending the best sequence to twelve
seconds, shortening holds, removing smoothing, and combining those timing changes
all failed to complete it. Timing alone did not fix this sequence.

The next comparison tests coherent joint targets against the current independent
world-position targets. A separate task-learning reference is specified in
[the rolling experiment](ROLLING_LEARNING_EXPERIMENT_2026-09-23.json); its learner
wrapper still needs implementation. It will diagnose learnability on this exact
body and will not replace the preserved UEMA learner. The measured hand-interface
improvement is the first integration task.

There is a directly relevant engineering baseline. Philipp, López and Triesch's
[June 2026 MIMo rolling study](https://arxiv.org/html/2606.17456v1) trained joint
control with an explicit rolling objective and found that available strength
strongly shaped performance. Their body ages, motor interface and 10 ms control
interval differ from our configuration. The declared private learning comparison isolates the command interface on
**our** body, with equal experience and matched physical starting states.

Exploration also needs a measured comparison. A
[June 2026 study of infant-inspired movement noise](https://arxiv.org/html/2606.16590v1)
found task-dependent benefits from moderate temporal correlation. Our smoother
and command holds already correlate movement over time. The follow-up therefore
compares their actual movement with learned synergy and feasible, persistent
exploration at equal movement and contact-force budgets, then measures acquisition
of selected hand actions and supported turns.

**Reproduce and inspect**

- [Physical test script](audit_hand_repertoire.py): 17 two-second stages, then
  five six-second stages with normal finger contacts and the same five with finger
  collisions removed only for diagnosis. No learner was loaded.
- [Command-conflict calculation](audit_hand_target_conflicts.py): joint limits and
  actual coupled-knuckle constraints, against all three saved processing stages.
- [Committed measurements](measurements/hand_capability_program_2026-09-23.json):
  definitions, individual joints, exact settings, source hashes and all results.
- Raw trajectories: `Saved/gof_takeover_20260923/hand_program/`.

The private body used the production settings from checkpoint 369250: age 24,
26 body targets, strength 0.47, 50 ms model steps, 250 ms command holds, and the
production smoothing. All joints remained active. The two- and six-second
batteries used different conditioning durations; their final errors are not a
matched estimate of a learning improvement.
