<!-- Shared copy of the sentence-by-sentence review notes (body-report-revision-ack-20261023), copied October 23, 2026, 09:58 AM PDT from the reviewing session's scratchpad C:/Users/bhart/AppData/Local/Temp/agent/c--Users-bhart-Documents-UEMA-devA-private/39f9510d-b517-49f4-ab68-2cf712580486/scratchpad/H84_BODY_REPAIR_review_notes_2026-10-23.md (sha256 473784a74ad92fe5e67dfb8b7f85af7e8bc78e806effb78ba21cb091460d6539), byte for byte below this comment.
Source HTML reviewed: docs/H84_BODY_REPAIR_EXPLAINED_2026-10-21.html, sha256 021b36a96c9a9c243bb2590adacf860246caf9bb7622928dcef6af533e725fab (saved October 23, 2026, 5:32 AM PDT).
The page after the operator-directed revision that these notes led to: sha256 5ba2f24acc4a16dea333b2be2c88db648afdce02c830b019a853a9298a321a1c.
Review date: October 23, 2026, 9:21 AM PDT. -->

# Review notes: docs/H84_BODY_REPAIR_EXPLAINED_2026-10-21.html

Reviewed version: sha256 021b36a9..., saved October 23, 2026, 5:32 AM PDT.
Reviewer: the builder, at the operator's request, October 23, 2026, 9:21 AM PDT.
Rules applied: the style guide (mechanism not metaphor; one word per thing; every check with its facts; numbers with
their meaning and scale; no vague size words; reading gate), AGENTS.md (lead with the consequence; no repeated
disclaimers), and the document hook .agent/hooks/worklog_time_check.py (it refuses the phrases marked HOOK).

Quoted page text is in backticks. The section labels are the page's own ids.

## Structure (the main cause of the reading load)

- S1. The page says `This page follows that same arm from the command to the result, then shows the repair`, but
  its first two sections (a2-diagnosis, consistency-lesson) are the newest work, placed before the screenshot. The
  reader meets "predictor", "goal", "combined pose error", "IK" and "fitted arm" before any of them is defined.
- S2. Numbering: the two newest sections carry no number; section 1 is the fourth section; "joint-calculation" sits
  between 3 and 4 with no number; "7" is hidden inside the collapsed not-adopted proposal; the nav lists 8 links and
  leaves out the two sections that come first.
- S3. The newest status of five repairs (`Latest report: ...`) is inside the collapsed "Earlier proposal" block,
  under a heading that says the proposal was not adopted.
- S4. Section 6 (48 records of the stopped run of October 22, 47 of 48 unverified) is not marked as superseded by
  the 72 matched records of October 23, which section a2-diagnosis presents first.
- S5. The footer's update notes run newest-to-oldest, but one is labelled `latest decision` (it is the 2:28 AM one;
  eight later decisions exist) and another says `see the newer decision below` while the newer one is above.
- S6. One thing, several names: the score is `combined pose error`, `pose-error score`, `combined difference` and
  `body error`; the fault is `cannot fit together`, `arm-shape error` and `inconsistency`; the hand point is `hand`,
  `palm`, `right palm` and `wrist`. The tab title (`MIMo: what broke, and what the repair must change`) differs
  from the H1 (`Why the arm missed - and what we are repairing.`).
- S7. Two near-duplicate phone instructions: `On a phone, swipe the picture sideways to see both arms.` and, in the
  caption right below, `On a phone, swipe the picture sideways to read both arms.`

## Errors of fact or of currency (checked against the files on October 23, 9:21 AM PDT)

- E1. Status box: `The next analysis uses those saved records; training is paused.` The analysis ran at 5:23 AM
  (mailbox a2-saved-record-diagnosis-result-20261023). After it: the lesson was built and checked (7:10 AM), its
  elbow term was repaired and re-checked (7:46 AM), a matched learning pair was packaged (8:49 AM) and authorized
  (8:50 AM and 9:07 AM), and a binding blocker was reported (9:06 AM). The box is four hours behind the bridge.
- E2. consistency-lesson: `Approved for implementation and a short code check. Not yet enabled for training.` and
  `the builder must report completed changes and any blocker by 5:50 AM PDT on October 23` are past; see E1.
- E3. a2-diagnosis details: `the builder will identify those records and recalculate the totals with and without them`.
  Done at 5:46 AM (mailbox a2-consistency-checkpoint-0550-20261023): the three records are plan step 496,270 body
  2, 496,280 body 2 and 496,300 body 3; without them, on 63 changed bodies of 69, the end score was below the start
  in 39, the palm points ended nearer in 43 (left) and 33 (right), and the body gap fell in 42 of 69.
- E4. a2-diagnosis: `The goal is the hand position in an earlier saved body pose.` The goal is the whole earlier
  saved pose (25 body parts, positions and turns); the hand point is one part of it (physical_goal.py
  pose_error_parts at f0183150b).
- E5. gaps, second card: `Repair: use allowed joint bends to check all the predicted arm parts together. The next
  drawing explains this.` That repair (the IK check at every choice) was not adopted; the adopted repair is the
  loss term in consistency-lesson. The "next drawing" is inside the collapsed block two sections later.
- E6. a2-diagnosis: `Grey rings show where an allowed, connected arm can put those parts.` Overstated. The grey arm
  is one fitted arm from a local fit; the 5:25 AM decision requires the label "a fitted allowed arm", not the
  nearest legal arm.
- E7. plan, step 4: `For the next draft run, disable both scheduled videos and event-triggered videos.` Done before
  the run of October 23 (`--video-every-min 0`, `--moment-video off`; mailbox
  a2-preparation-consolidated-result-20261023).
- E8. plan callout: `the builder is now assigned to compare the predicted arm shapes with those actual joints`. Done
  (5:23 AM).
- E9. footer: `Final source-package review remains open.` The package was reviewed and frozen at 8:49 AM
  (mailbox a2-final-package-20261023).
- E10. execution: `The closest bend still left about 58 degrees of difference.` The record holds 57.83 degrees
  (provenance.json hinge_residuals upperarm_L).
- E11. a2-diagnosis details: `The 72 outcomes come from nine recorded moments in eight simulated bodies.` The word
  `moments` is a banned synonym; they are nine planning model steps (496,260 to 496,340). HOOK.
- E12. Status, a2-diagnosis (three places) and the gallery text: `moved closer` / `move closer` / `moving closer`.
  The document hook refuses the phrase as a hidden operation; the measured thing is a distance at the end below
  the distance at the start. HOOK.
- E13. gaps, second card tag `Tests passed; calibration unresolved` describes the arm-fit code, but the card is
  about the predictor fault. The tag and the card do not match.
- E14. The source-code list names only the launched checkout f0183150b. The run of October 23 used 7dd0eabda and
  the lesson is at 71841320b (arm_consistency.py); neither is listed. The diagnosis provenance and captions files
  (a2_diagnosis_provenance.json, a2_diagnosis_captions.txt) are not linked.

Checked and found correct: the seven joint angles and limits (joint_detail.json); 7.5, 14.6 and 7.4 cm; 0.15 s
(3 model steps); the motor formula (decoder_kp 6.0 in the run's args.json, kd 0.3 default, position weight 8 per
metre; action_decoder.py 435-505); the score formula (1.80 cm, 11.0 degrees, 25 parts; physical_goal.py 15-23); the
keep rule (physical_goal.py 49-59); the a1 numbers (pose_rule_cases.json); 0.0000038147 cm = one float32 unit
(mailbox pose-learning-repairs-verified-20261022); 392 of 392, 0.811 and 0.733 cm; 274 files; 15 of 15 tests; the
calibration refusal; 100 updates, 496,250 to 496,350; 43, 45, 44, 38, 25 of 72; 7.49 cm; 19.48, 16.62, 23.63 cm.

## Sentence by sentence

### header
- `Why the arm missed - and what we are repairing.` A title with a trailing full stop; `we` is undefined (the overseer
  wrote the page; the operator reads it).
- `The hand did move: it moved farther from the earlier hand position used as the goal.` First use of `goal`
  with no definition.
- `the builder reports the command-handling and saving repairs passed their checks.` A check named without its facts
  (which checks, how many conditions, when). Rule: never name a check without its facts.
- `The IK/FK arm code passed its 15 tests.` `IK/FK` is first used here and defined two screens later, inside a
  collapsed block.
- `Its calibration check failed, so it remains outside the planner.` A failed condition without its own words
  (which check, what was measured: 33 of 100 left and 20 of 100 right fits ended with a gradient above the
  1e-10 radian line).
- `The short learning run finished: 100 learning updates and all eight worlds were saved.` `short` is a duration
  word; the run is unnamed (model step 496,250 to 496,350, October 23, 2:52 AM PDT).
- `The body moved closer to its earlier goal in 43 of 72 recorded movements.` HOOK (`moved closer`); the quantity
  is the pose-error score of 25 parts, not "the body"; the next section gives 45 of 72 for a distance, so one thing
  appears to have two numbers.
- `New calculated pictures on this page are not presented as successful reaches.` A disclaimer with no referent
  (which pictures).

### a2-diagnosis
- `The predictor sometimes draws an arm that cannot fit together.` `draws` is a metaphor; the predictor outputs a
  position and a turn per part. `predictor` is used here and defined in section 4.
- `Fixing that error is the next small repair.` `small` has no measure; `next` is stale (E1).
- `It has not yet explained why the body misses so badly.` `so badly` is a size word; the measured miss is 23.63
  cm against 19.48 cm at the start.
- `The predictor guesses where ...` / `It guesses each part separately.` / `Its guessed elbow ...` Software
  described as a person (`guesses`). Mechanism: the body head outputs 6 numbers per part from one shared vector,
  with no link length or hinge axis in the computation.
- `We checked 72 saved movements from the latest learning run.` `we` and `latest learning run` undefined; a
  "saved movement" is not defined (one command on one body with its start record, two predictions and end
  record).
- `In the largest-disagreement example, the predictor said the right hand would move closer to its goal.` HOOK
  (`move closer`); `said`; the selection rule for "largest disagreement" is not stated.
- `The goal is the hand position in an earlier saved body pose.` E4.
- Select labels `Lower combined pose error` / `Higher combined pose error` name a score defined only in section 4,
  under another name (S6). `typical` is not defined (the median-ranked outcome).
- `A lower score can still hide a hand moving farther from its goal.` Fine.
- `Grey rings show where an allowed, connected arm can put those parts.` E6.
- `The 72 outcomes come from nine recorded moments in eight simulated bodies.` E11.
- `Left palms moved closer in 44 outcomes; right palms in 38; both in 25.` HOOK; the measure is not stated.
- `That was not a count of wrists moving closer.` HOOK; refers to a label (`wrists toward in 45/72`) the reader
  never saw.
- `That can leave a small gap even in a goal picture.` `small` without its number (up to 0.26 cm in the goal
  panels).
- `For three saved records, the 5-millisecond delay did not explain the difference.` Which three, which
  difference: E3.
- `Larger arm-shape errors were only weakly associated with worse movement.` `weakly` without the number (rank
  correlation 0.14 to 0.27, where 0 is no relation and 1 is perfectly in step); third name for the fault (S6).

### consistency-lesson
- Heading `teach the predictor how an arm joins together`: metaphor; the mechanism is a loss term.
- `Approved for implementation and a short code check.` E2; `short`.
- Item 1: `lesson` is used without a definition (a loss term: a number the trainer lowers by changing weights).
- Item 2: the readings the penalty uses are not named (shoulder link, upper-arm link, elbow off-axis, hand ball),
  nor their units.
- Item 3 `Let the predictor change its own answers.` Metaphor. Mechanism: the term's gradient reaches the body
  head's weights only.
- Item 4: now measured (off mode equals the old code bit for bit; weight 0.2 from the screen); the page still
  reads as a plan.
- `Why not run IK for every choice? IK searches for joint bends ...` `IK` is used before its expansion.
- `the builder must report ... by 5:50 AM PDT on October 23` E2.
- `legal arm poses must receive almost no penalty` `almost no` is a size word; the declared line is known (the
  supervised loss at most the control's times 1.02).

### original (section 1)
- `The planner compares possible movement commands.` Incomplete: compares by predicted score; the score is defined
  three sections later.
- `The controller converts those requests into motor controls.` The operation is hidden here; section 3 holds it.
- `The blue dot marks the hand position requested by this command; it is different from the green square.` The
  measured gap is in the data (23.0 cm from the hand's start) and not given.
- `It moved 7.4 cm, largely the wrong way for this comparison.` `largely` is a vague size word; the gap grew from
  7.5 cm to 14.6 cm.
- The rest of the section passes.

### joints (section 2)
- `The shoulder nearly reached its limit.` `nearly` without the number in the heading (0.3 degrees from its
  limit).
- `One shoulder turn moved from 18.1° to −27.7°` Which turn (the forward/back turn).
- `The elbow also moved farther from its earlier bend.` The numbers exist (earlier −68.0°, start −50.2°, result
  −35.8°) and are not given.
- The rest passes.

### execution (section 3)
- Flow box `find a compromise between the requested hand position and all three requested turns`: `compromise`
  hides the operation (a damped least-squares solve of seven joint changes that minimizes the strength-weighted
  hand position error and the three turn errors).
- `The closest bend still left about 58 degrees of difference.` E10.
- `The command asked for a hand point below the floor.` The number is in the record (0.47 cm below).
- `The assigned repair records the complete prediction and joint movement so the next comparison can locate the
  disagreement.` Stale: the run of October 23 recorded them.
- `That matters for prediction, below.` Points three sections ahead.
- The rest passes (the motor formula is correct).

### joint-calculation (unnumbered)
- Eyebrow `Joint-level illustration of the proposed body check`: `proposed body check` is the not-adopted IK
  proposal; the section is useful on its own (which joints had to change) and should not be tied to it.
- `They do not predict time, force, balance or contact.` `balance` is a sense synonym under the one-word rule.
- The rest passes.

### rule (section 4)
- `The predictor is a learned program ...` The definition comes after three sections that used the word.
- `A command could get a lower total mainly by requesting less strength` `mainly` without the numbers (strength
  cost 0.076 to 0.009; goal term 0.099 to 0.082).
- `This combined difference is the pose-error score` S6.
- `The proposed arm check must cover it too.` Refers to the not-adopted proposal.
- The rest passes (formula and keep rule verified).

### gaps (section 5)
- Tag `Repair reported built`: not one of the repair states; `reported` hedges; the check facts are missing (16
  of 16, then 27 of 27 conditions).
- `the builder reports this repair and its checks are complete.` A check without its facts.
- `For example, it could place the hand farther from the elbow than the forearm and hand can span.` A hypothetical
  where measured facts exist (392 of 392 older predictions; median 1.02 cm on the new ones).
- `Repair: use allowed joint bends ...` E5. Tag mismatch E13.
- `the finding below describes that data` points into the collapsed block.

### evidence (section 6)
- No sentence says the 72 matched records supersede these 48 (S4).
- `These 48 records came from the stopped learning run.` Unnamed (the run of October 22, ended while its
  checkpoint was being written).
- `The position differences were tiny: at most 0.0000038147 cm.` `tiny`; it is one float32 rounding unit.
- `A new full-avatar forecast picture cannot be reconstructed from these summaries.` True for these 48; the run of
  October 23 recorded full predicted poses and the gallery shows them, which the sentence does not say.
- The rest passes.

### plan (earlier proposal)
- `Latest report:` S3; `the calculated arm matched closely` `closely` without the number (largest residual
  0.00000504 cm); `the builder is now assigned ...` E8.
- `The predictor then cannot prefer a command` `prefer` describes software as a person.
- `the builder reports checks for both paths` a check without its facts.
- `by more than the small error in the measured positions` `small` without the number (0.004 cm).
- `The typical elbow-position gap was about 0.8 cm on the left and 0.7 cm on the right.` `typical`, `about`: the
  medians are 0.81 and 0.73 cm.
- `the builder reports the stop/save repair passed checks against the recorded timing.` Without its facts (46 of 46
  conditions).
- `the overseer also required more frequent writes of detailed measurements.` The mechanism is hidden (the recorder
  writes its chunk every 5 model steps instead of every 100).
- `For the next draft run, disable ... videos.` E7.
- The IK and FK definitions and the drawing pass, but they are the only definitions on the page and are collapsed.

### checklist (section 8)
- Eyebrow `The checklist I should have used`: `I` is the overseer; the reader cannot tell.
- `Repeated preparations consumed hours.` No measured time.
- The rest passes.

### reviews
- `This replacement explains the arm calculation, labels completed work separately from proposed work, and keeps
  the original movement evidence.` A self-assessment with no evidence (AGENTS.md: a score must have evidence).

### footer
- S5, E9, E14.

## What the revision does

1. Puts the sections in the sequence a reader needs: the screenshot (1), the joints (2), the command to motion (3), the
   calculated shapes (4), the scoring change (5), the two faults with the state of each repair (6), the learning
   run and its 72 matched records (7), the repair under test with its status as of 9:23 AM PDT (8), the
   not-adopted proposal (9, collapsed), the superseded 48 records (10, collapsed), the checklist (11), the review
   record (12), sources by date.
2. Defines each term where it is first used: record, goal, hand point, command, controller, predictor, pose-error
   score, loss term, IK, FK, fitted allowed arm.
3. One name per thing: pose-error score; hand point; arm inconsistency; model step.
4. Every check carries its count, its time and its row name; every number its meaning and scale.
5. Removes the phrases the document hook refuses and the vague size words.
6. Keeps every section id the browser check expects (browser_check.py) and every asset.
