# Self-touch test in three layers (agent record, 2026-09-25)

Task: split the scheduled self-touch comparison
(`docs/gain_of_function_v3/probe_saved_self_touch_choices.py`, scheduler task `chosen_self_touch`)
into three layers reported separately, so a round in which no command can touch says "not
reachable from these starts" instead of "did not beat".

## Why

In the round at step 399,100 no candidate command touched the chosen skin patch in any of the 16
later starts (controller, holding still and the mean of four explorations all 0 of 16), so the
round could not tell a choice apart. An outside reviewer: that tests reachability and search, not
self-touch prediction.

## The three layers (result version `self_touch_layers_v4`, endpoint version 6)

A "new touch" is a self-contact that begins (a nonstructural onset: ordinary contact between
neighbouring parts, such as the thumb resting in its own palm, does not count) on a skin patch that
nothing touched at the saved moment. A "body area" is one of the world's named skin regions (a
hand, the trunk, an arm, a leg, the head; every onset records its two). **The self-touch under test
is a hand touching another body area** (`hand_to_body`): a hand region on one side and, on the other,
a region that is neither that hand nor that hand's own arm, which is the trunk, the head, a leg, or
the other hand or arm. This is the rule `hand_targets` uses to choose the reach targets. Every other
new touch (a finger on its own hand, an upper arm on the trunk, a hand on its own forearm) is scored
beside it and never decides the pass. A "pair" is two commands from the same saved moment: one made a
new touch on a patch within 50 ms, the other left that patch with no contact from the body's own parts.

1. **Reachability.** At each saved moment, does some command make a new touch within the contact
   predictor's 50 ms horizon, and does the moment offer a pair? Counted for a hand touching another
   body area and for every touch, at 50 ms and with every command held for its whole 250 ms order,
   with the region pairs of every new touch (`new_touch_region_pairs_50ms`). Moments that offer no pair
   at 50 ms are reported as not reachable and left out of layers 2 and 3.
2. **Prediction.** Does the contact head (its probability that an own part touches a patch within
   50 ms) foresee the touches? 50 of 100 is a coin flip; the goal line is 80, the lower end of "80 to
   90 = a ranking a planner can optimize" in `mimo_bridge/tools/command_ranking.py`. **Judged two
   ways, both of which must be above 50:**
   - *Whether*: pairs in which the touching command asked no body handle for a bigger position change,
     turn or strength than the other command did (`asked_move`: the size of each handle's asked move
     as `contact_head.order_features` reads it, every body handle, the eyes left out). Does the head
     give the toucher the higher chance on the touched patch? A head whose chance on every patch never
     falls as any handle's asked move grows gives the toucher at most a tie on these pairs, so it
     scores at most 50. Direction is not controlled here.
   - *Where*: crossed pairs of two commands of one moment that newly touched different patches, each
     leaving the other's patch without own-part contact (`crossed_pairs`). Half a point for each
     command given the higher chance on its own patch. A head that scores every patch from one summary
     of the command (its size, its displacement along one axis, anything) with gains of one sign favours
     the same command on both patches, so it scores exactly 50. A crossed pair whose two chances differ
     by 0.000001 or less on either patch is left out and counted (`unresolved_crossed_pairs`).
   Reported, not judged: the score over every pair (it can rise with the size of a move alone), the
   region pairs of the touches, and the score with only the forward model's latent varying and with
   only the order input varying.
3. **Selection.** On moments whose planner route reproduces the production planning record and whose
   planner takes its effort per body. For each touching command (not a finger curl, request known), its
   **matched command**: the first command of its own kind (family) from the same moment that touched
   nothing, not already matched, request known, differing beyond the fingers, with predicted wellness
   (the approximate evaluator's five-term sum one moment ahead, on the planner's route) within the
   evaluator's tie band of the toucher's (`causal_chain.TIE_BAND`, 0.0734 on the 0-to-5 sum, 1.5% of the
   scale), and leading by more than the band some other command that the toucher also leads by more
   than the band (predicted wellness makes both preferable to that command). **Judged:** is the
   planner's objective lower for the toucher than for its matched command? The objective is what the
   planner's sensory search minimises (`intention_generator.generate_action`): distance to its
   remembered goal, plus effort = 0.01 x the sum over every handle of the squared strength of the
   command's request as the planner's action space holds it (clamped to 0.0001 to 0.9999), plus the
   prospective-wellness term when the run turns it on (off on H84). The evaluator cannot tell the two
   apart, so general agreement between the objective and predicted wellness cannot favour either: 50 of
   100 is no preference, and the goal line of 80 sits on this share. **What a pass shows:** the planner
   prefers the commands that touched over same-kind commands that did not. It does not by itself
   separate a preference for the touch from a preference for the size of the move that made it, so
   every judged pair records how the two asked moves compare handle by handle (`move_relation`: the
   toucher's move smaller, bigger, equal or mixed) and which sense's part of the goal distance was
   lower for the toucher (`sense_parts`). Reported, not judged: the same comparison without requiring a
   command both lead (`unfiltered`), how often predicted wellness favoured the toucher over its partners
   and the size of those gaps (`wellness_gap_abs`), and why touching commands had no matched command.
   At a moment where the planner's hand stage ran (`hand_descent.stopped_by == 'completed'`; it
   re-chooses the fingers after the planner's search), every command is scored with one shared finger
   goal, the controller's.

**The decision rule, declared before any run.** Each judged score is pooled over its pairs (a tie
counts half) with a 95% interval from resampling whole bodies, each draw pooling its bodies' pairs
(the statistic of `command_ranking.share_interval`; a test holds the two intervals equal), and carries
the goal line of 80 and its distance to it. The verdict is an exact one-sided sign test over bodies: a
body votes with its share minus 50 once it has at least 5 judged pairs from at least 2 distinct touch
events (a start and a touching command; for a crossed pair, a start and its two touching commands),
because the pairs of one event share its predictions. A score is above 50 when, with every voting body
a fair coin, that many bodies above would come up at most 5 times in 100: every one of 5, 6 or 7 voting
bodies, or 7 of 8. Fewer than 5 voting bodies is "not decided" (4 of 4 comes up 1 time in 16). Declared
pass: whether, where and selection all above 50, over the touches of a hand on another body area.

**What the rule can reach (declared in the result, `expected_reachability`).**
- Layer 1: on the 8 saved worlds of checkpoint 400,350 each body's nearest untouched patch of another
  body area was 7.4 to 14.0 cm from a hand. Touching the nearest one within 50 ms needs 148 cm/s; the
  fastest hand closure in the 45 saved approach runs of 2026-09-24 was 2.74 cm in 50 ms (55 cm/s, with
  a deliberately slow 1.5 s ramp). Within a 250 ms order it needs 30 cm/s. So from such moments the
  hand-to-body scores are expected to be "not measured" at 50 ms.
- Layer 2: each judged score needs, in at least 5 bodies, 5 judged pairs from 2 touch events; the
  crossed pairs also need two commands of one moment that touched different patches.
- Layer 3 also needs predicted-wellness gaps beyond the tie band. `causal_chain.TIE_BAND_MEASUREMENT`
  (checkpoint 344,400; five candidate orders: the model's own, holding still, three babble-sized
  changes): the evaluator's predicted difference between two orders was 0.0014 (0.028% of the 0-to-5
  scale) at the median and 0.0039 (0.078%) at the 90th percentile, a nineteenth of the band (1.5% of
  the scale). Unless the reach commands' larger moves open larger gaps, selection is expected to read
  "not measured", and its declared pass cannot be reached until the evaluator resolves the
  differences between commands. Every round records the gaps it saw (`wellness_gap_abs`), and the
  `unfiltered` comparison is measured whenever a matched command exists.

## The commands at each saved moment

Every command goes through the production command contract, starts from the same complete snapshot
(every random stream included), is held for its whole 250 ms order, runs twice and must replay bit
for bit. Each command's executed wrist targets, their shift from where the wrists are, the arms'
executed strengths, the hand joint goal, the move it asked of each handle, and the order it would
execute with the controller's joint goal in place of its own are recorded beside its request.

- The earlier six: the controller's proposal, holding still (every handle requested where it is;
  the order smoother executes 0.35 of that request plus 0.65 of the previous order), four
  explorations with the earlier seeds.
- The order production itself sends at that moment, with its own eye command.
- A finger curl per hand: knuckle, middle and distal joints requested at 95% of their flexion
  limit; the joint smoother executes 0.35 of that request plus 0.65 of the previous joint goal,
  projected onto the hand's limits.
- Per hand, its two nearest untouched patches of another body area, each reached two ways:
  - "onto": the wrist requested so the hand would land 1 cm past the patch's nearest point,
    orientation held, the controller's strength. The smoother moves the executed wrist target about
    35% of that aim; each command records the share it actually moved (`executed_share_of_aim`).
  - "push": the request compensated for the smoother so the smoothed wrist target would land there,
    and the arm requested at the strength limit (which executes as 0.35 of the limit plus 0.65 of
    the previous strength). The execution transform's reach clamp (40 cm per axis at a range of
    motion of 1) cuts the compensated request once an aim component passes about 14 cm; each command
    records whether it was cut (`request_clipped`) and the round counts them
    (`requests_cut_by_the_reach_clamp`).

## First review (independent reviewer, relayed by the coordinator): dispositions

1. **Layer 3 was not the planner's own objective.** Fixed in v2: effort on each command's request as
   the planner's action space holds it (`planner_effort`); finger curls, finger-only differences and
   unknown requests left out; the hand stage's stopping rule recorded. v3 added the shared finger goal.
2. **"Wellness favours the toucher" was decided at float rounding.** Fixed in v2: only gaps beyond the
   tie band count.
3. **50 was not the null for either layer.** Fixed across v2 to v4; the final form is under "The three
   layers" (third review, items 1 to 3).
4. **Pooled pairs could call one body's noise "above chance".** Fixed by the declared sign rule, with
   the minimum per voting body (5 pairs from 2 touch events) and the reachability arithmetic.
5. **A pass could rest on touches inside one hand; the dashboard lead said the planner "picked".**
   Fixed; the self-touch under test is a hand on another body area; no lead says "picked".
6. **Descriptions described the request, not what executes.** Fixed as listed under "The commands".
7. **The noise-generator reset did nothing.** Confirmed by reading (`continuation_state.capture_learner`
   saves `noise_rng`; `copied_execution.capture/restore` call it); the reset was removed.
8. **Candidates were never checked against the planning record.** Fixed: the probe's own controller
   candidate must match the record's controller alternative within 0.001 on predicted wellness and
   goal distance; moments whose planner scores something the probe does not are left out with a reason.
9. to 13. **Validator gaps, empty reasons, doubled "not measured", this record's errors, the pause
   job's tests.** Fixed in v2 and extended since.

## Second review (report written 1:56:22 AM PDT, of v2 commit 72d6ccadd): dispositions

1. **MAJOR: "no bigger move" compared only whole-body sums.** v3 compared per limb and judged a patch
   contrast beside it; the third review showed both could still pass a head that does not know where
   a touch lands. v4 compares every handle and replaces the patch contrast with crossed pairs (third
   review, item 1).
2. **MAJOR: the layer-3 sign rule passed too often when the objective agrees with wellness in
   general.** Fixed: the toucher is compared directly with a matched command the evaluator scores the
   same, one judged pair per touching command. A test simulates exchangeable objectives over 400 rounds
   of 8 bodies: bodies vote above no more than half the time and the round passes at most 5 times in 100.
3. **MAJOR: the layer-3 control was not matched in roles.** Fixed: same kind (family), predicted
   wellness within the band, one matched command per toucher per moment. The size of the move is not
   matched; see the third review, item 2.
4. **MAJOR: "two different regions" was wider than the claim.** Fixed: `hand_to_body` (the reach
   targets' rule). Every judged score reports the region pairs of its touches.
5. **MAJOR: layer 3 had no reachability arithmetic.** Fixed in `EXPECTED_REACH`, with the gaps
   recorded every round.
6. **MAJOR: the goal line and the dashboard used the unjudged score.** Fixed: goal-line distances sit
   on judged scores only; the dashboard prints only judged numbers (a test holds the move-size
   counterexample's lead at "0 of 100 pairs in which it asked no bigger move" with no "60 of 100").
7. **MINOR: a bare "0.0734" in a verdict.** Fixed: shares of the 0-to-5 scale ("1.5%").
8. **MINOR: the effort convention was assumed.** Fixed: `planner_route_check` records
   `effort_per_body` and leaves a moment out of layer 3 when it is off.
9. **MINOR: the finger term.** Fixed: the controller's finger goal where the hand stage ran.
10. **MINOR: validator gaps.** Fixed and extended (third review, item 5).
11. **MINOR: the minimum counted pairs, not touches.** Fixed: `MIN_EVENTS_PER_BODY = 2`.
12. **MINOR: dashboard wording.** Fixed (third review, item 7).
13. **MINOR (the manifest is the coordinator's): `depends`.** Not mine to edit; see NEXT.
14. **NIT: the turn used 2 arccos(|dot|).** Fixed: 2 atan2(|v|, |w|).

## Third review (report written 3:06:36 AM PDT, of v3 commit 09adbf8fb): dispositions

1. **MAJOR: layer 2 passed a head that does not know where a touch lands** (chances scaled by how far
   the left wrist moves, the hand patches most sensitive; limb sums no bigger while the wrist moved
   more). Fixed as the review proposed: (a) "where" is judged on crossed pairs (the double dissociation),
   on which any head that scores every patch from one summary of the command with gains of one sign
   scores exactly 50; (b) "whether" is judged per handle, on what `order_features` reads for every body
   handle (the head reads every handle's position change, not only the 14 handles whose position the
   body executes), with "direction is not controlled" stated; (c) the patch contrast is removed rather
   than repaired, since crossed pairs answer "where" without its equal-sensitivity assumption; (d) the
   docstring, dashboard detail and this record state the class of head each rule bounds, and the pass
   lead no longer says "foresaw". Tests: the review's head (both reaches) reads above 80 over every
   pair, has no size-matched pair, and exactly 50 on crossed pairs; two parametrized heads that rank
   every patch by one random summary (a size, a signed displacement) read exactly 50 on crossed pairs.
2. **MAJOR: layer 3's matched command is not exchangeable with the toucher, so an objective that only
   prefers smaller moves passes.** Not removed: with the commands this probe builds, no same-kind
   command matches a toucher's move handle by handle (a reach aimed at the farther patch always moves
   the wrist farther; explorations differ at random). The review's second option is taken: the claim
   is narrowed to what is measured ("the planner prefers the commands that touched over same-kind
   commands that did not"), in the probe's docstring and acceptance text, the dashboard's detail and
   lead, and this record; every judged pair records how the two moves compare (`move_relation`) and
   which sense's part of the goal distance carried the preference (`sense_parts`), and the dashboard
   prints how many judged pairs had the toucher asking the smaller move. A test builds the review's
   case (an objective that only prefers smaller moves): selection passes and the record shows the
   toucher asked the smaller move in all 48 judged pairs. Each matched command now serves one toucher
   per moment. The null simulation now draws the objectives of explorations and reaches alike.
   The next test that would separate touch from move size is in NEXT.
3. **MAJOR: the layer-3 goal line sat on the toucher's raw share.** Fixed: the judged share is now the
   direct comparison with a command the evaluator scores the same, which general agreement between
   the objective and predicted wellness cannot raise; the goal line and its distance sit on it.
4. **MINOR: layer 2 counted the two sides of one touch as two events.** Fixed: events are keyed on
   (start, touching command); a test holds one touch with two patches at "not decided".
5. **MINOR: the validator did not tie a pooled value to its per-body shares.** Fixed: every judged
   score's value must equal the pair-weighted mean of its per-body shares; the review's mutation (value
   30 of 100 beside per-body shares of 100) is refused in the tests, with 37 other malformed results.
6. **MINOR (the manifest): `depends`.** Not mine to edit; the proposal in NEXT adds
   `intention_generator.py`.
7. **MINOR: dashboard wording.** Fixed: a score no body voted on says "no body had enough pairs to
   vote"; composite verdicts print "above a coin flip"; `_trend` counts the earlier single-choice rounds
   and the three-layer rounds apart; the whole-order length comes from the result's `tenure_steps`.
8. **NIT: `shared_finger_goal_starts` counted by object identity.** Fixed: it uses the hand-stage rule.
9. **NIT: patch-contrast admission and region pairs.** The patch contrast is removed; every judged
   score reports the region pairs of its touches.

## DONE

- Read in full the probe, its tests, `ability_rechecks.py`, `run_in_training_pause.py`,
  `gof_probe_harness.py`, `compare_hand_body_goals_2026-09-24.py`, `compare_synergy_physical.py`,
  `audit_self_touch_approach.py`, `probe_self_touch_approach_choice.py`, `probe_planner_candidates.py`,
  `hand_target_control.py`, `action_decoder.py`, `intention_generator.py`, `copied_execution.py`,
  `command_ranking.py`, `check_no_bare_decimals.py`; the planner, hand-stage, contact, execution and
  planning-record paths of `lem_model.py`, `predictive_state.transition`, `causal_chain.TIE_BAND` and
  its measurement, `hand_response.choose_hand_goal`, `continuation_state.capture_learner/restore_learner`,
  `touch_adapter.SKIN_REGIONS`, `env.contact_region`, `contact_head.order_features`; the first 700
  lines of `contact_head.py`.
- Measured from saved files (no model, no world): the 45 saved hand-to-body approach runs of
  2026-09-24 closed at most 0.15 to 2.74 cm of their gap in any 50 ms.
- World-only check (no learner; the memory guard allowed 6 GB inside the scheduler's saved window;
  its world folders were written at 12:33 AM PDT, 56 s in all) on the 8 saved worlds of checkpoint
  400,350: the target finder found 4 targets per world in 5 ms; each body's nearest untouched patch
  of another body area was 8.95, 7.49, 7.79, 11.62, 14.03, 7.39, 12.48 and 9.45 cm from a hand
  (median over all 32 targets 13.7 cm).
- First version (`self_touch_layers_v1`): commits 57ac9e47c (12:39:51 AM PDT) and 5d2e6e7dc
  (12:43:35 AM). The coordinator applied its validator branch (9d2eb056e, 12:47:58 AM) and dashboard
  words (18ca6c3bf, 12:50:17 AM); the scheduler started at 12:53:11 AM; the coordinator held the task
  (29f6392af, 12:56:57 AM; first_step 999999999).
- Second version (`self_touch_layers_v2`): commit 72d6ccadd (1:45:04 AM). Tests: 45 passed (1:43:36
  AM); scheduler and dashboard 88 passed, 1 skipped (1:44:14 AM).
- Third version (`self_touch_layers_v3`): commit 09adbf8fb (2:41:35 AM). Tests after the memory guard
  allowed 2 GB at 2:40:00 AM: 57 passed (2:40:04 AM); scheduler and dashboard 94 passed, 1 skipped
  (2:40:39 AM). The third review (3:06:36 AM) found items 1 to 3 above.
- Fourth version (`self_touch_layers_v4`): the probe, its tests, the validator branch, the dashboard's
  words and trend, the pause jobs file and this record. Tests after the guard allowed 2 GB (3:28:34 AM,
  41.8 of 68.8 GB committed): `test_saved_self_touch_choices.py` 63 passed (3:29:26 AM);
  `test_ability_rechecks.py` and `test_dashboard_staleness.py` 94 passed, 1 skipped (3:30:08 AM).
  Commit c14a5db28 (3:32:05 AM).
- No layered result was ever produced: `Saved/mimo_bridge/self_touch_layers` does not exist, and every
  completed `chosen_self_touch` round in the scheduler's status (steps 380,550 to 399,100) is the
  earlier single-choice form.

## IN PROGRESS

- A fourth independent read-only review, of v4 (commit c14a5db28), is running.

## NEXT (for the coordinator)

1. When a fresh review accepts v4, restart the ability scheduler so it loads the new validator (it
   changed again), then release the hold on `chosen_self_touch` (the manifest is yours; I did not edit
   it). Until the scheduler restarts, a scheduled run refuses at once (its scheduler started before the
   last change to `ability_rechecks.py`). The third reviewer's note on the hold's cost: while it holds,
   reachability goes unmeasured; a released round could pass only with hand-to-body touches within
   50 ms in at least 5 bodies, which the declared arithmetic does not expect.
2. Add to `chosen_self_touch`'s `depends` in `docs/gain_of_function_v3/ability_rechecks.json`:
   `Source/Python/mimo_bridge/causal_chain.py`, `Source/Python/mimo_bridge/touch_adapter.py`,
   `Source/Python/lem/model/contact_head.py` and `Source/Python/lem/model/intention_generator.py`
   (`action_decoder.py` is no longer read by v4). The manifest's title and hold note still describe the
   earlier single-choice test.
3. Restart the dashboard server so it loads the new words.
4. Run `docs/gain_of_function_v4/pause_jobs_self_touch_layers_2026-09-25.json` with
   `run_in_training_pause.py`. Where the full job's 3,600 s timeout comes from, from the 399,100
   round's file timestamps: 43.5 s from the declaration to the first start's predictions (loading),
   28.5 s between one start's end and the next start's predictions inside a body (the policy
   advancing at least 10 steps), 6.3 s for each of the 7 body changes, and about 3 s per start for
   12 branches of 50 ms (60 physics steps, at most 0.05 s per physics step including restores and
   writing). This form runs up to 34 branches of 250 ms per start, 850 physics steps: at most about
   42.5 s per start if the per-step cost holds (not measured for this form), so 24 starts take up to
   about 1,020 s, plus 16 x 28.5 s + 7 x 6.3 s + 43.5 s = 544 s. The hand inverse projection for up
   to 17 commands per start, twice per command (its own joint goal and the controller's), is not
   measured. The timeout leaves about 2,000 s for it and for error.
5. After the first full run, read `layers.reachability` first. If no moment offers a hand-to-body pair
   at 50 ms but some do within 250 ms, the next test is a whole-order version of layers 2 and 3 (the
   contact head predicts 50 ms, so it would be rolled forward moment by moment; not built). If none
   do even within 250 ms, `best_reach_closure_50ms_cm` and `requests_cut_by_the_reach_clamp` say how
   far the hands got and whether the clamp cut the pushes. For selection, read `wellness_gap_abs`
   against the band, then `move_relation` and `sense_parts` beside any verdict.
6. The test that would separate a preference for touch from a preference for move size (third review,
   item 2), not built: a mirrored reach per reach command, the same executed wrist shift reversed
   (compensated for the order smoother, with the reach's strength and joint goal), as each reach's
   matched command. It matches the wrist's move exactly, but not the fingers': their asked position
   change is the wrist shift plus the fingers' own drift (joint smoothing and the blended wrist
   orientation), the same for both, so the two finger moves differ in size unless that drift is
   zero. A version that holds the fingers still in both is needed before it is sound.
7. The planner action register entry `chosen_self_touch`
   (`docs/gain_of_function_v3/PLANNER_ACTIONS_2026-09-23.json`, owned by the takeover investigation)
   still names the single-choice comparison as its current action; its owner should record the
   layered test after the first round.
