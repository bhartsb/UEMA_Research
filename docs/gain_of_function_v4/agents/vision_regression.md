# Vision prediction regression after the 4:03 AM repair (agent findings, 2026-09-24 evening)

Question (coordinator, about 9:45 PM PDT): why did production's next-picture prediction get worse
again after this morning's repair, since when, and what repairs it. Rules: records first, no model
load beside training, no production change, no pause or control file written by this agent.

**Verdict so far (updated 7:18 PM PDT Sep 25).** The midday pause (1:26-3:48 PM) judged three rules (section 19).
Production's gradient clipping does not starve the vision head: at production's own clip scale a steady gradient
removes the offset in 60-200 updates. No drifting part keeps the offset alive: the two running centres, the memory
readout and the encoders and transformer move the prediction's constant 10 to 1,000 times less over 25 model steps
than the vision head's and the gaze copy's own updates do, and those updates move it by up to the offset's whole size
in that time, outward at 12 of 14 pairs. The fold is exact per acting call (its earlier misses were the memory bias
moving between acting steps). Next (NEXT 2, declared before any run): which lesson pushes the head (PULL), whether the
push is each update's own draw (NOISE), and whether the fold cuts the largest per-update kick by 3 or more (LEVER),
the rule that would license qualifying the prepared repair; and when the vision encoder stopped answering body 2's
face-down views (V2), which the records say does not share a cause with the offset (section 20). The last 20 full
reports (413,000-414,900) read a median ratio of 1.47. The earlier verdicts follow.

**Verdict of 11:48 AM PDT Sep 25.** The ablation ran in the pause of 10:41-11:21 AM PDT
(section 16). Removing, centring or scaling the memory bias at the vision head breaks the prediction at
frozen weights (1,641-1,658 times the unchanged picture's error), as declared. Folding the bias into the
head's own output bias keeps every prediction on the replay windows at 6 of 8 judged checkpoints and misses
on the live transitions, whose stored biases differ between the three acting steps (section 17). Under 400
updates of re-fitting on fixed inputs, the unchanged model loses its offset as fully as the folded one
(0.94-1.18 and 0.94-1.09 of the unchanged picture's error at all 11 checkpoints), and both give up about an
eighth of the eye command's learned benefit on the held-out bodies. So the head's geometry does not hold the
offset; something the re-fit holds fixed does. The prepared repair is not licensed. Built for the next pause
(NEXT): which moving part pushes the offset during production's own 25-step intervals (the two running
centres, the memory readout, the encoders and transformer), whether production's gradient clipping starves
the head (the re-fit repeated at production's clip factor), and a per-call check of the fold. The earlier
verdicts follow.

**Verdict of 10:39 AM PDT Sep 25.** The pause of 3:04-4:30 AM PDT judged the carrier rules
(section 12): the error is an offset every moment shares (C3 met: it carries a median 1.03 of the excess
at the nine readings with a ratio of 2 or more; line 0.5), and the memory bias decides which token each
attention head of the vision prediction head reads (C4 met: 1.02-1.03 of every winning margin from
386,750 on; line 0.8); the live lesson's construction, the replay mix, bf16 arithmetic and the lever
carry none of it (C1, C2, L not met; C5 not judged, its point 1.036 against 1.3). Without its offset the
model would beat the unchanged picture: error minus offset is 0.83-0.96 of the unchanged picture's at
the five checkpoints reading 1.8 or more. The saved replay windows then showed, without loading a model
(section 13), that the persistent memory's 192 state numbers have sat at plus or minus 1 in every saved
window since at least 385,975, so its bias is one constant vector for every body and moment: touch 242,
proprioception 204, vision 79 and motor 45 root mean square per number at 399,225 against tokens of about
5. The operator decided at 9:09 AM to test ablating or centring that contribution at the vision head on
the same saved scenes, the offset to go without damaging the eye-command effect. Built for the next
pause (section 14, NEXT): the literal arms (removed, centred, scaled), a function-preserving removal
(the fold), and a re-fit that shows whether the offset settles once the head no longer multiplies the
constant, with the eye-command guard. The repair that outcome would license is built behind a launch
flag whose default is today's model, on its own branch, not deployed (section 15). The earlier verdict
follows.

**Verdict of 2:35 AM PDT Sep 25.** Production still predicts the next picture worse than
"the picture stays the same" (its latest full reports read 1.48-3.27 times that error; 1.40 at 5 AM on
Sep 24, 3-5 through the day). The 11:31-11:44 PM pause judged four declared rules on 64 saved replay
windows: the morning's cause is not back, the growing attention direction does not follow the error,
the memory-bias rule is not met as written, and encoder drift is ruled out (section 8). Records then
showed that those frozen readings understate production's live error at the same weights (section 9),
and that production computes in bf16 while every frozen reading ran in fp32 (section 11). What keeps
changing: the vision head's attention gain (2.58 at update 380,550, 41.5 at 399,225, one direction
throughout), the memory bias added to the latent (RMS 5.5 before the Sept 23 deploy, 134 now, against
about 5 for the tokens it is added to), and the head's attention, which now reads one of its six input
tokens (spread 0.0003 or less after update 386,500). Neither the force growing the gain nor the carrier
of the error is named yet; the next pause job reads both, with its rules declared in NEXT.

Terms used below:
- **ratio**: next-picture error divided by the error of "the picture stays the same"
  (`prediction_audit.modalities.vision.prediction_mse / persistence_mse`, from
  `mimo_bridge/prediction_audit.py` on the delayed next-frame lesson, `lem_model.py` 2296-2326 and
  2745-2779). One report = 100 updates x 8 bodies = 800 predictions. 1 = no better than the unchanged
  picture, 0 = exact. It is not an eye-control skill score.
- **base head**: the vision prediction head plus the current token (residual prediction on).
  **Gaze copy**: the efference path that adds the eye command's expected picture change
  (`lem/model/efference.py` `vision_delta`). The full prediction is base + copy.
- **without-copy error**: `efference.gaze_copy.vision_err_without_copy`, the base head's own squared
  error (`efference.py` `_gauge_vision`); the unchanged picture scores about 0.008 on that scale.
- **memory bias**: the persistent memory's readout, added to the newest moment's six tokens after the
  transformer's final LayerNorm (`lem/model/forward_model.py` 402-407); `persistent_memory.bias_rms`
  is its root mean square (`lem_model.py` 2348). The final LayerNorm has a gain of RMS 3.97-4.89 and a
  bias of RMS 1.29-1.35 over segments 30-40 (weights file), so its tokens come out at an RMS of about 5.
- **attention gain**: the largest singular value of the vision head's attention value-then-output map
  `W_o W_v` (`prediction_heads.heads.vision.attn`): how much the head amplifies its strongest input
  direction on the way to its first MLP layer. That layer's 191 revived units read its output centred
  and LayerNormed (`lem/model/liveness.py` RevivedRows); its one retained unit reads it divided by the
  fixed 261.7 this morning's repair installed (`lem/model/input_conditioning.py`).

Durable sources written by this agent (numpy and the standard library only; they run beside training):
- `docs/gain_of_function_v4/vision_regression_records.py records` ->
  `docs/gain_of_function_v3/measurements/vision_regression_records_2026-09-24.json` (every report of
  segments 36-40, 1,000-update blocks, changepoint, launches, lineage summary).
- `docs/gain_of_function_v4/vision_regression_records.py weights --pair ...` ->
  `docs/gain_of_function_v3/measurements/vision_head_weights_lineage_2026-09-24.json` (61
  checkpoints from segment 30 on, archived ones found through `mimo_bridge.storage_locations`).

## DONE

### 1. When it began (records, 10:22 PM PDT)

| updates | time (PDT) | ratio (sum of errors / sum of unchanged-picture errors) | median without-copy error | copy's share of the base error removed | eye-correction units responsive |
|---|---|---|---|---|---|
| 381,001-382,000 | 5:11-5:41 AM | 1.40 | 20.9 | 0.99945 | 1.00 |
| 382,001-383,000 | 5:45-6:41 AM | 1.43 | 24.4 | 0.99942 | 1.00 |
| 383,001-384,000 | 6:44-7:57 AM | 1.64 | 25.6 | 0.99942 | 1.00 |
| 384,001-385,000 | 8:07-8:35 AM | 2.11 | 29.9 | 0.99942 | 1.00 |
| 385,001-386,000 | 8:51-9:15 AM | 1.85 | 28.9 | 0.99948 | 1.00 |
| 386,001-387,000 | 9:33-10:30 AM | 2.99 | 23.9 | 0.99891 | 1.00 |
| 387,001-388,000 | 10:33-11:10 AM | 4.80 | 20.5 | 0.99831 | 1.00 |
| 388,001-390,000 | 11:14 AM-12:44 PM | 4.28-4.44 | 15.1-9.2 | 0.9978-0.9963 | 1.00 |
| 390,001-394,000 | 12:47-5:54 PM | 3.26-4.31 | 10.6-6.8 | 0.9972-0.9946 | 1.00-0.93 |
| 394,001-397,000 | 6:09-8:12 PM | 2.94-5.47 | 6.8-5.8 | 0.9951-0.9930 | 0.89-0.81 |
| 397,001-398,300 (segment 40 from 9:46 PM) | 9:52-10:19 PM | 2.57-3.20 | 4.3-3.4 | 0.9959-0.9936 | 0.75 |

- 161 full reports from 4:07 AM to 10:19 PM PDT; the model beat the unchanged picture in 0 of them.
- Best single changepoint of log(ratio): after update 386,500 (9:51 AM PDT); geometric-mean ratio 1.72
  before, 3.72 after. The rise is gradual: 1.40 at 5 AM, 1.64 at 7 AM, 2.11 at 8 AM, 2.99 from 9:33 AM.
- The unchanged picture's error per report stayed 0.0056-0.0152: the scene did not get stiller.
- The eye-correction layer's responsive share stayed 1.00 until update 392,100 (4:24 PM), first read
  below it at 392,200 (0.98, 4:27 PM) and is 0.74-0.77 in segment 40: it falls hours after the onset,
  so it did not start it. The morning repair's line is 0.50.

### 2. What changed at the onset: nothing in flags or model code

- `args.json` of all 25 run folders of segments 37-40: identical except `--observer` (off at 4:03 AM,
  `record` from 6:31 AM, three hours before the onset, while the ratio stayed 1.4-1.6).
- Commits touching `lem/` or `mimo_bridge/` runtime code today: `ecfc9060f` (3:44 AM, the repair) and
  `a58c9b109` (9:32 AM, the developmental observer; its recorders read values the step already
  computed, `observer_model_lane.py`). The observer code ran uncommitted from 6:31 AM (those launch
  records name `ecfc9060f` with `--observer record` set). Later launches name `9a48d4a26`,
  `ccd25f38f`, `8638490c6`, `870c30f48`, `de9175abe`, `033c606fb`, `49ddaaee6`: dashboard, observer,
  scheduler, archiver and continuation code only.
- 23 launches since 4:03 AM, each resuming the previous run's newest checkpoint (list in the records
  file, `launches`). The first report after a resume reads 1.23-3.58, in line with its neighbours.

### 3. The same failure has been building since the Sept 23 deploy

Lineage summary (records file, `lineage`; every 5th record of all H84 folders):

| stretch | memory bias RMS (median) | ratio (median) | without-copy error |
|---|---|---|---|
| H84 start to segment 31 (updates 500-344,300; Sep 18-22) | 2.1-5.6 | 1.03-1.07 | no gaze copy |
| segment 32, the gain-of-function deploy (344,900-359,875; Sep 23 from 11:25 AM) | 20.5 (6.6 -> 50.8) | 1.07 | 0.008 -> 33.9 |
| segment 33 (360,300-369,250) | 71.1 | 1.32 | 32-66 |
| segments 34-36 (369,700-380,200) | 98-105 | 1.23-1.97 | 5-50 |
| segments 37-40 after the repair | 112-133 | 1.36-6.73 by run | 30 -> 3.4 |

- For 344,000 updates the model sat 3-7% worse than the unchanged picture and its heads' predicted
  change from the present picture was 0.00016-0.00043 (`pe.vision`, run medians; `lem_model.py`
  2697-2705); since the deploy the same reading is 3.4-66, the size of the base head's own output
  before the gaze copy cancels it.
- The deploy (seg31 -> seg32 `args.json`) switched on together: efference heads with the gaze copy,
  liveness repair, contact head, motor synergy, next-frame normalisation, the vestibular sense,
  normalized wellness ranking, controllability, learning progress and the causal chain.
- From the deploy the base head's error rose from 0.008 to 34-66 while the gaze copy learned to cancel
  it (share removed 0.9997): the prediction became the difference of two outputs with root mean squares
  of 5-8, where a typical one-moment picture change is about 0.09.

### 4. Weights along the lineage (weights file, read 10:22-10:40 PM PDT)

| checkpoint | attention gain (second largest) | memory readout weight norm | head output layer norm |
|---|---|---|---|
| 318,439-344,400 (seg30-31) | 0.27-0.56 (0.20-0.37) | 42-43 | 2.0-2.4 |
| 359,875 (end of seg32) | 0.89 (0.34) | 129 | 2.96 |
| 369,250 (end of seg33) | 1.44 (0.91) | 229 | 1.25 |
| 380,200 (morning stop) | 2.50 (0.56) | 268 | 1.40 |
| 380,550 (first after the repair) | 2.58 (0.53) | 269 | 1.42 |
| 385,975 (9:18 AM) | 7.14 (1.31) | 298 | 1.60 |
| 386,750 (9:57 AM) | 9.16 (1.55) | 301 | 1.59 |
| 389,850 (12:14 PM) | 14.16 (1.75) | 314 | 1.46 |
| 392,950 (4:52 PM) | 21.59 (1.86) | 323 | 1.19 |
| 397,325 (9:28 PM) | 36.11 (1.91) | 325 | 0.88 |
| 397,525 (seg40, 9:53 PM) | 36.12 (1.95) | 325 | not read |

- One direction grows. Before the deploy the top direction changed from checkpoint to checkpoint
  (cosine 0.00-0.74 between consecutive ones); since 380,200 it is the same direction (cosine
  0.97-1.00 between consecutive checkpoints). Its log growth since the repair is 0.154 per 1,000
  updates (doubling about every 4,500 updates), against 0.050 per 1,000 from 369,250 to 380,200.
- Over 25 updates it moves -1.2% to +3.0%, except +12.2% from 386,725 to 386,750 (9:56-9:57 AM),
  inside the ratio's changepoint.
- The direction it amplifies lies increasingly where the memory bias on the motor token varies: its
  length inside that slot's five main output directions rose from 0.29 (380,550) to 0.81 (397,525);
  a random direction scores 0.16 on this measure. The memory readout's per-slot top singular values
  (pre-deploy 8-10 for every slot) are now tactile 238, proprioception 201, vision 75, motor 40.
- The head's output layer shrank from 1.60 to 0.88 while the gain grew: downstream weights compensate.
- Across the 46 checkpoints after the repair, the gain and the median ratio of the 1,000 updates before
  each checkpoint have a rank correlation of 0.86 (1 = same order, 0 = unrelated). Both rise with time,
  so this fits a causal link without showing one; the ratio levels off at 3-5 from noon while the gain
  keeps growing from 14 to 36. The decomposition job tests the link on identical inputs.
- Encoder drift at the weight level is the normal rate: the vision encoder's weights moved 0.198 and
  0.210 of their size over 16,025 and 17,010 updates before the deploy (cosine 0.980, 0.978), 0.216
  and 0.214 in the deploy era, and 0.195 and 0.187 over 16,925 and 16,575 updates after the repair
  (cosine 0.981, 0.982). Whether the features drift at the normal rate is the functional anchor still
  to be measured (NEXT, `encoder_feature_anchor`); the evening audit's feature cosine 0.78 has no
  measured normal yet.
- This morning's measurement for comparison (`measurements/visual_regression_2026-09-24.json`): the
  first MLP layer's input RMS grew from 61.8 (375,906) to 261.7 (380,200) while the attention gain grew
  from 2.13 to 2.50, so this morning's input growth came mostly from the latent, not the gain.

### 5. The audit script repaired (unit-tested; its model-loading paths run in the pause)

`docs/gain_of_function_v3/audit_visual_regression_2026_09_24.py`:
- The frozen before/after comparison, path readings and part swaps are written to `result.json` before
  any training arm; the arms run after, per model, and a failed arm is recorded with its traceback
  (status names the failure; exit code 1). The 9:28 PM run lost everything because the comparison was
  printed only at the end.
- `condition_first_linear` conditions on top of an existing `InputScaledLinear` scale: the factor is
  the RMS of what the weight multiplies (raw input / 261.7); the weight is multiplied and the scale
  buffer set to 261.7 x factor, so the layer's own forward keeps dividing; Adam moments divide by the
  factor and its square; `undo` restores weight, scale, moments (and the forward of a plain Linear).
  The failed run replaced the forward and dropped the division, a start change of 1,465.8 against the
  0.00002 limit.
- The input cache is reused only when drawn from the same checkpoint with the same window count (a
  pause job's output folder stays fixed while its checkpoint changes).
- New `path_readings` on the same inputs: first-layer input RMS raw and divided by its scale, the
  retained unit's pre-activation, memory-bias and landing-token RMS, the memory bias's share of the
  attention output (exact split: values are linear), the share of the input's variation along its
  first principal direction (raw, and as the revived units read it) and that direction's cosine with
  the growing attention direction.
- Test: `Source/Python/mimo_bridge/tests/test_audit_visual_conditioning.py` (7 tests, synthetic heads,
  no checkpoint): conditioning on top of scale 50 keeps the start within the limit and restores
  exactly; the failed run's arithmetic breaks the start by more than 1,000 times the limit; a plain
  Linear is conditioned as this morning; an input already in measured units is left alone; the frozen
  comparison is on disk before the first training arm and a failed arm is recorded; the path readings
  run on a real `nn.ModuleDict` of revived rows; the growth attribution matches finite differences
  (plain descent and one AdamW update) and its retained and revived routes add up to the whole.

### 5a. Independent review of the new code (read-only reviewer, 10:24-10:41 PM PDT) and the fixes

- Crash found and fixed: `LivenessRepair.rows` is an `nn.ModuleDict`, which has no `.get()`; the audit's
  path readings and the decomposition's growth attribution called it, so both model-loading jobs would
  have failed on every checkpoint since the repair. Both now use `vision_row()`, and the two new tests
  run on a real `ModuleDict`.
- A failing path reading or model no longer loses the rest: the frozen phase saves after every
  reading; a failed conditioning check keeps the three finished arms and is recorded (exit code 1).
- The conditioning start limit scales with the prediction's size: 0.00002 up to values of 16, then
  proportionally (about ten float32 units in the last place of the largest value; the morning's
  changes were one unit, 0.0000019 and 0.0000038); the failed run's change was 1,465.8.
- The training arms add the gaze copy only on windows with a known executed order, as `predict()` and
  production's replay do (they counted every window before); the gaze copy's parts are read the same way.
- A fifth part swap moves the old prediction heads together with their revived rows: the evening
  audit's `old_prediction_heads` swap kept the new model's revived rows, a mix of the two models.
- The decomposition re-draws its shared inputs when the given file came from another checkpoint or
  window count, keeps the own-input and attribution readings in separate failure blocks, returns 1 when
  a planned reading failed, and names the learning rate the AdamW reading uses (the configured one:
  `load_checkpoint` restores it; this lineage runs `--movement-param-mod off`, so the trainer uses it too).
- Input caches are written atomically, keep the requested window count, and an unreadable cache is
  drawn again.
- Second pass (same reviewer, relayed by the coordinator at about 11 PM PDT): nothing in the changed
  parts would crash the pause, and the three first-pass defects are fixed. Its further findings, all
  fixed in the next commit:
  - the finite-difference test compared the weights with themselves (they were copied after the
    call); they are now copied before it, and the call is checked to leave the weights and their
    requires_grad flags as they were; the AdamW check uses a central difference, so it does not
    depend on the seed;
  - the decomposition read the inputs file's window count outside its error handling (a file
    without replay windows stopped the job before anything was written); now a bad file is drawn
    again;
  - the training arms called `eval()` on the liveness repair without checking that it exists;
  - the frozen phase's per-model readings (inputs, prediction, metrics, penalties, encoder features)
    are now inside one error handler: a failure records the error beside whatever that model
    already measured, and the other model and the training arms still run;
  - the decomposition's path and gaze-copy readings are wrapped the same way; every checkpoint's
    model is released in a `finally`, in both modes; the heads-with-rows swap note sits in
    `result['swap_notes']`, not among the models' readings.
- Recorded, not changed: the training arms and the growth attribution call
  `efference.vision_delta` directly, so they skip GazeCopy's gate that adds nothing before the first
  lived moment has set the latent's centre (`centre_weight <= 0`). Every checkpoint these jobs load
  restores that centre from its runtime state ("efference heads: running means restored" in the load
  log), so the gate would never apply to them.

### 6. Measurements written for a training pause (not run)

Jobs file `docs/gain_of_function_v4/pause_jobs_vision_regression_2026-09-24.json` for
`docs/gain_of_function_v4/run_in_training_pause.py`; each job loads one model at a time:

| job | memory need | what it measures |
|---|---|---|
| `vision_audit_unit_tests` | 2 GB | the five tests above, if they could not run beside training |
| `vision_audit_train64` | 12 GB | the fixed audit, before = qualified 380,500, after = the pause checkpoint, 64 saved moments, frozen comparison + path readings + part swaps, then the four isolated training arms including conditioning on top of 261.7 |
| `vision_lineage_decomposition` | 12 GB | 14 checkpoints (344,400 to the pause checkpoint) on the same 64 moments, 4 of them also on 16 moments of their own replay memory; per checkpoint the errors, path readings and the gaze copy's frozen 380,200 part vs trained part; for 6 checkpoints the growth attribution: the push on the attention gain from the prediction loss through the retained unit and through the revived units, from the penalty on the base head (before the repair) and on base + copy (since), and from one AdamW update with the checkpoint's own moments |
| `encoder_feature_anchor` | 12 GB | vision-encoder features of the same 32 raw pictures (8 bodies at 344,400, 380,200, 397,125 and the pause checkpoint) for 9 checkpoint pairs: two before the deploy (16,025 and 17,010 updates), two in the deploy era, the post-repair pairs 380,200 and 380,550 to 397,125, the audit's own pair, and two short pairs before and across the onset |

Scripts: the audit above and `docs/gain_of_function_v4/vision_lineage_decomposition.py` (new).
Outputs: `Saved/gof_takeover_20260924/repair/visual_audit_pause_train64/`,
`.../vision_lineage_decomposition/decomposition.json`, `.../vision_encoder_anchor/encoder_anchor.json`,
each written after every checkpoint so a failure keeps what was measured.

### 7. Tests

- The 7 unit tests passed at 10:48 PM PDT (2.1 s), run behind the memory guard (30.1 of 68.8 GB
  committed: training was stopped for the ability scheduler's saved comparison window), and again
  after the second review's fixes at 10:55 PM PDT (33.0 of 68.8 GB committed). Before that
  the guard had refused 1.5 GB at 9:58, 10:14, 10:20, 10:27 and 10:30 PM PDT. The job
  `vision_audit_unit_tests` stays first in the jobs file as a check of the code the pause runs.
  The model-loading code has not run: it belongs to a training pause (NEXT 1).

### 8. The pause's results against the declared rules (recorded 1:25 AM PDT Sep 25; every number re-read from the files)

Pause 11:31-11:44 PM PDT Sep 24, receipt `Saved/gof_observer_20260924/pauses/20260924_233147/receipt.json`:
checkpoint `H84_fresh_drivers_20260918_seg40_resume_20260924_232716/checkpoints/ckpt_step_399225`; the
four jobs exited 0 (the 7 unit tests passed in 2.1 s; audit 109 s; decomposition 428 s; anchor 74 s) and
training resumed. The decomposition's reading of 344,400 on the shared moments failed (`KeyError:
'intermind'`: that pre-deploy build expects an intermind token where the shared moments carry the
vestibular sense); a checkpoint listed for its own moments does not count as failed there, by design.

On the same 64 moments (drawn from the pause checkpoint's replay memory), next-picture error divided by
the unchanged picture's error (1 = no better than "the picture stays the same"): qualified model
380,500 **1.28**, pause model 399,225 **2.93**. Swapping 380,500's parts into 399,225 makes it worse in
every case: its efference running centres 10.9, heads and efference together 106, efference weights 488,
prediction heads 595, prediction heads with their revived rows 724.

| declared rule (as written in NEXT 2 before the pause) | measured | line | verdict |
|---|---|---|---|
| The morning's cause is back: the retained unit's effective input (raw / 261.7) at the pause checkpoint at least 2 times its 380,550 value, AND the conditioning arm lowers the held-out error by at least half against the corrected-penalty arm | effective input 18.6 against 0.97: 19.3 times. Held-out error (mean of the last 20 of 100 isolated updates, 32 held-out moments): conditioned 0.0109, corrected penalty 0.0158, 31.2% lower (this morning 74.8%) | 2 times; 50% | **not met** (the second part fails) |
| The growing attention direction degrades the head: across the checkpoints, the share of the first-layer input's variation along one direction rises with the ratio on the same 64 moments (no number was declared for "rises with") | rank correlation over the 13 checkpoints read on the shared moments (359,875-399,225): 0.05 (1 = same order, 0 = unrelated; p 0.88 by 20,000 permutations). The attention gain against the ratio: -0.02 | none declared | **not met** |
| The memory bias carries the head's input: its share of the attention output above 0.5 after the deploy and below 0.5 at 344,400 | 1.02-1.04 at all 12 post-deploy checkpoints; 6.03 at 344,400 (its own 16 moments) | 0.5 | **not met as written** (344,400 reads above the line) |
| Encoder drift: the post-repair pairs' mean feature cosine against the two pre-deploy pairs' | post-repair 0.776 (380,200 to 397,125) and 0.834 (380,550 to 397,125), mean 0.805; pre-deploy 0.858 and 0.739, mean 0.798 | the pre-deploy range | **ruled out** (inside the range) |

Measured facts beside the verdicts (none of them changes a verdict):
- Conditioning. The three unconditioned arms end within 0.5% of each other (0.01574, 0.01575, 0.01582):
  neither penalty changes the held-out error measurably. The unchanged picture scores 0.0076 on all 64
  moments (the held-out half's own value was not recorded). At 380,500 the input was already in
  measured units (255.3 / 261.7 = 0.98), so its conditioning arm equals its corrected-penalty arm.
- One direction. The share was 0.91-0.99 at every checkpoint from 375,906 on (0.55-0.61 at 359,875 and
  369,250) while the ratio moved between 1.13 and 6.50. The 9 checkpoints from 380,500 on give 0.02
  (share) and 0.27 (gain). The ratio was 1.13 at 389,850 and 392,950 (gain 14.2 and 21.6) and 2.93 at
  399,225 (gain 41.5).
- Memory bias. The "share" is the bias part's root mean square divided by the whole output's, so it
  exceeds 1 when the two parts partly cancel. At 344,400 the bias part (1.37) and the picture part
  (1.04) nearly cancel (whole output 0.23). The bias part is 1.3 times the picture part at 344,400 and
  12-42 times at every post-deploy checkpoint (359,875: 17; 369,250: 12; 375,906: 28; from 380,200 on:
  32-42).
- Encoder anchor. The two pre-deploy pairs differ by 0.12, which is this anchor's resolution. Deploy-era
  pairs 0.793 and 0.845; the audit's own pair (380,500 to 397,125) 0.759. The short pairs, 382,301 to
  385,975 (3,674 updates) 0.914 and 385,975 to 389,850 (3,875 updates, across the onset) 0.856, have no
  short pre-deploy pair to compare with.
- Growth attribution (a measurement with no declared rule; 6 checkpoints). The plain-gradient push on
  the gain from the next-picture loss (base head plus gaze copy against the next token, on the 64
  moments) is positive at 5 (0.07-1.03 per unit descent step) and negative at 380,550 (-0.04); it
  runs mostly through the retained unit at 380,200, 385,975, 397,125 and 399,225, through the revived
  units at 389,850. The penalty on base plus copy pushes 180-1,040 times less; the old penalty on the
  base alone pushes the gain down. The step implied by the checkpoint's SAVED AdamW moments
  (production's own average over about its last 10 updates of every lesson, plus weight decay; the 64
  moments play no part in it) shrinks the gain at 4 of 6 (-0.0027 to -0.0163 per update) and grows it
  at 2, while production's gain grew from 2.58 to 41.5. At 399,225 it is -0.005 per optimizer update
  against a local mean growth of +0.002 per optimizer update (397,325 to 399,225; production makes 1.5
  optimizer updates per model step, one live and one replay every second step). One saved momentum
  samples a quantity whose average is small against its spread (the gain's 25-update changes range
  -1.2% to +3.0%), so six of them cannot show the average push: the attribution has not named the force.

### 9. Records read beside the verdicts (1:11-1:25 AM PDT Sep 25, standard-library scripts, no model)

Two tests were declared in writing before they were computed (the IN PROGRESS entry, written after
1:11 AM PDT):
- **L1, do the frozen 64-moment readings measure production's quantity? Met, at the margin.** At the
  8 post-repair checkpoints production resumed from, the first full report after the resume against the
  frozen ratio (line: |ln(production / frozen)| at most the per-report noise, 0.321, a factor of 1.38,
  from 164 pairs of consecutive full reports after the repair):

  | resumed from | first full report (updates after the resume) | production | frozen | ln difference |
  |---|---|---|---|---|
  | 380,550 | 380,601-380,700 (51-150) | 1.73 | 1.26 | +0.322 |
  | 382,700 | 382,701-382,800 (1-100) | 1.40 | 1.68 | -0.184 |
  | 385,975 | 386,001-386,100 (26-125) | 2.28 | 1.57 | +0.374 |
  | 386,750 | 386,801-386,900 (51-150) | 4.03 | 3.90 | +0.033 |
  | 389,850 | 389,901-390,000 (51-150) | 2.26 | 1.13 | +0.690 |
  | 392,950 | 393,001-393,100 (51-150) | 2.91 | 1.13 | +0.943 |
  | 397,125 | none: the run lasted 18 minutes; its partial report (updates 1-75, 600 predictions) read 2.58 | - | 2.33 | (+0.10) |
  | 399,225 | 399,301-399,400 (76-175) | 5.09 | 2.93 | +0.554 |

  Production reads above the frozen value by more than the line at 5 of the 7 with a full report (the
  rule asked for 5 or more of 8); 380,550 exceeds the line by 0.0006, so with that one read as
  agreement it would be 4. Two deviations from the declaration: report windows are aligned to
  multiples of 100, so the first full report covers updates 1-100 to 76-175 after the resume, not 1-100;
  and 397,125 has no full report. Independent of both, the observer's per-update record (the live
  lesson's own error, `observer/model/*.npz`) over the first 10 updates after each resume, divided by
  the first report's unchanged-picture error, reads above the frozen value at all 6 checkpoints that
  have it: 382,700 2.22 (frozen 1.68), 385,975 1.75 (1.57), 386,750 4.37 (3.90), 389,850 1.54 (1.13),
  392,950 1.78 (1.13), 399,225 4.11 (2.93). Consequence: the frozen replay-window reading understates
  production's error at the same weights. The declared rules above were judged on it; what the live
  lesson has and the replay windows lack is a candidate carrier of production's excess (NEXT 1-2,
  job `vision_carrier_and_lever`, rules C1 and C5).
- **L2, is a report's error carried by a few updates? Not met.** The 5 largest of a report's 100
  updates carry a median 20.4% of its summed vision error after update 386,500 (128 reports) against
  14.8% between 382,700 and 386,500 (34 reports): 1.37 times, line 2. The post-onset rise is a level
  shift across updates, not spikes.

Read without a declared rule (what the stats already record; no verdict is drawn from them):
- **The vision head's attention concentrated onto one of its six input tokens.** The liveness monitor's
  attention spread for `prediction_heads.heads.vision.attn` (the entropy of its weights over the six
  newest tokens as a share of an even spread: 1 = equal weight on all six, 0 = all weight on one):
  median 0.36 in segment 32 (the deploy), 0.12-0.14 in segments 33-36, 0.026 from the repair to the
  onset (0.146 falling to 0.0003; 0.0018 at 385,900), at most 0.0003 after 386,500 (0.0000 on most
  reports). Like the gain it moves steadily with time, so its match with the ratio carries the same
  both-rise-with-time caveat. Which token wins is not recorded (rule C4 in NEXT 2 measures it).
- **Units of the vision head's first layer are revived again and again.** The liveness repair revives
  every unit silent over an 800-input-row window: it zeroes the unit's outgoing weights (their mean
  moves into the next layer's bias) and gives it fresh random input weights. After the first full
  revival (all 191 units at 359,884), 66% of segment 33's 100-update reports hold a revival of this
  layer (median 2 units, up to 46), 20% in segments 34-36, 19% from the repair to the onset (1-2 units),
  and 72% after the onset (median 1 unit; 24 at 398,644, 16 at 399,324). Over all 217 recorded
  revivals a switch changed the layer pair's output by at most 0.000064, against mean output sizes of
  0.9-5.7. Reports holding a revival read a
  median ratio of 3.76 (115 reports) after the onset against 3.14 without (44). The only revival
  records are the last of each 100-update window, so these are lower bounds.
- **The live vision gate kept closing:** `td_gate.vision` (the share of the model's own prediction
  blended into the tokens the live step acts on and appends to its context) medians 0.41 (segment 32),
  0.29, 0.20, 0.15 (repair to onset), 0.125 (after the onset).
- **The gain's jumps.** Between checkpoints 25 updates apart (mostly an overseer snapshot followed by a
  stop) the gain's log change per 1,000 updates has a median of 0.08 over 17 such intervals (0.06
  without the +12.2% jump from 386,725 to 386,750), against a median of 0.14 (range -0.03 to 0.46) over
  the 23 intervals of 175 updates or more: the updates before a stop do not grow it faster. That one
  jump (9:56-9:57 AM) holds no vision-head revival (the nearest at 386,494 and 386,980).
- **Now:** the last 8 full reports (400,000-400,700) read 1.48-3.27.

The two tests as they were declared (kept verbatim as the record of what was fixed before computing):
- **L1, do the frozen 64-moment
  readings measure production's quantity?** At the 8 post-repair checkpoints production resumed from
  (380,550; 382,700; 385,975; 386,750; 389,850; 392,950; 397,125; 399,225; each run folder's
  `args.json` `resume`), compare the frozen ratio on the 64 shared moments (decomposition.json) with
  production's ratio over the first full report after that resume (updates 1-100 from exactly those
  weights, 800 live predictions). Line: they agree when |ln(production / frozen)| is at most the
  per-report noise of ln(ratio), measured as the standard deviation of the differences between
  consecutive full reports of the post-repair stretch divided by the square root of 2. Verdict: the
  frozen reading stands in for production's if they agree at 6 or more of the 8; if production reads
  above frozen by more than the line at 5 or more of the 8, the replay-window reading understates
  production and the carrier is sought in what the live lesson has and replay lacks (the gate-blended
  context, the freshly encoded memory bias, the live moments themselves).
  **L2, is a report's error carried by a few updates?** From the observer's per-update vision error
  (`observer/model/*.npz` `next_pe[:, 0]`, the lesson's batch mean), for every full report with 100
  observed updates: the share of the report's summed error carried by its 5 largest updates. Line:
  spikes carry the post-onset rise if the median share after update 386,500 is at least twice the
  median share between 382,700 and 386,500 (a steady error level puts 5 of 100 updates near 5-15%).

### 10. The two label fixes (the reading notes below; 1:30 AM PDT Sep 25)

- `docs/gain_of_function_v3/audit_visual_regression_2026_09_24.py`: `metrics()` returns
  `copy_rms_all_windows` (was `copy_rms`: the gaze copy's root mean square over every window, a window
  without a known executed order counting as zero). The frozen comparison's penalties are
  `latent_penalty_all_senses_unweighted_base` and `latent_penalty_all_senses_unweighted_base_plus_copy`
  (were `regularizer_base` and `regularizer_total`: the latent penalty over every predicted sense, with
  no validity mask and without the lesson's weight of 0.01 / the number of predicted senses), read by
  the new `penalty_readings()`. `result.json` carries a `definitions` block for these three keys and for
  `isolated_training.weighted_loss` (vision only, weighted). The decomposition's rows take the new name
  through `metrics()`; its `gaze_copy.copy_rms` (only the windows that have a copy) keeps its name, now
  the only `copy_rms` in a row. Result files already on disk keep the old names.
- Tests (`Source/Python/mimo_bridge/tests/test_audit_visual_conditioning.py`, now 9): with the copy on
  half the windows, `copy_rms_all_windows` is its size on those windows times the square root of one
  half; the penalty counts a sense other than vision and carries no weight (4.5 and 6.5 on a
  hand-computed case); the written result names the definitions.

### 11. The next measurements, built and tested (1:30-2:29 AM PDT Sep 25; not yet run)

- `docs/gain_of_function_v4/vision_carrier_measurements.py measure`: the matched live and replay
  constructions with their one-ingredient swaps and three arithmetic arms (fp32, production's bf16,
  bf16 only at the prediction heads' input), the alignment check, the check against production's own
  audit of the newest transition, the error anatomy, the attention split, the context ingredients and
  the re-encoding gap (rules C1-C5 and F in NEXT), and the lever decomposition of the push on the
  attention gain (rule L). Nothing trains, steps an optimizer or writes a checkpoint; one model at a
  time; the result is written after every checkpoint; each stage (episode, matched readings, context
  ingredients, re-encoding gap, random windows, lever, summary) records its own failure and keeps the
  others. Reads the declared rules into its result.
- `Source/Python/mimo_bridge/tests/test_vision_carrier_measurements.py`: now 20 tests on synthetic
  rings, histories, heads and summary rows (listed in the jobs file). The first version's 12, with the
  audit file's 9: **21 passed in 2.4 s at 1:44 AM PDT** behind the memory guard (35.7 of 68.8 GB
  committed while another track's pause, started 1:42 AM, held training stopped); the reviewer re-ran
  them, 21 of 21. After the rewrite below, 26 of 29 passed at about 2:25 AM PDT; the three failures
  were a test input without a target, the draw allowing a window that ends on the newest moment
  (production's sampler never draws it: its start offsets reach at most count - need - 1 from the
  oldest slot, `replay.py` 159-171), and the precision arm's share computed as a closure instead of as
  the share of the bf16 excess it reproduces. All three are fixed: **29 of 29 passed in 2.9 s at 2:40
  AM PDT** behind the memory guard (33.2 of 68.8 GB committed; it had refused 2 GB at 2:29 and 2:33 AM).
  After the second review pass (section 11a), with 4 more tests (the void rule, the newest-half swap
  and the bias rounding, the heads-only bf16 arm, and the real vision head with its input scale and
  revived rows plus the efference gaze copy with and without its activation compensation, run under CPU
  bf16 autocast): **33 of 33 passed in 3.2 s at 2:57 AM PDT** behind the memory guard (40.6 of 68.8
  GB committed). The pause's first job runs them again as a check of the code the pause loads.
- Jobs file `docs/gain_of_function_v4/pause_jobs_vision_carrier_2026-09-25.json`:
  `vision_carrier_unit_tests` (2 GB) and `vision_carrier_and_lever` (12 GB; 11 checkpoints, the
  post-repair ones first and 344,400 last; 10 of them in the D: archive, found and hash-checked through
  `mimo_bridge.storage_locations`).
- What production's code fixed in the design (read, not assumed):
  - the companion saves the last four live moments per body: raw and gate-blended tokens, the context
    buffer each step used, the memory bias it added, poses and the three outgoing orders
    (`continuation_state.DATA_NAMES`); `motor_history[i]` carries moment i to moment i + 1, as the
    imagination chain reads it (`live_test._run_imagination_compiled`);
  - the live context appends the forward model's `ctx_update`, the step's INPUT tokens plus modality
    embeddings (`forward_model.py` 355-360), and those inputs are the gate-blended tokens
    (`lem_model.py` 2352-2383, 2419, 3406); the delayed lesson predicts from that context
    (`_nf_ctx`), while replay_update and the frozen audit build contexts from stored raw tokens; the
    memory bias joins only the newest moment's six tokens after the final LayerNorm, never the context;
  - the delayed lesson re-encodes the memory snapshot with current weights (`lem_model.py` 2291-2293),
    replay reads the bias from the stored memory state (4135-4139);
  - every production update is clipped at a gradient norm of 1.0 over all original parameters
    (`_clip_training_gradients`; `lem/master_config.yaml` 352-355, the file the bridge loads
    (`lem_model.py` 130): grad_clip 1.0, lr 0.0003, weight decay 0.01, betas 0.9 and 0.999); the
    100-update `training` records of 386,100-386,750 log pre-clip norms of
    140-1,197; the revived rows and the efference heads are clipped as their own groups;
  - **production computes every live and replay forward in bf16** (`lem/master_config.yaml` 279
    `precision: bf16`; `lem_model.py` 706-710 turn autocast on for any precision other than fp32 on the
    graphics card; the live step and replay_update run under it, 2243 and 4028; the live run's
    `trainer.log` reads `amp=True`), while the frozen audits so far ran in fp32 on the CPU. The
    persistent memory's readout (`persistent_memory.py` 58-59, a plain Linear) runs under that autocast,
    so the live memory bias is bf16-valued; the memory's encoder turns autocast off (48).

### 11a. Independent review of the new script and what changed (read-only reviewer; report before 2:29 AM PDT)

The reviewer read the script, its tests and the jobs file against production's code, ran the 21 tests
behind the memory guard (21 passed, 34.2 of 68.8 GB committed), parsed the job's argument list (11
`--lever` entries, 11 checkpoints) and confirmed every checkpoint and companion can be found (9 on D:,
400,350 on C:, each companion's size matching its receipt, each with an `optimizer.pt`). It verified as
correct: the pairing of each saved order with its moment; that `ctx_history[i]` is exactly the delayed
lesson's `_nf_ctx`; that the ring's newest slot is the histories' newest moment (replay.add precedes the
history appends in a step, and saves happen between steps); the window indexing and memory-state slot;
the live motor token; the shapes; the alignment check (a one-moment shift reads 0.07-0.12 relative on the
vision token, far above the 0.002 line); the exact logit split; the lever's algebra, head blocks, route
partition, Adam bias correction and sign; and that the 344,400 build has no path that crashes the matched
readings. Its 18 findings and what was done (all before any run):
- **Blocker, precision:** the live bias is bf16-valued, the replay arm recomputed it in fp32, so C1 could
  name the memory bias for a rounding difference. The replay bias is now rounded to the live bias's
  storage precision, and precision became its own rule (C5) with production's arithmetic (bf16 autocast
  around the whole forward) and a head-input-only arm (the prediction heads read the latent rounded to
  bf16; everything else fp32).
- **Nothing checked the live construction against production:** the companion holds production's own
  prediction and audit of the newest transition (`_nf_pred`, `last_prediction_audit`); the job now
  compares them (rule F gates C5 on it).
- **The current-token and memory-bias swaps are near-null by construction** (same encodings, same state):
  the matched design tests the context; the gap to what the lesson re-encodes is read separately
  (`reencoding_gap`: the memory re-encoded with the checkpoint's weights against the acting step's bias,
  and the saved raw frame re-encoded against the saved tokens).
- **The context swaps mix the gate blend with modality-embedding drift** (each live slot carries its own
  step's embedding): both are measured in the context's newest slots (`context_ingredients`), and C1
  attributes a slot swap to the gate blend only where the blend exceeds the drift.
- The "other sensory" swap included vision: it no longer does. C1-C3 pooled the pre-deploy 344,400: the
  population is now the post-repair checkpoints only. C2's random windows lacked production's action
  contract and window ages: production's draw (the hand mode's contract, ages recorded, split newer and
  older half) replaces them; the audit's draw is kept to reproduce the recorded 2.926 at 399,225. Failures
  before the alignment discarded the row: every stage now records its own error. The lever pooled foreign
  windows and shuffled across sets: it now reads each checkpoint's own moments (live transitions and its
  own production draw), shuffles within each set, and reports the shared windows separately. C4 picked one
  winner for all bodies and heads: each (body, head) now has its own, read on graded bodies. L's controls
  now void a reading beyond 0.001 (float32), the route is named by majority, and weight decay is reported
  per unit learning rate like the push. The summary is wrapped and tested. C1 names the smallest swap.
  Adam's second moment is floored at its own 1st percentile and amsgrad's maximum is used if present.
  344,400 runs last.
- Kept as a stated limit: the saved transitions were trained on before the checkpoint was written
  (1-3 updates), which lowers both constructions' error alike and the newest moments' more than older
  windows' (a bias of C2 toward "not harder"); the production check measures its size on the newest
  transition.

Second pass (same reviewer, over commit 06ccfa4a0; its report arrived after this agent's hand-back).
No blocker; it re-ran the 29 tests (29 passed, 35.9 of 68.8 GB committed), found
every checkpoint and companion (400,350 has since moved to D:), and confirmed the first pass's fixes in
the code, that CPU autocast reaches the attention projections (the heads' attention takes the slow path,
query not being key; the transformer's self-attention takes the fast path, which CPU autocast also
casts), that `last` is fp32 in production so `last.to(bfloat16)` is exactly what its input projection
applies, the production check's shapes and meaning, the re-encoding calls, the context indexing, the
draw's eligibility (about 221 of 5,871 target positions per body are eligible at 380,550) and that the
audit reproduction is the identical draw. Its findings and what was done (all tested by 2:57 AM PDT):
- **F could not tell the emulation from the construction's other differences, and its line (1.38) was
  looser than C5's (1.3):** F is now paired and tighter (within 1.14, half of C5's line in ln units)
  and requires the bf16 arm to be closer to production's audit than the fp32 arm, on the error and on
  the per-body prediction gap; its failure is read two ways ("both arms miss": the construction; "only
  bf16 misses": the emulation).
- **"Gate blend exceeds embedding drift" could not fail** (drift read only 1-3 steps back, where it is at
  most about 0.0014 of a token): the attribution now uses swaps of the newest half of the context,
  since drift grows with age and the blend does not; the context difference is also split by age third
  into the part the bodies share and the part that differs by body.
- **A failing precision arm would have taken the fp32 readings with it:** the fp32 arms now run first,
  each arithmetic arm (and the production draw's bf16 forward) in its own guarded step recording
  `precision_error`; a test runs the real vision head and gaze copy under CPU bf16 autocast.
- **The head-input arm missed the heads' own rounding:** a heads-only bf16 arm was added, and C5 now
  names one of three places (the latent input, inside the heads, the transformer).
- **C2's production draw had an fp32 bias while the newest moments' was bf16-rounded:** the draw's bias
  is now rounded like the live bias; the audit draw stays unrounded for the 2.926 reproduction.
- Minor: per-arm seconds are recorded and each checkpoint's row is saved after every stage, so a
  timeout keeps the stages read (the CPU, an i9-14900HX without AVX-512, has no measured bf16 speed);
  the job runs 6 full-bf16, 3 heads-only-bf16 and 3 input-rounded forwards of 8 bodies per checkpoint
  plus one bf16 64-window forward; a `live_bias_fp32` arm reads what the bias's bf16 rounding does to
  the error; the docstring says "reproduces", as the code computes; C1 names the swap with the largest
  closure; C4 takes one median over every (transition, body, head); a failed reproduction at 399,225
  voids every rule.

### 12. The carrier job against its declared rules (pause of 3:04-4:30 AM PDT Sep 25; recorded 9:55 AM PDT)

Receipt `Saved/gof_observer_20260924/pauses/20260925_030400/receipt.json`: the pause's own checkpoint
402,150; the 33 tests passed in 4.8 s; the measurement ran 4,749 s (3:10-4:29 AM) and exited 0, every stage
of all 11 checkpoints read. Result `Saved/gof_takeover_20260924/repair/vision_carrier_lever/carrier_lever.json`
(every number below re-read from it at 9:15-9:35 AM PDT; population: the 10 post-repair checkpoints).

| rule | measured | line | verdict |
|---|---|---|---|
| void check: the audit's own draw at 399,225 reproduces 2.926 | 2.926 | within 0.1% | reproduced: the rules are judged |
| C1 the live construction carries production's excess | live over replay 0.993 (5th-95th percentile 0.988-0.998, 80 clusters) | 1.3 | not met: on the same transitions the replay construction errs as the live one |
| C2 the newest moments are harder than the replay mix | 1.131 (5th percentile 1.003) | 1.38 | not met |
| C3 an offset every moment shares carries the error | median share of the excess 1.03 at the 9 readings with a ratio of 2 or more; 0.136 of the error at the 12 at or below 1.3 | 0.5 and 0.25 | **met** |
| C4 the memory bias holds the vision head's attention | the bias's share of each head's winning margin, median 1.017-1.032 at the 7 checkpoints from 386,750 | 0.8 | **met** |
| F the bf16 arm reproduces production's arithmetic | holds at 2 of 10; both arms miss production's own audit at 7 | two thirds | not met: the live construction differs from the lesson (1-2 updates later, re-encoded) |
| C5 bf16 arithmetic carries the excess | 1.036 (5th-95th 1.011-1.067); the heads' own bf16 arithmetic reproduces 0.76 of it, the rounding of their input -0.003 | 1.3 | not judged (F not met); below the line either way |
| L the memory bias's lever grows the attention gain | the lever carries 1.10-1.17 of a positive push at 4 checkpoints and 20.6 of a push near zero at 389,850; the push is negative at the other 5 | 7 of 10 at 0.7 | not met (5 of 10) |

Read beside the rules (no verdict drawn from them):
- In every body, two of the vision head's four attention heads read the touch token, one proprioception,
  one the motor token; the vision token's weight is 1.5e-12 to 2.9e-10 from 385,975 on (2.3e-6 and 9.4e-6
  at 382,700 and 380,550). The winning margins are 29-47 logits from 386,750 on, of which the memory
  bias gives 29.6-47.8 and the final-LayerNorm tokens (the latent without the bias) -0.9 to -1.5.
- The anatomy of the live construction: the base head's own change has a root mean square of 1.45-5.25
  and the gaze copy 1.45-5.31 per number; they cancel to a net 0.03-0.15, against a real change of
  0.06-0.12. Without the offset every moment shares, the error over the unchanged picture's reads 0.83
  (385,975), 0.94 (386,750), 0.96 (397,125), 0.92 (399,225) and 0.86 (400,350): the five checkpoints whose
  live ratio is 1.8 or more would beat the unchanged picture by 4-17%.

### 13. The persistent memory's bias is one constant (saved windows, no model; 9:25-10:31 AM PDT Sep 25)

The memory states of the saved replay windows (the audit's 64 of 399,225 and the decomposition's 16 of
344,400, 380,200, 385,975 and 389,850), through each checkpoint's memory readout taken from its
`model.safetensors` alone: first read at 9:27-9:29 AM PDT with a scratch script, then by
`docs/gain_of_function_v4/vision_memory_ablation.py records` at 10:31 AM PDT behind the memory guard, which
wrote `docs/gain_of_function_v3/measurements/vision_memory_bias_constancy_2026-09-25.json` (the same
numbers; every number below re-read from it).

| checkpoint | state numbers at magnitude above 0.99 | spread of the state between windows (RMS) | bias per slot, RMS per number (touch, proprioception, vision, motor) | moment-specific part (RMS, largest slot) |
|---|---|---|---|---|
| 344,400 (before the deploy) | 78.6% | 0.22 | 6.4, 5.4, 5.7, 5.4 | 0.68 |
| 380,200 | 99.97% | 0.0002 | 188, 186, 38, 11 | 0.0038 |
| 385,975 | 100% | 1e-7 | 219, 194, 51, 26 | 6e-6 |
| 389,850 | 100% | 4e-8 | 234, 200, 57, 28 | 2e-6 |
| 399,225 | 100% | 3e-8 | 242, 204, 79, 45 | 3e-6 |

- The memory's recurrent state saturated: from 385,975 on, each of its 192 numbers sits above 0.99 in size
  (its tanh limit is 1) and moves by 1e-7 or less between windows and bodies, so the bias it adds to the
  newest six tokens is the same vector everywhere. The
  persistent memory remembers nothing the heads can use; its readout's own bias vector is only 0.3-0.9, the
  constant is the readout weights times the fixed +-1 state.
- Consequences for the vision question: "centred" (constant removed, moment-specific part kept) and
  "removed" are the same operation today; and a constant added to every token the head reads can be moved
  into the head's own biases exactly when the attention weights are the same for every moment (they are:
  margins of 29-47 logits).
- One more consequence, arithmetic: with the state fixed at +-1 in 192 numbers, one Adam step of at most
  lr = 0.0003 per readout weight moves each bias number by up to 192 x 0.0003 = 0.058 per update, all in
  step; a constant input of that size multiplies Adam's per-weight step the same way wherever it is read
  (the attention's value weights read inputs whose mean part is 162-202, `lever.bias_input_mean_rms`).
- The saturation is its own defect, beside the vision one: it changes what every head reads. Its next
  test is in the pause job (per checkpoint, where it begins: the input projection's tanh or the
  recurrence, `memory_source`); a repair of the memory itself is the operator's decision (it changes the
  input of every head and the planner's memory-conditioned predictions).

### 14. The ablation measurement, built and tested (9:55-10:30 AM PDT Sep 25; it ran at 10:42 AM, section 16)

- `docs/gain_of_function_v4/vision_memory_ablation.py measure` (arms, readings and rules: its docstring, the
  result's `rules` field, and this file's NEXT 2 as committed in 95a2c6b9b) and `records` (section 13).
- `Source/Python/mimo_bridge/tests/test_vision_memory_ablation.py`: 14 tests on synthetic heads, memories,
  orders and rows (listed in the jobs file). With the carrier's 24 and the audit's 9 whose functions the
  script reuses: **47 passed in 8.5 s at 10:30 AM PDT** behind the memory guard (64.3 of 72.9 GB
  committed; it had refused 2 GB from 9:45 AM on).
- Jobs file `docs/gain_of_function_v4/pause_jobs_vision_memory_ablation_2026-09-25.json`:
  `vision_memory_ablation_unit_tests` (2 GB; the 47 tests again, as a check of the code the pause loads)
  and `vision_memory_ablation` (12 GB; 12 checkpoints, the pause's own first and 344,400 last; timeout
  7,200 s; its duration is unmeasured: the carrier job took 4,749 s for 11 checkpoints with bf16 arms
  this one does not run, and this one adds 14 re-fit runs of 400 updates per checkpoint). The records
  reading and the repair branch's tests ran beside training once the guard allowed (sections 13, 15).
- The production branch gained 399887f0e and 1f792d297 at 9:48 AM (the eye-row convention, executed
  fields and saccade flag, all behind switches that are off). The eyes-held order passes the model's own
  eye-row convention to `LemLite.gaze_to_handle` (legacy on every checkpoint this job reads).

### 15. The repair prepared behind a flag, not deployed (10:10 AM PDT Sep 25; worktree `C:/Users/bhart/Documents/UEMA_vision_fix`, branch `fix/vision-memory-values-2026-09-25` from 308635ffe)

What the ablation's NEXT 3 (as committed in 95a2c6b9b) licensed if R were met for Fv, built so it could ship
the moment the rule was met (R was not met, section 16):
- `--vision-memory-values add|exclude`, default `add` = today's model: no launch line carries the flag, and
  with it absent the prediction heads' state dict, the receipt, the parameters and every output are what
  they were.
- `exclude`: the vision prediction head's attention values read the newest tokens without the persistent
  memory's bias; the keys still read it, so every attention weight is unchanged. At the first heads call
  that carries the bias, its share of the attention output (through that call's attention weights, the
  mean over the call's bodies) joins the attention's output bias, so the prediction does not change at the
  switch; the change's size and its spread over the bodies are recorded (`vision_memory_fold`). No new
  parameter, no optimizer change. Needs `--persistent-memory on` and a vision head.
- The exact form (the bias removed at every call, not a stored constant), so it serves both outcomes
  where Fv settles: every call that runs the prediction heads after a forward with a memory bias passes
  that bias (the delayed lesson, the live step, the shuffled-order audit, replay,
  `predictive_state.transition` used by the planner, the imagination chain and the probes' rollouts,
  `imagine_step`, the legacy `predict_from_action` path, `gof_probe_harness.predict`, `causal_test`, the
  developmental representation probe, `action_effect_learning`). Under the flag a call without it raises,
  as a call without the current tokens does under residual prediction.
- The checkpoint records the switch (`heads.vision.memory_values_switched`). A build without the flag
  refuses such a checkpoint with a ValueError, not a RuntimeError, because the checkpoint loader
  re-initialises a component that raises a RuntimeError; its output bias holds the folded constant, so
  adding the bias again would count it twice. A checkpoint from before the switch loads into a flag build
  as not yet switched.
- Under production's bf16 autocast the as-is head computed its values from tokens up to 242 in size (bf16
  steps of 1.0); the excluded head computes them from tokens of about 5 (steps of 0.03) plus the fp32
  fold, so the switch also drops that rounding (the carrier's heads-only bf16 arm reproduced 0.76 of
  bf16's 3.6% error excess): a change at the switch in the direction of a smaller error.
- Not covered: the keys-and-values form (only if R licenses Fkv and not Fv) and the stored-constant form.
- Commits on the branch: 899d309ac (the repair, on 308635ffe), then a0717d5e8 merging the production head
  50e42b7f1 into it (production's 1f792d297 had appended LemLite arguments at the same place; both kept).
  It merges without a conflict with production at 3d68e6395 (`git merge-tree`, 10:39 AM PDT).
- Tests, on the merged branch (a0717d5e8) in the worktree, whose conftest puts the worktree's code first
  on the import path: `Source/Python/mimo_bridge/tests/test_vision_memory_values.py` **6 passed in 2.9 s at
  10:34 AM PDT** (guard: 64.6 of 72.9 GB); the suites of the paths it changed or merged
  (`test_transition_contract`, `test_efference_heads`, `test_eye_row_convention`,
  `test_executed_fields_only`) **97 passed in 21.1 s at 10:38 AM PDT** (guard at 4 GB: 62.3 of 72.9 GB).

### 16. The ablation job against its declared rules (pause of 10:41-11:21 AM PDT Sep 25; recorded 11:45-11:48 AM PDT)

Receipt `Saved/gof_observer_20260924/pauses/20260925_104133/receipt.json`: training stopped at 10:41 AM on the
pause's own checkpoint 410,400; the 47 unit tests passed in 6.3 s; the measurement ran 2,314.5 s (10:42-11:20
AM) and exited 0, every stage of all 12 checkpoints read with no error recorded; training resumed at 11:21 AM.
Result `Saved/gof_takeover_20260924/repair/vision_memory_ablation/memory_ablation.json`; every number below
re-read from it at 11:25-11:45 AM PDT (the job's log stops partway through the summary, at the script's
8,000-character print limit; the result file holds the whole summary). Population: the 11 post-repair
checkpoints 380,550-410,400; 344,400 is the reference.

Scale: the ratio is the vision prediction's error divided by the error of "the picture stays the same"
(1 = no better than the unchanged picture, 0 = exact). The eyes-held benefit is the share of the held-eyes
prediction's error that the actual eye command removes, each arm's offset taken out first (0 = the command
adds nothing; 0.050 at frozen weights over all live rows).

| rule (declared 9:45-10:15 AM) | measured | line | verdict |
|---|---|---|---|
| S: the memory bias has no moment-specific part | 3.5e-8 to 1.1e-7 of its constant at the 11 checkpoints (0.12 at 344,400) | at most 0.001 | met: removing and centring are one arm |
| E: the fold keeps every prediction (both folds; judged at the 8 checkpoints from 386,750) | live transitions: the largest change of any predicted number 0.000028-0.00125, above the line at all 8. Replay draws: 0.000008-0.000015 at 6 of 8; 0.000033 at 386,750 and 0.000035 at 389,850. The keys-and-values fold reads as the values-only fold at every set | 0.00002 at every set (three times the script's difference from the module's own forward, which read 0, is below the floor) | not met; section 17 says what the live change is |
| Z: a literal arm removes the offset at frozen weights | removed and centred 1,658 times the unchanged picture's error, scaled 1,641; the offset carries 0.9993-0.9995 of their error | ratio 1.3 and offset share 0.25 | not met, inside the declared prediction of 160-6,000 |
| G1 (frozen): the eye command's effect keeps its direction and size | every arm: pooled cosine 0.99995-1.00000 with A's response, size 0.9997-1.0000 of A's (11 checkpoints) | cosine 0.9; size 0.8-1.25 | holds |
| G2 (frozen): the eyes-held benefit does not fall | each arm minus A: removed and centred +0.0001 (5th-95th percentile -0.0011 to +0.0015), scaled +0.0003 (-0.0007 to +0.0017), folds +0.0002 (-0.00004 to +0.0006); 88 (checkpoint, body) clusters | fails only if the 95th percentile is below 0 | holds |
| R: under the re-fit the offset settles in a fold and stays in A | held-out ratio averaged over updates 220-400 and both folds: A 0.94-1.18 at all 11, the values-only fold 0.94-1.09 at all 11; A keeps an offset (1.3 or more) at 0 of 11, so part (ii) cannot be met; after the fold's re-fit the eyes-held benefit fell against A before it by 0.0071 (0.0012-0.0133), so the guard fails; the gaze forecast (G3) holds | (i) 1.3 at two thirds; (ii) A keeps an offset at 3 or more; (iii) G2 and G3 hold | not met; outcome "both settle" |

Read beside the rules (no verdict drawn from them):
- The guard's failure belongs to the re-fit, not to the fold. A's own re-fit lowers the eyes-held benefit by
  the same amount (0.0065, 5th-95th percentile 0.0011-0.0122): on the held-out rows it reads 0.051 before the
  re-fit, 0.045 after A's, 0.044 after the fold's and 0.038 after the removal's. Four hundred updates of the
  vision lesson on four bodies' moments give up about an eighth of the eye command's learned benefit on the
  other four. The same re-fit lowers the gaze head's 50 ms eye-forecast error for A by 0.020 degrees
  (0.005-0.036).
- The literal removal recovers under the re-fit: from 309-6,491 times the unchanged picture's error at update
  0 to 0.96-1.29 over updates 220-400, most of it within 60 updates.
- A stored constant (Fv_const) settles as the exact form does: 0.94-1.18.
- With new AdamW moments and no clipping ("fresh"), A settles (0.97-1.19) and the values-only fold does not stay
  settled (1.15-2.04: its held-out ratio climbs after about 100 updates at every checkpoint).
- At 5 of 11 checkpoints A's held-out ratio jumps to 2.3-14.7 at update 20 before settling (the fold training
  on bodies 0, 2, 4, 6); the values-only fold jumps at none.
- The re-fit's clipping: its fixed factors came from the vision lesson's gradient over the re-fit's core
  parameters, norms 0.16-33, factors 0.03-1.0. Production's live-update pre-clip norms in the 1,000 updates
  before each checkpoint (the 100-update means `avg_grad_norm` of the `training` records) had medians of 73-541
  and single records of 47-5,578: factors of 1/73-1/541, 12 to 410 times smaller than the re-fit's (per
  checkpoint and fold). This is the coordinator's lead (NEXT 2, rule K).
- For the persistent-memory track (another agent, branch `fix/persistent-memory-2026-09-25`): at every
  post-repair checkpoint the memory's GRU states and its retrieved vector are 100% saturated (81% at 344,400),
  and its input projection's tanh outputs 96.6-99.7% (98.3% at 344,400), from pre-activations of root mean
  square 11-17 (`memory_source` in each row).
- Corrections to the reading relayed after the pause: the moment-specific part's lower end is 3.5e-8, not 4e-8;
  E's largest judged live change is 0.00125 at 386,750 (0.00088 is 397,125's); on the replay draws E holds at 6
  of 8 judged checkpoints, not at all of them.

What the declared outcome licenses (NEXT 3 as written at 10:15 AM): "both settle": the head's geometry on fixed
inputs does not keep the offset alive; what the re-fit holds fixed does (the moving latent from the encoders'
and the transformer's training, the running centres, and the readout's updates from the other heads' lessons).
Next: the running-centre measurement (NEXT 2, rule RC, extended to every part the re-fit held fixed). The
prepared repair (section 15) is not licensed: neither R nor E is met. Its branch stays unmerged and undeployed.

### 17. Why the fold misses on the live transitions (records and algebra, 11:25-11:45 AM PDT; the pause job tests it)

- Only two things can differ between A and the fold. Both arms read the same latent and add the same gaze copy
  (`frozen_arms` computes the copy once from the latent and adds it to each arm's head output), so the gate
  blend, the top-down gate and the gaze copy cancel exactly between them. What differs is the attention output
  the head's MLP reads: (1) each moment's own bias minus the one folded constant, through that moment's
  attention weights and the value and output maps; (2) each moment's attention weights minus their mean, times
  the constant's value part.
- The second term is small on the live sets: from 386,750 on the weights' spread between moments is 1e-11 to
  4e-6 there, the same as on the replay draws (1e-11 to 3e-6), where the fold holds at 6 of 8.
- The first term is what the live sets have and the draws lack. The bias a live transition carries is its
  acting step's own, stored in bf16; it differs between moments by 0.033-0.066 root mean square per number
  (0.025-0.054% of its constant of 110-145 per number), while the draws' bias, recomputed from the saved memory
  states with one readout and rounded to bf16 the same way, differs by 0 to 0.00006.
- Why the acting steps' biases differ: the three saved transitions were acted one model step apart (steps
  410,396, 410,397 and 410,398 at the 410,400 checkpoint), with one or two optimizer updates of the memory readout
  between them. One update can move every bias number by up to 192 x 0.0003 = 0.058 (section 13), and bf16
  rounds values of 64-256 in steps of 0.5-1.0, so a move of that size flips the rounding of every number that
  sits within it of a rounding boundary.
- So E compared a fold made from one constant with three different biases. The repair folds at one call, and
  the bodies of one call share one bias (their memory states agree to 4e-8); at later calls it removes each
  call's own bias, so the readout's later movement stops reaching the vision head's values. That change is what
  the exact form is built to do, not an error of the fold.
- The draws' two misses (386,750 and 389,850) sit where the draws' attention weights spread most among the
  judged checkpoints (3.4e-6 and 1.5e-7; 1e-11 to 1.6e-9 at the other six): the second term.
- The pause job checks both (NEXT 2, E2): the fold made from each transition's own eight bodies, and the fold
  made from each moment's own bias (which leaves only the weights' spread).

### 18. The offset-drivers measurement, built and tested (11:49 AM-12:05 PM PDT Sep 25; nothing has run)

- `docs/gain_of_function_v4/vision_offset_drivers.py checkpoints` (E2, the centres' readings, K1-K5, production's
  saved steps) and `pairs` (RC); NEXT 2's declared text is its docstring and goes into its results. It reuses the
  ablation's constructions (the same live transitions and production draw, the same re-fit machinery), so its
  readings line up with section 16's.
- What the rules depend on, read in the code:
  - production clips five groups separately (`LemLite._clip_training_gradients`): the core; the efference heads,
    the gaze copy among them; the contact head; the motor synergy; and the liveness repair, the vision head's 191
    revived first-layer units among them. The logged pre-clip norm (`avg_grad_norm`: the live update's, a
    100-update mean) is the core's. So K4 trains only the core group's re-fit parameters and holds the copy and
    the revived rows fixed: the question is the core clip.
  - production's vision term carries the next-frame norm's weight (2.38-2.41 at 408,500-408,600,
    `next_frame_norm.weight_effective.vision` in the stats); K4 and K5 use each checkpoint's own.
  - the revived rows' centre moves on every training-mode forward of the vision head with gradient; the planner
    freezes the heads' weights during its loop (`intention_generator._freeze_params`), so its imagined rollouts
    do not move it; the gaze copy's centre moves once per lived training step (`efference.update_centres`).
  - the gradient pass runs the transformer with gradient on the 88 moments in chunks of at most 8 (the bridge
    attention over 768 tokens would otherwise hold several GB for the backward pass).
- Tests: `Source/Python/mimo_bridge/tests/test_vision_offset_drivers.py`, 14 tests on synthetic heads, memories,
  optimizer states, records and rows (listed in the jobs file). With the ablation's 14, the carrier's 24 and the
  audit's 9, whose functions it reuses: **61 passed in 9.1 s at 12:04 PM PDT** behind the memory guard (38.1 of
  68.8 GB committed).
- Jobs file `docs/gain_of_function_v4/pause_jobs_vision_offset_drivers_2026-09-25.json`:
  `vision_offset_drivers_unit_tests` (2 GB), `vision_offset_checkpoints` (12 GB; 12 checkpoints; timeout 7,200 s)
  and `vision_offset_pairs` (12 GB; 14 pairs, 28 checkpoints loaded one at a time; timeout 7,200 s). All 40
  checkpoint paths resolve through `mimo_bridge.storage_locations` (39 on D:, one on C:), each with its model,
  optimizer and episode companion. Measured cost on the ablation's comparable rows: 105-252 s per checkpoint with
  14 re-fit runs of 400 updates (2-17 s each); this job runs 10 per checkpoint plus the gradient pass.
- The model-loading paths have not run: they belong to a training pause (the guard refused 4 GB at 11:34 AM PDT
  with training up).

### 19. The offset-drivers jobs against their declared rules (pause of 1:26-3:48 PM PDT Sep 25; recorded 7:02-7:08 PM PDT)

Receipt `Saved/gof_observer_20260924/pauses/20260925_132646/receipt.json`: training stopped at 1:27 PM on
`ckpt_step_411150`; the 61 unit tests passed in 8.7 s (1:28 PM); `vision_offset_checkpoints` ran 2,019 s from 1:28 PM
and `vision_offset_pairs` 2,013 s from 3:13 PM, both exit 0, every stage of all 12 checkpoints and every arm of all 14
pairs read with no error recorded; after its swaps every later model reproduced its own prediction exactly (largest
difference 0 at all 14 pairs). Results `Saved/gof_takeover_20260924/repair/vision_offset_drivers/checkpoints.json`
and `pairs.json`; every number below re-read from them at 3:50-4:00 PM and 7:02-7:08 PM PDT (both logs stop partway
through the summary, at the script's 4,000-character print limit; the files hold it whole).

| rule (declared 11:49 AM-12:00 PM) | measured | line | verdict |
|---|---|---|---|
| E2: the fold's live misses are the bias's movement between acting steps | (i) the fold made per saved transition: the largest change of any prediction 0.0000048-0.000018 at 7 of the 8 judged checkpoints, 0.000036 at 386,750; (ii) the live biases' spread across the bodies of one transition 0 at all 8, between the transitions 0.00025-0.00048 of the constant | (i) 0.00002 at two thirds; (ii) within at most 1e-6 and between at least 1e-4 of the constant, at two thirds | met: the fold is exact per acting call |
| K: clipping starvation | the reference re-fit (core only, matched factor) settles at 10 of 11 (at 380,550 one fold jumps to 13.4 at update 340, window mean 1.71); at production's factor (1/73-1/541) the core-only re-fit's held-out ratio reads 0.93-1.07 at all 10 where the reference settled (production fails at 0 of 10); with eps removed the same to four digits; eps takes 0.003-0.03 in 100 of Adam's step for the median number of the head's output layer (half or more at 0 of 10) | starvation: production fails at two thirds AND eps removed settles AND eps takes half | "not starved" |
| RC: what moves the offset | 14 pairs, all 14 add up (the single parts' sum within 0.25 of their sizes of the all-parts reading). The two running centres carry half of the outward push at 0 of 14 pairs (positive at 6); the readout at 9 of 14 (positive at 11); the encoders and transformer at 4 (positive at 9). The head and the copy reduce the offset at 2 of 14 pairs and push it outward at 12 | a part is named at two thirds (10 of 14) for both its sign and its half share | "no single moving part carries it"; the head and copy push outward |

E2, read beside it: at 386,750 what the per-transition fold leaves is the attention weights' spread (the per-moment
fold misses by 0.000032 there, as its replay draw's 0.000033). The companions' saved biases, all 8 bodies at the 4
saved moments, read the same at all 11 post-repair checkpoints: within a moment 0, between moments 0.00027-0.00056 of
the constant.

K, read beside it:
- What sets the clip (K1): the vision head's attention holds 45-76 in 100 of the core group's saved second moment
  (Adam's running mean of each clipped squared gradient, over about the last 1,000 updates) at 10 post-repair
  checkpoints (28 in 100 at 380,550); the motor encoder 7-20 in 100 and the touch encoder 7-29. At 344,400, before the
  deploy, the vision head held less than 1 in 1,000. The vision head's own gradient is now most of the clip's
  denominator. The core's root summed second moment reads 1.03-1.06 at 10 of 11 checkpoints: every update clipped.
- What survives (K2): production's factor is 1/73 to 1/541 (medians of the 10-13 records before each checkpoint).
- K5: at production's factor the vision lesson's gradient is 0.008-0.08 of the root saved second moment for the
  head's output layer (0.16-0.83 for the gaze copy's output layer); on these 88 moments the vision lesson's core
  norm is 0.7 to 17 in 100 of production's pre-clip norm.
- The saved first moment points along the vision lesson's gradient on the attention and the first layer: cosine
  -0.84 to -1.0 at 8 checkpoints, +0.96 to +1.0 at 3 (397,125, 392,950 and 382,700). Production's recent steps on
  those tensors ran along one direction, with a sign that differs between checkpoints.
- Production's own saved step (lr x m / (sqrt(v) + eps), with weight decay) applied once moves the prediction's
  shared part by 0.012-0.14 per number through the head and copy (0.0001-0.001 through the readout); along the offset
  +0.08 to +1.6 (it grows the offset) at the 8 checkpoints where that cosine is negative, -0.16 to -0.24 (it shrinks
  it) at the 3 where it is positive.
- K4's other arms: the recorded factors applied in turn read as the constant factor (0.93-1.07); every re-fit
  parameter trained, the copy and the revived rows at factor 1, 0.95-1.05. At production's factor the held-out ratio
  comes down from 1.1-4.4 at update 0 to 1.0-1.2 within 60-200 updates.

RC, read beside it:
- Over 25 model steps the vision prediction's shared part moves (root mean square per number): through the vision
  head's weights 0.036-0.37, the gaze copy's weights 0.014-0.13, the memory readout 0.003-0.043, the encoders and
  transformer 0.002-0.018, the two running centres 0.0003-0.005; through all of them 0.033-0.38, as much as the offset
  itself (0.026-0.25 per number at these checkpoints). The offset is renewed within about 25 model steps.
- Along the offset (in norm units): the head +1.4 to +5.0 at 10 pairs and -0.07 to -0.71 at 4; the copy -1.57 to
  +1.25; all parts -0.91 to +5.19, positive at 12.
- The centres: over the 25 steps the gaze copy's centre moved 0.05-0.11 per number and the revived rows' 1.4-19 per
  number. What each leaves after subtracting is mostly a part every moment shares (the copy's input: 4.1-6.8 per
  number shared against 0.5-1.1 moment-specific; the revived rows' input: 17.5-231 against 0.3-9.0), but their
  movement moves the prediction 10 to 1,000 times less than the head's own weights do.

What the declared outcomes license (NEXT 3 as committed in c7e75dab2): K, "not starved": the clip is not the cause;
RC decides. RC: no single drifting part (the centres, the readout, the encoders and transformer) carries the offset;
the head and the copy push it outward: "production's own lessons grow it; their pulls on the head (imagination,
replay, the latent penalty) come next." That measurement is NEXT 2 (PULL), with the lever test declared beside it.

What this says, in one line each:
- The clip does not starve the head: at production's own scale a steady gradient removes the offset in 60-200
  updates.
- In production the head's and the copy's own updates move the prediction's constant by up to the offset's whole
  size every 25 steps, one way or the other; the offset is what those moves leave at any moment.
- Why one update moves the constant so far (arithmetic, measured next by LEVER): the head's value weights read the
  memory bias at the winning tokens (110-145 per number) and its output weights read an attention output whose
  constant part is 260-9,800 per number (the fold's output bias at 380,550 and 410,400); Adam moves each weight by up
  to about lr = 0.0003 per update whatever its gradient's size, so a weight that reads an input of size x moves its
  output by up to 0.0003 x, and 192 value weights reading 110-145 each can move one value output by up to 6-8.

### 20. Body 2's deaf vision token and the vision offset (records, 3:55 PM and 7:02-7:08 PM PDT)

- The finding (the proprioception owner's record,
  `C:/Users/bhart/Documents/UEMA_proprio_fix/docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md` section 16,
  read in full at 3:55 PM): body 2 has lain face down since 83,477 (Sept 19, 3:14 AM); its vision token's 64-s spread
  fell below the vision deaf line (0.20) at 167 of 227 checkpoints from 92,429 (Sept 19, 5:31 AM) on and off, and
  continuously since 397,025; its eyes' images still change as much as the other bodies' (mean absolute pixel change
  over 15 s 0.240 against 0.143-0.265).
- In that record's lineage file (`proprio_token_lineage_2026-09-25.json`, the memory's saved events): body 2's vision
  spread 0.46-0.50 through 79,404 (face up), 0.248 at 83,477 (the first face-down checkpoint), 0.20-0.25 at
  87,556-88,775, 0.191 at 92,429, 0.10-0.17 from 96,500 to 112,792, 0.441 at 116,823 and 0.23 at 120,702, 0.12-0.15
  from 123,975; body 0 0.37-0.53 throughout. 206 of the 227 checkpoints carry an episode companion, every one from
  63,506 to 127,975 among them.
- In the vision offset, body 2 is one body among eight. On the ablation's saved moments (the 11 post-repair
  checkpoints, `memory_ablation.json`, arm A) its mean error per moment is 0.0082 against 0.011-0.020 for the others,
  while its unchanged-picture error is 0.0017 against 0.0057-0.0133: its ratio reads 4.96 (1.49-1.97 for the others;
  377 at 386,750) because its token hardly changes, not because it errs more. It adds little to the pooled ratio's
  denominator; the offset itself is a vector every moment of every body shares (C3).
- A shared cause: not by the records. Body 2's vision went deaf on Sept 19 while the vision ratio stayed at 1.03-1.07
  until the Sept 23 deploy (section 3); the offset grows with the vision head's own updates on the memory bias's
  constant (section 19), which no view feeds; the encoder's features drifted at the pre-deploy rate (section 8). What
  the two share is a pattern: a learned part answering an input region with one constant (the vision encoder for body
  2's face-down views, the proprioception encoder for a crib heading, the vision head for its constant input). NEXT
  2's V2 re-renders body 2's face-down views from its saved worlds and reads them through the vision encoders of Sept
  19-25, to date the encoder's change and to say whether it is the encoder's or the views'.

### 21. The next measurements, built and tested (7:08-7:18 PM PDT Sep 25; nothing has run)

- `docs/gain_of_function_v4/vision_head_pulls.py measure` (PULL, NOISE, LEVER) and
  `docs/gain_of_function_v4/vision_body2_views.py measure` (V2); NEXT 2's declared text is their docstrings and goes
  into their results. The pulls reuse the offset-drivers' and the ablation's constructions (the same moments, the same
  re-fit forward, production's clip factor and next-frame weight); the imagination lesson runs production's own chain
  (`lem.live_test._run_imagination_compiled`) on the saved histories. V2 restores bodies with the probes' harness
  (`gof_probe_harness.Session`) and reads each encoder's weights by name from its `model.safetensors`.
- Tests: `Source/Python/mimo_bridge/tests/test_vision_head_pulls.py` (5) and `test_vision_body2_views.py` (4), on
  synthetic heads, optimizer states, weight files, tokens and tables. With the offset-drivers' 14 and the ablation's 14:
  **37 passed in 18.1 s at 7:15 PM PDT** behind the memory guard (58.8 of 69.7 GB committed; it refused 2 GB at 7:13
  PM with 61-62 GB committed, while the coordinator's 40-minute job ran). After a fix that keeps one model in memory
  while V2 moves from one view source to the next, the two new files' 9 passed again in 5.6 s at 7:19 PM.
- Jobs file `docs/gain_of_function_v4/pause_jobs_vision_pulls_body2_2026-09-25.json`: the tests (2 GB), 
  `vision_head_pulls` (12 GB; 13 checkpoints; timeout 5,400 s) and `vision_body2_views` (12 GB; 6 view sources, 24
  encoders; timeout 5,400 s). All 36 checkpoint paths resolve through `mimo_bridge.storage_locations` (D:), each with
  its weights, and each view source with its episode companion. Measured costs on comparable rows: the offset-drivers'
  checkpoints took 148-195 s each with 10 re-fit runs; the pulls job runs no re-fit but 3 lesson gradients, 8 per-body
  gradients and 66 lever passes per checkpoint; V2 restores 12 body worlds, renders 300 frames and runs 24 x 300
  encoder passes, at a cost not yet measured.
- Deleted at 7:18 PM (the coordinator's request; C: then read 28-29 GB free): this track's cached input windows, 618
  MB: the decomposition's `own_344400.pt`, `own_380200.pt`, `own_385975.pt`, `own_389850.pt` and the audit's
  `fixed_inputs.pt` of `visual_audit_397125`, `visual_audit_397125_train64` and `visual_audit_pause_train64`. Every
  number read from them is in the committed records (sections 8 and 13); the windows can be drawn again from each
  checkpoint's episode companion.

## IN PROGRESS

- Nothing. NEXT 1's jobs wait for a training pause.

## WAITING FOR A TEST WINDOW

- None: the one guarded run this round needed (the tests, 2 GB: `cd Source/Python && PYTHONPATH=.
  ../../.venv_mimo/Scripts/python.exe -m mimo_bridge.tools.commit_guard --need-gb 2 && PYTHONPATH=.
  ../../.venv_mimo/Scripts/python.exe -m pytest -q -p no:cacheprovider mimo_bridge/tests/test_vision_head_pulls.py
  mimo_bridge/tests/test_vision_body2_views.py mimo_bridge/tests/test_vision_offset_drivers.py
  mimo_bridge/tests/test_vision_memory_ablation.py`) ran in the coordinator's window at 7:15 PM: 37 passed.
- The model-loading and rendering work stays in the pause jobs file (NEXT 1): `vision_head_pulls` 12 GB and
  `vision_body2_views` 12 GB.

## Reading notes for the pause's results (coordinator, 10:56 PM PDT; the second review's full report)

Done 1:30 AM PDT Sep 25 (section 10): the first `copy_rms` is now `copy_rms_all_windows`, and the two
penalties are labelled as all-sense and unweighted.

The second review's complete report reached the coordinating session after the fixes in 353ff7309.
Two of its items are labels, not arithmetic: they change no number, and they are left for after the
pause (a script changed while the pause runs would split its jobs across two versions). Read the
results with them in mind:
- Two values in one decomposition row share the name `copy_rms`. `shared.copy_rms` (from the audit's
  `metrics`) averages over every window, so windows with no executed order add zeros and shrink it by
  the square root of the share of windows that had one; `shared.gaze_copy.copy_rms` covers only those
  windows. Compare the second with anything about the gaze copy's size.
- `regularizer_base` and `regularizer_total` are the latent penalty over every predicted sense,
  unweighted and unmasked; `isolated_training.weighted_loss` is the weighted, vision-only penalty the
  training arms use. They are different quantities.
- The training arms run AdamW without the liveness repair's per-row Adam-age hooks (attached only in
  the trainer's begin_step/end_step): negligible for rows revived at 380,200, not for rows revived
  just before a checkpoint.
After the pause: rename the first `copy_rms` (for example `copy_rms_all_windows`) and label the two
penalties as all-sense and unweighted.

## NEXT

The offset-drivers jobs ran in the midday pause and are judged in section 19; their declared text stays in
`vision_offset_drivers.py`'s docstring and in the results' `rules` fields. By the declared mapping (RC: the head and the
copy push the offset outward) the next measurement is the lessons' pulls on the head. Declared with it: the lever
test, which would license qualifying the prepared repair, and the body 2 dating job (the coordinator's request of 3:48
PM, section 20).

1. **Run `docs/gain_of_function_v4/pause_jobs_vision_pulls_body2_2026-09-25.json` in a training pause:** the tests
   (2 GB), `vision_head_pulls` (12 GB; the ablation's 11 post-repair checkpoints, 411,150 and 344,400 last) and
   `vision_body2_views` (12 GB; 6 view sources, 24 encoders). The two measurements are independent. Output
   `Saved/gof_takeover_20260924/repair/vision_head_pulls/pulls.json` and
   `Saved/gof_takeover_20260924/repair/vision_body2_views/views.json` (with `views.pt`, the rendered views in float16,
   up to about 270 MB, to delete once the result is recorded), each rewritten after every checkpoint, source and encoder.

2. **Rules, declared 7:08-7:14 PM PDT Sep 25, before any run** (the same text is in the two scripts' docstrings and
   goes into their results). Population: the post-repair checkpoints read (380,550 or later); 344,400 a reference,
   never pooled.

   - **PULL, which lesson pushes the offset outward.** For each lesson (the delayed next-frame lesson's vision term on
     the saved live transitions and replay's on the production draw, both at production's next-frame weight for
     vision; the imagination chain on the saved histories, at production's imagination weight), its gradient on the
     parameters whose updates moved the prediction's constant in RC (the vision head, its revived rows, the gaze
     copy's layers), turned into the step production's optimizer would give it alone, -lr f g / (sqrt(v) + eps) with
     the saved second moment and production's clip factors (the core at 1 / the median pre-clip norm, K2; the
     efference and liveness groups at 1). s = that step's move of the prediction's shared part along the offset,
     read on the other set's moments (the live lesson's on the draw, replay's on the live transitions,
     imagination's on the draw). A lesson pushes the offset outward when s > 0 at two thirds or more of the
     checkpoints read AND it carries at least half of the positive s among the three at two thirds or more. None
     named: "no lesson pushes it".
   - **NOISE, the push is each update's own draw:** the live lesson's steps made from each body's moments alone have a
     spread of s across the bodies at least 3 times the absolute value of their mean, at two thirds or more.
   - **LEVER, the fold cuts the largest per-update kick:** L = lr x the sum over those parameters of |d s / d theta|,
     the largest move along the offset that one step of at most lr per parameter can give (Adam's per-parameter step
     is at most about lr), under the geometry as it is (A) and under the fold (the values without the memory bias,
     its constant's share folded into the attention's output bias; exact per acting call, E2). Met when L_fold is at
     most L_A / 3 AND the attention's part of L_A is at least half of it, at two thirds or more of the checkpoints
     read. Beside it: the parts by group (attention; MLP with the revived rows; copy) and the same lever for 32 of the
     192 numbers of the shared part.
   - Reachability and arithmetic. PULL: production's saved step read +0.08 to +1.6 along the offset at 8 of 11
     checkpoints (section 19), so one lesson carrying half of the outward push is reachable; which lesson is what is
     unknown. NOISE: the saved momentum's sign differed between checkpoints (8 against 3) at |cosine| 0.84-1.0; if each
     update's push is its own moments' draw, the 8 bodies' steps disagree in sign. LEVER: the value weights read the
     constant (110-145 per number at the winning tokens) under A and the tokens (about 5 per number, a mean part of
     1.3) under the fold, so the value weights' part of L falls 20-30 times; the output weights' part falls with
     their input (the per-head value outputs, dominated by the constant under A); the MLP's and the copy's parts are
     the same under both (their inputs are the same at the switch). So L_fold / L_A is about (1 - a) + a / 25 with a
     the attention's share of L_A: at most 1/3 needs a of 0.69 or more. Why a third is enough: RC shows the offset is
     renewed by these kicks, so its size scales with them; a third of the kick is a ninth of the offset's squared
     error; the last 20 full reports (413,000-414,900, read at 7:05 PM) have a median vision ratio of 1.47 (1.24-2.47),
     and a ninth of that excess leaves about 1.05, the pre-deploy level (1.03-1.07).
   - Licences. LEVER met: qualify the prepared repair (NEXT 3 and 4). PULL names a lesson: its pull on the head becomes
     the target (its weight or its route into the head), declared then. NOISE met and LEVER not met: the kick is
     carried by the MLP or the copy; prepare a shape that normalises the head's attention output before its MLP (the
     retained unit read through the revived rows' centred LayerNorm), with its own rules.

   - **V2, when the vision encoder stopped answering body 2's face-down views.** Views re-rendered from saved worlds
     (the episode companions of 79,404, body 2 face up, the control; 83,477, its first face-down checkpoint; 92,429;
     123,975; 344,400; 411,150): body 2 and body 0 (face up, the live control) restored as the probes restore a body,
     the gaze reflex and spontaneous saccades off, the body held still and the eyes sent to a 5 by 5 grid of commands
     (azimuth -20 to +20 degrees, elevation -15 to +15), two model steps each, the second step's frame kept: 25 views
     per body and source, the patches and angles the encoder reads. Each set encoded by the vision encoders of 24
     checkpoints (71,511 to 127,975 on Sept 19, then 162,757, 220,675, 285,298, 344,400, 359,875, 380,550, 397,125,
     411,150). Spread = the token's standard deviation over the 25 views, mean over its 192 numbers; beside it the
     input's own spread and the realised eye angles.
     - An encoder answers body 2's face-down views when body 2's spread is at least half of body 0's from the same
       source under the same encoder; it is deaf to them at 0.35 or less (the lineage's 64-s readings, 0.07-0.12
       against live 0.32-0.51, give 0.37 or less).
     - The date: the first encoder checkpoint from which every face-down source reads deaf at every later encoder
       read. "The views, not the encoder": the encoders before 83,477, trained before body 2 lay face down, already
       read the face-down sets deaf.
     - Controls (a failure voids the date): body 0's sets read a spread of at least 0.2 under every encoder; body 2's
       face-up set (79,404) reads at least half of body 0's under the encoders before 83,477.
     - Reachability: a 40-degree sweep changes every view; the live bodies' 64-s spreads (0.32-0.51) come from natural
       viewing; body 2's face-down 64-s readings fell from 0.248 (83,477) to 0.10-0.17 (96,500-112,792), so if the
       encoder carries it the date falls between 83,477 and 96,500 (the readings came back once, 0.441 at 116,823).
     - Licences: "the encoder": a within-body anti-collapse lesson for vision (the proprioception owner's option A,
       adapted: the lesson would read the current token against a detached history, section 20's source) is the
       shape to prepare, with its own rules; "the views": the fix is behaviour (body 2 leaving the face-down pose),
       not the encoder. Either way the vision offset does not follow from it (section 20).

3. **The memory repair deploys first** (its review, `persistent_memory_review_2026-09-25.md`, question 5): how this
   track meets it.
   - PULL, NOISE, LEVER and V2 read saved checkpoints and change nothing in training; they can run before or after the
     memory deploy.
   - The memory's switch folds the constant into the readout's bias and keeps its size, so the vision head's value
     weights still read it after the switch: LEVER's reading holds for the switched model at its switch.
   - After the memory's switch the bias varies between bodies (the readout weight grows from 0). The prepared repair's
     exact form (the bias removed at every call) would remove that variation from the vision head's values too,
     against the memory repair's purpose. So if LEVER licenses it, the form to qualify after the memory deploy is
     the stored-constant one (the values read the latent minus the constant held at the vision switch; section 16's
     Fv_const settled as the exact form did, 0.94-1.18): the constant leaves the values, the memory's variation stays.
     It is a third value of `--vision-memory-values` on the prepared branch, built when LEVER is met; its fold is exact
     per call up to the bodies' spread at the switch, which the switch records.
   - Branches: `fix/vision-memory-values-2026-09-25` merges onto production plus the memory branch with three
     keep-both resolutions (review question 5). Deploy at a resume of its own, after the memory repair's kill window
     and at least 21 full reports after its switch, so each set of kill lines can be judged alone.

4. **Kill lines for the vision repair's deploy** (declared now, from records; the wander factors are the proprioception
   owner's measurement at 2:52 PM on the 482 full reports since 360,000, their record section 10):
   - K1, the switch: the first full report's vision ratio at most the median of the 20 before times one report's
     wander (2.56); the switch record's largest prediction change within E's line per call (0.00002 scaled with the
     largest value above 16; E2 read 0.0000048-0.000018 per call).
   - K2, the goal: the median vision ratio over full reports 2-21 at most the 20 before's median times a 20-report
     median's wander (2.10); and by 10,000 updates a 20-report median of at most 1.2 (the goal line is the
     pre-deploy 1.03-1.07; the arithmetic in 2 gives about 1.05 if the kick falls to a third).
   - K3, the eye command's learned effect: the gaze copy's share removed minus its shuffled-command control
     (`efference.gaze_copy`) at least half of its median over the 20 reports before, in 3 consecutive reports.
   - K4, the attention gain below 3 times its value at the resume through 10,000 updates (the weights file, section 4).
   - K5, any non-finite step.
   - Revert on any kill: pause the supervisor, snapshot_and_stop, resume the checkpoint before the switch without the
     flag (a switched checkpoint refuses a build without the flag, section 15), reattach the supervisor, lift the pause.
