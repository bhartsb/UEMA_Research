# The developmental observer: design, contracts and decisions

Addendum v4, Part 6 (sections 22-36): watch development without programming it. Built 2026-09-24.
This file maps every section of Part 6 to what was built, states each file's contract, and keeps
the log of every place the build departs from the addendum or had to decide something the
addendum leaves open (the operator's instruction: "if you disagree with anything then use your
best judgment but keep track"). Production runs the observer qualified on 2026-09-26 (entries 68-71):
detector version `observer_detectors_v10` (the recorder and detector branches' definitions together) and
analysis version `observer_analysis_v5` (both `tools/observer/daemon.py`), store version
`observer_store_v4` (store.py), counting rules `tracker_rules_v5` (tracker.py), recorder format
`observer_world_v2` (`observer_record.py`) and learner lane `observer_model_v1` (`observer_model_lane.py`).
The digester has run them since 10:48 AM PDT on 2026-09-26; no production trainer has run recorder v2
yet (section 2). Brian approved proposal `20260926_104649_4173cc` on September 27, 2026,
at 12:04 PM PDT. Its adoption and replay check finished that day at 02:14:43 PM PDT:
all 89 approved cases replayed identically and met their expected answers (section 6).
The [current delivery review](coordinator/audit_2026-09-27/observer_accomplishment_review_2026-09-28.md)
separates completed observer work from remaining production acceptance and detector coverage.
Branch `fix/observer-attribution-2026-10-08` (entry 72) changes the store to `observer_store_v5` and the counting
rules to `tracker_rules_v6`, and adds the evidence pins (`tools/observer/pins.py`); the detector and analysis versions
stay. A digester started before that code keeps the older versions until it is restarted.

## 1. What exists, in the addendum's layers (section 22)

| layer | where | runs in |
|---|---|---|
| 1 raw world stream | `Source/Python/mimo_bridge/observer_record.py` (WorldRecorder) | each world worker, only with `--observer record` |
| 1 learner lane | `Source/Python/mimo_bridge/observer_model_lane.py` (ModelLane) | the trainer's process, only with `--observer record` |
| 1 ordering, relaunches | `tools/observer/stream.py` | the digester's own process |
| geometry | `tools/observer/kinematics.py`, `features.py` | digester |
| 2 primitive events | `tools/observer/primitives.py` | digester |
| 3 episode segmenter | `tools/observer/segmenter.py` | digester |
| 4 composite detectors | `tools/observer/detectors.py` (52 catalog rows + 3 finger rows) | digester |
| intent/prediction join | `tools/observer/evidence.py` | digester |
| 5 development tracker | `tools/observer/tracker.py` | digester |
| 6 novelty miner | `tools/observer/novelty.py` | digester |
| lineage store | `tools/observer/store.py` (detections as columns, episodes with their novelty runs) | digester |
| section 24 fast lane | `tools/observer/fastlane.py` | digester |
| section 30 horizon | `tools/observer/horizon.py` | digester |
| section 27 validation | `tools/observer/validate.py` (scripted worlds; `--replay` re-digests the kept recordings), judged by the locked pass/fail lines (`tools/observer/validation_lock.py`, section 6), `baseline.py` | on demand; the scripted worlds and baselines need about 3.1 GB per world, the replay and re-summaries only the digester's memory |
| replay links | `tools/observer/replay.py` | digester |
| service | `tools/observer/daemon.py`, `python -m mimo_bridge.tools.observer watch` | kept alive by the training supervisor (`training_continuation.ensure_observer`) |
| 7 dashboard | the Development view of `tools/development_dashboard.py` (`/api/observer`, `tools/observer/view.py`) | the dashboard server |
| catalog | `docs/gain_of_function_v4/capability_catalog.json` | read by all |

`Source/Python/mimo_bridge/tools/` is outside the runtime fingerprint
(`prove_takeover_restart.runtime_fingerprint`), so the analysis changes without touching the code
production resumes on. Only the two recorder modules and three small hooks (datasource.py,
lem_model.py, train_mimo.py) are runtime code, and since 2026-09-26 one method validation calls,
`BridgeEnv.restart_clock` in env.py: it starts each validation case's virtual skin on the case's clock,
with the two helpers reset() calls in the same order (entry 71).

## 2. The one-way boundary (section 22 rule, section 36 "God-state leaks into UEMA")

- The recorder reads the simulator after the reply to the learner is built and, in a worker,
  after it is sent. MuJoCo calls: `mj_contactForce`, `mj_multiRay`, `mj_ray`, `mj_sizeModel`,
  `mj_saveModel`, all read-only on the model and the simulation's state; `mj_multiRay` uses mjData's
  scratch stack for its temporaries (MuJoCo 3.3.0 declares its data writable), which no step reads
  back (the 2026-09-25 review found the state and the stack's high-water mark identical with and
  without the recorders on a small world). No random number is drawn. Since
  observer_world_v2 (2026-09-25) it also reads each contact's frame (its normal) and receives the
  step's own arguments from the worker (motor targets, eye command, handle mask, raw commands, hand
  joint goal: `datasource._observe_frame`, after the reply), which it copies at a burst's first frame
  and never writes (`WorldRecorder.take_order`).
- The recorder is not in `env_kwargs` (a world snapshot check compares those) and the learner's
  observer attributes (`observer_per_body_errors`, `observer_moment`, `observer_contact`) start with
  neither `_` nor `last_`, so an episode save does not carry them.
- Proven by `tests/test_observer_record.py`: a recording world replies with the same bytes and ends
  in the same full saved state as a non-recording one (one-world source and the worker pool); a
  learner trained in closed loop with every recorder on ends with bit-identical weights,
  optimizer moments, losses and orders; no runtime module imports the analysis; the writers open
  no file but their own outputs. For v2 the file adds a world driven by orders and an eye command
  (production's path) whose recorded orders read back exactly, and checks every v2 field; these run
  in a training pause (crib worlds), and on the merged code they passed in the qualification pause of
  2026-09-26 (receipt Saved/gof_observer_20260924/pauses/20260926_104301_held, job
  observer_recorder_tests). `tests/test_observer_record_v2.py` checks the same fields on a
  small purpose-built world without torch (the squeeze's normals, the inner rings, a force past
  float16's limit, the orders and what is not kept, v1 chunks still read).
- At production scale (`docs/gain_of_function_v4/prove_observer.py`, result in
  `Saved/gof_observer_20260924/observer_proof/result.json`): two runs without the recorders
  differed from each other in 529 of 688 weight tensors (largest difference 0.0231: the GPU's own
  run-to-run spread); a run with them differed from one without in 49 tensors (largest 0.00694),
  inside that spread; same speed; 8 bodies x 25 records; the runtime fingerprint unchanged.
- In production since 2026-09-24 06:31 PDT (segment 38): the world recorder costs 2.0-2.6 ms per
  model step per body (inside a worker whose own step takes about 70 ms).
- Recorder v2 has been production's code since the deploy of 2026-09-26, but no production trainer has
  started on it: training has been stopped since 9:13 PM PDT on 2026-09-25 (the H84 watcher's pause,
  lifted only at the restart), so every production record so far is v1 (the active run,
  H84_fresh_drivers_20260918_seg40_resume_20260925_200754, holds v1 headers only). The first run on it is
  checked by `observer_recorder_pause_job.py after_resume --run <run>`: milliseconds per record against
  the review's expected 2.7-3.3, chunk size against 120-130 KB per 100 records, v2 headers with the four
  field groups, no recorder error.

## 3. Contracts

### observer_world_v1 (`<run>/observer/world/b<body>/`)
- `header_<pid>_<ms>.json`: entities (every infant body with its anatomical group, every object
  tree with kind / mocap / free / mass, every static geom with its kind; painted panels with their
  long axis, motif order and which end the first motif is at), joints with ranges, cameras, fovea
  and ring angles, sensors, handle and skin-patch names, author codes, the storage quanta, the disk
  budget and reserve, and `model_file` = `../models/<fingerprint[:32]>.mjb.gz` (the compiled world,
  once per kinematic structure).
- `chunk_<first step>_<pid>_<seq>.npz`, one row per model step (read with
  `observer_record.read_chunk`): time and world episode; provenance taken at the burst's first
  frame; qpos (whole multiples of 1e-6 m / 1e-6 rad / 1e-7, delta-coded); mocap; each object's root
  pose and visible / collidable flags; ctrl (float16); gaze rays per eye (centre + 8 at the fovea's
  half-angle): entity, distance, panel motif cell; contacts aggregated over the burst (pair, first
  and last frame, peak and mean normal force, point); the fast lane per frame (head acceleration
  and gyro magnitude, force from outside, largest contact force, force on the head); caregiver
  presence / object shown / touch / scene phase; the toy's scripted hand-off (0 none, 1 left, 2
  right); the world's own eye movers (saccade generator, reflex).
- `status_<pid>.json` (`enabled` is false once the trainer closed the recorder), `gaps.jsonl`
  (steps not written, with why), `recorder_error.json`.
- Measured in production: 93 KB per 100-step chunk per body (0.93 KB a step), 4.2 body-steps a
  second: about 0.34 GB a day for the eight worlds.

### observer_world_v2 (2026-09-25; the same files, v1 columns unchanged)
- The header's `fields` lists the groups written: `contact_normals`, `inner_rings`, `orders`,
  `fast_float32`. A reader treats a group a header does not list as NOT RECORDED (v1 lists none):
  `read_chunk` returns `fields`; `features.Track` sets `has_normals` / `has_inner_rings` and NaN
  normals on v1 tracks; every detector that needs a group measures nothing without it
  (`detectors.NEEDS_FIELDS`), and a babble summary made from such a recording leaves those
  capabilities out of `caps` and names them in `unmeasured_caps`, so the tracker calls the stage
  unmeasured instead of judging against a zero.
- `c_normal` [contact row, 3] float16: the force-weighted mean over the burst's frames of MuJoCo's
  contact normals, unit length, pointing from the pair's first entity (the lower code) to its second.
- `ray_in_ent` / `ray_in_dist` / `ray_in_cell` [record, eye, 16]: rings at `inner_rings_deg` (2 and 6
  degrees), 8 rays each, cast like the centre ray in a second call (the v1 bundle's values do not
  change).
- `fast` float32 (float16 stored a force above 65,504 N as infinity).
- `order_on` [record], `order_gaze` [record, 2], and one row per change of `order_row`,
  `order_motor` [k, handles, 9] float32, `order_mask` [k, handles] int8 (-1 none given),
  `order_hand_goal` [k, J]: the order each world was handed at a burst's first frame, kept only where
  it drove the body (not raw commands, not a scheduled rest's hold). `orders_in_force` reads them per
  record.
- Measured 2026-09-25 on recorded production states placed in the saved production model (95 timed
  records after 5 that build the header): 1.95 ms per record against v1's 1.25 (medians; +0.70 ms per
  body per step, most of it the 32 extra rays, in a worker whose own step takes about 70 ms); 98.0 KB
  against 70.8 KB per 100-record chunk with orders changing every 5 records at incompressible values.
  That v1 base was too small: the independent review measured 12 real production chunks (1,200
  records) at 92.5 KB per 100 records as stored and 127.6 KB with the v2 columns added as the branch
  writes them (+37.9%: the orders 23.3 KB, the inner rays 5.8, the 32-bit fast lane 4.7, the normals
  1.3), about +0.1 GB a day at this lineage's duty cycle. Each body folder's budget in production is
  1.25 GiB (12 GiB x 5/6 over 8 bodies, train_mimo.py; 1.5 GiB is the recorder's default when none is
  passed, `observer_record.DEFAULT_BUDGET_BYTES`), and digested chunks go after 72 hours unless an evidence
  pin keeps them (`pins.py`, entry 72). A NaN target
  equals itself when orders are compared, so an order held over its steps is one row (fixed after the
  review, 811689aaf).

### observer_model_v1 (`<run>/observer/model/`)
- Per step: author, latch, order sequence, rest, bout; next-moment error per sense (batch mean) and
  per body; each grounded wellness channel per body; the contact head's probabilities per body and
  patch (any / self / world) with each patch's base rate; each order's efference record (expected
  movement formed where it began, achieved, and the shuffled-order expectation); the planner's
  per-depth licence when it changes. `read_model_chunk`. 415 KB per 100 steps (all bodies): about
  0.19 GB a day.

### analysis (`<run>/observer/analysis/`)
- `detections.jsonl.gz`, `episodes.jsonl.gz`: gzip-compressed JSON lines, ONE MEMBER PER
  DIGESTION WINDOW. A detection: cap, sub-row, repetition key, body, pid, steps, measures,
  confidence, objects, parts, author and provenance class, caregiver presence / touch,
  `world_moved`, scripted, posture, evidence {prediction (a touch's verdict with its matched no-touch
  model steps and passes: `null_moments`, `null_passed`, `null_reach_steps`; for an act of the hands alone
  since observer_attribution_v1, one efference verdict per command in force during it over the command's whole
  hold, `intervals` with each command's `window`, the Predicted stage's units `commands_judged` and
  `commands_predicted`, and the act's verdict, every command passing, as a reading beside the stage),
  intention (for an act of the hands alone, per command:
  `intervals`), wellness (the act's change and its references: `reference_n/sum/sq` without the act,
  `all_n/sum/sq` whatever happened in them, `reference_reach_steps`), credit (an act of other parts: the
  command in force at the act's peak motion, `peak_step`, `credit_step`, `lag_records` 0; an act of the hands
  alone since observer_attribution_v1: no single command, `rule` `command_intervals`, its writer class and its
  `commands`), attribution and stages (acts of the hands alone only, entry 72)}, `first_author` and
  `first_provenance` (the writer of the act's first record, where `author` and `provenance` now name the writer
  class of its commands), `alone_inside` (H05), episode id,
  detector version. An episode (since v6 one effector's, or a pair of two effectors' starting within 0.5 s):
  id, effector, body, pid, steps, primitive tokens, capabilities detected, authors, caregiver
  flags, `toy_held`. Measured: 156 bytes a detection compressed (1,566 as plain
  JSON; production made 0.84 detections a body-step in its first hour, so plain files would have
  grown about 480 MB a day).
- `progress.json`: the digest state (per body: chunks done, finalized step per process, the last
  finalized step per capability key, read and window failures), the exposure, the fast-lane
  summaries and the record files' committed sizes, in ONE atomic write after every window.
  `state.json`, `exposure.json`, `fastlane.json` are the readers' copies.
- Restart safety: a digester that stops between an append and its commit cuts the record files
  back to the committed sizes when it starts again (`store.truncate`), so no window counts twice.
- Matched moments are placed by their distance from the act (analysis v5): the touch verdict's at 1, 2
  and 3 s before and after the onset, at the onset's place in its order, each given to the earliest act
  of the same row and skin patches that has it; the wellness references are stretches of the act's
  length ending 1 to 3 s before it starts (none for an act longer than 800 steps,
  `evidence.REFERENCE_MAX_STEPS`). A window reads ten chunks before its first unfinalized one and
  finalizes a touch act only once its last moment is recorded, so batch and live digests give the same
  controls: on segment 40's 9:33 AM run, digested with the statistics branch's code, 7,699 of the 7,717
  acts both found have the same control values, and the other 18 are acts the turn primitive finds with
  another end or beside an act only one digest found (the merged code's confirmed turns, entry 70, are
  built to remove those 18; not yet measured: section 9).
- Exposure bins (100 steps) keep their first and last recorded step (`lo`, `hi`), so a cut inside a bin
  counts only its records before the cut. Fast-lane bins are the exposure's (`fast_bin` 100; 1,000
  before v5), each spike found over the window's whole track and counted in the bin of its peak record;
  the learner's lane is read at world step + 1.
- `state.analysis_id`: one id per analysis folder; the lineage store notices a run digested again.
- `state.first_step`, `state.start_ms`: where and when the run's record begins (lineage cuts).
- `state.sealed`: a run that is no longer the active one with nothing left to digest; it is never
  digested again (raw-data housekeeping continues hourly).
- A different `ANALYSIS_VERSION` or `DETECTOR_VERSION` moves the folder aside
  (`analysis_<old detector version>_<time>` with `moved_aside.json`) and digests the raw chunks
  still on disk again. Detections of two definitions are never mixed; history older than the raw
  chunks' 72 hours is not re-digested.
- Raw-data housekeeping (`RunDigest.housekeeping`) removes a digested world chunk 72 hours after its file time
  (sooner, oldest first, while a body folder holds more than four fifths of its budget) and a learner-lane chunk
  starting more than 2,100 model steps before the earliest body's finalized step (any chunk of a sealed run) 72
  hours after its file time; never a chunk an evidence pin keeps, nothing of a run an invalid pin names,
  and nothing at all in a pass whose pin file cannot be read (entry 72). It never removes `causal_chain.jsonl`.

### evidence pins (`Saved/mimo_bridge/observer/pins.json`; `tools/observer/pins.py`, entry 72)
- Written by the coordinator, read by the digester at every pass, written by nothing in the observer:
  `{"pins": [{"id", "run", "steps": [first, last], "bodies", "until", "reason", "owner"}]}`. `id` unique in the
  file; `run` a run folder's name under `Saved/mimo_bridge/experiments` or its absolute path; `steps` model steps,
  both ends included; `bodies` the bodies whose world chunks it keeps (left out: every body); `until` optional, ISO
  8601 with its UTC offset or seconds since 1970.
- A pin keeps, in its run, every world chunk of its bodies whose records reach into its range, each chunk's range
  read from its own step column (an unreadable chunk is kept from its first step on), and every learner-lane chunk
  reaching into the range as the lane numbers it: one step after the world's (`evidence.TRAINER_LANE_LAG`) and up
  to `pins.LANE_REACH_STEPS` (10) further, so the efference record of the command in force at the pin's last step
  is kept. Housekeeping reports, per run, the pins naming it, the chunks they kept (world per body, learner lane)
  and whether `causal_chain.jsonl` exists.
- Failing closed. An EXPIRED pin is reported in every pass and KEEPS PROTECTING until the coordinator removes or
  renews it. A pin file that exists and cannot be read, or holds no list of pins, stops every deletion of that pass;
  a file starting with a UTF-8 BOM is read. An invalid entry is reported with why and stops every deletion in the
  run it names, or in every run when it names no usable run. A file read before and missing now keeps the pins last
  read for the digester's life: remove every pin by writing `{"pins": []}`, and replace the file atomically. A pin
  naming a run the digester does not list is reported. All of it is in `observer/daemon_status.json` under `pins`.

### the lineage (`Saved/mimo_bridge/observer/<lineage>/`)
- `store/`: the lineage store (`store_meta.json` naming a generation, `store_dets_<n>.npz`,
  `store_episodes_<n>.json`), saved at most every 30 minutes; a restarted digester reads only the
  records after the saved offsets. Store v4 (2026-09-25) adds per detection the wellness references'
  columns `wref_n/sum/sq` (stretches without the act) and `wall_n/sum/sq` (every stretch), and per
  episode its token counts and capabilities (`bag`, `bag_n`, `n_tok`, `caps`) for novelty's order
  control: 24 columns, 100 bytes a detection in memory (store v3: 18 columns, 76 bytes). `credit_moved`,
  a v3 column (-1 unmeasured, 0 the first record's command, 1 a later one), reads 2 for an earlier one
  since analysis v5. Store v5 (2026-10-08, entry 72) adds `attr` (how the act's writer is attributed: -2 the
  command at its peak, an act of other parts; -1 an act of the hands alone recorded before
  observer_attribution_v1, its writer not measured; 0 every command in force during it of one writer class; 1 of
  several), `alone` (H05's other digits inside the act: 1 within the line, 0 beyond it, -1 not measured), `sel`
  (planner selected: 0 a command was not the planner's, -1 not judged or not measured), `ach` (achieved: 1 yes, 0
  no, -1 not measured), and `pred_n`, `pred_k` (the commands of an act of the hands alone judged and predicted: the
  Predicted stage's units): 30 columns, 108 bytes a detection in memory.
- `development.json`: every capability's six stages with and without the caregiver, sub-rows, the
  timeline, novelty candidates, horizon, fast lanes (per body, `fast_lane_total` over the bodies,
  and `fast_lane_chance`: the same lanes of the random-order babble and limp recordings), recorder
  health, replays, the babble baseline used, the recording's first step and the lineage cuts.
- `replays/`: snippets and GIFs of first observations and novelty examples.
- `stage_state.json`, `stage_history.jsonl`: each capability row's stages ever reached, and every
  loss and return with its p-value and line (entry 38).

### baselines (`Saved/mimo_bridge/observer/baselines/<world signature>/<kind>.json`)
Per capability and sub-row (`baseline.chance_row`): count, occasions (per body and part), `acts_sq`
(the sum over occasions of the squared acts per occasion: babble's burst spread), body-steps, rate,
the main measure's 5th/50th/95th percentiles over every measured act (`measures`) and over each
occasion's first measured act (`measures_occasion`, with `measured_occasions`: the line the size
half of "seen" judges occasions against); for the novelty miner, how many babble episodes contain
each token run; the detector version (a summary of another version is not used); where its bodies
started (`start`: `reset_world` or `saved_worlds` with the checkpoint and its step); the recording
stays beside it (`baseline --resummarize <signature>` re-digests it without simulating). Since
2026-09-25: `recorded_fields` (the recorder's field groups every header of the recording carries)
and `unmeasured_caps` (capabilities left out of `caps` for want of one); a saved-world recording from
a newer checkpoint moves the previous comparison aside as `<kind>_ckpt_<step>` (never deleted) and
`baseline --compare <signature> --kind K --other K_ckpt_<step>` tests each capability's rate in the
two; kind `yoked` (production's own orders replayed in shuffled time order) with its `yoke` block
(source runs, orders per body, orders replayed, passes, method), and kind `yoked_sequence` (the same
orders in their recorded order: entry 67); both yokes are reported, never judged against. The comparison
since the deploy of 2026-09-26 is babble recorded under recorder v2 from checkpoint 415,075 (section 6).

## 4. Primitives and detectors (sections 22, 26)

52 catalog rows and three finger rows (section 32's enumeration, matched to the dashboard's
targeted hand actions): HX1 spread and gather, HX2 shape the palm, HX3 touch with one fingertip.
H05 has a sub-row per hand and digit: the finger-movement rows. Every threshold is in
`primitives.py` or `detectors.py` beside the code, with the measurement that set it. H05 detects a digit's
movement by the other four digits' END positions (within 5 degrees of where they began, on average); since
observer_attribution_v1 it also reports their largest movement inside the act (`others_max_excursion_deg`, their
mean `others_mean_excursion_deg`) and `alone_inside` (that mean within the same line), and the page's finger rows
read "moves; the other fingers end where they began" with the inside-the-act count beside them (entry 72).

Definitions changed on 2026-09-24 after measuring production (first 1,600-4,000 body-steps of
segment 38) against the random-order babble (20,000 body-steps):

| what | before | now | measured reason |
|---|---|---|---|
| contact | every on-step run | breaks of up to 2 steps (100 ms) are one touch | 28% of re-contacts came after a 1-2 step break; 36% of contacts lasted one step |
| weight shift (support_shift, P06) | the force-weighted centre of the floor contacts moving 2 cm | the centre of MASS moving 2 cm within 1 s, from a place it held for 250 ms, keeping half the shift 250 ms; P06 adds a 5-step-median support share moving the same way | the contact centre moved 2.8 cm at the median between two steps (limb contacts flicker); the centre of mass 0.35 cm over five steps (99.9th percentile 1.7 cm) |
| impact | the largest contact force above half a body weight | that force RISING by a body weight within one 10 ms frame | the largest RESTING contact carries 0.44 body weights at the median: the old line marked 34% of frames |
| support loss | a half-weight drop between two frames | outside force below a quarter weight for 30 ms right after carrying three quarters | one-frame half-weight drops came in 7% of frames (solver swings) |
| head jolt | 5 m/s2 from gravity | 2 g (19.6 m/s2) | 5 m/s2 was crossed by 6.9% of production frames; 2 g by 0.13% |
| salient (opens an episode) | every contact, every eye movement | not contacts with floor, walls, art, ceiling; not eye movements; an approach only when it ends within 20 cm | every episode ran to its 10 s cap with about 950 tokens |
| R01 reach | every object the hand got closer to | the one object it closed on most; the hand's own movement must do half the closing | one arm movement "reached" several toys (7.8 per 100 steps) |
| R02 repeat contact | every hand-object contact onset | a contact the hand made after its own approach | a limp body made 262 in 20,000 steps |
| G04 tracking | the object stays in the fovea while it moves | ... and the gaze turned at least 5 degrees with it | a limp body "tracked" 24 times |
| H02 hold (and H03, H09, R06, R07, O05) | an opposed grip counted even if nothing moved | an opposed grip counts without movement only if the hand closed 10 degrees on it | a limp hand lying against a box "held" it for 48 s |
| B03 hand to face | any hand-head contact on the face | the palm approached at 5 cm/s or more over 250 ms | a head turning into a resting hand is not this act |
| B05 opposite limb | any contact with the other side's limb | the hand at or past the body's midline (2 cm) | "cross-midline reach" (catalog) |
| rolls P02-P04 | blocked by any impact | blocked only by the caregiver's touch or a moving object the world moves | the impact line marked a third of frames: no roll could count |
| world-moved acts | counted | object-contact acts whose object the world moves (the toy, a critter) was moving are marked `world_moved` and never counted | a limp hand "repeated contact" 13 times per 1,000 steps from the toy |
| P01 head stabilization, P09/P10 translation | windows from where the digest window began | windows on absolute steps | a shifted grid re-emitted overlapping acts |

Definitions changed on 2026-09-24 evening after the second outside review (detectors v6 and v7;
measured on the same recorded chunks: production segments 38 and 39, 35,800 body-steps; random-order
babble and limp, 19,200 each):

| what | before | now | measured reason |
|---|---|---|---|
| episode (v6) | one per body; any salient act opened or extended it, up to 10 s | one per effector (left hand, right hand, gaze with the head, trunk with the whole-body fast events, left leg, right leg); 0.5 s of that effector's quiet ends it; none crosses an absolute 10 s block; finger, wrist and closing turns, continuing head turns and a hand touching itself ride inside without opening one; two effectors' fresh episodes starting within 0.5 s form a pair | 168 of 184 production episodes ran to the 10 s cap (babble 88 of 96); now 0-1 per effector |
| novelty explained (v6) | by the pooled capabilities of every detection in the episode | only by one detection covering every token kind of the run | the pooled rule contradicted its own docstring |
| fixation (v6, then v7) | v5: the centre ray on one target for 2 records; v6: 5 of the 9 foveal rays | v7: the thing the centre ray lands on (object, body part, adult, motif cell) for 2 records; the majority only where the centre ray lands on a surface or nothing | v6 lost looks at toys beyond about 15 cm (entry 44): ball fixations 29 under v6, 119 under v7 |
| G03 return look (v6) | any gap between two fixations of one target | the target must leave the fovea (no ray on it for a step) | a gaze that kept the target with a minority of its rays for 0.3 s "returned" |
| G04 tracking the mobile (v6) | aimed at its ceiling mount | aimed at its hanging parts | the mount does not move |
| support fall (v6) | the outside force at a frame minus the most of the next three | the running median of the previous 2 s minus the most of the next three | 85% of v5's production falls started a quarter body weight above the usual level (an impact's falling edge): 704 per 10,000 frames became 95 (babble 799 to 90, limp 0 and 0) |
| R02 repeat contact (v7) | its touch's start derived by the join from time_to_contact_s | the detector names the touch's first record (`onset_index`) | the act begins at its approach, the touch later |
| H02 hold, R06 carry (v8; H03, H09, R07, O05 through `held`) | the object moved 1 cm during the contact | the object's movement counts only while it is off its support (no floor contact, no contact with another object) | a resting hand sliding the light box along the floor made one 48.7 s "hold" and "carry" in each of the 8 limp bodies |
| HX1 spread and gather (v8) | the sum of the four fingers' side angles | the fan: the index finger's side angle minus the little finger's, rising 10-90% within 2 s (`fan_deg`, `rise_s`) | a sweep of all four fingers one way read as a spread, and limp hands' 6-70 s drifts made 40 acts |
| R08 placed release (v8) | the object slow in the record after the last contact | ... and at or under 0.03 m/s in the last contact record too | a box that slipped and fell within one record read as placed |
| H03 release (v8) | escape judged against the palm's speed | the object's own speed after contact ends | a hand withdrawing after letting go read as the object escaping |
| O01 push (v8) | the palm's velocity in the record before the object moved | when the palm was still then, its velocity in the object's first moving record | a push from rest contact was skipped |

Definitions changed on 2026-09-25 (the recorder branch's detectors v9, with the recorder's
observer_world_v2; entries 61-67; part of observer_detectors_v10 since the merge of 2026-09-26;
measured on the kept recordings with contact normals and inner rings rebuilt from each record's stored
state, `docs/gain_of_function_v4/observer_recorder_measurements.py`):

| what | before | now | measured reason |
|---|---|---|---|
| R04 opposing grasp | the angle at the object's centre between the thumb's and the other parts' mean contact POINTS, at least 90 degrees | THE GRIP (`detectors._grip`, the physics of a grasp; entry 67): two surfaces of one hand - thumb and fingers, thumb and palm, or fingers and palm (a palmar grasp) - touch a free object in the SAME physics frames, and in one record their pushes (force-weighted contact normals, hand into object) are at least 120 degrees apart (one exclusion), they balance (what the stronger exceeds the weaker by along their axis, the object's weight and acceleration and the pair's own friction can take), everything but the hand pushes the object up with less than half its weight (retained against gravity; its support not mainly the chest, a forearm, the other hand or the environment), and it moves with the palm (0.03 m/s over the record's own burst); nothing without recorded normals; the world's mocap toys left out | the 90-degree position line ran through the grasps this hand can make on the 6 cm boxes (83-126 degrees) and missed a real pad pinch (30-44); by normals the kept scripted grasps are the thumb on one face and the ring and middle fingers on ADJACENT faces (90, edge contacts up to 107.6); of babble's 123 thumb co-contacts in 120,000 body-steps the rebuild judged 35 (28%), all under 100 degrees, and the review placed every one of the 91 on a box or the ball: none on opposite faces; 30 of 147 babble and limp co-contacts touched in frames that never overlapped (the review's finding 3) |
| H02 opposed branch (H03, H09, R06, R07 through `held`) | thumb and fingers or palm touching the object in the same record | a gripping record (R04's grip) and the hand closing 10 degrees on the object | a palm and a thumb both on a box's top face were "opposed" (59.7 degrees apart by position); limp H02 1 to 0: that limp hold was opposed only by co-contact |
| H02 support | the floor or another object | CONTACT with the floor, another object or the body except both hands and forearms, and, where normals are recorded, also a FORCE: everything but the holding hand pushing the object up with half its weight or more (a forearm or the other hand is support exactly as far as it carries the object; a box squeezed between two palms rests on neither) | a box resting on the belly, moved by the body under a touching hand, read as carried; the contact rule for the forearms had been chosen on scripted boxes sitting 4-5.5 mm inside the forearm (the review's finding 5); by force alone a limp hand under a box the floor still touched "carried" it in all 8 reset-world limp bodies (entry 67) |
| fixations | the share of the nine v1 rays on the target | ... and `inner_share`, the share of the 17 inner rays (centre, rings at 2 and 6 degrees) on it, reported by fixations, G02, G03, G05 and R03; the centre-ray rule stays | a majority of the inner rays as the rule made holds less still and lost two thirds of the looks at objects 20-40 cm away (entry 63) |

O05 (shaking) is not among the detectors built on `held`, whatever the earlier rows of this section say:
it walks the hand-object contact intervals `_holds` lists and never reads `held`, so the recorder
branch's v9 left it unchanged and measured on v1 recordings as before (found by the 2026-09-25 review).
It reads the hand's turns, which the detector branch's confirmed turns change (the next table); no kept
recording holds anything, so no O05 count moved, and no validation case covers it.

On the kept recordings (20,000 body-steps each), v7 to v8: limp H02 8 to 0, R06 8 to 0, HX1 40 to 8;
random-order babble H02 27 to 3, R06 16 to 2, HX1 2,950 to 2,092, O01 66 to 69.

Definitions changed on 2026-09-25 and 26 by the detector branch (its detectors v9, merged with the
recorder branch's as observer_detectors_v10 on 2026-09-26: entry 70;
agents/observer_detectors_2026-09-25.md). Each was measured before it changed on the six kept babble
and limp recordings of world 74302686aba93fca and three production runs (segment 39; segment 40's
4:30 AM and 6:50 AM runs of 2026-09-25):

| what | before | now | measured reason |
|---|---|---|---|
| where a turn starts (`primitives._run_start`) | the lowest record since the previous turn's extreme | where the movement that reaches the peak starts: the low it rose from, at that low's last record within 5% of the line | in 1.7% of production's finger turns the recorded start came more than 0.5 s before the finger left its start level (one 17.4-degree bend was dated 2.85 s early) |
| when a turn is final (`primitives.turns_confirmed`, `daemon.final_cut`) | a turn ended where the read record ended | a turn is confirmed by a reversal of its line, a break in the record or 2 s without progress; the digester's cut moves back before any act on a turn still open | acts differing between one pass over segment 40's 9:33 AM run and live-sized windows with ten chunks of look-back: 10 of 7,927 (with twenty chunks, 7, each built on a turn the window's last record had not confirmed); now 1 of 8,158 (a trunk lift, P07: section 9) |
| the act timescale (H05, H07, H08 and HX2 join HX1) | none for those four; HX1 the whole turn's 10-90% rise within 2 s | the turn's fastest stretch moves the row's full line (15 degrees; 20 for a wrist axis) within 2 s (`line_s`; the whole turn's `rise_s` still reported) | every limp finger-isolation act rose over 8.5-16.5 s (fingers settling after a reset); read on the whole turn, the rule dropped 3-5 in 100 of production's spreading and palm-shaping acts, 3-6 in 10 of which moved the full line within 2 s |
| a hand act needs its own motors (H04, H05, H07, H08, HX1, HX2) | any movement of the joints | the moving joints' motors carried at least `MOTOR_ON` (0.001 of the command range) and their commands did positive work over the act (`command_peak`, `motor_work`) | a limp hand settling after a reset spread its fingers 30.8 degrees in 0.85 s, and no kinematic line separates that from a spread; every limp act's command was exactly 0 (17 of 17), production's at least 0.027. The cost: a finger relaxing against its spring is not counted, and production's palm shaping fell from 1,140 to 636 acts on segment 39 (entry 70) |
| hand episodes (`segmenter.hand_turns_from_rest`) | finger, wrist and closing turns never opened one | a finger, closing or wrist turn after 0.5 s of its own rest that rises within 2 s opens one | a hand moving only its fingers opened no episode; the three production runs gained 33.8-38.3 such openings per 1,000 body-steps (0 before) |
| H03 escape (`detectors._escape_mps`) | the object's speed in the two records after contact, under 0.10 m/s for a release | velocity along gravity is explained by a free fall from the last frame a part of the hand touched the object, the record's average against the fall's average over it | a box let go 3 cm above the floor falls at 0.25-0.35 m/s: v8 read it as escaping the hand |
| P05 support push | the force rise between the window's two end records | the rise holds: the least of the window's last 3 records stands the line above the most of the 3 ending at its start (`PUSH_HOLD_RECORDS`) | 90-98 of 100 P05 acts rested on one record of a floor force that flickers between 0 and 37 N from one 10 ms frame to the next; random-order babble 297 to 31 acts, production segment 39 186 to 16 |
| rolls P02-P04 (`detectors._body_drove`) | blocked only by the caregiver's touch or a moving object the world moves | also: the body's motors carried a command during the roll and did positive work over it | the locked positive case was a limp body turned by an outside torque; P02-P04 0 to 0 in every recording |

With them the validation cases changed (section 6): driven cases for the push and the rolls, the motor
commands recorded with every case, the grasp and palm-box cases rebuilt on a 2.5 cm cube, and a scripted
positive with a twin for every row the page showed beyond babble on 2026-09-25.

Not observable in this world (the crib): P11 climb onto a low platform, P12 back down legs first,
O09 put in a container, O10 take out. Their rows say why; they are never shown as "not developed".

## 5. Stages (section 25), each against its control (`tracker.py`)

| stage | rule | control |
|---|---|---|
| observed once | above babble in SIZE or in FREQUENCY: the share of OCCASIONS whose first measured act passes the 95th percentile of babble's occasions' first measured acts is above 5% (one-sided binomial; only when babble measured 20 or more occasions; rules v4), or it comes more often per body-step than babble on BOTH counts, occasions and single acts (the two-sample test of the next row, the larger of the two p-values); the smaller p-value of size and frequency is doubled (Bonferroni) when both were tested, and the row records which test passed (`by`). Dated where the running test of that reading began its last continuous stretch under the line (rules v5): a brief early pass that lapsed dates nothing | babble's occasion-level measure distribution, its occasions and its acts |
| repeated | 3+ separate OCCASIONS (an act starting at least 1 s after the previous act's start on the same body AND part - a hand, a finger, an object, the row's sub-row - opens a new one; a burst is one occasion), and more occasions AND more single acts per body-step than babble's in its own body-steps; each act count is divided by its burst spread E[n^2]/E[n] (quasi-Poisson; babble's from its summary's `acts_sq`). Dated like observed once | babble's occasions and acts, counted the same way; the exact conditional two-sample test (babble's count is a sample, not a known rate); both must pass (intersection-union: the larger p-value) |
| predicted before action | 3+ judged OCCASIONS, each judged by its first judged act, whose verdict beat its control more often than chance | movements: efference vs BOTH another body's command and the previous command's movement (persistence), chance one half; an act of the hands alone gives one unit per command in force during it (its own expectation against both controls over the command's whole hold), the act's commands one cluster: the passed and judged units divided by the design effect at its upper bound before the binomial test (`cluster_binom_above`; under no prediction skill it passed at most 4.72 rows in 100 at a line of 5: entry 72 (f)); one recorded before observer_attribution_v1, judged then on one command, is NOT JUDGED, with its count and its earlier reading beside the stage (rules v6, entry 72 (g)); touches: the contact head at least twice the start rate of skin untouched a record before, against the same verdict at the act's matched no-touch model steps (1, 2 and 3 s before and after the onset, at its place in its command's hold; Fisher's one-sided exact test); object effects: surprise over the act vs the 2 s before, chance one half |
| selected intentionally | 3+ OCCASIONS whose first act was a decisive planner choice (outcome 'best' beyond the tie band, reached the body), then more occasions per planner-authored step than per babble step (conditional binomial); with fewer decisive occasions it is not tested. An act of other parts is read on its credited command, the one in force at its peak motion. An act of the hands alone (rules v6, entry 72) is decisive only when every command in force during it was, with no hand joint goal asking for the other direction, and it counts for the planner or for babble only when all those commands came from that writer class; an act of several writers, or one recorded before observer_attribution_v1, counts for neither | the same run's babble-authored steps |
| generalized | repeated, acts on 3+ bodies from 2+ starting situations (posture at the start, objects involved) | - |
| retained | 2+ later windows of 3,000 steps after the window where the Repeats test first passed (no earlier than Seen's first pass); each window tested against that window's OCCASION count as a sample (the exact conditional test of two counts at 'half the rate') and judged together across every row, sub-row and no-overseer column that tests it with that window's Benjamini-Hochberg line; the newest window below its line is REGRESSED | the occasion count where it first repeated |

- The ladder is in order: a stage whose earlier stage is not reached keeps its numbers but is not
  reached (`waits_for`); observed once gates repeated, repeated gates the rest (generalized and
  retained included). A reached stage's first step is never earlier than its earlier stage's.
- Many tests at once: the four tested stages are judged with the Benjamini-Hochberg line across
  every row of the report that has enough acts to test (false discovery rate 5%). Each stage
  carries its p-value, its line (`alpha`) and `tests_together`. First production report with the
  line: 79 repeated tests, line 0.012; 88 novelty runs, line 0.0063. The line is q k / m (rules v5;
  before, the largest discovered p-value, which put the weakest pass exactly on its line).
- Counted = live (not an abandoned future, inside the run or across the lineage), not scripted, not
  probe / baseline / scheduled rest, no caregiver touch, not world-moved. Every stage is also
  computed on caregiver-absent acts (the no-overseer column). A stage with no control is
  UNMEASURED (None), never passed or failed.
- Lineage cuts: the supervisor resumes each continuation into a new run folder; a later run's first
  recorded step cuts every earlier run's steps from there (`daemon.lineage_cuts`), so steps a crash
  resume records again count once; a bin a cut falls inside counts only its records before the cut.
- Internal evidence under the behaviour: each row shows its grounded wellness change as one line, with
  the numbers in the line's tooltip, against stretches of the act's length ending 1 to 3 s before it
  (whatever happened in them, and those without the act); each occasion is one pair, the act minus the
  mean of its own stretches, with the 95% interval of those differences; never a stage.
- Beside the stages, never a pass (rules v6, entry 72): each row counts, over its acts of the hands alone recorded
  since observer_attribution_v1, two of the four separate stage readings (`act_stage_readings`: planner selected,
  no or not judged, since no noise band of the planner's goal distance is measured; achieved over each command's
  whole hold, yes, no or not measured), and H05's rows count `alone_inside`. A planner's selection is never read as
  intention, and the tie band of selected intentionally is unchanged.
- A stage the counting rules in force do not judge reads NOT JUDGED (the page's marker "Not judged"), with the reason
  and the count of acts (entry 72 (g)); "not measurable" stays for a stage whose control or records are missing.

## 6. Validation (section 27)

`python -m mimo_bridge.tools.observer validate --like <run>`: scripted positive, negative and
boundary cases in the production world, posed kinematically or, where a pose cannot make the forces,
driven by the body's own motors (92 cases since 2026-09-26, 89 of them in the lock's proposal, covering
28 detectors: end of this section; 66 cases covering 27 detectors from 2026-09-24 evening; 24 cases and
13 detectors before); the finger-curl sign; the painted panels' motif order by rendering; the
brief-versus-gradual harness with real physics; confusion counts per detector; replay determinism against the locked golden detections
(`observer_validation/locked/golden.json`; until 2026-09-25 `observer_validation/golden.json`, which
any run could rewrite). `validate --replay` re-digests the kept case recordings with the current
detectors and checks both the golden detections and every case's locked expectations, without a
world (1.5 GB, so beside training only while 6 GB stay free under the commit limit). Every verdict
reads its pass/fail lines from the locked set (end of this section).

Results (2026-09-24):
- 24 of 24 cases pass; the measured values match the scripted ones within 2e-5 degrees (index
  alone 40.0 deg, two open-close cycles 45.0, touch-then-close 30.0, wrist flexion 35.0, wrist
  rotation 35.0).
- The brief-versus-gradual harness: a 60 N push on the head for 10 ms against 1.2 N for 500 ms
  (the same 0.6 N s). The 10 ms peak separates them 107-fold (8.66 against 0.081 m/s2 away from
  gravity); a 2 s average separates them 4.9-fold: the fast lane's separation is 22 times the slow
  average's. The quiet record's peak is 0.081 m/s2.
- Four failures in the first runs were the scripts, not the detectors, and the scripts were fixed:
  the released box never left the hand (it was placed through the hand: the palm normal came from
  un-curling a curled finger and the cube kept the world's orientation); the pinch stopped 1.3 cm
  short (the thumb's joints alone cannot close it; the index now bends to meet it); the eyes aimed
  at the toy's resting place beyond their range (a box is now held 30 cm before the face); the
  fast-lane push was sideways on a head resting on the floor (friction held it; now upward).
- The scripted cases were recorded under detector v2. Each later definition was checked by
  `validate --replay` on the same recordings, and the golden file rewritten from it with the
  differences kept: v3 removed an R01 reach where the scripted box jumps onto a still hand; v5
  removed R02 touches the hand did not approach and made one reach's touch one act (it had been
  three, one per hand part). With v5, 24 of 24 recordings meet every expected present and absent
  detection.

The expansion of 2026-09-24 evening (observer_validation.md; review priority 9): 42 cases, 23 of
the 27 covered detectors with both a positive and a negative case. The grasp chain broken one link at
a time on a grasp settled by physics (contact without opposition, opposition without lift, lift
without retention, lift and hold, carry and place, release then withdraw, the toy moved into a still
hand and its twin, a resting hand pushing the box); G01, G03 (at 12 cm and at reaching distance, 30
cm), G05, R03, B06, H05, HX2 and HX1. Expectation qualifiers (@own, @world_moved, @scripted,
@caregiver_touch) judge the detectors' provenance flags. Five cases exposed detector defects, fixed as
v8 (section 4). Recorded in the paused production world at 9:05 PM PDT (checkpoint 397,125): 66 of 66
cases met every expectation on v8, the replay of the recordings gave the same detections for 66 of
66, and the finger-sign, panel-order and fast-lane checks passed. `validate dry-run` runs every case
in the saved world model without BridgeEnv (about 1 GB, beside training).

Babble from production's saved worlds (entry 49), recorded in the same pause from checkpoint 397,125
(`baseline --from-checkpoint`, 8 bodies x 2,500 steps, 588 s): random orders and a limp body. It is
the comparison since 9:16 PM PDT; the reset-world recordings are kept as `random_orders_reset_world`
and `limp_reset_world_vision_off`. On the same detectors, 7 of 55 rows differ between the two
starts at p < 0.01 (`baseline --compare`): repeat contact 25 against 132 occasions, reach 527
against 808, push 6 against 56, O06 0 against 26, return look 37 against 100, one-fingertip touch
327 against 204, whole-hand close after touch 66 against 36 (saved worlds first). Production's
bodies lie 0.59-0.65 m from the crib's centre and their actuators run at 33-41% of full force, so
chance reaching and toy contact are rarer and self-touch more common than a fresh world suggests.
`observer_validation/babble_rates.json` gives every detector's rate under each chance process.

Since the deploy of 2026-09-26 the comparison is babble recorded under recorder v2 from the held run's
checkpoint 415,075 (H84_fresh_drivers_20260918_seg40_resume_20260925_200754): random orders, a limp body
and random actuator commands, 8 bodies x 2,500 steps each with the run's vision, recorded 9:38-10:08 AM
PDT in the pause 20260926_093540_held, then placed and summarized with detectors v10 at 10:48-10:52 AM
(20260926_104815_held; receipts under Saved/gof_observer_20260924/pauses/). Each of the three summaries
carries all four v2 field groups and leaves no capability unmeasured. The checkpoint-397,125 recordings
stay beside them as `<kind>_ckpt_397125`, where R04 and the H02 family are unmeasured for want of
contact normals (entry 66).

Earlier babble baselines (`baseline --like <run> --kind random_orders|limp|random_ctrl [--vision like|on|off]`),
each 8 bodies x 2,500 steps = 20,000 body-steps from a world reset, digested with v5:
- `random_orders`: the body's own controller given random orders (orientation, a position within
  3 cm of each part, strength up to 0.3, held 250 ms): what stages and novelty are judged against.
  Recorded 2026-09-24 09:19-09:27 PDT with the eyes rendered as in training (entry 37): 10,910
  acts, 104 episodes. The first recording, vision off (9,433 acts), is kept as
  `random_orders_vision_off` for comparison.
- `limp`: no command: 88 acts (390 under v2: the removed ones were world-moved touches, "repeat
  contacts" without an approach, still-eye "tracking" and a limp hand's "holds"). Vision off.
- `random_ctrl`: uniform actuator noise: 12,099 acts. Reported, not judged against. Vision off.

### The pass/fail lines are locked (operator, 2026-09-25)

"Lock the pass/fail lines so the agent building a feature cannot edit them. Right now the same
agents that change the observer's detectors also rewrite its file of expected answers, golden.json."
Until then `validate --write-golden` overwrote golden.json with the detections of the moment, and
each case's expected-present and expected-absent lines and the known-defect register sat in
validate.py beside the detector code (entry 60).

- The locked set, `docs/gain_of_function_v4/observer_validation/locked/`: `expectations.json` (per
  case: expected present, expected absent, the scripted truth values and `qpos_hash`, the
  fingerprint of the recorded trajectory; plus the known-defect and boundary registers),
  `golden.json` (the detections each case's recording produced), `lock.json` (the SHA-256 of both
  files' canonical JSON, so line endings and indentation do not matter; who approved them, when,
  and the history).
- Verdicts read only the locked set (`validate --like`, `--replay`, `dry-run`). Each run first
  checks the files against lock.json and against their last commit; a changed, unreadable or
  uncommitted locked file fails every verdict (`lock.intact` false in the report). A case in
  `validate.CASES` that the locked set lacks is run and listed under `not_approved`, and never
  counts. A locked case missing from the code, one without a golden recording, or one whose new
  recording has a different fingerprint (`scenario_changed`: a box moved closer or a new pose is
  not the approved case) fails.
- Builders propose: `--write-golden` now writes `observer_validation/proposals/<id>/` (the proposed
  files and `proposal.json`, listing every case added, removed or changed against the locked set).
  Nothing a builder runs writes the locked folder.
- The operator approves, in their own terminal (from `Source/Python`, with the project's Python):
  `python -m mimo_bridge.tools.observer.validation_lock approve <proposal folder>`, or
  `approve --current` for the set as it stands. It refuses without an interactive console, shows
  the change against the locked set as it is at that moment, asks for the proposal's id and the
  approver's name, then writes the locked files and lock.json and commits them as "Operator <name>
  approves observer validation's pass/fail lines". `validation_lock status` prints the state and
  the proposals still waiting.
- Agents are stopped before the checks by two layers (commit cc5e57ed5): deny rules in
  `.agent/settings.json` refuse file edits to the locked folder, the lock module and its test, the
  guard hook and the three settings files that could switch the hook off; the PreToolUse hook
  `.agent/hooks/protect_validation_lock.py` refuses shell commands that name those paths together
  with a writing operation, and any run of `approve`. Checked live on 2026-09-25: edits of
  lock.json, validation_lock.py, its test, the hook and the settings file, a new file in the locked
  folder, and a Bash and a PowerShell copy naming the locked folder were all refused; reading the
  locked files with the Read tool or `cat` stays allowed.
- Limits: agents run as the same Windows user as the operator, so an agent that writes and runs
  its own script can still reach the files. The hash and last-commit checks then fail validation,
  and `tests/test_validation_lock.py` fails unless the newest commit touching the locked set is the
  operator's approval. The cost of the guard: agents, the coordinating session included, can no
  longer change the builder's settings files; the operator edits them.
- State when built (lock written 12:14 AM, committed 12:22 AM on 2026-09-25, d75d3aeaf): 66 cases,
  the lines exactly as the code held them at 50ce48c7a, and the golden detections recorded at 9:05
  PM PDT on 2026-09-24; marked not approved until the operator runs `approve --current`. The
  end-to-end replay waits for a training pause: the memory guard refused it beside training at
  12:27 AM (66.9 of 70.8 GB committed; 1.5 GB of work needs 6 GB left free above it).

### The qualified observer and the lock's proposal (2026-09-26; the operator's decision of 2026-09-27)

- What validation judges since the merge (entries 70-71): the 92 cases of `validate.CASES`, 89 of them
  in the proposal, covering 28 of the 55 detectors (26 with both a positive and a negative case; H08
  and O01 have a positive only). The support push, the two rolls, the two lifted grasp cases and the
  held pinch are driven by the body's own motors in the world's physics (the rolls' twin by an outside
  torque with every command zero); every case records its motor commands. The scripted truth values
  are judged: a case fails when its first detection of the capability reads one more than 5% off, or
  reads none (`validate.TRUTH_TOLERANCE`). The scenario
  fingerprint is `observer_scenario_v2` (`validate.SCENARIO_FINGERPRINT`): record by record the joint
  positions, the motor commands and the world's eye-mover flags, and once the world model's structure
  fingerprint (the boxes' sizes among it); the positions of what the world moves (the toy, the adult's
  head) are not in it. Every approved fingerprint changed with it, so all 66 locked cases read
  `scenario_changed` and no locked line is judged until the operator approves a new set: the
  qualification judged the code's own lines.
- The qualification (receipt Saved/gof_observer_20260924/pauses/20260926_104301_held, 10:43-10:47 AM PDT
  on 2026-09-26, the candidate tree at f271bc050): four gates, all passed. The world-free observer and
  dashboard suites exited 0; the crib-world recorder proofs exited 0; `validate --like` on the held run
  H84_fresh_drivers_20260918_seg40_resume_20260925_200754 exited 0 (92 cases, no unexpected failure or
  pass, exactly the 66 declared scenario changes, so every line of the code was met, R04's and H02's
  included, none excused; the finger-sign, panel-order and fast-lane checks passed); the crib-world
  end-to-end digest exited 0. The first attempt, from 9:35 AM, stopped on the virtual skin's clock
  (entry 71).
- The proposal, 20260926_104649_4173cc (in C:/Users/bhart/Documents/UEMA_observer_candidate/docs/
  gain_of_function_v4/observer_validation/proposals/, untracked as proposals are; written at 10:46 AM by
  that validate run): 89 cases. 64 locked cases change (every fingerprint, and the lines listed in
  agents/observer_detectors_2026-09-25.md, PROPOSAL, with grasp_opposition_no_lift's R04 and H02 turned
  absent: entry 67); 25 are added (24 of the detector branch's and the recorder branch's held pinch);
  p02_roll_to_side and p02_half_roll are removed; the golden detections are observer_detectors_v10's.
  Its expectations and golden files are identical, by the SHA-256 of their canonical JSON, to those of
  the 10:02 AM proposal 20260926_100239_b22fec (checked at 4:58 AM PDT on 2026-09-27).
- P02 (roll onto the side) is not locked (entry 71): p02_roll_to_side, p02_half_roll and
  p02_body_rolled_by_outside_torque run, are recorded and are reported beside the verdict, and stay out
  of the proposal's expectations, boundary register and golden detections (`validate.NOT_LOCKED`). The
  proposal's `why`, which the approval prints, names them and says the operator can refuse it and ask
  for one that keeps them. In the qualification's run p02_roll_to_side's "P02 present" was unmet and the
  other two's "P02 absent" met.
- The operator decided on 2026-09-27 to approve this proposal with P02 left out. The approval is theirs
  alone, run from their own terminal in Source/Python: `..\..\.venv_mimo\Scripts\python.exe -m
  mimo_bridge.tools.observer.validation_lock approve C:\Users\bhart\Documents\UEMA_observer_candidate\docs\gain_of_function_v4\observer_validation\proposals\20260926_104649_4173cc`.
  Approval was recorded by Brian at Sep 27, 2026, 12:04 PM PDT. The adoption and replay jobs
  ran from 02:14:30 PM to 02:14:43 PM PDT that day, both exiting successfully. All 89 approved
  recordings replayed identically and met their expected answers, with no unjudged locked case.
  Receipt: `Saved/gof_observer_20260924/pauses/20260927_141430_held/receipt.json`.

## 7. Production operation

- Current-run reporting correction, Sep 28, 2026: the service and CLI now pass the selected run
  explicitly to `build_report`. A failed empty launch sorted last and had incorrectly supplied
  recorder health, planning history and training-vision metadata. Four focused tests pass; the
  refreshed service selects the actual production run and retains the same historical detection
  and episode totals. Training state and its pause were unchanged. Evidence:
  [before/after report](coordinator/audit_2026-09-27/observer_current_run_repair_2026-09-28.json).

- Memory: the digester sets one math-library thread before importing numpy (1,545 MB of private
  memory at import with one thread per core, 67 MB with one), and shares one compiled model per
  kinematic structure across bodies (each holds a 380 MB physics arena; eight copies made the first
  production pass 4.8 GB). Steady state about 0.55 GB; about 1 GB while catching up a backlog.
- The digester works a backlog in windows of 20 chunks per body (widened up to 80 when one episode
  does not fit), commits after each and writes `daemon_heartbeat.json`; a window failing three
  passes in a row is set aside (`quarantined`, its steps not counted); a chunk that cannot be read
  three times is given up.
- The learner lane is waited for only while it writes (no error, no refusal in its status, a chunk
  closed in the last 10 minutes).
- The supervisor restarts a digester whose status and heartbeat went silent for 30 minutes
  (`OBSERVER_STALL_S`), recording the stop in `stalled_digester.json`.
- `digest` / `report` on the command line take the service's lock (never beside it) and never
  finalize the training's active run.
- The ability scheduler (another agent's system) pauses training every 3,000 updates. Two of its
  defects stopped training on 2026-09-24 and were fixed (section 8, entries 29-30).

## 8. Decisions and deviations log

Each entry: what the addendum says or leaves open, what was done, why.

1. **Section 24 "multi-timescale wellness": measured, not wired into wellness.** The addendum asks
   that fast transients not hide inside long averages. The observer measures them (world fast lane
   per 10 ms frame, the learner's error per body per 50 ms step, peaks, areas, recovery). Feeding
   them into the learner's wellness would change what it learns; that is a separate, qualified
   change (a learning intervention, with its own kill criterion), not measuring equipment. Trigger
   to revisit: the fast lane shows transients the learner's wellness misses, measured on production.
2. **Play areas (section 29) are not added to production.** A container zone or a low platform
   changes the learner's world mid-lineage: world and scene semantics are the operator's decision
   (the style guide, the parking test). The four detectors that need them exist and say "not observable
   in this world". Next step, the operator's: approve a scene option and the segment it starts at.
3. **No vision-language model** (section 23 reserves one for rare offline review). Gaze is ray
   geometry; wall regions are tagged by motif. Rendered audit pictures every 10 minutes, one body.
4. **The dashboard's four stages and the addendum's six are shown side by side, not merged.**
5. **Chance is babble through the body's own controller** (random orientation targets, position
   targets within 3 cm, random strength up to 0.3, held 250 ms), not uniform actuator noise: the
   chance process closest to the learner's own motor babble. `random_ctrl` and `limp` are recorded
   for reference. The hand controller rejects NaN position targets, so each order asks for where
   the part is, moved by up to 3 cm.
6. **"Observed once ... above a random/baseline threshold"** is read as a TEST: acts beyond babble's
   95th percentile come more often than babble's 5% (entry 22 replaced the first reading, "one act
   beyond the 95th percentile", which babble itself passes 95% of the time once it has 59 acts).
   Entry 32 adds the second reading of "above the threshold": more often than babble.
7. **The toy's hand-off is scripted**: toy holds, carries and episodes are marked and never counted.
8. **The eye cameras sit inside the head's sphere**: rays pass through the infant's own head.
9. **The east panel's motifs run the other way** (measured by rendering); validation re-checks it.
10. **One act counts once**: same capability, sub-row and key on the same step merge; an act
    overlapping one already finalized for the same key is not emitted again.
11. **Disk:** the recorder never writes below a free-space reserve (10 GB in production) or beyond
    its folder budget, and lists every dropped step. Digested raw chunks are kept 72 hours.
12. **The world model is saved once per kinematic structure, compressed** (20.8 MB raw, 0.26 MB).
13. **Found and fixed in another agent's code:** `LemLite._plan_settings` crashed a learner with
    the planner off; it now returns None.
14. **The supervisor keeps the digester alive** (`ensure_observer`).
15. **The digester's memory** (section 7): one math thread and one shared compiled model.
16. **Compressed records and a column store** (section 3): 156 bytes a detection on disk, 76 bytes
    in memory (the 18 columns of `store.FIELDS` in store v3; 45 before the credit and no-touch
    columns), the report judged from columns in one pass. Store v4: 24 columns, 100 bytes (section 3).
17. **Commit, truncate, seal, move aside** (section 3): a stopped digester never counts a window
    twice; a changed definition never mixes with the old.
18. **A stuck digester is restarted** (section 7) from its heartbeat, not a process check alone
    (a detector loop that never ended once looked alive for 660 CPU seconds).
19. **Primitive and detector definitions measured against production** (section 4 table).
20. **Episodes open on the infant's acts, not on resting contact** (section 4): with every floor
    contact salient, every episode ran to its 10 s cap. They still do under continuous babble (40
    episodes in 7,600 body-steps, 2026-09-24 review); see open items.
21. **Repetition counts occasions, not episodes or acts:** a burst of acts inside a second is one
    occasion (one scripted 12-step eye sweep produced 24 eye-movement acts). Episodes capped at 10 s
    made "3 separate episodes" a count of 10 s windows.
22. **Two-sample tests against babble** (entry 6, section 5): babble's own count is a sample of
    20,000 body-steps, so the chance comparison is the exact conditional test of two counts,
    not a Poisson test against babble's rate as a known constant.
23. **Many tests at once: the Benjamini-Hochberg line** (section 5), for the tracker's rows and the
    novelty runs; since rules v5 the line is q k / m, and retention's windows are judged with it too
    (entry 68).
24. **The ladder is in order** (section 5).
25. **Novelty is judged against babble** (section 36, "random babble labeled as skill"): a run babble
    makes as often is chance co-occurrence.
26. **World-moved acts are not counted** (section 4 table).
27. **Crash resumes count once across run folders** (lineage cuts, section 5).
28. **Evidence controls strengthened** (section 5): the efference comparison adds persistence (the
    shuffled-body control alone is beaten for free by any learner that knows the body follows its
    orders); surprise is read over the act, not its first step; the intention verdict needs the
    act's own order's planner record; no contact verdict before a patch has a base rate.
29. **Found and fixed in another agent's code (the ability scheduler), after it stopped training:**
    its eight-world save check called a present file missing. The file's path (a resume folder's
    companion save) was 262 characters, past Windows' 260; `Path.is_file()` says False there. It
    now checks through `storage_locations.long_path`, as the writer and `continuation_state.load`
    do (test `test_a_companion_past_260_characters_is_verified_not_called_missing`, which fails on
    the old check). The pause it had stopped with was re-verified with the fix and released
    training at 07:51:59 PDT.
30. **Found and fixed in the ability scheduler:** a task waiting for memory requested a pause whose
    lowest step came from the training statistics, which lag the snapshot the scheduler itself
    asked for (384,000 against the 384,025 checkpoint the gaze test waited for). The pause listed no
    task, released at once, and the 4 GB test ran beside the relaunched trainer: committed memory
    reached 68.5 of 68.8 GB ("the paging file is too small"), four stop-and-restart cycles in 30
    minutes. The pause now covers the task's checkpoint (`max(live step, checkpoint step)`; test
    `test_a_memory_window_covers_the_checkpoint_the_task_waits_for`, which fails on the old code).
31. **The segment's name shortened nothing:** `H84_fresh_drivers_20260918_seg38_observer` plus the
    supervisor's `_resume_<time>` put the companion save at 262 characters (entry 29). Future
    segment names should stay short, or every long-path reader must use `long_path`.
32. **Observed once passes by size OR by frequency** (section 5). With the size test alone, an act
    that happens far more often than babble but no bigger could never be "seen", so its
    "repeated" waited forever. Measured 2026-09-24 08:44 PDT: G01 had 297 occasions against 132.9
    expected from babble in the same body-steps, while its acts beyond babble's 95th percentile
    were no more common than babble's own 5% (share test p = 1.0). Both readings are the
    addendum's "above a random/baseline threshold"; the doubled p-value pays for having two
    chances. For a row seen by frequency, "repeated" adds only the 3-occasion minimum.
33. **A later stage never starts before an earlier one** (section 5): each stage took its own first
    qualifying act, so a later stage could be dated first. The page at 08:56 PDT showed "Left thumb
    moves alone" with varied starts at step 383,825 and repeats at 383,874, and the timeline put
    "Independent eye movement - varied starts" (383,288) before "seen" (383,331): generalized and
    retained used repeated's step from before the ladder moved it. Now every later stage's first
    step is at least the earlier stage's, generalized and retained wait for repeated like the
    others, and the 09:07 report had 0 rows out of order across rows, sub-rows and the no-overseer
    column.
34. **"Seen" is dated when its evidence first held**, as "repeats" always was: the first act by
    which the running test of the reading that passed was below the line, not the first act of
    the row (for a row seen by frequency that act says nothing on its own). Since rules v5 both are
    dated where that running test's last continuous stretch under the line begins (entry 69).
35. **The fast lane is shown against babble and a limp body** (section 24; the style guide: anchors
    before numbers). Measured 2026-09-24 on 32,000 frames of 10 ms each: support falls of a quarter
    body weight held 30 ms came 722 per 10,000 frames in production, 823 under random-order babble
    and 0 in a limp body, so they come with movement. The page now gives one line per lane with
    those three rates (the report's `fast_lane_chance`, read from the baselines' digests) instead
    of 16 per-body counts without a reference.
36. **The page speaks in words** (the reading gate): novelty tokens are clauses ("the left middle
    finger straightens, then the left hand opens", not "digit L middle straighten then curl L
    open"); simulated life is "summed over 8 bodies (2 minutes each)", since the sum read as each
    body's life; the horizon's three numbers share one line with one explanation; the planner's
    allowed look-ahead is stated as lem_model computes it (50 ms plus 50 ms per depth beating
    "nothing changes" in more than 50 of 100, counted from the shortest).
37. **Babble renders the eyes as training does** (section 36, "random babble labeled as skill"; the
    first open item of the 09-24 review). The eyes' motion reflex runs only on rendered images, and
    the first baselines ran with vision off, so babble had no reflex while production did.
    Re-recorded with the run's own vision setting (rendered at each burst's last frame, as
    train_mimo does): babble's independent eye movements came on 356 occasions instead of 154.
    Judged against it, "Independent eye movement" no longer repeats: 459 occasions against 466.4
    expected from babble (report of 09:29 PDT), where the vision-off babble had made it look like
    more than twice babble's rate. The hand rows moved little (finger spreading 81 to 87
    occasions, palm shaping 196 to 202). A baseline of the other vision setting is moved aside, never
    mixed, and the page says whether babble's eyes matched training's.
38. **Stage history** (section 32, "first seen, repeated, chosen, lost"; open item "stage history"
    of the 09-24 review). A stage that passed and later fails is listed on the timeline as lost, at
    the step it was found lost, until it passes again; every loss and return is appended with its
    p-value and line. A new detector or analysis version, new counting rules
    (`tracker.RULES_VERSION`) or a new babble recording starts the record afresh, with a `reset`
    line in the history: a changed definition or control is not a loss.
39. **Occasions are counted per body AND part** (independent review of 2026-09-24, finding 1: "merging
    both hands creates two of the five headline passes"). Merged per body, two hands moving at once
    were one occasion, and babble moves both hands almost without a pause: finger spreading's babble
    made 81 merged occasions against 305 and 306 per hand, so the learner's 224 merged occasions
    read 2.5 times babble's rate where each hand read 1.11 and 1.14 times (page of 09:07 PDT). The
    tracker and the babble summaries now count per (body, sub-row); the kept babble recordings were
    re-summarized without simulating again (peak 152 MB).
40. **"More often than babble" needs both counts** (the same finding, one level down). Within one
    hand, babble's movement still runs on without the 1-second pauses that separate occasions: at
    09:50 PDT finger spreading had 1,025 occasions against babble's 894 but 3,469 acts against
    babble's 4,578. Occasions alone reward pausing. The frequency reading of "seen" and the
    "repeated" stage now need occasions AND single acts above babble's (the larger p-value of the
    two, an intersection-union test), and a cell that passes one count and fails the other says
    which. Finger isolation passes both (1,281 acts against 816, 988 occasions against 689 at 09:40).
41. **The page, after the independent review** (32 findings; the reviewer's report is summarized in
    WORKLOG.md). Changed: "detected" (an act the detector found) is kept apart from "beyond babble";
    babble is defined in the first headline line; the headline is one point per line and names the
    capabilities, the planner's ties, the rows below babble and how many passing detectors have
    scripted test cases; the stage definitions state what the tracker computes (both counts, per
    part, the doubled line, the 95th percentile's direction, what "decisive" means); a size pass names
    its quantity and line ("322 of 1,589 acts had an amount over 34.0°"); the body map lists names and
    the stages any row reached; the regression watch says when nothing was checked; the eye rows'
    excluded acts name the world's eye movers, not the toy; the fast lane is per 100 s of one body's
    life with the variation rule spelled out and no goal line claimed; the horizon states its noise
    (3.5 per 100), its cap (150 ms) and its window; catalog codes and pattern hashes moved to
    tooltips; the site glossary no longer links words inside this view (its "capability", "stage" and
    "frame" mean other things). Removed: the wellness change under each row, until its reference
    (the same change over equal stretches without the act) is measured (the style guide: an anchor before
    a number); shown again since analysis v5 as one line against its references (entries 68-69).
42. **H84 stopped 1:29-3:36 PM on 2026-09-24: C: full; two of the causes were mine.** (a) Entry 30's
    fix to the ability scheduler was written at 08:04 AM, but the scheduler process had started at
    07:51 AM and never reloaded: at 12:59 and 1:05 PM it again opened pauses listing no test (lowest
    step 390,400 from lagging statistics, against the 390,425 checkpoint the gaze test waited for),
    released them at once and ran its 12 GB gaze test beside the relaunched trainer (1:13-1:26 PM).
    (b) The archiver moved nothing all day (the item above), so about 31 GB of checkpoints sat on
    C:. The memory squeeze grew Windows' page file onto C:, which fell to 0.8 GB free; the hourly
    watcher's 8 GB line stopped training, and its final save failed for lack of room: steps
    390,650-391,075 (425 updates, about 14 minutes of training) were not saved. Repaired: both
    processes restarted at 3:31 PM on their fixed code; training resumed from the last complete
    checkpoint, ckpt_step_390650, as segment 39 (a short name, so companion paths stay under 260
    characters), by `docs/gain_of_function_v4/resume_after_disk_stop_2026-09-24.py`, whose
    preconditions check that both processes started after their fixed files were written. The
    lost 425 updates re-run from the saved state; the observer's lineage cut drops the unsaved
    steps' records, so nothing is counted twice. The 7 scheduler stops between 9:57 AM and 1:12 PM
    (two of them the gaze test's retries) are measured in WORKLOG.md.
43. **Detector v6: per-effector episodes, fixations from the foveal bundle, support falls below the
    usual level** (review priorities 7 and 8; observer_segments.md; section 4's second table). Every
    non-gaze detection on the same chunks was unchanged (19,267 production, 11,689 babble, 56 limp).
    Units per 1,000 body-steps went from 5.1 to 145 in production (42 of them pairs).
44. **Detector v7: a look's target is the thing the centre ray lands on, not the majority of the nine
    rays** (a departure from the review's example rule). The review asked for fixations from the
    bundle, "for example: majority of foveal rays intersect the same entity/region for N
    milliseconds". Built that way (v6), the rule stopped seeing looks at toys: the ring of rays sits
    at the fovea's 12-degree half-angle, so a 6 cm toy fills 5 of 9 rays only within about 15 cm of
    the eye. Of production eye-steps with the centre ray on an object, 12.9% had 5 or more rays on
    it, and the scripted look at a box 32 cm away (g02_look_at_toy) failed. The majority did not
    make holds stiller either: two-record holds turned the eye 5 degrees or more in one step in 155
    of 455 majority holds (34%) and 206 of 620 centre-ray holds (33%). The addendum reads the same
    way: G02 is "foveal frustum remains on the same object", and section 23 asks for a small bundle
    "rather than one infinitesimal ray" to catch a target near the axis, not to demand that the
    target fill the fovea. v7 takes the centre ray's thing (an object, a body part, the adult, a
    motif cell); only where the centre ray lands on a surface or on nothing does the majority
    decide (a large thing seen through a gap: 52 of the 12,269 production left-eye steps whose
    target was a thing). Every measure the review listed is recorded (target, share of rays, aim
    error, dwell, target switches), and a return look still needs the target to leave every ray.
    Left eye, v6 to v7 on the same chunks: ball fixations 29 to 119, looks at the adult 1,045 to
    1,320; G02 864 to 1,174, G03 73 to 133, G05 135 to 235, R03 41 to 152; babble G02 282 to 401,
    R03 10 to 105; limp G02 1 to 31 (a still eye resting on the mobile or a critter: chance the
    babble and limp recordings measure). A bundle of the addendum's size (rings at 2-6 degrees) needs
    the recorder to cast it: open item.
45. **The learner's lane is read one record behind the world's, and each act is credited to the
    order in force when it moved most** (review priority 2; observer_stats.md). The trainer lets the
    learner read the frames of burst t - 1 before it latches the order for burst t, so lane step t
    describes world record t - 1 (segment 39: 3,304 of 3,304 efference records and 3,304 of 3,304
    planner latches). The join had read lane step s for world record s, so the touch verdict judged
    the prediction for the record before the touch (`evidence.TRAINER_LANE_LAG = 1` now). An act's
    peak is the record where its own parts moved most; the credited order is the one the recorder
    stamped on that record. On every kept record: 22,301 of 41,458 counted acts (53.8%) are credited
    to a later order than the one at their first record; the efference verdict changed for 10,173
    of 39,317 judged movement acts (57.6% passed before, 60.1% after). Since analysis v5 an act of the
    hands alone is credited to the order in force a record before its peak (entry 69).
46. **Two touch repairs in the join** (observer_stats.md): a touch of the infant's own body was judged
    on the outside-thing bit (457 of 1,619 hand and fingertip touch acts moved to the own-part bit),
    and a patch already touched a record before could carry the verdict with its probability of
    STAYING touched against a base rate of STARTING. The verdict now reads only patches untouched a
    record before, and only where the world record shows a touch starting on them (31 of 2,054
    touch acts lost their verdict).
47. **The occasion is the tested unit** (review priority 5; tracker rules v3 and v4; section 5's
    table). Acts inside one burst are correlated, so tests that counted them as independent trials
    were overconfident. Lineage store at 7:16 PM: 41,000 counted acts in 20,257 occasions (finger
    spreading: 9,655 acts in 2,953 occasions). The size half of "seen" uses each occasion's FIRST
    measured act, not the burst's largest: a longer burst would otherwise get more draws at the line.
48. **The touch verdict's chance rate is measured** (review priority 6). At matched no-touch moments
    (one a second, the act's own skin patches, a touch possible and none starting) the same verdict
    passes for whole-hand close after touch in 1,405 of 2,990 (47%), one-fingertip touch 4,361 of
    14,062 (31%), fingertips meeting 1,228 of 6,152 (20%), hand to face 388 of 1,846 (21%) and
    repeat contact 35 of 676 (5.2%); at touch onsets 78%, 70%, 36%, 16% and 8%. The first three
    are above their chance rate, the last two are not; against one half, fingertips meeting had
    read p 0.998.
49. **Babble from production's saved worlds replaces the reset-world babble as the comparison**
    (review priority 4; babble_saved_worlds.md). At checkpoint 396,050 production's bodies lay
    0.59-0.65 m from the crib's centre (reset-world babble reached 0.26-0.37 m in 2,500 steps), their
    actuators gave 33-41% of full force (fatigue under body function 'modulate'), and the caregiver
    was away from four of the eight. `baseline --from-checkpoint` restores each body as training
    resumes it, checks it field by field, and counts the caregiver's events without writing them.
    The reset-world recording moves aside as `<kind>_reset_world`.
50. **A run is sealed only after its recorders were quiet for 10 minutes, and a sealed run they
    write to again is reopened** (6fe437dd1). Segment 39's first run was sealed at 3:38 PM, two
    minutes after it started and before its first chunk: steps 390,650-392,949 (18,400 body-steps)
    never reached the report until the v7 re-digest.
51. **Planner verdict so far** (review priorities 1 and 3; planner_ties.md). In segments 36-39 all
    19,088 planner choices tied: the largest gap between the chosen order and the best alternative
    was 0.053 against a tie band of 0.0734 on the 0-to-5 wellness sum. The planner's objective is
    distance to a remembered goal moment plus effort (prospective wellness weight 0), so it cannot
    follow predicted wellness: the chosen order's rank by predicted wellness averages 2.98 of 5 over
    those 19,088 choices (3.0 = unrelated). Swapping in another body's order moves the touch forecast by 0.5% and the
    proprioception forecast by 2.4% of the whole predicted change (0% = the order is ignored), so
    the forward model is the first flat stage on the evidence so far. Depth cannot change a choice
    as built: the chain picks its first order with the depth-1 search before it imagines further,
    and forces one step for wrist-relative hands. The saved-state probe that separates the three
    stages and compares depth 1 with depth 3 on executed outcomes
    (`docs/gain_of_function_v4/probe_planner_candidates.py`) is registered with the ability scheduler
    from step 399,050, every 3,000 updates.
52. **The eye ability test's failed replays came from the graphics driver** (gaze_probe_replay.md).
    The integrated graphics drew one unchanged scene 400 times as 23 different pictures when the
    critters' semi-transparent wings overlapped (1 to 112 pixels, one brightness level each), and the
    test's digest includes the vision token computed from those pictures. The probe harness now gives
    a repeated scene the picture it got first (fingerprinting what MuJoCo hands OpenGL), and its
    snapshots carry eight per-frame trackers `save_full_state` leaves out (the mobile's chime gate
    among them).
53. **A commit of mine set the detector version back for two minutes** (ad71750c9, 8:02 PM). It built
    daemon.py from the working copy with another track's version line replaced by a remembered older
    text, after that track had committed v6; b12bde2bd restored the line at 8:04 PM. A file holding
    another track's hunk is now committed with that hunk's text read from HEAD at commit time.
54. **Detector v8: five defects the validation cases exposed** (section 4; 520357913). Each was
    measured on the kept limp and babble recordings before it was changed, and each change was
    approved against the scripted case that fails without it.
55. **Two limits of the grasp detectors stay open** (observer_validation.md, defects 6 and 7). R04's
    opposition angle is measured at the object's centre from contact positions; on the world's 6 cm
    boxes most grasps this hand can make read 83-126 degrees, so its 90-degree line runs through
    them, and a real pinch of the toy's thin pad reads 30-44 degrees. Judging opposition by contact
    NORMALS needs the recorder to store them (a recorder format change). H02's "opposed" is
    thumb-and-finger co-contact, not opposition; its fix depends on R04's.
56. **The evening's pause was finished by a second script** (continue_pause_2026-09-24_evening.py).
    A same-state vision audit became due while the pause runner recorded the limp bodies; the runner
    takes no new job and lifts the pause as soon as its own jobs end, which would have put a 12 GB
    audit beside the relaunched trainer (the squeeze that filled C: at 1:29 PM). The runner was
    stopped at 8:50 PM with its recording left running, and the continuation ran the rest, the
    audit included, before lifting the pause at 9:06:48 PM.
57. **The ability scheduler counts failures per test version** (301867ec1). At 9:12 PM it stopped
    training again for the eye test, whose same code had failed twice at 397,050: failures were
    counted per checkpoint, and the resume's new run folder made a new candidate. A test version is
    now the script plus the files it declares in `depends` (the probe harness for every task);
    two failures in a row hold it at any checkpoint until one of them changes. Until the eye test's
    second replay fix is confirmed, its manifest entry is held (`first_step` 999999999 with a
    written reason), as is the new remembered-hand-goal comparison until the scheduler runs on its
    validator.
58. **Vision prediction regressed again in production** (found by the dashboard track; outside the
    observer, recorded here because the planner's forward model is the first flat stage of entry
    51). On 8 identical saved moments the model qualified at step 380,500 predicts the next picture
    at 1.24 times the unchanged-picture error and the step-397,125 model at 3.09 times, worse on all
    8 (Saved/gof_takeover_20260924/repair/visual_audit_397125/result.json); the live audit reads 3.1
    to 5.4 times since 9:33 AM. The morning's regression (4.49 times worse, VISUAL_REGRESSION_2026-09-24.md)
    stopped production. The 64-moment audit with its isolated treatment arms stopped at 9:29 PM
    without a result: its input-conditioning arm was written for the heads before the morning's
    repair and drops the repaired layer's own input scaling. A dedicated track now fixes the audit and
    diagnoses the cause (docs/gain_of_function_v4/agents/vision_regression.md); training continues
    meanwhile, since no repair exists yet to stop for.
59. **Training stopped 9:28-9:45 PM because the supervisor read planned stops as failed starts**
    (49ddaaee6). It reset its failed-start count only after a trainer ran 10 minutes, and tonight's
    relaunches at 8:34, 9:06 and 9:24 PM were each stopped within 5 minutes, after saving a new
    checkpoint, by comparison windows and coordinated pauses (my vision-audit pause among them). After
    the 9:28 PM stop it refused to relaunch ("three failed starts; repair required"). A start that
    saved a newer checkpoint no longer counts as failed. The count lives in the supervisor's memory
    and a supervisor attaches only to a running trainer, so training resumed as segment 40
    (`H84_fresh_drivers_20260918_seg40`, a short name: every relaunch appends "_resume_<time>")
    from ckpt_step_397325 by `docs/gain_of_function_v4/resume_after_failed_start_guard_2026-09-24.py`,
    with a new supervisor on the fixed code.
60. **The pass/fail lines are locked; only the operator approves a change** (operator, 2026-09-25;
    section 6, commits d75d3aeaf, 10d58f10b, cc5e57ed5). The tracks that changed a detector also
    rewrote the answer it was judged against: `--write-golden` overwrote golden.json, and the
    changes of entry 54 were "approved against the scripted case" by the track that made them.
    Decisions made in building the lock:
    (a) the first locked set is the code's own lines at 50ce48c7a with the 9:05 PM golden
    detections, written by the coordinating session and marked not approved; verdicts use it
    meanwhile, since no other set was ever reviewed, and the operator's `approve --current` records
    the approval;
    (b) a case the lock lacks is reported and never counted, instead of failing validation, so a new
    case can be recorded and proposed without failing every run until it is approved;
    (c) the scenario is part of the case: a recording whose trajectory fingerprint differs from the
    approved one fails, since a builder could otherwise move the box instead of the line;
    (d) the known-defect register moved into the locked set: a builder can no longer mark its own
    failing case "known";
    (e) the guard also protects the builder's settings files, because `disableAllHooks` in any of them
    switches the hook off; the cost is that agents cannot change their settings.
61. **The recorder's second format, observer_world_v2** (2026-09-25, branch
    fix/observer-recorder-2026-09-25; section 3). Four field groups the analysis could not do without:
    contact normals (R04 and H02's opposition, entry 62), the inner foveal rings (entry 63), a 32-bit
    fast lane (float16 stored the kept case b04_fingertip_to_mouth's contact force, above 65,504 N, as
    infinity in 3 records, the only non-finite values in 188,144 kept records), and the order each world
    was handed (the yoked control, entry 64). The v1 columns keep their values (the inner rings are cast
    in a second call). A group a header does not list is not recorded, never zero: detectors that need
    it measure nothing, and a babble summary names the capabilities it cannot measure. Before any v2
    recording existed, the new definitions were measured on the kept v1 recordings with the normals and
    rings REBUILT from each record's stored state in the saved model (`tools/observer/reconstruct.py`,
    measuring only; the digester never mixes rebuilt and recorded fields): the rebuilt v1 bundle equals
    the stored one in 93-100% of rays in the validation cases and 89-93% in production, where the
    caregiver's toy switches between shapes (the recorder keeps one visibility flag per object, so the
    rebuild cannot know which shape was showing) and the adult's parts change at run time; the ring
    comparison of entry 63 is therefore also made on the records whose rebuilt rays equal the recorded
    ones (72% of production's). A rebuilt normal exists only for a pair still touching at a record's
    last frame: 73-75% of the contact rows of babble (random orders and random actuator commands),
    97-98% of the limp bodies', 79-80% of production's, 96-100% of the 66 validation cases'. The
    coverage that matters for R04 is the co-contacts' (the independent review of 2026-09-25, finding 8):
    babble made 123 thumb co-contact records (random actuator commands 52, reset-world orders 33,
    vision-off orders 34, saved-world orders 4) and the rebuild could judge 35 of them (28%); the limp
    body made 24 and the rebuild judged none. Placing each stored contact point on the object's faces,
    the review found none of babble's 91 co-contacts on a box or the ball with the thumb and a finger on
    opposite faces (same face or adjacent faces only; 32 on other shapes unchecked).
62. **R04 by contact normals, line 120 degrees; H02's opposed grip is R04's test; the rest of the body
    is support** (entry 55 closed; section 4; the line became one exclusion inside the grip, and the
    support a force, in entry 67). R04 compares the thumb's push with the fingers' and, apart,
    the palm's (summing them lets a palm pressing at right angles to the thumb pull the angle toward 90).
    The line moved from 90 to 120 degrees because by normals the kept scripted grasps are not opposing
    grasps: the thumb presses one face of the box, the ring finger lies on the top face, and the middle
    fingertip touches a side face next to the thumb's in 16-18 records of the three lifted cases (0.1 mm
    off it in grasp_opposition_no_lift), the index fingertip 5.4-5.6 mm off (the independent review's
    finding 4; 90 degrees, edge contacts up to 107.6; the forearm sits 4-5.5 mm inside the box), so a
    90-degree line runs through exactly that configuration, the defect being fixed. v8's position angles
    on those cases read 106.6-131.6 degrees. Of babble's 123 thumb co-contact records in 120,000
    body-steps the rebuild judged 35 (28%, entry 61): 0-100 degrees (15 at 0-30, 16 at 30-60, 3 at
    60-80, 1 at 80-100), none at 100 or more; opposite faces read about 180; production's one
    co-contact in 66,200 body-steps read under 30. H02's support now includes the
    infant's own body except the hands and forearms, the holding limbs: with the forearms as support no
    kept grasp case could be carried (each scripted box touches the forearm) and one babble pinning
    between forearm and palm (0.15 s) went; the other hand is no support either, since a box between two
    palms is held by each (the first v9 build counted it as support: found by the 2026-09-25 review; test
    test_observer_grasp_v9.py). Measured v8 to v9 on the kept recordings, acts counted as the summaries
    count them, with the final support rule (measured again on the evening of 2026-09-25: every count
    and angle equals the first rule's, under which the other hand and forearm were support): limp H02 1 to 0 (a limp palm and thumb base on a box on
    the floor, opposed in v8 only by co-contact; the rebuild placed no normal on it, so v9 could not
    judge it opposed: recorded normals will); random-order babble from saved worlds 0 to 0 for all six
    detectors; reset-world babble H02 3 to 3, R06 2 to 2; vision-off babble 1 to 1; random actuator
    commands H02 2 to 2, R06 2 to 2; R04 0 to 0 in every babble and in 66,200 production body-steps
    (segment 40, steps 402,150-411,100), where no detector of the six fired under either version. In
    the validation cases R04 goes from the five scripted grasps and H02 from three cases
    (grasp_opposition_no_lift, which lifts nothing and relied on that non-opposition, and the two h01
    palm-box cases, which never expected it): the locked lines "R04:R present" fail in four grasp cases
    and "H02:R present" in one, because the SCENARIOS are not opposing grasps (open item). O05 is not
    built on `held` (section 4) and did not change.
63. **The inner foveal rings do not replace the centre-ray fixation rule; each fixation reports their
    share.** Declared before measuring (agents/observer_recorder_2026-09-25.md, committed 2:50 PM PDT in
    4c306838a): the majority of the 17 inner rays (centre, 2 and 6 degrees) replaces v7's rule only if it
    meets every locked gaze expectation, makes holds at least 5 points stiller (p < 0.01), and keeps at
    least half the centre ray's looks at objects 20-40 cm away. It met the gaze cases. On 66,200
    production body-steps its holds jumped 5 degrees in one step MORE often: 54.4% of 2,306 against 49.0%
    of 2,462; on the 72% of records whose rebuilt nine rays equal the recorded ones (elsewhere a rebuilt
    ray can meet a shape that was not showing: the recorder keeps one visibility flag per object, and
    the caregiver's toy switches between shapes), 36.3% of 419 against 33.3% of 420 (2.9 points, z =
    0.9). It kept 65 of 197 looks at objects 20-40 cm away (33%; 44 of 125, 35%, where the rays agree;
    11% at 40-60 cm; 97.5% within 15 cm): "most of the fovea on a toy" holds only for toys near the eyes,
    as the 12-degree majority of entry 44 did. v7's rule stays; `inner_share` (the share of the 17 inner
    rays on the target) is reported with every fixation and by G02, G03 (of the return), G05 and R03 on
    v2 tracks (G03 and R03 since the review's fix, 3e396f091).
64. **The yoked control: the learner's own orders, re-anchored, in shuffled time order** (section 9's
    open item; `baseline --kind yoked`). An order's targets are world positions and orientations; replayed
    at another moment as they were, they would send each part back to where it was then. The yoke keeps
    what each order ASKED FOR - each handle's displacement and turn from its pose where the order began,
    its strengths, mask, hand goal and the eye command of each of its steps - and asks it again from the
    pose where the replayed order begins; whole orders, in a random permutation per pass, from
    production's saved worlds, body i replaying production body i's orders recorded before the
    checkpoint's step (only the learner's: planner, self choice, acting snapshot, reflex, babble noise;
    never a rest hold or a probe). It keeps the learner's order statistics and breaks their timing and
    context. It needs recorded orders (entry 61), so its first recording waits for production to run the
    v2 recorder: 500 orders per body, 2,500 production steps (segment 40 latched a learner's order every
    5 steps with no scheduled rest: planner 6,624, reflex noise 4,392, reflex 2,224 of 13,240 latches).
65. **Baselines are recorded again from each newer checkpoint and compared with the one they replace**
    (section 9's open item). A saved-world recording of another checkpoint no longer deletes the previous
    comparison: it moves aside as `<kind>_ckpt_<step>`, is re-digested with the running detectors and
    `baseline --compare --other` tests each capability's rate in the two; `random_ctrl` is recorded from
    saved worlds with the run's vision for the first time; all in the pause-jobs file
    `pause_jobs_observer_recorder_2026-09-25.json`, whose wrapper refuses to run on code other than this
    branch's merged recorder (an old recorder's baseline would replace the comparison and delete its
    recording).
66. **During the switch, production's v1 chunks under v9 undercount the H02 family and measure no R04.**
    The digester re-digests the raw chunks still on disk when the detector version changes; those are
    v1, so their R04 is not measured and H02 can hold only by carrying, while their body-steps count in
    the exposure. The bias is toward fewer acts than babble (a v2 babble recording measures both
    branches), never a false pass. It does NOT end when the v1 chunks leave the 72-hour window, as this
    entry first said: the exposure is one count per body shared by every row (daemon.py, tracker.py),
    and digested exposure stays in the lineage store after the raw chunks are deleted (the independent
    review of 2026-09-25, finding 7). No lineage surgery (the coordinator's decision of that evening):
    the dashboard marks R04 and the H02 family "Observable exposure begins at recorder v2"
    (view.row_view; test_observer_grasp_v9.py).
67. **The grasp read as physics** (the operator's review, adopted by the coordinator on the evening of
    2026-09-25, after the second independent review of the recorder branch,
    agents/observer_recorder_review_2026-09-25.md, 37aeebfa4). R04 moved from the single 120-degree line
    to the physics of a grip (`detectors._grip`): in one record, two surfaces of one hand touch a free
    object in overlapping physics frames (the recorder gathers a record's contacts over five frames, and
    30 of 147 kept babble and limp co-contacts had touched in frames that never overlapped: the review's
    finding 3), the pairs being thumb and fingers, thumb and palm, and fingers and palm (a palmar grasp,
    thumb-free, is an opposing grasp: the review's finding 5); their pushes are at least 120 degrees
    apart (the one exclusion the line became); they BALANCE: what the stronger push exceeds the weaker by
    along their axis, |F_a - F_b| sin(angle / 2), is no more than the object's weight and acceleration,
    m (g + |a|), and the pair's own friction, mu cos(angle / 2) (F_a + F_b), can take (straight against
    each other friction takes none, so a thumb pressing a box against a resting finger is a push); the
    object is RETAINED AGAINST GRAVITY by the hand: everything else - the floor, other objects, the chest
    and the rest of the body, both forearms, the other hand, the caregiver - pushes it up with less than
    half its weight, so its support does not come mainly from the chest, a forearm or the environment; and
    it moves with the palm (the object-to-palm distance changing at most 0.03 m/s over the record's own
    burst, H09's and R07's slip line; until 2026-09-26 the next record's change counted too, so a lift the
    hand could not keep never gripped: grasp_lift_no_retention's one lifted record moved 0.0001 m/s from the
    palm over its own burst and 0.49 m/s over the next one's). H02 holds by carrying the object off that support, or by a gripping record with the hand
    closing 10 degrees on it; its support is contact (the floor, another object, the body but both hands
    and forearms) and, wherever normals are recorded, that same force as well, so a forearm is support
    exactly as far as it carries the object (the contact rule for the forearms had been chosen on scripted
    boxes sitting inside the forearm: the review's finding 5, re-decided by physics). Force alone was not
    enough: a limp hand under a box the floor still touched carried most of its weight, and read by force
    alone the sliding arm "carried" it in all 8 reset-world limp bodies (measured the same night: H02 and
    R06 0 -> 8); by contact the floor still holds it. R04 leaves the world's mocap toys out (a pinch of the
    toy's pad is to be a grasp subtype of its own later, not a replacement). A toy clutched against the
    chest is therefore no grasp: its support comes mainly from the chest. On the same evening the
    detectors branch rebuilt the grasp cases on a 2.5 cm cube held between the thumb and the index on
    opposite faces (normals 163.5-164.3 degrees apart; the coordinator's decision of 8:20 PM PDT: fix the
    test object, do not weaken the definition): every grasp case changes scenario. One expectation and
    the grip disagreed: grasp_opposition_no_lift grips the cube without lifting it, so the floor carries
    it and R04 and H02 are absent under the grip, while the case expected both; the coordinator decided
    the same night that both are absent there (the definition stays) and that the lifted twin
    grasp_lift_and_hold keeps both present. Measured on the kept recordings in the night pause of
    2026-09-25, with normals rebuilt (agents/observer_recorder_2026-09-25.md), v8 to the grip: babble and
    limp H02 7 to 5, R06 5 to 5, R04 0 to 0; production's 66,200 body-steps 0 to 0 for all six detectors;
    of 321 babble and limp co-contacts 263 touched in overlapping frames, 55 could be fully judged, and
    none opposed at 120 degrees or more. With it, the same evening: whichever detector
    branch merges second says observer_detectors_v10 (tests/test_observer_detector_version.py); the three
    observer branches merge in one pause, a qualification gate (the pause runner's gate jobs: the
    crib-world proof, one validate on the merged tree read against declared failures, a fresh babble
    recording under v2, pause_jobs_observer_recorder_2026-09-25.json; merged and qualified on 2026-09-26:
    entry 71); and the yoked control gains a sequence-keeping variant,
    yoked_sequence (production's consecutive orders move each hand back the way it came, a mean cosine of
    -0.18 to -0.19 between consecutive displacements, which a shuffle breaks: the review's finding 6);
    both stay reported, never what stages are judged against.
68. **Statistics defects fixed before the lock** (observer statistics track,
    agents/observer_statistics_2026-09-25.md; analysis v5, rules v5, store v4). The Benjamini-Hochberg
    line is q k / m; retention's windows are tested against the first-repeat window's count as a sample
    and judged per window across rows (false "not kept" marks 0.12 of 20 rows per simulated lineage
    before, none after); novelty adds a within-unit order control (a run must also beat the same units'
    tokens in a random order); fast lanes cut at lineage cuts in 100-step bins and the learner's lane is
    read one record later; ten chunks of look-back (in live-sized windows 265 of a run's 7,733 acts, 3.4%,
    had differed from one pass over it, the live digest counting 7,560: turn-based finger, palm and wrist
    acts); a hand act is credited to the order a record before its peak; the wellness change under each
    row gets its references; each stage cell's tooltip gives its p-value against its line; a part's Kept
    cell reads its own regression.
69. **The independent review of that branch (agents/observer_statistics_review_2026-09-25.md) and its
    fixes.** The matched moments of both controls are placed by distance from the act, not by the
    digestion window (whole-hand close after touch: 35 in 100 in batch windows against 65 live before;
    67 in 100 in both after); touch rows' own Predicted tests are stricter, and 18 of their cells go from
    "waits for Repeats" to "not yet". Seen and Repeats are dated where their running test's last
    continuous stretch under the line begins (20 cells dated later, up to 24,672 steps; no verdict moved),
    so "Seen" reads "held since", as the page's note says; Kept still counts from the first pass. The
    page's chances are counts; Seen says "doubled" only with both tests; the wellness line carries no
    number. The credit lag is kept for the hands (43.2 against a placebo's 56.5 in 100) and dropped for
    the arms (47.6 against 53.3). A bin a cut falls inside counts its records before the cut.
70. **The detector branch's definitions and cases, merged with the recorder branch's as
    observer_detectors_v10** (2026-09-25 and 26; agents/observer_detectors_2026-09-25.md and its
    independent review, agents/observer_detectors_review_2026-09-25.md; section 4's detector-branch
    table, section 6). Both branches had named their definitions v9, so the merged set says v10 (the
    recorder review's finding 1; tests/test_observer_detector_version.py), and the digester re-digested
    every raw chunk still on disk under it. Decisions in it: a hand act and a roll count only when the
    body's own motors drove them (a command and positive work), because no kinematic line separates a
    limp hand settling from a spread; the cost, named: a finger relaxing against its spring is not counted, and production's
    palm shaping fell from 1,140 to 636 acts on segment 39 (the cupping hinge moved against its motor in
    36-43 of 100 of production's palm-shaping acts: section 9). P05's rise must hold over 3 records (the
    coordinator's fix of 8:40 PM PDT on 2026-09-25), since 90-98 of 100 of its acts were one record of a
    force that flickers from frame to frame. The grasp cases use a 2.5 cm cube, since no centred placement
    of the world's 6 cm box between the thumb and a fingertip kept the rest of the hand out of it (33,144
    tried; this hand opens at most 6.89 cm between the thumb and the index: the coordinator's decision of
    8:20 PM PDT, fix the test object, do not weaken the definition). Posed cases that drove limbs into the
    body or the floor harder than the whole body weighs (the kept b04 case read 74,639 N between the upper
    arm and the chest) were rebuilt on physical paths; every posed case now presses at most 87.7 N in the
    review's dry run, against the 43-46 N of the head resting on the floor. The lock's scenario
    fingerprint gained the motor commands, the eye movers' flags and the world model's structure, and the
    scripted truth values are judged (section 6). Every row the page showed beyond babble on 2026-09-25
    got a scripted positive and a twin.
71. **The merged observer qualified; P02 left out of the lock** (agents/observer_candidate_2026-09-26.md;
    the coordinator's decisions of the morning of 2026-09-26; the operator's decision of 2026-09-27;
    section 6). The candidate merge 75a7a3da1 (production d2dd4bab4, recorder ffd0c42af, detectors
    61336e593, statistics bb90aec09; daemon.py's two conflicting hunks resolved as in the recorder review's
    tree 97b7b6d80: analysis v5, detectors v10, and one loop over the detector branch's final cut and the
    statistics branch's hold-back until the cut stops moving) needed three fixes before its gates passed.
    (a) The no-lift grasp case crashed the dry run (d9e8309b3). (b) The lifted grasp cases' posed grips
    failed R04's balance line: the thumb pushed the 3.6 g cube 0.357 N against the index side's 0.71-0.89
    N, 1.81-2.56 times what the cube's weight and the pair's friction can take. A detector that cannot fire
    on its own positive case must not be locked, and the gate never excuses an R04 or H02 "present" line,
    so grasp_lift_and_hold and grasp_lift_no_retention are driven by the hand's own motors (4229ce4f1; in
    the saved world model both met every line in 10 of 10 noisy starts and in 8 neighbouring settings
    each). (c) validate set the
    world's clock back at every case and left the virtual skin's clock ahead, which stopped the first
    qualification's validate ("virtual skin time reversed"): `BridgeEnv.restart_clock` now starts the
    skin, the touch poolers and the per-patch exposure with the clock (f271bc050; reset() calls the same
    helpers in the same order, and training runs `--self-choice off`, so the env.py change refuses no
    resume). P02 (roll onto the side) is left out of the lock by the rule of (b): the roll synergies the
    build searches turn the upper body at most 29.4 degrees with the body's own motors in the dry run and
    29.7 in the qualification's validate, where p02_roll_to_side needs 70-120 (the twin rolled by an
    outside torque reaches 74.7 without P02). Its three cases run and are reported, never judged
    (`validate.NOT_LOCKED`), and return after the servo-strength deploy. Deployed on 2026-09-26: the v2
    babble placed and summarized at 10:48-10:52 AM PDT, and from 10:48 AM the digester on the new code
    moved all 40 run folders' analysis aside and digested their raw chunks again (the last, the active
    run's, at 10:56 AM). The operator decided on 2026-09-27 to approve the lock with P02 left out;
    the approval is the lock tool's, run from the operator's own terminal (section 6).
72. **An act of the hands alone is credited to no single command; four stage readings kept apart; evidence pins**
    (observer_attribution_v1, store v5, tracker rules v6; branch `fix/observer-attribution-2026-10-08`; the
    operator's instruction of 7:37 AM PDT, Oct 8, 2026 and its addition of the same morning, relayed by the
    coordinator; the observer source review, `D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/REVIEW.md`, findings F2,
    F3 and F8; the independent review of 4342f9194, `D:/UEMA_Research/external/D/UEMA_scratch/plan_review/OBSERVER_ATTRIBUTION_REVIEW.md`, 9:35
    AM PDT, Oct 8, 2026, two blockers and its fixes, repaired here). Measurements, all read only:
    `D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/observer/defect_recheck.json`, `compare_base_branch.json`,
    `predicted_null_simulation.json` and, in `review_fix/` beside them, `compare_production.json`,
    `compare_control_run.json` and `pin_edge_cases_fix.json`.
    (a) Attribution. From analysis v5 an act of the hands alone was credited to the command in force one record
    before the record of its largest world-frame motion, and its writer, intention verdict and efference verdict were
    read on that one command. Now `evidence.Joiner.command_intervals` lists every command in force during the act:
    the change from record r - 1 to record r belongs to the command in force at record r, and per command it gives the
    writer, the model step it was latched at, its records inside the act, the act measure's change during them (H05:
    the digit's mean flexion, degrees), that change's share of the act, the share of the joint path during which the
    command's own motors pushed the same way (`motors_same_way_share`), and whether its hand joint goal asked for the
    direction moved (`asked`); and over the command's WHOLE HOLD (all its consecutive records, inside the act or not)
    `hold_records`, `hold_change` and `hold_asked`. The commands of the 10 records before the act are listed as
    `history`. An act of one record (a touch at its onset record: HX3, B02) is listed under the command in force at
    that record, whose burst made it. A movement made during a command's records is not said to be caused by that
    command: a joint keeps moving after its command changed. The act's writer class is the class all its commands
    share, else `mixed`; its intention verdict needs every command to be a decisive planner choice that reached the
    body and did not ask for the other direction. Acts of other parts keep the command at their peak.
    (b) Measured on control run 1 (`H84_fresh_drivers_20260918_seg41c_0841`, resumed from checkpoint 481,800), a
    scratch copy of its preserved raw records digested in one pass by the base code (3a5bf510e) and by the branch:
    the same 9,547 detections; every act of other parts identical in every field (4,247), so every verdict of theirs
    is identical; the 5,300 acts of the hands alone differ in their evidence, their writer (`author` and
    `provenance`, 4,956) and, for the 1,316 H05 acts, the new measures. What is not identical outside the hands is the
    Predicted stage's pass line, shared across rows (the Benjamini-Hochberg line): on production's records 9 rows with
    no act of the hands alone (G01-G05, O04, O06, P05, P06) go from a line of 0.0243 (144 tests together) to 0.0094 (90),
    none of their verdicts changing. Production's own digest of control run 1 holds 9,548: two HX3 acts of the left
    thumb on body 2 where one pass finds one, and an R02 act on body 7 with another end, the same in the base code's
    single pass as in the branch's (the digestion window: section 3). On its 47 left-index H05 acts, each under 2 to 10
    commands (median 4): the credited writer was the planner for 34 and the motor controller for 13 (10 with its
    exploration noise, `reflex_noise`; 3 without, `reflex`), and now all 47 read `mixed`; the intention verdict passed
    for none, before and now; in 37 a command's hand joint goal asked for the direction opposite to the flexion made
    during its records; commands written by the planner were in force for 25 to 82 in 100 of an act's flexion path
    (median 51). (The commit message of 4342f9194 and this entry's first text said 13 with exploration noise.)
    (c) The shared episode of the first-finger investigation (body 5, model steps 483,793 to 483,802, index flexion
    +20.0 degrees), inside the act: command 96759 (the planner, latched at 483,790) during record 483,794, +0.96 degrees
    (4.8 in 100 of the act), its goal asking for flexion, its motors pushing the same way over the whole path; 96760
    (the motor controller with exploration noise, latched at 483,795) during 483,795 to 483,799, +14.71 degrees (73.5
    in 100), asking for flexion, motors the same way over 70 in 100 of the path; 96761 (the planner, latched at
    483,800) during 483,800 to 483,802, +4.34 degrees (21.7 in 100), its goal asking for extension, motors the same
    way over 26 in 100. The episode record's split from the joint angles agrees within 0.0005 degrees. Analysis v5 had
    credited the whole act to 96761. No output presents this act as demonstrated intentional bending.
    (d) Four stage readings of an act of the hands alone, each on its own evidence, none a stage pass and none read as
    intention (`evidence.Joiner.four_stages`):
    - planner selected: NO when a command in force during the act was not the planner's; when every one was, NOT
      JUDGED: the planner's own latch record, written before execution, gives each candidate's predicted goal distance
      (0 = the predicted next tokens equal the goal entry's) for the chosen command, holding still, the motor
      controller's command and the perturbed candidates, with the chosen command's rank and margin, and the hand
      stage's cost before and after its search; no noise band of that distance has been measured, so the rank is
      reported and never judged (review fix 3; 4342f9194 judged it with a band of zero). No band is invented here.
    - predicted: each command's own efference expectation against both controls, over the command's whole hold.
    - achieved: the act measure moved the way each command's hand joint goal asked, over the same whole hold: from the
      record before the command's first record to its last, the window its efference record grades (the lane's
      achieved movement matched the world's over exactly those records on all 47 left-index acts, defect_checks M1;
      on the whole run 17,857 of 17,857 efference records begin at their window's first record). 4342f9194 read
      achieved only on the command's records inside the act (review fix 2).
    - improved: not readable from one act.
    On the shared episode: planner selected NO (96760 was not the planner's; 96759's chosen command had the second
    lowest goal distance of those recorded, 96761's the lowest, both reported, not judged; the hand stage's cost fell from
    2.67 to 2.31 and from 2.58 to 2.29; the wellness evaluator tied both commands' candidates within its band of
    0.0734); predicted: 2 of its 3 commands (96760's own expectation missed by 3.70 cm against persistence's 3.12 cm, on
    world-frame records, source review finding F1, not repaired here); achieved NO over the whole holds (96759,
    483,790-483,794: -1.35 degrees where its goal asked +8.39; 96760, 483,795-483,799: +14.71 where it asked +18.10;
    96761, 483,800-483,804: +3.64 where it asked -1.81); improved not measured. On the 47 acts: planner selected NO for
    47 (each had a command that was not the planner's), every command predicted for 13, achieved for 15. The tie band
    is unchanged and a planner-written command is never labelled intentional. A Chosen stage at "not yet" shows that no
    intentional skill was demonstrated; the page says it does not show that the model learned no intentional behavior.
    (e) Moved alone. On the 47 acts, 7 kept the other four digits within the 5-degree line inside the act and 40 did
    not, the same verdict as the source review's check M3 on every act (largest difference 0.003 degrees).
    (f) THE PREDICTED RULE FOR AN ACT OF THE HANDS ALONE (review blocker 2). 4342f9194 read an act as predicted only
    when every command in force during it passed, and tested the acts against one half: on control run 1 that passed
    75 of 88 acts of one command and 0 of 31 of 13 or more, so a "not yet" read how many commands an act spanned. Now
    each command is one unit (`tracker.cluster_binom_above`): an occasion's first judged act gives its commands, the
    passed and judged counts are divided by the design effect at its upper bound, D = sum(m^2) / sum(m) over the
    occasions' command counts m (the commands of an act as alike as they can be), and the binomial tail against one
    half is read at those effective counts; with one unit per occasion it is the binomial test unchanged. Under no
    prediction skill (`predicted_null_simulation.json`: command counts drawn from control run 1's 4,642 acts of the
    hands alone, rows of 3 to 1,000 occasions, 20,000 simulated rows each up to 100 occasions and 5,000 beyond), the
    rule passed at most 4.72 in 100 rows (every command of an act sharing one draw, 1,000 occasions; two standard
    errors 0.62 in 100) against its line of 5 in 100, and 0.02 in 100 or fewer with independent commands; counting the
    commands as independent units passed up to 23.54 in 100. Its power: with 74 in 100 commands passing (the left
    index's rate on control run 1, 150 of 204), rows of 30 occasions pass in 87.48 in 100 simulations, against 0.09 in
    100 for the rule of 4342f9194. On control run 1's report, H05 reads passed again (2,453 of 3,257 commands over 986
    occasions; design effect 4.32), as it did under the base; H08 passes, its left-hand part does not (a p-value of
    0.0235 against a line of 0.0188). The act's "every command passing" stays as a reading beside the stage.
    (g) ACTS RECORDED BEFORE THE CHANGE (review blocker 1). Their records name one command, so the Predicted stage
    does not judge their movement verdicts: it reads NOT JUDGED, a new marker of the page, with that reason and their
    count, and keeps their EARLIER READING beside it (`earlier_rule`: the test of the rules before v6 on those acts,
    judged at the line of the family the earlier rule makes, and with the ladder: `stage_reached`). The page never says
    "not measurable" with another cause for them. On production's records (365,065 detections of 83 runs, read only),
    44 Predicted cells of 10 hand rows change from the base and no other stage does: 5 passed cells read not judged
    (H05 and its left little, right little, right middle and right thumb parts) and 39 cells that had not passed. The
    H05 cell reads: "Not judged: 43,912 acts of the hands alone were recorded before Oct 8, 2026, when an act's movement
    verdict was read on the one command it was credited to; the current rule judges every command in force during an
    act, and those records do not list them; under that withdrawn rule 26,723 of 33,181 occasions were predicted,
    enough to pass Predicted." The headline says "the learner predicted 0 before they happened under the current rule;
    1 more (Individual finger isolation) passed Predicted only under the single-command credit withdrawn on Oct 8,
    2026, and is not judged now", and the closest full behavior reads 2 of 4 stages with the same naming of the five
    withdrawn passes. The stage history's reset line at the rules change lists every stage that stood before and not
    after (`withdrawn`, with what it reads now), and the page's rule-change label names them. The plan's sentence that
    H05 "has reached 5 of its 6 stages" is stale from the digester's restart on this code. Chosen does not count those
    acts for either writer class, as before.
    (h) Versions. The detector and analysis versions stay: a change of either moves every run's analysis aside and
    digests again only the raw chunks still on disk, which would drop the history older than 72 hours. The store
    version changes (v5: `attr`, `alone`, `sel`, `ach`, `pred_n`, `pred_k`), so the restarted digester rebuilds the
    store from the record files. Until the digester runs the merged code and has saved its rebuilt store,
    `regression_pack.load_store` and `rebuild.py` refuse the store of the older version.
    (i) RESTART SEQUENCE after the merge (review section f): first restart the digester (the training supervisor's
    `ensure_observer` restarts only a missing or stalled one); confirm `rules_version` reads `tracker_rules_v6` in the
    lineage's `development.json` and `version` reads `observer_store_v5` in its `store/store_meta.json`; only then
    restart the dashboard server. A dashboard restarted first renders the new stage texts over counts made under the
    old rules (the review's page: H05 Predicted "Passed at step 383,684" under a title saying an act of the hands
    alone is credited to no single command). The running digester reads no pins until it is restarted on this code.
    (j) Evidence pins (section 3; source review finding F8: production's raw records of the index_L movements from
    checkpoint 481,800 were removed by the 72-hour housekeeping at 5:00 PM PDT, Oct 7). The coordinator writes the pin
    list; housekeeping keeps the pinned world and learner-lane chunks; an expired pin is reported and keeps
    protecting; an unreadable pin file stops every deletion of its pass. Review fix 1 closed three openings, each kept
    now in the reviewer's own cases (`pin_edge_cases_fix.json`): an invalid entry (an `until` without its UTC offset, a
    second pin with the same id, steps as text) stops its run's deletion, and one naming no usable run stops all; a
    chunk's range is read from its own step column (a process id given to two processes shrank a range to one step);
    the learner lane is kept by its own numbering, one step after the world's and up to the efference record of the
    command in force at the pin's last step. A pin file that starts with a UTF-8 BOM (the mark PowerShell writes) is
    read, and one read before and missing now keeps the pins last read. The digester's housekeeping is the observer's only deletion of production
    raw chunks; `offload_storage.py` moves a run's subfolders only on a person's command, with hashed copies.

## 9. Open items, each with its next executable test

From the reviews of 2026-09-24 and 2026-09-25 (statistics, code, recorder and detector reviews) and the
observer branches' records, not yet resolved:

- **What observer_attribution_v1 leaves open** (entry 72). (1) The efference verdict of an act of the hands alone
  still reads the efference head's world-frame displacement of the finger's handle, most of which is the hand being
  carried (the source review's finding F1); test: the review's check M1 on the pinned shared episode, the same verdict
  read relative to the wrist handle. (2) Acts of the hands alone recorded before the change keep their single-command
  credit in their record files; Chosen does not count them and Predicted reads them NOT JUDGED with their earlier
  reading beside it (touch verdicts read no writer and stay), so the hand rows' judged counts start again from the
  digester's restart on the merged code; the raw chunks still on disk (72 hours, and pinned ones) would be digested
  again only by a version change, which also moves every run's older analysis aside (entry 72 (h)). (3) Improvement
  with experience has no code path yet: one act cannot show it (entry 72 (d)). Its executable test, to be written with
  its row on the Tests page before it runs: for one finger's acts of the hands alone, the share of commands predicted
  and the share achieved in the first and in the last 3,000 model steps of a run whose evidence both windows hold,
  compared with the two-sample test the tracker uses for touches (`tracker.two_share_above`), the pass line the stage
  family's; no row exists in the test registry for it yet.

- **Babble is recorded from one checkpoint at a time** (entry 49, section 6). Recorded again under
  recorder v2 from checkpoint 415,075 on 2026-09-26 (random orders, limp, and, for the first time from
  saved worlds with the run's vision, random actuator commands), with the checkpoint-397,125 recordings
  kept beside it (entries 64-65). Open: both yoked controls (shuffled, and in recorded order: entry 67)
  refuse until 500 learner orders per body are recorded by v2, 2,500 production steps after the first
  trainer start on recorder v2 (section 2). Test: the yoked jobs in the first pause after those steps,
  and `baseline --compare 74302686aba93fca --kind random_orders --other random_orders_ckpt_397125` for
  what the newer checkpoint changed; then weekly from the newest checkpoint. Which control stages are
  judged against (random orders or a yoke) stays random orders until the operator chooses.
- **The grasp cases' remaining gaps** (entries 62, 67, 71). The grasp cases are rebuilt on a 2.5 cm
  cube held on opposite faces, grasp_opposition_no_lift expects no R04 or H02 (the floor carries its
  cube), and the two lifted cases are driven by the hand's own motors; in the qualification every R04
  and H02 line was met, none excused. Open: grasp_carry_and_place and the palm-box hold, release and
  escape cases still pose their lifted records on the floor grip, because the grip's settle in the air
  loses the cube (it moved 24.1 mm), and grasp_carry_and_place grips in one record only, at a balance use
  of 0.90 against the line of 1.0. Test: drive those cases by the hand's motors as the lifted cases are,
  then `predict` and a validate in a pause. Still to add (the validation track): the thumb and a finger
  on adjacent faces (R04 absent: the exclusion's edge), a box lying on the infant's belly moved by the
  body under a touching hand (H02 absent), a box lying on a moving forearm with the fingers curled on it
  (H02 absent: the forearm carries it), a palmar grasp of a lifted object (R04 present by fingers and
  palm), a thumb pressing an object against a resting finger (R04 absent: unbalanced), and later the pad
  pinch as a grasp subtype of its own.
- **Selected intentionally cannot be reached while the planner ties** (entry 51). Test: the saved-state
  probe at the ability scheduler's window from step 399,050 (`first_flat_stage_along_the_route` in
  its result); repair the first flat stage and run it again.
- **Validation covers 28 of 55 detectors** (the proposal of section 6; 26 of them with both a positive
  and a negative case). 17 observable detectors never fired under any chance process (babble_rates.json),
  so against babble's zero a few occasions pass "more often than babble": such a row needs a scripted
  positive before its pass is believed. The detector branch added a positive and a twin for every row
  the page showed beyond babble on 2026-09-25 (entry 70). Test: a case for any row that passes a stage
  without one. Closed by the detector branch (entry 70): the palm-box cases rebuilt on the settled grasp,
  a limp body's passive finger fan (the motor rule), H03's release into the air, the support push's
  driven case (p05_leg_pushes_the_floor and its twin in the air), and b04_fingertip_to_mouth on a
  physical path (its largest body contact 64.8 N instead of 74,639 N).
- **Approved validation completed; rolling coverage remains open** (entries 60, 71; section 6).
  Brian approved proposal 20260926_104649_4173cc with P02 left out on Sep 27, 2026, 12:04 PM PDT.
  Adoption and replay passed all 89 approved cases at 02:14:43 PM PDT. P02's three cases return to
  the lock only after the body's own motors can produce the declared positive rolling example.
  The missing rolling example does not invalidate the 89 completed checks.
- **The recorded foveal bundle was too wide to test "most of the fovea on a toy"** (entry 44).
  Closed 2026-09-25 (entries 61, 63): the recorder casts rings at 2 and 6 degrees; measured, the
  majority of the inner rays does not replace the centre-ray rule; fixations report `inner_share`.
  The measurement used rings rebuilt from stored states; test on the first v2 production recordings
  (none yet: section 2): the same comparison on recorded rings (the observer_recorder_measurements.py
  `rings` command reads them when present), reported beside entry 63.
- **Per-effector segmentation's cost** (observer_segments.md): units are 28 times as many as before
  per-effector episodes (145 per 1,000 body-steps against 5.1; 626 bytes each in memory, about 33 MB a
  day more; 95 of the 145 carry a novelty run, so run-less units could stay out of the lineage store),
  and hand onsets from rest added about 45% more units in production (about 13 MB a day more at 4.2
  body-steps a second: agents/observer_detectors_2026-09-25.md, defect 4). Test: the store's size per
  day once production records again, and the store item below. Closed by the detector branch (entry
  70): the digester's cut is one loop until it stops moving; a hand moving only its fingers opens an
  episode; a turn starts where the movement to its peak starts. Still open there: the segmenter's hand
  onsets use the whole turn's 10-90% rise, so a turn from rest with a slow lead-in opens no episode
  (test: the hand episodes the kept recordings gain when an onset reads the turn's fastest stretch, as
  the acts do).
- **A world resume drops eight per-frame trackers** (gaze_probe_replay.md): `BridgeEnv.save_full_state`
  leaves out eight attributes read in `_collect`, among them the mobile's chime gate
  (`_mobile_touch_prev`, `_chime_cool`), so training's episode continuation and every branch user
  restore them fresh. The probe harness carries them now; the root fix is adding them to
  `BridgeEnv._FULL_STATE_ATTRS` in env.py (runtime code: a new runtime fingerprint, and restore must
  accept checkpoints saved without them).
- **Disk C: lost about 22 GB within minutes of one training start** and got it back when training
  stopped: 39.0 GB free at 07:35:30 PDT (archiver, training stopped), 16.7 GB at 07:39 and 07:44
  (training running since 07:36); the page file grew 3.1 GB (40,000 to 43,180 MB); no file over
  200 MB was written under the user folder, ProgramData or Windows\Temp. NOT reproduced on the
  08:46 start: 42.0 GB free at 09:08 with training running since 08:46, the page file at its
  40,000 MB base. The drop coincided with the memory squeeze of entry 30: the ability scheduler's
  gaze test ran beside the relaunched trainer from 07:35:57 to 07:44:58 (its receipt), and a
  squeeze on 2026-09-08 did the same to C: (5 GB free, 24.3 GB after the trainer exited). So the
  likely cause is Windows' paging under a commit squeeze, which entry 30's fix prevents. Test if
  it recurs without a squeeze (administrator): `vssadmin list shadowstorage`, and Sysinternals
  RAMMap / handle on the trainer. The recorders stop writing below 10 GiB (10.7 GB) free, so
  checkpoints keep their room.
- **Memory margin:** with training running the machine sits 5.5-6.7 GB under its commit limit
  before the observer's own processes; the supervisor holds 2.5 GB (it imports torch to check
  checkpoints) and the dashboard server 0.8 GB (torch as well). The scripted worlds and new
  baselines (3.1 GB a world) fit only in a training pause.
- **Fixed 2026-09-24 3:31 PM: the checkpoint archiver skipped companion saves past 260
  characters** (`comp_src.exists()` without `long_path`), stopped its pass at the first one and
  moved nothing all day: 23 of 26 checkpoints stayed on C:. It now reads, copies and removes a
  companion through `storage_locations.long_path`
  (`test_a_companion_past_260_characters_moves_and_is_not_called_missing`, which fails with the old
  check). See entry 42.
- **The store grows with the lineage** (100 bytes of columns a detection in store v4, 76 in v3;
  episodes 28 times as many since v6 and about 45% more hand units since v10). Test: compact detections
  older than the raw chunks into per-window counts once the store passes 1 GB, and keep units without a
  novelty run out of the episode store.
- **A trunk lift (P07) depends on where a digestion window starts**: its baseline is the median height
  of the window's lying records, so 1 of 8,158 acts of segment 40's 9:33 AM run differs between one pass
  and live-sized windows (the turn primitive's own dependence, 22 of 7,733 acts under v8 with ten chunks
  of look-back, was closed by confirmed turns: entry 70). Test: a baseline from the lying records just
  before the lift. The merged code's batch-against-live agreement of the controls (18 of 7,717 acts
  differed under the statistics branch's code alone: section 3) is not yet measured. Test: the
  statistics track's `window-agreement` on that run with the merged code.
- **The digester's work per window**: ten chunks of look-back doubled a live-sized digest (84 s against
  39 s for a 2,000-step run of 8 bodies); production's digester took 5.6 s of each 60 s cycle at 7:20 PM
  on 2026-09-25. Test: each pass's time (`seconds` in Saved/mimo_bridge/observer/daemon_status.json) once
  production records again.
- **The credit lag is one record for every hand act** (right 57 to 61 times in 100: entry 69). Test:
  credit by direction, the order whose commanded direction the record's motion matches.
- **The new tooltips and the wellness line have no browser screenshot yet** (AGENTS.md). Test: the page
  at 390 px and at desktop width, from a dashboard server started after the deploy of 2026-09-26.
- **The detector branch's open items** (agents/observer_detectors_2026-09-25.md, NEXT). Production's
  hands moved against their own motors in 36-43 of 100 palm-shaping acts, 11-15 of 100 spreading acts
  and 2-4 of 100 finger-isolation acts (babble 18, 8 and 7), which the positive-work rule no longer
  counts; a rule reading the command's change over the act would separate a chosen relaxation (to be
  counted) from a push by contact or gravity (test: the command's change toward the movement on
  scripted pushed and relaxed cases). The saved-worlds limp body of checkpoint 397,125 still made 16
  brief fixations (G02), 3 reaches (R01), 2 one-fingertip touches (HX3), 2 eye-head coordinations (G05)
  and 1 hold (H02) (test: the same motor rule on the arm for R01 and on the eyes for G02 and G05, on the
  kept limp recordings). P05 reads each record's largest frame, so a force flickering inside every
  record reads as held (the options are in the record's defect 7, for the operator). The scenario
  fingerprint leaves out the positions of what the world moves (test: their determinism across two
  world builds). capability_catalog.json still describes HX1's timescale as a 10-90% rise and H03 by
  its v8 note (the catalog owner's text).
- **At 390 px the page's top navigation wraps onto a second row** (review finding 31). Since
  870c30f48 that second row no longer covers the header links (dashboard_staleness.md); the wrap
  itself remains. Test: the 390 px screenshot of the Now tab.

## 10. The second outside review (2026-09-24, 7 PM PDT) and the work it set

The review's verdict: the observer is good enough to say whether function emerges; stop widening
it and turn to the bottleneck it exposed, UEMA converting predictions into chosen physical action
(every planner choice ties, so no behaviour can pass "chosen"). Its milestone is now a headline line
on the Development tab: the first behaviour that is beyond matched babble, repeats, was predicted
before the act better than its control, and traces to a decisive planner choice (7:13 PM: none;
closest are finger-isolation rows with 3 of the 4, missing "chosen").

Agreed and in progress, each in its own track (findings files in docs/gain_of_function_v4/agents/):

| priority | work | track | findings file | status (9:25 PM PDT) |
|---|---|---|---|---|
| 1 | why planner alternatives tie: are candidate futures different (forward model), is predicted wellness different (evaluator), does selection follow it (planner); each candidate executed from the identical saved state | planner | planner_ties.md | records analysed (entry 51); the probe is registered with the ability scheduler from step 399,050 |
| 2 | credit each act to the order whose window holds its peak motion, not the order at its first step | observer statistics | observer_stats.md | done (entry 45) |
| 3 | depth 1 against depth 3 from identical saved states, both executed: does looking further ahead choose better physically | planner | planner_ties.md | in the same probe (entry 51) |
| 4 | babble from production's saved eight worlds, not a fresh world | babble | babble_saved_worlds.md | done: recorded from checkpoint 397,125, the comparison since 9:16 PM (section 6) |
| 5 | occasions, not correlated acts, as the unit of "predicted before action" | observer statistics | observer_stats.md | done for every stage (entry 47) |
| 6 | the contact verdict's empirical null from matched no-touch moments | observer statistics | observer_stats.md | done (entry 48) |
| 7 | per-effector episodes (left hand, right hand, gaze, trunk, legs), which also make novelty count occasions | observer episodes | observer_segments.md | done (entry 43) |
| 8 | gaze from the full foveal ray bundle (target, share of rays, angular error, dwell, switches) | observer episodes | observer_segments.md | done, with the centre ray choosing the target (entry 44) |
| 9 | scripted validation beyond 13 detectors: grasp broken link by link, and every detector the page reports beyond babble | validation | observer_validation.md | done: 66 cases, 27 detectors, all met on v8 (entries 54-55) |
| - | the fast lane's support fall below the usual level (before any fast-wellness experiment) | observer episodes | observer_segments.md | done (entry 43) |
| - | the eye ability test's failed exact replays (5 of 6 attempts at 5:44-6:02 PM, each a training stop) | ability test | gaze_probe_replay.md | first cause fixed (entry 52); failed again at full size (8:15 and 8:24 PM, the same arm); second cause found, its fix not yet confirmed; test held (entry 57) |
| - | dashboard sections pinned to old runs (the Now tab's activity at segment 36, model health from step 344,400, 24-hour times) | dashboard | dashboard_staleness.md | done (870c30f48); the catalogue's own text and registers in progress |

Deferred, with its trigger: the fast-stability wellness experiment (current wellness against current
plus a weak fast-stability term, judged on posture recovery, dangerous-transient frequency,
exploration and total movement) waits for the corrected support fall and the planner verdict. The
platform and container scenes stay out (entry 2) until the crib shows the primitive body functions.

Heavy runs (worlds or models: validation cases, saved-world babble, the planner probe, an eye-test
replay reproduction, dashboard screenshots, the explainer rebuild) do not run beside training; the
parent runs them together in one training pause once the tracks' code is ready, or through the
ability scheduler's saved pauses for the recurring planner probe.
