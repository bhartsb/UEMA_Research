# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE TRAINING PREFLIGHT (operator, Oct 5, 2026): seven gates that every new or changed training configuration passes
before it launches. A routine continuation of the exact same configuration launches without repeating them.

    PY -m mimo_bridge.tools.training_preflight check       (--current | --cmd-file FILE) [--set FLAG VALUE ...]
    PY -m mimo_bridge.tools.training_preflight new         (--current | --cmd-file FILE) [--set FLAG VALUE ...]
                                                           --what TEXT --by NAME
    PY -m mimo_bridge.tools.training_preflight gate        FINGERPRINT N --status PASS|N/A|FAIL --reason TEXT
                                                           --evidence E [--evidence E ...] --by NAME
    PY -m mimo_bridge.tools.training_preflight state       FINGERPRINT --checkpoint PATH --reason TEXT --by NAME
    PY -m mimo_bridge.tools.training_preflight repair      FINGERPRINT --purpose TEXT --target-gates 4,5,7
                                                           --repairs "A;B" --components "X;Y" --start CKPT --rollback CKPT
                                                           --stop-lines TEXT --stop-lines-row ROW --stop-criteria FILE
                                                           --reachability TEXT
                                                           --max-model-steps N --no-auto-continue TEXT
                                                           --remeasure 4=ROW ... --by NAME
    PY -m mimo_bridge.tools.training_preflight repair      FINGERPRINT --block-file FILE --by NAME   (an experiment's arm)
    PY -m mimo_bridge.tools.training_preflight continuation FINGERPRINT --block-file FILE --by NAME  (THE CONTINUATION)
    PY -m mimo_bridge.tools.training_preflight instance    FINGERPRINT --block-file FILE --by NAME  (THE FRESH INSTANCE)
    PY -m mimo_bridge.tools.training_preflight extension   FINGERPRINT --block-file FILE --by NAME  (THE EXTENSION)
    PY -m mimo_bridge.tools.training_preflight adoption    FINGERPRINT --block-file FILE --by NAME  (THE ADOPTION)
    PY -m mimo_bridge.tools.training_preflight recovery    FINGERPRINT --block-file FILE --src DIR [--registry FILE]
                                                           --by NAME  (THE LINKED RECOVERY OF A USED ADOPTION)
    PY -m mimo_bridge.tools.training_preflight link        FINGERPRINT --block-file FILE --src DIR [--registry FILE]
                                                           --by NAME  (THE LINKED CONTINUATION OF A STOPPED RECOVERY)
    PY -m mimo_bridge.tools.training_preflight launched    FINGERPRINT --run DIR [--continuation | --instance ID |
                                                           --extension | --adoption | --recovery | --link] --by NAME
                                                           (the relaunch script; --adoption, --recovery and --link
                                                           with --src)
    PY -m mimo_bridge.tools.training_preflight reference   --current --reason TEXT --by NAME
    PY -m mimo_bridge.tools.training_preflight fingerprint (--current | --cmd-file FILE) [--set FLAG VALUE ...]
                                                           [--src DIR]
    (check and fingerprint take --src DIR, the Source/Python folder of the checkout whose trainer code the launch runs,
    and check takes --registry FILE and --root DIR, a preflight folder; without them the checkout holding this file,
    production's registry and production's preflight folder are read)

`--current` reads the supervisor's recorded launch command (Saved/mimo_bridge/training_continuation.json); `--set`
applies the changes a relaunch would make (as the relaunch script's --set does).

A CONFIGURATION is what decides how training behaves:
  - the trainer's launch flags, except RUN_FLAGS (where a run writes, what it resumes from, how long it runs, what it
    shows and how often it reports);
  - the content of every file a kept flag names (a calibration or configuration file);
  - the content of the trainer's code, defined once in Source/Python/lem/trainer_code.py, which the admission check of
    a replacement hand calibration shares: every .py file under Source/Python/lem, mimo_bridge and tokenizer and the
    data files the trainer reads there (lem/master_config.yaml, mimo_bridge/assets/), without mimo_bridge's tools,
    explainer and experiments, any tests folder, any folder whose name starts with '.' and Python's caches (line ends
    normalized, so two checkouts of one commit agree, and a pytest run that rewrites the ignored
    mimo_bridge/.pytest_cache changes nothing).
Its FINGERPRINT is the first 16 hexadecimal characters of the SHA-256 of those three parts.

A RECORD, docs/gain_of_function_v4/preflight/<fingerprint>.json, holds the configuration's flags, the per-file hashes of
its code, what changed and who wrote it, and the seven gates, each {n, name, status, reason, evidence [..],
checked_at, checked_by}. status: OPEN (written, not judged), PASS, N/A (a written reason and at least one piece of
evidence: a registry row, a file, a commit), FAIL. Evidence is reused where an existing measurement answers the gate;
a new test runs only where no evidence answers it.

A QUALIFIED configuration is one whose record has every gate PASS, or N/A with a reason and evidence. It launches again
without a new record while its fingerprint is unchanged (a routine continuation: another checkpoint, run folder, run
length or report rate does not change the fingerprint). Every other change needs its own record.

THE MODEL STATE (the operator, Oct 5, 2026: "a preflight from one lineage/state must not automatically authorize an
arbitrary different checkpoint"): a record names the checkpoint(s) its gates were judged on (`states`; the first is the
--resume of the command it was written for). A launch is allowed only when its --resume CONTINUES one of them: the same
checkpoint, a later checkpoint of the same run, or a checkpoint of a run whose resume chain (each run's args.json
"resume") reaches that run at or after that step. A checkpoint of another run or branch, or one that predates the state
(and so may predate a repair), is refused until the gates that depend on the model's state are reconsidered for it and
it is added with `state`.

THE REPAIR RUN (the operator, Oct 5, 2026). Normal training needs every gate PASS or justified N/A. A controlled repair
or qualification run may launch with a gate reading FAIL - REPAIR TARGET only when (`repair`, checked at launch):
the exact failure is the run's stated purpose (its target gates); the repairs and affected components are named; a start
checkpoint and a rollback checkpoint are fixed, and the launch resumes exactly the start; the stop lines are declared in
the test registry before launch (their row exists) and given to the watcher as STOP CRITERIA (stop_criteria: entries of
the hourly watcher's declared criteria, {name, path: a stats field by dot path, op "<" or ">", line: a number,
from_run_hour, mode "alert" or "stop", note, new_field?}, at least one a stop; each path must read a number in the newest
full stats record of the start checkpoint's run, unless new_field says why that run's stats cannot carry it (a field
only this run's configuration writes): a misspelled path is never judged (the relaunch review of Oct 5, 2026, finding
F3). The relaunch script writes them into the new run folder's declared_criteria.json before the trainer starts, and
docs/h84_launch_2026-09-18/hourly_watch.py judges every stop entry at its 5-minute checks on the mean of the last hour's
records (finding F4), stops the run when a stop field reads Infinity or NaN in the newest record (the round-two review
of Oct 5, 2026, finding R3) or is absent from every record of the last hour once the run's records span an hour (the
hour of grace), and writes snapshot_and_stop on a breach; a stop field the start run leaves absent for an hour of its
records from its third on is refused, since the watcher would stop a run whose line holds (finding R2)); REACHABILITY
(the style guide, "No experiment launches on an unmeasured number", rule 4): the arithmetic showing each line can be reached
within the bound, written into the block; the run's length comes after the watcher's first judgement of every stop line
and the trainer's reading of its stop: the model steps to the first full stats record (from the start step and the
launch's --report-every), plus the start run's fastest pace x 330 s (the watcher's 30 s look for that record and its
300 s first check), plus 25 model steps (the trainer reads a stop when its model step is divisible by 25), and an
entry's from_run_hour and, for a new_field entry, the hour of grace move it later (first_readings; the re-check of Oct 5,
2026, finding N2, and finding R1); the duration is bounded (--max-steps within start + max_model_steps); every gate it
does not target reads PASS or justified N/A; it cannot continue into normal training by itself (the record's repair run
is used once: the relaunch script notes each attempt's run folder before its trainer starts, and the record is used as
soon as an attempt's folder holds training past the start, whether or not the script lives to mark it with `launched`
(finding R4); the script leaves the supervisor paused with the run named, so the supervisor relaunches nothing and
keeps the run's watcher running, and a later launch from any checkpoint is normal training); and after the run every
target gate is judged again (checked after the launch time) and normal training continues only a model state added
after the launch that is a checkpoint of the repair run's folder or of a run resumed from it, the checkpoint those gates
were judged on again (findings F9 and N3: the start checkpoint, whose weights the run never changed, no longer
authorizes it, and `state` refuses one that does not continue the repair run; `launched` needs the run's existing
folder, finding R5).

THE EXPERIMENT (the operator, Oct 6, 2026, on the attention cap's repair run): a repair run (role "repair") and its
matched control (role "control": a run of the same start, length, stop lines, seed and schedule with the repair
switched off, which repairs nothing) form one experiment, named in both records' repair_run blocks: {id, authorization
(the operator's words), authorized_by, authorized_at, arms {repair: FP, control: FP}, differs_only_in [flags],
exceptions {gate: {failure, why_measurable, evidence [..]}}, learning_signal {text, evidence}, outcome_validity {text,
evidence}}. A gate the run does not target stays FAIL; it no longer blocks this experiment's launch only when the
experiment names an exception for it: the failure, why the repair's effect can still be measured despite it, and the
evidence ("Keep each marked FAIL, with an explicit exception for this experiment-not PASS or N/A"). The learning
signal of the trained components and the validity of the outcome measurements are confirmed with evidence ("If either
is missing, that blocks this experiment"). The two arms' records must agree in start checkpoint, length, stop criteria
and code, and their flags may differ only in differs_only_in ("The cap setting is the only intended difference").
Each arm is used once like any repair run, except that a control block may declare repeats (the operator, Oct 6,
2026: "one cap-ON arm and two cap-OFF controls"): two control runs of one configuration share its record, each marked
launch or each attempt that trained past the start counts as one use, and the record stays a usable control until its
uses reach the declared number. The exceptions allow only these launches, never normal training.

THE TARGET OUTCOME (proxy decision pose-v2-implementation-cleared-20261015 of Oct 15, 2026, answer 4: "Implement and obtain focused review of the minimal target_outcome
alternative, initially scoped to this named pose pair. It replaces target_gates, not the seven gates themselves. Require
the actual registered defect, a concrete registered reading and executable prospective rule/inputs, bilateral
agreement, exact authorization/code/start/end pins, bounded one-use runs, preservation and remeasurement. Retain each of
the seven actual gate results and complete scoped exceptions for every FAIL. Refuse missing exceptions, mixed targets,
absent/non-executable reading, mismatched arms, unbounded runs or consumed authority. Historical admissions and
normal-training requirements stay intact."). An experiment (THE EXPERIMENT above) whose purpose is a registered outcome
rather than one of the seven gates names it in its experiment block as target_outcome {text, defect, reading_row, rule,
inputs [{name, source, sha256?}], reader {module, sha256}, pins {authorization {id, text_sha256, author}, code_commit,
code_hash, start_checkpoint, start_manifest_sha256, end_step, wall_cap_s, preserved_prefix}}; both arms' records carry
the same experiment block. Checked when a block is written and again at launch (target_outcome_problems,
target_outcome_launch_problems):
  - THE SCOPE: the experiment id and its two arms' fingerprints are an entry of TARGET_OUTCOME_ROUTES, which also pins the
    defect, the reading row, the authorizing decision and its text's sha256, the start checkpoint, the end step and the
    wall cap; nothing else is admitted by this route;
  - THE TARGET REPLACES target_gates: the repair arm names no target gate (a block naming both is refused: mixed
    targets) and its remeasure is exactly {"outcome": reading_row}; the control arm targets nothing, as before; no gate
    may read FAIL - REPAIR TARGET (set_gate refuses it, since no gate is targeted);
  - THE SEVEN GATES KEEP THEIR RESULTS: every gate reads PASS, justified N/A, or FAIL with a complete exception of the
    experiment {failure, why_measurable, evidence [..]} (the existing rule for a gate a repair run does not target); a
    FAIL without a complete exception, or a gate never judged, blocks the launch;
  - THE DEFECT is in the defect register (docs/gain_of_function_v4/defect_register.json) and open;
  - THE READING: the reading row is in the test registry, reads written or queued, and names the defect; the rule is
    written; the inputs are listed (an input that carries a sha256 hashes to it now); the reader is a Python file that
    hashes to its pinned sha256 and defines main;
  - THE AUTHORIZATION: the decision's message (Saved/gof_coordination/mailbox/<id>/message.json) holds text whose UTF-8
    sha256 is the pinned one and whose author line names the pinned author; no block of the preflight folder outside
    this experiment's two records carries that decision (consumed authority);
  - THE BOUNDS: start_checkpoint is the block's and the route's; max_model_steps equals end_step minus the start's model
    step; a launch's --max-steps equals end_step exactly (not less: the arms run the same length); the launch plan
    carries wall_cap_s for the run owner, which ends a run at it;
  - BILATERAL AGREEMENT: the other arm's record names the same experiment with the same target_outcome, authorization,
    arms and differs_only_in (arms_problems; the existing start, length, stop criteria and code agreement still hold);
  - AT LAUNCH: the configuration's code hash equals pins.code_hash; each arm is used once (repair_available, as every
    repair run); preserved_prefix names where the run owner preserves each arm's end state before its reading.
  CORRECTIONS OF OCT 15, 2026 (proxy decision pose-build-b-corrections-and-cpu-authority-20261015, ADMISSION,
  after the focused admission review; isolated, not merged):
  - ONE USE FOR BOTH ARMS: a block carrying a target outcome declares no repeats other than 1, whatever its role (the
    control's repeats of THE EXPERIMENT stay as before for experiments without a target outcome);
  - THE PINS ARE CHECKED, NOT ONLY WRITTEN: the route entry also pins code_commit, the start MANIFEST's path and sha256,
    preserved_prefix and each role's package folder; the block's pins must equal them, and the MANIFEST file itself must
    hash to the pinned sha256 now (at writing and at launch);
  - AT LAUNCH the command's --out lies inside its role's package folder, and that package's arm_continuation.json names
    the same fingerprint, code commit, preserved prefix and start MANIFEST sha256 as the pins;
  - IMPLEMENTATION IS NOT EXECUTION: the route's decision authorizes writing the records only; a launch judged in the
    tool's own preflight folder (PREFLIGHT_DIR, production) needs the route's execution entry {id, text_sha256, author}:
    a decision other than the route's decision, whose mailbox text hashes to the pin, carries the author line and names
    both arms' fingerprints and both package folders. A launch judged in any other folder (a rehearsal) is never launch
    authority and reads no execution decision.
Nothing here changes a record, a launch or a check that carries no target_outcome; a target_outcome authorizes only
its two runs, never normal training (each record's later launches are judged as before).

THE CONTINUATION (the operator's plan of Oct 11, 2026, docs/H84_PLAN_FOR_USEFUL_LEARNING_2026-10-11.html, "Prepare one
complete launch decision": "The current repair route has already used its allowed run. Prepare a narrowly named
continuation route, preserving the other holds; do not simply remove that restriction."). A record whose repair run
(role "repair") was used may take one continuation block (the record's key "continuation", written with `continuation`
from a JSON file): one more bounded run of the same configuration, from the repaired model. Every condition is checked
again at launch (continuation_block_problems, run_flag_problems, continuation_gate_problems; the review of Oct 11, 2026,
findings F1 to F8, made them as strict as written here):
  - the record's repair run launched and completed: its run folder (the marked run, or its first attempt that trained
    past the start) has an args.json whose max_steps names its end, and a stage_done.json reading ok at that model step
    with the time its trainer ended (finished_at); each of its target gates was judged again after that time;
  - the block's start checkpoint is exactly that run's end checkpoint, <run>/checkpoints/ckpt_step_<end>, and the launch
    resumes exactly the start (a later checkpoint is normal training); a rollback checkpoint is named;
  - the duration is bounded: --max-steps above the start's model step and at most start + max_model_steps;
  - the stop lines are declared in the test registry in a row of their own (stop_lines_row reads written or queued, and
    is not the repair run's stop_lines_row) and given to the watcher as stop_criteria, checked as a repair run's are
    (stop_criteria_problems, stop_field_problems on the start checkpoint's run, first_reading_problems at the launch's
    own length and report rate), with the reachability arithmetic written;
  - the run flags it relies on, outside the fingerprint (run_flags {flag: value}), each carried by the launch exactly;
    when the configuration runs the latent body, --latent-body-resume (an existing file) and --latent-body-output (a
    folder that does not exist yet);
  - the operator's authorization in the operator's words (authorization), with authorized_by and authorized_at;
  - a named purpose and the registry row of the completion jobs it promises (completion_jobs_row reads written or
    queued);
  - every gate that does not read PASS or justified N/A reads FAIL and has the block's exception {failure,
    why_measurable, evidence [..]} (exceptions {gate: ...}); an exception leaves its gate FAIL, never PASS or N/A: an
    exception carrying a status other than FAIL, or naming a gate that reads PASS or N/A, is refused; a gate never
    judged (OPEN or missing from the record), or still FAIL - REPAIR TARGET, is refused whatever the exception; an
    exception excuses the failure judged before the block was written (written_at_s), so a FAIL gate judged again
    since is refused until the block is written again;
  - no_auto_continue says how the run is kept from continuing into normal training.
launch_plan names such a launch kind "continuation" (block "continuation"), and the relaunch script treats it as it
treats a repair run: --max-steps reads the bound, the stop criteria go into the run folder's declared_criteria.json
before the trainer starts, the attempt is noted in the continuation block before the trainer starts and its launch
marked (note_attempt and mark_launched with block "continuation"), and the supervisor is left paused naming the run. It
is used once, counted as a repair run's uses are (its marked launch, or an attempt whose folder trained past the start),
and a used or launched block is not written again; a block whose attempts never trained may be written again (a retry
after a trainer that died before training needs a new --latent-body-output), its attempts carried into the new block.
A configuration has one record (one per fingerprint), so a configuration takes at most one continuation. The repair
run's block (its launch, attempts and run) is kept as written. A continuation authorizes nothing else: its exceptions
allow only its own launch, it adds no model state, and any other launch of the record, its start after its use
included, is judged as normal training, exactly as before.

THE FRESH INSTANCE (the operator's authorization of Oct 12, 2026, 6:22 AM PDT, D:/UEMA_Research/external/D/UEMA_scratch/body1_lesson_off/
OPERATOR_AUTHORIZATION_2026-10-12.txt: "implement and review the narrow preflight change that admits these two runs while
preserving its existing checks and all previous run records"). A configuration has one record and a record's repair run
is used once, so a configuration whose repair run was used could not be the matched control of a later experiment:
new_record refuses its existing record and set_repair_run refuses replacing its launched block. A fresh instance is that
control run, written in a file of its own, docs/gain_of_function_v4/preflight/instances/<fingerprint>/<experiment id>.json
(`instance`, from a JSON file): one bounded control run of the recorded configuration within a NEW experiment. It never
writes the configuration's record, so the record's gates, states, repair run, continuation and their launch history
stay byte for byte as they were. It is admitted only when (instance_problems, checked when written and at launch):
  - the record's own repair run was used (while it is unused, the control is written there, as before), and the
    instance's experiment is not that repair run's;
  - its role is control (it repairs nothing and targets no gate) and it is used once (no repeats);
  - its start checkpoint is a model state the record names (the gates were judged for it);
  - its stop lines are declared in a row of their own (not the used repair run's) that reads written or queued;
  - at launch it is judged exactly as an experiment's arm is (_repair_problems: every field, the stop criteria, the
    bound, every gate PASS, justified N/A or FAIL with the experiment's exception, and arms_problems against the other
    arm, whose record names this experiment in its repair run or in its own fresh instance), and each exception
    excuses only the failure judged before the instance was written (written_at_s), as a continuation's does.
At most one unused fresh instance of a configuration starts from a given checkpoint. launch_plan names its launch kind
"repair_run" with block "instance:<experiment id>", so the relaunch script bounds it, writes its stop criteria and leaves
the supervisor paused as for a repair run; its attempt and launch are noted in the instance file (note_attempt and
mark_launched with that block), used once as a repair run is. It adds no model state and authorizes nothing else:
every other launch of the configuration is judged exactly as before.

THE EXTENSION OF A STOPPED CONTINUATION (prepared Oct 12, 2026 for the operator's authorization of the body-1 extension,
D:/UEMA_Research/external/D/UEMA_scratch/body1_extension; on its own branch, for review). A configuration takes one continuation, used once; a
continuation that a stop line ended before its declared end leaves the authorized span unfinished, and every other route
refuses its end state (the continuation's start after its use, any later checkpoint and a fresh instance are judged as
before; normal training needs every gate PASS or justified N/A). An extension is that span's remainder: ONE bounded run
of the same configuration from exactly the end checkpoint the stopped continuation saved, ending at most at the
continuation's own declared end. It is written in a file of its own, docs/gain_of_function_v4/preflight/extensions/
<fingerprint>.json (`extension`, from a JSON file); the record is read and never written, so the record's gates, states,
repair run, continuation and their launch history stay byte for byte as they were. THE ROUTE ADMITS ONE EXTENSION ONLY
(the independent review of Oct 12, 2026, finding B-1: as first written it admitted one for any configuration whose
continuation stopped early): EXTENSION_ROUTE names it, configuration 2c3d82ca4ab4a870 whose continuation ran in
D:/UEMA_Research/external/D/UEMA_scratch/body1_continuation/run/B1C_skip13_484675, launched to end at model step 489,350 and stopped at 485,975
(its end checkpoint ckpt_step_485975). Another configuration, another continuation run, another stop step or another
end is refused when written, at launch and when its attempt or launch is noted (extension_route_problems); a standing
route for other configurations is the operator's decision and a change to this code. It is admitted only when
(extension_block_problems and continuation_gate_problems, checked when written and at launch):
  - it is the one extension EXTENSION_ROUTE names (the record's fingerprint, the continuation's run folder, the model
    step it stopped at and the end it was launched for);
  - the record's continuation was used (a marked launch, or an attempt that trained past its start) and its run STOPPED
    EARLY: the run's args.json max_steps names the end the continuation was launched for, within the continuation's
    bound (start + max_model_steps); its stage_done.json reads ok at a model step N after the start and BEFORE that end,
    with the time its trainer ended (finished_at); its checkpoint at N is complete (meta.json present). A continuation
    that reached its end has nothing left to extend;
  - the extension's start checkpoint is exactly <the continuation's run>/checkpoints/ckpt_step_<N>, and the launch
    resumes exactly it; a rollback checkpoint is named;
  - the duration is bounded: max_model_steps at most the continuation's end minus N, and --max-steps above N and at most
    N + max_model_steps: the continuation and its extension together never pass the end the continuation was launched for;
  - its stop lines are declared in a registry row of its own (not the repair run's or the continuation's) and its
    completion jobs in a row that is not the continuation's, each reading written or queued; its stop criteria are
    checked as a continuation's are (stop_criteria_problems, stop_field_problems on the start checkpoint's run, which is
    the continuation's run, first_reading_problems at the launch's own length and report rate), with its reachability;
  - the run flags it relies on (run_flags, exactly as a continuation's: with the latent body, the state it resumes, an
    existing file, and a new output folder); the latent body state it resumes is exactly the one the continuation's run
    saved, latest.pt in the continuation's --latent-body-output (finding B-3: any existing file was admitted);
  - the operator's authorization of THIS extension in the operator's words, with authorized_by and authorized_at, each
    text; the words and the time are not the continuation's (finding B-4: the continuation's authorization covered the
    continuation only, and the block could repeat it or carry a value that is not text); every other named field is
    text too, except stop_criteria, max_model_steps, run_flags and exceptions, which are checked as before;
  - every gate reads PASS, justified N/A, or FAIL with the block's exception {failure, why_measurable, evidence}, each
    written after that gate was last judged (continuation_gate_problems, unchanged; a FAIL - REPAIR TARGET or OPEN gate
    is refused);
  - no_auto_continue says how the run is kept from continuing into normal training.
launch_plan names its launch kind "continuation" with block "extension", so a launcher treats it as it treats the
continuation (bounded, its stop criteria written into the run folder, its attempt noted before the trainer starts and its
launch marked in the extension file with note_attempt and mark_launched, block "extension"). It is used once, counted as
a continuation's uses are; a used extension is not written again, and an unused one may be written again with its
attempts carried. One extension per configuration, of the continuation's run only: an extension's own run that stops
early is not extended again by this route. It adds no model state and authorizes nothing else: every other launch of
the configuration, the extension's start after its use included, is judged exactly as before.

THE ADOPTION OF A PRESERVED CHECKPOINT BY A NEW CONFIGURATION (proxy decision proxy-adopt-b1e-one-run-20261013 of
Oct 13, 2026, Saved/gof_coordination/mailbox/
proxy-adopt-b1e-one-run-20261013/message.json, section ADMISSION CORRECTION REQUIRED: "Authorize the narrow reviewed
project-tool change needed to represent this explicit checkpoint adoption and this one bounded continuation truthfully.
It must link the preserved predecessor, the code transition, this decision and the run budget, retaining all original
records and all other launch checks"; with the nine findings of the independent review of commit 67c895be0). A
continuation needs a completed repair run in its own record and starts exactly at that run's end checkpoint. A new
configuration (other code or another configuration flag, so another fingerprint and a record of its own) that starts
from a checkpoint trained under a different configuration has neither, and the decision forbids writing a run of
different code into the new record as if it had run under its fingerprint, clearing a used record, or recording a PASS
that was not measured. An adoption is ONE bounded run of a new configuration from a preserved checkpoint of another
configuration's run, written in a file of its own, docs/gain_of_function_v4/preflight/adoptions/<fingerprint>.json
(`adoption`, from a JSON file). The record is read and never written: its gates, its states and its history stay byte
for byte as they were, and no gate is qualified for normal training.
THE ROUTE ADMITS ONLY THE ADOPTIONS ADOPTION_ROUTES NAMES, and the entry pins every value the decision names (above
ADOPTION_ROUTES). Eleven entries (THE FIFTH, THE SIXTH and THE SEVENTH ENTRY below;
the eighth and the ninth, the arm-consistency matched pair's control and treatment branches; THE TENTH AND THE ELEVENTH ENTRY below). The first,
the decision's: the preserved checkpoint
D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run/checkpoints/ckpt_step_489350, start 489,350, end 494,025 (4,675
model steps), wall cap 20,683 s, and the configuration's fingerprint, filled in two steps: (1) the coordinator writes the
configuration's record (`new`, with the adoption's command, whose --resume is that checkpoint, so the record's first
state is the checkpoint its gates are judged on) and judges its seven gates FAIL (`gate`); (2) a reviewed change of the
entry's one line `fingerprint=None` to that record's fingerprint. This file lies outside the fingerprint's scope
(lem/trainer_code.py SKIP holds mimo_bridge/tools/), so step 2 leaves the fingerprint as written. An entry whose
fingerprint reads None admits nothing. Then `adoption FINGERPRINT --block-file` writes the adoption file. It is admitted
only when (adoption_problems; each condition is checked when the file is written, at launch, and when an attempt or a
launch is noted, except where named otherwise):
  - THE ROUTE (adoption_route_problems): the record's fingerprint is an entry's, and every value the entry pins is the
    block's, compared exactly: the start and predecessor checkpoint, the MANIFEST and its sha256, the model's sha256,
    the predecessor run, the outcome receipt and its sha256, the decision's id, its message's text sha256 and its
    author, the wall cap, the start and end steps, the commit the code transition starts from (its full hash or a prefix
    of at least 9 characters), and the sha256 of the latent body state the run resumes;
  - THE DECISION AND THE DELEGATION (adoption_decision_problems): the decision's message file names the decision and its
    "text" field hashes to the pinned sha256; the delegation record holds the operator's pinned sentences, and the
    authorization holds each of them verbatim;
  - THE PREDECESSOR (adoption_predecessor_problems): the block's predecessor {checkpoint, manifest_path,
    manifest_sha256, model_sha256, run, outcome_receipt, outcome_receipt_sha256}: its checkpoint is the start checkpoint,
    whose name gives start_step; the checkpoint's model.safetensors reads model_sha256, and the MANIFEST records that
    model_sha256; the latent body state the run resumes (run_flags --latent-body-resume) reads the sha256 the MANIFEST
    records for latent_body/latest.pt; the predecessor run is the preservation's copy of its folder, and its
    stage_done.json reads ok at start_step and is the file the MANIFEST records; the outcome receipt reads
    outcome_receipt_sha256. At launch the command resumes exactly the start checkpoint;
  - THE PRESERVATION (adoption_preservation_problems; adoption_preserved_files_problems when written and at launch): the
    MANIFEST exists, reads as a JSON object and reads manifest_sha256; the checkpoint holds meta.json and
    model.safetensors; every MANIFEST entry under the start checkpoint's folder, latent_body/latest.pt and every entry
    under episode_companion/ is a file of the preservation whose sha256 is the one the MANIFEST records;
  - THE CODE AND THE SETTINGS (adoption_record_problems, adoption_template_problems): the record names this
    configuration, and its flags, the hashes of the files they name and its code hash still compute to its fingerprint;
    the checkout's trainer code is the record's; the record's bytes read the sha256 the adoption file names
    (record_sha256: a record written to after the adoption, a gate judged again among the writes, needs the adoption
    written again); the record holds no repair run and no continuation, and the start checkpoint is a model state it
    names; the command template reads its pinned sha256, and (when written and at launch) the record's configuration
    flags are the template's. At launch: a command from a route's preserved checkpoint while its adoption is unused,
    whose configuration (a code file or a configuration flag) is not the adoption's, is refused with each difference
    named (_adoption_mismatch; its --resume is compared with the route's checkpoint before the adoption file is read);
    every flag but --resume, --out, --checkpoint-storage, --latent-body-resume, --latent-body-output and --max-steps
    reads as the template's; the run flags the block names (both latent body flags, a resume file that exists and an
    output folder that does not exist yet) are carried exactly; --out and --checkpoint-storage are named, do not exist
    yet, and neither they nor --latent-body-output lies in the preserved folder;
  - THE AUTHORIZATION (adoption_authorization_problems): decision {id, message_sha256, author}, code_transition
    {from_commit, to_commit, commits}, authorization (the operator's delegation sentences verbatim with the decision's
    reference: the text names the decision's id), authorized_by and authorized_at, each text; no other block of the
    preflight folder (a record's continuation or its repair run's experiment, an extension, a fresh instance's
    experiment, another adoption) carries the same decision id or the same decision message sha256, and no other block's
    authorization words lie within this one's, or this one's within them (whitespace aside);
  - THE BOUND: max_model_steps is end_step - start_step; at launch --max-steps reads exactly end_step; used once (a
    second launch after an attempt was noted is refused by the training row's record, and after an attempt trained or a
    launch was marked by the adoption file too; note_attempt and mark_launched refuse it);
  - THE REGISTRY ROWS (adoption_registry_problems; notes proxy-prediction-closure-20261013): only the training
    row the route pins may launch (stop_lines_row), and the completion jobs' row is never it; both are in the test
    registry; when written, at launch and when an attempt is noted each reads written or queued, the training row carries
    no block, and the completion jobs' row none but one waiting on this adoption (its waits_on names the training row or
    the decision); note_attempt and mark_launched record the attempt and the launch in the training row
    (ADOPTION_ROW_FIELD, through test_registry.update_row; no status is written, and completion rows are only read), and
    a training row whose record holds an attempt (noted, whether or not its trainer trained or was marked launched) or a
    launch, or another adoption's record, refuses a second launch. RESIDUAL LIMIT: the one use is held in two places, the
    adoption file and the training row's record (beside the run owner's own status and sealed last_run in that row, read
    as its status); a hand edit of both, or a launch by a tool that writes neither, is not seen by this check;
  - the stop criteria are checked as a continuation's at the launch's own report rate and length (when written and at
    launch), with the reachability arithmetic written; purpose, stop_lines, rollback_checkpoint and no_auto_continue are
    text; role adoption, no repeats, no target gate;
  - THE GATES (adoption_gate_problems, when written and at launch; the decision: "Keep each gate FAIL; no
    normal-training qualification is granted"): each of the seven gates reads FAIL with its reason and evidence, and the
    block names its complete exception {failure, relevance, why_measurable, evidence}, written after that gate was last
    judged; a gate reading PASS, N/A, OPEN or FAIL - REPAIR TARGET, or missing, is refused, and an exception never reads
    a status other than FAIL.
THE SECOND ENTRY, THE ARM-STAGE LEARNING RUN (proxy decision proxy-arm-learning-package-corrections-20261014 of
Oct 14, 2026, Saved/gof_coordination/mailbox/
proxy-arm-learning-package-corrections-20261014/message.json, item 1: "use one explicitly bounded adoption entry, not
another training arm or a fictitious repair target", "Keep single-use/history checks and the prohibition on normal
training"). Configuration c02c8947283b0b02: the command D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_learning_494025/command.json (the B1K
continuation's command with --arm-stage memory, new run, checkpoint, latent output, latent resume and recording
locations and --max-steps 496525; the recording folder by proxy-arm-m1-test-fix-recording-route-20261014)
on the derivative branch build/arm-stage-provenance-2026-10-14 at cd51f134f (its reset frames carry the measured
pose and its stage refuses a collapsed arm chain, proxy-arm-placeholder-repair-20261014; production's
641c931a6 plus the arm-stage commits: the stage replaces 30 of each body's 234 command numbers, both arms' wrist position
and wrist, lower-arm and upper-arm orientations, in the planner's first command with the pose stored with the selected
goal ring entry, and records its provenance and the achieved palm positions); the start is the preserved complete
494,025 of B1K_skip13_492800 (D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800: the checkpoint with its model, optimizer
and runtime state and replay memory, the eight saved worlds' companion and the latent body state, every file hashed in
its MANIFEST); start 494,025, end 496,525 (2,500 model steps), wall cap 5821 s (a bound the run owner enforces);
training row arm_stage_learning_494025_run. Every condition of THE ADOPTION applies to it unchanged, and no check is
changed for it: the seven gates read FAIL in its record, each with its complete exception; it is used once; the
configuration is never trained normally. Its fingerprint is the record's: a change of the derivative's trainer code or of the command's
configuration flags makes another fingerprint, so this entry and the record are written again.
launch_plan names its launch kind "continuation" with block "adoption", max_steps its end_step and wall_cap_s its wall cap (the
block's, which the route pins; the run owner enforces it), so a launcher treats it
as it treats the continuation (bounded, its stop criteria written into the run folder, its attempt noted before the
trainer starts and its launch marked with note_attempt and mark_launched, block "adoption", each given src and the
registry). It is used once: a used adoption is not written again (counted as a continuation's uses are), and once an
attempt is noted in its training row it is not launched or written again, its trainer trained or not. It adds no model state and authorizes nothing else: a configuration
ADOPTION_ROUTES lists is never trained normally (launch_refusal refuses its normal training, before and after the
adoption's use, whatever its gates read), and every other launch is judged exactly as before.

THE THIRD ENTRY, THE LINKED REPLACEMENT OF THE ARM-STAGE LEARNING RUN (proxy decisions
arm-reporting-device-fault-repair-20261015 and arm-device-result-accepted-20261015 of Oct 15, 2026, which ask for a linked replacement execution package from the preserved complete
494,025 that keeps the failed attempt and its source pins as history, admitted through a supported recorded route that
neither clears old records nor reuses an authorization). The second entry's run,
ARM1_skip13_494025, ended at model step 494,030 with an exception in the arm stage's reporting on the graphics card; its
adoption is used (its launch marked; its run trained past its start) and is kept with its training row's record, so
that configuration never launches again. The repair (build/arm-stage-device-repair-2026-10-15 at 578bf3973:
limits_effect and executed_turns read detached CPU float64 copies) changes trainer code, so the replacement is another
configuration, 50372bae86b0724a, admitted by THE ADOPTION as one more bounded run from the same preserved
checkpoint under an entry of its own: its own decision, record, command template
(D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_learning_494025_r2/command.json: new run, checkpoint, latent output and recording locations),
training row arm_stage_learning_494025_run2 and completion row; the same start 494,025, end 496,525 and wall cap 5821 s.
Two entries name one checkpoint: a launch of one configuration from it is refused while the other entry's adoption is
written and unused
(_adoption_mismatch); the second entry's adoption is used, so it refuses nothing. Every condition of THE ADOPTION
applies to it unchanged, and no check is changed for it.

THE FOURTH ENTRY, THE STAGE-OFF ARM OF THE ON/OFF COMPARISON (proxy decision
arm-onoff-reader-first-preparation-decision-20261015 of Oct 15, 2026, items D2 to D4: carry the completed ARM2 run as the ON arm and run one OFF arm from the same
preserved complete 494,025, whose command removes only the last --arm-stage memory pair and moves the four output
destinations of ARM2's command). The OFF arm is ARM2's configuration without the arm stage on the same code
(build/arm-stage-device-repair-2026-10-15 at 578bf3973), so it is another configuration, d1309fe930034fe4, admitted
by THE ADOPTION as one more bounded run from the same preserved checkpoint under an entry of its own: its own decision,
record, command template (D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_onoff_comparison_20261015/command.json),
training row arm_stage_onoff_off_run_494025 and completion row arm_stage_onoff_reading_494025; the same start 494,025 and
end 496,525, and its own wall cap of 7200 s (the decision's D4 as amended prospectively by
arm-off-prospective-cap7200-amendment-20261015: the measured 6328 s plus a policy reserve of 872 s). Three entries
name one checkpoint: a launch of one
configuration from it is refused while another entry's adoption is written and unused (_adoption_mismatch); the
second and third entries' adoptions are used, so they refuse nothing. Every condition of THE ADOPTION applies to it
unchanged, and no check is changed for it.

THE FIFTH ENTRY, THE S4R LEARNING-MEASUREMENT CONTINUATION (proxy decisions
overnight-measurement-qualification-route-decision-20261022 and overnight-finish-launch-connections-1045-20261022 of Oct 22,
2026: "Use the EXISTING bounded adoption mechanism"; "Prepare the route
with fingerprint/decision authorization unbound until the final runtime reply; an unbound entry must admit nothing").
Configuration b4ebcf3490cbd4fb: the S4R repair arm's command (record e307f78a744c22d8, source fd4236974) with new run,
checkpoint, latent output and recording locations, --resume and --latent-body-resume the preserved complete 496,000 of
S4R_skip13_494225 (D:/UEMA_Research/external/D/UEMA_preserved/s4_repair_S4R_skip13_494225) and --max-steps 496250, on
build/s4r-measure-recording-2026-10-22 at d6a12e317 (fd4236974 plus only the reviewed tenure-pair recording in
lem/model/efference.py); start 496,000, end 496,250 (250 model steps), wall cap 1200 s; training row
s4r_measure_496000_run. Its fingerprint and decision read None until the runtime decision binds them, so it admits
nothing. Every condition of THE ADOPTION applies to it unchanged, and no check is changed for it.

THE SIXTH ENTRY, THE POSE-RETURN LEARNING CONTINUATION (proxy decisions
overnight-pose-repair-result-next-learning-20261022 and overnight-pose-learning-monitor-timing-decision-20261022 of Oct
22, 2026: "the builder prepares the concrete package for one short LEARNING
continuation using this repaired planner"; "the actual launch still needs the decision on its exact manifest and
command"; "Authorize a narrow, explicit timing setting for this package/route"). The configuration is the measurement
continuation's command (configuration b4ebcf3490cbd4fb) with --physical-goals memory, --planner-search adam and
--planner-capture-every 1, new run, checkpoint, latent output and recording locations, --resume and
--latent-body-resume the preserved complete 496,250 of S4R_measure_496000 (D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000)
and --max-steps 496350, on repair/s4r-pose-objective-2026-10-22 at f0183150b (the reviewed base 4946d4e90, d6a12e317 plus
the pose-return selection and body-head input repair, and the capture-only recording commit); start 496,250, end 496,350
(100 model steps), wall cap 720 s; training row s4r_pose_learning_496250_run. ITS WATCHER TIMING: the entry pins a
5 s first-record look and 30 s to the first and each later rule check (watch_poll_s, watch_first_check_s),
the timing the run owner and the package's watcher use; the adoption block names the same (watch_timing {poll_s,
first_check_s}, compared exactly by adoption_route_problems), and the adoption's stop-line reachability is computed with
it (adoption_watch_timing); every other route keeps the 30 s look and 300 s checks, and the trainer still reads a stop
every CONTROL_POLL_STEPS (25) model steps. At the start run's fastest pace (0.268120 model steps/s, its full stats
records 496,200 and 496,250), the admission computes every stop entry first judged within 50 + ceil(0.268120 x 35)
+ 25 = 85 model steps; with the first quick check's own run time (7.0 s measured on the measurement run) the
judgement is written within 50 + ceil(0.268120 x 42) + 25 = 87 model steps; both before the run's 100. Its fingerprint and decision read None until the runtime decision binds them, so it
admits nothing. Every other condition of THE ADOPTION applies to it unchanged.

THE SEVENTH ENTRY, THE FRESH ATTEMPT a2 OF THE POSE-RETURN LEARNING CONTINUATION (proxy decision
pose-learning-repairs-decisions-20261022 of Oct 22, 2026, B2:
"prepare the fresh attempt/paths/route and manifest only after final source is pinned. Do not reuse a1 or mutate its
frozen package. No training authority is granted by this preparation."; and the
pose-package-a2-preparation-decision-20261022, item 5: "Authorize a2 draft preflight/Tests-row preparation and the
unbound seventh route entry with its existing focused adoption check after final source/configuration."). The sixth
entry's adoption is used: its run S4R_pose_learning_496250 (attempt s4r_pose_learning_496250_a1) was marked launched at
4:25 PM PDT and stopped at its first window with no complete endpoint; that entry and configuration 2819b74cc6afc95c
are kept and never admitted again. Configuration 83ff9dd7eb5e2699 is the sixth's command with new run, checkpoint,
latent output and recording locations (run S4R_pose_learning_a2_496250 in attempt s4r_pose_learning_496250_a2), with
--observer-chunk-steps 5 (each observer lane writes a chunk every 5 model steps; the trainer's default is 100) and no
scheduled or moment video (--video-every-min 0, --moment-video off), items 2 and 3 of the same decision, and every
other flag unchanged, on repair/s4r-pose-fields-s1-2026-10-22 at 7dd0eabda (f0183150b plus S1, the planned positions'
forward value equal to the live step's clamp; the command numbers no actuator reads held at the iteration-0 composite
in every candidate and in the causal chain's perturbed alternatives; the recording of the forecast poses and the hand
state; the capture flush connection, which writes a finished capture's arrays and world records at its end; and the
moment-video switch). The same start 496,250, end 496,350 (100 model steps), wall cap 720 s and pinned watcher timing
(a 5 s first-record look and 30 s checks, compared exactly with the block's watch_timing) as the sixth; training row
s4r_pose_learning_496250_a2_run. Two entries name one checkpoint: a launch of one configuration from it is refused
while the other entry's adoption is written and unused (_adoption_mismatch); the sixth entry's adoption is used, so it
refuses nothing. Its fingerprint and decision read None until the runtime decision binds them, so it admits nothing.
Every other condition of THE ADOPTION applies to it unchanged.

THE TENTH AND THE ELEVENTH ENTRY, THE MATCHED COMPARISON AT 496,250 (proxy decisions of Oct 24, 2026: reach-target-matched-launch-decision-20261024 for the OFF branch,
reach-target-matched-on-authority-20261024 for the ON branch, each with the conditions of
reach-target-matched-conditions-decision-20261024). Two configurations, one experiment with one cumulative 900 s budget:
a2's command (83ff9dd7eb5e2699) with --planner-target-candidate off (entry 10) or on (entry 11), --max-steps 496256,
--control-poll-every 1, --restore-readback on and new output locations, at the final branch commit; the same preserved
complete 496,250 of S4R_measure_496000, end 496,256 (6 model steps), wall cap 300 s; training rows
target_matched_496250_off_run and target_matched_496250_on_run; command templates the package's command_off.json and
command_on.json. Each entry is bound to its own decision (the one-decision/one-block rule is unchanged) and pins, beside
the existing keys: decision_message_sha256 (the message file's bytes), to_commit and code_hash (the code that runs; the
existing from_commit stays the code that trained the start, START's lineage.json code_commit d6a12e317), and
owner_bound {owner, owner_sha256, trainer_limit_s, save_reserve_s, exception}. OWNER-BOUND: a run that ends before any
watcher stop line can be judged carries none: its block names the same owner_bound (compared exactly), its stop_criteria
may be an empty list, and first_reading_problems is not applied; an owner bound is admitted only on a route shorter than
CONTROL_POLL_STEPS model steps whose wall cap is the bound's trainer limit; the pinned owner file must hash to its pinned
sha256 when the adoption is written, at launch and when its attempt or launch is noted (adoption_owner_problems); its
attempt is noted only with a claim whose runner process runs that file as its script (adoption_owner_claim_problems,
the first check of note_attempt) and its launch is marked only while the training row's recorded runner runs it
(adoption_owner_runner_problems, the first check of mark_launched): changing a block's metadata cannot admit another
owner. Both entries' commands pin --control-poll-every 1 (the trainer reads its control request at every model-step
boundary); CONTROL_POLL_STEPS (25) stays the default cadence the first-reading arithmetic of every other route uses. THE EXCEPTION, recorded in both entries and blocks: no full
50-model-step statistics window is available, so no long-window convergence claim. The seven gates of both records stay
FAIL with their exceptions; nothing here qualifies normal training. An entry that pins none of these keys is judged
exactly as before.

THE LINKED RECOVERY OF A USED ADOPTION (proxy decision proxy-adoption-recovery-489625-20261013 of Oct 13, 2026, Saved/gof_coordination/mailbox/
proxy-adoption-recovery-489625-20261013/message.json, item 7: "Keep original adoption fingerprint eea1e8bc7f446fb6 used,
its stopped receipt and its INCOMPLETE result. Register one linked recovery/extension naming that used record, actual
stopped checkpoint, this decision, exact code/settings, endpoint and remaining wall budget. If the tool cannot represent
recovery of an adoption, I authorize only the reviewed administrative change to represent THIS linked recovery"). An
adoption is used once and its configuration is never trained normally, so an adoption whose run was stopped before its
end leaves its authorized span unfinished and every other route refuses its end state: the used adoption file is not
written again, the record holds no repair run or continuation, the extension route names another configuration and a
fresh instance is a control run. A recovery is that span's remainder: ONE bounded run of the SAME configuration (the same
record, trainer code and flags) from the preserved copy of the stopped run's end checkpoint to the adoption's own end,
within the adoption's remaining wall time, written in a file of its own, docs/gain_of_function_v4/preflight/recoveries/
<fingerprint>.json (`recovery`, from a JSON file). The record and the used adoption's file are read and never written:
the record's gates stay as judged, and the adoption keeps its launch, its attempts, its stopped receipt, its INCOMPLETE
result and its registry rows' ends.
THE ROUTE ADMITS ONLY THE RECOVERIES RECOVERY_ROUTES NAMES, and the entry pins every value the decision names (above
RECOVERY_ROUTES). One entry: configuration eea1e8bc7f446fb6; the used adoption's file (its sha256), run, decision and
training row; the record's sha256; the stopped run's preserved end checkpoint D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/
run/checkpoints/ckpt_step_489625 with its MANIFEST, model, worlds companion and latent body state; the stopped receipt and
the INCOMPLETE outcome receipt; the decision; the trainer code's commit; the command template (the used adoption's own
launch command); the training row; the start 489,625, the end 494,025 and the wall cap 19,931 s with the receipts it is
computed from. It is admitted only when (recovery_problems; each condition is checked when the file is written, at
launch, and when an attempt or a launch is noted, except where named otherwise):
  - THE ROUTE (recovery_route_problems): every value the entry pins is the block's, compared exactly;
  - THE BUDGET (recovery_budget_problems): the wall cap is the used adoption's cap less its trainer's elapsed time, read
    from the run owner's log (the process creation time its one "trainer started" event records, to the time of its one
    "trainer ended" event), rounded down: floor(20683 - (1791913930.4329011 - 1791913178.843792)) = 19,931 s. The cap is
    a maximum the run owner enforces (the trainer's command carries no wall time), never a renewed allowance;
  - THE USED ADOPTION (recovery_link_problems): the adoption file of this configuration is the one the block names, reads
    its pinned sha256, is used with its launch marked for the pinned run, names the pinned decision and the pinned record
    sha256, ends at the same end with the original wall cap, and its code transition ends at the pinned commit; the
    record still reads that sha256 (no gate judged again since the adoption, no state added); the preservation's
    MANIFEST reads verdict pass at the stop step, names as its source the used adoption run's checkpoint at that step,
    records the worlds companion's pinned sha256 (equal to the one the checkpoint's episode_state.json names) and the
    preserved run's args.json, which resumed the adoption's start and was launched for the same end; the stopped receipt
    reads state stopped at the stop step, the trainer's exit code 0 and a preservation that passed at the start; the
    outcome receipt reads INCOMPLETE; the start continues the record's model state the adoption started from (continues:
    the preserved run's resume chain); the adoption's training row still records its launch of that run, reads status
    failed and its runner's state STOPPED, and its completion row reads failed; the stop criteria are the adoption's,
    unchanged;
  - THE DECISION AND THE DELEGATION (adoption_decision_problems), THE PRESERVATION AND THE PREDECESSOR
    (adoption_preservation_problems, adoption_preserved_files_problems when written and at launch,
    adoption_predecessor_problems: the preserved run's stage_done.json reads ok at the stop step; its model, latent body
    state and every preserved file of the start checkpoint and the worlds companion read the MANIFEST's sha256), THE CODE
    AND THE SETTINGS (adoption_record_problems with the checkout's trainer code; adoption_template_problems with the used
    adoption's launch command as the template), each checked as for the adoption, on this block;
  - THE AUTHORIZATION (recovery_authorization_problems): the words name this decision, and no other block of the
    preflight folder (the used adoption included) carries its decision id, its message sha256 or authorization words
    lying within it or around it; its authorized_at is not the used adoption's;
  - THE BOUND: start_step the stop step, end_step the adoption's end, max_model_steps their difference, wall_cap_s the
    pinned cap; at launch --max-steps reads exactly end_step; used once, held twice as the adoption's use is (the
    recovery file and its training row's record, field preflight_recovery);
  - THE GATES (adoption_gate_problems, recovery_exception_problems): each gate still reads FAIL as judged for the
    adoption, and the block carries the used adoption's exception for it unchanged {failure, relevance, why_measurable,
    evidence}, with carried_from and its applicability to the preserved same-lineage state written beside it; an
    exception never reads a status other than FAIL;
  - THE ROWS (adoption_registry_problems, field preflight_recovery): the pinned training row and a completion row that
    is neither it nor one of the used adoption's rows, each reading written or queued;
  - THE LAUNCH (_recovery_refusal): the launch resumes exactly the start, carries the run flags, reads every other flag as
    the template's, and names --out, --checkpoint-storage and --latent-body-output as new folders outside the
    preservations and outside the used adoption's run folder and latent body folder. While the recovery is unused, a
    launch from its start whose configuration is not the recovery's is refused naming each difference
    (_recovery_mismatch).
launch_plan names its launch kind "continuation" with block "recovery", max_steps its end_step and wall_cap_s its cap, so
a launcher treats it as it treats the adoption (bounded, its stop criteria written into the run folder, note_attempt and
mark_launched with block "recovery", each given src and the registry). It is used once; it adds no model state and
authorizes nothing else: the configuration is still never trained normally, and every other launch is judged exactly as
before.

THE LINKED CONTINUATION OF A STOPPED RECOVERY (proxy decision proxy-m6-result-next-run-20261013 of Oct 13, 2026, Saved/gof_coordination/mailbox/proxy-m6-result-next-run-20261013/
message.json: "Prepare one bounded continuation from the preserved complete 492800 state to the existing endpoint 494025
(1,225 model steps)", with items 3 to 6). A recovery is used once and its configuration is never trained normally, so a
recovery whose run was stopped before the adoption's end leaves the span unfinished and every route refuses its end
state: the used recovery's file is not written again, and RECOVERY_ROUTES names one entry. A link is that span's
remainder: ONE bounded run of the SAME configuration (the same record, trainer code and flags) from the preserved copy
of the stopped recovery run's end checkpoint to the adoption's own end, within the wall time the adoption's budget has
left after both trainers' lifetimes, written in a file of its own, docs/gain_of_function_v4/preflight/links/
<fingerprint>.json (`link`, from a JSON file). The record, the used adoption's file and the used recovery's file are
read and never written: the record's gates stay as judged, and the adoption and the recovery keep their launches, their
attempts, their stopped receipts, their INCOMPLETE results and their registry rows' ends. The recovery route and its
entry are unchanged.
THE ROUTE ADMITS ONLY THE LINKS LINK_ROUTES NAMES (one per configuration), and the entry pins every value the decision
names (above LINK_ROUTES). One entry: configuration eea1e8bc7f446fb6; the used recovery's file (its sha256), decision,
run, training row, start and cap; the used adoption's file (its sha256) and the original cap; the record's sha256; the
stopped recovery run's preserved end checkpoint D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run/checkpoints/
ckpt_step_492800 with its MANIFEST, model, worlds companion and latent body state; its stopped receipt and its INCOMPLETE
outcome receipt; the decision and its time; the trainer code's commit; the command template (the used recovery's own
launch command); the training row and the completion row; the start 492,800, the end 494,025 and the wall cap 12,338 s
with the two trainers' receipts it is computed from; the hourly watcher's carried clock; the run's one stop-policy
exception. It is admitted only when (link_problems; each condition is checked when the file is written, at launch, and
when an attempt or a launch is noted, except where named otherwise):
  - THE ROUTE (link_route_problems): every value the entry pins is the block's, compared exactly; the rollback checkpoint
    is the start, and the completion row and the authorized time are pinned too (the recovery route's independent
    review, findings m4 and m5);
  - THE BUDGET (link_budget_problems): the wall cap is the adoption's cap less both trainers' lifetimes, each read from
    its run owner's log (the process creation time its one "trainer started" event records, to the time of its one
    "trainer ended" event), rounded down: floor(20683 - (1791913930.4329011 - 1791913178.843792) - (1791933404.5869603 -
    1791925811.7154384)) = 12,338 s; the used recovery's own cap is the first of these alone, floor(20683 - 751.589) =
    19,931 s. The cap is a maximum the run owner enforces, never a renewed allowance;
  - THE CHAIN (link_parent_problems): the used recovery's and the used adoption's files read their pinned sha256, each
    used with its launch marked for its run; the recovery names the pinned decision, record sha256, span (489,625 to
    494,025), cap, training row and commit; the adoption keeps its original cap; the record still reads its pinned
    sha256 (no gate judged again, no state added); the stop criteria are the recovery's, unchanged (the adoption's six);
    the authorized time is neither earlier block's; the start continues the recovery's start and the record's first
    model state (the resume chain); the preservation's MANIFEST reads verdict pass at the stop step, names as its source
    the recovery run's checkpoint at that step, records the pinned worlds companion and the preserved run's args.json,
    which resumed the recovery's start and was launched for the same end; the stopped receipt reads state stopped at
    492,800, the trainer's exit code 0 and a preservation that passed at the start; the outcome receipt reads
    INCOMPLETE; the adoption's stopped receipt and INCOMPLETE outcome receipt still read the sha256 the recovery names;
    each earlier block's training row records its launch and reads status failed with its runner's state STOPPED, its
    completion row reads failed, and neither is one of the link's rows;
  - THE CARRIED CLOCK (link_clock_problems, link_first_reading_problems): the hourly watcher judges a declared stop line
    from its from_run_hour on, and its run-hour is the newest record's training seconds; the recovery's watcher wrapper
    (Carry in its watch/recover_hourly_watch.py) adds, before the run's own, the newest training seconds of the lineage's
    preserved runs (11,550.87 s for the recovery's start). The block names those runs, oldest first, and their sum, both
    pinned: the stopped continuation, the extension, the stopped adoption run and the stopped recovery run, 19,065.01 s.
    The route sums their preserved stats again, and each stop line must be first judged before the end on that clock
    (link_first_readings: first_readings with the carried seconds counted; with none carried, the same numbers). On the
    trainer's own clock alone the four lines from run-hour 1.25 would first be judged about 2,400 model steps in, after
    the 1,225-model-step span. RESIDUAL LIMIT: the wrapper is the package's; this check reads its inputs, not its run;
  - THE STOP-POLICY EXCEPTION (link_policy_problems; the decision, item 3): the block declares exactly the pinned one:
    rule M6 of the all-slot paths watch (docs/gain_of_function_v4/heads_all_slot_path_watch.py), sense proprioception,
    finite magnitude crossings, mode alert, with the decision's item 3 verbatim (held in the decision message's text),
    what still stops and its applicability. It is recorded here and applied by the watcher; the six declared stop
    criteria are unchanged, and no other rule's mode is named;
  - THE DECISION AND THE DELEGATION (adoption_decision_problems), THE PRESERVATION AND THE PREDECESSOR
    (adoption_preservation_problems, adoption_preserved_files_problems when written and at launch,
    adoption_predecessor_problems), THE CODE AND THE SETTINGS (adoption_record_problems with the checkout's trainer code;
    adoption_template_problems with the used recovery's launch command as the template), each checked as for the
    recovery, on this block;
  - THE AUTHORIZATION (link_authorization_problems): the words name this decision, and no other block of the preflight
    folder (the used adoption and recovery included) carries its decision id, its message sha256 or authorization words
    lying within it or around it;
  - THE BOUND: start_step the stop step, end_step the adoption's end, max_model_steps their difference (1,225),
    wall_cap_s the pinned cap; at launch --max-steps reads exactly end_step; used once, held twice as the recovery's use
    is (the link file and its training row's record, field preflight_link);
  - THE GATES (adoption_gate_problems, link_exception_problems): each gate still reads FAIL as judged, and the block
    carries the used recovery's exception for it unchanged {failure, relevance, why_measurable, evidence} (the adoption's,
    carried by the recovery), with carried_from and its applicability to the preserved 492,800 written beside it; an
    exception never reads a status other than FAIL;
  - THE ROWS (adoption_registry_problems, field preflight_link): the pinned training row and the pinned completion row,
    each reading written or queued;
  - THE LAUNCH (_link_refusal): the launch resumes exactly the start, carries the run flags, reads every other flag as
    the template's, and names --out, --checkpoint-storage and --latent-body-output as new folders outside the
    preservations and outside both earlier runs' run and latent body folders; --out's folder name is neither earlier
    run's (the trainer names the worlds companion folder after it; the recovery route's review, finding m6). While the
    link is unused, a launch from its start whose configuration is not the link's is refused naming each difference
    (_link_mismatch).
launch_plan names its launch kind "continuation" with block "link", max_steps its end_step and wall_cap_s its cap, so a
launcher treats it as it treats the recovery (note_attempt and mark_launched with block "link", each given src and the
registry). It is used once; it adds no model state and authorizes nothing else: the configuration is still never
trained normally, and every other launch is judged exactly as before.

EXACT FLAGS (finding F8): the trainer's parser accepts a unique prefix of a flag ("--max-step 0" reads as --max-steps), so
a command that names a flag the trainer's parser does not declare exactly is refused, and so is "--flag=value": the
preflight reads every flag as the trainer reads it, or it refuses.

THE REFERENCE: the configuration held when the rule was adopted (Oct 5, 2026) is written once with `reference`. It only
names what a new configuration changes. It gives no permission to launch: it was never preflighted (the operator: "A
routine continuation of the exact same qualified configuration does not repeat the entire preflight").

launch_refusal(cmd) -> None when the configuration may launch (normal training, the record's repair run, its
continuation, a fresh instance, the extension, the adoption, the recovery or the link), else the text that names every
gate or
condition that
blocks and why, or the differences from the
reference configuration when no record exists. The relaunch script calls it before it ends or
starts anything (docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from lem import trainer_code   # THE TRAINER CODE'S HASH, one definition shared with the hand calibration admission check

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "Source" / "Python"
PREFLIGHT_DIR = REPO / "docs" / "gain_of_function_v4" / "preflight"
BASELINE = "reference_configuration.json"
# the supervisor's record of the active run and its command (training_continuation.json); named apart from CONTINUATION,
# the continuation block's key below, which rebound this name on Oct 11, 2026 and broke --current (the continuation
# block review, note N3)
SUPERVISOR_RECORD = REPO / "Saved" / "mimo_bridge" / "training_continuation.json"

GATES = (
    (1, "Information flow", "Every signal the run needs survives to where it is needed (for the planner: all 26 "
                            "commanded handles reach the predictor and change what it predicts)."),
    (2, "Gradient flow", "Every component meant to learn receives a usable gradient from the lesson that teaches it."),
    (3, "Scale/conditioning", "No constant, normalization, attention logit or feature size overwhelms the variation "
                              "that carries the information."),
    (4, "Local stability", "Tiny input changes produce proportionately small physical consequences, not merely "
                           "similar internal numbers."),
    (5, "Causal controllability", "Changing a command predictably changes the corresponding predicted consequence and "
                                  "the actual body outcome."),
    (6, "Objective/test alignment", "Every pass and fail line of the run measures something necessary for gain of "
                                    "function directly; no stop on an unrelated internal statistic."),
    (7, "End-to-end causal dry run", "The whole chain is shown before long training: state, candidate command, a "
                                     "different predicted consequence, the planner's preference, the executed command, "
                                     "the corresponding physical consequence."),
)
# THE REPAIR RUN (the operator, Oct 5, 2026: "We want the preflight to prevent wasting training, not prevent the controlled
# training needed to repair a known failure"): a gate may read FAIL - REPAIR TARGET only in a record whose repair_run
# block names that gate as the run's purpose; normal training still needs every gate PASS or justified N/A.
REPAIR_TARGET = "FAIL - REPAIR TARGET"
STATUSES = ("OPEN", "PASS", "N/A", "FAIL", REPAIR_TARGET)
REGISTRY = REPO / "docs" / "gain_of_function_v4" / "test_registry.json"
REPAIR_FIELDS = ("purpose", "target_gates", "repairs", "affected_components", "start_checkpoint", "rollback_checkpoint",
                 "stop_lines", "stop_lines_row", "stop_criteria", "reachability", "max_model_steps", "no_auto_continue",
                 "remeasure")
# THE EXPERIMENT (the module text): a control run repairs nothing, so it names no target gate, repair or re-measurement
ROLES = ("repair", "control")
CONTROL_OPTIONAL = ("target_gates", "repairs", "remeasure")
EXCEPTION_FIELDS = ("failure", "why_measurable", "evidence")
# THE TARGET OUTCOME (the module text): the experiment's outcome in place of target gates, scoped by its route
TARGET_OUTCOME = "target_outcome"
TARGET_OUTCOME_FIELDS = ("text", "defect", "reading_row", "rule", "inputs", "reader", "pins")
TARGET_OUTCOME_PINS = ("authorization", "code_commit", "code_hash", "start_checkpoint", "start_manifest_sha256",
                       "end_step", "wall_cap_s", "preserved_prefix")
TARGET_OUTCOME_ROW_STATUSES = ("written", "queued")
DEFECT_REGISTER = REPO / "docs" / "gain_of_function_v4" / "defect_register.json"
MAILBOX = REPO / "Saved" / "gof_coordination" / "mailbox"
# each entry: experiment, arms {repair, control}, defect, reading_row, decision, decision_text_sha256, checkpoint,
# end_step, wall_cap_s, code_commit, start_manifest, start_manifest_sha256, preserved_prefix, packages {repair,
# control}, execution (None, or {id, text_sha256, author}: the exact execution decision) (written by the pose package's
# make_route_branch.py from its configuration)
TARGET_OUTCOME_ROUTE_PINS = ("code_commit", "start_manifest", "start_manifest_sha256", "preserved_prefix", "packages")
TARGET_OUTCOME_ROUTES = (
    # the pose read-out comparison of PROPRIOCEPTION_POSTURE_READOUT_REGRESSION (proxy decision pose-v2-implementation-cleared-20261015; package v2, D:/UEMA_Research/external/D/UEMA_scratch/pose_readout_fixture/COMPARISON_PACKAGE_v2.md;
    # the two packages D:/UEMA_Research/external/D/UEMA_scratch/pose_comparison_B_494025 and _A_494025)
    dict(experiment='pose_readout_comparison_494025', arms=dict(repair='27985bed6c687bf7', control='10d41243c4f07b9a'),
         defect='PROPRIOCEPTION_POSTURE_READOUT_REGRESSION', reading_row='pose_readout_comparison_reading_496500',
         decision='pose-v2-implementation-cleared-20261015',
         decision_text_sha256='55b1dc9f6ef9363c2d285f45d664644e6a8f0ead2f8142d1ab1e48925892de0b',
         checkpoint='D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025',
         end_step=496500, wall_cap_s=6339.0,
         code_commit='ff005b25407c59fbe86d342773744d290fd8bb12',
         start_manifest='D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/MANIFEST.json',
         start_manifest_sha256='b73e1e75c50214d1d478a0d67575fd7337560be224f9d5f40b02f9ea11555a11',
         preserved_prefix='pose_comparison_',
         packages=dict(repair='D:/UEMA_Research/external/D/UEMA_scratch/pose_comparison_B_494025', control='D:/UEMA_Research/external/D/UEMA_scratch/pose_comparison_A_494025'),
         execution=dict(id='pose-pair-conditional-execution-after-recorder-repair-20261016', text_sha256='ce2a67d5b298e19b4d4dd2853574d3e1e000e17892164e7448972048c51f4d8c', author="the overseer")),
)
# THE CONTINUATION (the module text): the fields its block names; exceptions may be empty when no gate fails
CONTINUATION = "continuation"
CONTINUATION_FIELDS = ("purpose", "start_checkpoint", "rollback_checkpoint", "stop_lines", "stop_lines_row",
                       "stop_criteria", "reachability", "max_model_steps", "no_auto_continue", "completion_jobs_row",
                       "authorization", "authorized_by", "authorized_at", "run_flags")
# the registry statuses a continuation's stop-lines and completion-jobs rows may read: declared for it, not yet run (the
# review of Oct 11, 2026, finding F1: a done row, the repair run's own among them, was accepted)
CONTINUATION_ROW_STATUSES = ("written", "queued")
# THE CONTINUATION'S RUN FLAGS (the review's finding F5): run flags are outside the fingerprint, so the block names the
# ones it relies on and the launch must carry them exactly; the relaunch script chooses --resume, --out and
# --checkpoint-storage, and --max-steps is the bound, so the block names none of them
CONTINUATION_CHOSEN_FLAGS = ("--resume", "--out", "--checkpoint-storage", "--max-steps")
LATENT_RUN_FLAGS = ("--latent-body-resume", "--latent-body-output")
# the record's blocks whose launches the relaunch script notes and marks
LAUNCH_BLOCKS = ("repair_run", CONTINUATION)
# THE FRESH INSTANCE (the module text): its files' folder under the preflight folder, and its launch block's name prefix
INSTANCES_DIR = "instances"
INSTANCE = "instance"
INSTANCE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
# THE EXTENSION (the module text): its files' folder under the preflight folder, and its launch block's name
EXTENSIONS_DIR = "extensions"
EXTENSION = "extension"
# THE ONE EXTENSION THE ROUTE ADMITS (the module text; the independent review of Oct 12, 2026, finding B-1): record
# 2c3d82ca4ab4a870's continuation, which ran in this folder, was launched to end at model step 489,350 (its args.json
# max_steps) and stopped at 485,975 (its stage_done.json; its end checkpoint ckpt_step_485975)
EXTENSION_ROUTE = dict(fingerprint="2c3d82ca4ab4a870",
                       continuation_run="D:/UEMA_Research/external/D/UEMA_scratch/body1_continuation/run/B1C_skip13_484675",
                       stopped_at=485975, end=489350)
# the extension's fields that are text (finding B-4); stop_criteria, max_model_steps, run_flags and exceptions have their
# own checks
EXTENSION_TEXT_FIELDS = ("purpose", "start_checkpoint", "rollback_checkpoint", "stop_lines", "stop_lines_row",
                         "reachability", "no_auto_continue", "completion_jobs_row", "authorization", "authorized_by",
                         "authorized_at")
# the latent body state file a run saves in its --latent-body-output folder (mimo_bridge/latent_body_shadow.py: "the
# output directory's latest.pt")
LATENT_STATE_FILE = "latest.pt"
# THE ADOPTION (the module text): its files' folder under the preflight folder, and its launch block's name
ADOPTIONS_DIR = "adoptions"
ADOPTION = "adoption"
# THE ADOPTIONS THE ROUTE ADMITS (the module text; proxy decision proxy-adopt-b1e-one-run-20261013 of Oct 13, 2026,
# section ADMISSION CORRECTION REQUIRED; the independent review of 67c895be0, findings 2 and 3: every value the decision
# names is pinned here and compared exactly). Each entry names: the configuration's fingerprint; the preserved predecessor
# checkpoint the run resumes; the preservation's MANIFEST.json, its sha256 and the folder under which preserved state is
# never written; the model's and the latent body state's sha256; the predecessor run's folder in the preservation
# (its stage_done.json one of the files the MANIFEST records; the coordinator's decision on the package review's finding
# 2); its outcome receipt and that file's sha256; the decision's id, its message file, the sha256 of that message's "text" field (UTF-8) and its
# author; the delegation record and the operator's quoted sentences in it (lines 7-9 and 12), each held verbatim by the
# authorization; the commit the code transition starts from; the command template and its sha256; the one training row
# that may launch (notes proxy-prediction-closure-20261013); the model steps the run starts and ends at; the
# wall cap in seconds. The fingerprint is filled in two steps: (1) the coordinator writes the
# configuration's record (training_preflight new, with the adoption's command) and judges its seven gates; (2) a reviewed
# change of the one line `fingerprint=None` below to that record's fingerprint. This file is outside the fingerprint's
# scope (lem/trainer_code.py SKIP holds mimo_bridge/tools/), so step 2 leaves the record's fingerprint as it was written.
# An entry whose fingerprint reads None admits nothing.
ADOPTION_ROUTES = (
    dict(fingerprint="eea1e8bc7f446fb6",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run/checkpoints/ckpt_step_489350",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/MANIFEST.json",
         manifest_sha256="81b2cff358ecd17edd36d65648d71a26bb744c6aed5f8b1e3c26f470186e9572",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="69e9997d6c32cf22ad3bed76f47e1168fdce6e7d51f376071dd6162f72e7a397",
         latent_sha256="68ee27286b1f412b8c43e30ec4da598f12eb5a8683c0e7ffb26742e0ae742978",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/body1_extension/reading/outcome_receipt.json",
         outcome_receipt_sha256="da924702b729be24d83565f46d8e6aec072c722213df04ea5f140ed93980e11a",
         decision="proxy-adopt-b1e-one-run-20261013",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "proxy-adopt-b1e-one-run-20261013/message.json",
         decision_text_sha256="5338b94e4ace134d049377b9514a59c2ddca86331af3b00926675b1bc18f7807",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="78717f9cd71b286099054f6de12749b60888c23e",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/body1_extension/command.json",
         command_template_sha256="21dcd5773966b7ca96693cd4edc2403513fc4e11b79b25133953318fdd035783",
         run_row="b1_adopt_B1A_skip13_489350",
         start=489350, end=494025, wall_cap_s=20683),
    # THE SECOND ENTRY, THE ARM-STAGE LEARNING RUN (the module text; proxy decision
    # proxy-arm-learning-package-corrections-20261014, item 1): one bounded run of configuration c02c8947283b0b02 (the
    # derivative branch build/arm-stage-provenance-2026-10-14 at cd51f134f with --arm-stage memory, the recording
    # route of proxy-arm-m1-test-fix-recording-route-20261014 and the reset-pose repair of
    # proxy-arm-placeholder-repair-20261014) from the
    # preserved complete 494,025 of B1K_skip13_492800 to 496,525, wall cap 5821 s
    dict(fingerprint="c02c8947283b0b02",
         checkpoint="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025",
         manifest="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/MANIFEST.json",
         manifest_sha256="b73e1e75c50214d1d478a0d67575fd7337560be224f9d5f40b02f9ea11555a11",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="8f99a3ce379303eb01c8364c9f40007a13e811c1b77f59b1b89c4058f44a315a",
         latent_sha256="6108b83ec565eb74ef493a770d377347cc83dfaeb10249fa1c07191ee7d660b4",
         predecessor_run="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_continue_492800/reading/outcome_receipt.json",
         outcome_receipt_sha256="1344f4f6e40665c3dce9d1f6e878e826d58dd4c0d3d6d93c130708c0241fab58",
         decision="proxy-arm-learning-package-corrections-20261014",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "proxy-arm-learning-package-corrections-20261014/message.json",
         decision_text_sha256="a9739e9ef2ad50861bbe787a61c81d0d221e511df7bee8cc5b627c541b9f3a7c",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_learning_494025/command.json",
         command_template_sha256="a6b71abf31b8a4599d1c6536047aef43804b3aafa204ad929f4b22b93421822e",
         run_row="arm_stage_learning_494025_run",
         start=494025, end=496525, wall_cap_s=5821),
    # THE THIRD ENTRY, THE LINKED REPLACEMENT (the module text; proxy decisions
    # arm-reporting-device-fault-repair-20261015 and arm-device-result-accepted-20261015): one bounded run of
    # configuration 50372bae86b0724a (the derivative branch build/arm-stage-device-repair-2026-10-15 at 578bf3973: the
    # second entry's code with the arm stage's reporting repair) from the same preserved complete 494,025 of
    # B1K_skip13_492800 to 496,525, wall cap 5821 s; the second entry's adoption is used and kept
    dict(fingerprint="50372bae86b0724a",
         checkpoint="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025",
         manifest="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/MANIFEST.json",
         manifest_sha256="b73e1e75c50214d1d478a0d67575fd7337560be224f9d5f40b02f9ea11555a11",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="8f99a3ce379303eb01c8364c9f40007a13e811c1b77f59b1b89c4058f44a315a",
         latent_sha256="6108b83ec565eb74ef493a770d377347cc83dfaeb10249fa1c07191ee7d660b4",
         predecessor_run="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_continue_492800/reading/outcome_receipt.json",
         outcome_receipt_sha256="1344f4f6e40665c3dce9d1f6e878e826d58dd4c0d3d6d93c130708c0241fab58",
         decision="arm-reporting-device-fault-repair-20261015",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "arm-reporting-device-fault-repair-20261015/message.json",
         decision_text_sha256="6ded0daf9b88cc477fe30e959cce179e9b2dbc63b59684a063159c4bd52b6443",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_learning_494025_r2/command.json",
         command_template_sha256="c2e8126dc77fc72072d489ffc4ae72e79cc483467977f5476904e481a0e4d286",
         run_row="arm_stage_learning_494025_run2",
         start=494025, end=496525, wall_cap_s=5821),
    # THE FOURTH ENTRY, THE STAGE-OFF ARM (the module text; proxy decision
    # arm-onoff-reader-first-preparation-decision-20261015, D2 to D4): one bounded run of configuration
    # d1309fe930034fe4 (ARM2's configuration without --arm-stage, on the derivative branch
    # build/arm-stage-device-repair-2026-10-15 at 578bf3973) from the same preserved complete 494,025 of
    # B1K_skip13_492800 to 496,525, wall cap 7200 s (D4 amended: 6328 measured + 872 reserve); the second and third
    # entries' adoptions are used and kept
    dict(fingerprint="d1309fe930034fe4",
         checkpoint="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025",
         manifest="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/MANIFEST.json",
         manifest_sha256="b73e1e75c50214d1d478a0d67575fd7337560be224f9d5f40b02f9ea11555a11",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="8f99a3ce379303eb01c8364c9f40007a13e811c1b77f59b1b89c4058f44a315a",
         latent_sha256="6108b83ec565eb74ef493a770d377347cc83dfaeb10249fa1c07191ee7d660b4",
         predecessor_run="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_continue_492800/reading/outcome_receipt.json",
         outcome_receipt_sha256="1344f4f6e40665c3dce9d1f6e878e826d58dd4c0d3d6d93c130708c0241fab58",
         decision="arm-onoff-reader-first-preparation-decision-20261015",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "arm-onoff-reader-first-preparation-decision-20261015/message.json",
         decision_text_sha256="1b7b30ccb6468f96280e27897d312233aaf1d022fd74c9078c903f71f0d74199",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_onoff_comparison_20261015/command.json",
         command_template_sha256="8edab7ae54d1270eb72402670e9b9d2066afb52eb79e268e0b028b6fc76f2ed1",
         run_row="arm_stage_onoff_off_run_494025",
         start=494025, end=496525, wall_cap_s=7200),
    # THE FIFTH ENTRY, THE S4R LEARNING-MEASUREMENT CONTINUATION (the module text; proxy decisions
    # overnight-measurement-qualification-route-decision-20261022 and overnight-finish-launch-connections-1045-20261022):
    # one bounded run of configuration b4ebcf3490cbd4fb (the S4R repair arm's configuration, record e307f78a744c22d8,
    # on fd4236974 plus only the reviewed tenure-pair recording, build/s4r-measure-recording-2026-10-22 at d6a12e317)
    # from the preserved complete 496,000 of S4R_skip13_494225 to 496,250, wall cap 1200 s. UNBOUND: its fingerprint and
    # decision read None until the runtime decision binds them, so it admits nothing
    dict(fingerprint="b4ebcf3490cbd4fb",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4_repair_S4R_skip13_494225/run/checkpoints/ckpt_step_496000",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4_repair_S4R_skip13_494225/MANIFEST.json",
         manifest_sha256="3cca330f2721dc88dba032eb91e6ac3db518bbc7b4f5f87e6e97e912e97ff458",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="9977cf5b7d02ba9521527a9d99a327c3d666a135976ce1d93520d417ad5cf6ad",
         latent_sha256="ccc3a927ac373c725b5182c666a9bba9fe18b25ea16e6a7b6454682ddaf7baa0",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4_repair_S4R_skip13_494225/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/blocker_prep_20261016/d1_d2/s4_repair_learning/owners/receipts_496225/"
                         "owner_end.json",
         outcome_receipt_sha256="a9b11b7a6c1258219a27faeec1536f949f5b239476e8da27f0cf0537ddf3ad30",
         decision="overnight-measurement-learning-authorized-20261022",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "overnight-measurement-learning-authorized-20261022/message.json",
         decision_text_sha256="0f76c0b4340f85a602d49d88d6d8064e8f77f97cc7843f500fd1328c5e4aa911",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="fd4236974ae9b81bf670ab813590c733f91c1d18",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/s4r_measure_496000_package/command.json",
         command_template_sha256="86d0eb8ee45bad617fa4f27f4d417754e22814a3e39c04a0be335186ecbf367b",
         run_row="s4r_measure_496000_run",
         start=496000, end=496250, wall_cap_s=1200),
    # THE SIXTH ENTRY, THE POSE-RETURN LEARNING CONTINUATION (the module text; proxy decisions
    # overnight-pose-repair-result-next-learning-20261022 and overnight-pose-learning-monitor-timing-decision-20261022):
    # one bounded run of the pose-return repair's configuration (repair/s4r-pose-objective-2026-10-22 at f0183150b: the
    # reviewed base 4946d4e90 plus the capture-only recording commit; from d6a12e317) from the preserved complete 496,250
    # of S4R_measure_496000 to 496,350, wall cap 720 s, the watcher's first-record look 5 s and checks 30 s.
    # UNBOUND: its fingerprint and decision read None until the runtime decision binds them, so it admits nothing
    dict(fingerprint="2819b74cc6afc95c",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run/checkpoints/ckpt_step_496250",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/MANIFEST.json",
         manifest_sha256="ccaab681c309d354178ab4f095f28e1520c64b3b8b2ca96a4399f2e49097e353",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="54893ddcd802711f05fc2db830b0d95b15cbf8af58b4e4762b9b260434e3ff4c",
         latent_sha256="d6da9591646cd61c58850f94a055ed7cc62a7ef55c4d89f4b9958c596bd49876",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/s4r_measure_496000_a1/evidence/owner/"
                         "arm_end.json",
         outcome_receipt_sha256="bba4852b9f0b7c387a9c9b9c3d2658e14dbba987c397920d06def09bb68f3ed1",
         decision="overnight-pose-learning-authorized-20261022",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "overnight-pose-learning-authorized-20261022/message.json",
         decision_text_sha256="c9c3061ddeacf98bb59f849099cb8e2eaaf432b074fd2835375ed00e72178611",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="d6a12e31725bc18dd81d187d77854cffb8a04260",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/s4r_pose_learning_496250_package/command.json",
         command_template_sha256="2ddab089b445a13a944e600550103a65f66d17824988519a0bc7d9bb7c4b9c0b",
         run_row="s4r_pose_learning_496250_run",
         start=496250, end=496350, wall_cap_s=720,
         watch_poll_s=5, watch_first_check_s=30),
    # THE SEVENTH ENTRY, THE FRESH ATTEMPT a2 OF THE POSE-RETURN LEARNING CONTINUATION (the module text; proxy
    # decisions pose-learning-repairs-decisions-20261022 B2 and pose-package-a2-preparation-decision-20261022 item 5):
    # one bounded run of configuration 83ff9dd7eb5e2699 (repair/s4r-pose-fields-s1-2026-10-22 at 7dd0eabda:
    # f0183150b plus S1, the decoder-inert command numbers held at the iteration-0 composite and in the perturbed
    # alternatives, the forecast-pose recording, the capture flush connection and the moment-video switch; from
    # d6a12e317; the command with --observer-chunk-steps 5, --video-every-min 0 and --moment-video off) from the
    # preserved complete 496,250 of S4R_measure_496000 to 496,350, wall cap 720 s, the watcher's first-record look 5 s
    # and checks 30 s; the sixth entry's adoption is used and kept. UNBOUND: its fingerprint and decision read None
    # until the runtime decision binds them, so it admits nothing
    dict(fingerprint="83ff9dd7eb5e2699",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run/checkpoints/ckpt_step_496250",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/MANIFEST.json",
         manifest_sha256="ccaab681c309d354178ab4f095f28e1520c64b3b8b2ca96a4399f2e49097e353",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="54893ddcd802711f05fc2db830b0d95b15cbf8af58b4e4762b9b260434e3ff4c",
         latent_sha256="d6da9591646cd61c58850f94a055ed7cc62a7ef55c4d89f4b9958c596bd49876",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/s4r_measure_496000_a1/evidence/owner/"
                         "arm_end.json",
         outcome_receipt_sha256="bba4852b9f0b7c387a9c9b9c3d2658e14dbba987c397920d06def09bb68f3ed1",
         decision="a2-bounded-learning-authorized-20261023",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "a2-bounded-learning-authorized-20261023/message.json",
         decision_text_sha256="c239d77e469c4daa2cc9443062dc7b1ba15f4b3f4a0ed0445867ec7a600225e4",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="d6a12e31725bc18dd81d187d77854cffb8a04260",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/s4r_pose_learning_496250_repair_package/"
                          "command.json",
         command_template_sha256="15a222d12685f14ff70ed12c13d56cfa44915a0296c83f4f5cdf385730e70008",
         run_row="s4r_pose_learning_496250_a2_run",
         start=496250, end=496350, wall_cap_s=720,
         watch_poll_s=5, watch_first_check_s=30),
    # THE EIGHTH ENTRY (BOUND), the arm-consistency matched pair's control branch: the preserved complete 496,350 of
    # S4R_pose_learning_a2_496250, 100 model steps to 496,450, its own training row and command template; fingerprint,
    # decision, decision_message and decision_text_sha256 bound by runtime decision
    # a2-consistency-pilot-authorized-20261023
    dict(fingerprint="a58629bf99a000f5",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/run/checkpoints/ckpt_step_496350",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/MANIFEST.json",
         manifest_sha256="6e30b5502962ccd01ac7ffad23a364c9843afcbcafa1ae9fcb9ce426f79fee7b",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="df36dc094a8135edbd5a4b3808f879e1e58b15bc579c47ba055f672fadad94d0",
         latent_sha256="c92786b870490176de574b1433bf41e8d8b003346170d8691f6ff95644a6f6f5",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/s4r_pose_learning_496250_a2/evidence/owner/"
                         "arm_end.json",
         outcome_receipt_sha256="03740080b4c8f993957e7afe248e66a6ff751f163566f8ffe3e76d770aca2fa9",
         decision="a2-consistency-pilot-authorized-20261023",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "a2-consistency-pilot-authorized-20261023/message.json",
         decision_text_sha256="c4b12b15e2cb3e419a4f61dd5709164ccc7a60158461e962992c21994308efc0",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="7dd0eabda866facdba382c801a0558314229c842",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/arm_consistency_496350_package/"
                          "command_control.json",
         command_template_sha256="e7c239d8fec0b4d5796fa1dc2849ac95137f86fb88e8ac5ce7b84fc020e63247",
         run_row="arm_consistency_496350_control_run",
         start=496350, end=496450, wall_cap_s=420,
         watch_poll_s=5, watch_first_check_s=30),
    # THE NINTH ENTRY (BOUND), the arm-consistency matched pair's treatment branch: the preserved complete 496,350 of
    # S4R_pose_learning_a2_496250, 100 model steps to 496,450, its own training row and command template; fingerprint,
    # decision, decision_message and decision_text_sha256 bound by runtime decision
    # a2-consistency-treatment-authorized-20261023
    dict(fingerprint="6caef5b863c6c1c0",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/run/checkpoints/ckpt_step_496350",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/MANIFEST.json",
         manifest_sha256="6e30b5502962ccd01ac7ffad23a364c9843afcbcafa1ae9fcb9ce426f79fee7b",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="df36dc094a8135edbd5a4b3808f879e1e58b15bc579c47ba055f672fadad94d0",
         latent_sha256="c92786b870490176de574b1433bf41e8d8b003346170d8691f6ff95644a6f6f5",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4r_pose_learning_S4R_pose_learning_a2_496250/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/s4r_pose_learning_496250_a2/evidence/owner/"
                         "arm_end.json",
         outcome_receipt_sha256="03740080b4c8f993957e7afe248e66a6ff751f163566f8ffe3e76d770aca2fa9",
         decision="a2-consistency-treatment-authorized-20261023",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "a2-consistency-treatment-authorized-20261023/message.json",
         decision_text_sha256="37c49fb0d7ac42c04f38727bc47e4f571a2623cf5ad79f4453f112a6c683c8de",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="7dd0eabda866facdba382c801a0558314229c842",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/arm_consistency_496350_package/"
                          "command_treatment.json",
         command_template_sha256="f3b17456980cf5fdc90814debd48fe30ff89d55912e282040c18f5eee57ed183",
         run_row="arm_consistency_496350_treatment_run",
         start=496350, end=496450, wall_cap_s=420,
         watch_poll_s=5, watch_first_check_s=30),
    # THE TENTH ENTRY, the matched comparison at 496,250, branch OFF (the reach-target-matched-launch-decision-20261024, with the
    # conditions of reach-target-matched-conditions-decision-20261024): configuration 32e8551c01083b97 (a2's command with
    # --planner-target-candidate off, --max-steps 496256, --control-poll-every 1, --restore-readback on and new
    # output locations) at the final branch commit, from the preserved complete 496,250 of S4R_measure_496000 to 496,256
    # (6 model steps), 300 s per trainer including its complete save; owner_bound (THE TENTH AND THE ELEVENTH ENTRY in the
    # module text); from_commit the code that trained the start (START's lineage.json code_commit d6a12e317), to_commit and
    # code_hash the code that runs
    dict(fingerprint="32e8551c01083b97",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run/checkpoints/ckpt_step_496250",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/MANIFEST.json",
         manifest_sha256="ccaab681c309d354178ab4f095f28e1520c64b3b8b2ca96a4399f2e49097e353",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="54893ddcd802711f05fc2db830b0d95b15cbf8af58b4e4762b9b260434e3ff4c",
         latent_sha256="d6da9591646cd61c58850f94a055ed7cc62a7ef55c4d89f4b9958c596bd49876",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/s4r_measure_496000_a1/evidence/owner/"
                         "arm_end.json",
         outcome_receipt_sha256="bba4852b9f0b7c387a9c9b9c3d2658e14dbba987c397920d06def09bb68f3ed1",
         decision="reach-target-matched-launch-decision-20261024",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "reach-target-matched-launch-decision-20261024/message.json",
         decision_text_sha256="4a7634000a3f36addc4f076c8ff08c4b1bd78ed05f60726aa1349705d0ece26d",
         decision_message_sha256="d4eff20a10e380de7909df943d7f82344db2f7619ab7cd0ef2a95d05f9d69558",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="d6a12e31725bc18dd81d187d77854cffb8a04260",
         to_commit="a48cac796693814666706fc8f392ec5cd7d60e9a",
         code_hash="b39d550b46c5ce2c9d7b39e562411b55d4eebb84ba7c1effd2fdd9817972c4ef",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/target_matched_496250_package_s2/command_off.json",
         command_template_sha256="8b8081e61d2ec45c4d4c0eed0acea6cdace2f90a9eb2bb8e08bd5f2b93aaf03e",
         run_row="target_matched_496250_s2_off_run",
         start=496250, end=496256, wall_cap_s=300,
         owner_bound=dict(owner="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/target_matched_496250_package_s2/matched_exec.py",
                          owner_sha256="4568a8d881e9e761c021f482637f500a976aac281ec43e801bd5740399ef11e4", trainer_limit_s=300, save_reserve_s=100,
                          exception=("no full 50-model-step statistics window is available in a run of 6 model steps (--report-every 50: t"
                                     "he first full stats record of a run resumed at 496,250 would come at 496,300), so the run carries no"
                                     " long-window watcher stop line and supports no long-window convergence claim; its bounds are its pin"
                                     "ned owner's: the trainer's tree ended at its row's claim + 300 s, the complete save included, and sn"
                                     "apshot_and_stop requested at the claim + 200 s when no end save has begun; the seven gates stay FAIL"
                                     " for normal training"))),
    # THE ELEVENTH ENTRY, the matched comparison at 496,250, branch ON (the reach-target-matched-on-authority-20261024, with the
    # conditions of reach-target-matched-conditions-decision-20261024): configuration 7f4f971dc00a4c90 (a2's command with
    # --planner-target-candidate on, --max-steps 496256, --control-poll-every 1, --restore-readback on and new
    # output locations) at the final branch commit, from the preserved complete 496,250 of S4R_measure_496000 to 496,256
    # (6 model steps), 300 s per trainer including its complete save; owner_bound (THE TENTH AND THE ELEVENTH ENTRY in the
    # module text); from_commit the code that trained the start (START's lineage.json code_commit d6a12e317), to_commit and
    # code_hash the code that runs
    dict(fingerprint="7f4f971dc00a4c90",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run/checkpoints/ckpt_step_496250",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/MANIFEST.json",
         manifest_sha256="ccaab681c309d354178ab4f095f28e1520c64b3b8b2ca96a4399f2e49097e353",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         model_sha256="54893ddcd802711f05fc2db830b0d95b15cbf8af58b4e4762b9b260434e3ff4c",
         latent_sha256="d6da9591646cd61c58850f94a055ed7cc62a7ef55c4d89f4b9958c596bd49876",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/s4r_measure_S4R_measure_496000/run",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/attempts/s4r_measure_496000_a1/evidence/owner/"
                         "arm_end.json",
         outcome_receipt_sha256="bba4852b9f0b7c387a9c9b9c3d2658e14dbba987c397920d06def09bb68f3ed1",
         decision="reach-target-matched-on-authority-20261024",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "reach-target-matched-on-authority-20261024/message.json",
         decision_text_sha256="4e243a3b056cf98e3425df8e8e3cf93f71ed78b57d81ee1da3027fa2dc9b8dda",
         decision_message_sha256="de5f76aba67b15040b425f3f10c5157422689ed9470de5fb10a7fa2284581c3b",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         from_commit="d6a12e31725bc18dd81d187d77854cffb8a04260",
         to_commit="a48cac796693814666706fc8f392ec5cd7d60e9a",
         code_hash="b39d550b46c5ce2c9d7b39e562411b55d4eebb84ba7c1effd2fdd9817972c4ef",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/target_matched_496250_package_s2/command_on.json",
         command_template_sha256="89ee20d204111d179c14b7c14a7a38c4e2a4dd009681794b6d943a34465d5263",
         run_row="target_matched_496250_s2_on_run",
         start=496250, end=496256, wall_cap_s=300,
         owner_bound=dict(owner="D:/UEMA_Research/external/D/UEMA_scratch/h84/d1_planner_recovery/target_matched_496250_package_s2/matched_exec.py",
                          owner_sha256="4568a8d881e9e761c021f482637f500a976aac281ec43e801bd5740399ef11e4", trainer_limit_s=300, save_reserve_s=100,
                          exception=("no full 50-model-step statistics window is available in a run of 6 model steps (--report-every 50: t"
                                     "he first full stats record of a run resumed at 496,250 would come at 496,300), so the run carries no"
                                     " long-window watcher stop line and supports no long-window convergence claim; its bounds are its pin"
                                     "ned owner's: the trainer's tree ended at its row's claim + 300 s, the complete save included, and sn"
                                     "apshot_and_stop requested at the claim + 200 s when no end save has begun; the seven gates stay FAIL"
                                     " for normal training"))),
)
# the flags the launch may set apart from the command template (the decision: "Replace only the resume/output/checkpoint-
# storage/latent-resume/latent-output locations and --max-steps (494025)"); every other flag reads as the template's
ADOPTION_TEMPLATE_REPLACED = ("--resume", "--out", "--checkpoint-storage", "--latent-body-resume", "--latent-body-output",
                              "--max-steps")
# an adoption's exception for a FAIL gate (the decision: "with each existing failure, evidence, relevance and why the run
# remains interpretable stated in the new record"; the review's finding 9)
ADOPTION_EXCEPTION_FIELDS = ("failure", "relevance", "why_measurable", "evidence")
# THE ADOPTION'S RECORD IN ITS TRAINING ROW (the review's finding 8; notes proxy-prediction-closure-20261013):
# the field of the training row (stop_lines_row) that holds the adoption's attempts and launch, written by note_attempt
# and mark_launched through test_registry.update_row; a second memory of the one use, apart from the adoption file.
# Completion rows are only read
ADOPTION_ROW_FIELD = "preflight_adoption"
# the MANIFEST entries hashed when the adoption is written and at launch (the review's finding 4): every file under the
# start checkpoint's folder, the latent body state and the episode companions
ADOPTION_COMPANION_DIR = "episode_companion/"
# the adoption block's fields, each required (exceptions too: a block naming one complete exception per gate)
ADOPTION_FIELDS = ("purpose", "predecessor", "code_transition", "decision", "authorization", "authorized_by",
                   "authorized_at", "start_checkpoint", "rollback_checkpoint", "start_step", "end_step",
                   "max_model_steps", "wall_cap_s", "run_flags", "stop_lines", "stop_lines_row", "stop_criteria",
                   "completion_jobs_row", "reachability", "no_auto_continue")
ADOPTION_TEXT_FIELDS = ("purpose", "authorization", "authorized_by", "authorized_at", "start_checkpoint",
                        "rollback_checkpoint", "stop_lines", "stop_lines_row", "completion_jobs_row", "reachability",
                        "no_auto_continue")
ADOPTION_PREDECESSOR_FIELDS = ("checkpoint", "manifest_path", "manifest_sha256", "model_sha256", "run", "outcome_receipt",
                               "outcome_receipt_sha256")
ADOPTION_CODE_FIELDS = ("from_commit", "to_commit", "commits")
ADOPTION_DECISION_FIELDS = ("id", "message_sha256", "author")
# THE LINKED RECOVERY OF A USED ADOPTION (the module text): its files' folder under the preflight folder, its launch
# block's name and the field of its training row that holds its attempts and launch (as ADOPTION_ROW_FIELD does)
RECOVERIES_DIR = "recoveries"
RECOVERY = "recovery"
RECOVERY_ROW_FIELD = "preflight_recovery"
# THE RECOVERIES THE ROUTE ADMITS (the module text; proxy decision proxy-adoption-recovery-489625-20261013, items
# 1, 2, 3 and 7: every value the decision names is pinned here and compared exactly). Each entry names: the configuration's
# fingerprint (its record exists, so the entry is filled when written); THE USED ADOPTION it continues: the sha256 of
# its adoption file (adoptions/<fingerprint>.json, read and never written), its decision, its run folder, its training
# row, the model step its run started from, its wall cap; the record's sha256 the adoption was written against; THE
# STOPPED RUN'S END: the model step it stopped at (the recovery's start), the preserved copy of its end checkpoint, the
# preservation's MANIFEST, its sha256 and the folder under which preserved state is never written, the preserved run
# folder, the model's, the worlds companion's and the latent body state's sha256; the stopped receipt (the run owner's
# arm_end.json) and the INCOMPLETE outcome receipt, each with its sha256; THE DECISION: its id, its message file, the
# sha256 of that message's "text" field (UTF-8) and its author, the delegation record and the operator's quoted
# sentences, each held verbatim by the authorization; THE CODE AND THE SETTINGS: the trainer code's commit and the
# command template (the used adoption's own launch command) with its sha256; THE BUDGET: the training row that may
# launch, the end, the remaining wall cap, and the receipts it is computed from (the run owner's log, the process creation
# time of the adoption's trainer and the time its end was observed).
RECOVERY_ROUTES = (
    dict(fingerprint="eea1e8bc7f446fb6",
         adoption_sha256="1d15cca55d5b85ea1f87079f753a4648636c3ba18d4d5600ddf06d99ff572280",
         adoption_decision="proxy-adopt-b1e-one-run-20261013",
         adoption_run="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/run/B1A_skip13_489350",
         adoption_row="b1_adopt_B1A_skip13_489350",
         adoption_start=489350,
         original_wall_cap_s=20683,
         record_sha256="e282bfc08fb4142b9b11780115898ebc85f526c466be40ad5b9da8c22e57d083",
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/run/checkpoints/ckpt_step_489625",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/MANIFEST.json",
         manifest_sha256="85f8f712d17f5f08d2db76993c84928e936b1fbfcb803d351d204b02da2a4a43",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/run",
         model_sha256="b274e85cf4fca6009e10e3dd993e39f316640132204341099581a9e6e505ba2b",
         companion_sha256="984482cc4515d4c3b6c2eee38af328765699a1dd38465a4e6e3ea417ae6b8b0a",
         latent_sha256="238f2a316489da17530885f9f963c7a3fb929a1ccd52521c28cba62ac7928d1b",
         stopped_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/receipts/adoption/arm_end.json",
         stopped_receipt_sha256="b21379f67f277dcd31bb5738487559f1beffc65effe48478831731b9e87a8d80",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/reading/outcome_receipt.json",
         outcome_receipt_sha256="c9e0c06eb42a1362a6c96594b83907672f9c2abf67d5177c355bd2274d800f2f",
         decision="proxy-adoption-recovery-489625-20261013",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "proxy-adoption-recovery-489625-20261013/message.json",
         decision_text_sha256="f382f32b0240a3ba4f9fea41c7e7b7280c98dc6d32b5d60fb23e0f0fd2e1a78f",
         decision_author="the overseer",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         code_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/command.json",
         command_template_sha256="e84a386dea90ebf9612cd23220d0e7f99d2579c5e49415905536122b305208b6",
         run_row="b1_recover_B1R_skip13_489625",
         start=489625, end=494025, wall_cap_s=19931,
         owner_log="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/receipts/owner.jsonl",
         trainer_created_s=1791913178.843792, trainer_exit_observed_s=1791913930.4329011),
)
# a recovery's exception for a FAIL gate: the used adoption's exception carried unchanged (ADOPTION_EXCEPTION_FIELDS),
# where it comes from, and its applicability to the preserved same-lineage state (the decision, item 7: "Retain the prior
# explicit gate exceptions and reconsider their applicability to this preserved same-lineage state")
RECOVERY_CARRIED_FIELDS = ("carried_from", "applicability")
# the recovery block's fields, each required (exceptions too: one complete exception per gate)
RECOVERY_FIELDS = ("purpose", "predecessor", "code", "decision", "authorization", "authorized_by", "authorized_at",
                   "start_checkpoint", "rollback_checkpoint", "start_step", "end_step", "max_model_steps", "wall_cap_s",
                   "run_flags", "stop_lines", "stop_lines_row", "stop_criteria", "completion_jobs_row", "reachability",
                   "no_auto_continue")
# the predecessor's fields: the used adoption (its file, the file's sha256, its decision, its run folder), the stop (the
# model step, the run owner's stopped receipt and its sha256) and the preserved end (as an adoption's predecessor names
# it: the checkpoint, the MANIFEST and its sha256, the model's and the companion's sha256, the preserved run folder, the
# INCOMPLETE outcome receipt and its sha256)
RECOVERY_PREDECESSOR_FIELDS = ("adoption", "adoption_sha256", "adoption_decision", "adoption_run", "stopped_step",
                               "stopped_receipt", "stopped_receipt_sha256", "checkpoint", "manifest_path",
                               "manifest_sha256", "model_sha256", "companion_sha256", "run", "outcome_receipt",
                               "outcome_receipt_sha256")
# the code the recovery runs: the commit (its full hash or a prefix of at least 9 characters) and the record's code hash
RECOVERY_CODE_FIELDS = ("commit", "code_hash")
# the outcome receipt's word for a shorter run (the decision: "its INCOMPLETE result")
INCOMPLETE = "INCOMPLETE"
# THE LINKED CONTINUATION OF A STOPPED RECOVERY (the module text): its files' folder under the preflight folder, its
# launch block's name and the field of its training row that holds its attempts and launch (as RECOVERY_ROW_FIELD does)
LINKS_DIR = "links"
LINK = "link"
LINK_ROW_FIELD = "preflight_link"
# THE LINKS THE ROUTE ADMITS (the module text; proxy decision proxy-m6-result-next-run-20261013, items 3 to 6:
# every value the decision names is pinned here and compared exactly). Each entry names: the configuration's
# fingerprint; THE CHAIN it continues: the sha256 of the used recovery's file (recoveries/<fingerprint>.json, read and
# never written), its decision, its run folder, its training row, the model step it started from and its wall cap; the
# sha256 of the used adoption's file and the adoption's original wall cap; the record's sha256; THE STOPPED RECOVERY
# RUN'S END: the model step it stopped at (the link's start), the preserved copy of its end checkpoint, the
# preservation's MANIFEST, its sha256 and the folder under which preserved state is never written, the preserved run
# folder, the model's, the worlds companion's and the latent body state's sha256; the stopped receipt (the run owner's
# arm_end.json) and the INCOMPLETE outcome receipt, each with its sha256; THE DECISION: its id, its message file, the
# sha256 of that message's "text" field (UTF-8), its author and its time (the message's created_at in PDT), the
# delegation record and the operator's quoted sentences; THE CODE AND THE SETTINGS: the trainer code's commit and the
# command template (the used recovery's own launch command) with its sha256; THE BUDGET: the training row and the
# completion row, the end, the remaining wall cap and the two trainers' receipts it is computed from (each run owner's
# log, the process creation time of its trainer and the time its end was observed); THE CARRIED CLOCK: the preserved
# runs, oldest first, whose newest records' training seconds the hourly watcher's wrapper adds before this run's own,
# and their sum; THE STOP-POLICY EXCEPTION: the one change of a stop rule the decision makes for this run (item 3).
LINK_ROUTES = (
    dict(fingerprint="eea1e8bc7f446fb6",
         record_sha256="e282bfc08fb4142b9b11780115898ebc85f526c466be40ad5b9da8c22e57d083",
         adoption_sha256="1d15cca55d5b85ea1f87079f753a4648636c3ba18d4d5600ddf06d99ff572280",
         original_wall_cap_s=20683,
         recovery_sha256="06d9fb15c5728dc28ed8abb847ba68b881030927e0372f8fe597a1be9cdfcd46",
         recovery_decision="proxy-adoption-recovery-489625-20261013",
         recovery_run="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/run/B1R_skip13_489625",
         recovery_row="b1_recover_B1R_skip13_489625",
         recovery_start=489625,
         recovery_wall_cap_s=19931,
         checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run/checkpoints/ckpt_step_492800",
         manifest="D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/MANIFEST.json",
         manifest_sha256="bde1568e7d9f14987dff906031b289a3021fe7a8f89859ab4cee908f7b6171b2",
         preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
         predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run",
         model_sha256="3e3b20270355c2d85db779be7ebb41168ff3bbede3582408f14ecd5e06f38b99",
         companion_sha256="4c91844c91971d06c42699ad252707787b176836728fcd9c2fabc935ba2317ec",
         latent_sha256="5b9bce03b670b93417c74ad0d692d18eafbcace5a359423f4cc8aa6e075769ef",
         stopped_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/receipts/recovery/arm_end.json",
         stopped_receipt_sha256="f55aec6ed3453eb3e78ec4dae53996b31d0f5050dbefd42c9b5e41a4e71d1727",
         outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/reading/outcome_receipt.json",
         outcome_receipt_sha256="0e2d7037cd63788b494573730a020fa7096a19b48e34d9221fb794f6dd215774",
         decision="proxy-m6-result-next-run-20261013",
         decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                          "proxy-m6-result-next-run-20261013/message.json",
         decision_text_sha256="49e9931da14da7b94ce78565573929453a620c8b7811dd8cc0b063787e488aa0",
         decision_author="the overseer",
         authorized_at="Oct 13, 2026 08:19:45 PM PDT",
         delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                           "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
         delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                           "You will determin launch conditions in my absense!"),
         code_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
         command_template="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/command.json",
         command_template_sha256="c396abf2d54c8621c23a73ec7365dfa364e65d1acadfd5920a93a54da0e85b9c",
         run_row="b1_continue_B1K_skip13_492800",
         completion_row="b1_continue_outcome_494025",
         start=492800, end=494025, wall_cap_s=12338,
         trainers=(("D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/receipts/owner.jsonl", 1791913178.843792, 1791913930.4329011),
                   ("D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/receipts/owner.jsonl", 1791925811.7154384, 1791933404.5869603)),
         carried_runs=("D:/UEMA_Research/external/D/UEMA_preserved/body1_continuation_B1C_skip13_484675/run",
                       "D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run",
                       "D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/run",
                       "D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run"),
         carried_s=19065.0122859478,
         stop_policy=dict(
             rule="M6", watch="docs/gain_of_function_v4/heads_all_slot_path_watch.py", sense="proprioception",
             crossing="finite", mode="alert",
             decision_words=(
                 "3. For this proposed continuation only, make FINITE proprioception M6 magnitude crossings alerts. "
                 "Keep the numeric gauge and each crossing recorded. Non-finite outputs remain stops. M6 crossings for "
                 "every other sense remain stops, as do all other current stop rules, including M4 combined prediction "
                 "error and M9 opposing fixed components; preserve the already-declared M3 treatment. Do not drop a "
                 "whole M6 report because its largest crossing is proprioception: any other sense's simultaneous "
                 "crossing must still stop. Do not invent a new pooled-sign or body-majority stopping threshold from "
                 "this single overlapping diagnostic set. This scoped change is supported by useful combined "
                 "predictions despite the magnitude crossing and the bounded remaining span, not a claim that all "
                 "future growth is harmless."))),
)
# the link block's fields, each required (exceptions too: one complete exception per gate): a recovery's, the carried
# clock {runs, seconds, basis} and the stop-policy exception {rule, watch, sense, crossing, mode, decision_words,
# still_stops, applicability}
LINK_FIELDS = RECOVERY_FIELDS + ("carried_clock", "stop_policy_exception")
# the predecessor's fields: the used recovery (its file, the file's sha256, its decision, its run folder), the used
# adoption (its file, the file's sha256), the stop (the model step, the run owner's stopped receipt and its sha256) and
# the preserved end (the checkpoint, the MANIFEST and its sha256, the model's and the companion's sha256, the preserved
# run folder, the INCOMPLETE outcome receipt and its sha256)
LINK_PREDECESSOR_FIELDS = ("recovery", "recovery_sha256", "recovery_decision", "recovery_run", "adoption",
                           "adoption_sha256", "stopped_step", "stopped_receipt", "stopped_receipt_sha256", "checkpoint",
                           "manifest_path", "manifest_sha256", "model_sha256", "companion_sha256", "run",
                           "outcome_receipt", "outcome_receipt_sha256")
LINK_CLOCK_FIELDS = ("runs", "seconds", "basis")
LINK_POLICY_PINNED = ("rule", "watch", "sense", "crossing", "mode", "decision_words")
LINK_POLICY_FIELDS = LINK_POLICY_PINNED + ("still_stops", "applicability")
# the files of a complete checkpoint the adoption checks (the trainer's checkpoint folder: meta.json names its model step,
# model.safetensors holds the model's weights)
CHECKPOINT_FILES = ("meta.json", "model.safetensors")
MODEL_FILE = "model.safetensors"
HEX_DIGITS = frozenset("0123456789abcdef")
# flags that say where a run writes, what it resumes from, how long it runs, what it shows and how often it reports:
# they change no trained number and no command
RUN_FLAGS = frozenset((
    "--out", "--resume", "--checkpoint-storage", "--max-steps", "--minutes", "--caregiver-record-dir",
    "--latent-body-output", "--latent-body-resume", "--latent-body-log-mb", "--live-view-every", "--live-state-every",
    "--live-view-body", "--live-view-size", "--checkpoint-every-min", "--video-every-min", "--video-seconds",
    "--explain-every-min", "--explain-keep-mb", "--explain-steps", "--explain-body", "--report-every",
    "--observer-reserve-gb"))
# the trainer code's scope, defined in lem/trainer_code.py (the module text: a CONFIGURATION)
CODE_ROOTS = trainer_code.ROOTS
CODE_SKIP = trainer_code.SKIP
TRAINER_FILE = "mimo_bridge/train_mimo.py"


class PreflightError(RuntimeError):
    pass


def clock() -> str:
    return time.strftime("%b %d, %Y %I:%M %p PDT").replace(" 0", " ")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return trainer_code.file_hash(path)


def flags_of(cmd) -> dict:
    """{flag: value} of a trainer command line (the arguments after `-m mimo_bridge.train_mimo`); a flag with no value
    reads True."""
    args = [str(a) for a in cmd]
    if "mimo_bridge.train_mimo" in args:
        args = args[args.index("mimo_bridge.train_mimo") + 1:]
    out, i = {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                out[a] = args[i + 1]
                i += 2
                continue
            out[a] = True
        i += 1
    return out


def apply_sets(cmd, sets):
    """The command with each (flag, value) of `sets` replaced or appended, as the relaunch script's --set does."""
    cmd = [str(a) for a in cmd]
    for flag, value in sets or []:
        flag = flag if flag.startswith("--") else "--" + flag
        if flag in cmd:
            i = cmd.index(flag)
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                cmd[i + 1] = str(value)
            else:
                cmd.insert(i + 1, str(value))
        else:
            cmd += [flag, str(value)]
    return cmd


def _add_argument_flags(tree) -> set:
    import ast
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            out |= {a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                    and a.value.startswith("--")}
    return out


def trainer_flags(src: Path = SRC) -> frozenset:
    """Every option string the trainer's parser declares, read from the source without importing it: the "--" string
    constants of the add_argument calls in mimo_bridge/train_mimo.py and in each module from which it imports a function
    named add_..._arguments (today mimo_bridge/latent_body_shadow.py, whose add_arguments(ap) parse_args calls)."""
    import ast
    path = Path(src) / TRAINER_FILE
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return frozenset()
    out = _add_argument_flags(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(a.name.startswith("add_") and "argument" in a.name
                                                    for a in node.names):
            parts = (node.module or "").split(".")
            if node.level:
                parts = ["mimo_bridge"] + [p for p in parts if p]
            module = Path(src).joinpath(*parts).with_suffix(".py")
            try:
                out |= _add_argument_flags(ast.parse(module.read_text(encoding="utf-8")))
            except (OSError, SyntaxError, ValueError):
                continue
    return frozenset(out)


def flag_problems(cmd, src: Path = SRC) -> list:
    """The flags of a trainer command line that the trainer would not read as the preflight reads them: a flag its
    parser does not declare exactly (argparse takes a unique prefix, so "--max-step 0" sets --max-steps) and a
    "--flag=value" form (flags_of reads the name and value as separate arguments). A trainer source that declares no
    flag at all is a problem too: then nothing can be checked."""
    declared = trainer_flags(src)
    args = [str(a) for a in cmd]
    if "mimo_bridge.train_mimo" in args:
        args = args[args.index("mimo_bridge.train_mimo") + 1:]
    if not declared:
        return ["the trainer's flags cannot be read from %s" % (Path(src) / TRAINER_FILE)]
    out = []
    for a in args:
        if not a.startswith("--"):
            continue
        if "=" in a:
            out.append("%s: write the flag and its value as two arguments" % a)
        elif a not in declared:
            near = sorted(f for f in declared if f.startswith(a))
            out.append("%s is not a flag of the trainer's parser%s" % (
                a, " (the parser would read it as %s)" % near[0] if len(near) == 1 else ""))
    return out


def newest_stats_record(run) -> dict:
    """The newest full stats record ("type": "mimo") of <run>/replay_training_stats.jsonl, read from the file's last 8 MB;
    {} when the file is missing or holds none."""
    path = Path(run) / "replay_training_stats.jsonl"
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8_000_000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("type") == "mimo":
            return record
    return {}


def field_value(record, path):
    """A record's value at a dot path, as the hourly watcher reads it (a digit step indexes a list), or None."""
    cur = record
    for k in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(k)
        elif isinstance(cur, (list, tuple)) and k.lstrip("-").isdigit():
            i = int(k)
            cur = cur[i] if -len(cur) <= i < len(cur) else None
        else:
            return None
    return cur


def code_hashes(src: Path = SRC) -> dict:
    """{relative path: hash} of the trainer's code (lem/trainer_code.py file_hashes; the scope is defined there)."""
    return trainer_code.file_hashes(src)


def configuration(cmd, src: Path = SRC) -> dict:
    """The configuration of a launch command: kept flags, the hashes of the files they name, the code's file hashes,
    and the fingerprint. code_hash is lem/trainer_code.py's trainer_code_hash of `src`, the hash an admission record of
    a replacement hand calibration must name."""
    flags = {k: v for k, v in flags_of(cmd).items() if k not in RUN_FLAGS}
    files = {}
    for k, v in sorted(flags.items()):
        if isinstance(v, str) and len(v) < 400:
            p = Path(v)
            if p.is_file():
                files[k] = _file_hash(p)
    code, code_hash = trainer_code.trainer_code(src)
    body = json.dumps({"flags": flags, "files": files, "code": code_hash}, sort_keys=True).encode()
    return dict(fingerprint=_sha(body)[:16], flags=flags, files=files, code_hash=code_hash, code=code)


FRESH = "fresh start (no --resume)"
STATE_GATES = (1, 2, 3, 4, 5, 7)


def _run_and_step(checkpoint):
    """(run folder, model step) of a checkpoint path <run>/checkpoints/ckpt_step_N."""
    p = Path(str(checkpoint))
    try:
        step = int(p.name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        step = None
    return p.parent.parent, step


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def continues(checkpoint, state, max_links: int = 64) -> bool:
    """True when `checkpoint` continues the model state `state`: the same checkpoint, a later checkpoint of the same
    run, or a checkpoint of a run whose resume chain (args.json "resume" of each run) reaches the state's run at or
    after the state's model step. A fresh start continues only a fresh-start state."""
    if not checkpoint or not state or state == FRESH:
        return not checkpoint and state == FRESH
    c_run, c_step = _run_and_step(checkpoint)
    q_run, q_step = _run_and_step(state)
    for _ in range(max_links):
        if _norm(c_run) == _norm(q_run):
            return c_step is not None and q_step is not None and c_step >= q_step
        try:
            parent = json.loads((c_run / "args.json").read_text(encoding="utf-8-sig")).get("resume")
        except (OSError, ValueError):
            return False
        if not parent:
            return False
        c_run, c_step = _run_and_step(parent)
    return False


def record_path(fingerprint: str, root: Path = PREFLIGHT_DIR) -> Path:
    return Path(root) / ("%s.json" % fingerprint)


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise PreflightError("the preflight record %s cannot be read: %s" % (path, exc)) from exc


def _save(path: Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def differences(conf: dict, base: dict) -> list:
    """Plain lines naming what differs between a configuration and the reference configuration."""
    out = []
    bf, cf = base.get("flags") or {}, conf.get("flags") or {}
    for k in sorted(set(bf) | set(cf)):
        if bf.get(k) != cf.get(k):
            out.append("flag %s: %s -> %s" % (k, bf.get(k, "absent"), cf.get(k, "absent")))
    bfi, cfi = base.get("files") or {}, conf.get("files") or {}
    for k in sorted(set(bfi) | set(cfi)):
        if bfi.get(k) != cfi.get(k) and bf.get(k) == cf.get(k):
            out.append("the file named by %s changed (%s)" % (k, cf.get(k)))
    bc, cc = base.get("code") or {}, conf.get("code") or {}
    changed = sorted(k for k in set(bc) | set(cc) if bc.get(k) != cc.get(k))
    if changed:
        shown = ", ".join(changed[:12]) + (" and %d more" % (len(changed) - 12) if len(changed) > 12 else "")
        out.append("code: %d file(s) differ: %s" % (len(changed), shown))
    return out


def gate_problem(gate: dict):
    """Why a gate blocks, or None when it reads PASS, or N/A with a reason and evidence."""
    name = "gate %s %s" % (gate.get("n"), gate.get("name"))
    status = gate.get("status")
    reason = str(gate.get("reason") or "").strip()
    evidence = [e for e in (gate.get("evidence") or []) if str(e).strip()]
    if status == "PASS":
        return None if evidence else "%s: PASS without evidence (name the row, file or commit that shows it)" % name
    if status == "N/A":
        if reason and evidence:
            return None
        return "%s: N/A needs a written reason and evidence (reason: %s; evidence: %s)" % (
            name, reason or "none", ", ".join(evidence) or "none")
    if status == "FAIL":
        return "%s: FAIL - %s (evidence: %s)" % (name, reason or "no reason written", ", ".join(evidence) or "none")
    if status == REPAIR_TARGET:
        return ("%s: FAIL - REPAIR TARGET - %s (evidence: %s); it allows only the launch of its repair run, never "
                "normal training" % (name, reason or "no reason written", ", ".join(evidence) or "none"))
    return "%s: %s - no result recorded" % (name, status or "OPEN")


def _registry_rows(registry: Path = REGISTRY) -> dict:
    try:
        data = json.loads(Path(registry).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {r.get("id"): r for r in data.get("rows") or [] if isinstance(r, dict)}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def stop_criteria_problems(crits) -> list:
    """What keeps a repair run's stop_criteria from being judged by the hourly watcher (hourly_watch.declared_criteria
    reads entries {name, path, op, line, from_run_hour, mode, note}): an entry that is not one, a missing name or stats
    field path, an op other than < or >, a line or from_run_hour that is not a number, a mode other than alert or stop,
    and a list with no stop (then no declared line can stop the run)."""
    if not isinstance(crits, list):
        return ["the repair run's stop criteria must be a list of the watcher's declared criteria (read %s)"
                % type(crits).__name__]
    out = []
    for i, c in enumerate(crits, 1):
        if not isinstance(c, dict):
            out.append("stop criterion %d is not an entry {name, path, op, line, mode}" % i)
            continue
        label = "stop criterion %d (%s)" % (i, c.get("name") or "no name")
        for k in ("name", "path"):
            if not str(c.get(k) or "").strip():
                out.append("%s has no %s" % (label, k))
        if c.get("op") not in ("<", ">"):
            out.append("%s: op must be < or > (reads %s)" % (label, c.get("op")))
        if not _number(c.get("line")):
            out.append("%s: line must be a number (reads %s)" % (label, c.get("line")))
        if not _number(c.get("from_run_hour", 0)) or c.get("from_run_hour", 0) < 0:
            out.append("%s: from_run_hour must be a number of hours, 0 or more (reads %s)" % (
                label, c.get("from_run_hour")))
        if c.get("mode") not in ("alert", "stop"):
            out.append("%s: mode must be alert or stop (reads %s)" % (label, c.get("mode")))
        if "new_field" in c and not str(c.get("new_field") or "").strip():
            out.append("%s: new_field, when given, says why the start run's stats cannot carry the field" % label)
    if crits and not any(isinstance(c, dict) and c.get("mode") == "stop" for c in crits):
        out.append("no stop criterion has mode stop: no declared line could stop the run")
    return out


# WHEN A REPAIR RUN'S STOP LINES ARE FIRST READ (the re-check of Oct 5, 2026, finding N2; the round-two review's R1 and
# R3): the trainer writes a full stats record when its model step is divisible by --report-every; the watcher, started at
# the trainer's start, looks for the first record every 30 s and makes its first 5-minute check 300 s after the look that
# finds it, then one every 300 s; the trainer reads a stop request when its model step is divisible by 25
# (docs/h84_launch_2026-09-18/hourly_watch.py main; mimo_bridge/train_mimo.py poll_control)
WATCH_POLL_S = 30
WATCH_FIRST_CHECK_S = 300
CONTROL_POLL_STEPS = 25
ABSENT_FIELD_WINDOW_S = 3600.0        # hourly_watch.ABSENT_FIELD_WINDOW_S: the hour of grace of an absent stop field
TRAINER_REPORT_EVERY = 100            # train_mimo.py: ap.add_argument("--report-every", type=int, default=100)


def stats_records(run) -> list:
    """Every full stats record ("type": "mimo") of <run>/replay_training_stats.jsonl, in the file's order; [] when the
    file is missing."""
    out = []
    try:
        with open(Path(run) / "replay_training_stats.jsonl", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("type") == "mimo":
                    out.append(record)
    except OSError:
        return []
    return out


def fastest_rate(records):
    """The fastest pace between two consecutive full stats records, in model steps per second (the most model steps a
    repair run can take in a given time at the start run's pace), or None."""
    best = None
    for a, b in zip(records, records[1:]):
        if all(_number(r.get(k)) for r in (a, b) for k in ("step", "elapsed_s")) \
                and b["elapsed_s"] > a["elapsed_s"] and b["step"] > a["step"]:
            pace = (b["step"] - a["step"]) / (b["elapsed_s"] - a["elapsed_s"])
            best = pace if best is None or pace > best else best
    return best


def _report_every(run, given=None) -> int:
    """`given` (a launch command's --report-every), else the start run's args.json report_every, else the trainer's
    default."""
    if given is not None:
        return int(given)
    try:
        value = json.loads((Path(run) / "args.json").read_text(encoding="utf-8-sig")).get("report_every")
        return int(value) if value else TRAINER_REPORT_EVERY
    except (OSError, ValueError, TypeError):
        return TRAINER_REPORT_EVERY


def first_readings(rr: dict, report_every=None, timing=None):
    """{stop entry's name: the most model steps a repair run can have taken when the watcher first judges that entry and
    the trainer has read a stop it writes}, from the start step, the report rate (`report_every`: the launch command's
    --report-every, else the start run's args.json, else 100) and the start run's fastest pace; or a text saying what
    cannot be read. An entry is first judged at the first check after the first full stats record (at most 330 s after
    it), and not before the newest record's run-hour reaches its from_run_hour; an entry with new_field (its field not
    seen in the start run's stats) may be absent, and its absence is judged once the records span an hour and the
    run-hour reaches from_run_hour + 1 (the hour of grace). A record that crosses such a time comes up to one report
    interval after it and the check up to 300 s after that record, so each later time adds `every` model steps and the
    pace x 300 s. `timing` (look seconds, check seconds): an adoption's route-pinned timing (adoption_watch_timing), in place
    of the 30 s look and the 300 s checks."""
    start_run, start_step = _run_and_step(rr.get("start_checkpoint") or "")
    rate = fastest_rate(stats_records(start_run))
    if rate is None or start_step is None:
        return "the start checkpoint's run (%s) does not give its pace between two full stats records" % start_run
    every = _report_every(start_run, report_every)
    to_first_record = every - (start_step % every) if start_step % every else every
    after = lambda seconds: int(math.ceil(rate * seconds))                        # noqa: E731
    poll_s, check_s = timing if timing else (WATCH_POLL_S, WATCH_FIRST_CHECK_S)
    out = {}
    for c in rr.get("stop_criteria") or []:
        if not isinstance(c, dict) or c.get("mode") != "stop":
            continue
        try:
            hour = float(c.get("from_run_hour", 0) or 0)
        except (TypeError, ValueError):
            hour = 0.0
        steps = to_first_record + after(poll_s + check_s)
        if hour > 0:
            steps = max(steps, after(hour * 3600 + check_s) + every)
        if str(c.get("new_field") or "").strip():
            steps = max(steps, max(to_first_record + after(ABSENT_FIELD_WINDOW_S + check_s),
                                   after(hour * 3600 + ABSENT_FIELD_WINDOW_S + check_s)) + every)
        out[str(c.get("name"))] = steps + CONTROL_POLL_STEPS
    return out


def first_reading_problems(rr: dict, report_every=None, run_steps=None, timing=None) -> list:
    """Each stop entry the watcher would first judge only after the repair run ends (first_readings): no such line would
    be judged while the run trains. The run ends `run_steps` model steps after its start (a launch's --max-steps minus
    the start step), else at its declared bound (max_model_steps)."""
    if not rr.get("start_checkpoint"):
        return []
    try:
        steps = int(run_steps if run_steps is not None else rr.get("max_model_steps") or 0)
    except (TypeError, ValueError):
        return []
    readings = first_readings(rr, report_every, timing)
    if isinstance(readings, str):
        return [readings + ", so when the watcher first reads the stop lines cannot be computed"]
    return ["the repair run ends %d model steps after its start, before the watcher first judges stop line %s and the "
            "trainer reads its stop, up to %d model steps in at the start run's fastest pace" % (steps, name, first)
            for name, first in readings.items() if steps <= first]


def trained_past_start(run, start_step) -> bool:
    """True when a run folder shows training past `start_step`: a full stats record, or stage_done.json naming a later
    model step."""
    if stats_records(run):
        return True
    try:
        return int(json.loads((Path(run) / "stage_done.json").read_text(encoding="utf-8")).get("steps") or 0) > int(
            start_step or 0)
    except (OSError, ValueError, TypeError):
        return False


def repair_repeats(rr: dict) -> int:
    """How many runs the block allows: its declared repeats (a control run only), else 1."""
    try:
        return max(1, int(rr.get("repeats") or 1))
    except (TypeError, ValueError):
        return 1


def repair_uses(rr: dict) -> list:
    """[(time text, time in seconds, run folder)] of each use of a record's repair run, once per run folder: its marked
    launches, then each attempt whose folder trained past the start."""
    marks = []
    if rr.get("launched_at"):
        marks.append((rr.get("launched_at"), rr.get("launched_at_s"), rr.get("run")))
    marks += [(m.get("at"), m.get("at_s"), m.get("run")) for m in rr.get("launches") or [] if isinstance(m, dict)]
    start = _step_of(rr.get("start_checkpoint"))
    trained = [(a.get("at"), a.get("at_s"), a.get("run")) for a in rr.get("attempts") or []
               if isinstance(a, dict) and a.get("run") and trained_past_start(a["run"], start)]
    out, seen = [], set()
    for use in marks + trained:
        key = _norm(use[2] or "")
        if key not in seen:
            seen.add(key)
            out.append(use)
    return out


def repair_available(rr: dict) -> bool:
    """True while a record's repair run may still be launched: its uses are fewer than its declared repeats."""
    return bool(rr) and len(repair_uses(rr)) < repair_repeats(rr)


def repair_launch(rr: dict):
    """(the time text, the time in seconds, the run folder) of a record's repair run once it is used: its marked launch,
    or else its first attempt whose run folder trained past the start (the round-two review's R4: the script can end
    before it marks the launch); (None, None, None) while unused."""
    if rr.get("launched_at"):
        return rr.get("launched_at"), rr.get("launched_at_s"), rr.get("run")
    start = _step_of(rr.get("start_checkpoint"))
    for a in rr.get("attempts") or []:
        if isinstance(a, dict) and a.get("run") and trained_past_start(a["run"], start):
            return a.get("at"), a.get("at_s"), a.get("run")
    return None, None, None


def descends_from(checkpoint, run_folder, max_links: int = 64) -> bool:
    """True when `checkpoint` is a checkpoint of `run_folder`, or of a run whose resume chain (each run's args.json
    "resume") reaches a checkpoint of `run_folder`."""
    if not checkpoint or not run_folder:
        return False
    c_run, _step = _run_and_step(checkpoint)
    for _ in range(max_links):
        if _norm(c_run) == _norm(run_folder):
            return True
        try:
            parent = json.loads((c_run / "args.json").read_text(encoding="utf-8-sig")).get("resume")
        except (OSError, ValueError):
            return False
        if not parent:
            return False
        c_run, _step = _run_and_step(parent)
    return False


def stop_field_problems(rr: dict) -> list:
    """Each stop criterion whose field reads no number in the newest full stats record of the start checkpoint's run, or
    is absent for an hour of its records from its third on, unless the entry's new_field says why that run's stats
    cannot carry it (the relaunch review of Oct 5, 2026, finding F3: a misspelled path would never be judged; the
    round-two review's R2: the watcher stops a run whose stop field is absent for an hour, though its line holds)."""
    crits = rr.get("stop_criteria")
    if not isinstance(crits, list) or not rr.get("start_checkpoint"):
        return []
    start_run, _step = _run_and_step(rr["start_checkpoint"])
    record = newest_stats_record(start_run)
    records = stats_records(start_run)[2:]           # a segment's first two records may lack late fields (finding N1)
    out = []
    for i, c in enumerate(crits, 1):
        if not isinstance(c, dict) or not str(c.get("path") or "").strip() or str(c.get("new_field") or "").strip():
            continue
        if not record:
            out.append("stop criterion %d (%s): %s holds no full stats record, so its field %s cannot be checked (name "
                       "why in new_field when only this run's configuration writes it)" % (
                           i, c.get("name"), Path(start_run) / "replay_training_stats.jsonl", c["path"]))
        elif not _number(field_value(record, c["path"])):
            out.append("stop criterion %d (%s): its field %s reads no number in the newest full stats record of the "
                       "start checkpoint's run (model step %s); the watcher would never judge it (name why in new_field "
                       "when only this run's configuration writes it)" % (i, c.get("name"), c["path"],
                                                                         record.get("step")))
        else:
            gap = longest_absence_s(records, c["path"])
            if gap >= ABSENT_FIELD_WINDOW_S:
                out.append("stop criterion %d (%s): its field %s reads no number for %.0f s in a row of the start run's "
                           "full stats records from its third on; the watcher would stop a repair run whose line holds "
                           "after %.0f s of absence (choose a field the start run writes in every record, or name why in "
                           "new_field)" % (i, c.get("name"), c["path"], gap, ABSENT_FIELD_WINDOW_S))
    return out


def longest_absence_s(records, path) -> float:
    """The longest time, by the records' elapsed_s, from the last record reading a finite number at `path` (or the first
    record) to the next one reading it (or the last record)."""
    longest, since = 0.0, None
    for r in records:
        t = r.get("elapsed_s")
        if not _number(t):
            continue
        if since is None:
            since = t
        if _number(field_value(r, path)):
            longest = max(longest, t - since)
            since = t
    if records and since is not None and _number(records[-1].get("elapsed_s")):
        longest = max(longest, records[-1]["elapsed_s"] - since)
    return longest


def repair_block_problems(rr: dict, registry: Path = REGISTRY, report_every=None, run_steps=None) -> list:
    """What a repair_run block lacks: every field named, the target gates among the seven, the duration bounded, a
    re-measurement named for each target gate, the stop lines' row declared in the test registry before launch, and
    stop criteria the hourly watcher can judge while the run trains, at least one a stop. `report_every` and `run_steps`
    are the launch command's report rate and run length, when a launch is judged (first_reading_problems)."""
    out = []
    role = rr.get("role", "repair")
    outcome = target_outcome_of(rr) is not None         # THE TARGET OUTCOME: it replaces target_gates
    for k in REPAIR_FIELDS:
        if role == "control" and k in CONTROL_OPTIONAL:
            continue
        if outcome and k == "target_gates":
            continue
        v = rr.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v):
            out.append("the repair run's %s is missing" % k.replace("_", " "))
    if role == "control" and rr.get("target_gates"):
        out.append("a control run repairs nothing and targets no gate (target_gates reads %s)" % rr["target_gates"])
    if "repeats" in rr:
        try:
            ok = int(rr["repeats"]) >= 1 and not isinstance(rr["repeats"], bool)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append("repeats must be a whole number of runs, 1 or more (reads %r)" % (rr["repeats"],))
        elif int(rr["repeats"]) > 1 and role != "control":
            out.append("only a control run may repeat (a repair run is used once)")
    out += experiment_problems(rr)
    out += target_outcome_problems(rr, registry)        # THE TARGET OUTCOME (the module text)
    if rr.get("stop_criteria"):
        out += stop_criteria_problems(rr["stop_criteria"])
        out += stop_field_problems(rr)
        out += first_reading_problems(rr, report_every, run_steps)
    targets = rr.get("target_gates") or []
    if any(not isinstance(n, int) or n not in [g[0] for g in GATES] for n in targets):
        out.append("the target gates must be numbers of the seven gates (read %s)" % targets)
    try:
        if int(rr.get("max_model_steps") or 0) <= 0:
            out.append("the repair run's duration is not bounded (max_model_steps must be above 0)")
    except (TypeError, ValueError):
        out.append("max_model_steps is not a number")
    remeasure = rr.get("remeasure") or {}
    for n in targets:
        if not str(remeasure.get(str(n)) or "").strip():
            out.append("no re-measurement is named for target gate %s (remeasure['%s'])" % (n, n))
    row = rr.get("stop_lines_row")
    if row:
        r = _registry_rows(registry).get(row)
        if r is None:
            out.append("the stop lines' row %s is not in the test registry (they are declared there before launch)" % row)
        elif r.get("status") == "cancelled":
            out.append("the stop lines' row %s is cancelled" % row)
    return out


def _evidence(v) -> bool:
    return isinstance(v, list) and any(str(e).strip() for e in v)


def experiment_problems(rr: dict) -> list:
    """What a repair_run block's experiment lacks (THE EXPERIMENT in the module text); [] for a repair run of no
    experiment. A control run exists only within an experiment."""
    role, exp = rr.get("role", "repair"), rr.get("experiment")
    out = []
    if role not in ROLES:
        out.append("the run's role must be repair or control (reads %s)" % role)
    if exp is None:
        return out + (["a control run exists only within an experiment: its experiment block is missing"]
                      if role == "control" else [])
    if not isinstance(exp, dict):
        return out + ["the experiment must be a block {id, authorization, arms, exceptions, ...}"]
    for k in ("id", "authorization", "authorized_by", "authorized_at"):
        if not str(exp.get(k) or "").strip():
            out.append("the experiment's %s is missing" % k.replace("_", " "))
    arms = exp.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ROLES) or not all(str(arms.get(r) or "").strip() for r in ROLES):
        out.append("the experiment names its two arms' configurations: arms {repair: FINGERPRINT, control: FINGERPRINT}")
    if not isinstance(exp.get("differs_only_in"), list) or not exp.get("differs_only_in"):
        out.append("the experiment names the flags in which its arms may differ (differs_only_in)")
    for k in ("learning_signal", "outcome_validity"):
        v = exp.get(k) if isinstance(exp.get(k), dict) else {}
        if not str(v.get("text") or "").strip() or not _evidence(v.get("evidence")):
            out.append("the experiment's %s needs its text and evidence (the operator, Oct 6, 2026: if either is "
                       "missing, that blocks this experiment)" % k.replace("_", " "))
    targets = set(rr.get("target_gates") or [])
    exceptions = exp.get("exceptions") or {}
    if not isinstance(exceptions, dict):
        return out + ["the experiment's exceptions must be a block {gate: {failure, why_measurable, evidence}}"]
    for key, x in exceptions.items():
        try:
            n = int(key)
        except (TypeError, ValueError):
            out.append("exception %r is not a gate number" % (key,))
            continue
        if n not in [g[0] for g in GATES]:
            out.append("exception %d is not one of the seven gates" % n)
        if n in targets:
            out.append("gate %d is a target of the run; an exception is for a gate the run does not target" % n)
        x = x if isinstance(x, dict) else {}
        for f in EXCEPTION_FIELDS:
            if not (_evidence(x.get(f)) if f == "evidence" else str(x.get(f) or "").strip()):
                out.append("the exception for gate %d lacks its %s" % (n, f.replace("_", " ")))
    return out


# ------------------------------------------------------------------ THE TARGET OUTCOME (the module text)
def target_outcome_of(rr: dict):
    """The experiment's target_outcome block of a repair_run block, or None (the block's experiment names none)."""
    exp = rr.get("experiment") if isinstance(rr, dict) else None
    return exp.get(TARGET_OUTCOME) if isinstance(exp, dict) and TARGET_OUTCOME in exp else None


def target_outcome_route(experiment_id):
    for route in TARGET_OUTCOME_ROUTES:
        if isinstance(route, dict) and route.get("experiment") == experiment_id:
            return route
    return None


def _defect_problems(defect_id, register=None) -> list:
    try:
        data = json.loads(Path(register or DEFECT_REGISTER).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return ["the defect register %s cannot be read (%s)" % (register or DEFECT_REGISTER, exc)]
    found = [d for d in data.get("defects") or [] if isinstance(d, dict) and d.get("id") == defect_id]
    if not found:
        return ["the target outcome's defect %s is not in the defect register" % defect_id]
    d = found[0]
    closed = d.get("state") == "VERIFIED" or (d.get("closed") and d.get("state") in ("FAILED", "SUPERSEDED"))
    return ["the target outcome's defect %s is closed (state %s)" % (defect_id, d.get("state"))] if closed else []


def _reader_problems(reader) -> list:
    import ast
    reader = reader if isinstance(reader, dict) else {}
    path, want = reader.get("module"), reader.get("sha256")
    if not _text(path) or not _is_sha256(want):
        return ["the target outcome's reader names no module file and sha256 (reader {module, sha256})"]
    got = _file_sha256(path)
    if got is None:
        return ["the target outcome's reader %s does not exist" % path]
    if got != want:
        return ["the target outcome's reader %s hashes %s, not the pinned %s" % (path, got, want)]
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, SyntaxError) as exc:
        return ["the target outcome's reader %s does not parse (%s)" % (path, exc)]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
        return ["the target outcome's reader %s defines no main" % path]
    return []


def _authorization_text_problems(auth) -> list:
    auth = auth if isinstance(auth, dict) else {}
    did, want, author = auth.get("id"), auth.get("text_sha256"), auth.get("author")
    if not _text(did) or not _is_sha256(want) or not _text(author):
        return ["the target outcome's authorization names its decision id, text sha256 and author"]
    path = MAILBOX / str(did) / "message.json"
    try:
        msg = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return ["the authorizing decision's message %s cannot be read (%s)" % (path, exc)]
    text = msg.get("text") if isinstance(msg, dict) else None
    if not isinstance(text, str):
        return ["the authorizing decision's message %s holds no text" % path]
    out = []
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != want:
        out.append("the authorizing decision's text hashes %s, not the pinned %s" % (got, want))
    if ("Author: %s." % author) not in text:
        out.append("the authorizing decision's text holds no author line naming %s" % author)
    return out


def route_pin_problems(route: dict, pins: dict) -> list:
    """THE PINS ARE CHECKED (the module text, corrections of Oct 15, 2026): the block's code commit, start MANIFEST sha256
    and preserved prefix equal the route entry's, and the route's start MANIFEST file hashes to its pinned sha256 now."""
    out = []
    missing = [k for k in TARGET_OUTCOME_ROUTE_PINS if not route.get(k)]
    if missing:
        return ["the route entry of %s lacks %s" % (route.get("experiment"), ", ".join(missing))]
    for what, theirs, mine in (("code commit", route["code_commit"], pins.get("code_commit")),
                               ("start manifest sha256", route["start_manifest_sha256"],
                                pins.get("start_manifest_sha256")),
                               ("preserved prefix", route["preserved_prefix"], pins.get("preserved_prefix"))):
        if theirs != mine:
            out.append("the target outcome's %s reads %r, the route's %r" % (what, mine, theirs))
    got = _file_sha256(route["start_manifest"])
    if got != route["start_manifest_sha256"]:
        out.append("the start MANIFEST %s hashes %s, not the pinned %s" % (route["start_manifest"], got,
                                                                          route["start_manifest_sha256"]))
    return out


def execution_problems(route: dict) -> list:
    """IMPLEMENTATION IS NOT EXECUTION (the module text): the route's execution entry, required for a launch judged in the
    production preflight folder."""
    ex = route.get("execution")
    if not isinstance(ex, dict):
        return ["the route names no execution decision: a launch of experiment %s in the production preflight folder "
                "needs the exact execution decision; the route's decision %s authorizes writing the records only"
                % (route.get("experiment"), route.get("decision"))]
    if ex.get("id") == route.get("decision"):
        return ["the route's execution decision %s is its implementation decision" % ex.get("id")]
    out = _authorization_text_problems(ex)
    if out:
        return ["the execution decision: %s" % x for x in out]
    try:
        text = json.loads((MAILBOX / str(ex["id"]) / "message.json").read_text(encoding="utf-8-sig"))["text"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ["the execution decision's message cannot be read (%s)" % exc]
    flat = text.replace("\\", "/")
    names = list((route.get("arms") or {}).values()) + [str(p).replace("\\", "/")
                                                         for p in (route.get("packages") or {}).values()]
    absent = [n for n in names if n not in flat]
    if absent:
        out.append("the execution decision's text does not name %s (both arms' fingerprints and package folders)"
                   % absent)
    return out


def target_outcome_package_problems(cmd, rr: dict, conf: dict, route: dict, pins: dict) -> list:
    """AT LAUNCH (the module text, corrections of Oct 15, 2026): the command's --out lies inside its role's package
    folder, whose arm_continuation.json names the same fingerprint, code commit, preserved prefix and start MANIFEST
    sha256 as the pins."""
    role = rr.get("role", "repair")
    pkg = ((route.get("packages") or {}).get(role))
    if not _text(pkg):
        return ["the route names no package folder for the %s arm" % role]
    out = []
    dest = flags_of(cmd).get("--out")
    if not _text(dest) or not _under(Path(dest).resolve(), Path(pkg).resolve()):
        out.append("--out %s does not lie inside the %s arm's package %s" % (dest, role, pkg))
    try:
        ac = json.loads((Path(pkg) / "arm_continuation.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return out + ["the %s arm's package configuration %s cannot be read (%s)" % (
            role, Path(pkg) / "arm_continuation.json", exc)]
    for what, theirs, mine in (("fingerprint", ac.get("fingerprint"), conf.get("fingerprint")),
                               ("code commit", (ac.get("code") or {}).get("commit"), pins.get("code_commit")),
                               ("preserved prefix", ac.get("preserved_prefix"), pins.get("preserved_prefix")),
                               ("start manifest sha256", (ac.get("route") or {}).get("manifest_sha256"),
                                pins.get("start_manifest_sha256"))):
        if theirs != mine:
            out.append("the %s arm's package names %s %r, the launch %r" % (role, what, theirs, mine))
    return out


def target_outcome_problems(rr: dict, registry: Path = REGISTRY) -> list:
    """What the target outcome of a repair_run block lacks or contradicts (THE TARGET OUTCOME in the module text), at
    writing and at launch; [] for a block whose experiment names none."""
    to = target_outcome_of(rr)
    if to is None:
        return []
    if not isinstance(to, dict):
        return ["the experiment's target_outcome must be a block {%s}" % ", ".join(TARGET_OUTCOME_FIELDS)]
    exp, role = rr.get("experiment") or {}, rr.get("role", "repair")
    out = []
    for k in TARGET_OUTCOME_FIELDS:
        v = to.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v):
            out.append("the target outcome's %s is missing" % k.replace("_", " "))
    pins = to.get("pins") if isinstance(to.get("pins"), dict) else {}
    for k in TARGET_OUTCOME_PINS:
        v = pins.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v):
            out.append("the target outcome's pin %s is missing" % k.replace("_", " "))
    if rr.get("target_gates"):
        out.append("a run whose experiment names a target outcome names no target gate (mixed targets: target_gates "
                   "reads %s)" % rr["target_gates"])
    if role == "repair" and (rr.get("remeasure") or {}) != {"outcome": to.get("reading_row")}:
        out.append("the repair arm's remeasure is exactly {\"outcome\": the reading row %s} (reads %s)" % (
            to.get("reading_row"), rr.get("remeasure")))
    # ONE USE FOR BOTH ARMS (the corrections of Oct 15, 2026): the control of a target outcome does not repeat either
    if "repeats" in rr and (rr["repeats"] != 1 or isinstance(rr["repeats"], bool)):
        out.append("a target-outcome arm is used once, whatever its role (repeats reads %r)" % (rr["repeats"],))
    route = target_outcome_route(exp.get("id"))
    if route is None:
        out.append("the experiment %s is not an entry of TARGET_OUTCOME_ROUTES: no target outcome is admitted outside "
                   "its named route" % exp.get("id"))
    else:
        auth = pins.get("authorization") if isinstance(pins.get("authorization"), dict) else {}
        for what, theirs, mine in (
                ("arms", route.get("arms"), exp.get("arms")), ("defect", route.get("defect"), to.get("defect")),
                ("reading row", route.get("reading_row"), to.get("reading_row")),
                ("decision", route.get("decision"), auth.get("id")),
                ("decision text sha256", route.get("decision_text_sha256"), auth.get("text_sha256")),
                ("end step", route.get("end_step"), pins.get("end_step")),
                ("wall cap", route.get("wall_cap_s"), pins.get("wall_cap_s"))):
            if theirs != mine:
                out.append("the target outcome's %s reads %r, the route's %r" % (what, mine, theirs))
        for what, path in (("pinned start checkpoint", pins.get("start_checkpoint")),
                           ("block's start checkpoint", rr.get("start_checkpoint"))):
            if _norm(path or "-") != _norm(route.get("checkpoint") or "+"):
                out.append("the %s %s is not the route's %s" % (what, path, route.get("checkpoint")))
        out += route_pin_problems(route, pins)
    start = _step_of(rr.get("start_checkpoint"))
    try:
        if start is None or int(rr.get("max_model_steps") or 0) != int(pins.get("end_step")) - start:
            out.append("max_model_steps %s is not the pinned end step %s minus the start's model step %s" % (
                rr.get("max_model_steps"), pins.get("end_step"), start))
    except (TypeError, ValueError):
        out.append("the pinned end step %r is not a model step" % (pins.get("end_step"),))
    if not _number(pins.get("wall_cap_s")) or not pins.get("wall_cap_s", 0) > 0:
        out.append("the pinned wall cap %r is not a number of seconds above 0" % (pins.get("wall_cap_s"),))
    if _text(to.get("defect")):
        out += _defect_problems(to["defect"])
    row_id = to.get("reading_row")
    row = _registry_rows(registry).get(row_id) if _text(row_id) else None
    if _text(row_id) and row is None:
        out.append("the reading row %s is not in the test registry" % row_id)
    elif row is not None:
        if row.get("status") not in TARGET_OUTCOME_ROW_STATUSES:
            out.append("the reading row %s reads %s, not one of %s" % (row_id, row.get("status"),
                                                                       ", ".join(TARGET_OUTCOME_ROW_STATUSES)))
        named = row.get("defect")
        if not (named == to.get("defect") or (isinstance(named, list) and to.get("defect") in named)):
            out.append("the reading row %s does not name the defect %s" % (row_id, to.get("defect")))
    if not _text(to.get("rule")):
        out.append("the target outcome's rule is not written")
    inputs = to.get("inputs")
    if isinstance(inputs, list):
        for i, x in enumerate(inputs, 1):
            if not isinstance(x, dict) or not _text(x.get("name")) or not _text(x.get("source")):
                out.append("the target outcome's input %d needs its name and source" % i)
            elif x.get("sha256") is not None and _file_sha256(x["source"]) != x["sha256"]:
                out.append("the target outcome's input %s (%s) does not hash to %s" % (x["name"], x["source"],
                                                                                    x["sha256"]))
    if to.get("reader") is not None:
        out += _reader_problems(to.get("reader"))
    if pins.get("authorization") is not None:
        out += _authorization_text_problems(pins.get("authorization"))
    return out


def target_outcome_authority_problems(rr: dict, fingerprint: str, root: Path = PREFLIGHT_DIR) -> list:
    """CONSUMED AUTHORITY (the module text): no block of the preflight folder outside this experiment's two records
    carries the authorizing decision (an experiment block of another id naming it in its target outcome, or any block
    whose decision id is it)."""
    to = target_outcome_of(rr)
    if not isinstance(to, dict):
        return []
    exp = rr.get("experiment") or {}
    did = ((to.get("pins") or {}).get("authorization") or {}).get("id")
    if not _text(did):
        return []
    mine = {_norm(record_path(str(fp), root)) for fp in (exp.get("arms") or {}).values()}
    out = []
    for where, other in _authorized_blocks(root):
        if other is None:
            out.append("%s cannot be read, so the reuse of the decision %s cannot be checked against it" % (where, did))
            continue
        if any(where.endswith("record %s" % Path(p).stem) for p in mine) and other.get("id") == exp.get("id"):
            continue                                    # this experiment's own block, in either arm's record
        o_to = other.get(TARGET_OUTCOME) if isinstance(other.get(TARGET_OUTCOME), dict) else {}
        odid = ((o_to.get("pins") or {}).get("authorization") or {}).get("id")
        if odid == did or ((other.get("decision") or {}).get("id") if isinstance(other.get("decision"), dict)
                           else None) == did:
            out.append("the decision %s already authorized %s: a decision authorizes one experiment" % (did, where))
    return out


def target_outcome_launch_problems(cmd, rr: dict, conf: dict, root: Path, registry: Path) -> list:
    """At launch (THE TARGET OUTCOME in the module text): every block check again, the code hash pin, the exact end
    step, and consumed authority."""
    to = target_outcome_of(rr)
    if to is None:
        return []
    out = list(target_outcome_problems(rr, registry))
    pins = (to.get("pins") if isinstance(to, dict) else None) or {}
    if pins.get("code_hash") != conf.get("code_hash"):
        out.append("the configuration's code hash %s is not the pinned %s" % (conf.get("code_hash"),
                                                                             pins.get("code_hash")))
    try:
        end = int(flags_of(cmd).get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    if end != pins.get("end_step"):
        out.append("--max-steps reads %s; a target-outcome arm runs exactly to the pinned end step %s" % (
            flags_of(cmd).get("--max-steps", "absent"), pins.get("end_step")))
    out += target_outcome_authority_problems(rr, conf["fingerprint"], root)
    route = target_outcome_route(((rr.get("experiment") or {}).get("id")))
    if route is not None:
        out += target_outcome_package_problems(cmd, rr, conf, route, pins)
        if _norm(Path(root).resolve()) == _norm(PREFLIGHT_DIR.resolve()):
            out += execution_problems(route)           # IMPLEMENTATION IS NOT EXECUTION (the module text)
    return out


def _exception_complete(x) -> bool:
    """True for an exception {failure, why_measurable, evidence [..]} with every field written."""
    return isinstance(x, dict) and all(_evidence(x.get(f)) if f == "evidence" else str(x.get(f) or "").strip()
                                       for f in EXCEPTION_FIELDS)


def gate_exception(rr: dict, n: int):
    """The experiment's complete exception for gate `n`, or None."""
    x = (((rr.get("experiment") or {}).get("exceptions") or {}).get(str(n)))
    return x if _exception_complete(x) else None


def arms_problems(rr: dict, conf: dict, root: Path) -> list:
    """For a run of an experiment: this record is the arm its role names, the other arm's record names the same
    experiment in the other role, and the two agree in start checkpoint, length, stop criteria and code, their flags
    differing only in the experiment's differs_only_in."""
    exp = rr.get("experiment")
    if not isinstance(exp, dict) or not isinstance(exp.get("arms"), dict):
        return []
    role = rr.get("role", "repair")
    other_role = "control" if role == "repair" else "repair"
    if exp["arms"].get(role) != conf["fingerprint"]:
        return ["this configuration (%s) is not the %s arm the experiment %s names (%s)" % (
            conf["fingerprint"], role, exp.get("id"), exp["arms"].get(role))]
    other = _load(record_path(str(exp["arms"].get(other_role) or "-"), root))
    if not other:
        return ["the experiment's %s arm %s has no preflight record" % (other_role, exp["arms"].get(other_role))]
    orr = other.get("repair_run") or {}
    if (orr.get("experiment") or {}).get("id") != exp.get("id") and not _instance_id_problem(exp.get("id")):
        # THE FRESH INSTANCE (the module text): the other arm may be a fresh instance of a configuration whose own repair
        # run was used; it is compared exactly as a repair_run block is
        held = _load(instance_path(str(exp["arms"].get(other_role)), str(exp.get("id")), root))
        if isinstance(held, dict) and isinstance(held.get("block"), dict):
            orr = held["block"]
    out = []
    if (orr.get("experiment") or {}).get("id") != exp.get("id") or orr.get("role", "repair") != other_role:
        out.append("the %s arm's record does not name experiment %s in the role %s" % (other_role, exp.get("id"),
                                                                                       other_role))
    if _norm(orr.get("start_checkpoint") or "-") != _norm(rr.get("start_checkpoint") or "+"):
        out.append("the arms start from different checkpoints")
    for k in ("max_model_steps", "stop_criteria"):
        if orr.get(k) != rr.get(k):
            out.append("the arms differ in %s" % k.replace("_", " "))
    if other.get("code_hash") != conf["code_hash"]:
        out.append("the arms run different code")
    # THE TARGET OUTCOME (the module text): bilateral agreement of the two arms' experiment blocks
    oexp = orr.get("experiment") if isinstance(orr.get("experiment"), dict) else {}
    if TARGET_OUTCOME in exp or TARGET_OUTCOME in oexp:
        for k in (TARGET_OUTCOME, "authorization", "authorized_by", "authorized_at", "arms", "differs_only_in"):
            if oexp.get(k) != exp.get(k):
                out.append("the arms' experiment blocks differ in %s (THE TARGET OUTCOME: both arms carry the same "
                           "experiment)" % k)
    allowed = set(exp.get("differs_only_in") or [])
    theirs, mine = other.get("flags") or {}, conf["flags"]
    beyond = sorted(k for k in set(theirs) | set(mine) if theirs.get(k) != mine.get(k) and k not in allowed)
    if beyond:
        out.append("the arms' flags differ beyond %s: %s" % (sorted(allowed), ", ".join(beyond)))
    return out


def _step_of(checkpoint):
    return _run_and_step(checkpoint)[1] if checkpoint else None


def _repair_refusal(cmd, rec: dict, rr: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the record's repair run and every condition of the operator's rule holds."""
    problems = _repair_problems(cmd, rec, rr, conf, root, registry)
    if not problems:
        return None
    return "\n".join(["BLOCKED: repair run of configuration %s (%s): %d condition(s) unmet:" % (
        conf["fingerprint"], record_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


def _repair_problems(cmd, rec: dict, rr: dict, conf: dict, root: Path, registry: Path) -> list:
    """Every condition of the operator's repair-run rule this launch of the block `rr` leaves unmet ([]: none)."""
    flags = flags_of(cmd)
    start_step = _step_of(rr.get("start_checkpoint"))
    try:
        end = int(flags.get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    # the launch's own report rate and run length reach the first-reading check (the round-two review's R1): a command
    # without --report-every runs at the trainer's default, and a --max-steps under the declared bound ends it sooner
    every = flags.get("--report-every")
    problems = repair_block_problems(
        rr, registry, report_every=int(every) if isinstance(every, str) and every.isdigit() else TRAINER_REPORT_EVERY,
        run_steps=end - start_step if start_step is not None and end > start_step else None)
    targets = set(rr.get("target_gates") or [])
    gates = {g.get("n"): g for g in rec.get("gates") or [] if isinstance(g, dict)}
    for n, name, _q in GATES:
        g = dict(gates.get(n) or {"n": n, "status": "OPEN"}, name=name)
        if n in targets:
            if g.get("status") != REPAIR_TARGET:
                problems.append("gate %d %s is the run's stated purpose and must read FAIL - REPAIR TARGET with its "
                                "reason and evidence (reads %s)" % (n, name, g.get("status") or "OPEN"))
            elif not str(g.get("reason") or "").strip() or not [e for e in g.get("evidence") or [] if str(e).strip()]:
                problems.append("gate %d %s: FAIL - REPAIR TARGET needs its reason and evidence" % (n, name))
        else:
            why = gate_problem(g)
            if why and g.get("status") == "FAIL" and gate_exception(rr, n):
                continue            # FAIL, with an exception of this run's experiment (THE EXPERIMENT)
            if why:
                problems.append(why + " (a gate the repair run does not target must read PASS or justified N/A, or "
                                      "FAIL with an exception of its experiment)")
    problems += arms_problems(rr, conf, root)
    problems += target_outcome_launch_problems(cmd, rr, conf, root, registry)      # THE TARGET OUTCOME
    bound = (start_step or 0) + int(rr.get("max_model_steps") or 0)
    if start_step is None:
        problems.append("the start checkpoint's model step cannot be read from its name (%s)" % rr.get("start_checkpoint"))
    elif not (start_step < end <= bound):
        problems.append("the launch is not bounded as declared: --max-steps reads %s, the run may end at most at model "
                        "step %d (start %d + %s)" % (flags.get("--max-steps", "absent"), bound, start_step,
                                                     rr.get("max_model_steps")))
    return problems


# ------------------------------------------------------------------ THE CONTINUATION (the module text)
def continuation_available(cb) -> bool:
    """True while a record's continuation block is written and unused (a marked launch, or an attempt whose folder trained
    past its start, uses it: repair_uses)."""
    return isinstance(cb, dict) and bool(cb) and not repair_uses(cb)


def stage_done(run) -> dict:
    """<run>/stage_done.json, which the trainer writes as it ends ({ok, steps, finished_at, ...}), or {}."""
    try:
        done = json.loads((Path(str(run)) / "stage_done.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return done if isinstance(done, dict) else {}


def repair_run_end(rr: dict):
    """(the repair run's folder, its end checkpoint, None) once the record's repair run launched and completed; else
    (the folder or None, None, the text naming why not). Its folder is the marked run, or its first attempt that trained
    past the start (repair_launch); its end is the --max-steps its trainer ran with (the folder's args.json max_steps);
    it completed when the folder's stage_done.json (written by the trainer as it ends) reads ok at that model step and
    names the time the trainer ended (finished_at, against which the target gates' judgement is read: finding F6)."""
    _at, _at_s, run = repair_launch(rr or {})
    if not run:
        return None, None, "the record's repair run never launched; a continuation continues a completed repair run"
    run = Path(str(run))
    try:
        end = int(json.loads((run / "args.json").read_text(encoding="utf-8-sig")).get("max_steps") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        end = 0
    if end <= 0:
        return run, None, ("the repair run's end cannot be read: %s names no max_steps above 0" % (run / "args.json"))
    done = stage_done(run)
    try:
        done_step = int(done.get("steps")) if done else None
    except (TypeError, ValueError):
        done_step = None
    if not done or done.get("ok") is not True or done_step != end:
        return run, None, ("the repair run did not complete: %s reads %s, where a completed run reads ok at its end, "
                           "model step %d" % (run / "stage_done.json", "ok %s at model step %s" % (
                               done.get("ok"), done.get("steps")) if done else "nothing", end))
    if not _number(done.get("finished_at")):
        return run, None, ("the repair run's end time cannot be read: %s names no finished_at, the time its trainer "
                           "ended, after which its target gates are judged again" % (run / "stage_done.json"))
    return run, run / "checkpoints" / ("ckpt_step_%d" % end), None


def run_flag_problems(cb: dict, rec: dict) -> list:
    """What the continuation's run_flags lack (the review of Oct 11, 2026, finding F5): a block {flag: value} of run flags
    (RUN_FLAGS, outside the fingerprint) other than those the relaunch script chooses and the bound, each value a text;
    when the configuration runs the latent body (--latent-body-hierarchy other than off), the state it resumes
    (--latent-body-resume, an existing file) and the folder it writes (--latent-body-output, a folder that does not yet
    exist, so no other run's saved work is written over). The launch must carry each exactly (_continuation_refusal)."""
    flags = cb.get("run_flags")
    if flags is None or flags == {}:
        return []                                   # reported as missing (CONTINUATION_FIELDS)
    if not isinstance(flags, dict):
        return ["the continuation's run flags must be a block {flag: value} (reads %s)" % type(flags).__name__]
    out = []
    for k, v in flags.items():
        if k not in RUN_FLAGS or k in CONTINUATION_CHOSEN_FLAGS:
            out.append("run flag %s: run_flags names run flags outside the fingerprint, never a configuration flag and "
                       "never %s, which the relaunch script chooses or the bound sets" % (
                           k, ", ".join(CONTINUATION_CHOSEN_FLAGS)))
        if not isinstance(v, str) or not v.strip():
            out.append("run flag %s needs its value as text (reads %r)" % (k, v))
    latent = str(((rec or {}).get("flags") or {}).get("--latent-body-hierarchy") or "off")
    if latent != "off":
        for k in LATENT_RUN_FLAGS:
            if not str(flags.get(k) or "").strip():
                out.append("the configuration runs the latent body (--latent-body-hierarchy %s): run_flags names %s (%s)"
                           % (latent, k, "the state it resumes" if k.endswith("resume") else "the folder it writes"))
    resume = flags.get("--latent-body-resume")
    if isinstance(resume, str) and resume.strip() and not Path(resume).is_file():
        out.append("the latent body state %s the continuation resumes is not a file" % resume)
    output = flags.get("--latent-body-output")
    if isinstance(output, str) and output.strip() and Path(output).exists():
        out.append("the latent body output folder %s exists already; a continuation writes a new folder, never over "
                   "saved work" % output)
    return out


def _as_continuation(texts):
    """The repair run's stop-line texts (stop_field_problems, first_reading_problems) said of the continuation."""
    return [t.replace("repair run", "continuation") for t in texts]


def continuation_block_problems(cb, rec: dict, registry: Path = REGISTRY, report_every=None, run_steps=None) -> list:
    """What a continuation block lacks (THE CONTINUATION in the module text), against its record `rec`: every field
    named; the record's repair run (role repair) launched and completed, and the start exactly its end checkpoint; the
    stop criteria checked as a repair run's; the duration bounded; the stop lines' and the completion jobs' rows in the
    test registry reading written or queued, the stop lines' row not the repair run's own; the run flags it relies on
    (with the latent body, its resume file existing and its output folder not yet existing); the exceptions' form, none
    naming a status other than FAIL. `report_every` and `run_steps` are the launch command's report rate and run length,
    when a launch is judged (first_reading_problems)."""
    if not isinstance(cb, dict):
        return ["the continuation must be a block {purpose, start_checkpoint, stop_criteria, authorization, ...}"]
    out = []
    for k in CONTINUATION_FIELDS:
        v = cb.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v) or (isinstance(v, str) and not v.strip()):
            out.append("the continuation's %s is missing" % k.replace("_", " "))
    if cb.get("role", CONTINUATION) != CONTINUATION:
        out.append("the continuation's role reads %s (a continuation is its own route, role continuation)" % cb["role"])
    if "repeats" in cb and (cb["repeats"] != 1 or isinstance(cb["repeats"], bool)):
        out.append("a continuation is used once (repeats reads %r)" % (cb["repeats"],))
    for k in ("target_gates", "experiment"):
        if cb.get(k):
            out.append("a continuation targets no gate and is no experiment's arm (%s reads %s)" % (k, cb[k]))
    rr = rec.get("repair_run") if isinstance(rec, dict) else None
    if not isinstance(rr, dict) or not rr:
        out.append("the record has no repair run; a continuation continues a completed repair run")
    elif rr.get("role", "repair") != "repair":
        out.append("a continuation continues a repair run (role repair); this record's run is a %s run" % rr.get("role"))
    else:
        run, end_checkpoint, why = repair_run_end(rr)
        if why:
            out.append(why)
        elif cb.get("start_checkpoint") and _norm(cb["start_checkpoint"]) != _norm(end_checkpoint):
            # the review of Oct 11, 2026, finding F4: continues() accepted a checkpoint of any run resumed from the end;
            # the plan names exactly the model the completed repair run produced
            out.append("the continuation's start checkpoint %s is not the repair run's end checkpoint %s (a continuation "
                       "resumes exactly the model the completed repair run %s produced)" % (
                           cb["start_checkpoint"], end_checkpoint, run))
    if cb.get("stop_criteria"):
        out += stop_criteria_problems(cb["stop_criteria"])
        out += _as_continuation(stop_field_problems(cb))
        out += _as_continuation(first_reading_problems(cb, report_every, run_steps))
    try:
        if int(cb.get("max_model_steps") or 0) <= 0 or isinstance(cb.get("max_model_steps"), bool):
            out.append("the continuation's duration is not bounded (max_model_steps must be above 0)")
    except (TypeError, ValueError):
        out.append("max_model_steps is not a number")
    rows = _registry_rows(registry)
    repair_row = rr.get("stop_lines_row") if isinstance(rr, dict) else None
    if cb.get("stop_lines_row") and repair_row and cb["stop_lines_row"] == repair_row:
        # finding F1: the plan of Oct 11, 2026: the old stop exception "does not carry over automatically"
        out.append("the continuation's stop lines' row %s is the repair run's own; a continuation declares its own stop "
                   "lines in its own row" % repair_row)
    for key, label in (("stop_lines_row", "the stop lines' row"), ("completion_jobs_row", "the completion jobs' row")):
        row = cb.get(key)
        if row:
            r = rows.get(row)
            if r is None:
                out.append("%s %s is not in the test registry (it is declared there before launch)" % (label, row))
            elif r.get("status") not in CONTINUATION_ROW_STATUSES:
                out.append("%s %s reads status %s; a continuation's row is declared for it and not yet run (status %s)"
                           % (label, row, r.get("status"), " or ".join(CONTINUATION_ROW_STATUSES)))
    out += run_flag_problems(cb, rec)
    exceptions = cb.get("exceptions", {})
    if not isinstance(exceptions, dict):
        return out + ["the continuation's exceptions must be a block {gate: {failure, why_measurable, evidence}}"]
    for key, x in exceptions.items():
        try:
            n = int(key)
        except (TypeError, ValueError):
            out.append("exception %r is not a gate number" % (key,))
            continue
        if n not in [g[0] for g in GATES]:
            out.append("exception %d is not one of the seven gates" % n)
            continue
        if not isinstance(x, dict):
            out.append("the exception for gate %d must be {failure, why_measurable, evidence}" % n)
            continue
        for f in EXCEPTION_FIELDS:
            if not (_evidence(x.get(f)) if f == "evidence" else str(x.get(f) or "").strip()):
                out.append("the exception for gate %d lacks its %s" % (n, f.replace("_", " ")))
        if "status" in x and x.get("status") != "FAIL":
            out.append("the exception for gate %d reads status %s: an exception leaves its gate FAIL, never PASS or N/A"
                       % (n, x.get("status")))
    return out


def continuation_gate_problems(rec: dict, cb: dict) -> list:
    """Each gate that blocks the continuation: a gate that does not read PASS or justified N/A must read FAIL with the
    block's complete exception; an exception naming a gate that reads PASS or N/A is refused (an exception is for a FAIL
    gate and leaves it FAIL); a gate never judged (OPEN), missing from the record, or still FAIL - REPAIR TARGET is refused
    whatever the exception; an exception excuses the failure judged before the block was written (written_at_s), so a
    FAIL gate judged again since is refused until the block is written again (the review of Oct 11, 2026, finding F3);
    each target gate of the repair run must have been judged again after that run ended (its stage_done.json
    finished_at; finding F6: the operator's rule says "after the run", not after its launch)."""
    rr = rec.get("repair_run") or {}
    run, _end, _why = repair_run_end(rr) if rr else (None, None, None)
    done = stage_done(run) if run else {}
    finished_s = done.get("finished_at") if _number(done.get("finished_at")) else None
    written_s = cb.get("written_at_s")
    targets = set(rr.get("target_gates") or [])
    exceptions = cb.get("exceptions") if isinstance(cb.get("exceptions"), dict) else {}
    gates = {g.get("n"): g for g in rec.get("gates") or [] if isinstance(g, dict)}
    out = []
    for n, name, _q in GATES:
        g = dict(gates.get(n) or {"n": n, "status": "OPEN"}, name=name)
        status = g.get("status") or "OPEN"
        x = exceptions.get(str(n))
        if status == REPAIR_TARGET:
            out.append("gate %d %s reads FAIL - REPAIR TARGET, the repair run's target: judge it again after that run "
                       "(PASS, N/A or FAIL); a continuation targets no gate" % (n, name))
            continue
        if n in targets and finished_s is not None and float(g.get("checked_at_s") or 0) <= float(finished_s):
            out.append("gate %d %s was a target of the repair run, which ended %s; it must be judged again after that "
                       "run's end before a continuation (last judged %s)" % (
                           n, name, done.get("finished_at_human") or finished_s, g.get("checked_at") or "never"))
        why = gate_problem(g)
        if why is None:
            if x is not None:
                out.append("gate %d %s reads %s and the continuation names an exception for it: an exception is for a "
                           "gate that reads FAIL, and leaves it FAIL" % (n, name, status))
            continue
        if status == "FAIL":
            if _exception_complete(x) and x.get("status", "FAIL") == "FAIL":
                if not _number(written_s):
                    out.append("gate %d %s: its exception cannot be tied to the failure it excuses: the block names no "
                               "written_at_s (write it with training_preflight continuation)" % (n, name))
                elif float(g.get("checked_at_s") or 0) > float(written_s):
                    out.append("gate %d %s was judged again since its exception was written (judged %s, the block "
                               "written %s): write the block again so its exception names the failure it excuses" % (
                                   n, name, g.get("checked_at") or g.get("checked_at_s"),
                                   cb.get("written_at") or written_s))
                continue                    # FAIL, with the continuation's own exception
            out.append(why + " (a continuation launches with a FAIL gate only with its exception {failure, "
                             "why_measurable, evidence})")
        else:
            out.append(why + " (an exception does not stand in for a gate never judged, or for a PASS or N/A without "
                             "its reason and evidence)")
    return out


def _continuation_refusal(cmd, rec: dict, cb: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the record's continuation and every condition holds (THE CONTINUATION)."""
    flags = flags_of(cmd)
    start_step = _step_of(cb.get("start_checkpoint"))
    try:
        end = int(flags.get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    every = flags.get("--report-every")
    problems = continuation_block_problems(
        cb, rec, registry,
        report_every=int(every) if isinstance(every, str) and every.isdigit() else TRAINER_REPORT_EVERY,
        run_steps=end - start_step if start_step is not None and end > start_step else None)
    problems += continuation_gate_problems(rec, cb)
    resume = flags.get("--resume")
    if not isinstance(resume, str) or _norm(resume) != _norm(cb.get("start_checkpoint") or ""):
        problems.append("the launch resumes %s, not the continuation's start %s" % (resume, cb.get("start_checkpoint")))
    run_flags = cb.get("run_flags") if isinstance(cb.get("run_flags"), dict) else {}
    for k, v in sorted(run_flags.items()):
        if flags.get(k) != v:                       # finding F5: compared exactly, as text
            problems.append("the launch's %s reads %s; the continuation names %s (run_flags, compared exactly)" % (
                k, flags.get(k, "absent"), v))
    try:
        bound = (start_step or 0) + int(cb.get("max_model_steps") or 0)
    except (TypeError, ValueError):
        bound = start_step or 0
    if start_step is None:
        problems.append("the start checkpoint's model step cannot be read from its name (%s)" % cb.get("start_checkpoint"))
    elif not (start_step < end <= bound):
        problems.append("the launch is not bounded as declared: --max-steps reads %s, the continuation may end at most "
                        "at model step %d (start %d + %s)" % (flags.get("--max-steps", "absent"), bound, start_step,
                                                              cb.get("max_model_steps")))
    if not problems:
        return None
    return "\n".join(["BLOCKED: continuation of configuration %s (%s): %d condition(s) unmet:" % (
        conf["fingerprint"], record_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


# ------------------------------------------------------------------ THE EXTENSION (the module text)
def extension_path(fingerprint: str, root: Path = PREFLIGHT_DIR) -> Path:
    """The extension file of configuration `fingerprint` (one per configuration)."""
    return Path(root) / EXTENSIONS_DIR / ("%s.json" % fingerprint)


def extension_of(fingerprint: str, root: Path = PREFLIGHT_DIR) -> dict:
    """The extension file's content ({} when the configuration has none)."""
    held = _load(extension_path(fingerprint, root))
    return held if isinstance(held, dict) else {}


def extension_available(xb) -> bool:
    """True while an extension block is written and unused (counted as a continuation's uses are: repair_uses)."""
    return isinstance(xb, dict) and bool(xb) and not repair_uses(xb)


def continuation_run_end(cb: dict):
    """(the continuation's run folder, the end checkpoint it stopped at, the model step N it stopped at, the end it was
    launched for, None) when the record's continuation was used and its run STOPPED EARLY (the module text); else
    (..., the text naming why not)."""
    _at, _at_s, run = repair_launch(cb or {})
    if not run:
        return None, None, None, None, ("the record's continuation was never used (no marked launch, no attempt that "
                                         "trained past its start); an extension continues a used continuation")
    run = Path(str(run))
    start = _step_of(cb.get("start_checkpoint"))
    try:
        bound = int(start or 0) + int(cb.get("max_model_steps") or 0)
    except (TypeError, ValueError):
        bound = None
    try:
        end = int(json.loads((run / "args.json").read_text(encoding="utf-8-sig")).get("max_steps") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        end = 0
    if start is None or bound is None or not (start < end <= bound):
        return run, None, None, None, ("the continuation's run %s names no end within its bound: args.json max_steps "
                                       "reads %s, the continuation's start %s and bound %s" % (run, end or "nothing",
                                                                                                start, bound))
    done = stage_done(run)
    try:
        n = int(done.get("steps")) if done else None
    except (TypeError, ValueError):
        n = None
    if not done or done.get("ok") is not True or n is None:
        return run, None, None, end, ("the continuation's run did not end with its own save: %s reads %s" % (
            run / "stage_done.json", "ok %s at model step %s" % (done.get("ok"), done.get("steps")) if done
            else "nothing"))
    if n >= end:
        return run, None, n, end, ("the continuation's run reached its end, model step %d (stage_done.json at %d): it "
                                   "has nothing left to extend" % (end, n))
    if n <= start:
        return run, None, n, end, ("the continuation's run ended at model step %d, not after its start %d: it trained "
                                   "nothing to continue" % (n, start))
    if not _number(done.get("finished_at")):
        return run, None, n, end, ("the continuation's run's end time cannot be read: %s names no finished_at" % (
            run / "stage_done.json"))
    ck = run / "checkpoints" / ("ckpt_step_%d" % n)
    if not (ck / "meta.json").is_file():
        return run, None, n, end, "the continuation's end checkpoint %s is not complete (no meta.json)" % ck
    return run, ck, n, end, None


def extension_route_problems(fingerprint, run=None, stopped_at=None, end=None) -> list:
    """Why an extension is not the one the route admits (EXTENSION_ROUTE; the module text, finding B-1), or []: the
    configuration's fingerprint, and, when given, the continuation's run folder, the model step it stopped at and the end
    it was launched for."""
    pin = EXTENSION_ROUTE
    if str(fingerprint or "") != pin["fingerprint"]:
        return ["the extension route admits one extension only, of configuration %s; configuration %s has none (an "
                "extension of another configuration is the operator's decision and a change to this code)" % (
                    pin["fingerprint"], fingerprint)]
    out = []
    if run is not None and _norm(run) != _norm(pin["continuation_run"]):
        out.append("the extension route admits only the extension of the continuation run %s; this record's "
                   "continuation ran in %s" % (pin["continuation_run"], run))
    if stopped_at is not None and stopped_at != pin["stopped_at"]:
        out.append("the extension route admits only the extension from model step %d, where the continuation run %s "
                   "stopped; this continuation stopped at %s" % (pin["stopped_at"], pin["continuation_run"], stopped_at))
    if end is not None and end != pin["end"]:
        out.append("the extension route admits only the extension to model step %d, the end the continuation run %s was "
                   "launched for; this continuation was launched to end at %s" % (pin["end"], pin["continuation_run"],
                                                                                 end))
    return out


def _as_extension(texts):
    """The repair run's stop-line texts (stop_field_problems, first_reading_problems) said of the extension."""
    return [t.replace("repair run", "extension") for t in texts]


def extension_block_problems(xb, rec: dict, registry: Path = REGISTRY, report_every=None, run_steps=None) -> list:
    """What an extension block lacks (THE EXTENSION in the module text), against its record `rec`: every field a
    continuation names; the record's continuation used and its run stopped early, and the start exactly the end
    checkpoint it saved; the stop criteria checked as a continuation's; the duration bounded by the continuation's own
    end; the stop lines' row its own and the completion jobs' row not the continuation's, both reading written or
    queued; the run flags; the exceptions' form."""
    if not isinstance(xb, dict):
        return ["the extension must be a block {purpose, start_checkpoint, stop_criteria, authorization, ...}"]
    out = []
    for k in CONTINUATION_FIELDS:
        v = xb.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v) or (isinstance(v, str) and not v.strip()):
            out.append("the extension's %s is missing" % k.replace("_", " "))
    for k in EXTENSION_TEXT_FIELDS:                 # finding B-4: a value that is not text passed the check above
        if xb.get(k) is not None and not isinstance(xb[k], str):
            out.append("the extension's %s must be text (reads %s %r)" % (k.replace("_", " "), type(xb[k]).__name__,
                                                                         xb[k]))
    if xb.get("role", EXTENSION) != EXTENSION:
        out.append("the extension's role reads %s (an extension is its own route, role extension)" % xb["role"])
    if "repeats" in xb and (xb["repeats"] != 1 or isinstance(xb["repeats"], bool)):
        out.append("an extension is used once (repeats reads %r)" % (xb["repeats"],))
    for k in ("target_gates", "experiment"):
        if xb.get(k):
            out.append("an extension targets no gate and is no experiment's arm (%s reads %s)" % (k, xb[k]))
    rec = rec if isinstance(rec, dict) else {}
    cb = rec.get(CONTINUATION) if isinstance(rec.get(CONTINUATION), dict) else {}
    rr = rec.get("repair_run") if isinstance(rec.get("repair_run"), dict) else {}
    run = n = end = None
    if not cb:
        out.append("the record has no continuation; an extension continues a continuation that stopped early")
    else:
        run, ck, n, end, why = continuation_run_end(cb)
        if why:
            out.append(why)
        elif xb.get("start_checkpoint") and _norm(xb["start_checkpoint"]) != _norm(ck):
            out.append("the extension's start checkpoint %s is not the end checkpoint %s the continuation's run %s "
                       "stopped at" % (xb["start_checkpoint"], ck, run))
    # THE ONE EXTENSION (finding B-1): this record, its continuation's run, the step it stopped at and its end
    out += extension_route_problems(rec.get("fingerprint"), run, n, end)
    # THE OPERATOR'S AUTHORIZATION OF THIS EXTENSION (finding B-4): its words and its time are not the continuation's
    for k, what in (("authorization", "words"), ("authorized_at", "time")):
        mine, theirs = xb.get(k), cb.get(k)
        if isinstance(mine, str) and isinstance(theirs, str) and mine.strip() \
                and " ".join(mine.split()) == " ".join(theirs.split()):
            out.append("the extension's %s repeats the continuation's %s: the operator's authorization of this "
                       "extension is given for it, in its own words at its own time" % (k.replace("_", " "), what))
    if xb.get("stop_criteria"):
        out += stop_criteria_problems(xb["stop_criteria"])
        out += _as_extension(stop_field_problems(xb))
        out += _as_extension(first_reading_problems(xb, report_every, run_steps))
    try:
        steps = int(xb.get("max_model_steps") or 0)
        if steps <= 0 or isinstance(xb.get("max_model_steps"), bool):
            out.append("the extension's duration is not bounded (max_model_steps must be above 0)")
        elif n is not None and end is not None and n + steps > end:
            out.append("the extension's max_model_steps %d passes the continuation's end: it stopped at model step %d "
                       "and was launched to end at %d, so at most %d model steps remain" % (steps, n, end, end - n))
    except (TypeError, ValueError):
        out.append("max_model_steps is not a number")
    rows = _registry_rows(registry)
    taken = {r: what for r, what in ((rr.get("stop_lines_row"), "the repair run's"),
                                     (cb.get("stop_lines_row"), "the continuation's")) if r}
    if xb.get("stop_lines_row") in taken:
        out.append("the extension's stop lines' row %s is %s own; an extension declares its stop lines in its own row" % (
            xb["stop_lines_row"], taken[xb["stop_lines_row"]]))
    if xb.get("completion_jobs_row") and xb["completion_jobs_row"] == cb.get("completion_jobs_row"):
        out.append("the extension's completion jobs' row %s is the continuation's own; the extension's completion jobs "
                   "are declared in their own row" % xb["completion_jobs_row"])
    for key, label in (("stop_lines_row", "the stop lines' row"), ("completion_jobs_row", "the completion jobs' row")):
        row = xb.get(key)
        if row:
            r = rows.get(row)
            if r is None:
                out.append("%s %s is not in the test registry (it is declared there before launch)" % (label, row))
            elif r.get("status") not in CONTINUATION_ROW_STATUSES:
                out.append("%s %s reads status %s; an extension's row is declared for it and not yet run (status %s)"
                           % (label, row, r.get("status"), " or ".join(CONTINUATION_ROW_STATUSES)))
    out += run_flag_problems(xb, rec)
    # THE LATENT BODY STATE (finding B-3): exactly the state the continuation's run saved, latest.pt in its
    # --latent-body-output, when the configuration runs the latent body or the block names a state to resume
    xflags = xb.get("run_flags") if isinstance(xb.get("run_flags"), dict) else {}
    cflags = cb.get("run_flags") if isinstance(cb.get("run_flags"), dict) else {}
    latent = str((rec.get("flags") or {}).get("--latent-body-hierarchy") or "off")
    resume = xflags.get("--latent-body-resume")
    named = isinstance(resume, str) and bool(resume.strip())
    if latent != "off" or named:
        saved = cflags.get("--latent-body-output")
        if not (isinstance(saved, str) and saved.strip()):
            out.append("the continuation names no --latent-body-output, so the latent body state the extension resumes "
                       "cannot be tied to the state the continuation's run saved")
        elif named and _norm(resume) != _norm(Path(saved) / LATENT_STATE_FILE):
            out.append("the extension resumes the latent body state %s; the continuation's run saved its state as %s "
                       "(its --latent-body-output's %s), which an extension resumes exactly" % (
                           resume, Path(saved) / LATENT_STATE_FILE, LATENT_STATE_FILE))
    exceptions = xb.get("exceptions", {})
    if not isinstance(exceptions, dict):
        return out + ["the extension's exceptions must be a block {gate: {failure, why_measurable, evidence}}"]
    for key, x in exceptions.items():
        try:
            g = int(key)
        except (TypeError, ValueError):
            out.append("exception %r is not a gate number" % (key,))
            continue
        if g not in [x_[0] for x_ in GATES]:
            out.append("exception %d is not one of the seven gates" % g)
            continue
        if not isinstance(x, dict):
            out.append("the exception for gate %d must be {failure, why_measurable, evidence}" % g)
            continue
        for f in EXCEPTION_FIELDS:
            if not (_evidence(x.get(f)) if f == "evidence" else str(x.get(f) or "").strip()):
                out.append("the exception for gate %d lacks its %s" % (g, f.replace("_", " ")))
        if "status" in x and x.get("status") != "FAIL":
            out.append("the exception for gate %d reads status %s: an exception leaves its gate FAIL, never PASS or N/A"
                       % (g, x.get("status")))
    return out


def _extension_refusal(cmd, rec: dict, xb: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the configuration's extension and every condition holds (THE EXTENSION)."""
    flags = flags_of(cmd)
    start_step = _step_of(xb.get("start_checkpoint"))
    try:
        end = int(flags.get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    every = flags.get("--report-every")
    problems = extension_block_problems(
        xb, rec, registry,
        report_every=int(every) if isinstance(every, str) and every.isdigit() else TRAINER_REPORT_EVERY,
        run_steps=end - start_step if start_step is not None and end > start_step else None)
    problems += [p.replace("a continuation", "an extension") for p in continuation_gate_problems(rec, xb)]
    resume = flags.get("--resume")
    if not isinstance(resume, str) or _norm(resume) != _norm(xb.get("start_checkpoint") or ""):
        problems.append("the launch resumes %s, not the extension's start %s" % (resume, xb.get("start_checkpoint")))
    run_flags = xb.get("run_flags") if isinstance(xb.get("run_flags"), dict) else {}
    for k, v in sorted(run_flags.items()):
        if flags.get(k) != v:
            problems.append("the launch's %s reads %s; the extension names %s (run_flags, compared exactly)" % (
                k, flags.get(k, "absent"), v))
    try:
        bound = (start_step or 0) + int(xb.get("max_model_steps") or 0)
    except (TypeError, ValueError):
        bound = start_step or 0
    if start_step is None:
        problems.append("the start checkpoint's model step cannot be read from its name (%s)" % xb.get("start_checkpoint"))
    elif not (start_step < end <= bound):
        problems.append("the launch is not bounded as declared: --max-steps reads %s, the extension may end at most at "
                        "model step %d (start %d + %s)" % (flags.get("--max-steps", "absent"), bound, start_step,
                                                           xb.get("max_model_steps")))
    if not problems:
        return None
    return "\n".join(["BLOCKED: extension of configuration %s (%s): %d condition(s) unmet:" % (
        conf["fingerprint"], extension_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


def set_extension(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY):
    """Write the configuration's extension (THE EXTENSION in the module text) in extensions/<fingerprint>.json after
    checking it (extension_block_problems, continuation_gate_problems). The record is read and never written. Refused
    once the extension was used (a marked launch, or an attempt that trained past its start); an unused one may be
    written again, its recorded attempts carried (as a continuation's). written_at_s ties each exception to the failure
    judged before it."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    if not str(by or "").strip():
        raise PreflightError("an extension needs who writes it (--by)")
    target = extension_path(fingerprint, root)
    old = extension_of(fingerprint, root).get("block") or {}
    if repair_uses(old):
        at, _at_s, run = repair_launch(old)
        raise PreflightError("the extension of configuration %s was used (%s, run %s); an extension is used once and a "
                             "used one is not written again" % (fingerprint, at or "an attempt trained past its start",
                                                                run))
    block = dict(block or {})
    if isinstance(block.get("exceptions"), dict):
        block["exceptions"] = {str(k): v for k, v in block["exceptions"].items()}
    problems = extension_block_problems(block, rec, registry)
    for k in ("attempts", "launches"):              # only the launch script records these (note_attempt, mark_launched)
        block.pop(k, None)
    carried = [a for a in old.get("attempts") or [] if isinstance(a, dict)]
    if carried:                                     # attempts that never trained, kept: one that trains later is the use
        block["attempts"] = carried
    block.update(role=EXTENSION, written_at=clock(), written_at_s=time.time(), written_by=str(by), launched_at=None,
                 launched_at_s=None, run=None)
    problems += [p.replace("a continuation", "an extension") for p in continuation_gate_problems(rec, block)]
    if problems:
        raise PreflightError("; ".join(problems))
    cb = rec.get(CONTINUATION) or {}
    _save(target, dict(fingerprint=fingerprint, record=str(path), record_sha256=_sha(path.read_bytes()),
                       continuation_run=str(repair_launch(cb)[2]), written_at=block["written_at"], written_by=str(by),
                       block=block))
    return block


# ------------------------------------------------------------------ THE FRESH INSTANCE (the module text)
def _instance_id_problem(experiment_id):
    """Why an experiment's id cannot name a fresh instance's file, or None."""
    text = str(experiment_id or "")
    if not text or text.startswith(".") or not set(text) <= INSTANCE_ID_CHARS:
        return ("the experiment's id names the fresh instance's file: letters, digits, '_', '-' or '.', not starting "
                "with '.' (reads %r)" % text)
    return None


def instance_path(fingerprint: str, experiment_id: str, root: Path = PREFLIGHT_DIR) -> Path:
    """The fresh instance file of configuration `fingerprint` in experiment `experiment_id`."""
    return Path(root) / INSTANCES_DIR / str(fingerprint) / ("%s.json" % experiment_id)


def instances(fingerprint: str, root: Path = PREFLIGHT_DIR) -> dict:
    """{experiment id: fresh instance file content} of a configuration; {} when it has none."""
    folder = Path(root) / INSTANCES_DIR / str(fingerprint)
    out = {}
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            held = _load(p)
            if isinstance(held, dict):
                out[p.stem] = held
    return out


def instance_problems(rec: dict, held: dict, experiment_id: str, registry: Path = REGISTRY) -> list:
    """What keeps a fresh instance (THE FRESH INSTANCE in the module text) from being one: its file names this record and
    experiment; the record's own repair run was used and is of another experiment; the block is a control run, used once;
    its start is a model state the record names; its stop lines are in a row of their own that reads written or
    queued. Checked when it is written and at its launch."""
    out = []
    held = held if isinstance(held, dict) else {}
    b = held.get("block") if isinstance(held.get("block"), dict) else {}
    exp = b.get("experiment") if isinstance(b.get("experiment"), dict) else {}
    bad = _instance_id_problem(experiment_id)
    if bad:
        out.append(bad)
    if held.get("fingerprint") != rec.get("fingerprint"):
        out.append("the fresh instance names configuration %s, not this record's %s" % (held.get("fingerprint"),
                                                                                          rec.get("fingerprint")))
    if held.get("experiment_id") != experiment_id or exp.get("id") != experiment_id:
        out.append("the fresh instance's file, its experiment_id and its block's experiment must name the same "
                   "experiment (%s, %s, %s)" % (experiment_id, held.get("experiment_id"), exp.get("id")))
    if b.get("role") != "control":
        out.append("a fresh instance is a control run, role control (reads %s); a repair is written in a record of "
                   "its own" % b.get("role"))
    if "repeats" in b and (b.get("repeats") != 1 or isinstance(b.get("repeats"), bool)):
        out.append("a fresh instance is used once (repeats reads %r)" % (b.get("repeats"),))
    rr = rec.get("repair_run") if isinstance(rec.get("repair_run"), dict) else {}
    if not repair_launch(rr)[0]:
        out.append("the record's own repair run was not used; a control of this configuration is written there "
                   "(training_preflight repair), not as a fresh instance")
    elif (rr.get("experiment") or {}).get("id") == experiment_id:
        out.append("experiment %s is the record's used repair run's; a fresh instance belongs to a new experiment"
                   % experiment_id)
    start = b.get("start_checkpoint")
    if start and not any(isinstance(s, dict) and _norm(s.get("checkpoint") or "") == _norm(start)
                         for s in rec.get("states") or []):
        out.append("the fresh instance's start %s is not a model state the record names (its gates were judged for "
                   "%s)" % (start, "; ".join(str(s.get("checkpoint")) for s in rec.get("states") or []
                                             if isinstance(s, dict)) or "none"))
    row = b.get("stop_lines_row")
    if row and row == rr.get("stop_lines_row"):
        out.append("the fresh instance's stop lines' row %s is the used repair run's; it declares its stop lines in a "
                   "row of its own" % row)
    elif row:
        r = _registry_rows(registry).get(row)
        if r is not None and r.get("status") not in CONTINUATION_ROW_STATUSES:
            out.append("the fresh instance's stop lines' row %s reads status %s; it is declared for this run and not "
                       "yet run (status %s)" % (row, r.get("status"), " or ".join(CONTINUATION_ROW_STATUSES)))
    return out


def instance_gate_problems(rec: dict, b: dict) -> list:
    """Each excepted FAIL gate judged again after the fresh instance was written (written_at_s): its exception excuses
    only the failure judged before (as a continuation's, the review of Oct 11, 2026, finding F3)."""
    written_s = b.get("written_at_s")
    exceptions = ((b.get("experiment") or {}).get("exceptions") or {})
    out = []
    for g in rec.get("gates") or []:
        if not isinstance(g, dict) or g.get("status") != "FAIL" or str(g.get("n")) not in exceptions:
            continue
        if not _number(written_s):
            out.append("gate %s %s: its exception cannot be tied to the failure it excuses: the fresh instance names no "
                       "written_at_s (write it with training_preflight instance)" % (g.get("n"), g.get("name")))
        elif float(g.get("checked_at_s") or 0) > float(written_s):
            out.append("gate %s %s was judged again since the fresh instance was written (judged %s, written %s): write "
                       "the instance again so its exception names the failure it excuses" % (
                           g.get("n"), g.get("name"), g.get("checked_at"), b.get("written_at")))
    return out


def _instance_matches(fingerprint: str, resume, root: Path) -> list:
    """[(experiment id, file content)] of the configuration's unused fresh instances that start from `resume`."""
    out = []
    for exp_id, held in instances(fingerprint, root).items():
        b = held.get("block") if isinstance(held.get("block"), dict) else {}
        if repair_available(b) and isinstance(resume, str) and b.get("start_checkpoint") \
                and _norm(resume) == _norm(b["start_checkpoint"]):
            out.append((exp_id, held))
    return out


def _instance_refusal(cmd, rec: dict, experiment_id: str, held: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the fresh instance `experiment_id` and every condition holds (THE FRESH INSTANCE)."""
    b = held.get("block") if isinstance(held.get("block"), dict) else {}
    problems = instance_problems(rec, held, experiment_id, registry)
    problems += instance_gate_problems(rec, b)
    problems += _repair_problems(cmd, rec, b, conf, root, registry)
    if not problems:
        return None
    return "\n".join(["BLOCKED: fresh instance %s of configuration %s (%s): %d condition(s) unmet:" % (
        experiment_id, conf["fingerprint"], instance_path(conf["fingerprint"], experiment_id, root), len(problems))]
        + ["  - " + p for p in problems])


def set_instance(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY):
    """Write a fresh instance (THE FRESH INSTANCE in the module text) of configuration `fingerprint`: the control run
    `block` (a repair_run block of role control with its experiment) in instances/<fingerprint>/<experiment id>.json.
    The configuration's record is read and never written. Refused when the block or the instance's conditions fail
    (repair_block_problems, instance_problems), when that instance was used (a marked launch, or an attempt that trained
    past its start; a used instance is not written again), and when another unused instance of the configuration
    starts from the same checkpoint. An unused instance may be written again; its recorded attempts are carried."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    if not str(by or "").strip():
        raise PreflightError("a fresh instance needs who writes it (--by)")
    block = dict(block or {})
    exp = dict(block.get("experiment")) if isinstance(block.get("experiment"), dict) else {}
    if isinstance(exp.get("exceptions"), dict):
        exp["exceptions"] = {str(k): v for k, v in exp["exceptions"].items()}
        block["experiment"] = exp
    experiment_id = str(exp.get("id") or "")
    block["target_gates"] = sorted({int(n) for n in block.get("target_gates") or []})
    block["remeasure"] = {str(k): str(v) for k, v in (block.get("remeasure") or {}).items()}
    problems = repair_block_problems(block, registry)
    problems += instance_problems(rec, dict(fingerprint=fingerprint, experiment_id=experiment_id, block=block),
                                  experiment_id, registry)
    if problems:
        raise PreflightError("; ".join(problems))
    target = instance_path(fingerprint, experiment_id, root)
    old = _load(target) or {}
    old_block = old.get("block") if isinstance(old.get("block"), dict) else {}
    if repair_uses(old_block):
        at, _at_s, run = repair_launch(old_block)
        raise PreflightError("the fresh instance %s of configuration %s was used (%s, run %s); a used instance is not "
                             "written again" % (experiment_id, fingerprint, at or "an attempt trained past its start",
                                                run))
    for other_id, _held in _instance_matches(fingerprint, block.get("start_checkpoint"), root):
        if other_id != experiment_id:
            raise PreflightError("the fresh instance %s of configuration %s starts from the same checkpoint and is "
                                 "unused; a launch from %s could not tell the two apart" % (
                                     other_id, fingerprint, block.get("start_checkpoint")))
    for k in ("attempts", "launches"):              # only the launch script records these (note_attempt, mark_launched)
        block.pop(k, None)
    carried = [a for a in old_block.get("attempts") or [] if isinstance(a, dict)]
    if carried:                                     # attempts that never trained, kept: one that trains later is the use
        block["attempts"] = carried
    block.update(written_at=clock(), written_at_s=time.time(), written_by=str(by), launched_at=None,
                 launched_at_s=None, run=None)
    _save(target, dict(fingerprint=fingerprint, experiment_id=experiment_id, record=str(path),
                       record_sha256=_sha(path.read_bytes()), written_at=block["written_at"], written_by=str(by),
                       block=block))
    return block


# ------------------------------------------------------------------ THE ADOPTION (the module text)
def adoption_path(fingerprint: str, root: Path = PREFLIGHT_DIR) -> Path:
    """The adoption file of configuration `fingerprint` (one per configuration)."""
    return Path(root) / ADOPTIONS_DIR / ("%s.json" % fingerprint)


def adoption_of(fingerprint: str, root: Path = PREFLIGHT_DIR) -> dict:
    """The adoption file's content ({} when the configuration has none)."""
    held = _load(adoption_path(fingerprint, root))
    return held if isinstance(held, dict) else {}


def adoption_available(ab) -> bool:
    """True while an adoption block is written and unused (counted as a continuation's uses are: repair_uses)."""
    return isinstance(ab, dict) and bool(ab) and not repair_uses(ab)


def _is_fingerprint(v) -> bool:
    return isinstance(v, str) and len(v) == 16 and set(v) <= HEX_DIGITS


def _is_sha256(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= HEX_DIGITS


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _text(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _under(path, folder) -> bool:
    """True when `path` is `folder` or lies inside it (letter case and slashes aside)."""
    p, f = _norm(path), _norm(folder).rstrip("\\/")
    return p == f or p.startswith(f + os.sep)


def _file_sha256(path):
    """The sha256 of a file's bytes, read in pieces of 1 MiB, or None when it cannot be read."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for piece in iter(lambda: f.read(1 << 20), b""):
                h.update(piece)
    except OSError:
        return None
    return h.hexdigest()


def adoption_route(fingerprint):
    """The ADOPTION_ROUTES entry of configuration `fingerprint`, or None (an entry whose fingerprint is not yet filled,
    None, matches no configuration)."""
    for route in ADOPTION_ROUTES:
        if isinstance(route, dict) and _is_fingerprint(route.get("fingerprint")) and route["fingerprint"] == fingerprint:
            return route
    return None


def _names_commit(pinned, named) -> bool:
    """True when `named` names the pinned commit: its full hash, or a prefix of it of at least 9 hexadecimal characters."""
    return isinstance(named, str) and 9 <= len(named) <= 40 and set(named) <= HEX_DIGITS and str(pinned).startswith(named)


def adoption_watch_timing(fingerprint):
    """(the watcher's first-record look in seconds, its seconds to the first and each later rule check) an adoption's
    stop lines are judged with (THE SIXTH ENTRY in the module text): the ADOPTION_ROUTES entry's pinned watch_poll_s and
    watch_first_check_s, else WATCH_POLL_S and WATCH_FIRST_CHECK_S."""
    route = adoption_route(fingerprint) if fingerprint else None
    route = route if isinstance(route, dict) else {}
    return (route.get("watch_poll_s", WATCH_POLL_S), route.get("watch_first_check_s", WATCH_FIRST_CHECK_S))


def adoption_route_problems(fingerprint, ab) -> list:
    """Why an adoption is not the one its route admits (ADOPTION_ROUTES; the module text), or []: the configuration's
    fingerprint is an entry's, and every value the entry pins is the block's, compared exactly: the start and predecessor
    checkpoint, the MANIFEST and its sha256, the model's sha256, the predecessor run, the outcome receipt and its sha256,
    the decision, its message's text sha256 and its author, the wall cap, the start and end steps, the commit the code
    transition starts from, the sha256 of the latent body state the run resumes, and the training row (stop_lines_row;
    the completion jobs' row is never it); and, where the entry pins it, the watcher's timing (watch_timing; an entry
    that pins none admits no block naming one)."""
    route = adoption_route(fingerprint)
    if route is None:
        named = ", ".join(str(r.get("fingerprint")) if _is_fingerprint(r.get("fingerprint")) else
                          "an entry whose fingerprint is not yet filled (None, admitting nothing)"
                          for r in ADOPTION_ROUTES if isinstance(r, dict)) or "none"
        return ["the adoption route admits only the configurations ADOPTION_ROUTES names (%s); configuration %s is not "
                "one (another adoption is a new decision and a reviewed change to this code)" % (named, fingerprint)]
    ab = ab if isinstance(ab, dict) else {}
    pred = ab.get("predecessor") if isinstance(ab.get("predecessor"), dict) else {}
    dec = ab.get("decision") if isinstance(ab.get("decision"), dict) else {}
    ct = ab.get("code_transition") if isinstance(ab.get("code_transition"), dict) else {}
    out = []
    for what, value in (("start checkpoint", ab.get("start_checkpoint")), ("predecessor's checkpoint",
                                                                           pred.get("checkpoint"))):
        if not isinstance(value, str) or _norm(value) != _norm(route["checkpoint"]):
            out.append("the adoption route of configuration %s admits only the preserved checkpoint %s; the adoption's %s "
                       "reads %s" % (fingerprint, route["checkpoint"], what, value))
    paths = (("preservation MANIFEST", pred.get("manifest_path"), route["manifest"]),
             ("predecessor run", pred.get("run"), route["predecessor_run"]),
             ("predecessor's outcome receipt", pred.get("outcome_receipt"), route["outcome_receipt"]))
    for what, value, pin in paths:
        if not isinstance(value, str) or _norm(value) != _norm(pin):
            out.append("the adoption route of configuration %s pins the %s %s; the adoption names %s" % (
                fingerprint, what, pin, value))
    if pred.get("manifest_sha256") != route["manifest_sha256"]:
        out.append("the adoption route of configuration %s admits only the preservation whose MANIFEST reads sha256 %s; "
                   "the adoption names %s" % (fingerprint, route["manifest_sha256"], pred.get("manifest_sha256")))
    if pred.get("model_sha256") != route["model_sha256"]:
        out.append("the adoption route of configuration %s pins the model sha256 %s; the adoption names %s" % (
            fingerprint, route["model_sha256"], pred.get("model_sha256")))
    if pred.get("outcome_receipt_sha256") != route["outcome_receipt_sha256"]:
        out.append("the adoption route of configuration %s pins the outcome receipt sha256 %s; the adoption names %s" % (
            fingerprint, route["outcome_receipt_sha256"], pred.get("outcome_receipt_sha256")))
    if dec.get("id") != route["decision"]:
        out.append("the adoption route of configuration %s pins the decision %s; the adoption names %s" % (
            fingerprint, route["decision"], dec.get("id")))
    if dec.get("message_sha256") != route["decision_text_sha256"]:
        out.append("the adoption route of configuration %s pins the decision message's text sha256 %s; the adoption names "
                   "%s" % (fingerprint, route["decision_text_sha256"], dec.get("message_sha256")))
    if dec.get("author") != route["decision_author"]:
        out.append("the adoption route of configuration %s pins the decision's author %r; the adoption names %r" % (
            fingerprint, route["decision_author"], dec.get("author")))
    if not _number(ab.get("wall_cap_s")) or ab.get("wall_cap_s") != route["wall_cap_s"]:
        out.append("the adoption route of configuration %s pins the wall cap %s s; the adoption's wall_cap_s reads %r" % (
            fingerprint, route["wall_cap_s"], ab.get("wall_cap_s")))
    if ab.get("start_step") != route["start"] or not _whole(ab.get("start_step")):
        out.append("the adoption route of configuration %s starts at model step %d; the adoption's start_step reads %r"
                   % (fingerprint, route["start"], ab.get("start_step")))
    if ab.get("end_step") != route["end"] or not _whole(ab.get("end_step")):
        out.append("the adoption route of configuration %s ends at model step %d; the adoption's end_step reads %r"
                   % (fingerprint, route["end"], ab.get("end_step")))
    if ab.get("stop_lines_row") != route["run_row"]:
        out.append("the adoption route of configuration %s pins the training row %s; the adoption's stop_lines_row "
                   "reads %r (only that row may launch)" % (fingerprint, route["run_row"], ab.get("stop_lines_row")))
    pinned = ("watch_poll_s" in route or "watch_first_check_s" in route)
    timing = ab.get("watch_timing")
    if pinned and timing != dict(poll_s=route.get("watch_poll_s"), first_check_s=route.get("watch_first_check_s")):
        out.append("the adoption route of configuration %s pins the watcher's first-record look %s s and checks %s s "
                   "(the run owner's and the watcher's own settings); the adoption's watch_timing reads %r" % (
                       fingerprint, route.get("watch_poll_s"), route.get("watch_first_check_s"), timing))
    if not pinned and timing is not None:
        out.append("the adoption route of configuration %s pins no watcher timing, so the watcher's %d s look and %d s "
                   "checks apply; the adoption's watch_timing reads %r" % (fingerprint, WATCH_POLL_S,
                                                                           WATCH_FIRST_CHECK_S, timing))
    # THE TENTH AND THE ELEVENTH ENTRY (the module text): the owner bound, its limits and run length, and the code that runs
    bound = route.get("owner_bound")
    if bound != ab.get("owner_bound"):
        out.append("the adoption route of configuration %s pins the run owner's bound %r; the adoption's owner_bound "
                   "reads %r (compared exactly; an entry that pins none admits no block naming one)" % (
                       fingerprint, bound, ab.get("owner_bound")))
    if bound is not None and (route["end"] - route["start"] >= CONTROL_POLL_STEPS
                              or not isinstance(bound, dict) or route["wall_cap_s"] != bound.get("trainer_limit_s")):
        out.append("the adoption route of configuration %s pins an owner bound on a run of %d model steps with a wall "
                   "cap of %s s: an owner bound is admitted only for a run shorter than %d model steps whose wall cap "
                   "is the bound's trainer limit" % (fingerprint, route["end"] - route["start"], route["wall_cap_s"],
                                                     CONTROL_POLL_STEPS))
    if "to_commit" in route and ct.get("to_commit") != route["to_commit"]:
        out.append("the adoption route of configuration %s pins the code it runs, commit %s; the adoption's "
                   "code_transition.to_commit reads %r (the full hash)" % (fingerprint, route["to_commit"],
                                                                           ct.get("to_commit")))
    if ab.get("completion_jobs_row") == route["run_row"]:
        out.append("the adoption's completion_jobs_row names the training row %s: a completion row cannot stand in for "
                   "it, nor it for a completion row" % route["run_row"])
    if not _names_commit(route["from_commit"], ct.get("from_commit")):
        out.append("the adoption route of configuration %s pins the code transition's start, commit %s; the adoption's "
                   "from_commit reads %r (the full hash or a prefix of it of at least 9 characters)" % (
                       fingerprint, route["from_commit"], ct.get("from_commit")))
    flags = ab.get("run_flags") if isinstance(ab.get("run_flags"), dict) else {}
    latent = flags.get("--latent-body-resume")
    got = _file_sha256(latent) if _text(latent) and Path(latent).is_file() else None
    if got != route["latent_sha256"]:
        out.append("the adoption route of configuration %s pins the latent body state's sha256 %s; the state the adoption "
                   "resumes, %s, reads %s" % (fingerprint, route["latent_sha256"], latent,
                                              "sha256 %s" % got if got else "nothing (not a file)"))
    return out


def adoption_decision_problems(route: dict, ab) -> list:
    """THE DECISION AND THE DELEGATION, read from their files (the review's finding 2, GAP3): the route's decision
    message reads as JSON naming the decision's id, and its "text" field's sha256 (UTF-8) is the pinned one; the
    delegation record holds each of the operator's quoted sentences the route pins, and the adoption's authorization
    holds each of them verbatim."""
    ab = ab if isinstance(ab, dict) else {}
    out = []
    message = Path(route["decision_message"])
    try:
        data = json.loads(message.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        data = None
        out.append("the decision's message %s cannot be read: %s" % (message, exc))
    if data is not None:
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("id") != route["decision"]:
            out.append("the decision's message %s does not name decision %s (it names %s)" % (
                message, route["decision"], data.get("id") if isinstance(data, dict) else "nothing"))
        digest = _sha(text.encode("utf-8")) if isinstance(text, str) else None
        if digest != route["decision_text_sha256"]:
            out.append("the decision's message %s holds a text whose sha256 reads %s; the route pins %s" % (
                message, digest, route["decision_text_sha256"]))
    if "decision_message_sha256" in route:          # THE TENTH AND THE ELEVENTH ENTRY: the message file's bytes
        got = _file_sha256(message) if message.is_file() else None
        if got != route["decision_message_sha256"]:
            out.append("the decision's message file %s reads sha256 %s; the route pins %s" % (
                message, got, route["decision_message_sha256"]))
    try:
        record = Path(route["delegation_record"]).read_text(encoding="utf-8-sig")
    except OSError as exc:
        record = None
        out.append("the delegation record %s cannot be read: %s" % (route["delegation_record"], exc))
    words = ab.get("authorization") if isinstance(ab.get("authorization"), str) else ""
    for sentence in route["delegation_words"]:
        if record is not None and sentence not in record:
            out.append("the delegation record %s does not hold the operator's sentence %r the route pins" % (
                route["delegation_record"], sentence))
        if sentence not in words:
            out.append("the adoption's authorization does not hold the operator's delegation sentence %r verbatim" %
                       sentence)
    return out


def adoption_template_problems(route: dict, rec, cmd=None, noting=False) -> list:
    """THE COMMAND TEMPLATE (the review's finding 3): the template file reads the pinned sha256; when written and at
    launch (not `noting`), the record's configuration flags are the template's (its flags outside RUN_FLAGS); at launch
    (`cmd` given), every flag of the launch command but ADOPTION_TEMPLATE_REPLACED reads as the template's, present or
    absent alike."""
    path = Path(route["command_template"])
    try:
        data = path.read_bytes()
        template = json.loads(data.decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        return ["the command template %s cannot be read: %s" % (path, exc)]
    out = []
    if _sha(data) != route["command_template_sha256"]:
        out.append("the command template %s reads sha256 %s; the route pins %s" % (
            path, _sha(data), route["command_template_sha256"]))
    if not isinstance(template, list):
        return out + ["the command template %s is not a command (a JSON list)" % path]
    tflags = flags_of(template)
    if not noting:
        wanted = {k: v for k, v in tflags.items() if k not in RUN_FLAGS}
        held = (rec or {}).get("flags") or {}
        differ = sorted(k for k in set(wanted) | set(held) if wanted.get(k) != held.get(k))
        if differ:
            out.append("the record's configuration flags are not the command template's: %s" % "; ".join(
                "%s reads %s, the template %s" % (k, held.get(k, "absent"), wanted.get(k, "absent")) for k in differ))
    if cmd is not None:
        cflags = flags_of(cmd)
        for k in sorted(set(tflags) | set(cflags)):
            if k not in ADOPTION_TEMPLATE_REPLACED and cflags.get(k) != tflags.get(k):
                out.append("the launch's %s reads %s; the command template reads %s (the decision replaces only %s)" % (
                    k, cflags.get(k, "absent"), tflags.get(k, "absent"), ", ".join(ADOPTION_TEMPLATE_REPLACED)))
    return out


def _adoption_manifest(pred: dict):
    """(the preservation MANIFEST's content, its sha256, None), or (None, its sha256 or None, the text naming why it cannot
    be read)."""
    path = Path(str(pred.get("manifest_path") or ""))
    if not path.is_file():
        return None, None, ("the preservation MANIFEST %s is missing: an adoption starts only from a preserved checkpoint "
                            "whose MANIFEST records it" % path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        return None, None, "the preservation MANIFEST %s cannot be read: %s" % (path, exc)
    digest = _sha(data)
    try:
        parsed = json.loads(data.decode("utf-8-sig"))
    except ValueError as exc:
        return None, digest, "the preservation MANIFEST %s cannot be read as JSON: %s" % (path, exc)
    if not isinstance(parsed, dict):
        return None, digest, "the preservation MANIFEST %s is not a JSON object" % path
    return parsed, digest, None


def adoption_preservation_problems(ab) -> list:
    """MISSING PRESERVATION (the module text): the MANIFEST missing, unreadable, or reading another sha256 than the
    block's; the predecessor checkpoint without meta.json or model.safetensors."""
    ab = ab if isinstance(ab, dict) else {}
    pred = ab.get("predecessor") if isinstance(ab.get("predecessor"), dict) else {}
    out = []
    if _text(pred.get("manifest_path")):
        _manifest, digest, why = _adoption_manifest(pred)
        if why:
            out.append(why)
        elif _is_sha256(pred.get("manifest_sha256")) and digest != pred["manifest_sha256"]:
            out.append("the preservation MANIFEST %s reads sha256 %s; the adoption names %s (the preservation record "
                       "changed, or it is another)" % (pred["manifest_path"], digest, pred["manifest_sha256"]))
    ck = pred.get("checkpoint") if _text(pred.get("checkpoint")) else ab.get("start_checkpoint")
    if _text(ck):
        for name in CHECKPOINT_FILES:
            if not (Path(ck) / name).is_file():
                out.append("the predecessor checkpoint %s holds no %s: the preserved state is incomplete" % (ck, name))
    return out


def adoption_preserved_files_problems(ab) -> list:
    """THE PRESERVED FILES (the review's finding 4, GAP6), when written and at launch: every MANIFEST entry under the
    start checkpoint's folder, the latent body state latent_body/latest.pt and every entry under episode_companion/ is a
    file of the preservation folder whose sha256 is the one the MANIFEST records; the MANIFEST records at least one of
    each, and the start checkpoint lies inside the preservation folder."""
    ab = ab if isinstance(ab, dict) else {}
    pred = ab.get("predecessor") if isinstance(ab.get("predecessor"), dict) else {}
    if not _text(pred.get("manifest_path")) or not _text(pred.get("checkpoint")):
        return []
    manifest = _adoption_manifest(pred)[0]
    if manifest is None:
        return []                                   # named by adoption_preservation_problems
    base = Path(pred["manifest_path"]).parent
    if not _under(pred["checkpoint"], base):
        return ["the predecessor checkpoint %s is not inside the preservation folder %s" % (pred["checkpoint"], base)]
    ck_rel = os.path.relpath(_norm(pred["checkpoint"]), _norm(base)).replace("\\", "/") + "/"
    latent_rel = "latent_body/" + LATENT_STATE_FILE
    groups = {"the start checkpoint's folder %s" % ck_rel: [], latent_rel: [], ADOPTION_COMPANION_DIR: []}
    keys = list(groups)
    for f in manifest.get("files") or []:
        if not isinstance(f, dict):
            continue
        rel = str(f.get("path") or "").replace("\\", "/")
        if rel.lower().startswith(ck_rel.lower()):
            groups[keys[0]].append(f)
        elif rel == latent_rel:
            groups[keys[1]].append(f)
        elif rel.startswith(ADOPTION_COMPANION_DIR):
            groups[keys[2]].append(f)
    out = []
    for name, entries in groups.items():
        if not entries:
            out.append("the preservation MANIFEST %s records no file of %s" % (pred["manifest_path"], name))
        for f in entries:
            path = base / str(f.get("path")).replace("\\", "/")
            got = _file_sha256(path) if path.is_file() else None
            if got != f.get("sha256"):
                out.append("the preserved file %s reads %s; the MANIFEST records sha256 %s" % (
                    path, "sha256 %s" % got if got else "nothing (not a file)", f.get("sha256")))
    return out


def adoption_predecessor_problems(ab) -> list:
    """WRONG PREDECESSOR (the module text): the predecessor's checkpoint is the start checkpoint, whose name gives
    start_step; its model.safetensors reads model_sha256 and the MANIFEST records that model_sha256; the latent body
    state resumed reads the sha256 the MANIFEST records for latent_body/latest.pt; the predecessor run's stage_done.json
    reads ok at start_step, and the run folder lies in the preservation folder with that stage_done.json one of the
    files the MANIFEST records, its sha256 the recorded one; the outcome receipt reads outcome_receipt_sha256. A file that
    is absent is named by the preservation's or the run flags' check."""
    ab = ab if isinstance(ab, dict) else {}
    pred = ab.get("predecessor") if isinstance(ab.get("predecessor"), dict) else {}
    out = []
    start, ck = ab.get("start_checkpoint"), pred.get("checkpoint")
    if _text(ck) and _text(start) and _norm(ck) != _norm(start):
        out.append("the adoption's start checkpoint %s is not its predecessor's checkpoint %s (an adoption resumes exactly "
                   "the preserved predecessor)" % (start, ck))
    if _text(start) and _whole(ab.get("start_step")) and _step_of(start) != ab["start_step"]:
        out.append("the start checkpoint %s names model step %s; the adoption's start_step reads %d" % (
            start, _step_of(start), ab["start_step"]))
    want = pred.get("model_sha256")
    if _text(ck) and _is_sha256(want) and (Path(ck) / MODEL_FILE).is_file():
        got = _file_sha256(Path(ck) / MODEL_FILE)
        if got != want:
            out.append("the predecessor's model %s reads sha256 %s; the adoption names %s" % (Path(ck) / MODEL_FILE, got,
                                                                                            want))
    manifest = _adoption_manifest(pred)[0] if _text(pred.get("manifest_path")) else None
    if manifest is not None and _is_sha256(want) and manifest.get("model_sha256") != want:
        out.append("the preservation MANIFEST %s records model sha256 %s; the adoption names %s" % (
            pred["manifest_path"], manifest.get("model_sha256"), want))
    flags = ab.get("run_flags") if isinstance(ab.get("run_flags"), dict) else {}
    latent = flags.get("--latent-body-resume")
    if _text(latent) and manifest is not None:
        recorded = sorted({str(f.get("sha256")) for f in manifest.get("files") or [] if isinstance(f, dict)
                           and str(f.get("path") or "").replace("\\", "/").endswith("latent_body/" + LATENT_STATE_FILE)})
        got = _file_sha256(latent) if Path(latent).is_file() else None
        if not recorded:
            out.append("the preservation MANIFEST %s records no latent_body/%s, so the latent body state the adoption "
                       "resumes cannot be tied to the predecessor" % (pred["manifest_path"], LATENT_STATE_FILE))
        elif got not in recorded:
            out.append("the latent body state %s the adoption resumes reads %s; the preservation MANIFEST records sha256 "
                       "%s for the predecessor's latent_body/%s" % (latent, "sha256 %s" % got if got else
                                                                    "nothing (not a file)", ", ".join(recorded),
                                                                    LATENT_STATE_FILE))
    if _text(pred.get("run")) and _whole(ab.get("start_step")):
        done = stage_done(pred["run"])
        if done.get("ok") is not True or done.get("steps") != ab["start_step"]:
            out.append("the predecessor run %s did not end with its own save at model step %d: its stage_done.json reads "
                       "%s" % (pred["run"], ab["start_step"], "ok %s at model step %s" % (done.get("ok"),
                                                                                        done.get("steps"))
                               if done else "nothing"))
    if _text(pred.get("run")) and manifest is not None:
        # THE PRESERVED RUN (the coordinator's decision on the package review's finding 2): the predecessor run is the
        # preservation's copy, and its stage_done.json is the file the MANIFEST records
        base = Path(pred["manifest_path"]).parent
        if not _under(pred["run"], base):
            out.append("the predecessor run %s is not inside the preservation folder %s: its stage_done.json is not one "
                       "the MANIFEST records" % (pred["run"], base))
        else:
            rel = os.path.relpath(_norm(Path(pred["run"]) / "stage_done.json"), _norm(base)).replace("\\", "/")
            recorded = [f for f in manifest.get("files") or [] if isinstance(f, dict)
                        and str(f.get("path") or "").replace("\\", "/").lower() == rel.lower()]
            got = _file_sha256(Path(pred["run"]) / "stage_done.json")
            if not recorded:
                out.append("the preservation MANIFEST %s records no %s, the predecessor run's stage_done.json" % (
                    pred["manifest_path"], rel))
            elif got != recorded[0].get("sha256"):
                out.append("the predecessor run's stage_done.json %s reads %s; the MANIFEST records sha256 %s" % (
                    Path(pred["run"]) / "stage_done.json", "sha256 %s" % got if got else "nothing (not a file)",
                    recorded[0].get("sha256")))
    receipt, receipt_sha = pred.get("outcome_receipt"), pred.get("outcome_receipt_sha256")
    if _text(receipt) and _is_sha256(receipt_sha):
        got = _file_sha256(receipt) if Path(receipt).is_file() else None
        if got != receipt_sha:
            out.append("the predecessor's outcome receipt %s reads %s; the adoption names sha256 %s" % (
                receipt, "sha256 %s" % got if got else "nothing (not a file)", receipt_sha))
    return out


def _authorized_blocks(root: Path, exclude=None) -> list:
    """[(where, block)] of every block of the preflight folder `root` that carries an authorization: each record's
    continuation and its repair run's experiment, each extension, each fresh instance's experiment, each adoption and
    each recovery (THE LINKED RECOVERY) and each link (THE LINKED CONTINUATION), but the file `exclude`; a file that
    cannot be read is listed with block None."""
    root, out = Path(root), []

    def read(p):
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            out.append((str(p), None))
            return {}
        return data if isinstance(data, dict) else {}
    for p in sorted(root.glob("*.json")):
        if p.name == BASELINE:
            continue
        rec = read(p)
        if isinstance(rec.get(CONTINUATION), dict) and rec[CONTINUATION]:
            out.append(("the continuation of record %s" % p.stem, rec[CONTINUATION]))
        exp = (rec.get("repair_run") or {}).get("experiment") if isinstance(rec.get("repair_run"), dict) else None
        if isinstance(exp, dict):
            out.append(("experiment %s of record %s" % (exp.get("id"), p.stem), exp))
    for p in sorted((root / EXTENSIONS_DIR).glob("*.json")):
        b = read(p).get("block")
        if isinstance(b, dict):
            out.append(("the extension %s" % p, b))
    for p in sorted((root / INSTANCES_DIR).glob("*/*.json")):
        b = read(p).get("block")
        if isinstance(b, dict) and isinstance(b.get("experiment"), dict):
            out.append(("the fresh instance %s" % p, b["experiment"]))
    for folder, what in ((ADOPTIONS_DIR, "the adoption"), (RECOVERIES_DIR, "the recovery"), (LINKS_DIR, "the link")):
        for p in sorted((root / folder).glob("*.json")):
            if exclude is not None and _norm(p) == _norm(exclude):
                continue
            b = read(p).get("block")
            if isinstance(b, dict):
                out.append(("%s %s" % (what, p), b))
    return out


def adoption_authorization_problems(fingerprint, ab, root: Path = PREFLIGHT_DIR) -> list:
    """REUSED AUTHORIZATION (the module text): the authorization names its decision's id (the operator's delegation
    words with the decision's reference); no other block of the preflight folder carries the same decision id or the
    same decision message sha256, and no other block's authorization words lie within this one's, or this one's within
    them (whitespace aside; the review's finding 2, GAP4)."""
    return _authorization_reuse_problems(ab, adoption_path(fingerprint, root), root, ADOPTION)


def _authorization_reuse_problems(ab, exclude, root: Path, noun: str) -> list:
    """adoption_authorization_problems for the block `ab` of the file `exclude`, its texts naming it `noun` (an adoption
    or THE LINKED RECOVERY's recovery: the same check on its own inputs)."""
    ab = ab if isinstance(ab, dict) else {}
    dec = ab.get("decision") if isinstance(ab.get("decision"), dict) else {}
    words, did, msha = ab.get("authorization"), dec.get("id"), dec.get("message_sha256")
    out = []
    if _text(did) and isinstance(words, str) and did not in words:
        out.append("the %s's authorization does not name its decision %s: it holds the operator's delegation words "
                   "verbatim with the decision's reference" % (noun, did))
    mine = " ".join(words.split()) if _text(words) else None
    for where, other in _authorized_blocks(root, exclude=exclude):
        if other is None:
            out.append("%s cannot be read, so the reuse of the %s's authorization cannot be checked against it"
                       % (where, noun))
            continue
        odec = other.get("decision") if isinstance(other.get("decision"), dict) else {}
        if _text(did) and odec.get("id") == did:
            out.append("the decision %s already authorized %s: a decision authorizes one block" % (did, where))
        if _text(msha) and odec.get("message_sha256") == msha:
            out.append("the decision message whose sha256 reads %s already authorized %s" % (msha, where))
        theirs = " ".join(other["authorization"].split()) if _text(other.get("authorization")) else None
        if mine and theirs and (theirs in mine or mine in theirs):
            out.append("the %s's authorization and the authorization words of %s lie one within the other: each "
                       "block carries its own authorization" % (noun, where))
    return out


def adoption_record_problems(fingerprint, rec, ab, code_hash=None, record_sha256=None,
                             root: Path = PREFLIGHT_DIR) -> list:
    """CHANGED CODE OR SETTINGS (the module text), in the record: it names this configuration, and its flags, the hashes
    of the files they name and its code hash still compute to its fingerprint; `code_hash` (the checkout's trainer code,
    when given) is the record's; the record's bytes read `record_sha256` (when given: the adoption file's); the record
    holds no repair run and no continuation; the start checkpoint is a model state the record names."""
    rec = rec if isinstance(rec, dict) else {}
    ab = ab if isinstance(ab, dict) else {}
    out = []
    if rec.get("fingerprint") != fingerprint:
        out.append("the record names configuration %s, not %s" % (rec.get("fingerprint"), fingerprint))
    body = json.dumps({"flags": rec.get("flags") or {}, "files": rec.get("files") or {}, "code": rec.get("code_hash")},
                      sort_keys=True).encode()
    if _sha(body)[:16] != fingerprint:
        out.append("the record's flags, the hashes of the files they name and its code hash compute to configuration %s, "
                   "not %s: its settings or code were changed after it was written" % (_sha(body)[:16], fingerprint))
    if code_hash is not None and code_hash != rec.get("code_hash"):
        out.append("the checkout's trainer code (hash %s) is not the code the record of configuration %s holds (%s): an "
                   "adoption runs exactly its record's code" % (code_hash, fingerprint, rec.get("code_hash")))
    for key, what in (("repair_run", "repair run"), (CONTINUATION, "continuation")):
        if rec.get(key):
            out.append("the record holds a %s; an adoption is the one bounded launch of a record written for it" % what)
    start = ab.get("start_checkpoint")
    if _text(start) and not any(isinstance(s, dict) and _norm(s.get("checkpoint") or "") == _norm(start)
                                for s in rec.get("states") or []):
        out.append("the adoption's start %s is not a model state the record names (its gates were judged for %s)" % (
            start, "; ".join(str(s.get("checkpoint")) for s in rec.get("states") or [] if isinstance(s, dict))
            or "none"))
    if record_sha256 is not None:
        try:
            now = _sha(record_path(fingerprint, root).read_bytes())
        except OSError:
            now = None
        if now != record_sha256:
            out.append("the record %s changed since the adoption was written (sha256 now %s, the adoption names %s): "
                       "write the adoption again" % (record_path(fingerprint, root), now, record_sha256))
    return out


def adoption_gate_problems(rec, ab) -> list:
    """THE GATES (the decision: "Keep each gate FAIL; no normal-training qualification is granted"; the review's finding
    1), when written and at launch: each of the seven gates reads FAIL with its reason and evidence, and the block names
    its complete exception {failure, relevance, why_measurable, evidence}, written after the gate was last judged
    (written_at_s). A gate reading PASS, N/A, OPEN or FAIL - REPAIR TARGET, or missing from the record, is refused."""
    gates = {g.get("n"): g for g in (rec or {}).get("gates") or [] if isinstance(g, dict)}
    ab = ab if isinstance(ab, dict) else {}
    exceptions = ab.get("exceptions") if isinstance(ab.get("exceptions"), dict) else {}
    written_s = ab.get("written_at_s")
    out = []
    for n, name, _q in GATES:
        g = gates.get(n) or {}
        status = g.get("status") or "OPEN (missing from the record)"
        if status != "FAIL":
            out.append("gate %d %s reads %s; an adoption launches only with every gate FAIL (the decision: \"Keep each "
                       "gate FAIL; no normal-training qualification is granted\")" % (n, name, status))
            continue
        if not _text(g.get("reason")) or not _evidence(g.get("evidence")):
            out.append("gate %d %s reads FAIL without its reason and evidence" % (n, name))
        x = exceptions.get(str(n))
        if not (isinstance(x, dict) and all(_evidence(x.get(f)) if f == "evidence" else _text(x.get(f))
                                            for f in ADOPTION_EXCEPTION_FIELDS)):
            out.append("gate %d %s reads FAIL and the adoption names no complete exception for it {%s}" % (
                n, name, ", ".join(ADOPTION_EXCEPTION_FIELDS)))
        elif not _number(written_s):
            out.append("gate %d %s: its exception cannot be tied to the failure it excuses: the adoption names no "
                       "written_at_s (write it with training_preflight adoption)" % (n, name))
        elif float(g.get("checked_at_s") or 0) > float(written_s):
            out.append("gate %d %s was judged again since its exception was written (judged %s, the adoption written %s): "
                       "write the adoption again so its exception names the failure it excuses" % (
                           n, name, g.get("checked_at") or g.get("checked_at_s"), ab.get("written_at") or written_s))
    return out


def _own_claim_problems(row, claim, registry: Path = REGISTRY, note_run=None) -> list:
    """THE ROW'S OWN CURRENT CLAIM (proxy decision overnight-admission-reader-final-watch-decisions-20261022, B1):
    at the note-attempt operation only, the training row reads status running because its own run owner claimed it
    (test_registry.job_started, file_lifecycle.claim_start) before writing into the attempt's folder. [] only when the
    caller's claim identity `claim` (run_id, runner_pid, runner_pid_created, generator, run) is that claim: the row's
    last_run is sealed, running and unended, carries that run_id and that runner identity, and the runner process is
    alive with that creation time; the claimed declaration is the row's; the claim's command is the FULL registered
    generator (the declaration's, not last_run.command, which keeps 60 tokens); the claim's run is the generator's --out
    inside the declared evidence folder and is the very folder note_attempt records (`note_run`, its own run argument;
    refused before any attempt is written); and the work record's attempt reads claimed for that run_id, for this row and
    under this declaration (overnight-noon-results-close-remaining-work-20261022). A foreign, stale, mismatched
    or consumed claim returns the reasons."""
    from mimo_bridge.tools import test_registry as TR, file_lifecycle as FL
    c = claim if isinstance(claim, dict) else {}
    last = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
    out = []
    if not TR.sealed(row):
        out.append("its last_run is not sealed")
    if not _text(c.get("run_id")) or last.get("run_id") != c.get("run_id"):
        out.append("the claim's run_id %r is not the row's current run %r" % (c.get("run_id"), last.get("run_id")))
    if last.get("state") != "running" or last.get("ended") or last.get("ended_at"):
        out.append("the row's run is not a current, unended claim (state %r)" % last.get("state"))
    rp, rc = c.get("runner_pid"), c.get("runner_pid_created")
    if not (_number(rp) and _number(rc) and last.get("runner_pid") == int(rp)
            and _number(last.get("runner_pid_created")) and abs(float(last["runner_pid_created"]) - float(rc)) < 1.0):
        out.append("the claim's runner %r is not the row's recorded runner %r" % (
            [rp, rc], [last.get("runner_pid"), last.get("runner_pid_created")]))
    else:
        now = TR.process_created(int(rp))
        if not _number(now) or abs(float(now) - float(rc)) >= 1.0:
            out.append("the recorded runner process %s is not alive with its recorded creation time (a stale claim)" % rp)
    binding = last.get("file_lifecycle") if isinstance(last.get("file_lifecycle"), dict) else {}
    decl = binding.get("declaration") if isinstance(binding.get("declaration"), dict) else {}
    if not decl or decl != row.get("file_lifecycle"):
        out.append("the claimed output declaration is absent or differs from the row's")
    gen = decl.get("generator")
    if not isinstance(gen, list) or [str(x) for x in gen] != [str(x) for x in (c.get("generator") or [])]:
        out.append("the claim's command is not the full registered generator")
    out_flag = gen[gen.index("--out") + 1] if isinstance(gen, list) and "--out" in gen[:-1] else None
    evidence = os.path.normcase(os.path.abspath(str(decl.get("evidence_dir") or "")))
    run = os.path.normcase(os.path.abspath(str(c.get("run") or "")))
    if not (_text(c.get("run")) and out_flag and run == os.path.normcase(os.path.abspath(str(out_flag)))
            and _text(decl.get("evidence_dir")) and run.startswith(evidence + os.sep)):
        out.append("the claim's run %r is not the generator's --out %r inside the declared evidence folder %r" % (
            c.get("run"), out_flag, decl.get("evidence_dir")))
    if not _text(note_run) or os.path.normcase(os.path.abspath(str(note_run))) != run:
        out.append("note_attempt's run %r is not the claim's run %r" % (note_run, c.get("run")))
    try:
        data = json.loads(Path(registry).read_text(encoding="utf-8-sig"))
        manifest = json.loads(FL.manifest_path(data["file_lifecycle_policy"], decl["work_id"]).read_text(
            encoding="utf-8"))
        att = [a for a in manifest.get("managed_attempts") or [] if isinstance(a, dict)
               and a.get("attempt_id") == decl.get("attempt_id")]
        if len(att) != 1 or att[0].get("state") != "claimed" or att[0].get("run_id") != c.get("run_id"):
            out.append("the work record's attempt %s is not claimed for run %s (state %r, run %r)" % (
                decl.get("attempt_id"), c.get("run_id"), att[0].get("state") if len(att) == 1 else None,
                att[0].get("run_id") if len(att) == 1 else None))
        elif att[0].get("row_id") != row.get("id") or att[0].get("declaration") != decl:
            out.append("the work record's attempt %s belongs to row %r or another declaration, not row %r under its "
                       "current declaration" % (decl.get("attempt_id"), att[0].get("row_id"), row.get("id")))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        out.append("the work record cannot be read: %s: %s" % (type(exc).__name__, exc))
    return out


def adoption_registry_problems(fingerprint, ab, registry: Path = REGISTRY, moment="launch", field=ADOPTION_ROW_FIELD,
                               noun=ADOPTION, claim=None, note_run=None) -> list:
    """THE REGISTRY ROWS (the review's finding 8, GAP9; notes proxy-prediction-closure-20261013, ADOPTION PACKAGE
    NOTES): the training row (stop_lines_row, the one ADOPTION_ROUTES pins) and the completion jobs' row are in the test
    registry. When written, at launch and when an attempt is noted (`moment` "written", "launch", "note"): each reads
    status written or queued; the training row carries no block (the registry's block_row), and the completion jobs'
    row carries none but one waiting on this adoption (its waits_on names the training row or the decision); and the
    training row's record of this adoption (ADOPTION_ROW_FIELD, written by note_attempt and mark_launched) holds no
    attempt and no launch: an attempt noted counts whether or not its trainer trained or was marked launched (a failed
    attempt is not retried by this route). When a launch is marked (`moment` "mark", after the run owner set the training
    row running) only a launch marked before refuses. A record of another adoption refuses. Completion rows are only
    read: their status stays the one their own jobs write. RESIDUAL LIMIT: the one use is held in two places, the
    adoption file and the training row's record (beside the run owner's own status and sealed last_run in that row, read
    as its status); a hand edit of both, or a launch by a tool that writes neither, is not seen here. THE LINKED
    RECOVERY's rows are judged here too, on its own inputs: `field` the training row's record field (RECOVERY_ROW_FIELD)
    and `noun` the word its texts use."""
    ab = ab if isinstance(ab, dict) else {}
    rows = _registry_rows(registry)
    run_row = ab.get("stop_lines_row")
    decision = (ab.get("decision") or {}).get("id") if isinstance(ab.get("decision"), dict) else None
    a_noun = ("an " if noun[:1] in "aeiou" else "a ") + noun
    out = []
    for key, label in (("stop_lines_row", "the training row"), ("completion_jobs_row", "the completion jobs' row")):
        row_id = ab.get(key)
        if not _text(row_id):
            continue
        row = rows.get(row_id)
        if row is None:
            out.append("%s %s is not in the test registry (it is declared there before launch)" % (label, row_id))
            continue
        if moment == "mark":
            continue
        # B1 (overnight-admission-reader-final-watch-decisions-20261022): at the note-attempt operation only, the
        # training row's OWN current claim, proved by the caller's claim identity (_own_claim_problems), is admitted
        own = (moment == "note" and key == "stop_lines_row" and row.get("status") == "running" and claim is not None)
        own_why = _own_claim_problems(row, claim, registry, note_run) if own else []
        if row.get("status") not in CONTINUATION_ROW_STATUSES and not (own and not own_why):
            out.append("%s %s reads status %s; %s's row is declared for it and not yet run (status %s)%s" % (
                label, row_id, row.get("status"), a_noun, " or ".join(CONTINUATION_ROW_STATUSES),
                "; its own claim is not proved: %s" % "; ".join(own_why) if own_why else ""))
        block = row.get("blocked") if isinstance(row.get("blocked"), dict) else None
        if block and _text(block.get("reason")):
            waits = str(block.get("waits_on") or "")
            if key != "completion_jobs_row" or not ((_text(run_row) and run_row in waits)
                                                    or (_text(decision) and decision in waits)):
                out.append("%s %s is blocked (%s; it waits on %s): only the completion jobs' row may carry a block, "
                           "one waiting on this %s's training row %s" % (
                               label, row_id, block.get("reason"), waits or "not stated", noun, run_row))
    row = rows.get(run_row) if _text(run_row) else None
    held = row.get(field) if isinstance(row, dict) else None
    if held is not None:
        if not isinstance(held, dict) or held.get("fingerprint") != fingerprint:
            out.append("the training row %s records the launch of another %s (configuration %s): a row records "
                       "one %s" % (run_row, noun, held.get("fingerprint") if isinstance(held, dict) else held, noun))
        elif held.get("launched_at"):
            out.append("the training row %s records this %s's launch marked %s (run %s): %s launches "
                       "once, whatever its %s file reads" % (run_row, noun, held.get("launched_at"), held.get("run"),
                                                             a_noun, noun))
        elif moment != "mark" and held.get("attempts"):
            last = held["attempts"][-1] if isinstance(held["attempts"][-1], dict) else {}
            out.append("the training row %s records this %s's attempt (run %s, noted %s): %s launches "
                       "once, and a noted attempt counts whether or not its trainer trained or was marked launched "
                       "(notes proxy-prediction-closure-20261013)" % (run_row, noun, last.get("run"),
                                                                              last.get("at"), a_noun))
    return out


def _record_in_training_row(fingerprint, ab, registry, run, by, kind, field=ADOPTION_ROW_FIELD, noun=ADOPTION):
    """Write the attempt (kind "attempt") or the launch (kind "launch") into the training row's record of this adoption
    (ADOPTION_ROW_FIELD of the stop_lines_row), through test_registry.update_row (the registry's one writer: a lock beside
    the file, only this field of the one row; no status is written). Raises PreflightError when the row cannot be
    written. THE LINKED RECOVERY's record is written the same way into its own field (`field`, RECOVERY_ROW_FIELD)."""
    from mimo_bridge.tools import test_registry as TR    # imported here: the registry's writer, needed only for this
    entry = dict(run=str(Path(str(run)).resolve()), at=clock(), at_s=time.time(), by=str(by))

    def change(row):
        held = row.get(field) if isinstance(row.get(field), dict) else None
        held = dict(held) if held else dict(fingerprint=fingerprint, decision=(ab.get("decision") or {}).get("id"),
                                            start_checkpoint=ab.get("start_checkpoint"), attempts=[], launched_at=None,
                                            launched_at_s=None, run=None)
        if kind == "attempt":
            held["attempts"] = list(held.get("attempts") or []) + [entry]
        else:
            held.update(launched_at=entry["at"], launched_at_s=entry["at_s"], run=entry["run"], launched_by=entry["by"])
        return {field: held}
    try:
        written = TR.update_row(ab.get("stop_lines_row"), change, path=registry)
    except TR.RegistryError as exc:
        raise PreflightError("the %s's %s could not be recorded in its training row %s: %s" % (
            noun, kind, ab.get("stop_lines_row"), exc)) from exc
    if not written:
        raise PreflightError("the %s's %s could not be recorded: no single row of %s has id %s" % (
            noun, kind, registry, ab.get("stop_lines_row")))


def _as_adoption(texts):
    """The repair run's or continuation's texts said of the adoption."""
    return [t.replace("a repair run", "an adoption").replace("repair run", "adoption").replace(
        "a continuation", "an adoption").replace("the continuation", "the adoption") for t in texts]


def adoption_block_problems(ab, rec: dict, registry: Path = REGISTRY, report_every=None, run_steps=None,
                            noting=False) -> list:
    """What an adoption block lacks (THE ADOPTION in the module text): every field, text where text is named, the
    predecessor's, the code transition's and the decision's fields, role adoption, used once, the bound (max_model_steps
    is end_step - start_step, wall_cap_s above 0) and the exceptions' form {failure, relevance, why_measurable,
    evidence}; when written and at launch (not `noting`) also the stop criteria (checked as a continuation's at the
    launch's own report rate and length) and the run flags (both latent body flags, an existing resume file, an output
    folder that does not exist yet)."""
    if not isinstance(ab, dict):
        return ["the adoption must be a block {purpose, predecessor, code_transition, decision, authorization, ...}"]
    out = []
    # THE OWNER-BOUND RUN (THE TENTH AND THE ELEVENTH ENTRY in the module text): its route pins owner_bound, so
    # its block may name no watcher stop line
    owner_bound = (adoption_route(rec.get("fingerprint")) or {}).get("owner_bound") if isinstance(rec, dict) else None
    for k in ADOPTION_FIELDS:
        v = ab.get(k)
        if k == "stop_criteria" and owner_bound is not None and v == []:
            continue
        if v is None or (isinstance(v, (str, list, dict)) and not v) or (isinstance(v, str) and not v.strip()):
            out.append("the adoption's %s is missing" % k.replace("_", " "))
    if not isinstance(ab.get("exceptions"), dict):
        out.append("the adoption's exceptions must be a block {gate: {%s}} (reads %r)" % (
            ", ".join(ADOPTION_EXCEPTION_FIELDS), ab.get("exceptions")))
    for k in ADOPTION_TEXT_FIELDS:
        if ab.get(k) is not None and not isinstance(ab[k], str):
            out.append("the adoption's %s must be text (reads %s %r)" % (k.replace("_", " "), type(ab[k]).__name__,
                                                                        ab[k]))
    for key, fields in (("predecessor", ADOPTION_PREDECESSOR_FIELDS), ("code_transition", ADOPTION_CODE_FIELDS),
                        ("decision", ADOPTION_DECISION_FIELDS)):
        sub = ab.get(key)
        if sub is None or sub == {}:
            continue                                # named missing above
        if not isinstance(sub, dict):
            out.append("the adoption's %s must be a block {%s}" % (key.replace("_", " "), ", ".join(fields)))
            continue
        for f in fields:
            v = sub.get(f)
            if f == "commits":
                if not (isinstance(v, list) and v and all(_text(c) for c in v)):
                    out.append("the adoption's code transition names its commits as a list of text (reads %r)" % (v,))
            elif not _text(v):
                out.append("the adoption's %s %s must be text (reads %r)" % (key.replace("_", " "), f, v))
            elif f.endswith("sha256") and not _is_sha256(v):
                out.append("the adoption's %s %s must be a sha256, 64 lowercase hexadecimal characters (reads %r)" % (
                    key.replace("_", " "), f, v))
    ct = ab.get("code_transition") if isinstance(ab.get("code_transition"), dict) else {}
    if _text(ct.get("from_commit")) and ct.get("from_commit") == ct.get("to_commit"):
        out.append("the adoption's code transition names one commit as both its from_commit and its to_commit (%s)"
                   % ct["from_commit"])
    if ab.get("role", ADOPTION) != ADOPTION:
        out.append("the adoption's role reads %s (an adoption is its own route, role adoption)" % ab["role"])
    if "repeats" in ab and (ab["repeats"] != 1 or isinstance(ab["repeats"], bool)):
        out.append("an adoption is used once (repeats reads %r)" % (ab["repeats"],))
    for k in ("target_gates", "experiment"):
        if ab.get(k):
            out.append("an adoption targets no gate and is no experiment's arm (%s reads %s)" % (k, ab[k]))
    s, e, m = ab.get("start_step"), ab.get("end_step"), ab.get("max_model_steps")
    for k, v in (("start_step", s), ("end_step", e), ("max_model_steps", m)):
        if v is not None and not _whole(v):
            out.append("the adoption's %s must be a whole number of model steps (reads %r)" % (k, v))
    if _whole(s) and _whole(e) and _whole(m):
        if m <= 0 or e <= s:
            out.append("the adoption's duration is not bounded: start_step %d, end_step %d, max_model_steps %d" % (s, e, m))
        elif m != e - s:
            out.append("the adoption's max_model_steps %d is not its end_step minus its start_step (%d - %d = %d)" % (
                m, e, s, e - s))
    w = ab.get("wall_cap_s")
    if w is not None and not (_number(w) and w > 0):
        out.append("the adoption's wall_cap_s must be a number of seconds above 0 (reads %r)" % (w,))
    if not noting:
        if ab.get("stop_criteria"):
            out += stop_criteria_problems(ab["stop_criteria"])
            out += _as_adoption(stop_field_problems(ab))
            out += _as_adoption(first_reading_problems(ab, report_every, run_steps, timing=adoption_watch_timing(
                rec.get("fingerprint") if isinstance(rec, dict) else None)))
        out += _as_adoption(run_flag_problems(ab, rec))
        named = ab.get("run_flags") if isinstance(ab.get("run_flags"), dict) else None
        if named is not None:
            for k in LATENT_RUN_FLAGS:
                if not _text(named.get(k)):
                    out.append("the adoption's run_flags name %s (%s): the run resumes the predecessor's preserved "
                               "latent body state and writes a new folder" % (
                                   k, "the state it resumes" if k.endswith("resume") else "the folder it writes"))
    exceptions = ab.get("exceptions")
    for key, x in (exceptions.items() if isinstance(exceptions, dict) else ()):
        try:
            gate_n = int(key)
        except (TypeError, ValueError):
            out.append("the adoption's exception %r is not a gate number" % (key,))
            continue
        if gate_n not in [g[0] for g in GATES]:
            out.append("the adoption's exception %d is not one of the seven gates" % gate_n)
            continue
        if not isinstance(x, dict):
            out.append("the adoption's exception for gate %d must be {%s}" % (gate_n,
                                                                             ", ".join(ADOPTION_EXCEPTION_FIELDS)))
            continue
        for f in ADOPTION_EXCEPTION_FIELDS:
            if not (_evidence(x.get(f)) if f == "evidence" else _text(x.get(f))):
                out.append("the adoption's exception for gate %d lacks its %s" % (gate_n, f.replace("_", " ")))
        if "status" in x and x.get("status") != "FAIL":
            out.append("the adoption's exception for gate %d reads status %s: an exception leaves its gate FAIL, never "
                       "PASS or N/A" % (gate_n, x.get("status")))
    return out


def adoption_owner_problems(fingerprint, rec) -> list:
    """THE TENTH AND THE ELEVENTH ENTRY (the module text), when written, at launch and when an attempt or a launch is
    noted: a route that pins code_hash admits only a record holding that code hash; a route that pins owner_bound admits
    only while its pinned owner file hashes to the pinned owner_sha256."""
    route = adoption_route(fingerprint)
    if route is None:
        return []
    out = []
    if "code_hash" in route and (rec or {}).get("code_hash") != route["code_hash"]:
        out.append("the adoption route of configuration %s pins the trainer code hash %s; the record holds %s" % (
            fingerprint, route["code_hash"], (rec or {}).get("code_hash")))
    bound = route.get("owner_bound")
    if isinstance(bound, dict):
        owner = bound.get("owner")
        got = _file_sha256(owner) if _text(owner) and Path(owner).is_file() else None
        if got != bound.get("owner_sha256"):
            out.append("the owner-bound route of configuration %s pins its run owner %s at sha256 %s; that file reads %s"
                       % (fingerprint, owner, bound.get("owner_sha256"), got or "nothing (not a file)"))
    return out


def _owner_script(args):
    """The script a python command line runs: the first argument after the interpreter and its options (-W and -X take
    a value); None for -c or -m, or when there is none."""
    rest, i = [str(a) for a in list(args or [])[1:]], 0
    while i < len(rest):
        a = rest[i]
        if a in ("-c", "-m"):
            return None
        if a in ("-W", "-X"):
            i += 2
            continue
        if a.startswith("-") and len(a) > 1:
            i += 1
            continue
        return a
    return None


def _owner_process_problems(fingerprint, bound, pid, created, what) -> list:
    """[] when process `pid`, created at `created`, runs the pinned owner file of an owner-bound route as its script."""
    try:
        import psutil
        proc = psutil.Process(int(pid))
        args, born = [str(a) for a in proc.cmdline()], proc.create_time()
    except Exception as exc:                                                    # noqa: BLE001
        return ["the owner-bound route of configuration %s admits only its pinned run owner %s; %s (pid %r) cannot be "
                "read (%s: %s)" % (fingerprint, bound.get("owner"), what, pid, type(exc).__name__, exc)]
    out = []
    if not (_number(created) and abs(float(born) - float(created)) < 1.0):
        out.append("%s (pid %s) was created at %r, not at the recorded %r (another process)" % (what, pid, born, created))
    script = _owner_script(args)
    if script is None or _norm(script) != _norm(bound.get("owner") or ""):
        out.append("%s (pid %s) is not the pinned run owner %s of configuration %s: it runs %r (command line %r)" % (
            what, pid, bound.get("owner"), fingerprint, script, args[:6]))
    return out


def adoption_owner_claim_problems(fingerprint, claim) -> list:
    """THE TENTH AND THE ELEVENTH ENTRY, first at the note-attempt operation of an adoption: on an owner-bound route the
    attempt is noted only with a claim whose runner process (runner_pid, created at runner_pid_created) runs the pinned
    owner file as its script (the file's bytes adoption_owner_problems checks); [] on every other route."""
    route = adoption_route(fingerprint)
    bound = route.get("owner_bound") if isinstance(route, dict) else None
    if not isinstance(bound, dict):
        return []
    c = claim if isinstance(claim, dict) else {}
    return _owner_process_problems(fingerprint, bound, c.get("runner_pid"), c.get("runner_pid_created"),
                                   "the claim's runner process")


def adoption_owner_runner_problems(fingerprint, registry) -> list:
    """THE TENTH AND THE ELEVENTH ENTRY, first at the mark-launched operation of an adoption: on an owner-bound route the
    launch is marked only while the training row's recorded runner (last_run.runner_pid, created at
    runner_pid_created: the process that claimed the row) runs the pinned owner file as its script; [] on every other
    route."""
    route = adoption_route(fingerprint)
    bound = route.get("owner_bound") if isinstance(route, dict) else None
    if not isinstance(bound, dict):
        return []
    try:
        from mimo_bridge.tools import test_registry as TR
        rows = TR.rows_with_id(TR.load(Path(registry)), route["run_row"])
        last = rows[0].get("last_run") if len(rows) == 1 and isinstance(rows[0].get("last_run"), dict) else {}
    except Exception as exc:                                                    # noqa: BLE001
        return ["the owner-bound route of configuration %s: its training row %s cannot be read (%s: %s)" % (
            fingerprint, route["run_row"], type(exc).__name__, exc)]
    return _owner_process_problems(fingerprint, bound, last.get("runner_pid"), last.get("runner_pid_created"),
                                   "the training row's recorded runner")


def adoption_problems(fingerprint, held, rec, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY, code_hash=None,
                      report_every=None, run_steps=None, noting=False) -> list:
    """Every condition of THE ADOPTION (the module text) the adoption file `held` ({fingerprint, record_sha256, block})
    leaves unmet against its record `rec`: the block (adoption_block_problems), the route's pins
    (adoption_route_problems), the decision's message and the delegation record (adoption_decision_problems), the
    command template (adoption_template_problems), the preservation (adoption_preservation_problems), the predecessor
    (adoption_predecessor_problems), the authorization (adoption_authorization_problems) and the record
    (adoption_record_problems: `code_hash` is the checkout's trainer code); when written and at launch (not `noting`)
    also the preserved files' hashes (adoption_preserved_files_problems), the gates (adoption_gate_problems: each FAIL
    with its complete exception) and the registry rows (adoption_registry_problems)."""
    held = held if isinstance(held, dict) else {}
    ab = held.get("block") if isinstance(held.get("block"), dict) else {}
    out = []
    if held.get("fingerprint") != fingerprint:
        out.append("the adoption file names configuration %s, not %s" % (held.get("fingerprint"), fingerprint))
    out += adoption_block_problems(ab, rec, registry, report_every, run_steps, noting)
    out += adoption_route_problems(fingerprint, ab)
    route = adoption_route(fingerprint)
    if route is not None:
        out += adoption_decision_problems(route, ab)
        out += adoption_template_problems(route, rec, noting=noting)
    out += adoption_preservation_problems(ab)
    out += adoption_predecessor_problems(ab)
    out += adoption_authorization_problems(fingerprint, ab, root)
    out += adoption_record_problems(fingerprint, rec, ab, code_hash, held.get("record_sha256"), root)
    out += adoption_owner_problems(fingerprint, rec)      # THE TENTH AND THE ELEVENTH ENTRY
    if not noting:
        out += adoption_preserved_files_problems(ab)
        out += adoption_gate_problems(rec if isinstance(rec, dict) else {}, ab)
        out += adoption_registry_problems(fingerprint, ab, registry, "launch")
    return out


def _adoption_launch_folder_problems(flags: dict, route: dict) -> list:
    """THE LAUNCH'S OWN FOLDERS (the review's finding 3, GAP8): the launch names --out and --checkpoint-storage, each a
    folder that does not exist yet; neither they nor --latent-body-output lies in the route's preserved folder or in the
    preservation's own folder (the MANIFEST's)."""
    out = []
    keep = (route["preserved_root"], str(Path(route["manifest"]).parent))
    for flag in ("--out", "--checkpoint-storage", "--latent-body-output"):
        value = flags.get(flag)
        if not _text(value):                         # absent, or a flag with no value (flags_of reads True)
            if flag != "--latent-body-output":
                out.append("the launch names no %s: an adoption writes its run and its checkpoints into new folders" % flag)
            continue
        if flag != "--latent-body-output" and Path(value).exists():
            out.append("the launch's %s %s exists already: an adoption writes a new folder, never over saved work" % (
                flag, value))
        for folder in keep:
            if _under(value, folder):
                out.append("the launch's %s %s lies in the preserved folder %s: preserved state is never written" % (
                    flag, value, folder))
    return out


def _adoption_refusal(cmd, rec: dict, held: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the configuration's adoption and every condition holds (THE ADOPTION): the adoption's
    conditions at the launch's own report rate and length, the launch resuming exactly the start, the block's run flags
    carried exactly, every other flag as the command template's, its own new folders, and --max-steps reading exactly
    end_step."""
    ab = held.get("block") if isinstance(held.get("block"), dict) else {}
    flags = flags_of(cmd)
    start_step = _step_of(ab.get("start_checkpoint")) if _text(ab.get("start_checkpoint")) else None
    try:
        end = int(flags.get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    every = flags.get("--report-every")
    problems = adoption_problems(
        conf["fingerprint"], held, rec, root, registry, code_hash=conf["code_hash"],
        report_every=int(every) if isinstance(every, str) and every.isdigit() else TRAINER_REPORT_EVERY,
        run_steps=end - start_step if start_step is not None and end > start_step else None)
    route = adoption_route(conf["fingerprint"])
    if route is not None:
        problems += adoption_template_problems(route, rec, cmd=cmd, noting=True)
        problems += _adoption_launch_folder_problems(flags, route)
    resume = flags.get("--resume")
    if not isinstance(resume, str) or _norm(resume) != _norm(ab.get("start_checkpoint") or ""):
        problems.append("the launch resumes %s, not the adoption's start %s" % (resume, ab.get("start_checkpoint")))
    adopted_flags = ab.get("run_flags") if isinstance(ab.get("run_flags"), dict) else {}
    for k, v in sorted(adopted_flags.items()):
        if flags.get(k) != v:
            problems.append("the launch's %s reads %s; the adoption names %s (run_flags, compared exactly)" % (
                k, flags.get(k, "absent"), v))
    if not _whole(ab.get("end_step")) or end != ab["end_step"]:
        problems.append("the launch is not bounded as declared: --max-steps reads %s; the adoption ends exactly at model "
                        "step %s (its end_step: start %s + %s model steps)" % (
                            flags.get("--max-steps", "absent"), ab.get("end_step"), ab.get("start_step"),
                            ab.get("max_model_steps")))
    problems = list(dict.fromkeys(problems))         # a template line said twice (written and at launch) once
    if not problems:
        return None
    return "\n".join(["BLOCKED: adoption of configuration %s (%s): %d condition(s) unmet:" % (
        conf["fingerprint"], adoption_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


def _adoption_mismatch(conf: dict, resume, root: Path):
    """The refusal of a launch from a route's preserved checkpoint while that route's adoption is unused, when the
    launch's configuration is not the adoption's (CHANGED CODE OR SETTINGS, the module text: its code or a configuration
    flag differs, so its fingerprint differs), naming each difference; else None. The launch's --resume is compared with
    the route's checkpoint before the route's adoption file is read (the review's finding 6, GAP10: an unreadable file
    of the route then concerns only launches from that checkpoint)."""
    for route in ADOPTION_ROUTES:
        fp = route.get("fingerprint") if isinstance(route, dict) else None
        if not _is_fingerprint(fp) or fp == conf["fingerprint"]:
            continue
        if not isinstance(resume, str) or _norm(resume) != _norm(route.get("checkpoint") or ""):
            continue
        ab = adoption_of(fp, root).get("block") or {}
        if adoption_available(ab):
            rec = _load(record_path(fp, root)) or {}
            diff = differences(conf, rec) or ["(no flag or code difference found)"]
            return "\n".join(["BLOCKED: adoption of configuration %s (%s): the command's configuration %s is not the "
                              "adoption's; an adoption runs exactly its record's code and settings. It differs in:" % (
                                  fp, adoption_path(fp, root), conf["fingerprint"])] + ["  - " + d for d in diff])
    return None


def _own_adoption_at(conf: dict, resume, root: Path):
    """THE PAIR'S SIBLING ADOPTIONS (a2-admission-repair-authorized-20261023): the adoption file of the launch's own
    configuration when that configuration has its OWN bound route (ADOPTION_ROUTES) whose checkpoint is exactly the
    launch's --resume and its OWN adoption there is written, unused and starts exactly at --resume; else None. It grants
    nothing: launch_refusal then judges the launch by that adoption's full conditions (_adoption_refusal) alone."""
    route = adoption_route(conf["fingerprint"])
    if route is None or not isinstance(resume, str) or _norm(resume) != _norm(route.get("checkpoint") or ""):
        return None
    held = adoption_of(conf["fingerprint"], root)
    ab = held.get("block") if isinstance(held.get("block"), dict) else {}
    if not adoption_available(ab) or not _text(ab.get("start_checkpoint")) \
            or _norm(resume) != _norm(ab["start_checkpoint"]):
        return None
    return held


def set_adoption(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY,
                 src: Path = SRC):
    """Write the configuration's adoption (THE ADOPTION in the module text) in adoptions/<fingerprint>.json after checking
    it (adoption_problems, with the trainer code of `src`: the Source/Python folder of the checkout the trainer runs
    from). The record is read and never written: its gates stay as judged. Refused once the adoption was used (a marked
    launch, or an attempt that trained past its start); an unused one may be written again, its recorded attempts
    carried (as a continuation's). written_at_s ties each exception to the failure judged before it, and record_sha256
    ties the adoption to the record as it reads now."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    if not str(by or "").strip():
        raise PreflightError("an adoption needs who writes it (--by)")
    target = adoption_path(fingerprint, root)
    old = adoption_of(fingerprint, root).get("block") or {}
    if repair_uses(old):
        at, _at_s, run = repair_launch(old)
        raise PreflightError("the adoption of configuration %s was used (%s, run %s); an adoption is used once and a used "
                             "one is not written again" % (fingerprint, at or "an attempt trained past its start", run))
    ab = dict(block or {})
    if isinstance(ab.get("exceptions"), dict):
        ab["exceptions"] = {str(k): v for k, v in ab["exceptions"].items()}
    for k in ("attempts", "launches"):              # only the launch script records these (note_attempt, mark_launched)
        ab.pop(k, None)
    carried = [a for a in old.get("attempts") or [] if isinstance(a, dict)]
    if carried:                                     # attempts that never trained, kept: one that trains later is the use
        ab["attempts"] = carried
    ab.setdefault("role", ADOPTION)
    ab.update(written_at=clock(), written_at_s=time.time(), written_by=str(by), launched_at=None, launched_at_s=None,
              run=None)
    held = dict(fingerprint=fingerprint, record=str(path), record_sha256=_sha(path.read_bytes()),
                route=dict(adoption_route(fingerprint) or {}), written_at=ab["written_at"], written_by=str(by), block=ab)
    problems = adoption_problems(fingerprint, held, rec, root, registry, code_hash=trainer_code.trainer_code(src)[1])
    if problems:
        raise PreflightError("; ".join(problems))
    _save(target, held)
    return ab


# ------------------------------------------------------------------ THE LINKED RECOVERY OF A USED ADOPTION (the module text)
def recovery_path(fingerprint: str, root: Path = PREFLIGHT_DIR) -> Path:
    """The recovery file of configuration `fingerprint` (one per configuration)."""
    return Path(root) / RECOVERIES_DIR / ("%s.json" % fingerprint)


def recovery_of(fingerprint: str, root: Path = PREFLIGHT_DIR) -> dict:
    """The recovery file's content ({} when the configuration has none)."""
    held = _load(recovery_path(fingerprint, root))
    return held if isinstance(held, dict) else {}


def recovery_available(rb) -> bool:
    """True while a recovery block is written and unused (counted as a continuation's uses are: repair_uses)."""
    return isinstance(rb, dict) and bool(rb) and not repair_uses(rb)


def recovery_route(fingerprint):
    """The RECOVERY_ROUTES entry of configuration `fingerprint`, or None."""
    for route in RECOVERY_ROUTES:
        if isinstance(route, dict) and _is_fingerprint(route.get("fingerprint")) and route["fingerprint"] == fingerprint:
            return route
    return None


def _recovery_words(texts):
    """The texts of the adoption's checks, which the recovery reuses on its own block, said of the recovery."""
    out = []
    for t in texts:
        for old, new in (("training_preflight adoption", "training_preflight recovery"), ("An adoption", "A recovery"),
                         ("an adoption", "a recovery"), ("the adoption", "the recovery"), ("adoption's", "recovery's"),
                         ("a recovery is the one bounded launch of a record written for it",
                          "a recovery continues, unchanged, the record its used adoption was written against"),
                         ('(the decision: "Keep each gate FAIL',
                          '(the used adoption\'s decision, which the recovery keeps: "Keep each gate FAIL')):
            t = t.replace(old, new)
        out.append(t)
    return out


def recovery_route_problems(fingerprint, rb) -> list:
    """Why a recovery is not the one its route admits (RECOVERY_ROUTES; the module text), or []: the configuration's
    fingerprint is an entry's, and every value the entry pins is the block's, compared exactly: the start and the
    predecessor's checkpoint, the MANIFEST, the preserved run, the used adoption's run, the stopped receipt and the outcome
    receipt, the sha256 of the MANIFEST, the model, the worlds companion, the receipts and the used adoption's file, the
    used adoption's decision, the stop step, the decision, its message's text sha256 and its author, the wall cap, the
    start and end steps, the training row (the completion jobs' row is never it, nor the used adoption's training row), the
    commit of the trainer code, and the sha256 of the latent body state the run resumes."""
    route = recovery_route(fingerprint)
    if route is None:
        named = ", ".join(str(r.get("fingerprint")) for r in RECOVERY_ROUTES if isinstance(r, dict)) or "none"
        return ["the recovery route admits only the configurations RECOVERY_ROUTES names (%s); configuration %s is not "
                "one (another recovery is a new decision and a reviewed change to this code)" % (named, fingerprint)]
    rb = rb if isinstance(rb, dict) else {}
    pred = rb.get("predecessor") if isinstance(rb.get("predecessor"), dict) else {}
    dec = rb.get("decision") if isinstance(rb.get("decision"), dict) else {}
    code = rb.get("code") if isinstance(rb.get("code"), dict) else {}
    pin = "the recovery route of configuration %s" % fingerprint
    out = []
    for what, value in (("start checkpoint", rb.get("start_checkpoint")), ("predecessor's checkpoint",
                                                                           pred.get("checkpoint"))):
        if not isinstance(value, str) or _norm(value) != _norm(route["checkpoint"]):
            out.append("%s admits only the stopped run's preserved end checkpoint %s; the recovery's %s reads %s" % (
                pin, route["checkpoint"], what, value))
    for what, value, key in (("preservation MANIFEST", pred.get("manifest_path"), "manifest"),
                             ("preserved run", pred.get("run"), "predecessor_run"),
                             ("used adoption's run", pred.get("adoption_run"), "adoption_run"),
                             ("stopped receipt", pred.get("stopped_receipt"), "stopped_receipt"),
                             ("outcome receipt", pred.get("outcome_receipt"), "outcome_receipt")):
        if not isinstance(value, str) or _norm(value) != _norm(route[key]):
            out.append("%s pins the %s %s; the recovery names %s" % (pin, what, route[key], value))
    for what, key in (("preservation MANIFEST's", "manifest_sha256"), ("model's", "model_sha256"),
                      ("worlds companion's", "companion_sha256"), ("outcome receipt's", "outcome_receipt_sha256"),
                      ("stopped receipt's", "stopped_receipt_sha256"), ("used adoption file's", "adoption_sha256")):
        if pred.get(key) != route[key]:
            out.append("%s pins the %s sha256 %s; the recovery names %s" % (pin, what, route[key], pred.get(key)))
    if pred.get("adoption_decision") != route["adoption_decision"]:
        out.append("%s continues the adoption of decision %s; the recovery names %s" % (
            pin, route["adoption_decision"], pred.get("adoption_decision")))
    if pred.get("stopped_step") != route["start"] or not _whole(pred.get("stopped_step")):
        out.append("%s continues the run stopped at model step %d; the recovery's stopped_step reads %r" % (
            pin, route["start"], pred.get("stopped_step")))
    if dec.get("id") != route["decision"]:
        out.append("%s pins the decision %s; the recovery names %s" % (pin, route["decision"], dec.get("id")))
    if dec.get("message_sha256") != route["decision_text_sha256"]:
        out.append("%s pins the decision message's text sha256 %s; the recovery names %s" % (
            pin, route["decision_text_sha256"], dec.get("message_sha256")))
    if dec.get("author") != route["decision_author"]:
        out.append("%s pins the decision's author %r; the recovery names %r" % (pin, route["decision_author"],
                                                                               dec.get("author")))
    if not _number(rb.get("wall_cap_s")) or rb.get("wall_cap_s") != route["wall_cap_s"]:
        out.append("%s pins the remaining wall cap %s s; the recovery's wall_cap_s reads %r" % (
            pin, route["wall_cap_s"], rb.get("wall_cap_s")))
    if rb.get("start_step") != route["start"] or not _whole(rb.get("start_step")):
        out.append("%s starts at model step %d; the recovery's start_step reads %r" % (pin, route["start"],
                                                                                       rb.get("start_step")))
    if rb.get("end_step") != route["end"] or not _whole(rb.get("end_step")):
        out.append("%s ends at model step %d; the recovery's end_step reads %r" % (pin, route["end"], rb.get("end_step")))
    if rb.get("stop_lines_row") != route["run_row"]:
        out.append("%s pins the training row %s; the recovery's stop_lines_row reads %r (only that row may launch)" % (
            pin, route["run_row"], rb.get("stop_lines_row")))
    if rb.get("completion_jobs_row") in (route["run_row"], route["adoption_row"]):
        out.append("the recovery's completion_jobs_row names the training row %s: a completion row cannot stand in for a "
                   "training row (its own or the used adoption's)" % rb.get("completion_jobs_row"))
    if not _names_commit(route["code_commit"], code.get("commit")):
        out.append("%s runs the trainer code of commit %s; the recovery's code commit reads %r (the full hash or a prefix "
                   "of it of at least 9 characters)" % (pin, route["code_commit"], code.get("commit")))
    flags = rb.get("run_flags") if isinstance(rb.get("run_flags"), dict) else {}
    latent = flags.get("--latent-body-resume")
    got = _file_sha256(latent) if _text(latent) and Path(latent).is_file() else None
    if got != route["latent_sha256"]:
        out.append("%s pins the latent body state's sha256 %s; the state the recovery resumes, %s, reads %s" % (
            pin, route["latent_sha256"], latent, "sha256 %s" % got if got else "nothing (not a file)"))
    return out


def recovery_budget_problems(route: dict) -> list:
    """THE BUDGET (the module text; the decision, item 3): the run owner's log holds one "trainer started" event whose
    process creation time is the pinned one and one "trainer ended" event at the pinned time, and the pinned wall cap is
    the used adoption's cap less that elapsed time, rounded down."""
    out = []
    events = []
    try:
        with open(route["owner_log"], encoding="utf-8-sig") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict):
                    events.append(e)
    except OSError as exc:
        return ["the run owner's log %s cannot be read: %s, so the recovery's wall cap cannot be tied to the used "
                "adoption's trainer time" % (route["owner_log"], exc)]
    for name, key, pinned in (("trainer started", "created", route["trainer_created_s"]),
                              ("trainer ended", "at_s", route["trainer_exit_observed_s"])):
        found = [e.get(key) for e in events if e.get("event") == name]
        if found != [pinned]:
            out.append("the run owner's log %s records %s for its '%s' event(s); the recovery route pins one, its %s %r"
                       % (route["owner_log"], found or "none", name, key, pinned))
    elapsed = route["trainer_exit_observed_s"] - route["trainer_created_s"]
    left = int(math.floor(route["original_wall_cap_s"] - elapsed)) if elapsed > 0 else None
    if left is None or route["wall_cap_s"] != left:
        out.append("the recovery route's wall cap %s s is not the used adoption's %s s less its trainer's %.3f s (from "
                   "the process creation to the observed end), rounded down (%s s): the cap is what remains, never a "
                   "renewed allowance" % (route["wall_cap_s"], route["original_wall_cap_s"], elapsed, left))
    return out


def recovery_link_problems(fingerprint, rb, route, rec, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY) -> list:
    """THE USED ADOPTION (the module text): what links the recovery to the used adoption of its configuration, to its
    stopped run and its preserved end, and keeps the adoption's history as it reads: the adoption file, used, read byte
    for byte; the record unchanged since the adoption; the block's code hash the record's; the stop criteria and the
    authorized time; the start's lineage; the preservation's source, step, verdict, worlds companion and the preserved
    run's args.json; the stopped receipt; the INCOMPLETE outcome; the used adoption's registry rows."""
    rb = rb if isinstance(rb, dict) else {}
    pred = rb.get("predecessor") if isinstance(rb.get("predecessor"), dict) else {}
    out = []
    # the used adoption's file: named by the block, read byte for byte
    apath = adoption_path(fingerprint, root)
    if not _text(pred.get("adoption")) or _norm(pred["adoption"]) != _norm(apath):
        out.append("the recovery's predecessor names the adoption file %s; the used adoption of configuration %s is %s"
                   % (pred.get("adoption"), fingerprint, apath))
    try:
        data = apath.read_bytes()
    except OSError:
        data = None
    got = _sha(data) if data is not None else None
    if got != route["adoption_sha256"]:
        out.append("the used adoption's file %s reads %s; the recovery route pins sha256 %s: the used adoption is kept "
                   "byte for byte (its block, its attempts and its launch)" % (
                       apath, "sha256 %s" % got if got else "nothing (not a file)", route["adoption_sha256"]))
    try:
        held = json.loads(data.decode("utf-8-sig")) if data else {}
    except ValueError:
        held = {}
    held = held if isinstance(held, dict) else {}
    ab = held.get("block") if isinstance(held.get("block"), dict) else {}
    if not repair_uses(ab):
        out.append("the adoption of configuration %s was not used (no marked launch, no attempt that trained past its "
                   "start): a recovery continues a used adoption's stopped run, and an unused adoption launches as "
                   "itself" % fingerprint)
    elif not ab.get("launched_at") or _norm(ab.get("run") or "") != _norm(route["adoption_run"]):
        out.append("the used adoption's marked launch names run %s; the recovery route pins the run %s" % (
            ab.get("run") if ab.get("launched_at") else "none (no launch marked)", route["adoption_run"]))
    if (ab.get("decision") or {}).get("id") != route["adoption_decision"]:
        out.append("the used adoption names decision %s; the recovery route continues the adoption of decision %s" % (
            (ab.get("decision") or {}).get("id"), route["adoption_decision"]))
    if held.get("record_sha256") != route["record_sha256"]:
        out.append("the used adoption was written against the record sha256 %s; the recovery route pins %s" % (
            held.get("record_sha256"), route["record_sha256"]))
    if ab.get("start_step") != route["adoption_start"] or ab.get("end_step") != route["end"]:
        out.append("the used adoption ran from model step %s to %s; the recovery route continues the adoption from %d to "
                   "%d, and ends at its end" % (ab.get("start_step"), ab.get("end_step"), route["adoption_start"],
                                               route["end"]))
    if ab.get("wall_cap_s") != route["original_wall_cap_s"]:
        out.append("the used adoption's wall cap reads %r; the recovery's remaining cap is computed from the original %s s"
                   % (ab.get("wall_cap_s"), route["original_wall_cap_s"]))
    ct = ab.get("code_transition") if isinstance(ab.get("code_transition"), dict) else {}
    if not _names_commit(route["code_commit"], ct.get("to_commit")):
        out.append("the used adoption's code transition ends at %r; the recovery runs the code of commit %s, which must be "
                   "the adoption's own" % (ct.get("to_commit"), route["code_commit"]))
    # the record: unchanged since the adoption was written, its code the block's
    try:
        now = _sha(record_path(fingerprint, root).read_bytes())
    except OSError:
        now = None
    if now != route["record_sha256"]:
        out.append("the record %s reads sha256 %s; the used adoption was written against %s (the recovery route's pin): a "
                   "gate judged again or a state added since then is not carried by the recovery" % (
                       record_path(fingerprint, root), now, route["record_sha256"]))
    code = rb.get("code") if isinstance(rb.get("code"), dict) else {}
    if _text(code.get("code_hash")) and code.get("code_hash") != (rec or {}).get("code_hash"):
        out.append("the recovery's code names code hash %s; the record of configuration %s holds %s" % (
            code.get("code_hash"), fingerprint, (rec or {}).get("code_hash")))
    # the stop lines and the authorized time
    if "stop_criteria" in rb and rb.get("stop_criteria") != ab.get("stop_criteria"):
        out.append("the recovery's stop criteria are not the used adoption's: every declared stop line is kept as it was "
                   "(the decision, item 5: all other active stop thresholds remain unchanged)")
    mine, theirs = rb.get("authorized_at"), ab.get("authorized_at")
    if _text(mine) and _text(theirs) and " ".join(mine.split()) == " ".join(theirs.split()):
        out.append("the recovery's authorized_at repeats the used adoption's (%s): the recovery's decision is its own, at "
                   "its own time" % theirs)
    # the lineage: the start continues the model state the used adoption started from
    start = rb.get("start_checkpoint")
    if _text(start) and _text(ab.get("start_checkpoint")) and not continues(start, ab["start_checkpoint"]):
        out.append("the recovery's start %s does not continue the model state %s the used adoption started from (its "
                   "run's args.json resume chain): a recovery continues the same lineage" % (start,
                                                                                             ab["start_checkpoint"]))
    # the preservation: its source is the used adoption's run at the stop step; its companion; the preserved args.json
    manifest = _adoption_manifest(pred)[0] if _text(pred.get("manifest_path")) else None
    if manifest is not None:
        source = manifest.get("checkpoint")
        srun, sstep = _run_and_step(source) if _text(source) else (None, None)
        if srun is None or _norm(srun) != _norm(route["adoption_run"]) or sstep != route["start"] \
                or manifest.get("step") != route["start"]:
            out.append("the preservation MANIFEST %s records the source checkpoint %s at model step %s; the recovery's "
                       "start is the preserved copy of the used adoption's run %s at model step %d" % (
                           pred["manifest_path"], source, manifest.get("step"), route["adoption_run"], route["start"]))
        if manifest.get("verdict") != "pass":
            out.append("the preservation MANIFEST %s reads verdict %s; a recovery starts only from a preservation that "
                       "passed" % (pred["manifest_path"], manifest.get("verdict")))
        companions = sorted(str(f.get("sha256")) for f in manifest.get("files") or [] if isinstance(f, dict)
                            and str(f.get("path") or "").replace("\\", "/").startswith(ADOPTION_COMPANION_DIR))
        if manifest.get("companion_sha256") != route["companion_sha256"] or manifest.get("companion_equal") is not True \
                or manifest.get("companion_sha256_in_episode_state") != route["companion_sha256"] \
                or companions != [route["companion_sha256"]]:
            out.append("the preservation MANIFEST %s records the worlds companion %s (the checkpoint's episode_state.json "
                       "names %s, equal: %s; companion files %s); the recovery route pins sha256 %s" % (
                           pred["manifest_path"], manifest.get("companion_sha256"),
                           manifest.get("companion_sha256_in_episode_state"), manifest.get("companion_equal"),
                           companions or "none", route["companion_sha256"]))
        base = Path(pred["manifest_path"]).parent
        if _text(pred.get("run")) and _under(pred["run"], base):
            rel = os.path.relpath(_norm(Path(pred["run"]) / "args.json"), _norm(base)).replace("\\", "/")
            recorded = [f for f in manifest.get("files") or [] if isinstance(f, dict)
                        and str(f.get("path") or "").replace("\\", "/").lower() == rel.lower()]
            args_file = Path(pred["run"]) / "args.json"
            got = _file_sha256(args_file) if args_file.is_file() else None
            if not recorded or got != recorded[0].get("sha256"):
                out.append("the preserved run's args.json %s reads %s; the MANIFEST records %s" % (
                    args_file, "sha256 %s" % got if got else "nothing (not a file)",
                    recorded[0].get("sha256") if recorded else "no such file"))
            else:
                try:
                    args = json.loads(args_file.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    args = {}
                args = args if isinstance(args, dict) else {}
                if not _text(args.get("resume")) or _norm(args["resume"]) != _norm(ab.get("start_checkpoint") or "") \
                        or args.get("max_steps") != route["end"]:
                    out.append("the preserved run's args.json resumes %s and ends at %s; the used adoption started from "
                               "%s and was launched for model step %d" % (args.get("resume"), args.get("max_steps"),
                                                                          ab.get("start_checkpoint"), route["end"]))
    # the stopped receipt: the run owner stopped the run at the stop step, its trainer ended and its end was preserved
    receipt, receipt_sha = pred.get("stopped_receipt"), pred.get("stopped_receipt_sha256")
    if _text(receipt):
        got = _file_sha256(receipt) if Path(receipt).is_file() else None
        if got != receipt_sha:
            out.append("the stopped receipt %s reads %s; the recovery names sha256 %s" % (
                receipt, "sha256 %s" % got if got else "nothing (not a file)", receipt_sha))
        else:
            try:
                end = json.loads(Path(receipt).read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                end = {}
            end = end if isinstance(end, dict) else {}
            kept = end.get("preserved") if isinstance(end.get("preserved"), dict) else {}
            if end.get("state") != "stopped" or end.get("step") != route["start"] or end.get("trainer_exit_code") != 0 \
                    or kept.get("verdict") != "pass" or _norm(end.get("run") or "") != _norm(route["adoption_run"]) \
                    or _norm(end.get("preserved_checkpoint") or "") != _norm(route["checkpoint"]):
                out.append("the stopped receipt %s reads state %s at model step %s (run %s, trainer exit code %s, "
                           "preservation %s at %s); a recovery continues the run %s stopped at %d, its trainer ending with "
                           "exit code 0 and its end preserved at %s with verdict pass" % (
                               receipt, end.get("state"), end.get("step"), end.get("run"), end.get("trainer_exit_code"),
                               kept.get("verdict"), end.get("preserved_checkpoint"), route["adoption_run"],
                               route["start"], route["checkpoint"]))
    # the INCOMPLETE result: kept, never a manufactured success
    receipt = pred.get("outcome_receipt")
    if _text(receipt) and Path(receipt).is_file():
        try:
            outcome = json.loads(Path(receipt).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            outcome = {}
        text = str((outcome if isinstance(outcome, dict) else {}).get("outcome") or "")
        if INCOMPLETE not in text:
            out.append("the outcome receipt %s reads %r; the used adoption's result is %s and the recovery keeps it (it "
                       "makes no successful predecessor)" % (receipt, text[:200], INCOMPLETE))
    # the used adoption's registry rows keep their ends
    rows = _registry_rows(registry)
    arow = rows.get(route["adoption_row"])
    if arow is None:
        out.append("the used adoption's training row %s is not in the test registry %s" % (route["adoption_row"],
                                                                                         registry))
    else:
        kept = arow.get(ADOPTION_ROW_FIELD) if isinstance(arow.get(ADOPTION_ROW_FIELD), dict) else {}
        if kept.get("fingerprint") != fingerprint or not kept.get("launched_at") \
                or _norm(kept.get("run") or "") != _norm(route["adoption_run"]):
            out.append("the used adoption's training row %s does not record its launch of run %s (its record reads %s): "
                       "the adoption's use is kept in both places" % (route["adoption_row"], route["adoption_run"],
                                                                     arow.get(ADOPTION_ROW_FIELD)))
        last = arow.get("last_run") if isinstance(arow.get("last_run"), dict) else {}
        if arow.get("status") != "failed" or last.get("state") != "STOPPED":
            out.append("the used adoption's training row %s reads status %s and its runner's state %s; the stopped "
                       "adoption's row keeps status failed and the state STOPPED" % (
                           route["adoption_row"], arow.get("status"), last.get("state")))
    done_row = ab.get("completion_jobs_row")
    if _text(done_row):
        crow = rows.get(done_row)
        if crow is None:
            out.append("the used adoption's completion row %s is not in the test registry %s" % (done_row, registry))
        elif crow.get("status") != "failed":
            out.append("the used adoption's completion row %s reads status %s; its INCOMPLETE reading keeps status failed"
                       % (done_row, crow.get("status")))
        if rb.get("completion_jobs_row") == done_row:
            out.append("the recovery's completion jobs' row is the used adoption's %s: the recovery's completion jobs are "
                       "declared in a row of their own" % done_row)
    return out


def recovery_exception_problems(rb, ab) -> list:
    """THE GATES (the module text; the decision, item 7: "Retain the prior explicit gate exceptions and reconsider their
    applicability to this preserved same-lineage state"): each exception of the block carries the used adoption's
    exception for its gate unchanged (failure, relevance, why_measurable, evidence) and writes carried_from and its
    applicability beside it."""
    rb = rb if isinstance(rb, dict) else {}
    exceptions = rb.get("exceptions") if isinstance(rb.get("exceptions"), dict) else {}
    carried = (ab or {}).get("exceptions") if isinstance((ab or {}).get("exceptions"), dict) else {}
    out = []
    for n, name, _q in GATES:
        x, c = exceptions.get(str(n)), carried.get(str(n))
        if not isinstance(x, dict):
            continue                                # named by adoption_gate_problems
        if not isinstance(c, dict):
            out.append("gate %d %s: the used adoption names no exception for it, so none is carried" % (n, name))
            continue
        changed = [f for f in ADOPTION_EXCEPTION_FIELDS if x.get(f) != c.get(f)]
        if changed:
            out.append("the recovery's exception for gate %d %s changes the used adoption's %s: an inherited exception is "
                       "carried as written, its applicability to this state written beside it, never a new measurement"
                       % (n, name, ", ".join(changed)))
        for f in RECOVERY_CARRIED_FIELDS:
            if not _text(x.get(f)):
                out.append("the recovery's exception for gate %d %s lacks its %s" % (n, name, f.replace("_", " ")))
    return out


def recovery_authorization_problems(fingerprint, rb, root: Path = PREFLIGHT_DIR) -> list:
    """REUSED AUTHORIZATION, as for the adoption (_authorization_reuse_problems), on the recovery's own file: the used
    adoption's file is one of the blocks it is compared with."""
    return _authorization_reuse_problems(rb, recovery_path(fingerprint, root), root, RECOVERY)


def recovery_block_problems(rb, rec: dict, registry: Path = REGISTRY, report_every=None, run_steps=None,
                            noting=False) -> list:
    """What a recovery block lacks (the module text): every field, text where text is named, the predecessor's, the
    code's and the decision's fields, role recovery, used once, the bound (max_model_steps is end_step - start_step,
    wall_cap_s above 0) and the exceptions' form {failure, relevance, why_measurable, evidence, carried_from,
    applicability}; when written and at launch (not `noting`) also the stop criteria (checked as a continuation's at the
    launch's own report rate and length) and the run flags (both latent body flags, an existing resume file, an output
    folder that does not exist yet)."""
    if not isinstance(rb, dict):
        return ["the recovery must be a block {purpose, predecessor, code, decision, authorization, ...}"]
    out = []
    for k in RECOVERY_FIELDS:
        v = rb.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v) or (isinstance(v, str) and not v.strip()):
            out.append("the recovery's %s is missing" % k.replace("_", " "))
    if not isinstance(rb.get("exceptions"), dict):
        out.append("the recovery's exceptions must be a block {gate: {%s}} (reads %r)" % (
            ", ".join(ADOPTION_EXCEPTION_FIELDS + RECOVERY_CARRIED_FIELDS), rb.get("exceptions")))
    for k in ADOPTION_TEXT_FIELDS:
        if rb.get(k) is not None and not isinstance(rb[k], str):
            out.append("the recovery's %s must be text (reads %s %r)" % (k.replace("_", " "), type(rb[k]).__name__,
                                                                        rb[k]))
    for key, fields in (("predecessor", RECOVERY_PREDECESSOR_FIELDS), ("code", RECOVERY_CODE_FIELDS),
                        ("decision", ADOPTION_DECISION_FIELDS)):
        sub = rb.get(key)
        if sub is None or sub == {}:
            continue                                # named missing above
        if not isinstance(sub, dict):
            out.append("the recovery's %s must be a block {%s}" % (key, ", ".join(fields)))
            continue
        for f in fields:
            v = sub.get(f)
            if f == "stopped_step":
                if not _whole(v):
                    out.append("the recovery's predecessor stopped_step must be a whole number of model steps (reads %r)"
                               % (v,))
            elif not _text(v):
                out.append("the recovery's %s %s must be text (reads %r)" % (key, f, v))
            elif (f.endswith("sha256") or f == "code_hash") and not _is_sha256(v):
                out.append("the recovery's %s %s must be a sha256, 64 lowercase hexadecimal characters (reads %r)" % (
                    key, f, v))
    if rb.get("role", RECOVERY) != RECOVERY:
        out.append("the recovery's role reads %s (a recovery is its own route, role recovery)" % rb["role"])
    if "repeats" in rb and (rb["repeats"] != 1 or isinstance(rb["repeats"], bool)):
        out.append("a recovery is used once (repeats reads %r)" % (rb["repeats"],))
    for k in ("target_gates", "experiment"):
        if rb.get(k):
            out.append("a recovery targets no gate and is no experiment's arm (%s reads %s)" % (k, rb[k]))
    s, e, m = rb.get("start_step"), rb.get("end_step"), rb.get("max_model_steps")
    for k, v in (("start_step", s), ("end_step", e), ("max_model_steps", m)):
        if v is not None and not _whole(v):
            out.append("the recovery's %s must be a whole number of model steps (reads %r)" % (k, v))
    if _whole(s) and _whole(e) and _whole(m):
        if m <= 0 or e <= s:
            out.append("the recovery's duration is not bounded: start_step %d, end_step %d, max_model_steps %d" % (s, e, m))
        elif m != e - s:
            out.append("the recovery's max_model_steps %d is not its end_step minus its start_step (%d - %d = %d)" % (
                m, e, s, e - s))
    w = rb.get("wall_cap_s")
    if w is not None and not (_number(w) and w > 0):
        out.append("the recovery's wall_cap_s must be a number of seconds above 0 (reads %r)" % (w,))
    if not noting:
        if rb.get("stop_criteria"):
            out += stop_criteria_problems(rb["stop_criteria"])
            out += _recovery_words(_as_adoption(stop_field_problems(rb)))
            out += _recovery_words(_as_adoption(first_reading_problems(rb, report_every, run_steps)))
        out += _recovery_words(_as_adoption(run_flag_problems(rb, rec)))
        named = rb.get("run_flags") if isinstance(rb.get("run_flags"), dict) else None
        if named is not None:
            for k in LATENT_RUN_FLAGS:
                if not _text(named.get(k)):
                    out.append("the recovery's run_flags name %s (%s): the run resumes the stopped run's preserved "
                               "latent body state and writes a new folder" % (
                                   k, "the state it resumes" if k.endswith("resume") else "the folder it writes"))
    exceptions = rb.get("exceptions")
    for key, x in (exceptions.items() if isinstance(exceptions, dict) else ()):
        try:
            gate_n = int(key)
        except (TypeError, ValueError):
            out.append("the recovery's exception %r is not a gate number" % (key,))
            continue
        if gate_n not in [g[0] for g in GATES]:
            out.append("the recovery's exception %d is not one of the seven gates" % gate_n)
            continue
        if not isinstance(x, dict):
            out.append("the recovery's exception for gate %d must be {%s}" % (
                gate_n, ", ".join(ADOPTION_EXCEPTION_FIELDS + RECOVERY_CARRIED_FIELDS)))
            continue
        if "status" in x and x.get("status") != "FAIL":
            out.append("the recovery's exception for gate %d reads status %s: an exception leaves its gate FAIL, never "
                       "PASS or N/A" % (gate_n, x.get("status")))
    return out


def recovery_problems(fingerprint, held, rec, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY, code_hash=None,
                      report_every=None, run_steps=None, noting=False) -> list:
    """Every condition of THE LINKED RECOVERY (the module text) the recovery file `held` ({fingerprint, record_sha256,
    block}) leaves unmet against its record `rec`: the block (recovery_block_problems), the route's pins
    (recovery_route_problems), the budget (recovery_budget_problems), the decision's message and the delegation record
    (adoption_decision_problems), the command template (adoption_template_problems), the used adoption and the stopped
    run (recovery_link_problems), the preservation and the predecessor (adoption_preservation_problems,
    adoption_predecessor_problems), the authorization (recovery_authorization_problems) and the record
    (adoption_record_problems: `code_hash` is the checkout's trainer code); when written and at launch (not `noting`) also
    the preserved files' hashes (adoption_preserved_files_problems), the gates (adoption_gate_problems,
    recovery_exception_problems) and the registry rows (adoption_registry_problems, field RECOVERY_ROW_FIELD)."""
    held = held if isinstance(held, dict) else {}
    rb = held.get("block") if isinstance(held.get("block"), dict) else {}
    rec = rec if isinstance(rec, dict) else {}
    out = []
    if held.get("fingerprint") != fingerprint:
        out.append("the recovery file names configuration %s, not %s" % (held.get("fingerprint"), fingerprint))
    out += recovery_block_problems(rb, rec, registry, report_every, run_steps, noting)
    out += recovery_route_problems(fingerprint, rb)
    route = recovery_route(fingerprint)
    if route is not None:
        out += recovery_budget_problems(route)
        out += _recovery_words(adoption_decision_problems(route, rb))
        out += adoption_template_problems(route, rec, noting=noting)
        out += recovery_link_problems(fingerprint, rb, route, rec, root, registry)
        if held.get("record_sha256") != route["record_sha256"]:
            out.append("the recovery file was written against the record sha256 %s; the recovery route pins %s" % (
                held.get("record_sha256"), route["record_sha256"]))
    out += _recovery_words(adoption_preservation_problems(rb))
    out += _recovery_words(adoption_predecessor_problems(rb))
    out += recovery_authorization_problems(fingerprint, rb, root)
    out += _recovery_words(adoption_record_problems(fingerprint, rec, {}, code_hash, None, root))
    if not noting:
        out += _recovery_words(adoption_preserved_files_problems(rb))
        out += _recovery_words(adoption_gate_problems(rec, rb))
        out += recovery_exception_problems(rb, adoption_of(fingerprint, root).get("block") or {})
        out += adoption_registry_problems(fingerprint, rb, registry, "launch", field=RECOVERY_ROW_FIELD, noun=RECOVERY)
    return list(dict.fromkeys(out))


def _recovery_launch_folder_problems(flags: dict, route: dict, ab: dict) -> list:
    """THE LAUNCH'S OWN FOLDERS: as an adoption's (_adoption_launch_folder_problems: --out and --checkpoint-storage named
    and new, none of the three in the preservations), and none of --out, --checkpoint-storage and --latent-body-output in
    the used adoption's run folder or its latent body folder (the stopped run is kept as it is)."""
    out = _recovery_words(_adoption_launch_folder_problems(flags, route))
    kept = [(route["adoption_run"], "the used adoption's run folder")]
    latent = ((ab or {}).get("run_flags") or {}).get("--latent-body-output") if isinstance(ab, dict) else None
    if _text(latent):
        kept.append((latent, "the used adoption's latent body folder"))
    for flag in ("--out", "--checkpoint-storage", "--latent-body-output"):
        value = flags.get(flag)
        if not _text(value):
            continue
        for folder, what in kept:
            if _under(value, folder):
                out.append("the launch's %s %s lies in %s %s: the stopped run is kept as it is" % (flag, value, what,
                                                                                                  folder))
    return out


def _recovery_refusal(cmd, rec: dict, held: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the configuration's recovery and every condition holds (THE LINKED RECOVERY): the
    recovery's conditions at the launch's own report rate and length, the launch resuming exactly the start, the block's
    run flags carried exactly, every other flag as the command template's, its own new folders, and --max-steps reading
    exactly end_step."""
    rb = held.get("block") if isinstance(held.get("block"), dict) else {}
    flags = flags_of(cmd)
    start_step = _step_of(rb.get("start_checkpoint")) if _text(rb.get("start_checkpoint")) else None
    try:
        end = int(flags.get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    every = flags.get("--report-every")
    problems = recovery_problems(
        conf["fingerprint"], held, rec, root, registry, code_hash=conf["code_hash"],
        report_every=int(every) if isinstance(every, str) and every.isdigit() else TRAINER_REPORT_EVERY,
        run_steps=end - start_step if start_step is not None and end > start_step else None)
    route = recovery_route(conf["fingerprint"])
    if route is not None:
        problems += adoption_template_problems(route, rec, cmd=cmd, noting=True)
        problems += _recovery_launch_folder_problems(flags, route, adoption_of(conf["fingerprint"], root).get("block"))
    resume = flags.get("--resume")
    if not isinstance(resume, str) or _norm(resume) != _norm(rb.get("start_checkpoint") or ""):
        problems.append("the launch resumes %s, not the recovery's start %s" % (resume, rb.get("start_checkpoint")))
    named = rb.get("run_flags") if isinstance(rb.get("run_flags"), dict) else {}
    for k, v in sorted(named.items()):
        if flags.get(k) != v:
            problems.append("the launch's %s reads %s; the recovery names %s (run_flags, compared exactly)" % (
                k, flags.get(k, "absent"), v))
    if not _whole(rb.get("end_step")) or end != rb["end_step"]:
        problems.append("the launch is not bounded as declared: --max-steps reads %s; the recovery ends exactly at model "
                        "step %s (its end_step: start %s + %s model steps)" % (
                            flags.get("--max-steps", "absent"), rb.get("end_step"), rb.get("start_step"),
                            rb.get("max_model_steps")))
    problems = list(dict.fromkeys(problems))
    if not problems:
        return None
    return "\n".join(["BLOCKED: recovery of configuration %s (%s): %d condition(s) unmet:" % (
        conf["fingerprint"], recovery_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


def _recovery_mismatch(conf: dict, resume, root: Path):
    """The refusal of a launch from a recovery route's start checkpoint while that route's recovery is unused, when the
    launch's configuration is not the recovery's (CHANGED CODE OR SETTINGS: its code or a configuration flag differs, so
    its fingerprint differs), naming each difference; else None. The launch's --resume is compared with the route's
    checkpoint before the route's recovery file is read."""
    for route in RECOVERY_ROUTES:
        fp = route.get("fingerprint") if isinstance(route, dict) else None
        if not _is_fingerprint(fp) or fp == conf["fingerprint"]:
            continue
        if not isinstance(resume, str) or _norm(resume) != _norm(route.get("checkpoint") or ""):
            continue
        rb = recovery_of(fp, root).get("block") or {}
        if recovery_available(rb):
            rec = _load(record_path(fp, root)) or {}
            diff = differences(conf, rec) or ["(no flag or code difference found)"]
            return "\n".join(["BLOCKED: recovery of configuration %s (%s): the command's configuration %s is not the "
                              "recovery's; a recovery runs exactly its record's code and settings, those of its used "
                              "adoption. It differs in:" % (fp, recovery_path(fp, root), conf["fingerprint"])]
                             + ["  - " + d for d in diff])
    return None


def set_recovery(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY,
                 src: Path = SRC):
    """Write the configuration's recovery (THE LINKED RECOVERY in the module text) in recoveries/<fingerprint>.json after
    checking it (recovery_problems, with the trainer code of `src`: the Source/Python folder of the checkout the trainer
    runs from). The record and the used adoption's file are read and never written. Refused once the recovery was used (a
    marked launch, or an attempt that trained past its start); an unused one may be written again, its recorded attempts
    carried (as a continuation's). written_at_s ties each exception to the failure judged before it, and record_sha256
    ties the recovery to the record as it reads now."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    if not str(by or "").strip():
        raise PreflightError("a recovery needs who writes it (--by)")
    target = recovery_path(fingerprint, root)
    old = recovery_of(fingerprint, root).get("block") or {}
    if repair_uses(old):
        at, _at_s, run = repair_launch(old)
        raise PreflightError("the recovery of configuration %s was used (%s, run %s); a recovery is used once and a used "
                             "one is not written again" % (fingerprint, at or "an attempt trained past its start", run))
    rb = dict(block or {})
    if isinstance(rb.get("exceptions"), dict):
        rb["exceptions"] = {str(k): v for k, v in rb["exceptions"].items()}
    for k in ("attempts", "launches"):              # only the launch script records these (note_attempt, mark_launched)
        rb.pop(k, None)
    carried = [a for a in old.get("attempts") or [] if isinstance(a, dict)]
    if carried:                                     # attempts that never trained, kept: one that trains later is the use
        rb["attempts"] = carried
    rb.setdefault("role", RECOVERY)
    rb.update(written_at=clock(), written_at_s=time.time(), written_by=str(by), launched_at=None, launched_at_s=None,
              run=None)
    apath = adoption_path(fingerprint, root)
    held = dict(fingerprint=fingerprint, record=str(path), record_sha256=_sha(path.read_bytes()), adoption=str(apath),
                adoption_sha256=_sha(apath.read_bytes()) if apath.is_file() else None,
                route=dict(recovery_route(fingerprint) or {}), written_at=rb["written_at"], written_by=str(by), block=rb)
    problems = recovery_problems(fingerprint, held, rec, root, registry, code_hash=trainer_code.trainer_code(src)[1])
    if problems:
        raise PreflightError("; ".join(problems))
    _save(target, held)
    return rb


# ------------------------------------------------------------------ THE LINKED CONTINUATION OF A STOPPED RECOVERY
def link_path(fingerprint: str, root: Path = PREFLIGHT_DIR) -> Path:
    """The link file of configuration `fingerprint` (one per configuration)."""
    return Path(root) / LINKS_DIR / ("%s.json" % fingerprint)


def link_of(fingerprint: str, root: Path = PREFLIGHT_DIR) -> dict:
    """The link file's content ({} when the configuration has none)."""
    held = _load(link_path(fingerprint, root))
    return held if isinstance(held, dict) else {}


def link_available(lb) -> bool:
    """True while a link block is written and unused (counted as a continuation's uses are: repair_uses)."""
    return isinstance(lb, dict) and bool(lb) and not repair_uses(lb)


def link_route(fingerprint):
    """The LINK_ROUTES entry of configuration `fingerprint`, or None."""
    for route in LINK_ROUTES:
        if isinstance(route, dict) and _is_fingerprint(route.get("fingerprint")) and route["fingerprint"] == fingerprint:
            return route
    return None


def _link_words(texts):
    """The texts of the adoption's checks, which the link reuses on its own block, said of the link."""
    out = []
    for t in texts:
        for old, new in (("training_preflight adoption", "training_preflight link"), ("An adoption", "A link"),
                         ("an adoption", "a link"), ("the adoption", "the link"), ("adoption's", "link's"),
                         ("a link is the one bounded launch of a record written for it",
                          "a link continues, unchanged, the record its used adoption and recovery were written against"),
                         ('(the decision: "Keep each gate FAIL',
                          '(the used adoption\'s decision, which the link keeps: "Keep each gate FAIL')):
            t = t.replace(old, new)
        out.append(t)
    return out


def link_route_problems(fingerprint, lb) -> list:
    """Why a link is not the one its route admits (LINK_ROUTES; the module text), or []: the configuration's fingerprint
    is an entry's, and every value the entry pins is the block's, compared exactly: the start, rollback and
    predecessor's checkpoint, the MANIFEST, the preserved run, the used recovery's run, the stopped receipt and the
    outcome receipt, the sha256 of the MANIFEST, the model, the worlds companion, the receipts, the used recovery's and
    the used adoption's files, the used recovery's decision, the stop step, the decision, its message's text sha256, its
    author and its time, the wall cap, the start and end steps, the training row and the completion row, the commit of the
    trainer code, and the sha256 of the latent body state the run resumes."""
    route = link_route(fingerprint)
    if route is None:
        named = ", ".join(str(r.get("fingerprint")) for r in LINK_ROUTES if isinstance(r, dict)) or "none"
        return ["the link route admits only the configurations LINK_ROUTES names (%s); configuration %s is not one "
                "(another link is a new decision and a reviewed change to this code)" % (named, fingerprint)]
    lb = lb if isinstance(lb, dict) else {}
    pred = lb.get("predecessor") if isinstance(lb.get("predecessor"), dict) else {}
    dec = lb.get("decision") if isinstance(lb.get("decision"), dict) else {}
    code = lb.get("code") if isinstance(lb.get("code"), dict) else {}
    pin = "the link route of configuration %s" % fingerprint
    out = []
    for what, value in (("start checkpoint", lb.get("start_checkpoint")),
                        ("rollback checkpoint", lb.get("rollback_checkpoint")),
                        ("predecessor's checkpoint", pred.get("checkpoint"))):
        if not isinstance(value, str) or _norm(value) != _norm(route["checkpoint"]):
            out.append("%s admits only the stopped recovery run's preserved end checkpoint %s; the link's %s reads %s"
                       % (pin, route["checkpoint"], what, value))
    for what, value, key in (("preservation MANIFEST", pred.get("manifest_path"), "manifest"),
                             ("preserved run", pred.get("run"), "predecessor_run"),
                             ("used recovery's run", pred.get("recovery_run"), "recovery_run"),
                             ("stopped receipt", pred.get("stopped_receipt"), "stopped_receipt"),
                             ("outcome receipt", pred.get("outcome_receipt"), "outcome_receipt")):
        if not isinstance(value, str) or _norm(value) != _norm(route[key]):
            out.append("%s pins the %s %s; the link names %s" % (pin, what, route[key], value))
    for what, key in (("preservation MANIFEST's", "manifest_sha256"), ("model's", "model_sha256"),
                      ("worlds companion's", "companion_sha256"), ("outcome receipt's", "outcome_receipt_sha256"),
                      ("stopped receipt's", "stopped_receipt_sha256"), ("used recovery file's", "recovery_sha256"),
                      ("used adoption file's", "adoption_sha256")):
        if pred.get(key) != route[key]:
            out.append("%s pins the %s sha256 %s; the link names %s" % (pin, what, route[key], pred.get(key)))
    if pred.get("recovery_decision") != route["recovery_decision"]:
        out.append("%s continues the recovery of decision %s; the link names %s" % (
            pin, route["recovery_decision"], pred.get("recovery_decision")))
    if pred.get("stopped_step") != route["start"] or not _whole(pred.get("stopped_step")):
        out.append("%s continues the recovery run stopped at model step %d; the link's stopped_step reads %r" % (
            pin, route["start"], pred.get("stopped_step")))
    if dec.get("id") != route["decision"]:
        out.append("%s pins the decision %s; the link names %s" % (pin, route["decision"], dec.get("id")))
    if dec.get("message_sha256") != route["decision_text_sha256"]:
        out.append("%s pins the decision message's text sha256 %s; the link names %s" % (
            pin, route["decision_text_sha256"], dec.get("message_sha256")))
    if dec.get("author") != route["decision_author"]:
        out.append("%s pins the decision's author %r; the link names %r" % (pin, route["decision_author"],
                                                                           dec.get("author")))
    if lb.get("authorized_at") != route["authorized_at"]:
        out.append("%s pins the decision's time %r (its message's created_at in PDT); the link's authorized_at reads %r"
                   % (pin, route["authorized_at"], lb.get("authorized_at")))
    if not _number(lb.get("wall_cap_s")) or lb.get("wall_cap_s") != route["wall_cap_s"]:
        out.append("%s pins the remaining wall cap %s s; the link's wall_cap_s reads %r" % (
            pin, route["wall_cap_s"], lb.get("wall_cap_s")))
    if lb.get("start_step") != route["start"] or not _whole(lb.get("start_step")):
        out.append("%s starts at model step %d; the link's start_step reads %r" % (pin, route["start"],
                                                                                   lb.get("start_step")))
    if lb.get("end_step") != route["end"] or not _whole(lb.get("end_step")):
        out.append("%s ends at model step %d; the link's end_step reads %r" % (pin, route["end"], lb.get("end_step")))
    if lb.get("stop_lines_row") != route["run_row"]:
        out.append("%s pins the training row %s; the link's stop_lines_row reads %r (only that row may launch)" % (
            pin, route["run_row"], lb.get("stop_lines_row")))
    if lb.get("completion_jobs_row") != route["completion_row"]:
        out.append("%s pins the completion row %s; the link's completion_jobs_row reads %r" % (
            pin, route["completion_row"], lb.get("completion_jobs_row")))
    if not _names_commit(route["code_commit"], code.get("commit")):
        out.append("%s runs the trainer code of commit %s; the link's code commit reads %r (the full hash or a prefix of "
                   "it of at least 9 characters)" % (pin, route["code_commit"], code.get("commit")))
    flags = lb.get("run_flags") if isinstance(lb.get("run_flags"), dict) else {}
    latent = flags.get("--latent-body-resume")
    got = _file_sha256(latent) if _text(latent) and Path(latent).is_file() else None
    if got != route["latent_sha256"]:
        out.append("%s pins the latent body state's sha256 %s; the state the link resumes, %s, reads %s" % (
            pin, route["latent_sha256"], latent, "sha256 %s" % got if got else "nothing (not a file)"))
    return out


def link_budget_problems(route: dict) -> list:
    """THE BUDGET (the module text; the decision, item 5: "remaining time allowance calculated from actual trainer
    lifetimes and the existing budget"): each run owner's log the route names holds one "trainer started" event whose
    process creation time is the pinned one and one "trainer ended" event at the pinned time; the pinned wall cap is the
    adoption's cap less both trainers' lifetimes, rounded down, and the used recovery's cap is the adoption's less the
    first, rounded down."""
    out = []
    for log, created, ended in route["trainers"]:
        events = []
        try:
            with open(log, encoding="utf-8-sig") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(e, dict):
                        events.append(e)
        except OSError as exc:
            out.append("the run owner's log %s cannot be read: %s, so the link's wall cap cannot be tied to that "
                       "trainer's lifetime" % (log, exc))
            continue
        for name, key, pinned in (("trainer started", "created", created), ("trainer ended", "at_s", ended)):
            found = [e.get(key) for e in events if e.get("event") == name]
            if found != [pinned]:
                out.append("the run owner's log %s records %s for its '%s' event(s); the link route pins one, its %s %r"
                           % (log, found or "none", name, key, pinned))
    lives = [ended - created for _log, created, ended in route["trainers"]]
    if not lives or any(x <= 0 for x in lives):
        return out + ["the link route's trainer lifetimes %s are not each above 0 s" % lives]
    left = int(math.floor(route["original_wall_cap_s"] - sum(lives)))
    if route["wall_cap_s"] != left:
        out.append("the link route's wall cap %s s is not the adoption's %s s less both trainers' %s s (each from its "
                   "process creation to its observed end), rounded down (%s s): the cap is what remains, never a renewed "
                   "allowance" % (route["wall_cap_s"], route["original_wall_cap_s"],
                                  " + ".join("%.3f" % x for x in lives), left))
    first = int(math.floor(route["original_wall_cap_s"] - lives[0]))
    if route["recovery_wall_cap_s"] != first:
        out.append("the link route's recovery cap %s s is not the adoption's %s s less the first trainer's %.3f s, "
                   "rounded down (%s s)" % (route["recovery_wall_cap_s"], route["original_wall_cap_s"], lives[0], first))
    return out


def link_clock_problems(route: dict, lb) -> list:
    """THE CARRIED CLOCK (the module text): the block's carried_clock names the route's carried runs, oldest first, and
    its seconds; each run is a preserved copy (under the route's preserved folder) whose stats hold a full record with
    its training seconds; the newest records' training seconds summed are the pinned seconds (the hourly watcher's
    wrapper adds them before this run's own, Carry.E of the recovery's watch/recover_hourly_watch.py)."""
    lb = lb if isinstance(lb, dict) else {}
    clock = lb.get("carried_clock") if isinstance(lb.get("carried_clock"), dict) else {}
    pin = "the link route of configuration %s" % route["fingerprint"]
    out = []
    runs, pinned = clock.get("runs"), [str(r) for r in route["carried_runs"]]
    if not isinstance(runs, list) or len(runs) != len(pinned) or not all(
            isinstance(r, str) and _norm(r) == _norm(p) for r, p in zip(runs, pinned)):
        out.append("%s pins the hourly watcher's carried runs %s (oldest first); the link's carried_clock names %r" % (
            pin, pinned, runs))
    if not _number(clock.get("seconds")) or clock.get("seconds") != route["carried_s"]:
        out.append("%s pins the carried clock %r s; the link's carried_clock reads %r" % (pin, route["carried_s"],
                                                                                         clock.get("seconds")))
    total, readable = 0.0, True
    for run in pinned:
        if not _under(run, route["preserved_root"]):
            out.append("the carried run %s is not a preserved copy (under %s): the watcher reads the lineage's "
                       "preserved runs" % (run, route["preserved_root"]))
        records = stats_records(run)
        last = records[-1].get("elapsed_s") if records else None
        if not _number(last):
            out.append("the carried run %s holds no full stats record with its training seconds (elapsed_s)" % run)
            readable = False
            continue
        total += float(last)
    if readable and abs(total - float(route["carried_s"])) > 1e-6:
        out.append("the carried runs' newest full stats records read %r training seconds in all; %s pins %r" % (
            total, pin, route["carried_s"]))
    return out


def link_first_readings(lb: dict, carried_s=0.0, report_every=None):
    """first_readings on the hourly watcher's carried clock: the watcher's run-hour is the newest record's training
    seconds, raised by `carried_s` (the training seconds of the preserved runs its wrapper carries), so an entry's
    from_run_hour is reached after its hours less carried_s of this run's own; with carried_s 0 the same numbers as
    first_readings. {stop entry's name: the most model steps the run can have taken when the watcher first judges that
    entry and the trainer has read a stop it writes}, or a text saying what cannot be read."""
    start_run, start_step = _run_and_step(lb.get("start_checkpoint") or "")
    rate = fastest_rate(stats_records(start_run))
    if rate is None or start_step is None:
        return "the start checkpoint's run (%s) does not give its pace between two full stats records" % start_run
    every = _report_every(start_run, report_every)
    to_first_record = every - (start_step % every) if start_step % every else every
    after = lambda seconds: int(math.ceil(rate * seconds))                        # noqa: E731
    carried = float(carried_s or 0.0)
    out = {}
    for c in lb.get("stop_criteria") or []:
        if not isinstance(c, dict) or c.get("mode") != "stop":
            continue
        try:
            hour = float(c.get("from_run_hour", 0) or 0)
        except (TypeError, ValueError):
            hour = 0.0
        own_s = hour * 3600 - carried
        steps = to_first_record + after(WATCH_POLL_S + WATCH_FIRST_CHECK_S)
        if own_s > 0:
            steps = max(steps, after(own_s + WATCH_FIRST_CHECK_S) + every)
        if str(c.get("new_field") or "").strip():
            steps = max(steps, max(to_first_record + after(ABSENT_FIELD_WINDOW_S + WATCH_FIRST_CHECK_S),
                                   after(max(own_s, 0.0) + ABSENT_FIELD_WINDOW_S + WATCH_FIRST_CHECK_S)) + every)
        out[str(c.get("name"))] = steps + CONTROL_POLL_STEPS
    return out


def link_first_reading_problems(lb: dict, carried_s=0.0, report_every=None, run_steps=None) -> list:
    """Each stop entry the watcher would first judge only after the link ends (link_first_readings on the carried
    clock): no such line would be judged while the run trains. The run ends `run_steps` model steps after its start (a
    launch's --max-steps minus the start step), else at its declared bound (max_model_steps)."""
    if not lb.get("start_checkpoint"):
        return []
    try:
        steps = int(run_steps if run_steps is not None else lb.get("max_model_steps") or 0)
    except (TypeError, ValueError):
        return []
    readings = link_first_readings(lb, carried_s, report_every)
    if isinstance(readings, str):
        return [readings + ", so when the watcher first reads the stop lines cannot be computed"]
    return ["the link ends %d model steps after its start, before the watcher first judges stop line %s (its run-hour "
            "counting the %.1f s it carries) and the trainer reads its stop, up to %d model steps in at the start run's "
            "fastest pace" % (steps, name, float(carried_s or 0.0), first)
            for name, first in readings.items() if steps <= first]


def link_policy_problems(route: dict, lb) -> list:
    """THE STOP-POLICY EXCEPTION (the module text; the decision, item 3): the block's stop_policy_exception is one block
    of its named fields, each text; its rule, watch, sense, crossing, mode and decision words are the route's, compared
    exactly; the decision words lie in the decision message's text."""
    lb = lb if isinstance(lb, dict) else {}
    pol = lb.get("stop_policy_exception")
    if pol is None or pol == {}:
        return []                                   # named missing by link_block_problems
    if not isinstance(pol, dict):
        return ["the link's stop_policy_exception must be a block {%s} (reads %s)" % (", ".join(LINK_POLICY_FIELDS),
                                                                                    type(pol).__name__)]
    want = route["stop_policy"]
    out = []
    for f in LINK_POLICY_FIELDS:
        if not _text(pol.get(f)):
            out.append("the link's stop_policy_exception lacks its %s" % f.replace("_", " "))
    for f in LINK_POLICY_PINNED:
        if _text(pol.get(f)) and pol.get(f) != want[f]:
            out.append("the link route of configuration %s pins the stop-policy exception's %s %r; the link's reads %r"
                       % (route["fingerprint"], f.replace("_", " "), want[f], pol.get(f)))
    beyond = sorted(str(k) for k in pol if k not in LINK_POLICY_FIELDS)
    if beyond:
        out.append("the link's stop_policy_exception names %s beyond its fields {%s}: the decision makes one scoped "
                   "exception" % (", ".join(beyond), ", ".join(LINK_POLICY_FIELDS)))
    try:
        text = json.loads(Path(route["decision_message"]).read_text(encoding="utf-8-sig")).get("text")
    except (OSError, ValueError, AttributeError):
        text = None                                 # named by adoption_decision_problems
    if isinstance(text, str) and want["decision_words"] not in text:
        out.append("the link route's stop-policy words are not in the decision message %s: the exception is the "
                   "decision's own words, verbatim" % route["decision_message"])
    return out


def _stopped_receipt_problems(path, sha256, what, run, step, preserved_checkpoint, namer="the link") -> list:
    """A stopped run's receipt (the run owner's arm_end.json), read when it reads `sha256`: state stopped at `step`, its
    trainer's exit code 0, its end preserved with verdict pass at `preserved_checkpoint`, its run `run`."""
    if not _text(path):
        return []
    got = _file_sha256(path) if Path(path).is_file() else None
    if got != sha256:
        return ["%s %s reads %s; %s names sha256 %s" % (what, path, "sha256 %s" % got if got else
                                                       "nothing (not a file)", namer, sha256)]
    try:
        end = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        end = {}
    end = end if isinstance(end, dict) else {}
    kept = end.get("preserved") if isinstance(end.get("preserved"), dict) else {}
    if end.get("state") != "stopped" or end.get("step") != step or end.get("trainer_exit_code") != 0 \
            or kept.get("verdict") != "pass" or _norm(end.get("run") or "") != _norm(run) \
            or _norm(end.get("preserved_checkpoint") or "") != _norm(preserved_checkpoint):
        return ["%s %s reads state %s at model step %s (run %s, trainer exit code %s, preservation %s at %s); the link "
                "continues the run %s stopped at %s, its trainer ending with exit code 0 and its end preserved at %s with "
                "verdict pass" % (what, path, end.get("state"), end.get("step"), end.get("run"),
                                  end.get("trainer_exit_code"), kept.get("verdict"), end.get("preserved_checkpoint"),
                                  run, step, preserved_checkpoint)]
    return []


def _incomplete_problems(path, sha256, what, namer="the link") -> list:
    """An outcome receipt that reads `sha256` and whose outcome text holds INCOMPLETE (kept, never a manufactured
    success)."""
    if not _text(path):
        return []
    got = _file_sha256(path) if Path(path).is_file() else None
    if got != sha256:
        return ["%s %s reads %s; %s names sha256 %s" % (what, path, "sha256 %s" % got if got else
                                                       "nothing (not a file)", namer, sha256)]
    try:
        outcome = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        outcome = {}
    text = str((outcome if isinstance(outcome, dict) else {}).get("outcome") or "")
    if INCOMPLETE not in text:
        return ["%s %s reads %r; that run's result is %s and the link keeps it (it makes no successful predecessor)" % (
            what, path, text[:200], INCOMPLETE)]
    return []


def link_parent_problems(fingerprint, lb, route, rec, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY) -> list:
    """THE CHAIN (the module text): what links the link to the used recovery of its configuration and, through it, to
    the used adoption, to the stopped recovery run and its preserved end, and keeps both earlier blocks' history as it
    reads: the recovery's and the adoption's files, used, read byte for byte; the record unchanged; the block's code hash
    the record's; the stop criteria and the authorized time; the start's lineage; the preservation's source, step,
    verdict, worlds companion and the preserved run's args.json; the stopped receipt; the INCOMPLETE outcome; the
    adoption's receipts as the recovery names them; both earlier blocks' registry rows."""
    lb = lb if isinstance(lb, dict) else {}
    pred = lb.get("predecessor") if isinstance(lb.get("predecessor"), dict) else {}
    out = []
    files = {}
    for key, what, path in (("recovery", "used recovery", recovery_path(fingerprint, root)),
                            ("adoption", "used adoption", adoption_path(fingerprint, root))):
        if not _text(pred.get(key)) or _norm(pred[key]) != _norm(path):
            out.append("the link's predecessor names the %s file %s; the %s of configuration %s is %s" % (
                what, pred.get(key), what, fingerprint, path))
        try:
            data = path.read_bytes()
        except OSError:
            data = None
        got = _sha(data) if data is not None else None
        if got != route["%s_sha256" % key]:
            out.append("the %s's file %s reads %s; the link route pins sha256 %s: the %s is kept byte for byte (its "
                       "block, its attempts and its launch)" % (what, path, "sha256 %s" % got if got else
                                                                "nothing (not a file)", route["%s_sha256" % key], what))
        try:
            held = json.loads(data.decode("utf-8-sig")) if data else {}
        except ValueError:
            held = {}
        held = held if isinstance(held, dict) else {}
        files[key] = (held, held.get("block") if isinstance(held.get("block"), dict) else {})
    rheld, rb = files["recovery"]
    _aheld, ab = files["adoption"]
    rpred = rb.get("predecessor") if isinstance(rb.get("predecessor"), dict) else {}
    # the used recovery: used, its launch marked for the pinned run; its decision, record, span, cap and code
    if not repair_uses(rb):
        out.append("the recovery of configuration %s was not used (no marked launch, no attempt that trained past its "
                   "start): a link continues a used recovery's stopped run, and an unused recovery launches as itself"
                   % fingerprint)
    elif not rb.get("launched_at") or _norm(rb.get("run") or "") != _norm(route["recovery_run"]):
        out.append("the used recovery's marked launch names run %s; the link route pins the run %s" % (
            rb.get("run") if rb.get("launched_at") else "none (no launch marked)", route["recovery_run"]))
    if (rb.get("decision") or {}).get("id") != route["recovery_decision"]:
        out.append("the used recovery names decision %s; the link route continues the recovery of decision %s" % (
            (rb.get("decision") or {}).get("id"), route["recovery_decision"]))
    if rheld.get("record_sha256") != route["record_sha256"]:
        out.append("the used recovery was written against the record sha256 %s; the link route pins %s" % (
            rheld.get("record_sha256"), route["record_sha256"]))
    if rb.get("start_step") != route["recovery_start"] or rb.get("end_step") != route["end"]:
        out.append("the used recovery ran from model step %s to %s; the link route continues the recovery from %d to "
                   "%d, and ends at its end" % (rb.get("start_step"), rb.get("end_step"), route["recovery_start"],
                                               route["end"]))
    if rb.get("wall_cap_s") != route["recovery_wall_cap_s"]:
        out.append("the used recovery's wall cap reads %r; the link route pins %s s" % (rb.get("wall_cap_s"),
                                                                                     route["recovery_wall_cap_s"]))
    if rb.get("stop_lines_row") != route["recovery_row"]:
        out.append("the used recovery's training row reads %r; the link route pins %s" % (rb.get("stop_lines_row"),
                                                                                        route["recovery_row"]))
    if not _names_commit(route["code_commit"], (rb.get("code") or {}).get("commit")):
        out.append("the used recovery ran the code of commit %r; the link runs the code of commit %s, which must be the "
                   "recovery's own" % ((rb.get("code") or {}).get("commit"), route["code_commit"]))
    # the used adoption before it: used, its launch marked for the run the recovery names, its original cap
    if not repair_uses(ab) or not ab.get("launched_at") \
            or _norm(ab.get("run") or "") != _norm(rpred.get("adoption_run") or ""):
        out.append("the used adoption's marked launch names run %s; the used recovery continues the adoption run %s" % (
            ab.get("run") if ab.get("launched_at") else "none (no launch marked)", rpred.get("adoption_run")))
    if ab.get("wall_cap_s") != route["original_wall_cap_s"]:
        out.append("the used adoption's wall cap reads %r; the link's remaining cap is computed from the original %s s"
                   % (ab.get("wall_cap_s"), route["original_wall_cap_s"]))
    # the record: unchanged since the adoption was written, its code the block's
    try:
        now = _sha(record_path(fingerprint, root).read_bytes())
    except OSError:
        now = None
    if now != route["record_sha256"]:
        out.append("the record %s reads sha256 %s; the used adoption and recovery were written against %s (the link "
                   "route's pin): a gate judged again or a state added since then is not carried by the link" % (
                       record_path(fingerprint, root), now, route["record_sha256"]))
    code = lb.get("code") if isinstance(lb.get("code"), dict) else {}
    if _text(code.get("code_hash")) and code.get("code_hash") != (rec or {}).get("code_hash"):
        out.append("the link's code names code hash %s; the record of configuration %s holds %s" % (
            code.get("code_hash"), fingerprint, (rec or {}).get("code_hash")))
    # the stop lines and the authorized time
    if "stop_criteria" in lb and lb.get("stop_criteria") != rb.get("stop_criteria"):
        out.append("the link's stop criteria are not the used recovery's: every declared stop line is kept as it was (the "
                   "decision, item 3: all other current stop rules remain stops)")
    mine = lb.get("authorized_at")
    for theirs, whose in ((rb.get("authorized_at"), "the used recovery's"), (ab.get("authorized_at"),
                                                                            "the used adoption's")):
        if _text(mine) and _text(theirs) and " ".join(mine.split()) == " ".join(theirs.split()):
            out.append("the link's authorized_at repeats %s (%s): the link's decision is its own, at its own time" % (
                whose, theirs))
    # the lineage: the start continues the recovery's start and the record's first model state
    start = lb.get("start_checkpoint")
    first = next((s.get("checkpoint") for s in (rec or {}).get("states") or [] if isinstance(s, dict)), None)
    for state, what in ((rb.get("start_checkpoint"), "the used recovery started from"),
                        (first, "the record's gates were judged on")):
        if _text(start) and _text(state) and not continues(start, state):
            out.append("the link's start %s does not continue the model state %s %s (its run's args.json resume chain): a "
                       "link continues the same lineage" % (start, state, what))
    # the preservation: its source is the recovery's run at the stop step; its companion; the preserved args.json
    manifest = _adoption_manifest(pred)[0] if _text(pred.get("manifest_path")) else None
    if manifest is not None:
        source = manifest.get("checkpoint")
        srun, sstep = _run_and_step(source) if _text(source) else (None, None)
        if srun is None or _norm(srun) != _norm(route["recovery_run"]) or sstep != route["start"] \
                or manifest.get("step") != route["start"]:
            out.append("the preservation MANIFEST %s records the source checkpoint %s at model step %s; the link's start "
                       "is the preserved copy of the used recovery's run %s at model step %d" % (
                           pred["manifest_path"], source, manifest.get("step"), route["recovery_run"], route["start"]))
        if manifest.get("verdict") != "pass":
            out.append("the preservation MANIFEST %s reads verdict %s; a link starts only from a preservation that passed"
                       % (pred["manifest_path"], manifest.get("verdict")))
        companions = sorted(str(f.get("sha256")) for f in manifest.get("files") or [] if isinstance(f, dict)
                            and str(f.get("path") or "").replace("\\", "/").startswith(ADOPTION_COMPANION_DIR))
        if manifest.get("companion_sha256") != route["companion_sha256"] or manifest.get("companion_equal") is not True \
                or manifest.get("companion_sha256_in_episode_state") != route["companion_sha256"] \
                or companions != [route["companion_sha256"]]:
            out.append("the preservation MANIFEST %s records the worlds companion %s (the checkpoint's episode_state.json "
                       "names %s, equal: %s; companion files %s); the link route pins sha256 %s" % (
                           pred["manifest_path"], manifest.get("companion_sha256"),
                           manifest.get("companion_sha256_in_episode_state"), manifest.get("companion_equal"),
                           companions or "none", route["companion_sha256"]))
        base = Path(pred["manifest_path"]).parent
        if _text(pred.get("run")) and _under(pred["run"], base):
            rel = os.path.relpath(_norm(Path(pred["run"]) / "args.json"), _norm(base)).replace("\\", "/")
            recorded = [f for f in manifest.get("files") or [] if isinstance(f, dict)
                        and str(f.get("path") or "").replace("\\", "/").lower() == rel.lower()]
            args_file = Path(pred["run"]) / "args.json"
            got = _file_sha256(args_file) if args_file.is_file() else None
            if not recorded or got != recorded[0].get("sha256"):
                out.append("the preserved run's args.json %s reads %s; the MANIFEST records %s" % (
                    args_file, "sha256 %s" % got if got else "nothing (not a file)",
                    recorded[0].get("sha256") if recorded else "no such file"))
            else:
                try:
                    args = json.loads(args_file.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    args = {}
                args = args if isinstance(args, dict) else {}
                if not _text(args.get("resume")) or _norm(args["resume"]) != _norm(rb.get("start_checkpoint") or "") \
                        or args.get("max_steps") != route["end"]:
                    out.append("the preserved run's args.json resumes %s and ends at %s; the used recovery started from "
                               "%s and was launched for model step %d" % (args.get("resume"), args.get("max_steps"),
                                                                          rb.get("start_checkpoint"), route["end"]))
    # the stopped receipt and the INCOMPLETE outcome of the recovery run; the adoption run's, as the recovery names them
    out += _stopped_receipt_problems(pred.get("stopped_receipt"), pred.get("stopped_receipt_sha256"),
                                     "the stopped receipt", route["recovery_run"], route["start"], route["checkpoint"])
    out += _incomplete_problems(pred.get("outcome_receipt"), pred.get("outcome_receipt_sha256"), "the outcome receipt")
    out += _stopped_receipt_problems(rpred.get("stopped_receipt"), rpred.get("stopped_receipt_sha256"),
                                     "the used adoption's stopped receipt", rpred.get("adoption_run"),
                                     rpred.get("stopped_step"), rpred.get("checkpoint"), namer="the used recovery")
    out += _incomplete_problems(rpred.get("outcome_receipt"), rpred.get("outcome_receipt_sha256"),
                                "the used adoption's outcome receipt", namer="the used recovery")
    # both earlier blocks' registry rows keep their ends
    rows = _registry_rows(registry)
    for block, field, what in ((rb, RECOVERY_ROW_FIELD, "used recovery"), (ab, ADOPTION_ROW_FIELD, "used adoption")):
        run_row, done_row = block.get("stop_lines_row"), block.get("completion_jobs_row")
        row = rows.get(run_row) if _text(run_row) else None
        if row is None:
            out.append("the %s's training row %s is not in the test registry %s" % (what, run_row, registry))
        else:
            kept = row.get(field) if isinstance(row.get(field), dict) else {}
            if kept.get("fingerprint") != fingerprint or not kept.get("launched_at") \
                    or _norm(kept.get("run") or "") != _norm(block.get("run") or ""):
                out.append("the %s's training row %s does not record its launch of run %s (its record reads %s): the "
                           "%s's use is kept in both places" % (what, run_row, block.get("run"), row.get(field), what))
            last = row.get("last_run") if isinstance(row.get("last_run"), dict) else {}
            if row.get("status") != "failed" or last.get("state") != "STOPPED":
                out.append("the %s's training row %s reads status %s and its runner's state %s; the stopped run's row "
                           "keeps status failed and the state STOPPED" % (what, run_row, row.get("status"),
                                                                          last.get("state")))
        crow = rows.get(done_row) if _text(done_row) else None
        if crow is None:
            out.append("the %s's completion row %s is not in the test registry %s" % (what, done_row, registry))
        elif crow.get("status") != "failed":
            out.append("the %s's completion row %s reads status %s; its INCOMPLETE reading keeps status failed" % (
                what, done_row, crow.get("status")))
        for key in ("stop_lines_row", "completion_jobs_row"):
            if _text(lb.get(key)) and lb.get(key) in (run_row, done_row):
                out.append("the link's %s is the %s's row %s: the link's rows are its own" % (key, what, lb[key]))
    return out


def link_exception_problems(lb, rb) -> list:
    """THE GATES (the module text; the decision, item 4: "Preserve the old stop and INCOMPLETE verdict"): each exception
    of the block carries the used recovery's exception for its gate unchanged (failure, relevance, why_measurable,
    evidence) and writes carried_from and its applicability beside it."""
    lb = lb if isinstance(lb, dict) else {}
    exceptions = lb.get("exceptions") if isinstance(lb.get("exceptions"), dict) else {}
    carried = (rb or {}).get("exceptions") if isinstance((rb or {}).get("exceptions"), dict) else {}
    out = []
    for n, name, _q in GATES:
        x, c = exceptions.get(str(n)), carried.get(str(n))
        if not isinstance(x, dict):
            continue                                # named by adoption_gate_problems
        if not isinstance(c, dict):
            out.append("gate %d %s: the used recovery names no exception for it, so none is carried" % (n, name))
            continue
        changed = [f for f in ADOPTION_EXCEPTION_FIELDS if x.get(f) != c.get(f)]
        if changed:
            out.append("the link's exception for gate %d %s changes the used recovery's %s: an inherited exception is "
                       "carried as written, its applicability to this state written beside it, never a new measurement"
                       % (n, name, ", ".join(changed)))
        for f in RECOVERY_CARRIED_FIELDS:
            if not _text(x.get(f)):
                out.append("the link's exception for gate %d %s lacks its %s" % (n, name, f.replace("_", " ")))
    return out


def link_authorization_problems(fingerprint, lb, root: Path = PREFLIGHT_DIR) -> list:
    """REUSED AUTHORIZATION, as for the adoption (_authorization_reuse_problems), on the link's own file: the used
    adoption's and the used recovery's files are among the blocks it is compared with."""
    return _authorization_reuse_problems(lb, link_path(fingerprint, root), root, LINK)


def link_block_problems(lb, rec: dict, registry: Path = REGISTRY, report_every=None, run_steps=None, noting=False,
                        carried_s=0.0) -> list:
    """What a link block lacks (the module text): every field, text where text is named, the predecessor's, the code's,
    the decision's and the carried clock's fields, role link, used once, the bound (max_model_steps is end_step -
    start_step, wall_cap_s above 0) and the exceptions' form; when written and at launch (not `noting`) also the stop
    criteria (their form, their fields in the start run's stats, and each judged before the end on the carried clock,
    `carried_s`, at the launch's own report rate and length) and the run flags (both latent body flags, an existing
    resume file, an output folder that does not exist yet)."""
    if not isinstance(lb, dict):
        return ["the link must be a block {purpose, predecessor, code, decision, authorization, ...}"]
    out = []
    for k in LINK_FIELDS:
        v = lb.get(k)
        if v is None or (isinstance(v, (str, list, dict)) and not v) or (isinstance(v, str) and not v.strip()):
            out.append("the link's %s is missing" % k.replace("_", " "))
    if not isinstance(lb.get("exceptions"), dict):
        out.append("the link's exceptions must be a block {gate: {%s}} (reads %r)" % (
            ", ".join(ADOPTION_EXCEPTION_FIELDS + RECOVERY_CARRIED_FIELDS), lb.get("exceptions")))
    for k in ADOPTION_TEXT_FIELDS:
        if lb.get(k) is not None and not isinstance(lb[k], str):
            out.append("the link's %s must be text (reads %s %r)" % (k.replace("_", " "), type(lb[k]).__name__, lb[k]))
    for key, fields in (("predecessor", LINK_PREDECESSOR_FIELDS), ("code", RECOVERY_CODE_FIELDS),
                        ("decision", ADOPTION_DECISION_FIELDS)):
        sub = lb.get(key)
        if sub is None or sub == {}:
            continue                                # named missing above
        if not isinstance(sub, dict):
            out.append("the link's %s must be a block {%s}" % (key, ", ".join(fields)))
            continue
        for f in fields:
            v = sub.get(f)
            if f == "stopped_step":
                if not _whole(v):
                    out.append("the link's predecessor stopped_step must be a whole number of model steps (reads %r)"
                               % (v,))
            elif not _text(v):
                out.append("the link's %s %s must be text (reads %r)" % (key, f, v))
            elif (f.endswith("sha256") or f == "code_hash") and not _is_sha256(v):
                out.append("the link's %s %s must be a sha256, 64 lowercase hexadecimal characters (reads %r)" % (
                    key, f, v))
    clock = lb.get("carried_clock")
    if clock is not None and clock != {}:
        if not isinstance(clock, dict):
            out.append("the link's carried_clock must be a block {%s}" % ", ".join(LINK_CLOCK_FIELDS))
        else:
            if not isinstance(clock.get("runs"), list) or not all(_text(r) for r in clock["runs"]):
                out.append("the link's carried_clock runs must be a list of run folders (reads %r)" % (clock.get("runs"),))
            if not (_number(clock.get("seconds")) and clock["seconds"] >= 0):
                out.append("the link's carried_clock seconds must be a number of seconds, 0 or more (reads %r)" % (
                    clock.get("seconds"),))
            if not _text(clock.get("basis")):
                out.append("the link's carried_clock lacks its basis")
    if lb.get("role", LINK) != LINK:
        out.append("the link's role reads %s (a link is its own route, role link)" % lb["role"])
    if "repeats" in lb and (lb["repeats"] != 1 or isinstance(lb["repeats"], bool)):
        out.append("a link is used once (repeats reads %r)" % (lb["repeats"],))
    for k in ("target_gates", "experiment"):
        if lb.get(k):
            out.append("a link targets no gate and is no experiment's arm (%s reads %s)" % (k, lb[k]))
    s, e, m = lb.get("start_step"), lb.get("end_step"), lb.get("max_model_steps")
    for k, v in (("start_step", s), ("end_step", e), ("max_model_steps", m)):
        if v is not None and not _whole(v):
            out.append("the link's %s must be a whole number of model steps (reads %r)" % (k, v))
    if _whole(s) and _whole(e) and _whole(m):
        if m <= 0 or e <= s:
            out.append("the link's duration is not bounded: start_step %d, end_step %d, max_model_steps %d" % (s, e, m))
        elif m != e - s:
            out.append("the link's max_model_steps %d is not its end_step minus its start_step (%d - %d = %d)" % (
                m, e, s, e - s))
    w = lb.get("wall_cap_s")
    if w is not None and not (_number(w) and w > 0):
        out.append("the link's wall_cap_s must be a number of seconds above 0 (reads %r)" % (w,))
    if not noting:
        if lb.get("stop_criteria"):
            out += stop_criteria_problems(lb["stop_criteria"])
            out += _link_words(_as_adoption(stop_field_problems(lb)))
            out += link_first_reading_problems(lb, carried_s, report_every, run_steps)
        out += _link_words(_as_adoption(run_flag_problems(lb, rec)))
        named = lb.get("run_flags") if isinstance(lb.get("run_flags"), dict) else None
        if named is not None:
            for k in LATENT_RUN_FLAGS:
                if not _text(named.get(k)):
                    out.append("the link's run_flags name %s (%s): the run resumes the stopped recovery run's preserved "
                               "latent body state and writes a new folder" % (
                                   k, "the state it resumes" if k.endswith("resume") else "the folder it writes"))
    exceptions = lb.get("exceptions")
    for key, x in (exceptions.items() if isinstance(exceptions, dict) else ()):
        try:
            gate_n = int(key)
        except (TypeError, ValueError):
            out.append("the link's exception %r is not a gate number" % (key,))
            continue
        if gate_n not in [g[0] for g in GATES]:
            out.append("the link's exception %d is not one of the seven gates" % gate_n)
            continue
        if not isinstance(x, dict):
            out.append("the link's exception for gate %d must be {%s}" % (
                gate_n, ", ".join(ADOPTION_EXCEPTION_FIELDS + RECOVERY_CARRIED_FIELDS)))
            continue
        if "status" in x and x.get("status") != "FAIL":
            out.append("the link's exception for gate %d reads status %s: an exception leaves its gate FAIL, never PASS "
                       "or N/A" % (gate_n, x.get("status")))
    return out


def link_problems(fingerprint, held, rec, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY, code_hash=None,
                  report_every=None, run_steps=None, noting=False) -> list:
    """Every condition of THE LINKED CONTINUATION (the module text) the link file `held` ({fingerprint, record_sha256,
    block}) leaves unmet against its record `rec`: the block (link_block_problems, on the route's carried clock), the
    route's pins (link_route_problems), the budget (link_budget_problems), the decision's message and the delegation
    record (adoption_decision_problems), the command template (adoption_template_problems), the chain (the used recovery,
    the used adoption and the stopped recovery run: link_parent_problems), the carried clock (link_clock_problems), the
    stop-policy exception (link_policy_problems), the preservation and the predecessor (adoption_preservation_problems,
    adoption_predecessor_problems), the authorization (link_authorization_problems) and the record
    (adoption_record_problems: `code_hash` is the checkout's trainer code); when written and at launch (not `noting`) also
    the preserved files' hashes (adoption_preserved_files_problems), the gates (adoption_gate_problems,
    link_exception_problems) and the registry rows (adoption_registry_problems, field LINK_ROW_FIELD)."""
    held = held if isinstance(held, dict) else {}
    lb = held.get("block") if isinstance(held.get("block"), dict) else {}
    rec = rec if isinstance(rec, dict) else {}
    route = link_route(fingerprint)
    out = []
    if held.get("fingerprint") != fingerprint:
        out.append("the link file names configuration %s, not %s" % (held.get("fingerprint"), fingerprint))
    out += link_block_problems(lb, rec, registry, report_every, run_steps, noting,
                               carried_s=route["carried_s"] if route is not None else 0.0)
    out += link_route_problems(fingerprint, lb)
    if route is not None:
        out += link_budget_problems(route)
        out += _link_words(adoption_decision_problems(route, lb))
        out += adoption_template_problems(route, rec, noting=noting)
        out += link_parent_problems(fingerprint, lb, route, rec, root, registry)
        out += link_clock_problems(route, lb)
        out += link_policy_problems(route, lb)
        if held.get("record_sha256") != route["record_sha256"]:
            out.append("the link file was written against the record sha256 %s; the link route pins %s" % (
                held.get("record_sha256"), route["record_sha256"]))
    out += _link_words(adoption_preservation_problems(lb))
    out += _link_words(adoption_predecessor_problems(lb))
    out += link_authorization_problems(fingerprint, lb, root)
    out += _link_words(adoption_record_problems(fingerprint, rec, {}, code_hash, None, root))
    if not noting:
        out += _link_words(adoption_preserved_files_problems(lb))
        out += _link_words(adoption_gate_problems(rec, lb))
        out += link_exception_problems(lb, recovery_of(fingerprint, root).get("block") or {})
        out += adoption_registry_problems(fingerprint, lb, registry, "launch", field=LINK_ROW_FIELD, noun=LINK)
    return list(dict.fromkeys(out))


def _link_launch_folder_problems(flags: dict, route: dict, root: Path) -> list:
    """THE LAUNCH'S OWN FOLDERS: as an adoption's (_adoption_launch_folder_problems: --out and --checkpoint-storage
    named and new, none of the three in the preservations); none of --out, --checkpoint-storage and
    --latent-body-output in the used recovery's or the used adoption's run folder or latent body folder (the stopped
    runs are kept as they are); and --out's folder name not either earlier run's, since the trainer writes the worlds
    companions into <--caregiver-record-dir>/checkpoint_state/<--out's folder name> (the recovery route's independent
    review, finding m6)."""
    out = _link_words(_adoption_launch_folder_problems(flags, route))
    rb = recovery_of(route["fingerprint"], root).get("block") or {}
    ab = adoption_of(route["fingerprint"], root).get("block") or {}
    runs = [(route["recovery_run"], "the used recovery's run folder")]
    if _text(ab.get("run")):
        runs.append((ab["run"], "the used adoption's run folder"))
    kept = list(runs)
    for block, whose in ((rb, "the used recovery's"), (ab, "the used adoption's")):
        latent = (block.get("run_flags") or {}).get("--latent-body-output") if isinstance(block, dict) else None
        if _text(latent):
            kept.append((latent, "%s latent body folder" % whose))
    for flag in ("--out", "--checkpoint-storage", "--latent-body-output"):
        value = flags.get(flag)
        if not _text(value):
            continue
        for folder, what in kept:
            if _under(value, folder):
                out.append("the launch's %s %s lies in %s %s: the stopped runs are kept as they are" % (flag, value, what,
                                                                                                        folder))
    if _text(flags.get("--out")):
        name = Path(str(flags["--out"]).replace("\\", "/").rstrip("/")).name
        for run, what in runs:
            if name.lower() == Path(str(run).replace("\\", "/").rstrip("/")).name.lower():
                out.append("the launch's --out %s has the folder name of %s %s: the trainer writes the worlds companions "
                           "into <--caregiver-record-dir>/checkpoint_state/%s, which holds that run's" % (
                               flags["--out"], what, run, name))
    return out


def _link_refusal(cmd, rec: dict, held: dict, conf: dict, root: Path, registry: Path):
    """None when this launch is the configuration's link and every condition holds (THE LINKED CONTINUATION): the
    link's conditions at the launch's own report rate and length, the launch resuming exactly the start, the block's run
    flags carried exactly, every other flag as the command template's, its own new folders, and --max-steps reading
    exactly end_step."""
    lb = held.get("block") if isinstance(held.get("block"), dict) else {}
    flags = flags_of(cmd)
    start_step = _step_of(lb.get("start_checkpoint")) if _text(lb.get("start_checkpoint")) else None
    try:
        end = int(flags.get("--max-steps") or 0)
    except (TypeError, ValueError):
        end = 0
    every = flags.get("--report-every")
    problems = link_problems(
        conf["fingerprint"], held, rec, root, registry, code_hash=conf["code_hash"],
        report_every=int(every) if isinstance(every, str) and every.isdigit() else TRAINER_REPORT_EVERY,
        run_steps=end - start_step if start_step is not None and end > start_step else None)
    route = link_route(conf["fingerprint"])
    if route is not None:
        problems += adoption_template_problems(route, rec, cmd=cmd, noting=True)
        problems += _link_launch_folder_problems(flags, route, root)
    resume = flags.get("--resume")
    if not isinstance(resume, str) or _norm(resume) != _norm(lb.get("start_checkpoint") or ""):
        problems.append("the launch resumes %s, not the link's start %s" % (resume, lb.get("start_checkpoint")))
    named = lb.get("run_flags") if isinstance(lb.get("run_flags"), dict) else {}
    for k, v in sorted(named.items()):
        if flags.get(k) != v:
            problems.append("the launch's %s reads %s; the link names %s (run_flags, compared exactly)" % (
                k, flags.get(k, "absent"), v))
    if not _whole(lb.get("end_step")) or end != lb["end_step"]:
        problems.append("the launch is not bounded as declared: --max-steps reads %s; the link ends exactly at model step "
                        "%s (its end_step: start %s + %s model steps)" % (
                            flags.get("--max-steps", "absent"), lb.get("end_step"), lb.get("start_step"),
                            lb.get("max_model_steps")))
    problems = list(dict.fromkeys(problems))
    if not problems:
        return None
    return "\n".join(["BLOCKED: link of configuration %s (%s): %d condition(s) unmet:" % (
        conf["fingerprint"], link_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


def _link_mismatch(conf: dict, resume, root: Path):
    """The refusal of a launch from a link route's start checkpoint while that route's link is unused, when the launch's
    configuration is not the link's (CHANGED CODE OR SETTINGS: its code or a configuration flag differs, so its
    fingerprint differs), naming each difference; else None. The launch's --resume is compared with the route's
    checkpoint before the route's link file is read."""
    for route in LINK_ROUTES:
        fp = route.get("fingerprint") if isinstance(route, dict) else None
        if not _is_fingerprint(fp) or fp == conf["fingerprint"]:
            continue
        if not isinstance(resume, str) or _norm(resume) != _norm(route.get("checkpoint") or ""):
            continue
        lb = link_of(fp, root).get("block") or {}
        if link_available(lb):
            rec = _load(record_path(fp, root)) or {}
            diff = differences(conf, rec) or ["(no flag or code difference found)"]
            return "\n".join(["BLOCKED: link of configuration %s (%s): the command's configuration %s is not the link's; "
                              "a link runs exactly its record's code and settings, those of its used recovery and "
                              "adoption. It differs in:" % (fp, link_path(fp, root), conf["fingerprint"])]
                             + ["  - " + d for d in diff])
    return None


def set_link(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY,
             src: Path = SRC):
    """Write the configuration's link (THE LINKED CONTINUATION in the module text) in links/<fingerprint>.json after
    checking it (link_problems, with the trainer code of `src`: the Source/Python folder of the checkout the trainer runs
    from). The record, the used adoption's file and the used recovery's file are read and never written. Refused once the
    link was used (a marked launch, or an attempt that trained past its start); an unused one may be written again, its
    recorded attempts carried (as a continuation's). written_at_s ties each exception to the failure judged before it,
    and record_sha256 ties the link to the record as it reads now."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    if not str(by or "").strip():
        raise PreflightError("a link needs who writes it (--by)")
    target = link_path(fingerprint, root)
    old = link_of(fingerprint, root).get("block") or {}
    if repair_uses(old):
        at, _at_s, run = repair_launch(old)
        raise PreflightError("the link of configuration %s was used (%s, run %s); a link is used once and a used one is "
                             "not written again" % (fingerprint, at or "an attempt trained past its start", run))
    lb = dict(block or {})
    if isinstance(lb.get("exceptions"), dict):
        lb["exceptions"] = {str(k): v for k, v in lb["exceptions"].items()}
    for k in ("attempts", "launches"):              # only the launch script records these (note_attempt, mark_launched)
        lb.pop(k, None)
    carried = [a for a in old.get("attempts") or [] if isinstance(a, dict)]
    if carried:                                     # attempts that never trained, kept: one that trains later is the use
        lb["attempts"] = carried
    lb.setdefault("role", LINK)
    lb.update(written_at=clock(), written_at_s=time.time(), written_by=str(by), launched_at=None, launched_at_s=None,
              run=None)
    apath, rpath = adoption_path(fingerprint, root), recovery_path(fingerprint, root)
    held = dict(fingerprint=fingerprint, record=str(path), record_sha256=_sha(path.read_bytes()), adoption=str(apath),
                adoption_sha256=_sha(apath.read_bytes()) if apath.is_file() else None, recovery=str(rpath),
                recovery_sha256=_sha(rpath.read_bytes()) if rpath.is_file() else None,
                route=dict(link_route(fingerprint) or {}), written_at=lb["written_at"], written_by=str(by), block=lb)
    problems = link_problems(fingerprint, held, rec, root, registry, code_hash=trainer_code.trainer_code(src)[1])
    if problems:
        raise PreflightError("; ".join(problems))
    _save(target, held)
    return lb


def launch_plan(cmd, root: Path = PREFLIGHT_DIR, src: Path = SRC) -> dict:
    """{"kind": "repair_run" | "continuation" | "normal", "block", "fingerprint", "start_step", "max_steps",
    "stop_criteria"}: a launch whose --resume is the start checkpoint of its record's repair run, not yet launched, is
    that repair run (block "repair_run"); one whose --resume is the start checkpoint of its record's continuation, not
    yet launched, is that continuation (block "continuation", THE CONTINUATION). For either the relaunch script sets
    --max-steps to the bound, writes the stop criteria into the run folder's declared_criteria.json, notes and marks the
    launch in the named block, and leaves the supervisor paused with the run named (no automatic continuation). A launch
    whose --resume is the start checkpoint of one unused fresh instance of the configuration is that instance (kind
    "repair_run", block "instance:<experiment id>", THE FRESH INSTANCE), bounded and noted as a repair run is. A launch
    whose --resume is the start checkpoint of the configuration's unused extension is that extension (kind
    "continuation", block "extension", THE EXTENSION), bounded and noted as the continuation is. A launch whose --resume
    is the start checkpoint of the configuration's unused adoption is that adoption (kind "continuation", block
    "adoption", THE ADOPTION; max_steps its end_step, wall_cap_s its wall cap), bounded and noted as the continuation
    is. A launch whose --resume
    is the start checkpoint of the configuration's unused recovery is that recovery (kind "continuation", block
    "recovery", THE LINKED RECOVERY; max_steps its end_step, wall_cap_s its remaining wall cap), bounded and noted as the
    adoption is. A launch whose --resume is the start checkpoint of the configuration's unused link is that link (kind
    "continuation", block "link", THE LINKED CONTINUATION; max_steps its end_step, wall_cap_s its remaining wall cap),
    bounded and noted as the recovery is. Every other launch is normal training. Whether the launch may start is
    launch_refusal's judgement."""
    conf = configuration(cmd, src)
    rec = _load(record_path(conf["fingerprint"], root)) or {}
    rr = rec.get("repair_run") or {}
    cb = rec.get(CONTINUATION) or {}
    resume = flags_of(cmd).get("--resume")

    def plan(kind, name, block):
        start = _step_of(block["start_checkpoint"]) or 0
        try:
            bound = start + int(block.get("max_model_steps") or 0)
        except (TypeError, ValueError):
            bound = start
        out = dict(kind=kind, block=name, fingerprint=conf["fingerprint"], start_step=start, max_steps=bound,
                   stop_criteria=list(block.get("stop_criteria") or []))
        to = target_outcome_of(block)
        if isinstance(to, dict):            # THE TARGET OUTCOME: the run owner compares the cap with its own
            out["wall_cap_s"] = (to.get("pins") or {}).get("wall_cap_s")
        return out
    for kind, block, available in (("repair_run", rr, repair_available(rr) if rr else False),
                                   (CONTINUATION, cb, continuation_available(cb))):
        if available and isinstance(resume, str) and block.get("start_checkpoint") \
                and _norm(resume) == _norm(block["start_checkpoint"]):
            return plan(kind, kind, block)
    # THE EXTENSION (the module text): a launch kind "continuation", block "extension", while unused
    xb = (extension_of(conf["fingerprint"], root).get("block") or {}) if rec else {}
    if extension_available(xb) and isinstance(resume, str) and xb.get("start_checkpoint") \
            and _norm(resume) == _norm(xb["start_checkpoint"]):
        return plan(CONTINUATION, EXTENSION, xb)
    # THE ADOPTION (the module text): a launch kind "continuation", block "adoption", while unused
    ab = (adoption_of(conf["fingerprint"], root).get("block") or {}) if rec else {}
    if adoption_available(ab) and isinstance(resume, str) and _text(ab.get("start_checkpoint")) \
            and _norm(resume) == _norm(ab["start_checkpoint"]):
        # the adoption's wall cap rides in its plan, as the recovery's and the link's do: a launcher compares it
        # with the cap it enforces (the arm-stage launcher refuses a plan whose cap is not its own)
        return dict(plan(CONTINUATION, ADOPTION, ab), wall_cap_s=ab.get("wall_cap_s"))
    # THE LINKED RECOVERY (the module text): a launch kind "continuation", block "recovery", while unused
    rb = (recovery_of(conf["fingerprint"], root).get("block") or {}) if rec else {}
    if recovery_available(rb) and isinstance(resume, str) and _text(rb.get("start_checkpoint")) \
            and _norm(resume) == _norm(rb["start_checkpoint"]):
        return dict(plan(CONTINUATION, RECOVERY, rb), wall_cap_s=rb.get("wall_cap_s"))
    # THE LINKED CONTINUATION (the module text): a launch kind "continuation", block "link", while unused
    lb = (link_of(conf["fingerprint"], root).get("block") or {}) if rec else {}
    if link_available(lb) and isinstance(resume, str) and _text(lb.get("start_checkpoint")) \
            and _norm(resume) == _norm(lb["start_checkpoint"]):
        return dict(plan(CONTINUATION, LINK, lb), wall_cap_s=lb.get("wall_cap_s"))
    if rec:
        matches = _instance_matches(conf["fingerprint"], resume, root)
        if len(matches) == 1:
            return plan("repair_run", "%s:%s" % (INSTANCE, matches[0][0]), matches[0][1]["block"])
    return dict(kind="normal", block=None, fingerprint=conf["fingerprint"], start_step=None, max_steps=None,
                stop_criteria=None)


def launch_refusal(cmd, root: Path = PREFLIGHT_DIR, src: Path = SRC, registry: Path = REGISTRY):
    """None when this launch command's configuration may launch; else the text naming what blocks it and why."""
    flags_wrong = flag_problems(cmd, src)
    if flags_wrong:
        return "\n".join(["BLOCKED: the command names flags the trainer would read otherwise than this check does:"]
                         + ["  - " + p for p in flags_wrong])
    conf = configuration(cmd, src)
    # THE ADOPTION (the module text): a launch from the start of a route's unused adoption whose configuration (its code
    # or a configuration flag) is not the adoption's is refused, naming each difference; THE PAIR'S SIBLING ADOPTIONS
    # (a2-admission-repair-authorized-20261023): except a launch whose own configuration has its own bound route
    # at this exact checkpoint and its own unused adoption starting there (_own_adoption_at), which is judged below by
    # that adoption's full conditions (_adoption_refusal) and by nothing else
    mismatch = _adoption_mismatch(conf, flags_of(cmd).get("--resume"), root)
    own = _own_adoption_at(conf, flags_of(cmd).get("--resume"), root) if mismatch else None
    if mismatch and own is None:
        return mismatch
    # THE LINKED RECOVERY (the module text): likewise for a launch from the start of a route's unused recovery
    mismatch = _recovery_mismatch(conf, flags_of(cmd).get("--resume"), root)
    if mismatch:
        return mismatch
    # THE LINKED CONTINUATION (the module text): likewise for a launch from the start of a route's unused link
    mismatch = _link_mismatch(conf, flags_of(cmd).get("--resume"), root)
    if mismatch:
        return mismatch
    base = _load(Path(root) / BASELINE)
    rec = _load(record_path(conf["fingerprint"], root))
    if rec is None:
        lines = ["BLOCKED: configuration %s has no preflight record (%s)." % (
            conf["fingerprint"], record_path(conf["fingerprint"], root))]
        if base:
            diff = differences(conf, base)
            if base.get("fingerprint") == conf["fingerprint"]:
                lines.append("It is the reference configuration (held when the rule was adopted, never preflighted).")
            else:
                lines.append("It differs from the reference configuration %s in:" % base.get("fingerprint"))
                lines += ["  - " + d for d in diff] or ["  - (no flag or code difference found)"]
        else:
            lines.append("No reference configuration is recorded (%s)." % (Path(root) / BASELINE))
        lines.append("Write it: training_preflight new ... --what TEXT --by NAME; then each of the 7 gates PASS, or N/A "
                     "with a reason and evidence (training_preflight gate ...).")
        return "\n".join(lines)
    if own is not None:              # THE PAIR'S SIBLING ADOPTIONS: the own adoption's full conditions decide, alone
        return _adoption_refusal(cmd, rec, own, conf, root, registry)
    rr = rec.get("repair_run") or {}
    resume = flags_of(cmd).get("--resume")
    if rr and repair_available(rr) and isinstance(resume, str) and rr.get("start_checkpoint") \
            and _norm(resume) == _norm(rr["start_checkpoint"]):
        return _repair_refusal(cmd, rec, rr, conf, root, registry)
    # THE CONTINUATION (the module text): judged only for its own start while unused; every other launch, its start after
    # its use included, is normal training below, judged exactly as before
    cb = rec.get(CONTINUATION) or {}
    if continuation_available(cb) and isinstance(resume, str) and cb.get("start_checkpoint") \
            and _norm(resume) == _norm(cb["start_checkpoint"]):
        return _continuation_refusal(cmd, rec, cb, conf, root, registry)
    # THE EXTENSION (the module text): judged only for its own start while unused; every other launch, its start after
    # its use included, is judged below exactly as before
    xb = extension_of(conf["fingerprint"], root).get("block") or {}
    if extension_available(xb) and isinstance(resume, str) and xb.get("start_checkpoint") \
            and _norm(resume) == _norm(xb["start_checkpoint"]):
        return _extension_refusal(cmd, rec, xb, conf, root, registry)
    # THE ADOPTION (the module text): judged only for its own start while unused; every other launch, its start after its
    # use included, is judged below exactly as before
    held = adoption_of(conf["fingerprint"], root)
    ab = held.get("block") or {}
    if adoption_available(ab) and isinstance(resume, str) and _text(ab.get("start_checkpoint")) \
            and _norm(resume) == _norm(ab["start_checkpoint"]):
        return _adoption_refusal(cmd, rec, held, conf, root, registry)
    # THE LINKED RECOVERY (the module text): judged only for its own start while unused; every other launch, its start
    # after its use included, is judged below exactly as before
    held_r = recovery_of(conf["fingerprint"], root)
    rb = held_r.get("block") or {}
    if recovery_available(rb) and isinstance(resume, str) and _text(rb.get("start_checkpoint")) \
            and _norm(resume) == _norm(rb["start_checkpoint"]):
        return _recovery_refusal(cmd, rec, held_r, conf, root, registry)
    # THE LINKED CONTINUATION (the module text): judged only for its own start while unused; every other launch, its
    # start after its use included, is judged below exactly as before
    held_l = link_of(conf["fingerprint"], root)
    lb = held_l.get("block") or {}
    if link_available(lb) and isinstance(resume, str) and _text(lb.get("start_checkpoint")) \
            and _norm(resume) == _norm(lb["start_checkpoint"]):
        return _link_refusal(cmd, rec, held_l, conf, root, registry)
    # THE FRESH INSTANCE (the module text): judged only for its own start while unused; every other launch is normal
    # training below, judged exactly as before
    matches = _instance_matches(conf["fingerprint"], resume, root)
    if len(matches) > 1:
        return ("BLOCKED: configuration %s has %d unused fresh instances starting from %s (%s); a launch cannot tell them "
                "apart" % (conf["fingerprint"], len(matches), resume, ", ".join(m[0] for m in matches)))
    if matches:
        return _instance_refusal(cmd, rec, matches[0][0], matches[0][1], conf, root, registry)
    # THE ADOPTION (the module text; the review's finding 1, GAP1b): a configuration ADOPTION_ROUTES lists is never
    # trained normally, whatever its gates read, before or after its adoption's use
    listed = adoption_route(conf["fingerprint"])
    recovered = recovery_route(conf["fingerprint"])        # THE LINKED RECOVERY: its configuration is never trained
    also = "" if recovered is None else (", and as its linked recovery (RECOVERY_ROUTES), once, from %s to model step %d"
                                         % (recovered["checkpoint"], recovered["end"]))
    linked = link_route(conf["fingerprint"])                # THE LINKED CONTINUATION: likewise never trained normally
    if linked is not None:
        also += (", and as its linked continuation (LINK_ROUTES), once, from %s to model step %d" % (
            linked["checkpoint"], linked["end"]))
    if listed is not None:
        return ("BLOCKED: configuration %s is the adoption's (ADOPTION_ROUTES): it launches only as its adoption, once, "
                "from %s to model step %d%s; every other launch of it is normal training, refused (the decision: \"Keep "
                "each gate FAIL; no normal-training qualification is granted\")" % (
                    conf["fingerprint"], listed["checkpoint"], listed["end"], also))
    if recovered is not None:
        return ("BLOCKED: configuration %s is the linked recovery's (RECOVERY_ROUTES): it launches only as its recovery, "
                "once, from %s to model step %d; every other launch of it is normal training, refused" % (
                    conf["fingerprint"], recovered["checkpoint"], recovered["end"]))
    if linked is not None:
        return ("BLOCKED: configuration %s is the linked continuation's (LINK_ROUTES): it launches only as its link, "
                "once, from %s to model step %d; every other launch of it is normal training, refused" % (
                    conf["fingerprint"], linked["checkpoint"], linked["end"]))
    gates = {g.get("n"): g for g in rec.get("gates") or [] if isinstance(g, dict)}
    launched_at, launched_s, launched_run = repair_launch(rr)
    problems = []
    for n, name, _question in GATES:
        g = gates.get(n) or {"n": n, "name": name, "status": "OPEN"}
        why = gate_problem(dict(g, name=name))
        if why:
            problems.append(why)
        elif launched_s and n in (rr.get("target_gates") or []) \
                and float(g.get("checked_at_s") or 0) <= float(launched_s):
            problems.append("gate %d %s was the target of the repair run launched %s; it must be measured again after "
                            "that run before normal training (last judged %s)" % (
                                n, name, launched_at, g.get("checked_at") or "never"))
    if not problems:
        checkpoint = flags_of(cmd).get("--resume")
        checkpoint = checkpoint if isinstance(checkpoint, str) else None
        pool = [s for s in rec.get("states") or [] if isinstance(s, dict)]
        if launched_s:
            # AFTER THE REPAIR RUN (the relaunch review of Oct 5, 2026, finding F9): the target gates were judged again
            # on the repaired model, so only a state added after the launch (the checkpoint they were judged on)
            # authorizes normal training; the start checkpoint, whose weights the run never changed, no longer does
            pool = [s for s in pool if float(s.get("at_s") or 0) >= float(launched_s)
                    and descends_from(s.get("checkpoint"), launched_run)]     # the re-check's finding N3
            if not any(continues(checkpoint, s.get("checkpoint")) for s in pool):
                return ("BLOCKED: configuration %s ran its repair run (launched %s, run %s); normal training continues "
                        "only a model state added after that launch, the checkpoint its target gates were judged on "
                        "again (states added since: %s); checkpoint %s is not one. Add it: training_preflight state %s "
                        "--checkpoint PATH --reason TEXT --by NAME." % (
                            conf["fingerprint"], launched_at, launched_run,
                            "; ".join(str(s.get("checkpoint")) for s in pool) or "none", checkpoint or FRESH,
                            conf["fingerprint"]))
            return None
        states = [s.get("checkpoint") for s in pool]
        if any(continues(checkpoint, s) for s in states):
            return None
        return "\n".join([
            "BLOCKED: configuration %s passed its gates on the model state(s) %s; checkpoint %s does not continue them "
            "(another run or branch, or a checkpoint from before that state, which may predate a repair)." % (
                conf["fingerprint"], "; ".join(map(str, states)) or "none recorded", checkpoint or FRESH),
            "Reconsider the gates that depend on the model's state (%s) for this checkpoint, then add it: "
            "training_preflight state %s --checkpoint PATH --reason TEXT --by NAME." % (
                ", ".join("%d %s" % (n, nm) for n, nm, _q in GATES if n in STATE_GATES), conf["fingerprint"])])
    return "\n".join(["BLOCKED: configuration %s (%s): %d of 7 gates block launch:" % (
        conf["fingerprint"], record_path(conf["fingerprint"], root), len(problems))] + ["  - " + p for p in problems])


def new_record(cmd, what: str, by: str, root: Path = PREFLIGHT_DIR, src: Path = SRC) -> Path:
    """Write the record of a configuration with its seven gates OPEN and what differs from the reference."""
    if not str(what).strip() or not str(by).strip():
        raise PreflightError("a record needs what changed and who writes it")
    flags_wrong = flag_problems(cmd, src)
    if flags_wrong:
        raise PreflightError("; ".join(flags_wrong))
    conf = configuration(cmd, src)
    path = record_path(conf["fingerprint"], root)
    if path.exists():
        raise PreflightError("a record for configuration %s exists: %s" % (conf["fingerprint"], path))
    base = _load(Path(root) / BASELINE)
    resume = flags_of(cmd).get("--resume")
    _save(path, dict(
        fingerprint=conf["fingerprint"], created_at=clock(), by=str(by), what=str(what),
        states=[dict(checkpoint=resume if isinstance(resume, str) else FRESH, at=clock(), at_s=time.time(), by=str(by),
                     reason="the state the gates are judged on (the command's --resume)")],
        differences_from_reference=differences(conf, base) if base else ["no reference configuration"],
        flags=conf["flags"], files=conf["files"], code_hash=conf["code_hash"], code=conf["code"],
        gates=[dict(n=n, name=name, question=q, status="OPEN", reason="", evidence=[], checked_at=None,
                    checked_by=None) for n, name, q in GATES]))
    return path


def set_gate(fingerprint: str, n: int, status: str, reason: str, evidence, by: str, root: Path = PREFLIGHT_DIR):
    """Write one gate's result into a configuration's record."""
    if status not in STATUSES[1:]:
        raise PreflightError("status must be one of PASS, N/A, FAIL, %s" % REPAIR_TARGET)
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    evidence = [str(e) for e in (evidence or []) if str(e).strip()]
    if status == REPAIR_TARGET:
        rr = rec.get("repair_run") or {}
        if int(n) not in (rr.get("target_gates") or []):
            raise PreflightError("gate %s may read %s only when the record's repair run names it as its purpose "
                                 "(training_preflight repair ...)" % (n, REPAIR_TARGET))
        if not str(reason or "").strip() or not evidence:
            raise PreflightError("%s needs the failure's reason and its evidence" % REPAIR_TARGET)
    for g in rec.get("gates") or []:
        if g.get("n") == int(n):
            g.update(status=status, reason=str(reason or ""), evidence=evidence, checked_at=clock(),
                     checked_at_s=time.time(), checked_by=str(by))
            why = gate_problem(g)
            if why and status not in ("FAIL", REPAIR_TARGET):
                raise PreflightError(why)
            _save(path, rec)
            return g
    raise PreflightError("the record has no gate %s" % n)


def set_repair_run(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY):
    """Write a record's repair_run block, the operator's conditions for a controlled repair or qualification run: the
    exact failure(s) it repairs (target gates), the repairs and affected components named, the start and rollback
    checkpoints, the stop lines declared in the registry before launch, a bounded duration, how automatic continuation
    is prevented, and the measurement that judges each target gate again after the run."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    block = dict(block)
    block["target_gates"] = sorted({int(n) for n in block.get("target_gates") or []})
    block["remeasure"] = {str(k): str(v) for k, v in (block.get("remeasure") or {}).items()}
    problems = repair_block_problems(block, registry)
    problems += target_outcome_authority_problems(block, fingerprint, root)          # THE TARGET OUTCOME
    if problems:
        raise PreflightError("; ".join(problems))
    used = repair_launch(rec.get("repair_run") or {})[0]
    if used:
        raise PreflightError("this record's repair run was launched %s; a new repair run needs its own record" % used)
    block.update(written_at=clock(), written_by=str(by), launched_at=None, launched_at_s=None, run=None)
    rec["repair_run"] = block
    if not any(_norm(s.get("checkpoint")) == _norm(block["start_checkpoint"]) for s in rec.get("states") or []):
        rec.setdefault("states", []).append(dict(checkpoint=block["start_checkpoint"], at=clock(), at_s=time.time(),
                                                 by=str(by), reason="the repair run's start checkpoint"))
    _save(path, rec)
    return block


def set_continuation(fingerprint: str, block: dict, by: str, root: Path = PREFLIGHT_DIR, registry: Path = REGISTRY):
    """Write a record's continuation block (THE CONTINUATION in the module text) after checking it
    (continuation_block_problems): refused for a record whose repair run never launched or did not complete, and once
    the record's continuation was used or marked launched (an attempt that trained past its start, or a launch mark).
    A block whose attempts never trained may be written again (the re-check of Oct 11, 2026: an attempt whose trainer
    died before its first full stats record left its latent output folder behind, which run_flag_problems refuses, so
    the retry needs a new --latent-body-output): its recorded attempts are carried into the new block, so an attempt
    that trains later still counts as the one use (the review's finding F2: a rewrite that dropped them let that
    training count as no use). The time it is written (written_at_s) ties each exception to the failure judged before
    it (finding F3). The repair run's block is not touched, and no model state is added. The gates and the launch
    command are judged at launch (launch_refusal)."""
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    if not str(by or "").strip():
        raise PreflightError("a continuation needs who writes it (--by)")
    old = rec.get(CONTINUATION) if isinstance(rec.get(CONTINUATION), dict) else {}
    if repair_uses(old):                            # a launch mark, or an attempt that trained past its start
        at, _at_s, run = repair_launch(old)
        raise PreflightError(
            "this record's continuation was used (%s, run %s); a continuation is used once, and a used or launched block "
            "is not written again. A configuration has one record (one per fingerprint), so this configuration takes no "
            "second continuation" % (("launched %s" % at) if old.get("launched_at") else
                                     "an attempt trained past its start", run))
    block = dict(block or {})
    if isinstance(block.get("exceptions"), dict):
        block["exceptions"] = {str(k): v for k, v in block["exceptions"].items()}
    problems = continuation_block_problems(block, rec, registry)
    if problems:
        raise PreflightError("; ".join(problems))
    for k in ("attempts", "launches"):              # only the relaunch script records these (note_attempt, mark_launched)
        block.pop(k, None)
    carried = [a for a in old.get("attempts") or [] if isinstance(a, dict)]
    if carried:                                     # attempts that never trained, kept: one that trains later is the use
        block["attempts"] = carried
    block.update(role=CONTINUATION, written_at=clock(), written_at_s=time.time(), written_by=str(by), launched_at=None,
                 launched_at_s=None, run=None)
    rec[CONTINUATION] = block
    _save(path, rec)
    return block


def _launch_block(rec, fingerprint: str, block: str):
    """(the record's block `block`, its name in messages): the repair run's or the continuation's."""
    if block not in LAUNCH_BLOCKS:
        raise PreflightError("the block must be one of %s (reads %r)" % (", ".join(LAUNCH_BLOCKS), block))
    noun = "repair run" if block == "repair_run" else "continuation"
    b = (rec or {}).get(block)
    if not b:
        raise PreflightError("no %s in the record of configuration %s" % (noun, fingerprint))
    return b, noun


def _launch_target(fingerprint: str, block: str, root: Path, src=None, registry=None):
    """(the file that holds the launch block `block`, that file's content, the block, its name in messages): the
    record's repair run or continuation (_launch_block), or a fresh instance (block "instance:<experiment id>", THE
    FRESH INSTANCE), whose own file holds its launches: the record is not written for it. The extension (block
    "extension", THE EXTENSION) likewise: its own file holds its launches. The adoption (block "adoption", THE ADOPTION)
    likewise, checked again first (adoption_problems with `noting`: the block, the route's pins, the decision's message,
    the delegation record, the template's sha256, the preservation, the predecessor, the authorization, and the record
    with the trainer code of `src`, which is required: the Source/Python folder of the checkout the trainer runs from,
    the same one the run owner gives launch_refusal; the review's finding 5). The recovery (block "recovery", THE LINKED
    RECOVERY) likewise, checked again first (recovery_problems with `noting`, src required, the used adoption's rows read
    in `registry`, production's when None). The link (block "link", THE LINKED CONTINUATION) likewise (link_problems
    with `noting`, src required, the earlier blocks' rows read in `registry`, production's when None)."""
    if block == LINK:
        if src is None:
            raise PreflightError("a link's attempt and launch are noted with src, the Source/Python folder of the "
                                 "checkout the trainer runs from (the src the run owner gives launch_refusal)")
        path = link_path(fingerprint, root)
        held = _load(path)
        b = (held or {}).get("block")
        if not isinstance(b, dict) or not b:
            raise PreflightError("no link of configuration %s (%s)" % (fingerprint, path))
        rec = _load(record_path(fingerprint, root))
        if rec is None:
            raise PreflightError("no record for configuration %s" % fingerprint)
        bad = link_problems(fingerprint, held, rec, root, Path(registry) if registry else REGISTRY,
                            code_hash=trainer_code.trainer_code(src)[1], noting=True)
        if bad:
            raise PreflightError("; ".join(bad))
        return path, held, b, "link"
    if block == RECOVERY:
        if src is None:
            raise PreflightError("a recovery's attempt and launch are noted with src, the Source/Python folder of the "
                                 "checkout the trainer runs from (the src the run owner gives launch_refusal)")
        path = recovery_path(fingerprint, root)
        held = _load(path)
        b = (held or {}).get("block")
        if not isinstance(b, dict) or not b:
            raise PreflightError("no recovery of configuration %s (%s)" % (fingerprint, path))
        rec = _load(record_path(fingerprint, root))
        if rec is None:
            raise PreflightError("no record for configuration %s" % fingerprint)
        bad = recovery_problems(fingerprint, held, rec, root, Path(registry) if registry else REGISTRY,
                                code_hash=trainer_code.trainer_code(src)[1], noting=True)
        if bad:
            raise PreflightError("; ".join(bad))
        return path, held, b, "recovery"
    if block == ADOPTION:
        if src is None:
            raise PreflightError("an adoption's attempt and launch are noted with src, the Source/Python folder of the "
                                 "checkout the trainer runs from (the src the run owner gives launch_refusal)")
        path = adoption_path(fingerprint, root)
        held = _load(path)
        b = (held or {}).get("block")
        if not isinstance(b, dict) or not b:
            raise PreflightError("no adoption of configuration %s (%s)" % (fingerprint, path))
        rec = _load(record_path(fingerprint, root))
        if rec is None:
            raise PreflightError("no record for configuration %s" % fingerprint)
        bad = adoption_problems(fingerprint, held, rec, root, code_hash=trainer_code.trainer_code(src)[1], noting=True)
        if bad:
            raise PreflightError("; ".join(bad))
        return path, held, b, "adoption"
    if block == EXTENSION:
        path = extension_path(fingerprint, root)
        held = _load(path)
        b = (held or {}).get("block")
        if not isinstance(b, dict) or not b:
            raise PreflightError("no extension of configuration %s (%s)" % (fingerprint, path))
        bad = extension_route_problems(fingerprint, (held or {}).get("continuation_run"))
        if bad:                                     # finding B-1: an extension file the route does not admit
            raise PreflightError("; ".join(bad))
        return path, held, b, "extension"
    if isinstance(block, str) and block.startswith(INSTANCE + ":"):
        experiment_id = block.split(":", 1)[1]
        bad = _instance_id_problem(experiment_id)
        if bad:
            raise PreflightError(bad)
        path = instance_path(fingerprint, experiment_id, root)
        held = _load(path)
        b = (held or {}).get("block")
        if not isinstance(b, dict) or not b:
            raise PreflightError("no fresh instance %s of configuration %s (%s)" % (experiment_id, fingerprint, path))
        return path, held, b, "fresh instance"
    path = record_path(fingerprint, root)
    rec = _load(path)
    b, noun = _launch_block(rec, fingerprint, block)
    return path, rec, b, noun


def mark_launched(fingerprint: str, run: str, by: str, root: Path = PREFLIGHT_DIR, block: str = "repair_run",
                  src=None, registry=None):
    """Written by the relaunch script when it starts a repair run (block "repair_run"), a continuation (block
    "continuation"), a fresh instance (block "instance:<experiment id>"), an extension (block "extension") or an
    adoption (block "adoption": the run owner passes src, the Source/Python folder of the checkout the trainer runs
    from, and the launch is also recorded in the adoption's training row of `registry`, production's test registry
    when None), or a recovery (block "recovery", as an adoption, its training row's record field RECOVERY_ROW_FIELD), or
    a link (block "link", as a recovery, its training row's record field LINK_ROW_FIELD): the run is used once; after a
    repair run, normal training of this record needs every target gate judged again after this time."""
    path, rec, rr, noun = _launch_target(fingerprint, block, root, src, registry)
    if block == ADOPTION:                           # THE TENTH AND THE ELEVENTH ENTRY: the row's runner is the owner
        bad = adoption_owner_runner_problems(fingerprint, Path(registry) if registry else REGISTRY)
        if bad:
            raise PreflightError("; ".join(bad))
    if rr.get("launched_at") and repair_repeats(rr) <= 1:
        raise PreflightError("this %s was launched already (%s)" % (noun, rr["launched_at"]))
    if not str(run or "").strip() or not Path(str(run)).is_dir():
        # the round-two review's R5: an empty or missing run folder made every later state "not one" of the repair run
        raise PreflightError("name the %s's folder, an existing folder (--run), to mark its launch: read %r"
                             % (noun, run))
    resolved = str(Path(str(run)).resolve())
    marked = ([rr.get("run")] if rr.get("launched_at") else []) + [
        m.get("run") for m in rr.get("launches") or [] if isinstance(m, dict)]
    if any(_norm(m or "") == _norm(resolved) for m in marked):
        raise PreflightError("run %s is marked launched already" % resolved)
    if len(marked) >= repair_repeats(rr):
        raise PreflightError("this %s's %d launch(es) are marked already" % (noun, repair_repeats(rr)))
    if block in (ADOPTION, RECOVERY):               # THE ADOPTION: its training row's record (the review's finding 8)
        registry = Path(registry) if registry else REGISTRY
        field = ADOPTION_ROW_FIELD if block == ADOPTION else RECOVERY_ROW_FIELD
        bad = adoption_registry_problems(fingerprint, rr, registry, "mark", field=field, noun=block)
        if bad:
            raise PreflightError("; ".join(bad))
        _record_in_training_row(fingerprint, rr, registry, run, by, "launch", field=field, noun=block)
    if block == LINK:                               # THE LINKED CONTINUATION: its training row's record, as a recovery's
        registry = Path(registry) if registry else REGISTRY
        bad = adoption_registry_problems(fingerprint, rr, registry, "mark", field=LINK_ROW_FIELD, noun=LINK)
        if bad:
            raise PreflightError("; ".join(bad))
        _record_in_training_row(fingerprint, rr, registry, run, by, "launch", field=LINK_ROW_FIELD, noun=LINK)
    if not rr.get("launched_at"):
        rr.update(launched_at=clock(), launched_at_s=time.time(), run=resolved, launched_by=str(by))
    else:                               # a repeated control's later run (THE EXPERIMENT in the module text)
        rr.setdefault("launches", []).append(dict(at=clock(), at_s=time.time(), run=resolved, by=str(by)))
    _save(path, rec)
    return rr


def note_attempt(fingerprint: str, run: str, by: str, root: Path = PREFLIGHT_DIR, block: str = "repair_run",
                 src=None, registry=None, claim=None):
    """Written by the relaunch script before a repair run's (block "repair_run"), a continuation's (block
    "continuation"), a fresh instance's (block "instance:<experiment id>"), an extension's (block "extension") or an
    adoption's (block "adoption": the run owner passes src, the Source/Python folder of the checkout the trainer runs
    from, and the attempt is also recorded in the adoption's training row of `registry`, production's test registry
    when None) trainer starts: the attempt's run folder. A block one of whose attempts trained past the start is used
    (repair_launch, repair_uses), whether or not the script lived to mark the launch (the round-two review's R4). A
    recovery's (block "recovery") is noted as an adoption's, its training row's record field RECOVERY_ROW_FIELD; a link's
    (block "link") as a recovery's, its training row's record field LINK_ROW_FIELD."""
    path, rec, rr, noun = _launch_target(fingerprint, block, root, src, registry)
    if block == ADOPTION:                           # THE TENTH AND THE ELEVENTH ENTRY: the pinned owner's own claim
        bad = adoption_owner_claim_problems(fingerprint, claim)
        if bad:
            raise PreflightError("; ".join(bad))
    if not repair_available(rr):
        raise PreflightError("this %s was launched already (%s; %d of %d use(s))" % (
            noun, repair_launch(rr)[0], len(repair_uses(rr)), repair_repeats(rr)))
    if not Path(str(run)).is_dir():
        raise PreflightError("the attempt's run folder does not exist: %s" % run)
    if block in (ADOPTION, RECOVERY):               # THE ADOPTION: its training row's record (the review's finding 8)
        registry = Path(registry) if registry else REGISTRY
        field = ADOPTION_ROW_FIELD if block == ADOPTION else RECOVERY_ROW_FIELD
        bad = adoption_registry_problems(fingerprint, rr, registry, "note", field=field, noun=block,
                                         claim=claim, note_run=run)   # B1: the owner's claim identity and this run
        if bad:
            raise PreflightError("; ".join(bad))
        _record_in_training_row(fingerprint, rr, registry, run, by, "attempt", field=field, noun=block)
    if block == LINK:                               # THE LINKED CONTINUATION: its training row's record, as a recovery's
        registry = Path(registry) if registry else REGISTRY
        bad = adoption_registry_problems(fingerprint, rr, registry, "note", field=LINK_ROW_FIELD, noun=LINK)
        if bad:
            raise PreflightError("; ".join(bad))
        _record_in_training_row(fingerprint, rr, registry, run, by, "attempt", field=LINK_ROW_FIELD, noun=LINK)
    rr.setdefault("attempts", []).append(dict(run=str(Path(str(run)).resolve()), at=clock(), at_s=time.time(),
                                              by=str(by)))
    _save(path, rec)
    return rr["attempts"][-1]


def add_state(fingerprint: str, checkpoint: str, reason: str, by: str, root: Path = PREFLIGHT_DIR):
    """Add a model state the record's gates hold for, after the gates that depend on the model's state were
    reconsidered for it (the reason says how)."""
    if not str(reason).strip() or not str(by).strip() or not str(checkpoint).strip():
        raise PreflightError("a state needs its checkpoint, the reason the gates hold for it, and who writes it")
    path = record_path(fingerprint, root)
    rec = _load(path)
    if rec is None:
        raise PreflightError("no record for configuration %s" % fingerprint)
    launched_at, launched_s, launched_run = repair_launch(rec.get("repair_run") or {})
    if launched_s and not descends_from(checkpoint, launched_run):
        # the re-check of Oct 5, 2026, finding N3: after the repair run only the repaired model's checkpoints are states
        raise PreflightError("the repair run of this record launched %s in %s; a state added after it is a checkpoint of "
                             "that run or of a run resumed from it, and %s is not one" % (
                                 launched_at, launched_run, checkpoint))
    rec.setdefault("states", []).append(dict(checkpoint=str(checkpoint), at=clock(), at_s=time.time(), by=str(by),
                                             reason=str(reason)))
    _save(path, rec)
    return rec["states"][-1]


def reference(cmd, reason: str, by: str, root: Path = PREFLIGHT_DIR, src: Path = SRC) -> Path:
    """Write the reference configuration once (it names differences; it gives no permission to launch)."""
    path = Path(root) / BASELINE
    if path.exists():
        raise PreflightError("a reference configuration exists: %s" % path)
    if not str(reason).strip() or not str(by).strip():
        raise PreflightError("the reference needs its reason and who writes it")
    conf = configuration(cmd, src)
    _save(path, dict(status="REFERENCE", fingerprint=conf["fingerprint"], written_at=clock(), by=str(by),
                     reason=str(reason), flags=conf["flags"], files=conf["files"], code_hash=conf["code_hash"],
                     code=conf["code"]))
    return path


def _command(a):
    if a.current:
        cmd = json.loads(SUPERVISOR_RECORD.read_text(encoding="utf-8"))["command"]
    elif a.cmd_file:
        # utf-8-sig: a file written with a byte-order mark (PowerShell 5.1's Out-File) reads as the same list, as the
        # relaunch's --cmd-file reads it (the independent review of Oct 9, 2026, finding F4)
        cmd = json.loads(Path(a.cmd_file).read_text(encoding="utf-8-sig"))
    else:
        raise PreflightError("name the command: --current or --cmd-file FILE")
    return apply_sets(cmd, a.set)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="training_preflight", description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["check", "new", "gate", "state", "repair", "continuation", "instance",
                                       "extension", "adoption", "recovery", "link", "launched", "reference",
                                       "fingerprint"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--purpose")
    ap.add_argument("--target-gates", default="")
    ap.add_argument("--repairs", default="")
    ap.add_argument("--components", default="")
    ap.add_argument("--start")
    ap.add_argument("--rollback")
    ap.add_argument("--stop-lines")
    ap.add_argument("--stop-lines-row")
    ap.add_argument("--stop-criteria", help="a JSON file: the list of the watcher's declared criteria for the run")
    ap.add_argument("--block-file", help="repair: the whole repair_run block as a JSON file (an experiment's arm); "
                                         "continuation, instance, extension, adoption, recovery, link: that block as a "
                                         "JSON file")
    ap.add_argument("--continuation", action="store_true",
                    help="launched: mark the record's continuation launched, not its repair run")
    ap.add_argument("--instance", default="", metavar="EXPERIMENT_ID",
                    help="launched: mark the configuration's fresh instance in this experiment launched")
    ap.add_argument("--extension", action="store_true",
                    help="launched: mark the configuration's extension launched")
    ap.add_argument("--adoption", action="store_true",
                    help="launched: mark the configuration's adoption launched (with --src, and --registry when not "
                         "production's)")
    ap.add_argument("--recovery", action="store_true",
                    help="launched: mark the configuration's recovery launched (with --src, and --registry when not "
                         "production's)")
    ap.add_argument("--link", action="store_true",
                    help="launched: mark the configuration's link launched (with --src, and --registry when not "
                         "production's)")
    ap.add_argument("--src", help="launched --adoption, --recovery or --link, recovery, link, check and fingerprint: the "
                                  "Source/Python folder of the checkout the trainer runs from")
    ap.add_argument("--registry", help="launched --adoption, --recovery or --link, recovery, link and check: the test "
                                       "registry holding the rows (production's when not given)")
    ap.add_argument("--root", help="check: the preflight folder to judge against (production's when not given)")
    ap.add_argument("--reachability", help="the arithmetic showing each stop and pass line can be reached within the "
                                           "bound (the style guide, No experiment launches on an unmeasured number, rule 4)")
    ap.add_argument("--max-model-steps", type=int, default=0)
    ap.add_argument("--no-auto-continue")
    ap.add_argument("--remeasure", action="append", default=[], metavar="GATE=ROW")
    ap.add_argument("--run")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--current", action="store_true")
    ap.add_argument("--cmd-file")
    ap.add_argument("--set", nargs=2, action="append", default=[], metavar=("FLAG", "VALUE"))
    ap.add_argument("--what")
    ap.add_argument("--by")
    ap.add_argument("--reason")
    ap.add_argument("--status")
    ap.add_argument("--evidence", action="append", default=[])
    a = ap.parse_args(argv)
    try:
        if a.action == "gate":
            fingerprint, n = a.args[0], int(a.args[1])
            g = set_gate(fingerprint, n, a.status, a.reason, a.evidence, a.by)
            print("gate %d %s: %s, written %s" % (g["n"], g["name"], g["status"], g["checked_at"]))
            return 0
        if a.action == "state":
            s = add_state(a.args[0], a.checkpoint or "", a.reason or "", a.by or "")
            print("state added %s, %s" % (s["checkpoint"], s["at"]))
            return 0
        if a.action == "repair" and a.block_file:
            # the whole block from a JSON file (an experiment's arm: role, experiment, exceptions)
            b = set_repair_run(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")), a.by or "")
            print("repair run written (%s): target gates %s, start %s, at most %d model steps" % (
                b.get("role", "repair"), b["target_gates"], b["start_checkpoint"], b["max_model_steps"]))
            return 0
        if a.action == "repair":
            split = lambda s: [x.strip() for x in str(s or "").split(";") if x.strip()]  # noqa: E731
            block = dict(purpose=a.purpose or "", target_gates=[int(x) for x in split(a.target_gates.replace(",", ";"))],
                         repairs=split(a.repairs), affected_components=split(a.components),
                         start_checkpoint=a.start or "", rollback_checkpoint=a.rollback or "",
                         stop_lines=a.stop_lines or "", stop_lines_row=a.stop_lines_row or "",
                         stop_criteria=(json.loads(Path(a.stop_criteria).read_text(encoding="utf-8-sig"))
                                        if a.stop_criteria else []),
                         reachability=a.reachability or "",
                         max_model_steps=a.max_model_steps, no_auto_continue=a.no_auto_continue or "",
                         remeasure=dict(x.split("=", 1) for x in a.remeasure if "=" in x))
            b = set_repair_run(a.args[0], block, a.by or "")
            print("repair run written: target gates %s, start %s, at most %d model steps" % (
                b["target_gates"], b["start_checkpoint"], b["max_model_steps"]))
            return 0
        if a.action == "continuation":
            if not a.block_file:
                raise PreflightError("continuation needs --block-file, the continuation block as a JSON file")
            b = set_continuation(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")),
                                 a.by or "")
            print("continuation written: start %s, at most %d model steps, exceptions for gates %s" % (
                b["start_checkpoint"], int(b["max_model_steps"]), sorted(b.get("exceptions") or {}) or "none"))
            return 0
        if a.action == "instance":
            if not a.block_file:
                raise PreflightError("instance needs --block-file, the fresh instance's block as a JSON file")
            b = set_instance(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")), a.by or "")
            print("fresh instance written (%s): experiment %s, start %s, at most %d model steps, exceptions for gates %s"
                  % (instance_path(a.args[0], b["experiment"]["id"]), b["experiment"]["id"], b["start_checkpoint"],
                     int(b["max_model_steps"]), sorted((b["experiment"].get("exceptions") or {})) or "none"))
            return 0
        if a.action == "extension":
            if not a.block_file:
                raise PreflightError("extension needs --block-file, the extension's block as a JSON file")
            b = set_extension(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")), a.by or "")
            print("extension written (%s): start %s, at most %d model steps, exceptions for gates %s" % (
                extension_path(a.args[0]), b["start_checkpoint"], int(b["max_model_steps"]),
                sorted(b.get("exceptions") or {}) or "none"))
            return 0
        if a.action == "adoption":
            if not a.block_file:
                raise PreflightError("adoption needs --block-file, the adoption's block as a JSON file")
            b = set_adoption(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")), a.by or "")
            print("adoption written (%s): start %s, model steps %d to %d, decision %s, exceptions for gates %s" % (
                adoption_path(a.args[0]), b["start_checkpoint"], b["start_step"], b["end_step"],
                b["decision"]["id"], sorted(b.get("exceptions") or {}) or "none"))
            return 0
        if a.action == "recovery":
            if not a.block_file:
                raise PreflightError("recovery needs --block-file, the recovery's block as a JSON file")
            if not a.src:
                raise PreflightError("recovery needs --src, the Source/Python folder of the checkout the trainer runs "
                                     "from (its trainer code is compared with the record's)")
            b = set_recovery(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")), a.by or "",
                             registry=Path(a.registry) if a.registry else REGISTRY, src=Path(a.src))
            print("recovery written (%s): start %s, model steps %d to %d, wall cap %s s, decision %s, exceptions for "
                  "gates %s" % (recovery_path(a.args[0]), b["start_checkpoint"], b["start_step"], b["end_step"],
                                b["wall_cap_s"], b["decision"]["id"], sorted(b.get("exceptions") or {}) or "none"))
            return 0
        if a.action == "link":
            if not a.block_file:
                raise PreflightError("link needs --block-file, the link's block as a JSON file")
            if not a.src:
                raise PreflightError("link needs --src, the Source/Python folder of the checkout the trainer runs from "
                                     "(its trainer code is compared with the record's)")
            b = set_link(a.args[0], json.loads(Path(a.block_file).read_text(encoding="utf-8-sig")), a.by or "",
                         registry=Path(a.registry) if a.registry else REGISTRY, src=Path(a.src))
            print("link written (%s): start %s, model steps %d to %d, wall cap %s s, decision %s, exceptions for gates %s"
                  % (link_path(a.args[0]), b["start_checkpoint"], b["start_step"], b["end_step"], b["wall_cap_s"],
                     b["decision"]["id"], sorted(b.get("exceptions") or {}) or "none"))
            return 0
        if a.action == "launched":
            if not a.run:
                raise PreflightError("launched needs --run, the %s's folder" % (
                    "continuation" if a.continuation else "fresh instance" if a.instance else
                    "extension" if a.extension else "adoption" if a.adoption else "recovery" if a.recovery else
                    "link" if a.link else "repair run"))
            if sum(map(bool, (a.continuation, a.instance, a.extension, a.adoption, a.recovery, a.link))) > 1:
                raise PreflightError("launched marks one block: --continuation, --instance, --extension, --adoption, "
                                     "--recovery or --link, not more")
            if a.link:
                rr = mark_launched(a.args[0], a.run, a.by or "", block=LINK,
                                   src=Path(a.src) if a.src else None, registry=a.registry or None)
                print("link marked launched %s" % rr["launched_at"])
                return 0
            if a.recovery:
                rr = mark_launched(a.args[0], a.run, a.by or "", block=RECOVERY,
                                   src=Path(a.src) if a.src else None, registry=a.registry or None)
                print("recovery marked launched %s" % rr["launched_at"])
                return 0
            if a.adoption:
                rr = mark_launched(a.args[0], a.run, a.by or "", block=ADOPTION,
                                   src=Path(a.src) if a.src else None, registry=a.registry or None)
                print("adoption marked launched %s" % rr["launched_at"])
                return 0
            if a.extension:
                rr = mark_launched(a.args[0], a.run, a.by or "", block=EXTENSION)
                print("extension marked launched %s" % rr["launched_at"])
                return 0
            if a.instance:
                rr = mark_launched(a.args[0], a.run, a.by or "", block="%s:%s" % (INSTANCE, a.instance))
                print("fresh instance %s marked launched %s" % (a.instance, rr["launched_at"]))
                return 0
            if a.continuation:
                rr = mark_launched(a.args[0], a.run, a.by or "", block=CONTINUATION)
                print("continuation marked launched %s" % rr["launched_at"])
                return 0
            rr = mark_launched(a.args[0], a.run, a.by or "")
            print("repair run marked launched %s" % rr["launched_at"])
            return 0
        cmd = _command(a)
        # --src: the trainer code of another checkout (the launch's own; a recovery runs its pinned checkout's code)
        kw = dict(src=Path(a.src)) if a.src else {}
        if a.action == "fingerprint":
            print(configuration(cmd, **kw)["fingerprint"])
            return 0
        if a.action == "new":
            print("written", new_record(cmd, a.what or "", a.by or ""))
            return 0
        if a.action == "reference":
            print("written", reference(cmd, a.reason or "", a.by or ""))
            return 0
        judged = dict(kw, **({"registry": Path(a.registry)} if a.registry else {}))
        if a.root:                                  # a preflight folder other than production's (a scratch copy)
            judged["root"] = Path(a.root)
        why = launch_refusal(cmd, **judged)
        print(why or "ALLOWED: configuration %s may launch." % configuration(cmd, **kw)["fingerprint"])
        return 1 if why else 0
    except PreflightError as exc:
        print("preflight: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
