# Preserve learning while adding local skin and an observing chooser

The candidate changes `contact_profile=mechanical_v2` to `mechanical_v2_local`
and `self_choice=off` to `shadow`. The original learner continues to select the
actual commands and receive all learning updates. The separate candidate observes
the actual body and records its proposed choices; it is not authorized to control
the body. The experiment does not establish a learned developmental ability.

## Preservation already verified

Preparation from the complete native checkpoint at step 4,012,825 preserved all
390 original model tensors and all 328 earned Adam states exactly by name. It
copied the original runtime, curriculum metadata and lineage files byte for byte.
The full eight-world episode retained physical poses, motion, current observation,
executed orders, replay, self histories, recording prefixes and the old quality
and exposure histories. All eight clocks were 4,445.500000073825 simulated seconds.

The previous operator applied impairment uniformly within each body row. Its
existing quality/history is therefore inherited by that body's original skin
samples at the boundary. Local dynamics begin on the next physical tick. No
earlier impaired experience is relabeled as healthy, and no old model is imported.

Receipt: `D:/UEMA_Research/external/D/UEMA_recordings/local_shadow_preflight_cpu_20260913/checkpoints/ckpt_step_4012825/continuity_migration.json`.
This is an explicitly labeled CPU copy: its copied CUDA RNG is omitted only
because CPU execution cannot consume it. It must never become the production
checkpoint. The production branch of the preparation tool preserves CUDA RNG.

The atomic writer's focused physical/learner test also verifies exact old
weights, Adam, runtime and episode through actual publication and reload.
Three migration tests passed on September 13.

## Native eight-world validation record

`Saved/mimo_bridge/native_local_shadow_validation_20260913/partial_report.json`
records the first successful native startup and two actual learning updates.
All eight physical clocks advanced by 0.1 second without reset; the shadow
observed twice and authored no physical command. Complete checkpoint 4,012,827
was published.

The second native startup rejected the saved actor's execution fingerprint.
The failure is retained in `pass_2/console.log`; it occurred before another
learning update. This was a continuation defect, not a behavioral failure. The
running production trainer was not stopped by this test. The corrected independent
native second startup has now passed, as recorded below.

The review reproduced a defect in the old execution hash: interning an unchanged
Python string constant can change `marshal.dumps(code)`. The corrected v2 hash
encodes semantic code fields and constants canonically, binds defaults and loaded
primitive tables, and stores a full descriptor for later comparisons. Sixteen
execution/actor tests passed, including changed-motor-table rejection, interning,
independent processes and actual actor save/resume. The old receipt stored only
its hash, so this reproduced defect cannot retrospectively identify which field
caused the first native pair's mismatch.

**The fresh native pair passed completely.**
`Saved/mimo_bridge/native_local_shadow_validation_20260913b/report.json` records
two actual eight-body native training runs, with a full checkpoint and a separate
process startup between them. Steps advanced 4,012,825 to 4,012,827 to 4,012,829.
Every physical clock advanced from 4,445.5 to 4,445.6 to 4,445.7 seconds, without
reset. Every body ran 4,683 local skin samples and delivered all 53 within-body
spatial-moment rows in the actual frame. The observer retained its original birth
step, continued its observation count from two to four, and never authored a
physical command. Both final runtime/physical checkpoints loaded and validated.

This establishes native functional continuation on CPU. Production GPU headroom,
throughput and first-update stability must still be checked at the actual latest-
checkpoint boundary. It does not license the observer's proposed choices or claim
that four copied updates acquired a new ability. An extra comparison of the two
first-pass studies found matching displayed losses but nonidentical trained
tensors (largest difference 1.74e-6 after two updates; checkpoint cadence also
differed). No bitwise equality of separately trained runs is claimed. The original
pre-learning tensor/Adam preservation check is exact.

**The actual learning-opportunity audit was also exercised.** The native
continuation saved step 4,012,831 after six observer calls and retained one
completed comparison across all eight bodies for vision, touch, audio and
proprioception. Each comparison contains the actual prediction error, the error
with a shuffled body command, and the error from assuming the observation stays
unchanged. One fixed anchor is retained; the required retention panel contains
eight examples, so the meter correctly credits no established learning progress
yet. The next scheduled audit is pending and survives the full checkpoint.

Audit scheduling uses the global training step modulo five, rather than the
number of observer calls. Because the original birth step 4,012,825 was divisible
by five, the original four-observation pair had already completed an audit. The
additional two-update continuation was therefore unnecessary for that branch.
Its initial readback validator also wrongly rejected the next pending audit and
then required the forward model to omit extra uncounted motor outputs. Both
validation mistakes were corrected; the saved native training run itself passed.
The receipt is `Saved/mimo_bridge/native_local_shadow_audit_20260913/report.json`.
No additional training was run to repair the validator.

Reviewing the later reference-refresh branch found a separate defect: loading
new reference weights did not invalidate cached motor embeddings. That branch
has now been repaired, including cache invalidation on restore. Two focused
actual-model tests verify the resulting forecasts against fresh caches. The
short native continuation did not reach the eight-example retention branch;
the focused tests provide that branch's verification. Production starts a fresh
shadow under the repaired execution fingerprint.

## Exact boundary preparation

The trainer owner first requests a complete final checkpoint while controlling
the existing supervisor's restart race. Select that newest finished checkpoint;
do not adopt the older copied-study source merely because it passed preservation.
With `$sourceCheckpoint` set to that actual checkpoint and `$sameStep` to its
recorded step, run from the repository with `PYTHONPATH=Source/Python`:

```powershell
.venv_mimo/Scripts/python.exe -m mimo_bridge.tools.prepare_local_shadow $sourceCheckpoint `
  --out "D:/UEMA_Research/external/D/UEMA_recordings/checkpoint_runs/H80_local_shadow_20260913/checkpoints/ckpt_step_$sameStep" `
  --calibration D:/UEMA_Research/external/D/UEMA_recordings/shadow_local_calibration_20260913/shadow_calibration.json `
  --episode-storage D:/UEMA_Research/external/D/UEMA_recordings/H80_local_shadow_20260913/checkpoint_state
```

Do not supply `--cpu-probe` to this production preparation. The tool refuses any
old tensor/Adam loss, unexpected new parameter, changed historical episode,
incorrect target step, missing continuation, source actor, or mismatched profile.
It publishes runtime and physical companion inside the checkpoint transaction.

Reuse the supervisor's exact current native command and change only these flags:

| Flag | Value |
|---|---|
| `--resume` | The newly prepared checkpoint at the actual final source step |
| `--out` | A new local-shadow run directory for current logs |
| `--checkpoint-storage` | That run's configured checkpoint directory |
| `--contact-profile` | `mechanical_v2_local` |
| `--self-choice` | `shadow` |
| `--self-choice-receipt` | `D:/UEMA_Research/external/D/UEMA_recordings/shadow_local_calibration_20260913/shadow_calibration.json` |

Retain all other body, sensory, cadence, replay, caregiver, stillness and learning
settings. Do not lower thresholds or turn the shadow's calibration into a claim
that its choices improve function.

## Calibration scope

The local-skin calibration replays one actual lived-in world exactly and measures
action-dependent sensory distinctions under the native frame contract. Its
pressure separation was 41,621 Pa; the coarsest sufficient precision was three
bits, rather than an arbitrary high quality-index demand. That receipt supplies
the shadow's declared operating criteria, not a universal pain threshold, a
forecast-accuracy qualification or proof across every body. Native eight-world
startup checks compatibility; independent held-out physical choices must still
earn any later promotion to body control.
