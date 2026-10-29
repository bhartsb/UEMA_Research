# Self learning: implementation and measured limits

September 13, 2026. This report covers copied learning experiments, local sensing,
and checkpoint preservation. It does not award a developmental ability.

The current-function input is now preserved locally and the future predictor can
learn changes from actual commands and elapsed time. The copied predictor still
fails its practical forecast requirement in fresh worlds. A second obstruction
is physical: the passive-contact test has controllable contact, but no measurable
loss of sensing to recover. Neither result justifies turning on a new chooser.

## What was implemented

`lem/model/self_state.py` adds the optional `grounded_residual` prediction mode.
All existing parameters keep their names and shapes. It retains the learned
local representation, recurrent belief, and relation memory. It explicitly
distinguishes the observed current function from the learned current readout.
An exact observed input is never counted as a learned prediction.

New future dynamics start from observed local function and learn a residual over
physical seconds. A recurrent state reads every executed command and duration.
The old learned belief and the actual local touch fields remain inputs. Action
normalization is fitted only on development commands and stored in the weights.
Healthy sites and affected sites retain separate prediction lessons, preventing
many unchanged sites from concealing a bad local prediction. Actual repeated
physics branches can provide a separate action-contrast lesson; unrelated
parallel bodies cannot be labeled as alternative outcomes of one state.

The subsequently added spatial attention supplies real retinal/tactile region
features to local self queries. Its output starts at zero and the old global
sensory tokens remain unchanged. The saved fitting data below predates this
spatial route. Its numerical results therefore do not establish that the new
spatial route predicts contact.

`mimo_bridge/self_candidate.py` can import only the new dynamics and their Adam
histories into the newest learner. It refuses to overwrite already earned
destination dynamics. An older copied whole checkpoint must never replace the
newer production brain or its physical episode.

The native verification against intact H78 checkpoint 3,995,492 preserved all
364 original tensors and all 308 earned optimizer states exactly. It imported
18 new dynamics tensors and 16 dynamics optimizer states. Four newer spatial
matrices remained neutral. Receipt:
`Saved/mimo_bridge/self_candidate_latest_verified_20260913/report.json`.

## What the copied experiments found

Action-error removal compares action-dependent predictions with ignoring the
action. Zero means no improvement; one means perfect action differences;
negative means worse than ignoring the action. The declared requirement is at
least 75% removal in every confirmation world, corresponding to prediction RMS
error at most half the measured action separation. This is a chosen engineering
resolution requirement, not a biological developmental threshold.

| Attempt | Data and result | Decision |
|---|---|---|
| Persistence plus learned residual, 480 updates | Original eight regression worlds: action-error removal about −15% to +4% | Failed |
| Paired actual action contrasts, 960 updates | Eight development worlds: about 8–24% removal | Development only |
| Paired contrasts plus command scaling, 960 updates | Development: about 13–34%; lowest-cost candidate selected in 7/8 worlds | Holding was best in all eight. This is not evidence of gaining a skill |
| One-second prediction, 1,600 updates | Development motor-capacity removal 79–97%; skin prediction failed | Freeze weights and test separate domains on new worlds |
| Five-second frozen candidate, fresh worlds 8300–8307 | Removal −97.5% to +3.6% | Failed; no adoption |
| One-second frozen candidate, same fresh physical branches | Motor removal listed below; skin failed | Neither domain qualified |

| Fresh world | Motor action-error removed | Motor mean absolute force-gain error | Worst local motor error |
|---|---:|---:|---:|
| 8300 | 61.9% | 0.101 percentage points | 1.72 points |
| 8301 | 66.3% | 0.132 points | 2.05 points |
| 8302 | 59.2% | 0.146 points | 2.77 points |
| 8303 | 82.8% | 0.105 points | 1.48 points |
| 8304 | −23.0% | 0.144 points | 2.27 points |
| 8305 | 53.9% | 0.135 points | 2.66 points |
| 8306 | 24.4% | 0.123 points | 2.07 points |
| 8307 | 38.9% | 0.115 points | 2.82 points |

Small averages still conceal individual skin errors as large as 0.55 of the
precision-index range. Forecast uncertainty was much too broad to license a
chooser. A positive result in development did not transfer reliably to new
worlds; the failed confirmation is retained.

Evidence folders:

- `Saved/mimo_bridge/self_grounded_dynamics_20260913`
- `Saved/mimo_bridge/self_grounded_contrast_20260913`
- `Saved/mimo_bridge/self_grounded_calibrated_20260913`
- `D:/UEMA_Research/external/D/UEMA_recordings/self_grounded_one_second_20260913`
- `D:/UEMA_Research/external/D/UEMA_recordings/self_grounded_fresh_20260913`
- `D:/UEMA_Research/external/D/UEMA_recordings/self_grounded_domains_20260913`

Each folder contains its declaration, result and copied evidence. The first
development scripts were extended while earlier variants were still running;
their completion-time source hashes do not certify the executing source. Keep
the declared flags and this provenance limitation. The one-second and fresh
confirmation runs captured launch hashes; the fresh producer reports no source
change during its run. No confirmation outcomes were used for another fit.

## Contact and sensing: the physical limitation

Skin function is a precision index, not a percentage of wellness, pain, or usable
knowledge. Bits are rounded from `2 + 21q`; below full function, the pressure grid
spacing is `100,000 / (2**bits - 1)` pascals. At full function the actual operator
bypasses quantization. A theoretical 23-bit grid must not invent lost distinctions
when the original float32 signal is unchanged. Motor function is different: its
first field is an actual available actuator-force fraction.

The original top-of-hand source often pushed the hand away before the decision
started. Holding then recovered automatically. A new passive sphere below the
hand exposed another issue: it actually contacted the lower arm. The fixture now
selects the skin body actually contacted at decision time and records contact
geometry, force, pressure, sensed pressure, and actual precision changes.

In the corrected measured scene, an ordinary command maintained 7.488 N against
the sphere. The adapter divided it by 97.92 cm² of active sensor support, yielding
764.7 Pa on average. Other commands released the contact. All q values stayed at
one and no pressure observation changed. Source removal and movement therefore
provided no restoration opportunity. Same-state repeated trajectories matched
exactly. This is physical contact control, not learned purposeful behavior.

Report: `D:/UEMA_Research/external/D/UEMA_recordings/function_support_pressure_area_20260913/report.json`.
Earlier support reports retain their original values. Their full-function
theoretical-grid collision count and recomputed initial contact force were
corrected in the later audit; the unchanged-sensing verdict was unaffected.

Sensor interpolation support is not a resolved tissue contact footprint.
MuJoCo provides contact geometry and a parameterized soft contact model; its
solver penetration must not silently become a measured biological indentation.
Any improved pressure transducer must state its geometry/compliance assumptions,
conserve normal force, and remain invariant to solver-contact splitting and
sensor resampling. [MuJoCo contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#soft-contact-model)

## The overnight checkpoint failure and repair

H78 stopped at 04:27 after C: filled during `runtime_state.pt`. The model and
optimizer directory had already been published, so step 3,998,932 contained a
truncated runtime archive. The supervisor resumed the preceding intact step
3,995,492 at 04:28. That process then stopped at 04:29 before a new learning step
with a CUDA unknown error; its crash-save also encountered full storage. The
supervisor itself died trying to write its status JSON to the full drive. The
CUDA error is recorded separately; the log does not establish its cause.

The repaired transaction writes model, optimizer, runtime and physical episode
before publishing the checkpoint directory. A failed same-step replacement
retains its predecessor. New checkpoints require runtime evidence; corrupt or
missing runtime can no longer silently reset learned working state. The
supervisor skips corrupt new checkpoints, retains earlier valid fallback, and
keeps supervising when its status file cannot be written.

`--checkpoint-storage` allows a per-run directory on the recording drive while
retaining normal run logs. The external checkpoint parent gets the actual
arguments. Each automatic continuation receives its own external checkpoint
directory. Pre-write space estimates include shared tensor storage, serialization
overhead and a 64 MiB margin. Atomic publication still protects against another
process filling the drive concurrently.

Targeted verification includes old-weight/Adam preservation, real physical
continuation, action/time sensitivity, affected/healthy balancing, invalid
candidate rejection before mutation, full-disk fault injection, corrupt-runtime
rejection, external-storage lookup and actor save/resume. These tests establish
the implementation contracts, not a gain of function.

## Adoption and the next acquisition

Keep the newest intact brain and physical episode. Add the grounded extension
through the verified native migration and candidate-only initializer. Continue
learning on actual modulation; qualification of a frozen acting candidate remains
separate. Do not replace the existing chooser with the failed copied candidate.

Conserving full function alone makes stillness a complete solution in many
healthy scenes. Acquisition therefore also needs a bounded reason to learn new
controllable distinctions while the body remains viable. The parallel learning-
opportunity work compares actual-action prediction with shuffled-action evidence,
tracks reducible improvement, and limits repeated interest after mastery. It must
influence candidate choices and survive uncontrolled noise and sensor duplication;
a new dashboard gauge alone is insufficient. No rolling, toy-touching, or other
named-skill reward is warranted by this repair.
