# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The developmental observer's analysis: watch development without programming it.

Addendum v4 Part 6 (docs/gain_of_function_v4/UEMA_Embodied_Gain_of_Function_Addendum_v4.html);
design, contracts and the decisions log in docs/gain_of_function_v4/OBSERVER_DESIGN.md.

The recorders (mimo_bridge/observer_record.py in each world worker, mimo_bridge/
observer_model_lane.py in the trainer) write files under <run>/observer/. Everything in this
package runs in its OWN process and reads those files only; nothing here is imported by the
trainer, a world or the learner (tests/test_observer_record.py pins the boundary). The layers,
in the addendum's section 22 order:

  stream.py      1 raw world stream: ordered chunks per body, relaunch branches pruned, gaps
  kinematics.py  the body's geometry from the saved world model (finger curl, wrist axes, face
                 and mouth zones, trunk roll, eye axes), exactly as the simulator placed it
  features.py    per-step measurements of one body over a window (the "track")
  primitives.py  2 primitive events: contact on/off, finger curl, hand in view, fixation,
                 object lift/drop, support shift, posture transitions, transients
  segmenter.py   3 episodes: primitives grouped into time windows
  detectors.py   4 the catalog's 53 detectors (and the finger-movement rows), measurements first
  evidence.py    the intent/prediction join: provenance, the learner's prediction before the
                 act, the planner's selection
  tracker.py     5 development tracker: observed, repeated, predicted, selected, generalized,
                 retained, against babble baselines
  novelty.py     6 recurring event sequences no named detector explains
  fastlane.py    section 24: peaks, areas, recovery of 50-100 ms transients
  horizon.py     section 30: how far ahead the learner's imagination is accurate
  validate.py    section 27: scripted positive / negative / boundary worlds, replay determinism
  baseline.py    babble-only and limp physics runs: each detector's chance rate
  replay.py      replay links: an episode rendered from its record
  daemon.py      digests new chunks, writes <run>/observer/analysis/, frees digested raw data
                 7 the dashboard reads the analysis files (tools/development_dashboard.py)
"""

import os as _os

# One math-library thread in the observer's own processes. With the default (one per core), the
# numpy and scipy imports alone committed 1,545 MB of private memory in the digester, against 67 MB
# with one thread (measured 2026-09-24 beside the live trainer, whose commit headroom this shares).
# Set here, before this package imports numpy; a variable already set by the caller is kept.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_name, "1")
