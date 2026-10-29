# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""One detector version names one set of definitions (the independent review of 2026-09-25, finding 1,
and the coordinator's decision of that evening).

The digester re-digests a run, and a babble summary counts as current, only while the version string
matches (OBSERVER_DESIGN.md section 3). Two branches changed the definitions under the same name,
"observer_detectors_v9": the recorder branch (fix/observer-recorder-2026-09-25: R04 and the H02 family
read as the physics of a grip from contact normals) and the detectors branch
(fix/observer-detectors-2026-09-25: the motor gate, turns known once confirmed, the escape gravity does
not explain). Whichever of the two merges second says "observer_detectors_v10" and updates
test_observer_segments.py's pin, even though the three observer branches merge in one pause, so no two
definitions ever share a name. A later definition change bumps the version and this table with it.
"""
from mimo_bridge.tools.observer import daemon as DM
from mimo_bridge.tools.observer import detectors as D
from mimo_bridge.tools.observer import primitives as P


def _recorder_branch() -> bool:
    return hasattr(D, "_grip") and hasattr(D, "NEEDS_FIELDS")


def _detectors_branch() -> bool:
    return hasattr(D, "MOTOR_ON") and hasattr(P, "turns_confirmed")


def test_the_detector_version_names_the_definitions_this_tree_holds():
    held = (_recorder_branch(), _detectors_branch())
    want = {(True, False): "observer_detectors_v9",       # one branch alone
            (False, True): "observer_detectors_v9",
            (True, True): "observer_detectors_v11"}       # Oct 12: onset attribution and two-hand contact timing
    assert held in want, held
    assert DM.DETECTOR_VERSION == want[held], (
        "this tree holds the recorder branch's definitions: %s, the detectors branch's: %s; they are "
        "named %s. The branch merged second sets DETECTOR_VERSION to %s (daemon.py) and updates "
        "test_observer_segments.py's pin" % (held[0], held[1], DM.DETECTOR_VERSION, want[held]))
