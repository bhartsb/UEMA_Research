# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Per-body presence vector: how much each named thing is available to the infant now.

PRESENCE means availability, not quality and not value. A number of 1 says the
thing is there for the infant to see or reach at this moment; 0 says it is not
there at all. Nothing here says whether the thing is good, wanted, or attended to.

The inputs are the two dictionaries the environment already reports every step:
the task block (`info["task"]`, built in `MimoEnv._observation`) and the
caregiver block (`info["caregiver"]`, the `Caregiver.diagnostics` fields plus the
mode, reply count, object style and event count added alongside them). Both ride
on every `BridgeFrame` as `frame.meta["task"]` and `frame.meta["caregiver"]`.

WHICH KEY EACH ENTITY READS, and whether the environment reports it graded or
yes/no:

  caregiver          graded  `adult_visibility`, the adult body's alpha, which
                             `caregiver_visual.pose_adult` sets to
                             (0.95 m - distance) / 0.20 m clipped to 0..1, so the
                             adult fades over the last 20 cm of walking away.
                             Gated by `adult_present` (which the caregiver's own
                             diagnostics define as adult_visibility > 0).
  toy_ball           yes/no  style 0. See the style mapping below.
  toy_pad            yes/no  style 1.
  toy_ring           yes/no  style 2.
  mobile             yes/no  `mobile_present`.
  squeaker           graded  `squeaker_view_deg`, the angle between the left eye
                             camera's forward axis and the squeaker panel.
  critter_dragonfly  graded  `dragonfly_deg`, the same eye angle, and only while
                             `critter_visitor` reads "dragonfly".
  critter_bee        graded  `bee_deg`, likewise, and only while
                             `critter_visitor` reads "bee".
  own_hand_seen      graded  `hand_view_deg`, the same eye angle to the nearer
                             hand. THE CONTROL ENTITY: it belongs to the body, so
                             the infant carries it everywhere and it never
                             withdraws on an adult's schedule the way the world's
                             entities do.

THE THREE TOY STYLES. The adult brings one toy at a time and `object_style`
selects which. The integer is checked three ways in the source rather than
guessed: `Caregiver.object_geoms` lists the shapes in the order target_geom,
caregiver_pad, caregiver_ring_*; `Caregiver.tick` speaks the phrase
('ball', 'cloth', 'ring')[style] when it presents one; and
`caregiver_language.OBJECT_NAMES` is ('ball', 'pad', 'ring'). So 0 is the ball,
1 is the pad, 2 is the ring. Only the selected style can read above 0; the other
two read 0, because `Caregiver._appearance` sets every other style's shapes to
alpha 0 with collisions switched off.

THE ANGLE RULE. An eye angle becomes presence as 1 - degrees / 30, clipped to
0..1: 1.0 when the thing sits on the eye's forward axis, 0.5 at 15 degrees off,
0.0 at 30 degrees or further. The 30 degrees is the environment's own line, not a
new constant: `MimoEnv._observation` sets `hand_in_view` and `squeaker_seen` true
below 30 degrees and calls it "the 60 deg wide field". So presence above 0 holds
exactly when the environment's own yes/no flag is true, and the graded value adds
where inside that field the thing sits.

GRADED WHERE THE SIGNAL IS GRADED. The caregiver's reported block carries
`object_present` and `mobile_present` as yes/no only; the fractional
`object_visibility` and `mobile_visibility` that `Caregiver._availability`
computes are kept in the caregiver's own state and in its recorded tape, but
`Caregiver.diagnostics` does not export them. This module reads the fractional
value when it is present in the dictionary it is handed and falls back to the
yes/no flag when it is not, so it gives 0/1 on a live trainer frame and the
fractional fade when replayed against a caregiver tape record. The same holds for
the style key, which the reported block calls `object_style` and the tape record
calls `style`.

A MISSING KEY READS 0.0. It is never an error and never a guess. A value that is
not a finite number reads 0.0 for the same reason.

WHAT IS LEFT OUT, and why:
  * the weight boxes (`box_touch`, `box_touch_hand`) - reported as a list of
    classes currently in contact, which is touching, not availability; the crib
    scene these runs use reports an empty list on every step.
  * the adult's hands and torso - posed and drawn, but the environment reports no
    separate visibility or presence number for them; they fade with the adult's
    own alpha, which `caregiver` already carries.
  * the adult's speech (`last_utterance`) - an event that starts and ends, not a
    thing standing in the world with an availability between 0 and 1.
  * the individual mobile elements - the environment reports the mobile as one
    body (`mobile_present`, `mobile_speed_cms`, `mobile_touched`), never element
    by element.
  * the crib walls, floor and support fixture - no presence or visibility field is
    reported for them anywhere in the task or caregiver block.

COST. Measured 2026-09-20 on this machine, CPU only with the processor hidden,
against real explainer snapshot dictionaries from run H84: `entity_presence_batch`
for 8 bodies takes 24 microseconds per call at best and 27 at the median of 5
blocks of 20,000 calls, and one body through `entity_presence` takes 3.4
microseconds. For scale, that run logged 759.5 ms per model step, so the batch
call is 0.004% of one step. It is dictionary reads, arithmetic and one tensor
build - no randomness and no model call, so the same dictionaries always produce
the same numbers.
"""
from __future__ import annotations

import math
from typing import Sequence

# Fixed order. This tuple is the module's contract: a stored presence vector's
# column meaning comes from this order, so appending is safe and reordering or
# removing a name invalidates every vector written before the change.
ENTITY_NAMES: tuple[str, ...] = (
    "caregiver",
    "toy_ball",
    "toy_pad",
    "toy_ring",
    "mobile",
    "squeaker",
    "critter_dragonfly",
    "critter_bee",
    "own_hand_seen",
)

# The half-width of the environment's own wide visual field, in degrees. Read off
# MimoEnv._observation, where hand_in_view and squeaker_seen are both set true
# below this angle.
VIEW_FIELD_HALF_DEG = 30.0

# object_style / style integer -> the entity name that integer selects. Taken from
# Caregiver.object_geoms, the presentation phrase in Caregiver.tick, and
# caregiver_language.OBJECT_NAMES, which all agree.
TOY_STYLE_TO_NAME: dict[int, str] = {0: "toy_ball", 1: "toy_pad", 2: "toy_ring"}

ENTITY_DOC: dict[str, str] = {
    "caregiver":
        "0 means the adult is out of sight, 1 means the adult is standing in full "
        "view beside the crib, and a value in between is the adult part way "
        "through fading out as they walk off or fading back in as they return.",
    "toy_ball":
        "0 means the ball is not the toy the adult has out, or it has been carried "
        "away, and 1 means the ball is the toy out now and within reach.",
    "toy_pad":
        "0 means the soft pad is not the toy the adult has out, or it has been "
        "carried away, and 1 means the pad is the toy out now and within reach.",
    "toy_ring":
        "0 means the ring is not the toy the adult has out, or it has been carried "
        "away, and 1 means the ring is the toy out now and within reach.",
    "mobile":
        "0 means the mobile has been lifted up out of the crib, and 1 means it is "
        "hanging overhead where the infant can see and bat it.",
    "squeaker":
        "0 means the squeaker panel by the feet is 30 degrees or more off where the "
        "eyes point, so it is outside the visual field, 1 means it is straight "
        "ahead, and 0.5 means it is 15 degrees off to one side.",
    "critter_dragonfly":
        "0 means the dragonfly is not the visitor right now or is 30 degrees or "
        "more off where the eyes point, 1 means it is straight ahead, and 0.5 "
        "means it is 15 degrees off to one side.",
    "critter_bee":
        "0 means the bee is not the visitor right now or is 30 degrees or more off "
        "where the eyes point, 1 means it is straight ahead, and 0.5 means it is "
        "15 degrees off to one side.",
    "own_hand_seen":
        "0 means neither of the infant's own hands is within 30 degrees of where "
        "the eyes point, 1 means a hand is straight ahead in the middle of the "
        "view, and 0.5 means the nearer hand is 15 degrees off to one side.",
}


def _number(source: dict, key: str):
    """The finite float behind `key`, or None when the key is absent, is not a
    number, or holds something that is not finite. Booleans count as numbers:
    True is 1.0 and False is 0.0."""
    if not isinstance(source, dict):
        return None
    value = source.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _from_angle(source: dict, key: str) -> float:
    """An eye angle in degrees turned into presence: 1.0 on the forward axis,
    0.0 at the 30 degree edge of the wide field, straight line in between."""
    degrees = _number(source, key)
    if degrees is None:
        return 0.0
    return _clip01(1.0 - abs(degrees) / VIEW_FIELD_HALF_DEG)


def _graded_or_flag(source: dict, graded_key: str, flag_key: str) -> float:
    """The fractional value when the dictionary carries one, otherwise the yes/no
    flag as 1.0 or 0.0, otherwise 0.0 because the key is missing."""
    value = _number(source, graded_key)
    if value is None:
        value = _number(source, flag_key)
    return 0.0 if value is None else _clip01(value)


def entity_presence(task: dict, caregiver: dict) -> list[float]:
    """One body's presence vector: len(ENTITY_NAMES) numbers in 0..1, in the order
    ENTITY_NAMES gives.

    task: the environment's task block for this body (frame.meta["task"]).
    caregiver: the environment's caregiver block for this body
        (frame.meta["caregiver"]), or a caregiver tape record, or the caregiver's
        own state dictionary - all three carry the keys read here.
    Either may be None or an empty dictionary; every entity then reads 0.0.
    """
    task = task if isinstance(task, dict) else {}
    caregiver = caregiver if isinstance(caregiver, dict) else {}

    # The adult. Gone means 0 whatever the fade value says; present means the fade
    # value itself, which is 1.0 while the adult stands at the crib side.
    adult_flag = _number(caregiver, "adult_present")
    if adult_flag is not None and adult_flag <= 0.0:
        adult = 0.0
    else:
        adult = _graded_or_flag(caregiver, "adult_visibility", "adult_present")

    # The three toy styles. Only the style currently selected can read above 0;
    # the other two styles are drawn at alpha 0 with collisions off, so they are
    # not available to the infant at all.
    style = _number(caregiver, "object_style")
    if style is None:
        style = _number(caregiver, "style")       # the caregiver tape's own name
    toy_out = 0.0 if style is None else _graded_or_flag(
        caregiver, "object_visibility", "object_present")
    selected = None if style is None else TOY_STYLE_TO_NAME.get(int(style))

    mobile = _graded_or_flag(caregiver, "mobile_visibility", "mobile_present")

    # The critters visit one at a time; the environment writes the angle key only
    # for the one currently visiting, and names it in critter_visitor.
    visitor = task.get("critter_visitor")
    visitor = visitor if isinstance(visitor, str) else ""
    dragonfly = _from_angle(task, "dragonfly_deg") if visitor == "dragonfly" else 0.0
    bee = _from_angle(task, "bee_deg") if visitor == "bee" else 0.0

    values = {
        "caregiver": adult,
        "toy_ball": toy_out if selected == "toy_ball" else 0.0,
        "toy_pad": toy_out if selected == "toy_pad" else 0.0,
        "toy_ring": toy_out if selected == "toy_ring" else 0.0,
        "mobile": mobile,
        "squeaker": _from_angle(task, "squeaker_view_deg"),
        "critter_dragonfly": dragonfly,
        "critter_bee": bee,
        "own_hand_seen": _from_angle(task, "hand_view_deg"),
    }
    return [values[name] for name in ENTITY_NAMES]


def entity_presence_batch(tasks: Sequence[dict],
                          caregivers: Sequence[dict]) -> "torch.Tensor":
    """The presence vectors for a whole batch of bodies as a [B, N] float32 tensor
    on the processor, where B is the number of bodies and N is len(ENTITY_NAMES).
    Row b is exactly entity_presence(tasks[b], caregivers[b]).

    The two sequences must be the same length: a mismatch pairs one body's task
    block with another body's caregiver block, which is a wiring mistake rather
    than a missing key, so it raises instead of reading 0.
    """
    import torch

    tasks = list(tasks)
    caregivers = list(caregivers)
    if len(tasks) != len(caregivers):
        raise ValueError("entity_presence_batch needs one caregiver block per task "
                         "block; got %d task and %d caregiver blocks"
                         % (len(tasks), len(caregivers)))
    rows = [entity_presence(t, c) for t, c in zip(tasks, caregivers)]
    if not rows:
        return torch.zeros((0, len(ENTITY_NAMES)), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)
