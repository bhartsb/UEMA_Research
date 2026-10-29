# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Every rule in entity_presence, checked on hand-made dictionaries and then on
real explainer snapshots written by a training run."""
import json
import math
import re
from pathlib import Path

import pytest

from mimo_bridge.entity_presence import (ENTITY_DOC, ENTITY_NAMES,
                                         TOY_STYLE_TO_NAME, VIEW_FIELD_HALF_DEG,
                                         entity_presence, entity_presence_batch)

REPO = Path(__file__).resolve().parents[4]
NAMED_RUN = (REPO / "Saved" / "mimo_bridge" / "experiments"
             / "H84_fresh_drivers_20260918_seg8_slot_names" / "explain")


def presence(task=None, caregiver=None):
    """The presence vector as a name -> value dictionary, which keeps the
    assertions below readable without depending on column order."""
    return dict(zip(ENTITY_NAMES, entity_presence(task or {}, caregiver or {})))


def adult(visibility=1.0, present=True, **rest):
    block = dict(adult_visibility=visibility, adult_present=present)
    block.update(rest)
    return block


# ---------------------------------------------------------------- the contract

# One world per value each entity's sentence claims it can read, as the (task, caregiver)
# pair that produces it. This table is what turns the sentences from prose into a checked
# claim: every "N means ..." in a sentence has to appear here, and every world here has to
# make entity_presence return exactly that N. Until 2026-09-21 the check was that the
# sentence contained the characters "0 " and "1 " and the word "means" twice, which a
# sentence naming the WRONG scale passes - "0 means straight ahead and 1 means out of
# sight" would have been accepted with the numbers exactly backwards.
ANGLE_SCALE = {
    0.0: VIEW_FIELD_HALF_DEG,               # the edge of the wide field: nothing is available
    0.5: VIEW_FIELD_HALF_DEG / 2.0,         # half a field off to one side
    1.0: 0.0,                               # straight ahead on the eye's forward axis
}


def _angle_worlds(key, extra=None):
    return {value: [(dict({key: degrees}, **(extra or {})), {})]
            for value, degrees in ANGLE_SCALE.items()}


def _toy_worlds(style):
    """A toy reads 1 only while it is the style the adult has out AND it is out; it reads 0
    both when another style is out and when its own style has been carried away."""
    others = [s for s in TOY_STYLE_TO_NAME if s != style]
    return {
        0.0: [({}, adult(object_style=style, object_present=False)),
              ({}, adult(object_style=others[0], object_present=True))],
        1.0: [({}, adult(object_style=style, object_present=True))],
    }


SCALE_WORLDS = {
    "caregiver": {0.0: [({}, adult(0.0, False))], 1.0: [({}, adult(1.0, True))]},
    "toy_ball": _toy_worlds(0),
    "toy_pad": _toy_worlds(1),
    "toy_ring": _toy_worlds(2),
    "mobile": {0.0: [({}, dict(mobile_present=False))],
               1.0: [({}, dict(mobile_present=True))]},
    "squeaker": _angle_worlds("squeaker_view_deg"),
    "critter_dragonfly": _angle_worlds("dragonfly_deg", dict(critter_visitor="dragonfly")),
    "critter_bee": _angle_worlds("bee_deg", dict(critter_visitor="bee")),
    "own_hand_seen": _angle_worlds("hand_view_deg"),
}


def test_entity_names_are_unique_and_documented():
    """The documented scale of every entity, checked against the number the function returns
    in a world built to produce it - not against the shape of the sentence.

    What is checked: which values the sentence claims this entity can read; that a world
    exists for each of them and that entity_presence returns exactly that value in it; and,
    for the four entities read from an eye angle, that the angle the sentence puts at 0 is
    the wide field's own half-width and the angle it puts at 0.5 is half of that. What is
    not checked, because no mechanical test can read English: whether the words describing
    an end match the world that produces it for the five entities whose scale carries no
    number - the adult, the three toys and the mobile."""
    assert len(set(ENTITY_NAMES)) == len(ENTITY_NAMES)
    assert set(ENTITY_DOC) == set(ENTITY_NAMES)
    assert set(SCALE_WORLDS) == set(ENTITY_NAMES), (
        "an entity was added without a world that shows what its two ends read")
    for name, sentence in ENTITY_DOC.items():
        assert sentence.strip().endswith("."), name
        pieces = re.split(r"\b(0\.5|0|1) means\b", sentence)
        clauses = {float(pieces[i]): pieces[i + 1] for i in range(1, len(pieces), 2)}
        assert {0.0, 1.0} <= set(clauses), ("the sentence must say what 0 and what 1 mean", name)
        assert set(clauses) == set(SCALE_WORLDS[name]), (
            "the sentence and the worlds disagree about which values this entity can read",
            name, sorted(clauses), sorted(SCALE_WORLDS[name]))
        for value, worlds in SCALE_WORLDS[name].items():
            for task, caregiver in worlds:
                got = dict(zip(ENTITY_NAMES, entity_presence(task, caregiver)))[name]
                assert got == pytest.approx(value), (
                    "the sentence says this world reads %s and the function returns %s"
                    % (value, got), name, task, caregiver)
        # every angle the sentence names has to be the field's own half-width or half of it
        for degrees in re.findall(r"(\d+(?:\.\d+)?) degrees", sentence):
            assert float(degrees) in (VIEW_FIELD_HALF_DEG, VIEW_FIELD_HALF_DEG / 2.0), (
                "the sentence names %s degrees; the wide field's half-width is %s"
                % (degrees, VIEW_FIELD_HALF_DEG), name)
        # and for an entity read from an eye angle, each angle has to sit beside the value it
        # produces: the field's edge at 0, half a field off at 0.5, no angle at all at 1
        if 0.5 in clauses:
            assert "%g degrees" % VIEW_FIELD_HALF_DEG in clauses[0.0], (name, clauses[0.0])
            assert "%g degrees" % (VIEW_FIELD_HALF_DEG / 2.0) in clauses[0.5], (name, clauses[0.5])
            assert "degrees" not in clauses[1.0], (
                "the end that reads 1 is the forward axis, which is no angle at all",
                name, clauses[1.0])


def test_vector_length_matches_the_name_list():
    assert len(entity_presence({}, {})) == len(ENTITY_NAMES)


# ------------------------------------------------------------------ the adult

def test_caregiver_absent_reads_zero_and_fully_visible_reads_one():
    assert presence(caregiver=adult(0.0, False))["caregiver"] == 0.0
    assert presence(caregiver=adult(1.0, True))["caregiver"] == 1.0


def test_caregiver_half_faded_reads_between():
    value = presence(caregiver=adult(0.4, True))["caregiver"]
    assert 0.0 < value < 1.0
    assert value == pytest.approx(0.4)


def test_caregiver_gone_beats_a_stale_visibility():
    # adult_present false settles it even if a visibility number is still there
    assert presence(caregiver=adult(0.9, False))["caregiver"] == 0.0


# ------------------------------------------------------------- the three toys

def test_only_the_style_that_is_out_can_read_above_zero():
    for style, name in TOY_STYLE_TO_NAME.items():
        out = presence(caregiver=adult(object_style=style, object_present=True))
        assert out[name] == 1.0, name
        others = [n for n in TOY_STYLE_TO_NAME.values() if n != name]
        assert [out[n] for n in others] == [0.0, 0.0], name


def test_the_style_that_is_out_reads_zero_once_it_is_carried_away():
    out = presence(caregiver=adult(object_style=1, object_present=False))
    assert out["toy_pad"] == 0.0
    assert out["toy_ball"] == 0.0 and out["toy_ring"] == 0.0


def test_a_fractional_object_visibility_is_carried_through():
    # the caregiver's own state and its recorded tape carry the fade; the
    # reported block carries the yes/no flag alone
    out = presence(caregiver=dict(style=2, object_present=True,
                                  object_visibility=0.25))
    assert out["toy_ring"] == pytest.approx(0.25)
    assert 0.0 < out["toy_ring"] < 1.0


def test_an_unknown_style_integer_leaves_every_toy_at_zero():
    out = presence(caregiver=adult(object_style=7, object_present=True))
    assert [out[n] for n in TOY_STYLE_TO_NAME.values()] == [0.0, 0.0, 0.0]


# ----------------------------------------------------------------- the mobile

def test_mobile_present_reads_one_and_lifted_away_reads_zero():
    assert presence(caregiver=dict(mobile_present=True))["mobile"] == 1.0
    assert presence(caregiver=dict(mobile_present=False))["mobile"] == 0.0


def test_mobile_half_lifted_reads_between():
    value = presence(caregiver=dict(mobile_present=True,
                                    mobile_visibility=0.6))["mobile"]
    assert 0.0 < value < 1.0
    assert value == pytest.approx(0.6)


# ---------------------------------------------- the angle rule, shared by four

@pytest.mark.parametrize("name, keys", [
    ("squeaker", dict(squeaker_view_deg=0.0)),
    ("own_hand_seen", dict(hand_view_deg=0.0)),
    ("critter_dragonfly", dict(critter_visitor="dragonfly", dragonfly_deg=0.0)),
    ("critter_bee", dict(critter_visitor="bee", bee_deg=0.0)),
])
def test_straight_ahead_reads_one(name, keys):
    assert presence(task=keys)[name] == 1.0


@pytest.mark.parametrize("name, key, extra", [
    ("squeaker", "squeaker_view_deg", {}),
    ("own_hand_seen", "hand_view_deg", {}),
    ("critter_dragonfly", "dragonfly_deg", dict(critter_visitor="dragonfly")),
    ("critter_bee", "bee_deg", dict(critter_visitor="bee")),
])
def test_half_a_field_off_reads_half_and_the_field_edge_reads_zero(name, key, extra):
    half = presence(task=dict({key: 0.5 * VIEW_FIELD_HALF_DEG}, **extra))[name]
    assert half == pytest.approx(0.5)
    assert 0.0 < half < 1.0
    assert presence(task=dict({key: VIEW_FIELD_HALF_DEG}, **extra))[name] == 0.0
    assert presence(task=dict({key: 179.0}, **extra))[name] == 0.0


def test_only_the_visiting_critter_can_read_above_zero():
    task = dict(critter_visitor="dragonfly", dragonfly_deg=5.0, bee_deg=5.0)
    out = presence(task=task)
    assert out["critter_dragonfly"] > 0.0
    assert out["critter_bee"] == 0.0
    task["critter_visitor"] = "bee"
    out = presence(task=task)
    assert out["critter_bee"] > 0.0
    assert out["critter_dragonfly"] == 0.0
    task["critter_visitor"] = "none"
    out = presence(task=task)
    assert out["critter_bee"] == 0.0 and out["critter_dragonfly"] == 0.0


def test_the_hand_is_read_from_the_body_not_from_the_caregiver_block():
    # the control entity: it has to stay at its value with the adult gone and
    # every world entity switched off
    out = presence(task=dict(hand_view_deg=0.0),
                   caregiver=adult(0.0, False, object_present=False,
                                   mobile_present=False))
    assert out["own_hand_seen"] == 1.0
    assert out["caregiver"] == 0.0 and out["mobile"] == 0.0


# ----------------------------------------------------- missing and broken keys

def test_empty_dictionaries_read_all_zero_and_raise_nothing():
    assert entity_presence({}, {}) == [0.0] * len(ENTITY_NAMES)


def test_none_in_place_of_a_block_reads_all_zero():
    assert entity_presence(None, None) == [0.0] * len(ENTITY_NAMES)


@pytest.mark.parametrize("junk", [float("nan"), float("inf"), float("-inf"),
                                  "yes", None, [1.0], {}])
def test_a_value_that_is_not_a_finite_number_reads_zero(junk):
    out = presence(task=dict(squeaker_view_deg=junk, hand_view_deg=junk,
                             critter_visitor="bee", bee_deg=junk),
                   caregiver=dict(adult_visibility=junk, object_style=junk,
                                  object_present=junk, mobile_present=junk))
    assert out["squeaker"] == 0.0
    assert out["own_hand_seen"] == 0.0
    assert out["critter_bee"] == 0.0
    assert out["caregiver"] == 0.0
    assert out["mobile"] == 0.0


def test_one_key_present_leaves_the_rest_at_zero():
    out = presence(task=dict(hand_view_deg=0.0))
    assert out["own_hand_seen"] == 1.0
    assert sum(v for n, v in out.items() if n != "own_hand_seen") == 0.0


def test_every_value_stays_inside_zero_to_one_however_wild_the_input():
    out = presence(task=dict(squeaker_view_deg=-400.0, hand_view_deg=-0.0,
                             critter_visitor="dragonfly", dragonfly_deg=-12.0),
                   caregiver=adult(9.9, True, object_style=0, object_present=True,
                                   object_visibility=4.0, mobile_visibility=-3.0))
    for name, value in out.items():
        assert 0.0 <= value <= 1.0, name


def test_the_same_dictionaries_give_the_same_numbers_twice():
    task = dict(hand_view_deg=7.5, squeaker_view_deg=20.0)
    caregiver = adult(0.3, True, object_style=2, object_present=True)
    assert entity_presence(task, caregiver) == entity_presence(task, caregiver)


# ----------------------------------------------------------------- batch form

def eight_bodies():
    tasks, caregivers = [], []
    for b in range(8):
        tasks.append(dict(hand_view_deg=3.0 * b,
                          squeaker_view_deg=90.0 - 10.0 * b,
                          critter_visitor=("dragonfly", "bee", "none")[b % 3],
                          dragonfly_deg=4.0 * b, bee_deg=2.0 * b))
        caregivers.append(adult(visibility=b / 7.0, present=b > 0,
                                object_style=b % 3, object_present=b % 2 == 0,
                                mobile_present=b < 5))
    return tasks, caregivers


def test_batch_shape_is_bodies_by_names():
    tasks, caregivers = eight_bodies()
    out = entity_presence_batch(tasks, caregivers)
    assert tuple(out.shape) == (8, len(ENTITY_NAMES))
    assert str(out.dtype) == "torch.float32"


def test_batch_rows_match_the_single_body_form():
    tasks, caregivers = eight_bodies()
    out = entity_presence_batch(tasks, caregivers)
    for b, (task, caregiver) in enumerate(zip(tasks, caregivers)):
        assert out[b].tolist() == pytest.approx(entity_presence(task, caregiver))


def test_batch_of_no_bodies_is_an_empty_tensor_of_the_right_width():
    out = entity_presence_batch([], [])
    assert tuple(out.shape) == (0, len(ENTITY_NAMES))


def test_mismatched_batch_lengths_raise():
    with pytest.raises(ValueError):
        entity_presence_batch([{}, {}], [{}])


# ------------------------------------------------------------- real snapshots

def real_snapshots(limit=4):
    """Explainer snapshots written by a real run: the named H84 segment first,
    then any other experiment's explain folder."""
    folders = [NAMED_RUN] + sorted(
        (REPO / "Saved" / "mimo_bridge" / "experiments").glob("*/explain"),
        reverse=True)
    found, seen = [], set()
    for folder in folders:
        if not folder.is_dir() or folder in seen:
            continue
        seen.add(folder)
        for path in sorted(folder.glob("step_*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    snapshot = json.load(handle)
            except (OSError, ValueError):
                continue        # a snapshot still being written is skipped
            task = (snapshot.get("senses") or {}).get("task")
            caregiver = snapshot.get("caregiver")
            if isinstance(task, dict) and isinstance(caregiver, dict):
                found.append((path, task, caregiver))
            if len(found) >= limit:
                return found
    return found


def test_real_snapshots_give_finite_values_inside_zero_to_one():
    snapshots = real_snapshots()
    if len(snapshots) < 3:
        pytest.skip("fewer than 3 readable explainer snapshots under Saved/")
    for path, task, caregiver in snapshots:
        values = entity_presence(task, caregiver)
        assert len(values) == len(ENTITY_NAMES), path.name
        for name, value in zip(ENTITY_NAMES, values):
            assert isinstance(value, float), (path.name, name)
            assert math.isfinite(value), (path.name, name)
            assert 0.0 <= value <= 1.0, (path.name, name, value)


def test_real_snapshots_agree_between_the_single_and_batch_forms():
    snapshots = real_snapshots()
    if len(snapshots) < 3:
        pytest.skip("fewer than 3 readable explainer snapshots under Saved/")
    tasks = [task for _, task, _ in snapshots]
    caregivers = [caregiver for _, _, caregiver in snapshots]
    out = entity_presence_batch(tasks, caregivers)
    assert tuple(out.shape) == (len(snapshots), len(ENTITY_NAMES))
    for row, task, caregiver in zip(out, tasks, caregivers):
        assert row.tolist() == pytest.approx(entity_presence(task, caregiver))


def test_real_snapshots_keep_the_one_toy_at_a_time_rule():
    snapshots = real_snapshots()
    if len(snapshots) < 3:
        pytest.skip("fewer than 3 readable explainer snapshots under Saved/")
    for path, task, caregiver in snapshots:
        out = dict(zip(ENTITY_NAMES, entity_presence(task, caregiver)))
        above_zero = [n for n in TOY_STYLE_TO_NAME.values() if out[n] > 0.0]
        assert len(above_zero) <= 1, (path.name, above_zero)
