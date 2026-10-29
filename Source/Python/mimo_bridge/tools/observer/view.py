# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The dashboard's Development view (addendum v4 section 32): every sentence built here.

Reads the lineage report the digester writes (Saved/mimo_bridge/observer/<lineage>/development.json)
and its service status; the page (development_dashboard.js) only lays the sentences out. Wording
follows the dashboard's rules: each number says what it counts and against what; a stage without
its control reads "unmeasured", never passed or failed; scripted, probe and caregiver-caused acts
are named and never counted.

Revised after the independent review of 2026-09-24 (OBSERVER_DESIGN.md entries 39-40): stage
definitions state what tracker.py computes; "detected" (an act the detector found) is kept apart
from "beyond babble" (the Seen stage); babble is defined where the page first uses it; a number
without its reference point is not shown: the learner's wellness change over its acts is shown since
2026-09-25 against the same change over stretches of the act's length just before it, whatever happened in
them and without the act, as one plain line with its numbers in the line's tooltip (wellness_view).
Chances are pictured counts ('29 in 10,000'), never decimals.

Revised 2026-09-30 (operator approved): the comparison the code calls babble is named "random flailing" on
the page, defined where first used; each stage carries a plain question (STAGE_QUESTIONS); every cell
has one of four markers (passed, not yet, not judged yet, not measurable: MARKERS) and says what passing
takes and how far away it is, from the stage's own test inverted at its pass line (tracker: needed_*);
a stage that waits for an earlier one still shows its number and whether it would pass today; each row
opens with one sentence on how close it is to a full behaviour (lead_sentence); and dated copies of the
report (history.py) can be shown and compared (build(at=..., compare=...), history_view).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from . import history as HI
from . import tracker as TR

STEP_S = 0.05
RET_BLOCK_STEPS = 3000
STAGE_TITLES = [
    ("observed_once", "Seen",
     "Beyond random flailing (the same bodies in this world moved by random orders, with no learner; the code "
     "calls it babble). Either it comes more often than flailing in the same amount of simulated life, counted "
     "both as single acts and as occasions (an occasion is an act that starts at least 1 second after the "
     "previous act on the same body and part, such as one hand or one finger), or more of its occasions than "
     "chance allows start with an act of a size only 5 in 100 flailing occasions start with (for distances and "
     "delays: under the value only 5 in 100 flailing occasions start under). Each row gets both chances, so its "
     "pass line is twice as strict. All rows are judged together, and the pass line keeps chance passes to at "
     "most 5 in 100 of the rows that pass."),
    ("repeated", "Repeats",
     "At least 3 occasions, and more occasions and more single acts than random flailing in the same amount of "
     "simulated life (occasions alone would reward pauses: flailing's movement runs on without them; single acts "
     "are compared allowing for how they bunch into bursts). An occasion is an act that starts at least 1 second "
     "after the previous act on the same body and part. All rows are judged together, as for Seen."),
    ("predicted_before_action", "Predicted",
     "Before the act, the learner's own prediction beat its comparison on more occasions than chance allows; "
     "each occasion counts once, judged by its first act that could be checked. Movements: the prediction "
     "made with the body's own command must beat the same prediction made with another body's command and beat "
     "repeating the last movement (chance: 50 in 100). For an act of the hands alone each command in force during "
     "it is checked over its whole hold and counts once, and an act's commands are counted as one bunch; acts of "
     "the hands alone recorded before Oct 8, 2026, whose records name one command, are not judged. "
     "Touches: the predicted chance of a touch starting must "
     "be at least twice the skin's usual rate, more often than the same test passes at moments with no touch "
     "starting (that rate is measured for each row). Effects on objects: the act must be no bigger a surprise "
     "than the 2 seconds before it."),
    ("selected_intentionally", "Chosen",
     "For at least 3 occasions, the planner's prediction ranked the command best by more than the prediction's "
     "own error margin and the command reached the body; and the act comes more often while the planner writes "
     "the commands than while the motor controller with its exploration noise writes them. An act of "
     "other parts is credited to the command in force when it moved most. An act of the hands alone is credited to "
     "no single command: every command in force during it must pass, none may have asked the joints to move the "
     "other way, and an act whose commands came from more than one writer counts for neither side. This stage reads "
     "the wellness evaluator's tie band; a 'not yet' here does not show that the model learned no intentional "
     "behavior."),
    ("generalized", "Varied starts",
     "Repeats, and its acts appear on at least 3 of the 8 bodies and from at least 2 starting situations (the "
     "posture at the start plus the objects involved). It does not test that the act repeats in each situation."),
    ("retained", "Kept",
     "After it first repeated, two later blocks of 3,000 steps (2.5 minutes of each body's life each) each keep "
     "at least half the rate of occasions of the block where it first repeated (where its Repeats test first "
     "passed; the Repeats cell is dated where that test has passed at every occasion since, which can be later). "
     "A block fails only when a statistical test puts it below half, counting the first-repeat block's own "
     "occasions as a sample too; each block is judged together with every other row's test of that block, and "
     "the pass line keeps chance failures to at most 5 in 100 of the blocks it fails."),
]
# one plain question per stage (operator, 2026-09-30)
STAGE_QUESTIONS = {
    "observed_once": "Does it happen more than random flailing?",
    "repeated": "Does it keep happening, more than flailing?",
    "predicted_before_action": "Did the model see it coming?",
    "selected_intentionally": "Did the planner pick it on purpose?",
    "generalized": "Does it happen on different bodies and from different positions?",
    "retained": "Is it still there later?",
}
# the one set of markers every cell uses: its state -> the word the page shows. "Not judged" (tracker rules v6): the
# stage's evidence exists, but the counting rules in force do not judge it (acts of the hands alone recorded before
# Oct 8, 2026, whose records name one command); the cell says why and how many
MARKERS = {"reached": "Passed", "not": "Not yet", "waits": "Not judged yet", "unmeasured": "Not measurable",
           "notjudged": "Not judged"}
# the stage each stage waits for (tracker: gated)
EARLIER = {"repeated": "observed_once", "predicted_before_action": "repeated", "selected_intentionally": "repeated",
           "generalized": "repeated", "retained": "repeated"}
FULL_BEHAVIOUR = ("A full behavior passes the first four stages: Seen, Repeats, Predicted and Chosen.")
GROUP_ORDER = ["Hand & finger", "Self & body", "Gaze", "Reach & grasp", "Posture & locomotion", "Object play"]
DIGIT_WORDS = {"index": "index finger", "middle": "middle finger", "ring": "ring finger", "little": "little finger",
               "thumb": "thumb"}
SIDE_WORDS = {"L": "left", "R": "right"}
VALIDATION = os.path.join("docs", "gain_of_function_v4", "observer_validation", "validation_report.json")


def _read(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _num(n) -> str:
    return "{:,}".format(int(round(n))) if isinstance(n, (int, float)) else "an unrecorded number of"


def _plural(n, one: str, many: Optional[str] = None) -> str:
    return "%s %s" % (_num(n), one if isinstance(n, (int, float)) and int(round(n)) == 1 else (many or one + "s"))


def _life(steps: float) -> str:
    """Body-steps as simulated life: 50 ms each (summed over bodies where the caller says so)."""
    s = float(steps) * STEP_S
    if s < 120:
        return _plural(round(s), "second")
    if s < 7200:
        return _plural(round(s / 60), "minute")
    return "%.1f hours" % (s / 3600)


def _title(key: str) -> str:
    return next((t for k, t, _ in STAGE_TITLES if k == key), key)


def _dec(x: float) -> str:
    """A count expected by chance, to one decimal."""
    return ("%.1f" % x).rstrip("0").rstrip(".")


def _amount(x) -> str:
    """An expected count: one decimal below 100, whole numbers with separators above."""
    if x is None:
        return "not measured"
    return _num(x) if x >= 100 else _dec(x)


def _fmt(v) -> str:
    if v is None:
        return "?"
    a = abs(float(v))
    if a >= 100:
        return _num(v)
    if a >= 10:
        return "%.1f" % v
    return ("%.2f" % v).rstrip("0").rstrip(".")


def _chance(st: dict) -> str:
    exp = st.get("expected_by_chance")
    return "" if exp is None else "; flailing: %s in the same time" % _amount(exp)


UNITS = [("_radps", " radians a second"), ("_deg", "°"), ("_ms", " ms"), ("_cm", " cm"), ("_s", " s"), ("_n", " N")]


def measure_words(key: str):
    """A detector's main measure (capability_catalog.json 'quality') as words and a unit."""
    for suffix, unit in UNITS:
        if key.endswith(suffix):
            words = key[:-len(suffix)]
            break
    else:
        words, unit = key, ""
    words = words.replace("_", " ")
    if words.startswith("min "):
        words = "minimum " + words[4:]
    return words, unit


PART_WORDS = {"arm_lower": "forearm", "arm_upper": "upper arm", "leg_lower": "lower leg", "leg_upper": "thigh",
              "hand_palm": "palm", "root": "hips"}
OBJECT_WORDS = {"target": "ball", "box_light": "light box", "box_medium": "medium box", "box_heavy": "heavy box",
                "art": "painted wall panels"}
POSTURE_WORDS = {"back": "its back", "side": "its side", "belly": "its belly", "between": "an in-between position",
                 "upright": "upright"}
TRANSIENT_WORDS = {"head_jolt": "a sudden jolt of the head", "head_spin": "a sudden turn of the head",
                   "impact": "an impact", "support_loss": "a loss of support"}


def part_words(name: str) -> str:
    """A body part or object name from the recorder's header, in words: 'arm_L_lower' is 'the left
    forearm', 'finger_R_thumb' 'the right thumb', 'target' 'the ball'."""
    bits = name.split("_")
    side = next((SIDE_WORDS[b] for b in bits if b in SIDE_WORDS), None)
    rest = [b for b in bits if b not in SIDE_WORDS]
    if len(rest) == 2 and rest[0] == "finger":
        noun = DIGIT_WORDS.get(rest[1], rest[1] + " finger")
    else:
        key = "_".join(rest)
        noun = PART_WORDS.get(key) or OBJECT_WORDS.get(key) or OBJECT_WORDS.get(name) or key.replace("_", " ")
    return "the %s %s" % (side, noun) if side else "the " + noun


def token_words(tok: str) -> str:
    """One episode token (segmenter.token) as a plain clause."""
    kind, _, rest = tok.partition(":")
    a = rest.split(":") if rest else []
    side = SIDE_WORDS.get(a[0], a[0]) if a else ""
    if kind == "touch":
        x, _, y = rest.partition("~")
        return "%s touches %s" % (part_words(x), part_words(y))
    if kind == "curl" and len(a) == 2:
        return "the %s hand %s" % (side, "opens" if a[1] == "open" else "closes")
    if kind == "digit" and len(a) == 3:
        return "the %s %s %s" % (side, DIGIT_WORDS.get(a[1], a[1]), "bends" if a[2] == "bend" else "straightens")
    if kind == "wrist" and len(a) == 3:
        verb = {"flexion": "bends toward the palm" if a[2] == "+" else "bends back",
                "rotation": "turns" + (" one way" if a[2] == "+" else " the other way"),
                "deviation": "tilts sideways"}.get(a[1], a[1])
        return "the %s wrist %s" % (side, verb)
    if kind == "hand_view" and a:
        return "the eyes find the %s hand%s" % (side, " in the centre of view" if len(a) > 1 else "")
    if kind == "look" and a:
        return "the eyes rest on " + part_words(a[0])
    if kind == "eye_move":
        return "the eyes jump"
    if kind == "head_turn":
        return "the head turns"
    if kind == "object_moves" and a:
        return part_words(a[0]) + " moves"
    if kind in ("lift", "drop") and a:
        return part_words(a[0]) + (" is lifted" if kind == "lift" else " is dropped")
    if kind == "support_shift":
        return "the weight shifts"
    if kind == "posture" and a:
        frm, _, to = a[0].partition(">")
        return "the body moves from %s to %s" % (POSTURE_WORDS.get(frm, frm), POSTURE_WORDS.get(to, to))
    if kind == "transient" and a:
        return TRANSIENT_WORDS.get(a[0], a[0].replace("_", " "))
    if kind == "approach" and len(a) == 2:
        return "the %s hand moves toward %s" % (side, part_words(a[1]))
    if kind == "caregiver" and a:
        return {"touch": "the caregiver touches the body", "arrive": "the caregiver arrives",
                "leave": "the caregiver leaves"}.get(a[0], "the caregiver " + a[0])
    return tok.replace("_", " ").replace(":", " ")


def _size_detail(st: dict) -> str:
    """A size result with its quantity and line: '233 of 1,183 occasions started with an amount over
    34.7°, which only 5 in 100 flailing occasions do'."""
    line = st.get("beyond_line")
    if line is None or not st.get("measure"):
        return "%s of %s occasions started bigger than 95 in 100 flailing occasions do" % (
            _num(st.get("beyond")), _num(st.get("measured")))
    words, unit = measure_words(st["measure"])
    lower = st.get("direction") == "lower"
    article = "an" if words[:1] in "aeiou" else "a"
    return "%s of %s occasions started with %s %s %s %s%s, which only 5 in 100 flailing occasions do" % (
        _num(st.get("beyond")), _num(st.get("measured")), article, words, "under" if lower else "over",
        _fmt(line), unit)


def _per(d: int) -> str:
    return "a million" if d == 10 ** 6 else _num(d)


def _count_in(p: float, d: int, digits: int = 0) -> str:
    x = p * d
    return ("%.*f" % (digits, x)).rstrip("0").rstrip(".") if digits else _num(x)


def _denominator(p: float) -> int:
    """The smallest of 100, 1,000, ... a million over which p is a count of at least 5."""
    d = 100
    while d < 10 ** 6 and p * d < 5:
        d *= 10
    return d


def _chance_words(p) -> str:
    """A chance as a pictured count (the pictured-counts rule; independent review of 2026-09-25, finding
    4): '29 in 10,000', '5 in 100', '21 in a million', 'under 1 in a million'."""
    if p is None:
        return "not tested"
    p = float(p)
    if p < 5e-7:
        return "under 1 in a million"
    d = _denominator(p)
    return "%s in %s" % (_count_in(p, d), _per(d))


def _chances(p: float, line: float) -> tuple:
    """A p-value and the line it is judged by as counts over one shared number where they are alike
    ('29 in 10,000' against '36 in 10,000'; a decimal is added only where the two would round to the
    same count), each over its own where the p-value is far smaller ('21 in a million' against '5 in
    100')."""
    p, line = float(p), float(line)
    if line <= 0:
        return _chance_words(p), "0 (no test of its family passed)"
    d = _denominator(line)
    if p * d < 1:
        return _chance_words(p), "%s in %s" % (_count_in(line, d), _per(d))
    digits = 0
    while digits < 3 and p != line and _count_in(p, d, digits) == _count_in(line, d, digits):
        digits += 1
    return "%s in %s" % (_count_in(p, d, digits), _per(d)), "%s in %s" % (_count_in(line, d, digits), _per(d))


# What each tested stage's p-value is the chance of (stage_tip)
P_MEANS = {
    "observed_once": "evidence this strong if the learner were random flailing",
    "repeated": "both counts this far above flailing's if the learner were random flailing (the larger of the two "
                "counts' chances)",
    "predicted_before_action": "this many occasions predicted (for acts of the hands alone: this many commands, an "
                               "act's commands counted as one bunch) if the prediction were no better than its control",
    "selected_intentionally": "this many more occasions while the planner writes the commands if its choice made no "
                              "difference",
}


def stage_tip(key: str, st: dict) -> Optional[str]:
    """The cell's tooltip (OBSERVER_DESIGN.md section 9, review finding 24: a pass at its line showed no
    margin): the stage's p-value against the line it was judged by, what the p-value is the chance of,
    and how many tests shared the line. None where no test was made (unmeasured, not tested, or a stage
    without one: Varied starts)."""
    if key == "retained":
        p, line = st.get("worst_window_p"), st.get("worst_window_alpha")
        if p is None or line is None:
            return None
        n = next((h.get("tests_together") for h in st.get("history") or []
                  if h.get("window") == st.get("worst_window")), None)
        pw, lw = _chances(p, line)
        return ("Its lowest later block of 3,000 steps: a chance of %s of a count this low or lower if the rate had "
                "held at half the first-repeat rate; that block's line %s%s. A block counts as fallen below half only "
                "at or under its line." % (pw, lw, " (%s tested it together)" % _plural(n, "row") if n else ""))
    p, line = st.get("p_value"), st.get("alpha")
    if p is None or line is None or key not in P_MEANS:
        return None
    n = st.get("tests_together")
    how = (" (the Benjamini-Hochberg line over %s: at most 5 in 100 of the passes it allows are expected to be "
           "chance)" % _plural(n, "test")) if n else " (this row judged alone: 5 in 100)"
    means = P_MEANS[key]
    if key == "observed_once" and st.get("share_p") is not None and st.get("rate_p") is not None:
        # doubled only where both of the row's chances were tested (the review found 73 of 156 tested
        # Seen cells with the frequency test alone, their p-value not doubled, and the tooltip saying it was)
        means += ", doubled because the row had two chances (size and frequency)"
    pw, lw = _chances(p, line)
    return "A chance of %s: %s. Pass line%s: %s. A stage passes at or under its line." % (means, pw, how, lw)


def stage_text(key: str, st: dict, row: dict, ctx: Optional[dict] = None) -> dict:
    """One stage cell, with its tooltip (stage_tip) where a test was made; a stage that waits for an
    earlier one keeps its own test's numbers, says whether it would pass today, and is not reached."""
    out = _stage_cell(key, st, row, ctx)
    tip = stage_tip(key, st) if st.get("reached") is not None else None
    if key == "observed_once" and st.get("measured") and st.get("reached") is False:
        tip = "By size: %s.%s" % (_size_detail(dict(st, beyond=st.get("beyond") or 0)), (" " + tip) if tip else "")
    if tip and st.get("waits_for"):
        tip += " It is not reached until %s is." % _title(st["waits_for"])
    if tip:
        out["tip"] = tip
    return out


def _family_line(key: str, st: dict, ctx: dict) -> float:
    """The line this cell's p-value must reach to pass, every other test of its family held at its
    p-value (tracker.pass_line_for over the report's family: ctx 'families'); the line it was judged by
    when it already reaches it. Without the family (a cell built alone): one more discovery than the
    line it was judged by allows, q (k + 1) / m, or the plain 5 in 100."""
    p, alpha = st.get("p_value"), st.get("alpha")
    if p is not None and alpha is not None and p <= alpha:
        return float(alpha)
    fam = (ctx.get("families") or {}).get(key)
    if fam is not None:
        others = list(fam)
        if p is not None and p in others:
            others.remove(p)
        return TR.pass_line_for(others)
    m = st.get("tests_together")
    if alpha is not None and m:
        return min(TR.ALPHA, TR.ALPHA * (int(round(float(alpha) * m / TR.ALPHA)) + 1) / float(m))
    return TR.ALPHA


def _attribution_counts(st: dict) -> str:
    """'; 12 occasions whose acts ran under commands of more than one writer and 40 recorded before each command was
    listed count for neither side' (tracker rules v6), or '' when there are none."""
    mixed, old = int(st.get("mixed_writer_occasions") or 0), int(st.get("writer_not_measured_occasions") or 0)
    parts = []
    if mixed:
        parts.append("%s whose acts of the hands alone ran under commands of more than one writer" % _plural(mixed, "occasion"))
    if old:
        parts.append("%s recorded before each command of an act of the hands alone was listed" % _plural(old, "occasion"))
    return ("; %s count for neither side" % " and ".join(parts)) if parts else ""


def _about(n: Optional[int], have, unit: str = "") -> Optional[str]:
    """'about 905 occasions (51 more)': a needed count with its distance from the count now."""
    if n is None:
        return None
    gap = int(n) - int(have or 0)
    return "about %s%s%s" % (_num(n), unit, " (%s more)" % _num(gap) if gap > 0 else "")


def _freq_inputs(st: dict, row: dict, ctx: dict) -> Optional[tuple]:
    """The inputs of "more often than flailing on both counts" from a cell: (the learner's body-steps over
    flailing's, flailing's occasions, the learner's acts per occasion, flailing's single acts, both burst
    spreads); the row's spreads where an older report's cell lacks its own. None when they are missing."""
    occ, exp, b_occ = st.get("occasions"), st.get("expected_by_chance"), st.get("babble_occasions")
    rep = (ctx.get("stages") or {}).get("repeated") or {}
    Eb = st.get("babble_steps") or rep.get("babble_steps")
    if st.get("exposure_steps") and Eb:
        ratio = float(st["exposure_steps"]) / float(Eb)
    elif b_occ and exp is not None:
        ratio = float(exp) / float(b_occ)
    elif row.get("exposure_steps") and Eb:
        ratio = float(row["exposure_steps"]) / float(Eb)
    else:
        return None
    if b_occ is None and exp is not None:
        b_occ = float(exp) / ratio
    if b_occ is None:
        return None
    b_acts = st.get("babble_acts")
    if b_acts is None and st.get("expected_acts") is not None:
        b_acts = float(st["expected_acts"]) / ratio
    a = (float(st["acts"]) / occ) if occ and st.get("acts") else float(row.get("acts_per_occasion") or 1.0)
    phi = st.get("dispersion") or row.get("dispersion") or 1.0
    phi_b = st.get("babble_dispersion") or row.get("babble_dispersion") or 1.0
    return ratio, b_occ, a, b_acts or 0.0, phi, phi_b


def _acts_hold_back(st: dict, row: dict, ctx: dict) -> str:
    """'. The count holding it back is single acts: 23,708 against flailing's 38,656' when the single-act
    half of the test is the one further from passing (its p-value the larger), else ''."""
    inp = _freq_inputs(st, row, ctx)
    occ, acts, exp_acts = st.get("occasions"), st.get("acts"), st.get("expected_acts")
    if inp is None or not occ or acts is None or exp_acts is None:
        return ""
    ratio, b_occ, _, b_acts, phi, phi_b = inp
    p_occ = TR.two_sample_above(int(occ), ratio, int(round(b_occ)), 1.0)
    p_acts = TR.two_sample_above_dispersed(float(acts), ratio, phi, b_acts, 1.0, phi_b)
    if p_acts <= p_occ:
        return ""
    return ". The count holding it back is single acts: %s against flailing's %s" % (_num(acts), _amount(exp_acts))


def _frequency_needed(st: dict, row: dict, ctx: dict, line: float, chances: int, least: int) -> Optional[int]:
    """The occasions at which "more often than flailing on both counts" would pass (tracker.needed_occasions),
    single acts rising with the occasions at the cell's own acts per occasion."""
    inp = _freq_inputs(st, row, ctx)
    if inp is None:
        return None
    occ = st.get("occasions")
    n = TR.needed_occasions(line, *inp, chances=chances, least=least)
    # a cell that has not passed needs more than it has (an older report's fallbacks can land below)
    return None if n is None else max(int(n), int(occ or 0) + 1)


def would_pass(key: str, st: dict) -> bool:
    """Whether a stage's own test passes now, whatever the earlier stages say."""
    p, alpha = st.get("p_value"), st.get("alpha")
    tested = p is not None and alpha is not None and p <= alpha
    if key == "repeated":
        return tested and int(st.get("occasions") or 0) >= TR.MIN_COUNT
    if key == "predicted_before_action":
        return tested and int(st.get("predicted") or 0) >= TR.MIN_COUNT
    if key == "selected_intentionally":
        return tested and int(st.get("decisive") or 0) >= TR.MIN_COUNT
    if key == "generalized":
        return int(st.get("bodies") or 0) >= 3 and int(st.get("contexts") or 0) >= 2
    if key == "retained":
        return False
    return tested


def _measure(key: str, st: dict, row: dict, ctx: dict) -> tuple:
    """(what was measured, what passing takes and how far away it is) for a cell that has not passed;
    either may be None. Every count is the stage's own; 'passing needs' inverts its own test at its pass
    line with every other input held (tracker: needed_*)."""
    occ, exp = st.get("occasions"), st.get("expected_by_chance")
    if key == "observed_once":
        chances = 2 if (st.get("share_p") is not None and st.get("rate_p") is not None) else 1
        line = _family_line(key, st, ctx)
        parts, needs = [], []
        if st.get("measured"):
            k = TR.needed_beyond(line, int(st["measured"]), chances)
            have = int(st.get("beyond") or 0)
            # the measure and its line are in the cell's tooltip (stage_text): the cell keeps the counts
            parts.append("%s of %s occasions started bigger than 95 in 100 flailing occasions do" % (
                _num(have), _num(st["measured"])))
            needs.append("about %s%s" % (_num(k), " (%s more)" % _num(k - have) if k > have else "") if k else
                         "more than all %s" % _num(st["measured"]))
        if exp is not None and occ:
            n = _frequency_needed(st, row, ctx, line, chances, 1)
            how = "%s against flailing's %s in the same time" % (_plural(occ, "occasion"), _amount(exp))
            parts.append(how + _acts_hold_back(st, row, ctx))
            needs.append(_about(n, occ, " occasions") or "a count this page cannot compute")
        if not parts:
            return "No acts yet", None
        if len(parts) == 2:
            return ("Two ways to pass. By size: %s; passing needs %s. By how often: %s; passing needs %s" % (
                parts[0], needs[0], parts[1], needs[1])), None
        if st.get("measured"):
            return parts[0], "passing needs " + needs[0]
        return parts[0], "passing needs " + needs[0]
    if key == "repeated":
        occ = int(occ or 0)
        if exp is None:
            return _plural(occ, "occasion"), "passing needs at least 3"
        n = _frequency_needed(st, row, ctx, _family_line(key, st, ctx), 1, TR.MIN_COUNT)
        what = "%s against flailing's %s in the same time" % (_plural(occ, "occasion"), _amount(exp))
        what += _acts_hold_back(st, row, ctx)
        return what, ("passing needs %s" % _about(n, occ, " occasions") if n else
                      "passing needs at least 3" if occ < TR.MIN_COUNT else "not past its pass line yet")
    if key == "predicted_before_action":
        n, k = int(st.get("judged") or 0), int(st.get("predicted") or 0)
        # tracker rules v6: an act of the hands alone recorded before observer_attribution_v1 was judged on one command
        # only; it is not judged now, and the cell says how many
        old = int(st.get("acts_read_on_one_credited_order_not_judged") or 0)
        not_judged = ("; %s of the hands alone recorded before each command was listed %s not judged" % (
            _plural(old, "act"), "is" if old == 1 else "are")) if old else ""
        if not n:
            return "No acts to judge yet" + not_judged, None
        if n < TR.MIN_COUNT:
            return ("%s of %s occasions predicted%s%s" % (_num(k), _num(n), _touch_null(st), not_judged),
                    "passing needs at least 3 judged occasions")
        touch = bool(st.get("null_moments")) and st.get("null_passed") is not None
        if st.get("commands_judged") and not touch:
            # tracker rules v6: the units are commands, an act's commands counted as one bunch (cluster_binom_above)
            nc, kc = int(st["commands_judged"]), int(st.get("commands_predicted") or 0)
            design = float(st.get("design_effect") or 1.0)
            what = "%s of %s commands predicted, over %s%s" % (_num(kc), _num(nc), _plural(n, "occasion"), not_judged)
            need = TR.needed_predicted(_family_line(key, st, ctx), nc, design=design)
            return what, ("passing needs about %s of %s commands%s" % (
                _num(need), _num(nc), " (%s more)" % _num(need - kc) if need > kc else "") if need else
                "even all %s commands predicted would not pass at this number" % _num(nc))
        what = "%s of %s occasions predicted%s%s" % (_num(k), _num(n), _touch_null(st), not_judged)
        need = TR.needed_predicted(_family_line(key, st, ctx), n, st.get("null_passed") if touch else None,
                                   st.get("null_moments") if touch else None)
        return what, ("passing needs about %s of %s%s" % (_num(need), _num(n), " (%s more)" % _num(need - k)
                                                           if need > k else "") if need else
                      "even all %s predicted would not pass at this number of occasions" % _num(n))
    if key == "selected_intentionally":
        d = int(st.get("decisive") or 0)
        if ctx.get("planner_tied") and d < TR.MIN_COUNT:
            # WHAT A TIE IS (own-command analysis review, 2026-10-02): the band is the evaluator's own error on a
            # difference, wider than the candidates' predicted spread in every H84 window since Sep 23
            # (docs/gain_of_function_v4/agents/hold_loss_history_2026-10-02.json), so the count says nothing about
            # the forward model
            judged = ctx.get("choices_judged")
            return ("A choice counts as chosen only when the chosen command's predicted wellness beats every "
                    "alternative's by more than the tie band, the evaluator's own error on a difference between two "
                    "candidates%s. On H84 the band has been wider than the candidates' predicted spread in every report "
                    "window since Sep 23, so this count is no evidence about the forward model, and it does not show "
                    "that the model learned no intentional behavior: the planner selects by predicted goal distance, "
                    "recorded beside each act of the hands alone%s" % (
                        "; in the current run that happened in %s of %s choices" % (
                            _num(ctx.get("choices_best") or 0), _num(judged)) if judged else "",
                        _attribution_counts(st))), None
        if d < TR.MIN_COUNT:
            return ("%s of the 3 occasions the planner must pick decisively before this is tested (decisively: its "
                    "prediction ranked the command best by more than its own error margin)%s" % (
                        _num(d), _attribution_counts(st))), None
        cm, cs = int(st.get("planner_occasions") or 0), int(st.get("babble_occasions") or 0)
        em, es = st.get("planner_steps"), st.get("reflex_steps")
        if em is None and st.get("planner_rate_per_1000") and cm:
            em = 1000.0 * cm / st["planner_rate_per_1000"]
        if es is None and st.get("babble_rate_per_1000") and cs:
            es = 1000.0 * cs / st["babble_rate_per_1000"]
        if not em or not es:
            return "%s while the planner wrote the commands" % _plural(cm, "occasion"), None
        need = TR.needed_planner_occasions(_family_line(key, st, ctx), cs, em, es)
        return ("%s while the planner wrote the commands against %s expected at the motor controller's rate" % (
            _plural(cm, "occasion"), _amount(cs * em / es)), "passing needs %s" % _about(need, cm, " occasions"))
    if key == "generalized":
        return ("On %s, from %s" % (_plural(st.get("bodies") or 0, "body", "bodies"),
                                    _plural(st.get("contexts") or 0, "starting situation")),
                "passing needs 3 bodies and 2 starting situations")
    if key == "retained":
        if st.get("windows_after") is None:
            return None, None
        hist = st.get("history") or []
        if hist and hist[-1].get("below"):
            return "Regressed: the newest block of 3,000 steps fell below half the rate at which it first repeated", None
        if any(h.get("below") for h in hist):
            return "Not kept: a later block of 3,000 steps fell below half the first-repeat rate", None
        ref = st.get("reference_count")
        return ("%s of the 2 later blocks of 3,000 steps (2.5 minutes of each body's life) recorded so far" %
                _num(st.get("windows_after") or 0),
                "passing needs 2, each keeping at least half the rate of the block where it first repeated%s" % (
                    " (%s in that block)" % _plural(ref, "occasion") if ref is not None else ""))
    return None, None


def _cell(state: str, detail: str, **kw) -> dict:
    """A cell in the one set of markers: the marker the page shows, the detail beside it, and the whole
    sentence ('Not yet: ...', 'Passed at step ...', 'Not judged until ...', 'Not measurable: ...')."""
    marker = MARKERS[state]
    if state == "waits":
        text = detail
    elif state == "reached":
        text = "Passed " + detail[:1].lower() + detail[1:]
    elif detail.startswith(("Regressed", "Not kept")):
        text = detail
    else:
        text = "%s: %s" % (marker, detail if detail[:1].isdigit() else detail[:1].lower() + detail[1:])
    return dict(state=state, marker=marker, detail=detail, text=text, **kw)


def not_judged_detail(st: dict) -> str:
    """The cell of a stage the counting rules in force do not judge (tracker rules v6; independent review of 4342f9194,
    blocker 1): how many acts, why, and the withdrawn earlier reading beside them."""
    n = int(st.get("acts_read_on_one_credited_order_not_judged") or 0)
    text = ("%s of the hands alone %s recorded before Oct 8, 2026, when an act's movement verdict was read on the one "
            "command it was credited to; the current rule judges every command in force during an act, and those "
            "records do not list them" % (_plural(n, "act"), "was" if n == 1 else "were"))
    er = st.get("earlier_rule") or {}
    if er.get("judged"):
        text += "; under that withdrawn rule %s of %s occasions were predicted, %s" % (
            _num(er.get("predicted") or 0), _num(er["judged"]),
            ("enough to pass Predicted" if er.get("stage_reached") else
             "enough to pass its test, while the stage waits for Repeats" if er.get("passes_at_line") else
             "not enough to pass"))
    return text


def withdrawn_rows(caps: Dict[str, dict]) -> List[str]:
    """The rows and parts whose Predicted stage is not judged under tracker rules v6 while, under the withdrawn earlier
    rule, it would be reached (its test at that rule's line and the ladder: Repeats reached) (blocker 1: a headline
    never changes only because evidence stopped being judged)."""
    out = []
    for cid, row in caps.items():
        cands = [(row.get("name") or cid, row.get("stages") or {})]
        for sk, sr in (row.get("subs") or {}).items():
            cands.append(("%s (%s)" % (row.get("name") or cid, sub_label(cid, sk)), sr.get("stages") or {}))
        for name, stages in cands:
            st = stages.get("predicted_before_action") or {}
            if st.get("not_judged") and (st.get("earlier_rule") or {}).get("stage_reached"):
                out.append(name)
    return out


def _stage_cell(key: str, st: dict, row: dict, ctx: Optional[dict] = None) -> dict:
    """One stage cell: its state (MARKERS), the detail the page shows beside the marker, and the whole
    sentence. ctx: 'stages' (the same row's other stages), 'planner_tied' (every planner choice of the
    current run tied, so Chosen cannot be reached), 'choices_judged' and 'choices_best' (those choices),
    'families' (each tested stage's p-values across the report, for the pass line a cell must reach)."""
    ctx = ctx or {}
    reached = st.get("reached")
    waits = st.get("waits_for")
    earlier = EARLIER.get(key)
    if not waits and reached is False and earlier and ((ctx.get("stages") or {}).get(earlier) or {}).get("reached") is False:
        # the tracker marks a stage as waiting only when its own test passed; one that fails its own test
        # while the earlier stage has not passed is not judged either, and passing it alone would not do
        waits = earlier
    if waits:
        first = "Not judged until %s passes." % _title(waits)
        what, need = _measure(key, dict(st, waits_for=None), row, ctx)
        if key == "retained" or what is None:
            return _cell("waits", first + " Nothing to measure until then.", would_pass=False)
        if would_pass(key, st):
            return _cell("waits", "%s %s; it would pass today." % (first, what), would_pass=True)
        if need:
            return _cell("waits", "%s %s; it would not pass today; %s." % (first, what, need), would_pass=False)
        return _cell("waits", "%s %s. It would not pass today." % (first, what), would_pass=False)
    if reached is True:
        step = st.get("first_step")
        if key == "retained":
            return _cell("reached", "In all %s later blocks of 3,000 steps so far, it kept at least half the rate of "
                                    "the block where it first repeated" % _num(st.get("windows_after") or 0))
        text = "At step " + _num(step) if step is not None else "At an unrecorded step"
        if key == "observed_once" and st.get("by") == "size":
            text += ": bigger than random flailing (%s)" % _size_detail(st)
        elif key == "observed_once" and st.get("by") == "frequency":
            text += ": more often than random flailing on both counts (%s%s)" % (
                _plural(st.get("occasions"), "occasion"), _chance(st))
        if key == "repeated":
            seen = (ctx.get("stages") or {}).get("observed_once") or {}
            if seen.get("by") == "frequency":
                text += " (the same counts as Seen, with at least 3 occasions)"
            elif st.get("expected_by_chance") is not None:
                text += " (on both counts: %s%s)" % (_plural(st.get("occasions"), "occasion"), _chance(st))
        if key == "predicted_before_action" and st.get("commands_judged") and not st.get("null_moments"):
            text += " (%s of %s commands predicted, over %s)" % (
                _num(st.get("commands_predicted") or 0), _num(st.get("commands_judged")),
                _plural(st.get("judged"), "occasion"))
        elif key == "predicted_before_action":
            text += " (%s of %s occasions predicted%s)" % (_num(st.get("predicted")), _num(st.get("judged")),
                                                           _touch_null(st))
        if key == "selected_intentionally":
            text += " (%s)" % _plural(st.get("decisive"), "decisive occasion")
        if key == "generalized" and st.get("bodies") is not None:
            text += " (%s, %s)" % (_plural(st.get("bodies"), "body", "bodies"),
                                   _plural(st.get("contexts"), "starting situation"))
        return _cell("reached", text)
    if reached is None and st.get("not_judged"):
        # tracker rules v6 (independent review of 4342f9194, blocker 1): the true reason and the count, and the
        # withdrawn earlier reading beside them; never "not measurable" with another cause
        return _cell("notjudged", not_judged_detail(st))
    if reached is None:
        if not row.get("observable", True):
            return _cell("unmeasured", "Not observable in this world")
        note = st.get("note") or ""
        if "not rebuilt" in note:
            short = "this dated copy was rebuilt without it"
        elif "babble" in note or key in ("observed_once", "repeated"):
            short = "random flailing is not recorded for this world yet"
        elif "prediction record" in note:
            short = "no learner prediction record for these acts"
        elif "no order" in note or "no command" in note:
            short = "the planner wrote no command"
        elif "babble-authored" in note:
            short = "no steps written by the motor controller to compare"
        elif key in ("generalized", "retained"):
            short = "waits for Repeats to be measured"
        else:
            short = note or "its control is not measured yet"
        return _cell("unmeasured", short[:1].upper() + short[1:])
    what, need = _measure(key, st, row, ctx)
    if not what and not need:
        return _cell("not", "No acts yet.")
    if what and need:
        return _cell("not", "%s; %s." % (what, need))
    return _cell("not", (what or need) + ".")


WELLNESS_MIN = 5          # occasions paired with each reference before its comparison is shown


def _signed(x: float) -> str:
    """A small signed change to two significant digits, never in exponent form: '+0.0042', '-0.012'."""
    import numpy as np
    return ("+" if x >= 0 else "-") + np.format_float_positional(abs(float(x)), precision=2, unique=False,
                                                                 fractional=False, trim="-")


def _interval(d: float, ci: float) -> str:
    return "95%% interval %s to %s" % (_signed(d - ci), _signed(d + ci))


def _compared(d: Optional[float], ci: Optional[float]) -> Optional[str]:
    """'more', 'less' or 'same' (0 inside the 95% interval), None without a difference."""
    if d is None or ci is None:
        return None
    return "more" if d - ci > 0 else "less" if d + ci < 0 else "same"


WELLNESS_VERDICT = {
    "same": "changed by the same amount during this act as in the same length of time just before it, as far as "
            "the measurement can tell",
    "more": "went up more (or down less) during this act than in the same length of time just before it",
    "less": "went up less (or down more) during this act than in the same length of time just before it",
}
WITHOUT_VERDICT = {
    "same": "changed by the same amount during the act as in them, as far as the measurement can tell",
    "more": "went up more (or down less) during the act than in them",
    "less": "went up less (or down more) during the act than in them",
}


def wellness_view(w: dict) -> Optional[dict]:
    """The learner's wellness change under a behaviour against its two references (tracker.wellness_summary),
    as one plain line with every number in its tooltip (independent review of 2026-09-25, finding 4: the
    sentence it replaces carried 12 numbers in the row). The references: stretches of the act's length
    ending 1 to 3 s before it starts, whatever happened in them, and those without the act. Both, never
    the second alone (the coordinator's decision of 2026-09-25). None until WELLNESS_MIN occasions are
    paired with the first reference."""
    if not w or (w.get("acts_n") or 0) < WELLNESS_MIN or (w.get("all_n") or 0) < WELLNESS_MIN:
        return None
    v_all = _compared(w.get("difference_all"), w.get("difference_all_ci95"))
    if v_all is None:
        return None
    v_without = (_compared(w.get("difference"), w.get("difference_ci95"))
                 if (w.get("reference_n") or 0) >= WELLNESS_MIN else None)
    share = w.get("in_act_share")
    text = ("Wellness (the sum of the learner's grounded wellness channels, each 0 to 1) " + WELLNESS_VERDICT[v_all])
    if share is not None and share >= 0.5:
        # a row that runs almost without pause: the time just before one act is mostly another act of it
        # (the independent review of 2026-09-25, finding 4), so the comparison says little on its own
        text += " (the same behaviour was already going on in %d of every 100 of those stretches)" % int(round(100 * share))
    if v_without == v_all:
        text += ", with or without the act in that time."
    elif v_without is not None:
        text += ". Against only the stretches in which the act did not happen, wellness %s." % WITHOUT_VERDICT[v_without]
    else:
        text += "."
    tip = ("The first act of each of %s changed wellness by %s on average; from one occasion to the next that change "
           "spreads by %s (one standard deviation). The references are stretches of the act's length ending 1 to 3 s "
           "before it starts, one a second, at the act's own place in its order. %s had such stretches (%s%s): "
           "over them wellness changed by %s on average, and the act minus its own stretches is %s (%s; each "
           "occasion one sample; 0 = the same change)." % (
               _plural(w["acts_n"], "occasion"), _signed(w["acts_mean"]), _signed(w["acts_sd"] or 0.0).lstrip("+"),
               _plural(w["all_n"], "occasion"), _plural(w.get("all_stretches"), "stretch", "stretches"),
               "" if share is None else "; the same behaviour went on in %d in 100 of them" % int(round(100 * share)),
               _signed(w["all_mean"]), _signed(w["difference_all"]),
               _interval(w["difference_all"], w["difference_all_ci95"])))
    if v_without is not None:
        tip += (" %s had stretches without the act (%s): over them wellness changed by %s, and the act minus "
                "those is %s (%s)." % (
                    _plural(w["reference_n"], "occasion"), _plural(w.get("reference_stretches"), "stretch", "stretches"),
                    _signed(w["reference_mean"]), _signed(w["difference"]),
                    _interval(w["difference"], w["difference_ci95"])))
        if v_without == "less" and v_all != "less":
            # below the stretches without the act, not below the stretches in general: those stretches are the
            # quieter moments between acts
            tip += (" Wellness rose more (or fell less) in the stretches without the act than in the stretches in "
                    "general, so a change below theirs is not by itself a cost of the act.")
    return dict(text=text, tip=tip)


def _touch_null(st: dict) -> str:
    """A touch row's measured chance: how often the same verdict passes with no touch starting."""
    rate, n = st.get("null_rate"), st.get("null_moments")
    if rate is None or not n:              # movements and object effects: chance is 50 in 100, in the key
        return ""
    return "; with no touch starting, the same test passes %d in 100" % int(round(100 * rate))


def sub_label(cap: str, sub: str) -> str:
    """A sub-row key in words: 'L' is the left hand (the left eye for gaze rows, the left arm for
    the arm-into-view row), 'L:thumb' the left thumb, 'toward_L' toward the left."""
    parts = sub.split(":")
    if cap in ("H05", "HX3") and len(parts) == 2:
        return "%s %s" % (SIDE_WORDS.get(parts[0], parts[0]), DIGIT_WORDS.get(parts[1], parts[1]))
    if cap == "H06" and len(parts) == 2:
        return "%s thumb to %s" % (SIDE_WORDS.get(parts[0], parts[0]), DIGIT_WORDS.get(parts[1], parts[1]))
    if len(parts) == 2 and parts[0] in SIDE_WORDS:
        return "%s %s" % (SIDE_WORDS[parts[0]], parts[1].replace("_", " "))
    if sub in SIDE_WORDS:
        noun = "eye" if cap.startswith("G") else "arm" if cap == "B06" else "hand"
        return "%s %s" % (SIDE_WORDS[sub], noun)
    if sub.startswith("toward_") and sub[7:] in SIDE_WORDS:
        return "toward the " + SIDE_WORDS[sub[7:]]
    return " and ".join(part_words(p)[4:] for p in sub.split("+"))


def _replay_key(st: dict):
    """The replay index key of a stage's example (daemon.replay_key): its episode id, or the act's
    own run, body, process and step when it began outside every episode."""
    if st.get("example"):
        return st["example"]
    a = st.get("act") or {}
    if a:
        return "%s:b%s:p%s:s%s" % (a.get("run"), a.get("body"), a.get("pid"), a.get("step0"))
    return None


def _were(n) -> str:
    return "was" if int(n) == 1 else "were"


def _and(words: List[str]) -> str:
    return words[0] if len(words) == 1 else "%s and %s" % (", ".join(words[:-1]), words[-1])


def lead_sentence(cells: List[dict], observable: bool = True) -> str:
    """The one sentence a row opens with: how close it is to a full behaviour (the first four stages,
    FULL_BEHAVIOUR), from its own cells."""
    if not observable:
        return "Not measurable in this world."
    state = {c["key"]: c.get("state") for c in cells}
    passed = [k for k in MILESTONE_STAGES if state.get(k) == "reached"]
    missing = [k for k in MILESTONE_STAGES if k not in passed]
    if not missing:
        return "A full behavior: it passed Seen, Repeats, Predicted and Chosen."
    if len(missing) == 1:
        return "Closest to a full behavior: only %s is missing." % _title(missing[0])
    if not passed:
        if state.get("observed_once") == "unmeasured":
            return "Not measurable yet: its comparison, random flailing, is not recorded for this world."
        return "No stage passed yet: it does not yet happen more than random flailing."
    return "Passed %s; a full behavior still needs %s." % (_and([_title(k) for k in passed]),
                                                          _and([_title(k) for k in missing]))


def alone_note(row: dict) -> Optional[str]:
    """H05's other digits INSIDE the act (detectors.H05 `alone_inside`; tracker `alone_inside`), as one sentence, or
    None when the row holds no act it was measured on."""
    a = row.get("alone_inside") or {}
    yes, no, old = int(a.get("yes") or 0), int(a.get("no") or 0), int(a.get("not_measured") or 0)
    if yes + no == 0:
        return None
    text = ("Detected acts need the other four digits to END within %s degrees of where they began on average; inside "
            "the act they also stayed within it in %s of %s acts measured" % (
                _num(a.get("line_deg") or 5), _num(yes), _num(yes + no)))
    if old:
        text += " (%s more %s recorded before this was measured)" % (_num(old), _were(old))
    return text + "."


def row_view(row: dict, replays: dict, ctx: Optional[dict] = None) -> dict:
    ctx = ctx or {}
    stages = row.get("stages") or {}
    counted = row.get("counted") or 0
    base = row.get("baseline_rate_per_1000_steps")
    exposure = row.get("exposure_steps") or 0
    if not row.get("observable", True):
        summary = "Not observable in this world: " + (row.get("why_not") or "no reason recorded") + \
                  " This row is not a finding of zero ability."
    elif counted == 0:
        summary = "Not detected in %s of simulated life summed over the bodies (counted acts only)." % _life(exposure)
    else:
        summary = "Detected %s in %s of simulated life summed over the bodies" % (_plural(counted, "time"), _life(exposure))
        bound = row.get("baseline_bound_per_1000_steps")
        if base is not None and exposure:
            summary += "; random flailing would produce about %s in the same time" % _amount(base * exposure / 1000.0)
        elif bound is not None and exposure:
            summary += ("; random flailing never produced it (at most %s in the same time, the 95%% upper bound of a "
                        "count of 0)"
                        % _amount(bound * exposure / 1000.0))
        summary += "."
    extra = []
    from .detectors import NEEDS_FIELDS
    if row.get("id") in NEEDS_FIELDS and row.get("observable", True):
        # no lineage surgery (the coordinator's decision of the evening of 2026-09-25): the exposure is one
        # count per body shared by every row, and it holds body-steps recorded before the recorder kept
        # contact normals, in which R04 and the H02 family could see nothing
        extra.append("Observable exposure begins at recorder v2 (observer_world_v2, which records the contact "
                     "normals this detector reads): the simulated life counted here also holds body-steps "
                     "recorded before it, in which this detector could see nothing.")
    if row.get("scripted"):
        n = row["scripted"]
        if row.get("id") == "G01":
            extra.append("%s more %s made by the world's own eye movers (the spontaneous saccade generator and the "
                         "eye reflex), not the learner, and %s not counted." % (_num(n), _were(n), "is" if n == 1 else "are"))
        else:
            extra.append("%s more %s carried by the toy's scripted hand-off and %s not counted." % (
                _num(n), _were(n), "is" if n == 1 else "are"))
    if row.get("caregiver_involved"):
        n = row["caregiver_involved"]
        extra.append("%s more happened while the caregiver touched the body and %s not counted." % (
            _num(n), "is" if n == 1 else "are"))
    if row.get("world_moved"):
        n = row["world_moved"]
        extra.append("%s more began while a toy or critter the world moves was moving into the act, and %s not counted." % (
            _num(n), "is" if n == 1 else "are"))
    if alone_note(row):
        extra.append(alone_note(row))
    validated = ctx.get("validated")
    if validated is not None and row.get("observable", True) and row.get("id") not in validated \
            and (stages.get("observed_once") or {}).get("reached"):
        extra.append("This detector has no scripted test case yet (validation covers %s of the catalog's detectors)."
                     % _num(len(validated)))
    regressed_text = ("REGRESSED: the newest block of 3,000 steps fell below half the rate at which it first repeated."
                      if row.get("regressed") else None)
    ex = _replay_key(stages.get("observed_once") or {})
    rp = replays.get(ex) if ex else None
    changes = ctx.get("changes") or {}
    cells = [dict(key=k, title=t, question=STAGE_QUESTIONS[k], **stage_text(k, stages.get(k) or {}, row,
                                                                          dict(ctx, stages=stages)))
             for k, t, _ in STAGE_TITLES]
    out = dict(id=row["id"], key=row["id"], name=row.get("name"), group=row.get("group"),
               lead=lead_sentence(cells, row.get("observable", True)), track=HI.row_numbers(stages),
               changes=changes.get(row["id"]), summary=summary, notes=extra,
               wellness=wellness_view(row.get("wellness") or {}) if counted and row.get("observable", True) else None,
               regressed_text=regressed_text, observable=row.get("observable", True), regressed=bool(row.get("regressed")),
               stages=cells,
               no_overseer=[dict(key=k, title=t, question=STAGE_QUESTIONS[k],
                                 **stage_text(k, (row.get("no_overseer") or {}).get(k) or {}, row,
                                              dict(ctx, stages=row.get("no_overseer") or {})))
                            for k, t, _ in STAGE_TITLES] if row.get("no_overseer") else None,
               replay=("/files/" + rp["gif_path"]) if rp and rp.get("gif_path") else None)
    subs = []
    for sk, sr in sorted((row.get("subs") or {}).items()):
        sst = sr.get("stages") or {}
        own = dict(row, **{k: v for k, v in sr.items() if k not in ("stages", "subs", "no_overseer")})
        key = "%s|%s" % (row["id"], sk)
        sub_cells = [dict(key=k, title=t, question=STAGE_QUESTIONS[k], **stage_text(k, sst.get(k) or {}, own,
                                                                                  dict(ctx, stages=sst)))
                     for k, t, _ in STAGE_TITLES]
        lead = lead_sentence(sub_cells, row.get("observable", True))
        note = alone_note(sr)
        subs.append(dict(sub=sk, key=key, label=sub_label(row["id"], sk), counted=sr.get("counted", 0),
                         lead=(lead + " " + note) if (note and lead) else (lead or note),
                         notes=[note] if note else [], track=HI.row_numbers(sst),
                         changes=changes.get(key), stages=sub_cells))
    out["subs"] = subs
    return out


def finger_rows(caps: Dict[str, dict], views: Dict[str, dict]) -> List[dict]:
    """The finger-movement rows (section 32): each digit's isolated movement per hand, the thumb
    meeting each finger, the wrist's bends and turns, spreading, cupping and single-fingertip
    touch. Rows with no record yet are listed too, as not detected."""
    out = []
    subs = {s["sub"]: s for s in (views.get("H05") or {}).get("subs") or []}
    for side in ("L", "R"):
        for digit in ("thumb", "index", "middle", "ring", "little"):
            key = "%s:%s" % (side, digit)
            s = subs.get(key)
            # WHAT "ALONE" MEASURES (observer_attribution_v1, 2026-10-08): the other digits' END positions, not their
            # movement inside the act (detectors.H05); the row's note gives the inside-the-act count where measured
            out.append(dict(cap="H05", key="H05|" + key,
                            label="%s %s moves; the other fingers end where they began" % (
                                SIDE_WORDS[side].capitalize(), DIGIT_WORDS[digit]),
                            counted=(s or {}).get("counted", 0), lead=(s or {}).get("lead") or
                            "No stage passed yet: it has not been detected.", track=(s or {}).get("track"),
                            changes=(s or {}).get("changes"), notes=(s or {}).get("notes") or [],
                            stages=(s or {}).get("stages") or [
                                dict(key=k, title=t, question=STAGE_QUESTIONS[k], state="not", marker=MARKERS["not"],
                                     detail="Not detected yet.", text="Not yet: not detected yet.")
                                for k, t, _ in STAGE_TITLES]))
    for cap, label in (("H06", "Thumb meets a fingertip"), ("H07", "Wrist bends"), ("H08", "Wrist turns"),
                       ("HX1", "Fingers spread and gather"), ("HX2", "Palm cups and flattens"),
                       ("HX3", "One fingertip touches something")):
        v = views.get(cap)
        if v:
            out.append(dict(cap=cap, key=cap, label=label, counted=caps.get(cap, {}).get("counted", 0), lead=v["lead"],
                            track=v["track"], changes=v.get("changes"), stages=v["stages"]))
    return out


REGIONS = [
    ("Fingers", ["H05", "H06", "HX1", "HX2", "HX3", "B02"]),
    ("Hands", ["H01", "H02", "H03", "H04", "H09", "B01", "R04", "R05", "R06", "R07", "R08"]),
    ("Wrists", ["H07", "H08"]),
    ("Arms", ["R01", "R02", "R03", "B05", "B06"]),
    ("Head and face", ["P01", "B03", "B04"]),
    ("Trunk", ["P02", "P03", "P04", "P06", "P07", "P08"]),
    ("Legs", ["P05", "P09", "P10"]),
    ("Gaze", ["G01", "G02", "G03", "G04", "G05"]),
    ("Objects", ["O01", "O02", "O03", "O04", "O05", "O06", "O07", "O08", "O11", "O12"]),
]


def _names(names: List[str], most: int = 4) -> str:
    if len(names) <= most:
        return ", ".join(names)
    return "%s and %s more" % (", ".join(names[:most]), _num(len(names) - most))


def body_map(caps: Dict[str, dict]) -> List[dict]:
    """Each body region: which of its capabilities are beyond random flailing (babble) and which stages
    any of them reached (section 32's body map). Voice is listed as not measured: this world gives the
    observer no voice record."""
    out = []
    for region, ids in REGIONS:
        beyond = [cid for cid in ids if (((caps.get(cid) or {}).get("stages") or {}).get("observed_once") or {}).get("reached") is True]
        reached = [t for k, t, _ in STAGE_TITLES
                   if any((((caps.get(cid) or {}).get("stages") or {}).get(k) or {}).get("reached") is True for cid in ids)]
        measured = any((((caps.get(cid) or {}).get("stages") or {}).get("observed_once") or {}).get("reached") is not None
                       for cid in ids)
        if beyond:
            text = "%s of %s beyond random flailing: %s. Stages passed: %s." % (
                _num(len(beyond)), _num(len(ids)), _names([(caps.get(c) or {}).get("name", c) for c in beyond]),
                ", ".join(reached))
        elif measured:
            text = "%s; none beyond random flailing yet" % _plural(len(ids), "capability", "capabilities")
        else:
            text = "%s; not measurable yet: random flailing is not recorded" % _plural(
                len(ids), "capability", "capabilities")
        out.append(dict(region=region, stage=(reached[-1] if reached else None), text=text))
    out.append(dict(region="Voice", stage=None, text="Not measured: the observer has no voice record in this world."))
    return out


# (key, what, size, whether the size is judged against the lane's usual variation too). A support
# fall is measured below the usual level itself (fastlane.support_fall: the running median of the
# previous 2 s minus the most of the next three frames), a series that sits at 0, so its variation
# clause would add nothing.
FAST_LANES = [
    ("head_jolt", "Sudden rises in the head's acceleration", "at least 2 m/s², a fifth of gravity's pull", True),
    ("head_spin", "Sudden rises in the head's turning speed", "at least 1 radian a second, a sixth of a turn per second",
     True),
    ("support_drop", "Falls of the support from outside the body below its usual level",
     "a quarter of body weight or more under the median of the previous 2 s, held 30 ms", False),
    ("max_contact_force", "Sudden rises in the largest single contact force", "at least a quarter of body weight", True),
]


def _per10k(r: dict) -> str:
    v = (r or {}).get("spikes_per_10000")
    if v is None:
        return "not measured"
    if v == 0:
        return "0"
    return "%.1f" % v if v < 10 else _num(v)


def fast_lines(total: dict, chance: dict, n_bodies: int) -> List[str]:
    """Section 24's fast lane, one line per lane: the learner's bodies against random-order babble
    and a body with no commands in the same world (daemon.baseline_fast), per 10,000 frames of 10 ms
    (100 s of one body's life)."""
    out = []
    for key, what, size, varies in FAST_LANES:
        r = total.get(key) or {}
        if not r.get("samples"):
            continue
        line = ("%s (%s%s): %s per 100 s of one body's life over the %d bodies; random flailing %s; a body with no "
                "commands %s." % (
                    what, size, ", and more than 6 times the usual variation over the previous 2 s" if varies else "",
                    _per10k(r), n_bodies, _per10k(chance.get("random_orders", {}).get(key)),
                    _per10k(chance.get("limp", {}).get(key))))
        if key == "head_jolt" and r.get("spikes"):
            rec = r.get("mean_recovery_s")
            b = (chance.get("random_orders") or {}).get(key) or {}
            line += " Largest: %.1f g beyond gravity's pull%s%s." % (
                (r.get("largest") or 0) / 9.81,
                "" if not b.get("largest") else " (random flailing's largest: %.1f g)" % (b["largest"] / 9.81),
                "" if rec is None else "; back to the usual level after %d ms on average" % round(1000 * rec))
        out.append(line)
    if out:
        out.append("No goal line is set for these counts: fewer can mean less movement, not better control.")
    return out


def _vision_note(base: dict) -> str:
    """Whether babble saw what training sees: the eyes' motion reflex runs only on rendered images."""
    b, t = base.get("vision"), base.get("training_vision")
    if b is None or t is None:
        return ""
    if b == t:
        return "Its eyes' images were %s, as in training." % ("on" if b else "off")
    return ("Its eyes' images were %s while training's are %s, so the eyes' motion reflex %s: the gaze rows' "
            "chance rates do not match training yet." % ("on" if b else "off", "on" if t else "off",
                                                          "ran only in random flailing" if b else "ran only in training"))


MILESTONE_STAGES = ("observed_once", "repeated", "predicted_before_action", "selected_intentionally")


def milestone(caps: Dict[str, dict]) -> dict:
    """The first complete behaviour (outside review, 2026-09-24): beyond matched babble, repeats on
    separate occasions, predicted before the act better than its control, and traced to a decisive
    planner choice - the four stages together. Capability rows and their parts (a finger, a hand)
    both count. Returns the rows that pass all four, or the closest ones and what they lack."""
    rows = []
    for cid, row in caps.items():
        if not row.get("observable", True):
            continue
        cands = [(row.get("name") or cid, row.get("stages") or {})]
        for sk, sr in (row.get("subs") or {}).items():
            cands.append(("%s (%s)" % (row.get("name") or cid, sub_label(cid, sk)), sr.get("stages") or {}))
        for name, st in cands:
            have = [k for k in MILESTONE_STAGES if (st.get(k) or {}).get("reached") is True]
            rows.append((len(have), name, [k for k in MILESTONE_STAGES if k not in have]))
    done = [n for c, n, _ in rows if c == len(MILESTONE_STAGES)]
    best = max((c for c, _, _ in rows), default=0)
    closest = [(n, miss) for c, n, miss in rows if c == best and best > 0]
    return dict(done=done, best=best, closest=closest, withdrawn=withdrawn_rows(caps))


def milestone_line(m: dict) -> str:
    if m["done"]:
        return ("First full behavior reached: %s passes all four stages (Seen, Repeats, Predicted, "
                "Chosen)." % _names(m["done"]))
    # the Chosen stage reads the wellness evaluator's tie band (register entry
    # OBSERVER_SELECTED_STAGE_READS_UNOPTIMIZED_EVALUATOR): its 'not yet' shows no intentional skill demonstrated, and
    # never that the model learned no intentional behavior (the operator's addition of Oct 8, 2026)
    chosen_note = (" The Chosen stage reads the wellness evaluator's tie band, so this does not show that the model "
                   "learned no intentional behavior.")
    # tracker rules v6 (independent review of 4342f9194, blocker 1): a Predicted pass withdrawn with the single-command
    # credit is named, never dropped silently from the count
    gone = m.get("withdrawn") or []
    withdrawn_note = ("" if not gone else
                      " Predicted is not judged now for %s: %s passed it only under the single-command credit "
                      "withdrawn on Oct 8, 2026, and %s acts of the hands alone were recorded before that date." % (
                          _names(gone, most=3), "it" if len(gone) == 1 else "they", "its" if len(gone) == 1 else "their"))
    if not m["closest"]:
        return ("The first full behavior (more than random flailing, repeated, predicted before the act, and "
                "traced to a decisive planner choice) has not happened yet; no behavior has passed any of the four."
                + withdrawn_note + chosen_note)
    missing = sorted({_title(k) for _, miss in m["closest"] for k in miss})
    return ("The first full behavior (more than random flailing, repeated, predicted before the act, and traced "
            "to a decisive planner choice) has not happened yet. Closest, with %s of 4: %s; missing: %s." % (
                _num(m["best"]), _names([n for n, _ in m["closest"]], most=3), ", ".join(missing))
            + withdrawn_note
            + (chosen_note if "selected_intentionally" in {k for _, miss in m["closest"] for k in miss} else ""))


def _run_started(name: Optional[str]) -> Optional[str]:
    """'…_resume_20260924_092822' as '9:28 AM', or None."""
    try:
        stamp = name.rsplit("_resume_", 1)[1]
        t = time.strptime(stamp, "%Y%m%d_%H%M%S")
        return time.strftime("%I:%M %p", t).lstrip("0")
    except (AttributeError, IndexError, ValueError):
        return None


def stage_families(caps: Dict[str, dict]) -> Dict[str, List[float]]:
    """Each tested stage's p-values across the report (rows, caregiver-away columns and parts), the
    family its Benjamini-Hochberg line was drawn over (tracker._all_stages)."""
    out: Dict[str, List[float]] = {k: [] for k in TR.FAMILY_STAGES}
    for row in caps.values():
        sources = [row.get("stages") or {}, row.get("no_overseer") or {}]
        sources += [sr.get("stages") or {} for sr in (row.get("subs") or {}).values()]
        for src in sources:
            for k in TR.FAMILY_STAGES:
                p = (src.get(k) or {}).get("p_value")
                if p is not None:
                    out[k].append(float(p))
    return out


def clock(t: Optional[float]) -> str:
    """'Sep 29, 11:37 AM PDT' in this machine's local time (Pacific)."""
    if not t:
        return "an unrecorded time"
    lt = time.localtime(float(t))
    return "%s %d, %s %s" % (time.strftime("%b", lt), lt.tm_mday, time.strftime("%I:%M %p", lt).lstrip("0"),
                             "PDT" if lt.tm_isdst else "PST")


def copy_label(entry: dict) -> str:
    """A dated copy's label: its date and time, its training step, and whether it was rebuilt."""
    return "%s · step %s%s" % (clock(entry.get("time")), _num(entry.get("step")),
                               " · rebuilt later" if entry.get("rebuilt") else "")


def _report_label(rep: dict, name: Optional[str]) -> str:
    if name:
        e = HI.parse_name(name) or {}
        return copy_label(dict(e, rebuilt=bool(rep.get("rebuilt")) or e.get("rebuilt")))
    return "now (%s · step %s%s)" % (clock(rep.get("generated")), _num(HI.report_step(rep)),
                                     "" if rep.get("last_step") is not None else " or later")


def _rows_of(rep: dict) -> Dict[str, dict]:
    out = {}
    for cid, row in (rep.get("capabilities") or {}).items():
        out[cid] = row.get("stages") or {}
        for sk, sr in (row.get("subs") or {}).items():
            out["%s|%s" % (cid, sk)] = sr.get("stages") or {}
    return out


def _went(a, b, what: str) -> str:
    verb = "rose" if b > a else "fell"
    return "%s %s from %s to %s" % (what, verb, _amount(a) if isinstance(a, float) and a < 100 else _num(a),
                                    _amount(b) if isinstance(b, float) and b < 100 else _num(b))


def row_change(old: dict, new: dict) -> Optional[List[str]]:
    """What changed in one row from an older report to a newer one: stages gained or lost, and its
    occasions and flailing's count for the same time, up or down. None when nothing changed."""
    o, n = HI.row_numbers(old or {}), HI.row_numbers(new or {})
    out = []
    gained = [_title(k) for i, k in enumerate(HI.STAGES) if n[2] >> i & 1 and not o[2] >> i & 1]
    lost = [_title(k) for i, k in enumerate(HI.STAGES) if o[2] >> i & 1 and not n[2] >> i & 1]
    if gained:
        out.append("Gained %s." % _and(gained))
    if lost:
        out.append("Lost %s." % _and(lost))
    if o[0] is not None and n[0] is not None and o[0] != n[0]:
        line = _went(o[0], n[0], "Occasions")
        if o[1] is not None and n[1] is not None and round(o[1], 1) != round(n[1], 1):
            line += "; flailing's count for the same time " + _went(o[1], n[1], "")[1:]
        out.append(line + ".")
    return out or None


def compare_view(shown: dict, shown_name: Optional[str], other: dict, other_name: Optional[str],
                 resets: List[dict]) -> tuple:
    """The comparison of the report shown with another one: (its header, {row key: change lines}),
    always from the older report to the newer one."""
    a = dict(rep=shown, name=shown_name, time=float(shown.get("generated") or 0))
    b = dict(rep=other, name=other_name, time=float(other.get("generated") or 0))
    old, new = (b, a) if b["time"] <= a["time"] else (a, b)
    rules = HI.compare_rules(dict(time=old["time"], rules=HI.rules_key(old["rep"])),
                             dict(time=new["time"], rules=HI.rules_key(new["rep"])), resets)
    ro, rn = _rows_of(old["rep"]), _rows_of(new["rep"])
    changes = {}
    for key in sorted(set(ro) | set(rn)):
        c = row_change(ro.get(key) or {}, rn.get(key) or {})
        if c:
            changes[key] = c
    gained = sum(1 for c in changes.values() for line in c if line.startswith("Gained"))
    lost = sum(1 for c in changes.values() for line in c if line.startswith("Lost"))
    head = dict(older=_report_label(old["rep"], old["name"]), newer=_report_label(new["rep"], new["name"]),
                like_for_like=rules["like"], rows_changed=len(changes))
    head["text"] = "Changes from %s to %s: %s gained a stage, %s lost one (a row is a behavior or one of its " \
                   "parts, such as one finger)." % (head["older"], head["newer"], _plural(gained, "row"), _plural(lost, "row"))
    if not rules["like"]:
        when = "; ".join("%s (step %s)" % (clock(r["time"]), _num(r.get("step"))) for r in rules["resets"])
        head["warning"] = ("Not like for like: the counting rules, the detectors or the random-flailing recording "
                           "changed between these two dates%s, so some changes come from the new rules, not from the "
                           "learner." % (" (%s)" % when if when else ""))
    return head, changes


def _withdrawn_text(withdrawn) -> str:
    """'; 1 stage that stood before does not stand under the new rules: Individual finger isolation Predicted (not
    judged)' for a reset's withdrawn stages (daemon.stage_history), or ''."""
    w = list(withdrawn or [])
    if not w:
        return ""
    items = ["%s %s (%s)" % (x.get("cap"), _title(x.get("stage")), x.get("now")) for x in w]
    return "; %s that stood before %s not stand under the new rules: %s" % (
        _plural(len(w), "stage"), "does" if len(w) == 1 else "do", _names(items, most=3))


def history_view(root: str) -> dict:
    """The dated copies for the date list and the small charts (history.py): each copy's label and counting
    rules, the rule changes between them, and per row the occasions against flailing's count for the same
    time at every copy and now."""
    saved = os.path.join(root, "Saved", "mimo_bridge")
    sup = _read(os.path.join(saved, "training_continuation.json"), {}) or {}
    active = sup.get("active_run")
    if not active:
        return dict(available=False, entries=[], sentence="No active training run is recorded.")
    name = os.path.basename(active)
    lineage = name.split("_seg")[0] if "_seg" in name else name
    live_folder = os.path.join(saved, "observer", lineage)
    folder = os.path.join(HI.history_root(), lineage)
    present = HI.drive_present(folder)
    entries = HI.load_entries(folder) if present else []
    resets = HI.read_resets(live_folder)
    live = _read(os.path.join(live_folder, "development.json"))
    points = [dict(e, label=copy_label(e)) for e in entries]
    if live:
        s = HI.summary(live)
        points.append(dict(s, file="now", time=float(live.get("generated") or time.time()), step=s["step"],
                           rebuilt=False, label=_report_label(live, None)))
    groups = []
    for i, e in enumerate(points):
        if i and not HI.compare_rules(points[i - 1], e, resets)["like"]:
            groups.append(i)
    t0 = points[0]["time"] if points else None
    series: Dict[str, list] = {}
    for i, e in enumerate(points):
        for key, nums in (e.get("rows") or {}).items():
            series.setdefault(key, []).append([i] + list(nums))
    note = None
    if not present:
        note = ("The history disk (%s) is not connected, so no dated copy can be listed; the page shows only "
                "now." % os.path.splitdrive(folder)[0])
    elif not entries:
        note = "No dated copy has been written yet: the observer writes one every 4 hours."
    return dict(available=True, folder=folder, drive=present, note=note,
                entries=[dict(file=e["file"], label=e["label"], time=e["time"], step=e.get("step"),
                              rebuilt=bool(e.get("rebuilt"))) for e in points],
                breaks=groups,
                rule_changes=[dict(time=r["time"], step=r.get("step"), withdrawn=r.get("withdrawn") or [],
                                   label="Counting rules changed %s (step %s)%s" % (
                                       clock(r["time"]), _num(r.get("step")), _withdrawn_text(r.get("withdrawn"))))
                              for r in resets if t0 is None or r["time"] >= t0 - 86400],
                series=series)


def build(root: str, at: Optional[str] = None, compare: Optional[str] = None) -> dict:
    """The page. at: a dated copy's file name (history.py) to show instead of the live report; compare:
    another copy's name, or 'now', whose differences each row then shows."""
    saved = os.path.join(root, "Saved", "mimo_bridge")
    status = _read(os.path.join(saved, "observer", "daemon_status.json"), {}) or {}
    sup = _read(os.path.join(saved, "training_continuation.json"), {}) or {}
    active = sup.get("active_run")
    now = time.time()
    beat = _read(os.path.join(saved, "observer", "daemon_heartbeat.json"), {}) or {}
    last = max(float(status.get("time", 0) or 0), float(status.get("finished", 0) or 0), float(beat.get("time", 0) or 0))
    service = dict(running=bool(status) and now - last < 600,
                   last_cycle_age_s=(now - last) if last else None,
                   ok=status.get("ok"), error=(status.get("error") or "")[-600:] or None,
                   memory_mb=status.get("memory_mb"))
    if not active:
        return dict(available=False, sentence="No active training run is recorded, so the observer has nothing to digest.",
                    service=service)
    name = os.path.basename(active)
    lineage = name.split("_seg")[0] if "_seg" in name else name
    rep_path = os.path.join(saved, "observer", lineage, "development.json")
    rep = _read(rep_path)
    live = rep
    hist_folder = os.path.join(HI.history_root(), lineage)
    shown_copy = dict(kind="live", file=None)
    if at and at != "now":
        snap = HI.load_report(hist_folder, at)
        if snap is None:
            return dict(available=False, service=service,
                        sentence="The dated copy %s could not be read from %s." % (at, hist_folder))
        rep = snap
        shown_copy = dict(kind="copy", file=at, label=_report_label(snap, at), rebuilt=bool(snap.get("rebuilt")))
        if snap.get("rebuilt"):
            shown_copy["note"] = ("This copy was rebuilt on %s from the observer's records, under today's counting rules. "
                             "It leaves out %s." % (clock((snap["rebuilt"] or {}).get("at")),
                                                    _and((snap["rebuilt"] or {}).get("missing") or ["nothing"])))
    if not rep:
        rec = os.path.isdir(os.path.join(active, "observer", "world"))
        return dict(available=False, service=service,
                    sentence=("The developmental observer is recording this run, and its first report has not "
                              "been written yet." if rec else
                              "The active run was launched without the observer's recorders (--observer record), "
                              "so no developmental observation exists for it."))
    caps = rep.get("capabilities") or {}
    replays = {k: dict(v, gif_path=(os.path.relpath(os.path.join(os.path.dirname(rep_path), v["gif"]), root).replace("\\", "/")
                                    if v.get("gif") else None)) for k, v in (rep.get("replays") or {}).items()}
    hz = rep.get("horizon") or {}
    ch = hz.get("choices") or {}
    planner_tied = bool(ch.get("judged")) and not ch.get("best")
    conf = (_read(os.path.join(root, VALIDATION), {}) or {}).get("confusion")
    validated = set(conf) if isinstance(conf, dict) else None
    comparison, changes = None, {}
    if compare:
        other = live if compare == "now" else HI.load_report(hist_folder, compare)
        if other is None:
            comparison = dict(text=("The live report could not be read, so nothing is compared." if compare == "now"
                                    else "The dated copy %s could not be read, so nothing is compared." % compare))
        else:
            comparison, changes = compare_view(rep, shown_copy.get("file"), other, None if compare == "now" else compare,
                                               HI.read_resets(os.path.dirname(rep_path)))
    families = stage_families({c: r for c, r in caps.items() if c != "O12"})
    ctx = dict(planner_tied=planner_tied, validated=validated, families=families, changes=changes,
               choices_judged=ch.get("judged"), choices_best=ch.get("best") or 0)
    # O12's stages are judged by a family of their own (daemon.build_report: a separate track_all)
    views = {cid: row_view(r, replays, dict(ctx, families=stage_families({cid: r})) if cid == "O12" else ctx)
             for cid, r in caps.items()}

    def reached(stage):
        return [cid for cid in sorted(caps) if ((caps[cid].get("stages") or {}).get(stage) or {}).get("reached") is True]
    names = {cid: caps[cid].get("name", cid) for cid in caps}
    observable = sum(1 for r in caps.values() if r.get("observable", True))
    steps = sum((v.get("steps") or 0) for v in (rep.get("exposure") or {}).values())
    n_bodies = len(rep.get("exposure") or {})
    lived = "%s of simulated life summed over %s bodies (%s each)" % (_life(steps), n_bodies,
                                                                     _life(steps / max(1, n_bodies)))
    base = rep.get("baseline") or {}
    lines = []
    if base.get("available"):
        seen, rep_, pred, chosen, kept = (reached(s) for s in ("observed_once", "repeated", "predicted_before_action",
                                                               "selected_intentionally", "retained"))
        lines.append("Random flailing is the comparison: the same bodies in this world moved by random commands, with "
                     "no learner (the code calls it babble).")
        lines.append("Across %s, %s of the %s capabilities observable here happen more often or bigger than random "
                     "flailing%s." % (
            lived, _num(len(seen)), _num(observable), (": " + _names([names[c] for c in seen])) if seen else ""))

        def predicted_stage(cid):
            return (caps[cid].get("stages") or {}).get("predicted_before_action") or {}
        # tracker rules v6 (independent review of 4342f9194, blocker 1): a Predicted pass withdrawn with the
        # single-command credit is named here, and so is every capability whose Predicted is not judged
        unjudged = [cid for cid in sorted(caps) if predicted_stage(cid).get("not_judged")]
        gone = [cid for cid in unjudged if (predicted_stage(cid).get("earlier_rule") or {}).get("stage_reached")]
        if seen:
            lines.append("Of these, %s %s on separate occasions more often than flailing, and the learner predicted %s "
                         "before %s happened%s." % (
                             _num(len(rep_)), "repeats" if len(rep_) == 1 else "repeat", _num(len(pred)),
                             "it" if len(pred) == 1 else "they",
                             (" under the current rule; %s more (%s) passed Predicted only under the single-command "
                              "credit withdrawn on Oct 8, 2026, and %s not judged now" % (
                                  _num(len(gone)), _names([names[c] for c in gone]),
                                  "is" if len(gone) == 1 else "are")) if gone else ""))
        if unjudged:
            lines.append("Predicted is not judged for %s: %s acts of the hands alone were recorded before Oct 8, 2026, "
                         "when each act's movement verdict was read on the one command it was credited to, and the "
                         "current rule judges every command in force during an act." % (
                             _plural(len(unjudged), "capability", "capabilities"),
                             "its" if len(unjudged) == 1 else "their"))
        if planner_tied:
            lines.append("No capability passed Chosen: in the current run the chosen command's predicted wellness beat "
                         "every alternative's by more than the tie band (the evaluator's own error on a difference "
                         "between two candidates) in 0 of %s choices. The band has been wider than the candidates' "
                         "predicted spread in every report window since Sep 23, so this count is no evidence about the "
                         "forward model." % _num(ch.get("judged")))
        else:
            lines.append("The planner chose %s decisively." % _plural(len(chosen), "capability", "capabilities"))
        if rep_ and not kept:
            lines.append("None is kept yet: keeping needs two later blocks of 2.5 minutes of each body's life.")
        below = []
        tested = [cid for cid in caps if (((caps[cid].get("stages") or {}).get("repeated") or {}).get("below_babble_p")) is not None]
        for cid in tested:
            r = (caps[cid].get("stages") or {}).get("repeated") or {}
            if r["below_babble_p"] <= 0.05 / max(1, len(tested)):
                below.append((r["below_babble_p"], cid, r))
        if below:
            below.sort()
            _, cid, r = below[0]
            lines.append("%s happen less often than random flailing, for example %s (%s against flailing's %s)." % (
                _plural(len(below), "capability", "capabilities"), names[cid], _plural(r.get("occasions") or 0, "occasion"),
                _amount(r.get("expected_by_chance"))))
        lines.append(milestone_line(milestone(caps)))
        if seen and validated is not None:
            n_ok = sum(1 for c in seen if c in validated)
            lines.append("%s of these %s detectors %s scripted test cases (%s of the catalog's %s have)." % (
                _num(n_ok), _num(len(seen)), "has" if n_ok == 1 else "have", _num(len(validated)), _num(len(caps))))
    else:
        lines.append("Across %s the observer detected %s acts. Whether any capability happens more than random "
                     "flailing (the same bodies moved by random orders, with no learner; the code calls it babble) is "
                     "not measurable yet: this world has no recording of it with the current detectors." % (
                         lived, _num(((rep.get("counts") or {}).get("live_detections")) or 0)))
    groups = []
    for g in GROUP_ORDER:
        rows = [views[c["id"]] for c in sorted(caps.values(), key=lambda r: r["id"]) if c.get("group") == g]
        if rows:
            n_seen = sum(1 for r in rows if (((caps[r["id"]].get("stages") or {}).get("observed_once") or {}).get("reached")))
            groups.append(dict(title=g, rows=rows, summary="%s · %s · %s beyond random flailing" % (
                g, _plural(len(rows), "capability", "capabilities"), _num(n_seen))))
    timeline = []
    first = (rep.get("recording") or {}).get("first_step")
    if first is not None:
        timeline.append(dict(step=first, text="Step %s: the observer began recording. A stage reached at the first "
                                               "recorded steps may have been reached earlier, unseen." % _num(first)))
        timeline.append(dict(step=first, note=True, text="Dates are recomputed at every report: Seen and Repeats are "
                             "dated at the occasion from which their test has passed at every later occasion so far "
                             "(a brief early pass that lapsed does not date them), and a later stage is never dated "
                             "before the one it waits for, so stages can share a date."))

    def when(step):
        if first is None or step < first:
            return "Step %s" % _num(step)
        into = "under 1 second" if (step - first) * STEP_S < 1 else _life(step - first)
        return "Step %s (%s into each body's recorded life)" % (_num(step), into)
    for ev in (rep.get("timeline") or [])[-60:]:
        if ev["stage"].startswith("lost:"):
            timeline.append(dict(step=ev["step"], lost=True, text="%s: %s - lost '%s': it passed in an earlier "
                                 "report and no longer does." % (when(ev["step"]), ev.get("name"), _title(ev["stage"][5:]))))
            continue
        t = next((s for k, s, _ in STAGE_TITLES if k == ev["stage"]), ev["stage"])
        timeline.append(dict(step=ev["step"], text="%s: %s - %s." % (when(ev["step"]), ev.get("name"),
                                                                     "regressed" if ev["stage"] == "regressed" else t.lower())))
    novelty = []
    shown = []
    for c in rep.get("novelty") or []:
        rp = replays.get((c.get("examples") or [None])[0])
        if c.get("chance_measured"):
            chance = "; random flailing: %s in the same time" % _amount(c.get("expected_by_chance") or 0)
        else:
            chance = "; how often random flailing makes it is not measured yet"
        # the order control (novelty.order_test): the same episodes with their own events shuffled
        if c.get("order_expected") is not None:
            chance += "; with each of those episodes' own events in a random order, %s would show it" % _amount(
                c["order_expected"])
        toks = list(c.get("tokens") or [])
        words = ", then ".join(token_words(t) for t in toks)
        text = ("%s%s. Seen in %s (an episode is one body part's stretch of activity - a hand, the eyes and head, "
                "the trunk or a leg - ending after half a second of quiet, or two parts' episodes starting within "
                "half a second of each other) on %s%s. %s%% of them came in the "
                "later half of the record (50%% = as common early as late)%s." % (
                    words[:1].upper(), words[1:], _plural(c.get("episodes"), "episode"),
                    _plural(c.get("bodies"), "body", "bodies"), chance, int(round(100 * (c.get("later_share") or 0))),
                    "; it changes an object" if c.get("world_effect") else ""))
        if len(toks) > 1 and list(reversed(toks)) in shown:
            text += " The same events as a pattern above, in the other order."
        shown.append(toks)
        novelty.append(dict(label=c.get("label"), text=text,
                            replay=("/files/" + rp["gif_path"]) if rp and rp.get("gif_path") else None))
    regress = [dict(id=v["id"], text="%s: %s" % (v["name"], v["regressed_text"])) for v in views.values() if v["regressed"]]
    # a part (one hand, one finger, one target) regresses on its own record too: listed by its own flag,
    # and counted among the checked (until 2026-09-25 the watch counted whole rows only, so a part's
    # regression never reached it)
    for cid in sorted(caps):
        for sk, sr in sorted((caps[cid].get("subs") or {}).items()):
            if sr.get("regressed"):
                regress.append(dict(id=cid, sub=sk, text="%s (%s): REGRESSED: the newest block of 3,000 steps fell "
                                    "below half the rate at which it first repeated." % (
                                        caps[cid].get("name", cid), sub_label(cid, sk))))
    checked = sum(1 for r in caps.values() if (((r.get("stages") or {}).get("retained") or {}).get("windows_after")))
    checked += sum(1 for r in caps.values() for sr in (r.get("subs") or {}).values()
                   if (((sr.get("stages") or {}).get("retained") or {}).get("windows_after")))
    if regress:
        regression_note = ""
    elif not checked:
        regression_note = ("Not checked yet: a capability can regress only after it repeats and a later block of "
                           "3,000 steps (2.5 minutes of each body's life) is recorded; none has one yet.")
    else:
        regression_note = ("No repeating capability or part of one fell below half its first-repeat rate in its newest "
                           "block (%s checked)." % _num(checked))
    horizon = []
    depths = hz.get("depths") or []
    if depths:
        w = [int(round(100 * d["beats_do_nothing_share"])) for d in depths]
        horizon.append("Looking 50 ms ahead, the imagined next moment came closer to what happened than 'nothing "
                       "changes' did in %d of the last 100 or so updates%s." % (
                           w[0], ", so it lost more often than it won" if w[0] < 50 else ""))
        if len(depths) > 1:
            horizon.append("Looking %s ahead it won %s." % (
                " and ".join("%d ms" % d["horizon_ms"] for d in depths[1:]), " and ".join(str(x) for x in w[1:])))
        horizon.append("Each of these running shares has a noise of about 3.5 per 100 (one standard deviation). 50 "
                       "would only tie 'nothing changes': a floor, not the goal; how far these are from planning well "
                       "is unmeasured.")
    if hz.get("planner_max_steps"):
        lic = int(hz.get("licensed_depth") or 0)
        horizon.append("The planner may look %d ms ahead now, %d ms at most. A depth adds 50 ms when it and every "
                       "shorter depth beat 'nothing changes' in more than 50 of 100; %s." % (
                           50 * int(hz["planner_max_steps"]), 50 * max(1, len(depths)),
                           "no depth has earned it yet" if lic == 0 else "%s have" % _num(lic)))
    if ch:
        started = _run_started((rep.get("runs") or [None])[-1])
        horizon.append("In the current run%s, the chosen command's predicted wellness beat every alternative's by more "
                       "than the tie band (the evaluator's own error on a difference between two candidates) in %s of "
                       "%s choices%s." % (
                           " (since %s PDT)" % started if started else "", _num(ch.get("best") or 0),
                           _num(ch.get("judged") or 0),
                           ": every choice tied, so no act can pass Chosen. While the band is wider than the candidates' "
                           "predicted spread, a tie is certain whatever the forward model predicts, so the count is no "
                           "evidence about the forward model" if planner_tied else ""))
    horizon.append("Not measured yet: whether looking further ahead changes which plan is chosen. The test: from the "
                   "same saved moments, let the planner choose at 50 ms and at 150 ms, run both, and compare what the "
                   "bodies did.")
    fast = fast_lines(rep.get("fast_lane_total") or {}, rep.get("fast_lane_chance") or {}, n_bodies)
    health = []
    drive = os.path.splitdrive(active)[0] or "The run's disk"
    for run, h in (rep.get("health") or {}).items():
        rows = sorted((h.get("bodies") or {}).items(), key=lambda kv: int(kv[0]))
        recs = {int((s.get("status") or {}).get("records") or 0) for _, s in rows}
        if rows and len(recs) == 1 and not any((s.get("status") or {}).get("dropped") or s.get("error") or s.get("gaps")
                                               for _, s in rows):
            n = recs.pop()
            health.append("All %d bodies: %s recorded each in the current run (%s of each body's life), none dropped." % (
                len(rows), _plural(n, "step"), _life(n)))
            rows = []
        for b, s in rows:
            st = s.get("status") or {}
            health.append("Body %d: %s recorded, %s dropped%s%s." % (
                int(b) + 1, _plural(st.get("records") or 0, "step"), _num(st.get("dropped") or 0),
                "; a recorder ERROR stopped its recording" if s.get("error") else "",
                ("; %s listed" % _plural(s["gaps"], "gap")) if s.get("gaps") else ""))
        if h.get("disk_free_gib") is not None:
            reserve = h.get("reserve_gib")
            health.append("Disk %s has %.1f GB free; the recorders stop writing below %s so checkpoints keep their room." % (
                drive, h["disk_free_gib"] * 1.073741824,
                ("%.1f GB" % (reserve * 1.073741824)) if reserve else "their reserve"))
    exposure = rep.get("exposure") or {}
    away = {int(b): float(v.get("absent_steps") or 0) for b, v in exposure.items()}
    away_steps = sum(away.values())
    no_overseer_label = ("With the caregiver away (%s of simulated life summed over bodies %s)" % (
        _life(away_steps), ", ".join(str(b + 1) for b in sorted(away) if away[b] > 0))
        if away_steps else "With the caregiver away (the caregiver was never away)")
    return dict(available=True, headline=" ".join(lines), headline_lines=lines, generated=rep.get("generated"),
                service=service, runs=rep.get("runs"),
                stages=[dict(key=k, title=t, meaning=m, question=STAGE_QUESTIONS[k]) for k, t, m in STAGE_TITLES],
                markers=MARKERS, full_behaviour=FULL_BEHAVIOUR, lineage=lineage, shown=shown_copy,
                comparison=comparison, last_step=HI.report_step(rep),
                groups=groups, fingers=finger_rows(caps, views), body_map=body_map(caps),
                timeline=timeline, novelty=novelty, regressions=regress, regression_note=regression_note,
                horizon=horizon, fast=fast, health=health, no_overseer_label=no_overseer_label,
                baseline=(("Random flailing (the code calls it babble), the comparison: the same bodies in this world "
                           "driven by random orders with no learner. Every 250 ms each controlled part gets a random orientation, a position up to %s cm "
                           "away on each axis, and a strength between 0 (no force) and %s (1 is the controller's full "
                           "force). %s, %s of simulated life summed over its bodies. %s" % (
                               _num(base.get("order_reach_cm") or 3), base.get("order_strength") or 0.3,
                               ("Started from production's own saved worlds at step %s" % _num(base.get("checkpoint_step"))
                                if base.get("start") == "saved_worlds" and base.get("checkpoint_step") is not None
                                else "Started from a fresh world"),
                               _life(base.get("steps") or 0), _vision_note(base)))
                          if base.get("available") else
                          "Random flailing (the code calls it babble) is not recorded for this world yet (python -m "
                          "mimo_bridge.tools.observer baseline)."),
                roles=rep.get("roles"),
                links=dict(design="/files/docs/gain_of_function_v4/OBSERVER_DESIGN.md",
                           catalog="/files/docs/gain_of_function_v4/capability_catalog.json",
                           validation="/files/docs/gain_of_function_v4/observer_validation/validation_report.json",
                           report="/files/" + os.path.relpath(rep_path, root).replace("\\", "/")))
