# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The section 11 scanner flags bare score decimals and passes counts, units and named-end percentages."""
from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, page_text


def _numbers(text):
    return [n for n, _ in bare_decimals(text)]


def test_bare_scores_and_scientific_notation_are_found():
    assert _numbers("direction match 0.14, the live model 0.63") == ["0.14", "0.63"]
    assert _numbers("foresaw 7.3e-06 of the change") == ["7.3e-06"]
    assert _numbers("distance 0.9999999268563718 from the goal line") == ["0.9999999268563718"]


def test_counts_units_and_percentages_pass():
    text = ("32 out of 100 orders; 0.8 cm off; 0.25 s later; 20% of the way from 'no difference' to "
            "'exact'; 1.25 steps per second; 12.5 degrees; step 344,400; spec v3.6; 2026-09-22")
    assert _numbers(text) == []


def test_a_typed_launch_setting_is_a_literal_not_a_score():
    assert _numbers("--babble-noise 0.05 (was --babble-noise 0.03); --finger-armature=0.0004") == []
    assert _numbers(page_text("<p>run it with <code>--decoder-kp 0.5 x</code> then</p>")) == []
    assert _numbers("the score 0.05 of --babble-noise") == ["0.05"]


def test_long_floats_and_doubled_sentences_are_found():
    from mimo_bridge.tools.check_no_bare_decimals import doubled_sentences, long_numbers
    assert long_numbers("distance 0.9999999268563718 and 1.0000000000000002, step 344,400") == [
        "0.9999999268563718", "1.0000000000000002"]
    assert long_numbers("0.0000073% of the way; 12.5 degrees; 1,234.5 steps") == []
    para = "When only the gaze command changed, the picture moved 2% of the way."
    assert doubled_sentences("Scale: " + para + ". " + para) != []
    assert doubled_sentences("Scale: the whole thing.. next") != []
    assert doubled_sentences("One short line. Another different longer sentence here with words.") == []
    assert doubled_sentences("Wait... then go on with the sentence here now.") == []


def test_page_text_drops_scripts_styles_and_tags():
    markup = "<style>.a{opacity:0.5}</style><script>var x=0.25;</script><p>turned <b>0.3</b> cm</p>"
    assert _numbers(page_text(markup)) == []
    assert _numbers(page_text("<p>score <b>0.3</b></p>")) == ["0.3"]


# ------------------------------------------------------------ numbers of 1 or more (fix-scanner 2026-09-23)
# Each example is a sentence a page printed. bare_decimals passes every one of them (the number is not
# below 1, or a letter sits before it), which is why unanchored_numbers exists.
from mimo_bridge.tools.check_no_bare_decimals import readout_problems, unanchored_numbers  # noqa: E402


def _rules(text):
    return [(n, rule) for n, rule, _ in unanchored_numbers(text)]


POSTURE_0922 = ("A head turn the body made itself was predicted about as well as the same turn pushed from outside "
                "(error ratio 1.003; above 1 = its own turn predicted better, 1 = no difference).")
TOUR_BEFORE = "the run tab's fresh-vision share reads 1.0 when that holds"
DRIFT_CHIPS = ('<span class="chip w">ungraded-channel drift x1.0</span><span class="chip g">causal error '
               '0.012-&gt;0.012 (+6% worse)</span>')
BODY_0923 = ("Of the tenure pose head's hidden units, 0% had an input above -3, where a unit still responds (0% = "
             "the head returns the same number whatever the state and order, 100% = every unit responds); their "
             "inputs ranged from -32.9 to -7.1.")


def test_a_score_of_1_or_more_after_a_score_word_is_found():
    """The posture probe's headline of 2026-09-22 19:40 (gof_probes/index.jsonl) and the explainer's
    senses tour before verify-fix-explainer (walkthrough.py at ddf4d3922)."""
    assert _numbers(POSTURE_0922) == [] and _numbers(TOUR_BEFORE) == []
    assert _rules(POSTURE_0922) == [("1.003", "score")]
    assert _rules(TOUR_BEFORE) == [("1.0", "score")]
    assert _rules("the ratio was 1.36") == [("1.36", "score")]
    assert _rules("replay loss 2.31 and falling") == [("2.31", "score")]


def test_a_score_word_beside_a_unit_a_count_or_a_scale_end_passes():
    assert _rules("a margin of 1.5 cm; a share of 12.5%; the error was 2.3 times the do-nothing error") == []
    # the explainer's own wording (walkthrough.py, schema.py)
    assert _rules("the run tab's fresh-vision share reads 100 of every 100 steps when that holds") == []
    assert _rules("Touch pleasure vs pain, on the scale 1.0 = all your weight on one small spot") == []
    assert _rules("a slot that beat all 8 in all 3 windows reads a standing of 63.6 in 100") == []
    # a range whose unit follows its second end (the Now page's WHO windows; the old dashboard's chip)
    assert _rules("WHO standing alone window: 6.9–16.9 months; a margin of 1.2 to 2.5 cm") == []
    assert _rules("vision foresaw 0.14% of the change; the probe's share reads 1.0 in every record") == [
        ("1.0", "score")], "'in every record' is not a count: nothing says in every how many"


def test_a_multiplier_with_nothing_named_after_it_is_found():
    """The old dashboard's chip before verify-fix-dashboards (dashboard_history/dashboard_2026-09-23_0300.html):
    below 1 a letter precedes it and above 1 it is not below 1, so bare_decimals passed both."""
    assert _numbers(page_text(DRIFT_CHIPS)) == ["0.012", "0.012"], "the next chip's own decimals, not the drift"
    assert _rules(page_text(DRIFT_CHIPS)) == [("x1.0", "multiplier")], "a tag boundary is not an anchor"
    assert _rules("ungraded-channel drift x0.8;") == [("x0.8", "multiplier")]
    assert _rules("the error grew 1.3x.") == [("1.3x", "multiplier")]
    assert _rules("the undamaged residual runs measured 1.16 to 2.45 times; no jump") == [("2.45 times", "multiplier")]
    assert _rules("ungraded-channel drift x1;") == [("x1", "multiplier")], "a whole number after a score word"


def test_a_multiplier_that_names_what_it_multiplies_passes():
    # dashboard.py's drift row and fault chip, viewer.js's memory ring, schema.py's attention mask
    assert _rules("hearing 4.2 times its first value, inter-mind the same as its first value.") == []
    assert _rules("RECONSTRUCTION BROKEN on 2 sense(s), up to 47x next-state error") == []
    assert _rules("the transformer itself reads 64 moments = 6.4 s of body time, so this ring reaches 3x further "
                  "back") == []
    assert _rules("a fixed 64x64 true/false mask blocks every column later than the row") == []
    assert _rules("its closing line, 1 time the readings' own standard error") == []
    # an earlier multiplier of the same sentence names it (dashboard.py's reconstruction row)
    assert _rules("It sits at 2.3 times its next-state error - above 1 time is expected with residual prediction "
                  "on, and undamaged residual runs measured 1.16 to 2.45 times; it becomes a FAULT only on a jump "
                  "past 4 times its own recent value.") == []
    # the explainer's segment 31 page, 2026-09-23: the sentence says what is multiplied; a repeat count;
    # a count out of 100
    assert _rules("how good it felt multiplied by (1 + the curiosity drive) - 1x for a moment whose prediction "
                  "error sat outside the learnable band, 2x at the band's midpoint") == []
    assert _rules("imagine: predict, feed back, ×3") == []
    assert _rules("pressure, shear, slip, area, vib x4, self-touch share, own parts on it /4, push direction x3") == []
    assert _rules("ranks one fall against another correctly 99.8 times in 100 above 30 cm") == []


def test_a_raw_hidden_unit_input_value_is_found():
    """The body probe's reader-liveness sentence of 2026-09-23 01:59 (gof_probes/index.jsonl) and the
    addendum's pose-head paragraph (typeset minus signs)."""
    assert _numbers(BODY_0923) == []
    assert _rules(BODY_0923) == [("-3", "unit input"), ("-32.9", "unit input")]
    assert _rules("all 256 units sit at −29 to −5.5, where the activation function (GELU) outputs 0") == [
        ("−29", "unit input")]
    assert _rules("which alone puts every unit's input near −17, far below the −3 line") == [
        ("−17", "unit input")]
    # a count of the things feeding a unit is not an input value, nor is a scale's named end
    assert _rules("the motor pool's units read inputs from 26 handles") == []
    assert _rules("how many units ever respond (0 = dead: it outputs its bias for every input, 100 = every unit "
                  "responds)") == []


def test_counts_units_clock_times_dates_versions_and_commits_pass_every_rule():
    # commit 2e3746163 holds only digits around one "e" (Oct 5, 2026: read as a number in exponent form until then)
    text = ("at 3:44 AM on 2026-09-23, spec v3.6 (UEMA_Spec_v3_6.txt section 19.2), commit d416e834e, step "
            "344,400 of 1,234.5 steps, 1.25 steps per second, 12.5 degrees, 7.07 of 8.19 GB, 32 out of 100 "
            "orders, 7 in every billion, 2.4 self-contact patches per body, a 192-wide core with 4 blocks "
            "(14.4 million parameters), commit 2e3746163, HEAD 2e374616")
    assert readout_problems(text) == []
    assert _numbers("foresaw 7.3e-06, 1e+20, 1e5 and 3e10 of it") == ["7.3e-06", "1e+20", "1e5", "3e10"]


def test_readout_problems_holds_the_bare_decimals_and_the_new_rules_together():
    found = readout_problems("direction match 0.14; " + POSTURE_0922)
    assert [(n, rule) for n, rule, _ in found] == [("0.14", "bare decimal"), ("1.003", "score")]


def test_a_unit_after_a_number_must_be_a_whole_word():
    """The explainer's second recheck of Oct 15, 2026: "the 0.00001 perturbation" passed because "per" matched the start
    of "perturbation". Since the decision explainer-part2-rule-confirmed-20261015 every caller reads each unit as a
    whole word (UNIT_AFTER is UNIT_AFTER_STRICT); the list used until then (UNIT_AFTER_PREFIX) excused every one of the
    prefixed texts below. Whole units, their plurals, "percent" and "per cent" still pass, and so do a launch setting
    and a commit's short hash."""
    from mimo_bridge.tools import check_no_bare_decimals as C
    assert C.UNIT_AFTER is C.UNIT_AFTER_STRICT
    prefixed = ("the 0.00001 perturbation", "a 0.3 perception share", "a 0.4 period", "a 0.2 secondary term",
                "a 0.6 daylight level", "the 0.7 hourly mean", "the 0.1 section score", "the 0.2 cmd share")
    for text in prefixed:
        assert [n for n, _ in bare_decimals(text)] == [text.split()[1]], text
        after = text[text.index(text.split()[1]) + len(text.split()[1]):]
        assert C.UNIT_AFTER_PREFIX.match(after), text
    whole = ("0.5 percent", "0.5 per cent", "0.5 per handle", "0.25 s later", "0.25 seconds", "0.3 days", "0.5 hours",
             "0.8 cm", "0.4 model steps", "0.5 degrees", "0.2 x the error", "0.5 bytes", "0.5%", "0.5 min")
    for text in whole:
        assert bare_decimals(text) == [], text
    assert bare_decimals("--babble-noise 0.05 at commit 2e3746163") == []
