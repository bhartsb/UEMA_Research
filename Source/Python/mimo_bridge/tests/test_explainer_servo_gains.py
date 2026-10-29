# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The explainer's handle entry states the run's own servo stiffness (fix-rows-r1-schema.py,
2026-09-23).

Until then the glossary typed "a separate eye servo 40 times stiffer than the limb servo
(stiffness 80 against 2)" and "a limb needs about 250 ms". 2 is ActionDecoder's class default and
H83's setting; all 50 H84 segments ran --decoder-kp 6 with --gaze-kp 80 (13.3 times), and no file
measures a limb's arrival time at stiffness 6. schema.servo_fill now reads the three settings from
the run's args.json and says so when a launch did not record them."""
import os
import re

from mimo_bridge.explainer import schema as SCH
from mimo_bridge.tools.check_no_bare_decimals import bare_decimals

H84_ARGS = {"decoder_kp": 6.0, "gaze_kp": 80.0, "gaze_kd": 3.0}   # seg31 args.json


def _handle_text(args, run_name):
    fill = SCH.run_fill({}, args, run_name)
    text = SCH.DOC["term.handle"]["w"]
    for key, val in fill.items():                      # the same pass build_explainer.render makes
        text = text.replace(key, val)
    return fill, " ".join(text.split())


def test_the_h84_page_states_h84_s_own_servo_stiffness():
    fill, text = _handle_text(H84_ARGS, "H84")
    assert fill["{eye_limb_kp_ratio}"] == "13.3"
    assert fill["{limb_kp}"] == "6" and fill["{gaze_kp}"] == "80"
    assert "13.3 times stiffer than the limb servo in H84" in text, text
    assert "80 against 6" in text, text
    assert "40 times" not in text and "against 2" not in text, text
    # the limb's arrival time at stiffness 6 is unmeasured, so the page no longer types one
    assert "about 250 ms" not in text
    assert "at stiffness 6 has not been measured" in text, text
    assert "At H84's eye stiffness 80 and damping 3, a sweep measured the eye reaching a new " \
           "target in 130 ms" in text, text
    assert not re.findall(r"\{[a-z][a-z0-9_]*\}", text), text
    assert not bare_decimals(text), bare_decimals(text)


def test_the_number_is_computed_from_the_run_not_typed():
    # H83's setting: the old typed sentence happened to describe this run, and now it is computed
    _, text = _handle_text({"decoder_kp": 2.0, "gaze_kp": 80.0, "gaze_kd": 3.0}, "H83")
    assert "40.0 times stiffer than the limb servo in H83" in text and "80 against 2" in text
    assert "When the limb servo was tuned at stiffness 2, a 28.6 degree shoulder turn" in text
    assert "has not been measured" not in text, text
    # eye gains the sweep did not use: the 130 ms is attributed to its own gains, not this run's
    _, text = _handle_text({"decoder_kp": 6.0, "gaze_kp": 40.0, "gaze_kd": 2.0}, "RX")
    assert "6.7 times stiffer" in text and "40 against 6" in text
    assert "at eye stiffness 80 and damping 3" in text
    assert "at RX's eye stiffness 40 and damping 2 the eye's arrival time has not been " \
           "measured" in text, text


def test_a_launch_that_did_not_record_the_gains_says_so_and_prints_no_other_run_s_number():
    fill, text = _handle_text({}, "R0")
    assert "did not record --decoder-kp or --gaze-kp" in text, text
    assert "train_mimo's default values are 8 for the limbs and 80 for the eye" in text, text
    assert "13.3" not in text and "40 times" not in text and "against 2" not in text
    assert fill["{eye_limb_kp_ratio}"].startswith("not stated")
    assert "did not record --gaze-kp or --gaze-kd" in text
    assert "because its launch did not record --decoder-kp" in text
    assert not re.findall(r"\{[a-z][a-z0-9_]*\}", text), text
    assert not bare_decimals(text), bare_decimals(text)
    # one of the two recorded: the recorded one is printed, the missing one named with its default
    _, text = _handle_text({"gaze_kp": 80.0, "gaze_kd": 3.0}, "R1")
    assert "did not record --decoder-kp, so" in text, text
    assert "train_mimo's default is 8 for the limbs; it recorded --gaze-kp 80" in text, text


def test_the_named_defaults_are_train_mimo_s_own():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_mimo.py")
    src = open(path, encoding="utf-8").read()

    def default(flag):
        m = re.search(r'add_argument\("' + re.escape(flag) + r'",\s*type=float,\s*default=([0-9.]+)', src)
        assert m, f"train_mimo.py no longer declares {flag} as a float with a default"
        return float(m.group(1))
    assert default("--decoder-kp") == SCH.DECODER_KP_DEFAULT
    assert default("--gaze-kp") == SCH.GAZE_KP_DEFAULT
