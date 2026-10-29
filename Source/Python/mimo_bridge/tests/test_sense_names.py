# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""One name per sense on the explainer and the development dashboard: the model registry's own
(vision, tactile, audio, proprioception, motor, intermind, vestibular), defined where each page first
names it. The operator's standing rule (2026-09-19, applied to the senses 2026-09-25): never rename a
part; the code's own word, one word per thing, a plain definition where it is first used. Until
2026-09-25 the dashboard called the senses "balance-sense", "body-position", "movement-order",
"touch", "sound", "seeing" and "hearing", and the explainer "touch", "hearing", "body sense" and "the
vestibular sense", while both pages' diagrams and code paths said tactile, audio, proprioception."""
import json
import os
import re

from mimo_bridge.explainer import schema as SCH
from mimo_bridge.explainer import walkthrough as WT
from mimo_bridge.tools import development_dashboard as DD
from mimo_bridge.tools import lineage_series as LS

REGISTRY = ("vision", "tactile", "audio", "proprioception", "motor", "intermind", "vestibular")
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# every source whose text reaches either page, the two older diagnostic pages the dashboard still
# serves (tools/dashboard.py, tools/lifetime.py; 2026-09-25) and the posture probe, whose result
# files the dashboard prints as the probe's own summary
PAGE_SOURCES = ("explainer/schema.py", "explainer/viewer.js", "explainer/walkthrough.py",
                "explainer/capture.py", "tools/build_explainer.py", "tools/check_explainer_page.py",
                "tools/development_dashboard.py", "tools/development_dashboard.js",
                "tools/development_dashboard.html", "tools/development_dashboard_terms.js",
                "tools/lineage_series.py", "tools/experiment_board.py",
                "tools/review_completed_developmental.py", "tools/training_brief.py",
                "tools/developmental_status.py", "tools/dashboard.py", "tools/lifetime.py",
                "tools/gof_probe_posture.py")
# the second names the pages used for a sense. "touch" stays a word for a contact (a touch that
# begins, self-touch); it is caught only where it named the sense: its token, encoder, rows, input,
# or a place in a list of senses
OLD_NAMES = (r"balance[- ]sense", r"\bbody[- ]sense\b", r"\bbody[- ]position\b", r"\bmovement-order\b",
             r"\bhearing\b", r"\bvestibular sense\b", r"\bseeing reconstruction\b", r"\binter-mind\b",
             r"\bpredict(?:ing|ion)? (?:the next instant of )?(?:SEEING|TOUCH|the BODY-SENSE)\b",
             r"internal-state sense",
             r"(?<!self-)\btouch[ -](?:token|encoder|rows?|sense|channels?|reconstruction|prediction|"
             r"numbers?|input|information)\b",
             r"\b(?:vision|tactile|audio|proprioception|motor), touch\b",
             r"\btouch, (?:vision|audio|proprioception|motor|hearing)\b")
# the one place an old name is DATA, not page text: lineage_series matches the unit text older posture
# probe results carry ('the balance sense'), so those files still read right
DATA_MATCH = re.compile(r"'[^']*' in unit_text")


def _src(rel):
    return open(os.path.join(PKG, rel), encoding="utf-8").read()


def test_every_sense_name_is_the_registry_s():
    assert SCH.SENSE_WORDS == {m: m for m in REGISTRY}
    js = re.search(r"const SENSE_WORD = \{(.*?)\};", _src("explainer/viewer.js"), re.S).group(1)
    assert dict(re.findall(r'(\w+): "([^"]+)"', js)) == {m: m for m in REGISTRY}
    assert all(k == name for k, name in LS.ALL_SENSES) and {k for k, _ in LS.ALL_SENSES} == {
        "vision", "tactile", "proprioception", "audio"}
    board = _src("tools/experiment_board.py")
    assert "(('proprioception','Proprioception'),('tactile','Tactile'),('vision','Vision'))" in board
    for m in ("vestibular", "motor", "intermind"):
        assert DD._part(f"prediction_heads.heads.{m}.attn") == f"{m} prediction head's input selection"
    # one name for every sense's reconstruction decoder (the vestibular one was 'vestibular
    # reconstruction, stage 1' beside 'the vision reconstruction decoder' until 2026-09-25)
    for m in REGISTRY:
        assert DD._part(f"recon_decoders.{m}.0") == f"the {m} reconstruction decoder"
    # the explainer's sense boxes are titled by the name the box itself shows
    for m in REGISTRY:
        assert SCH.DOC[f"node.sense.{m}"]["t"].split(" (")[0] == m, SCH.DOC[f"node.sense.{m}"]["t"]
    # the older diagnostic page: its table and the reading legend that defines each name
    from mimo_bridge.tools import dashboard as LEGACY
    assert LEGACY.SENSE_NAMES == {m: m for m in REGISTRY}
    legend = LEGACY.sense_names_legend()
    for m in REGISTRY:
        assert f"{m} is {SCH.SENSE_DEFINITIONS[m]}".replace("'", "&#x27;") in legend, m
    # the history page defines the two senses its yardsticks name, from the same table
    assert ('SENSE_DEFINITIONS[m]}" for m in ("proprioception", "vestibular")'
            in _src("tools/lifetime.py"))


def test_the_old_sense_names_never_come_back():
    # the control: each name the pages used before 2026-09-25 is caught, and the words that stay (a
    # touch as a contact, the flag, a verb, a plural) are not
    caught = lambda s: any(re.search(p, s, re.I) for p in OLD_NAMES)
    for old in ("balance-sense reconstruction", "the balance sense", "body sense", "body-position",
                "Body position", "movement-order encoder", "hearing's blend", "the vestibular sense",
                "the touch token", "touch rows", "vision, touch, hearing", "touch, vision",
                "the seeing reconstruction decoder", "audio and inter-mind",
                "Predicting the next instant of SEEING", "Balance and internal-state sense"):
        assert caught(old), old
    for kept in ("two self-touch numbers", "a touch that begins", "--vestibular-sense on",
                 "body positions and rotations", "what the body senses", "touch comfort",
                 "Own-body touch foreseen", "the tactile token", "Is it seeing a new picture every moment?",
                 "audio and intermind"):
        assert not caught(kept), kept
    hits = []
    for rel in PAGE_SOURCES:
        for n, line in enumerate(_src(rel).splitlines(), 1):
            text = DATA_MATCH.sub("", line)
            for pat in OLD_NAMES:
                for m in re.finditer(pat, text, re.I):
                    hits.append(f"{rel}:{n}: {m.group(0)!r} in {line.strip()[:110]!r}")
    assert not hits, "a sense is named by another word than the registry's:\n  " + "\n  ".join(hits)


def test_each_sense_is_defined_where_the_pages_first_name_it(tmp_path):
    assert set(SCH.SENSE_DEFINITIONS) == set(REGISTRY)
    # the explainer: the tour's senses step names and defines every sense the run carries
    mods = ["vision", "tactile", "audio", "proprioception", "motor", "vestibular"]
    receipt = {"active_modalities": mods, "forward_model": {"_seq_length": 24, "_n_modalities": 6,
                                                            "n_layers": 2, "n_heads": 8}}
    senses = next(st for st in WT.build_steps(receipt, {}) if st["id"] == "senses")
    for m in mods:
        assert f"{m} is {SCH.SENSE_DEFINITIONS[m]}" in senses["body"], m
    # the dashboard's glossary carries the same definition for each sense
    terms = _src("tools/development_dashboard_terms.js")
    for m in REGISTRY:
        key = "Motor (sense)" if m == "motor" else m.capitalize()
        entry = re.search(r"'%s':'((?:[^'\\]|\\.)*)'" % re.escape(key), terms)
        assert entry, f"the glossary has no entry {key!r}"
        text = entry.group(1).replace("\\'", "'")
        want = SCH.SENSE_DEFINITIONS[m]
        assert text == want[:1].upper() + want[1:] + ".", (key, text)
    # the Model health card defines a sense where it names one of its parts, and only then
    card = DD.liveness_card({"liveness": {"layers": {"recon_decoders.vestibular.0": 0.2,
                                                     "motor_controller.residual_blocks.1.0": 0.9}}},
                            measurements=tmp_path)
    # since 2026-09-30 (the operator: the model alert was "gobbledygook") definitions live in the
    # card's glossary (behind its Definitions link) and in the flagged part's own issue
    shown = json.dumps([card["glossary"], card["issues"]], ensure_ascii=False)
    assert "Vestibular is " + SCH.SENSE_DEFINITIONS["vestibular"] + "." in shown
    assert "Motor is " not in shown, "the motor controller is not the motor sense"


def test_the_posture_probe_names_the_sense_it_scored_and_older_results_still_read():
    """gof_probe_posture.py wrote "the balance sense" and "body position sense, standing in for the
    balance sense" into its results until 2026-09-25; new results name the vestibular or the
    proprioception input with its plain meaning. The dashboard reads both the same way: a vestibular
    reading is a direct one, a proprioception reading a stand-in that is not counted."""
    from mimo_bridge.tools import gof_probe_posture as P
    sense, label = P._token_sense(["vision", "vestibular"])
    assert sense == "vestibular" and label.startswith("vestibular input (the body's load and balance")
    sense, label = P._token_sense(["vision", "proprioception"])
    assert sense == "proprioception" and label.startswith("proprioception input (where each tracked body part is")
    assert "standing in for the vestibular input" in label
    old_vest = ("share of the real change in what the body sensed one moment after a head order (the balance "
                "sense), against holding still, that the prediction foresaw")
    old_prop = ("share of the real change in what the body sensed one moment after a head order (body position "
                "sense, standing in for the balance sense), against holding still, that the prediction foresaw")
    for unit, lived, direct in ((P.headline_unit("vestibular"), P._token_sense(["vestibular"])[1], True),
                                (old_vest, "", True),
                                (P.headline_unit("proprioception"), P._token_sense(["proprioception"])[1], False),
                                (old_prop, "", False)):
        words = LS.probe_words({"metric": "posture_neck_share_50ms", "value": 0.2, "unit": unit,
                                "lived": lived}, "posture")
        assert words["stand_in"] is (not direct), (unit, words)
        assert (words["label"] == "Vestibular prediction test") is direct, (unit, words["label"])


def test_tactile_comfort_is_the_code_s_name_with_its_definition_from_the_wellness_code():
    """The wellness term tactile_comfort (lem/model/wellness.py) was "touch comfort" on the explainer
    and the older diagnostic page until 2026-09-25. One line defines it, from
    lem/model/contact_wellness.py contact_response, and its numbers are held to that code here."""
    import inspect
    from lem.model import contact_wellness as CW
    hits = [f"{rel}:{n}" for rel in PAGE_SOURCES
            for n, line in enumerate(_src(rel).splitlines(), 1)
            if re.search(r"\btouch[ -]comfort\b", line, re.I)]
    assert not hits, f"the wellness term is called by another name than the code's: {hits}"
    # the definition's numbers are the formula's: comfort = (0.5 + 0.35 x pleasant) x (1 - pain), the
    # pleasant hill peaking at TACTILE_PLEASANT_PEAK of the 100 kPa unit
    src = inspect.getsource(CW.contact_response)
    assert "comfort = (0.5 + 0.35 * pleasant) * (1 - pain)" in src
    d = SCH.TACTILE_COMFORT_DEFINITION
    assert "50% without contact or remaining load history" in d and "up to 85%" in d and "the largest skin-patch aversion" in d
    assert f"near {CW.TACTILE_PLEASANT_PEAK * 100:g} kPa" in d
    # the explainer's entry opens with it; the dashboard's glossary carries the same words
    entry = SCH.DOC["term.eval-tactile-comfort"]
    assert entry["t"].startswith("Tactile comfort:") and entry["w"].startswith("Tactile comfort is {tactile_comfort_definition}.") and SCH.run_fill({}, {}, "fixture")["{tactile_comfort_definition}"] == d
    terms = _src("tools/development_dashboard_terms.js")
    glossary = re.search(r"'Tactile comfort':'((?:[^'\\]|\\.)*)'", terms).group(1).replace("\\'", "'")
    assert glossary == d[:1].upper() + d[1:] + "."
