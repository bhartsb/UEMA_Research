# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A harness that trains LemLite with self-state learning builds its worlds with the body-function producer
(2026-09-27): the depth gates' and the stored-token drift's identity harnesses built them without it, and their first
step stopped ("self perception requires the body-function producer"), which cost the motor gate its pause. This reads
every script under docs/gain_of_function_v4 as text: a file that builds LemLite with self_state "learn" passes
body_function in each literal SingleEnvSource or PoolSource call (a call that expands a keyword dictionary, **kw, is
taken to carry the trainer's settings, as train_mimo.env_kwargs_for writes them)."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def _calls(text, name):
    """The argument text of each call `name(...)` in `text`, balanced on parentheses."""
    out = []
    for m in re.finditer(r"\b%s\(" % name, text):
        depth, i = 1, m.end()
        while depth and i < len(text):
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        out.append(text[m.end():i - 1])
    return out


def worlds_without_body_function(text):
    """The world builders a script calls without the body-function producer, when it trains self-state learning."""
    if not any(re.search(r"self_state\s*=\s*['\"]learn['\"]", args) for args in _calls(text, "LemLite")):
        return []
    return ["%s(%s)" % (name, " ".join(args.split())[:120]) for name in ("SingleEnvSource", "PoolSource")
            for args in _calls(text, name) if "body_function" not in args and "**" not in args]


def test_the_check_finds_the_harness_that_stopped_and_passes_its_repair():
    stopped = ('m = LemLite(device="cpu", seed=0, persistent_memory="on", self_state="learn",\n'
               '            overrides={"model": {"residual_prediction": True}})\n'
               'srcs = [SingleEnvSource(scene="reach", run_dir=str(d) + "_w%d" % b, vision=False,\n'
               '                        seed=5 + b) for b in range(a.bodies)]\n')
    assert worlds_without_body_function(stopped)
    assert not worlds_without_body_function(stopped.replace("seed=5 + b)", 'seed=5 + b, body_function="modulate")'))
    assert not worlds_without_body_function(stopped.replace('self_state="learn"', 'self_state="off"'))
    assert not worlds_without_body_function(stopped.replace('vision=False,', '**kw,'))


def test_every_harness_with_self_state_learning_builds_worlds_with_the_body_function_producer():
    missing = []
    for path in sorted((REPO / "docs" / "gain_of_function_v4").rglob("*.py")):
        for call in worlds_without_body_function(path.read_text(encoding="utf-8", errors="replace")):
            missing.append("%s: %s" % (path.relative_to(REPO), call))
    assert not missing, missing
