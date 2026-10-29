# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The documentation index (docs/index.html + docs/source_of_truth.json) stays true.

The page reads source_of_truth.json when served and falls back to a copy embedded in the page when
opened from disk (a browser cannot read another local file from file://). These tests keep that
copy identical to the JSON and every path the index names pointing at a file that exists, so the
index cannot quietly go stale.
"""
import json
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[4] / "docs"


def _index():
    return json.loads((DOCS / "source_of_truth.json").read_text(encoding="utf-8"))


def _embedded():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="fallback">(.*?)</script>', html, re.S)
    assert m, "index.html lost its embedded copy of source_of_truth.json"
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_the_page_copy_equals_the_json():
    assert _embedded() == _index(), (
        "docs/index.html's embedded copy differs from docs/source_of_truth.json; regenerate the "
        "page's fallback block from the JSON")


def test_every_master_names_one_file_and_a_known_state():
    d = _index()
    questions = [m["question"] for m in d["masters"]]
    assert len(questions) == len(set(questions)), "a question appears twice"
    for m in d["masters"]:
        assert m["state"] in ("current", "behind", "pending"), m
        assert m["file"] and m["note"], m


def test_every_named_path_exists_unless_it_is_not_written_yet():
    d = _index()
    missing = []
    for m in d["masters"]:
        if m["state"] != "pending" and not (DOCS / m["file"]).exists():
            missing.append(m["file"])
        for a in m.get("also", []):
            if " " not in a and not (DOCS / a).exists():
                missing.append(a)
    for h in d["history"]:
        for f in h["files"]:
            if not (DOCS / f).exists():
                missing.append(f)
    assert not missing, "the index names files that do not exist: %s" % missing


def test_no_banned_wording():
    text = (DOCS / "index.html").read_text(encoding="utf-8").lower()
    for word in ("honest", "truthful"):
        assert word not in text, word
