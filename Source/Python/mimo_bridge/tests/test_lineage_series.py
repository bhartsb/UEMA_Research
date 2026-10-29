# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""tools/lineage_series.py: the lineage's gauges, read incrementally.

Forks are dropped at each child's resume step, each stats file is read from where the last read
stopped, and a field a record does not carry is unmeasured (None), never 0.
"""
import json
from pathlib import Path

import pytest

from mimo_bridge.tools import lineage_series as ls
from mimo_bridge.tools import development_dashboard as dd


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rec(step, **extra):
    row = {"type": "mimo", "step": step, "timestamp": 1_790_000_000 + step, "model_steps_per_s": 1.0}
    row.update(extra)
    return row


def audit(vis=0.001, tac=0.002, pro=-0.001, pers=0.05):
    return {"modalities": {m: {"improvement_over_shuffled": v, "improvement_over_persistence": pers}
                           for m, v in (("vision", vis), ("tactile", tac), ("proprioception", pro))}}


def stats(run, rows, launch=None):
    run.mkdir(parents=True, exist_ok=True)
    lines = []
    if launch:
        lines.append(json.dumps(dict({"type": "launch"}, **launch)))
    lines += [json.dumps(r) for r in rows]
    (run / "replay_training_stats.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def lineage_fixture(root):
    saved = root / "Saved/mimo_bridge"
    parent = saved / "experiments/H1"
    child = saved / "experiments/H1_seg2"
    stray = saved / "experiments/H1_seg2_resume_fork"
    write(parent / "args.json", {"n_envs": 8, "toy_hand_off": None})
    write(child / "args.json", {"n_envs": 8, "toy_hand_off": "follow",
                                "resume": str(parent / "checkpoints/ckpt_step_300")})
    write(stray / "args.json", {"resume": str(parent / "checkpoints/ckpt_step_300")})
    # the parent ran on to step 500 after the child's checkpoint: 400 and 500 are a discarded future
    stats(parent, [rec(s, prediction_audit=audit()) for s in (100, 200, 300, 400, 500)],
          launch={"step": 0, "code_commit": "aaaa1111", "timestamp": 1_790_000_000})
    stats(child, [rec(s, prediction_audit=audit(vis=0.004)) for s in (400, 500, 600)],
          launch={"step": 300, "code_commit": "bbbb2222", "timestamp": 1_790_000_300})
    stats(stray, [rec(s, prediction_audit=audit(vis=9.0)) for s in (400, 500)])
    write(saved / "training_continuation.json", {"active_run": str(child), "timestamp": 1_790_000_700,
                                                 "status": "training"})
    return saved, parent, child, stray


def test_forks_are_dropped_and_off_chain_runs_are_never_read(tmp_path):
    saved, parent, child, stray = lineage_fixture(tmp_path)
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys=["act.vision"], now=1_790_001_000)
    assert out["read"]["files"] == 2, "only the chain's two runs are read"
    assert out["read"]["dropped_fork_records"] == 2, "the parent's steps 400 and 500 are dropped"
    steps = out["x"]["step"]
    assert steps == [100, 200, 300, 400, 500, 600]
    # 400 and 500 come from the child (vision 0.004), never from the parent or the fork (9.0)
    values = dict(zip(steps, out["series"]["act.vision"]))
    assert values[300] == 0.001 and values[400] == 0.004 and values[500] == 0.004
    assert 9.0 not in out["series"]["act.vision"]
    assert out["x"]["segment"] == [0, 0, 0, 1, 1, 1]


def test_segments_carry_commit_and_the_changed_launch_settings(tmp_path):
    lineage_fixture(tmp_path)
    segs = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys=["act.vision"], now=1)["segments"]
    assert [s["run"] for s in segs] == ["H1", "H1_seg2"]
    assert segs[0]["code_commit"] == "aaaa1111" and segs[0]["fresh"] is True
    assert segs[1]["code_commit"] == "bbbb2222" and segs[1]["label"] == "seg2"
    # the values IN FORCE are compared, and the row says what the resumed worlds did with an env
    # setting (reviews/dashboards.md M8)
    ch = segs[1]["changed"]
    assert [(c["flag"], c["was"], c["now"]) for c in ch] == [("--toy-hand-off", "not set", "follow")]
    assert ch[0]["text"].startswith("--toy-hand-off follow (was --toy-hand-off not set)")
    assert segs[1]["flags"]["causal_chain"] is None and "uncommitted changes not recorded" in segs[1]["code"]
    assert (segs[1]["first_step"], segs[1]["last_step"], segs[1]["records"]) == (400, 600, 3)


def test_reads_are_incremental_and_a_partial_line_waits(tmp_path):
    run = tmp_path / "run"
    stats(run, [rec(100, prediction_audit=audit()), rec(200, prediction_audit=audit())])
    path = run / "replay_training_stats.jsonl"
    size = path.stat().st_size
    rows, _, first = ls.read_rows(path, tmp_path / "cache")
    assert len(rows) == 2 and first["bytes_read"] == size
    rows, _, again = ls.read_rows(path, tmp_path / "cache")
    assert again["bytes_read"] == 0 and again["from_cache"]
    ls._MEMO.clear()                      # a new process: the cache file alone must serve
    rows, _, fresh = ls.read_rows(path, tmp_path / "cache")
    assert len(rows) == 2 and fresh["bytes_read"] == 0
    added = json.dumps(rec(300, prediction_audit=audit())) + "\n"
    partial = json.dumps(rec(400))[:40]
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(added + partial)
    rows, _, grown = ls.read_rows(path, tmp_path / "cache")
    assert [r[0] for r in rows] == [100, 200, 300]
    assert grown["bytes_read"] == len(added.encode("utf-8")), "only the new complete line is read"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(rec(400))[40:] + "\n")
    rows, _, done = ls.read_rows(path, tmp_path / "cache")
    assert [r[0] for r in rows] == [100, 200, 300, 400]


def test_a_rewritten_file_is_read_again_from_the_start(tmp_path):
    run = tmp_path / "run"
    stats(run, [rec(100, prediction_audit=audit()), rec(200, prediction_audit=audit())])
    path = run / "replay_training_stats.jsonl"
    ls.read_rows(path, tmp_path / "cache")
    stats(run, [rec(7, prediction_audit=audit(vis=0.5)), rec(8, prediction_audit=audit(vis=0.5)),
                rec(9, prediction_audit=audit(vis=0.5))])
    rows, _, st = ls.read_rows(path, tmp_path / "cache")
    assert [r[0] for r in rows] == [7, 8, 9] and st["restarted"]


def test_a_missing_field_is_unmeasured_never_zero(tmp_path):
    saved = tmp_path / "Saved/mimo_bridge"
    run = saved / "experiments/R"
    write(run / "args.json", {})
    stats(run, [rec(100), rec(200, prediction_audit=audit()), rec(300)])
    write(saved / "training_continuation.json", {"active_run": str(run), "timestamp": 1, "status": "training"})
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys=["act.vision", "cc.chosen_best", "posture.mean"], now=1)
    assert out["series"]["act.vision"] == [None, 0.001, None]
    assert out["series"]["cc.chosen_best"] is None, "a block no record carries is null, not zeros"
    assert out["series"]["posture.mean"] is None
    assert out["defs"]["act.vision"]["latest"]["step"] == 200
    assert out["defs"]["cc.chosen_best"]["latest"] is None
    assert "get" not in out["defs"]["act.vision"], "no function leaks into the JSON"
    json.dumps(out, allow_nan=False)


def test_posture_reads_body_orientation_roll(tmp_path):
    r = rec(1, body_orientation=[{"valid": True, "roll_degrees": -10.0}, {"valid": True, "roll_degrees": 175.0},
                                 {"valid": False, "roll_degrees": 90.0}], rolled_over_fraction=0.75)
    assert ls.SERIES["posture.mean"]["get"](r) == pytest.approx(92.5)
    assert ls.SERIES["posture.max"]["get"](r) == 175.0
    assert ls.SERIES["posture.mean"]["get"](rec(1)) is None


def test_downsampling_never_merges_two_segments(tmp_path):
    lineage_fixture(tmp_path)
    out = ls.build(tmp_path, cache_dir=tmp_path / "cache", keys=["act.vision"], max_buckets=1, now=1)
    assert out["x"]["segment"] == [0, 1]
    assert out["x"]["step"] == [300, 600]
    assert out["series"]["act.vision"] == [0.001, 0.004]
    assert out['x']['start_step'] == [100, 400]
    assert out['x']['counts']['act.vision'] == [3, 3]
    assert out['x']['aggregation'] == 'mean of available report windows'


def probe_line(item, run, step, value, goal=None, checkpoint=None, probe="gaze_counterfactual"):
    return json.dumps({"probe": probe, "item": item, "run": run, "checkpoint": checkpoint or f"x/{run}/checkpoints/ckpt_step_{step}",
                       "lineage_step": step, "time": 1, "path": "p.json",
                       "headline": {"metric": "shift agreement", "value": value, "unit": "", "floor": 0.0,
                                    "goal": goal, "distance_to_goal": None, "lived": "0 = the view does not move"}})


def test_chart_averages_expose_their_interval_and_count_without_inventing_missing_values():
    index = list(ls.SERIES).index('act.vision')
    rows = []
    for step, value in ((100, 1.0), (200, None), (300, 3.0)):
        values = [None] * len(ls.SERIES)
        values[index] = value
        rows.append((step, step + 1000, values))
    chain = [dict(index=0, run='example', rows=rows)]
    axis, values = ls._bucketed(chain, ['act.vision', 'act.tactile'], 1)
    assert axis['start_step'] == [100] and axis['step'] == [300]
    assert axis['counts']['act.vision'] == [2]
    assert values['act.vision'] == [2.0] and values['act.tactile'] is None
    assert ls._latest(chain, 'act.vision')['value'] == 3.0


def test_probe_results_keep_only_the_ancestry(tmp_path):
    saved, parent, child, stray = lineage_fixture(tmp_path)
    index = saved / "gof_probes/index.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text("\n".join([probe_line(7, "H1", 200, 0.1), probe_line(7, "H1", 500, 0.9),
                                probe_line(7, "H1_seg2", 600, 0.3), probe_line(7, "H1_seg2_resume_fork", 500, 0.8),
                                "{broken"]) + "\n", encoding="utf-8")
    training = dd.training_state(tmp_path, 1)
    probes = ls.probe_results(tmp_path, dd.ancestry_bounds(tmp_path, training))
    entry = probes["items"]["7"][0]
    assert entry["points"] == [[200, 0.1], [600, 0.3]], "the parent's step 500 and the fork are not the lineage"
    assert entry["latest"]["value"] == 0.3 and probes["other_lineage"] == 2
    assert ls.probe_results(tmp_path / "nowhere", {})["index_found"] is False


def test_the_ten_items_with_absent_blocks_say_unmeasured_and_why():
    record = rec(1, prediction_audit=audit(), cmd_response_cos_held=0.15,
                 body_orientation=[{"valid": True, "roll_degrees": 3.0}])
    panel = ls.gain_of_function(record, {"causal_chain": "off"}, {"items": {}, "index_found": False})
    assert panel["total"] == 10 and panel["measured"] == 0
    first = panel["items"][0]
    assert first["measured"] is False and first["value"] is None
    assert "--causal-chain is off" in first["unmeasured_because"]
    second = panel["items"][1]
    assert "absent from this run's launch settings" in second["unmeasured_because"]
    # every number in a percentage between two named ends (DESIGN.md section 11)
    assert "vision 0.1%" in first["related"] and "no validated success target" in first["related"]
    assert "physics alone" in second["related"] and "15% of the way" in second["related"],         "the physics-alone reference is the goal line of following"
    assert "gof_probes/index.jsonl" in panel["items"][6]["unmeasured_because"]


def test_the_ten_items_with_present_blocks_are_measured_with_ends_goal_and_distance():
    """Rewritten 2026-09-23 (fix-dashboards) to the new truth: item 5 is measured by how well the
    body follows synergy-written orders, never by the map's reconstruction (reviews/dashboards.md
    M5), and item 6 by self-touch onsets against a control, never by the any-contact score that
    counts persisting contacts (B1, reviews/b4 M3); a record carrying only those older fields leaves
    both items unmeasured and says which field it lacks."""
    record = rec(1, causal_chain={"chosen_best_share": 0.6, "alternatives_n": 4},
                 controllability={"agreement_mean": 0.2, "control_mean": 0.01},
                 learning_progress={"value_mean": 0.003}, wellness_rank={"mode": "normalized",
                 "winner_margin_share": {"integrity": 0.2, "coherence": 0.7}},
                 synergy={"explained_share": 0.4}, contact_control={"head_auc": 0.71},
                 vestibular={"improvement_over_persistence": 0.1})
    probes = {"index_found": True, "items": {"8": [{"probe": "body_counterfactual", "points": [[5, 0.2]],
              "latest": {"value": 0.2, "unit": "cm", "floor": 0.0, "goal": None, "distance_to_goal": None,
                         "lived": "0 = orders predict the same movement", "lineage_step": 5}}]}}
    panel = ls.gain_of_function(record, {"causal_chain": "on"}, probes)
    by = {row["item"]: row for row in panel["items"]}
    assert panel["measured"] == 6
    one = by[1]
    assert one["measured"] and one["value"] == 0.6 and one["big"] == "60 in 100 plans"
    assert "No validated success target: the planner does not aim at predicted wellness" in ls.item_text(one)
    assert "0% = never" not in ls.item_text(one) and "short of the goal line" not in ls.item_text(one)
    assert by[4]["value"] == 0.7 and "No success threshold has been validated" in ls.item_text(by[4])
    assert not by[5]["measured"] and "synergy.following" in by[5]["unmeasured_because"]
    assert not by[6]["measured"] and "onset_auc_self" in by[6]["unmeasured_because"]
    assert by[8]["measured"] and by[8]["source"] == "probe index: body_counterfactual"
    assert by[8]["probes"][0]["value"] == 0.2
    assert not by[7]["measured"] and not by[9]["measured"]


def test_the_headline_counts_items_and_quotes_action_use_with_its_scale():
    """Rewritten 2026-09-23 (fix-dashboards): the abilities lead (reviews/dashboards.md M6), and one
    report window is never set against a line (single windows are noisy)."""
    panel = ls.gain_of_function(rec(1), {}, {})
    head = ls.headline(panel, rec(1, prediction_audit=audit(vis=0.0005, tac=0.0009, pro=-0.0005)))
    assert head["title"].startswith("No gain-of-function item is shown working yet")
    assert "None of the 10 items has a direct measurement" in head["detail"]
    assert "0.09% of the way to exact (tactile)" in head["detail"] and "too noisy" in head["detail"]
    assert "unmeasured" in ls.headline(panel, {})["detail"]


def test_the_step_rate_stop_reason_reads_in_plain_words():
    text = ls.plain_stop_reason("model steps per second 0.397 under one third of own hour-1 baseline 1.249 "
                                "two hourly checks running (sustained step-time blow-up)")
    assert "0.397 model steps per second" in text and "1.249" in text and "0.416" in text
    assert ls.plain_stop_reason("declared criterion 'x' breached") == "\"declared criterion 'x' breached\""


def test_a_stop_is_named_after_the_program_that_wrote_it():
    """The motor fixes kill-line watcher writes a 'writer' field; a stop without one is the hourly
    watcher's (the motor fixes deploy branch's review, point 10: every stop was named the hourly
    watcher's)."""
    assert ls.stop_writer({}).startswith("the hourly watcher program")
    assert ls.stop_writer(None).startswith("the hourly watcher program")
    motor = ls.stop_writer({"writer": "docs/gain_of_function_v4/motor_fixes_watch.py"})
    assert motor.startswith("the motor fixes kill-line watcher") and "hourly" not in motor
    assert ls.stop_writer({"writer": "some_other_program.py"}) == "the program some_other_program.py"
    # the trainer's own stop after repeated physical faults names itself (the second review, point 7)
    assert ls.stop_writer({"writer": "mimo_bridge/train_mimo.py: repeated physical faults"}).startswith(
        "the trainer itself")
    src = (Path(ls.__file__).resolve().parents[1] / "train_mimo.py").read_text(encoding="utf-8")
    i = src.index("Path(out,'stop_reason.json').write_text(")
    assert "writer='mimo_bridge/train_mimo.py: repeated physical faults'" in src[i:i + 400]


def test_every_series_definition_names_its_scale_and_goal_state():
    for key, spec in ls.SERIES.items():
        assert spec["title"] and len(spec["lo"]) == 2 and spec["lo"][1], key
        assert spec["goal"] is None or (len(spec["goal"]) == 2 and spec["goal"][1]), key
        assert spec["better"] in ("higher", "lower"), key
    for card in ls.CARDS:
        for panel in card["panels"]:
            assert all(k in ls.SERIES for k in panel["keys"]), card["id"]
            assert len(panel["keys"]) <= 3, "at most three lines per chart (palette validated for three)"
    assert {c["item"] for c in ls.CARDS if c["item"]} == set(range(1, 11))


# ------------------------------------------------------------------ the cache's code-version tag
# verify-fix-dashboards, open issue (2026-09-23): the cache file names carried no code version, so a
# server on older code and a newer build overwrote each other's cache under one name, and a changed
# series definition kept serving rows the old code had extracted (the signature named only the keys).
def _one_file(tmp_path):
    run = tmp_path / "run"
    stats(run, [rec(100, prediction_audit=audit()), rec(200, prediction_audit=audit())])
    return run / "replay_training_stats.jsonl"


def test_a_cache_written_by_other_code_is_never_read_and_each_version_keeps_its_own(tmp_path, monkeypatch):
    path = _one_file(tmp_path)
    size, cache = path.stat().st_size, tmp_path / "cache"
    ls._MEMO.clear()
    _, _, first = ls.read_rows(path, cache)
    assert first["bytes_read"] == size
    tag = ls.cache_tag(ls.signature())
    files = [p.name for p in cache.glob("f_*.json")]
    assert len(files) == 1 and files[0].endswith("_" + tag + ".json"), files
    assert json.loads((cache / files[0]).read_text(encoding="utf-8"))["tag"] == tag
    # the same keys under another source of lineage_series.py: a cache of its own, read from the start
    monkeypatch.setattr(ls, "SOURCE_TAG", "other-code-000000")
    ls._MEMO.clear()
    rows, _, other = ls.read_rows(path, cache)
    assert other["bytes_read"] == size and not other["from_cache"] and [r[0] for r in rows] == [100, 200]
    assert len(list(cache.glob("f_*.json"))) == 2, "two code versions keep two files"
    # back on the first code, in a new process: its own file serves, nothing is re-read
    monkeypatch.undo()
    ls._MEMO.clear()
    _, _, again = ls.read_rows(path, cache)
    assert again["bytes_read"] == 0 and again["from_cache"]
    # a file that carries this code's name but another tag in its contents is refused
    mine = cache / files[0]
    forged = dict(json.loads(mine.read_text(encoding="utf-8")), tag="copied-from-other-code")
    mine.write_text(json.dumps(forged), encoding="utf-8")
    ls._MEMO.clear()
    _, _, refused = ls.read_rows(path, cache)
    assert refused["bytes_read"] == size and not refused["from_cache"]


def test_a_third_code_version_leaves_only_the_newest_other_cache(tmp_path, monkeypatch):
    """Each code version would otherwise leave one file per segment behind; the newest other
    version's file stays (a running server on older code), older ones and the untagged file of
    the code before tags go."""
    import os
    path = _one_file(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    untagged = ls._cache_file(cache, path, "x").name.rsplit("_", 1)[0] + ".json"
    (cache / untagged).write_text("{}", encoding="utf-8")
    os.utime(cache / untagged, (1_000, 1_000))
    names = []
    for i, source in enumerate(("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc")):
        monkeypatch.setattr(ls, "SOURCE_TAG", source)
        ls._MEMO.clear()
        ls.read_rows(path, cache)
        names.append(ls._cache_file(cache, path, ls.cache_tag(ls.signature())).name)
        os.utime(cache / names[-1], (2_000 + i, 2_000 + i))
    left = sorted(p.name for p in cache.glob("f_*.json"))
    assert left == sorted(names[1:]), left


def test_the_tag_follows_the_source_and_every_page_cache_carries_one(tmp_path, monkeypatch):
    src = tmp_path / "m.py"
    src.write_text("A = 1\n", encoding="utf-8")
    one = ls.file_tag(src)
    src.write_text("A = 2\n", encoding="utf-8")
    assert len(one) == 16 and ls.file_tag(src) != one
    assert ls.file_tag(tmp_path / "missing.py") == ""
    assert ls.SOURCE_TAG, "the loaded lineage_series.py is tagged"
    assert ls.cache_tag("s") != ls.cache_tag("t")
    # the old dashboard's compact cache names its extractor's source; the lifetime page its file's
    from mimo_bridge.tools import dashboard, lifetime
    assert dashboard._COMPACT_SIG.endswith(":" + dashboard._source_tag(dashboard._compact))
    assert len(dashboard._source_tag(dashboard._compact)) == 16 and len(lifetime.SOURCE_TAG) == 16
    seen = []

    def spy(path, cache, extractor=None, sig=None):
        seen.append(sig)
        return [], {}, {}
    monkeypatch.setattr(ls, "read_rows", spy)
    run = tmp_path / "Saved/mimo_bridge/experiments/R"
    stats(run, [rec(1)])
    lifetime.sweep({str(run): None})
    assert seen and seen[0].endswith(":" + lifetime.SOURCE_TAG), seen
