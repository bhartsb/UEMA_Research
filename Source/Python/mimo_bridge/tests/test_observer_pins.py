# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Evidence pins (tools/observer/pins.py; the operator's instruction of 7:37 AM PDT, Oct 8, 2026): the raw records of open
investigations are never deleted by the digester's housekeeping; and the independent review of 4342f9194 (9:35 AM PDT,
Oct 8, 2026, fix 1): every doubt keeps records.

Each test fails on the code before the change, which deleted every digested chunk past 72 hours and read no pin list,
or on 4342f9194, which released the chunks of an invalid pin, bounded a chunk by its neighbour's name and kept the
learner lane's chunks by the world's step numbers:
  * world chunks of the pinned bodies and learner-lane chunks reaching into a pinned step range are kept; others go;
  * an expired pin is reported and keeps protecting until it is removed;
  * an unreadable pin file stops every deletion of the pass and says why; a byte-order mark is read;
  * an invalid pin naming a run stops that run's deletion and is reported; one naming no usable run stops all;
  * a chunk's range is read from its own step column, so a process id given to a second process shrinks nothing;
  * a lane chunk is kept by the lane's own numbering: one step behind the world, and up to the efference record of the
    command in force at the pin's last step;
  * a pin file read before and missing now keeps the pins last read.
"""
import json
import os
import time

import numpy as np

from mimo_bridge.tools.observer import daemon as DM
from mimo_bridge.tools.observer import pins as PN

OLD = time.time() - 4 * 24 * 3600          # four days: past the 72 hours of daemon.RAW_KEEP_S


def _chunk(folder, first, pid, seq, n=100):
    path = os.path.join(folder, "chunk_%09d_%d_%05d.npz" % (first, pid, seq))
    np.savez(path, step=np.arange(first, first + n))
    os.utime(path, (OLD, OLD))
    return os.path.basename(path)


def _run(tmp_path, name="H99_seg01"):
    """A run folder: body 0 and body 1 with three world chunks each (steps 1000-1299), a learner lane with three
    chunks, a causal chain, every chunk digested and four days old, the run sealed."""
    run = tmp_path / name
    names = {}
    for body in (0, 1):
        folder = run / "observer" / "world" / ("b%d" % body)
        folder.mkdir(parents=True)
        (folder / "header_11_1000.json").write_text(json.dumps(dict(budget_bytes=10 ** 12)), encoding="utf-8")
        names[body] = [_chunk(str(folder), 1000 + 100 * k, 11 + body, k) for k in range(3)]
    lane = run / "observer" / "model"
    lane.mkdir(parents=True)
    names["model"] = [_chunk(str(lane), 1000 + 100 * k, 99, k) for k in range(3)]
    (run / "causal_chain.jsonl").write_text("", encoding="utf-8")
    dg = DM.RunDigest(str(run))
    for body in (0, 1):
        dg.state["bodies"][str(body)] = dict(done={n: dict(pid=11 + body) for n in names[body]},
                                              final={str(11 + body): 1299})
    dg.state["sealed"] = time.time()
    return run, dg, names


def _one_body_run(tmp_path, name, world, lane):
    """A sealed run with body 0's world chunks and lane chunks given as (first step, process id, sequence)."""
    run = tmp_path / name
    b0 = run / "observer" / "world" / "b0"
    model = run / "observer" / "model"
    b0.mkdir(parents=True)
    model.mkdir(parents=True)
    (b0 / "header_11_1000.json").write_text(json.dumps(dict(budget_bytes=10 ** 12)), encoding="utf-8")
    names = [_chunk(str(b0), *c) for c in world]
    for c in lane:
        _chunk(str(model), *c)
    dg = DM.RunDigest(str(run))
    dg.state["bodies"]["0"] = dict(done={n: dict(pid=0) for n in names}, final={"0": 10 ** 6})
    dg.state["sealed"] = time.time()
    return run, dg, b0, model


def _left(folder):
    return sorted(n for n in os.listdir(folder) if n.startswith("chunk_"))


def test_pinned_chunks_survive_housekeeping_past_the_keep_time(tmp_path):
    run, dg, names = _run(tmp_path)
    pins = PN.Pins(entries=[dict(id="finger_v2", run=os.path.basename(str(run)), steps=[1150, 1160], bodies=[1],
                                 reason="first-finger investigation")])
    rep = dg.housekeeping(pins=pins)
    assert _left(run / "observer" / "world" / "b0") == [], "body 0 is not pinned: its old chunks go"
    assert _left(run / "observer" / "world" / "b1") == [names[1][1]], "the chunk holding steps 1100-1199 stays"
    assert _left(run / "observer" / "model") == [names["model"][1]], "the lane chunk of the range stays"
    assert rep["pins"] == ["finger_v2"] and rep["pinned_kept"] == dict(world={"1": 1}, model=1)
    assert rep["command_history"] is True and rep["blocked"] is None and rep["removed"] == 7


def test_without_pins_housekeeping_removes_as_before(tmp_path):
    run, dg, _ = _run(tmp_path)
    rep = dg.housekeeping()
    assert rep["removed"] == 9 and rep["pins"] == []
    assert _left(run / "observer" / "world" / "b1") == [] and _left(run / "observer" / "model") == []


def test_an_expired_pin_is_reported_and_still_protects(tmp_path):
    run, dg, names = _run(tmp_path)
    pins = PN.Pins(entries=[dict(id="old", run=str(run), steps=[1000, 1050], until="2026-10-01T00:00:00-07:00")])
    summary = pins.summary(known_runs=[str(run)])
    assert summary["expired"] == ["old"] and summary["pins"][0]["state"].startswith("EXPIRED")
    dg.housekeeping(pins=pins)
    assert _left(run / "observer" / "world" / "b0") == [names[0][0]], "expiry never releases evidence"


def test_an_unreadable_pin_file_stops_every_deletion(tmp_path):
    run, dg, _ = _run(tmp_path)
    path = tmp_path / "pins.json"
    path.write_text("{not json", encoding="utf-8")
    pins = PN.Pins(str(path))
    rep = dg.housekeeping(pins=pins)
    assert pins.blocks_all and rep["removed"] == 0 and "cannot be read" in rep["blocked"]
    assert len(_left(run / "observer" / "world" / "b0")) == 3 and len(_left(run / "observer" / "model")) == 3
    assert PN.Pins(str(tmp_path / "missing.json")).pins == [] and not PN.Pins(str(tmp_path / "missing.json")).blocks_all


def test_a_pin_file_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(dict(pins=[dict(id="a", run="H99_seg01", steps=[1, 2])])), encoding="utf-8-sig")
    pins = PN.Pins(str(path))
    assert pins.error is None and [p["id"] for p in pins.pins] == ["a"]


def test_an_invalid_pin_naming_a_run_stops_that_runs_deletion_and_is_reported(tmp_path):
    """Review fix 1: a pin whose 'until' lacks its UTC offset, a second pin with the same id and steps written as text
    each keep their run's records instead of releasing them."""
    run, dg, _ = _run(tmp_path)
    other, dg_other, _ = _run(tmp_path, "H99_seg02")
    for bad in (dict(id="no_offset", run=os.path.basename(str(run)), steps=[1150, 1160], until="2026-11-08"),
                dict(id="text_steps", run=os.path.basename(str(run)), steps=["1150", "1160"])):
        pins = PN.Pins(entries=[bad])
        rep = dg.housekeeping(pins=pins)
        assert rep["removed"] == 0 and "invalid pin %s" % bad["id"] in rep["blocked"]
        assert len(_left(run / "observer" / "world" / "b0")) == 3 and len(_left(run / "observer" / "model")) == 3
        assert pins.summary(known_runs=[str(run)])["blocked_runs"][os.path.basename(str(run))]
    twice = PN.Pins(entries=[dict(id="a", run=os.path.basename(str(run)), steps=[1150, 1160]),
                             dict(id="a", run=os.path.basename(str(run)), steps=[1200, 1210])])
    assert [p["id"] for p in twice.pins] == ["a"] and twice.blocks_run(str(run))
    # a run no invalid pin names is housekept as before
    assert dg_other.housekeeping(pins=PN.Pins(entries=[bad]))["removed"] == 9


def test_an_invalid_pin_naming_no_usable_run_stops_every_deletion(tmp_path):
    run, dg, _ = _run(tmp_path)
    pins = PN.Pins(entries=[dict(id="no_run", steps=[1150, 1160])])
    rep = dg.housekeeping(pins=pins)
    assert pins.blocks_all and rep["removed"] == 0 and "no usable run" in rep["blocked"]


def test_a_malformed_pin_is_reported_and_the_others_stand(tmp_path):
    run, _, _ = _run(tmp_path)
    pins = PN.Pins(entries=[dict(id="good", run=str(run), steps=[1000, 1010]),
                            dict(id="bad", run=str(run), steps=[1010, 1000]),
                            dict(id="no_offset", run=str(run), steps=[1, 2], until="2026-11-01T00:00:00"),
                            dict(id="elsewhere", run="H99_seg77", steps=[1, 2])])
    s = pins.summary(known_runs=[str(run)])
    assert [p["id"] for p in pins.pins] == ["good", "elsewhere"]
    assert sorted(i["entry"]["id"] for i in s["invalid"]) == ["bad", "no_offset"]
    assert s["unknown_runs"] == ["elsewhere"] and len(pins.blocks_run(str(run))) == 2


def test_a_chunk_range_is_read_from_its_own_steps_when_a_process_id_was_given_twice(tmp_path):
    """Review fix 1: process A (1,000-1,199) and a later process B (483,700-483,899) with the same process id in one
    body folder; 4342f9194 gave B's first chunk the range 483,700-483,700 and removed it."""
    run, dg, b0, _ = _one_body_run(tmp_path, "H99_pidreuse", [(1000, 5, 0), (1100, 5, 1), (483700, 5, 0),
                                                              (483800, 5, 1, 37)], [])
    r = PN.chunk_ranges(str(b0))
    assert r == {"chunk_000001000_5_00000.npz": (1000, 1099), "chunk_000001100_5_00001.npz": (1100, 1199),
                 "chunk_000483700_5_00000.npz": (483700, 483799), "chunk_000483800_5_00001.npz": (483800, 483836)}
    rep = dg.housekeeping(pins=PN.Pins(entries=[dict(id="episode", run=os.path.basename(str(run)),
                                                     steps=[483773, 483822])]))
    assert _left(b0) == ["chunk_000483700_5_00000.npz", "chunk_000483800_5_00001.npz"] and rep["removed"] == 2
    assert PN.pinned_names([dict(id="a", steps=[1136, 1200])], r) == {"chunk_000001100_5_00001.npz": ["a"]}


def test_the_lane_chunk_holding_the_pins_last_step_is_kept(tmp_path):
    """Review fix 1: lane step t holds the learner's records of world step t - 1, so the lane chunk starting at 483,823
    holds world step 483,822's; the efference record of the command in force there comes up to a hold later."""
    run, dg, _, model = _one_body_run(tmp_path, "H99_lanelag", [(483723, 3, 0)],
                                      [(483723, 9, 0), (483823, 9, 1), (483923, 9, 2)])
    rep = dg.housekeeping(pins=PN.Pins(entries=[dict(id="episode", run=os.path.basename(str(run)),
                                                     steps=[483773, 483822])]))
    assert _left(model) == ["chunk_000483723_9_00000.npz", "chunk_000483823_9_00001.npz"]
    assert rep["pinned_kept"]["model"] == 2


def test_a_pin_file_read_before_and_missing_now_keeps_its_pins(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(dict(pins=[dict(id="a", run="H99_seg01", steps=[1, 2])])), encoding="utf-8")
    first = PN.Pins(str(path))
    path.unlink()
    second = PN.Pins(str(path), previous=first)
    assert [p["id"] for p in second.pins] == ["a"] and second.missing_file
    assert "stand until a file with an empty list is written" in second.summary()["missing_file"]
    path.write_text(json.dumps(dict(pins=[])), encoding="utf-8")
    assert PN.Pins(str(path), previous=second).pins == [], "an empty list removes every pin"
