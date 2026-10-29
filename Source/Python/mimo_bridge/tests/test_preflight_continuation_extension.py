# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE EXTENSION OF A STOPPED CONTINUATION (training_preflight.py, the module text; prepared Oct 12, 2026 for the
operator's authorization of the body-1 extension, on its own branch for review).

The world of these tests mirrors record 2c3d82ca4ab4a870 on stand-ins, in pytest's temporary folder: a configuration
whose repair run (start 1,000, bound 500) was launched, completed at 1,500 and its target gate judged again; whose
continuation (from 1,500, bound 1,000, launched to end at 2,500) was used and STOPPED EARLY at 1,800 by a stop line, its
end checkpoint saved. The extension is the span's remainder: 1,800 to at most 2,500. Nothing here touches a real record.

THE ONE EXTENSION (the independent review of Oct 12, 2026, finding B-1): the route admits only the extension
EXTENSION_ROUTE names (production: configuration 2c3d82ca4ab4a870, continuation run B1C_skip13_484675, stopped at
485,975, launched to end at 489,350). The fixture `pin` points EXTENSION_ROUTE at a test world's own continuation for one
test (pytest's monkeypatch restores it); the tests of (6) read the route as written in the code.

What is held:
  (1) without an extension the stopped continuation's end is judged as normal training and refused (today's tool);
  (2) the extension, written in its own file, admits exactly that launch; the record stays byte-identical;
  (3) every condition refuses when it does not hold: the bound past the continuation's end (block and launch), another
      start, the continuation's or the repair run's rows, a row already run, a continuation that reached its end or was
      never used, a gate judged again after the extension was written, an existing latent output, another --resume;
  (4) used once: an attempt that trained past the start uses it, the same command is then normal training and refused,
      the file is not written again; an attempt that never trained may be written again, its attempts carried;
  (5) every other route is unchanged (the continuation's own start after its use is refused as normal training);
  (6) THE ONE EXTENSION (finding B-1): another configuration, another continuation run, another stop step and another end
      are refused when written, at launch and when an attempt is noted; the route names production's one extension;
  (7) ONE TEST PER CONDITION THE REVIEW FOUND UNTESTED (finding B-2; each fails on a copy of the tool with that condition
      deleted: the package's mutation check X_PF_MUT): the end checkpoint's meta.json, stage_done ok true, finished_at,
      stopped after its start, the continuation's end within its own bound, every required field (the operator's words
      first), repeats;
  (8) the latent body state is exactly the continuation's saved latest.pt (finding B-3);
  (9) the operator's words and time are text and not the continuation's (finding B-4).
"""
import json
import time
from pathlib import Path

import pytest

from mimo_bridge.tools import training_preflight as TP

FLAGS = ("--intention", "--resume", "--out", "--report-every", "--max-steps", "--checkpoint-storage",
         "--proprio-token-skip", "--latent-body-output", "--latent-body-resume", "--latent-body-hierarchy")
TRAINER = ("import argparse\n\n\ndef parse_args(argv=None):\n    ap = argparse.ArgumentParser()\n"
           + "".join('    ap.add_argument("%s")\n' % f for f in FLAGS) + "    return ap.parse_args(argv)\n")
STOP_CRITERIA = [dict(name="vision_causal_error", path="causal_pe_per_modality.vision", op=">", line=0.04,
                      from_run_hour=0, mode="stop", note="S1")]
ROWS = dict(repair_stop_row="written", cont_stop_row="written", cont_done_row="written", ext_stop_row="written",
            ext_done_row="written", done_row="done")
CONT_WORDS = "the operator's words"
CONT_AT = "Oct 12, 2026 5:36 PM PDT"
PRODUCTION_ROUTE = dict(fingerprint="2c3d82ca4ab4a870",
                        continuation_run="D:/UEMA_Research/external/D/UEMA_scratch/body1_continuation/run/B1C_skip13_484675",
                        stopped_at=485975, end=489350)


def _src(tmp_path):
    src = tmp_path / "src"
    for rel, text in (("lem/model/core.py", "x = 1\n"), ("mimo_bridge/train_mimo.py", TRAINER)):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return src


def _registry(tmp_path):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(dict(rows=[dict(id=k, status=v) for k, v in ROWS.items()])), encoding="utf-8")
    return p


def _stats(run: Path, steps_elapsed):
    run.mkdir(parents=True, exist_ok=True)
    (run / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(
        type="mimo", step=s, elapsed_s=e, causal_pe_per_modality=dict(vision=0.01))) + "\n"
        for s, e in steps_elapsed), encoding="utf-8")


def _end(run: Path, max_steps: int, stopped_at: int):
    """A run's args.json naming its end, its stage_done.json and its checkpoint at the model step it ended."""
    (run / "args.json").write_text(json.dumps(dict(max_steps=max_steps, resume=None)), encoding="utf-8")
    (run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=stopped_at, finished_at=time.time())),
                                         encoding="utf-8")
    ck = run / "checkpoints" / ("ckpt_step_%d" % stopped_at)
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "meta.json").write_text("{}", encoding="utf-8")
    return ck


def _exceptions(gates=range(1, 8)):
    return {str(n): dict(failure="gate %d fails" % n, why_measurable="the readings decide on posture directly",
                         evidence=["row_%d" % n]) for n in gates}


def _ext_block(start, lat, **more):
    b = dict(purpose="the remainder of the stopped continuation's span", start_checkpoint=str(start),
             rollback_checkpoint=str(start), stop_lines="S1", stop_lines_row="ext_stop_row",
             stop_criteria=STOP_CRITERIA, reachability="the first check reads S1 about 330 model steps in",
             max_model_steps=700, no_auto_continue="the owner starts the trainer with --max-steps and no supervisor",
             completion_jobs_row="ext_done_row", authorization="the operator's words for the extension",
             authorized_by="the operator", authorized_at="Oct 12, 2026 11:00 PM PDT",
             run_flags={"--latent-body-output": str(lat)}, exceptions=_exceptions())
    b.update(more)
    return b


def _world(tmp_path, continuation_stopped_at=1800, latent=False):
    """The stand-in record and runs (the module text). latent: the configuration runs the latent body
    (--latent-body-hierarchy shadow, as record 2c3d82ca4ab4a870's), so the continuation names the latent state it resumed
    and the folder its run saved latest.pt in (lat_c)."""
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    exp = tmp_path / "experiments"
    a = exp / "run_a"
    start = a / "checkpoints" / "ckpt_step_1000"
    start.mkdir(parents=True)
    (a / "args.json").write_text('{"resume": null}', encoding="utf-8")
    _stats(a, ((900, 167.0), (1000, 440.0)))
    base = ["python", "-u", "-m", "mimo_bridge.train_mimo", "--intention", "on", "--resume", str(start), "--out",
            "C:/runs/x", "--report-every", "100", "--proprio-token-skip", "1,3"]
    if latent:
        base += ["--latent-body-hierarchy", "shadow"]
    fp = Path(TP.new_record(base, "configuration X", "tester", root=root, src=src)).stem
    TP.set_repair_run(fp, dict(purpose="repair", target_gates=[1], repairs=["the fix"], affected_components=["ENC"],
                               start_checkpoint=str(start), rollback_checkpoint=str(start), stop_lines="S1",
                               stop_lines_row="repair_stop_row", stop_criteria=STOP_CRITERIA, reachability="r",
                               max_model_steps=500, no_auto_continue="n", remeasure={"1": "row_1b"}),
                      "tester", root=root, registry=reg)
    TP.set_gate(fp, 1, TP.REPAIR_TARGET, "fails today", ["row_1"], "tester", root=root)
    for n in range(2, 8):
        TP.set_gate(fp, n, "FAIL", "fails at the start", ["row_%d" % n], "tester", root=root)
    # the repair run: noted, trained, marked, completed at 1,500; its target gate judged again after its end
    r = exp / "run_r"
    r.mkdir(parents=True)
    TP.note_attempt(fp, str(r), "launcher", root=root)
    _stats(r, ((1100, 120.0), (1200, 340.0), (1300, 560.0), (1400, 780.0), (1500, 1000.0)))
    TP.mark_launched(fp, str(r), "launcher", root=root)
    r_end = _end(r, 1500, 1500)
    time.sleep(0.02)
    TP.set_gate(fp, 1, "FAIL", "judged again after the repair run", ["row_1b"], "tester", root=root)
    # the continuation: written, used, STOPPED EARLY at continuation_stopped_at (launched to end at 2,500)
    cont_flags = {"--latent-body-output": str(tmp_path / "lat_c")}
    if latent:
        lat_r = tmp_path / "lat_r" / "latest.pt"           # the state the continuation resumed (the repair run's)
        lat_r.parent.mkdir(parents=True)
        lat_r.write_text("state r", encoding="utf-8")
        cont_flags["--latent-body-resume"] = str(lat_r)
    TP.set_continuation(fp, dict(
        purpose="the continuation", start_checkpoint=str(r_end), rollback_checkpoint=str(r_end), stop_lines="S1",
        stop_lines_row="cont_stop_row", stop_criteria=STOP_CRITERIA, reachability="r", max_model_steps=1000,
        no_auto_continue="n", completion_jobs_row="cont_done_row", authorization=CONT_WORDS,
        authorized_by="the operator", authorized_at=CONT_AT, run_flags=cont_flags, exceptions=_exceptions()),
        "tester", root=root, registry=reg)
    c = exp / "run_c"
    c.mkdir(parents=True)
    TP.note_attempt(fp, str(c), "launcher", root=root, block=TP.CONTINUATION)
    _stats(c, ((1600, 108.0), (1700, 328.0), (1800, 548.0)))
    TP.mark_launched(fp, str(c), "launcher", root=root, block=TP.CONTINUATION)
    c_end = _end(c, 2500, continuation_stopped_at)
    sets = [("--resume", str(c_end)), ("--max-steps", "2500"), ("--latent-body-output", str(tmp_path / "lat_x"))]
    if latent:
        saved = tmp_path / "lat_c" / "latest.pt"           # the state the continuation's run saved
        saved.parent.mkdir(parents=True)
        saved.write_text("state c", encoding="utf-8")
        sets.append(("--latent-body-resume", str(saved)))
    cmd = TP.apply_sets(base, sets)
    rec = TP._load(TP.record_path(fp, root))
    return dict(src=src, root=root, reg=reg, fp=fp, base=base, c=c, c_end=c_end, r_end=r_end, cmd=cmd,
                lat=tmp_path / "lat_x", stopped_at=continuation_stopped_at, end=2500,
                c_run=TP.repair_launch(rec["continuation"])[2], tmp=tmp_path)


@pytest.fixture
def pin(monkeypatch):
    """pin(world, **changes): EXTENSION_ROUTE names the world's own continuation (its record, its run as the record
    stores it, the step it stopped at, its end), with `changes` applied; restored after the test."""
    def _pin(w, **changes):
        route = dict(fingerprint=w["fp"], continuation_run=str(w["c_run"]), stopped_at=w["stopped_at"], end=w["end"])
        route.update(changes)
        monkeypatch.setattr(TP, "EXTENSION_ROUTE", route)
        return w
    return _pin


def _refusal(w, cmd=None):
    return TP.launch_refusal(cmd or w["cmd"], root=w["root"], src=w["src"], registry=w["reg"])


def _write(w, **more):
    return TP.set_extension(w["fp"], _ext_block(w["c_end"], w["lat"], **more), "tester", root=w["root"],
                            registry=w["reg"])


def _write_without(w, key):
    b = _ext_block(w["c_end"], w["lat"])
    b.pop(key)
    return TP.set_extension(w["fp"], b, "tester", root=w["root"], registry=w["reg"])


# ------------------------------------------------------------------------------ (1) and (2)
def test_without_an_extension_the_stopped_continuations_end_is_normal_training_and_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    why = _refusal(w)
    assert why and "7 gates block launch" in why, why
    assert TP.launch_plan(w["cmd"], root=w["root"], src=w["src"])["kind"] == "normal"


def test_the_extension_admits_exactly_its_launch_and_the_record_stays_byte_identical(tmp_path, pin):
    w = pin(_world(tmp_path))
    rec_path = TP.record_path(w["fp"], w["root"])
    before = rec_path.read_bytes()
    b = _write(w)
    assert rec_path.read_bytes() == before
    held = json.loads(TP.extension_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
    assert held["block"]["role"] == "extension" and held["record_sha256"] == TP._sha(before)
    assert held["continuation_run"] == str(w["c"].resolve()) or Path(held["continuation_run"]) == w["c"].resolve()
    assert _refusal(w) is None
    plan = TP.launch_plan(w["cmd"], root=w["root"], src=w["src"])
    assert (plan["kind"], plan["block"], plan["start_step"], plan["max_steps"]) == ("continuation", "extension", 1800,
                                                                                     2500)
    assert plan["stop_criteria"] == STOP_CRITERIA and b["written_at_s"] > 0
    assert rec_path.read_bytes() == before


# ------------------------------------------------------------------------------ (3)
def test_the_launch_past_the_continuations_end_and_a_block_bound_past_it_are_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError, match="passes the continuation's end"):
        _write(w, max_model_steps=701)
    _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--max-steps", "2501")]))
    assert why and "not bounded as declared" in why, why


def test_another_start_another_resume_and_another_rows_are_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    other = w["c"] / "checkpoints" / "ckpt_step_1700"
    with pytest.raises(TP.PreflightError, match="is not the end checkpoint"):
        TP.set_extension(w["fp"], _ext_block(other, w["lat"]), "tester", root=w["root"], registry=w["reg"])
    with pytest.raises(TP.PreflightError, match="the continuation's own"):
        _write(w, stop_lines_row="cont_stop_row")
    with pytest.raises(TP.PreflightError, match="the repair run's own"):
        _write(w, stop_lines_row="repair_stop_row")
    with pytest.raises(TP.PreflightError, match="completion jobs' row cont_done_row is the continuation's own"):
        _write(w, completion_jobs_row="cont_done_row")
    with pytest.raises(TP.PreflightError, match="reads status done"):
        _write(w, completion_jobs_row="done_row")
    _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--resume", str(w["r_end"]))]))
    assert why and "7 gates block launch" in why, why                     # another --resume: normal training
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--latent-body-output", str(w["lat"]) + "_other")]))
    assert why and "run_flags, compared exactly" in why, why


def test_the_extension_refusal_names_a_launch_that_resumes_another_checkpoint(tmp_path, pin):
    """launch_refusal routes a launch to the extension only when its --resume is the extension's start, so the
    launch-time comparison of --resume is read here directly (the review's probe of its mutation M20)."""
    w = pin(_world(tmp_path))
    _write(w)
    xb = TP.extension_of(w["fp"], w["root"])["block"]
    rec = TP._load(TP.record_path(w["fp"], w["root"]))
    cmd = TP.apply_sets(w["cmd"], [("--resume", str(w["r_end"]))])
    why = TP._extension_refusal(cmd, rec, xb, TP.configuration(cmd, w["src"]), w["root"], w["reg"])
    assert why and "not the extension's start" in why, why


def test_a_continuation_that_reached_its_end_or_was_never_used_has_no_extension(tmp_path, pin):
    reached = pin(_world(tmp_path / "reached", continuation_stopped_at=2500))
    with pytest.raises(TP.PreflightError, match="reached its end"):
        _write(reached)
    unused = pin(_world(tmp_path / "unused"))
    rec = TP._load(TP.record_path(unused["fp"], unused["root"]))
    rec["continuation"].update(launched_at=None, launched_at_s=None, run=None, attempts=[])
    TP._save(TP.record_path(unused["fp"], unused["root"]), rec)
    with pytest.raises(TP.PreflightError, match="was never used"):
        _write(unused)


def test_a_gate_judged_again_after_the_extension_was_written_and_an_existing_latent_output_are_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    time.sleep(0.02)
    TP.set_gate(w["fp"], 4, "FAIL", "judged again", ["row_4b"], "tester", root=w["root"])
    why = _refusal(w)
    assert why and "gate 4" in why and "judged again since its exception was written" in why, why
    w2 = pin(_world(tmp_path / "lat"))
    w2["lat"].mkdir(parents=True)
    with pytest.raises(TP.PreflightError, match="exists already"):
        _write(w2)


# ------------------------------------------------------------------------------ (4)
def test_the_extension_is_used_once(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    run = tmp_path / "experiments" / "run_x"
    run.mkdir(parents=True)
    TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)
    assert _refusal(w) is None                                          # the attempt has not trained yet
    _stats(run, ((1900, 108.0),))
    why = _refusal(w)
    assert why and "7 gates block launch" in why, why                   # used: the same command is normal training
    with pytest.raises(TP.PreflightError, match="was used"):
        _write(w)
    rr = TP.mark_launched(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)
    assert rr["launched_at"] and rr["run"] == str(run.resolve())
    with pytest.raises(TP.PreflightError, match="launched already"):
        TP.mark_launched(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)


def test_an_attempt_that_never_trained_may_be_written_again_with_its_attempts_carried(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    run = tmp_path / "experiments" / "run_x"
    run.mkdir(parents=True)
    TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)
    b = TP.set_extension(w["fp"], _ext_block(w["c_end"], tmp_path / "lat_x2"), "tester", root=w["root"],
                         registry=w["reg"])
    assert [a["run"] for a in b["attempts"]] == [str(run.resolve())]
    _stats(run, ((1900, 108.0),))                                       # the carried attempt trains later: the use
    cmd = TP.apply_sets(w["cmd"], [("--latent-body-output", str(tmp_path / "lat_x2"))])
    why = _refusal(w, cmd)
    assert why and "7 gates block launch" in why, why


# ------------------------------------------------------------------------------ (5)
def test_every_other_route_is_unchanged(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    cont_start = TP.apply_sets(w["base"], [("--resume", str(w["r_end"])), ("--max-steps", "2500"),
                                           ("--latent-body-output", str(tmp_path / "lat_c"))])
    why = _refusal(w, cont_start)
    assert why and "7 gates block launch" in why, why                   # the used continuation's start
    assert TP.launch_plan(cont_start, root=w["root"], src=w["src"])["kind"] == "normal"
    assert TP.main(["launched", w["fp"], "--run", str(tmp_path), "--extension", "--continuation", "--by", "t"]) == 2


# ------------------------------------------------------------------------------ (6) THE ONE EXTENSION (finding B-1)
def test_the_route_names_productions_one_extension():
    assert TP.EXTENSION_ROUTE == PRODUCTION_ROUTE


def test_another_configuration_is_refused_when_written_at_launch_and_when_noted(tmp_path, pin):
    w = _world(tmp_path)                       # EXTENSION_ROUTE as written in the code: production's configuration
    with pytest.raises(TP.PreflightError, match="admits one extension only, of configuration 2c3d82ca4ab4a870"):
        _write(w)
    assert not TP.extension_path(w["fp"], w["root"]).exists()
    pin(w)                                      # written while the route named this world, then the route as written
    _write(w)
    pin(w, fingerprint=PRODUCTION_ROUTE["fingerprint"])
    why = _refusal(w)
    assert why and "admits one extension only, of configuration 2c3d82ca4ab4a870" in why, why
    run = tmp_path / "experiments" / "run_x"
    run.mkdir(parents=True)
    with pytest.raises(TP.PreflightError, match="admits one extension only"):
        TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)
    with pytest.raises(TP.PreflightError, match="admits one extension only"):
        TP.mark_launched(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)


def test_another_continuation_run_is_refused_when_written_and_when_noted(tmp_path, pin):
    w = pin(_world(tmp_path), continuation_run=str(tmp_path / "experiments" / "run_other"))
    with pytest.raises(TP.PreflightError, match="admits only the extension of the continuation run"):
        _write(w)
    pin(w)
    _write(w)
    pin(w, continuation_run=str(tmp_path / "experiments" / "run_other"))
    why = _refusal(w)
    assert why and "admits only the extension of the continuation run" in why, why
    run = tmp_path / "experiments" / "run_x"
    run.mkdir(parents=True)
    with pytest.raises(TP.PreflightError, match="admits only the extension of the continuation run"):
        TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.EXTENSION)


def test_another_stop_step_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path), stopped_at=1700)
    with pytest.raises(TP.PreflightError, match="admits only the extension from model step 1700"):
        _write(w)


def test_another_end_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path), end=2400)
    with pytest.raises(TP.PreflightError, match="admits only the extension to model step 2400"):
        _write(w)


# ------------------------------------------------------------------------------ (7) one test per condition (B-2)
def test_an_end_checkpoint_without_its_meta_json_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    (w["c_end"] / "meta.json").unlink()
    with pytest.raises(TP.PreflightError, match=r"is not complete \(no meta.json\)"):
        _write(w)


def test_a_continuation_whose_end_save_does_not_read_ok_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    done = json.loads((w["c"] / "stage_done.json").read_text(encoding="utf-8"))
    for ok in (False, None, "true"):
        done["ok"] = ok
        (w["c"] / "stage_done.json").write_text(json.dumps(done), encoding="utf-8")
        with pytest.raises(TP.PreflightError, match="did not end with its own save"):
            _write(w)


def test_a_continuation_end_without_finished_at_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    done = json.loads((w["c"] / "stage_done.json").read_text(encoding="utf-8"))
    for finished in (None, "late", float("nan")):
        if finished is None:
            done.pop("finished_at", None)
        else:
            done["finished_at"] = finished
        (w["c"] / "stage_done.json").write_text(json.dumps(done), encoding="utf-8")
        with pytest.raises(TP.PreflightError, match="end time cannot be read"):
            _write(w)


def test_a_continuation_that_stopped_at_or_before_its_start_is_refused(tmp_path, pin):
    for at in (1500, 1400):                     # the continuation started at 1,500
        w = pin(_world(tmp_path / str(at), continuation_stopped_at=at))
        with pytest.raises(TP.PreflightError, match="not after its start 1500"):
            TP.set_extension(w["fp"], _ext_block(w["c_end"], w["lat"], max_model_steps=2500 - at), "tester",
                             root=w["root"], registry=w["reg"])


def test_a_continuation_launched_past_its_own_bound_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path), end=2600)         # the continuation's bound: start 1,500 + 1,000 = 2,500
    (w["c"] / "args.json").write_text(json.dumps(dict(max_steps=2600, resume=None)), encoding="utf-8")
    with pytest.raises(TP.PreflightError, match="names no end within its bound"):
        _write(w)


def test_the_operators_words_are_required(tmp_path, pin):
    w = pin(_world(tmp_path))
    for key in ("authorization", "authorized_by", "authorized_at"):
        with pytest.raises(TP.PreflightError, match="the extension's %s is missing" % key.replace("_", " ")):
            _write_without(w, key)
        for empty in ("", "   \n "):
            with pytest.raises(TP.PreflightError, match="the extension's %s is missing" % key.replace("_", " ")):
                _write(w, **{key: empty})
    assert not TP.extension_path(w["fp"], w["root"]).exists()


@pytest.mark.parametrize("key", [k for k in TP.CONTINUATION_FIELDS
                                 if k not in ("authorization", "authorized_by", "authorized_at")])
def test_every_other_required_field_is_required(tmp_path, pin, key):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError, match="the extension's %s is missing" % key.replace("_", " ")):
        _write_without(w, key)


def test_an_extension_that_repeats_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    for repeats in (2, 0, True, "1"):
        with pytest.raises(TP.PreflightError, match="an extension is used once"):
            _write(w, repeats=repeats)
    _write(w, repeats=1)


# ------------------------------------------------------------------------------ (8) the latent state (finding B-3)
def test_the_latent_state_resumed_is_the_one_the_continuations_run_saved(tmp_path, pin):
    w = pin(_world(tmp_path, latent=True))
    saved = tmp_path / "lat_c" / "latest.pt"
    other = tmp_path / "lat_r" / "latest.pt"    # an existing file: the state the continuation itself resumed
    with pytest.raises(TP.PreflightError, match="which an extension resumes exactly"):
        _write(w, run_flags={"--latent-body-output": str(w["lat"]), "--latent-body-resume": str(other)})
    with pytest.raises(TP.PreflightError, match="names --latent-body-resume"):
        _write(w)                               # the latent body runs: the state it resumes is named
    _write(w, run_flags={"--latent-body-output": str(w["lat"]), "--latent-body-resume": str(saved)})
    assert _refusal(w) is None


# ------------------------------------------------------------------------------ (9) the operator's words (finding B-4)
def test_the_operators_words_must_be_text(tmp_path, pin):
    w = pin(_world(tmp_path))
    for key, value in (("authorization", ["the operator's words"]), ("authorization", 12),
                       ("authorized_by", {"who": "the operator"}), ("authorized_at", 1791880000.0),
                       ("purpose", ["a purpose"]), ("start_checkpoint", 1800)):
        with pytest.raises(TP.PreflightError, match="the extension's %s must be text" % key.replace("_", " ")):
            _write(w, **{key: value})


def test_the_continuations_words_are_not_the_extensions(tmp_path, pin):
    w = pin(_world(tmp_path))
    for words in (CONT_WORDS, "  the operator's\nwords "):
        with pytest.raises(TP.PreflightError, match="authorization repeats the continuation's words"):
            _write(w, authorization=words)


def test_the_continuations_time_is_not_the_extensions(tmp_path, pin):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError, match="authorized at repeats the continuation's time"):
        _write(w, authorized_at=CONT_AT)
