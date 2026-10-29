# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""docs/gain_of_function_v4/memory_starter_test.py, the memory starter test's harness, after its two reviews
(docs/gain_of_function_v4/agents/memory_starter_test_review_a_2026-09-26.md and memory_starter_test_review_2026-09-26.md)
and its re-review (memory_starter_test_review2_2026-09-26.md).

The first round's tests each hold one fix and fail on the harness as first reviewed (2ddb504d5): the floor under the
line, contradicting bands, the reductions stated, the controls' gate, the required readings (S5), the judge's per-arm
preconditions, resumed arms flagged and held to an exact continuation, the start read again inside the readings, slots
paired by position, the timeline's band never narrower than rounding, the job's deadline ending the trainer's whole
tree, --force starting again from the copy, the harness file pinned at prepare, and the declared floor recomputed from
the lineage file.
The second round's tests each fail on the harness as re-reviewed (340fe6fd4, 147ad6583): a trainer run ended from
outside is recorded before it starts and the continued arm passes its checks; a continued launch stopped while building
its worlds is not held to them, while a launched run missing a world, another digest, or two launches with one log
still fail; a failed precondition writes the same NO VERDICT into every slot; the fifth arm reads a removal that leaves
growth beside a stopped detach as a contradiction; the judge refuses readings older than an arm receipt; the declared
reading states what VALID needs at 4 worlds and the controls' receipt names each slot's VALID; the smoke's pass holds
every gate (S5, the copy's skins, S8 the saves' resume state, S9 the projection), with --force and a kept pass; the
controls' gate holds its length and the smoke's start weights; the resume-state reader loads real torch saves; every job
waits for the heavy-job lock; a process still alive 30 s after the harness ended its trainer's tree is recorded.
No trainer, model or world runs: the folders point at a temporary directory and the reader and the trainer are replaced
where a test needs them. Measured light: pure Python, plus short child processes (two of them import torch on the CPU).
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SCRIPT = REPO / "docs/gain_of_function_v4/memory_starter_test.py"
JOBS = [REPO / ("docs/gain_of_function_v4/pause_jobs_2026-09-26_memory_starter_test%s.json" % s)
        for s in ("", "_arm5_template", "_longer_template")]
COMMIT = "a" * 40
BLOB = "b" * 40
READER = "c" * 40
SLOTS_NOW = ["vision", "tactile", "audio", "proprioception", "motor", "vestibular"]
SLOTS_THEN = ["vision", "tactile", "audio", "proprioception", "motor", "intermind"]
LAYOUT = ("WARNING: touch inside rays fixed_v1 (world _w%d): 58 inside tests ran with directions sha256 a112217f...; "
          "skin layout 4683 points, digest %s\n")
USUAL = "0581069917fae467"
REPAIRED = ("motor_controller__residual_blocks__0__0", "persistent_memory__outcome__0")


@pytest.fixture
def H(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("memory_starter_harness_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(mod, "START_ROOT", tmp_path / "scratch" / "start_344400", raising=False)
    monkeypatch.setattr(mod, "RESULTS", tmp_path / "results")
    (tmp_path / "scratch").mkdir()
    return mod


def prep_of(H):
    return dict(ok=True, commit=COMMIT, harness=dict(blob=BLOB, committed=True), start_step=H.START_STEP,
                reader=dict(commit=READER, blob="r" * 40, dir="unused", script="unused"),
                python_path="unused", start_checkpoint="unused", base_command=["PY", "-m", "mimo_bridge.train_mimo"],
                start_skins=dict(usual_layout_in_every_world=True))


def block(i, step, detached=(), removed=(), leak=None, state_leak=None, drop=()):
    """One report's memory_starter block as the patch writes it (lem/model/memory_starter.py window_summary)."""
    heads = [s for s in SLOTS_NOW if s != "motor"]
    rows = {"delayed.pred_head.vision": None, "replay.pred_head.vision": None, "delayed.gaze_copy": "gaze_copy",
            "replay.gaze_copy": "gaze_copy", "delayed.efference_body": "efference_body"}
    req = {k: dict(request_norm_mean=1e-3, detached=r in detached) for k, r in rows.items() if r not in removed}
    mem = dict(constant_rms=6.0 + 0.01 * i, constant_rms_per_slot={s: 6.0 for s in SLOTS_NOW},
               slope_since_first_step_per_update={s: 1e-3 for s in SLOTS_NOW},
               constant_rms_step_std=0.01, constant_rms_step_std_per_slot={s: 0.01 for s in SLOTS_NOW})
    if i > 0:
        mem["slope_since_previous_report_per_update"] = {s: 1e-3 for s in SLOTS_NOW}
    ms = dict(slots=list(SLOTS_NOW), memory=mem, requests=req,
              site_totals={s: dict(request_norm_mean=1e-3, unassigned_share_max=1e-7 if leak is None else leak)
                           for s in ("delayed", "lived", "replay")},
              state_requests={"delayed.needs": dict(request_norm_mean=1e-4),
                              "delayed.readout": dict(request_norm_mean=1e-3)},
              state_site_totals={"delayed": dict(request_norm_mean=1e-3, unassigned_share_per_body_max=(
                  1e-7 if state_leak is None else state_leak))},
              attention={site: {h: dict(weight_mean=[[1 / 6.0] * 6, [1 / 6.0] * 6]) for h in heads}
                         for site in ("delayed", "lived", "replay")},
              motor_first_layer=dict(layer="motor_controller.residual_blocks.0.0", live_share=0.94, live_units=120),
              vision_error=dict(head_alone=0.5, head_plus_gaze_copy=0.45))
    for k in drop:
        ms.pop(k, None)
    return dict(type="mimo", step=step, timestamp=1000.0 + 2.0 * step, memory_starter=ms)


def write_arm(H, name, growth_value, detached=None, removed=(), commit=COMMIT, digest="d" * 64, leak_at=None,
              resumed=None, layout_digest=USUAL, config_detached=None):
    """A finished arm as the harness leaves it: its folder, receipt, stats (launch, config, reports) and log; each trainer
    run recorded with the stats' launch count at its start, as run_trainer records it."""
    det, rem = H._detach_set(name)
    detached = det if detached is None else detached
    folder = H.arms_root() / H.TAGS[name]
    out = folder / ("H84_fresh_drivers_20260918_st_" + H.TAGS[name])
    out.mkdir(parents=True, exist_ok=True)
    log = folder / "arm_1.log"
    log.write_text("".join(LAYOUT % (w, layout_digest if w == 2 else USUAL) for w in range(4)), encoding="utf-8")
    logs = [str(log)]
    recs = [dict(type="launch", step=H.START_STEP, code_commit=commit),
            dict(type="memory_starter_config", step=H.START_STEP,
                 detached=sorted(detached if config_detached is None else config_detached), removed=sorted(rem),
                 start_digest=digest, resumed=None)]
    steps = [H.START_STEP + 100, H.START_STEP + 200, H.START_STEP + 300]
    for i, s in enumerate(steps):
        recs.append(block(i, s, detached=det, removed=rem, leak=1e-3 if leak_at == s else None))
    if resumed is not None:
        at, restored = resumed
        log2 = folder / "arm_2.log"
        log2.write_text("".join(LAYOUT % (w, USUAL) for w in range(4)), encoding="utf-8")
        logs.append(str(log2))
        recs += [dict(type="launch", step=at, code_commit=commit),
                 dict(type="memory_starter_config", step=at, detached=sorted(detached), removed=sorted(rem),
                      start_digest="e" * 64,
                      resumed=(dict(saved_at_step=at, liveness_windows_restored=["x"]) if restored else None))]
    (out / "replay_training_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    t = time.time()
    receipt = dict(name=name, complete=True, commit=commit, harness=dict(blob=BLOB), updates=H.UPDATES,
                   stats=str(out / "replay_training_stats.jsonl"), logs=logs,
                   segments=[dict(state="ended", log=l, started=t - 1000 + i, launches_before=i)
                             for i, l in enumerate(logs)],
                   paired_part=["PY", "-m", "mimo_bridge.train_mimo", "--same"], command_first=[], finished=t)
    (folder / "arm_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return growth_value


def write_readings(H, growths, slot_growths=None):
    """readings.json as `readings` writes it: the start and each arm read by the pinned reader, and when it started."""
    start = {H.START_STEP: dict(constant_rms=5.0, slots_ordered=[[s, 5.0] for s in SLOTS_THEN])}

    def slots(n, g):
        return [[s, 5.0 + (slot_growths or {}).get(n, {}).get(s, g)] for s in SLOTS_NOW]
    arms = {n: dict(exit=0, reader_commit=READER,
                    constants={H.END_STEP: dict(constant_rms=5.0 + g, slots_ordered=slots(n, g))})
            for n, g in growths.items()}
    t = time.time()
    out = dict(reader=dict(commit=READER), start=dict(exit=0, reader_commit=READER, constants=start), arms=arms,
               started=t, finished=t)
    (H.SCRATCH / "readings.json").write_text(json.dumps(out), encoding="utf-8")


def four(H, **over):
    base = dict(control=4.0, control2=4.1, detach_gaze_copy=0.1, detach_efference_body=4.0, detach_both=0.12)
    specs = {n: {} for n in base}
    for key, value in over.items():
        n, field = key.split("__")
        specs[n][field] = value
    g = {n: write_arm(H, n, base[n], **specs[n]) for n in base}
    write_readings(H, g)
    return g


def judged(H, monkeypatch):
    monkeypatch.setattr(H, "_prepared", lambda: prep_of(H))
    code = H.judge()
    return code, json.loads((H.SCRATCH / "judged.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ a stand-in trainer
class Killed(BaseException):
    """The harness ended from outside (the watchdog on a pause lift, the runner at its timeout): no Python code after
    the trainer's start runs, as when the whole process tree is ended."""


def save(out, step):
    c = Path(out) / "checkpoints" / ("ckpt_step_%d" % step)
    c.mkdir(parents=True, exist_ok=True)
    for f in ("model.safetensors", "optimizer.pt", "runtime_state.pt", "episode_state.json", "lineage.json"):
        (c / f).write_text("x", encoding="utf-8")


def stand_in_trainer(H, name, plan, seen=None):
    """run_measured replaced by a stand-in trainer. plan[k] describes call k: `worlds` (the world lines its log shows),
    `launch` (the step it launches from; None: stopped while building its worlds, so no launch record), `reports` and
    `saves` (steps), `per_update_s` (the time between reports), `drop` (block keys left out), `end` ('return', 'reserve'
    or 'killed'), `peak_gb`. `seen` collects what attempt.json held as each trainer started. Returns (run_measured,
    calls) with calls [(command, log path)]."""
    det, rem = H._detach_set(name)
    calls = []

    def run_measured(cmd, cwd, env, log_path):
        p = plan[len(calls)]
        calls.append((list(cmd), str(log_path)))
        if seen is not None:
            seen.append(json.loads((Path(log_path).parent / "attempt.json").read_text(encoding="utf-8")))
        out = Path(cmd[cmd.index("--out") + 1])
        Path(log_path).write_text("".join(LAYOUT % (w, USUAL) for w in range(p.get("worlds", 4))), encoding="utf-8")
        if p.get("launch") is not None:
            t0 = time.time()
            at = p["launch"]
            recs = [dict(type="launch", timestamp=t0, step=at, resumed_from="x", code_commit=COMMIT),
                    dict(type="memory_starter_config", timestamp=t0, step=at, detached=sorted(det), removed=sorted(rem),
                         start_digest="d" * 64,
                         resumed=None if at == H.START_STEP else dict(saved_at_step=at,
                                                                     liveness_windows_restored=list(REPAIRED)))]
            for s in p.get("reports", []):
                b = block((s - H.START_STEP) // H.REPORT_EVERY - 1, s, detached=det, removed=rem, drop=p.get("drop", ()))
                b["timestamp"] = t0 + p.get("per_update_s", 2.0) * (s - at)
                recs.append(b)
            with open(out / "replay_training_stats.jsonl", "a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r) + "\n" for r in recs))
            for s in p.get("saves", []):
                save(out, s)
        end = p.get("end", "return")
        if end == "killed":
            raise Killed()
        reserve = dict(committed_gb=63.0, limit_gb=68.8) if end == "reserve" else None
        return (0 if end == "return" else 1), dict(tree_private_peak_gb=p.get("peak_gb", 21.0),
                                                     stopped_at_reserve=reserve, ended_at_deadline=None)
    return run_measured, calls


def reports(a, b):
    return list(range(a, b, 100))


# ------------------------------------------------------------------------------------------------ the declared reading
def test_the_floor_is_the_lineages_own_random_walk_fit(H):
    """The floor's constants are the lineage file's own fit (29 pre-deploy intervals), recomputed here; and the floor at
    the window's 3,391 updates is sqrt 2 x 0.00403 x sqrt(3,391) = 0.332 per number."""
    w = H.window_arithmetic()
    assert w["intervals"] == H.WALK_FIT["intervals"] == 29
    assert abs(w["sigma_overall"] - H.WALK_FIT["overall"]) < 1e-9
    assert all(abs(a - b) < 1e-9 for a, b in zip(w["sigma_slots"], H.WALK_FIT["slots"]))
    assert w["slot_names_then"] == list(H.WALK_FIT["slot_names_then"])
    assert H.UPDATES == 3391 and abs(H.floor_of() - 0.3322) < 1e-3
    assert abs(w["stopped_arm_reads_stopped_at_least"] - 0.9214) < 1e-3
    assert abs(w["growth_8_bodies"] - (9.885 - 5.6558)) < 1e-3
    assert w["table"][-1]["growth_over_floor"] > 10 and w["table"][-1]["slowest_slot_growth_over_floor"] > 4


def test_the_declared_reading_states_what_valid_needs_at_4_worlds(H):
    """The re-review (SEMANTIC): at 4 worlds the growth and the wander are unmeasured, and VALID needs 16 in 100 of the
    8-body growth overall and 16 to 40 in 100 per slot (touch highest). The declared shares are the lineage's own
    numbers, and the declared reading states them."""
    v = H.window_arithmetic()["valid_needs"]
    assert round(100 * v["overall"]["share_of_8_body_growth"]) == H.DECLARED_VALID_SHARES["overall"] == 16
    shares = [round(100 * s["share_of_8_body_growth"]) for s in v["slots"]]
    assert shares == list(H.DECLARED_VALID_SHARES["slots"]) == [26, 40, 16, 34, 16, 27]
    assert v["slots"][shares.index(max(shares))]["slot_then"] == "tactile"
    assert abs(v["overall"]["valid_needs_growth_above"] - 2 * H.floor_of()) < 1e-6     # the declared fit's 10 digits
    doc = " ".join(H.__doc__.split())
    assert "AT 4 WORLDS THE GROWTH AND THE WANDER ARE UNMEASURED" in doc
    assert "all six slots together 16 in 100" in doc and "touch 40 (the highest)" in doc


def test_a_truly_stopped_arm_reads_stopped_against_the_floor(H):
    """Review 2's synthetic case: controls at 4.0 and 4.0 and a resolution of 0.03, a stopped gaze-copy arm at 0.15 and a
    stopped both-cut arm at 0.10. Without a floor the line was 0.042 and both read REDUCED ("another path"); against the
    declared floor they read STOPPED and the verdict names the gaze copy."""
    grow = dict(control=4.0, control2=4.0, detach_gaze_copy=0.15, detach_efference_body=4.0, detach_both=0.10)
    out = H.judge_growth(grow, 0.03)
    assert out["line_from"] == "the declared floor" and abs(out["line"] - H.floor_of()) < 1e-12
    assert out["bands"] == dict(detach_gaze_copy="STOPPED", detach_efference_body="CONTINUED", detach_both="STOPPED")
    assert out["verdict"] == "the gaze copy's training is the starter"
    no_floor = H.judge_growth(grow, 0.03, floor=0.0)
    assert no_floor["bands"]["detach_gaze_copy"] == "REDUCED"      # the reviewed behaviour, kept reachable


def test_contradicting_bands_name_no_starter(H):
    """Review 2 (MAJOR, from review A's MINOR): the gaze-copy cut STOPPED while cutting both readers CONTINUED named the
    gaze copy; cutting both includes that cut, so it is NO VERDICT, named as a contradiction."""
    out = H.judge_growth(dict(control=4.0, control2=4.1, detach_gaze_copy=0.1, detach_efference_body=4.0,
                              detach_both=4.0), 0.03)
    assert out["valid"] and out["verdict"].startswith("NO VERDICT: contradicting bands")
    assert out["contradictions"] and "the gaze copy" in out["contradictions"][0]
    both_stop = H.judge_growth(dict(control=4.0, control2=4.1, detach_gaze_copy=4.0, detach_efference_body=4.0,
                                    detach_both=0.1), 0.03)
    assert both_stop["verdict"].startswith("both contribute")


def test_cuts_that_reduce_but_do_not_stop_state_their_share(H):
    """Review 2 (SEMANTIC): three cuts each removing 76 in 100 of the growth read only "another gradient or update path";
    the verdict now says what each cut removed."""
    out = H.judge_growth(dict(control=4.0, control2=4.0, detach_gaze_copy=0.96, detach_efference_body=0.96,
                              detach_both=0.96), 0.03)
    assert out["verdict"].startswith("another gradient or update path")
    assert "removed 76 in 100" in out["verdict"] and "REDUCED" in out["verdict"]


def test_the_no_verdict_rule_names_the_next_length(H):
    r = H.next_test(0.5, 0.33, 3391)
    assert r["updates"] == 9000 and "--updates 9000" in r["next"]
    assert "operator" in H.next_test(-0.1, 0.33, 3391)["next"]
    assert H.next_test(4.0, 0.33, 3391)["updates"] >= 3391 + H.SAVE_EVERY


def test_slots_are_paired_by_position_across_the_vestibular_switch(H):
    """The sixth slot is intermind at 344,400 and vestibular after the deploy's load: paired by name it had no growth."""
    start = dict(constants={H.START_STEP: dict(constant_rms=5.0, slots_ordered=[[s, 5.0] for s in SLOTS_THEN])})
    end = dict(constants={str(H.END_STEP): dict(constant_rms=7.0,
                                                slots_ordered=[[s, 5.0 + k] for k, s in enumerate(SLOTS_NOW)])})
    assert H.growth(end, start) == 2.0
    assert H.growth(end, start, 5) == 5.0 and H.growth(end, start, 0) == 0.0


def test_the_timeline_band_is_never_narrower_than_rounding(H):
    """Review 2 (MINOR): where the two controls agree the band had zero width, so a 1e-7 difference read as a departure."""
    def rec(step, v):
        return dict(step=step, memory_starter=dict(memory=dict(constant_rms=v), x=dict(y=v)))
    streams = dict(control=[rec(1, 6.0), rec(2, 6.0)], control2=[rec(1, 6.0), rec(2, 6.0)],
                   detach_gaze_copy=[rec(1, 6.0 + 1e-7), rec(2, 6.0 + 1e-7)],
                   detach_both=[rec(1, 6.0), rec(2, 6.5)])
    t = H.timeline(streams, resolution=0.01)
    assert t["detach_gaze_copy"] == {}
    assert t["detach_both"]["x.y"]["first_step"] == 2 and t["detach_both"]["memory.constant_rms"]["first_step"] == 2


# ------------------------------------------------------------------------------------------------ the gates
def test_the_smoke_gates_the_readings_the_operator_required(H):
    """Review 2 (MAJOR): reports without attention, the motor layer's live units or vision's error passed every gate."""
    good = [block(i, H.START_STEP + 100 * (i + 1), detached=("gaze_copy", "efference_body")) for i in range(2)]
    assert H.required_readings(good)["S5"]
    for drop in ("attention", "motor_first_layer", "vision_error", "state_requests"):
        bad = [block(i, H.START_STEP + 100 * (i + 1), drop=(drop,)) for i in range(2)]
        assert not H.required_readings(bad)["S5"], drop
    no_slope = [block(0, H.START_STEP + 100), block(0, H.START_STEP + 200)]          # the second lacks its slope
    assert not H.required_readings(no_slope)["S5"]


def test_the_accounting_holds_every_site_the_state_and_the_cut(H):
    both = ("gaze_copy", "efference_body")
    ok = [block(i, 1 + i, detached=both) for i in range(2)]
    assert H.accounting(ok, detach=both)["S1"] and H.accounting(ok, detach=both)["S2"]
    assert not H.accounting([block(0, 1, detached=both, leak=2e-4)], detach=both)["S1"]
    assert not H.accounting([block(0, 1, detached=both, state_leak=2e-4)], detach=both)["S1"]
    assert not H.accounting([block(0, 1)], detach=both)["S2"]                              # nothing cut
    assert not H.accounting([block(0, 1, detached=("gaze_copy",))], control=True)["S2"]    # a control that cuts
    removed = [block(0, 1, removed=("gaze_copy",))]
    assert H.accounting(removed, remove=("gaze_copy",))["S2"]
    assert not H.accounting([block(0, 1)], remove=("gaze_copy",))["S2"]                   # the removed reader reports


def test_every_world_of_every_launch_must_carry_the_fixed_layout(H, tmp_path):
    """S6 on recorded trainer runs, each linked to the launch records its stats gained (the re-review's MAJOR finding).
    The negative cases of the first round still fail: a missing world line, another digest, two launches with one log;
    a run that did not launch is not held to the 4 worlds, and records that cannot be linked fail."""
    def log(name, text):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return str(p)
    good = log("good.log", "".join(LAYOUT % (w, USUAL) for w in range(4)) + LAYOUT.replace("_w%d", "_preflight") % USUAL)
    other = log("other.log", "".join(LAYOUT % (w, "2daece73bcc3994c" if w == 1 else USUAL) for w in range(4)))
    silent = log("silent.log", "resume: step 344400\n")        # the default switch logs nothing
    partial = log("partial.log", "".join(LAYOUT % (w, USUAL) for w in range(3)))
    t = time.time()

    def runs(*logs):                  # run i started at t + 10 i, with the stats' launch count then
        return [dict(log=lg, launches_before=n, started=t + 10 * i) for i, (lg, n) in enumerate(logs)]

    def launches(*at):                # the stats' launch records, written `at` seconds after t
        return [dict(type="launch", step=344400 + 500 * i, timestamp=t + a) for i, a in enumerate(at)]
    assert H.touch_layout(runs((good, 0)), launches(1))["S6"]
    assert not H.touch_layout(runs((other, 0)), launches(1))["S6"]                         # another digest
    assert not H.touch_layout(runs((silent, 0)), launches(1))["S6"]                        # no world line at all
    assert not H.touch_layout(runs((partial, 0)), launches(1))["S6"]                       # a missing world line
    assert not H.touch_layout(runs((good, 0)), launches(1, 11))["S6"]                      # two launches, one log
    assert not H.touch_layout(runs((good, 0), (silent, 1)), launches(1, 11))["S6"]         # the second launch's log
    assert H.touch_layout(runs((good, 0), (good, 1)), launches(1, 11))["S6"]
    # a run stopped while building its worlds wrote no launch record: its partial log is not held to the 4 worlds
    kept = H.touch_layout(runs((good, 0), (partial, 1), (good, 1)), launches(1, 21))
    assert kept["S6"] and kept["not_launched_runs"] == ["partial.log"], kept
    # ... but another layout in any log fails, launched or not
    assert not H.touch_layout(runs((good, 0), (other, 1), (good, 1)), launches(1, 21))["S6"]
    # records that cannot be linked fail
    assert not H.touch_layout(runs((good, 1)), launches(1, 11))["S6"]          # a launch before the first recorded run
    assert not H.touch_layout([dict(log=good, started=t)], launches(1))["S6"]  # a run recorded without its count
    assert not H.touch_layout(runs((good, 0)), launches(-100))["S6"]           # a launch before its run started
    assert not H.touch_layout([], launches(1))["S6"] and not H.touch_layout([], [])["S6"]


def test_an_arm_ended_from_outside_after_its_first_save_continues_and_passes_its_checks(H, monkeypatch):
    """The re-review's MAJOR finding, its first probe: the harness is ended from outside after the arm's first saves (the
    watchdog on a pause lift ends the whole tree), and the arm's job runs again and continues from 345,000. The ended
    run is in attempt.json before its trainer started, so the continued arm's 2 launches have 2 logs and arm_checks
    passes. Without the killed run's record (the re-reviewed harness) the same records fail."""
    monkeypatch.setattr(H, "wait_for_guard", lambda need, wait_s=900.0: dict(admitted=True))
    prep = prep_of(H)
    folder = H.arms_root() / H.TAGS["control"]
    folder.mkdir(parents=True)
    seen = []
    fake, calls = stand_in_trainer(H, "control", [
        dict(launch=H.START_STEP, reports=reports(H.START_STEP + 100, 345001), saves=[344500, 345000], end="killed"),
        dict(launch=345000, reports=reports(345100, H.END_STEP) + [H.END_STEP],
             saves=list(range(345500, H.END_STEP, 500)) + [H.END_STEP])], seen)
    monkeypatch.setattr(H, "run_measured", fake)
    with pytest.raises(Killed):
        H.run_trainer(prep, "control", folder, H.UPDATES, H.SAVE_EVERY, "arm", H.TAGS["control"])
    first = seen[0]["segments"]
    assert first and first[-1]["log"] == calls[0][1] and first[-1]["launches_before"] == 0, first
    assert first[-1]["state"] == "running" and first[-1]["started"] <= time.time()
    res = H.run_trainer(prep, "control", folder, H.UPDATES, H.SAVE_EVERY, "arm", H.TAGS["control"])
    assert res["complete"] and calls[1][0][calls[1][0].index("--resume") + 1].endswith("ckpt_step_345000")
    receipt = dict(res, commit=COMMIT, harness=dict(blob=BLOB), updates=H.UPDATES, finished=time.time())
    chk = H.arm_checks("control", prep, receipt)
    assert chk["ok"], chk["problems"]
    assert chk["launch"]["resumed_at"] == [345000]
    assert [r["ended_from_outside"] for r in chk["touch_layout"]["runs"]] == [True, False]
    # the re-reviewed harness left the ended run unrecorded: 2 launches, 1 log
    lost = H.arm_checks("control", prep, dict(receipt, segments=receipt["segments"][1:]))
    assert not lost["ok"] and any("precede" in p for p in lost["touch_layout"]["problems"]), lost["problems"]


def test_a_continued_launch_stopped_while_building_its_worlds_is_not_held_to_them(H, monkeypatch):
    """The re-review's MAJOR finding, its second probe: the harness stops the arm at its memory reserve after saves, the
    continued launch is stopped again while 2 of its 4 worlds are built (no launch record, nothing trained), and the
    job runs again and completes from 345,500. arm_checks passes; the same partial log in a run that DID launch fails."""
    monkeypatch.setattr(H, "wait_for_guard", lambda need, wait_s=900.0: dict(admitted=True))
    prep = prep_of(H)

    def run(folder, second_launches):
        folder.mkdir(parents=True)
        fake, calls = stand_in_trainer(H, "control", [
            dict(launch=H.START_STEP, reports=reports(H.START_STEP + 100, 345701), saves=[344500, 345000, 345500],
                 end="reserve"),
            dict(worlds=2, launch=345500 if second_launches else None, end="reserve"),
            dict(launch=345500, reports=reports(345600, H.END_STEP) + [H.END_STEP],
                 saves=list(range(346000, H.END_STEP, 500)) + [H.END_STEP])])
        monkeypatch.setattr(H, "run_measured", fake)
        job1 = H.run_trainer(prep, "control", folder, H.UPDATES, H.SAVE_EVERY, "arm", H.TAGS["control"])
        assert not job1["complete"] and len(calls) == 2
        res = H.run_trainer(prep, "control", folder, H.UPDATES, H.SAVE_EVERY, "arm", H.TAGS["control"])
        assert res["complete"] and calls[2][0][calls[2][0].index("--resume") + 1].endswith("ckpt_step_345500")
        return H.arm_checks("control", prep, dict(res, commit=COMMIT, harness=dict(blob=BLOB), updates=H.UPDATES))
    chk = run(H.arms_root() / "c1", second_launches=False)
    assert chk["ok"], chk["problems"]
    assert len(chk["touch_layout"]["not_launched_runs"]) == 1 and chk["launch"]["resumed_at"] == [345500]
    launched = run(H.arms_root() / "c1b", second_launches=True)      # a missing world line in a run that launched
    assert not launched["ok"] and any("worlds [0, 1]" in p for p in launched["touch_layout"]["problems"])


# ------------------------------------------------------------------------------------------------ the judge, end to end
def test_the_judge_reads_the_declared_verdict_on_clean_arms_and_flags_a_resume(H, monkeypatch):
    four(H, control__resumed=(346000, True))
    code, j = judged(H, monkeypatch)
    assert code == 0 and j["overall"]["verdict"] == "the gaze copy's training is the starter", j["preconditions"]
    assert j["resumed"] == {"control": [346000]}
    assert j["preconditions"]["passed"]


@pytest.mark.parametrize("breakage,why", [
    (dict(detach_both__leak_at=344600), "not accounted"),
    (dict(detach_gaze_copy__config_detached=["efference_body"]), "names detached"),
    (dict(detach_both__commit="f" * 40), "code"),
    (dict(detach_efference_body__digest="0" * 64), "different trained weights"),
    (dict(control2__layout_digest="2daece73bcc3994c"), "fixed layout"),
    (dict(control__resumed=(346000, False)), "not an exact continuation"),
])
def test_the_judge_rechecks_every_arm_and_refuses_a_verdict_when_one_fails(H, monkeypatch, breakage, why):
    """Review 2 (MAJOR): the judge compared commands only. It now rechecks, from each arm's own records, the accounting
    and the cut at every report, the launch's switch record, one code commit, one start digest, the fixed layout in
    every world, and that every resume restored the saved state; any failure is NO VERDICT with its reason."""
    four(H, **breakage)
    code, j = judged(H, monkeypatch)
    assert code == 4 and j["overall"]["verdict"].startswith("NO VERDICT"), j["overall"]["verdict"]
    assert any(why in r for r in j["preconditions"]["reasons"]), j["preconditions"]["reasons"]


@pytest.mark.parametrize("breakage", [dict(detach_both__leak_at=344600),
                                      dict(detach_gaze_copy__config_detached=["efference_body"])])
def test_a_failed_precondition_writes_the_same_no_verdict_into_every_slot(H, monkeypatch, breakage):
    """The re-review (SEMANTIC), its two probe cases: the both-cut arm unaccounted at one report, or the gaze-copy arm's
    launch naming another cut. The overall read NO VERDICT while all six slots still named the gaze copy. Every slot now
    carries the overall NO VERDICT and keeps its bands, for reading only."""
    four(H, **breakage)
    code, j = judged(H, monkeypatch)
    nv = j["overall"]["verdict"]
    assert code == 4 and nv.startswith("NO VERDICT")
    assert len(j["per_slot"]) == 6
    for s, v in j["per_slot"].items():
        assert v["verdict"] == nv and v["valid"] is False, (s, v["verdict"])
        assert v["verdict_before_preconditions"] == "the gaze copy's training is the starter"
        assert v["bands"] == dict(detach_gaze_copy="STOPPED", detach_efference_body="CONTINUED", detach_both="STOPPED")


@pytest.mark.parametrize("remove,value,expected", [
    ("remove_gaze_copy", 2.0, "NO VERDICT: contradicting bands"),        # REDUCED beside a STOPPED detach
    ("remove_gaze_copy", 4.0, "NO VERDICT: contradicting bands"),        # CONTINUED beside a STOPPED detach
    ("remove_gaze_copy", 0.1, "direct"),
    ("remove_efference_body", 0.1, "indirect"),
    ("remove_efference_body", 4.05, "the reader's forward signal is not needed for the growth either (CONTINUED)"),
])
def test_the_fifth_arm_reads_a_removal_that_leaves_growth_beside_a_stopped_detach_as_a_contradiction(
        H, monkeypatch, remove, value, expected):
    """Review A's MINOR, carried by the re-review: a REDUCED removal beside a STOPPED detach read "the reader's forward
    signal is not needed for the growth either". Removing the forward signal removes its gradient path too, so it is a
    contradiction and reads NO VERDICT; the other readings are unchanged."""
    g = four(H)
    g[remove] = write_arm(H, remove, value)
    write_readings(H, g)
    code, j = judged(H, monkeypatch)
    assert code == 0 and j["overall"]["valid"], j["preconditions"]
    assert j["fifth_arm"][remove]["reading"].startswith(expected), j["fifth_arm"]


def test_one_code_commit_across_every_arm_the_fifth_included(H, monkeypatch):
    """The judge's own one-commit check (the re-review: redundant for the four arms, whose own checks catch a second
    commit) is what holds a fifth arm to the four's commit: its receipt on another commit makes the whole reading NO
    VERDICT, not only the fifth arm's line."""
    g = four(H)
    g["remove_gaze_copy"] = write_arm(H, "remove_gaze_copy", 0.1, commit="f" * 40)
    write_readings(H, g)
    code, j = judged(H, monkeypatch)
    assert code == 4 and any("more than one code commit" in r for r in j["preconditions"]["reasons"]), j["preconditions"]


def test_the_fifth_arm_carries_the_overall_no_verdict_when_a_precondition_fails(H, monkeypatch):
    g = four(H, detach_both__leak_at=344600)
    g["remove_gaze_copy"] = write_arm(H, "remove_gaze_copy", 0.1)
    write_readings(H, g)
    code, j = judged(H, monkeypatch)
    assert code == 4 and j["fifth_arm"]["remove_gaze_copy"]["reading"] == j["overall"]["verdict"]


def test_the_judge_refuses_readings_older_than_an_arm_receipt(H, monkeypatch):
    """The re-review (MINOR): an arm run again (--force) after its readings was judged with its old reading. Every arm
    receipt must be written before the readings started."""
    four(H)
    code, j = judged(H, monkeypatch)
    assert code == 0, j["preconditions"]
    rd = json.loads((H.SCRATCH / "readings.json").read_text(encoding="utf-8"))
    path = H.arms_root() / H.TAGS["detach_both"] / "arm_receipt.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["finished"] = rd["started"] + 5.0
    path.write_text(json.dumps(rec), encoding="utf-8")
    code, j = judged(H, monkeypatch)
    assert code == 4 and j["overall"]["verdict"].startswith("NO VERDICT")
    assert any("older than the arm receipts of ['detach_both']" in r for r in j["preconditions"]["reasons"])
    rd.pop("started")                      # readings from before the fix carry no time: refused too
    (H.SCRATCH / "readings.json").write_text(json.dumps(rd), encoding="utf-8")
    code, j = judged(H, monkeypatch)
    assert code == 4 and any("unrecorded time" in r for r in j["preconditions"]["reasons"])


def test_the_judge_refuses_readings_from_another_reader(H, monkeypatch):
    four(H)
    rd = json.loads((H.SCRATCH / "readings.json").read_text(encoding="utf-8"))
    rd["start"]["reader_commit"] = "9" * 40
    (H.SCRATCH / "readings.json").write_text(json.dumps(rd), encoding="utf-8")
    code, j = judged(H, monkeypatch)
    assert code == 4 and any("another reader" in r for r in j["preconditions"]["reasons"])


# ------------------------------------------------------------------------------------------------ the queue's steps
def test_a_cut_arm_refuses_without_the_controls_pass(H, monkeypatch):
    """Review 2 (MAJOR): the three cut arms ran even when the controls ruled out a verdict."""
    monkeypatch.setattr(H, "_prepared", lambda: prep_of(H))
    (H.SCRATCH / "smoke_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(blob=BLOB))),
                                                  encoding="utf-8")
    ran = []
    monkeypatch.setattr(H, "run_trainer", lambda *a, **k: ran.append(a) or dict(complete=True, command_first=[],
                                                                              checkpoints=[], ended_at_deadline=False))
    for name in ("detach_gaze_copy", "detach_efference_body", "detach_both", "remove_gaze_copy"):
        with pytest.raises(SystemExit, match="controls|fifth arm"):
            H.arm(name)
    assert not ran
    H.arms_root().mkdir(parents=True, exist_ok=True)
    (H.arms_root() / "controls_receipt.json").write_text(json.dumps(dict(ok=False, commit=COMMIT, harness=dict(
        blob=BLOB), updates=H.UPDATES, reason="NO VERDICT: the controls' growth ...")), encoding="utf-8")
    with pytest.raises(SystemExit, match="controls"):
        H.arm("detach_both")
    (H.arms_root() / "controls_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(
        blob=BLOB), updates=H.UPDATES)), encoding="utf-8")
    assert H.arm("detach_both") == 0 and len(ran) == 1
    H.arm("control")                                    # the controls themselves never wait for it
    assert len(ran) == 2


def test_the_controls_gate_holds_its_length_and_the_smokes_start_weights(H, monkeypatch):
    """The re-review (MINOR): no test drove the controls' length and digest checks. A passing controls' receipt for
    another length admits no cut arm; a control that began from other trained weights than the smoke's fails the
    controls' check."""
    monkeypatch.setattr(H, "_prepared", lambda: prep_of(H))
    (H.SCRATCH / "smoke_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(blob=BLOB),
                                                                  start_digest="d" * 64)), encoding="utf-8")
    ran = []
    monkeypatch.setattr(H, "run_trainer", lambda *a, **k: ran.append(a) or dict(complete=True, command_first=[],
                                                                              checkpoints=[], ended_at_deadline=False))
    H.arms_root().mkdir(parents=True, exist_ok=True)
    (H.arms_root() / "controls_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(
        blob=BLOB), updates=H.UPDATES + 500)), encoding="utf-8")
    with pytest.raises(SystemExit, match="length"):
        H.arm("detach_both")
    assert not ran
    four(H, control2__digest="0" * 64)
    monkeypatch.setattr(H, "read_records", lambda *a, **k: pytest.fail("the check read saves after a failed digest"))
    assert H.controls() == 4
    rec = json.loads((H.arms_root() / "controls_receipt.json").read_text(encoding="utf-8"))
    assert not rec["ok"] and "differ between the smoke and the controls" in rec["reason"], rec["reason"]


def controls_reader(H, values, slot_values=None):
    """read_records replaced: the start at 5.0 per number in every slot, each control's end at 5.0 plus its growth
    (per slot where given)."""
    def reader(root, out, reader, steps=None):
        if Path(root) == H.START_ROOT:
            return dict(exit=0, reader_commit=READER, constants={H.START_STEP: dict(
                constant_rms=5.0, slots_ordered=[[s, 5.0] for s in SLOTS_THEN])})
        name = {H.TAGS[n]: n for n in H.CONTROLS}[Path(root).name]
        per = (slot_values or {}).get(name, {})
        return dict(exit=0, reader_commit=READER, constants={H.END_STEP: dict(
            constant_rms=5.0 + values[name], slots_ordered=[[s, 5.0 + per.get(s, 0.0)] for s in SLOTS_NOW])})
    return reader


def test_the_controls_check_passes_only_when_the_controls_can_give_a_verdict(H, monkeypatch):
    monkeypatch.setattr(H, "_prepared", lambda: prep_of(H))
    for growth_c2, ok in ((4.1, True), (0.2, False)):
        four(H)
        (H.SCRATCH / "smoke_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(blob=BLOB),
                                                                      start_digest="d" * 64)), encoding="utf-8")
        monkeypatch.setattr(H, "read_records", controls_reader(H, dict(control=4.0, control2=growth_c2)))
        code = H.controls()
        rec = json.loads((H.arms_root() / "controls_receipt.json").read_text(encoding="utf-8"))
        assert (code == 0) == ok and rec["ok"] == ok, rec["reason"]
        if not ok:
            assert rec["reason"].startswith("NO VERDICT") and rec["next_test"]["updates"] > H.UPDATES


def test_the_controls_receipt_names_each_slots_valid(H, monkeypatch):
    """The re-review (SEMANTIC): the controls' check gated the overall constant only. Its receipt now names each slot's
    VALID, so a slot that cannot decide is known before the cut arms run; the overall VALID alone gates them. Here
    touch grew 0.5 per number in both controls, under twice its floor (0.976)."""
    monkeypatch.setattr(H, "_prepared", lambda: prep_of(H))
    four(H)
    (H.SCRATCH / "smoke_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(blob=BLOB),
                                                                  start_digest="d" * 64)), encoding="utf-8")
    per = {s: 4.0 for s in SLOTS_NOW}
    per["tactile"] = 0.5
    monkeypatch.setattr(H, "read_records", controls_reader(H, dict(control=4.0, control2=4.1),
                                                           dict(control=per, control2=dict(per, vision=4.1))))
    assert H.controls() == 0
    rec = json.loads((H.arms_root() / "controls_receipt.json").read_text(encoding="utf-8"))
    assert rec["ok"] and rec["slots_that_cannot_decide"] == ["tactile"], rec.get("per_slot")
    t = rec["per_slot"]["tactile"]
    assert t["valid"] is False and abs(t["needs_growth_above"] - 2 * H.floor_of(1)) < 1e-12
    assert t["declared_share_of_8_body_growth_needed"] == 40 and t["slot_at_start"] == "tactile"
    assert rec["per_slot"]["vestibular"]["slot_at_start"] == "intermind" and rec["per_slot"]["vision"]["valid"]


def test_the_readings_read_the_start_again_with_the_pinned_reader(H, monkeypatch):
    """Review A and review 2 (MAJOR): the start was read once, at prepare, by the live working tree's reader. The
    readings also record when they started, for the judge's stale-readings check."""
    prep = dict(prep_of(H), start_reading=dict(constants={H.START_STEP: dict(constant_rms=999.0)}))
    monkeypatch.setattr(H, "_prepared", lambda: prep)
    four(H)
    calls = []

    def reader(root, out, reader, steps=None):
        calls.append((Path(root), reader["commit"], steps))
        return dict(exit=0, reader_commit=reader["commit"], constants={H.START_STEP: dict(constant_rms=5.0)})
    monkeypatch.setattr(H, "read_records", reader)
    t0 = time.time()
    assert H.readings() == 0
    out = json.loads((H.SCRATCH / "readings.json").read_text(encoding="utf-8"))
    assert calls[0] == (H.START_ROOT, READER, [H.START_STEP])
    assert out["start"]["constants"][str(H.START_STEP)]["constant_rms"] == 5.0
    assert all(c[1] == READER for c in calls) and len(calls) == 1 + 5
    assert t0 <= out["started"] <= out["finished"] <= time.time()


def test_force_starts_the_arm_again_from_the_copy(H, monkeypatch):
    """Review A (MINOR): --force on a complete arm resumed its own last save and trained 0 updates."""
    monkeypatch.setattr(H, "_prepared", lambda: prep_of(H))
    (H.SCRATCH / "smoke_receipt.json").write_text(json.dumps(dict(ok=True, commit=COMMIT, harness=dict(blob=BLOB))),
                                                  encoding="utf-8")
    folder = H.arms_root() / H.TAGS["control"]
    (folder / "old_run").mkdir(parents=True)
    (folder / "attempt.json").write_text("{}", encoding="utf-8")
    (folder / "arm_receipt.json").write_text(json.dumps(dict(complete=True, commit=COMMIT, harness=dict(blob=BLOB),
                                                             updates=H.UPDATES)), encoding="utf-8")
    seen = []
    monkeypatch.setattr(H, "run_trainer", lambda prep, name, f, *a: seen.append(sorted(p.name for p in f.iterdir()))
                        or dict(complete=True, command_first=[], checkpoints=[], ended_at_deadline=False))
    assert H.arm("control") == 0 and seen == []            # kept without --force
    assert H.arm("control", force=True) == 0 and seen == [[]]


def test_every_step_refuses_a_harness_file_other_than_prepares(H, monkeypatch):
    """Review 2 (MINOR): the declared reading lived in a working-tree file nothing pinned."""
    (H.SCRATCH / "prepare_receipt.json").write_text(json.dumps(prep_of(H)), encoding="utf-8")
    monkeypatch.setattr(H, "_git", lambda *a, **k: COMMIT)
    monkeypatch.setattr(H, "harness_identity", lambda: dict(blob=BLOB, committed=True))
    assert H._prepared()["commit"] == COMMIT
    monkeypatch.setattr(H, "harness_identity", lambda: dict(blob="0" * 40, committed=True))
    with pytest.raises(SystemExit, match="harness file changed"):
        H._prepared()


def test_the_start_is_proved_to_be_the_checkpoint_segment_32_resumed(H, monkeypatch, tmp_path):
    """The archiver's move record names segment 32's resume path as its source and each file's sha256; a changed file
    or another source is refused."""
    ck = tmp_path / "archive" / "ckpt_step_344400"
    ck.mkdir(parents=True)
    (ck / "lineage.json").write_text(json.dumps(dict(step=344400, code_commit="30c0c7a8e")), encoding="utf-8")
    (ck / "model.safetensors").write_bytes(b"weights")
    c_path = "C:\\run\\seg31\\checkpoints\\ckpt_step_344400"
    first = tmp_path / "seg32" / "ckpt_step_344525"
    first.mkdir(parents=True)
    (first / "lineage.json").write_text(json.dumps(dict(code_commit="637a3d00c", resumed_from=c_path)), encoding="utf-8")
    receipt = tmp_path / "continuation_receipt.json"
    receipt.write_text(json.dumps(dict(checkpoint=c_path, command=["PY", "-m", "mimo_bridge.train_mimo"])),
                       encoding="utf-8")
    manifest = tmp_path / "move_manifest.jsonl"
    row = dict(event="moved", source=c_path, destination=str(ck),
               files={"model.safetensors": H._sha256(ck / "model.safetensors"),
                      "lineage.json": H._sha256(ck / "lineage.json")})
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    for name, value in (("START_CKPT", ck), ("FIRST_DEPLOY_SAVE", first), ("TRAINED_BY", receipt),
                        ("MOVE_MANIFEST", manifest)):
        monkeypatch.setattr(H, name, value)
    assert H.start_identity()["ok"]
    (ck / "model.safetensors").write_bytes(b"other weights")
    bad = H.start_identity()
    assert not bad["ok"] and any("differs" in p for p in bad["problems"])


def test_the_jobs_end_their_trainer_before_the_runner_does(H):
    """Review 2 (MAJOR): the harness takes a deadline below its job's timeout (the runner ends the tree at the timeout,
    20fcd3c7c); every heavy job's --deadline-s is at least 300 s under its timeout, and the cut arms come after the
    controls' check."""
    jobs = json.loads(JOBS[0].read_text(encoding="utf-8"))
    names = [j["name"] for j in jobs]
    assert names.index("memory_starter_controls") > names.index("memory_starter_arm_control2")
    assert all(names.index("memory_starter_controls") < names.index("memory_starter_arm_" + n) for n in H.CUTS)
    for j in jobs:
        if j["need_gb"] >= 10 and j["name"] != "memory_starter_prepare":
            d = float(j["cmd"][j["cmd"].index("--deadline-s") + 1])
            assert d <= j["timeout_s"] - 300, j["name"]


def test_no_job_gives_up_while_another_runners_heavy_job_holds_the_lock(H):
    """The coordinator (Sep 26, 9 PM PDT): every job's guard_wait_s at least 10,800 s, and at least an arm's own timeout
    (16,000 s), so a job waits out another runner's heavy job, an arm of this test included, instead of being refused
    (the runner waits guard_wait_s for the heavy-job lock and the memory guard before the job starts)."""
    for path in JOBS:
        for j in json.loads(path.read_text(encoding="utf-8")):
            assert j["guard_wait_s"] >= max(10800, H.ARM_TIMEOUT_S), (path.name, j["name"], j["guard_wait_s"])


def test_the_deadline_ends_the_trainers_whole_tree(H, monkeypatch, tmp_path):
    """A child that starts a grandchild and sleeps: at the deadline the harness ends both and records it."""
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(H, "_guard", lambda: SimpleNamespace(commit_reading=lambda: None))   # no reserve stop here
    monkeypatch.setattr(H, "DEADLINE_S", 1.0)
    monkeypatch.setattr(H, "T0", time.time())
    monkeypatch.setattr(H, "SAMPLE_EVERY_S", 0.2)
    pid_file = tmp_path / "grandchild.pid"
    child = ("import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
             "open(r'%s', 'w').write(str(p.pid)); time.sleep(60)" % pid_file)
    t0 = time.time()
    code, rec = H.run_measured([sys.executable, "-c", child], tmp_path, None, tmp_path / "run.log")
    assert time.time() - t0 < 30 and rec["ended_at_deadline"]
    assert rec["ended_at_deadline"]["alive_30_s_after_the_end"] == []
    deadline = time.time() + 10
    gp = int(pid_file.read_text()) if pid_file.exists() else None
    while gp is not None and psutil.pid_exists(gp) and time.time() < deadline:
        time.sleep(0.2)
    assert gp is None or not psutil.pid_exists(gp) or psutil.Process(gp).status() == psutil.STATUS_ZOMBIE


def test_a_process_alive_after_the_end_is_recorded(H, monkeypatch, tmp_path):
    """The re-review's note: after ending a trainer's tree the harness waited 30 s and went on silently whether or not
    every process had exited. A process still alive then is now recorded with the stop."""
    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(H, "_guard", lambda: SimpleNamespace(commit_reading=lambda: None))
    monkeypatch.setattr(H, "DEADLINE_S", 0.5)
    monkeypatch.setattr(H, "T0", time.time())
    monkeypatch.setattr(H, "SAMPLE_EVERY_S", 0.1)
    real = psutil.wait_procs

    def wait_procs(procs, timeout=None):              # one ended process reported still alive
        gone, alive = real(procs, timeout=timeout)
        return gone, list(alive) + [SimpleNamespace(pid=999999)]
    monkeypatch.setattr(psutil, "wait_procs", wait_procs)
    code, rec = H.run_measured([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, None,
                               tmp_path / "run.log")
    assert rec["ended_at_deadline"]["alive_30_s_after_the_end"] == [999999]


# ------------------------------------------------------------------------------------------------ the smoke, end to end
def runtime_state(step, detached, removed=(), with_resume=True, windows=REPAIRED):
    """A save's runtime state as LemLite.save_checkpoint writes it (the parts resume_summary reads)."""
    rt = {"vis_adv_ema": 0.0, "liveness_repair": {k: {"centre_mean": [0.0], "centre_weight": 1.0} for k in REPAIRED}}
    if with_resume:
        rt["memory_starter_resume"] = dict(
            version="memory_starter_v2", step=step,
            starter=dict(detached=sorted(detached), removed=sorted(removed), first=(344401, [6.0] * 6, 6.0),
                         previous=None, c_hat=None, window={}),
            liveness_windows={k: dict(seen_max=None, act_sum=None, seen_n=7, probe_x=None) for k in windows})
    else:
        rt["memory_starter_resume_error"] = "RuntimeError: a stand-in failure"
    return rt


def smoke_setup(H, monkeypatch, plan, name="detach_both", skins=True, no_resume_at=(), windows=REPAIRED):
    """smoke() with the trainer, the pinned reader and the resume-state reader replaced; returns the trainer's calls."""
    prep = dict(prep_of(H), start_skins=dict(usual_layout_in_every_world=skins))
    monkeypatch.setattr(H, "_prepared", lambda: prep)
    monkeypatch.setattr(H, "wait_for_guard", lambda need, wait_s=900.0: dict(admitted=True))
    fake, calls = stand_in_trainer(H, name, plan)
    monkeypatch.setattr(H, "run_measured", fake)

    def reader(root, out, reader, steps=None):
        cks = sorted(Path(root).glob("*/checkpoints/ckpt_step_*"))
        return dict(exit=0, reader_commit=READER, constants={H._step_of(c): dict(constant_rms=6.0) for c in cks})
    monkeypatch.setattr(H, "read_records", reader)
    det, rem = H._detach_set(name)

    def resume_reader(prep, store, out):
        rows = [dict(checkpoint=str(c), step=H._step_of(c),
                     **H.resume_summary(runtime_state(H._step_of(c), det, rem, H._step_of(c) not in no_resume_at,
                                                      windows)))
                for c in H.checkpoints_in(store)]
        return dict(exit=0, saves=rows)
    monkeypatch.setattr(H, "read_resume_states", resume_reader)
    return calls


def smoke_plan(**over):
    p = dict(launch=344400, reports=[344500, 344600], saves=[344500, 344600])
    p.update(over)
    return [p]


def smoke_receipt(H):
    return json.loads((H.SCRATCH / "smoke_receipt.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case,gate", [
    ("clean", None),
    ("a report without attention", "S5_required_readings"),
    ("the copy's saved skins another layout", "S6_every_world_the_fixed_layout"),
    ("a save without the resume state", "S8_every_save_carries_the_resume_state"),
    ("the liveness windows short of a repaired layer", "S8_every_save_carries_the_resume_state"),
    ("3.0 s per update", "S9_an_arm_fits_its_job"),
    ("the process tree at 27 GB", "S9_an_arm_fits_its_job"),
])
def test_the_smokes_pass_holds_every_gate(H, monkeypatch, case, gate):
    """The re-review (MINOR): no test drove smoke()'s pass (planted: S5 or the copy's skins left out of it went
    unnoticed); a save without the starter's resume state showed only at a resume; the projection of an arm's time
    and memory was no part of the pass. smoke() runs here with the trainer and the readers replaced: each case fails
    exactly its gate and the smoke, and the clean case passes every gate."""
    over = {"a report without attention": dict(drop=("attention",)), "3.0 s per update": dict(per_update_s=3.0),
            "the process tree at 27 GB": dict(peak_gb=27.0)}.get(case, {})
    smoke_setup(H, monkeypatch, smoke_plan(**over), skins=case != "the copy's saved skins another layout",
                no_resume_at=(344600,) if case == "a save without the resume state" else (),
                windows=REPAIRED[:1] if case == "the liveness windows short of a repaired layer" else REPAIRED)
    code = H.smoke()
    rec = smoke_receipt(H)
    gates = {k: v for k, v in rec["gates"].items() if k.startswith("S")}
    assert set(gates) == {"S1_every_site_and_the_state_accounted", "S2_the_cut_as_declared", "S3_every_reader_reports",
                          "S4_saves_read", "S5_required_readings", "S6_every_world_the_fixed_layout",
                          "S7_launch_names_cut_digest_commit", "S8_every_save_carries_the_resume_state",
                          "S9_an_arm_fits_its_job"}
    if gate is None:
        assert code == 0 and rec["ok"] and all(gates.values()), gates
        assert abs(rec["projection"]["seconds_per_update"] - 2.0) < 1e-9 and rec["projection"]["within_bounds"]
    else:
        assert code == 3 and not rec["ok"], gates
        assert [k for k, v in gates.items() if not v] == [gate], gates


def test_the_smoke_continues_after_an_outside_kill_keeps_a_pass_and_force_starts_again(H, monkeypatch):
    """The re-review (MINOR): a smoke continued after an outside kill failed S6 and had no --force. Here the first run
    is ended from outside after the save at 344,500; the job again continues from it and passes (the projection
    counts no time between the runs); run again, the pass is kept; --force starts again from the copy."""
    calls = smoke_setup(H, monkeypatch, [
        dict(launch=344400, reports=[344500], saves=[344500], end="killed"),
        dict(launch=344500, reports=[344600], saves=[344600]),
        dict(launch=344400, reports=[344500, 344600], saves=[344500, 344600])])
    with pytest.raises(Killed):
        H.smoke()
    time.sleep(0.05)
    assert H.smoke() == 0, smoke_receipt(H)["gates"]
    rec = smoke_receipt(H)
    assert calls[1][0][calls[1][0].index("--resume") + 1].endswith("ckpt_step_344500")
    assert abs(rec["projection"]["seconds_per_update"] - 2.0) < 1e-9 and rec["launch"]["resumed_at"] == [344500]
    assert H.smoke() == 0 and len(calls) == 2                       # the pass is kept: no trainer ran
    assert H.smoke(force=True) == 0 and len(calls) == 3
    assert calls[2][0][calls[2][0].index("--resume") + 1] == prep_of(H)["start_checkpoint"]
    rec = smoke_receipt(H)
    assert rec["ok"] and len(rec["run"]["segments"]) == 1 and rec["launch"]["resumed_at"] == []


def test_the_projection_reads_the_trainers_own_rate_within_each_process(H, tmp_path):
    """S9's rate: from each launch record to its process's reports; the hours between an ended run and its continuation
    never count. Past 0.6 of the arm's timeout the projection fails."""
    stats = tmp_path / "stats.jsonl"
    recs = [dict(type="launch", step=344400, timestamp=0.0), dict(type="mimo", step=344500, timestamp=200.0),
            dict(type="launch", step=344500, timestamp=10000.0), dict(type="mimo", step=344600, timestamp=10200.0)]
    stats.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert abs(H.seconds_per_update(stats) - 2.0) < 1e-12
    seg = [dict(commit=dict(tree_private_peak_gb=21.2))]
    assert H.projection_of(stats, seg)["within_bounds"]
    line = H.PROJECTION_SHARE * H.ARM_TIMEOUT_S / H.DEFAULT_UPDATES          # 2.83 s per update
    recs[3]["timestamp"] = 10000.0 + 100 * (2 * line - 2.0) + 1.0            # the mean just past the line
    stats.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert not H.projection_of(stats, seg)["within_bounds"]
    assert not H.projection_of(tmp_path / "missing.jsonl", seg)["within_bounds"]
    recs[3]["timestamp"] = 10200.0                                           # 2 s per update again
    stats.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    assert H.projection_of(stats, seg)["within_bounds"]
    assert not H.projection_of(stats, [dict()])["within_bounds"]                # no peak measured
    assert not H.projection_of(stats, [dict(commit=dict(tree_private_peak_gb=27.0))])["within_bounds"]


def test_the_resume_state_reader_loads_each_save_as_the_trainer_does(H, monkeypatch, tmp_path):
    """S8's reader runs the harness's resume_states in a child process, which loads each runtime_state.pt with torch
    (weights_only=False, on the CPU) as LemLite.load_checkpoint does. A save written with memory_starter_resume passes;
    one written with memory_starter_resume_error fails with that error; another arm's state fails. Two child processes
    import torch on the CPU."""
    from types import SimpleNamespace
    monkeypatch.setattr(H, "_guard", lambda: SimpleNamespace(commit_reading=lambda: None))
    store = tmp_path / "checkpoints"
    make = "\n".join([
        "import sys, torch",
        "from pathlib import Path",
        "store = Path(sys.argv[1])",
        "rep = {k: {'centre_mean': torch.zeros(4), 'centre_weight': 1.0} for k in %r}" % (REPAIRED,),
        "for step, ok in ((344500, True), (344600, False)):",
        "    ck = store / ('ckpt_step_%d' % step)",
        "    ck.mkdir(parents=True)",
        "    (ck / 'model.safetensors').write_bytes(b'x')",
        "    rt = {'vis_adv_ema': 0.0, 'liveness_repair': rep}",
        "    if ok:",
        "        rt['memory_starter_resume'] = {'version': 'memory_starter_v2', 'step': step, 'starter': {",
        "            'detached': ['efference_body', 'gaze_copy'], 'removed': [], 'first': (344401, [6.0] * 6, 6.0),",
        "            'c_hat': torch.ones(6, 8)}, 'liveness_windows': {k: {'seen_max': torch.ones(4),",
        "            'act_sum': torch.ones(4), 'seen_n': 7, 'probe_x': None} for k in rep}}",
        "    else:",
        "        rt['memory_starter_resume_error'] = 'RuntimeError: a stand-in failure'",
        "    torch.save(rt, str(ck / 'runtime_state.pt'))"])
    subprocess.run([sys.executable, "-c", make, str(store)], check=True, timeout=300,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    prep = dict(prep_of(H), python_path=str(REPO / "Source" / "Python"))
    rs = H.read_resume_states(prep, store, tmp_path / "resume_states.json")
    assert rs["exit"] == 0 and [r["step"] for r in rs["saves"]] == [344500, 344600], rs
    good = H.resume_gate(rs["saves"], "detach_both", [344500])
    assert good["S8"], good
    assert rs["saves"][0]["liveness_window_layers"] == sorted(REPAIRED) and rs["saves"][0]["starter_first_step"] == 344401
    bad = H.resume_gate(rs["saves"], "detach_both", [344500, 344600])
    assert not bad["S8"] and any("stand-in failure" in p for p in bad["problems"]), bad
    assert not H.resume_gate(rs["saves"], "control", [344500])["S8"]           # another arm's state
    assert not H.resume_gate(rs["saves"], "detach_both", [345000])["S8"]       # a save that was not read
