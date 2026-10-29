# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE RECORDS LOCKSTEP'S JOB (docs/gain_of_function_v4/comparison_records/records_lockstep.py).

The job's own logic with the trainer arms replaced by a fake that writes what an arm writes (its world records, its
end-of-run checkpoint with an episode companion, its model lane, the branch's comparison records, its arm record): the
refusals before any copy, the verdicts (pass, fail, invalid), the repeat of a run every time and the repeat of b run
only when a equals its repeat and differs from b (v3), the deterministic mode's line and warnings read from each arm's
log (v3), the discard in its finally path,
the window of the bundle review's FIX-4 (model steps 481,800 to 481,805, the end save ckpt_step_481806) and the parts of
its FIX-5 (every checkpoint leaf, the companion's parts, the lane's and the world records' columns; the parts two
launches of one code differ in reported and not judged). The learning-on package it reuses lives on its own branch
(C:/Users/bhart/Documents/UEMA_wt/learning_on_stop_limits); the module is skipped when that folder is absent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[4]
JOB_FILE = REPO / "docs/gain_of_function_v4/comparison_records/records_lockstep.py"
LEARNING_ON = Path("C:/Users/bhart/Documents/UEMA_wt/learning_on_stop_limits/docs/gain_of_function_v4/"
                   "learning_on_stop_limits/learning_on.py")
if not LEARNING_ON.is_file():
    pytest.skip("the learning-on package's folder is absent", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("records_lockstep", JOB_FILE)
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)

HASH_A, HASH_B = "a" * 64, "b" * 64
N = J.STEPS                                       # 6 model steps: 481,800 to 481,805


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(tmp_path):
    ck = tmp_path / "ckpt" / "ckpt_step_481800"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"model numbers")
    companion = tmp_path / "companion.pt.gz"
    companion.write_bytes(b"episode")
    (ck / "episode_state.json").write_text(json.dumps(dict(storage_path=str(companion), sha256=sha(companion))))
    trees = dict(a=tmp_path / "code_a", b=tmp_path / "code_b")
    for t in trees.values():
        t.mkdir()
    a = SimpleNamespace(scratch=str(tmp_path / "job"), out=str(tmp_path / "result.json"), code_a=str(trees["a"]),
                        code_b=str(trees["b"]), code_b_hash=HASH_B, command=str(J.COMMAND), deadline_s=600.0)
    hashes = {str(trees["a"]): J.CODE_A_HASH, str(trees["b"]): HASH_B}
    return a, ck, sha(ck / "model.safetensors"), hashes


def write_world(run: Path, body: int, motor_shift=0.0, header="chunk_000481800_1_00000.json", qpos_shift=0):
    """One world chunk of model steps 481,800 to 481,805 in the recorder's command columns: commands latched at records
    0 and 5, the eye command per record, a quantized position column, the header's name beside it."""
    folder = run / "observer" / "world" / ("b%d" % body)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / header).write_text(json.dumps(dict(fields="fake")))
    g = np.random.default_rng(body)
    motor = g.normal(0, 10, (2, 26, 9)).astype(np.float32) + np.float32(motor_shift)
    np.savez(folder / "chunk_000481800_1_00000.npz", header=np.asarray(header),
             order_on=np.ones(N, bool), order_row=np.asarray([0, 5]), order_motor=motor,
             order_mask=np.full((2, 26), -1, np.int8), order_hand_goal=np.full((2, 44), np.nan, np.float32),
             order_gaze=g.normal(0, 0.1, (N, 2)).astype(np.float32), author=np.full(N, 3, np.int8),
             latched=np.asarray([1, 0, 0, 0, 0, 1], bool), order_seq=np.asarray([1, 1, 1, 1, 1, 2]),
             bout=np.ones(N, bool), qpos_raw=np.arange(N * 3, dtype=np.float64).reshape(N, 3) + qpos_shift)


def write_checkpoint(run: Path, weight_shift=0.0, nonfinite_steps=0, companion=None):
    from safetensors.torch import save_file
    ck = run / "checkpoints" / ("ckpt_step_%d" % J.END_SAVE)
    ck.mkdir(parents=True, exist_ok=True)
    save_file({"forward_model.w": torch.arange(6.0).reshape(2, 3) + weight_shift, "heads.b": torch.ones(4)},
              str(ck / "model.safetensors"))
    torch.save({"optimizer": {"state": {0: {"exp_avg": torch.full((3,), 0.5), "step": torch.tensor(7.0)}},
                              "param_groups": [{"lr": 1e-4, "params": [0]}]}}, str(ck / "optimizer.pt"))
    torch.save({"efference": {"centre": torch.zeros(5)}, "nonfinite_steps": nonfinite_steps,
                "vis_adv_ema": 0.25, "plan_win_ema": [0.5, 0.75]}, str(ck / "runtime_state.pt"))
    (ck / "fake_companion.json").write_text(json.dumps(companion))


def companion_of(run: Path, learner_rng="r0", worlds="w0"):
    """The fake companion: the parts continuation_state.load returns, with the arm's own folder in a text (the trainer
    part holds paths of the arm's folder; each arm runs in its own)."""
    return {"version": 3, "config": {"n_envs": 8},
            "learner": {"data": {"_last_plan_inputs": {"decision": {"parameter_versions": [[id(run), 4], [7, 9]]}},
                                 "_nf_ctx": [1.0, 2.0]},
                        "torch_rng": learner_rng, "objects": {}},
            "trainer": {"step": J.END_SAVE, "frame_meta": [{"record": str(run / "observer" / "world")}]},
            "worlds": [{"payload": worlds}]}


def read_fake_companion(folder, arm_root):
    """J.companion_parts' shape, read from the fake companion instead of continuation_state.load."""
    state = json.loads((Path(folder) / "fake_companion.json").read_text())
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            for sub, v in value.items():
                out["%s.%s" % (key, sub)] = J.value_digest(v, arm_root)
        else:
            out[key] = J.value_digest(value, arm_root)
    for part in ("data", "objects"):
        for name, v in (state["learner"].get(part) or {}).items():
            out["learner.%s.%s" % (part, name)] = J.value_digest(v, arm_root)
    return out


def write_lane(run: Path, records_columns: bool, pe_shift=0.0):
    lane = run / "observer" / "model"
    lane.mkdir(parents=True, exist_ok=True)
    cols = dict(step=np.arange(481800, 481800 + N), next_pe=np.full((N, 5), 0.01, np.float32),
                next_pe_body=np.full((N, 8, 5), 0.01, np.float32) + np.float32(pe_shift),
                wellness=np.full((N, 3, 8), 0.5, np.float32))
    if records_columns:
        cols.update(persist_pe_body=np.full((N, 8, 5), 0.02, np.float32),
                    vision_no_copy_body=np.full((N, 8), 0.3, np.float32), gaze_copy_applied=np.ones(N, np.int8))
    np.savez(lane / "chunk_000481800_1_00000.npz", **cols)


def write_stats(run: Path, nan_steps=0, audio_ratio=1.3617183717846408):
    """The arm's stats stream as the trainer writes it at the end of a 6-model-step run: a full record of model step
    481,806 (type mimo with prediction_audit_per_body), which learning_on.run_arm reads with record_stop. Its values
    are those of row records_lockstep_481800's arm A (audio 1361.7 in 1,000, above the learning-on file's 100-model-step
    gross line 1119.1)."""
    run.mkdir(parents=True, exist_ok=True)
    ratios = dict(tactile=0.9305916472820346, proprioception=0.5950041611278907, vision=0.7350867051561488,
                  audio=audio_ratio)
    rec = dict(type="mimo", step=481806, nan_steps=nan_steps, timestamp=1.0, elapsed_s=200.0,
               prediction_audit_per_body=dict(senses={s: dict(without_body_2=dict(ratio=v)) for s, v in ratios.items()}),
               hand_control=dict(joint_error_deg=15.137943457189943))
    with open(run / "replay_training_stats.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(type="startup", step=481800)) + "\n" + json.dumps(rec) + "\n")


def write_records(run: Path, steps=range(481800, 481800 + N)):
    folder = run / "comparison_records"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "header_1_2.json").write_text(json.dumps(dict(version="comparison_records_v1",
                                                             spike=dict(line=0.14, run=3))))
    (folder / "steps_1.jsonl").write_text("".join(json.dumps(dict(step=s, group_norms=dict(core=1.0))) + "\n"
                                                  for s in steps))
    (folder / "replay_passes_1.jsonl").write_text("".join(
        json.dumps(dict(step=s, windows=dict(body=[0, 3]))) + "\n" for s in (481802, 481804, 481806)))
    spikes = folder / "vision_spike" / "checkpoints" / "ckpt_step_481805"
    spikes.mkdir(parents=True)
    (spikes / "model.safetensors").write_bytes(b"spike weights")


class FakeArm:
    """Stands for one trainer arm: writes what the arm and its trainer write (its full stats record among them) and
    ends with exit code 0, or, with the variant alive, keeps running until it is ended."""
    pid = 4242
    variant = {}
    envs = {}

    def __init__(self, cmd, env=None, **_):
        name = cmd[cmd.index("--arm") + 1]
        assert cmd[cmd.index("--steps") + 1] == str(N)
        scratch = Path(cmd[cmd.index("--scratch") + 1])
        run = scratch / "run"
        v = self.variant.get(name, self.variant.get("lockstep_b", {}) if name == "lockstep_b_again" else {})
        for b in range(8):
            write_world(run, b, motor_shift=v.get("motor_shift", 0.0) if b == 3 else 0.0,
                        header=v.get("world_header", "chunk_000481800_1_00000.json"),
                        qpos_shift=v.get("qpos_shift", 0) if b == 2 else 0)
        write_checkpoint(run, weight_shift=v.get("weight_shift", 0.0), nonfinite_steps=v.get("nonfinite_steps", 0),
                         companion=companion_of(run, learner_rng=v.get("learner_rng", "r0"),
                                                worlds=v.get("worlds", "w0")))
        write_lane(run, records_columns=(name in ("lockstep_b", "lockstep_b_again") and not v.get("no_lane_columns")),
                   pe_shift=v.get("pe_shift", 0.0))
        if name == "lockstep_b" and not v.get("no_records"):
            write_records(run)
        write_stats(run, nan_steps=v.get("nan_steps", 0))
        log = "fake arm\n"
        if (env or {}).get("UEMA_LOCKSTEP_DETERMINISTIC") == "1" and not v.get("no_determinism_line"):
            log += ("lockstep determinism: use_deterministic_algorithms=True warn_only=True cudnn.deterministic=True "
                    "cudnn.benchmark=False mem_efficient_sdp=%s cudnn_sdp=False flash_sdp=False math_sdp=True "
                    "math_sdp_bf16_reduction=True CUBLAS_WORKSPACE_CONFIG=%s PYTHONHASHSEED=%s torch=2.6.0+cu124\n"
                    % (v.get("mem_efficient_sdp", False), env["CUBLAS_WORKSPACE_CONFIG"], env["PYTHONHASHSEED"]))
            log += "lockstep card memory: max_memory_reserved_mib=7012.0 max_memory_allocated_mib=6544.5\n"
        if v.get("attention_warning"):
            log += ("UserWarning: Memory Efficient attention defaults to a non-deterministic algorithm. To explicitly "
                    "enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).\n")
        for _ in range(v.get("nondeterministic_warnings", 0)):
            log += "UserWarning: index_add_cuda_ does not have a deterministic implementation, but you set ...\n"
        (scratch / "arm.log").write_text(log)
        FakeArm.envs[name] = dict(env or {})
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(dict(problems=[])))
        self.code = None if v.get("alive") else 0

    def poll(self):
        return self.code


def run_job(tmp_path, variant=None, hashes_change=None, **job_kw):
    a, ck, model_sha, hashes = setup(tmp_path)
    if hashes_change:
        hashes.update(hashes_change(a))
    FakeArm.variant = variant or {}
    FakeArm.envs = {}
    started = []

    def popen(cmd, **kw):
        started.append(cmd[cmd.index("--arm") + 1])
        return FakeArm(cmd, **kw)
    kw = dict(paused=lambda: (True, "paused"), during=lambda own, root: [], priority=lambda procs: {},
              listing=lambda proc: [])
    kw.update(job_kw)
    code = J.job(a, popen=popen, contain=False, checkpoint=ck, preconditions_fn=lambda: [], supervisor_env={},
                 machine=lambda: (None, []), gpu=lambda: {}, expected_model=model_sha,
                 code_hash=lambda tree: hashes.get(str(tree), "c" * 64), sleep=lambda s: None,
                 read_companion=read_fake_companion, **kw)
    return code, json.loads(Path(a.out).read_text()), a, started


def test_equal_arms_with_the_records_read_pass_and_every_weight_is_discarded(tmp_path):
    code, out, a, started = run_job(tmp_path)
    assert code == 0 and out["verdict"] == "pass", out
    assert started == ["lockstep_a", "lockstep_b", "lockstep_a_again"], "v3: the repeat runs every time"
    assert out["a_vs_a_again"] is not None and J.all_equal(out["a_vs_a_again"])
    for arm in started:
        assert FakeArm.envs[arm]["UEMA_LOCKSTEP_DETERMINISTIC"] == "1"
        assert FakeArm.envs[arm]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and FakeArm.envs[arm]["PYTHONHASHSEED"] == "0"
        assert out["determinism"][arm]["memory"].startswith("lockstep card memory:")
        assert out["determinism"][arm]["line"].startswith("lockstep determinism: use_deterministic_algorithms=True")
        assert out["determinism"][arm]["warnings"] == []
    ab = out["a_vs_b"]
    assert ab["commands"]["compared"] == 8 * 6 and ab["weights"]["equal"]
    assert out["model_steps"] == [481800, 481805] and out["end_save"] == 481806
    assert out["records_b"]["readings"]["step_lines"] == list(range(481800, 481806))
    # FIX-5: every part compared; the lane judged on production's columns, the branch's own three listed beside them
    assert ab["companion"]["equal"] and ab["companion"]["compared"] >= 6
    assert ab["lane"]["equal"] and set(ab["lane"]["b_only"]) == {"0:persist_pe_body", "0:vision_no_copy_body",
                                                                 "0:gaze_copy_applied"}
    assert ab["world"]["equal"] and ab["world"]["compared"] == 8 * 11, "12 columns per body, the header not judged"
    assert ab["weights"]["files"]["runtime_state.pt"]["tensors"] == 5, "plain numbers are leaves too"
    record = json.loads((Path(a.scratch) / "job_record.json").read_text())
    removed = {Path(r["path"]).name for r in record["discarded"]["removed"]}
    assert {"checkpoint_copy", "checkpoints"} <= removed and record["discarded_spike_snapshots"]
    for arm in ("lockstep_a", "lockstep_b", "lockstep_a_again"):
        run = Path(a.scratch) / ("arm_" + arm) / "run"
        assert not (run / "checkpoints").exists() and (run / "observer" / "world" / "b0").is_dir()
    assert not (Path(a.scratch) / "arm_lockstep_b" / "run" / "comparison_records" / "vision_spike" / "checkpoints").exists()
    assert (Path(a.scratch) / "arm_lockstep_b" / "run" / "comparison_records" / "steps_1.jsonl").is_file()


def test_a_command_difference_with_a_repeating_configuration_reads_fail(tmp_path):
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_b=dict(motor_shift=1e-3)))
    assert code == 1 and out["verdict"] == "fail", out
    assert started == ["lockstep_a", "lockstep_b", "lockstep_a_again", "lockstep_b_again"]
    assert J.all_equal(out["b_vs_b_again"])
    diff = out["a_vs_b"]["commands"]["differing"]
    assert {d["body"] for d in diff} == {3} and len(diff) == 6
    assert out["a_vs_a_again"]["commands"]["equal"]


def test_a_weight_difference_alone_reads_fail_and_names_the_tensor(tmp_path):
    code, out, _, _ = run_job(tmp_path, variant=dict(lockstep_b=dict(weight_shift=1e-6)))
    assert code == 1 and out["verdict"] == "fail"
    w = out["a_vs_b"]["weights"]["files"]["model.safetensors"]
    assert w["differing"] == ["forward_model.w"] and w["largest_difference"] > 0
    assert out["a_vs_b"]["commands"]["equal"]


@pytest.mark.parametrize("variant, part, name", [
    (dict(nonfinite_steps=1), "weights", "nonfinite_steps"),            # a plain number of runtime_state.pt
    (dict(learner_rng="r1"), "companion", "learner.torch_rng"),           # the companion's learner part
    (dict(pe_shift=1e-4), "lane", "0:next_pe_body"),                      # a lane column production writes
    (dict(qpos_shift=1), "world", "b2:qpos_raw"),                         # a world record column
])
def test_a_difference_in_any_judged_part_reads_fail_and_is_named(tmp_path, variant, part, name):
    """FIX-5: a change confined to a part the first version did not compare (non-tensor numbers, the companion, the
    lane's existing columns, the world records beyond the command) is caught."""
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_b=variant))
    assert code == 1 and out["verdict"] == "fail" and started[-1] == "lockstep_b_again", out["verdict_reason"]
    c = out["a_vs_b"][part]
    named = (c["files"]["runtime_state.pt"]["differing"] if part == "weights" else c["differing"])
    assert named == [name] and part in out["verdict_reason"]
    assert all(out["a_vs_b"][k]["equal"] for k in ("commands", "weights", "companion", "lane", "world") if k != part)


def test_parts_two_launches_of_one_code_differ_in_are_reported_and_not_judged(tmp_path):
    """The companion's 'worlds' and each world chunk's 'header' differ between two launches of one code (the bundle
    review's base-against-base probe): a difference there alone reads pass, and is reported."""
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_b=dict(
        worlds="w1", world_header="chunk_000481800_2_00000.json")))
    assert code == 0 and out["verdict"] == "pass" and started == ["lockstep_a", "lockstep_b", "lockstep_a_again"], out
    assert out["a_vs_b"]["companion"]["not_judged"] == {"worlds": False}
    assert out["a_vs_b"]["world"]["not_judged"]["b0:header"] is False
    assert out["not_judged"] == {"companion": ["worlds"], "world": ["header"]}
    # each arm's own folder in a text is read as <arm>, and the planner's id() addresses are left out: equal
    assert out["a_vs_b"]["companion"]["differing"] == []


def test_a_configuration_that_does_not_repeat_reads_invalid(tmp_path):
    code, out, _, _ = run_job(tmp_path, variant=dict(lockstep_b=dict(motor_shift=1e-3),
                                                     lockstep_a_again=dict(motor_shift=2e-3)))
    assert code == 2 and out["verdict"] == "invalid" and "does not repeat" in out["verdict_reason"]


@pytest.mark.parametrize("variant, why", [(dict(no_records=True), "no comparison records header"),
                                          (dict(no_lane_columns=True), "no persist_pe_body column")])
def test_missing_records_read_fail(tmp_path, variant, why):
    code, out, _, _ = run_job(tmp_path, variant=dict(lockstep_b=variant))
    assert code == 1 and out["verdict"] == "fail" and any(why in p for p in out["records_b"]["problems"]), out


def test_refusals_come_before_any_copy(tmp_path):
    with pytest.raises(SystemExit, match="not the declared"):
        run_job(tmp_path, hashes_change=lambda a: {a.code_b: "d" * 64})
    assert not list((tmp_path / "job").glob("arm_*"))
    with pytest.raises(SystemExit, match="same trainer code"):
        _same_code(tmp_path / "second")


def _same_code(tmp_path):
    """The branch's tree declared with production's own hash: the lockstep would compare a code with itself."""
    tmp_path.mkdir()
    a, ck, model_sha, hashes = setup(tmp_path)
    a.code_b_hash = J.CODE_A_HASH
    hashes[a.code_b] = J.CODE_A_HASH
    J.job(a, popen=lambda cmd, **kw: FakeArm(cmd), contain=False, checkpoint=ck, preconditions_fn=lambda: [],
          supervisor_env={}, machine=lambda: (None, []), gpu=lambda: {}, expected_model=model_sha,
          code_hash=lambda tree: hashes.get(str(tree)), sleep=lambda s: None, read_companion=read_fake_companion)


def test_the_commands_are_read_from_the_world_records_as_the_trainer_wrote_them(tmp_path):
    run = tmp_path / "run"
    write_world(run, 0)
    rows = J.latched_commands(run, bodies=(0,))[0]
    assert len(rows) == 6 and rows[0]["motor"].shape == (26, 9) and rows[0]["hand_goal"] is None
    assert np.array_equal(rows[4]["motor"], rows[0]["motor"]) and not np.array_equal(rows[5]["motor"], rows[0]["motor"])
    assert "no world record" in J.latched_commands(run, bodies=(1,))[1]


def test_the_window_ends_where_repeats_are_on_record():
    """FIX-4: the arms run 6 model steps (481,800 to 481,805, --max-steps 481,806); the end save ckpt_step_481806
    follows the replay pass after 481,805, whose weights the command of 481,806 used in two runs of one
    configuration."""
    assert (J.STEP, J.STEPS, J.LAST, J.END_SAVE) == (481800, 6, 481805, 481806)
    assert J.REPLAY_AFTER == (481801, 481803, 481805) and J.REPLAY_AFTER[-1] + 1 == J.END_SAVE
    cmd = J.arm_command("lockstep_a", Path("copy"), Path("scratch"))
    assert cmd[cmd.index("--steps") + 1] == "6"


def test_a_full_stats_record_is_read_with_the_declared_lines_and_passes(tmp_path):
    """Row records_lockstep_481800's INVALID: the arm's full stats record of model step 481,806 reached
    learning_on.record_stop with limits {} (KeyError 'gross'). Every fake arm here writes that record with arm A's
    values; the declared lines read it and stop nothing, so the row passes; the learning-on limits file's
    100-model-step lines would have stopped the arm at that record (audio 1361.7 in 1,000 against 1119.1)."""
    code, out, a, started = run_job(tmp_path)
    assert code == 0 and out["verdict"] == "pass", out
    record = json.loads((Path(a.scratch) / "job_record.json").read_text())
    assert record["arm_runs"]["lockstep_a"]["records"] == 1 and "stopped" not in record["arm_runs"]["lockstep_a"]
    assert out["limits"]["gross"]["audio"] == "inf" and out["exception"] is None and out["ended_before_discard"] == []
    rec = [json.loads(x) for x in (Path(a.scratch) / "arm_lockstep_a" / "run" / "replay_training_stats.jsonl")
           .read_text().splitlines()][-1]
    assert J.L.record_stop(rec, J.LIMITS) is None
    limits_file = Path("D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/learning_on/limits.json")
    if limits_file.is_file():
        why = J.L.record_stop(rec, json.loads(limits_file.read_text(encoding="utf-8")))
        assert why and why.startswith("audio's prediction ratio 1361.7"), why


def test_a_non_finite_update_still_stops_the_arm_and_the_row_reads_invalid(tmp_path):
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_a=dict(nan_steps=2)))
    assert code == 2 and out["verdict"] == "invalid" and started == ["lockstep_a"]
    assert "not finite" in out["verdict_reason"] and "lockstep_a was stopped (reading)" in out["verdict_reason"]


def test_an_exception_with_the_arm_alive_ends_it_and_waits_before_any_discard(tmp_path, monkeypatch):
    """run_arm raises while arm A still runs (here its priority reading fails): the job records the exception, ends
    the arm's tree and waits for it (the end function) and only then discards; the row reads invalid."""
    order = []

    def failing_priority(procs):
        raise RuntimeError("planted: the reading failed while the arm wrote its end save")

    def end(proc):
        order.append(("end", proc.code))
        proc.code = 1                                   # the tree ended and waited for
        return dict(ended=1, wait_limit_s=60.0, waited_s=0.1, alive_after_wait=[])
    real_discard = J.L.discard_weights

    def discard(scratch, arms):
        order.append(("discard", tuple(arms)))
        return real_discard(scratch, arms)
    monkeypatch.setattr(J.L, "discard_weights", discard)
    code, out, a, started = run_job(tmp_path, variant=dict(lockstep_a=dict(alive=True)), priority=failing_priority,
                                    end=end)
    assert code == 2 and out["verdict"] == "invalid" and started == ["lockstep_a"]
    assert order == [("end", None), ("discard", ("lockstep_a",))], order
    assert out["exception"]["type"] == "RuntimeError" and "planted" in out["exception"]["message"]
    assert out["ended_before_discard"][0]["arm"] == "lockstep_a" and out["ended_before_discard"][0]["end"]["ended"] == 1
    assert "the job raised RuntimeError" in out["verdict_reason"]
    assert not (Path(a.scratch) / "arm_lockstep_a" / "run" / "checkpoints").exists(), "discarded after the end"


def test_a_repeat_that_differs_reads_invalid_even_when_a_and_b_are_equal(tmp_path):
    """v3 (row records_lockstep_2_481800): launches 1 and 2 agreed while launch 3 parted, so a and b being equal counts
    only when a and a_again are equal too."""
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_a_again=dict(motor_shift=1e-3)))
    assert code == 2 and out["verdict"] == "invalid", out
    assert started == ["lockstep_a", "lockstep_b", "lockstep_a_again"] and J.all_equal(out["a_vs_b"])
    assert "does not repeat" in out["verdict_reason"] and "equal in every judged part" in out["verdict_reason"]


def test_an_arm_without_the_determinism_line_reads_invalid(tmp_path):
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_b=dict(no_determinism_line=True)))
    assert code == 2 and out["verdict"] == "invalid" and started == ["lockstep_a", "lockstep_b"], out
    assert "lockstep_b: arm.log holds no 'lockstep determinism:' line" in out["verdict_reason"]


def test_the_warnings_of_operations_without_a_deterministic_version_are_recorded(tmp_path):
    code, out, _, _ = run_job(tmp_path, variant=dict(lockstep_a=dict(nondeterministic_warnings=3)))
    assert code == 0 and out["verdict"] == "pass", out
    det = out["determinism"]["lockstep_a"]
    assert det["warning_lines"] == 3 and len(det["warnings"]) == 1 and "index_add_cuda_" in det["warnings"][0]


def test_every_differing_name_is_kept():
    """Row records_lockstep_2_481800's result listed 50 of 555 differing tensors; v3 keeps every name."""
    names = ["t%03d" % i for i in range(120)]
    a = dict(digests={"model.safetensors": {n: ["s", "d", "x"] for n in names}}, model_values={})
    b = dict(digests={"model.safetensors": {n: ["s", "d", "y"] for n in names}}, model_values={})
    w = J.compare_tensors(a, b)["files"]["model.safetensors"]
    assert w["differing_count"] == 120 and w["differing"] == names
    p = J.compare_parts({n: 1 for n in names}, {n: 2 for n in names})
    assert p["differing_count"] == 120 and p["differing"] == names


def test_the_arm_entry_switches_the_deterministic_mode_on_and_reports_it(monkeypatch):
    """The arm entry's deterministic_mode, run in this process and switched back after: PyTorch reports the mode on,
    warn-only, cuDNN deterministic and benchmark off, and the line the job reads holds those settings."""
    entry = REPO / "docs/gain_of_function_v4/comparison_records/records_lockstep_trial_trainer.py"
    spec = importlib.util.spec_from_file_location("records_lockstep_trial_trainer_under_test", entry)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    cuda = torch.backends.cuda
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    sdp = (cuda.mem_efficient_sdp_enabled(), cuda.cudnn_sdp_enabled(), cuda.flash_sdp_enabled(), cuda.math_sdp_enabled(),
           cuda.fp16_bf16_reduction_math_sdp_allowed())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    try:
        line = E.deterministic_mode()
        assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        assert not cuda.mem_efficient_sdp_enabled() and not cuda.cudnn_sdp_enabled() and not cuda.flash_sdp_enabled()
        assert cuda.math_sdp_enabled() and cuda.fp16_bf16_reduction_math_sdp_allowed()
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[2], before[3]
        cuda.enable_mem_efficient_sdp(sdp[0])
        cuda.enable_cudnn_sdp(sdp[1])
        cuda.enable_flash_sdp(sdp[2])
        cuda.enable_math_sdp(sdp[3])
        cuda.allow_fp16_bf16_reduction_math_sdp(sdp[4])
    assert line.startswith(J.DETERMINISM_LINE)
    assert J.determinism_of_line(line) == [], "the entry's line shows every setting the job requires"
    assert J.determinism_of(Path("no such arm.log")) == dict(line=None, missing_settings=list(J.DETERMINISM_SETTINGS),
                                                            warnings=[], warning_lines=0, attention_warnings=[],
                                                            memory=None)


def test_a_difference_of_b_that_its_own_repeat_does_not_hold_reads_invalid(tmp_path):
    """The review of Oct 10, BLOCKER 2: a equal to a_again shows one configuration repeats, not that b's launch does; a
    difference of a and b that b_again does not repeat reads invalid, never fail."""
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_b=dict(motor_shift=1e-3),
                                                           lockstep_b_again=dict(motor_shift=2e-3)))
    assert code == 2 and out["verdict"] == "invalid", out
    assert started == ["lockstep_a", "lockstep_b", "lockstep_a_again", "lockstep_b_again"]
    assert "b's launch does not repeat" in out["verdict_reason"]


def test_b_repeat_does_not_run_when_a_and_b_are_equal_or_a_does_not_repeat(tmp_path):
    _, out, _, started = run_job(tmp_path / "equal")
    assert out["verdict"] == "pass" and "lockstep_b_again" not in started and out["b_vs_b_again"] is None
    _, out, _, started = run_job(tmp_path / "no_repeat", variant=dict(lockstep_b=dict(motor_shift=1e-3),
                                                                     lockstep_a_again=dict(motor_shift=2e-3)))
    assert out["verdict"] == "invalid" and "lockstep_b_again" not in started


def test_a_determinism_line_missing_a_setting_reads_invalid(tmp_path):
    """The review of Oct 10, FIX 2: the line must show every setting, here the efficient attention switched off."""
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_a=dict(mem_efficient_sdp=True)))
    assert code == 2 and out["verdict"] == "invalid" and started == ["lockstep_a"], out
    assert "does not show mem_efficient_sdp=False" in out["verdict_reason"]
    assert out["determinism"]["lockstep_a"]["missing_settings"] == ["mem_efficient_sdp=False"]


def test_an_attention_warning_reads_invalid(tmp_path):
    """The second reader of Oct 10, FIX 6: with the efficient, cuDNN and flash back ends off, PyTorch's attention warning
    means the switch did not hold; the row reads invalid and the series ends."""
    code, out, _, started = run_job(tmp_path, variant=dict(lockstep_a=dict(attention_warning=True)))
    assert code == 2 and out["verdict"] == "invalid" and started == ["lockstep_a"], out
    det = out["determinism"]["lockstep_a"]
    assert det["warning_lines"] == 1 and "defaults to a non-deterministic algorithm" in det["attention_warnings"][0]
    assert "a non-deterministic attention ran" in out["verdict_reason"]


def test_the_card_memory_line_is_read():
    """The arm entry's peak card memory after the trainer returns (the math attention's cost, measured) is read."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "arm.log"
        log.write_text("x\nlockstep card memory: max_memory_reserved_mib=7012.0 max_memory_allocated_mib=6544.5\n")
        assert J.determinism_of(log)["memory"].endswith("max_memory_allocated_mib=6544.5")
