# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE LATENT CONTRACT CHECK'S JOB (docs/gain_of_function_v4/latent_contract_repair/latent_contract_check.py).

The job's own logic, with the trainer arm replaced by a fake that replays recorded H84 rows (latent_contract_rows_481802.npz)
through the real shadow hook, as the arm's trainer would call it: its refusals before any copy, its verdicts (pass,
fail, invalid), and the discard in its finally path. The learning-on package it reuses lives on its own branch
(C:/Users/bhart/Documents/UEMA_wt/learning_on_stop_limits); the module is skipped when that folder is absent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mimo_bridge.tests import test_latent_contract_repair as R
from mimo_bridge.tools import latent_body_start_state as S

HERE = Path(__file__).resolve()
JOB_FILE = R.REPO / "docs/gain_of_function_v4/latent_contract_repair/latent_contract_check.py"
LEARNING_ON = Path("C:/Users/bhart/Documents/UEMA_wt/learning_on_stop_limits/docs/gain_of_function_v4/"
                   "learning_on_stop_limits/learning_on.py")
if not LEARNING_ON.is_file():
    pytest.skip("the learning-on package's folder is absent", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("latent_contract_check", JOB_FILE)
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(tmp_path, *, extra_change=None):
    """A fake checkpoint, a declared start, a command naming it, a limits file and the job's arguments."""
    ck = tmp_path / "ckpt" / "ckpt_step_481800"
    ck.mkdir(parents=True)
    (ck / "model.safetensors").write_bytes(b"model numbers")
    companion = tmp_path / "companion.pt.gz"
    companion.write_bytes(b"episode")
    (ck / "episode_state.json").write_text(json.dumps(dict(storage_path=str(companion), sha256=sha(companion))))
    config = S.load_config(R.NEW_CONFIG)
    start = tmp_path / "start" / "latent_start.pt"
    S.write_start(start, S.fresh_start(R.TREE, config),
                  S.start_record("fresh", R.NEW_CONFIG, config, R.TREE, R.LEVELS, "trunk_v1"))
    cmd = json.loads(J.REFERENCE.read_text(encoding="utf-8"))
    for flag, value in (("--latent-body-config", str(R.NEW_CONFIG)), ("--latent-body-resume", str(start))) + (
            extra_change or ()):
        cmd[cmd.index(flag) + 1] = value
    command = tmp_path / "command.json"
    command.write_text(json.dumps(cmd, indent=1))
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps(dict(gross={})))
    old_state = tmp_path / "old_shadow" / "latest.pt"
    R.hook_run(R.arguments(tmp_path, R.OLD_CONFIG, name="old_shadow"), servo=False, contract=R.OLD)
    a = SimpleNamespace(scratch=str(tmp_path / "job"), out=str(tmp_path / "result.json"), steps=R.N_FRAMES,
                        command=str(command), command_sha256=sha(command), config=str(R.NEW_CONFIG),
                        config_sha256=S.text_sha256(R.NEW_CONFIG), start=str(start), old_config=str(R.OLD_CONFIG),
                        old_state=str(old_state), limits=str(limits), limits_sha256=sha(limits), deadline_s=600.0)
    # the job's code-hash refusal is tested on its own (test_the_second_attempt_refuses_another_code_or_record); here
    # the code tree's own hash is declared, so the job's other parts are tested on whichever commit runs them
    from lem.trainer_code import trainer_code_hash
    a.expected_code_hash = trainer_code_hash()
    return a, ck, sha(ck / "model.safetensors")


class FakeArm:
    """Stands for the trainer arm: writes what the arm and its trainer write (arm.log, run/args.json, the shadow's
    folder through the real hook, the arm record) and ends with `code`."""
    pid = 4242
    envs = []

    def __init__(self, cmd, config, servo=True, code=0, env=None, no_args=False, **_):
        FakeArm.envs.append(dict(env or {}))
        scratch = Path(cmd[cmd.index("--scratch") + 1])
        if no_args:
            # the second attempt's arm (row latent_contract_check_2_481800): refused at its command's sha256 before
            # train_mimo.main, it wrote its log and its record and nothing else
            with (scratch / "arm.log").open("a", encoding="utf-8") as log:
                log.write("refused\n")
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(dict(problems=["the command file differs"])))
            self.code = 2
            return
        args = R.arguments(scratch, config, name="latent_body", resume=self.start)
        R.hook_run(args, servo=servo, contract=R.NEW if servo else R.OLD)
        (scratch / "run").mkdir(parents=True, exist_ok=True)
        (scratch / "run" / "args.json").write_text(json.dumps(dict(
            finger_joint_servo="on" if servo else "off", latent_body_config=str(config),
            latent_body_resume=self.start)))
        with (scratch / "arm.log").open("a", encoding="utf-8") as log:
            log.write("fake arm\n")
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(dict(problems=[] if code == 0 else ["exit"])))
        self.code = code

    def poll(self):
        return self.code


def run_job(tmp_path, a, ck, model_sha, config=None, code=0, preconditions_fn=lambda: [], no_args=False):
    FakeArm.start = a.start
    FakeArm.envs = []

    def popen(cmd, env=None, **_):
        return FakeArm(cmd, config or R.NEW_CONFIG, code=code, env=env, no_args=no_args)
    preserved = iter([dict(x=1), dict(x=1)])
    return J.job(a, popen=popen, contain=False, checkpoint=ck, preconditions_fn=preconditions_fn, supervisor_env={},
                 machine=lambda: (None, []), gpu=lambda: {}, expected_model=model_sha,
                 preserved=lambda: next(preserved), sleep=lambda s: None)


def test_the_command_differs_from_the_finger_arm_only_in_the_shadows_configuration_and_start():
    cmd = json.loads(J.COMMAND.read_text(encoding="utf-8"))
    ref = json.loads(J.REFERENCE.read_text(encoding="utf-8"))
    assert sha(J.REFERENCE) == J.REFERENCE_SHA256
    assert sorted(J.command_differences(ref, cmd)) == sorted(J.CHANGED)
    values = J.flag_values(cmd)
    # the command names the repaired configuration by its path in the checkout the command was built in (the latent
    # repair's worktree); in another checkout of the same commit the file is compared by its name and its text
    named = Path(values["--latent-body-config"][0])
    assert named.name == R.NEW_CONFIG.name and values["--finger-joint-servo"] == ["on"]
    if named.is_file():
        assert S.text_sha256(named) == S.text_sha256(R.NEW_CONFIG)
    assert J.command_problems(cmd, ref, named, values["--latent-body-resume"][0]) == []


def test_a_passing_job_judges_the_shadow_and_discards_the_learned_states(tmp_path):
    a, ck, model_sha = setup(tmp_path)
    assert run_job(tmp_path, a, ck, model_sha) == 0
    out = json.loads(Path(a.out).read_text())
    assert out["verdict"] == "pass" and out["validity_problems"] == [] and out["shadow"]["verdict"] == "pass"
    record = json.loads((Path(a.scratch) / "job_record.json").read_text())
    removed = [Path(r["path"]).name for r in record["discarded"]["removed"]]
    assert "latest.pt" in removed and "checkpoint_copy" in removed
    shadow = Path(a.scratch) / "arm_latent_check" / "latent_body"
    assert not (shadow / "latest.pt").exists() and (shadow / "trajectory.jsonl").is_file()
    assert Path(a.start).is_file() and S.sha256_file(a.start) == record["start"]["record"]["sha256"]


def test_the_sep30_configuration_in_the_arm_reads_fail(tmp_path):
    a, ck, model_sha = setup(tmp_path)
    assert run_job(tmp_path, a, ck, model_sha, config=R.OLD_CONFIG) == 1
    assert json.loads(Path(a.out).read_text())["verdict"] == "fail"


def test_an_arm_that_ends_badly_reads_invalid_and_is_still_discarded(tmp_path):
    a, ck, model_sha = setup(tmp_path)
    assert run_job(tmp_path, a, ck, model_sha, code=1) == 2
    out = json.loads(Path(a.out).read_text())
    assert out["verdict"] == "invalid" and any("exit code" in p for p in out["validity_problems"])
    assert not (Path(a.scratch) / "arm_latent_check" / "checkpoint_copy").exists()


@pytest.mark.parametrize("change, why", [("third_flag", "differs from the matched comparison"),
                                         ("config_sha", "config file"), ("start_file", "sidecar"),
                                         ("steps", "lies outside"), ("relative", "relative path"),
                                         ("busy", "runs the trainer")])
def test_the_job_refuses_before_any_copy(tmp_path, change, why):
    extra = (("--babble-noise", "0.06"),) if change == "third_flag" else None
    a, ck, model_sha = setup(tmp_path, extra_change=extra)
    kw = {}
    if change == "config_sha":
        a.config_sha256 = "0" * 64
    if change == "start_file":
        with open(a.start, "ab") as f:
            f.write(b"x")
    if change == "steps":
        a.steps = 3
    if change == "relative":
        a.scratch = "job"
    if change == "busy":
        kw["preconditions_fn"] = lambda: ["process 1 runs the trainer"]
    with pytest.raises(SystemExit, match="refused.*" + why):
        run_job(tmp_path, a, ck, model_sha, **kw)
    assert not (tmp_path / "job" / "arm_latent_check").exists()


def admission_setup(tmp_path, bound):
    """The second attempt's form: an admission record binding trainer code `bound`, named by the command in place of
    the reference's record."""
    from lem.trainer_code import trainer_code_hash
    record = tmp_path / "admission_record.json"
    record.write_text(json.dumps(dict(code_hash=bound)))
    a, ck, model_sha = setup(tmp_path, extra_change=(("--replace-saved-hand-calibration", str(record)),))
    a.expected_code_hash, a.admission_record, a.admission_sha256 = trainer_code_hash(), str(record), sha(record)
    return a, ck, model_sha


def test_the_second_attempt_runs_with_its_declared_code_and_admission_record(tmp_path):
    from lem.trainer_code import trainer_code_hash
    a, ck, model_sha = admission_setup(tmp_path, trainer_code_hash())
    a.expected_excluded = None
    assert run_job(tmp_path, a, ck, model_sha) == 0
    record = json.loads((Path(a.scratch) / "job_record.json").read_text())
    assert record["admission"]["code_hash"] == trainer_code_hash() == record["code_tree"]["expected"]


@pytest.mark.parametrize("change, why", [("hash", "not the declared"), ("record_binds", "binds trainer code"),
                                         ("record_sha", "admission record"), ("no_record", "not exactly")])
def test_the_second_attempt_refuses_another_code_or_record(tmp_path, change, why):
    from lem.trainer_code import trainer_code_hash
    a, ck, model_sha = admission_setup(tmp_path, "f" * 64 if change == "record_binds" else trainer_code_hash())
    if change == "hash":
        a.expected_code_hash = "e" * 64
    if change == "record_sha":
        a.admission_sha256 = "0" * 64
    if change == "no_record":
        a.admission_record = None                   # the command still names a third changed value
    with pytest.raises(SystemExit, match="refused.*" + why):
        run_job(tmp_path, a, ck, model_sha)
    assert not (tmp_path / "job" / "arm_latent_check").exists()


def test_the_arm_runs_the_command_file_the_job_names(tmp_path):
    """Row latent_contract_check_2_481800 read INVALID: the arm entry named latent_check_command.json whatever the
    job's --command, so the second attempt's arm read the first attempt's file. The job hands its --command to the arm
    (COMMAND_ENV), and the entry reads it."""
    a, ck, model_sha = setup(tmp_path)
    assert run_job(tmp_path, a, ck, model_sha) == 0
    assert FakeArm.envs[-1][J.COMMAND_ENV] == str(Path(a.command).resolve())
    record = json.loads((Path(a.scratch) / "job_record.json").read_text())
    assert record["arm_command_file"] == str(Path(a.command).resolve())
    entry = J.ARM_ENTRY.read_text(encoding="utf-8")
    assert 'COMMAND_ENV = "%s"' % J.COMMAND_ENV in entry and "os.environ.get(COMMAND_ENV)" in entry


def test_an_arm_refused_before_training_reads_invalid_and_the_after_readings_are_taken(tmp_path):
    """The second attempt's arm wrote no run/args.json; v1's judgment raised FileNotFoundError and the job ended
    without reading the preserved files and the original again. v2 judges the shadow only after an arm that ran."""
    a, ck, model_sha = setup(tmp_path)
    assert run_job(tmp_path, a, ck, model_sha, no_args=True) == 2
    out = json.loads(Path(a.out).read_text())
    assert out["verdict"] == "invalid" and out["shadow"] is None
    assert any("exit code 2" in p for p in out["validity_problems"])
    record = json.loads((Path(a.scratch) / "job_record.json").read_text())
    assert "preserved_after" in record and record["original_changed"] == []
    assert not (Path(a.scratch) / "arm_latent_check" / "checkpoint_copy").exists()


def test_the_declared_lines_let_the_end_record_through_and_keep_the_other_stops():
    """Row latent_contract_check_2_481800's second run: the trainer's end record of model step 481,810 (10 model steps)
    was judged against the learning-on file's 100-model-step gross lines and stopped the arm (audio 1141.2 in 1,000
    against 1119.1). The declared LIMITS read that record and stop nothing; a non-finite update still stops."""
    ratios = dict(tactile=0.9305916472820346, proprioception=0.5950041611278907, vision=0.7350867051561488,
                  audio=1.1412)
    rec = dict(type="mimo", step=481810, nan_steps=0, timestamp=1.0, elapsed_s=80.0,
               prediction_audit_per_body=dict(senses={k: dict(without_body_2=dict(ratio=v)) for k, v in ratios.items()}),
               hand_control=dict(joint_error_deg=12.19))
    assert J.L.record_stop(rec, J.LIMITS) is None
    assert all(v["line"] == float("inf") for v in J.LIMITS["gross"].values())
    limits_file = Path("D:/UEMA_Research/external/D/UEMA_scratch/finger_investigation/learning_on/limits.json")
    if limits_file.is_file():
        why = J.L.record_stop(rec, json.loads(limits_file.read_text(encoding="utf-8")))
        assert why and why.startswith("audio's prediction ratio 1141.2"), why
    rec["nan_steps"] = 1
    assert "not finite" in J.L.record_stop(rec, J.LIMITS)
    rec["nan_steps"] = 0
    del rec["hand_control"]
    assert "finger joint error" in J.L.record_stop(rec, J.LIMITS)
